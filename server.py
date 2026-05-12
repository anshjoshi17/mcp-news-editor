import asyncio
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

import httpx
from supabase import create_client, Client

import mcp.server.stdio
import mcp.types as types
from mcp.server.models import InitializationOptions

# MCP import location differs across versions, so keep a safe fallback.
try:
    from mcp.server.lowlevel import Server, NotificationOptions
except ImportError:
    from mcp.server import Server, NotificationOptions

# New Gemini SDK
from google import genai

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("professional-news-editor")

# ---------- Configuration ----------
API_BASE = os.getenv("API_BASE_URL")  # Your backend on Render
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROCESSED_COLUMN = os.getenv("PROCESSED_COLUMN", "professional_rewrite")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is required")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- Professional News Editor Prompt ----------
EDITOR_PROMPT_TEMPLATE = """You are a senior news editor at a major international publication.
Rewrite the following article in a clear, authoritative, and engaging journalistic style.
Follow these rules strictly:
- Inverted pyramid: most important facts first.
- Short paragraphs (2-3 sentences each).
- Remove all promotional or clickbait language.
- Keep all factual claims, names, dates, and numbers unchanged.
- Add a strong, neutral, and informative headline (max 12 words).
- Use active voice and avoid clichés.
- Do NOT add opinions or extra facts.

Output must be valid JSON with exactly two fields: "headline" and "body".

Original article:
Title: {title}
Content: {content}
"""

# ---------- Helper: Fetch articles without professional rewrite ----------
async def fetch_articles_to_process(limit: int = 3):
    """
    Fetch articles that still have NULL in professional_rewrite column.
    First try direct Supabase query.
    If not available, fallback to your API endpoint.
    """
    try:
        result = (
            supabase.table("ai_news")
            .select("id, title, ai_content, short_desc")
            .is_(PROCESSED_COLUMN, "null")
            .limit(limit)
            .execute()
        )
        articles = result.data or []
        if articles:
            return articles
    except Exception as e:
        logger.exception("Supabase fetch failed: %s", e)

    try:
        if not API_BASE:
            return []

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{API_BASE}/api/news",
                params={"limit": limit, "no_professional": "true"},
                timeout=10.0,
            )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", []) or []
    except Exception as e:
        logger.exception("API fallback failed: %s", e)

    return []

# ---------- Core rewriting using Gemini ----------
async def rewrite_professionally(title: str, original_content: str) -> dict:
    """Send prompt to Gemini and parse JSON response."""
    prompt = EDITOR_PROMPT_TEMPLATE.format(title=title, content=original_content)

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        ),
    )

    raw_text = (response.text or "").strip()

    # Extract JSON from response (Gemini may add markdown or extra text)
    if "```json" in raw_text:
        raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw_text:
        raw_text = raw_text.split("```", 1)[1].split("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: preserve whatever came back
        parsed = {"headline": title, "body": raw_text or original_content}

    return {
        "headline": parsed.get("headline", title),
        "body": parsed.get("body", original_content),
    }

async def process_article(article: dict):
    """Rewrite one article and update Supabase."""
    article_id = article["id"]
    original_title = article.get("title", "")
    base_content = article.get("ai_content") or article.get("short_desc") or ""

    if not base_content:
        logger.warning("Article %s has no content, skipping", article_id)
        return

    try:
        rewritten = await rewrite_professionally(original_title, base_content)
        full_text = f"{rewritten['headline']}\n\n{rewritten['body']}"

        (
            supabase.table("ai_news")
            .update(
                {
                    PROCESSED_COLUMN: full_text,
                    "professional_rewrite_at": datetime.utcnow().isoformat(),
                }
            )
            .eq("id", article_id)
            .execute()
        )
        logger.info("Professional rewrite done for article %s", article_id)
    except Exception as e:
        logger.exception("Error rewriting article %s: %s", article_id, e)

async def fetch_and_process_batch():
    """Main batch function: get up to 3 unprocessed articles, rewrite, update DB."""
    articles = await fetch_articles_to_process(limit=3)
    if not articles:
        logger.info("[%s] No articles awaiting professional rewrite.", datetime.utcnow().isoformat())
        return

    logger.info("Processing %s article(s)...", len(articles))
    for art in articles:
        await process_article(art)

# ---------- MCP Server Setup ----------
server = Server("professional-news-editor")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="process_professional_rewrite",
            description="Fetch up to 3 articles without professional rewrite, rewrite them with Gemini, and store the result.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "process_professional_rewrite":
        await fetch_and_process_batch()
        return [types.TextContent(type="text", text="Batch processed.")]
    raise ValueError(f"Unknown tool: {name}")

# ---------- Background scheduler (runs every 60 seconds) ----------
async def background_worker():
    while True:
        try:
            await fetch_and_process_batch()
        except Exception as e:
            logger.exception("Background worker error: %s", e)
        await asyncio.sleep(60)

# ---------- Main entry point ----------
async def main():
    asyncio.create_task(background_worker())

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="professional-news-editor",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions()
                ),
            ),
            notification_options=NotificationOptions(tools_changed=True),
        )

if __name__ == "__main__":
    asyncio.run(main())