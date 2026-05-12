import asyncio
import os
import json
from datetime import datetime
from dotenv import load_dotenv

import httpx
import google.generativeai as genai
from supabase import create_client, Client

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

load_dotenv()

# ---------- Configuration ----------
API_BASE = os.getenv("API_BASE_URL")                # Your backend on Render
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROCESSED_COLUMN = os.getenv("PROCESSED_COLUMN", "professional_rewrite")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-1.5-pro

# ---------- Professional News Editor Prompt (Gemini expects plain text) ----------
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
    First try using your API with a custom query param (recommended).
    If not available, fallback to direct Supabase query.
    """
    # Option A: Use a direct Supabase query (most reliable)
    # This assumes your table has the column 'professional_rewrite'
    result = supabase.table("ai_news")\
        .select("id, title, ai_content, short_desc")\
        .is_(PROCESSED_COLUMN, "null")\
        .limit(limit)\
        .execute()
    articles = result.data
    if articles:
        return articles

    # Option B: Try your API endpoint with a special param (if you added it)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{API_BASE}/api/news",
                params={"limit": limit, "no_professional": "true"},
                timeout=10.0
            )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
    except Exception as e:
        print(f"API fallback failed: {e}")
    return []

# ---------- Core rewriting using Gemini ----------
async def rewrite_professionally(title: str, original_content: str) -> dict:
    """Send prompt to Gemini and parse JSON response."""
    prompt = EDITOR_PROMPT_TEMPLATE.format(title=title, content=original_content)
    
    # Gemini expects synchronous call, but we wrap in a thread to avoid blocking asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: gemini_model.generate_content(prompt)
    )
    
    raw_text = response.text.strip()
    # Extract JSON from response (Gemini may add markdown or extra text)
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0]
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[1].split("```")[0]
    raw_text = raw_text.strip()
    
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: create basic structure
        parsed = {"headline": title, "body": raw_text}
    
    return {
        "headline": parsed.get("headline", title),
        "body": parsed.get("body", original_content)
    }

async def process_article(article: dict):
    """Rewrite one article and update Supabase."""
    article_id = article["id"]
    original_title = article.get("title", "")
    # Use the already AI‑rewritten content as base (or original description)
    base_content = article.get("ai_content") or article.get("short_desc") or ""
    if not base_content:
        print(f"⚠️ Article {article_id} has no content, skipping")
        return

    try:
        rewritten = await rewrite_professionally(original_title, base_content)
        full_text = f"{rewritten['headline']}\n\n{rewritten['body']}"

        supabase.table("ai_news")\
            .update({
                PROCESSED_COLUMN: full_text,
                "professional_rewrite_at": datetime.utcnow().isoformat()
            })\
            .eq("id", article_id)\
            .execute()
        print(f"✅ Professional rewrite done for article {article_id}")
    except Exception as e:
        print(f"❌ Error rewriting article {article_id}: {e}")

async def fetch_and_process_batch():
    """Main batch function: get up to 3 unprocessed articles, rewrite, update DB."""
    articles = await fetch_articles_to_process(limit=3)
    if not articles:
        print(f"[{datetime.utcnow().isoformat()}] No articles awaiting professional rewrite. Sleeping...")
        return
    print(f"Processing {len(articles)} article(s)...")
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
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "process_professional_rewrite":
        await fetch_and_process_batch()
        return [types.TextContent(type="text", text="Batch processed.")]
    else:
        raise ValueError(f"Unknown tool: {name}")

# ---------- Background scheduler (runs every 60 seconds) ----------
async def background_worker():
    while True:
        await fetch_and_process_batch()
        await asyncio.sleep(60)   # exactly 1 minute

# ---------- Main entry point ----------
async def main():
    # Start background worker as a separate task
    asyncio.create_task(background_worker())
    # Run MCP stdio server (for manual tools via Claude Desktop etc.)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="professional-news-editor",
                server_version="1.0.0"
            ),
            notification_options=NotificationOptions(tools_changed=True)
        )

if __name__ == "__main__":
    asyncio.run(main())