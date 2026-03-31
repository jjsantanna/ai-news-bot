#!/usr/bin/env python3
"""
daily_ai_feed.py — Fetch top AI news from 10 curated sources and send to Telegram.

Usage:
    python3 daily_ai_feed.py           # send to Telegram
    python3 daily_ai_feed.py --print   # print only, no Telegram
"""

import argparse
import feedparser
import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "../private/credentials.json")

# Top 10 AI RSS sources
SOURCES = [
    {"name": "MIT Technology Review – AI",     "url": "https://www.technologyreview.com/feed/"},
    {"name": "The Verge – AI",                 "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"},
    {"name": "Wired – AI",                     "url": "https://www.wired.com/feed/tag/ai/latest/rss"},
    {"name": "VentureBeat – AI",               "url": "https://venturebeat.com/category/ai/feed/"},
    {"name": "TechCrunch – AI",                "url": "https://techcrunch.com/category/artificial-intelligence/feed/"},
    {"name": "Google AI Blog",                 "url": "https://blog.research.google/feeds/posts/default/-/artificial%20intelligence"},
    {"name": "OpenAI Blog",                    "url": "https://openai.com/blog/rss.xml"},
    {"name": "Hugging Face Blog",              "url": "https://huggingface.co/blog/feed.xml"},
    {"name": "Import AI (Jack Clark)",         "url": "https://importai.substack.com/feed"},
    {"name": "AI News (artificialintelligence-news.com)", "url": "https://www.artificialintelligence-news.com/feed/"},
]

MAX_ITEMS_PER_SOURCE = 2   # top 2 per source = up to 20 items total
LOOKBACK_HOURS = 26        # include articles from last 26h (covers overnight)

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        return {}
    with open(CREDENTIALS_FILE) as f:
        return json.load(f)

def send_telegram(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.load(resp)
    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")
    print(f"✅ Sent via Telegram. Message ID: {result['result']['message_id']}")

def entry_date(entry):
    """Return a timezone-aware datetime for a feed entry, or epoch if unavailable."""
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                import time
                ts = time.mktime(t)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass
    return datetime.fromtimestamp(0, tz=timezone.utc)

def truncate(text, max_len=120):
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    return text[:max_len] + "…" if len(text) > max_len else text

def fetch_feed(source):
    try:
        feed = feedparser.parse(source["url"])
        return feed.entries[:MAX_ITEMS_PER_SOURCE * 3]  # grab extras to filter by date
    except Exception as e:
        print(f"  ⚠️  {source['name']}: {e}", file=sys.stderr)
        return []

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="Print only, don't send to Telegram")
    args = parser.parse_args()

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    today = datetime.now(tz=timezone(timedelta(hours=1)))  # Amsterdam time

    print(f"🔍 Fetching AI news ({today.strftime('%A, %B %d %Y')})…\n")

    sections = []
    total_items = 0

    for source in SOURCES:
        print(f"  📡 {source['name']}…")
        entries = fetch_feed(source)

        # Filter recent entries
        recent = [e for e in entries if entry_date(e) >= cutoff]
        if not recent:
            # Fallback: just take latest even if older
            recent = entries[:1]

        items = recent[:MAX_ITEMS_PER_SOURCE]
        if not items:
            continue

        lines = [f"📰 {source['name']}"]
        for item in items:
            title = getattr(item, "title", "No title").strip()
            link  = getattr(item, "link", "")
            summary = truncate(getattr(item, "summary", ""), 100)
            lines.append(f"• {title}")
            lines.append(f"  🔗 {link}")
            if summary:
                lines.append(f"  {summary}")
        sections.append("\n".join(lines))
        total_items += len(items)

    header = (
        f"🤖 Daily AI Feed — {today.strftime('%A, %b %d')}\n"
        f"Top {total_items} stories from {len(sections)} sources\n"
        f"{'─' * 30}"
    )

    body = f"\n\n{header}\n\n" + "\n\n".join(sections)

    print(f"\n{'='*50}")
    print(body)
    print(f"{'='*50}\n")

    if args.print:
        print("(--print mode: not sending to Telegram)")
        return

    creds = load_credentials()
    tg = creds.get("telegram", {})
    token   = tg.get("bot_token") or tg.get("token") or os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = tg.get("ai_news_chat_id") or tg.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("❌ Telegram credentials not found. Run with --print or set credentials.", file=sys.stderr)
        sys.exit(1)

    # Telegram has a 4096 char limit — split if needed
    MAX_MSG = 4000
    if len(body) <= MAX_MSG:
        send_telegram(body, token, chat_id)
    else:
        # Send header + each section as separate messages
        send_telegram(f"\n{header}", token, chat_id)
        chunk = ""
        for section in sections:
            if len(chunk) + len(section) + 2 > MAX_MSG:
                send_telegram(chunk.strip(), token, chat_id)
                chunk = section + "\n\n"
            else:
                chunk += section + "\n\n"
        if chunk.strip():
            send_telegram(chunk.strip(), token, chat_id)

    print("Done!")

if __name__ == "__main__":
    main()
