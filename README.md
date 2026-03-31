# 🤖 AI News Bot

Fetches top AI news from 10 curated RSS sources and sends a daily digest to a Telegram group.

## Sources

- MIT Technology Review – AI
- The Verge – AI
- Wired – AI
- VentureBeat – AI
- TechCrunch – AI
- Google AI Blog
- OpenAI Blog
- Hugging Face Blog
- Import AI (Jack Clark)
- AI News (artificialintelligence-news.com)

## Usage

```bash
pip install feedparser
python3 daily_ai_feed.py          # send to Telegram
python3 daily_ai_feed.py --print  # dry run, print only
```

## Config

Credentials are loaded from `../private/credentials.json` (never committed):

```json
{
  "telegram": {
    "bot_token": "YOUR_BOT_TOKEN",
    "ai_news_chat_id": "YOUR_CHAT_ID"
  }
}
```

## License

MIT
