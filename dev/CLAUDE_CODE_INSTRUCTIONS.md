# Claude Code Implementation Guide

## Quick Start with Claude Code

### Step 1: Open Claude Code in your terminal

```bash
cd /path/to/your/projects
claude
```

### Step 2: Initialize the project

Copy and paste this prompt to Claude Code:

---

**PROMPT TO GIVE CLAUDE CODE:**

```
I want to build a Telegram moderation bot with a dynamic knowledge/FAQ system. I have a complete project specification. Please:

1. Read the PROJECT_SPEC.md file I'm about to share
2. Create the complete project structure
3. Implement all the files according to the spec
4. Set up the configuration and Docker files

The bot should:
- Moderate content (detect spam, scams, hate speech, auto-delete or flag for review)
- Answer questions by routing to appropriate knowledge bases (FAQ, Rules, Product Info, etc.)
- Be fully dynamic - support adding new knowledge bases without code changes
- Use Claude API for LLM operations

Tech stack: Python 3.11+, python-telegram-bot v21+, Anthropic Claude API, PostgreSQL, SQLAlchemy async

Here's the full specification:

[PASTE THE ENTIRE CONTENTS OF PROJECT_SPEC.md HERE]

Please implement this project step by step, starting with the project structure and core files.
```

---

### Step 3: Alternative - Reference the file directly

If you have the PROJECT_SPEC.md in your current directory:

```
Read PROJECT_SPEC.md and implement the complete Telegram moderation bot project according to the specification. Create all files and directories as specified.
```

---

## Recommended Implementation Order

Ask Claude Code to implement in this order for best results:

### Phase 1: Project Setup
```
Create the project structure and these files:
1. requirements.txt
2. .env.example
3. docker-compose.yml
4. Dockerfile
5. src/__init__.py
6. src/config.py
```

### Phase 2: Models
```
Create the model files:
1. src/models/__init__.py
2. src/models/enums.py
3. src/models/schemas.py
4. src/models/database.py
```

### Phase 3: LLM Services
```
Create the LLM service files:
1. src/services/__init__.py
2. src/services/llm/__init__.py
3. src/services/llm/prompts.py
4. src/services/llm/client.py
5. src/services/llm/classifier.py
```

### Phase 4: Knowledge System
```
Create the knowledge service files:
1. src/services/knowledge/__init__.py
2. src/services/knowledge/registry.py
3. src/services/knowledge/router.py
4. src/services/knowledge/retriever.py
5. src/services/knowledge/generator.py
6. src/utils/embeddings.py
```

### Phase 5: Moderation System
```
Create the moderation service files:
1. src/services/moderation/__init__.py
2. src/services/moderation/service.py
```

### Phase 6: Bot Handlers
```
Create the bot handler files:
1. src/bot/__init__.py
2. src/bot/app.py
3. src/bot/handlers/__init__.py
4. src/bot/handlers/message_handler.py
5. src/bot/handlers/command_handler.py
6. src/bot/handlers/callback_handler.py
```

### Phase 7: Main Entry & Scripts
```
Create the main entry point and scripts:
1. src/main.py
2. scripts/seed_knowledge.py
```

---

## Testing Commands for Claude Code

After implementation, ask Claude Code to:

```
1. Verify all imports work: python -c "from src.main import main"
2. Run the seed script: python -m scripts.seed_knowledge
3. Start the bot: python -m src.main
```

---

## Common Issues & Fixes

### Issue: Circular imports
**Ask Claude Code:**
```
Check for circular imports in the project and fix them by using lazy imports or restructuring
```

### Issue: Missing __init__.py files
**Ask Claude Code:**
```
Ensure all packages have proper __init__.py files with appropriate exports
```

### Issue: Async/sync mixing
**Ask Claude Code:**
```
The Anthropic client is sync but we're using async handlers. Either:
1. Use asyncio.to_thread() for LLM calls
2. Or use the entire codebase synchronously
Please fix the async/sync mixing throughout the project.
```

---

## Customization Prompts

### Add a new knowledge base type:
```
Add a new knowledge base type called "TUTORIALS" that handles how-to questions. Update the enums, add example items in the seed script, and ensure the router knows about it.
```

### Add database persistence:
```
Implement database persistence using SQLAlchemy async. Add:
1. Database connection setup in src/database/connection.py
2. Repository pattern in src/database/repositories/
3. Update services to use database instead of in-memory storage
4. Add Alembic migrations
```

### Add Redis caching:
```
Add Redis caching for:
1. Whitelist lookups
2. Keyword lists
3. Knowledge base metadata
Use the redis-py async client.
```

### Add webhook support (for production):
```
Convert from polling to webhook mode for production deployment. Add:
1. Webhook setup in src/bot/app.py
2. FastAPI/Starlette server for receiving webhooks
3. Health check endpoint
```

---

## Environment Setup Checklist

Before running, ensure you have:

- [ ] Telegram bot token from @BotFather
- [ ] Anthropic API key
- [ ] OpenAI API key (for embeddings) OR set USE_LOCAL_EMBEDDINGS=true
- [ ] Admin chat IDs (your Telegram user ID)
- [ ] Monitored chat IDs (the groups/channels to moderate)

Get your Telegram user ID by messaging @userinfobot on Telegram.

---

## Full Single Prompt (For Advanced Users)

If you want Claude Code to do everything at once:

```
Implement a complete Telegram moderation bot with the following features:

MODERATION:
- Classify message intent using Claude API (spam, scam, hate_speech, harassment, etc.)
- Whitelist system for trusted users
- Keyword scanner for flagged words
- Auto-delete for severe violations (spam, scam, adult_content)
- Admin review queue for borderline cases (hate_speech, harassment, misinformation)
- Admin approval/rejection via inline buttons

KNOWLEDGE SYSTEM:
- Dynamic knowledge base registry (FAQ, Rules, Product Info, etc. - extensible)
- LLM-powered routing to determine which KB answers a question
- Semantic search within selected KBs (using embeddings)
- RAG response generation using retrieved context
- Source attribution in responses

MESSAGE FLOW:
1. Classify message type (question/statement/greeting/command)
2. If question -> route to knowledge system -> generate answer
3. If statement -> run moderation pipeline -> delete/review/allow
4. If greeting -> simple response
5. If command -> handle admin commands

ADMIN COMMANDS:
/whitelist <user_id> - Add to whitelist
/unwhitelist <user_id> - Remove from whitelist  
/addkeyword <word> - Add flagged keyword
/addkb <slug> <name> <description> - Create knowledge base
/addfaq <kb_slug> <question> | <answer> - Add FAQ item
/stats - Show statistics

TECH STACK:
- Python 3.11+
- python-telegram-bot v21+ (async)
- Anthropic Claude API
- OpenAI embeddings (or local fallback)
- Pydantic for schemas
- SQLAlchemy for database models

Create the complete project with proper structure, all services, handlers, and a seed script for initial data.
```

---

## Support

If you encounter issues:

1. Check that all environment variables are set correctly
2. Ensure your bot has admin permissions in monitored chats
3. Verify API keys are valid
4. Check logs for specific error messages

For Claude API issues, check: https://docs.anthropic.com/
For python-telegram-bot issues, check: https://python-telegram-bot.org/
