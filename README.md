# Telegram Content Moderation Bot with Dynamic Knowledge System

An intelligent Telegram bot that combines content moderation with a dynamic knowledge/FAQ system powered by Claude AI.

## Features

### 🛡️ Content Moderation
- **LLM-powered Intent Classification**: Automatically detects spam, scams, hate speech, harassment, and more
- **Keyword Scanning**: Configurable keyword-based filtering
- **Smart Actions**: Auto-delete serious violations, flag questionable content for admin review
- **Whitelist System**: Trusted users bypass moderation
- **Admin Controls**: Commands to manage keywords, whitelist, and review flagged content

### 💡 Knowledge System
- **Dynamic Knowledge Bases**: Create multiple knowledge bases (FAQ, Rules, Product Info, etc.)
- **Intelligent Routing**: LLM determines which knowledge base(s) can answer a question
- **RAG (Retrieval-Augmented Generation)**: Semantic search + LLM generation for accurate answers
- **Easy Content Management**: Add FAQs and knowledge items via commands

### 🤖 Message Processing
- **Message Classification**: Automatically identifies questions, statements, commands, greetings
- **Context-Aware Responses**: Different handling based on message type
- **Multi-stage Pipeline**: Classification → Moderation → Knowledge Lookup → Response

## Tech Stack

- **Python 3.12+**
- **uv** - Fast Python package installer and resolver
- **python-telegram-bot v21+** - Telegram Bot API
- **Anthropic Claude API** - LLM for classification, routing, and generation
- **OpenAI Embeddings** - Semantic search (with local fallback)
- **PostgreSQL + SQLAlchemy** - Database (optional for production)
- **Redis** - Caching (optional for production)
- **Pydantic** - Configuration and data validation

## Quick Start

### 1. Create a Telegram Bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow instructions
3. Save your bot token

### 2. Get API Keys

- **Anthropic API Key**: Get from [Anthropic Console](https://console.anthropic.com/)
- **OpenAI API Key** (optional): Get from [OpenAI Platform](https://platform.openai.com/)
  - Or set `USE_LOCAL_EMBEDDINGS=true` to use local embeddings

### 3. Set Up Environment

```bash
# Clone or create project directory
cd telegram-mod-bot

# Copy environment template
cp .env.example .env

# Edit .env with your values
nano .env
```

Required environment variables:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
ADMIN_CHAT_IDS=[your_telegram_user_id]
MONITORED_CHAT_IDS=[-1001234567890]  # Your group chat ID
ANTHROPIC_API_KEY=your_anthropic_key_here
```

To find your Telegram user ID:
- Message [@userinfobot](https://t.me/userinfobot) to get your ID
- Add bot to group, send message, check logs for chat ID

### 4. Set Up Development Environment with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. Follow these steps:

#### Install uv (if not already installed)

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**

```bash
pip install uv
```

Verify installation:

```bash
uv --version
```

#### Create and Activate Virtual Environment

```bash
# Create a virtual environment with Python 3.12+
uv venv

# Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows (Command Prompt):
# .venv\Scripts\activate

# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

#### Install Dependencies

```bash
# Install all dependencies from pyproject.toml
uv pip install -e .
```

This installs all packages specified in `pyproject.toml` into your virtual environment.

### 5. Seed Knowledge Base (Optional)

```bash
python -m scripts.seed_knowledge
```

This creates default knowledge bases with sample content.

### 6. Run the Bot

```bash
python -m src.main
```

## Admin Commands

Once the bot is running, admins can use these commands:

### User Management
- `/whitelist <user_id>` - Add user to whitelist (bypass moderation)
- `/unwhitelist <user_id>` - Remove user from whitelist

### Content Moderation
- `/addkeyword <word>` - Add flagged keyword for detection

### Knowledge Base Management
- `/addkb <slug> <name> <description>` - Create new knowledge base
- `/addfaq <kb_slug> <question> | <answer>` - Add FAQ item
- `/stats` - View statistics

### Examples

```
/addkb support "Technical Support" "Help with technical issues"
/addfaq support How do I reset my password? | Click Settings > Security > Reset Password
/addkeyword scam
/whitelist 123456789
```

## Docker Deployment

For production deployment with PostgreSQL and Redis:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop services
docker-compose down
```

## Configuration

### Moderation Settings

Edit `src/config.py` to customize moderation behavior:

```python
# Auto-delete these intents
auto_delete_intents = {
    "spam", "scam", "adult_content"
}

# Send these for admin review
admin_review_intents = {
    "hate_speech", "harassment", "misinformation"
}

# Default flagged keywords
flagged_keywords = {
    "crypto", "investment", "guaranteed returns", "click here",
    "free money", "dm me", "limited time"
}
```

### Knowledge System Settings

```python
# Number of relevant items to retrieve
knowledge_retrieval_top_k = 3

# Minimum similarity threshold (0.0 - 1.0)
knowledge_similarity_threshold = 0.7
```

## Architecture

### Message Flow

```
Incoming Message
    ↓
Message Classification (LLM)
    ↓
├─→ Question → Knowledge Routing → RAG Generation → Response
├─→ Statement → Moderation → Auto-delete / Review / Allow
├─→ Greeting → Simple Response
└─→ Command → Handler
```

### Knowledge System Flow

```
User Question
    ↓
Knowledge Router (LLM) - "Which KB can answer this?"
    ↓
Knowledge Retriever - Semantic search in selected KBs
    ↓
Knowledge Generator (LLM) - Generate answer from context
    ↓
Response with sources
```

### Moderation Flow

```
Message Content
    ↓
Keyword Scanner → Flagged keywords
    ↓
Intent Classifier (LLM) → Intent + Confidence
    ↓
Action Decision
    ↓
├─→ Auto-delete (spam, scam, NSFW)
├─→ Admin Review (hate speech, harassment)
└─→ Allow (safe content)
```

## Project Structure

### Directory Tree

```
telegram-mod-bot/
├── src/
│   ├── bot/              # Telegram bot handlers
│   ├── services/         # LLM, Knowledge, Moderation services
│   ├── models/           # Data models and schemas
│   ├── database/         # Database models and repos
│   ├── utils/            # Embeddings and helpers
│   ├── config.py         # Configuration
│   └── main.py           # Entry point
├── scripts/              # Utilities
├── tests/                # Tests
├── .env.example          # Environment template
├── pyproject.toml        # Project dependencies and metadata
├── docker-compose.yml    # Docker setup
└── README.md            # This file
```

### Core Components

#### Configuration & Models

- [src/config.py](src/config.py) - Pydantic settings with environment variables
- [src/models/enums.py](src/models/enums.py) - Enumerations (MessageType, ModerationIntent, etc.)
- [src/models/schemas.py](src/models/schemas.py) - Pydantic schemas for data validation
- [src/models/database.py](src/models/database.py) - SQLAlchemy database models
- [src/core/interfaces.py](src/core/interfaces.py) - Abstract base classes
- [src/core/exceptions.py](src/core/exceptions.py) - Custom exception classes

### 🤖 LLM Services

- [src/services/llm/client.py](src/services/llm/client.py) - Claude API wrapper with JSON parsing
- [src/services/llm/prompts.py](src/services/llm/prompts.py) - All prompt templates for classification, routing, and generation
- [src/services/llm/classifier.py](src/services/llm/classifier.py) - Message type & moderation intent classification

### 💡 Knowledge System

- [src/services/knowledge/registry.py](src/services/knowledge/registry.py) - Central KB registry with in-memory storage
- [src/services/knowledge/router.py](src/services/knowledge/router.py) - LLM-powered routing to select relevant KBs
- [src/services/knowledge/retriever.py](src/services/knowledge/retriever.py) - Semantic search using embeddings
- [src/services/knowledge/generator.py](src/services/knowledge/generator.py) - RAG answer generation with sources

### 🛡️ Moderation

- [src/services/moderation/service.py](src/services/moderation/service.py) - Complete content moderation pipeline with keyword scanning and LLM classification

### Bot Application

- [src/bot/app.py](src/bot/app.py) - Bot setup and configuration with all handlers
- [src/bot/handlers/message_handler.py](src/bot/handlers/message_handler.py) - Main message processing (classification → knowledge lookup → moderation)
- [src/bot/handlers/command_handler.py](src/bot/handlers/command_handler.py) - Admin commands (/whitelist, /addkb, /addfaq, etc.)
- [src/bot/handlers/callback_handler.py](src/bot/handlers/callback_handler.py) - Inline button callbacks for admin review

### Utilities & Infrastructure

- [src/utils/embeddings.py](src/utils/embeddings.py) - OpenAI embeddings with local fallback
- [src/utils/helpers.py](src/utils/helpers.py) - General utility functions
- [src/main.py](src/main.py) - Main entry point that initializes and starts the bot

### Scripts & Configuration Files

- [scripts/seed_knowledge.py](scripts/seed_knowledge.py) - Seed default knowledge bases with sample FAQ, rules, and product info
- [pyproject.toml](pyproject.toml) - Project metadata and Python dependencies (telegram-bot, anthropic, openai, pydantic, etc.)
- [docker-compose.yml](docker-compose.yml) - Docker orchestration with PostgreSQL and Redis
- [Dockerfile](Dockerfile) - Container definition for the bot
- [.env.example](.env.example) - Environment variables template
- [.gitignore](.gitignore) - Git ignore patterns

### Implementation Details

**Message Processing Pipeline:**

1. **Classification** - Determines message type (question/statement/greeting/command) using quick pattern matching + LLM
2. **Knowledge Lookup** - For questions: routing → semantic retrieval → RAG generation
3. **Content Moderation** - For statements: keyword scanning → LLM intent classification → action (allow/delete/review)
4. **Response** - Bot responds, deletes, or flags for admin review based on classification

**Admin Commands Implemented:**

- `/whitelist <user_id>`, `/unwhitelist <user_id>` - User whitelist management
- `/addkeyword <word>` - Add flagged keywords dynamically
- `/addkb <slug> <name> <description>` - Create knowledge bases without code changes
- `/addfaq <kb_slug> <question> | <answer>` - Add FAQ items on-the-fly
- `/stats` - View statistics (whitelisted users, keywords, KB counts)

**Moderation System Features:**

- Whitelist bypass for trusted users
- Keyword scanning with regex patterns
- LLM intent classification with confidence scores
- Configurable actions: auto-delete (spam, scams, NSFW), admin review (hate speech, harassment), or allow
- Admin review workflow with approve/delete/whitelist buttons

**Knowledge System Features:**

- Dynamic KB registry (add/remove knowledge bases at runtime)
- LLM-powered routing (determines which KBs can answer each question)
- Semantic search with OpenAI embeddings or local fallback
- RAG answer generation using retrieved context
- Source attribution (shows which KBs contributed to answer)

## How It Works

### 1. Message Classification

The bot uses Claude to classify every incoming message:
- **Question**: Routes to knowledge system
- **Statement**: Runs through moderation
- **Greeting**: Simple friendly response
- **Command**: Handled by command system

### 2. Content Moderation

Two-stage approach:
1. **Keyword Scanning**: Fast pattern matching for flagged terms
2. **LLM Classification**: Contextual understanding of intent

Actions based on severity:
- **Auto-delete**: Immediate removal + admin notification
- **Review**: Send to admins with approve/delete buttons
- **Allow**: Message passes moderation

### 3. Knowledge System

Three-component RAG pipeline:
1. **Router**: LLM decides which knowledge bases are relevant
2. **Retriever**: Semantic search finds relevant content
3. **Generator**: LLM synthesizes answer from retrieved context

### 4. Dynamic Management

Admins can:
- Add/remove knowledge bases without code changes
- Add FAQ items via commands
- Update keywords and whitelist in real-time
- Review and decide on flagged content

## Troubleshooting

### Bot doesn't respond in group

1. Make sure bot is added to the group as admin
2. Check that group chat ID is in `MONITORED_CHAT_IDS`
3. Verify bot has permission to read messages

### Moderation not working

1. Check `ADMIN_CHAT_IDS` is set correctly
2. Verify Anthropic API key is valid
3. Check logs for errors

### Knowledge system not answering

1. Run seed script: `python -m scripts.seed_knowledge`
2. Verify knowledge bases have content: `/stats`
3. Check question matches KB routing hints

### Rate limiting errors

If you hit API rate limits:
- Reduce message volume
- Add caching layer (Redis)
- Consider upgrading API tier

## Development

### Adding New Dependencies

To add a new package to the project:

```bash
# Add a new dependency
uv pip install <package-name>

# Update pyproject.toml manually to include it in dependencies
```

Or edit `pyproject.toml` directly and reinstall:

```bash
uv pip install -e .
```

### Running Tests

```bash
# Install test dependencies if needed
uv pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Adding a New Knowledge Base

```python
from src.services.knowledge.registry import knowledge_registry
from src.models.schemas import KnowledgeBase
from src.models.enums import KnowledgeBaseType

kb = KnowledgeBase(
    slug="tutorials",
    name="Tutorials",
    description="Step-by-step guides",
    type=KnowledgeBaseType.TUTORIALS,
    routing_hints=["how to", "tutorial", "guide", "step by step"],
    example_questions=["How do I...?"],
    priority=7
)
knowledge_registry.register(kb)
```

### Adding New Moderation Intents

1. Add to `ModerationIntent` enum in `src/models/enums.py`
2. Update `MODERATION_CLASSIFIER_PROMPT` in `src/services/llm/prompts.py`
3. Configure action in `src/config.py`

## Project Origin

This project was developed using AI-assisted development with Claude Code. The original development specifications, instructions, and planning documents used to create this bot are preserved in the [dev/](dev/) folder for reference and transparency.

### Development Files

The [dev/](dev/) folder contains:
- **CLAUDE_CODE_INSTRUCTIONS.md** - Original instructions and guidelines for the AI-assisted development process
- **PROJECT_SPEC.md** - Complete project specifications, architecture decisions, and implementation details

These files provide insight into the project's design philosophy, architectural decisions, and development methodology. They can be useful for understanding the rationale behind implementation choices or for extending the project with new features.

## License

MIT License - See [LICENSE](LICENSE) file for details

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review logs for errors

## Roadmap

- [ ] Persistent database integration
- [ ] Multi-language support
- [ ] Analytics dashboard
- [ ] Custom LLM model fine-tuning
- [ ] Image/video content moderation
- [ ] Automated testing suite
- [ ] Web admin panel

## Credits

Built with:
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [Anthropic Claude API](https://www.anthropic.com/)
- [OpenAI Embeddings](https://platform.openai.com/)
