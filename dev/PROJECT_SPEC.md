# Telegram Content Moderation Bot with Dynamic Knowledge System

## Project Overview

Build a Telegram bot that combines intelligent content moderation with a dynamic knowledge/FAQ system. The bot should:

1. **Moderate content** - Detect spam, scams, hate speech, etc. and auto-delete or flag for admin review
2. **Answer questions** - Route user questions to appropriate knowledge bases and generate answers using RAG
3. **Be fully dynamic** - Support adding new knowledge bases, keywords, and rules without code changes

## Tech Stack

- **Language**: Python 3.11+
- **Telegram**: python-telegram-bot v21+
- **LLM**: Anthropic Claude API (claude-sonnet-4-20250514)
- **Database**: PostgreSQL with SQLAlchemy (async)
- **Embeddings**: OpenAI text-embedding-3-small (or sentence-transformers for local)
- **Cache**: Redis (optional, for production)
- **Config**: Pydantic Settings

## Project Structure

```
telegram-mod-bot/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── app.py                      # Bot application setup
│   │   ├── handlers/
│   │   │   ├── __init__.py
│   │   │   ├── message_handler.py      # Main message processing
│   │   │   ├── command_handler.py      # Admin commands
│   │   │   └── callback_handler.py     # Button callbacks
│   │   └── middleware/
│   │       ├── __init__.py
│   │       └── classifier.py           # Message type classifier
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── interfaces.py               # Abstract base classes
│   │   └── exceptions.py               # Custom exceptions
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── client.py               # LLM client wrapper
│   │   │   ├── prompts.py              # All prompt templates
│   │   │   └── classifier.py           # Intent classification
│   │   │
│   │   ├── moderation/
│   │   │   ├── __init__.py
│   │   │   ├── service.py              # Main moderation service
│   │   │   ├── keyword_scanner.py      # Keyword detection
│   │   │   └── actions.py              # Delete/Review actions
│   │   │
│   │   └── knowledge/
│   │       ├── __init__.py
│   │       ├── router.py               # Context routing logic
│   │       ├── registry.py             # Knowledge base registry
│   │       ├── retriever.py            # Semantic search
│   │       └── generator.py            # RAG response generation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enums.py                    # All enumerations
│   │   ├── schemas.py                  # Pydantic models
│   │   └── database.py                 # SQLAlchemy models
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py               # DB connection setup
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── base.py                 # Base repository
│   │       ├── knowledge_repo.py       # Knowledge CRUD
│   │       ├── moderation_repo.py      # Moderation logs
│   │       └── user_repo.py            # User/whitelist
│   │
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py               # Text embedding utilities
│       └── helpers.py                  # General helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_moderation.py
│   ├── test_knowledge.py
│   └── test_routing.py
│
├── scripts/
│   ├── seed_knowledge.py               # Seed initial data
│   └── migrate.py                      # Run migrations
│
├── alembic/                            # Database migrations
│   ├── env.py
│   └── versions/
│
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── alembic.ini
└── README.md
```

---

## Configuration

### `.env.example`

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here
ADMIN_CHAT_IDS=[123456789,987654321]
MONITORED_CHAT_IDS=[-1001234567890]

# LLM
ANTHROPIC_API_KEY=your_anthropic_key_here
LLM_MODEL=claude-sonnet-4-20250514

# Embeddings (choose one)
OPENAI_API_KEY=your_openai_key_here
# Or use local embeddings
USE_LOCAL_EMBEDDINGS=false

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/modbot
REDIS_URL=redis://localhost:6379

# Moderation
WHITELISTED_USER_IDS=[]
```

### `src/config.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Set, List
from src.models.enums import ModerationIntent


class Settings(BaseSettings):
    # Telegram
    telegram_bot_token: str
    admin_chat_ids: List[int] = Field(default_factory=list)
    monitored_chat_ids: List[int] = Field(default_factory=list)
    
    # LLM
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-20250514"
    
    # Embeddings
    openai_api_key: str = ""
    use_local_embeddings: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/modbot"
    redis_url: str = "redis://localhost:6379"
    
    # Moderation
    whitelisted_user_ids: Set[int] = Field(default_factory=set)
    
    flagged_keywords: Set[str] = Field(default_factory=lambda: {
        "crypto", "investment", "guaranteed returns", "click here",
        "free money", "dm me", "limited time", "act now", "wire transfer"
    })
    
    auto_delete_intents: Set[str] = Field(default_factory=lambda: {
        ModerationIntent.SPAM.value,
        ModerationIntent.SCAM.value,
        ModerationIntent.ADULT_CONTENT.value
    })
    
    admin_review_intents: Set[str] = Field(default_factory=lambda: {
        ModerationIntent.HATE_SPEECH.value,
        ModerationIntent.HARASSMENT.value,
        ModerationIntent.MISINFORMATION.value
    })
    
    # Knowledge System
    knowledge_retrieval_top_k: int = 3
    knowledge_similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

---

## Models

### `src/models/enums.py`

```python
from enum import Enum


class MessageType(str, Enum):
    """Classification of incoming message type."""
    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class ModerationIntent(str, Enum):
    """Intents for content moderation."""
    SPAM = "spam"
    SCAM = "scam"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SELF_PROMOTION = "self_promotion"
    OFF_TOPIC = "off_topic"
    MISINFORMATION = "misinformation"
    ADULT_CONTENT = "adult_content"
    VIOLENCE = "violence"
    SAFE = "safe"
    UNCLEAR = "unclear"


class ModerationAction(str, Enum):
    """Actions to take on moderated content."""
    ALLOW = "allow"
    DELETE = "delete"
    REVIEW = "review"


class KnowledgeBaseType(str, Enum):
    """Types of knowledge bases."""
    FAQ = "faq"
    RULES = "rules"
    PRODUCT_INFO = "product_info"
    POLICIES = "policies"
    TUTORIALS = "tutorials"
    ANNOUNCEMENTS = "announcements"
    CUSTOM = "custom"
```

### `src/models/schemas.py`

```python
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.models.enums import (
    MessageType, ModerationIntent, ModerationAction, KnowledgeBaseType
)


# ============== Message Classification ==============

class MessageClassification(BaseModel):
    message_type: MessageType
    confidence: int = Field(ge=0, le=100)
    detected_language: str = "en"
    requires_moderation: bool = True
    requires_knowledge_lookup: bool = False


# ============== Moderation ==============

class ModerationResult(BaseModel):
    intent: ModerationIntent
    confidence: int = Field(ge=0, le=100)
    explanation: str
    keywords_found: List[str] = []
    action: ModerationAction = ModerationAction.ALLOW


# ============== Knowledge System ==============

class KnowledgeBase(BaseModel):
    id: Optional[int] = None
    name: str
    slug: str
    description: str
    type: KnowledgeBaseType
    routing_hints: List[str] = []
    example_questions: List[str] = []
    priority: int = 0
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class KnowledgeItem(BaseModel):
    id: Optional[int] = None
    knowledge_base_id: int
    question: Optional[str] = None
    title: Optional[str] = None
    content: str
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class RoutingDecision(BaseModel):
    should_answer: bool
    selected_knowledge_bases: List[str] = []
    confidence: int = Field(ge=0, le=100)
    reasoning: str
    fallback_response: Optional[str] = None


class RetrievedContext(BaseModel):
    knowledge_base_slug: str
    knowledge_base_name: str
    items: List[KnowledgeItem]
    relevance_scores: List[float] = []


class KnowledgeResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]] = []
    confidence: int = Field(ge=0, le=100)
    knowledge_bases_used: List[str] = []


# ============== Processing Result ==============

class ProcessingResult(BaseModel):
    classification: MessageClassification
    moderation: Optional[ModerationResult] = None
    knowledge_response: Optional[KnowledgeResponse] = None
    should_delete: bool = False
    should_respond: bool = False
    response_text: Optional[str] = None
```

### `src/models/database.py`

```python
from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, DateTime,
    Boolean, JSON, Float, ForeignKey, Text, Index
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class WhitelistedUser(Base):
    __tablename__ = "whitelisted_users"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, unique=True, index=True)
    username = Column(String, nullable=True)
    added_by = Column(BigInteger)
    added_at = Column(DateTime, default=datetime.utcnow)
    reason = Column(String, nullable=True)


class FlaggedKeyword(Base):
    __tablename__ = "flagged_keywords"
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True, index=True)
    severity = Column(String, default="medium")
    added_at = Column(DateTime, default=datetime.utcnow)


class ModerationLog(Base):
    __tablename__ = "moderation_logs"
    
    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger)
    chat_id = Column(BigInteger)
    user_id = Column(BigInteger)
    content_preview = Column(String(500))
    detected_intent = Column(String)
    action_taken = Column(String)
    confidence_score = Column(Integer)
    keywords_found = Column(JSON, nullable=True)
    admin_decision = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('idx_moderation_chat_user', 'chat_id', 'user_id'),
    )


class PendingReview(Base):
    __tablename__ = "pending_reviews"
    
    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger)
    chat_id = Column(BigInteger)
    user_id = Column(BigInteger)
    content = Column(Text)
    detected_intent = Column(String)
    confidence_score = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_resolved = Column(Boolean, default=False)


class KnowledgeBaseDB(Base):
    __tablename__ = "knowledge_bases"
    
    id = Column(Integer, primary_key=True)
    slug = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    type = Column(String, nullable=False)
    routing_hints = Column(JSON, default=list)
    example_questions = Column(JSON, default=list)
    priority = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    items = relationship("KnowledgeItemDB", back_populates="knowledge_base", cascade="all, delete-orphan")


class KnowledgeItemDB(Base):
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), index=True)
    question = Column(Text, nullable=True)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    tags = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    embedding = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    knowledge_base = relationship("KnowledgeBaseDB", back_populates="items")
    
    __table_args__ = (
        Index('idx_knowledge_item_kb', 'knowledge_base_id', 'is_active'),
    )
```

---

## LLM Services

### `src/services/llm/prompts.py`

```python
"""All prompt templates for LLM operations."""

MESSAGE_CLASSIFIER_PROMPT = """You are a message classifier. Analyze the incoming message and classify its type.

Classification types:
- question: User is asking something, seeking information, or needs help
- statement: Regular message, opinion, or content (not asking for information)
- command: Bot command (starts with /)
- greeting: Simple greeting like "hi", "hello", "hey"
- unknown: Cannot determine

Respond with JSON only:
{
    "message_type": "<type>",
    "confidence": <0-100>,
    "detected_language": "<language_code>",
    "requires_moderation": <true if statement/unknown, false otherwise>,
    "requires_knowledge_lookup": <true if question, false otherwise>
}"""


KNOWLEDGE_ROUTER_PROMPT = """You are a knowledge routing assistant. Given a user's question and available knowledge bases, determine which knowledge base(s) can best answer the question.

Available Knowledge Bases:
{knowledge_bases}

User Question: "{question}"

Analyze:
1. Does this question match any knowledge base's domain?
2. Which knowledge base(s) are most relevant?
3. If no good match, provide a helpful fallback response.

Respond with JSON only:
{{
    "should_answer": <true if any KB can help>,
    "selected_knowledge_bases": ["<slug1>", "<slug2>"],
    "confidence": <0-100>,
    "reasoning": "<brief explanation>",
    "fallback_response": "<response if no KB matches, null otherwise>"
}}"""


RAG_GENERATION_PROMPT = """You are a helpful assistant answering questions using the provided context.

Context from knowledge base(s):
{context}

User Question: "{question}"

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't fully answer the question, acknowledge what you can answer and what's missing
3. Be concise but complete
4. Be friendly and helpful

Provide your answer:"""


MODERATION_CLASSIFIER_PROMPT = """You are a content moderation assistant. Analyze the message and classify its intent.

Intent classifications:
- spam: Unsolicited promotional content, repetitive messages
- scam: Fraudulent schemes, phishing, too-good-to-be-true offers
- hate_speech: Content targeting protected groups
- harassment: Personal attacks, bullying, intimidation
- self_promotion: Excessive self-advertising (mild violation)
- off_topic: Unrelated to channel purpose
- misinformation: False or misleading information
- adult_content: NSFW material
- violence: Threats or glorification of violence
- safe: Normal, acceptable content
- unclear: Cannot determine with confidence

Respond with JSON only:
{
    "intent": "<intent>",
    "confidence": <0-100>,
    "explanation": "<brief explanation>",
    "keywords_found": ["<keyword1>", "<keyword2>"]
}"""
```

### `src/services/llm/client.py`

```python
"""Unified LLM client wrapper."""
import json
import logging
from typing import Dict, Any
import anthropic

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for LLM API calls."""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
    
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Make a completion request."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            raise
    
    async def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Make a completion request and parse JSON response."""
        response_text = await self.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=0.0
        )
        
        try:
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {response_text}")
            raise ValueError(f"Invalid JSON from LLM: {e}")


llm_client = LLMClient()
```

### `src/services/llm/classifier.py`

```python
"""Message and intent classification using LLM."""
import logging
from src.services.llm.client import llm_client
from src.services.llm.prompts import MESSAGE_CLASSIFIER_PROMPT, MODERATION_CLASSIFIER_PROMPT
from src.models.schemas import MessageClassification, ModerationResult
from src.models.enums import MessageType, ModerationIntent, ModerationAction
from src.config import settings

logger = logging.getLogger(__name__)


class MessageClassifier:
    """Classifies incoming messages."""
    
    async def classify(self, text: str) -> MessageClassification:
        """Classify the message type."""
        
        # Quick checks
        if text.startswith('/'):
            return MessageClassification(
                message_type=MessageType.COMMAND,
                confidence=100,
                requires_moderation=False,
                requires_knowledge_lookup=False
            )
        
        text_lower = text.lower().strip()
        greetings = {'hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon'}
        
        if text_lower in greetings or (len(text_lower) < 20 and any(g in text_lower for g in greetings)):
            return MessageClassification(
                message_type=MessageType.GREETING,
                confidence=90,
                requires_moderation=False,
                requires_knowledge_lookup=False
            )
        
        # Use LLM
        try:
            result = await llm_client.complete_json(
                system_prompt=MESSAGE_CLASSIFIER_PROMPT,
                user_message=f"Classify this message: \"{text}\""
            )
            
            return MessageClassification(
                message_type=MessageType(result["message_type"]),
                confidence=result["confidence"],
                detected_language=result.get("detected_language", "en"),
                requires_moderation=result.get("requires_moderation", True),
                requires_knowledge_lookup=result.get("requires_knowledge_lookup", False)
            )
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return MessageClassification(
                message_type=MessageType.STATEMENT,
                confidence=50,
                requires_moderation=True,
                requires_knowledge_lookup=False
            )


class IntentClassifier:
    """Classifies message intent for moderation."""
    
    async def classify(self, text: str, username: str = None) -> ModerationResult:
        """Classify content intent."""
        try:
            msg = f"Analyze this message for moderation:\n\nMessage: \"{text}\""
            if username:
                msg += f"\nFrom user: @{username}"
            
            result = await llm_client.complete_json(
                system_prompt=MODERATION_CLASSIFIER_PROMPT,
                user_message=msg
            )
            
            intent = ModerationIntent(result["intent"])
            
            # Determine action
            if intent.value in settings.auto_delete_intents:
                action = ModerationAction.DELETE
            elif intent.value in settings.admin_review_intents:
                action = ModerationAction.REVIEW
            else:
                action = ModerationAction.ALLOW
            
            return ModerationResult(
                intent=intent,
                confidence=result["confidence"],
                explanation=result["explanation"],
                keywords_found=result.get("keywords_found", []),
                action=action
            )
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return ModerationResult(
                intent=ModerationIntent.UNCLEAR,
                confidence=0,
                explanation="Classification failed",
                action=ModerationAction.REVIEW
            )


message_classifier = MessageClassifier()
intent_classifier = IntentClassifier()
```

---

## Knowledge Services

### `src/services/knowledge/registry.py`

```python
"""Knowledge Base Registry."""
from typing import Dict, List, Optional
import logging
from src.models.schemas import KnowledgeBase, KnowledgeItem
from src.models.enums import KnowledgeBaseType

logger = logging.getLogger(__name__)


class KnowledgeRegistry:
    """Central registry for all knowledge bases."""
    
    def __init__(self):
        self._knowledge_bases: Dict[str, KnowledgeBase] = {}
        self._items: Dict[str, List[KnowledgeItem]] = {}
    
    def register(self, kb: KnowledgeBase) -> None:
        """Register a knowledge base."""
        self._knowledge_bases[kb.slug] = kb
        if kb.slug not in self._items:
            self._items[kb.slug] = []
        logger.info(f"Registered KB: {kb.name} ({kb.slug})")
    
    def unregister(self, slug: str) -> None:
        """Remove a knowledge base."""
        self._knowledge_bases.pop(slug, None)
        self._items.pop(slug, None)
    
    def get(self, slug: str) -> Optional[KnowledgeBase]:
        """Get KB by slug."""
        return self._knowledge_bases.get(slug)
    
    def get_all_active(self) -> List[KnowledgeBase]:
        """Get all active KBs sorted by priority."""
        return sorted(
            [kb for kb in self._knowledge_bases.values() if kb.is_active],
            key=lambda x: x.priority,
            reverse=True
        )
    
    def add_item(self, kb_slug: str, item: KnowledgeItem) -> None:
        """Add item to a KB."""
        if kb_slug not in self._items:
            self._items[kb_slug] = []
        self._items[kb_slug].append(item)
    
    def add_items(self, kb_slug: str, items: List[KnowledgeItem]) -> None:
        """Add multiple items to a KB."""
        for item in items:
            self.add_item(kb_slug, item)
    
    def get_items(self, kb_slug: str) -> List[KnowledgeItem]:
        """Get all items for a KB."""
        return self._items.get(kb_slug, [])
    
    def get_routing_context(self) -> str:
        """Generate context for router."""
        lines = []
        for kb in self.get_all_active():
            lines.append(f"""
Knowledge Base: {kb.name}
Slug: {kb.slug}
Type: {kb.type.value}
Description: {kb.description}
Example Questions: {', '.join(kb.example_questions[:3]) if kb.example_questions else 'N/A'}
Keywords: {', '.join(kb.routing_hints[:5]) if kb.routing_hints else 'N/A'}
""")
        return "\n---\n".join(lines)


knowledge_registry = KnowledgeRegistry()


def setup_default_knowledge_bases():
    """Register default KBs."""
    knowledge_registry.register(KnowledgeBase(
        slug="faq",
        name="Frequently Asked Questions",
        description="Common questions and answers",
        type=KnowledgeBaseType.FAQ,
        routing_hints=["how do i", "what is", "can i", "help", "question", "how to"],
        example_questions=["How do I get started?", "What are the features?", "How can I contact support?"],
        priority=10
    ))
    
    knowledge_registry.register(KnowledgeBase(
        slug="rules",
        name="Community Rules",
        description="Rules, guidelines, and policies",
        type=KnowledgeBaseType.RULES,
        routing_hints=["rules", "allowed", "banned", "permitted", "guidelines", "policy", "can i post"],
        example_questions=["What are the rules?", "Is promotion allowed?", "What gets you banned?"],
        priority=9
    ))
    
    knowledge_registry.register(KnowledgeBase(
        slug="product",
        name="Product Information",
        description="Product features, pricing, and details",
        type=KnowledgeBaseType.PRODUCT_INFO,
        routing_hints=["price", "cost", "feature", "product", "plan", "subscription", "pricing"],
        example_questions=["How much does it cost?", "What's included?", "Is there a free plan?"],
        priority=8
    ))
```

### `src/services/knowledge/router.py`

```python
"""Knowledge routing service."""
import logging
from src.services.llm.client import llm_client
from src.services.llm.prompts import KNOWLEDGE_ROUTER_PROMPT
from src.services.knowledge.registry import knowledge_registry
from src.models.schemas import RoutingDecision

logger = logging.getLogger(__name__)


class KnowledgeRouter:
    """Routes questions to appropriate KBs."""
    
    async def route(self, question: str) -> RoutingDecision:
        """Determine which KB(s) should handle this question."""
        
        kb_context = knowledge_registry.get_routing_context()
        
        if not kb_context.strip():
            return RoutingDecision(
                should_answer=False,
                selected_knowledge_bases=[],
                confidence=0,
                reasoning="No knowledge bases available",
                fallback_response="I don't have any information sources configured."
            )
        
        try:
            prompt = KNOWLEDGE_ROUTER_PROMPT.format(
                knowledge_bases=kb_context,
                question=question
            )
            
            result = await llm_client.complete_json(
                system_prompt="You are a routing assistant. Respond with JSON only.",
                user_message=prompt
            )
            
            return RoutingDecision(
                should_answer=result["should_answer"],
                selected_knowledge_bases=result.get("selected_knowledge_bases", []),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                fallback_response=result.get("fallback_response")
            )
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return RoutingDecision(
                should_answer=False,
                selected_knowledge_bases=[],
                confidence=0,
                reasoning=str(e),
                fallback_response="I'm having trouble finding information. Please try again."
            )


knowledge_router = KnowledgeRouter()
```

### `src/services/knowledge/retriever.py`

```python
"""Retrieves relevant content from KBs."""
import logging
from typing import List
import numpy as np

from src.services.knowledge.registry import knowledge_registry
from src.models.schemas import KnowledgeItem, RetrievedContext
from src.utils.embeddings import get_embedding, cosine_similarity
from src.config import settings

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Retrieves relevant items using semantic search."""
    
    def __init__(self):
        self.top_k = settings.knowledge_retrieval_top_k
        self.threshold = settings.knowledge_similarity_threshold
    
    async def retrieve(self, question: str, kb_slugs: List[str]) -> List[RetrievedContext]:
        """Retrieve relevant items from specified KBs."""
        results = []
        question_embedding = await get_embedding(question)
        
        for slug in kb_slugs:
            kb = knowledge_registry.get(slug)
            if not kb:
                continue
            
            items = knowledge_registry.get_items(slug)
            if not items:
                continue
            
            scored_items = []
            for item in items:
                if item.embedding:
                    score = cosine_similarity(question_embedding, item.embedding)
                else:
                    score = self._keyword_score(question, item)
                scored_items.append((item, score))
            
            scored_items.sort(key=lambda x: x[1], reverse=True)
            top_items = [(item, score) for item, score in scored_items[:self.top_k] if score >= self.threshold]
            
            if top_items:
                results.append(RetrievedContext(
                    knowledge_base_slug=slug,
                    knowledge_base_name=kb.name,
                    items=[item for item, _ in top_items],
                    relevance_scores=[score for _, score in top_items]
                ))
        
        return results
    
    def _keyword_score(self, question: str, item: KnowledgeItem) -> float:
        """Keyword matching fallback."""
        q_words = set(question.lower().split())
        item_text = f"{item.question or ''} {item.title or ''} {item.content}".lower()
        i_words = set(item_text.split())
        
        if not i_words or not q_words:
            return 0.0
        
        return len(q_words & i_words) / len(q_words)


knowledge_retriever = KnowledgeRetriever()
```

### `src/services/knowledge/generator.py`

```python
"""RAG response generation."""
import logging
from typing import List
from src.services.llm.client import llm_client
from src.services.llm.prompts import RAG_GENERATION_PROMPT
from src.models.schemas import RetrievedContext, KnowledgeResponse

logger = logging.getLogger(__name__)


class KnowledgeGenerator:
    """Generates answers using RAG."""
    
    async def generate(self, question: str, contexts: List[RetrievedContext]) -> KnowledgeResponse:
        """Generate answer from contexts."""
        
        if not contexts:
            return KnowledgeResponse(
                answer="I couldn't find relevant information to answer your question.",
                sources=[],
                confidence=0,
                knowledge_bases_used=[]
            )
        
        context_text = self._format_contexts(contexts)
        
        try:
            prompt = RAG_GENERATION_PROMPT.format(context=context_text, question=question)
            
            answer = await llm_client.complete(
                system_prompt="Answer based only on the provided context.",
                user_message=prompt,
                max_tokens=500
            )
            
            sources = []
            kb_used = []
            for ctx in contexts:
                kb_used.append(ctx.knowledge_base_slug)
                for item in ctx.items:
                    sources.append({
                        "kb": ctx.knowledge_base_name,
                        "title": item.title or item.question or "Item"
                    })
            
            return KnowledgeResponse(
                answer=answer,
                sources=sources[:5],
                confidence=self._calc_confidence(contexts),
                knowledge_bases_used=list(set(kb_used))
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return KnowledgeResponse(
                answer="I encountered an error. Please try again.",
                sources=[],
                confidence=0,
                knowledge_bases_used=[]
            )
    
    def _format_contexts(self, contexts: List[RetrievedContext]) -> str:
        parts = []
        for ctx in contexts:
            parts.append(f"=== {ctx.knowledge_base_name} ===")
            for item in ctx.items:
                if item.question:
                    parts.append(f"Q: {item.question}\nA: {item.content}")
                elif item.title:
                    parts.append(f"**{item.title}**\n{item.content}")
                else:
                    parts.append(item.content)
                parts.append("")
        return "\n".join(parts)
    
    def _calc_confidence(self, contexts: List[RetrievedContext]) -> int:
        scores = [s for ctx in contexts for s in ctx.relevance_scores]
        return min(100, int((sum(scores) / len(scores)) * 100)) if scores else 50


knowledge_generator = KnowledgeGenerator()
```

---

## Moderation Service

### `src/services/moderation/service.py`

```python
"""Main moderation service."""
import re
import logging
from typing import Tuple, Optional

from src.config import settings
from src.services.llm.classifier import intent_classifier
from src.models.schemas import ModerationResult
from src.models.enums import ModerationAction

logger = logging.getLogger(__name__)


class ModerationService:
    """Content moderation service."""
    
    def __init__(self):
        self.whitelisted_users = set(settings.whitelisted_user_ids)
        self.flagged_keywords = settings.flagged_keywords
    
    def is_whitelisted(self, user_id: int) -> bool:
        return user_id in self.whitelisted_users
    
    def add_to_whitelist(self, user_id: int) -> None:
        self.whitelisted_users.add(user_id)
    
    def remove_from_whitelist(self, user_id: int) -> None:
        self.whitelisted_users.discard(user_id)
    
    def scan_keywords(self, content: str) -> list[str]:
        """Scan for flagged keywords."""
        content_lower = content.lower()
        found = []
        for keyword in self.flagged_keywords:
            pattern = rf'\b{re.escape(keyword.lower())}\b'
            if re.search(pattern, content_lower):
                found.append(keyword)
        return found
    
    async def moderate_content(
        self,
        content: str,
        user_id: int,
        username: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[ModerationResult, str]:
        """Main moderation pipeline. Returns (result, action)."""
        
        # Whitelist check
        if self.is_whitelisted(user_id):
            return ModerationResult(
                intent="safe",
                confidence=100,
                explanation="User whitelisted",
                action=ModerationAction.ALLOW
            ), ModerationAction.ALLOW.value
        
        # Keyword scan
        keywords_found = self.scan_keywords(content)
        
        # LLM classification
        result = await intent_classifier.classify(content, username)
        result.keywords_found = list(set(result.keywords_found + keywords_found))
        
        # Adjust action if keywords found but LLM uncertain
        if keywords_found and result.confidence < 80 and result.action == ModerationAction.ALLOW:
            result.action = ModerationAction.REVIEW
        
        return result, result.action.value


moderation_service = ModerationService()
```

---

## Embeddings Utility

### `src/utils/embeddings.py`

```python
"""Embedding utilities."""
import logging
from typing import List
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


async def get_embedding(text: str) -> List[float]:
    """Get embedding for text."""
    
    if settings.use_local_embeddings:
        return _get_local_embedding(text)
    
    # Use OpenAI embeddings
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"OpenAI embedding failed, using local: {e}")
        return _get_local_embedding(text)


def _get_local_embedding(text: str) -> List[float]:
    """Simple local embedding fallback."""
    import hashlib
    
    words = text.lower().split()
    dim = settings.embedding_dimension
    vector = [0.0] * dim
    
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % dim] += 1.0
    
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = [v / norm for v in vector]
    
    return vector


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity."""
    v1, v2 = np.array(vec1), np.array(vec2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0.0
```

---

## Bot Application

### `src/bot/app.py`

```python
"""Bot application setup."""
import logging
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters

from src.config import settings

logger = logging.getLogger(__name__)


def create_bot_application() -> Application:
    """Create and configure the bot application."""
    
    app = Application.builder().token(settings.telegram_bot_token).build()
    
    # Import handlers here to avoid circular imports
    from src.bot.handlers.message_handler import handle_message
    from src.bot.handlers.command_handler import (
        start_command, whitelist_command, unwhitelist_command,
        add_keyword_command, add_kb_command, add_faq_command, stats_command
    )
    from src.bot.handlers.callback_handler import handle_callback
    
    # Command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("whitelist", whitelist_command))
    app.add_handler(CommandHandler("unwhitelist", unwhitelist_command))
    app.add_handler(CommandHandler("addkeyword", add_keyword_command))
    app.add_handler(CommandHandler("addkb", add_kb_command))
    app.add_handler(CommandHandler("addfaq", add_faq_command))
    app.add_handler(CommandHandler("stats", stats_command))
    
    # Message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Callback handler
    app.add_handler(CallbackQueryHandler(handle_callback))
    
    # Error handler
    app.add_error_handler(error_handler)
    
    return app


async def error_handler(update, context):
    """Global error handler."""
    logger.error(f"Error: {context.error}", exc_info=context.error)
```

### `src/bot/handlers/message_handler.py`

```python
"""Main message handler."""
import logging
from telegram import Update
from telegram.ext import ContextTypes

from src.config import settings
from src.services.llm.classifier import message_classifier
from src.services.moderation.service import moderation_service
from src.services.knowledge.router import knowledge_router
from src.services.knowledge.retriever import knowledge_retriever
from src.services.knowledge.generator import knowledge_generator
from src.models.enums import MessageType, ModerationAction

logger = logging.getLogger(__name__)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming messages."""
    
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    
    if chat.id not in settings.monitored_chat_ids:
        return
    
    if not message.text:
        return
    
    try:
        # Classify message
        classification = await message_classifier.classify(message.text)
        
        logger.info(f"Classified: {classification.message_type}, conf={classification.confidence}")
        
        # Handle by type
        if classification.message_type == MessageType.COMMAND:
            return
        
        if classification.message_type == MessageType.GREETING:
            await message.reply_text("Hello! 👋 How can I help you today?")
            return
        
        # Handle questions
        if classification.requires_knowledge_lookup:
            response = await _handle_question(message.text)
            if response:
                await message.reply_text(response, parse_mode="Markdown")
        
        # Run moderation
        if classification.requires_moderation:
            await _handle_moderation(message, user, context)
            
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)


async def _handle_question(question: str) -> str | None:
    """Handle question through knowledge system."""
    
    routing = await knowledge_router.route(question)
    
    if not routing.should_answer:
        return routing.fallback_response
    
    contexts = await knowledge_retriever.retrieve(question, routing.selected_knowledge_bases)
    response = await knowledge_generator.generate(question, contexts)
    
    text = response.answer
    if response.knowledge_bases_used:
        text += f"\n\n_📚 Source: {', '.join(response.knowledge_bases_used)}_"
    
    return text


async def _handle_moderation(message, user, context) -> None:
    """Run moderation pipeline."""
    
    result, action = await moderation_service.moderate_content(
        content=message.text,
        user_id=user.id,
        username=user.username
    )
    
    if action == ModerationAction.DELETE.value:
        await _delete_message(message, result, context)
    elif action == ModerationAction.REVIEW.value:
        await _request_review(message, result, context)


async def _delete_message(message, result, context) -> None:
    """Delete and notify."""
    try:
        await message.delete()
        for admin_id in settings.admin_chat_ids:
            try:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=f"🗑️ *Deleted*\nUser: @{message.from_user.username or message.from_user.id}\nIntent: `{result.intent}`\nContent: `{message.text[:100]}...`",
                    parse_mode="Markdown"
                )
            except:
                pass
    except Exception as e:
        logger.error(f"Delete failed: {e}")


async def _request_review(message, result, context) -> None:
    """Send for admin review."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve_{message.chat_id}_{message.message_id}"),
            InlineKeyboardButton("🗑️ Delete", callback_data=f"delete_{message.chat_id}_{message.message_id}")
        ],
        [InlineKeyboardButton("⚪ Whitelist User", callback_data=f"whitelist_{message.from_user.id}")]
    ])
    
    for admin_id in settings.admin_chat_ids:
        try:
            await context.bot.send_message(
                chat_id=admin_id,
                text=f"🔍 *Review*\n\nUser: @{message.from_user.username or message.from_user.id}\nIntent: `{result.intent}` ({result.confidence}%)\n\n```\n{message.text[:400]}\n```",
                parse_mode="Markdown",
                reply_markup=keyboard
            )
        except:
            pass
```

### `src/bot/handlers/command_handler.py`

```python
"""Admin command handlers."""
import logging
from telegram import Update
from telegram.ext import ContextTypes

from src.config import settings
from src.services.moderation.service import moderation_service
from src.services.knowledge.registry import knowledge_registry
from src.models.schemas import KnowledgeBase, KnowledgeItem
from src.models.enums import KnowledgeBaseType

logger = logging.getLogger(__name__)


def is_admin(user_id: int) -> bool:
    return user_id in settings.admin_chat_ids


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start and /help."""
    await update.message.reply_text(
        "🤖 *Moderation Bot*\n\n"
        "*Admin Commands:*\n"
        "/whitelist <user_id> - Add to whitelist\n"
        "/unwhitelist <user_id> - Remove from whitelist\n"
        "/addkeyword <word> - Add flagged keyword\n"
        "/addkb <slug> <name> <description> - Add knowledge base\n"
        "/addfaq <kb_slug> <question> | <answer> - Add FAQ item\n"
        "/stats - View statistics",
        parse_mode="Markdown"
    )


async def whitelist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add user to whitelist."""
    if not is_admin(update.effective_user.id):
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /whitelist <user_id>")
        return
    
    try:
        user_id = int(context.args[0])
        moderation_service.add_to_whitelist(user_id)
        await update.message.reply_text(f"✅ User {user_id} whitelisted")
    except ValueError:
        await update.message.reply_text("Invalid user ID")


async def unwhitelist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove from whitelist."""
    if not is_admin(update.effective_user.id):
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /unwhitelist <user_id>")
        return
    
    try:
        user_id = int(context.args[0])
        moderation_service.remove_from_whitelist(user_id)
        await update.message.reply_text(f"✅ User {user_id} removed from whitelist")
    except ValueError:
        await update.message.reply_text("Invalid user ID")


async def add_keyword_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add flagged keyword."""
    if not is_admin(update.effective_user.id):
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /addkeyword <keyword>")
        return
    
    keyword = " ".join(context.args).lower()
    settings.flagged_keywords.add(keyword)
    await update.message.reply_text(f"✅ Keyword '{keyword}' added")


async def add_kb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add knowledge base."""
    if not is_admin(update.effective_user.id):
        return
    
    if len(context.args) < 3:
        await update.message.reply_text("Usage: /addkb <slug> <name> <description>")
        return
    
    slug = context.args[0].lower()
    name = context.args[1]
    description = " ".join(context.args[2:])
    
    kb = KnowledgeBase(
        slug=slug,
        name=name,
        description=description,
        type=KnowledgeBaseType.CUSTOM,
        priority=5
    )
    knowledge_registry.register(kb)
    await update.message.reply_text(f"✅ Knowledge base '{name}' ({slug}) created")


async def add_faq_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add FAQ item. Usage: /addfaq <kb_slug> <question> | <answer>"""
    if not is_admin(update.effective_user.id):
        return
    
    text = " ".join(context.args) if context.args else ""
    
    if "|" not in text or len(text.split("|")) < 2:
        await update.message.reply_text("Usage: /addfaq <kb_slug> <question> | <answer>")
        return
    
    parts = text.split("|")
    first_part = parts[0].strip().split()
    
    if len(first_part) < 2:
        await update.message.reply_text("Usage: /addfaq <kb_slug> <question> | <answer>")
        return
    
    kb_slug = first_part[0]
    question = " ".join(first_part[1:])
    answer = "|".join(parts[1:]).strip()
    
    kb = knowledge_registry.get(kb_slug)
    if not kb:
        await update.message.reply_text(f"❌ Knowledge base '{kb_slug}' not found")
        return
    
    item = KnowledgeItem(
        knowledge_base_id=0,
        question=question,
        content=answer
    )
    knowledge_registry.add_item(kb_slug, item)
    await update.message.reply_text(f"✅ FAQ added to '{kb_slug}':\nQ: {question}\nA: {answer[:100]}...")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show stats."""
    if not is_admin(update.effective_user.id):
        return
    
    kbs = knowledge_registry.get_all_active()
    kb_info = "\n".join([f"  • {kb.name} ({kb.slug}): {len(knowledge_registry.get_items(kb.slug))} items" for kb in kbs])
    
    await update.message.reply_text(
        f"📊 *Statistics*\n\n"
        f"*Moderation:*\n"
        f"  • Whitelisted users: {len(moderation_service.whitelisted_users)}\n"
        f"  • Flagged keywords: {len(moderation_service.flagged_keywords)}\n\n"
        f"*Knowledge Bases:*\n{kb_info or '  None'}",
        parse_mode="Markdown"
    )
```

### `src/bot/handlers/callback_handler.py`

```python
"""Callback query handler for admin buttons."""
import logging
from telegram import Update
from telegram.ext import ContextTypes

from src.config import settings
from src.services.moderation.service import moderation_service

logger = logging.getLogger(__name__)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle admin decision callbacks."""
    query = update.callback_query
    user_id = query.from_user.id
    
    if user_id not in settings.admin_chat_ids:
        await query.answer("Not authorized", show_alert=True)
        return
    
    await query.answer()
    data = query.data
    
    if data.startswith("approve_"):
        await query.edit_message_text(
            query.message.text + "\n\n✅ *Approved*",
            parse_mode="Markdown"
        )
    
    elif data.startswith("delete_"):
        parts = data.split("_")
        chat_id, message_id = int(parts[1]), int(parts[2])
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            await query.edit_message_text(
                query.message.text + "\n\n🗑️ *Deleted*",
                parse_mode="Markdown"
            )
        except Exception as e:
            await query.edit_message_text(
                query.message.text + f"\n\n❌ *Failed: {e}*",
                parse_mode="Markdown"
            )
    
    elif data.startswith("whitelist_"):
        target_user_id = int(data.split("_")[1])
        moderation_service.add_to_whitelist(target_user_id)
        await query.edit_message_text(
            query.message.text + f"\n\n⚪ *User {target_user_id} whitelisted*",
            parse_mode="Markdown"
        )
```

---

## Main Entry Point

### `src/main.py`

```python
"""Main entry point."""
import logging
import asyncio

from src.bot.app import create_bot_application
from src.services.knowledge.registry import setup_default_knowledge_bases

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    """Start the bot."""
    logger.info("Setting up knowledge bases...")
    setup_default_knowledge_bases()
    
    logger.info("Creating bot application...")
    app = create_bot_application()
    
    logger.info("Starting bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
```

---

## Requirements

### `requirements.txt`

```
python-telegram-bot[all]>=21.0
anthropic>=0.18.0
openai>=1.0.0
pydantic>=2.0
pydantic-settings>=2.0
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29.0
alembic>=1.13.0
numpy>=1.24.0
python-dotenv>=1.0
redis>=5.0
```

---

## Docker Setup

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  bot:
    build: .
    env_file: .env
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: modbot
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]
```

---

## Initial Data Seeding Script

### `scripts/seed_knowledge.py`

```python
"""Seed initial knowledge base data."""
from src.services.knowledge.registry import knowledge_registry, setup_default_knowledge_bases
from src.models.schemas import KnowledgeItem


def seed_faq():
    """Seed FAQ items."""
    items = [
        KnowledgeItem(
            knowledge_base_id=0,
            question="How do I get started?",
            content="Welcome! To get started, simply introduce yourself and tell us what you're interested in. Our community is here to help!"
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="How can I contact support?",
            content="You can reach our support team by sending a direct message to any admin, or email support@example.com."
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="What are the community features?",
            content="Our community offers: Discussion channels, Q&A support, Resource sharing, Regular events, and Networking opportunities."
        ),
    ]
    knowledge_registry.add_items("faq", items)


def seed_rules():
    """Seed rules."""
    items = [
        KnowledgeItem(
            knowledge_base_id=0,
            title="General Rules",
            content="1. Be respectful to all members. 2. No spam or self-promotion without permission. 3. Keep discussions on-topic. 4. No hate speech or harassment. 5. Follow Telegram's Terms of Service."
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="Is self-promotion allowed?",
            content="Limited self-promotion is allowed in designated channels only. Excessive promotion or spam will result in warnings or bans."
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="What happens if I break the rules?",
            content="Rule violations result in: 1st offense - Warning, 2nd offense - Temporary mute, 3rd offense - Ban. Severe violations may result in immediate ban."
        ),
    ]
    knowledge_registry.add_items("rules", items)


def seed_product():
    """Seed product info."""
    items = [
        KnowledgeItem(
            knowledge_base_id=0,
            question="How much does it cost?",
            content="We offer three plans: Free (basic features), Pro ($9.99/month - full features), Enterprise (custom pricing - unlimited everything + priority support)."
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="Is there a free trial?",
            content="Yes! We offer a 14-day free trial of our Pro plan. No credit card required to start."
        ),
    ]
    knowledge_registry.add_items("product", items)


def main():
    print("Setting up default knowledge bases...")
    setup_default_knowledge_bases()
    
    print("Seeding FAQ...")
    seed_faq()
    
    print("Seeding Rules...")
    seed_rules()
    
    print("Seeding Product Info...")
    seed_product()
    
    print("Done! Knowledge bases seeded.")
    
    # Print summary
    for kb in knowledge_registry.get_all_active():
        items = knowledge_registry.get_items(kb.slug)
        print(f"  {kb.name}: {len(items)} items")


if __name__ == "__main__":
    main()
```

---

## Getting Started Instructions

1. **Create Telegram Bot**: Message @BotFather, create bot, get token
2. **Set up environment**: Copy `.env.example` to `.env` and fill in values
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Seed knowledge**: `python -m scripts.seed_knowledge`
5. **Run bot**: `python -m src.main`

For production:
1. Set up PostgreSQL database
2. Run migrations with Alembic
3. Use Docker Compose: `docker-compose up -d`

---

## Key Features Summary

- **Message Classification**: LLM-powered classification (question/statement/greeting)
- **Content Moderation**: Keyword scanning + LLM intent classification
- **Knowledge System**: Dynamic KB registry + LLM routing + RAG generation
- **Admin Controls**: Whitelist, keyword management, KB management via commands
- **Review System**: Admin approval workflow for flagged content
