"""
Pydantic schemas for data validation and serialization.

This module defines all Pydantic models used throughout the application
for request/response validation, data transfer, and API interactions.
Organized into sections: Message Classification, Moderation, Knowledge System.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.models.enums import (
    MessageType, ModerationIntent, ModerationAction, KnowledgeBaseType
)


# ============== Message Classification ==============

class MessageClassification(BaseModel):
    """
    Result of message type classification.

    Contains the classified message type, confidence score, and flags
    indicating whether moderation or knowledge lookup is required.

    Attributes:
        message_type: The classified type (question, statement, etc.)
        confidence: Classification confidence (0-100)
        detected_language: Detected language code (default: "en")
        requires_moderation: Whether message needs moderation check
        requires_knowledge_lookup: Whether message needs KB query
    """

    message_type: MessageType
    confidence: int = Field(ge=0, le=100)
    detected_language: str = "en"
    requires_moderation: bool = True
    requires_knowledge_lookup: bool = False


# ============== Moderation ==============

class ModerationResult(BaseModel):
    """
    Result of content moderation analysis.

    Contains the detected intent, confidence, explanation, and
    recommended action for the message.

    Attributes:
        intent: The detected moderation intent
        confidence: Detection confidence (0-100)
        explanation: Human-readable explanation of the classification
        keywords_found: List of flagged keywords detected
        action: Recommended action (ALLOW, DELETE, REVIEW)
    """

    intent: ModerationIntent
    confidence: int = Field(ge=0, le=100)
    explanation: str
    keywords_found: List[str] = []
    action: ModerationAction = ModerationAction.ALLOW


# ============== Knowledge System ==============

class KnowledgeBase(BaseModel):
    """
    Knowledge base configuration and metadata.

    Represents a collection of related knowledge items (FAQ, rules, etc.).
    Used for routing questions and organizing content.

    Attributes:
        id: Database ID (optional, for persistence)
        name: Display name of the knowledge base
        slug: URL-friendly identifier
        description: What this KB contains
        type: Category of knowledge base
        routing_hints: Keywords for routing questions
        example_questions: Sample questions this KB can answer
        priority: Routing priority (higher = checked first)
        is_active: Whether this KB is enabled
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

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
    """
    Individual piece of knowledge within a knowledge base.

    Represents a single FAQ, rule, or information snippet.
    Can have embeddings for semantic search.

    Attributes:
        id: Database ID (optional, for persistence)
        knowledge_base_id: ID of parent knowledge base
        question: Question text (for FAQ items)
        title: Title (for article-style items)
        content: The actual content/answer
        tags: Tags for categorization
        metadata: Additional metadata
        embedding: Vector embedding for semantic search
        is_active: Whether this item is enabled
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

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
    """
    Result of knowledge base routing decision.

    Determines which knowledge bases should be queried for a question.

    Attributes:
        should_answer: Whether any KB can answer the question
        selected_knowledge_bases: List of KB slugs to query
        confidence: Routing confidence (0-100)
        reasoning: Explanation of routing decision
        fallback_response: Response if no KB matches
    """

    should_answer: bool
    selected_knowledge_bases: List[str] = []
    confidence: int = Field(ge=0, le=100)
    reasoning: str
    fallback_response: Optional[str] = None


class RetrievedContext(BaseModel):
    """
    Retrieved context from a knowledge base.

    Contains relevant items and scores from semantic search.

    Attributes:
        knowledge_base_slug: Slug of source KB
        knowledge_base_name: Display name of source KB
        items: Retrieved knowledge items
        relevance_scores: Relevance scores for each item
    """

    knowledge_base_slug: str
    knowledge_base_name: str
    items: List[KnowledgeItem]
    relevance_scores: List[float] = []


class KnowledgeResponse(BaseModel):
    """
    Final generated response from knowledge system.

    Contains the RAG-generated answer with sources and metadata.

    Attributes:
        answer: Generated answer text
        sources: Source citations
        confidence: Overall confidence (0-100)
        knowledge_bases_used: List of KBs that contributed
    """

    answer: str
    sources: List[Dict[str, str]] = []
    confidence: int = Field(ge=0, le=100)
    knowledge_bases_used: List[str] = []


# ============== Processing Result ==============

class ProcessingResult(BaseModel):
    """
    Complete message processing result.

    Aggregates all processing stages: classification, moderation,
    and knowledge response.

    Attributes:
        classification: Message classification result
        moderation: Moderation result (if applicable)
        knowledge_response: Knowledge system response (if applicable)
        should_delete: Whether message should be deleted
        should_respond: Whether bot should respond
        response_text: Generated response text
    """

    classification: MessageClassification
    moderation: Optional[ModerationResult] = None
    knowledge_response: Optional[KnowledgeResponse] = None
    should_delete: bool = False
    should_respond: bool = False
    response_text: Optional[str] = None
