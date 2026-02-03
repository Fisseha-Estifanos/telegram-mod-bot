"""
SQLAlchemy database models for persistent storage.

This module defines all database tables using SQLAlchemy ORM:
- WhitelistedUser: Users exempt from moderation
- FlaggedKeyword: Keywords to detect in messages
- ModerationLog: History of moderation actions
- PendingReview: Messages awaiting admin review
- KnowledgeBaseDB: Knowledge base configurations
- KnowledgeItemDB: Individual knowledge items

These models are used for production deployments with PostgreSQL.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Text,
    Index,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class WhitelistedUser(Base):
    """
    Represents a whitelisted user.
    """

    __tablename__ = "whitelisted_users"

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, unique=True, index=True)
    username = Column(String, nullable=True)
    added_by = Column(BigInteger)
    added_at = Column(DateTime, default=datetime.utcnow)
    reason = Column(String, nullable=True)


class FlaggedKeyword(Base):
    """
    Represents a flagged keyword.
    """

    __tablename__ = "flagged_keywords"

    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True, index=True)
    severity = Column(String, default="medium")
    added_at = Column(DateTime, default=datetime.utcnow)


class ModerationLog(Base):
    """
    Represents a moderation log entry.
    """

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

    __table_args__ = (Index("idx_moderation_chat_user", "chat_id", "user_id"),)


class PendingReview(Base):
    """
    Represents a pending review.
    """

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
    """
    Represents a knowledge base.
    """

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

    items = relationship(
        "KnowledgeItemDB", back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class KnowledgeItemDB(Base):
    """
    Represents a knowledge item.
    """

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

    __table_args__ = (Index("idx_knowledge_item_kb", "knowledge_base_id", "is_active"),)
