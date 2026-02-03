"""
Application configuration module.

This module defines all configuration settings for the Telegram moderation bot,
loaded from environment variables using Pydantic Settings. Includes settings for
Telegram bot, LLM APIs, embeddings, database, and moderation rules.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Set, List
from src.models.enums import ModerationIntent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class defines all configuration parameters for the bot, including:
    - Telegram bot credentials and chat IDs
    - LLM API keys and models (Anthropic Claude)
    - Embedding service configuration (OpenAI or local)
    - Database and Redis URLs
    - Moderation rules (keywords, intents, actions)
    - Knowledge system parameters

    All settings are loaded from .env file or environment variables.
    """

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

    flagged_keywords: Set[str] = Field(
        default_factory=lambda: {
            "crypto",
            "investment",
            "guaranteed returns",
            "click here",
            "free money",
            "dm me",
            "limited time",
            "act now",
            "wire transfer",
        }
    )

    auto_delete_intents: Set[str] = Field(
        default_factory=lambda: {
            ModerationIntent.SPAM.value,
            ModerationIntent.SCAM.value,
            ModerationIntent.ADULT_CONTENT.value,
        }
    )

    admin_review_intents: Set[str] = Field(
        default_factory=lambda: {
            ModerationIntent.HATE_SPEECH.value,
            ModerationIntent.HARASSMENT.value,
            ModerationIntent.MISINFORMATION.value,
        }
    )

    # Knowledge System
    knowledge_retrieval_top_k: int = 3
    knowledge_similarity_threshold: float = 0.7

    class Config:
        """
        Configuration for the application settings.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
