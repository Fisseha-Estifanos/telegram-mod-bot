"""Custom exceptions."""


class ModBotException(Exception):
    """Base exception for the bot."""
    pass


class ConfigurationError(ModBotException):
    """Configuration related errors."""
    pass


class LLMError(ModBotException):
    """LLM API related errors."""
    pass


class ModerationError(ModBotException):
    """Moderation service errors."""
    pass


class KnowledgeBaseError(ModBotException):
    """Knowledge base related errors."""
    pass


class DatabaseError(ModBotException):
    """Database operation errors."""
    pass
