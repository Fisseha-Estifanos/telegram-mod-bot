"""
Enumeration types for the moderation bot.

This module defines all enum classes used throughout the application:
- MessageType: Classification of incoming messages
- ModerationIntent: Types of harmful or inappropriate content
- ModerationAction: Actions to take on moderated content
- KnowledgeBaseType: Categories of knowledge bases
"""

from enum import Enum


class MessageType(str, Enum):
    """
    Classification of incoming message types.

    Used by the message classifier to determine how to process
    each incoming message.
    """

    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class ModerationIntent(str, Enum):
    """
    Intent classification for content moderation.

    Categorizes potentially harmful or inappropriate content types
    that require moderation action. Used by the LLM classifier to
    identify problematic messages.
    """

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
    """
    Actions to take on moderated content.

    Defines the possible moderation responses:
    - ALLOW: Message passes moderation
    - DELETE: Automatically delete the message
    - REVIEW: Flag for admin review before taking action
    """

    ALLOW = "allow"
    DELETE = "delete"
    REVIEW = "review"


class KnowledgeBaseType(str, Enum):
    """
    Categories of knowledge bases.

    Defines different types of knowledge bases that can be created
    to organize information. Each type helps the routing system
    understand what kind of questions the KB can answer.
    """

    FAQ = "faq"
    RULES = "rules"
    PRODUCT_INFO = "product_info"
    POLICIES = "policies"
    TUTORIALS = "tutorials"
    ANNOUNCEMENTS = "announcements"
    CUSTOM = "custom"
