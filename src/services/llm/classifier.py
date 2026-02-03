"""
Message and intent classification using LLM.

This module provides classifiers for determining message types and
moderation intents using Claude AI. Includes optimizations for
common cases (commands, greetings) before calling the LLM.
"""

import logging
from src.services.llm.client import llm_client
from src.services.llm.prompts import MESSAGE_CLASSIFIER_PROMPT, MODERATION_CLASSIFIER_PROMPT
from src.models.schemas import MessageClassification, ModerationResult
from src.models.enums import MessageType, ModerationIntent, ModerationAction
from src.config import settings

logger = logging.getLogger(__name__)


class MessageClassifier:
    """
    Classifies incoming messages into types.

    Determines whether a message is a question, statement, command,
    or greeting. Uses fast pattern matching for common cases,
    falls back to LLM for complex classification.
    """

    async def classify(self, text: str) -> MessageClassification:
        """
        Classify the type of an incoming message.

        Performs quick pattern matching for commands and greetings,
        then uses LLM classification for other message types.

        Args:
            text: The message text to classify

        Returns:
            MessageClassification with type, confidence, and processing flags

        Examples:
            >>> await classifier.classify("/start")
            MessageClassification(message_type=MessageType.COMMAND, ...)
            >>> await classifier.classify("How do I get started?")
            MessageClassification(message_type=MessageType.QUESTION, ...)
        """

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
    """
    Classifies message intent for content moderation.

    Analyzes message content to detect harmful intents such as spam,
    scams, hate speech, etc. Uses Claude AI for contextual understanding
    and automatically determines the appropriate moderation action.
    """

    async def classify(self, text: str, username: str = None) -> ModerationResult:
        """
        Classify the moderation intent of a message.

        Uses LLM to analyze message content and context to detect
        potentially harmful intents. Automatically maps intent to
        moderation action based on configuration.

        Args:
            text: The message text to analyze
            username: Optional username of the message sender

        Returns:
            ModerationResult with intent, confidence, explanation, and action

        Raises:
            Returns UNCLEAR intent with REVIEW action if classification fails

        Examples:
            >>> await classifier.classify("Buy crypto now! Guaranteed 100x returns!")
            ModerationResult(intent=ModerationIntent.SCAM, action=ModerationAction.DELETE, ...)
        """
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
