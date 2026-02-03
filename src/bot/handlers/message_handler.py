"""
Main message handler for the Telegram bot.

This module processes all incoming non-command text messages through
a multi-stage pipeline:
1. Message classification (question, statement, greeting)
2. Knowledge lookup for questions (routing -> retrieval -> generation)
3. Content moderation (keyword scan -> LLM classification -> action)

Different message types receive different processing and responses.
"""

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
    """
    Process incoming text messages.

    Main handler for all non-command messages. Performs classification,
    knowledge lookup for questions, and content moderation.

    Args:
        update: Telegram update object containing the message
        context: Bot context for accessing bot methods

    Processing flow:
        1. Check if message is from a monitored chat
        2. Classify message type
        3. Route questions through knowledge system
        4. Run moderation on statements
        5. Take appropriate action (respond, delete, flag for review)
    """

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
    """
    Process question through knowledge system (RAG pipeline).

    Performs three-stage RAG processing:
    1. Routing: Determine which knowledge bases can answer
    2. Retrieval: Semantic search for relevant content
    3. Generation: LLM generates answer from context

    Args:
        question: User's question text

    Returns:
        Generated answer with sources, or None if no answer available
    """

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
    """
    Run content moderation pipeline.

    Checks message content for policy violations and takes
    appropriate action based on detected intent.

    Args:
        message: Telegram message object
        user: User who sent the message
        context: Bot context for actions
    """

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
    """
    Delete message and notify admins.

    Removes the message from chat and sends notification to all
    configured admin users with details about the deletion.

    Args:
        message: Message to delete
        result: Moderation result with intent details
        context: Bot context for deletion
    """
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
    """
    Send message to admins for manual review.

    Creates an admin notification with approve/delete/whitelist
    buttons for human review of flagged content.

    Args:
        message: Message needing review
        result: Moderation result with intent and confidence
        context: Bot context for sending notifications
    """
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
