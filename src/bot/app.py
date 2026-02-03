"""
Bot application setup and configuration.

This module creates and configures the Telegram bot application
with all handlers, middleware, and error handling. Sets up:
- Command handlers for admin operations
- Message handler for content processing
- Callback handler for inline button interactions
- Global error handler for exception logging
"""

import logging
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters

from src.config import settings

logger = logging.getLogger(__name__)


def create_bot_application() -> Application:
    """
    Create and configure the Telegram bot application.

    Builds the bot application with all necessary handlers:
    - Command handlers: /start, /help, /whitelist, /addkb, etc.
    - Message handler: Processes all text messages
    - Callback handler: Handles inline button clicks
    - Error handler: Logs exceptions globally

    Returns:
        Configured Application instance ready to run

    Note:
        Handlers are imported inside function to avoid circular imports
    """

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
    """
    Global error handler for all unhandled exceptions.

    Logs all errors that occur during message processing to help
    with debugging and monitoring.

    Args:
        update: Telegram update that caused the error (may be None)
        context: Bot context containing error details
    """
    logger.error(f"Error: {context.error}", exc_info=context.error)
