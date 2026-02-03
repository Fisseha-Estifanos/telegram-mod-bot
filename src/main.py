"""
Main entry point for the Telegram moderation bot.

This module initializes and starts the bot application, including:
- Setting up logging configuration
- Initializing default knowledge bases
- Creating the bot application with all handlers
- Starting the polling loop to receive messages
"""

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
    """
    Start the Telegram bot application.

    This function performs the following steps:
    1. Sets up default knowledge bases (FAQ, Rules, Product Info)
    2. Creates the bot application with all handlers
    3. Starts polling for incoming messages

    The bot will run continuously until interrupted.
    """
    logger.info("Setting up knowledge bases...")
    setup_default_knowledge_bases()

    logger.info("Creating bot application...")
    app = create_bot_application()

    logger.info("Starting bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
