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
            query.message.text + "\n\n✅ *Approved*", parse_mode="Markdown"
        )

    elif data.startswith("delete_"):
        parts = data.split("_")
        chat_id, message_id = int(parts[1]), int(parts[2])
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            await query.edit_message_text(
                query.message.text + "\n\n🗑️ *Deleted*", parse_mode="Markdown"
            )
        except Exception as e:
            await query.edit_message_text(
                query.message.text + f"\n\n❌ *Failed: {e}*", parse_mode="Markdown"
            )

    elif data.startswith("whitelist_"):
        target_user_id = int(data.split("_")[1])
        moderation_service.add_to_whitelist(target_user_id)
        await query.edit_message_text(
            query.message.text + f"\n\n⚪ *User {target_user_id} whitelisted*",
            parse_mode="Markdown",
        )
