"""
Admin command handlers for bot management.

This module provides command handlers for admin users to manage
the bot configuration, including:
- User whitelist management (/whitelist, /unwhitelist)
- Keyword moderation (/addkeyword)
- Knowledge base management (/addkb, /addfaq)
- Statistics and monitoring (/stats)

All commands check for admin authorization before execution.
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes

from src.config import settings
from src.services.moderation.service import moderation_service
from src.services.knowledge.registry import knowledge_registry
from src.models.schemas import KnowledgeBase, KnowledgeItem
from src.models.enums import KnowledgeBaseType

logger = logging.getLogger(__name__)


def is_admin(user_id: int) -> bool:
    """
    Check if user is authorized admin.

    Args:
        user_id: Telegram user ID to check

    Returns:
        True if user is in admin_chat_ids, False otherwise
    """
    return user_id in settings.admin_chat_ids


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start and /help commands.

    Displays welcome message with list of available admin commands
    and their usage.

    Args:
        update: Telegram update object
        context: Bot context
    """
    await update.message.reply_text(
        "🤖 *Moderation Bot*\n\n"
        "*Admin Commands:*\n"
        "/whitelist <user_id> - Add to whitelist\n"
        "/unwhitelist <user_id> - Remove from whitelist\n"
        "/addkeyword <word> - Add flagged keyword\n"
        "/addkb <slug> <name> <description> - Add knowledge base\n"
        "/addfaq <kb_slug> <question> | <answer> - Add FAQ item\n"
        "/stats - View statistics",
        parse_mode="Markdown"
    )


async def whitelist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add user to whitelist."""
    if not is_admin(update.effective_user.id):
        return

    if not context.args:
        await update.message.reply_text("Usage: /whitelist <user_id>")
        return

    try:
        user_id = int(context.args[0])
        moderation_service.add_to_whitelist(user_id)
        await update.message.reply_text(f"✅ User {user_id} whitelisted")
    except ValueError:
        await update.message.reply_text("Invalid user ID")


async def unwhitelist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove from whitelist."""
    if not is_admin(update.effective_user.id):
        return

    if not context.args:
        await update.message.reply_text("Usage: /unwhitelist <user_id>")
        return

    try:
        user_id = int(context.args[0])
        moderation_service.remove_from_whitelist(user_id)
        await update.message.reply_text(f"✅ User {user_id} removed from whitelist")
    except ValueError:
        await update.message.reply_text("Invalid user ID")


async def add_keyword_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add flagged keyword."""
    if not is_admin(update.effective_user.id):
        return

    if not context.args:
        await update.message.reply_text("Usage: /addkeyword <keyword>")
        return

    keyword = " ".join(context.args).lower()
    settings.flagged_keywords.add(keyword)
    await update.message.reply_text(f"✅ Keyword '{keyword}' added")


async def add_kb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add knowledge base."""
    if not is_admin(update.effective_user.id):
        return

    if len(context.args) < 3:
        await update.message.reply_text("Usage: /addkb <slug> <name> <description>")
        return

    slug = context.args[0].lower()
    name = context.args[1]
    description = " ".join(context.args[2:])

    kb = KnowledgeBase(
        slug=slug,
        name=name,
        description=description,
        type=KnowledgeBaseType.CUSTOM,
        priority=5
    )
    knowledge_registry.register(kb)
    await update.message.reply_text(f"✅ Knowledge base '{name}' ({slug}) created")


async def add_faq_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add FAQ item. Usage: /addfaq <kb_slug> <question> | <answer>"""
    if not is_admin(update.effective_user.id):
        return

    text = " ".join(context.args) if context.args else ""

    if "|" not in text or len(text.split("|")) < 2:
        await update.message.reply_text("Usage: /addfaq <kb_slug> <question> | <answer>")
        return

    parts = text.split("|")
    first_part = parts[0].strip().split()

    if len(first_part) < 2:
        await update.message.reply_text("Usage: /addfaq <kb_slug> <question> | <answer>")
        return

    kb_slug = first_part[0]
    question = " ".join(first_part[1:])
    answer = "|".join(parts[1:]).strip()

    kb = knowledge_registry.get(kb_slug)
    if not kb:
        await update.message.reply_text(f"❌ Knowledge base '{kb_slug}' not found")
        return

    item = KnowledgeItem(
        knowledge_base_id=0,
        question=question,
        content=answer
    )
    knowledge_registry.add_item(kb_slug, item)
    await update.message.reply_text(f"✅ FAQ added to '{kb_slug}':\nQ: {question}\nA: {answer[:100]}...")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show stats."""
    if not is_admin(update.effective_user.id):
        return

    kbs = knowledge_registry.get_all_active()
    kb_info = "\n".join([f"  • {kb.name} ({kb.slug}): {len(knowledge_registry.get_items(kb.slug))} items" for kb in kbs])

    await update.message.reply_text(
        f"📊 *Statistics*\n\n"
        f"*Moderation:*\n"
        f"  • Whitelisted users: {len(moderation_service.whitelisted_users)}\n"
        f"  • Flagged keywords: {len(moderation_service.flagged_keywords)}\n\n"
        f"*Knowledge Bases:*\n{kb_info or '  None'}",
        parse_mode="Markdown"
    )
