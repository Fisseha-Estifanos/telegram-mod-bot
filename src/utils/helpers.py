"""General helper utilities."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_markdown(text: str) -> str:
    """Escape special markdown characters."""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def format_user_mention(user_id: int, username: Optional[str] = None) -> str:
    """Format user mention for display."""
    if username:
        return f"@{username}"
    return f"User {user_id}"


def extract_user_id_from_mention(mention: str) -> Optional[int]:
    """Extract user ID from mention string."""
    try:
        # Handle @username format
        if mention.startswith('@'):
            return None

        # Handle numeric ID
        return int(mention)
    except (ValueError, AttributeError):
        return None
