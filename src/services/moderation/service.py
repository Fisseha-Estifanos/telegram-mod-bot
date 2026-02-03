"""Main moderation service."""
import re
import logging
from typing import Tuple, Optional

from src.config import settings
from src.services.llm.classifier import intent_classifier
from src.models.schemas import ModerationResult
from src.models.enums import ModerationAction

logger = logging.getLogger(__name__)


class ModerationService:
    """Content moderation service."""

    def __init__(self):
        self.whitelisted_users = set(settings.whitelisted_user_ids)
        self.flagged_keywords = settings.flagged_keywords

    def is_whitelisted(self, user_id: int) -> bool:
        return user_id in self.whitelisted_users

    def add_to_whitelist(self, user_id: int) -> None:
        self.whitelisted_users.add(user_id)

    def remove_from_whitelist(self, user_id: int) -> None:
        self.whitelisted_users.discard(user_id)

    def scan_keywords(self, content: str) -> list[str]:
        """Scan for flagged keywords."""
        content_lower = content.lower()
        found = []
        for keyword in self.flagged_keywords:
            pattern = rf'\b{re.escape(keyword.lower())}\b'
            if re.search(pattern, content_lower):
                found.append(keyword)
        return found

    async def moderate_content(
        self,
        content: str,
        user_id: int,
        username: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[ModerationResult, str]:
        """Main moderation pipeline. Returns (result, action)."""

        # Whitelist check
        if self.is_whitelisted(user_id):
            return ModerationResult(
                intent="safe",
                confidence=100,
                explanation="User whitelisted",
                action=ModerationAction.ALLOW
            ), ModerationAction.ALLOW.value

        # Keyword scan
        keywords_found = self.scan_keywords(content)

        # LLM classification
        result = await intent_classifier.classify(content, username)
        result.keywords_found = list(set(result.keywords_found + keywords_found))

        # Adjust action if keywords found but LLM uncertain
        if keywords_found and result.confidence < 80 and result.action == ModerationAction.ALLOW:
            result.action = ModerationAction.REVIEW

        return result, result.action.value


moderation_service = ModerationService()
