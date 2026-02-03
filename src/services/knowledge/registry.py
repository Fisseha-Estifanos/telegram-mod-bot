"""
Knowledge Base Registry.

Central registry for managing all knowledge bases and their items.
Provides in-memory storage and management of knowledge bases,
with support for dynamic registration, item management, and routing context.
"""

from typing import Dict, List, Optional
import logging
from src.models.schemas import KnowledgeBase, KnowledgeItem
from src.models.enums import KnowledgeBaseType

logger = logging.getLogger(__name__)


class KnowledgeRegistry:
    """
    Central registry for all knowledge bases.

    Manages knowledge bases and their items in memory. Provides methods
    for registration, retrieval, and generating routing context for the
    LLM router.

    Attributes:
        _knowledge_bases: Dictionary mapping slugs to knowledge bases
        _items: Dictionary mapping KB slugs to lists of items
    """

    def __init__(self):
        self._knowledge_bases: Dict[str, KnowledgeBase] = {}
        self._items: Dict[str, List[KnowledgeItem]] = {}

    def register(self, kb: KnowledgeBase) -> None:
        """Register a knowledge base."""
        self._knowledge_bases[kb.slug] = kb
        if kb.slug not in self._items:
            self._items[kb.slug] = []
        logger.info(f"Registered KB: {kb.name} ({kb.slug})")

    def unregister(self, slug: str) -> None:
        """Remove a knowledge base."""
        self._knowledge_bases.pop(slug, None)
        self._items.pop(slug, None)

    def get(self, slug: str) -> Optional[KnowledgeBase]:
        """Get KB by slug."""
        return self._knowledge_bases.get(slug)

    def get_all_active(self) -> List[KnowledgeBase]:
        """Get all active KBs sorted by priority."""
        return sorted(
            [kb for kb in self._knowledge_bases.values() if kb.is_active],
            key=lambda x: x.priority,
            reverse=True
        )

    def add_item(self, kb_slug: str, item: KnowledgeItem) -> None:
        """Add item to a KB."""
        if kb_slug not in self._items:
            self._items[kb_slug] = []
        self._items[kb_slug].append(item)

    def add_items(self, kb_slug: str, items: List[KnowledgeItem]) -> None:
        """Add multiple items to a KB."""
        for item in items:
            self.add_item(kb_slug, item)

    def get_items(self, kb_slug: str) -> List[KnowledgeItem]:
        """Get all items for a KB."""
        return self._items.get(kb_slug, [])

    def get_routing_context(self) -> str:
        """
        Generate formatted context for LLM router.

        Creates a text summary of all active knowledge bases with their
        metadata, used by the router to decide which KBs can answer a question.

        Returns:
            Formatted string containing all active KB descriptions
        """
        lines = []
        for kb in self.get_all_active():
            lines.append(f"""
Knowledge Base: {kb.name}
Slug: {kb.slug}
Type: {kb.type.value}
Description: {kb.description}
Example Questions: {', '.join(kb.example_questions[:3]) if kb.example_questions else 'N/A'}
Keywords: {', '.join(kb.routing_hints[:5]) if kb.routing_hints else 'N/A'}
""")
        return "\n---\n".join(lines)


knowledge_registry = KnowledgeRegistry()


def setup_default_knowledge_bases():
    """
    Register default knowledge bases.

    Creates and registers three default knowledge bases:
    - FAQ: Frequently asked questions
    - Rules: Community rules and guidelines
    - Product: Product information and pricing

    Called during bot initialization to set up core knowledge bases.
    """
    knowledge_registry.register(KnowledgeBase(
        slug="faq",
        name="Frequently Asked Questions",
        description="Common questions and answers",
        type=KnowledgeBaseType.FAQ,
        routing_hints=["how do i", "what is", "can i", "help", "question", "how to"],
        example_questions=["How do I get started?", "What are the features?", "How can I contact support?"],
        priority=10
    ))

    knowledge_registry.register(KnowledgeBase(
        slug="rules",
        name="Community Rules",
        description="Rules, guidelines, and policies",
        type=KnowledgeBaseType.RULES,
        routing_hints=["rules", "allowed", "banned", "permitted", "guidelines", "policy", "can i post"],
        example_questions=["What are the rules?", "Is promotion allowed?", "What gets you banned?"],
        priority=9
    ))

    knowledge_registry.register(KnowledgeBase(
        slug="product",
        name="Product Information",
        description="Product features, pricing, and details",
        type=KnowledgeBaseType.PRODUCT_INFO,
        routing_hints=["price", "cost", "feature", "product", "plan", "subscription", "pricing"],
        example_questions=["How much does it cost?", "What's included?", "Is there a free plan?"],
        priority=8
    ))
