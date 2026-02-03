"""Seed initial knowledge base data."""

import os
import sys

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.knowledge.registry import (
    knowledge_registry,
    setup_default_knowledge_bases,
)
from src.models.schemas import KnowledgeItem


def seed_faq():
    """Seed FAQ items."""
    items = [
        KnowledgeItem(
            knowledge_base_id=0,
            question="How do I get started?",
            content="Welcome! To get started, simply introduce yourself and tell us what you're interested in. Our community is here to help!",  # noqa: E501
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="How can I contact support?",
            content="You can reach our support team by sending a direct message to any admin, or email support@example.com.",  # noqa: E501
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="What are the community features?",
            content="Our community offers: Discussion channels, Q&A support, Resource sharing, Regular events, and Networking opportunities.",  # noqa: E501
        ),
    ]
    knowledge_registry.add_items("faq", items)


def seed_rules():
    """Seed rules."""
    items = [
        KnowledgeItem(
            knowledge_base_id=0,
            title="General Rules",
            content="1. Be respectful to all members. 2. No spam or self-promotion without permission. 3. Keep discussions on-topic. 4. No hate speech or harassment. 5. Follow Telegram's Terms of Service.",  # noqa: E501
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="Is self-promotion allowed?",
            content="Limited self-promotion is allowed in designated channels only. Excessive promotion or spam will result in warnings or bans.",  # noqa: E501
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="What happens if I break the rules?",
            content="Rule violations result in: 1st offense - Warning, 2nd offense - Temporary mute, 3rd offense - Ban. Severe violations may result in immediate ban.",  # noqa: E501
        ),
    ]
    knowledge_registry.add_items("rules", items)


def seed_product():
    """Seed product info."""
    items = [
        KnowledgeItem(
            knowledge_base_id=0,
            question="How much does it cost?",
            content="We offer three plans: Free (basic features), Pro ($9.99/month - full features), Enterprise (custom pricing - unlimited everything + priority support).",  # noqa: E501
        ),
        KnowledgeItem(
            knowledge_base_id=0,
            question="Is there a free trial?",
            content="Yes! We offer a 14-day free trial of our Pro plan. No credit card required to start.",  # noqa: E501
        ),
    ]
    knowledge_registry.add_items("product", items)


def main():
    """
    Main function to seed the knowledge base.
    """
    print("Setting up default knowledge bases...")
    setup_default_knowledge_bases()

    print("Seeding FAQ...")
    seed_faq()

    print("Seeding Rules...")
    seed_rules()

    print("Seeding Product Info...")
    seed_product()

    print("Done! Knowledge bases seeded.")

    # Print summary
    for kb in knowledge_registry.get_all_active():
        items = knowledge_registry.get_items(kb.slug)
        print(f"  {kb.name}: {len(items)} items")


if __name__ == "__main__":
    main()
