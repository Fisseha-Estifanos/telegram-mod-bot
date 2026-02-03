"""Retrieves relevant content from KBs."""
import logging
from typing import List
import numpy as np

from src.services.knowledge.registry import knowledge_registry
from src.models.schemas import KnowledgeItem, RetrievedContext
from src.utils.embeddings import get_embedding, cosine_similarity
from src.config import settings

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Retrieves relevant items using semantic search."""

    def __init__(self):
        self.top_k = settings.knowledge_retrieval_top_k
        self.threshold = settings.knowledge_similarity_threshold

    async def retrieve(self, question: str, kb_slugs: List[str]) -> List[RetrievedContext]:
        """Retrieve relevant items from specified KBs."""
        results = []
        question_embedding = await get_embedding(question)

        for slug in kb_slugs:
            kb = knowledge_registry.get(slug)
            if not kb:
                continue

            items = knowledge_registry.get_items(slug)
            if not items:
                continue

            scored_items = []
            for item in items:
                if item.embedding:
                    score = cosine_similarity(question_embedding, item.embedding)
                else:
                    score = self._keyword_score(question, item)
                scored_items.append((item, score))

            scored_items.sort(key=lambda x: x[1], reverse=True)
            top_items = [(item, score) for item, score in scored_items[:self.top_k] if score >= self.threshold]

            if top_items:
                results.append(RetrievedContext(
                    knowledge_base_slug=slug,
                    knowledge_base_name=kb.name,
                    items=[item for item, _ in top_items],
                    relevance_scores=[score for _, score in top_items]
                ))

        return results

    def _keyword_score(self, question: str, item: KnowledgeItem) -> float:
        """Keyword matching fallback."""
        q_words = set(question.lower().split())
        item_text = f"{item.question or ''} {item.title or ''} {item.content}".lower()
        i_words = set(item_text.split())

        if not i_words or not q_words:
            return 0.0

        return len(q_words & i_words) / len(q_words)


knowledge_retriever = KnowledgeRetriever()
