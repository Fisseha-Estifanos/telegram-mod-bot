"""RAG response generation."""
import logging
from typing import List
from src.services.llm.client import llm_client
from src.services.llm.prompts import RAG_GENERATION_PROMPT
from src.models.schemas import RetrievedContext, KnowledgeResponse

logger = logging.getLogger(__name__)


class KnowledgeGenerator:
    """Generates answers using RAG."""

    async def generate(self, question: str, contexts: List[RetrievedContext]) -> KnowledgeResponse:
        """Generate answer from contexts."""

        if not contexts:
            return KnowledgeResponse(
                answer="I couldn't find relevant information to answer your question.",
                sources=[],
                confidence=0,
                knowledge_bases_used=[]
            )

        context_text = self._format_contexts(contexts)

        try:
            prompt = RAG_GENERATION_PROMPT.format(context=context_text, question=question)

            answer = await llm_client.complete(
                system_prompt="Answer based only on the provided context.",
                user_message=prompt,
                max_tokens=500
            )

            sources = []
            kb_used = []
            for ctx in contexts:
                kb_used.append(ctx.knowledge_base_slug)
                for item in ctx.items:
                    sources.append({
                        "kb": ctx.knowledge_base_name,
                        "title": item.title or item.question or "Item"
                    })

            return KnowledgeResponse(
                answer=answer,
                sources=sources[:5],
                confidence=self._calc_confidence(contexts),
                knowledge_bases_used=list(set(kb_used))
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return KnowledgeResponse(
                answer="I encountered an error. Please try again.",
                sources=[],
                confidence=0,
                knowledge_bases_used=[]
            )

    def _format_contexts(self, contexts: List[RetrievedContext]) -> str:
        parts = []
        for ctx in contexts:
            parts.append(f"=== {ctx.knowledge_base_name} ===")
            for item in ctx.items:
                if item.question:
                    parts.append(f"Q: {item.question}\nA: {item.content}")
                elif item.title:
                    parts.append(f"**{item.title}**\n{item.content}")
                else:
                    parts.append(item.content)
                parts.append("")
        return "\n".join(parts)

    def _calc_confidence(self, contexts: List[RetrievedContext]) -> int:
        scores = [s for ctx in contexts for s in ctx.relevance_scores]
        return min(100, int((sum(scores) / len(scores)) * 100)) if scores else 50


knowledge_generator = KnowledgeGenerator()
