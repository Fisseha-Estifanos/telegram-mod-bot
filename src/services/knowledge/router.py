"""Knowledge routing service."""
import logging
from src.services.llm.client import llm_client
from src.services.llm.prompts import KNOWLEDGE_ROUTER_PROMPT
from src.services.knowledge.registry import knowledge_registry
from src.models.schemas import RoutingDecision

logger = logging.getLogger(__name__)


class KnowledgeRouter:
    """Routes questions to appropriate KBs."""

    async def route(self, question: str) -> RoutingDecision:
        """Determine which KB(s) should handle this question."""

        kb_context = knowledge_registry.get_routing_context()

        if not kb_context.strip():
            return RoutingDecision(
                should_answer=False,
                selected_knowledge_bases=[],
                confidence=0,
                reasoning="No knowledge bases available",
                fallback_response="I don't have any information sources configured."
            )

        try:
            prompt = KNOWLEDGE_ROUTER_PROMPT.format(
                knowledge_bases=kb_context,
                question=question
            )

            result = await llm_client.complete_json(
                system_prompt="You are a routing assistant. Respond with JSON only.",
                user_message=prompt
            )

            return RoutingDecision(
                should_answer=result["should_answer"],
                selected_knowledge_bases=result.get("selected_knowledge_bases", []),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                fallback_response=result.get("fallback_response")
            )
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return RoutingDecision(
                should_answer=False,
                selected_knowledge_bases=[],
                confidence=0,
                reasoning=str(e),
                fallback_response="I'm having trouble finding information. Please try again."
            )


knowledge_router = KnowledgeRouter()
