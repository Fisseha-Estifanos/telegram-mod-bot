"""Unified LLM client wrapper."""
import json
import logging
from typing import Dict, Any
import anthropic

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for LLM API calls."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Make a completion request."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            raise

    async def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Make a completion request and parse JSON response."""
        response_text = await self.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=0.0
        )

        try:
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {response_text}")
            raise ValueError(f"Invalid JSON from LLM: {e}")


llm_client = LLMClient()
