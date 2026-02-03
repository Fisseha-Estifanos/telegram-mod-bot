"""Abstract base classes and interfaces."""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.models.schemas import (
    MessageClassification, ModerationResult, RoutingDecision,
    RetrievedContext, KnowledgeResponse
)


class MessageClassifierInterface(ABC):
    """Interface for message classification."""

    @abstractmethod
    async def classify(self, text: str) -> MessageClassification:
        """Classify a message."""
        pass


class IntentClassifierInterface(ABC):
    """Interface for intent classification."""

    @abstractmethod
    async def classify(self, text: str, username: Optional[str] = None) -> ModerationResult:
        """Classify content intent."""
        pass


class KnowledgeRouterInterface(ABC):
    """Interface for knowledge routing."""

    @abstractmethod
    async def route(self, question: str) -> RoutingDecision:
        """Route question to appropriate knowledge bases."""
        pass


class KnowledgeRetrieverInterface(ABC):
    """Interface for knowledge retrieval."""

    @abstractmethod
    async def retrieve(self, question: str, kb_slugs: List[str]) -> List[RetrievedContext]:
        """Retrieve relevant content from knowledge bases."""
        pass


class KnowledgeGeneratorInterface(ABC):
    """Interface for knowledge generation."""

    @abstractmethod
    async def generate(self, question: str, contexts: List[RetrievedContext]) -> KnowledgeResponse:
        """Generate answer from contexts."""
        pass
