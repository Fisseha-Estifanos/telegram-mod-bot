"""Embedding utilities."""
import logging
from typing import List
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


async def get_embedding(text: str) -> List[float]:
    """Get embedding for text."""

    if settings.use_local_embeddings:
        return _get_local_embedding(text)

    # Use OpenAI embeddings
    try:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"OpenAI embedding failed, using local: {e}")
        return _get_local_embedding(text)


def _get_local_embedding(text: str) -> List[float]:
    """Simple local embedding fallback."""
    import hashlib

    words = text.lower().split()
    dim = settings.embedding_dimension
    vector = [0.0] * dim

    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % dim] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity."""
    v1, v2 = np.array(vec1), np.array(vec2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0.0
