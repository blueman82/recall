"""Embedding layer for recall."""

from recall.embedding.ollama import EMBED_PREFIX, EmbeddingError, OllamaClient

__all__ = ["OllamaClient", "EmbeddingError", "EMBED_PREFIX"]
