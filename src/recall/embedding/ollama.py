"""Ollama embedding client with async httpx and mxbai query prefix support.

This module provides an async HTTP client for the Ollama embeddings API with:
- Query prefix support for mxbai-embed-large model
- Exponential backoff retry logic for network resilience
- Batch embedding support for efficiency
- Configurable timeout handling
"""

import asyncio
import logging
from typing import Any, List

import httpx

logger = logging.getLogger(__name__)

# Query prefix for mxbai-embed-large model
# Documents do NOT get this prefix, only queries
EMBED_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    pass


class OllamaClient:
    """Async HTTP client for Ollama embeddings API.

    Provides embedding generation with retry logic and batch processing.
    Supports mxbai-embed-large query prefix for optimal retrieval.

    Args:
        host: Ollama server host URL (default: "http://localhost:11434")
        model: Embedding model name (default: "mxbai-embed-large")
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> async with OllamaClient() as client:
        ...     # Query embedding (with prefix)
        ...     query_emb = await client.embed("What is Python?", is_query=True)
        ...     # Document embedding (no prefix)
        ...     doc_emb = await client.embed("Python is a programming language.")
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mxbai-embed-large",
        timeout: float = 30.0,
    ):
        """Initialize Ollama client.

        Args:
            host: Ollama server host URL
            model: Embedding model name
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - close client."""
        await self.close()

    async def _request_with_retry(
        self,
        payload: dict,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> dict:
        """Make HTTP request to Ollama API with exponential backoff retry.

        Args:
            payload: Request payload dictionary
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Initial delay in seconds (default: 1.0)

        Returns:
            Response JSON data

        Raises:
            EmbeddingError: If request fails after all retries
        """
        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{self.host}/api/embed",
                    json=payload,
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result

            except httpx.ConnectError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # 1s, 2s, 4s
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise EmbeddingError(
                        f"Failed to connect to Ollama after {max_retries} attempts: {e}"
                    ) from e

            except httpx.TimeoutException as e:
                raise EmbeddingError(
                    f"Request timeout after {self.timeout}s. "
                    f"Consider increasing timeout for model {self.model}"
                ) from e

            except httpx.HTTPStatusError as e:
                raise EmbeddingError(
                    f"Ollama API error: {e.response.status_code} - {e.response.text}"
                ) from e

            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Request error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise EmbeddingError(
                        f"Ollama API request failed after {max_retries} attempts: {e}"
                    ) from e

        # Should not reach here, but just in case
        raise EmbeddingError(f"Unexpected error: {last_error}")

    async def embed(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed
            is_query: If True, apply mxbai query prefix for search queries

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If text is empty

        Example:
            >>> client = OllamaClient()
            >>> # For search queries - adds prefix
            >>> query_emb = await client.embed("What is Python?", is_query=True)
            >>> # For documents - no prefix
            >>> doc_emb = await client.embed("Python is a language.")
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Apply query prefix for mxbai-embed-large when is_query=True
        if is_query and "mxbai" in self.model.lower():
            prefixed_text = f"{EMBED_PREFIX}{text}"
        else:
            prefixed_text = text

        payload = {
            "model": self.model,
            "input": prefixed_text,
        }

        data = await self._request_with_retry(payload)
        embeddings = data.get("embeddings")

        if not embeddings or len(embeddings) == 0:
            raise EmbeddingError("No embedding returned from Ollama API")

        embedding: list[float] = embeddings[0]
        return embedding

    async def embed_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        The Ollama /api/embed endpoint supports multiple inputs in a single
        request, which is more efficient than individual calls.

        Args:
            texts: List of input texts to embed
            is_query: If True, apply mxbai query prefix for search queries
            batch_size: Number of texts per API call (default: 32)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If texts list is empty

        Example:
            >>> client = OllamaClient()
            >>> texts = ["doc1", "doc2", "doc3"]
            >>> embeddings = await client.embed_batch(texts)
            >>> len(embeddings)
            3
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Apply query prefix if needed
            if is_query and "mxbai" in self.model.lower():
                processed_batch = [f"{EMBED_PREFIX}{text}" for text in batch]
            else:
                processed_batch = batch

            payload = {
                "model": self.model,
                "input": processed_batch,
            }

            data = await self._request_with_retry(payload)
            embeddings = data.get("embeddings")

            if not embeddings or len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"Expected {len(batch)} embeddings, got {len(embeddings) if embeddings else 0}"
                )

            all_embeddings.extend(embeddings)

        return all_embeddings
