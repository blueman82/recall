"""Unit tests for Ollama embedding client."""

import pytest
import httpx

from recall.embedding.ollama import OllamaClient, EmbeddingError, EMBED_PREFIX


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        client = OllamaClient()
        assert client.host == "http://localhost:11434"
        assert client.model == "mxbai-embed-large"
        assert client.timeout == 30.0

    def test_custom_values(self):
        """Test custom initialization values."""
        client = OllamaClient(
            host="http://custom:8080/",
            model="custom-model",
            timeout=60.0,
        )
        assert client.host == "http://custom:8080"  # Trailing slash stripped
        assert client.model == "custom-model"
        assert client.timeout == 60.0

    def test_host_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from host."""
        client = OllamaClient(host="http://localhost:11434/")
        assert client.host == "http://localhost:11434"


class TestEmbedPrefix:
    """Tests for query prefix handling."""

    def test_prefix_constant(self):
        """Test EMBED_PREFIX constant value."""
        assert EMBED_PREFIX == "Represent this sentence for searching relevant passages: "


class TestEmbed:
    """Tests for single text embedding."""

    @pytest.mark.asyncio
    async def test_embed_document_no_prefix(self, httpx_mock):
        """Test document embedding does not add prefix."""
        mock_embedding = [0.1, 0.2, 0.3]
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": [mock_embedding]},
        )

        async with OllamaClient() as client:
            result = await client.embed("test document", is_query=False)

        assert result == mock_embedding
        request = httpx_mock.get_request()
        assert request.content == b'{"model":"mxbai-embed-large","input":"test document"}'

    @pytest.mark.asyncio
    async def test_embed_query_with_prefix(self, httpx_mock):
        """Test query embedding adds mxbai prefix."""
        mock_embedding = [0.1, 0.2, 0.3]
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": [mock_embedding]},
        )

        async with OllamaClient() as client:
            result = await client.embed("test query", is_query=True)

        assert result == mock_embedding
        request = httpx_mock.get_request()
        expected_input = f"{EMBED_PREFIX}test query"
        import json
        payload = json.loads(request.content)
        assert payload["input"] == expected_input

    @pytest.mark.asyncio
    async def test_embed_query_no_prefix_non_mxbai(self, httpx_mock):
        """Test query embedding without prefix for non-mxbai models."""
        mock_embedding = [0.1, 0.2, 0.3]
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": [mock_embedding]},
        )

        async with OllamaClient(model="nomic-embed-text") as client:
            result = await client.embed("test query", is_query=True)

        assert result == mock_embedding
        request = httpx_mock.get_request()
        import json
        payload = json.loads(request.content)
        assert payload["input"] == "test query"  # No prefix

    @pytest.mark.asyncio
    async def test_embed_empty_text_raises(self):
        """Test that empty text raises ValueError."""
        async with OllamaClient() as client:
            with pytest.raises(ValueError, match="Text cannot be empty"):
                await client.embed("")

    @pytest.mark.asyncio
    async def test_embed_whitespace_only_raises(self):
        """Test that whitespace-only text raises ValueError."""
        async with OllamaClient() as client:
            with pytest.raises(ValueError, match="Text cannot be empty"):
                await client.embed("   ")

    @pytest.mark.asyncio
    async def test_embed_no_embeddings_returned(self, httpx_mock):
        """Test error when no embeddings returned."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": []},
        )

        async with OllamaClient() as client:
            with pytest.raises(EmbeddingError, match="No embedding returned"):
                await client.embed("test")


class TestEmbedBatch:
    """Tests for batch embedding."""

    @pytest.mark.asyncio
    async def test_embed_batch_single_batch(self, httpx_mock):
        """Test batch embedding with single batch."""
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": mock_embeddings},
        )

        async with OllamaClient() as client:
            texts = ["text1", "text2", "text3"]
            result = await client.embed_batch(texts)

        assert result == mock_embeddings
        request = httpx_mock.get_request()
        import json
        payload = json.loads(request.content)
        assert payload["input"] == texts

    @pytest.mark.asyncio
    async def test_embed_batch_multiple_batches(self, httpx_mock):
        """Test batch embedding with multiple batches."""
        batch1_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        batch2_embeddings = [[0.5, 0.6]]

        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": batch1_embeddings},
        )
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": batch2_embeddings},
        )

        async with OllamaClient() as client:
            texts = ["text1", "text2", "text3"]
            result = await client.embed_batch(texts, batch_size=2)

        assert result == batch1_embeddings + batch2_embeddings

    @pytest.mark.asyncio
    async def test_embed_batch_with_query_prefix(self, httpx_mock):
        """Test batch embedding with query prefix."""
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": mock_embeddings},
        )

        async with OllamaClient() as client:
            texts = ["query1", "query2"]
            result = await client.embed_batch(texts, is_query=True)

        assert result == mock_embeddings
        request = httpx_mock.get_request()
        import json
        payload = json.loads(request.content)
        expected_inputs = [f"{EMBED_PREFIX}query1", f"{EMBED_PREFIX}query2"]
        assert payload["input"] == expected_inputs

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list_raises(self):
        """Test that empty texts list raises ValueError."""
        async with OllamaClient() as client:
            with pytest.raises(ValueError, match="Texts list cannot be empty"):
                await client.embed_batch([])

    @pytest.mark.asyncio
    async def test_embed_batch_wrong_count(self, httpx_mock):
        """Test error when wrong number of embeddings returned."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": [[0.1, 0.2]]},  # Only 1, expected 3
        )

        async with OllamaClient() as client:
            with pytest.raises(EmbeddingError, match="Expected 3 embeddings"):
                await client.embed_batch(["t1", "t2", "t3"])


class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_connect_error(self, httpx_mock, mocker):
        """Test retry on connection errors."""
        # Mock asyncio.sleep to avoid actual delays
        mock_sleep = mocker.patch("asyncio.sleep", return_value=None)

        # First two calls fail, third succeeds
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": [[0.1, 0.2]]},
        )

        async with OllamaClient() as client:
            result = await client.embed("test")

        assert result == [0.1, 0.2]
        # Should have slept twice (1s, then 2s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, httpx_mock, mocker):
        """Test error after all retries exhausted."""
        mocker.patch("asyncio.sleep", return_value=None)

        # All calls fail
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        async with OllamaClient() as client:
            with pytest.raises(EmbeddingError, match="Failed to connect.*after 3 attempts"):
                await client.embed("test")

    @pytest.mark.asyncio
    async def test_timeout_error_no_retry(self, httpx_mock):
        """Test that timeout errors do not retry."""
        httpx_mock.add_exception(httpx.TimeoutException("Request timed out"))

        async with OllamaClient() as client:
            with pytest.raises(EmbeddingError, match="Request timeout"):
                await client.embed("test")

    @pytest.mark.asyncio
    async def test_http_status_error_no_retry(self, httpx_mock):
        """Test that HTTP status errors do not retry."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            status_code=500,
            text="Internal Server Error",
        )

        async with OllamaClient() as client:
            with pytest.raises(EmbeddingError, match="Ollama API error: 500"):
                await client.embed("test")


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self, httpx_mock):
        """Test that context manager properly closes client."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": [[0.1]]},
        )

        client = OllamaClient()
        async with client:
            await client.embed("test")
            assert client._client is not None

        # After exiting, client should be closed
        assert client._client is None

    @pytest.mark.asyncio
    async def test_manual_close(self, httpx_mock):
        """Test manual close method."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:11434/api/embed",
            json={"embeddings": [[0.1]]},
        )

        client = OllamaClient()
        await client.embed("test")
        assert client._client is not None

        await client.close()
        assert client._client is None
