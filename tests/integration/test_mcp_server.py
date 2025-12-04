"""Integration tests for the Recall MCP server.

These tests verify that the MCP server correctly exposes all memory operations
as tools and that the tools function correctly end-to-end.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recall.config import RecallSettings
from recall.memory.types import MemoryType, RelationType
from recall.storage.chromadb import ChromaStore
from recall.storage.hybrid import HybridStore
from recall.storage.sqlite import SQLiteStore
from recall.embedding.ollama import OllamaClient


def unique_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_embedding_client():
    """Create mock OllamaClient for testing."""
    client = AsyncMock(spec=OllamaClient)
    # Return consistent embeddings for mxbai dimension
    client.embed.return_value = [0.1] * 1024
    return client


@pytest.fixture
def ephemeral_store(mock_embedding_client):
    """Create HybridStore with ephemeral stores for testing."""
    sqlite = SQLiteStore(ephemeral=True)
    chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())

    store = HybridStore(
        sqlite_store=sqlite,
        chroma_store=chroma,
        embedding_client=mock_embedding_client,
    )
    yield store
    sqlite.close()


class TestRecallSettings:
    """Tests for RecallSettings configuration."""

    def test_default_settings(self):
        """Test that default settings are applied correctly."""
        settings = RecallSettings()

        assert settings.ollama_host == "http://localhost:11434"
        assert settings.ollama_model == "mxbai-embed-large"
        assert settings.collection_name == "memories"
        assert settings.log_level == "INFO"
        assert settings.default_namespace == "global"
        assert settings.default_importance == 0.5
        assert settings.default_token_budget == 4000

    def test_env_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("RECALL_OLLAMA_HOST", "http://custom:11434")
        monkeypatch.setenv("RECALL_OLLAMA_MODEL", "custom-model")
        monkeypatch.setenv("RECALL_LOG_LEVEL", "DEBUG")

        settings = RecallSettings()

        assert settings.ollama_host == "http://custom:11434"
        assert settings.ollama_model == "custom-model"
        assert settings.log_level == "DEBUG"

    def test_importance_validation(self):
        """Test that importance is validated."""
        # Valid importance
        settings = RecallSettings(default_importance=0.8)
        assert settings.default_importance == 0.8

        # Invalid importance should raise
        with pytest.raises(ValueError):
            RecallSettings(default_importance=1.5)

        with pytest.raises(ValueError):
            RecallSettings(default_importance=-0.1)


class TestMCPToolHandlers:
    """Tests for MCP tool handler functions."""

    @pytest.mark.asyncio
    async def test_memory_store_tool_success(self, ephemeral_store):
        """Test memory_store_tool with valid input."""
        # Import the tool handler
        from recall.__main__ import memory_store_tool

        # Patch the global hybrid_store
        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result = await memory_store_tool(
                content="Test memory content",
                memory_type="preference",
                namespace="global",
                importance=0.8,
            )

        assert result["success"] is True
        assert result["id"] is not None
        assert result["content_hash"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_memory_store_tool_invalid_type(self, ephemeral_store):
        """Test memory_store_tool with invalid memory type."""
        from recall.__main__ import memory_store_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result = await memory_store_tool(
                content="Test content",
                memory_type="invalid_type",
            )

        assert result["success"] is False
        assert "Invalid memory_type" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_store_tool_not_initialized(self):
        """Test memory_store_tool when server not initialized."""
        from recall.__main__ import memory_store_tool

        with patch("recall.__main__.hybrid_store", None):
            result = await memory_store_tool(
                content="Test content",
            )

        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_recall_tool_success(self, ephemeral_store):
        """Test memory_recall_tool with stored memories."""
        from recall.__main__ import memory_store_tool, memory_recall_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Store a memory first
            await memory_store_tool(
                content="User prefers dark mode",
                memory_type="preference",
                namespace="global",
            )

            # Recall memories
            result = await memory_recall_tool(
                query="dark mode preferences",
                n_results=5,
            )

        assert result["success"] is True
        assert "memories" in result
        assert "total" in result

    @pytest.mark.asyncio
    async def test_memory_recall_tool_with_filters(self, ephemeral_store):
        """Test memory_recall_tool with namespace and type filters."""
        from recall.__main__ import memory_store_tool, memory_recall_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Store memories with different types
            await memory_store_tool(
                content="Preference memory",
                memory_type="preference",
                namespace="global",
            )
            await memory_store_tool(
                content="Decision memory",
                memory_type="decision",
                namespace="project:test",
            )

            # Recall with filter
            result = await memory_recall_tool(
                query="memory",
                namespace="global",
                memory_type="preference",
            )

        assert result["success"] is True
        # All returned memories should match filter criteria
        for memory in result["memories"]:
            assert memory["namespace"] == "global"
            assert memory["type"] == "preference"

    @pytest.mark.asyncio
    async def test_memory_recall_tool_invalid_type(self, ephemeral_store):
        """Test memory_recall_tool with invalid memory type filter."""
        from recall.__main__ import memory_recall_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result = await memory_recall_tool(
                query="test",
                memory_type="invalid_type",
            )

        assert result["success"] is False
        assert "Invalid memory_type" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_relate_tool_success(self, ephemeral_store):
        """Test memory_relate_tool creates edge between memories."""
        from recall.__main__ import memory_store_tool, memory_relate_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Create two memories
            result1 = await memory_store_tool(
                content="New information",
                memory_type="decision",
            )
            result2 = await memory_store_tool(
                content="Old information",
                memory_type="decision",
            )

            # Create relationship
            result = await memory_relate_tool(
                source_id=result1["id"],
                target_id=result2["id"],
                relation="supersedes",
                weight=1.0,
            )

        assert result["success"] is True
        assert result["edge_id"] is not None

    @pytest.mark.asyncio
    async def test_memory_relate_tool_invalid_relation(self, ephemeral_store):
        """Test memory_relate_tool with invalid relation type."""
        from recall.__main__ import memory_store_tool, memory_relate_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result1 = await memory_store_tool(content="Memory 1")
            result2 = await memory_store_tool(content="Memory 2")

            result = await memory_relate_tool(
                source_id=result1["id"],
                target_id=result2["id"],
                relation="invalid_relation",
            )

        assert result["success"] is False
        assert "Invalid relation" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_relate_tool_nonexistent_memory(self, ephemeral_store):
        """Test memory_relate_tool with nonexistent memory ID."""
        from recall.__main__ import memory_store_tool, memory_relate_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result1 = await memory_store_tool(content="Existing memory")

            result = await memory_relate_tool(
                source_id=result1["id"],
                target_id="nonexistent_id",
                relation="relates_to",
            )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_context_tool_success(self, ephemeral_store):
        """Test memory_context_tool generates markdown context."""
        from recall.__main__ import memory_store_tool, memory_context_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Store some memories
            await memory_store_tool(
                content="User prefers vim keybindings",
                memory_type="preference",
            )
            await memory_store_tool(
                content="Decided to use FastAPI",
                memory_type="decision",
            )

            # Generate context
            result = await memory_context_tool(
                project="testproject",
                token_budget=4000,
            )

        assert result["success"] is True
        assert "context" in result
        assert "token_estimate" in result
        assert "# Memory Context" in result["context"]

    @pytest.mark.asyncio
    async def test_memory_context_tool_with_query(self, ephemeral_store):
        """Test memory_context_tool with search query."""
        from recall.__main__ import memory_store_tool, memory_context_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            await memory_store_tool(
                content="Relevant memory",
                memory_type="preference",
            )

            result = await memory_context_tool(
                query="relevant",
                token_budget=2000,
            )

        assert result["success"] is True
        assert result["token_estimate"] <= 2000

    @pytest.mark.asyncio
    async def test_memory_forget_tool_by_id(self, ephemeral_store):
        """Test memory_forget_tool deletes by ID."""
        from recall.__main__ import memory_store_tool, memory_forget_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Store a memory
            store_result = await memory_store_tool(
                content="Memory to delete",
            )
            memory_id = store_result["id"]

            # Delete by ID
            result = await memory_forget_tool(
                memory_id=memory_id,
            )

        assert result["success"] is True
        assert result["deleted_count"] == 1
        assert memory_id in result["deleted_ids"]

    @pytest.mark.asyncio
    async def test_memory_forget_tool_by_query(self, ephemeral_store):
        """Test memory_forget_tool deletes by semantic search."""
        from recall.__main__ import memory_store_tool, memory_forget_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Store memories
            await memory_store_tool(content="Memory about topic A")
            await memory_store_tool(content="Memory about topic B")

            # Delete by query
            result = await memory_forget_tool(
                query="topic",
                n_results=10,
            )

        assert result["success"] is True
        assert result["deleted_count"] > 0

    @pytest.mark.asyncio
    async def test_memory_forget_tool_not_confirmed(self, ephemeral_store):
        """Test memory_forget_tool respects confirm flag."""
        from recall.__main__ import memory_store_tool, memory_forget_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            store_result = await memory_store_tool(content="Memory")

            result = await memory_forget_tool(
                memory_id=store_result["id"],
                confirm=False,
            )

        assert result["success"] is False
        assert "not confirmed" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_forget_tool_invalid_params(self, ephemeral_store):
        """Test memory_forget_tool requires either id or query."""
        from recall.__main__ import memory_forget_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Neither id nor query
            result = await memory_forget_tool()

        assert result["success"] is False
        assert "Must provide either" in result["error"]


class TestMCPServerIntegration:
    """End-to-end integration tests for MCP server functionality."""

    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self, ephemeral_store):
        """Test complete memory lifecycle: store -> recall -> relate -> forget."""
        from recall.__main__ import (
            memory_store_tool,
            memory_recall_tool,
            memory_relate_tool,
            memory_forget_tool,
        )

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # 1. Store memories
            result1 = await memory_store_tool(
                content="Original design decision",
                memory_type="decision",
                namespace="project:test",
                importance=0.7,
            )
            assert result1["success"] is True
            id1 = result1["id"]

            result2 = await memory_store_tool(
                content="Updated design decision",
                memory_type="decision",
                namespace="project:test",
                importance=0.9,
            )
            assert result2["success"] is True
            id2 = result2["id"]

            # 2. Create relationship (new supersedes old)
            relate_result = await memory_relate_tool(
                source_id=id2,
                target_id=id1,
                relation="supersedes",
            )
            assert relate_result["success"] is True

            # 3. Recall memories
            recall_result = await memory_recall_tool(
                query="design decision",
                namespace="project:test",
                include_related=True,
            )
            assert recall_result["success"] is True
            assert recall_result["total"] >= 1

            # 4. Forget old memory
            forget_result = await memory_forget_tool(
                memory_id=id1,
            )
            assert forget_result["success"] is True
            assert forget_result["deleted_count"] == 1

    @pytest.mark.asyncio
    async def test_deduplication(self, ephemeral_store):
        """Test that duplicate content returns existing memory ID."""
        from recall.__main__ import memory_store_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            content = "Unique content for dedup test"

            # Store first time
            result1 = await memory_store_tool(
                content=content,
                namespace="global",
            )
            assert result1["success"] is True
            id1 = result1["id"]

            # Store same content again
            result2 = await memory_store_tool(
                content=content,
                namespace="global",
            )
            assert result2["success"] is True
            assert result2["id"] == id1  # Should return same ID

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, ephemeral_store):
        """Test that memories in different namespaces are isolated."""
        from recall.__main__ import memory_store_tool, memory_recall_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Store in global
            await memory_store_tool(
                content="Global memory",
                namespace="global",
            )

            # Store in project namespace
            await memory_store_tool(
                content="Project memory",
                namespace="project:myapp",
            )

            # Query only global
            result = await memory_recall_tool(
                query="memory",
                namespace="global",
            )

            # Should only get global memories
            for memory in result["memories"]:
                assert memory["namespace"] == "global"

    @pytest.mark.asyncio
    async def test_importance_filtering(self, ephemeral_store):
        """Test min_importance filtering in recall."""
        from recall.__main__ import memory_store_tool, memory_recall_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            # Store with different importance levels
            await memory_store_tool(
                content="High importance memory",
                importance=0.9,
            )
            await memory_store_tool(
                content="Low importance memory",
                importance=0.2,
            )

            # Query with min_importance filter
            result = await memory_recall_tool(
                query="memory",
                min_importance=0.5,
            )

            # Should only get high importance memories
            for memory in result["memories"]:
                assert memory["importance"] >= 0.5

    @pytest.mark.asyncio
    async def test_all_memory_types(self, ephemeral_store):
        """Test storage and retrieval of all memory types."""
        from recall.__main__ import memory_store_tool, memory_recall_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            for mem_type in MemoryType:
                result = await memory_store_tool(
                    content=f"Content for {mem_type.value}",
                    memory_type=mem_type.value,
                )
                assert result["success"] is True

            # Verify all types can be recalled
            for mem_type in MemoryType:
                result = await memory_recall_tool(
                    query=mem_type.value,
                    memory_type=mem_type.value,
                )
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_all_relation_types(self, ephemeral_store):
        """Test creation of all relation types."""
        from recall.__main__ import memory_store_tool, memory_relate_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            source = await memory_store_tool(content="Source memory")

            for rel_type in RelationType:
                target = await memory_store_tool(
                    content=f"Target for {rel_type.value}"
                )

                result = await memory_relate_tool(
                    source_id=source["id"],
                    target_id=target["id"],
                    relation=rel_type.value,
                )
                assert result["success"] is True


class TestMCPServerErrorHandling:
    """Tests for error handling in MCP server."""

    @pytest.mark.asyncio
    async def test_empty_content_error(self, ephemeral_store):
        """Test that empty content returns error."""
        from recall.__main__ import memory_store_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result = await memory_store_tool(
                content="",
            )

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_importance_error(self, ephemeral_store):
        """Test that invalid importance returns error."""
        from recall.__main__ import memory_store_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result = await memory_store_tool(
                content="Test",
                importance=1.5,  # Invalid
            )

        assert result["success"] is False
        assert "importance" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_memory_not_found_error(self, ephemeral_store):
        """Test that nonexistent memory ID returns error."""
        from recall.__main__ import memory_forget_tool

        with patch("recall.__main__.hybrid_store", ephemeral_store):
            result = await memory_forget_tool(
                memory_id="nonexistent_id",
            )

        assert result["success"] is False
        assert "not found" in result["error"]
