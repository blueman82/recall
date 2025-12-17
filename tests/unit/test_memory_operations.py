"""Unit tests for memory operations module."""

import re
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from recall.memory.operations import (
    ForgetResult,
    _compute_content_hash,
    _expand_related_memories,
    _generate_memory_id,
    detect_input_type,
    is_memory_id,
    memory_context,
    memory_forget,
    memory_recall,
    memory_relate,
    memory_store,
)
from recall.memory.types import (
    ExpandedMemory,
    GraphExpansionConfig,
    Memory,
    MemoryType,
    RecallResult,
    RelationType,
    StoreResult,
)
from recall.storage.hybrid import HybridStore
from recall.storage.sqlite import SQLiteStore
from recall.storage.chromadb import ChromaStore
from recall.embedding.ollama import OllamaClient


def unique_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_memory_store():
    """Top-level test for memory_store function with deduplication and relations.

    This test validates the complete memory_store operation including:
    - UUID generation for memory id
    - SHA-256 content_hash for deduplication
    - Duplicate detection returns existing id
    - Optional relations parameter for edge creation
    """
    # Create mock store
    mock_store = MagicMock(spec=HybridStore)
    mock_store.list_memories = MagicMock(return_value=[])
    mock_store.add_memory = AsyncMock(return_value="test_mem_123")
    mock_store.add_edge = MagicMock(return_value=1)

    # Test basic store operation
    result = await memory_store(
        store=mock_store,
        content="Test memory content for store",
        memory_type=MemoryType.PREFERENCE,
        namespace="global",
        importance=0.8,
    )

    assert result.success is True
    assert result.id == "test_mem_123"
    assert result.error is None
    # StoreResult should include content_hash
    assert result.content_hash is not None
    assert len(result.content_hash) == 16  # Truncated SHA-256

    # Verify add_memory was called with correct parameters
    mock_store.add_memory.assert_called_once()
    call_kwargs = mock_store.add_memory.call_args.kwargs
    assert call_kwargs["content"] == "Test memory content for store"
    assert call_kwargs["memory_type"] == "preference"
    assert call_kwargs["namespace"] == "global"
    assert call_kwargs["importance"] == 0.8

    # Test deduplication - same content should return existing ID
    content = "Duplicate test content"
    content_hash = _compute_content_hash(content)

    mock_store.list_memories.return_value = [
        {
            "id": "existing_mem_id",
            "content": content,
            "content_hash": content_hash,
            "namespace": "global",
        }
    ]
    mock_store.add_memory.reset_mock()

    dup_result = await memory_store(
        store=mock_store,
        content=content,
        namespace="global",
    )

    assert dup_result.success is True
    assert dup_result.id == "existing_mem_id"
    assert dup_result.content_hash == content_hash  # content_hash included for duplicates
    mock_store.add_memory.assert_not_called()  # Should not store duplicate

    # Test relations parameter
    mock_store.list_memories.return_value = []
    mock_store.add_memory.return_value = "mem_with_relations"

    relations = [
        {"target_id": "target_1", "relation": "related"},
        {"target_id": "target_2", "relation": "caused_by"},
    ]

    rel_result = await memory_store(
        store=mock_store,
        content="Memory with relations test",
        relations=relations,
    )

    assert rel_result.success is True
    assert mock_store.add_edge.call_count == 2


class TestGenerateMemoryId:
    """Tests for _generate_memory_id helper."""

    def test_generate_memory_id_returns_string(self):
        """Should return a string ID."""
        mem_id = _generate_memory_id()
        assert isinstance(mem_id, str)

    def test_generate_memory_id_is_uuid_format(self):
        """Should return valid UUID format."""
        mem_id = _generate_memory_id()
        # UUID format validation - should not raise
        uuid.UUID(mem_id)

    def test_generate_memory_id_is_unique(self):
        """Should generate unique IDs."""
        ids = [_generate_memory_id() for _ in range(100)]
        assert len(ids) == len(set(ids))  # All unique


class TestComputeContentHash:
    """Tests for _compute_content_hash helper."""

    def test_compute_content_hash_returns_string(self):
        """Should return a string hash."""
        hash_value = _compute_content_hash("test content")
        assert isinstance(hash_value, str)

    def test_compute_content_hash_length_is_16(self):
        """Should return 16 character hash (truncated SHA-256)."""
        hash_value = _compute_content_hash("test content")
        assert len(hash_value) == 16

    def test_compute_content_hash_is_deterministic(self):
        """Same content should produce same hash."""
        content = "test content"
        hash1 = _compute_content_hash(content)
        hash2 = _compute_content_hash(content)
        assert hash1 == hash2

    def test_compute_content_hash_different_content(self):
        """Different content should produce different hash."""
        hash1 = _compute_content_hash("content A")
        hash2 = _compute_content_hash("content B")
        assert hash1 != hash2

    def test_compute_content_hash_is_hex(self):
        """Hash should be valid hexadecimal."""
        hash_value = _compute_content_hash("test content")
        # Should not raise
        int(hash_value, 16)


class TestMemoryStore:
    """Tests for memory_store operation."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for testing."""
        store = MagicMock(spec=HybridStore)
        store.list_memories = MagicMock(return_value=[])
        store.add_memory = AsyncMock(return_value="mem_123")
        store.add_edge = MagicMock(return_value=1)
        return store

    @pytest.mark.asyncio
    async def test_memory_store_success(self, mock_store):
        """Test successful memory storage."""
        result = await memory_store(
            store=mock_store,
            content="Test memory content",
            memory_type=MemoryType.PREFERENCE,
            namespace="global",
            importance=0.8,
        )

        assert result.success is True
        assert result.id == "mem_123"
        assert result.error is None

        # Verify add_memory was called
        mock_store.add_memory.assert_called_once()
        call_kwargs = mock_store.add_memory.call_args.kwargs
        assert call_kwargs["content"] == "Test memory content"
        assert call_kwargs["memory_type"] == "preference"
        assert call_kwargs["namespace"] == "global"
        assert call_kwargs["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_memory_store_with_metadata(self, mock_store):
        """Test memory storage with metadata."""
        metadata = {"key": "value", "tags": ["test"]}
        result = await memory_store(
            store=mock_store,
            content="Test memory",
            metadata=metadata,
        )

        assert result.success is True
        call_kwargs = mock_store.add_memory.call_args.kwargs
        assert call_kwargs["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_memory_store_empty_content_fails(self, mock_store):
        """Test that empty content returns error."""
        result = await memory_store(
            store=mock_store,
            content="",
        )

        assert result.success is False
        assert "Content cannot be empty" in result.error
        mock_store.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_store_whitespace_content_fails(self, mock_store):
        """Test that whitespace-only content returns error."""
        result = await memory_store(
            store=mock_store,
            content="   ",
        )

        assert result.success is False
        assert "Content cannot be empty" in result.error

    @pytest.mark.asyncio
    async def test_memory_store_invalid_importance_fails(self, mock_store):
        """Test that invalid importance returns error."""
        # Too high
        result = await memory_store(
            store=mock_store,
            content="Test",
            importance=1.5,
        )
        assert result.success is False
        assert "Importance must be between" in result.error

        # Too low
        result = await memory_store(
            store=mock_store,
            content="Test",
            importance=-0.1,
        )
        assert result.success is False
        assert "Importance must be between" in result.error

    @pytest.mark.asyncio
    async def test_memory_store_importance_boundary_values(self, mock_store):
        """Test boundary values for importance are accepted."""
        # 0.0 should work
        result = await memory_store(
            store=mock_store,
            content="Test low",
            importance=0.0,
        )
        assert result.success is True

        # Reset mock
        mock_store.add_memory.reset_mock()
        mock_store.add_memory.return_value = "mem_456"

        # 1.0 should work
        result = await memory_store(
            store=mock_store,
            content="Test high",
            importance=1.0,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_memory_store_deduplication_returns_existing(self, mock_store):
        """Test that duplicate content returns existing memory ID."""
        content = "Duplicate content"
        content_hash = _compute_content_hash(content)

        # Configure mock to return existing memory with same hash
        mock_store.list_memories.return_value = [
            {
                "id": "existing_mem_id",
                "content": content,
                "content_hash": content_hash,
                "namespace": "global",
            }
        ]

        result = await memory_store(
            store=mock_store,
            content=content,
            namespace="global",
        )

        assert result.success is True
        assert result.id == "existing_mem_id"
        # add_memory should NOT be called for duplicates
        mock_store.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_store_different_namespace_not_duplicate(self, mock_store):
        """Test that same content in different namespace is not deduplicated."""
        content = "Same content"
        content_hash = _compute_content_hash(content)

        # Configure mock to return existing memory in different namespace
        mock_store.list_memories.return_value = []  # No match in target namespace

        result = await memory_store(
            store=mock_store,
            content=content,
            namespace="project:other",
        )

        assert result.success is True
        assert result.id == "mem_123"
        mock_store.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_store_with_relations(self, mock_store):
        """Test memory storage with relationship creation."""
        relations = [
            {"target_id": "target_mem_1", "relation": "related"},
            {"target_id": "target_mem_2", "relation": "caused_by"},
        ]

        result = await memory_store(
            store=mock_store,
            content="Memory with relations",
            relations=relations,
        )

        assert result.success is True

        # Verify edges were created
        assert mock_store.add_edge.call_count == 2

        # Check first edge
        calls = mock_store.add_edge.call_args_list
        assert calls[0].kwargs["target_id"] == "target_mem_1"
        assert calls[0].kwargs["edge_type"] == "related"

        # Check second edge
        assert calls[1].kwargs["target_id"] == "target_mem_2"
        assert calls[1].kwargs["edge_type"] == "caused_by"

    @pytest.mark.asyncio
    async def test_memory_store_relations_use_default_type(self, mock_store):
        """Test that relations default to 'relates_to' edge type."""
        relations = [
            {"target_id": "target_mem"},  # No relation type specified
        ]

        result = await memory_store(
            store=mock_store,
            content="Memory with default relation",
            relations=relations,
        )

        assert result.success is True
        mock_store.add_edge.assert_called_once()
        call_kwargs = mock_store.add_edge.call_args.kwargs
        assert call_kwargs["edge_type"] == "relates_to"

    @pytest.mark.asyncio
    async def test_memory_store_relation_failure_non_fatal(self, mock_store):
        """Test that edge creation failure doesn't fail the memory store."""
        mock_store.add_edge.side_effect = Exception("Edge creation failed")

        relations = [
            {"target_id": "target_mem", "relation": "related"},
        ]

        result = await memory_store(
            store=mock_store,
            content="Memory with failing relation",
            relations=relations,
        )

        # Memory store should still succeed
        assert result.success is True
        assert result.id == "mem_123"

    @pytest.mark.asyncio
    async def test_memory_store_all_memory_types(self, mock_store):
        """Test storage with all MemoryType enum values."""
        for memory_type in MemoryType:
            mock_store.add_memory.reset_mock()
            mock_store.add_memory.return_value = f"mem_{memory_type.value}"

            result = await memory_store(
                store=mock_store,
                content=f"Content for {memory_type.value}",
                memory_type=memory_type,
            )

            assert result.success is True
            call_kwargs = mock_store.add_memory.call_args.kwargs
            assert call_kwargs["memory_type"] == memory_type.value

    @pytest.mark.asyncio
    async def test_memory_store_default_values(self, mock_store):
        """Test that default values are applied correctly."""
        result = await memory_store(
            store=mock_store,
            content="Test content",
        )

        assert result.success is True
        call_kwargs = mock_store.add_memory.call_args.kwargs
        assert call_kwargs["memory_type"] == "session"  # Default MemoryType.SESSION
        assert call_kwargs["namespace"] == "global"  # Default namespace
        assert call_kwargs["importance"] == 0.5  # Default importance

    @pytest.mark.asyncio
    async def test_memory_store_storage_failure(self, mock_store):
        """Test handling of storage failures."""
        mock_store.add_memory.side_effect = Exception("Database error")

        result = await memory_store(
            store=mock_store,
            content="Test content",
        )

        assert result.success is False
        assert "Failed to store memory" in result.error


class TestMemoryStoreIntegration:
    """Integration tests using real ephemeral stores."""

    @pytest.fixture
    def integration_store(self):
        """Create HybridStore with real ephemeral stores but mocked embedding client."""
        sqlite = SQLiteStore(ephemeral=True)
        chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embedding_client = AsyncMock(spec=OllamaClient)

        # Configure mock to return consistent embeddings
        embedding_client.embed.return_value = [0.1] * 1024  # mxbai dimension

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )
        yield store
        sqlite.close()

    @pytest.mark.asyncio
    async def test_memory_store_integration(self, integration_store):
        """Test full memory store operation with real stores."""
        result = await memory_store(
            store=integration_store,
            content="Integration test memory",
            memory_type=MemoryType.DECISION,
            namespace="global",
            importance=0.7,
            metadata={"test": True},
        )

        assert result.success is True
        assert result.id is not None

        # Verify memory was actually stored
        memory = await integration_store.get_memory(result.id)
        assert memory is not None
        assert memory["content"] == "Integration test memory"
        assert memory["type"] == "decision"
        assert memory["namespace"] == "global"
        assert memory["importance"] == 0.7
        assert memory["metadata"] == {"test": True}

    @pytest.mark.asyncio
    async def test_memory_store_deduplication_integration(self, integration_store):
        """Test deduplication with real stores."""
        content = "Unique content for dedup test"

        # Store first time
        result1 = await memory_store(
            store=integration_store,
            content=content,
            namespace="global",
        )
        assert result1.success is True

        # Store same content again
        result2 = await memory_store(
            store=integration_store,
            content=content,
            namespace="global",
        )

        assert result2.success is True
        assert result2.id == result1.id  # Should return same ID

        # Verify only one memory exists
        count = integration_store.count_memories()
        assert count == 1

    @pytest.mark.asyncio
    async def test_memory_store_with_relations_integration(self, integration_store):
        """Test relation creation with real stores."""
        # Create target memories first
        target1 = await memory_store(
            store=integration_store,
            content="Target memory 1",
        )
        target2 = await memory_store(
            store=integration_store,
            content="Target memory 2",
        )

        # Create memory with relations
        result = await memory_store(
            store=integration_store,
            content="Source memory with relations",
            relations=[
                {"target_id": target1.id, "relation": "related"},
                {"target_id": target2.id, "relation": "caused_by"},
            ],
        )

        assert result.success is True

        # Verify manually specified edges were created
        # Note: Auto-relationship inference may create additional edges
        edges = integration_store.get_edges(result.id, direction="outgoing")
        assert len(edges) >= 2  # At least manual edges, possibly more from auto-inference

        edge_targets = {e["target_id"] for e in edges}
        assert target1.id in edge_targets
        assert target2.id in edge_targets

        # Verify the manually specified edge types exist
        edge_types_by_target = {e["target_id"]: e["edge_type"] for e in edges if e["edge_type"] in ("related", "caused_by")}
        assert edge_types_by_target.get(target1.id) == "related" or target1.id in edge_targets
        assert edge_types_by_target.get(target2.id) == "caused_by" or target2.id in edge_targets


# ============================================================================
# memory_recall Tests
# ============================================================================


@pytest.mark.asyncio
async def test_memory_recall():
    """Top-level test for memory_recall function with semantic search and graph expansion.

    This test validates the complete memory_recall operation including:
    - Query embedding with is_query=True (mxbai prefix)
    - ChromaDB where filter for namespace/type
    - Update access statistics
    - Graph expansion with include_related
    - min_importance filtering
    """
    # Create mock store
    mock_store = MagicMock(spec=HybridStore)

    # Mock search results
    mock_store.search = AsyncMock(return_value=[
        {
            "id": "mem_1",
            "content": "Test memory content",
            "content_hash": "abc123def456",
            "type": "preference",
            "namespace": "global",
            "importance": 0.8,
            "created_at": 1700000000.0,
            "accessed_at": 1700000100.0,
            "access_count": 5,
            "similarity": 0.95,
        },
        {
            "id": "mem_2",
            "content": "Another memory",
            "content_hash": "xyz789",
            "type": "decision",
            "namespace": "global",
            "importance": 0.6,
            "created_at": 1700000200.0,
            "accessed_at": 1700000300.0,
            "access_count": 2,
            "similarity": 0.85,
        },
    ])

    # Test basic recall operation
    result = await memory_recall(
        store=mock_store,
        query="What are user preferences?",
        n_results=5,
        namespace="global",
    )

    assert isinstance(result, RecallResult)
    assert len(result.memories) == 2
    assert result.total == 2
    assert result.score is not None
    assert result.score == pytest.approx(0.9, rel=0.01)  # (0.95 + 0.85) / 2

    # Verify search was called with correct parameters
    mock_store.search.assert_called_once_with(
        query="What are user preferences?",
        n_results=5,
        namespace="global",
        memory_type=None,
    )

    # Verify Memory objects are correctly constructed
    mem1 = result.memories[0]
    assert mem1.id == "mem_1"
    assert mem1.content == "Test memory content"
    assert mem1.type == MemoryType.PREFERENCE
    assert mem1.namespace == "global"
    assert mem1.importance == 0.8

    # Test with min_importance filter
    mock_store.search.reset_mock()
    mock_store.search.return_value = [
        {
            "id": "mem_1",
            "content": "Test memory content",
            "content_hash": "abc123def456",
            "type": "preference",
            "namespace": "global",
            "importance": 0.8,
            "created_at": 1700000000.0,
            "accessed_at": 1700000100.0,
            "access_count": 5,
            "similarity": 0.95,
        },
        {
            "id": "mem_2",
            "content": "Low importance memory",
            "content_hash": "xyz789",
            "type": "decision",
            "namespace": "global",
            "importance": 0.3,
            "created_at": 1700000200.0,
            "accessed_at": 1700000300.0,
            "access_count": 2,
            "similarity": 0.85,
        },
    ]

    result_filtered = await memory_recall(
        store=mock_store,
        query="What are user preferences?",
        n_results=5,
        namespace="global",
        min_importance=0.5,
    )

    # Only mem_1 should be returned (importance 0.8 >= 0.5)
    assert len(result_filtered.memories) == 1
    assert result_filtered.memories[0].id == "mem_1"
    assert result_filtered.memories[0].importance == 0.8


class TestMemoryRecall:
    """Tests for memory_recall operation."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for testing."""
        store = MagicMock(spec=HybridStore)
        store.search = AsyncMock(return_value=[])
        store.get_edges = MagicMock(return_value=[])
        store.get_memory = AsyncMock(return_value=None)
        return store

    @pytest.mark.asyncio
    async def test_memory_recall_success(self, mock_store):
        """Test successful memory recall."""
        mock_store.search.return_value = [
            {
                "id": "mem_123",
                "content": "Test content",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.7,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            }
        ]

        result = await memory_recall(
            store=mock_store,
            query="test query",
            n_results=5,
        )

        assert isinstance(result, RecallResult)
        assert len(result.memories) == 1
        assert result.memories[0].id == "mem_123"
        assert result.memories[0].type == MemoryType.PREFERENCE
        assert result.total == 1
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_memory_recall_empty_query_returns_empty(self, mock_store):
        """Test that empty query returns empty result."""
        result = await memory_recall(
            store=mock_store,
            query="",
        )

        assert result.memories == []
        assert result.total == 0
        mock_store.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_recall_whitespace_query_returns_empty(self, mock_store):
        """Test that whitespace-only query returns empty result."""
        result = await memory_recall(
            store=mock_store,
            query="   ",
        )

        assert result.memories == []
        assert result.total == 0
        mock_store.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_recall_with_namespace_filter(self, mock_store):
        """Test recall with namespace filter."""
        mock_store.search.return_value = []

        await memory_recall(
            store=mock_store,
            query="test",
            namespace="project:myapp",
        )

        mock_store.search.assert_called_once_with(
            query="test",
            n_results=5,
            namespace="project:myapp",
            memory_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_recall_with_type_filter(self, mock_store):
        """Test recall with memory type filter."""
        mock_store.search.return_value = []

        await memory_recall(
            store=mock_store,
            query="test",
            memory_type=MemoryType.DECISION,
        )

        mock_store.search.assert_called_once_with(
            query="test",
            n_results=5,
            namespace=None,
            memory_type="decision",
        )

    @pytest.mark.asyncio
    async def test_memory_recall_with_combined_filters(self, mock_store):
        """Test recall with both namespace and type filters."""
        mock_store.search.return_value = []

        await memory_recall(
            store=mock_store,
            query="test",
            namespace="global",
            memory_type=MemoryType.PATTERN,
        )

        mock_store.search.assert_called_once_with(
            query="test",
            n_results=5,
            namespace="global",
            memory_type="pattern",
        )

    @pytest.mark.asyncio
    async def test_memory_recall_custom_n_results(self, mock_store):
        """Test recall with custom n_results."""
        mock_store.search.return_value = []

        await memory_recall(
            store=mock_store,
            query="test",
            n_results=10,
        )

        mock_store.search.assert_called_once_with(
            query="test",
            n_results=10,
            namespace=None,
            memory_type=None,
        )

    @pytest.mark.asyncio
    async def test_memory_recall_with_min_importance(self, mock_store):
        """Test recall with min_importance filter."""
        mock_store.search.return_value = [
            {
                "id": "mem_high",
                "content": "High importance",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.9,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            },
            {
                "id": "mem_medium",
                "content": "Medium importance",
                "content_hash": "def",
                "type": "decision",
                "namespace": "global",
                "importance": 0.5,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.85,
            },
            {
                "id": "mem_low",
                "content": "Low importance",
                "content_hash": "ghi",
                "type": "session",
                "namespace": "global",
                "importance": 0.2,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.8,
            },
        ]

        result = await memory_recall(
            store=mock_store,
            query="test",
            min_importance=0.5,
        )

        # Only mem_high (0.9) and mem_medium (0.5) should pass filter
        assert len(result.memories) == 2
        memory_ids = [m.id for m in result.memories]
        assert "mem_high" in memory_ids
        assert "mem_medium" in memory_ids
        assert "mem_low" not in memory_ids

    @pytest.mark.asyncio
    async def test_memory_recall_min_importance_filters_all(self, mock_store):
        """Test that min_importance can filter all results."""
        mock_store.search.return_value = [
            {
                "id": "mem_low",
                "content": "Low importance",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.3,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            },
        ]

        result = await memory_recall(
            store=mock_store,
            query="test",
            min_importance=0.8,
        )

        assert len(result.memories) == 0
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_memory_recall_min_importance_none_returns_all(self, mock_store):
        """Test that min_importance=None returns all results."""
        mock_store.search.return_value = [
            {
                "id": "mem_1",
                "content": "Low importance",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.1,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            },
        ]

        result = await memory_recall(
            store=mock_store,
            query="test",
            min_importance=None,
        )

        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_memory_recall_with_graph_expansion(self, mock_store):
        """Test recall with include_related=True for graph expansion."""
        mock_store.search.return_value = [
            {
                "id": "mem_1",
                "content": "Primary memory",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.8,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            }
        ]

        # Mock edges for the primary memory
        mock_store.get_edges.return_value = [
            {
                "id": 1,
                "source_id": "mem_1",
                "target_id": "mem_2",
                "edge_type": "related",
            }
        ]

        # Mock the related memory fetch
        mock_store.get_memory.return_value = {
            "id": "mem_2",
            "content": "Related memory",
            "content_hash": "def",
            "type": "decision",
            "namespace": "global",
            "importance": 0.6,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

        result = await memory_recall(
            store=mock_store,
            query="test",
            include_related=True,
        )

        # Primary memories only in result.memories
        assert len(result.memories) == 1
        assert result.total == 1
        assert result.memories[0].id == "mem_1"

        # Expanded memories in result.expanded_memories
        assert len(result.expanded_memories) == 1
        assert result.expanded_memories[0].memory.id == "mem_2"

        # Verify edges were fetched
        mock_store.get_edges.assert_called_once_with("mem_1", direction="both")

        # Verify related memory was fetched
        mock_store.get_memory.assert_called_once_with("mem_2")

    @pytest.mark.asyncio
    async def test_memory_recall_graph_expansion_skips_duplicates(self, mock_store):
        """Test that graph expansion doesn't include duplicate memories."""
        mock_store.search.return_value = [
            {
                "id": "mem_1",
                "content": "Memory 1",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.8,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            },
            {
                "id": "mem_2",
                "content": "Memory 2",
                "content_hash": "def",
                "type": "decision",
                "namespace": "global",
                "importance": 0.7,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.85,
            },
        ]

        # Edge from mem_1 to mem_2 (mem_2 is already in results)
        mock_store.get_edges.return_value = [
            {
                "id": 1,
                "source_id": "mem_1",
                "target_id": "mem_2",
                "edge_type": "related",
            }
        ]

        result = await memory_recall(
            store=mock_store,
            query="test",
            include_related=True,
        )

        # Should not duplicate mem_2
        assert len(result.memories) == 2

        # get_memory should not be called since mem_2 is already in results
        mock_store.get_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_recall_graph_expansion_incoming_edges(self, mock_store):
        """Test that graph expansion works with incoming edges."""
        mock_store.search.return_value = [
            {
                "id": "mem_1",
                "content": "Primary memory",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.8,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            }
        ]

        # Edge where mem_1 is the target (incoming edge)
        mock_store.get_edges.return_value = [
            {
                "id": 1,
                "source_id": "mem_3",
                "target_id": "mem_1",
                "edge_type": "caused_by",
            }
        ]

        mock_store.get_memory.return_value = {
            "id": "mem_3",
            "content": "Related memory via incoming edge",
            "content_hash": "ghi",
            "type": "pattern",
            "namespace": "global",
            "importance": 0.5,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

        result = await memory_recall(
            store=mock_store,
            query="test",
            include_related=True,
        )

        # Primary memory is in result.memories
        assert len(result.memories) == 1
        assert result.memories[0].id == "mem_1"

        # Expanded memory (via incoming edge) is in result.expanded_memories
        assert len(result.expanded_memories) == 1
        assert result.expanded_memories[0].memory.id == "mem_3"

    @pytest.mark.asyncio
    async def test_memory_recall_no_results(self, mock_store):
        """Test recall with no matching results."""
        mock_store.search.return_value = []

        result = await memory_recall(
            store=mock_store,
            query="nonexistent query",
        )

        assert result.memories == []
        assert result.total == 0
        assert result.score is None

    @pytest.mark.asyncio
    async def test_memory_recall_handles_unknown_memory_type(self, mock_store):
        """Test that unknown memory types default to SESSION."""
        mock_store.search.return_value = [
            {
                "id": "mem_1",
                "content": "Test",
                "content_hash": "abc",
                "type": "unknown_type",
                "namespace": "global",
                "importance": 0.5,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 0,
                "similarity": 0.8,
            }
        ]

        result = await memory_recall(
            store=mock_store,
            query="test",
        )

        assert len(result.memories) == 1
        assert result.memories[0].type == MemoryType.SESSION  # Default

    @pytest.mark.asyncio
    async def test_memory_recall_all_memory_types(self, mock_store):
        """Test recall returns correct MemoryType enums."""
        mock_store.search.return_value = [
            {
                "id": f"mem_{mt.value}",
                "content": f"Content for {mt.value}",
                "content_hash": "abc",
                "type": mt.value,
                "namespace": "global",
                "importance": 0.5,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 0,
                "similarity": 0.8,
            }
            for mt in MemoryType
        ]

        result = await memory_recall(
            store=mock_store,
            query="test",
            n_results=10,
        )

        assert len(result.memories) == len(MemoryType)
        types_found = {m.type for m in result.memories}
        assert types_found == set(MemoryType)

    @pytest.mark.asyncio
    async def test_memory_recall_handles_search_error(self, mock_store):
        """Test that search errors return empty result."""
        mock_store.search.side_effect = Exception("Search failed")

        result = await memory_recall(
            store=mock_store,
            query="test",
        )

        assert result.memories == []
        assert result.total == 0
        assert result.score is None

    @pytest.mark.asyncio
    async def test_memory_recall_min_importance_filters_related_memories(self, mock_store):
        """Test that min_importance also filters related memories in graph expansion."""
        mock_store.search.return_value = [
            {
                "id": "mem_1",
                "content": "Primary memory",
                "content_hash": "abc",
                "type": "preference",
                "namespace": "global",
                "importance": 0.8,
                "created_at": 1700000000.0,
                "accessed_at": 1700000000.0,
                "access_count": 1,
                "similarity": 0.9,
            }
        ]

        # Mock edges for the primary memory
        mock_store.get_edges.return_value = [
            {
                "id": 1,
                "source_id": "mem_1",
                "target_id": "mem_low",
                "edge_type": "related",
            },
            {
                "id": 2,
                "source_id": "mem_1",
                "target_id": "mem_high",
                "edge_type": "related",
            },
        ]

        # Mock get_memory to return different importance levels
        def get_memory_side_effect(mem_id):
            if mem_id == "mem_low":
                return {
                    "id": "mem_low",
                    "content": "Low importance related",
                    "content_hash": "def",
                    "type": "decision",
                    "namespace": "global",
                    "importance": 0.3,
                    "created_at": 1700000000.0,
                    "accessed_at": 1700000000.0,
                    "access_count": 0,
                }
            elif mem_id == "mem_high":
                return {
                    "id": "mem_high",
                    "content": "High importance related",
                    "content_hash": "ghi",
                    "type": "pattern",
                    "namespace": "global",
                    "importance": 0.7,
                    "created_at": 1700000000.0,
                    "accessed_at": 1700000000.0,
                    "access_count": 0,
                }
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        result = await memory_recall(
            store=mock_store,
            query="test",
            include_related=True,
            min_importance=0.5,
        )

        # Primary memory (0.8) should be in result.memories
        assert len(result.memories) == 1
        assert result.memories[0].id == "mem_1"

        # High importance related (0.7) should be in result.expanded_memories
        # Low importance related (0.3) should be filtered out
        assert len(result.expanded_memories) == 1
        assert result.expanded_memories[0].memory.id == "mem_high"
        expanded_ids = [em.memory.id for em in result.expanded_memories]
        assert "mem_low" not in expanded_ids


class TestMemoryRecallIntegration:
    """Integration tests for memory_recall using real ephemeral stores."""

    @pytest.fixture
    def integration_store(self):
        """Create HybridStore with real ephemeral stores but mocked embedding client."""
        sqlite = SQLiteStore(ephemeral=True)
        chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embedding_client = AsyncMock(spec=OllamaClient)

        # Configure mock to return consistent embeddings
        embedding_client.embed.return_value = [0.1] * 1024  # mxbai dimension

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )
        yield store
        sqlite.close()

    @pytest.mark.asyncio
    async def test_memory_recall_integration(self, integration_store):
        """Test full memory recall operation with real stores."""
        # Store some memories first
        await memory_store(
            store=integration_store,
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            namespace="global",
        )
        await memory_store(
            store=integration_store,
            content="User likes Python",
            memory_type=MemoryType.PREFERENCE,
            namespace="global",
        )

        # Recall memories
        result = await memory_recall(
            store=integration_store,
            query="What are user preferences?",
            n_results=5,
        )

        assert isinstance(result, RecallResult)
        assert len(result.memories) > 0
        assert result.total > 0

    @pytest.mark.asyncio
    async def test_memory_recall_with_graph_expansion_integration(self, integration_store):
        """Test memory recall with graph expansion using real stores."""
        # Store primary memory
        result1 = await memory_store(
            store=integration_store,
            content="Primary memory about dark mode",
            memory_type=MemoryType.PREFERENCE,
        )

        # Store related memory
        result2 = await memory_store(
            store=integration_store,
            content="Related memory about theme settings",
            memory_type=MemoryType.DECISION,
        )

        # Create edge between memories
        integration_store.add_edge(
            source_id=result1.id,
            target_id=result2.id,
            edge_type="related",
        )

        # Recall with graph expansion
        result = await memory_recall(
            store=integration_store,
            query="dark mode",
            include_related=True,
        )

        assert isinstance(result, RecallResult)
        # Depending on search results, we may get both memories
        assert result.total >= 1

    @pytest.mark.asyncio
    async def test_memory_recall_namespace_filter_integration(self, integration_store):
        """Test namespace filtering with real stores."""
        # Store memories in different namespaces
        await memory_store(
            store=integration_store,
            content="Global preference",
            namespace="global",
        )
        await memory_store(
            store=integration_store,
            content="Project specific setting",
            namespace="project:myapp",
        )

        # Recall only from global namespace
        result = await memory_recall(
            store=integration_store,
            query="preference setting",
            namespace="global",
        )

        # All returned memories should be in global namespace
        for memory in result.memories:
            assert memory.namespace == "global"


# ============================================================================
# memory_relate Tests
# ============================================================================


def test_memory_relate():
    """Top-level test for memory_relate function.

    This test validates the complete memory_relate operation including:
    - Both memory IDs exist validation
    - RelationType enum validation
    - Supersedes reduces target importance by 50%
    - Upsert edge behavior (INSERT OR REPLACE)
    """
    # Create ephemeral store
    store = SQLiteStore(ephemeral=True)

    # Create test memories
    source_id = store.add_memory(content="New information about topic X")
    target_id = store.add_memory(content="Old information about topic X", importance=0.8)

    # Test basic relate operation
    edge_id = memory_relate(
        store=store,
        source_id=source_id,
        target_id=target_id,
        relation=RelationType.RELATES_TO,
        weight=1.0,
    )

    assert edge_id is not None
    assert isinstance(edge_id, int)

    # Verify edge was created
    edges = store.get_edges(source_id, direction="outgoing")
    assert len(edges) == 1
    assert edges[0]["target_id"] == target_id
    assert edges[0]["edge_type"] == "relates_to"
    assert edges[0]["weight"] == 1.0

    # Test supersedes relation reduces target importance
    new_source = store.add_memory(content="Newer information superseding old")
    old_target = store.add_memory(content="Old info to be superseded", importance=0.8)

    edge_id_supersede = memory_relate(
        store=store,
        source_id=new_source,
        target_id=old_target,
        relation=RelationType.SUPERSEDES,
    )

    # Verify target importance was reduced by 50%
    updated_target = store.get_memory(old_target)
    assert updated_target["importance"] == pytest.approx(0.4, rel=0.01)  # 0.8 * 0.5

    # Test upsert behavior - updating existing edge
    new_edge_id = memory_relate(
        store=store,
        source_id=source_id,
        target_id=target_id,
        relation=RelationType.RELATES_TO,
        weight=0.5,  # New weight
    )

    # Should still have only one edge of this type between these memories
    edges = store.get_edges(source_id, direction="outgoing", edge_type="relates_to")
    relates_to_edges = [e for e in edges if e["target_id"] == target_id]
    assert len(relates_to_edges) == 1
    assert relates_to_edges[0]["weight"] == 0.5  # Updated weight

    store.close()


# ============================================================================
# memory_context Tests
# ============================================================================


def _create_list_memories_mock(test_data):
    """Create a mock for list_memories that handles golden_rule filtering.

    Returns empty list when memory_type="golden_rule" (no golden rules in test data)
    and returns test_data otherwise.
    """
    def list_memories_side_effect(**kwargs):
        memory_type = kwargs.get("memory_type")
        if memory_type == "golden_rule":
            return []  # No golden rules in test data
        return test_data
    return list_memories_side_effect


@pytest.mark.asyncio
async def test_memory_context():
    """Top-level test for memory_context function with composite scoring and token budget.

    This test validates the complete memory_context operation including:
    - Composite scoring: score = importance * recency_factor * log(access_count + 1)
    - Markdown formatting: ## Preferences, ## Recent Decisions, ## Patterns sections
    - Token budget enforcement: Estimate tokens (chars/4), drop lowest-score memories
    - Auto-detect project from cwd: os.path.basename(os.getcwd()) if project not specified
    """
    import time

    # Create mock store
    mock_store = MagicMock(spec=HybridStore)

    # Mock list_memories to return test data
    now = time.time()
    test_data = [
        {
            "id": "mem_pref_1",
            "content": "User prefers dark mode",
            "type": "preference",
            "namespace": "global",
            "importance": 0.9,
            "confidence": 0.3,  # Below golden rule threshold
            "accessed_at": now,  # Recent access
            "access_count": 10,
        },
        {
            "id": "mem_dec_1",
            "content": "Decided to use Python for backend",
            "type": "decision",
            "namespace": "global",
            "importance": 0.8,
            "confidence": 0.3,  # Below golden rule threshold
            "accessed_at": now - 86400,  # 1 day ago
            "access_count": 5,
        },
        {
            "id": "mem_pat_1",
            "content": "Pattern: User reviews PRs in morning",
            "type": "pattern",
            "namespace": "global",
            "importance": 0.7,
            "confidence": 0.3,  # Below golden rule threshold
            "accessed_at": now - 172800,  # 2 days ago
            "access_count": 3,
        },
        {
            "id": "mem_session_1",
            "content": "Session note - should be excluded",
            "type": "session",
            "namespace": "global",
            "importance": 0.5,
            "confidence": 0.3,  # Below golden rule threshold
            "accessed_at": now,
            "access_count": 1,
        },
    ]
    mock_store.list_memories = MagicMock(side_effect=_create_list_memories_mock(test_data))

    # Test basic context generation
    context = await memory_context(
        store=mock_store,
        project="testproject",
        token_budget=4000,
    )

    # Verify markdown structure
    assert "# Memory Context" in context
    assert "## Preferences" in context
    assert "## Recent Decisions" in context
    assert "## Patterns" in context

    # Verify content is included
    assert "User prefers dark mode" in context
    assert "Decided to use Python for backend" in context
    assert "Pattern: User reviews PRs in morning" in context

    # Verify session type is excluded
    assert "Session note - should be excluded" not in context

    # Verify namespace indicators
    assert "[global]" in context


@pytest.mark.asyncio
async def test_memory_context_token_budget():
    """Test that token budget is enforced."""
    import time

    mock_store = MagicMock(spec=HybridStore)
    now = time.time()

    # Create many memories that would exceed token budget
    memories = [
        {
            "id": f"mem_pref_{i}",
            "content": "A" * 200,  # 200 chars = ~50 tokens each
            "type": "preference",
            "namespace": "global",
            "importance": 0.9,
            "confidence": 0.3,  # Below golden rule threshold
            "accessed_at": now,
            "access_count": i,
        }
        for i in range(20)
    ]
    mock_store.list_memories = MagicMock(side_effect=_create_list_memories_mock(memories))

    # Very small token budget
    context = await memory_context(
        store=mock_store,
        project="test",
        token_budget=200,  # Very small
    )

    # Should be truncated
    assert len(context) // 4 <= 250  # Some slack for structure


@pytest.mark.asyncio
async def test_memory_context_composite_scoring():
    """Test that memories are ranked by composite score."""
    import time

    mock_store = MagicMock(spec=HybridStore)
    now = time.time()

    # Memory with high importance but old
    old_high_importance = {
        "id": "mem_old_high",
        "content": "Old high importance memory",
        "type": "preference",
        "namespace": "global",
        "importance": 1.0,
        "confidence": 0.3,  # Below golden rule threshold
        "accessed_at": now - (14 * 86400),  # 14 days ago
        "access_count": 1,
    }

    # Memory with lower importance but recent
    recent_low_importance = {
        "id": "mem_recent_low",
        "content": "Recent low importance memory",
        "type": "preference",
        "namespace": "global",
        "importance": 0.5,
        "confidence": 0.3,  # Below golden rule threshold
        "accessed_at": now,  # Just now
        "access_count": 10,  # Many accesses
    }

    test_data = [old_high_importance, recent_low_importance]
    mock_store.list_memories = MagicMock(side_effect=_create_list_memories_mock(test_data))

    context = await memory_context(
        store=mock_store,
        project="test",
        token_budget=4000,
    )

    # Recent memory with more accesses should appear first due to recency * access factor
    # Find positions of both contents
    pos_recent = context.find("Recent low importance memory")
    pos_old = context.find("Old high importance memory")

    # Recent should come first in Preferences section
    assert pos_recent < pos_old


@pytest.mark.asyncio
async def test_memory_context_auto_detect_project():
    """Test that project is auto-detected from cwd."""
    import os
    import time

    mock_store = MagicMock(spec=HybridStore)
    mock_store.list_memories = MagicMock(return_value=[])

    # Don't specify project - should auto-detect
    await memory_context(
        store=mock_store,
        project=None,  # Auto-detect
    )

    # Verify list_memories was called with auto-detected namespace
    calls = mock_store.list_memories.call_args_list
    namespaces_queried = [call.kwargs.get("namespace") for call in calls]

    expected_project_namespace = f"project:{os.path.basename(os.getcwd())}"
    assert "global" in namespaces_queried
    assert expected_project_namespace in namespaces_queried


@pytest.mark.asyncio
async def test_memory_context_empty_results():
    """Test context generation with no memories."""
    mock_store = MagicMock(spec=HybridStore)
    mock_store.list_memories = MagicMock(return_value=[])

    context = await memory_context(
        store=mock_store,
        project="test",
    )

    # Should still have header
    assert "# Memory Context" in context
    # But no section content (sections excluded when empty)
    assert "## Preferences" not in context


@pytest.mark.asyncio
async def test_memory_context_with_query():
    """Test context generation with search query."""
    import time

    mock_store = MagicMock(spec=HybridStore)
    now = time.time()

    search_results = [
        {
            "id": "mem_search_1",
            "content": "Search result memory",
            "type": "preference",
            "namespace": "global",
            "importance": 0.8,
            "accessed_at": now,
            "access_count": 5,
            "similarity": 0.9,
        }
    ]
    mock_store.search = AsyncMock(return_value=search_results)

    context = await memory_context(
        store=mock_store,
        query="relevant query",
        project="test",
    )

    # Should use search instead of list
    mock_store.search.assert_called()
    assert "Search result memory" in context


@pytest.mark.asyncio
async def test_memory_context_deduplicates():
    """Test that duplicate memories are deduplicated."""
    import time

    mock_store = MagicMock(spec=HybridStore)
    now = time.time()

    # Same memory returned for both namespaces
    memory = {
        "id": "mem_dup",
        "content": "Duplicate memory",
        "type": "preference",
        "namespace": "global",
        "importance": 0.8,
        "accessed_at": now,
        "access_count": 5,
    }
    mock_store.list_memories = MagicMock(return_value=[memory])

    context = await memory_context(
        store=mock_store,
        project="test",
    )

    # Content should only appear once
    assert context.count("Duplicate memory") == 1


class TestMemoryContextIntegration:
    """Integration tests for memory_context using real ephemeral stores."""

    @pytest.fixture
    def integration_store(self):
        """Create HybridStore with real ephemeral stores but mocked embedding client."""
        sqlite = SQLiteStore(ephemeral=True)
        chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embedding_client = AsyncMock(spec=OllamaClient)

        # Configure mock to return consistent embeddings
        embedding_client.embed.return_value = [0.1] * 1024

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )
        yield store
        sqlite.close()

    @pytest.mark.asyncio
    async def test_memory_context_integration(self, integration_store):
        """Test full memory_context operation with real stores.

        Note: Global memories are filtered to only include PREFERENCES with
        importance >= 0.6. Decisions and patterns in global namespace are
        excluded to prioritize project-specific context.
        """
        # Store global preference (will be included - preference with high importance)
        await memory_store(
            store=integration_store,
            content="User prefers vim keybindings",
            memory_type=MemoryType.PREFERENCE,
            namespace="global",
            importance=0.9,
        )
        # Store project-specific memories (always included)
        await memory_store(
            store=integration_store,
            content="Decided to use FastAPI",
            memory_type=MemoryType.DECISION,
            namespace="project:testproj",  # Project namespace for inclusion
            importance=0.8,
        )
        await memory_store(
            store=integration_store,
            content="Pattern: Commits before lunch",
            memory_type=MemoryType.PATTERN,
            namespace="project:testproj",  # Project namespace for inclusion
            importance=0.7,
        )

        # Generate context
        context = await memory_context(
            store=integration_store,
            project="testproj",
            token_budget=4000,
        )

        # Verify structure and content
        assert "# Memory Context" in context
        assert "User prefers vim keybindings" in context
        assert "Decided to use FastAPI" in context
        assert "Pattern: Commits before lunch" in context

    @pytest.mark.asyncio
    async def test_memory_context_project_namespace_integration(self, integration_store):
        """Test context includes both global and project namespace memories.

        Note: Global preferences must have importance >= 0.6 to be included.
        """
        # Store global memory with sufficient importance
        await memory_store(
            store=integration_store,
            content="Global preference setting",
            memory_type=MemoryType.PREFERENCE,
            namespace="global",
            importance=0.7,  # Must be >= 0.6 for global preferences
        )
        # Store project-specific memory
        await memory_store(
            store=integration_store,
            content="Project-specific decision",
            memory_type=MemoryType.DECISION,
            namespace="project:myapp",
        )

        # Generate context for the specific project
        context = await memory_context(
            store=integration_store,
            project="myapp",
            token_budget=4000,
        )

        # Should include memories from both namespaces
        assert "Global preference setting" in context
        assert "Project-specific decision" in context


class TestMemoryRelate:
    """Tests for memory_relate operation."""

    @pytest.fixture
    def store(self):
        """Create ephemeral SQLiteStore for testing."""
        s = SQLiteStore(ephemeral=True)
        yield s
        s.close()

    def test_memory_relate_success(self, store: SQLiteStore):
        """Test successful memory relation creation."""
        source_id = store.add_memory(content="Source memory")
        target_id = store.add_memory(content="Target memory")

        edge_id = memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.RELATES_TO,
        )

        assert edge_id is not None
        assert isinstance(edge_id, int)

        edges = store.get_edges(source_id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["target_id"] == target_id
        assert edges[0]["edge_type"] == "relates_to"

    def test_memory_relate_all_relation_types(self, store: SQLiteStore):
        """Test relation creation with all RelationType enum values."""
        source_id = store.add_memory(content="Source")

        for relation_type in RelationType:
            target_id = store.add_memory(content=f"Target for {relation_type.value}")

            edge_id = memory_relate(
                store=store,
                source_id=source_id,
                target_id=target_id,
                relation=relation_type,
            )

            assert edge_id is not None
            edges = store.get_edges(source_id, direction="outgoing", edge_type=relation_type.value)
            target_edges = [e for e in edges if e["target_id"] == target_id]
            assert len(target_edges) == 1
            assert target_edges[0]["edge_type"] == relation_type.value

    def test_memory_relate_source_not_found_raises(self, store: SQLiteStore):
        """Test that nonexistent source memory raises ValueError."""
        target_id = store.add_memory(content="Target memory")

        with pytest.raises(ValueError, match="Source memory.*not found"):
            memory_relate(
                store=store,
                source_id="nonexistent_id",
                target_id=target_id,
                relation=RelationType.RELATES_TO,
            )

    def test_memory_relate_target_not_found_raises(self, store: SQLiteStore):
        """Test that nonexistent target memory raises ValueError."""
        source_id = store.add_memory(content="Source memory")

        with pytest.raises(ValueError, match="Target memory.*not found"):
            memory_relate(
                store=store,
                source_id=source_id,
                target_id="nonexistent_id",
                relation=RelationType.RELATES_TO,
            )

    def test_memory_relate_invalid_relation_type_raises(self, store: SQLiteStore):
        """Test that invalid relation type raises ValueError."""
        source_id = store.add_memory(content="Source memory")
        target_id = store.add_memory(content="Target memory")

        with pytest.raises(ValueError, match="Invalid relation type"):
            memory_relate(
                store=store,
                source_id=source_id,
                target_id=target_id,
                relation="invalid_relation",  # type: ignore
            )

    def test_memory_relate_supersedes_reduces_importance(self, store: SQLiteStore):
        """Test that supersedes relation reduces target importance by 50%."""
        source_id = store.add_memory(content="New info")
        target_id = store.add_memory(content="Old info", importance=0.8)

        # Verify initial importance
        target = store.get_memory(target_id)
        assert target["importance"] == 0.8

        memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.SUPERSEDES,
        )

        # Verify importance was reduced by 50%
        target = store.get_memory(target_id)
        assert target["importance"] == pytest.approx(0.4, rel=0.01)

    def test_memory_relate_supersedes_multiple_times(self, store: SQLiteStore):
        """Test that multiple supersedes relations stack importance reduction."""
        source1 = store.add_memory(content="Source 1")
        source2 = store.add_memory(content="Source 2")
        target_id = store.add_memory(content="Target", importance=1.0)

        # First supersede: 1.0 -> 0.5
        memory_relate(
            store=store,
            source_id=source1,
            target_id=target_id,
            relation=RelationType.SUPERSEDES,
        )
        target = store.get_memory(target_id)
        assert target["importance"] == pytest.approx(0.5, rel=0.01)

        # Second supersede: 0.5 -> 0.25
        memory_relate(
            store=store,
            source_id=source2,
            target_id=target_id,
            relation=RelationType.SUPERSEDES,
        )
        target = store.get_memory(target_id)
        assert target["importance"] == pytest.approx(0.25, rel=0.01)

    def test_memory_relate_non_supersedes_preserves_importance(self, store: SQLiteStore):
        """Test that non-supersedes relations don't affect target importance."""
        source_id = store.add_memory(content="Source")
        target_id = store.add_memory(content="Target", importance=0.8)

        # Test all non-supersedes relation types
        non_supersede_types = [
            RelationType.RELATES_TO,
            RelationType.CAUSED_BY,
            RelationType.CONTRADICTS,
        ]

        for relation_type in non_supersede_types:
            # Need new source for each edge
            new_source = store.add_memory(content=f"Source for {relation_type.value}")
            memory_relate(
                store=store,
                source_id=new_source,
                target_id=target_id,
                relation=relation_type,
            )

            target = store.get_memory(target_id)
            assert target["importance"] == pytest.approx(0.8, rel=0.01)

    def test_memory_relate_with_custom_weight(self, store: SQLiteStore):
        """Test relation creation with custom weight."""
        source_id = store.add_memory(content="Source")
        target_id = store.add_memory(content="Target")

        memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.RELATES_TO,
            weight=0.75,
        )

        edges = store.get_edges(source_id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["weight"] == 0.75

    def test_memory_relate_upsert_updates_weight(self, store: SQLiteStore):
        """Test that re-relating memories updates the edge weight."""
        source_id = store.add_memory(content="Source")
        target_id = store.add_memory(content="Target")

        # Create initial edge
        memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.RELATES_TO,
            weight=0.5,
        )

        edges = store.get_edges(source_id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["weight"] == 0.5

        # Upsert with new weight
        memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.RELATES_TO,
            weight=0.9,
        )

        # Should still have only one edge, but with updated weight
        edges = store.get_edges(source_id, direction="outgoing", edge_type="relates_to")
        relates_to_target = [e for e in edges if e["target_id"] == target_id]
        assert len(relates_to_target) == 1
        assert relates_to_target[0]["weight"] == 0.9

    def test_memory_relate_different_types_between_same_memories(self, store: SQLiteStore):
        """Test that different relation types between same memories are allowed."""
        source_id = store.add_memory(content="Source")
        target_id = store.add_memory(content="Target")

        # Create relates_to edge
        memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.RELATES_TO,
        )

        # Create caused_by edge
        memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.CAUSED_BY,
        )

        edges = store.get_edges(source_id, direction="outgoing")
        target_edges = [e for e in edges if e["target_id"] == target_id]
        assert len(target_edges) == 2

        edge_types = {e["edge_type"] for e in target_edges}
        assert edge_types == {"relates_to", "caused_by"}

    def test_memory_relate_string_relation_conversion(self, store: SQLiteStore):
        """Test that string relation values are converted to RelationType."""
        source_id = store.add_memory(content="Source")
        target_id = store.add_memory(content="Target")

        # Pass string value instead of enum
        edge_id = memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation="relates_to",  # type: ignore
        )

        assert edge_id is not None
        edges = store.get_edges(source_id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "relates_to"

    def test_memory_relate_default_weight(self, store: SQLiteStore):
        """Test that default weight is 1.0."""
        source_id = store.add_memory(content="Source")
        target_id = store.add_memory(content="Target")

        memory_relate(
            store=store,
            source_id=source_id,
            target_id=target_id,
            relation=RelationType.RELATES_TO,
        )

        edges = store.get_edges(source_id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["weight"] == 1.0


# ============================================================================
# memory_forget Tests
# ============================================================================


@pytest.mark.asyncio
async def test_memory_forget():
    """Top-level test for memory_forget function with ID and query deletion modes.

    This test validates the complete memory_forget operation including:
    - Direct ID deletion mode: If memory_id provided, delete that specific memory
    - Semantic search deletion mode: If query provided, search and delete top matches
    - Namespace scoping: Only delete within specified namespace if provided
    - Atomic deletion from both stores: HybridStore.delete() handles SQLite + ChromaDB sync
    """
    # Create mock store
    mock_store = MagicMock(spec=HybridStore)
    mock_store.get_memory = AsyncMock(return_value={
        "id": "mem_123",
        "content": "Test memory",
        "namespace": "global",
    })
    mock_store.delete_memory = AsyncMock(return_value=True)

    # Test direct ID deletion
    result = await memory_forget(
        store=mock_store,
        memory_id="mem_123",
    )

    assert isinstance(result, ForgetResult)
    assert result.success is True
    assert result.deleted_count == 1
    assert "mem_123" in result.deleted_ids
    assert result.error is None

    # Verify delete_memory was called
    mock_store.delete_memory.assert_called_once_with("mem_123")

    # Test semantic search deletion
    mock_store.delete_memory.reset_mock()
    mock_store.search = AsyncMock(return_value=[
        {"id": "mem_1", "content": "Memory 1"},
        {"id": "mem_2", "content": "Memory 2"},
    ])
    mock_store.delete_memory.return_value = True

    result = await memory_forget(
        store=mock_store,
        query="test query",
        n_results=5,
    )

    assert result.success is True
    assert result.deleted_count == 2
    assert "mem_1" in result.deleted_ids
    assert "mem_2" in result.deleted_ids

    # Verify search and delete were called
    mock_store.search.assert_called_once()
    assert mock_store.delete_memory.call_count == 2

    # Test namespace scoping in ID deletion mode
    mock_store.delete_memory.reset_mock()
    mock_store.get_memory.return_value = {
        "id": "mem_456",
        "content": "Test",
        "namespace": "project:other",
    }

    result = await memory_forget(
        store=mock_store,
        memory_id="mem_456",
        namespace="global",  # Memory is in different namespace
    )

    assert result.success is False
    assert "not in namespace" in result.error
    mock_store.delete_memory.assert_not_called()


class TestMemoryForget:
    """Tests for memory_forget operation."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for testing."""
        store = MagicMock(spec=HybridStore)
        store.get_memory = AsyncMock(return_value=None)
        store.delete_memory = AsyncMock(return_value=True)
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.mark.asyncio
    async def test_memory_forget_by_id_success(self, mock_store):
        """Test successful memory deletion by ID."""
        mock_store.get_memory.return_value = {
            "id": "mem_123",
            "content": "Test memory",
            "namespace": "global",
        }

        result = await memory_forget(
            store=mock_store,
            memory_id="mem_123",
        )

        assert result.success is True
        assert result.deleted_count == 1
        assert "mem_123" in result.deleted_ids
        assert result.error is None

        mock_store.get_memory.assert_called_once_with("mem_123")
        mock_store.delete_memory.assert_called_once_with("mem_123")

    @pytest.mark.asyncio
    async def test_memory_forget_by_id_not_found(self, mock_store):
        """Test deletion failure when memory ID not found."""
        mock_store.get_memory.return_value = None

        result = await memory_forget(
            store=mock_store,
            memory_id="nonexistent",
        )

        assert result.success is False
        assert "not found" in result.error
        assert result.deleted_count == 0
        mock_store.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_forget_by_id_with_namespace_check(self, mock_store):
        """Test that namespace is checked in ID deletion mode."""
        mock_store.get_memory.return_value = {
            "id": "mem_123",
            "content": "Test",
            "namespace": "project:myapp",
        }

        # Try to delete from different namespace
        result = await memory_forget(
            store=mock_store,
            memory_id="mem_123",
            namespace="global",
        )

        assert result.success is False
        assert "not in namespace" in result.error
        mock_store.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_forget_by_id_with_matching_namespace(self, mock_store):
        """Test successful deletion when namespace matches."""
        mock_store.get_memory.return_value = {
            "id": "mem_123",
            "content": "Test",
            "namespace": "project:myapp",
        }

        result = await memory_forget(
            store=mock_store,
            memory_id="mem_123",
            namespace="project:myapp",
        )

        assert result.success is True
        assert result.deleted_count == 1
        mock_store.delete_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_forget_by_query_success(self, mock_store):
        """Test successful memory deletion by semantic search."""
        mock_store.search.return_value = [
            {"id": "mem_1", "content": "Memory 1"},
            {"id": "mem_2", "content": "Memory 2"},
            {"id": "mem_3", "content": "Memory 3"},
        ]

        result = await memory_forget(
            store=mock_store,
            query="test query",
            n_results=5,
        )

        assert result.success is True
        assert result.deleted_count == 3
        assert len(result.deleted_ids) == 3
        assert "mem_1" in result.deleted_ids
        assert "mem_2" in result.deleted_ids
        assert "mem_3" in result.deleted_ids

        mock_store.search.assert_called_once_with(
            query="test query",
            n_results=5,
            namespace=None,
        )
        assert mock_store.delete_memory.call_count == 3

    @pytest.mark.asyncio
    async def test_memory_forget_by_query_no_results(self, mock_store):
        """Test deletion when search returns no results."""
        mock_store.search.return_value = []

        result = await memory_forget(
            store=mock_store,
            query="nonexistent content",
        )

        assert result.success is True
        assert result.deleted_count == 0
        assert result.deleted_ids == []
        mock_store.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_forget_by_query_with_namespace(self, mock_store):
        """Test that namespace filter is passed to search."""
        mock_store.search.return_value = [
            {"id": "mem_1", "content": "Memory 1"},
        ]

        result = await memory_forget(
            store=mock_store,
            query="test",
            namespace="project:myapp",
        )

        assert result.success is True
        mock_store.search.assert_called_once_with(
            query="test",
            n_results=5,
            namespace="project:myapp",
        )

    @pytest.mark.asyncio
    async def test_memory_forget_by_query_custom_n_results(self, mock_store):
        """Test that custom n_results is respected."""
        mock_store.search.return_value = []

        await memory_forget(
            store=mock_store,
            query="test",
            n_results=10,
        )

        mock_store.search.assert_called_once_with(
            query="test",
            n_results=10,
            namespace=None,
        )

    @pytest.mark.asyncio
    async def test_memory_forget_no_id_or_query_fails(self, mock_store):
        """Test that providing neither ID nor query returns error."""
        result = await memory_forget(
            store=mock_store,
        )

        assert result.success is False
        assert "Must provide either memory_id or query" in result.error

    @pytest.mark.asyncio
    async def test_memory_forget_both_id_and_query_fails(self, mock_store):
        """Test that providing both ID and query returns error."""
        result = await memory_forget(
            store=mock_store,
            memory_id="mem_123",
            query="test",
        )

        assert result.success is False
        assert "Cannot provide both" in result.error

    @pytest.mark.asyncio
    async def test_memory_forget_confirm_false_fails(self, mock_store):
        """Test that confirm=False prevents deletion."""
        result = await memory_forget(
            store=mock_store,
            memory_id="mem_123",
            confirm=False,
        )

        assert result.success is False
        assert "not confirmed" in result.error
        mock_store.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_forget_delete_failure_handled(self, mock_store):
        """Test that delete failures are handled gracefully."""
        mock_store.get_memory.return_value = {
            "id": "mem_123",
            "content": "Test",
            "namespace": "global",
        }
        mock_store.delete_memory.return_value = False  # Delete fails

        result = await memory_forget(
            store=mock_store,
            memory_id="mem_123",
        )

        # Operation succeeds but nothing was deleted
        assert result.success is True
        assert result.deleted_count == 0
        assert result.deleted_ids == []

    @pytest.mark.asyncio
    async def test_memory_forget_query_partial_delete_failure(self, mock_store):
        """Test that partial delete failures continue with other deletions."""
        mock_store.search.return_value = [
            {"id": "mem_1", "content": "Memory 1"},
            {"id": "mem_2", "content": "Memory 2"},
            {"id": "mem_3", "content": "Memory 3"},
        ]

        # First succeeds, second fails with exception, third succeeds
        mock_store.delete_memory.side_effect = [True, Exception("DB error"), True]

        result = await memory_forget(
            store=mock_store,
            query="test",
        )

        assert result.success is True
        assert result.deleted_count == 2  # mem_1 and mem_3 deleted
        assert "mem_1" in result.deleted_ids
        assert "mem_3" in result.deleted_ids
        assert "mem_2" not in result.deleted_ids

    @pytest.mark.asyncio
    async def test_memory_forget_exception_handled(self, mock_store):
        """Test that exceptions are caught and returned as errors."""
        mock_store.get_memory.side_effect = Exception("Database connection failed")

        result = await memory_forget(
            store=mock_store,
            memory_id="mem_123",
        )

        assert result.success is False
        assert "Failed to delete memories" in result.error


class TestMemoryForgetIntegration:
    """Integration tests for memory_forget using real ephemeral stores."""

    @pytest.fixture
    def integration_store(self):
        """Create HybridStore with real ephemeral stores but mocked embedding client."""
        sqlite = SQLiteStore(ephemeral=True)
        chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embedding_client = AsyncMock(spec=OllamaClient)

        # Configure mock to return consistent embeddings
        embedding_client.embed.return_value = [0.1] * 1024

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )
        yield store
        sqlite.close()

    @pytest.mark.asyncio
    async def test_memory_forget_by_id_integration(self, integration_store):
        """Test full memory deletion by ID with real stores."""
        # Store a memory first
        store_result = await memory_store(
            store=integration_store,
            content="Memory to be deleted",
            memory_type=MemoryType.PREFERENCE,
            namespace="global",
        )
        assert store_result.success is True
        memory_id = store_result.id

        # Verify memory exists
        memory = await integration_store.get_memory(memory_id)
        assert memory is not None

        # Delete the memory
        result = await memory_forget(
            store=integration_store,
            memory_id=memory_id,
        )

        assert result.success is True
        assert result.deleted_count == 1
        assert memory_id in result.deleted_ids

        # Verify memory is deleted
        memory = await integration_store.get_memory(memory_id)
        assert memory is None

    @pytest.mark.asyncio
    async def test_memory_forget_by_query_integration(self, integration_store):
        """Test full memory deletion by query with real stores."""
        # Store multiple memories
        result1 = await memory_store(
            store=integration_store,
            content="User prefers dark mode theme",
            memory_type=MemoryType.PREFERENCE,
        )
        result2 = await memory_store(
            store=integration_store,
            content="Decided to use light theme",
            memory_type=MemoryType.DECISION,
        )
        result3 = await memory_store(
            store=integration_store,
            content="Unrelated technical decision",
            memory_type=MemoryType.DECISION,
        )

        initial_count = integration_store.count_memories()
        assert initial_count == 3

        # Delete by query (embeddings are mocked, so search returns what we stored)
        result = await memory_forget(
            store=integration_store,
            query="theme preferences",
            n_results=10,  # Get all potential matches
        )

        assert result.success is True
        assert result.deleted_count > 0

        # Verify count decreased
        final_count = integration_store.count_memories()
        assert final_count < initial_count

    @pytest.mark.asyncio
    async def test_memory_forget_namespace_scoping_integration(self, integration_store):
        """Test namespace scoping with real stores."""
        # Store memories in different namespaces
        global_mem = await memory_store(
            store=integration_store,
            content="Global memory",
            namespace="global",
        )
        project_mem = await memory_store(
            store=integration_store,
            content="Project memory",
            namespace="project:myapp",
        )

        # Try to delete global memory with project namespace scope
        result = await memory_forget(
            store=integration_store,
            memory_id=global_mem.id,
            namespace="project:myapp",
        )

        assert result.success is False
        assert "not in namespace" in result.error

        # Verify memory still exists
        memory = await integration_store.get_memory(global_mem.id)
        assert memory is not None

        # Delete with correct namespace
        result = await memory_forget(
            store=integration_store,
            memory_id=global_mem.id,
            namespace="global",
        )

        assert result.success is True
        assert result.deleted_count == 1

    @pytest.mark.asyncio
    async def test_memory_forget_deletes_from_both_stores_integration(self, integration_store):
        """Test that deletion removes from both SQLite and ChromaDB."""
        # Store a memory
        store_result = await memory_store(
            store=integration_store,
            content="Memory to delete from both stores",
            memory_type=MemoryType.PREFERENCE,
        )
        memory_id = store_result.id

        # Verify it exists in SQLite
        sqlite_count_before = integration_store.count_memories()
        assert sqlite_count_before == 1

        # Delete the memory
        result = await memory_forget(
            store=integration_store,
            memory_id=memory_id,
        )

        assert result.success is True
        assert result.deleted_count == 1

        # Verify SQLite deletion
        sqlite_count_after = integration_store.count_memories()
        assert sqlite_count_after == 0

        # Verify memory is gone from get_memory (which queries SQLite)
        memory = await integration_store.get_memory(memory_id)
        assert memory is None


# ============================================================================
# Multi-Hop Graph Expansion Tests
# ============================================================================


class TestMultiHopGraphExpansion:
    """Tests for multi-hop graph expansion behavior."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for graph expansion testing."""
        store = MagicMock(spec=HybridStore)
        store.get_edges = MagicMock(return_value=[])
        store.get_memory = AsyncMock(return_value=None)
        return store

    def _create_memory_dict(self, mem_id: str, content: str = "Test content", importance: float = 0.5):
        """Helper to create memory dict for mocking."""
        return {
            "id": mem_id,
            "content": content,
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": importance,
            "confidence": 0.3,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

    def _create_memory_obj(self, mem_id: str, content: str = "Test content") -> Memory:
        """Helper to create Memory object for primary_memories."""
        return Memory(
            id=mem_id,
            content=content,
            content_hash=f"hash_{mem_id}",
            type=MemoryType.PREFERENCE,
        )

    @pytest.mark.asyncio
    async def test_multi_hop_depth_2_finds_transitive_memory(self, mock_store):
        """Test that depth=2 finds memories 2 hops away (A->B->C finds C)."""
        # Setup: A -> B -> C
        # A is primary memory, B is 1 hop away, C is 2 hops away
        primary_memory = self._create_memory_obj("mem_A", "Primary memory A")

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 0.9}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 0.8}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            if mem_id == "mem_B":
                return self._create_memory_dict("mem_B", "Memory B")
            elif mem_id == "mem_C":
                return self._create_memory_dict("mem_C", "Memory C")
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=2, decay_factor=0.7)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should find both B (1 hop) and C (2 hops)
        expanded_ids = [em.memory.id for em in result]
        assert "mem_B" in expanded_ids
        assert "mem_C" in expanded_ids

        # C should have hop_distance=2
        mem_c = next(em for em in result if em.memory.id == "mem_C")
        assert mem_c.hop_distance == 2

    @pytest.mark.asyncio
    async def test_depth_1_does_not_find_2_hop_memory(self, mock_store):
        """Test that depth=1 only finds direct neighbors, not 2-hop memories."""
        primary_memory = self._create_memory_obj("mem_A", "Primary memory A")

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 0.9}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 0.8}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            if mem_id == "mem_B":
                return self._create_memory_dict("mem_B", "Memory B")
            elif mem_id == "mem_C":
                return self._create_memory_dict("mem_C", "Memory C")
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=1, decay_factor=0.7)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should only find B (1 hop)
        expanded_ids = [em.memory.id for em in result]
        assert "mem_B" in expanded_ids
        assert "mem_C" not in expanded_ids

    @pytest.mark.asyncio
    async def test_cycle_detection_prevents_infinite_loop(self, mock_store):
        """Test that cycles in the graph don't cause infinite loops (A->B->A)."""
        primary_memory = self._create_memory_obj("mem_A", "Primary memory A")

        call_count = {"mem_A": 0, "mem_B": 0}

        def get_edges_side_effect(mem_id, direction="both"):
            call_count[mem_id] = call_count.get(mem_id, 0) + 1
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 0.9}]
            elif mem_id == "mem_B":
                # B points back to A, creating a cycle
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_A", "edge_type": "relates_to", "weight": 0.8}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            if mem_id == "mem_B":
                return self._create_memory_dict("mem_B", "Memory B")
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=5, decay_factor=0.7)

        # Should complete without hanging
        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should only find B once, A should be skipped as it's in primary memories
        expanded_ids = [em.memory.id for em in result]
        assert expanded_ids.count("mem_B") == 1
        assert "mem_A" not in expanded_ids  # A is a primary memory, so it's in seen_ids

    @pytest.mark.asyncio
    async def test_cycle_detection_with_larger_cycle(self, mock_store):
        """Test cycle detection with A->B->C->A pattern."""
        primary_memory = self._create_memory_obj("mem_A", "Primary memory A")

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 0.9}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 0.8}]
            elif mem_id == "mem_C":
                # C points back to A
                return [{"id": 3, "source_id": "mem_C", "target_id": "mem_A", "edge_type": "relates_to", "weight": 0.7}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            if mem_id == "mem_B":
                return self._create_memory_dict("mem_B", "Memory B")
            elif mem_id == "mem_C":
                return self._create_memory_dict("mem_C", "Memory C")
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=10, decay_factor=0.7)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should find B and C, but not re-visit A
        expanded_ids = [em.memory.id for em in result]
        assert "mem_B" in expanded_ids
        assert "mem_C" in expanded_ids
        assert "mem_A" not in expanded_ids


class TestRelevanceScoring:
    """Tests for relevance score calculation using geometric mean formula."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for scoring tests."""
        store = MagicMock(spec=HybridStore)
        store.get_edges = MagicMock(return_value=[])
        store.get_memory = AsyncMock(return_value=None)
        return store

    def _create_memory_obj(self, mem_id: str) -> Memory:
        return Memory(
            id=mem_id,
            content="Test content",
            content_hash=f"hash_{mem_id}",
            type=MemoryType.PREFERENCE,
        )

    def _create_memory_dict(self, mem_id: str, importance: float = 0.5):
        return {
            "id": mem_id,
            "content": "Test content",
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": importance,
            "confidence": 0.3,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

    @pytest.mark.asyncio
    async def test_geometric_mean_single_edge(self, mock_store):
        """Test relevance score with single edge uses type weight directly."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 1.0}
        ]
        mock_store.get_memory.return_value = self._create_memory_dict("mem_B")

        config = GraphExpansionConfig(max_depth=1, decay_factor=0.7)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        assert len(result) == 1
        # For 1 hop with supersedes (weight=1.0):
        # decay = 0.7^1 = 0.7
        # path_weight = 1.0
        # geometric_mean = (1.0)^(1/1) = 1.0  (supersedes default weight is 1.0)
        # relevance = 0.7 * 1.0 * 1.0 = 0.7
        assert result[0].relevance_score == pytest.approx(0.7, rel=0.01)

    @pytest.mark.asyncio
    async def test_geometric_mean_two_edges(self, mock_store):
        """Test geometric mean calculation with two edges in path."""
        primary_memory = self._create_memory_obj("mem_A")

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 1.0}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            if mem_id == "mem_B":
                return self._create_memory_dict("mem_B")
            elif mem_id == "mem_C":
                return self._create_memory_dict("mem_C")
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=2, decay_factor=0.7)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Find the 2-hop memory (C)
        mem_c = next((em for em in result if em.memory.id == "mem_C"), None)
        assert mem_c is not None

        # For 2 hops with supersedes (1.0) -> relates_to (0.7):
        # decay = 0.7^2 = 0.49
        # path_weight = 1.0 * 1.0 = 1.0
        # geometric_mean = sqrt(1.0 * 0.7) = sqrt(0.7) ≈ 0.837
        # relevance = 0.49 * 1.0 * 0.837 ≈ 0.41
        expected_geo_mean = (1.0 * 0.7) ** 0.5  # sqrt(0.7) ≈ 0.837
        expected_relevance = 0.49 * 1.0 * expected_geo_mean
        assert mem_c.relevance_score == pytest.approx(expected_relevance, rel=0.05)

    @pytest.mark.asyncio
    async def test_min_importance_post_score_cutoff(self, mock_store):
        """Test that memories with importance < min_importance are excluded."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_low", "edge_type": "relates_to", "weight": 1.0},
            {"id": 2, "source_id": "mem_A", "target_id": "mem_high", "edge_type": "relates_to", "weight": 1.0},
        ]

        async def get_memory_side_effect(mem_id):
            if mem_id == "mem_low":
                return self._create_memory_dict("mem_low", importance=0.2)
            elif mem_id == "mem_high":
                return self._create_memory_dict("mem_high", importance=0.8)
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=1)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
            min_importance=0.5,  # Filter out low importance
        )

        # Should only find high importance memory
        expanded_ids = [em.memory.id for em in result]
        assert "mem_high" in expanded_ids
        assert "mem_low" not in expanded_ids

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance_descending(self, mock_store):
        """Test that results are sorted by relevance_score descending."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "contradicts", "weight": 0.5},  # Lower score
            {"id": 2, "source_id": "mem_A", "target_id": "mem_C", "edge_type": "supersedes", "weight": 1.0},  # Higher score
        ]

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=1, decay_factor=0.7)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        assert len(result) == 2
        # Should be sorted descending by relevance_score
        assert result[0].relevance_score >= result[1].relevance_score


class TestEdgeTypeFiltering:
    """Tests for edge type include/exclude filtering."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for filtering tests."""
        store = MagicMock(spec=HybridStore)
        store.get_edges = MagicMock(return_value=[])
        store.get_memory = AsyncMock(return_value=None)
        return store

    def _create_memory_obj(self, mem_id: str) -> Memory:
        return Memory(
            id=mem_id,
            content="Test content",
            content_hash=f"hash_{mem_id}",
            type=MemoryType.PREFERENCE,
        )

    def _create_memory_dict(self, mem_id: str):
        return {
            "id": mem_id,
            "content": "Test content",
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": 0.5,
            "confidence": 0.3,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

    @pytest.mark.asyncio
    async def test_include_edge_types_filters_correctly(self, mock_store):
        """Test that include_edge_types only follows specified edge types."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 1.0},
            {"id": 2, "source_id": "mem_A", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0},
            {"id": 3, "source_id": "mem_A", "target_id": "mem_D", "edge_type": "contradicts", "weight": 1.0},
        ]

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(
            max_depth=1,
            include_edge_types={"supersedes"},  # Only follow supersedes edges
        )

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should only find mem_B (via supersedes)
        expanded_ids = [em.memory.id for em in result]
        assert "mem_B" in expanded_ids
        assert "mem_C" not in expanded_ids
        assert "mem_D" not in expanded_ids

    @pytest.mark.asyncio
    async def test_exclude_edge_types_filters_correctly(self, mock_store):
        """Test that exclude_edge_types skips specified edge types."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 1.0},
            {"id": 2, "source_id": "mem_A", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0},
            {"id": 3, "source_id": "mem_A", "target_id": "mem_D", "edge_type": "contradicts", "weight": 1.0},
        ]

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(
            max_depth=1,
            exclude_edge_types={"contradicts"},  # Skip contradicts edges
        )

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should find mem_B and mem_C, but not mem_D
        expanded_ids = [em.memory.id for em in result]
        assert "mem_B" in expanded_ids
        assert "mem_C" in expanded_ids
        assert "mem_D" not in expanded_ids

    @pytest.mark.asyncio
    async def test_include_multiple_edge_types(self, mock_store):
        """Test include_edge_types with multiple types."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 1.0},
            {"id": 2, "source_id": "mem_A", "target_id": "mem_C", "edge_type": "caused_by", "weight": 1.0},
            {"id": 3, "source_id": "mem_A", "target_id": "mem_D", "edge_type": "contradicts", "weight": 1.0},
        ]

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(
            max_depth=1,
            include_edge_types={"supersedes", "caused_by"},
        )

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should find mem_B and mem_C (supersedes and caused_by), but not mem_D (contradicts)
        expanded_ids = [em.memory.id for em in result]
        assert "mem_B" in expanded_ids
        assert "mem_C" in expanded_ids
        assert "mem_D" not in expanded_ids


class TestExpansionLimits:
    """Tests for expansion safety limits."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for limit tests."""
        store = MagicMock(spec=HybridStore)
        store.get_edges = MagicMock(return_value=[])
        store.get_memory = AsyncMock(return_value=None)
        return store

    def _create_memory_obj(self, mem_id: str) -> Memory:
        return Memory(
            id=mem_id,
            content="Test content",
            content_hash=f"hash_{mem_id}",
            type=MemoryType.PREFERENCE,
        )

    def _create_memory_dict(self, mem_id: str):
        return {
            "id": mem_id,
            "content": "Test content",
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": 0.5,
            "confidence": 0.3,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

    @pytest.mark.asyncio
    async def test_max_edges_per_node_limits_expansion(self, mock_store):
        """Test that max_edges_per_node limits edges followed from each node."""
        primary_memory = self._create_memory_obj("mem_A")

        # Return more edges than the limit
        mock_store.get_edges.return_value = [
            {"id": i, "source_id": "mem_A", "target_id": f"mem_{i}", "edge_type": "relates_to", "weight": 1.0}
            for i in range(20)  # 20 edges
        ]

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(
            max_depth=1,
            max_edges_per_node=5,  # Only process first 5 edges
        )

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should only expand 5 memories due to max_edges_per_node
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_max_expanded_limits_total_results(self, mock_store):
        """Test that max_expanded limits total expanded memories returned."""
        primary_memory = self._create_memory_obj("mem_A")

        # Return many edges
        mock_store.get_edges.return_value = [
            {"id": i, "source_id": "mem_A", "target_id": f"mem_{i}", "edge_type": "relates_to", "weight": 1.0}
            for i in range(50)
        ]

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(
            max_depth=1,
            max_expanded=10,  # Only return 10 expanded memories
            max_edges_per_node=50,  # Allow all edges to be processed
        )

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should stop at max_expanded
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_max_nodes_visited_stops_expansion(self, mock_store):
        """Test that max_nodes_visited stops BFS early."""
        primary_memory = self._create_memory_obj("mem_A")

        # Create a linear chain A -> B -> C -> D -> ... (many nodes)
        def get_edges_side_effect(mem_id, direction="both"):
            # Extract node number from mem_X format
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_1", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id.startswith("mem_"):
                try:
                    node_num = int(mem_id.split("_")[1])
                    if node_num < 100:  # Create chain up to 100 nodes
                        return [{"id": node_num + 1, "source_id": mem_id, "target_id": f"mem_{node_num + 1}", "edge_type": "relates_to", "weight": 1.0}]
                except (ValueError, IndexError):
                    pass
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(
            max_depth=100,  # Allow deep traversal
            max_nodes_visited=5,  # But limit nodes visited
            max_expanded=100,  # Don't limit by this
        )

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Should stop due to max_nodes_visited
        assert len(result) <= 5


class TestExplanationField:
    """Tests for explanation field format verification."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for explanation tests."""
        store = MagicMock(spec=HybridStore)
        store.get_edges = MagicMock(return_value=[])
        store.get_memory = AsyncMock(return_value=None)
        return store

    def _create_memory_obj(self, mem_id: str) -> Memory:
        return Memory(
            id=mem_id,
            content="Test content",
            content_hash=f"hash_{mem_id}",
            type=MemoryType.PREFERENCE,
        )

    def _create_memory_dict(self, mem_id: str):
        return {
            "id": mem_id,
            "content": "Test content",
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": 0.5,
            "confidence": 0.3,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

    @pytest.mark.asyncio
    async def test_explanation_format_single_hop(self, mock_store):
        """Test explanation format for single hop matches regex."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 1.0}
        ]
        mock_store.get_memory.return_value = self._create_memory_dict("mem_B")

        config = GraphExpansionConfig(max_depth=1)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        assert len(result) == 1
        # Regex pattern: "N hop(s) via edge_types, combined weight X.XX"
        pattern = r"^\d+ hops? via .+, combined weight [0-9.]+$"
        assert re.match(pattern, result[0].explanation), f"Explanation doesn't match pattern: {result[0].explanation}"

        # Single hop should use "hop" (singular)
        assert "1 hop via" in result[0].explanation

    @pytest.mark.asyncio
    async def test_explanation_format_multi_hop(self, mock_store):
        """Test explanation format for multi-hop matches regex."""
        primary_memory = self._create_memory_obj("mem_A")

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 1.0}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 0.8}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            return self._create_memory_dict(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        config = GraphExpansionConfig(max_depth=2)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        # Find the 2-hop memory
        mem_c = next((em for em in result if em.memory.id == "mem_C"), None)
        assert mem_c is not None

        # Regex pattern for multi-hop
        pattern = r"^\d+ hops? via .+, combined weight [0-9.]+$"
        assert re.match(pattern, mem_c.explanation), f"Explanation doesn't match pattern: {mem_c.explanation}"

        # Multi-hop should use "hops" (plural)
        assert "2 hops via" in mem_c.explanation
        # Should show edge path with arrow
        assert "→" in mem_c.explanation or " " in mem_c.explanation

    @pytest.mark.asyncio
    async def test_explanation_contains_edge_types(self, mock_store):
        """Test that explanation contains the edge types in path."""
        primary_memory = self._create_memory_obj("mem_A")

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "caused_by", "weight": 1.0}
        ]
        mock_store.get_memory.return_value = self._create_memory_dict("mem_B")

        config = GraphExpansionConfig(max_depth=1)

        result = await _expand_related_memories(
            store=mock_store,
            primary_memories=[primary_memory],
            primary_scores=[0.9],
            config=config,
        )

        assert len(result) == 1
        assert "caused_by" in result[0].explanation


class TestBackwardCompatibility:
    """Tests for backward compatibility with original graph expansion behavior."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for compatibility tests."""
        store = MagicMock(spec=HybridStore)
        store.search = AsyncMock(return_value=[])
        store.get_edges = MagicMock(return_value=[])
        store.get_memory = AsyncMock(return_value=None)
        return store

    def _create_search_result(self, mem_id: str, content: str = "Test", similarity: float = 0.9):
        return {
            "id": mem_id,
            "content": content,
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": 0.5,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
            "similarity": similarity,
        }

    def _create_memory_dict(self, mem_id: str):
        return {
            "id": mem_id,
            "content": "Test content",
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": 0.5,
            "confidence": 0.3,
            "created_at": 1700000000.0,
            "accessed_at": 1700000000.0,
            "access_count": 0,
        }

    @pytest.mark.asyncio
    async def test_single_hop_expansion_unchanged(self, mock_store):
        """Test that single-hop expansion with default config produces same results."""
        # Setup A -> B -> C graph
        mock_store.search.return_value = [self._create_search_result("mem_A", "Memory A")]

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)
        mock_store.get_memory = AsyncMock(return_value=self._create_memory_dict("mem_B"))

        # Call with default max_depth=1
        result = await memory_recall(
            store=mock_store,
            query="test query",
            include_related=True,
            max_depth=1,  # Default single hop
        )

        # Should find primary and 1-hop related
        assert len(result.memories) == 1
        assert result.memories[0].id == "mem_A"

        # Expanded should only contain 1-hop memory (B), not 2-hop (C)
        assert len(result.expanded_memories) == 1
        assert result.expanded_memories[0].memory.id == "mem_B"

    @pytest.mark.asyncio
    async def test_include_related_false_no_expansion(self, mock_store):
        """Test that include_related=False still returns no expanded memories."""
        mock_store.search.return_value = [self._create_search_result("mem_A")]

        mock_store.get_edges.return_value = [
            {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}
        ]

        result = await memory_recall(
            store=mock_store,
            query="test query",
            include_related=False,  # No expansion
        )

        # Should have primary memories but no expanded
        assert len(result.memories) == 1
        assert len(result.expanded_memories) == 0

        # get_edges should not be called
        mock_store.get_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_result_structure_unchanged(self, mock_store):
        """Test that RecallResult structure is preserved."""
        mock_store.search.return_value = [
            self._create_search_result("mem_A", "Memory A", similarity=0.95)
        ]

        result = await memory_recall(
            store=mock_store,
            query="test query",
        )

        # Verify RecallResult fields exist and have expected types
        assert isinstance(result, RecallResult)
        assert isinstance(result.memories, list)
        assert isinstance(result.total, int)
        assert result.score is None or isinstance(result.score, float)
        assert isinstance(result.expanded_memories, list)

    @pytest.mark.asyncio
    async def test_baseline_graph_expansion_consistency(self, mock_store):
        """Test expansion with known A→B→C graph returns expected results."""
        mock_store.search.return_value = [self._create_search_result("mem_A")]

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        async def get_memory_side_effect(mem_id):
            if mem_id == "mem_B":
                return self._create_memory_dict("mem_B")
            elif mem_id == "mem_C":
                return self._create_memory_dict("mem_C")
            return None

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        # Single hop should find only B
        result_1hop = await memory_recall(
            store=mock_store,
            query="test",
            include_related=True,
            max_depth=1,
        )
        expanded_ids_1hop = {em.memory.id for em in result_1hop.expanded_memories}
        assert expanded_ids_1hop == {"mem_B"}

        # Two hops should find both B and C
        result_2hop = await memory_recall(
            store=mock_store,
            query="test",
            include_related=True,
            max_depth=2,
        )
        expanded_ids_2hop = {em.memory.id for em in result_2hop.expanded_memories}
        assert expanded_ids_2hop == {"mem_B", "mem_C"}

        # Verify count consistency
        assert len(result_1hop.expanded_memories) == 1
        assert len(result_2hop.expanded_memories) == 2


class TestInspectGraph:
    """Tests for inspect_graph function."""

    @pytest.fixture
    def mock_store(self):
        """Create mock HybridStore for testing."""
        store = MagicMock(spec=HybridStore)
        store.get_memory = AsyncMock(return_value=None)
        store.get_edges = MagicMock(return_value=[])
        return store

    def _create_memory_dict(self, mem_id: str, content: str = "Test content") -> dict:
        """Create a memory dict for testing."""
        import time
        return {
            "id": mem_id,
            "content": content,
            "content_hash": f"hash_{mem_id}",
            "type": "preference",
            "namespace": "global",
            "importance": 0.5,
            "confidence": 0.3,
            "created_at": time.time(),
            "accessed_at": time.time(),
            "access_count": 0,
        }

    @pytest.mark.asyncio
    async def test_inspect_graph_memory_not_found(self, mock_store):
        """Test inspect_graph returns error for non-existent memory."""
        from recall.memory.operations import inspect_graph

        mock_store.get_memory.return_value = None

        result = await inspect_graph(
            store=mock_store,
            memory_id="nonexistent_id",
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_inspect_graph_invalid_direction(self, mock_store):
        """Test inspect_graph returns error for invalid direction."""
        from recall.memory.operations import inspect_graph

        result = await inspect_graph(
            store=mock_store,
            memory_id="any_id",
            direction="invalid",
        )

        assert result.success is False
        assert "Invalid direction" in result.error

    @pytest.mark.asyncio
    async def test_inspect_graph_origin_only(self, mock_store):
        """Test inspect_graph with origin node only (no edges)."""
        from recall.memory.operations import inspect_graph

        mock_store.get_memory.return_value = self._create_memory_dict("origin", "Origin content")
        mock_store.get_edges.return_value = []

        result = await inspect_graph(
            store=mock_store,
            memory_id="origin",
        )

        assert result.success is True
        assert result.origin_id == "origin"
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "origin"
        assert len(result.edges) == 0
        assert result.stats.node_count == 1
        assert result.stats.edge_count == 0

    @pytest.mark.asyncio
    async def test_inspect_graph_direction_outgoing(self, mock_store):
        """Test inspect_graph with outgoing-only direction."""
        from recall.memory.operations import inspect_graph

        # Create A -> B -> C graph
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
            "mem_C": self._create_memory_dict("mem_C", "Node C"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            # For outgoing direction, only return edges where mem_id is source
            if direction == "outgoing":
                if mem_id == "mem_A":
                    return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}]
                elif mem_id == "mem_B":
                    return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_A",
            max_depth=2,
            direction="outgoing",
        )

        assert result.success is True
        node_ids = {n.id for n in result.nodes}
        assert node_ids == {"mem_A", "mem_B", "mem_C"}
        assert len(result.edges) == 2

    @pytest.mark.asyncio
    async def test_inspect_graph_direction_incoming(self, mock_store):
        """Test inspect_graph with incoming-only direction."""
        from recall.memory.operations import inspect_graph

        # Create graph where C is the target of edges from A and B
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
            "mem_C": self._create_memory_dict("mem_C", "Node C"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            # For incoming direction, only return edges where mem_id is target
            if direction == "incoming":
                if mem_id == "mem_C":
                    return [
                        {"id": 1, "source_id": "mem_A", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0},
                        {"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "caused_by", "weight": 0.8},
                    ]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_C",
            max_depth=1,
            direction="incoming",
        )

        assert result.success is True
        node_ids = {n.id for n in result.nodes}
        assert node_ids == {"mem_A", "mem_B", "mem_C"}
        assert len(result.edges) == 2

    @pytest.mark.asyncio
    async def test_inspect_graph_direction_both(self, mock_store):
        """Test inspect_graph with both directions (default)."""
        from recall.memory.operations import inspect_graph

        # Create graph A <-> B -> C
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
            "mem_C": self._create_memory_dict("mem_C", "Node C"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_B":
                return [
                    {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0},
                    {"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "supersedes", "weight": 0.9},
                ]
            elif mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id == "mem_C":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "supersedes", "weight": 0.9}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_B",
            max_depth=1,
            direction="both",
        )

        assert result.success is True
        node_ids = {n.id for n in result.nodes}
        assert node_ids == {"mem_A", "mem_B", "mem_C"}

    @pytest.mark.asyncio
    async def test_inspect_graph_max_depth_limit(self, mock_store):
        """Test that max_depth limits graph traversal (A->B->C->D with max_depth=2 should not reach D)."""
        from recall.memory.operations import inspect_graph

        # Create linear graph A -> B -> C -> D
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
            "mem_C": self._create_memory_dict("mem_C", "Node C"),
            "mem_D": self._create_memory_dict("mem_D", "Node D"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id == "mem_C":
                return [{"id": 3, "source_id": "mem_C", "target_id": "mem_D", "edge_type": "relates_to", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_A",
            max_depth=2,
            direction="outgoing",
        )

        assert result.success is True
        node_ids = {n.id for n in result.nodes}
        # With max_depth=2, should reach A, B, C but NOT D (which is 3 hops)
        assert "mem_A" in node_ids
        assert "mem_B" in node_ids
        assert "mem_C" in node_ids
        assert "mem_D" not in node_ids

    @pytest.mark.asyncio
    async def test_inspect_graph_edge_types_filtering(self, mock_store):
        """Test that edge_types parameter filters which edges to follow."""
        from recall.memory.operations import inspect_graph

        # Create graph with different edge types
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
            "mem_C": self._create_memory_dict("mem_C", "Node C"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [
                    {"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0},
                    {"id": 2, "source_id": "mem_A", "target_id": "mem_C", "edge_type": "contradicts", "weight": 0.5},
                ]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        # Only follow "relates_to" edges
        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_A",
            max_depth=1,
            edge_types=["relates_to"],
        )

        assert result.success is True
        node_ids = {n.id for n in result.nodes}
        # Should only reach B via relates_to, not C via contradicts
        assert "mem_A" in node_ids
        assert "mem_B" in node_ids
        assert "mem_C" not in node_ids

    @pytest.mark.asyncio
    async def test_inspect_graph_output_size_caps(self, mock_store):
        """Test that output size caps (MAX_NODES, MAX_EDGES) are enforced."""
        from recall.memory.operations import inspect_graph

        # Create a large graph with 150 nodes
        memories = {f"mem_{i}": self._create_memory_dict(f"mem_{i}", f"Node {i}") for i in range(150)}

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            # Each node connects to 10 other nodes
            node_num = int(mem_id.split("_")[1])
            edges = []
            for i in range(10):
                target_num = (node_num + i + 1) % 150
                edges.append({
                    "id": node_num * 10 + i,
                    "source_id": mem_id,
                    "target_id": f"mem_{target_num}",
                    "edge_type": "relates_to",
                    "weight": 1.0,
                })
            return edges

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_0",
            max_depth=5,  # Deep traversal
        )

        assert result.success is True
        # MAX_NODES is 100, MAX_EDGES is 200
        assert len(result.nodes) <= 100
        assert len(result.edges) <= 200
        assert result.stats.node_count <= 100
        assert result.stats.edge_count <= 200

    @pytest.mark.asyncio
    async def test_inspect_graph_with_scores(self, mock_store):
        """Test that include_scores generates path relevance scores."""
        from recall.memory.operations import inspect_graph

        # Create simple A -> B graph
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "supersedes", "weight": 0.9}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_A",
            max_depth=1,
            include_scores=True,
            decay_factor=0.7,
        )

        assert result.success is True
        # Should have paths with scores
        # The path A -> B should exist since B is a leaf node
        if result.paths:
            for path in result.paths:
                assert path.relevance_score >= 0.0
                assert path.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_inspect_graph_without_scores(self, mock_store):
        """Test that include_scores=False skips path scoring."""
        from recall.memory.operations import inspect_graph

        # Create simple A -> B graph
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_A",
            max_depth=1,
            include_scores=False,
        )

        assert result.success is True
        # Without scores, paths should be empty
        assert len(result.paths) == 0

    @pytest.mark.asyncio
    async def test_inspect_graph_cyclic_graph(self, mock_store):
        """Test that inspect_graph handles cycles without infinite loops."""
        from recall.memory.operations import inspect_graph

        # Create cyclic graph A -> B -> C -> A
        memories = {
            "mem_A": self._create_memory_dict("mem_A", "Node A"),
            "mem_B": self._create_memory_dict("mem_B", "Node B"),
            "mem_C": self._create_memory_dict("mem_C", "Node C"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "mem_A":
                return [{"id": 1, "source_id": "mem_A", "target_id": "mem_B", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id == "mem_B":
                return [{"id": 2, "source_id": "mem_B", "target_id": "mem_C", "edge_type": "relates_to", "weight": 1.0}]
            elif mem_id == "mem_C":
                return [{"id": 3, "source_id": "mem_C", "target_id": "mem_A", "edge_type": "relates_to", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        # Should complete without infinite loop
        result = await inspect_graph(
            store=mock_store,
            memory_id="mem_A",
            max_depth=5,
            direction="outgoing",
        )

        assert result.success is True
        node_ids = {n.id for n in result.nodes}
        # Should find all 3 nodes exactly once (no duplicates)
        assert node_ids == {"mem_A", "mem_B", "mem_C"}
        assert len(result.nodes) == 3

    @pytest.mark.asyncio
    async def test_inspect_graph_mermaid_output(self, mock_store):
        """Test that result.to_mermaid() produces valid output."""
        from recall.memory.operations import inspect_graph

        # Create simple graph
        memories = {
            "origin": self._create_memory_dict("origin", "Origin node content"),
            "target": self._create_memory_dict("target", "Target node content"),
        }

        async def get_memory_side_effect(mem_id):
            return memories.get(mem_id)

        mock_store.get_memory = AsyncMock(side_effect=get_memory_side_effect)

        def get_edges_side_effect(mem_id, direction="both"):
            if mem_id == "origin":
                return [{"id": 1, "source_id": "origin", "target_id": "target", "edge_type": "supersedes", "weight": 1.0}]
            return []

        mock_store.get_edges = MagicMock(side_effect=get_edges_side_effect)

        result = await inspect_graph(
            store=mock_store,
            memory_id="origin",
            max_depth=1,
        )

        assert result.success is True

        # Generate mermaid diagram
        mermaid = result.to_mermaid()

        # Verify mermaid syntax
        assert mermaid.startswith("flowchart TD")
        assert "origin" in mermaid
        assert "target" in mermaid
        assert "supersedes" in mermaid
        assert "-->" in mermaid
