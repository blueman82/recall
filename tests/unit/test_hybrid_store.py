"""Unit tests for HybridStore - coordinated SQLite/ChromaDB operations."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recall.storage.hybrid import HybridStore, HybridStoreError
from recall.storage.sqlite import SQLiteStore, SQLiteStoreError
from recall.storage.chromadb import ChromaStore, StorageError as ChromaStorageError
from recall.embedding.ollama import OllamaClient, EmbeddingError


def unique_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


class TestHybridStoreInit:
    """Tests for HybridStore initialization."""

    def test_init_with_components(self):
        """Test initialization with component stores."""
        sqlite = SQLiteStore(ephemeral=True)
        chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embedding_client = OllamaClient()

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )

        assert store._sqlite is sqlite
        assert store._chroma is chroma
        assert store._embedding_client is embedding_client
        assert store._sync_on_write is True
        assert store._chroma_available is True

        sqlite.close()

    def test_init_sync_on_write_false(self):
        """Test initialization with sync_on_write disabled."""
        sqlite = SQLiteStore(ephemeral=True)
        chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embedding_client = OllamaClient()

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
            sync_on_write=False,
        )

        assert store._sync_on_write is False
        sqlite.close()


class TestHybridStoreCreate:
    """Tests for HybridStore.create factory method."""

    @pytest.mark.asyncio
    async def test_create_ephemeral(self):
        """Test creating ephemeral HybridStore."""
        store = await HybridStore.create(
            ephemeral=True,
            collection_name=unique_collection_name(),
        )

        assert store._sqlite.ephemeral is True
        assert store._chroma.ephemeral is True
        await store.close()

    @pytest.mark.asyncio
    async def test_create_with_custom_options(self):
        """Test creating HybridStore with custom options."""
        store = await HybridStore.create(
            ephemeral=True,
            collection_name=unique_collection_name(),
            ollama_host="http://localhost:11434",
            ollama_model="mxbai-embed-large",
            sync_on_write=False,
        )

        assert store._sync_on_write is False
        assert store._embedding_client.host == "http://localhost:11434"
        assert store._embedding_client.model == "mxbai-embed-large"
        await store.close()


class TestAddMemory:
    """Tests for HybridStore.add_memory."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock stores for testing."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        # Configure mock collection
        chroma._collection = MagicMock()

        return sqlite, chroma, embedding_client

    @pytest.mark.asyncio
    async def test_add_memory_success(self, mock_stores):
        """Test successful memory addition with ChromaDB sync."""
        sqlite, chroma, embedding_client = mock_stores

        sqlite.add_memory.return_value = "mem_123"
        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "mem_123", "operation": "add"}
        ]
        embedding_client.embed.return_value = [0.1, 0.2, 0.3]

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )

        result = await store.add_memory(
            content="Test memory",
            memory_type="fact",
            namespace="default",
            importance=0.8,
        )

        assert result == "mem_123"

        # Verify SQLite was called
        sqlite.add_memory.assert_called_once_with(
            content="Test memory",
            memory_type="fact",
            namespace="default",
            importance=0.8,
            metadata=None,
            memory_id=None,
        )

        # Verify embedding was generated (not a query)
        embedding_client.embed.assert_called_once_with("Test memory", is_query=False)

        # Verify ChromaDB was called
        chroma._collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memory_sqlite_failure_raises(self, mock_stores):
        """Test that SQLite failure raises HybridStoreError."""
        sqlite, chroma, embedding_client = mock_stores

        sqlite.add_memory.side_effect = SQLiteStoreError("Database error")

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )

        with pytest.raises(HybridStoreError, match="Failed to add memory"):
            await store.add_memory(content="Test")

    @pytest.mark.asyncio
    async def test_add_memory_chroma_failure_non_fatal(self, mock_stores):
        """Test that ChromaDB failure is non-fatal (operation succeeds)."""
        sqlite, chroma, embedding_client = mock_stores

        sqlite.add_memory.return_value = "mem_123"
        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "mem_123", "operation": "add"}
        ]
        embedding_client.embed.return_value = [0.1, 0.2, 0.3]
        chroma._collection.add.side_effect = Exception("ChromaDB error")

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )

        # Should not raise - ChromaDB failures are non-fatal
        result = await store.add_memory(content="Test")

        assert result == "mem_123"
        # ChromaDB should be marked as unavailable
        assert store._chroma_available is False

    @pytest.mark.asyncio
    async def test_add_memory_embedding_failure_non_fatal(self, mock_stores):
        """Test that embedding failure is non-fatal."""
        sqlite, chroma, embedding_client = mock_stores

        sqlite.add_memory.return_value = "mem_123"
        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "mem_123", "operation": "add"}
        ]
        embedding_client.embed.side_effect = EmbeddingError("Ollama unavailable")

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
        )

        # Should not raise - embedding failures are non-fatal
        result = await store.add_memory(content="Test")

        assert result == "mem_123"
        # ChromaDB add should not have been called
        chroma._collection.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_memory_sync_disabled(self, mock_stores):
        """Test that ChromaDB sync is skipped when disabled."""
        sqlite, chroma, embedding_client = mock_stores

        sqlite.add_memory.return_value = "mem_123"

        store = HybridStore(
            sqlite_store=sqlite,
            chroma_store=chroma,
            embedding_client=embedding_client,
            sync_on_write=False,
        )

        result = await store.add_memory(content="Test")

        assert result == "mem_123"
        # No embedding should be generated
        embedding_client.embed.assert_not_called()
        # ChromaDB should not be called
        chroma._collection.add.assert_not_called()


class TestGetMemory:
    """Tests for HybridStore.get_memory."""

    @pytest.mark.asyncio
    async def test_get_memory_found(self):
        """Test getting existing memory."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        expected = {
            "id": "mem_123",
            "content": "Test",
            "type": "fact",
            "namespace": "default",
        }
        sqlite.get_memory.return_value = expected

        store = HybridStore(sqlite, chroma, embedding_client)
        result = await store.get_memory("mem_123")

        assert result == expected
        sqlite.get_memory.assert_called_once_with("mem_123")

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        """Test getting non-existent memory."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.get_memory.return_value = None

        store = HybridStore(sqlite, chroma, embedding_client)
        result = await store.get_memory("nonexistent")

        assert result is None


class TestDeleteMemory:
    """Tests for HybridStore.delete_memory."""

    @pytest.mark.asyncio
    async def test_delete_memory_success(self):
        """Test successful memory deletion."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.delete_memory.return_value = True

        store = HybridStore(sqlite, chroma, embedding_client)
        result = await store.delete_memory("mem_123")

        assert result is True
        sqlite.delete_memory.assert_called_once_with("mem_123")
        chroma.delete.assert_called_once_with(["mem_123"])

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self):
        """Test deleting non-existent memory."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.delete_memory.return_value = False

        store = HybridStore(sqlite, chroma, embedding_client)
        result = await store.delete_memory("nonexistent")

        assert result is False
        # ChromaDB delete should not be called
        chroma.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_memory_chroma_failure_non_fatal(self):
        """Test that ChromaDB failure during delete is non-fatal."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.delete_memory.return_value = True
        chroma.delete.side_effect = ChromaStorageError("Delete failed")

        store = HybridStore(sqlite, chroma, embedding_client)
        result = await store.delete_memory("mem_123")

        # Should succeed even if ChromaDB fails
        assert result is True


class TestSearch:
    """Tests for HybridStore.search."""

    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful semantic search."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        # Mock embedding generation
        embedding_client.embed.return_value = [0.1, 0.2, 0.3]

        # Mock ChromaDB search results
        chroma.query.return_value = {
            "ids": ["mem_1", "mem_2"],
            "documents": ["doc 1", "doc 2"],
            "metadatas": [{"namespace": "default"}, {"namespace": "default"}],
            "distances": [0.1, 0.3],
        }

        # Mock SQLite enrichment
        sqlite.get_memory.side_effect = [
            {"id": "mem_1", "content": "doc 1", "importance": 0.8},
            {"id": "mem_2", "content": "doc 2", "importance": 0.6},
        ]
        sqlite.touch_memory.return_value = True

        store = HybridStore(sqlite, chroma, embedding_client)
        results = await store.search("test query", n_results=2)

        assert len(results) == 2
        assert results[0]["id"] == "mem_1"
        assert "similarity" in results[0]

        # Verify query embedding used is_query=True
        embedding_client.embed.assert_called_once_with("test query", is_query=True)

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with namespace and type filters."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        embedding_client.embed.return_value = [0.1, 0.2, 0.3]
        chroma.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }

        store = HybridStore(sqlite, chroma, embedding_client)
        await store.search("query", namespace="ns1", memory_type="fact")

        # Verify filter was passed to ChromaDB
        chroma.query.assert_called_once()
        call_args = chroma.query.call_args
        assert call_args.kwargs["where"] == {"namespace": "ns1", "type": "fact"}

    @pytest.mark.asyncio
    async def test_search_fallback_to_fts(self):
        """Test fallback to FTS when ChromaDB fails."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        embedding_client.embed.return_value = [0.1, 0.2, 0.3]
        chroma.query.side_effect = ChromaStorageError("ChromaDB unavailable")
        sqlite.search_fts.return_value = [
            {"id": "mem_1", "content": "doc 1", "rank": -1.5}
        ]

        store = HybridStore(sqlite, chroma, embedding_client)
        results = await store.search("test query")

        assert len(results) == 1
        assert results[0]["id"] == "mem_1"
        # ChromaDB should be marked unavailable
        assert store._chroma_available is False

    @pytest.mark.asyncio
    async def test_search_embedding_failure_raises(self):
        """Test that embedding failure raises error."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        embedding_client.embed.side_effect = EmbeddingError("Ollama down")

        store = HybridStore(sqlite, chroma, embedding_client)

        with pytest.raises(HybridStoreError, match="Failed to generate query embedding"):
            await store.search("test query")


class TestSearchFTS:
    """Tests for HybridStore.search_fts."""

    def test_search_fts_success(self):
        """Test FTS search delegation."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.search_fts.return_value = [
            {"id": "mem_1", "content": "test", "rank": -1.5}
        ]

        store = HybridStore(sqlite, chroma, embedding_client)
        results = store.search_fts("test", namespace="ns1", limit=5)

        assert len(results) == 1
        sqlite.search_fts.assert_called_once_with(
            query="test",
            namespace="ns1",
            memory_type=None,
            limit=5,
        )


class TestOutboxProcessing:
    """Tests for outbox processing."""

    @pytest.mark.asyncio
    async def test_process_outbox_success(self):
        """Test successful outbox processing."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        # Setup mock collection
        chroma._collection = MagicMock()

        # Mock pending outbox entries
        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "mem_1", "operation": "add"},
            {"id": 2, "memory_id": "mem_2", "operation": "add"},
        ]

        # Mock memory retrieval
        sqlite.get_memory.side_effect = [
            {"id": "mem_1", "content": "doc 1", "namespace": "default", "type": "fact"},
            {"id": "mem_2", "content": "doc 2", "namespace": "default", "type": "fact"},
        ]

        # Mock embedding generation
        embedding_client.embed.return_value = [0.1, 0.2, 0.3]

        store = HybridStore(sqlite, chroma, embedding_client)
        processed = await store.process_outbox(batch_size=10)

        assert processed == 2
        assert embedding_client.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_process_outbox_deleted_memory(self):
        """Test processing outbox for deleted memory."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        # Mock pending outbox entry for deleted memory
        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "deleted_mem", "operation": "add"},
        ]
        sqlite.get_memory.return_value = None  # Memory was deleted

        store = HybridStore(sqlite, chroma, embedding_client)
        processed = await store.process_outbox()

        assert processed == 1
        # Entry should be marked as processed
        sqlite.mark_outbox_processed.assert_called_once_with(1)

    def test_get_outbox_status(self):
        """Test getting outbox status."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "mem_1"},
            {"id": 2, "memory_id": "mem_2"},
        ]

        store = HybridStore(sqlite, chroma, embedding_client)
        status = store.get_outbox_status()

        assert status["pending"] == 2
        assert status["chroma_available"] is True


class TestEdgeOperations:
    """Tests for graph edge operations."""

    def test_add_edge(self):
        """Test edge addition delegation."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.add_edge.return_value = 1

        store = HybridStore(sqlite, chroma, embedding_client)
        result = store.add_edge("mem_a", "mem_b", edge_type="causes", weight=0.9)

        assert result == 1
        sqlite.add_edge.assert_called_once_with(
            source_id="mem_a",
            target_id="mem_b",
            edge_type="causes",
            weight=0.9,
            metadata=None,
        )

    def test_get_edges(self):
        """Test edge retrieval delegation."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.get_edges.return_value = [
            {"id": 1, "source_id": "mem_a", "target_id": "mem_b"}
        ]

        store = HybridStore(sqlite, chroma, embedding_client)
        results = store.get_edges("mem_a", direction="outgoing")

        assert len(results) == 1
        sqlite.get_edges.assert_called_once_with(
            memory_id="mem_a",
            direction="outgoing",
            edge_type=None,
        )

    def test_delete_edge(self):
        """Test edge deletion delegation."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.delete_edge.return_value = True

        store = HybridStore(sqlite, chroma, embedding_client)
        result = store.delete_edge(1)

        assert result is True
        sqlite.delete_edge.assert_called_once_with(1)


class TestUtilityOperations:
    """Tests for utility operations."""

    def test_list_memories(self):
        """Test list_memories delegation."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.list_memories.return_value = [
            {"id": "mem_1", "content": "doc 1"}
        ]

        store = HybridStore(sqlite, chroma, embedding_client)
        results = store.list_memories(namespace="ns1", limit=10)

        assert len(results) == 1
        sqlite.list_memories.assert_called_once()

    def test_count_memories(self):
        """Test count_memories delegation."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.count_memories.return_value = 42

        store = HybridStore(sqlite, chroma, embedding_client)
        result = store.count_memories()

        assert result == 42

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing both stores."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        sqlite.clear.return_value = 10

        store = HybridStore(sqlite, chroma, embedding_client)
        result = await store.clear()

        assert result == 10
        sqlite.clear.assert_called_once()
        chroma.clear.assert_called_once()


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_resources(self):
        """Test that context manager properly closes resources."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        async with HybridStore(sqlite, chroma, embedding_client) as store:
            pass

        embedding_client.close.assert_called_once()
        sqlite.close.assert_called_once()


class TestChromaAvailability:
    """Tests for ChromaDB availability tracking."""

    @pytest.mark.asyncio
    async def test_chroma_marked_unavailable_on_failure(self):
        """Test that ChromaDB is marked unavailable after failures."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        chroma._collection = MagicMock()
        sqlite.add_memory.return_value = "mem_123"
        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "mem_123", "operation": "add"}
        ]
        embedding_client.embed.return_value = [0.1, 0.2, 0.3]
        chroma._collection.add.side_effect = Exception("Connection refused")

        store = HybridStore(sqlite, chroma, embedding_client)
        assert store.chroma_available is True

        await store.add_memory(content="Test")

        assert store.chroma_available is False

    @pytest.mark.asyncio
    async def test_chroma_marked_available_on_success(self):
        """Test that ChromaDB is marked available after success."""
        sqlite = MagicMock(spec=SQLiteStore)
        chroma = MagicMock(spec=ChromaStore)
        embedding_client = AsyncMock(spec=OllamaClient)

        chroma._collection = MagicMock()
        sqlite.add_memory.return_value = "mem_123"
        sqlite.get_pending_outbox.return_value = [
            {"id": 1, "memory_id": "mem_123", "operation": "add"}
        ]
        embedding_client.embed.return_value = [0.1, 0.2, 0.3]

        store = HybridStore(sqlite, chroma, embedding_client)
        store._chroma_available = False  # Start unavailable

        await store.add_memory(content="Test")

        assert store.chroma_available is True


class TestIntegration:
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
    async def test_add_and_get_memory(self, integration_store):
        """Test adding and retrieving a memory."""
        mem_id = await integration_store.add_memory(
            content="Python is great",
            memory_type="fact",
            namespace="tech",
            importance=0.9,
        )

        memory = await integration_store.get_memory(mem_id)

        assert memory is not None
        assert memory["content"] == "Python is great"
        assert memory["type"] == "fact"
        assert memory["namespace"] == "tech"
        assert memory["importance"] == 0.9

    @pytest.mark.asyncio
    async def test_add_and_delete_memory(self, integration_store):
        """Test adding and deleting a memory."""
        mem_id = await integration_store.add_memory(content="Temporary")

        deleted = await integration_store.delete_memory(mem_id)
        assert deleted is True

        memory = await integration_store.get_memory(mem_id)
        assert memory is None

    @pytest.mark.asyncio
    async def test_list_and_count(self, integration_store):
        """Test listing and counting memories."""
        await integration_store.add_memory(content="Memory 1", namespace="ns1")
        await integration_store.add_memory(content="Memory 2", namespace="ns1")
        await integration_store.add_memory(content="Memory 3", namespace="ns2")

        all_memories = integration_store.list_memories()
        assert len(all_memories) == 3

        ns1_memories = integration_store.list_memories(namespace="ns1")
        assert len(ns1_memories) == 2

        count = integration_store.count_memories()
        assert count == 3

    @pytest.mark.asyncio
    async def test_edge_operations(self, integration_store):
        """Test graph edge operations."""
        mem_a = await integration_store.add_memory(content="Memory A")
        mem_b = await integration_store.add_memory(content="Memory B")

        edge_id = integration_store.add_edge(
            source_id=mem_a,
            target_id=mem_b,
            edge_type="related",
        )
        assert edge_id is not None

        edges = integration_store.get_edges(mem_a, direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["target_id"] == mem_b

        deleted = integration_store.delete_edge(edge_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_fts_search(self, integration_store):
        """Test FTS search functionality."""
        await integration_store.add_memory(
            content="Python programming language",
            namespace="tech",
        )
        await integration_store.add_memory(
            content="JavaScript runs in browser",
            namespace="tech",
        )

        results = integration_store.search_fts("Python", namespace="tech")

        assert len(results) == 1
        assert "Python" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_clear(self, integration_store):
        """Test clearing all stores."""
        await integration_store.add_memory(content="Memory 1")
        await integration_store.add_memory(content="Memory 2")

        count = await integration_store.clear()
        assert count == 2

        remaining = integration_store.count_memories()
        assert remaining == 0
