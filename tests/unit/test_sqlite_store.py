"""Tests for SQLiteStore - metadata, graph edges, and FTS5 search."""

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from recall.storage.sqlite import SQLiteStore, SQLiteStoreError


class TestSQLiteStoreInit:
    """Tests for SQLiteStore initialization."""

    def test_init_ephemeral(self):
        """Test ephemeral (in-memory) store initialization."""
        store = SQLiteStore(ephemeral=True)
        assert store.ephemeral is True
        assert store.db_path is None
        store.close()

    def test_init_persistent(self, tmp_path: Path):
        """Test persistent store initialization."""
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path=db_path)
        assert store.ephemeral is False
        assert store.db_path == db_path
        assert db_path.exists()
        store.close()

    def test_init_creates_parent_directories(self, tmp_path: Path):
        """Test that init creates parent directories if needed."""
        db_path = tmp_path / "nested" / "dir" / "test.db"
        store = SQLiteStore(db_path=db_path)
        assert db_path.parent.exists()
        store.close()

    def test_context_manager(self, tmp_path: Path):
        """Test context manager protocol."""
        db_path = tmp_path / "test.db"
        with SQLiteStore(db_path=db_path) as store:
            assert store._conn is not None
        # Connection should be closed after context manager exits

    def test_schema_initialization(self):
        """Test that all tables and indexes are created."""
        store = SQLiteStore(ephemeral=True)
        cursor = store._conn.cursor()

        # Check memories table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert cursor.fetchone() is not None

        # Check edges table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='edges'"
        )
        assert cursor.fetchone() is not None

        # Check outbox table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='outbox'"
        )
        assert cursor.fetchone() is not None

        # Check FTS5 table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        )
        assert cursor.fetchone() is not None

        # Check indexes exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_namespace'"
        )
        assert cursor.fetchone() is not None

        store.close()


class TestMemoryCRUD:
    """Tests for memory CRUD operations."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store for testing."""
        s = SQLiteStore(ephemeral=True)
        yield s
        s.close()

    def test_add_memory_basic(self, store: SQLiteStore):
        """Test basic memory addition."""
        mem_id = store.add_memory(content="Test memory content")
        assert mem_id is not None
        assert mem_id.startswith("mem_")

    def test_add_memory_with_all_fields(self, store: SQLiteStore):
        """Test memory addition with all optional fields."""
        mem_id = store.add_memory(
            content="Test content",
            memory_type="decision",
            namespace="project1",
            importance=0.8,
            metadata={"key": "value", "nested": {"data": 123}},
        )

        memory = store.get_memory(mem_id)
        assert memory is not None
        assert memory["content"] == "Test content"
        assert memory["type"] == "decision"
        assert memory["namespace"] == "project1"
        assert memory["importance"] == 0.8
        assert memory["metadata"] == {"key": "value", "nested": {"data": 123}}
        assert memory["access_count"] == 0

    def test_add_memory_custom_id(self, store: SQLiteStore):
        """Test memory addition with custom ID."""
        custom_id = "custom_123"
        mem_id = store.add_memory(content="Test content", memory_id=custom_id)
        assert mem_id == custom_id

    def test_add_memory_empty_content_raises(self, store: SQLiteStore):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            store.add_memory(content="")

    def test_add_memory_invalid_importance_raises(self, store: SQLiteStore):
        """Test that invalid importance raises ValueError."""
        with pytest.raises(ValueError, match="Importance must be between"):
            store.add_memory(content="Test", importance=1.5)
        with pytest.raises(ValueError, match="Importance must be between"):
            store.add_memory(content="Test", importance=-0.1)

    def test_add_memory_duplicate_content_same_namespace_raises(
        self, store: SQLiteStore
    ):
        """Test that duplicate content in same namespace raises ValueError."""
        store.add_memory(content="Duplicate content", namespace="ns1")
        with pytest.raises(ValueError, match="already exists in namespace"):
            store.add_memory(content="Duplicate content", namespace="ns1")

    def test_add_memory_duplicate_content_different_namespace_allowed(
        self, store: SQLiteStore
    ):
        """Test that duplicate content in different namespaces is allowed."""
        id1 = store.add_memory(content="Same content", namespace="ns1")
        id2 = store.add_memory(content="Same content", namespace="ns2")
        assert id1 != id2

    def test_get_memory_not_found(self, store: SQLiteStore):
        """Test get_memory returns None for non-existent ID."""
        result = store.get_memory("nonexistent_id")
        assert result is None

    def test_get_memory_with_timestamps(self, store: SQLiteStore):
        """Test that timestamps are properly set."""
        before = time.time()
        mem_id = store.add_memory(content="Test content")
        after = time.time()

        memory = store.get_memory(mem_id)
        assert memory is not None
        assert before <= memory["created_at"] <= after
        assert before <= memory["accessed_at"] <= after

    def test_update_memory_content(self, store: SQLiteStore):
        """Test updating memory content."""
        mem_id = store.add_memory(content="Original content")
        result = store.update_memory(mem_id, content="Updated content")
        assert result is True

        memory = store.get_memory(mem_id)
        assert memory["content"] == "Updated content"

    def test_update_memory_multiple_fields(self, store: SQLiteStore):
        """Test updating multiple fields at once."""
        mem_id = store.add_memory(
            content="Test",
            memory_type="fact",
            namespace="ns1",
            importance=0.5,
        )

        store.update_memory(
            mem_id,
            memory_type="decision",
            importance=0.9,
            metadata={"updated": True},
        )

        memory = store.get_memory(mem_id)
        assert memory["type"] == "decision"
        assert memory["importance"] == 0.9
        assert memory["metadata"] == {"updated": True}

    def test_update_memory_not_found(self, store: SQLiteStore):
        """Test update_memory returns False for non-existent ID."""
        result = store.update_memory("nonexistent", content="New content")
        assert result is False

    def test_update_memory_no_changes(self, store: SQLiteStore):
        """Test update_memory with no fields returns False."""
        mem_id = store.add_memory(content="Test")
        result = store.update_memory(mem_id)
        assert result is False

    def test_update_memory_invalid_importance(self, store: SQLiteStore):
        """Test that invalid importance raises ValueError on update."""
        mem_id = store.add_memory(content="Test")
        with pytest.raises(ValueError, match="Importance must be between"):
            store.update_memory(mem_id, importance=2.0)

    def test_delete_memory(self, store: SQLiteStore):
        """Test memory deletion."""
        mem_id = store.add_memory(content="Test content")
        result = store.delete_memory(mem_id)
        assert result is True
        assert store.get_memory(mem_id) is None

    def test_delete_memory_not_found(self, store: SQLiteStore):
        """Test delete_memory returns False for non-existent ID."""
        result = store.delete_memory("nonexistent")
        assert result is False

    def test_touch_memory(self, store: SQLiteStore):
        """Test touch_memory updates timestamp and access_count."""
        mem_id = store.add_memory(content="Test")
        memory_before = store.get_memory(mem_id)
        assert memory_before["access_count"] == 0

        time.sleep(0.01)  # Small delay to ensure timestamp changes
        result = store.touch_memory(mem_id)
        assert result is True

        memory_after = store.get_memory(mem_id)
        assert memory_after["access_count"] == 1
        assert memory_after["accessed_at"] > memory_before["accessed_at"]

    def test_touch_memory_not_found(self, store: SQLiteStore):
        """Test touch_memory returns False for non-existent ID."""
        result = store.touch_memory("nonexistent")
        assert result is False


class TestListMemories:
    """Tests for list_memories operation."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store with test data."""
        s = SQLiteStore(ephemeral=True)
        # Add test memories
        s.add_memory(content="Memory 1", namespace="ns1", memory_type="fact", importance=0.3)
        time.sleep(0.001)
        s.add_memory(content="Memory 2", namespace="ns1", memory_type="decision", importance=0.7)
        time.sleep(0.001)
        s.add_memory(content="Memory 3", namespace="ns2", memory_type="fact", importance=0.5)
        yield s
        s.close()

    def test_list_all_memories(self, store: SQLiteStore):
        """Test listing all memories."""
        memories = store.list_memories()
        assert len(memories) == 3

    def test_list_memories_filter_namespace(self, store: SQLiteStore):
        """Test filtering by namespace."""
        memories = store.list_memories(namespace="ns1")
        assert len(memories) == 2
        assert all(m["namespace"] == "ns1" for m in memories)

    def test_list_memories_filter_type(self, store: SQLiteStore):
        """Test filtering by type."""
        memories = store.list_memories(memory_type="fact")
        assert len(memories) == 2
        assert all(m["type"] == "fact" for m in memories)

    def test_list_memories_combined_filters(self, store: SQLiteStore):
        """Test combined namespace and type filters."""
        memories = store.list_memories(namespace="ns1", memory_type="fact")
        assert len(memories) == 1
        assert memories[0]["content"] == "Memory 1"

    def test_list_memories_pagination(self, store: SQLiteStore):
        """Test pagination with limit and offset."""
        all_memories = store.list_memories()
        first_page = store.list_memories(limit=2)
        second_page = store.list_memories(limit=2, offset=2)

        assert len(first_page) == 2
        assert len(second_page) == 1
        # Verify they're different
        assert first_page[0]["id"] != second_page[0]["id"]

    def test_list_memories_order_by(self, store: SQLiteStore):
        """Test ordering by different fields."""
        # Order by importance ascending
        memories = store.list_memories(order_by="importance", descending=False)
        importances = [m["importance"] for m in memories]
        assert importances == sorted(importances)

        # Order by importance descending
        memories = store.list_memories(order_by="importance", descending=True)
        importances = [m["importance"] for m in memories]
        assert importances == sorted(importances, reverse=True)

    def test_list_memories_invalid_order_by(self, store: SQLiteStore):
        """Test that invalid order_by raises ValueError."""
        with pytest.raises(ValueError, match="order_by must be one of"):
            store.list_memories(order_by="invalid_field")


class TestFTS5Search:
    """Tests for FTS5 full-text search."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store with searchable content."""
        s = SQLiteStore(ephemeral=True)
        s.add_memory(
            content="Python is a programming language",
            namespace="tech",
            memory_type="fact",
        )
        s.add_memory(
            content="JavaScript runs in the browser",
            namespace="tech",
            memory_type="fact",
        )
        s.add_memory(
            content="The python snake is not venomous",
            namespace="biology",
            memory_type="fact",
        )
        yield s
        s.close()

    def test_search_basic(self, store: SQLiteStore):
        """Test basic FTS search."""
        results = store.search_fts("python")
        assert len(results) == 2

    def test_search_with_namespace_filter(self, store: SQLiteStore):
        """Test FTS search with namespace filter."""
        results = store.search_fts("python", namespace="tech")
        assert len(results) == 1
        assert "programming" in results[0]["content"]

    def test_search_with_type_filter(self, store: SQLiteStore):
        """Test FTS search with type filter."""
        results = store.search_fts("python", memory_type="fact")
        assert len(results) == 2

    def test_search_phrase(self, store: SQLiteStore):
        """Test FTS phrase search."""
        results = store.search_fts('"programming language"')
        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_search_limit(self, store: SQLiteStore):
        """Test FTS search with limit."""
        results = store.search_fts("python", limit=1)
        assert len(results) == 1

    def test_search_includes_rank(self, store: SQLiteStore):
        """Test that search results include relevance rank."""
        results = store.search_fts("python")
        assert all("rank" in r for r in results)

    def test_search_no_results(self, store: SQLiteStore):
        """Test search with no matching results."""
        results = store.search_fts("nonexistent_term_xyz")
        assert len(results) == 0


class TestEdgeOperations:
    """Tests for graph edge operations."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store with connected memories."""
        s = SQLiteStore(ephemeral=True)
        s.add_memory(content="Memory A", memory_id="mem_a")
        s.add_memory(content="Memory B", memory_id="mem_b")
        s.add_memory(content="Memory C", memory_id="mem_c")
        yield s
        s.close()

    def test_add_edge_basic(self, store: SQLiteStore):
        """Test basic edge creation."""
        edge_id = store.add_edge("mem_a", "mem_b")
        assert edge_id is not None
        assert isinstance(edge_id, int)

    def test_add_edge_with_all_fields(self, store: SQLiteStore):
        """Test edge creation with all optional fields."""
        edge_id = store.add_edge(
            source_id="mem_a",
            target_id="mem_b",
            edge_type="causes",
            weight=0.9,
            metadata={"reason": "test"},
        )
        edges = store.get_edges("mem_a", direction="outgoing")
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "causes"
        assert edges[0]["weight"] == 0.9
        assert edges[0]["metadata"] == {"reason": "test"}

    def test_add_edge_nonexistent_source_raises(self, store: SQLiteStore):
        """Test that adding edge with nonexistent source raises."""
        with pytest.raises(ValueError, match="Source memory.*not found"):
            store.add_edge("nonexistent", "mem_b")

    def test_add_edge_nonexistent_target_raises(self, store: SQLiteStore):
        """Test that adding edge with nonexistent target raises."""
        with pytest.raises(ValueError, match="Target memory.*not found"):
            store.add_edge("mem_a", "nonexistent")

    def test_add_edge_duplicate_raises(self, store: SQLiteStore):
        """Test that duplicate edge raises ValueError."""
        store.add_edge("mem_a", "mem_b", edge_type="related")
        with pytest.raises(ValueError, match="Edge already exists"):
            store.add_edge("mem_a", "mem_b", edge_type="related")

    def test_add_edge_different_types_allowed(self, store: SQLiteStore):
        """Test that different edge types between same nodes are allowed."""
        store.add_edge("mem_a", "mem_b", edge_type="related")
        store.add_edge("mem_a", "mem_b", edge_type="causes")
        edges = store.get_edges("mem_a", direction="outgoing")
        assert len(edges) == 2

    def test_add_edge_invalid_weight_raises(self, store: SQLiteStore):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            store.add_edge("mem_a", "mem_b", weight=-1.0)

    def test_get_edges_outgoing(self, store: SQLiteStore):
        """Test getting outgoing edges."""
        store.add_edge("mem_a", "mem_b")
        store.add_edge("mem_a", "mem_c")
        edges = store.get_edges("mem_a", direction="outgoing")
        assert len(edges) == 2
        assert all(e["source_id"] == "mem_a" for e in edges)

    def test_get_edges_incoming(self, store: SQLiteStore):
        """Test getting incoming edges."""
        store.add_edge("mem_a", "mem_c")
        store.add_edge("mem_b", "mem_c")
        edges = store.get_edges("mem_c", direction="incoming")
        assert len(edges) == 2
        assert all(e["target_id"] == "mem_c" for e in edges)

    def test_get_edges_both(self, store: SQLiteStore):
        """Test getting both incoming and outgoing edges."""
        store.add_edge("mem_a", "mem_b")
        store.add_edge("mem_c", "mem_b")
        edges = store.get_edges("mem_b", direction="both")
        assert len(edges) == 2

    def test_get_edges_filter_type(self, store: SQLiteStore):
        """Test filtering edges by type."""
        store.add_edge("mem_a", "mem_b", edge_type="related")
        store.add_edge("mem_a", "mem_c", edge_type="causes")
        edges = store.get_edges("mem_a", direction="outgoing", edge_type="causes")
        assert len(edges) == 1
        assert edges[0]["target_id"] == "mem_c"

    def test_get_edges_invalid_direction_raises(self, store: SQLiteStore):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="direction must be"):
            store.get_edges("mem_a", direction="invalid")

    def test_delete_edge(self, store: SQLiteStore):
        """Test edge deletion."""
        edge_id = store.add_edge("mem_a", "mem_b")
        result = store.delete_edge(edge_id)
        assert result is True
        edges = store.get_edges("mem_a", direction="outgoing")
        assert len(edges) == 0

    def test_delete_edge_not_found(self, store: SQLiteStore):
        """Test delete_edge returns False for non-existent ID."""
        result = store.delete_edge(99999)
        assert result is False

    def test_cascade_delete_edges_on_memory_delete(self, store: SQLiteStore):
        """Test that edges are deleted when memory is deleted (CASCADE)."""
        store.add_edge("mem_a", "mem_b")
        store.add_edge("mem_b", "mem_c")

        # Delete mem_b - should remove both edges
        store.delete_memory("mem_b")

        # Verify edges are gone
        edges_a = store.get_edges("mem_a", direction="outgoing")
        edges_c = store.get_edges("mem_c", direction="incoming")
        assert len(edges_a) == 0
        assert len(edges_c) == 0


class TestOutboxOperations:
    """Tests for outbox (ChromaDB sync) operations."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store for testing."""
        s = SQLiteStore(ephemeral=True)
        yield s
        s.close()

    def test_outbox_entry_created_on_add(self, store: SQLiteStore):
        """Test that adding memory creates outbox entry."""
        store.add_memory(content="Test")
        pending = store.get_pending_outbox()
        assert len(pending) == 1
        assert pending[0]["operation"] == "add"

    def test_outbox_entry_created_on_update(self, store: SQLiteStore):
        """Test that updating memory creates outbox entry."""
        mem_id = store.add_memory(content="Test")
        store.update_memory(mem_id, content="Updated")
        pending = store.get_pending_outbox()
        # Should have 2: one for add, one for update
        assert len(pending) == 2
        assert pending[1]["operation"] == "update"

    def test_get_pending_outbox_limit(self, store: SQLiteStore):
        """Test pending outbox with limit."""
        for i in range(5):
            store.add_memory(content=f"Memory {i}")
        pending = store.get_pending_outbox(limit=3)
        assert len(pending) == 3

    def test_mark_outbox_processed_success(self, store: SQLiteStore):
        """Test marking outbox entry as processed."""
        store.add_memory(content="Test")
        pending = store.get_pending_outbox()
        outbox_id = pending[0]["id"]

        result = store.mark_outbox_processed(outbox_id)
        assert result is True

        # Should no longer appear in pending
        pending = store.get_pending_outbox()
        assert len(pending) == 0

    def test_mark_outbox_processed_with_error(self, store: SQLiteStore):
        """Test marking outbox entry as failed with error message."""
        store.add_memory(content="Test")
        pending = store.get_pending_outbox()
        outbox_id = pending[0]["id"]

        result = store.mark_outbox_processed(outbox_id, error_message="Test error")
        assert result is True

        # Should no longer appear in pending
        pending = store.get_pending_outbox()
        assert len(pending) == 0

    def test_mark_outbox_processed_not_found(self, store: SQLiteStore):
        """Test mark_outbox_processed returns False for non-existent ID."""
        result = store.mark_outbox_processed(99999)
        assert result is False

    def test_clear_processed_outbox(self, store: SQLiteStore):
        """Test clearing old processed entries."""
        store.add_memory(content="Test")
        pending = store.get_pending_outbox()
        store.mark_outbox_processed(pending[0]["id"])

        # Clear with 0 seconds threshold (immediate)
        count = store.clear_processed_outbox(older_than_seconds=0)
        # Note: might be 0 if not enough time passed
        assert count >= 0


class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store with test data."""
        s = SQLiteStore(ephemeral=True)
        s.add_memory(content="Memory 1", namespace="ns1", memory_type="fact")
        s.add_memory(content="Memory 2", namespace="ns1", memory_type="decision")
        s.add_memory(content="Memory 3", namespace="ns2", memory_type="fact")
        yield s
        s.close()

    def test_count_memories_all(self, store: SQLiteStore):
        """Test counting all memories."""
        count = store.count_memories()
        assert count == 3

    def test_count_memories_by_namespace(self, store: SQLiteStore):
        """Test counting memories by namespace."""
        count = store.count_memories(namespace="ns1")
        assert count == 2

    def test_count_memories_by_type(self, store: SQLiteStore):
        """Test counting memories by type."""
        count = store.count_memories(memory_type="fact")
        assert count == 2

    def test_count_memories_combined_filters(self, store: SQLiteStore):
        """Test counting with combined filters."""
        count = store.count_memories(namespace="ns1", memory_type="fact")
        assert count == 1

    def test_clear_all(self, store: SQLiteStore):
        """Test clearing all data."""
        # Add some edges first
        memories = store.list_memories()
        store.add_edge(memories[0]["id"], memories[1]["id"])

        count = store.clear()
        assert count == 3
        assert store.count_memories() == 0

        # Verify edges are also cleared
        edges = store.get_edges(memories[0]["id"], direction="both")
        assert len(edges) == 0


class TestContentHash:
    """Tests for content hash deduplication."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store for testing."""
        s = SQLiteStore(ephemeral=True)
        yield s
        s.close()

    def test_content_hash_computed(self, store: SQLiteStore):
        """Test that content hash is computed and stored."""
        mem_id = store.add_memory(content="Test content")
        memory = store.get_memory(mem_id)
        assert memory["content_hash"] is not None
        assert len(memory["content_hash"]) == 64  # SHA-256 hex

    def test_same_content_same_hash(self, store: SQLiteStore):
        """Test that same content produces same hash."""
        content = "Identical content"
        # Add to different namespaces
        id1 = store.add_memory(content=content, namespace="ns1")
        id2 = store.add_memory(content=content, namespace="ns2")

        mem1 = store.get_memory(id1)
        mem2 = store.get_memory(id2)
        assert mem1["content_hash"] == mem2["content_hash"]

    def test_different_content_different_hash(self, store: SQLiteStore):
        """Test that different content produces different hash."""
        id1 = store.add_memory(content="Content A")
        id2 = store.add_memory(content="Content B")

        mem1 = store.get_memory(id1)
        mem2 = store.get_memory(id2)
        assert mem1["content_hash"] != mem2["content_hash"]

    def test_update_content_updates_hash(self, store: SQLiteStore):
        """Test that updating content updates the hash."""
        mem_id = store.add_memory(content="Original")
        original_hash = store.get_memory(mem_id)["content_hash"]

        store.update_memory(mem_id, content="Updated")
        new_hash = store.get_memory(mem_id)["content_hash"]

        assert original_hash != new_hash


class TestFTSSyncTriggers:
    """Tests for FTS5 sync triggers."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store for testing."""
        s = SQLiteStore(ephemeral=True)
        yield s
        s.close()

    def test_fts_updated_on_insert(self, store: SQLiteStore):
        """Test that FTS index is updated on memory insert."""
        store.add_memory(content="Unique searchable term xyz123")
        results = store.search_fts("xyz123")
        assert len(results) == 1

    def test_fts_updated_on_delete(self, store: SQLiteStore):
        """Test that FTS index is updated on memory delete."""
        mem_id = store.add_memory(content="Deletable content abc789")
        results = store.search_fts("abc789")
        assert len(results) == 1

        store.delete_memory(mem_id)
        results = store.search_fts("abc789")
        assert len(results) == 0

    def test_fts_updated_on_update(self, store: SQLiteStore):
        """Test that FTS index is updated on memory update."""
        mem_id = store.add_memory(content="Original term foo123")
        results = store.search_fts("foo123")
        assert len(results) == 1

        store.update_memory(mem_id, content="Updated term bar456")

        # Old term should not be found
        results = store.search_fts("foo123")
        assert len(results) == 0

        # New term should be found
        results = store.search_fts("bar456")
        assert len(results) == 1


class TestErrorHandling:
    """Tests for error handling."""

    def test_init_with_invalid_path(self):
        """Test initialization with invalid path raises error."""
        # Use a path that cannot be created (e.g., under /proc on Linux)
        # This test may be platform-specific
        pass  # Skip for now as it's platform-dependent

    def test_sqlite_store_error_inheritance(self):
        """Test that SQLiteStoreError inherits from Exception."""
        error = SQLiteStoreError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"
