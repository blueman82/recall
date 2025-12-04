"""Unit tests for ChromaDB store."""

import uuid

import pytest

from recall.storage.chromadb import ChromaStore, StorageError


def unique_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"test_{uuid.uuid4().hex[:8]}"


class TestChromaStoreInit:
    """Tests for ChromaStore initialization."""

    def test_ephemeral_mode_creates_in_memory_client(self):
        """Ephemeral mode should use in-memory storage."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        assert store.ephemeral is True
        assert store.db_path is None

    def test_custom_collection_name(self):
        """Should accept custom collection name."""
        collection = unique_collection_name()
        store = ChromaStore(ephemeral=True, collection_name=collection)
        assert store.collection_name == collection

    def test_default_collection_name(self):
        """Default collection name should be 'memories'."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        # Create a new store with default collection to check default
        default_store = ChromaStore.__new__(ChromaStore)
        assert ChromaStore.__init__.__defaults__[1] == "memories"

    def test_collection_uses_cosine_distance(self):
        """Collection should be created with cosine distance metric."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        # Verify cosine metric is set in collection metadata
        metadata = store._collection.metadata
        assert metadata.get("hnsw:space") == "cosine"


class TestChromaStoreAdd:
    """Tests for ChromaStore add operations."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store with unique collection for each test."""
        return ChromaStore(ephemeral=True, collection_name=unique_collection_name())

    def test_add_single_document(self, store):
        """Should add a single document with embedding."""
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["test document"]
        metadatas = [{"type": "test"}]

        ids = store.add(embeddings, documents, metadatas)

        assert len(ids) == 1
        assert store.count() == 1

    def test_add_multiple_documents(self, store):
        """Should add multiple documents."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["doc 1", "doc 2"]
        metadatas = [{"type": "a"}, {"type": "b"}]

        ids = store.add(embeddings, documents, metadatas)

        assert len(ids) == 2
        assert store.count() == 2

    def test_add_without_metadata(self, store):
        """Should allow adding documents without metadata."""
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["test document"]

        ids = store.add(embeddings, documents)

        assert len(ids) == 1
        assert store.count() == 1

    def test_add_empty_list_returns_empty(self, store):
        """Adding empty list should return empty list."""
        ids = store.add([], [])
        assert ids == []
        assert store.count() == 0

    def test_add_length_mismatch_raises_error(self, store):
        """Should raise ValueError on length mismatch."""
        embeddings = [[0.1, 0.2, 0.3]]
        documents = ["doc 1", "doc 2"]  # Mismatch

        with pytest.raises(ValueError, match="Length mismatch"):
            store.add(embeddings, documents)

    def test_add_metadata_length_mismatch_raises_error(self, store):
        """Should raise ValueError when metadata length mismatches."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["doc 1", "doc 2"]
        metadatas = [{"type": "a"}]  # Only one metadata

        with pytest.raises(ValueError, match="Length mismatch"):
            store.add(embeddings, documents, metadatas)

    def test_generated_ids_are_unique(self, store):
        """Generated IDs should be unique."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        documents = ["doc 1", "doc 2", "doc 3"]

        ids = store.add(embeddings, documents)

        assert len(ids) == len(set(ids))  # All unique


class TestChromaStoreQuery:
    """Tests for ChromaStore query operations."""

    @pytest.fixture
    def populated_store(self):
        """Create store with test data."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        # Add test documents with distinct embeddings
        embeddings = [
            [1.0, 0.0, 0.0],  # Points in x direction
            [0.0, 1.0, 0.0],  # Points in y direction
            [0.0, 0.0, 1.0],  # Points in z direction
        ]
        documents = ["x document", "y document", "z document"]
        metadatas = [
            {"namespace": "ns1", "type": "a"},
            {"namespace": "ns1", "type": "b"},
            {"namespace": "ns2", "type": "a"},
        ]
        store.add(embeddings, documents, metadatas)
        return store

    def test_query_returns_results(self, populated_store):
        """Query should return matching documents."""
        # Query with vector pointing mostly in x direction
        query = [0.9, 0.1, 0.0]
        results = populated_store.query(query, n_results=1)

        assert len(results["ids"]) == 1
        assert results["documents"][0] == "x document"

    def test_query_returns_correct_structure(self, populated_store):
        """Query results should have correct structure."""
        query = [1.0, 0.0, 0.0]
        results = populated_store.query(query, n_results=2)

        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert len(results["ids"]) == 2

    def test_query_with_where_filter(self, populated_store):
        """Query should filter by metadata."""
        query = [0.5, 0.5, 0.0]  # Between x and y
        results = populated_store.query(
            query, n_results=3, where={"namespace": "ns1"}
        )

        # Should only return ns1 documents (x and y)
        assert len(results["ids"]) == 2
        for metadata in results["metadatas"]:
            assert metadata["namespace"] == "ns1"

    def test_query_with_type_filter(self, populated_store):
        """Query should filter by type metadata."""
        query = [0.5, 0.5, 0.5]
        results = populated_store.query(query, n_results=3, where={"type": "a"})

        # Should return type "a" documents (x and z)
        assert len(results["ids"]) == 2
        for metadata in results["metadatas"]:
            assert metadata["type"] == "a"

    def test_query_empty_store_returns_empty(self):
        """Querying empty store should return empty results."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        results = store.query([1.0, 0.0, 0.0], n_results=5)

        assert results["ids"] == []
        assert results["documents"] == []

    def test_cosine_similarity_ordering(self, populated_store):
        """Results should be ordered by cosine similarity."""
        # Query exactly matching x direction
        query = [1.0, 0.0, 0.0]
        results = populated_store.query(query, n_results=3)

        # First result should be x document (most similar)
        assert results["documents"][0] == "x document"
        # Distances should be ascending (lower = more similar for cosine)
        distances = results["distances"]
        assert distances[0] <= distances[1] <= distances[2]


class TestChromaStoreDelete:
    """Tests for ChromaStore delete operations."""

    @pytest.fixture
    def store(self):
        """Create ephemeral store with unique collection for each test."""
        return ChromaStore(ephemeral=True, collection_name=unique_collection_name())

    def test_delete_by_ids(self, store):
        """Should delete documents by IDs."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["doc 1", "doc 2"]
        ids = store.add(embeddings, documents)

        assert store.count() == 2
        store.delete([ids[0]])
        assert store.count() == 1

    def test_delete_empty_list_does_nothing(self, store):
        """Deleting empty list should not error."""
        store.add([[0.1, 0.2, 0.3]], ["doc"])
        store.delete([])  # Should not raise
        assert store.count() == 1


class TestChromaStoreGet:
    """Tests for ChromaStore get operations."""

    @pytest.fixture
    def populated_store(self):
        """Create store with test data."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["doc 1", "doc 2"]
        metadatas = [{"namespace": "ns1"}, {"namespace": "ns2"}]
        store.add(embeddings, documents, metadatas)
        return store

    def test_get_by_where_filter(self, populated_store):
        """Should get documents by metadata filter."""
        results = populated_store.get(where={"namespace": "ns1"})

        assert len(results["ids"]) == 1
        assert results["documents"][0] == "doc 1"

    def test_get_returns_correct_structure(self, populated_store):
        """Get results should have correct structure."""
        results = populated_store.get(where={"namespace": "ns1"})

        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results


class TestChromaStoreClear:
    """Tests for ChromaStore clear operations."""

    def test_clear_removes_all_documents(self):
        """Clear should remove all documents."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["doc 1", "doc 2"]
        store.add(embeddings, documents)

        assert store.count() == 2
        deleted = store.clear()
        assert deleted == 2
        assert store.count() == 0

    def test_clear_empty_store_returns_zero(self):
        """Clearing empty store should return 0."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        deleted = store.clear()
        assert deleted == 0

    def test_collection_usable_after_clear(self):
        """Collection should be usable after clear."""
        store = ChromaStore(ephemeral=True, collection_name=unique_collection_name())
        store.add([[0.1, 0.2, 0.3]], ["doc"])
        store.clear()

        # Should be able to add again
        ids = store.add([[0.4, 0.5, 0.6]], ["new doc"])
        assert len(ids) == 1
        assert store.count() == 1
