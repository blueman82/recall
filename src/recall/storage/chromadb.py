"""ChromaDB storage layer for vector operations.

This module provides a wrapper around ChromaDB for vector storage with support for:
- Persistent storage (production) via PersistentClient
- Ephemeral storage (testing) via EphemeralClient
- Cosine distance metric (critical for mxbai-embed-large compatibility)
- Metadata filtering for namespace/type queries
"""

import time
from pathlib import Path
from typing import Any, Optional

import chromadb  # type: ignore[import-not-found]
from chromadb.api.models.Collection import Collection  # type: ignore[import-not-found]


class StorageError(Exception):
    """Custom exception for storage-related errors."""

    pass


class ChromaStore:
    """Vector storage layer using ChromaDB.

    Provides storage for embeddings with metadata, semantic search,
    and support for both persistent and ephemeral modes.

    Args:
        db_path: Path to ChromaDB persistent storage directory.
                 Defaults to ~/.docvec/chroma_db for sharing with docvec MCP.
        collection_name: Name of the collection (default: "memories")
        ephemeral: If True, use in-memory storage for testing (default: False)

    Attributes:
        db_path: Path to database storage (None if ephemeral)
        collection_name: Name of the active collection
        ephemeral: Whether using ephemeral storage
        _client: ChromaDB client instance
        _collection: ChromaDB collection instance
        _id_counter: Counter for generating unique IDs
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: str = "memories",
        ephemeral: bool = False,
    ):
        """Initialize ChromaStore with persistent or ephemeral storage.

        Args:
            db_path: Path to ChromaDB persistent storage directory.
                     Defaults to ~/.docvec/chroma_db if not ephemeral.
            collection_name: Name of the collection to use
            ephemeral: If True, use in-memory EphemeralClient for testing

        Raises:
            StorageError: If database initialization fails
        """
        self.collection_name = collection_name
        self.ephemeral = ephemeral
        self._id_counter = 0

        # Set default path if not provided and not ephemeral
        if ephemeral:
            self.db_path = None
        else:
            self.db_path = db_path or Path.home() / ".docvec" / "chroma_db"

        try:
            if ephemeral:
                # Use in-memory client for testing
                self._client = chromadb.EphemeralClient()
            else:
                # Ensure database directory exists
                if self.db_path is not None:
                    self.db_path.mkdir(parents=True, exist_ok=True)
                # Initialize persistent ChromaDB client
                self._client = chromadb.PersistentClient(path=str(self.db_path))

            # Get or create collection with cosine distance
            self._collection = self._get_or_create_collection()

        except Exception as e:
            raise StorageError(f"Failed to initialize ChromaDB storage: {e}") from e

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one with cosine distance.

        Returns:
            ChromaDB Collection instance

        Raises:
            StorageError: If collection operations fail
        """
        try:
            # Use cosine similarity - critical for mxbai-embed-large
            # NOT L2 which performs poorly with this model
            collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return collection

        except Exception as e:
            raise StorageError(f"Failed to get or create collection: {e}") from e

    def _generate_id(self) -> str:
        """Generate unique, sortable ID using timestamp and sequence.

        Returns:
            Unique ID string in format: timestamp_sequence
        """
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{self._id_counter}"
        self._id_counter += 1
        return unique_id

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> list[str]:
        """Add embeddings with documents and metadata to storage.

        Args:
            embeddings: List of embedding vectors
            documents: List of document content strings
            metadatas: Optional list of metadata dictionaries

        Returns:
            List of generated IDs for the added documents

        Raises:
            StorageError: If add operation fails
            ValueError: If input lists have different lengths
        """
        if metadatas is not None and len(embeddings) != len(metadatas):
            raise ValueError(
                f"Length mismatch: embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        if len(embeddings) != len(documents):
            raise ValueError(
                f"Length mismatch: embeddings={len(embeddings)}, documents={len(documents)}"
            )

        if not embeddings:
            return []

        try:
            # Generate unique IDs for each document
            ids = [self._generate_id() for _ in range(len(embeddings))]

            # Add to ChromaDB collection using explicit keyword arguments
            # to avoid mypy type inference issues with **dict unpacking
            if metadatas is not None:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,  # type: ignore[arg-type]
                    documents=documents,
                    metadatas=metadatas,  # type: ignore[arg-type]
                )
            else:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,  # type: ignore[arg-type]
                    documents=documents,
                )

            return ids

        except Exception as e:
            raise StorageError(f"Failed to add documents to storage: {e}") from e

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Perform semantic search with optional metadata filtering.

        Args:
            query_embedding: Query vector to search for
            n_results: Number of results to return (default: 5)
            where: Optional metadata filter dict for namespace/type filtering
                   (e.g., {"namespace": "project1"}, {"type": "decision"})

        Returns:
            Dictionary with search results containing:
                - ids: List of document IDs
                - documents: List of document content
                - metadatas: List of metadata dictionaries
                - distances: List of distance scores (lower is better for cosine)

        Raises:
            StorageError: If search operation fails
        """
        try:
            # Perform semantic search using explicit keyword arguments
            # to avoid mypy type inference issues with **dict unpacking
            include_fields: list[str] = ["documents", "metadatas", "distances"]
            if where is not None:
                results = self._collection.query(
                    query_embeddings=[query_embedding],  # type: ignore[arg-type]
                    n_results=n_results,
                    include=include_fields,  # type: ignore[arg-type]
                    where=where,  # type: ignore[arg-type]
                )
            else:
                results = self._collection.query(
                    query_embeddings=[query_embedding],  # type: ignore[arg-type]
                    n_results=n_results,
                    include=include_fields,  # type: ignore[arg-type]
                )

            # ChromaDB returns results wrapped in lists (for batch queries)
            # Extract the first (and only) result set
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }

        except Exception as e:
            raise StorageError(f"Failed to search documents: {e}") from e

    def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Raises:
            StorageError: If delete operation fails
        """
        if not ids:
            return

        try:
            self._collection.delete(ids=ids)

        except Exception as e:
            raise StorageError(f"Failed to delete documents: {e}") from e

    def count(self) -> int:
        """Get total number of documents in collection.

        Returns:
            Number of documents stored

        Raises:
            StorageError: If count operation fails
        """
        try:
            return self._collection.count()

        except Exception as e:
            raise StorageError(f"Failed to count documents: {e}") from e

    def get(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> dict:
        """Get documents by IDs or metadata filter.

        Args:
            ids: Optional list of document IDs to retrieve
            where: Optional metadata filter dict

        Returns:
            Dictionary with document data containing:
                - ids: List of document IDs
                - documents: List of document content
                - metadatas: List of metadata dictionaries

        Raises:
            StorageError: If get operation fails
        """
        try:
            get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
            if ids is not None:
                get_kwargs["ids"] = ids
            if where is not None:
                get_kwargs["where"] = where

            results = self._collection.get(**get_kwargs)

            return {
                "ids": results["ids"] if results["ids"] else [],
                "documents": results["documents"] if results["documents"] else [],
                "metadatas": results["metadatas"] if results["metadatas"] else [],
            }

        except Exception as e:
            raise StorageError(f"Failed to get documents: {e}") from e

    def clear(self) -> int:
        """Delete all documents by dropping and recreating the collection.

        Returns:
            Number of documents deleted

        Raises:
            StorageError: If clear operation fails
        """
        try:
            current_count = self._collection.count()
            if current_count == 0:
                return 0

            # Delete and recreate collection atomically
            self._client.delete_collection(self.collection_name)
            self._collection = self._get_or_create_collection()
            return current_count

        except Exception as e:
            raise StorageError(f"Failed to clear collection: {e}") from e
