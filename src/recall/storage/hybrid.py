"""Hybrid storage layer coordinating SQLite and ChromaDB operations.

This module provides a HybridStore that coordinates SQLite (metadata, graph, FTS)
and ChromaDB (vector storage) operations with eventual consistency via the
outbox pattern.

Key principles:
- SQLite is the source of truth for all metadata
- ChromaDB contains embeddings + minimal metadata for search
- Outbox pattern ensures eventual consistency
- Graceful handling of ChromaDB unavailability
- Embedding generation happens at store time via OllamaClient
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from recall.embedding.ollama import EmbeddingError, OllamaClient
from recall.storage.chromadb import ChromaStore, StorageError as ChromaStorageError
from recall.storage.sqlite import SQLiteStore, SQLiteStoreError

logger = logging.getLogger(__name__)


class HybridStoreError(Exception):
    """Custom exception for hybrid storage operations."""

    pass


class HybridStore:
    """Coordinated storage layer combining SQLite and ChromaDB.

    Uses the outbox pattern for eventual consistency:
    1. All writes go to SQLite first (source of truth)
    2. Outbox entry is created in same SQLite transaction
    3. ChromaDB sync is attempted immediately
    4. If ChromaDB fails, outbox entry remains for later retry

    Args:
        sqlite_store: SQLiteStore instance for metadata/graph/FTS
        chroma_store: ChromaStore instance for vector operations
        embedding_client: OllamaClient for generating embeddings
        sync_on_write: If True, attempt ChromaDB sync on each write (default: True)

    Example:
        >>> async with HybridStore.create() as store:
        ...     mem_id = await store.add_memory("Important fact", memory_type="fact")
        ...     results = await store.search("relevant query", n_results=5)
    """

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        chroma_store: ChromaStore,
        embedding_client: OllamaClient,
        sync_on_write: bool = True,
    ):
        """Initialize HybridStore with component stores.

        Args:
            sqlite_store: SQLiteStore instance
            chroma_store: ChromaStore instance
            embedding_client: OllamaClient instance
            sync_on_write: Whether to sync to ChromaDB immediately on writes
        """
        self._sqlite = sqlite_store
        self._chroma = chroma_store
        self._embedding_client = embedding_client
        self._sync_on_write = sync_on_write
        self._chroma_available = True

    @classmethod
    async def create(
        cls,
        sqlite_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        collection_name: str = "memories",
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "mxbai-embed-large",
        ephemeral: bool = False,
        sync_on_write: bool = True,
    ) -> "HybridStore":
        """Create a HybridStore with new component instances.

        Factory method for convenient HybridStore creation with
        default configurations.

        Args:
            sqlite_path: Path to SQLite database (default: ~/.recall/recall.db)
            chroma_path: Path to ChromaDB storage (default: ~/.docvec/chroma_db)
            collection_name: ChromaDB collection name (default: "memories")
            ollama_host: Ollama server host (default: http://localhost:11434)
            ollama_model: Embedding model name (default: mxbai-embed-large)
            ephemeral: Use in-memory storage for testing (default: False)
            sync_on_write: Sync to ChromaDB on writes (default: True)

        Returns:
            Configured HybridStore instance

        Raises:
            HybridStoreError: If store initialization fails
        """
        try:
            sqlite_store = SQLiteStore(db_path=sqlite_path, ephemeral=ephemeral)
            chroma_store = ChromaStore(
                db_path=chroma_path,
                collection_name=collection_name,
                ephemeral=ephemeral,
            )
            embedding_client = OllamaClient(
                host=ollama_host,
                model=ollama_model,
            )

            return cls(
                sqlite_store=sqlite_store,
                chroma_store=chroma_store,
                embedding_client=embedding_client,
                sync_on_write=sync_on_write,
            )

        except (SQLiteStoreError, ChromaStorageError) as e:
            raise HybridStoreError(f"Failed to create HybridStore: {e}") from e

    async def close(self) -> None:
        """Close all underlying stores and clients."""
        await self._embedding_client.close()
        self._sqlite.close()
        # ChromaDB doesn't require explicit close

    async def __aenter__(self) -> "HybridStore":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - close all resources."""
        await self.close()

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        namespace: str = "default",
        importance: float = 0.5,
        confidence: float = 0.3,
        metadata: Optional[dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Add a new memory with embedding to both stores.

        Writes to SQLite first (source of truth), generates embedding,
        then syncs to ChromaDB. If ChromaDB sync fails, outbox entry
        remains for later retry.

        Args:
            content: The memory content text
            memory_type: Type of memory (e.g., 'fact', 'decision', 'context')
            namespace: Namespace for organizing memories
            importance: Importance score from 0.0 to 1.0
            confidence: Confidence score from 0.0 to 1.0 (default: 0.3)
            metadata: Optional additional metadata as dict
            memory_id: Optional custom ID (auto-generated if not provided)

        Returns:
            The ID of the created memory

        Raises:
            HybridStoreError: If SQLite write fails (ChromaDB failures are non-fatal)
            ValueError: If content is empty or scores out of range
        """
        try:
            # Step 1: Add to SQLite (source of truth)
            # This also creates an outbox entry in the same transaction
            mem_id = self._sqlite.add_memory(
                content=content,
                memory_type=memory_type,
                namespace=namespace,
                importance=importance,
                confidence=confidence,
                metadata=metadata,
                memory_id=memory_id,
            )

            # Step 2: Attempt ChromaDB sync if enabled
            if self._sync_on_write:
                await self._sync_memory_to_chroma(mem_id, content, namespace, memory_type)

            return mem_id

        except (SQLiteStoreError, ValueError) as e:
            raise HybridStoreError(f"Failed to add memory: {e}") from e

    async def _sync_memory_to_chroma(
        self,
        memory_id: str,
        content: str,
        namespace: str,
        memory_type: str,
    ) -> bool:
        """Sync a single memory to ChromaDB.

        Generates embedding and adds to ChromaDB. If successful,
        marks the outbox entry as processed.

        Args:
            memory_id: Memory ID to sync
            content: Memory content for embedding
            namespace: Memory namespace
            memory_type: Memory type

        Returns:
            True if sync succeeded, False otherwise
        """
        try:
            # Generate embedding for the content (not a query, so no prefix)
            embedding = await self._embedding_client.embed(content, is_query=False)

            # Add to ChromaDB with minimal metadata
            # ChromaDB needs memory_id as the document ID for correlation
            self._chroma._collection.add(
                ids=[memory_id],
                embeddings=[embedding],  # type: ignore[arg-type]
                documents=[content],
                metadatas=[{
                    "namespace": namespace,
                    "type": memory_type,
                }],
            )

            # Mark outbox entry as processed
            self._mark_outbox_processed_for_memory(memory_id)
            self._chroma_available = True

            logger.debug(f"Successfully synced memory {memory_id} to ChromaDB")
            return True

        except EmbeddingError as e:
            logger.warning(f"Embedding generation failed for memory {memory_id}: {e}")
            self._mark_outbox_failed_for_memory(memory_id, str(e))
            return False

        except ChromaStorageError as e:
            logger.warning(f"ChromaDB sync failed for memory {memory_id}: {e}")
            self._chroma_available = False
            self._mark_outbox_failed_for_memory(memory_id, str(e))
            return False

        except Exception as e:
            logger.warning(f"Unexpected error syncing memory {memory_id}: {e}")
            self._chroma_available = False
            self._mark_outbox_failed_for_memory(memory_id, str(e))
            return False

    def _mark_outbox_processed_for_memory(self, memory_id: str) -> None:
        """Mark the latest outbox entry for a memory as processed."""
        pending = self._sqlite.get_pending_outbox(limit=100)
        for entry in pending:
            if entry["memory_id"] == memory_id:
                self._sqlite.mark_outbox_processed(entry["id"])
                break

    def _mark_outbox_failed_for_memory(self, memory_id: str, error: str) -> None:
        """Mark the latest outbox entry for a memory as failed."""
        pending = self._sqlite.get_pending_outbox(limit=100)
        for entry in pending:
            if entry["memory_id"] == memory_id:
                self._sqlite.mark_outbox_processed(entry["id"], error_message=error)
                break

    async def get_memory(self, memory_id: str) -> Optional[dict[str, Any]]:
        """Get a memory by ID from SQLite.

        Args:
            memory_id: The memory ID to retrieve

        Returns:
            Memory dict or None if not found

        Raises:
            HybridStoreError: If get operation fails
        """
        try:
            return self._sqlite.get_memory(memory_id)
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to get memory: {e}") from e

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        namespace: Optional[str] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory in both stores.

        Updates SQLite first, then syncs changes to ChromaDB.
        If content is updated, a new embedding is generated.

        Args:
            memory_id: The memory ID to update
            content: New content (optional)
            memory_type: New type (optional)
            namespace: New namespace (optional)
            importance: New importance score (optional)
            confidence: New confidence score (optional)
            metadata: New metadata dict (optional)

        Returns:
            True if memory was updated, False if not found

        Raises:
            HybridStoreError: If update operation fails
        """
        try:
            # Update SQLite (source of truth)
            updated = self._sqlite.update_memory(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                namespace=namespace,
                importance=importance,
                confidence=confidence,
                metadata=metadata,
            )

            if not updated:
                return False

            # If content or searchable fields changed, sync to ChromaDB
            if self._sync_on_write and (content is not None or namespace is not None or memory_type is not None):
                memory = self._sqlite.get_memory(memory_id)
                if memory:
                    # Delete old entry and add new one
                    try:
                        self._chroma.delete([memory_id])
                    except ChromaStorageError:
                        pass  # May not exist in ChromaDB yet

                    await self._sync_memory_to_chroma(
                        memory_id,
                        memory["content"],
                        memory["namespace"],
                        memory["type"],
                    )

            return True

        except (SQLiteStoreError, ValueError) as e:
            raise HybridStoreError(f"Failed to update memory: {e}") from e

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from both stores.

        Deletes from SQLite first (source of truth), then ChromaDB.

        Args:
            memory_id: The memory ID to delete

        Returns:
            True if memory was deleted, False if not found

        Raises:
            HybridStoreError: If delete operation fails
        """
        try:
            # Delete from SQLite (source of truth)
            deleted = self._sqlite.delete_memory(memory_id)

            if deleted:
                # Also delete from ChromaDB (best effort)
                try:
                    self._chroma.delete([memory_id])
                except ChromaStorageError as e:
                    logger.warning(f"Failed to delete from ChromaDB: {e}")

            return deleted

        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to delete memory: {e}") from e

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        n_results: int = 5,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic search using ChromaDB, enriched with SQLite metadata.

        Generates query embedding, searches ChromaDB for similar vectors,
        then enriches results with full metadata from SQLite.

        Args:
            query: Search query text
            n_results: Number of results to return (default: 5)
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)

        Returns:
            List of memory dicts with similarity scores

        Raises:
            HybridStoreError: If search operation fails
        """
        try:
            # Generate query embedding (with mxbai prefix)
            query_embedding = await self._embedding_client.embed(query, is_query=True)

            # Build where filter for ChromaDB
            where: Optional[dict] = None
            if namespace or memory_type:
                where = {}
                if namespace:
                    where["namespace"] = namespace
                if memory_type:
                    where["type"] = memory_type

            # Search ChromaDB
            chroma_results = self._chroma.query(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where,
            )

            # Enrich with SQLite metadata
            enriched_results = []
            for i, memory_id in enumerate(chroma_results["ids"]):
                # Get full metadata from SQLite
                memory = self._sqlite.get_memory(memory_id)
                if memory:
                    # Add similarity score (convert distance to similarity)
                    distance = chroma_results["distances"][i]
                    # Cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity: 1 - (distance / 2)
                    similarity = 1 - (distance / 2)
                    memory["similarity"] = similarity
                    enriched_results.append(memory)

                    # Touch memory to update access stats
                    self._sqlite.touch_memory(memory_id)

            return enriched_results

        except EmbeddingError as e:
            raise HybridStoreError(f"Failed to generate query embedding: {e}") from e
        except ChromaStorageError as e:
            # Fall back to FTS search if ChromaDB unavailable
            logger.warning(f"ChromaDB search failed, falling back to FTS: {e}")
            self._chroma_available = False
            return self.search_fts(query, namespace=namespace, memory_type=memory_type, limit=n_results)
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to enrich search results: {e}") from e

    def search_fts(
        self,
        query: str,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform full-text search using SQLite FTS5.

        Fallback search when ChromaDB is unavailable or for
        keyword-based queries.

        Args:
            query: FTS5 search query
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)
            limit: Maximum number of results (default: 10)

        Returns:
            List of matching memory dicts

        Raises:
            HybridStoreError: If FTS search fails
        """
        try:
            return self._sqlite.search_fts(
                query=query,
                namespace=namespace,
                memory_type=memory_type,
                limit=limit,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"FTS search failed: {e}") from e

    # =========================================================================
    # Outbox Processing
    # =========================================================================

    async def process_outbox(self, batch_size: int = 10) -> int:
        """Process pending outbox entries for ChromaDB sync.

        Retries failed ChromaDB syncs from the outbox queue.
        Should be called periodically for eventual consistency.

        Args:
            batch_size: Number of entries to process (default: 10)

        Returns:
            Number of entries successfully processed
        """
        pending = self._sqlite.get_pending_outbox(limit=batch_size)
        processed = 0

        for entry in pending:
            memory_id = entry["memory_id"]
            operation = entry["operation"]

            if operation in ("add", "update"):
                memory = self._sqlite.get_memory(memory_id)
                if memory:
                    success = await self._sync_memory_to_chroma(
                        memory_id,
                        memory["content"],
                        memory["namespace"],
                        memory["type"],
                    )
                    if success:
                        processed += 1
                else:
                    # Memory was deleted, mark as processed
                    self._sqlite.mark_outbox_processed(entry["id"])
                    processed += 1

        return processed

    def get_outbox_status(self) -> dict[str, int]:
        """Get outbox queue status.

        Returns:
            Dict with counts of pending, processed, and failed entries
        """
        pending = self._sqlite.get_pending_outbox(limit=1000)
        return {
            "pending": len(pending),
            "chroma_available": self._chroma_available,
        }

    # =========================================================================
    # Graph Operations (delegated to SQLite)
    # =========================================================================

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "related",
        weight: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Add an edge between two memories.

        Graph operations are handled entirely by SQLite.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            edge_type: Type of relationship (default: 'related')
            weight: Edge weight (default: 1.0)
            metadata: Optional edge metadata

        Returns:
            The ID of the created edge

        Raises:
            HybridStoreError: If add operation fails
        """
        try:
            return self._sqlite.add_edge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata,
            )
        except (SQLiteStoreError, ValueError) as e:
            raise HybridStoreError(f"Failed to add edge: {e}") from e

    def get_edges(
        self,
        memory_id: str,
        direction: str = "both",
        edge_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get edges connected to a memory.

        Args:
            memory_id: The memory ID to get edges for
            direction: 'outgoing', 'incoming', or 'both' (default: 'both')
            edge_type: Filter by edge type (optional)

        Returns:
            List of edge dicts

        Raises:
            HybridStoreError: If get operation fails
        """
        try:
            return self._sqlite.get_edges(
                memory_id=memory_id,
                direction=direction,
                edge_type=edge_type,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to get edges: {e}") from e

    def delete_edge(self, edge_id: int) -> bool:
        """Delete an edge by ID.

        Args:
            edge_id: The edge ID to delete

        Returns:
            True if edge was deleted, False if not found

        Raises:
            HybridStoreError: If delete operation fails
        """
        try:
            return self._sqlite.delete_edge(edge_id)
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to delete edge: {e}") from e

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def list_memories(
        self,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        """List memories with optional filtering.

        Args:
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)
            limit: Maximum number of results (default: 100)
            offset: Number of results to skip (default: 0)
            order_by: Field to sort by (default: 'created_at')
            descending: Sort descending (default: True)

        Returns:
            List of memory dicts

        Raises:
            HybridStoreError: If list operation fails
        """
        try:
            return self._sqlite.list_memories(
                namespace=namespace,
                memory_type=memory_type,
                limit=limit,
                offset=offset,
                order_by=order_by,
                descending=descending,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to list memories: {e}") from e

    def count_memories(
        self,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> int:
        """Count memories with optional filtering.

        Args:
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)

        Returns:
            Number of matching memories

        Raises:
            HybridStoreError: If count operation fails
        """
        try:
            return self._sqlite.count_memories(
                namespace=namespace,
                memory_type=memory_type,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to count memories: {e}") from e

    async def clear(self) -> int:
        """Delete all data from both stores.

        Returns:
            Number of memories deleted

        Raises:
            HybridStoreError: If clear operation fails
        """
        try:
            count = self._sqlite.clear()
            try:
                self._chroma.clear()
            except ChromaStorageError as e:
                logger.warning(f"Failed to clear ChromaDB: {e}")
            return count
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to clear stores: {e}") from e

    # =========================================================================
    # Validation Events Operations
    # =========================================================================

    def add_validation_event(
        self,
        memory_id: str,
        event_type: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Add a validation event for a memory.

        Args:
            memory_id: ID of the memory being validated
            event_type: Type of event ('applied', 'succeeded', 'failed')
            context: Optional context string (JSON)
            session_id: Optional session ID

        Returns:
            The ID of the created validation event

        Raises:
            HybridStoreError: If add operation fails
        """
        try:
            return self._sqlite.add_validation_event(
                memory_id=memory_id,
                event_type=event_type,
                context=context,
                session_id=session_id,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to add validation event: {e}") from e

    def get_validation_events(
        self,
        memory_id: str,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get validation events for a memory.

        Args:
            memory_id: ID of the memory
            event_type: Filter by event type (optional)
            limit: Maximum number of results (default: 100)

        Returns:
            List of validation event dicts

        Raises:
            HybridStoreError: If get operation fails
        """
        try:
            return self._sqlite.get_validation_events(
                memory_id=memory_id,
                event_type=event_type,
                limit=limit,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to get validation events: {e}") from e

    # =========================================================================
    # File Activity Operations
    # =========================================================================

    def add_file_activity(
        self,
        file_path: str,
        action: str,
        session_id: Optional[str] = None,
        project_root: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Add a file activity record.

        Args:
            file_path: Path to the file
            action: Action performed ('read', 'write', 'edit', 'multiedit')
            session_id: Optional session ID
            project_root: Optional project root directory
            file_type: Optional file type (e.g., 'python', 'typescript')
            metadata: Optional additional metadata

        Returns:
            The ID of the created file activity record

        Raises:
            HybridStoreError: If add operation fails
        """
        try:
            return self._sqlite.add_file_activity(
                file_path=file_path,
                action=action,
                session_id=session_id,
                project_root=project_root,
                file_type=file_type,
                metadata=metadata,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to add file activity: {e}") from e

    def get_file_activity(
        self,
        file_path: Optional[str] = None,
        action: Optional[str] = None,
        project_root: Optional[str] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Get file activity records.

        Args:
            file_path: Filter by file path (optional)
            action: Filter by action (optional)
            project_root: Filter by project root (optional)
            limit: Maximum number of results (default: 100)
            since: Filter to activities after this timestamp (optional)

        Returns:
            List of file activity dicts

        Raises:
            HybridStoreError: If get operation fails
        """
        try:
            return self._sqlite.get_file_activity(
                file_path=file_path,
                action=action,
                project_root=project_root,
                limit=limit,
                since=since,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to get file activity: {e}") from e

    def get_recent_files(
        self,
        project_root: Optional[str] = None,
        limit: int = 20,
        days: int = 14,
    ) -> list[dict[str, Any]]:
        """Get recently accessed files with aggregated activity.

        Args:
            project_root: Filter by project root (optional)
            limit: Maximum number of files to return (default: 20)
            days: Look back this many days (default: 14)

        Returns:
            List of dicts with file_path, last_action, last_accessed, access_count

        Raises:
            HybridStoreError: If get operation fails
        """
        try:
            return self._sqlite.get_recent_files(
                project_root=project_root,
                limit=limit,
                days=days,
            )
        except SQLiteStoreError as e:
            raise HybridStoreError(f"Failed to get recent files: {e}") from e

    @property
    def chroma_available(self) -> bool:
        """Check if ChromaDB is available for operations."""
        return self._chroma_available
