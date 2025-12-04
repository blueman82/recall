"""SQLite storage layer for metadata, graph edges, and full-text search.

This module provides a SQLite-based storage layer for Recall with support for:
- Memory metadata (id, content, type, namespace, importance, timestamps)
- Graph edges (relationships between memories)
- FTS5 full-text search index
- Outbox table for ChromaDB sync

The SQLiteStore class manages all SQLite operations and uses content_hash
for deduplication within namespaces.
"""

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class SQLiteStoreError(Exception):
    """Custom exception for SQLite storage-related errors."""

    pass


class SQLiteStore:
    """SQLite storage layer for memory metadata, graph, and FTS.

    Provides storage for memory metadata, relationships between memories (edges),
    full-text search via FTS5, and an outbox for ChromaDB sync operations.

    Args:
        db_path: Path to SQLite database file.
                 Defaults to ~/.recall/recall.db
        ephemeral: If True, use in-memory storage for testing (default: False)

    Attributes:
        db_path: Path to database file (None if ephemeral)
        ephemeral: Whether using ephemeral storage
        _conn: SQLite connection instance
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        ephemeral: bool = False,
    ):
        """Initialize SQLiteStore with persistent or ephemeral storage.

        Args:
            db_path: Path to SQLite database file.
                     Defaults to ~/.recall/recall.db if not ephemeral.
            ephemeral: If True, use in-memory database for testing

        Raises:
            SQLiteStoreError: If database initialization fails
        """
        self.ephemeral = ephemeral

        # Set default path if not provided and not ephemeral
        if ephemeral:
            self.db_path = None
        else:
            self.db_path = db_path or Path.home() / ".recall" / "recall.db"

        try:
            if ephemeral:
                # Use in-memory database for testing
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                # Ensure database directory exists
                if self.db_path is not None:
                    self.db_path.parent.mkdir(parents=True, exist_ok=True)
                # Initialize persistent SQLite connection
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

            # Enable row factory for dict-like access
            self._conn.row_factory = sqlite3.Row

            # Enable foreign key constraints
            self._conn.execute("PRAGMA foreign_keys = ON")

            # Initialize schema
            self._init_schema()

        except sqlite3.Error as e:
            raise SQLiteStoreError(f"Failed to initialize SQLite storage: {e}") from e

    def _init_schema(self) -> None:
        """Initialize database schema with all tables and indexes.

        Creates the following tables:
        - memories: Core memory metadata
        - edges: Graph relationships between memories
        - outbox: Queue for ChromaDB sync operations
        - memories_fts: FTS5 virtual table for full-text search

        Raises:
            SQLiteStoreError: If schema initialization fails
        """
        try:
            cursor = self._conn.cursor()

            # Create memories table with content_hash for deduplication
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'general',
                    namespace TEXT NOT NULL DEFAULT 'default',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT,
                    UNIQUE(content_hash, namespace)
                )
            """)

            # Create indexes for common query patterns
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_namespace
                ON memories(namespace)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_accessed_at
                ON memories(accessed_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_content_hash
                ON memories(content_hash)
            """)

            # Create edges table for graph relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL DEFAULT 'related',
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE,
                    UNIQUE(source_id, target_id, edge_type)
                )
            """)

            # Create indexes for edge queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_source
                ON edges(source_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_target
                ON edges(target_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_type
                ON edges(edge_type)
            """)

            # Create outbox table for ChromaDB sync
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    processed_at REAL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)

            # Create index for outbox processing
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outbox_status
                ON outbox(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outbox_created_at
                ON outbox(created_at)
            """)

            # Create FTS5 virtual table for full-text search
            # Note: FTS5 tables don't support IF NOT EXISTS in the same way
            # Check if table exists first
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='memories_fts'
            """)
            if cursor.fetchone() is None:
                cursor.execute("""
                    CREATE VIRTUAL TABLE memories_fts USING fts5(
                        id,
                        content,
                        type,
                        namespace,
                        content='memories',
                        content_rowid='rowid'
                    )
                """)

                # Create triggers to keep FTS index in sync
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, id, content, type, namespace)
                        VALUES (NEW.rowid, NEW.id, NEW.content, NEW.type, NEW.namespace);
                    END
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, id, content, type, namespace)
                        VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.type, OLD.namespace);
                    END
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, id, content, type, namespace)
                        VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.type, OLD.namespace);
                        INSERT INTO memories_fts(rowid, id, content, type, namespace)
                        VALUES (NEW.rowid, NEW.id, NEW.content, NEW.type, NEW.namespace);
                    END
                """)

            self._conn.commit()

        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to initialize schema: {e}") from e

    @staticmethod
    def _compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash of content for deduplication.

        Args:
            content: The content to hash

        Returns:
            Hex-encoded SHA-256 hash string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _generate_id(self) -> str:
        """Generate unique, sortable ID using timestamp and random suffix.

        Returns:
            Unique ID string in format: mem_timestamp_random
        """
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        import secrets

        random_suffix = secrets.token_hex(4)
        return f"mem_{timestamp}_{random_suffix}"

    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        namespace: str = "default",
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Add a new memory to storage.

        Args:
            content: The memory content text
            memory_type: Type of memory (e.g., 'fact', 'decision', 'context')
            namespace: Namespace for organizing memories
            importance: Importance score from 0.0 to 1.0
            metadata: Optional additional metadata as dict
            memory_id: Optional custom ID (auto-generated if not provided)

        Returns:
            The ID of the created memory

        Raises:
            SQLiteStoreError: If add operation fails
            ValueError: If a memory with same content exists in namespace
        """
        if not content:
            raise ValueError("Content cannot be empty")

        if importance < 0.0 or importance > 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")

        content_hash = self._compute_content_hash(content)
        now = time.time()
        mem_id = memory_id or self._generate_id()

        try:
            cursor = self._conn.cursor()

            # Serialize metadata as JSON if provided
            import json

            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute(
                """
                INSERT INTO memories (
                    id, content, content_hash, type, namespace,
                    importance, created_at, accessed_at, access_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    mem_id,
                    content,
                    content_hash,
                    memory_type,
                    namespace,
                    importance,
                    now,
                    now,
                    metadata_json,
                ),
            )

            # Add to outbox for ChromaDB sync
            cursor.execute(
                """
                INSERT INTO outbox (memory_id, operation, created_at)
                VALUES (?, 'add', ?)
                """,
                (mem_id, now),
            )

            self._conn.commit()
            return mem_id

        except sqlite3.IntegrityError as e:
            self._conn.rollback()
            if "UNIQUE constraint failed: memories.content_hash" in str(e):
                raise ValueError(
                    f"Memory with same content already exists in namespace '{namespace}'"
                ) from e
            raise SQLiteStoreError(f"Failed to add memory: {e}") from e
        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to add memory: {e}") from e

    def get_memory(self, memory_id: str) -> Optional[dict[str, Any]]:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID to retrieve

        Returns:
            Memory dict or None if not found

        Raises:
            SQLiteStoreError: If get operation fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, content, content_hash, type, namespace,
                       importance, created_at, accessed_at, access_count, metadata
                FROM memories WHERE id = ?
                """,
                (memory_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            import json

            return {
                "id": row["id"],
                "content": row["content"],
                "content_hash": row["content_hash"],
                "type": row["type"],
                "namespace": row["namespace"],
                "importance": row["importance"],
                "created_at": row["created_at"],
                "accessed_at": row["accessed_at"],
                "access_count": row["access_count"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            }

        except sqlite3.Error as e:
            raise SQLiteStoreError(f"Failed to get memory: {e}") from e

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        namespace: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: The memory ID to update
            content: New content (optional)
            memory_type: New type (optional)
            namespace: New namespace (optional)
            importance: New importance score (optional)
            metadata: New metadata dict (optional)

        Returns:
            True if memory was updated, False if not found

        Raises:
            SQLiteStoreError: If update operation fails
            ValueError: If update would create duplicate content in namespace
        """
        if importance is not None and (importance < 0.0 or importance > 1.0):
            raise ValueError("Importance must be between 0.0 and 1.0")

        try:
            cursor = self._conn.cursor()

            # Build dynamic update query
            updates = []
            params: list[Any] = []

            if content is not None:
                updates.append("content = ?")
                params.append(content)
                updates.append("content_hash = ?")
                params.append(self._compute_content_hash(content))

            if memory_type is not None:
                updates.append("type = ?")
                params.append(memory_type)

            if namespace is not None:
                updates.append("namespace = ?")
                params.append(namespace)

            if importance is not None:
                updates.append("importance = ?")
                params.append(importance)

            if metadata is not None:
                import json

                updates.append("metadata = ?")
                params.append(json.dumps(metadata))

            if not updates:
                return False

            params.append(memory_id)

            cursor.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                params,
            )

            if cursor.rowcount == 0:
                return False

            # Add to outbox for ChromaDB sync
            cursor.execute(
                """
                INSERT INTO outbox (memory_id, operation, created_at)
                VALUES (?, 'update', ?)
                """,
                (memory_id, time.time()),
            )

            self._conn.commit()
            return True

        except sqlite3.IntegrityError as e:
            self._conn.rollback()
            if "UNIQUE constraint failed: memories.content_hash" in str(e):
                raise ValueError(
                    "Update would create duplicate content in namespace"
                ) from e
            raise SQLiteStoreError(f"Failed to update memory: {e}") from e
        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to update memory: {e}") from e

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        This also deletes associated edges due to CASCADE constraint
        and removes from FTS index via trigger.

        Args:
            memory_id: The memory ID to delete

        Returns:
            True if memory was deleted, False if not found

        Raises:
            SQLiteStoreError: If delete operation fails
        """
        try:
            cursor = self._conn.cursor()

            # Note: outbox entry will also be deleted via CASCADE
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

            if cursor.rowcount == 0:
                return False

            self._conn.commit()
            return True

        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to delete memory: {e}") from e

    def touch_memory(self, memory_id: str) -> bool:
        """Update accessed_at timestamp and increment access_count.

        Args:
            memory_id: The memory ID to touch

        Returns:
            True if memory was touched, False if not found

        Raises:
            SQLiteStoreError: If touch operation fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                UPDATE memories
                SET accessed_at = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (time.time(), memory_id),
            )

            if cursor.rowcount == 0:
                return False

            self._conn.commit()
            return True

        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to touch memory: {e}") from e

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
            SQLiteStoreError: If list operation fails
        """
        allowed_order_fields = {
            "created_at",
            "accessed_at",
            "importance",
            "access_count",
        }
        if order_by not in allowed_order_fields:
            raise ValueError(f"order_by must be one of: {allowed_order_fields}")

        try:
            cursor = self._conn.cursor()

            # Build query with optional filters
            query = """
                SELECT id, content, content_hash, type, namespace,
                       importance, created_at, accessed_at, access_count, metadata
                FROM memories
            """
            conditions = []
            params: list[Any] = []

            if namespace is not None:
                conditions.append("namespace = ?")
                params.append(namespace)

            if memory_type is not None:
                conditions.append("type = ?")
                params.append(memory_type)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            order_dir = "DESC" if descending else "ASC"
            query += f" ORDER BY {order_by} {order_dir}"
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            import json

            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "content_hash": row["content_hash"],
                    "type": row["type"],
                    "namespace": row["namespace"],
                    "importance": row["importance"],
                    "created_at": row["created_at"],
                    "accessed_at": row["accessed_at"],
                    "access_count": row["access_count"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                }
                for row in rows
            ]

        except sqlite3.Error as e:
            raise SQLiteStoreError(f"Failed to list memories: {e}") from e

    def search_fts(
        self,
        query: str,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform full-text search on memories.

        Args:
            query: FTS5 search query (supports AND, OR, NOT, phrases, etc.)
            namespace: Filter by namespace (optional)
            memory_type: Filter by type (optional)
            limit: Maximum number of results (default: 10)

        Returns:
            List of matching memory dicts with relevance scores

        Raises:
            SQLiteStoreError: If search operation fails
        """
        try:
            cursor = self._conn.cursor()

            # Join FTS results with memories table for full data
            query_sql = """
                SELECT m.id, m.content, m.content_hash, m.type, m.namespace,
                       m.importance, m.created_at, m.accessed_at, m.access_count,
                       m.metadata, bm25(memories_fts) as rank
                FROM memories_fts fts
                JOIN memories m ON fts.id = m.id
                WHERE memories_fts MATCH ?
            """
            params: list[Any] = [query]

            if namespace is not None:
                query_sql += " AND m.namespace = ?"
                params.append(namespace)

            if memory_type is not None:
                query_sql += " AND m.type = ?"
                params.append(memory_type)

            query_sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            cursor.execute(query_sql, params)
            rows = cursor.fetchall()

            import json

            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "content_hash": row["content_hash"],
                    "type": row["type"],
                    "namespace": row["namespace"],
                    "importance": row["importance"],
                    "created_at": row["created_at"],
                    "accessed_at": row["accessed_at"],
                    "access_count": row["access_count"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "rank": row["rank"],
                }
                for row in rows
            ]

        except sqlite3.Error as e:
            raise SQLiteStoreError(f"Failed to search memories: {e}") from e

    # =========================================================================
    # Edge (Graph) Operations
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

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            edge_type: Type of relationship (default: 'related')
            weight: Edge weight (default: 1.0)
            metadata: Optional edge metadata

        Returns:
            The ID of the created edge

        Raises:
            SQLiteStoreError: If add operation fails
            ValueError: If source or target memory doesn't exist
        """
        if weight < 0.0:
            raise ValueError("Weight must be non-negative")

        try:
            cursor = self._conn.cursor()

            # Verify source and target exist
            cursor.execute("SELECT id FROM memories WHERE id = ?", (source_id,))
            if cursor.fetchone() is None:
                raise ValueError(f"Source memory '{source_id}' not found")

            cursor.execute("SELECT id FROM memories WHERE id = ?", (target_id,))
            if cursor.fetchone() is None:
                raise ValueError(f"Target memory '{target_id}' not found")

            import json

            metadata_json = json.dumps(metadata) if metadata else None
            now = time.time()

            cursor.execute(
                """
                INSERT INTO edges (source_id, target_id, edge_type, weight, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_id, target_id, edge_type, weight, now, metadata_json),
            )

            edge_id = cursor.lastrowid
            self._conn.commit()
            return edge_id  # type: ignore[return-value]

        except sqlite3.IntegrityError as e:
            self._conn.rollback()
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(
                    f"Edge already exists: {source_id} -> {target_id} ({edge_type})"
                ) from e
            raise SQLiteStoreError(f"Failed to add edge: {e}") from e
        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to add edge: {e}") from e

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
            SQLiteStoreError: If get operation fails
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError("direction must be 'outgoing', 'incoming', or 'both'")

        try:
            cursor = self._conn.cursor()
            results = []

            if direction in ("outgoing", "both"):
                query = """
                    SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
                    FROM edges WHERE source_id = ?
                """
                params: list[Any] = [memory_id]
                if edge_type is not None:
                    query += " AND edge_type = ?"
                    params.append(edge_type)

                cursor.execute(query, params)
                results.extend(cursor.fetchall())

            if direction in ("incoming", "both"):
                query = """
                    SELECT id, source_id, target_id, edge_type, weight, created_at, metadata
                    FROM edges WHERE target_id = ?
                """
                params = [memory_id]
                if edge_type is not None:
                    query += " AND edge_type = ?"
                    params.append(edge_type)

                cursor.execute(query, params)
                results.extend(cursor.fetchall())

            import json

            return [
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "edge_type": row["edge_type"],
                    "weight": row["weight"],
                    "created_at": row["created_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                }
                for row in results
            ]

        except sqlite3.Error as e:
            raise SQLiteStoreError(f"Failed to get edges: {e}") from e

    def delete_edge(self, edge_id: int) -> bool:
        """Delete an edge by ID.

        Args:
            edge_id: The edge ID to delete

        Returns:
            True if edge was deleted, False if not found

        Raises:
            SQLiteStoreError: If delete operation fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))

            if cursor.rowcount == 0:
                return False

            self._conn.commit()
            return True

        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to delete edge: {e}") from e

    # =========================================================================
    # Outbox Operations (for ChromaDB sync)
    # =========================================================================

    def get_pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get pending outbox entries for ChromaDB sync.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of pending outbox entries

        Raises:
            SQLiteStoreError: If get operation fails
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, memory_id, operation, created_at, status
                FROM outbox
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "memory_id": row["memory_id"],
                    "operation": row["operation"],
                    "created_at": row["created_at"],
                    "status": row["status"],
                }
                for row in rows
            ]

        except sqlite3.Error as e:
            raise SQLiteStoreError(f"Failed to get pending outbox: {e}") from e

    def mark_outbox_processed(
        self,
        outbox_id: int,
        error_message: Optional[str] = None,
    ) -> bool:
        """Mark an outbox entry as processed.

        Args:
            outbox_id: The outbox entry ID
            error_message: Error message if processing failed

        Returns:
            True if entry was updated, False if not found

        Raises:
            SQLiteStoreError: If update operation fails
        """
        try:
            cursor = self._conn.cursor()
            status = "failed" if error_message else "processed"

            cursor.execute(
                """
                UPDATE outbox
                SET status = ?, processed_at = ?, error_message = ?
                WHERE id = ?
                """,
                (status, time.time(), error_message, outbox_id),
            )

            if cursor.rowcount == 0:
                return False

            self._conn.commit()
            return True

        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to mark outbox processed: {e}") from e

    def clear_processed_outbox(self, older_than_seconds: int = 86400) -> int:
        """Clear old processed outbox entries.

        Args:
            older_than_seconds: Delete entries older than this many seconds

        Returns:
            Number of entries deleted

        Raises:
            SQLiteStoreError: If delete operation fails
        """
        try:
            cursor = self._conn.cursor()
            cutoff = time.time() - older_than_seconds

            cursor.execute(
                """
                DELETE FROM outbox
                WHERE status = 'processed' AND processed_at < ?
                """,
                (cutoff,),
            )

            count = cursor.rowcount
            self._conn.commit()
            return count

        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to clear processed outbox: {e}") from e

    # =========================================================================
    # Utility Methods
    # =========================================================================

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
            SQLiteStoreError: If count operation fails
        """
        try:
            cursor = self._conn.cursor()

            query = "SELECT COUNT(*) FROM memories"
            conditions = []
            params: list[Any] = []

            if namespace is not None:
                conditions.append("namespace = ?")
                params.append(namespace)

            if memory_type is not None:
                conditions.append("type = ?")
                params.append(memory_type)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            cursor.execute(query, params)
            result = cursor.fetchone()
            return int(result[0]) if result else 0

        except sqlite3.Error as e:
            raise SQLiteStoreError(f"Failed to count memories: {e}") from e

    def clear(self) -> int:
        """Delete all data from all tables.

        Returns:
            Number of memories deleted

        Raises:
            SQLiteStoreError: If clear operation fails
        """
        try:
            cursor = self._conn.cursor()

            # Get count before deletion
            cursor.execute("SELECT COUNT(*) FROM memories")
            result = cursor.fetchone()
            count = int(result[0]) if result else 0

            # Delete in correct order to respect foreign keys
            cursor.execute("DELETE FROM outbox")
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM memories")

            self._conn.commit()
            return count

        except sqlite3.Error as e:
            self._conn.rollback()
            raise SQLiteStoreError(f"Failed to clear database: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def __enter__(self) -> "SQLiteStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
