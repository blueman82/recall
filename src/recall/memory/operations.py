"""Memory operations for the recall system.

This module provides high-level operations for storing and managing memories,
including deduplication, embedding generation, and optional relationship creation.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from recall.memory.types import Memory, MemoryType, RecallResult, RelationType, StoreResult
from recall.storage.hybrid import HybridStore
from recall.storage.sqlite import SQLiteStore


def _generate_memory_id() -> str:
    """Generate a globally unique memory ID using UUID4.

    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())


def _compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for deduplication.

    Uses the first 16 characters of the hex digest for a compact hash.

    Args:
        content: The content to hash

    Returns:
        Truncated hex-encoded SHA-256 hash string (16 characters)
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


async def memory_store(
    store: HybridStore,
    content: str,
    memory_type: MemoryType = MemoryType.SESSION,
    namespace: str = "global",
    importance: float = 0.5,
    metadata: Optional[dict[str, Any]] = None,
    relations: Optional[list[dict[str, str]]] = None,
) -> StoreResult:
    """Store a new memory with semantic indexing, deduplication, and optional relations.

    Handles the complete memory storage workflow:
    1. Generate unique ID and content hash
    2. Check for duplicate content in the namespace
    3. Store via HybridStore (SQLite + ChromaDB with embeddings)
    4. Optionally create edges to related memories

    Args:
        store: HybridStore instance for storage operations
        content: The memory content text
        memory_type: Type of memory (MemoryType enum, default: SESSION)
        namespace: Scope of the memory ('global' or 'project:{name}')
        importance: Importance score from 0.0 to 1.0 (default: 0.5)
        metadata: Optional additional metadata as dict
        relations: Optional list of dicts with {target_id, relation} to create edges

    Returns:
        StoreResult with success status, memory id (if successful), or error message

    Example:
        >>> store = await HybridStore.create(ephemeral=True)
        >>> result = await memory_store(
        ...     store=store,
        ...     content="User prefers dark mode",
        ...     memory_type=MemoryType.PREFERENCE,
        ...     namespace="project:myapp",
        ...     importance=0.8,
        ... )
        >>> if result.success:
        ...     print(f"Stored memory: {result.id}")
    """
    # Validate inputs
    if not content or not content.strip():
        return StoreResult(success=False, error="Content cannot be empty")

    if not 0.0 <= importance <= 1.0:
        return StoreResult(
            success=False,
            error=f"Importance must be between 0.0 and 1.0, got {importance}",
        )

    # Compute content hash for deduplication (truncated 16-char version)
    content_hash = _compute_content_hash(content)

    # Check for existing memory with same content hash in the namespace
    # Query SQLite directly through the HybridStore's internal SQLite store
    # SQLite stores full SHA-256 (64 chars), we compare with truncated (16 chars)
    existing_memories = store.list_memories(namespace=namespace, limit=1000)
    for memory in existing_memories:
        stored_hash = memory.get("content_hash", "")
        # Compare truncated hashes - SQLite stores full 64-char hash
        if stored_hash.startswith(content_hash) or stored_hash == content_hash:
            # Duplicate found - return existing ID with content_hash
            return StoreResult(
                success=True,
                id=memory["id"],
                content_hash=content_hash,
            )

    # Generate new memory ID
    memory_id = _generate_memory_id()

    try:
        # Convert MemoryType enum to string value for storage
        type_value = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)

        # Store memory via HybridStore
        stored_id = await store.add_memory(
            content=content,
            memory_type=type_value,
            namespace=namespace,
            importance=importance,
            metadata=metadata,
            memory_id=memory_id,
        )

        # Create edges for relations if provided
        if relations:
            for relation in relations:
                target_id = relation.get("target_id")
                relation_type = relation.get("relation", "related")

                if target_id:
                    try:
                        store.add_edge(
                            source_id=stored_id,
                            target_id=target_id,
                            edge_type=relation_type,
                        )
                    except Exception as e:
                        # Log but don't fail the whole operation for edge creation failures
                        # The memory was already stored successfully
                        pass

        return StoreResult(success=True, id=stored_id, content_hash=content_hash)

    except ValueError as e:
        return StoreResult(success=False, error=str(e))
    except Exception as e:
        return StoreResult(success=False, error=f"Failed to store memory: {e}")


async def memory_recall(
    store: HybridStore,
    query: str,
    n_results: int = 5,
    namespace: Optional[str] = None,
    memory_type: Optional[MemoryType] = None,
    min_importance: Optional[float] = None,
    include_related: bool = False,
) -> RecallResult:
    """Recall memories using semantic search with optional graph expansion.

    Performs semantic search using ChromaDB vector similarity, applies filters,
    updates access statistics, and optionally expands results via graph edges.

    Args:
        store: HybridStore instance for storage operations
        query: Search query text (will be embedded with mxbai query prefix)
        n_results: Maximum number of primary results (default: 5)
        namespace: Filter by namespace (optional)
        memory_type: Filter by memory type (optional)
        min_importance: Minimum importance score filter (0.0 to 1.0, optional)
        include_related: If True, include related memories via graph edges (default: False)

    Returns:
        RecallResult containing matched memories, total count, and average score

    Example:
        >>> store = await HybridStore.create(ephemeral=True)
        >>> result = await memory_recall(
        ...     store=store,
        ...     query="What are user preferences?",
        ...     namespace="project:myapp",
        ...     min_importance=0.5,
        ...     include_related=True,
        ... )
        >>> for memory in result.memories:
        ...     print(f"{memory.id}: {memory.content}")
    """
    if not query or not query.strip():
        return RecallResult(memories=[], total=0, score=None)

    try:
        # Convert MemoryType enum to string value for filtering
        type_filter = memory_type.value if isinstance(memory_type, MemoryType) else memory_type

        # Perform semantic search via HybridStore
        # HybridStore.search() uses is_query=True for embedding (adds mxbai prefix)
        # and builds where filter for namespace/type
        search_results = await store.search(
            query=query,
            n_results=n_results,
            namespace=namespace,
            memory_type=type_filter,
        )

        # Convert search results to Memory objects
        memories: list[Memory] = []
        scores: list[float] = []

        for result in search_results:
            # Filter by min_importance if specified
            result_importance = result.get("importance", 0.5)
            if min_importance is not None and result_importance < min_importance:
                continue

            # Extract memory type - convert string to MemoryType enum if possible
            result_type = result.get("type", "session")
            try:
                mem_type = MemoryType(result_type)
            except ValueError:
                # Use SESSION as default if type doesn't match enum
                mem_type = MemoryType.SESSION

            # Build Memory object from result
            memory = Memory(
                id=result["id"],
                content=result["content"],
                content_hash=result.get("content_hash", ""),
                type=mem_type,
                namespace=result.get("namespace", "global"),
                importance=result_importance,
                created_at=datetime.fromtimestamp(result.get("created_at", datetime.now().timestamp())),
                accessed_at=datetime.fromtimestamp(result.get("accessed_at", datetime.now().timestamp())),
                access_count=result.get("access_count", 0),
            )
            memories.append(memory)

            # Track similarity scores
            if "similarity" in result:
                scores.append(result["similarity"])

        # Expand via graph edges if include_related is True
        related_memories: list[Memory] = []
        if include_related and memories:
            seen_ids = {m.id for m in memories}

            for memory in memories:
                # Get edges connected to this memory (both directions)
                edges = store.get_edges(memory.id, direction="both")

                for edge in edges:
                    # Get the related memory ID (could be source or target)
                    related_id = edge["target_id"] if edge["source_id"] == memory.id else edge["source_id"]

                    # Skip if we've already included this memory
                    if related_id in seen_ids:
                        continue

                    # Fetch the related memory
                    related_result = await store.get_memory(related_id)
                    if related_result:
                        # Filter related memory by min_importance if specified
                        related_importance = related_result.get("importance", 0.5)
                        if min_importance is not None and related_importance < min_importance:
                            seen_ids.add(related_id)  # Mark as seen to avoid re-fetching
                            continue

                        # Convert to Memory object
                        related_type = related_result.get("type", "session")
                        try:
                            rel_mem_type = MemoryType(related_type)
                        except ValueError:
                            rel_mem_type = MemoryType.SESSION

                        related_memory = Memory(
                            id=related_result["id"],
                            content=related_result["content"],
                            content_hash=related_result.get("content_hash", ""),
                            type=rel_mem_type,
                            namespace=related_result.get("namespace", "global"),
                            importance=related_importance,
                            created_at=datetime.fromtimestamp(related_result.get("created_at", datetime.now().timestamp())),
                            accessed_at=datetime.fromtimestamp(related_result.get("accessed_at", datetime.now().timestamp())),
                            access_count=related_result.get("access_count", 0),
                        )
                        related_memories.append(related_memory)
                        seen_ids.add(related_id)

        # Combine primary results with related memories
        all_memories = memories + related_memories

        # Calculate average score if we have scores
        avg_score: Optional[float] = None
        if scores:
            avg_score = sum(scores) / len(scores)

        return RecallResult(
            memories=all_memories,
            total=len(all_memories),
            score=avg_score,
        )

    except Exception as e:
        # Return empty result on error instead of raising
        return RecallResult(memories=[], total=0, score=None)


async def memory_context(
    store: HybridStore,
    query: Optional[str] = None,
    project: Optional[str] = None,
    token_budget: int = 4000,
    n_results: int = 20,
) -> str:
    """Fetch relevant memories and format them for context injection in session_start hooks.

    Fetches memories from global and project namespaces, ranks by composite
    score (recency + importance + access), formats as markdown sections,
    and truncates to token budget.

    Composite scoring formula:
        score = importance * recency_factor * log(access_count + 1)

    Where recency_factor decays based on time since last access.

    Args:
        store: HybridStore instance for storage operations
        query: Optional search query to filter relevant memories
        project: Project namespace (auto-detected from cwd if not specified)
        token_budget: Maximum tokens for context (default: 4000, estimated as chars/4)
        n_results: Maximum number of memories to fetch per namespace (default: 20)

    Returns:
        Markdown-formatted context string with sections:
        - ## Preferences
        - ## Recent Decisions
        - ## Patterns

    Example:
        >>> store = await HybridStore.create(ephemeral=True)
        >>> context = await memory_context(store, token_budget=2000)
        >>> print(context)  # Markdown sections with relevant memories
    """
    import math
    import os
    import time

    # Auto-detect project from cwd if not specified
    if project is None:
        project_name = os.path.basename(os.getcwd())
        project_namespace = f"project:{project_name}"
    else:
        project_namespace = f"project:{project}" if not project.startswith("project:") else project

    # Collect memories - prioritize project-specific, filter globals
    all_memories: list[dict[str, Any]] = []

    # 1. Get ALL project-specific memories (these are always relevant)
    project_memories = store.list_memories(
        namespace=project_namespace,
        limit=n_results,
        order_by="accessed_at",
        descending=True,
    )
    # Tag project memories for boosting
    for mem in project_memories:
        mem["_is_project"] = True
    all_memories.extend(project_memories)

    # 2. Get ONLY global PREFERENCES (user preferences apply everywhere)
    #    Skip global patterns/decisions - they're usually project-specific context
    global_memories = store.list_memories(
        namespace="global",
        limit=n_results,
        order_by="importance",
        descending=True,
    )
    # Filter to only preferences with importance >= 0.6
    global_preferences = [
        mem for mem in global_memories
        if mem.get("type") == "preference" and mem.get("importance", 0.5) >= 0.6
    ]
    # Tag global memories
    for mem in global_preferences:
        mem["_is_project"] = False
    all_memories.extend(global_preferences)

    # 3. If query provided, also do semantic search for relevant globals
    if query:
        try:
            search_results = await store.search(
                query=query,
                n_results=min(5, n_results),  # Limit semantic global results
                namespace="global",
            )
            for mem in search_results:
                mem["_is_project"] = False
            all_memories.extend(search_results)
        except Exception:
            pass  # Semantic search failed, continue with what we have

    # Remove duplicates by ID
    seen_ids: set[str] = set()
    unique_memories: list[dict[str, Any]] = []
    for memory in all_memories:
        if memory["id"] not in seen_ids:
            seen_ids.add(memory["id"])
            unique_memories.append(memory)

    # Calculate composite scores
    now = time.time()
    scored_memories: list[tuple[float, dict[str, Any]]] = []

    for memory in unique_memories:
        importance = memory.get("importance", 0.5)
        access_count = memory.get("access_count", 0)
        accessed_at = memory.get("accessed_at", now)
        is_project = memory.get("_is_project", False)

        # Recency factor: decay based on time since last access
        # Using exponential decay with half-life of 7 days
        age_seconds = now - accessed_at
        age_days = age_seconds / (24 * 60 * 60)
        half_life_days = 7
        recency_factor = math.pow(0.5, age_days / half_life_days)

        # Composite score: importance * recency_factor * log(access_count + 1)
        # Add 1 to access_count to handle log(0) case
        access_factor = math.log(access_count + 1) + 1  # +1 base so 0 accesses gives score > 0
        score = importance * recency_factor * access_factor

        # BOOST: Project-specific memories get 3x priority over globals
        if is_project:
            score *= 3.0

        scored_memories.append((score, memory))

    # Sort by score descending
    scored_memories.sort(key=lambda x: x[0], reverse=True)

    # Group memories by type
    preferences: list[tuple[float, dict[str, Any]]] = []
    decisions: list[tuple[float, dict[str, Any]]] = []
    patterns: list[tuple[float, dict[str, Any]]] = []

    for score, memory in scored_memories:
        memory_type = memory.get("type", "session")
        if memory_type == "preference":
            preferences.append((score, memory))
        elif memory_type == "decision":
            decisions.append((score, memory))
        elif memory_type == "pattern":
            patterns.append((score, memory))
        # Skip session type for context injection

    # Build markdown sections
    def format_memory(memory: dict[str, Any]) -> str:
        """Format a single memory as a bullet point."""
        content = memory.get("content", "")
        namespace = memory.get("namespace", "global")
        # Indicate source namespace
        source = "[global]" if namespace == "global" else f"[{namespace}]"
        return f"- {content} {source}"

    def estimate_tokens(text: str) -> int:
        """Estimate tokens as chars/4."""
        return len(text) // 4

    # Build context with token budget enforcement
    sections: list[str] = []
    current_tokens = 0
    max_tokens = token_budget

    # Header
    header = "# Memory Context\n\n"
    current_tokens += estimate_tokens(header)
    sections.append(header)

    # Process sections in priority order
    section_data = [
        ("## Preferences\n\n", preferences),
        ("## Recent Decisions\n\n", decisions),
        ("## Patterns\n\n", patterns),
    ]

    for section_header, memories in section_data:
        if not memories:
            continue

        section_content = section_header
        section_tokens = estimate_tokens(section_header)

        for score, memory in memories:
            entry = format_memory(memory) + "\n"
            entry_tokens = estimate_tokens(entry)

            if current_tokens + section_tokens + entry_tokens > max_tokens:
                # Token budget exceeded, stop adding memories
                break

            section_content += entry
            section_tokens += entry_tokens

        # Only add section if it has content beyond header
        if section_content != section_header:
            sections.append(section_content + "\n")
            current_tokens += section_tokens

        if current_tokens >= max_tokens:
            break

    return "".join(sections).strip()


@dataclass
class ForgetResult:
    """Result of a memory forget operation.

    Indicates whether deletion succeeded and provides details about
    what was deleted or an error message if it failed.

    Attributes:
        success: Whether the operation succeeded
        deleted_ids: List of memory IDs that were deleted
        deleted_count: Number of memories deleted
        error: Error message (if failed)
    """
    success: bool
    deleted_ids: list[str] = field(default_factory=list)
    deleted_count: int = 0
    error: Optional[str] = None


async def memory_forget(
    store: HybridStore,
    memory_id: Optional[str] = None,
    query: Optional[str] = None,
    namespace: Optional[str] = None,
    n_results: int = 5,
    confirm: bool = True,
) -> ForgetResult:
    """Delete memories by ID or semantic search.

    Supports two modes:
    1. Direct ID deletion: If memory_id is provided, delete that specific memory
    2. Semantic search deletion: If query is provided, search and delete top matches

    Deletion is atomic - memories are removed from both SQLite and ChromaDB
    via HybridStore.delete_memory().

    Args:
        store: HybridStore instance for storage operations
        memory_id: Specific memory ID to delete (direct deletion mode)
        query: Search query to find memories to delete (search deletion mode)
        namespace: Filter deletion to specific namespace (optional)
        n_results: Number of search results to delete when using query mode (default: 5)
        confirm: If True, proceed with deletion (for future confirmation support)

    Returns:
        ForgetResult with success status, deleted IDs, count, or error message

    Raises:
        ValueError: If neither memory_id nor query is provided

    Example:
        >>> store = await HybridStore.create(ephemeral=True)
        >>> # Direct ID deletion
        >>> result = await memory_forget(store, memory_id="mem_123")
        >>> # Semantic search deletion
        >>> result = await memory_forget(store, query="outdated preferences", n_results=3)
    """
    # Validate inputs - must provide either memory_id or query
    if memory_id is None and query is None:
        return ForgetResult(
            success=False,
            error="Must provide either memory_id or query for deletion",
        )

    if memory_id is not None and query is not None:
        return ForgetResult(
            success=False,
            error="Cannot provide both memory_id and query - use one mode at a time",
        )

    if not confirm:
        return ForgetResult(
            success=False,
            error="Deletion not confirmed",
        )

    deleted_ids: list[str] = []

    try:
        if memory_id is not None:
            # Direct ID deletion mode
            # Verify memory exists and optionally check namespace
            memory = await store.get_memory(memory_id)
            if memory is None:
                return ForgetResult(
                    success=False,
                    error=f"Memory '{memory_id}' not found",
                )

            # Check namespace if specified
            if namespace is not None and memory.get("namespace") != namespace:
                return ForgetResult(
                    success=False,
                    error=f"Memory '{memory_id}' not in namespace '{namespace}'",
                )

            # Delete the memory
            deleted = await store.delete_memory(memory_id)
            if deleted:
                deleted_ids.append(memory_id)

        else:
            # Semantic search deletion mode
            # Search for memories matching the query
            # Note: query is guaranteed to be str here due to validation above
            assert query is not None  # mypy: narrow type from str | None to str
            search_results = await store.search(
                query=query,
                n_results=n_results,
                namespace=namespace,
            )

            if not search_results:
                return ForgetResult(
                    success=True,
                    deleted_ids=[],
                    deleted_count=0,
                )

            # Delete each matching memory
            for result in search_results:
                mem_id = result["id"]
                try:
                    deleted = await store.delete_memory(mem_id)
                    if deleted:
                        deleted_ids.append(mem_id)
                except Exception:
                    # Continue with other deletions even if one fails
                    pass

        return ForgetResult(
            success=True,
            deleted_ids=deleted_ids,
            deleted_count=len(deleted_ids),
        )

    except Exception as e:
        return ForgetResult(
            success=False,
            deleted_ids=deleted_ids,
            deleted_count=len(deleted_ids),
            error=f"Failed to delete memories: {e}",
        )


def memory_relate(
    store: SQLiteStore,
    source_id: str,
    target_id: str,
    relation: RelationType,
    weight: float = 1.0,
) -> int:
    """Create a relationship between two memories.

    Validates both memories exist, validates relation type, and upserts the edge.
    Special handling for 'supersedes' relation which reduces target importance by 50%.

    Args:
        store: SQLiteStore instance for storage operations
        source_id: ID of the source memory
        target_id: ID of the target memory
        relation: Type of relationship (RelationType enum)
        weight: Edge weight (default: 1.0)

    Returns:
        The ID of the created/updated edge

    Raises:
        ValueError: If source or target memory doesn't exist, or relation is invalid

    Example:
        >>> store = SQLiteStore(ephemeral=True)
        >>> source_id = store.add_memory(content="New info about X")
        >>> target_id = store.add_memory(content="Old info about X")
        >>> edge_id = memory_relate(
        ...     store=store,
        ...     source_id=source_id,
        ...     target_id=target_id,
        ...     relation=RelationType.SUPERSEDES,
        ... )
        >>> # Target memory importance is now reduced by 50%
    """
    # Validate relation type is a valid RelationType enum
    if not isinstance(relation, RelationType):
        # Try to convert string to RelationType
        try:
            relation = RelationType(relation)
        except (ValueError, KeyError):
            valid_values = [r.value for r in RelationType]
            raise ValueError(
                f"Invalid relation type. Must be one of: {valid_values}"
            )

    # Validate source memory exists
    source_memory = store.get_memory(source_id)
    if source_memory is None:
        raise ValueError(f"Source memory '{source_id}' not found")

    # Validate target memory exists
    target_memory = store.get_memory(target_id)
    if target_memory is None:
        raise ValueError(f"Target memory '{target_id}' not found")

    # Handle 'supersedes' relation special case - reduce target importance
    if relation == RelationType.SUPERSEDES:
        current_importance = target_memory.get("importance", 0.5)
        new_importance = current_importance * 0.5
        store.update_memory(target_id, importance=new_importance)

    # Upsert edge (INSERT OR REPLACE)
    # First check if edge already exists with same source, target, and type
    existing_edges = store.get_edges(source_id, direction="outgoing", edge_type=relation.value)
    existing_edge = None
    for edge in existing_edges:
        if edge["target_id"] == target_id:
            existing_edge = edge
            break

    if existing_edge:
        # Update existing edge - delete and recreate with new weight
        store.delete_edge(existing_edge["id"])

    # Create the edge
    edge_id = store.add_edge(
        source_id=source_id,
        target_id=target_id,
        edge_type=relation.value,
        weight=weight,
    )

    return edge_id
