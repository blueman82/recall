"""Memory operations for the recall system.

This module provides high-level operations for storing and managing memories,
including deduplication, embedding generation, and optional relationship creation.
"""

import hashlib
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from recall.memory.types import (
    ApplyResult,
    ExpandedMemory,
    GraphEdge,
    GraphExpansionConfig,
    GraphInspectionResult,
    GraphNode,
    GraphPath,
    GraphStats,
    Memory,
    MemoryType,
    OutcomeResult,
    RecallResult,
    RelationType,
    StoreResult,
    ValidateResult,
)
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


def _dict_to_memory(data: dict[str, Any]) -> Memory:
    """Convert a storage dict to a Memory dataclass.

    Handles MemoryType enum conversion with fallback to SESSION if the type
    value is not recognized.

    Args:
        data: Dictionary containing memory data from storage

    Returns:
        Memory dataclass instance
    """
    # Extract memory type - convert string to MemoryType enum if possible
    result_type = data.get("type", "session")
    try:
        mem_type = MemoryType(result_type)
    except ValueError:
        # Use SESSION as default if type doesn't match enum
        mem_type = MemoryType.SESSION

    return Memory(
        id=data["id"],
        content=data["content"],
        content_hash=data.get("content_hash", ""),
        type=mem_type,
        namespace=data.get("namespace", "global"),
        importance=data.get("importance", 0.5),
        confidence=data.get("confidence", 0.3),
        created_at=datetime.fromtimestamp(data.get("created_at", datetime.now().timestamp())),
        accessed_at=datetime.fromtimestamp(data.get("accessed_at", datetime.now().timestamp())),
        access_count=data.get("access_count", 0),
        metadata=data.get("metadata"),
    )


async def _expand_related_memories(
    store: HybridStore,
    primary_memories: list[Memory],
    primary_scores: list[float],
    config: GraphExpansionConfig,
    min_importance: float = 0.0,
) -> list[ExpandedMemory]:
    """Expand from seed memories using BFS traversal with relevance scoring.

    Performs breadth-first search from the given seed memories, following edges
    up to max_depth hops. Each expanded memory receives a relevance score based
    on hop distance, edge weights, and edge type importance.

    Relevance score formula:
        relevance = decay_factor^hop_distance * path_weight_product * geometric_mean(type_weights)

    Args:
        store: HybridStore instance for edge traversal
        primary_memories: List of starting memories from semantic search
        primary_scores: List of similarity scores for each primary memory
        config: GraphExpansionConfig controlling traversal behavior
        min_importance: Minimum relevance score cutoff (skip if relevance_score < min_importance)

    Returns:
        List of ExpandedMemory objects discovered through graph traversal,
        sorted by relevance_score descending
    """
    # Track visited nodes to prevent cycles
    seen_ids: set[str] = {m.id for m in primary_memories}
    expanded: list[ExpandedMemory] = []
    nodes_visited = 0

    # BFS queue: (memory_id, hop_distance, path_edges, path_weight)
    # path_edges is a list of edge type strings
    # path_weight is the product of edge weights along the path
    queue: deque[tuple[str, int, list[str], float]] = deque()

    # Initialize queue with primary memories (direct edges from them)
    for memory in primary_memories:
        queue.append((memory.id, 0, [], 1.0))

    while queue and len(expanded) < config.max_expanded and nodes_visited < config.max_nodes_visited:
        current_id, hop_distance, path_edges, path_weight = queue.popleft()
        nodes_visited += 1

        # Stop if we've exceeded max depth
        if hop_distance >= config.max_depth:
            continue

        # Get edges from current node (both directions)
        edges = store.get_edges(current_id, direction="both")

        # Safety guard: limit edges per node
        edges = edges[: config.max_edges_per_node]

        for edge in edges:
            # Determine the related memory ID (could be source or target)
            related_id = edge["target_id"] if edge["source_id"] == current_id else edge["source_id"]

            # Skip if already seen
            if related_id in seen_ids:
                continue

            edge_type = edge.get("edge_type", "relates_to")

            # Apply edge type filtering
            if config.include_edge_types is not None and edge_type not in config.include_edge_types:
                continue
            if config.exclude_edge_types is not None and edge_type in config.exclude_edge_types:
                continue

            # Calculate new path metrics
            edge_weight = edge.get("weight", 1.0)
            new_path_weight = path_weight * edge_weight
            new_path_edges = path_edges + [edge_type]
            new_hop_distance = hop_distance + 1

            # Compute relevance score
            # decay_factor^hop_distance
            decay_component = config.decay_factor ** new_hop_distance

            # Geometric mean of type weights for edges in path
            type_weights_product = 1.0
            for e_type in new_path_edges:
                type_weights_product *= config.edge_type_weights.get(e_type, 0.7)
            geometric_mean = type_weights_product ** (1.0 / len(new_path_edges))

            # Final relevance score
            relevance_score = decay_component * new_path_weight * geometric_mean

            # Fetch the related memory
            related_data = await store.get_memory(related_id)
            if related_data is None:
                seen_ids.add(related_id)
                continue

            # Filter by memory's importance field (backward compatibility)
            memory_importance = related_data.get("importance", 0.5)
            if memory_importance < min_importance:
                seen_ids.add(related_id)
                continue

            # Convert to Memory object
            related_memory = _dict_to_memory(related_data)

            # Generate explanation string
            # Example: "2 hops via supersedes → relates_to, combined weight 0.42"
            edge_path_str = " → ".join(new_path_edges)
            explanation = f"{new_hop_distance} hop{'s' if new_hop_distance != 1 else ''} via {edge_path_str}, combined weight {relevance_score:.2f}"

            # Create ExpandedMemory
            expanded_memory = ExpandedMemory(
                memory=related_memory,
                relevance_score=relevance_score,
                hop_distance=new_hop_distance,
                path=new_path_edges,
                edge_weight_product=new_path_weight,
                explanation=explanation,
            )

            expanded.append(expanded_memory)
            seen_ids.add(related_id)

            # Add to queue for further expansion if within depth limit
            if new_hop_distance < config.max_depth:
                queue.append((related_id, new_hop_distance, new_path_edges, new_path_weight))

            # Early termination if we've collected enough
            if len(expanded) >= config.max_expanded:
                break

    # Sort by relevance score descending
    expanded.sort(key=lambda x: x.relevance_score, reverse=True)

    return expanded


async def inspect_graph(
    store: HybridStore,
    memory_id: str,
    max_depth: int = 2,
    direction: str = "both",
    edge_types: Optional[list[str]] = None,
    include_scores: bool = True,
    decay_factor: float = 0.7,
) -> GraphInspectionResult:
    """Inspect the graph structure around a memory node using BFS traversal.

    Performs read-only breadth-first search from the origin memory, collecting
    all nodes and edges within max_depth hops. Returns structured data for
    visualization including Mermaid diagram generation.

    This function is READ-ONLY - it does not modify any data in the store.

    Args:
        store: HybridStore instance for storage operations
        memory_id: ID of the memory to start inspection from
        max_depth: Maximum number of hops to traverse (default: 2)
        direction: Edge traversal direction - "outgoing", "incoming", or "both" (default: "both")
        edge_types: Optional list of edge types to include (None means all)
        include_scores: If True, compute relevance scores for paths (default: True)
        decay_factor: Factor by which relevance decays per hop (default: 0.7)

    Returns:
        GraphInspectionResult with nodes, edges, paths, stats, and to_mermaid() support

    Example:
        >>> store = await HybridStore.create(ephemeral=True)
        >>> result = await inspect_graph(store, "mem_123", max_depth=2)
        >>> if result.success:
        ...     print(f"Found {result.stats.node_count} nodes")
        ...     print(result.to_mermaid())  # Mermaid diagram
    """
    # Output size caps to prevent oversized responses
    MAX_NODES = 100
    MAX_EDGES = 200
    MAX_PATHS = 50

    # Validate direction parameter
    if direction not in ("outgoing", "incoming", "both"):
        return GraphInspectionResult(
            success=False,
            error=f"Invalid direction '{direction}'. Must be 'outgoing', 'incoming', or 'both'.",
        )

    # Verify origin memory exists
    origin_data = await store.get_memory(memory_id)
    if origin_data is None:
        return GraphInspectionResult(
            success=False,
            error=f"Origin memory '{memory_id}' not found",
        )

    # Convert edge type filters to set for O(1) lookup
    edge_types_set: Optional[set[str]] = None
    if edge_types is not None:
        edge_types_set = set(edge_types)

    # Build GraphExpansionConfig for scoring
    config = GraphExpansionConfig(
        max_depth=max_depth,
        decay_factor=decay_factor,
        include_edge_types=edge_types_set,
        exclude_edge_types=None,
    )

    # Initialize result collections
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    paths: list[GraphPath] = []

    # Track visited nodes and edges to avoid duplicates
    seen_node_ids: set[str] = set()
    seen_edge_keys: set[tuple[str, str, str]] = set()  # (source_id, target_id, edge_type)

    # Edge ID counter
    edge_id_counter = 0

    # Track maximum depth actually reached
    max_depth_reached = 0

    # BFS queue: (memory_id, depth, path_node_ids, path_edge_types, path_weight)
    queue: deque[tuple[str, int, list[str], list[str], float]] = deque()

    # Add origin node
    origin_memory = _dict_to_memory(origin_data)
    origin_node = GraphNode.from_memory(origin_memory)
    nodes.append(origin_node)
    seen_node_ids.add(memory_id)

    # Initialize queue with origin
    queue.append((memory_id, 0, [memory_id], [], 1.0))

    try:
        while queue and len(nodes) < MAX_NODES:
            current_id, depth, path_node_ids, path_edge_types, path_weight = queue.popleft()

            # Track maximum depth reached
            if depth > max_depth_reached:
                max_depth_reached = depth

            # Stop expanding if we've exceeded max depth
            if depth >= max_depth:
                # Record path for leaf nodes (if we have paths to record)
                if len(path_node_ids) > 1 and len(paths) < MAX_PATHS and include_scores:
                    # Compute relevance score
                    relevance_score = _compute_path_relevance(
                        path_edge_types, path_weight, config
                    )
                    paths.append(GraphPath(
                        node_ids=path_node_ids.copy(),
                        edge_types=path_edge_types.copy(),
                        total_weight=path_weight,
                        relevance_score=relevance_score,
                    ))
                continue

            # Get edges from current node based on direction
            edge_list = store.get_edges(current_id, direction=direction)

            # Track if this node has any valid outgoing edges
            has_children = False

            for edge in edge_list:
                if len(edges) >= MAX_EDGES:
                    break

                edge_type = edge.get("edge_type", "relates_to")

                # Apply edge type filtering (only include types in the list if specified)
                if edge_types_set is not None and edge_type not in edge_types_set:
                    continue

                # Determine the related memory ID based on direction
                source_id = edge["source_id"]
                target_id = edge["target_id"]

                # For "outgoing", we follow edges where current is source
                # For "incoming", we follow edges where current is target
                # For "both", we follow in both directions
                if direction == "outgoing" and source_id != current_id:
                    continue
                if direction == "incoming" and target_id != current_id:
                    continue

                related_id = target_id if source_id == current_id else source_id

                # Create edge key for deduplication
                edge_key = (source_id, target_id, edge_type)
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)

                    # Create GraphEdge
                    edge_weight = edge.get("weight", 1.0)
                    graph_edge = GraphEdge(
                        id=edge_id_counter,
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=edge_type,
                        weight=edge_weight,
                    )
                    edges.append(graph_edge)
                    edge_id_counter += 1

                # Add related node if not seen
                if related_id not in seen_node_ids and len(nodes) < MAX_NODES:
                    seen_node_ids.add(related_id)
                    has_children = True

                    # Fetch related memory
                    related_data = await store.get_memory(related_id)
                    if related_data is not None:
                        related_memory = _dict_to_memory(related_data)
                        related_node = GraphNode.from_memory(related_memory)
                        nodes.append(related_node)

                        # Calculate new path metrics
                        edge_weight = edge.get("weight", 1.0)
                        new_path_weight = path_weight * edge_weight
                        new_path_node_ids = path_node_ids + [related_id]
                        new_path_edge_types = path_edge_types + [edge_type]

                        # Add to queue for further expansion
                        queue.append((
                            related_id,
                            depth + 1,
                            new_path_node_ids,
                            new_path_edge_types,
                            new_path_weight,
                        ))

            # Record path for leaf nodes (nodes with no unvisited children)
            if not has_children and len(path_node_ids) > 1 and len(paths) < MAX_PATHS and include_scores:
                relevance_score = _compute_path_relevance(
                    path_edge_types, path_weight, config
                )
                paths.append(GraphPath(
                    node_ids=path_node_ids.copy(),
                    edge_types=path_edge_types.copy(),
                    total_weight=path_weight,
                    relevance_score=relevance_score,
                ))

    except Exception as e:
        return GraphInspectionResult(
            success=False,
            error=f"Error during graph traversal: {e}",
        )

    # Build stats
    stats = GraphStats(
        node_count=len(nodes),
        edge_count=len(edges),
        max_depth_reached=max_depth_reached,
        origin_id=memory_id,
    )

    # Sort paths by relevance score descending
    paths.sort(key=lambda p: p.relevance_score, reverse=True)

    return GraphInspectionResult(
        success=True,
        origin_id=memory_id,
        nodes=nodes,
        edges=edges,
        paths=paths[:MAX_PATHS],  # Ensure we don't exceed limit
        stats=stats,
    )


def _compute_path_relevance(
    edge_types: list[str],
    path_weight: float,
    config: GraphExpansionConfig,
) -> float:
    """Compute relevance score for a path using GraphExpansionConfig formula.

    Formula: decay^hop * weight_product * geometric_mean(type_weights)

    Args:
        edge_types: List of edge types in the path
        path_weight: Product of edge weights along the path
        config: GraphExpansionConfig with decay_factor and edge_type_weights

    Returns:
        Relevance score between 0.0 and 1.0
    """
    if not edge_types:
        return 1.0

    hop_distance = len(edge_types)

    # decay_factor^hop_distance
    decay_component = config.decay_factor ** hop_distance

    # Geometric mean of type weights for edges in path
    type_weights_product = 1.0
    for e_type in edge_types:
        type_weights_product *= config.edge_type_weights.get(e_type, 0.7)
    geometric_mean = type_weights_product ** (1.0 / len(edge_types))

    # Final relevance score
    return decay_component * path_weight * geometric_mean


async def memory_recall(
    store: HybridStore,
    query: str,
    n_results: int = 5,
    namespace: Optional[str] = None,
    memory_type: Optional[MemoryType] = None,
    min_importance: Optional[float] = None,
    include_related: bool = False,
    max_depth: int = 1,
    max_expanded: int = 20,
    decay_factor: float = 0.7,
    include_edge_types: Optional[list[str]] = None,
    exclude_edge_types: Optional[list[str]] = None,
) -> RecallResult:
    """Recall memories using semantic search with optional multi-hop graph expansion.

    Performs semantic search using ChromaDB vector similarity, applies filters,
    updates access statistics, and optionally expands results via graph edges
    using configurable multi-hop traversal.

    Args:
        store: HybridStore instance for storage operations
        query: Search query text (will be embedded with mxbai query prefix)
        n_results: Maximum number of primary results (default: 5)
        namespace: Filter by namespace (optional)
        memory_type: Filter by memory type (optional)
        min_importance: Minimum importance score filter (0.0 to 1.0, optional)
        include_related: If True, include related memories via graph edges (default: False)
        max_depth: Maximum number of hops for graph expansion (default: 1)
        max_expanded: Maximum number of expanded memories to return (default: 20)
        decay_factor: Factor by which relevance decays per hop (default: 0.7)
        include_edge_types: Optional list of edge types to include (None means all)
        exclude_edge_types: Optional list of edge types to exclude (None means none)

    Returns:
        RecallResult containing matched memories, total count, average score, and
        expanded_memories list when include_related=True

    Example:
        >>> store = await HybridStore.create(ephemeral=True)
        >>> result = await memory_recall(
        ...     store=store,
        ...     query="What are user preferences?",
        ...     namespace="project:myapp",
        ...     min_importance=0.5,
        ...     include_related=True,
        ...     max_depth=2,
        ...     decay_factor=0.8,
        ... )
        >>> for memory in result.memories:
        ...     print(f"{memory.id}: {memory.content}")
        >>> for expanded in result.expanded_memories:
        ...     print(f"{expanded.memory.id}: {expanded.explanation}")
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
        expanded_memories: list[ExpandedMemory] = []
        if include_related and memories:
            # Convert list parameters to sets for O(1) lookup
            include_types_set: Optional[set[str]] = None
            if include_edge_types is not None:
                include_types_set = set(include_edge_types)

            exclude_types_set: Optional[set[str]] = None
            if exclude_edge_types is not None:
                exclude_types_set = set(exclude_edge_types)

            # Build GraphExpansionConfig from parameters
            config = GraphExpansionConfig(
                max_depth=max_depth,
                decay_factor=decay_factor,
                include_edge_types=include_types_set,
                exclude_edge_types=exclude_types_set,
                max_expanded=max_expanded,
            )

            # Call _expand_related_memories() helper
            expanded_memories = await _expand_related_memories(
                store=store,
                primary_memories=memories,
                primary_scores=scores,
                config=config,
                min_importance=min_importance if min_importance is not None else 0.0,
            )

        # Calculate average score if we have scores
        avg_score: Optional[float] = None
        if scores:
            avg_score = sum(scores) / len(scores)

        return RecallResult(
            memories=memories,
            total=len(memories),
            score=avg_score,
            expanded_memories=expanded_memories,
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

    # 4. Get ALL golden rules (they always appear regardless of namespace)
    # Golden rules are either type=golden_rule OR confidence >= 0.9
    all_golden_rules: list[dict[str, Any]] = []

    # Get project golden rules
    project_golden = store.list_memories(
        namespace=project_namespace,
        memory_type="golden_rule",
        limit=n_results,
    )
    all_golden_rules.extend(project_golden)

    # Get global golden rules
    global_golden = store.list_memories(
        namespace="global",
        memory_type="golden_rule",
        limit=n_results,
    )
    all_golden_rules.extend(global_golden)

    # Also find high-confidence memories that qualify as golden rules
    high_confidence_project = [
        mem for mem in store.list_memories(namespace=project_namespace, limit=100)
        if mem.get("confidence", 0.3) >= 0.9 and mem.get("type") != "golden_rule"
    ]
    all_golden_rules.extend(high_confidence_project)

    high_confidence_global = [
        mem for mem in store.list_memories(namespace="global", limit=100)
        if mem.get("confidence", 0.3) >= 0.9 and mem.get("type") != "golden_rule"
    ]
    all_golden_rules.extend(high_confidence_global)

    # Remove duplicates by ID
    seen_ids: set[str] = set()
    unique_memories: list[dict[str, Any]] = []
    golden_rules: list[dict[str, Any]] = []

    # First, collect golden rules (they get priority)
    for memory in all_golden_rules:
        if memory["id"] not in seen_ids:
            seen_ids.add(memory["id"])
            golden_rules.append(memory)

    # Then collect other memories, excluding golden rules
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

    # RFC 2119 preamble - defines semantics for requirement keywords
    # This ensures the consuming LLM knows MUST/SHOULD/MAY have precise meanings
    rfc2119_preamble = (
        "The key words \"MUST\", \"MUST NOT\", \"REQUIRED\", \"SHALL\", \"SHALL NOT\", "
        "\"SHOULD\", \"SHOULD NOT\", \"RECOMMENDED\", \"MAY\", and \"OPTIONAL\" in these "
        "memories are to be interpreted as described in RFC 2119.\n\n"
        "---\n\n"
    )
    current_tokens += estimate_tokens(rfc2119_preamble)
    sections.append(rfc2119_preamble)

    # Golden Rules section - always included, NOT subject to token budget
    # These are constitutional principles that must always be visible
    if golden_rules:
        golden_section = "## Golden Rules\n\n"
        for memory in golden_rules:
            content = memory.get("content", "")
            namespace = memory.get("namespace", "global")
            source = "[global]" if namespace == "global" else f"[{namespace}]"
            golden_section += f"- {content} {source}\n"
        golden_section += "\n"
        sections.append(golden_section)
        # Golden rules don't count against token budget - they're mandatory

    # Process regular sections in priority order (these respect token budget)
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


def _is_golden_rule(memory: dict[str, Any]) -> bool:
    """Check if a memory qualifies as a golden rule.

    A memory is a golden rule if:
    - Its type is 'golden_rule', OR
    - Its confidence is >= 0.9

    Args:
        memory: Memory dict to check

    Returns:
        True if the memory is a golden rule
    """
    mem_type = memory.get("type", "session")
    confidence = memory.get("confidence", 0.3)
    return mem_type == "golden_rule" or confidence >= 0.9


async def memory_forget(
    store: HybridStore,
    memory_id: Optional[str] = None,
    query: Optional[str] = None,
    namespace: Optional[str] = None,
    n_results: int = 5,
    confirm: bool = True,
    force: bool = False,
) -> ForgetResult:
    """Delete memories by ID or semantic search.

    Supports two modes:
    1. Direct ID deletion: If memory_id is provided, delete that specific memory
    2. Semantic search deletion: If query is provided, search and delete top matches

    Deletion is atomic - memories are removed from both SQLite and ChromaDB
    via HybridStore.delete_memory().

    Golden rules (type=golden_rule or confidence >= 0.9) are protected from
    deletion unless force=True is specified.

    Args:
        store: HybridStore instance for storage operations
        memory_id: Specific memory ID to delete (direct deletion mode)
        query: Search query to find memories to delete (search deletion mode)
        namespace: Filter deletion to specific namespace (optional)
        n_results: Number of search results to delete when using query mode (default: 5)
        confirm: If True, proceed with deletion (for future confirmation support)
        force: If True, allow deletion of golden rules (default: False)

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
        >>> # Force delete a golden rule
        >>> result = await memory_forget(store, memory_id="golden_123", force=True)
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

            # Check golden rule protection
            if _is_golden_rule(memory) and not force:
                return ForgetResult(
                    success=False,
                    error=f"Memory '{memory_id}' is a golden rule and cannot be deleted. "
                    "Use force=True to override this protection.",
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

            # Delete each matching memory (skip golden rules unless force=True)
            skipped_golden: list[str] = []
            for result in search_results:
                mem_id = result["id"]

                # Check golden rule protection
                if _is_golden_rule(result) and not force:
                    skipped_golden.append(mem_id)
                    continue

                try:
                    deleted = await store.delete_memory(mem_id)
                    if deleted:
                        deleted_ids.append(mem_id)
                except Exception:
                    # Continue with other deletions even if one fails
                    pass

            # If all results were golden rules, return error
            if skipped_golden and not deleted_ids:
                return ForgetResult(
                    success=False,
                    error=f"All matching memories are golden rules and cannot be deleted. "
                    f"Skipped: {skipped_golden}. Use force=True to override this protection.",
                )

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


async def _check_golden_promotion(
    store: HybridStore,
    memory_id: str,
    new_confidence: float,
) -> bool:
    """Check and perform golden rule promotion if eligible.

    Promotes a memory to GOLDEN_RULE type when confidence reaches 0.9.
    Only PREFERENCE, DECISION, and PATTERN types can be promoted.
    Original type is preserved in metadata.promoted_from.

    Args:
        store: HybridStore instance for storage operations
        memory_id: ID of the memory to check
        new_confidence: The updated confidence score

    Returns:
        True if memory was promoted, False otherwise
    """
    if new_confidence < 0.9:
        return False

    # Fetch the memory to check its type
    memory = await store.get_memory(memory_id)
    if memory is None:
        return False

    current_type = memory.get("type", "session")

    # Only these types can be promoted to golden rule
    promotable_types = {"preference", "decision", "pattern"}
    if current_type not in promotable_types:
        return False

    # Already a golden rule
    if current_type == "golden_rule":
        return False

    # Preserve original type in metadata
    metadata = memory.get("metadata") or {}
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    metadata["promoted_from"] = current_type

    # Update to golden rule type
    await store.update_memory(
        memory_id,
        memory_type="golden_rule",
        metadata=metadata,
    )

    return True


async def memory_validate(
    store: HybridStore,
    memory_id: str,
    success: bool,
    adjustment: float = 0.1,
) -> ValidateResult:
    """Validate a memory and adjust its confidence score.

    Adjusts confidence based on whether the memory was useful:
    - Success: confidence += adjustment (max 1.0)
    - Failure: confidence -= adjustment * 1.5 (min 0.0)

    Automatically promotes to GOLDEN_RULE when confidence reaches 0.9.

    Args:
        store: HybridStore instance for storage operations
        memory_id: ID of the memory to validate
        success: Whether the memory application was successful
        adjustment: Base confidence adjustment (default: 0.1)

    Returns:
        ValidateResult with old/new confidence, promotion status, or error

    Example:
        >>> result = await memory_validate(store, "mem_123", success=True)
        >>> print(f"Confidence: {result.old_confidence} -> {result.new_confidence}")
    """
    # Validate adjustment range
    if not 0.0 <= adjustment <= 1.0:
        return ValidateResult(
            success=False,
            error=f"Adjustment must be between 0.0 and 1.0, got {adjustment}",
        )

    # Fetch the memory
    memory = await store.get_memory(memory_id)
    if memory is None:
        return ValidateResult(
            success=False,
            error=f"Memory '{memory_id}' not found",
        )

    old_confidence = memory.get("confidence", 0.3)

    # Calculate new confidence
    if success:
        new_confidence = min(1.0, old_confidence + adjustment)
    else:
        # Failure reduces confidence faster (1.5x adjustment)
        new_confidence = max(0.0, old_confidence - (adjustment * 1.5))

    # Update the memory's confidence
    await store.update_memory(memory_id, confidence=new_confidence)

    # Check for golden rule promotion
    promoted = await _check_golden_promotion(store, memory_id, new_confidence)

    return ValidateResult(
        success=True,
        memory_id=memory_id,
        old_confidence=old_confidence,
        new_confidence=new_confidence,
        promoted=promoted,
    )


async def memory_apply(
    store: HybridStore,
    memory_id: str,
    context: str,
    session_id: Optional[str] = None,
) -> ApplyResult:
    """Record that a memory is being applied.

    Creates a validation event to track when a memory is used in practice.
    This starts the TRY phase of the validation loop.

    Args:
        store: HybridStore instance for storage operations
        memory_id: ID of the memory being applied
        context: Description of how/where the memory is being applied
        session_id: Optional session identifier

    Returns:
        ApplyResult with event ID or error

    Example:
        >>> result = await memory_apply(
        ...     store, "mem_123",
        ...     context="Applying dark mode preference to UI settings"
        ... )
        >>> if result.success:
        ...     # Later record the outcome
        ...     await memory_outcome(store, "mem_123", success=True)
    """
    # Verify memory exists
    memory = await store.get_memory(memory_id)
    if memory is None:
        return ApplyResult(
            success=False,
            error=f"Memory '{memory_id}' not found",
        )

    try:
        # Record the 'applied' event
        event_id = store.add_validation_event(
            memory_id=memory_id,
            event_type="applied",
            context=context,
            session_id=session_id,
        )

        # Update accessed_at timestamp
        await store.update_memory(memory_id)  # This updates accessed_at automatically

        return ApplyResult(
            success=True,
            memory_id=memory_id,
            event_id=event_id,
        )

    except Exception as e:
        return ApplyResult(
            success=False,
            error=f"Failed to record memory application: {e}",
        )


async def memory_outcome(
    store: HybridStore,
    memory_id: str,
    success: bool,
    error_msg: Optional[str] = None,
    session_id: Optional[str] = None,
) -> OutcomeResult:
    """Record the outcome of a memory application and adjust confidence.

    Records whether applying a memory succeeded or failed, creating a
    validation event and adjusting confidence accordingly.

    Args:
        store: HybridStore instance for storage operations
        memory_id: ID of the memory that was applied
        success: Whether the application was successful
        error_msg: Optional error message if failed
        session_id: Optional session identifier

    Returns:
        OutcomeResult with updated confidence and promotion status

    Example:
        >>> # After applying a memory and observing the result
        >>> result = await memory_outcome(store, "mem_123", success=False,
        ...     error_msg="User rejected the suggestion")
        >>> print(f"New confidence: {result.new_confidence}")
    """
    # Verify memory exists
    memory = await store.get_memory(memory_id)
    if memory is None:
        return OutcomeResult(
            success=False,
            error=f"Memory '{memory_id}' not found",
        )

    try:
        # Record the outcome event
        event_type = "succeeded" if success else "failed"
        context = error_msg if error_msg else ("Success" if success else "Failed")

        store.add_validation_event(
            memory_id=memory_id,
            event_type=event_type,
            context=context,
            session_id=session_id,
        )

        # Adjust confidence via memory_validate
        validate_result = await memory_validate(store, memory_id, success)

        if not validate_result.success:
            return OutcomeResult(
                success=False,
                error=validate_result.error,
            )

        return OutcomeResult(
            success=True,
            memory_id=memory_id,
            outcome_success=success,
            new_confidence=validate_result.new_confidence,
            promoted=validate_result.promoted,
        )

    except Exception as e:
        return OutcomeResult(
            success=False,
            error=f"Failed to record outcome: {e}",
        )
