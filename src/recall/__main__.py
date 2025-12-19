"""MCP server entry point for the Recall memory system.

This module provides the main entry point for the MCP server with:
- CLI argument parsing for flexible configuration
- Pydantic Settings for environment variable support
- Component initialization in dependency order
- Tool registration for all memory operations
- Signal handling for graceful shutdown
- Comprehensive logging to stderr (CRITICAL for MCP stdio)

Usage:
    python -m recall [options]

    Options:
        --sqlite-path PATH      SQLite database path
        --chroma-path PATH      ChromaDB storage path
        --collection NAME       Collection name (default: memories)
        --ollama-host HOST      Ollama server host (default: http://localhost:11434)
        --ollama-model MODEL    Embedding model name (default: mxbai-embed-large)
        --log-level LEVEL       Logging level (default: INFO)
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from recall.config import RecallSettings
from recall.memory.operations import (
    EdgeForgetResult,
    ForgetResult,
    edge_forget,
    inspect_graph,
    is_memory_id,
    memory_apply,
    memory_context,
    memory_forget,
    memory_outcome,
    memory_recall,
    memory_relate,
    memory_store,
    memory_validate,
)
from recall.memory.types import MemoryType, RelationType
from recall.storage.hybrid import HybridStore
from recall.validation import (
    analyze_memory_health,
    check_supersedes,
    detect_contradictions,
)

# Initialize FastMCP server
mcp = FastMCP("recall")

# Global components (initialized in main)
hybrid_store: Optional[HybridStore] = None

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging to stderr (never stdout for MCP servers).

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Note:
        STDIO-based MCP servers must never write to stdout as it corrupts
        JSON-RPC messages. All logging goes to stderr.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger to write to stderr
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Critical: never use stdout in MCP servers
    )

    logger.info(f"Logging initialized at {log_level.upper()} level")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments with configuration defaults.

    Returns:
        Parsed arguments namespace

    Configuration precedence:
        1. CLI arguments (highest priority)
        2. Environment variables (RECALL_ prefix)
        3. Defaults (lowest priority)
    """
    # Load settings for defaults
    settings = RecallSettings()

    parser = argparse.ArgumentParser(
        description="Recall MCP server for AI memory management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Direct tool invocation mode (for hooks)
    parser.add_argument(
        "--call",
        type=str,
        metavar="TOOL_NAME",
        help="Directly invoke a tool by name (memory_store, memory_recall, memory_context, memory_forget, memory_relate)",
    )
    parser.add_argument(
        "--args",
        type=str,
        default="{}",
        help="JSON arguments for the tool (used with --call)",
    )

    # Storage configuration
    parser.add_argument(
        "--sqlite-path",
        type=str,
        default=str(settings.sqlite_path) if settings.sqlite_path else None,
        help="SQLite database path (default: ~/.recall/recall.db)",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=str(settings.chroma_path) if settings.chroma_path else None,
        help="ChromaDB storage path (default: ~/.recall/chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.collection_name,
        help="ChromaDB collection name",
    )

    # Ollama configuration
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=settings.ollama_host,
        help="Ollama server host URL",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=settings.ollama_model,
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=settings.ollama_timeout,
        help="Ollama request timeout in seconds",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


async def initialize_components(args: argparse.Namespace) -> HybridStore:
    """Initialize HybridStore with all component dependencies.

    Initialization follows the dependency graph:
    1. Parse paths from args
    2. Create HybridStore with all dependencies

    Args:
        args: Parsed CLI arguments

    Returns:
        Configured HybridStore instance

    Raises:
        Exception: If any component initialization fails
    """
    logger.info("Initializing components...")

    # Parse paths
    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else None
    chroma_path = Path(args.chroma_path) if args.chroma_path else None

    logger.info(
        f"Configuration: "
        f"sqlite_path={sqlite_path}, "
        f"chroma_path={chroma_path}, "
        f"collection={args.collection}, "
        f"ollama_host={args.ollama_host}, "
        f"ollama_model={args.ollama_model}"
    )

    # Create HybridStore with factory method
    store = await HybridStore.create(
        sqlite_path=sqlite_path,
        chroma_path=chroma_path,
        collection_name=args.collection,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        ephemeral=False,
        sync_on_write=True,
    )

    logger.info("HybridStore initialized successfully")
    return store


# =============================================================================
# MCP Tool Handlers - Memory Store
# =============================================================================


@mcp.tool()
async def memory_store_tool(
    content: str,
    memory_type: str = "session",
    namespace: str = "global",
    importance: float = 0.5,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Store a new memory with semantic indexing, deduplication, and automatic relationship inference.

    Automatically finds semantically similar existing memories and creates
    relationship edges using LLM classification. This builds the knowledge
    graph automatically - no manual edge creation needed.

    Args:
        content: The memory content text
        memory_type: Type of memory (preference, decision, pattern, session)
        namespace: Scope of the memory ('global' or 'project:{name}')
        importance: Importance score from 0.0 to 1.0 (default: 0.5)
        metadata: Optional additional metadata as dict

    Returns:
        Result dictionary with:
        - success: Boolean indicating operation success
        - id: Memory ID (if successful)
        - content_hash: Content hash for deduplication
        - auto_relationships: List of automatically inferred relationships, each with:
            - target_id: ID of the related memory
            - relation: Relationship type (relates_to, supersedes, caused_by, contradicts)
            - confidence: LLM confidence in the relationship (0.0-1.0)
            - reason: Brief explanation of why the relationship was created
        - error: Error message (if failed)
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        # Convert string to MemoryType enum
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid memory_type: {memory_type}. "
                f"Must be one of: {[t.value for t in MemoryType]}",
            }

        result = await memory_store(
            store=hybrid_store,
            content=content,
            memory_type=mem_type,
            namespace=namespace,
            importance=importance,
            metadata=metadata,
        )

        # Include auto_relationships in response
        auto_relationships = getattr(result, 'auto_relationships', [])

        return {
            "success": result.success,
            "id": result.id,
            "content_hash": result.content_hash,
            "auto_relationships": auto_relationships,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"memory_store_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - Memory Recall
# =============================================================================


@mcp.tool()
async def memory_recall_tool(
    query: str,
    n_results: int = 5,
    namespace: Optional[str] = None,
    memory_type: Optional[str] = None,
    min_importance: Optional[float] = None,
    include_related: bool = False,
    max_depth: int = 1,
    max_expanded: int = 20,
    decay_factor: float = 0.7,
    include_edge_types: Optional[list[str]] = None,
    exclude_edge_types: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Recall memories using semantic search with optional multi-hop graph expansion.

    Performs semantic search using ChromaDB vector similarity, applies filters,
    and optionally expands results via graph edges using configurable multi-hop
    traversal.

    Args:
        query: Search query text (will be embedded with mxbai query prefix)
        n_results: Maximum number of primary results (default: 5)
        namespace: Filter by namespace (optional, e.g., 'global' or 'project:myapp')
        memory_type: Filter by memory type (optional, e.g., 'preference', 'decision')
        min_importance: Minimum importance score filter (0.0 to 1.0, optional)
        include_related: If True, include related memories via graph edges (default: False)
        max_depth: Maximum number of hops for graph expansion (default: 1)
        max_expanded: Maximum number of expanded memories to return (default: 20)
        decay_factor: Factor by which relevance decays per hop (default: 0.7)
        include_edge_types: Optional list of edge types to include (None means all).
            Valid types: relates_to, supersedes, caused_by, contradicts
        exclude_edge_types: Optional list of edge types to exclude (None means none).
            Valid types: relates_to, supersedes, caused_by, contradicts

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - memories: List of primary memory dicts with id, content, type, etc.
        - total: Total count of primary memories returned
        - score: Average similarity score of primary memories (or None)
        - expanded: List of expanded memory dicts (when include_related=True) with:
            - id: Memory ID
            - content: Memory content
            - type: Memory type
            - relevance_score: Combined relevance score (0.0 to 1.0)
            - hop_distance: Number of edges traversed to reach this memory
            - path: List of edge types in traversal order
            - explanation: Human-readable relevance explanation
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        # Convert string to MemoryType enum if provided
        mem_type = None
        if memory_type:
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid memory_type: {memory_type}. "
                    f"Must be one of: {[t.value for t in MemoryType]}",
                }

        result = await memory_recall(
            store=hybrid_store,
            query=query,
            n_results=n_results,
            namespace=namespace,
            memory_type=mem_type,
            min_importance=min_importance,
            include_related=include_related,
            max_depth=max_depth,
            max_expanded=max_expanded,
            decay_factor=decay_factor,
            include_edge_types=include_edge_types,
            exclude_edge_types=exclude_edge_types,
        )

        # Convert Memory objects to dicts for JSON serialization
        memories_data = []
        for memory in result.memories:
            memories_data.append({
                "id": memory.id,
                "content": memory.content,
                "content_hash": memory.content_hash,
                "type": memory.type.value,
                "namespace": memory.namespace,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat(),
                "accessed_at": memory.accessed_at.isoformat(),
                "access_count": memory.access_count,
            })

        # Convert ExpandedMemory objects to dicts for JSON serialization
        expanded_data = []
        for expanded in result.expanded_memories:
            expanded_data.append({
                "id": expanded.memory.id,
                "content": expanded.memory.content,
                "type": expanded.memory.type.value,
                "relevance_score": expanded.relevance_score,
                "hop_distance": expanded.hop_distance,
                "path": expanded.path,
                "explanation": expanded.explanation,
            })

        return {
            "success": True,
            "memories": memories_data,
            "total": result.total,
            "score": result.score,
            "expanded": expanded_data,
        }

    except Exception as e:
        logger.error(f"memory_recall_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - Graph Inspection
# =============================================================================


@mcp.tool()
async def memory_inspect_graph_tool(
    memory_id: str,
    max_depth: int = 2,
    direction: str = "both",
    edge_types: Optional[list[str]] = None,
    include_scores: bool = True,
    decay_factor: float = 0.7,
    output_format: str = "json",
) -> dict[str, Any]:
    """Inspect the graph structure around a memory node.

    Performs read-only breadth-first search from the origin memory, collecting
    all nodes and edges within max_depth hops. Returns structured data for
    visualization including Mermaid diagram generation.

    Args:
        memory_id: ID of the memory to start inspection from
        max_depth: Maximum number of hops to traverse (default: 2)
        direction: Edge traversal direction - "outgoing", "incoming", or "both" (default: "both")
        edge_types: Optional list of edge types to include (None means all).
            Valid types: relates_to, supersedes, caused_by, contradicts
        include_scores: If True, compute relevance scores for paths (default: True)
        decay_factor: Factor by which relevance decays per hop (default: 0.7)
        output_format: Output format - "json" or "mermaid" (default: "json")

    Returns:
        Dictionary with:
        - success: Boolean indicating operation success
        - origin_id: The starting memory ID
        - nodes: List of node dicts with id, content_preview, type, confidence, importance
        - edges: List of edge dicts with id, source_id, target_id, edge_type, weight
        - paths: List of path dicts with node_ids, edge_types, total_weight, relevance_score
        - stats: Dict with node_count, edge_count, max_depth_reached, origin_id
        - mermaid: Mermaid diagram string (only when output_format='mermaid')
        - error: Error message (if failed)
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    # Validate direction parameter
    if direction not in ("outgoing", "incoming", "both"):
        return {
            "success": False,
            "error": f"Invalid direction: {direction}. Must be one of: outgoing, incoming, both",
        }

    # Validate output_format parameter
    if output_format not in ("json", "mermaid"):
        return {
            "success": False,
            "error": f"Invalid output_format: {output_format}. Must be one of: json, mermaid",
        }

    try:
        result = await inspect_graph(
            store=hybrid_store,
            memory_id=memory_id,
            max_depth=max_depth,
            direction=direction,
            edge_types=edge_types,
            include_scores=include_scores,
            decay_factor=decay_factor,
        )

        if not result.success:
            return {
                "success": False,
                "error": result.error or f"Memory not found: {memory_id}",
            }

        # Format response based on output_format
        if output_format == "mermaid":
            return {
                "success": True,
                "origin_id": result.origin_id,
                "mermaid": result.to_mermaid(),
                "stats": {
                    "node_count": result.stats.node_count,
                    "edge_count": result.stats.edge_count,
                    "max_depth_reached": result.stats.max_depth_reached,
                    "origin_id": result.stats.origin_id,
                },
            }

        # JSON format - convert dataclasses to dicts
        nodes_data = []
        for node in result.nodes:
            nodes_data.append({
                "id": node.id,
                "content_preview": node.content_preview,
                "type": node.memory_type,
                "confidence": node.confidence,
                "importance": node.importance,
            })

        edges_data = []
        for edge in result.edges:
            edges_data.append({
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
            })

        paths_data = []
        for path in result.paths:
            paths_data.append({
                "node_ids": path.node_ids,
                "edge_types": path.edge_types,
                "total_weight": path.total_weight,
                "relevance_score": path.relevance_score,
            })

        return {
            "success": True,
            "origin_id": result.origin_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "paths": paths_data,
            "stats": {
                "node_count": result.stats.node_count,
                "edge_count": result.stats.edge_count,
                "max_depth_reached": result.stats.max_depth_reached,
                "origin_id": result.stats.origin_id,
            },
        }

    except Exception as e:
        logger.error(f"memory_inspect_graph_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - Memory Relate
# =============================================================================


@mcp.tool()
async def memory_relate_tool(
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Create a relationship between two memories.

    Args:
        source_id: ID of the source memory
        target_id: ID of the target memory
        relation: Type of relationship (relates_to, supersedes, caused_by, contradicts)
        weight: Edge weight (default: 1.0)

    Returns:
        Result dictionary with success status and edge_id
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        # Convert string to RelationType enum
        try:
            rel_type = RelationType(relation)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid relation: {relation}. "
                f"Must be one of: {[r.value for r in RelationType]}",
            }

        # memory_relate uses SQLiteStore directly
        edge_id = memory_relate(
            store=hybrid_store._sqlite,
            source_id=source_id,
            target_id=target_id,
            relation=rel_type,
            weight=weight,
        )

        return {
            "success": True,
            "edge_id": edge_id,
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"memory_relate_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - Memory Context
# =============================================================================


@mcp.tool()
async def memory_context_tool(
    query: Optional[str] = None,
    project: Optional[str] = None,
    token_budget: int = 4000,
) -> dict[str, Any]:
    """Fetch relevant memories and format them for context injection.

    Args:
        query: Optional search query to filter relevant memories
        project: Project namespace (auto-detected from cwd if not specified)
        token_budget: Maximum tokens for context (default: 4000)

    Returns:
        Dictionary with success status and formatted markdown context
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        context = await memory_context(
            store=hybrid_store,
            query=query,
            project=project,
            token_budget=token_budget,
        )

        return {
            "success": True,
            "context": context,
            "token_estimate": len(context) // 4,
        }

    except Exception as e:
        logger.error(f"memory_context_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - Memory Forget
# =============================================================================


@mcp.tool()
async def memory_forget_tool(
    memory_id: Optional[str] = None,
    query: Optional[str] = None,
    input: Optional[str] = None,
    namespace: Optional[str] = None,
    n_results: int = 5,
    confirm: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Delete memories by ID or semantic search.

    Golden rules (type=golden_rule or confidence >= 0.9) are protected from
    deletion unless force=True is specified.

    Args:
        memory_id: Specific memory ID to delete (direct deletion mode).
        query: Search query to find memories to delete (search deletion mode).
        input: Smart parameter that auto-detects if value is an ID or query.
        namespace: Filter deletion to specific namespace (optional).
        n_results: Number of search results to delete in query mode (default: 5).
        confirm: If True, proceed with deletion (default: True).
        force: If True, allow deletion of golden rules (default: False).

    Returns:
        Result dictionary with success status, deleted_ids, and deleted_count.

    Note:
        If both memory_id and query are None but input is provided, the function
        auto-detects whether input is a memory ID or search query.

    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    # Auto-detect input type if using smart parameter
    if memory_id is None and query is None and input is not None:
        if is_memory_id(input):
            memory_id = input
        else:
            query = input

    try:
        result = await memory_forget(
            store=hybrid_store,
            memory_id=memory_id,
            query=query,
            namespace=namespace,
            n_results=n_results,
            confirm=confirm,
            force=force,
        )

        return {
            "success": result.success,
            "deleted_ids": result.deleted_ids,
            "deleted_count": result.deleted_count,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"memory_forget_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - File Activity
# =============================================================================


@mcp.tool()
async def file_activity_add(
    file_path: str,
    action: str,
    session_id: Optional[str] = None,
    project_root: Optional[str] = None,
    file_type: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Record a file activity event.

    Used by PostToolUse hooks to track what files have been accessed.

    Args:
        file_path: Path to the file that was accessed
        action: Type of action (read, write, edit, multiedit)
        session_id: Optional session ID for grouping activities
        project_root: Optional project root directory
        file_type: Optional file type (e.g., 'python', 'typescript')
        metadata: Optional additional metadata

    Returns:
        Result dictionary with success status and activity_id
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        activity_id = hybrid_store.add_file_activity(
            file_path=file_path,
            action=action,
            session_id=session_id,
            project_root=project_root,
            file_type=file_type,
            metadata=metadata,
        )

        return {
            "success": True,
            "activity_id": activity_id,
        }

    except Exception as e:
        logger.error(f"file_activity_add failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def file_activity_recent(
    project_root: Optional[str] = None,
    limit: int = 20,
    days: int = 14,
) -> dict[str, Any]:
    """Get recently accessed files with aggregated activity.

    Args:
        project_root: Filter by project root (optional)
        limit: Maximum number of files to return (default: 20)
        days: Look back this many days (default: 14)

    Returns:
        Dictionary with success status and list of recent files
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        files = hybrid_store.get_recent_files(
            project_root=project_root,
            limit=limit,
            days=days,
        )

        return {
            "success": True,
            "files": files,
            "count": len(files),
        }

    except Exception as e:
        logger.error(f"file_activity_recent failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - Memory Validation
# =============================================================================


@mcp.tool()
async def memory_validate_tool(
    memory_id: str,
    success: bool,
    adjustment: float = 0.1,
) -> dict[str, Any]:
    """Validate a memory and adjust its confidence score.

    Adjusts confidence based on whether the memory was useful:
    - Success: confidence += adjustment (max 1.0)
    - Failure: confidence -= adjustment * 1.5 (min 0.0)

    Automatically promotes to GOLDEN_RULE when confidence reaches 0.9.

    Args:
        memory_id: ID of the memory to validate
        success: Whether the memory application was successful
        adjustment: Base confidence adjustment (default: 0.1)

    Returns:
        Result with old/new confidence, promotion status, or error
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        result = await memory_validate(
            store=hybrid_store,
            memory_id=memory_id,
            success=success,
            adjustment=adjustment,
        )

        return {
            "success": result.success,
            "memory_id": result.memory_id,
            "old_confidence": result.old_confidence,
            "new_confidence": result.new_confidence,
            "promoted": result.promoted,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"memory_validate_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_apply_tool(
    memory_id: str,
    context: str,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Record that a memory is being applied.

    Creates a validation event to track when a memory is used in practice.
    This starts the TRY phase of the validation loop.

    Args:
        memory_id: ID of the memory being applied
        context: Description of how/where the memory is being applied
        session_id: Optional session identifier

    Returns:
        Result with event ID or error
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        result = await memory_apply(
            store=hybrid_store,
            memory_id=memory_id,
            context=context,
            session_id=session_id,
        )

        return {
            "success": result.success,
            "memory_id": result.memory_id,
            "event_id": result.event_id,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"memory_apply_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_outcome_tool(
    memory_id: str,
    success: bool,
    error_msg: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Record the outcome of a memory application and adjust confidence.

    Records whether applying a memory succeeded or failed, creating a
    validation event and adjusting confidence accordingly.

    Args:
        memory_id: ID of the memory that was applied
        success: Whether the application was successful
        error_msg: Optional error message if failed
        session_id: Optional session identifier

    Returns:
        Result with updated confidence and promotion status
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        result = await memory_outcome(
            store=hybrid_store,
            memory_id=memory_id,
            success=success,
            error_msg=error_msg,
            session_id=session_id,
        )

        return {
            "success": result.success,
            "memory_id": result.memory_id,
            "outcome_success": result.outcome_success,
            "new_confidence": result.new_confidence,
            "promoted": result.promoted,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"memory_outcome_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# MCP Tool Handlers - Validation & Analysis
# =============================================================================


@mcp.tool()
async def memory_detect_contradictions(
    memory_id: str,
    similarity_threshold: float = 0.7,
    create_edges: bool = True,
) -> dict[str, Any]:
    """Detect memories that contradict a given memory.

    Uses semantic search to find similar memories, then LLM reasoning
    to determine if they actually contradict each other.

    Args:
        memory_id: ID of the memory to check for contradictions
        similarity_threshold: Minimum similarity for considering contradictions (default: 0.7)
        create_edges: Whether to create CONTRADICTS edges (default: True)

    Returns:
        Result with list of contradicting memory IDs and edges created
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        result = await detect_contradictions(
            store=hybrid_store,
            memory_id=memory_id,
            similarity_threshold=similarity_threshold,
            create_edges=create_edges,
        )

        return {
            "success": result.error is None,
            "memory_id": result.memory_id,
            "contradictions": result.contradictions,
            "edges_created": result.edges_created,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"memory_detect_contradictions failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_check_supersedes(
    memory_id: str,
    create_edge: bool = True,
) -> dict[str, Any]:
    """Check if a memory should supersede another based on validation history.

    A newer memory supersedes an older one when it consistently succeeds
    where the older one failed on similar topics.

    Args:
        memory_id: ID of the (potentially newer) memory to check
        create_edge: Whether to create SUPERSEDES edge (default: True)

    Returns:
        Result with superseded memory ID if applicable
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        result = await check_supersedes(
            store=hybrid_store,
            memory_id=memory_id,
            create_edge=create_edge,
        )

        return {
            "success": result.error is None,
            "memory_id": result.memory_id,
            "superseded_id": result.superseded_id,
            "edge_created": result.edge_created,
            "reason": result.reason,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"memory_check_supersedes failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_analyze_health(
    namespace: Optional[str] = None,
    include_contradictions: bool = True,
    include_low_confidence: bool = True,
    include_stale: bool = True,
    stale_days: int = 30,
) -> dict[str, Any]:
    """Analyze the health of memories in the system.

    Checks for unresolved contradictions, low-confidence memories,
    and stale memories that haven't been validated recently.

    Args:
        namespace: Limit analysis to specific namespace (optional)
        include_contradictions: Check for contradictions (default: True)
        include_low_confidence: Find low-confidence memories (default: True)
        include_stale: Find stale memories (default: True)
        stale_days: Days without validation to consider stale (default: 30)

    Returns:
        Dictionary with categorized issues and recommendations
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        issues = await analyze_memory_health(
            store=hybrid_store,
            namespace=namespace,
            include_contradictions=include_contradictions,
            include_low_confidence=include_low_confidence,
            include_stale=include_stale,
            stale_days=stale_days,
        )

        return {
            "success": True,
            **issues,
        }

    except Exception as e:
        logger.error(f"memory_analyze_health failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# Memory Inspection Tools
# =============================================================================


@mcp.tool()
async def memory_count_tool(
    namespace: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> dict[str, Any]:
    """Count memories with optional filters.

    Provides a quick count of memories in the system, optionally
    filtered by namespace and/or memory type.

    Args:
        namespace: Filter by namespace (optional, e.g., 'global' or 'project:myapp')
        memory_type: Filter by type (optional, e.g., 'preference', 'decision', 'golden_rule')

    Returns:
        Dictionary with count and applied filters
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        count = hybrid_store.count_memories(
            namespace=namespace,
            memory_type=memory_type,
        )
        return {
            "success": True,
            "count": count,
            "filters": {
                "namespace": namespace,
                "memory_type": memory_type,
            },
        }
    except Exception as e:
        logger.error(f"memory_count_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def memory_list_tool(
    namespace: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "created_at",
    descending: bool = True,
) -> dict[str, Any]:
    """List memories with filtering and pagination.

    Browse memories in the system with optional filters and pagination.
    Useful for auditing, exploring, or debugging memory contents.

    Args:
        namespace: Filter by namespace (optional)
        memory_type: Filter by type (optional)
        limit: Maximum number of results (default: 100, max: 1000)
        offset: Number of results to skip for pagination (default: 0)
        order_by: Field to sort by (default: 'created_at', options: 'created_at', 'accessed_at', 'importance', 'confidence')
        descending: Sort in descending order (default: True)

    Returns:
        Dictionary with list of memories and pagination info
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    # Clamp limit to prevent excessive results
    limit = min(limit, 1000)

    try:
        memories = hybrid_store.list_memories(
            namespace=namespace,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
            order_by=order_by,
            descending=descending,
        )
        return {
            "success": True,
            "memories": memories,
            "count": len(memories),
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(memories) == limit,
            },
            "filters": {
                "namespace": namespace,
                "memory_type": memory_type,
                "order_by": order_by,
                "descending": descending,
            },
        }
    except Exception as e:
        logger.error(f"memory_list_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
async def validation_history_tool(
    memory_id: str,
    event_type: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get validation event history for a memory.

    Shows the history of validation events (applied, succeeded, failed)
    for a specific memory. Useful for understanding why a memory has
    its current confidence score.

    Args:
        memory_id: ID of the memory to get history for
        event_type: Filter by event type (optional: 'applied', 'succeeded', 'failed')
        limit: Maximum number of events to return (default: 50)

    Returns:
        Dictionary with validation events and summary statistics
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

    try:
        events = hybrid_store.get_validation_events(
            memory_id=memory_id,
            event_type=event_type,
            limit=limit,
        )

        # Calculate summary stats
        success_count = sum(1 for e in events if e.get("event_type") == "succeeded")
        failure_count = sum(1 for e in events if e.get("event_type") == "failed")
        applied_count = sum(1 for e in events if e.get("event_type") == "applied")

        return {
            "success": True,
            "memory_id": memory_id,
            "events": events,
            "summary": {
                "total_events": len(events),
                "applied": applied_count,
                "succeeded": success_count,
                "failed": failure_count,
                "success_rate": (
                    success_count / (success_count + failure_count)
                    if (success_count + failure_count) > 0
                    else None
                ),
            },
        }
    except Exception as e:
        logger.error(f"validation_history_tool failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# =============================================================================
# Direct Tool Invocation (for hooks)
# =============================================================================


async def call_tool_directly(
    tool_name: str,
    args_json: str,
    store: HybridStore,
) -> dict[str, Any]:
    """Directly invoke a tool without MCP protocol overhead.

    This is used by Claude Code hooks for fast, direct tool invocation.

    Args:
        tool_name: Name of the tool to call (memory_store, memory_recall, etc.)
        args_json: JSON string of arguments for the tool
        store: Initialized HybridStore

    Returns:
        Tool result as dictionary
    """
    global hybrid_store
    hybrid_store = store

    try:
        tool_args = json.loads(args_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON arguments: {e}"}

    # Map tool names to handlers
    tool_handlers = {
        "memory_store": memory_store_tool,
        "memory_recall": memory_recall_tool,
        "memory_inspect_graph": memory_inspect_graph_tool,
        "memory_relate": memory_relate_tool,
        "memory_context": memory_context_tool,
        "memory_forget": memory_forget_tool,
        "memory_validate": memory_validate_tool,
        "memory_apply": memory_apply_tool,
        "memory_outcome": memory_outcome_tool,
        "memory_detect_contradictions": memory_detect_contradictions,
        "memory_check_supersedes": memory_check_supersedes,
        "memory_analyze_health": memory_analyze_health,
        "memory_count": memory_count_tool,
        "memory_list": memory_list_tool,
        "validation_history": validation_history_tool,
        "file_activity_add": file_activity_add,
        "file_activity_recent": file_activity_recent,
    }

    handler = tool_handlers.get(tool_name)
    if not handler:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}. Available: {list(tool_handlers.keys())}",
        }

    try:
        result = await handler(**tool_args)
        return result
    except TypeError as e:
        return {"success": False, "error": f"Invalid arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Tool execution failed: {e}"}


def run_direct_call(args: argparse.Namespace) -> None:
    """Run a direct tool call and print result to stdout.

    Args:
        args: Parsed CLI arguments with --call and --args
    """
    setup_logging("WARNING")  # Quiet logging for direct calls

    async def _run():
        store = await initialize_components(args)
        try:
            result = await call_tool_directly(args.call, args.args, store)
            print(json.dumps(result))
        finally:
            await store.close()

    asyncio.run(_run())


# =============================================================================
# Signal Handling
# =============================================================================


def handle_shutdown(signum: int, frame: Any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown.

    Logs shutdown signal and performs cleanup before exit.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # Note: FastMCP handles cleanup automatically
    sys.exit(0)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for MCP server.

    Workflow:
    1. Parse CLI arguments
    2. If --call provided, run direct tool invocation and exit
    3. Setup logging
    4. Initialize components
    5. Register signal handlers
    6. Run MCP server with stdio transport

    Note:
        Uses stdio transport for MCP communication. All logging
        goes to stderr to avoid corrupting JSON-RPC messages on stdout.
    """
    global hybrid_store

    # Parse arguments
    args = parse_arguments()

    # Direct tool invocation mode (for hooks)
    if args.call:
        run_direct_call(args)
        return

    # Setup logging to stderr
    setup_logging(args.log_level)

    logger.info("Starting Recall MCP Server...")

    try:
        # Note: We need to initialize components before running the MCP server
        # FastMCP manages its own event loop, so we use run_until_complete
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Initialize HybridStore
        hybrid_store = loop.run_until_complete(initialize_components(args))

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        logger.info("MCP server ready, starting stdio transport...")

        # Run MCP server with stdio transport
        # This blocks until server shuts down
        # Note: mcp.run() is synchronous and manages its own event loop
        mcp.run(transport="stdio")

    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if hybrid_store is not None:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(hybrid_store.close())
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


if __name__ == "__main__":
    main()
