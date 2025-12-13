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
    ForgetResult,
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
    """Store a new memory with semantic indexing and deduplication.

    Args:
        content: The memory content text
        memory_type: Type of memory (preference, decision, pattern, session)
        namespace: Scope of the memory ('global' or 'project:{name}')
        importance: Importance score from 0.0 to 1.0 (default: 0.5)
        metadata: Optional additional metadata as dict

    Returns:
        Result dictionary with success status, memory id, and content_hash
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

        return {
            "success": result.success,
            "id": result.id,
            "content_hash": result.content_hash,
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
) -> dict[str, Any]:
    """Recall memories using semantic search with optional graph expansion.

    Args:
        query: Search query text
        n_results: Maximum number of results (default: 5)
        namespace: Filter by namespace (optional)
        memory_type: Filter by memory type (optional)
        min_importance: Minimum importance score filter (0.0 to 1.0, optional)
        include_related: If True, include related memories via graph edges

    Returns:
        Dictionary with memories list, total count, and average score
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

        return {
            "success": True,
            "memories": memories_data,
            "total": result.total,
            "score": result.score,
        }

    except Exception as e:
        logger.error(f"memory_recall_tool failed: {e}", exc_info=True)
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
    namespace: Optional[str] = None,
    n_results: int = 5,
    confirm: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Delete memories by ID or semantic search.

    Golden rules (type=golden_rule or confidence >= 0.9) are protected from
    deletion unless force=True is specified.

    Args:
        memory_id: Specific memory ID to delete (direct deletion mode)
        query: Search query to find memories to delete (search deletion mode)
        namespace: Filter deletion to specific namespace (optional)
        n_results: Number of search results to delete in query mode (default: 5)
        confirm: If True, proceed with deletion (default: True)
        force: If True, allow deletion of golden rules (default: False)

    Returns:
        Result dictionary with success status, deleted_ids, and deleted_count
    """
    if hybrid_store is None:
        return {"success": False, "error": "Server not initialized"}

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
        "memory_relate": memory_relate_tool,
        "memory_context": memory_context_tool,
        "memory_forget": memory_forget_tool,
        "memory_validate": memory_validate_tool,
        "memory_apply": memory_apply_tool,
        "memory_outcome": memory_outcome_tool,
        "memory_detect_contradictions": memory_detect_contradictions,
        "memory_check_supersedes": memory_check_supersedes,
        "memory_analyze_health": memory_analyze_health,
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
