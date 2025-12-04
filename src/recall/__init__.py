"""Recall - AI memory and context management.

This package provides the Recall MCP server for persistent memory storage
and retrieval with semantic search capabilities.

Main components:
- memory.operations: High-level memory operations (store, recall, relate, context, forget)
- storage.hybrid: Coordinated SQLite + ChromaDB storage layer
- config: Pydantic Settings for configuration management

Usage:
    # Run as MCP server
    python -m recall

    # Or use the CLI
    recall --help
"""

__all__ = ["main"]
__version__ = "0.1.0"


def main() -> None:
    """Main entry point for the Recall MCP server."""
    from recall.__main__ import main as _main
    _main()
