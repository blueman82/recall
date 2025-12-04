# Recall - Long-term Memory System for Claude Code

## Project Overview

Recall is an MCP (Model Context Protocol) server that provides persistent memory storage and retrieval with semantic search capabilities. It enables Claude Code to remember user preferences, decisions, patterns, and session context across conversations.

## Quick Start

```bash
# Run as MCP server
uv run python -m recall

# Show help
uv run python -m recall --help

# Run tests
uv run pytest tests/
```

## Architecture

### Storage Layer (Hybrid Storage)
- **SQLite**: Structured metadata, memory records, relationships (edges)
- **ChromaDB**: Vector embeddings for semantic search
- **Ollama**: Local embedding generation via mxbai-embed-large model

### Core Components

```
src/recall/
├── __init__.py          # Package entry point
├── __main__.py          # MCP server with FastMCP tools
├── config.py            # Pydantic Settings configuration
├── memory/
│   ├── types.py         # Memory, Edge, MemoryType, RelationType
│   └── operations.py    # store, recall, relate, context, forget
├── storage/
│   ├── sqlite.py        # SQLite store implementation
│   ├── chromadb.py      # ChromaDB store implementation
│   └── hybrid.py        # Coordinated SQLite + ChromaDB layer
└── embedding/
    └── ollama.py        # Ollama embedding client
```

### Data Types

**MemoryType** (enum):
- `preference` - User preferences or settings
- `decision` - Design or implementation decisions
- `pattern` - Recognized patterns or recurring behaviors
- `session` - Session-related information

**RelationType** (enum):
- `relates_to` - General relationship
- `supersedes` - One memory replaces another
- `caused_by` - Causal relationship
- `contradicts` - Conflicting information

**Namespace** format:
- `global` - Cross-project memories
- `project:{name}` - Project-scoped memories

## MCP Tools

The server exposes 5 tools via FastMCP:

### memory_store_tool
Store a new memory with semantic indexing and deduplication.

### memory_recall_tool
Recall memories using semantic search with optional graph expansion.

### memory_relate_tool
Create a relationship between two memories.

### memory_context_tool
Fetch and format relevant memories for context injection.

### memory_forget_tool
Delete memories by ID or semantic search.

## Configuration

Settings are loaded via Pydantic Settings with `RECALL_` prefix:

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| RECALL_SQLITE_PATH | ~/.recall/recall.db | SQLite database path |
| RECALL_CHROMA_PATH | ~/.recall/chroma_db | ChromaDB storage path |
| RECALL_COLLECTION_NAME | memories | ChromaDB collection name |
| RECALL_OLLAMA_HOST | http://localhost:11434 | Ollama server URL |
| RECALL_OLLAMA_MODEL | mxbai-embed-large | Embedding model |
| RECALL_OLLAMA_TIMEOUT | 30 | Request timeout (seconds) |
| RECALL_LOG_LEVEL | INFO | Logging level |
| RECALL_DEFAULT_NAMESPACE | global | Default namespace |
| RECALL_DEFAULT_IMPORTANCE | 0.5 | Default importance score |
| RECALL_DEFAULT_TOKEN_BUDGET | 4000 | Default token budget |

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=recall

# Run specific test file
uv run pytest tests/integration/test_mcp_server.py

# Run specific test class
uv run pytest tests/integration/test_mcp_server.py::TestMCPToolHandlers
```

## Development Patterns

### Async Operations
All memory operations (store, recall, forget, context) are async. Use `await` when calling.

### Error Handling
Operations return result objects with `success` boolean and optional `error` message.

### Deduplication
Content is hashed (SHA-256, truncated to 16 chars) for deduplication within namespaces.

### Graph Expansion
Set `include_related=True` in recall to follow relationship edges.

## Important Notes

- **STDIO Transport**: MCP uses stdio - all logging goes to stderr, never stdout
- **Ollama Dependency**: Requires Ollama running locally with mxbai-embed-large model
- **Signal Handling**: SIGINT/SIGTERM trigger graceful shutdown
- **Python 3.13+**: Requires Python 3.13 or later
