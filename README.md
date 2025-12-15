# Recall

Long-term memory system for MCP-compatible AI assistants with semantic search and relationship tracking.

## Features

- **Persistent Memory Storage**: Store preferences, decisions, patterns, and session context
- **Semantic Search**: Find relevant memories using natural language queries via ChromaDB vectors
- **Memory Relationships**: Create edges between memories (supersedes, relates_to, caused_by, contradicts)
- **Namespace Isolation**: Global memories vs project-scoped memories
- **Context Generation**: Auto-format memories for session context injection
- **Deduplication**: Content-hash based duplicate detection

## Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/recall.git
cd recall

# Install with uv
uv sync

# Ensure Ollama is running with required models
ollama pull mxbai-embed-large  # Required: embeddings for semantic search
ollama pull llama3.2           # Optional: session summarization for auto-capture hook
ollama serve
```

## Usage

### Run as MCP Server

```bash
uv run python -m recall
```

### CLI Options

```bash
uv run python -m recall --help

Options:
  --sqlite-path PATH      SQLite database path (default: ~/.recall/recall.db)
  --chroma-path PATH      ChromaDB storage path (default: ~/.recall/chroma_db)
  --collection NAME       ChromaDB collection name (default: memories)
  --ollama-host HOST      Ollama server URL (default: http://localhost:11434)
  --ollama-model MODEL    Embedding model (default: mxbai-embed-large)
  --ollama-timeout SECS   Request timeout (default: 30)
  --log-level LEVEL       DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
```

## meta-mcp Configuration

Add Recall to your meta-mcp `servers.json`:

```json
{
  "recall": {
    "command": "uv",
    "args": [
      "run",
      "--directory",
      "/path/to/recall",
      "python",
      "-m",
      "recall"
    ],
    "env": {
      "RECALL_LOG_LEVEL": "INFO",
      "RECALL_OLLAMA_HOST": "http://localhost:11434",
      "RECALL_OLLAMA_MODEL": "mxbai-embed-large"
    },
    "description": "Long-term memory system with semantic search",
    "tags": ["memory", "context", "semantic-search"]
  }
}
```

Or for Claude Code / other MCP clients (`claude.json`):

```json
{
  "mcpServers": {
    "recall": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/recall",
        "python",
        "-m",
        "recall"
      ],
      "env": {
        "RECALL_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RECALL_SQLITE_PATH` | `~/.recall/recall.db` | SQLite database file path |
| `RECALL_CHROMA_PATH` | `~/.recall/chroma_db` | ChromaDB persistent storage directory |
| `RECALL_COLLECTION_NAME` | `memories` | ChromaDB collection name |
| `RECALL_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `RECALL_OLLAMA_MODEL` | `mxbai-embed-large` | Embedding model name |
| `RECALL_OLLAMA_TIMEOUT` | `30` | Ollama request timeout in seconds |
| `RECALL_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `RECALL_DEFAULT_NAMESPACE` | `global` | Default namespace for memories |
| `RECALL_DEFAULT_IMPORTANCE` | `0.5` | Default importance score (0.0-1.0) |
| `RECALL_DEFAULT_TOKEN_BUDGET` | `4000` | Default token budget for context |

## MCP Tool Examples

### memory_store_tool

Store a new memory with semantic indexing:

```json
{
  "content": "User prefers dark mode in all applications",
  "memory_type": "preference",
  "namespace": "global",
  "importance": 0.8,
  "metadata": {"source": "explicit_request"}
}
```

Response:
```json
{
  "success": true,
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content_hash": "a1b2c3d4e5f67890"
}
```

### memory_recall_tool

Search memories by semantic similarity:

```json
{
  "query": "user interface preferences",
  "n_results": 5,
  "namespace": "global",
  "memory_type": "preference",
  "min_importance": 0.5,
  "include_related": true
}
```

Response:
```json
{
  "success": true,
  "memories": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "User prefers dark mode in all applications",
      "type": "preference",
      "namespace": "global",
      "importance": 0.8,
      "created_at": "2024-01-15T10:30:00",
      "accessed_at": "2024-01-15T14:22:00",
      "access_count": 3
    }
  ],
  "total": 1,
  "score": 0.92
}
```

### memory_relate_tool

Create a relationship between memories:

```json
{
  "source_id": "mem_new_123",
  "target_id": "mem_old_456",
  "relation": "supersedes",
  "weight": 1.0
}
```

Response:
```json
{
  "success": true,
  "edge_id": 42
}
```

### memory_context_tool

Generate formatted context for session injection:

```json
{
  "query": "coding style preferences",
  "project": "myproject",
  "token_budget": 4000
}
```

Response:
```json
{
  "success": true,
  "context": "# Memory Context\n\n## Preferences\n\n- User prefers dark mode [global]\n- Use 2-space indentation [project:myproject]\n\n## Recent Decisions\n\n- Decided to use FastAPI for the backend [project:myproject]\n",
  "token_estimate": 125
}
```

### memory_forget_tool

Delete memories by ID or semantic search:

```json
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "confirm": true
}
```

Or delete by search:
```json
{
  "query": "outdated preferences",
  "namespace": "project:oldproject",
  "n_results": 10,
  "confirm": true
}
```

Response:
```json
{
  "success": true,
  "deleted_ids": ["550e8400-e29b-41d4-a716-446655440000"],
  "deleted_count": 1
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Server (FastMCP)                     │
│  memory_store │ memory_recall │ memory_relate │ memory_forget │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │     HybridStore       │
                │  (Coordinated Layer)  │
                └───────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│   SQLiteStore │   │  ChromaStore  │   │ OllamaClient  │
│  (Metadata &  │   │   (Vector     │   │  (Embedding   │
│   Relations)  │   │   Search)     │   │  Generation)  │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=recall --cov-report=html

# Type checking
uv run mypy src/recall

# Run specific integration tests
uv run pytest tests/integration/test_mcp_server.py -v
```

## Requirements

- Python 3.13+
- Ollama with:
  - `mxbai-embed-large` model (required for semantic search)
  - `llama3.2` model (optional, for session auto-capture hook)
- ~500MB disk space for ChromaDB indices

## License

MIT
