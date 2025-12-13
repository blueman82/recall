# Recall - Long-term Memory System for MCP Clients

## Project Overview

Recall is an MCP (Model Context Protocol) server that provides persistent memory storage and retrieval with semantic search capabilities. It enables MCP-compatible clients (Claude Code, Claude Desktop, Cursor, etc.) to remember user preferences, decisions, patterns, and session context across conversations.

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
├── validation.py        # Contradiction detection, auto-supersede
├── memory/
│   ├── types.py         # Memory, Edge, MemoryType, RelationType, confidence
│   └── operations.py    # store, recall, relate, context, forget, validate, apply, outcome
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
- `file_context` - File activity tracking (what files were touched)
- `golden_rule` - High-confidence memories (constitutional principles)

**RelationType** (enum):
- `relates_to` - General relationship
- `supersedes` - One memory replaces another
- `caused_by` - Causal relationship
- `contradicts` - Conflicting information

**Namespace** format:
- `global` - Cross-project memories
- `project:{name}` - Project-scoped memories

**Confidence Score**:
- Range: 0.0 to 1.0 (default: 0.3)
- Validated through usage via the validation loop
- Success increases confidence, failure decreases it
- Memories at confidence >= 0.9 are golden rules

### Golden Rules

Golden rules are high-confidence memories that represent validated, constitutional principles:

- **Auto-promotion**: Memories with confidence >= 0.9 are automatically promoted to `golden_rule` type
- **Eligible types**: Only `preference`, `decision`, and `pattern` can be promoted
- **Protected**: Golden rules cannot be deleted unless `force=True` is specified
- **Always visible**: Appear in context regardless of token budget or recency
- **Original type preserved**: Stored in `metadata.promoted_from`

## RFC 2119 Requirement Language

Memories in Recall follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) semantics for requirement keywords. When writing memories, use these keywords with their precise meanings:

| Keyword | Meaning | Use For |
|---------|---------|---------|
| **MUST** / **REQUIRED** / **SHALL** | Absolute requirement | Hard rules that cannot be ignored |
| **MUST NOT** / **SHALL NOT** | Absolute prohibition | Actions that are never allowed |
| **SHOULD** / **RECOMMENDED** | Strong suggestion, exceptions require justification | Preferences with valid escape hatches |
| **SHOULD NOT** / **NOT RECOMMENDED** | Strong discouragement | Generally avoid unless justified |
| **MAY** / **OPTIONAL** | Truly optional | Nice-to-haves |

### Writing Effective Memories

**Weak (avoid):**
```
User prefers pytest over unittest
```

**Strong (preferred):**
```
You MUST use pytest for all tests. You MUST NOT use unittest.
```

### Why This Matters

Soft language ("prefers", "wants", "likes") gets rationalized and ignored under time pressure. RFC 2119 keywords have defined semantics that are harder to bypass.

The `memory_context_tool` automatically injects an RFC 2119 preamble so the consuming LLM knows these keywords have precise meanings.

## MCP Tools

The server exposes 16 tools via FastMCP:

### memory_store_tool
Store a new memory with semantic indexing and deduplication.

### memory_recall_tool
Recall memories using semantic search with optional graph expansion.

### memory_relate_tool
Create a relationship between two memories.

### memory_context_tool
Fetch and format relevant memories for context injection.

### memory_forget_tool
Delete memories by ID or semantic search. Golden rules are protected from deletion.

### memory_validate_tool
Validate a memory and adjust its confidence score based on success/failure.

### memory_apply_tool
Record that a memory is being applied (TRY phase of validation loop).

### memory_outcome_tool
Record the outcome of applying a memory (LEARN phase of validation loop).

### memory_count_tool
Count memories with optional namespace and type filters.

### memory_list_tool
List memories with filtering and pagination support.

### validation_history_tool
Get validation event history for a memory to understand confidence changes.

### file_activity_add
Record file activity events (used by PostToolUse hooks).

### file_activity_recent
Get recently accessed files with aggregated activity.

## Validation Loop (ELF-Inspired)

Recall implements a validation loop to build confidence in memories through practical use:

```
TRY → BREAK → ANALYZE → LEARN
 ↑                        ↓
 └────────────────────────┘
```

### Workflow

1. **TRY** - Apply a memory using `memory_apply_tool`
   - Records an "applied" validation event
   - Updates access timestamp

2. **BREAK** - Memory application either succeeds or fails
   - Success: Memory was useful and correct
   - Failure: Memory led to errors or was rejected

3. **ANALYZE** - Evaluate what happened
   - Collect error messages and context
   - Determine if memory was helpful

4. **LEARN** - Update confidence using `memory_outcome_tool`
   - Success: `confidence += 0.1` (max 1.0)
   - Failure: `confidence -= 0.15` (min 0.0)
   - Auto-promote to golden rule at confidence >= 0.9

### Example Usage

```python
# 1. TRY: Apply a memory
result = await memory_apply_tool(
    memory_id="mem_123",
    context="Using dark mode preference for UI settings",
    session_id="session_456"
)

# 2. BREAK: Attempt to use the memory
# ... application code ...

# 3. ANALYZE: Check if it worked
success = (error_count == 0)

# 4. LEARN: Record the outcome
outcome = await memory_outcome_tool(
    memory_id="mem_123",
    success=success,
    error_msg="User rejected setting" if not success else None,
    session_id="session_456"
)

# Check if promoted to golden rule
if outcome["promoted"]:
    print(f"Memory promoted to golden rule! Confidence: {outcome['new_confidence']}")
```

### Contradiction Detection

The validation system also detects contradictions between memories:

- **Semantic similarity**: ChromaDB finds similar memories (threshold: 0.7)
- **LLM reasoning**: Ollama determines if they actually contradict
- **Edge creation**: `CONTRADICTS` edges link conflicting memories
- **Auto-supersede**: Better-performing memories replace worse ones

### File Activity Tracking

Files accessed during tool operations are automatically tracked:

- **Action types**: `read`, `write`, `edit`, `multiedit`
- **Project context**: Grouped by project root directory
- **File type detection**: Automatically inferred from extension
- **Recent files**: Query by project and time window

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
| RECALL_DEFAULT_CONFIDENCE | 0.3 | Default confidence score |
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
All memory operations (store, recall, forget, context, validate, apply, outcome) are async. Use `await` when calling.

### Error Handling
Operations return result objects with `success` boolean and optional `error` message.

### Deduplication
Content is hashed (SHA-256, truncated to 16 chars) for deduplication within namespaces.

### Graph Expansion
Set `include_related=True` in recall to follow relationship edges.

### Confidence Building
- Memories start at confidence 0.3 by default
- Use the validation loop to build confidence through practical application
- Golden rules (confidence >= 0.9) gain special protection and visibility
- Low-confidence memories (< 0.15) are candidates for deletion

## Important Notes

- **STDIO Transport**: MCP uses stdio - all logging goes to stderr, never stdout
- **Ollama Dependencies**:
  - `mxbai-embed-large` model for embeddings (required)
  - `llama3.2` model for contradiction detection and auto-supersede (optional)
- **Signal Handling**: SIGINT/SIGTERM trigger graceful shutdown
- **Python 3.13+**: Requires Python 3.13 or later
- **Golden Rule Protection**: Golden rules cannot be deleted without `force=True` flag
