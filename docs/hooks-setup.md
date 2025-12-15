# Claude Code Hooks Setup

This guide explains how to configure Claude Code hooks for aggressive memory capture, context injection, and automated learning with Recall.

## Overview

Recall provides **6 hooks** for Claude Code:

| Hook | File | Purpose |
|------|------|---------|
| **SessionStart** | `recall-context.py` | Load relevant memories at session start |
| **SessionEnd** | `recall-capture.py` | Summarize and store session memories |
| **PostSessionEnd** | `recall-monitor.py` | Health check and memory analysis |
| **PostToolUse** | `recall-track.py` | Track file Read/Write/Edit operations |
| **PreToolUse** | `recall-precontext.py` | Inject context before Bash/Write |
| **Notification** | `recall-errors.py` | Capture error patterns |

These hooks are **optional** - Recall works as a standard MCP server without them. The hooks enhance the experience by automating memory capture, context injection, and learning.

## Quick Start

### Full Hook Configuration

Add all hooks to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "uv run --directory /path/to/recall python hooks/recall-context.py",
        "timeout": 10000
      }
    ],
    "SessionEnd": [
      {
        "command": "uv run --directory /path/to/recall python hooks/recall-capture.py",
        "timeout": 30000
      }
    ],
    "PostSessionEnd": [
      {
        "command": "uv run --directory /path/to/recall python hooks/recall-monitor.py",
        "timeout": 30000
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Read|Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory /path/to/recall python hooks/recall-track.py",
            "timeout": 5000
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash|Write",
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory /path/to/recall python hooks/recall-precontext.py",
            "timeout": 3000
          }
        ]
      }
    ],
    "Notification": [
      {
        "matcher": {"type": "error"},
        "hooks": [
          {
            "type": "command",
            "command": "uv run --directory /path/to/recall python hooks/recall-errors.py",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

Replace `/path/to/recall` with your actual Recall installation path.

## Prerequisites

- Claude Code installed and configured
- Recall MCP server set up (see main README)
- Python 3.10+ with `uv` package manager
- Ollama installed with models (for SessionEnd and Monitor hooks)

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.2      # For session summarization
ollama pull mxbai-embed-large  # For embeddings
```

---

## Hook Details

### 1. SessionStart - Context Injection

**File:** `recall-context.py`

**Purpose:** Load relevant memories at the start of each Claude Code session.

**How it works:**
1. Detects the current project from working directory
2. Calls `memory_context` to fetch relevant memories
3. Outputs context as markdown for Claude to ingest

**Configuration:**
```json
"SessionStart": [
  {
    "command": "uv run --directory /path/to/recall python hooks/recall-context.py",
    "timeout": 10000
  }
]
```

**Customization:**
```python
# Adjust token budget in recall-context.py
result = call_recall("memory_context", {
    "project": namespace,
    "token_budget": 8000,  # Increase from default 4000
})
```

---

### 2. SessionEnd - Memory Capture

**File:** `recall-capture.py`

**Purpose:** Summarize session transcripts and store important information as memories.

**How it works:**
1. Reads session transcript from `CLAUDE_TRANSCRIPT_PATH`
2. Sends to Ollama `llama3.2` for summarization
3. Extracts structured memories (preferences, decisions, patterns)
4. Stores each memory via `memory_store`

**Configuration:**
```json
"SessionEnd": [
  {
    "command": "uv run --directory /path/to/recall python hooks/recall-capture.py",
    "timeout": 30000
  }
]
```

**Customization:**
```python
# Change summarization model in recall-capture.py
summary_json = summarize_with_ollama(transcript, model="mistral")

# Filter memory types
if memory_type not in ["decision", "preference"]:
    continue
```

---

### 3. PostSessionEnd - Health Monitoring

**File:** `recall-monitor.py`

**Purpose:** Automated health checks on the memory system after each session.

**How it works:**
1. Runs health check across all namespaces
2. Uses Anthropic API (optional) for deep analysis
3. Outputs warnings/suggestions to stderr
4. Stores analysis results as session memories

**Configuration:**
```json
"PostSessionEnd": [
  {
    "command": "uv run --directory /path/to/recall python hooks/recall-monitor.py",
    "timeout": 30000
  }
]
```

**Environment Variables:**
```bash
RECALL_MONITOR_ENABLED=true      # Enable monitoring (default: false)
RECALL_ANTHROPIC_API_KEY=sk-...  # For deep analysis
RECALL_MONITOR_DEEP=true         # Enable deep Opus analysis
```

---

### 4. PostToolUse - File Activity Tracking

**File:** `recall-track.py`

**Purpose:** Track every file Read/Write/Edit/MultiEdit operation.

**How it works:**
1. Captures file path, action type, and project context
2. Stores in recall's `file_activity` table
3. Enables "recently touched files" queries

**Configuration:**
```json
"PostToolUse": [
  {
    "matcher": "Read|Write|Edit|MultiEdit",
    "hooks": [
      {
        "type": "command",
        "command": "uv run --directory /path/to/recall python hooks/recall-track.py",
        "timeout": 5000
      }
    ]
  }
]
```

**Querying File Activity:**
```python
# Get recently touched files
file_activity_recent(
    project_root="/path/to/project",
    hours=24
)
```

---

### 5. PreToolUse - Context Injection Before Actions

**File:** `recall-precontext.py`

**Purpose:** Inject relevant memory reminders BEFORE certain operations.

**How it works:**
1. Extracts keywords from Bash command or Write file path
2. Searches recall for relevant memories
3. Outputs reminders to prevent mistakes

**Configuration:**
```json
"PreToolUse": [
  {
    "matcher": "Bash|Write",
    "hooks": [
      {
        "type": "command",
        "command": "uv run --directory /path/to/recall python hooks/recall-precontext.py",
        "timeout": 3000
      }
    ]
  }
]
```

**Example Output:**
```markdown
# Memory Reminder
*Before running: `npm install express...`*

- [GOLDEN RULE] You MUST use pnpm. You MUST NOT use npm.
- [PREFERENCE] This project uses TypeScript strict mode
```

**Trigger Keywords:**
- Package managers: npm, pnpm, yarn, pip, uv
- Git operations: git push, git commit, git checkout
- Build tools: pytest, jest, docker, make

---

### 6. Notification - Error Pattern Capture

**File:** `recall-errors.py`

**Purpose:** Automatically capture error patterns for future debugging.

**How it works:**
1. Catches error notifications from Claude Code
2. Categorizes errors (syntax, import, permission, etc.)
3. Deduplicates similar errors
4. Stores as pattern memories

**Configuration:**
```json
"Notification": [
  {
    "matcher": {"type": "error"},
    "hooks": [
      {
        "type": "command",
        "command": "uv run --directory /path/to/recall python hooks/recall-errors.py",
        "timeout": 5000
      }
    ]
  }
]
```

**Error Categories Captured:**
- Node.js: module not found, syntax/type/reference errors
- Python: import, syntax, indentation, type, attribute errors
- Git: fatal errors, merge conflicts, SSH auth issues
- Docker: daemon errors, image not found, port conflicts
- Build: linker errors, TypeScript/Rust compiler errors
- Network: connection refused, timeout, DNS failures

---

## Troubleshooting

### Hook Not Running

1. Check file path is correct and absolute
2. Verify Python/uv is in PATH
3. Check file permissions: `chmod +x hooks/*.py`
4. Check hook logs: `~/.claude/hooks/logs/`

### No Context Loaded (SessionStart)

1. Verify Recall MCP server starts without errors
2. Check you have stored memories: `memory_list_tool()`
3. Increase timeout if needed

### Session Not Captured (SessionEnd)

1. Verify Ollama is running: `ollama serve`
2. Check model is installed: `ollama pull llama3.2`
3. Verify `CLAUDE_TRANSCRIPT_PATH` is set

### PreToolUse Not Showing Reminders

1. Check you have relevant memories stored
2. Verify the command/file matches trigger keywords
3. Check logs: `~/.claude/hooks/logs/recall-precontext.log`

### Debug Mode

Test hooks manually:

```bash
# Test context hook
uv run --directory /path/to/recall python hooks/recall-context.py

# Test capture hook (with mock transcript)
CLAUDE_TRANSCRIPT_PATH=/tmp/test.txt uv run --directory /path/to/recall python hooks/recall-capture.py

# Test file tracking
echo '{"tool_name": "Write", "tool_input": {"file_path": "/tmp/test.py"}}' | \
  uv run --directory /path/to/recall python hooks/recall-track.py

# Test error capture
echo '{"type": "error", "message": "ModuleNotFoundError: No module named foo"}' | \
  uv run --directory /path/to/recall python hooks/recall-errors.py
```

---

## Performance

| Hook | Typical Time | Notes |
|------|--------------|-------|
| SessionStart | ~1-2s | Direct tool call |
| SessionEnd | ~5-15s | Depends on transcript length |
| PostSessionEnd | ~5-30s | Deep analysis takes longer |
| PostToolUse | ~100ms | Very fast file tracking |
| PreToolUse | ~200ms | Quick memory lookup |
| Notification | ~100ms | Fast error capture |

All hooks run asynchronously and don't block Claude Code operations.

---

## Security Notes

- Hooks run with your user permissions
- All data stays local (processed by local Ollama)
- Memories are stored in local SQLite database
- No data is sent to external services (unless you enable Anthropic API for monitoring)

---

## Hook Interaction Flow

```
Session Start
     │
     ├── SessionStart: Load context ──────────────────┐
     │                                                │
     ▼                                                │
┌─────────────────────────────────────────────────────┴───┐
│                   Claude Code Session                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ User requests: "Run npm install"                 │   │
│  └──────────────────────────────────────────────────┘   │
│           │                                              │
│           ▼                                              │
│  PreToolUse: "Reminder: Use pnpm, not npm!"             │
│           │                                              │
│           ▼                                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Tool executes: Bash("pnpm install")              │   │
│  └──────────────────────────────────────────────────┘   │
│           │                                              │
│           ▼                                              │
│  PostToolUse: Track file operations                      │
│           │                                              │
│           ▼                                              │
│  [If error] Notification: Capture error pattern          │
│                                                          │
└──────────────────────────────────────────────────────────┘
     │
     ├── SessionEnd: Summarize & store memories
     │
     └── PostSessionEnd: Health check & analysis
```

---

## See Also

- [Graph Expansion](./GRAPH_EXPANSION.md) - Understanding memory relationships
- [Validation Loop](./VALIDATION_LOOP.md) - Building confidence in memories
- [Main README](../README.md) - General Recall documentation
