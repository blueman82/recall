# Claude Code Hooks Setup

This guide explains how to configure Claude Code hooks for automatic memory capture and context injection with Recall.

## Overview

Recall provides two hooks for Claude Code:

1. **SessionStart** (`recall-context.py`): Loads relevant memories at the start of each session
2. **SessionEnd** (`recall-capture.py`): Summarizes and stores important session information

These hooks are **optional** - Recall works as a standard MCP server without them. The hooks enhance the experience by automating memory capture and context injection.

## Prerequisites

- Claude Code installed and configured
- Recall MCP server set up (see main README)
- Ollama installed with `llama3.2` model (for session summarization)
- Python 3.10+ with `uv` package manager

### Install Ollama (for SessionEnd hook)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the summarization model
ollama pull llama3.2
```

## Configuration

### 1. Locate Your Recall Installation

The hooks need to know where Recall is installed. By default, they look in:

```
~/.local/share/recall/
/opt/recall/
<hook-parent-directory>/  (relative to hook file)
```

### 2. Configure Claude Code Settings

Edit `~/.claude/settings.json` to add the hooks:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "command": "python /path/to/recall/hooks/recall-context.py",
        "timeout": 10000
      }
    ],
    "SessionEnd": [
      {
        "command": "python /path/to/recall/hooks/recall-capture.py",
        "timeout": 30000
      }
    ]
  }
}
```

Replace `/path/to/recall` with your actual Recall installation path.

### 3. Alternative: Using uv run

If you installed Recall as a project with `uv`, you can use:

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
    ]
  }
}
```

## How It Works

### SessionStart Hook

1. Detects the current project from working directory
2. Calls `memory_context` via Recall's `--call` mode
3. Outputs relevant memories as markdown context
4. Claude Code injects this context at session start

The hook is fast (~1-2 seconds) because it uses direct tool invocation instead of MCP protocol.

### SessionEnd Hook

1. Reads the session transcript from `CLAUDE_TRANSCRIPT_PATH`
2. Sends to Ollama `llama3.2` for summarization
3. Extracts structured memories (preferences, decisions, patterns)
4. Stores each memory via Recall's `memory_store` tool

This enables zero-friction memory capture - you don't need to explicitly save anything.

## Troubleshooting

### Hook not running

Check that:
- The hook file path is correct and absolute
- Python is in your PATH
- The file has execute permissions (`chmod +x hooks/*.py`)

### No context loaded

Check that:
- Recall MCP server can start without errors
- You have stored memories (try `memory_recall` manually)
- The hook isn't timing out (increase `timeout` if needed)

### Session not being captured

Check that:
- Ollama is running (`ollama serve`)
- The `llama3.2` model is installed (`ollama pull llama3.2`)
- `CLAUDE_TRANSCRIPT_PATH` is set (this is automatic in Claude Code)

### Debug mode

Add stderr output to see what's happening:

```bash
# Test context hook
python /path/to/recall/hooks/recall-context.py 2>&1

# Test capture hook (with mock transcript)
CLAUDE_TRANSCRIPT_PATH=/tmp/test.txt python /path/to/recall/hooks/recall-capture.py 2>&1
```

## Customization

### Change summarization model

Edit `recall-capture.py` and change the model name:

```python
summary_json = summarize_with_ollama(transcript, model="mistral")
```

### Adjust token budget

Edit `recall-context.py` to change context size:

```python
result = call_recall("memory_context", {
    "project": namespace,
    "token_budget": 8000,  # Increase from default 4000
})
```

### Filter memory types

Modify `recall-capture.py` to only store certain types:

```python
# Only store decisions and preferences
if memory_type not in ["decision", "preference"]:
    continue
```

## Security Notes

- Hooks run with your user permissions
- Transcript data stays local (processed by local Ollama)
- Memories are stored in local SQLite database
- No data is sent to external services

## Performance

- SessionStart hook: ~1-2 seconds (direct tool call)
- SessionEnd hook: ~5-15 seconds (depends on transcript length and Ollama speed)

Hooks run asynchronously and don't block Claude Code startup or shutdown.
