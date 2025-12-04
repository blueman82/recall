#!/usr/bin/env python3
"""Claude Code SessionEnd hook for capturing session memories.

This hook runs at the end of each Claude Code session and summarizes
the conversation to store important decisions, preferences, and patterns.
It uses Ollama for local summarization and recall MCP for storage.

Usage:
    Configure in ~/.claude/settings.json:
    {
        "hooks": {
            "SessionEnd": [
                {
                    "command": "python /path/to/recall/hooks/recall-capture.py",
                    "timeout": 30000
                }
            ]
        }
    }

Environment:
    CLAUDE_TRANSCRIPT_PATH: Path to session transcript (set by Claude Code)
    CLAUDE_SESSION_ID: Unique session identifier (set by Claude Code)

The hook reads the transcript, summarizes it with Ollama, and stores
relevant memories. Failures are handled gracefully.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_project_namespace() -> str:
    """Derive project namespace from current working directory.

    Returns:
        Namespace string in format 'project:{name}' or 'global'
    """
    cwd = os.getcwd()
    project_name = Path(cwd).name

    project_indicators = [
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
    ]

    for indicator in project_indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"

    return "global"


def read_transcript() -> Optional[str]:
    """Read session transcript from Claude Code environment.

    Returns:
        Transcript content or None if unavailable
    """
    transcript_path = os.environ.get("CLAUDE_TRANSCRIPT_PATH")

    if not transcript_path:
        return None

    try:
        path = Path(transcript_path)
        if path.exists():
            return path.read_text()
    except Exception:
        pass

    return None


def summarize_with_ollama(transcript: str, model: str = "llama3.2") -> Optional[str]:
    """Summarize transcript using local Ollama model.

    Args:
        transcript: Session transcript text
        model: Ollama model name (default: llama3.2)

    Returns:
        Summarized memories or None on failure
    """
    # Truncate very long transcripts to last 8000 chars
    if len(transcript) > 8000:
        transcript = "...(truncated)...\n" + transcript[-8000:]

    prompt = f"""Analyze this Claude Code session transcript and extract key memories.

For each memory, identify:
1. Type: preference (user preferences), decision (technical decisions), pattern (recurring behaviors), or session (session-specific)
2. Content: A clear, concise statement

Format output as JSON array:
[
  {{"type": "decision", "content": "Decided to use FastAPI for the backend"}},
  {{"type": "preference", "content": "User prefers TypeScript over JavaScript"}}
]

Only include significant, reusable memories. Skip trivial or one-off items.
If no significant memories, return empty array: []

Transcript:
{transcript}

JSON output:"""

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=25,
        )

        if result.returncode != 0:
            return None

        # Extract JSON from response
        output = result.stdout.strip()

        # Handle markdown code blocks
        if "```json" in output:
            start = output.find("```json") + 7
            end = output.find("```", start)
            output = output[start:end].strip()
        elif "```" in output:
            start = output.find("```") + 3
            end = output.find("```", start)
            output = output[start:end].strip()

        return output

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        # Ollama not installed
        return None
    except Exception:
        return None


def call_recall(tool_name: str, args: dict) -> dict:
    """Call recall MCP tool directly via --call mode.

    Args:
        tool_name: Name of the tool (memory_store, etc.)
        args: Dictionary of tool arguments

    Returns:
        Tool result as dictionary, or error dict on failure
    """
    try:
        recall_paths = [
            Path(__file__).parent.parent,
            Path.home() / ".local" / "share" / "recall",
            Path("/opt/recall"),
        ]

        recall_dir = None
        for path in recall_paths:
            if (path / "src" / "recall" / "__main__.py").exists():
                recall_dir = path
                break

        if recall_dir is None:
            cmd = [
                "uv", "run", "python", "-m", "recall",
                "--call", tool_name,
                "--args", json.dumps(args),
            ]
        else:
            cmd = [
                "uv", "run",
                "--directory", str(recall_dir),
                "python", "-m", "recall",
                "--call", tool_name,
                "--args", json.dumps(args),
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=recall_dir or Path.cwd(),
        )

        if result.returncode != 0:
            return {"success": False, "error": f"recall failed: {result.stderr}"}

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "recall timed out"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON response: {e}"}
    except FileNotFoundError:
        return {"success": False, "error": "uv or python not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def store_memories(memories: list, namespace: str) -> int:
    """Store extracted memories in recall.

    Args:
        memories: List of memory dicts with type and content
        namespace: Namespace for storage

    Returns:
        Number of successfully stored memories
    """
    stored = 0

    for memory in memories:
        memory_type = memory.get("type", "session")
        content = memory.get("content", "")

        if not content or len(content) < 10:
            continue

        # Map simplified types to MemoryType enum values
        type_map = {
            "preference": "preference",
            "decision": "decision",
            "pattern": "pattern",
            "session": "session",
        }
        mem_type = type_map.get(memory_type, "session")

        # Determine importance based on type
        importance_map = {
            "preference": 0.7,
            "decision": 0.8,
            "pattern": 0.6,
            "session": 0.4,
        }
        importance = importance_map.get(mem_type, 0.5)

        result = call_recall("memory_store", {
            "content": content,
            "memory_type": mem_type,
            "namespace": namespace,
            "importance": importance,
        })

        if result.get("success"):
            stored += 1

    return stored


def main():
    """Main hook entry point.

    Reads transcript, summarizes with Ollama, and stores memories.
    All errors are caught to prevent blocking Claude Code.
    """
    try:
        # Read session transcript
        transcript = read_transcript()

        if not transcript or len(transcript) < 100:
            # No meaningful transcript to process
            return

        # Determine project namespace
        namespace = get_project_namespace()

        # Summarize with Ollama
        summary_json = summarize_with_ollama(transcript)

        if not summary_json:
            # Ollama not available or failed - skip silently
            return

        # Parse memories
        try:
            memories = json.loads(summary_json)
            if not isinstance(memories, list):
                return
        except json.JSONDecodeError:
            return

        if not memories:
            return

        # Store memories
        stored = store_memories(memories, namespace)

        # Optional: log for debugging
        if stored > 0:
            session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")
            print(
                f"<!-- recall-capture: stored {stored} memories from session {session_id} -->",
                file=sys.stderr,
            )

    except Exception as e:
        # Silently fail - don't block Claude Code
        print(f"<!-- recall-capture hook error: {e} -->", file=sys.stderr)


if __name__ == "__main__":
    main()
