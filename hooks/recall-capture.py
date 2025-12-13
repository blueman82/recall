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

Input (via stdin JSON from Claude Code):
    {
        "session_id": "abc123",
        "transcript_path": "~/.claude/projects/.../session.jsonl",
        "cwd": "/path/to/project",
        "hook_event_name": "SessionEnd",
        "reason": "exit"
    }

The hook reads the transcript, summarizes it with Ollama, and stores
relevant memories. Failures are handled gracefully.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def read_hook_input() -> dict:
    """Read hook input from stdin.

    Claude Code passes hook data as JSON via stdin.

    Returns:
        Dictionary with hook input data, or empty dict if unavailable
    """
    try:
        # Check if stdin has data (non-blocking check)
        if sys.stdin.isatty():
            return {}

        stdin_data = sys.stdin.read()
        if stdin_data:
            return json.loads(stdin_data)
    except (json.JSONDecodeError, IOError):
        pass

    return {}


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


def read_transcript(transcript_path: Optional[str]) -> Optional[str]:
    """Read session transcript from provided path.

    Args:
        transcript_path: Path to the transcript file (from hook input)

    Returns:
        Transcript content or None if unavailable
    """
    if not transcript_path:
        return None

    try:
        # Expand ~ to home directory
        path = Path(transcript_path).expanduser()
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
            ["ollama", "run", model],
            input=prompt,
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
        else:
            # Extract first complete JSON array from plain text response
            start = output.find("[")
            if start != -1:
                # Find matching closing bracket by counting
                depth = 0
                for i, char in enumerate(output[start:], start):
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            output = output[start:i + 1].strip()
                            break

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
    from datetime import datetime

    # Read hook input from stdin (Claude Code passes JSON)
    hook_input = read_hook_input()
    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path")
    cwd = hook_input.get("cwd", os.getcwd())

    # Log hook invocation for verification
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-capture.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | SessionEnd fired | session={session_id} | transcript={transcript_path}\n")

    try:
        # Read session transcript
        transcript = read_transcript(transcript_path)

        if not transcript or len(transcript) < 100:
            # No meaningful transcript to process
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | No transcript or too short | len={len(transcript) if transcript else 0}\n")
            return

        # Determine project namespace using cwd from hook input
        os.chdir(cwd)  # Change to session's working directory
        namespace = get_project_namespace()

        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | Processing transcript | len={len(transcript)} | namespace={namespace}\n")

        # Summarize with Ollama
        summary_json = summarize_with_ollama(transcript)

        if not summary_json:
            # Ollama not available or failed - skip silently
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | Ollama summarization failed\n")
            return

        # Parse memories
        try:
            memories = json.loads(summary_json)
            if not isinstance(memories, list):
                return
        except json.JSONDecodeError:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | JSON parse failed | raw={summary_json[:200]}\n")
            return

        if not memories:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | No memories extracted\n")
            return

        # Store memories
        stored = store_memories(memories, namespace)

        # Log result
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | Stored {stored} memories from session {session_id}\n")

    except BrokenPipeError:
        # Claude Code closed connection before we finished - this is fine
        pass
    except Exception as e:
        # Log error but don't block Claude Code
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | ERROR: {e}\n")


if __name__ == "__main__":
    main()
