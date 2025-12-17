#!/usr/bin/env python3
"""Claude Code / Factory SessionEnd hook for capturing session memories.

This hook runs at the end of each session and summarizes the conversation
to store important decisions, preferences, and patterns.
It uses Ollama for local summarization and recall MCP for storage.

SessionEnd reasons: clear, logout, prompt_input_exit, other

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-capture.py",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
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
import tempfile
from pathlib import Path
from typing import Optional


def run_background(data_file: str) -> None:
    """Background worker - does actual capture work."""
    try:
        with open(data_file) as f:
            hook_input = json.load(f)
        os.unlink(data_file)
        _do_capture(hook_input)
    except Exception:
        pass


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
    # Truncate very long transcripts to last 16000 chars (increased for better context)
    if len(transcript) > 16000:
        transcript = "...(truncated)...\n" + transcript[-16000:]

    prompt = f"""Analyze this Claude Code session transcript and extract key memories.

IMPORTANT: Use RFC 2119 requirement language for all memories:
- "You MUST..." for absolute requirements
- "You MUST NOT..." for absolute prohibitions
- "You SHOULD..." for strong recommendations
- "Use X for Y" for decisions (imperative, not descriptive)

DO NOT use soft language like "User prefers", "User wants", "User likes".
Transform preferences into commands: "User prefers X" → "You MUST use X."

For each memory, identify:
1. Type: preference, decision, pattern, or session
2. Content: An imperative statement using RFC 2119 keywords

Format output as JSON array:
[
  {{"type": "decision", "content": "Use FastAPI for the backend. Do not use Flask or Django."}},
  {{"type": "preference", "content": "You MUST use TypeScript for all code. You MUST NOT create JavaScript files."}},
  {{"type": "pattern", "content": "You MUST write tests BEFORE implementation (TDD)."}}
]

Only include significant, reusable memories. Skip:
- Trivial or one-off items
- Information already captured in previous sessions
- Implementation details (focus on decisions and preferences)

If no significant NEW memories, return empty array: []

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
            # User's development location
            Path.home() / "Documents" / "Github" / "recall",
            # Relative to this hook file (if hook is in recall repo)
            Path(__file__).parent.parent,
            # Common install locations
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


def check_duplicate(content: str, namespace: str, threshold: float = 0.85) -> bool:
    """Check if a similar memory already exists.

    Args:
        content: Memory content to check
        namespace: Namespace to search in
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        True if a similar memory exists, False otherwise
    """
    try:
        result = call_recall("memory_recall", {
            "query": content,
            "n_results": 1,
            "namespace": namespace,
        })

        if not result.get("success") or not result.get("results"):
            return False

        # Check if top result is similar enough to be a duplicate
        top_result = result["results"][0]
        similarity = top_result.get("score", 0)

        # Also check for exact content match (different similarity scoring)
        existing_content = top_result.get("content", "")
        if existing_content.strip().lower() == content.strip().lower():
            return True

        return similarity >= threshold

    except Exception:
        # On error, allow storage (fail open)
        return False


def store_memories(memories: list, namespace: str) -> tuple[int, int]:
    """Store extracted memories in recall with deduplication.

    Args:
        memories: List of memory dicts with type and content
        namespace: Namespace for storage

    Returns:
        Tuple of (stored count, skipped duplicate count)
    """
    stored = 0
    skipped = 0

    for memory in memories:
        memory_type = memory.get("type", "session")
        content = memory.get("content", "")

        if not content or len(content) < 10:
            continue

        # Check for duplicates before storing
        if check_duplicate(content, namespace):
            skipped += 1
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
        # RFC 2119 MUST statements get higher importance
        importance_map = {
            "preference": 0.8,  # Increased for RFC 2119 compliance
            "decision": 0.8,
            "pattern": 0.7,    # Increased for RFC 2119 compliance
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

    return stored, skipped


def _do_capture(hook_input: dict) -> None:
    """Actual capture work - runs in background."""
    from datetime import datetime

    session_id = hook_input.get("session_id") or hook_input.get("sessionId", "unknown")
    transcript_path = hook_input.get("transcript_path") or hook_input.get("transcriptPath")
    cwd = hook_input.get("cwd", os.getcwd())

    log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-capture.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | SessionEnd fired | session={session_id} | transcript={transcript_path}\n")

        transcript = read_transcript(transcript_path)
        if not transcript or len(transcript) < 100:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | No transcript or too short | len={len(transcript) if transcript else 0}\n")
            return

        os.chdir(cwd)
        namespace = get_project_namespace()

        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | Processing transcript | len={len(transcript)} | namespace={namespace}\n")

        summary_json = summarize_with_ollama(transcript)
        if not summary_json:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | Ollama summarization failed\n")
            return

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

        stored, skipped = store_memories(memories, namespace)

        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | Stored {stored} memories, skipped {skipped} duplicates from session {session_id}\n")

    except Exception as e:
        try:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | ERROR: {e}\n")
        except Exception:
            pass


def main():
    """Main hook entry point - fire and forget."""
    # Handle background mode
    if len(sys.argv) > 2 and sys.argv[1] == "--background":
        run_background(sys.argv[2])
        return

    hook_input = read_hook_input()
    if not hook_input:
        return

    try:
        fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="recall-capture-")
        with os.fdopen(fd, "w") as f:
            json.dump(hook_input, f)

        subprocess.Popen(
            [sys.executable, __file__, "--background", temp_path],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        _do_capture(hook_input)


if __name__ == "__main__":
    main()
