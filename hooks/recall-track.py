#!/usr/bin/env python3
"""Claude Code / Factory PostToolUse hook for tracking tool operations.

This hook runs after tool calls complete and records activity in recall.
Tracks file operations, searches, web fetches, and subagent tasks.

Supported tools: Task, Bash, Glob, Grep, Read, Write, Edit, MultiEdit, WebFetch, WebSearch, mcp__*

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Task|Bash|Glob|Grep|Read|Write|Edit|MultiEdit|WebFetch|WebSearch|mcp__.*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-track.py",
                            "timeout": 5
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
    {
        "tool_name": "Write",
        "tool_input": {"file_path": "/path/to/file.py", "content": "..."},
        "tool_response": {"success": true, "filePath": "/path/to/file.py"},
        "session_id": "abc123",
        "cwd": "/project/root"
    }

The hook extracts relevant info and stores it in recall.
Failures are handled gracefully to avoid blocking the agent.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# File extension to type mapping
FILE_TYPE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}


def read_hook_input() -> dict:
    """Read hook input from stdin.

    Claude Code passes hook data as JSON via stdin.

    Returns:
        Dictionary with hook input data, or empty dict if unavailable
    """
    try:
        if sys.stdin.isatty():
            return {}

        stdin_data = sys.stdin.read()
        if stdin_data:
            return json.loads(stdin_data)
    except (json.JSONDecodeError, IOError):
        pass

    return {}


def get_file_type(file_path: str) -> Optional[str]:
    """Get file type from extension.

    Args:
        file_path: Path to the file

    Returns:
        File type string or None if unknown
    """
    ext = Path(file_path).suffix.lower()
    return FILE_TYPE_MAP.get(ext)


def extract_file_path(tool_name: str, tool_input: dict) -> Optional[str]:
    """Extract file path from tool input.

    Args:
        tool_name: Name of the tool (Read, Write, Edit, MultiEdit)
        tool_input: Tool input dictionary

    Returns:
        File path or None if not found
    """
    # All these tools use file_path parameter
    return tool_input.get("file_path")


def get_action(tool_name: str) -> str:
    """Convert tool name to action string.

    Args:
        tool_name: Name of the tool

    Returns:
        Action string (read, write, edit, multiedit)
    """
    return tool_name.lower()


def find_project_root(file_path: str) -> Optional[str]:
    """Find project root from file path.

    Walks up directory tree looking for project indicators.

    Args:
        file_path: Path to the file

    Returns:
        Project root path or None
    """
    project_indicators = [
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        ".project",
        "pom.xml",
        "build.gradle",
    ]

    try:
        current = Path(file_path).resolve().parent
        while current != current.parent:
            for indicator in project_indicators:
                if (current / indicator).exists():
                    return str(current)
            current = current.parent
    except Exception:
        pass

    return None


def call_recall(tool_name: str, args: dict) -> dict:
    """Call recall MCP tool directly via --call mode.

    Args:
        tool_name: Name of the tool (file_activity_add, etc.)
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
            timeout=4,
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


def track_file_activity(tool_name: str, tool_input: dict, session_id: str, cwd: str, log_path: Path) -> None:
    """Track file-related tool activity."""
    from datetime import datetime

    file_path = extract_file_path(tool_name, tool_input)
    if not file_path:
        return

    action = get_action(tool_name)
    file_type = get_file_type(file_path)
    project_root = find_project_root(file_path) or cwd

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | {action} | {file_path} | project={project_root}\n")

    result = call_recall("file_activity_add", {
        "file_path": file_path,
        "action": action,
        "session_id": session_id,
        "project_root": project_root,
        "file_type": file_type,
    })

    if not result.get("success"):
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | WARN: {result.get('error', 'unknown error')}\n")


def track_search_activity(tool_name: str, tool_input: dict, session_id: str, cwd: str, log_path: Path) -> None:
    """Track search tool activity (Glob, Grep)."""
    from datetime import datetime

    pattern = tool_input.get("pattern", "") or str(tool_input.get("patterns", []))
    path = tool_input.get("path", cwd)

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | {tool_name.lower()} | pattern={pattern[:50]} | path={path}\n")


def track_web_activity(tool_name: str, tool_input: dict, tool_response: dict, session_id: str, log_path: Path) -> None:
    """Track web tool activity (WebFetch, WebSearch)."""
    from datetime import datetime

    if tool_name == "WebFetch":
        url = tool_input.get("url", "")
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | webfetch | url={url[:100]}\n")
    elif tool_name == "WebSearch":
        query = tool_input.get("query", "")
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | websearch | query={query[:100]}\n")


def track_task_activity(tool_input: dict, tool_response: dict, session_id: str, log_path: Path) -> None:
    """Track Task (subagent) activity."""
    from datetime import datetime

    description = tool_input.get("description", "")
    subagent_type = tool_input.get("subagent_type", "")

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | task | type={subagent_type} | desc={description[:50]}\n")


def track_bash_activity(tool_input: dict, tool_response: dict, session_id: str, log_path: Path) -> None:
    """Track Bash command activity."""
    from datetime import datetime

    command = tool_input.get("command", "")[:100]

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | bash | cmd={command}\n")


def track_mcp_activity(tool_name: str, tool_input: dict, tool_response: dict, session_id: str, log_path: Path) -> None:
    """Track MCP tool activity."""
    from datetime import datetime

    # Parse mcp__server__tool format
    parts = tool_name.split("__")
    server = parts[1] if len(parts) >= 2 else "unknown"
    mcp_tool = parts[2] if len(parts) >= 3 else "unknown"

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | mcp | server={server} | tool={mcp_tool}\n")


def main():
    """Main hook entry point.

    Reads tool call data, extracts relevant info, and stores activity.
    All errors are caught to prevent blocking the agent.
    """
    from datetime import datetime

    # Supported tools
    FILE_TOOLS = {"Read", "Write", "Edit", "MultiEdit"}
    SEARCH_TOOLS = {"Glob", "Grep"}
    WEB_TOOLS = {"WebFetch", "WebSearch"}

    # Read hook input from stdin
    hook_input = read_hook_input()

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response", {})
    session_id = hook_input.get("session_id")
    cwd = hook_input.get("cwd", os.getcwd())

    # Log hook invocation for debugging
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-track.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Route to appropriate tracker based on tool type
        if tool_name in FILE_TOOLS:
            track_file_activity(tool_name, tool_input, session_id, cwd, log_path)

        elif tool_name in SEARCH_TOOLS:
            track_search_activity(tool_name, tool_input, session_id, cwd, log_path)

        elif tool_name in WEB_TOOLS:
            track_web_activity(tool_name, tool_input, tool_response, session_id, log_path)

        elif tool_name == "Task":
            track_task_activity(tool_input, tool_response, session_id, log_path)

        elif tool_name == "Bash":
            track_bash_activity(tool_input, tool_response, session_id, log_path)

        elif tool_name.startswith("mcp__"):
            track_mcp_activity(tool_name, tool_input, tool_response, session_id, log_path)

    except BrokenPipeError:
        # Agent closed connection - this is fine
        pass
    except Exception as e:
        # Log error but don't block the agent
        try:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | ERROR: {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
