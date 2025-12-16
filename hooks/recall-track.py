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


def main():
    """Main hook entry point.

    Reads tool call data, extracts file info, and stores activity.
    All errors are caught to prevent blocking Claude Code.
    """
    from datetime import datetime

    # Read hook input from stdin
    hook_input = read_hook_input()

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    session_id = hook_input.get("session_id")
    cwd = hook_input.get("cwd", os.getcwd())

    # Log hook invocation for debugging
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-track.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Only process file-related tools
        if tool_name not in ("Read", "Write", "Edit", "MultiEdit"):
            return

        # Extract file path from tool input
        file_path = extract_file_path(tool_name, tool_input)
        if not file_path:
            return

        # Determine action and file type
        action = get_action(tool_name)
        file_type = get_file_type(file_path)

        # Find project root
        project_root = find_project_root(file_path) or cwd

        # Log the activity
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {action} | {file_path} | project={project_root}\n")

        # Store file activity via recall
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

    except BrokenPipeError:
        # Claude Code closed connection - this is fine
        pass
    except Exception as e:
        # Log error but don't block Claude Code
        try:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | ERROR: {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
