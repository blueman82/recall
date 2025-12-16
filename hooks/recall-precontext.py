#!/usr/bin/env python3
"""Claude Code / Factory PreToolUse hook for injecting relevant memory context.

This hook runs BEFORE tool calls and injects relevant memories as context
reminders. This helps prevent mistakes by surfacing relevant preferences,
patterns, and past decisions.

Supported tools: Task, Bash, Glob, Grep, Read, Edit, Write, WebFetch, WebSearch, mcp__*

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Task|Bash|Glob|Grep|Read|Edit|Write|WebFetch|WebSearch|mcp__.*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-precontext.py",
                            "timeout": 5
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
    {
        "tool_name": "Bash",
        "tool_input": {"command": "npm install express"},
        "session_id": "abc123",
        "cwd": "/project/root"
    }

Output (to stdout):
    Relevant context/reminders based on the tool being used.
    Example: "Reminder: This project uses pnpm, not npm."

The hook extracts key terms from the tool input and searches recall
for relevant memories, outputting them as context.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Keywords that trigger memory lookups for Bash commands
BASH_TRIGGER_KEYWORDS = {
    # Package managers
    "npm": ["package manager", "npm", "pnpm", "yarn"],
    "pnpm": ["package manager", "pnpm"],
    "yarn": ["package manager", "yarn"],
    "pip": ["pip", "uv", "python package"],
    "uv": ["uv", "pip", "python package"],

    # Git operations
    "git push": ["git push", "dual repo", "remote"],
    "git commit": ["git commit", "commit message"],
    "git checkout": ["git branch", "checkout"],

    # Build/test commands
    "pytest": ["pytest", "test", "testing"],
    "jest": ["jest", "test", "testing"],
    "docker": ["docker", "container"],
    "make": ["makefile", "build"],
}

# File patterns that trigger memory lookups for Write
WRITE_TRIGGER_PATTERNS = {
    ".py": ["python", "coding style"],
    ".ts": ["typescript", "coding style"],
    ".tsx": ["react", "typescript", "component"],
    ".js": ["javascript", "coding style"],
    ".jsx": ["react", "javascript", "component"],
    "Dockerfile": ["docker", "container"],
    "docker-compose": ["docker", "compose"],
    ".env": ["environment", "secrets", "config"],
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


def get_project_namespace() -> str:
    """Derive project namespace from current working directory.

    Returns:
        Namespace string in format 'project:{name}' or 'global'
    """
    cwd = os.getcwd()
    project_name = Path(cwd).name

    project_indicators = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for indicator in project_indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"

    return "global"


def extract_search_terms(tool_name: str, tool_input: dict) -> list[str]:
    """Extract search terms from tool input based on tool type.

    Args:
        tool_name: Name of the tool (Bash, Write, etc.)
        tool_input: Tool input dictionary

    Returns:
        List of search terms to query recall
    """
    terms = []

    if tool_name == "Bash":
        command = tool_input.get("command", "").lower()

        # Check for trigger keywords
        for keyword, search_terms in BASH_TRIGGER_KEYWORDS.items():
            if keyword in command:
                terms.extend(search_terms)

        # Extract first word as potential command
        first_word = command.split()[0] if command.split() else ""
        if first_word and first_word not in terms:
            terms.append(first_word)

    elif tool_name == "Write":
        file_path = tool_input.get("file_path", "")

        # Check for file pattern triggers
        for pattern, search_terms in WRITE_TRIGGER_PATTERNS.items():
            if pattern in file_path.lower():
                terms.extend(search_terms)

        # Add file extension as term
        ext = Path(file_path).suffix
        if ext:
            terms.append(ext.lstrip("."))

    return list(set(terms))  # Deduplicate


def call_recall(tool_name: str, args: dict) -> dict:
    """Call recall MCP tool directly via --call mode.

    Args:
        tool_name: Name of the tool (memory_recall, etc.)
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
            timeout=3,  # Short timeout for pre-tool hooks
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


def format_reminder(memories: list[dict], tool_name: str, tool_input: dict) -> Optional[str]:
    """Format relevant memories as a reminder string.

    Args:
        memories: List of memory dictionaries from recall
        tool_name: Name of the tool being used
        tool_input: Tool input dictionary

    Returns:
        Formatted reminder string or None if no relevant memories
    """
    if not memories:
        return None

    # Filter to high-confidence or high-importance memories
    relevant = [
        m for m in memories
        if m.get("confidence", 0) >= 0.5 or m.get("importance", 0) >= 0.7
        or m.get("type") == "golden_rule"
    ]

    if not relevant:
        return None

    # Build reminder
    lines = ["# Memory Reminder"]

    # Add context about what triggered this
    if tool_name == "Bash":
        command = tool_input.get("command", "")[:50]
        lines.append(f"*Before running: `{command}...`*\n")
    elif tool_name == "Write":
        file_path = Path(tool_input.get("file_path", "")).name
        lines.append(f"*Before writing to: `{file_path}`*\n")

    # Add relevant memories (max 3)
    for mem in relevant[:3]:
        mem_type = mem.get("type", "memory").upper()
        content = mem.get("content", "")

        # Truncate long content
        if len(content) > 200:
            content = content[:200] + "..."

        # Mark golden rules specially
        if mem.get("type") == "golden_rule":
            lines.append(f"**[GOLDEN RULE]** {content}")
        else:
            lines.append(f"- [{mem_type}] {content}")

    return "\n".join(lines)


def main():
    """Main hook entry point.

    Reads tool call data, searches for relevant memories, and outputs
    context reminders. Failures are silent to avoid blocking Claude Code.
    """
    try:
        # Read hook input from stdin
        hook_input = read_hook_input()

        tool_name = hook_input.get("tool_name", "")
        tool_input = hook_input.get("tool_input", {})

        # Only process supported tools
        if tool_name not in ("Bash", "Write"):
            return

        # Extract search terms
        search_terms = extract_search_terms(tool_name, tool_input)
        if not search_terms:
            return

        # Get project namespace
        namespace = get_project_namespace()

        # Search for relevant memories
        query = " ".join(search_terms)
        result = call_recall("memory_recall_tool", {
            "query": query,
            "n_results": 5,
            "namespace": namespace,
            "include_related": True,
            "max_depth": 1,
        })

        if not result.get("success"):
            return

        memories = result.get("memories", [])

        # Also check global namespace if project namespace returned few results
        if len(memories) < 2 and namespace != "global":
            global_result = call_recall("memory_recall_tool", {
                "query": query,
                "n_results": 3,
                "namespace": "global",
            })
            if global_result.get("success"):
                memories.extend(global_result.get("memories", []))

        # Format and output reminder
        reminder = format_reminder(memories, tool_name, tool_input)
        if reminder:
            print(reminder)

    except BrokenPipeError:
        # Claude Code closed connection - this is fine
        pass
    except Exception:
        # Silently fail - don't block Claude Code
        pass


if __name__ == "__main__":
    main()
