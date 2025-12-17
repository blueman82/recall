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


def extract_query(tool_name: str, tool_input: dict) -> Optional[str]:
    """Extract search query from tool input - pass raw text to semantic search.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input dictionary

    Returns:
        Query string for recall semantic search, or None if nothing useful
    """
    if tool_name == "Bash":
        return tool_input.get("command", "")

    elif tool_name in ("Write", "Edit", "MultiEdit", "Read"):
        file_path = tool_input.get("file_path", "")
        return f"{Path(file_path).name} {Path(file_path).suffix}"

    elif tool_name == "Task":
        prompt = tool_input.get("prompt", "")
        description = tool_input.get("description", "")
        subagent_type = tool_input.get("subagent_type", "")
        return f"{subagent_type} {description} {prompt}"

    elif tool_name == "Glob":
        patterns = tool_input.get("patterns", [])
        if isinstance(patterns, list):
            return " ".join(patterns)
        return tool_input.get("pattern", "")

    elif tool_name == "Grep":
        return tool_input.get("pattern", "")

    elif tool_name == "WebFetch":
        return tool_input.get("url", "")

    elif tool_name == "WebSearch":
        return tool_input.get("query", "")

    elif tool_name.startswith("mcp__"):
        # MCP tools - use server and tool name
        parts = tool_name.split("__")
        return " ".join(parts[1:])

    return None


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
            timeout=5,  # Graph expansion needs more time
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
    context reminders. Failures are silent to avoid blocking the agent.
    """
    # Supported tools for memory context injection
    SUPPORTED_TOOLS = {
        "Task", "Bash", "Glob", "Grep", "Read", "Edit", "MultiEdit",
        "Write", "WebFetch", "WebSearch",
    }

    try:
        # Read hook input from stdin
        hook_input = read_hook_input()

        tool_name = hook_input.get("tool_name") or hook_input.get("toolName", "")
        tool_input = hook_input.get("tool_input") or hook_input.get("toolInput", {})
        cwd = hook_input.get("cwd")

        # Change to session's working directory if provided
        if cwd:
            os.chdir(cwd)

        # Check if tool is supported (including MCP tools)
        is_mcp = tool_name.startswith("mcp__")
        if tool_name not in SUPPORTED_TOOLS and not is_mcp:
            return

        # Extract query from tool input
        query = extract_query(tool_name, tool_input)
        if not query or not query.strip():
            return

        # Get project namespace
        namespace = get_project_namespace()
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

        # Include graph-expanded memories
        for expanded in result.get("expanded", []):
            memories.append(expanded)

        # Also check global namespace if project namespace returned few results
        if len(memories) < 2 and namespace != "global":
            global_result = call_recall("memory_recall_tool", {
                "query": query,
                "n_results": 3,
                "namespace": "global",
                "include_related": True,
                "max_depth": 1,
            })
            if global_result.get("success"):
                memories.extend(global_result.get("memories", []))
                for expanded in global_result.get("expanded", []):
                    memories.append(expanded)

        # Format and output reminder
        reminder = format_reminder(memories, tool_name, tool_input)
        if reminder:
            print(reminder)

    except BrokenPipeError:
        # Agent closed connection - this is fine
        pass
    except Exception:
        # Silently fail - don't block the agent
        pass


if __name__ == "__main__":
    main()
