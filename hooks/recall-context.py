#!/usr/bin/env python3
"""Claude Code SessionStart hook for loading relevant memory context.

This hook runs at the start of each Claude Code session and injects
relevant memories as system context. It uses the recall MCP --call mode
for direct, fast tool invocation without MCP protocol overhead.

Usage:
    Configure in ~/.claude/settings.json:
    {
        "hooks": {
            "SessionStart": [
                {
                    "command": "python /path/to/recall/hooks/recall-context.py",
                    "timeout": 10000
                }
            ]
        }
    }

The hook outputs markdown context that Claude Code will see at session start.
Failures are handled gracefully - they don't block Claude Code.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def get_project_namespace() -> str:
    """Derive project namespace from current working directory.

    Returns:
        Namespace string in format 'project:{name}' or 'global'
    """
    cwd = os.getcwd()
    project_name = Path(cwd).name

    # Check for common project indicators
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


def call_recall(tool_name: str, args: dict) -> dict:
    """Call recall MCP tool directly via --call mode.

    Args:
        tool_name: Name of the tool (memory_context, etc.)
        args: Dictionary of tool arguments

    Returns:
        Tool result as dictionary, or error dict on failure
    """
    try:
        # Find the recall module - try multiple locations
        recall_paths = [
            # Relative to this hook file
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
            # Try using uv run directly (installed globally)
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
            return {
                "success": False,
                "error": f"recall failed: {result.stderr}",
            }

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

    Loads relevant memory context and outputs it as markdown.
    All errors are caught to prevent blocking Claude Code.
    """
    try:
        # Determine project namespace
        namespace = get_project_namespace()

        # Call memory_context tool
        result = call_recall("memory_context", {
            "project": namespace.replace("project:", "") if namespace.startswith("project:") else None,
            "token_budget": 4000,
        })

        if result.get("success") and result.get("context"):
            context = result["context"]
            # Only output if there's actual content
            if context and context.strip() and context != "# Memory Context\n\n_No memories found._":
                print(context)
                print()  # blank line
                print("---")
                print("**Memory tip:** When you notice user preferences, technical decisions, or patterns worth remembering, use `memory_store_tool` to save them.")

    except Exception as e:
        # Silently fail - don't block Claude Code
        # Optionally log to stderr for debugging
        print(f"<!-- recall-context hook error: {e} -->", file=sys.stderr)


if __name__ == "__main__":
    main()
