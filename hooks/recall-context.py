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
from datetime import datetime, timedelta
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


def get_project_root() -> str | None:
    """Get the project root directory.

    Returns:
        Project root path string or None if not in a project
    """
    cwd = os.getcwd()

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
            return cwd

    return None


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


def format_recent_files(files: list[dict]) -> str:
    """Format recent files as markdown list.

    Args:
        files: List of file activity dicts from get_recent_files()

    Returns:
        Formatted markdown string
    """
    if not files:
        return ""

    lines = ["## Recent Files\n"]

    for file_info in files:
        file_path = file_info.get("file_path", "")
        last_action = file_info.get("last_action", "accessed")
        last_accessed = file_info.get("last_accessed", 0)

        # Convert timestamp to relative time
        try:
            accessed_time = datetime.fromtimestamp(last_accessed)
            now = datetime.now()
            delta = now - accessed_time

            if delta.days == 0:
                time_str = "today"
            elif delta.days == 1:
                time_str = "yesterday"
            else:
                time_str = f"{delta.days} days ago"
        except (ValueError, OSError):
            time_str = "recently"

        lines.append(f"- {file_path} ({last_action} {time_str})")

    return "\n".join(lines)


def call_get_recent_files(project_root: str | None) -> list[dict]:
    """Call the recall internal API to get recent files.

    Args:
        project_root: Project root path or None

    Returns:
        List of file activity dicts
    """
    try:
        # Import recall modules directly for internal API access
        recall_paths = [
            Path(__file__).parent.parent,
            Path.home() / ".local" / "share" / "recall",
            Path("/opt/recall"),
        ]

        recall_dir = None
        for path in recall_paths:
            if (path / "src" / "recall").exists():
                recall_dir = path
                break

        if recall_dir is None:
            return []

        # Add src to sys.path temporarily
        src_path = str(recall_dir / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Import and use HybridStore
        from recall.config import settings
        from recall.storage.hybrid import HybridStore

        # Create store and get recent files
        import asyncio

        async def _get_files():
            async with HybridStore.create(
                sqlite_path=settings.sqlite_path,
                chroma_path=settings.chroma_path,
                collection_name=settings.collection_name,
                ollama_host=settings.ollama_host,
                ollama_model=settings.ollama_model,
            ) as store:
                return store.get_recent_files(
                    project_root=project_root,
                    limit=10,
                    days=14,
                )

        return asyncio.run(_get_files())

    except Exception as e:
        # Silently fail - this is optional context
        print(f"<!-- Error getting recent files: {e} -->", file=sys.stderr)
        return []


def main():
    """Main hook entry point.

    Loads relevant memory context and outputs it as markdown.
    All errors are caught to prevent blocking Claude Code.
    """
    try:
        # Determine project namespace and root
        namespace = get_project_namespace()
        project_root = get_project_root()

        # Call memory_context tool
        result = call_recall("memory_context", {
            "project": namespace.replace("project:", "") if namespace.startswith("project:") else None,
            "token_budget": 4000,
        })

        # Collect output sections
        output_sections = []

        # Add memory context if available
        if result.get("success") and result.get("context"):
            context = result["context"]
            # Only output if there's actual content
            if context and context.strip() and context != "# Memory Context\n\n_No memories found._":
                output_sections.append(context)

        # Add recent files if in a project
        if project_root:
            recent_files = call_get_recent_files(project_root)
            if recent_files:
                files_section = format_recent_files(recent_files)
                if files_section:
                    output_sections.append(files_section)

        # Output all sections
        if output_sections:
            print("\n\n".join(output_sections))
            print()  # blank line
            print("---")
            print("**Memory tip:** When you notice user preferences, technical decisions, or patterns worth remembering, use `memory_store_tool` to save them.")

    except Exception as e:
        # Silently fail - don't block Claude Code
        # Optionally log to stderr for debugging
        print(f"<!-- recall-context hook error: {e} -->", file=sys.stderr)


if __name__ == "__main__":
    main()
