#!/usr/bin/env python3
"""Claude Code / Factory SessionStart hook for loading relevant memory context.

This hook runs at the start of each session and injects relevant memories
as system context. It uses Ollama (llama3.2) for intelligent curation
and synthesis of memories.

SessionStart matchers: startup, resume, clear, compact

Architecture:
    Phase 1: Fetch raw memories via recall --call mode
    Phase 2: Curate with Ollama for synthesis and prioritization

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-context.py",
                            "timeout": 10
                        }
                    ]
                }
            ]
        }
    }

The hook outputs markdown context that the agent will see at session start.
Failures are handled gracefully - they don't block the agent.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_project_namespace() -> tuple[str, str]:
    """Derive project namespace and name from current working directory.

    Returns:
        Tuple of (namespace string, project name)
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
            return f"project:{project_name}", project_name

    return "global", project_name


def call_recall(tool_name: str, args: dict) -> dict:
    """Call recall MCP tool directly via --call mode.

    Args:
        tool_name: Name of the tool (memory_list_tool, etc.)
        args: Dictionary of tool arguments

    Returns:
        Tool result as dictionary, or error dict on failure
    """
    try:
        # Find the recall module - try multiple locations
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
            timeout=5,  # Reduced timeout to leave room for Ollama
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


def fetch_raw_memories(project_namespace: str) -> list[dict]:
    """Fetch raw memories from both project and global namespaces.

    Args:
        project_namespace: The project namespace (e.g., 'project:recall')

    Returns:
        List of memory dicts with type, content, importance, confidence
    """
    all_memories = []

    # Fetch project memories
    project_result = call_recall("memory_list", {
        "namespace": project_namespace,
        "limit": 50,
        "order_by": "importance",
        "descending": True,
    })
    if project_result.get("success") and project_result.get("memories"):
        for mem in project_result["memories"]:
            mem["_source"] = "project"
        all_memories.extend(project_result["memories"])

    # Fetch global memories (preferences and golden rules)
    global_result = call_recall("memory_list", {
        "namespace": "global",
        "limit": 30,
        "order_by": "importance",
        "descending": True,
    })
    if global_result.get("success") and global_result.get("memories"):
        # Only include universal global memories (preferences and golden rules)
        # Skip patterns/decisions - they're usually context-specific incidents
        for mem in global_result["memories"]:
            mem_type = mem.get("type", "")
            # Only preferences and golden rules apply across all projects
            if mem_type in ("preference", "golden_rule"):
                mem["_source"] = "global"
                all_memories.append(mem)

    return all_memories


def curate_with_ollama(
    memories: list[dict],
    project_name: str,
    model: str = "llama3.2",
) -> Optional[str]:
    """Use Ollama to intelligently curate and synthesize memories.

    Args:
        memories: List of raw memory dicts
        project_name: Name of the current project
        model: Ollama model to use (default: llama3.2)

    Returns:
        Curated markdown context, or None on failure
    """
    if not memories:
        return None

    # Format memories for Ollama
    memory_lines = []
    for mem in memories:
        source = mem.get("_source", "unknown")
        mem_type = mem.get("type", "unknown")
        content = mem.get("content", "")
        importance = mem.get("importance", 0.5)
        confidence = mem.get("confidence", 0.3)

        # Format: [source|type|imp:X.X|conf:X.X] content
        memory_lines.append(
            f"[{source}|{mem_type}|imp:{importance:.1f}|conf:{confidence:.1f}] {content}"
        )

    memory_text = "\n".join(memory_lines)

    prompt = f"""You are curating memories for a Claude Code session.
Project: {project_name}

Raw memories (format: [source|type|importance|confidence] content):
{memory_text}

INSTRUCTIONS:
1. Synthesize duplicate/similar memories into single statements
2. Use RFC 2119 language (MUST, MUST NOT, SHOULD, SHOULD NOT)
3. Prioritize by: golden_rule > high-confidence > project-specific > global
4. Remove redundant or contradictory information (keep higher confidence)
5. Output ONLY the curated markdown, no explanations

OUTPUT FORMAT:
# Memory Context

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in these memories are to be interpreted as described in RFC 2119.

---

## Golden Rules
- [highest priority rules, if any]

## Preferences
- [user preferences]

## Patterns
- [coding patterns]

## Recent Decisions
- [decisions, if any]

OUTPUT:"""

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=6,  # Leave headroom within 10s hook timeout
        )

        if result.returncode != 0:
            return None

        output = result.stdout.strip()

        # Validate output looks like markdown
        if not output or "Memory Context" not in output:
            return None

        return output

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        # Ollama not installed
        return None
    except Exception:
        return None


def fallback_context(memories: list[dict]) -> str:
    """Generate simple context when Ollama is unavailable.

    Args:
        memories: List of memory dicts

    Returns:
        Basic markdown context
    """
    if not memories:
        return ""

    # RFC 2119 preamble
    lines = [
        "# Memory Context",
        "",
        'The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", '
        '"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in these '
        "memories are to be interpreted as described in RFC 2119.",
        "",
        "---",
        "",
    ]

    # Group by type
    by_type: dict[str, list[str]] = {
        "golden_rule": [],
        "preference": [],
        "pattern": [],
        "decision": [],
        "other": [],
    }

    for mem in memories:
        mem_type = mem.get("type", "other")
        content = mem.get("content", "")
        source = mem.get("_source", "")
        namespace_tag = f" [{source}]" if source else ""

        if mem_type in by_type:
            by_type[mem_type].append(f"- {content}{namespace_tag}")
        else:
            by_type["other"].append(f"- {content}{namespace_tag}")

    # Output sections
    if by_type["golden_rule"]:
        lines.append("## Golden Rules")
        lines.extend(by_type["golden_rule"])
        lines.append("")

    if by_type["preference"]:
        lines.append("## Preferences")
        lines.extend(by_type["preference"])
        lines.append("")

    if by_type["pattern"]:
        lines.append("## Patterns")
        lines.extend(by_type["pattern"])
        lines.append("")

    if by_type["decision"]:
        lines.append("## Recent Decisions")
        lines.extend(by_type["decision"])
        lines.append("")

    return "\n".join(lines)


def main():
    """Main hook entry point.

    Two-phase approach:
    1. Fetch raw memories from recall
    2. Curate with Ollama (or fallback to simple formatting)

    All errors are caught to prevent blocking Claude Code.
    """
    try:
        # Determine project namespace
        namespace, project_name = get_project_namespace()

        # Phase 1: Fetch raw memories
        memories = fetch_raw_memories(namespace)

        if not memories:
            # No memories to show
            return

        # Phase 2: Curate with Ollama
        context = curate_with_ollama(memories, project_name)

        # Fallback if Ollama fails
        if not context:
            context = fallback_context(memories)

        if context and context.strip():
            print(context)
            print()
            print("---")
            print("**Memory tip:** When you notice user preferences, technical decisions, or patterns worth remembering, use `memory_store_tool` to save them.")

    except Exception as e:
        # Silently fail - don't block Claude Code
        print(f"<!-- recall-context hook error: {e} -->", file=sys.stderr)


if __name__ == "__main__":
    main()
