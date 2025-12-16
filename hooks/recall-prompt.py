#!/usr/bin/env python3
"""Claude Code / Factory UserPromptSubmit hook for memory context injection.

This hook runs BEFORE the user's prompt is processed, allowing injection of
relevant memory context based on the prompt content. This helps Claude/Droid
make better decisions by surfacing relevant preferences and past decisions.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-prompt.py",
                            "timeout": 5
                        }
                    ]
                }
            ]
        }
    }

Input (via stdin JSON):
    {
        "session_id": "abc123",
        "transcript_path": "/path/to/transcript.jsonl",
        "cwd": "/project/root",
        "permission_mode": "default",
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Create a new React component for user settings"
    }

Output:
    - stdout: Additional context to inject (shown to Claude/Droid)
    - JSON with decision: "block" to reject prompt, or additionalContext
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def read_hook_input() -> dict:
    """Read hook input from stdin."""
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
    """Derive project namespace from current working directory."""
    cwd = os.getcwd()
    project_name = Path(cwd).name
    
    project_indicators = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for indicator in project_indicators:
        if Path(cwd, indicator).exists():
            return f"project:{project_name}"
    return "global"


def extract_search_terms(prompt: str) -> list[str]:
    """Extract search terms from the user's prompt."""
    prompt_lower = prompt.lower()
    terms = set()
    
    for keyword, search_terms in PROMPT_TRIGGERS.items():
        if keyword in prompt_lower:
            terms.update(search_terms)
    
    # Also extract potential file extensions mentioned
    import re
    extensions = re.findall(r'\.(py|ts|tsx|js|jsx|rs|go|java|rb|php|swift|kt|cs)\b', prompt_lower)
    terms.update(extensions)
    
    return list(terms)


def call_recall(tool_name: str, args: dict) -> dict:
    """Call recall MCP tool directly via --call mode."""
    try:
        recall_paths = [
            Path(__file__).parent.parent,
            Path.home() / "Documents" / "Github" / "recall",
            Path.home() / ".local" / "share" / "recall",
            Path("/opt/recall"),
        ]
        
        recall_dir = None
        for path in recall_paths:
            if (path / "src" / "recall" / "__main__.py").exists():
                recall_dir = path
                break
        
        if recall_dir is None:
            cmd = ["uv", "run", "python", "-m", "recall", "--call", tool_name, "--args", json.dumps(args)]
        else:
            cmd = ["uv", "run", "--directory", str(recall_dir), "python", "-m", "recall", "--call", tool_name, "--args", json.dumps(args)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3, cwd=recall_dir or Path.cwd())
        
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


def format_context(memories: list[dict], prompt: str) -> Optional[str]:
    """Format relevant memories as context for the prompt."""
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
    
    lines = ["# Relevant Memories for This Request", ""]
    
    # Separate golden rules
    golden_rules = [m for m in relevant if m.get("type") == "golden_rule"]
    other = [m for m in relevant if m.get("type") != "golden_rule"]
    
    if golden_rules:
        lines.append("## Golden Rules (MUST follow)")
        for mem in golden_rules[:3]:
            content = mem.get("content", "")[:300]
            lines.append(f"- {content}")
        lines.append("")
    
    if other:
        lines.append("## Relevant Context")
        for mem in other[:5]:
            mem_type = mem.get("type", "memory").upper()
            content = mem.get("content", "")[:200]
            lines.append(f"- [{mem_type}] {content}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Main hook entry point."""
    try:
        hook_input = read_hook_input()
        prompt = hook_input.get("prompt", "")
        cwd = hook_input.get("cwd", os.getcwd())
        
        if not prompt or len(prompt) < 5:
            return
        
        # Change to session's working directory
        if cwd:
            os.chdir(cwd)
        
        # Extract search terms from prompt
        search_terms = extract_search_terms(prompt)
        
        # Always search with the prompt itself too
        query_parts = search_terms + [prompt[:100]]
        query = " ".join(query_parts)
        
        if not query.strip():
            return
        
        namespace = get_project_namespace()
        
        # Search for relevant memories
        result = call_recall("memory_recall_tool", {
            "query": query,
            "n_results": 10,
            "namespace": namespace,
            "include_related": True,
            "max_depth": 1,
        })
        
        memories = []
        if result.get("success"):
            memories = result.get("memories", [])
        
        # Also check global namespace
        if namespace != "global":
            global_result = call_recall("memory_recall_tool", {
                "query": query,
                "n_results": 5,
                "namespace": "global",
            })
            if global_result.get("success"):
                memories.extend(global_result.get("memories", []))
        
        # Format and output context
        context = format_context(memories, prompt)
        if context:
            # Output as JSON for structured response
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            }
            print(json.dumps(output))
    
    except BrokenPipeError:
        pass
    except Exception:
        # Silently fail - don't block the prompt
        pass


if __name__ == "__main__":
    main()
