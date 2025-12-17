#!/usr/bin/env python3
"""Claude Code / Factory SubagentStop hook for tracking subagent (Task) results.

This hook runs when a subagent (Task tool call) finishes. It captures
the subagent's work for memory context, enabling better coordination
between subagents and the main agent.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "SubagentStop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-subagent.py",
                            "timeout": 10
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
        "hook_event_name": "SubagentStop",
        "stop_hook_active": false
    }

Output:
    - JSON with decision: "block" to continue, or nothing to allow stop
"""

import json
import os
import subprocess
import sys
from datetime import datetime
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


def read_transcript_tail(transcript_path: Optional[str], lines: int = 30) -> Optional[str]:
    """Read the last N lines of the transcript."""
    if not transcript_path:
        return None
    
    try:
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return None
        
        content = path.read_text()
        all_lines = content.strip().split('\n')
        return '\n'.join(all_lines[-lines:])
    except Exception:
        return None


def extract_subagent_summary(transcript_tail: str) -> Optional[str]:
    """Extract a summary of what the subagent accomplished.
    
    Parses the transcript to find the subagent's final output/report.
    """
    if not transcript_tail:
        return None
    
    # Look for common subagent completion patterns
    lines = transcript_tail.split('\n')
    summary_lines = []
    
    for line in lines[-20:]:  # Focus on last 20 lines
        # Skip empty lines and tool metadata
        if not line.strip():
            continue
        if line.startswith('{') and '"tool' in line:
            continue
        
        # Look for result/summary indicators
        lower_line = line.lower()
        if any(indicator in lower_line for indicator in [
            'completed', 'finished', 'done', 'result', 'summary',
            'found', 'created', 'updated', 'fixed', 'implemented'
        ]):
            summary_lines.append(line.strip())
    
    if summary_lines:
        return ' | '.join(summary_lines[-3:])  # Last 3 relevant lines
    
    return None


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
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, cwd=recall_dir or Path.cwd())
        
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


def record_subagent_activity(session_id: str, namespace: str, summary: Optional[str]) -> None:
    """Record subagent activity for tracking."""
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-subagent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(log_path, "a") as f:
            summary_preview = (summary[:100] + "...") if summary and len(summary) > 100 else summary
            f.write(f"{datetime.now().isoformat()} | SUBAGENT_STOP | session={session_id} | namespace={namespace} | summary={summary_preview}\n")
    except Exception:
        pass


def main():
    """Main hook entry point.
    
    Tracks subagent completion for:
    1. Analytics on subagent usage
    2. Capturing subagent results for future reference
    3. Enabling memory sharing between subagents
    """
    try:
        hook_input = read_hook_input()
        session_id = hook_input.get("session_id", "unknown")
        transcript_path = hook_input.get("transcript_path")
        cwd = hook_input.get("cwd", os.getcwd())
        stop_hook_active = hook_input.get("stop_hook_active", False)
        
        # Prevent infinite loops
        if stop_hook_active:
            return
        
        # Change to session's working directory
        if cwd:
            os.chdir(cwd)
        
        namespace = get_project_namespace()
        
        # Read recent transcript to understand what the subagent did
        transcript_tail = read_transcript_tail(transcript_path)
        summary = extract_subagent_summary(transcript_tail)
        
        # Record the activity
        record_subagent_activity(session_id, namespace, summary)
        
        # If we have a meaningful summary, store it as session context
        if summary and len(summary) > 20:
            call_recall("memory_store_tool", {
                "content": f"Subagent completed: {summary}",
                "memory_type": "session",
                "namespace": namespace,
                "importance": 0.4,  # Lower importance - session-specific
                "confidence": 0.7,
                "metadata": {
                    "source": "recall-subagent",
                    "session_id": session_id,
                    "is_subagent_result": True,
                },
            })
        
        # Allow the subagent to stop (don't output anything to block)
    
    except BrokenPipeError:
        pass
    except Exception as e:
        # Log error but don't block
        try:
            log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-subagent.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | ERROR | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
