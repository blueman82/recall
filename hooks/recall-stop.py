#!/usr/bin/env python3
"""Claude Code / Factory Stop hook for capturing session learnings.

This hook runs when the agent finishes responding. It can analyze the
conversation to identify important patterns, decisions, or issues that
should be remembered for future sessions.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-stop.py",
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
        "hook_event_name": "Stop",
        "stop_hook_active": false
    }

Output:
    - JSON with decision: "block" to continue, or nothing to allow stop
    - If blocking, must provide "reason" for the agent to continue
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


def read_transcript_tail(transcript_path: Optional[str], lines: int = 50) -> Optional[str]:
    """Read the last N lines of the transcript."""
    if not transcript_path:
        return None
    
    try:
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return None
        
        content = path.read_text()
        # Get last N lines
        all_lines = content.strip().split('\n')
        return '\n'.join(all_lines[-lines:])
    except Exception:
        return None


def analyze_for_learnings(transcript_tail: str) -> list[dict]:
    """Analyze transcript for potential learnings.
    
    Returns list of dicts with type and content for memories to store.
    This is a lightweight heuristic analysis - heavy analysis is done in SessionEnd.
    """
    learnings = []
    
    # Look for error patterns that were resolved
    if "error" in transcript_tail.lower() and ("fixed" in transcript_tail.lower() or "resolved" in transcript_tail.lower()):
        # There was an error that got fixed - potential pattern to remember
        pass  # Let SessionEnd capture the full context
    
    # Look for explicit memory storage requests
    if "remember" in transcript_tail.lower() or "note that" in transcript_tail.lower():
        # User asked to remember something - SessionEnd will capture
        pass
    
    return learnings


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


def record_stop_event(session_id: str, namespace: str) -> None:
    """Record that the agent stopped (for analytics/tracking)."""
    try:
        # This could be used to track session patterns
        # For now, we just log it
        log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-stop.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | STOP | session={session_id} | namespace={namespace}\n")
    except Exception:
        pass


def main():
    """Main hook entry point.
    
    The Stop hook is primarily for:
    1. Recording that the agent stopped (analytics)
    2. Optionally blocking the stop if work is incomplete
    
    Heavy analysis is deferred to SessionEnd hook.
    """
    try:
        hook_input = read_hook_input()
        session_id = hook_input.get("session_id") or hook_input.get("sessionId", "unknown")
        transcript_path = hook_input.get("transcript_path") or hook_input.get("transcriptPath")
        cwd = hook_input.get("cwd", os.getcwd())
        stop_hook_active = hook_input.get("stop_hook_active", False)
        
        # Prevent infinite loops - if stop hook already ran, don't block again
        if stop_hook_active:
            return
        
        # Change to session's working directory
        if cwd:
            os.chdir(cwd)
        
        namespace = get_project_namespace()
        
        # Record the stop event
        record_stop_event(session_id, namespace)
        
        # Read recent transcript
        transcript_tail = read_transcript_tail(transcript_path, lines=30)
        
        if transcript_tail:
            # Light analysis for immediate learnings
            learnings = analyze_for_learnings(transcript_tail)
            
            # Store any immediate learnings
            for learning in learnings:
                call_recall("memory_store_tool", {
                    "content": learning["content"],
                    "memory_type": learning.get("type", "pattern"),
                    "namespace": namespace,
                    "importance": 0.6,
                    "metadata": {
                        "source": "recall-stop",
                        "session_id": session_id,
                    },
                })
        
        # Allow the stop to proceed (don't output anything to block)
        # Full session analysis happens in SessionEnd hook
    
    except BrokenPipeError:
        pass
    except Exception as e:
        # Log error but don't block
        try:
            log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-stop.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | ERROR | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
