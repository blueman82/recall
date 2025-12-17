#!/usr/bin/env python3
"""Claude Code / Factory PreCompact hook for preserving important context.

This hook runs BEFORE a compact operation (context window compression).
It extracts and stores important information that might be lost during
compaction, ensuring critical decisions and context are preserved.

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "PreCompact": [
                {
                    "matcher": "auto|manual",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-compact.py",
                            "timeout": 15
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
        "hook_event_name": "PreCompact",
        "trigger": "auto",
        "custom_instructions": ""
    }

Output:
    - Stores important context before it's compressed away
    - Does not block compaction
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


def read_transcript(transcript_path: Optional[str]) -> Optional[str]:
    """Read the full transcript."""
    if not transcript_path:
        return None
    
    try:
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return None
        return path.read_text()
    except Exception:
        return None


def extract_key_context(transcript: str) -> list[dict]:
    """Extract key context items from transcript before compaction.
    
    Focuses on:
    - Decisions made
    - Errors encountered and how they were resolved
    - Files modified
    - User corrections/preferences expressed
    """
    context_items = []
    
    if not transcript:
        return context_items
    
    # Parse transcript lines (JSONL format)
    lines = transcript.strip().split('\n')
    
    decisions = []
    errors_resolved = []
    user_corrections = []
    
    for line in lines:
        try:
            entry = json.loads(line)
            
            # Look for assistant messages with decision indicators
            if entry.get("role") == "assistant":
                content = entry.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
                
                lower_content = content.lower()
                
                # Detect decisions
                if any(phrase in lower_content for phrase in [
                    "i'll use", "i will use", "let's use", "we should use",
                    "i've decided", "the approach is", "implementing with"
                ]):
                    decisions.append(content[:200])
                
                # Detect error resolution
                if any(phrase in lower_content for phrase in [
                    "fixed by", "resolved by", "the issue was", "the problem was",
                    "solution:", "fix:"
                ]):
                    errors_resolved.append(content[:200])
            
            # Look for user corrections
            elif entry.get("role") == "user":
                content = entry.get("content", "")
                if isinstance(content, str):
                    lower_content = content.lower()
                    if any(phrase in lower_content for phrase in [
                        "no,", "don't", "instead", "actually", "wrong",
                        "that's not", "use X instead", "prefer"
                    ]):
                        user_corrections.append(content[:200])
        
        except (json.JSONDecodeError, AttributeError):
            continue
    
    # Convert to context items
    if decisions:
        context_items.append({
            "type": "decision",
            "content": f"Session decisions before compact: {' | '.join(decisions[-3:])}",
        })
    
    if errors_resolved:
        context_items.append({
            "type": "pattern",
            "content": f"Errors resolved in session: {' | '.join(errors_resolved[-3:])}",
        })
    
    if user_corrections:
        context_items.append({
            "type": "preference",
            "content": f"User corrections noted: {' | '.join(user_corrections[-3:])}",
        })
    
    return context_items


def summarize_with_ollama(transcript: str, model: str = "llama3.2") -> Optional[str]:
    """Use Ollama to extract key context before compaction."""
    if not transcript or len(transcript) < 500:
        return None
    
    # Truncate to last 12000 chars for speed
    if len(transcript) > 12000:
        transcript = "...(earlier context)...\n" + transcript[-12000:]
    
    prompt = f"""This conversation is about to be compacted (compressed). Extract ONLY the most critical information that should be remembered.

Focus on:
1. Technical decisions made (e.g., "Using FastAPI for backend")
2. User preferences expressed (e.g., "User prefers TypeScript")
3. Errors that were resolved and how
4. Important context that would be lost

Format as JSON array (max 5 items):
[
  {{"type": "decision", "content": "Brief decision..."}},
  {{"type": "preference", "content": "User preference..."}},
  {{"type": "pattern", "content": "Error pattern and fix..."}}
]

Return empty array [] if nothing critical to preserve.

Transcript:
{transcript}

JSON:"""

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode != 0:
            return None
        
        output = result.stdout.strip()
        
        # Extract JSON
        if "```json" in output:
            start = output.find("```json") + 7
            end = output.find("```", start)
            output = output[start:end].strip()
        elif "```" in output:
            start = output.find("```") + 3
            end = output.find("```", start)
            output = output[start:end].strip()
        elif "[" in output:
            start = output.find("[")
            depth = 0
            for i, char in enumerate(output[start:], start):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        output = output[start:i + 1].strip()
                        break
        
        return output
    
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    except Exception:
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


def main():
    """Main hook entry point.
    
    Preserves important context before compaction by:
    1. Extracting key decisions, preferences, and patterns
    2. Storing them in recall for future reference
    3. Logging the compaction event
    """
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-compact.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        hook_input = read_hook_input()
        session_id = hook_input.get("session_id") or hook_input.get("sessionId", "unknown")
        transcript_path = hook_input.get("transcript_path") or hook_input.get("transcriptPath")
        cwd = hook_input.get("cwd", os.getcwd())
        trigger = hook_input.get("trigger", "unknown")  # "auto" or "manual"
        custom_instructions = hook_input.get("custom_instructions", "")
        
        # Change to session's working directory
        if cwd:
            os.chdir(cwd)
        
        namespace = get_project_namespace()
        
        # Log the compaction event
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | PRE_COMPACT | trigger={trigger} | session={session_id} | namespace={namespace}\n")
        
        # Read transcript
        transcript = read_transcript(transcript_path)
        
        if not transcript or len(transcript) < 500:
            # Not enough content to preserve
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | SKIP | Transcript too short\n")
            return
        
        # Try Ollama summarization first
        summary_json = summarize_with_ollama(transcript)
        
        context_items = []
        if summary_json:
            try:
                context_items = json.loads(summary_json)
                if not isinstance(context_items, list):
                    context_items = []
            except json.JSONDecodeError:
                context_items = []
        
        # Fallback to heuristic extraction
        if not context_items:
            context_items = extract_key_context(transcript)
        
        if not context_items:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | SKIP | No critical context found\n")
            return
        
        # Store the context items
        stored = 0
        for item in context_items[:5]:  # Max 5 items
            content = item.get("content", "")
            mem_type = item.get("type", "session")
            
            if not content or len(content) < 10:
                continue
            
            # Add pre-compact marker
            content = f"[Pre-compact preservation] {content}"
            
            result = call_recall("memory_store_tool", {
                "content": content,
                "memory_type": mem_type,
                "namespace": namespace,
                "importance": 0.6,  # Moderate importance
                "confidence": 0.6,
                "metadata": {
                    "source": "recall-compact",
                    "trigger": trigger,
                    "session_id": session_id,
                },
            })
            
            if result.get("success"):
                stored += 1
        
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | STORED | {stored} items preserved before compact\n")
    
    except BrokenPipeError:
        pass
    except Exception as e:
        try:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | ERROR | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
