#!/usr/bin/env python3
"""Claude Code / Factory Notification hook for capturing notifications.

This hook runs when the agent emits notifications. It captures error patterns,
permission requests, and idle notifications for recall tracking.

Notification types:
- permission_prompt: Permission requests from the agent
- idle_prompt: When agent is waiting for user input (60+ seconds idle)
- error: Error notifications (custom, not standard Claude Code)

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "Notification": [
                {
                    "matcher": "permission_prompt|idle_prompt",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python /path/to/recall/hooks/recall-errors.py",
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
        "hook_event_name": "Notification",
        "message": "Agent needs your permission to use Bash",
        "notification_type": "permission_prompt"
    }

The hook extracts notification information and logs/stores relevant patterns.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Common error patterns and their categories
ERROR_PATTERNS = {
    # Node.js / JavaScript
    r"Cannot find module": "node_module_not_found",
    r"SyntaxError:": "javascript_syntax_error",
    r"TypeError:": "javascript_type_error",
    r"ReferenceError:": "javascript_reference_error",
    r"ENOENT": "file_not_found",
    r"EACCES": "permission_denied",

    # Python
    r"ModuleNotFoundError": "python_import_error",
    r"ImportError": "python_import_error",
    r"SyntaxError": "python_syntax_error",
    r"IndentationError": "python_indentation_error",
    r"TypeError": "python_type_error",
    r"AttributeError": "python_attribute_error",
    r"KeyError": "python_key_error",
    r"ValueError": "python_value_error",

    # Git
    r"fatal:": "git_fatal_error",
    r"merge conflict": "git_merge_conflict",
    r"Permission denied \(publickey\)": "git_ssh_auth_error",

    # Docker
    r"Cannot connect to the Docker daemon": "docker_daemon_error",
    r"no such image": "docker_image_not_found",
    r"port is already allocated": "docker_port_conflict",

    # Build tools
    r"error: linker": "linker_error",
    r"error\[E\d+\]": "rust_compiler_error",
    r"error TS\d+": "typescript_compiler_error",

    # Network
    r"ECONNREFUSED": "connection_refused",
    r"ETIMEDOUT": "connection_timeout",
    r"getaddrinfo ENOTFOUND": "dns_resolution_failed",
}


def read_hook_input() -> dict:
    """Read hook input from stdin.

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


def categorize_error(message: str, details: dict) -> tuple[str, str]:
    """Categorize an error based on its message and details.

    Args:
        message: Error message string
        details: Additional error details

    Returns:
        Tuple of (category, subcategory)
    """
    # Combine message and stderr for pattern matching
    full_text = message
    if details.get("stderr"):
        full_text += "\n" + details["stderr"]
    if details.get("stdout"):
        full_text += "\n" + details["stdout"]

    # Check against known patterns
    for pattern, category in ERROR_PATTERNS.items():
        if re.search(pattern, full_text, re.IGNORECASE):
            return category.split("_")[0], category

    # Default categorization based on tool
    tool = details.get("tool", "unknown").lower()
    return tool, f"{tool}_error"


def extract_error_context(message: str, details: dict) -> dict:
    """Extract useful context from error.

    Args:
        message: Error message
        details: Error details

    Returns:
        Dictionary with extracted context
    """
    context = {
        "tool": details.get("tool"),
        "command": details.get("command"),
        "file_path": details.get("file_path"),
        "exit_code": details.get("exit_code"),
    }

    # Extract file paths from error message
    file_matches = re.findall(r'(?:/[\w./\-]+)+\.[\w]+', message)
    if file_matches:
        context["mentioned_files"] = file_matches[:5]

    # Extract line numbers
    line_matches = re.findall(r'line (\d+)', message, re.IGNORECASE)
    if line_matches:
        context["line_numbers"] = [int(l) for l in line_matches[:5]]

    # Clean up None values
    return {k: v for k, v in context.items() if v is not None}


def format_error_memory(message: str, details: dict, category: str, subcategory: str) -> str:
    """Format error information as a memory content string.

    Args:
        message: Error message
        details: Error details
        category: Error category
        subcategory: Error subcategory

    Returns:
        Formatted memory content
    """
    lines = [f"Error Pattern [{subcategory}]:"]

    # Add command/file context
    if details.get("command"):
        cmd = details["command"][:100]
        lines.append(f"Command: {cmd}")
    if details.get("file_path"):
        lines.append(f"File: {details['file_path']}")

    # Add error message (truncated)
    error_text = message[:500]
    lines.append(f"Error: {error_text}")

    # Add stderr snippet if available
    if details.get("stderr"):
        stderr = details["stderr"][:300]
        lines.append(f"Stderr: {stderr}")

    # Add timestamp
    lines.append(f"Occurred: {datetime.now().isoformat()}")

    return "\n".join(lines)


def call_recall(tool_name: str, args: dict) -> dict:
    """Call recall MCP tool directly via --call mode.

    Args:
        tool_name: Name of the tool
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
            timeout=5,
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


def check_duplicate_error(namespace: str, subcategory: str, message: str) -> bool:
    """Check if a similar error was already captured recently.

    Args:
        namespace: Memory namespace
        subcategory: Error subcategory
        message: Error message

    Returns:
        True if duplicate found, False otherwise
    """
    try:
        result = call_recall("memory_recall_tool", {
            "query": f"{subcategory} {message[:50]}",
            "n_results": 3,
            "namespace": namespace,
            "memory_type": "pattern",
        })

        if not result.get("success"):
            return False

        # Check if any recent memory has very similar content
        for mem in result.get("memories", []):
            content = mem.get("content", "")
            if subcategory in content and message[:30] in content:
                return True

        return False

    except Exception:
        return False


def handle_permission_prompt(hook_input: dict, log_path: Path) -> None:
    """Handle permission_prompt notifications."""
    message = hook_input.get("message", "")
    session_id = hook_input.get("session_id")

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | PERMISSION | {message[:100]}\n")


def handle_idle_prompt(hook_input: dict, log_path: Path) -> None:
    """Handle idle_prompt notifications."""
    message = hook_input.get("message", "")
    session_id = hook_input.get("session_id")

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | IDLE | {message[:100]}\n")


def handle_error_notification(hook_input: dict, log_path: Path) -> None:
    """Handle error notifications and store as patterns."""
    message = hook_input.get("message", "")
    details = hook_input.get("details", {})
    session_id = hook_input.get("session_id")

    if not message:
        return

    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} | ERROR | {message[:100]}\n")

    namespace = get_project_namespace()
    category, subcategory = categorize_error(message, details)

    if check_duplicate_error(namespace, subcategory, message):
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | SKIP | Duplicate error\n")
        return

    context = extract_error_context(message, details)
    content = format_error_memory(message, details, category, subcategory)

    result = call_recall("memory_store_tool", {
        "content": content,
        "memory_type": "pattern",
        "namespace": namespace,
        "importance": 0.6,
        "metadata": {
            "source": "recall-errors",
            "category": category,
            "subcategory": subcategory,
            "session_id": session_id,
            **context,
        },
    })

    if result.get("success"):
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | STORED | {subcategory}\n")
    else:
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} | FAIL | {result.get('error')}\n")


def main():
    """Main hook entry point.

    Handles different notification types:
    - permission_prompt: Logs permission requests
    - idle_prompt: Logs idle notifications
    - error: Captures and stores error patterns
    """
    log_path = Path.home() / ".claude" / "hooks" / "logs" / "recall-errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hook_input = read_hook_input()

        # Get notification type (new format) or type (legacy format)
        notification_type = hook_input.get("notification_type") or hook_input.get("type", "")

        if notification_type == "permission_prompt":
            handle_permission_prompt(hook_input, log_path)

        elif notification_type == "idle_prompt":
            handle_idle_prompt(hook_input, log_path)

        elif notification_type == "error":
            handle_error_notification(hook_input, log_path)

        else:
            # Log unknown notification types for debugging
            message = hook_input.get("message", "")[:50]
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | UNKNOWN | type={notification_type} | {message}\n")

    except BrokenPipeError:
        pass
    except Exception as e:
        try:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()} | EXCEPTION | {e}\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
