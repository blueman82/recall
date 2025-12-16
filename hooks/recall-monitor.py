#!/usr/bin/env python3
"""Claude Code / Factory SessionEnd hook for memory system monitoring.

This hook runs at the end of each session and performs automated
health checks on the recall memory system:
- Quick Haiku check for common issues (~$0.001)
- Deep Opus analysis if issues found (~$0.10)
- Outputs warnings/suggestions to stderr
- Stores analysis results as memories

Usage:
    Configure in ~/.claude/settings.json (Claude Code) or ~/.factory/settings.json (Factory):
    {
        "hooks": {
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "uv run --directory /path/to/recall python hooks/recall-monitor.py",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }

Environment Variables:
    RECALL_MONITOR_ENABLED: Set to "true" to enable (default: false)
    RECALL_MONITOR_DEEP: Set to "true" for deep Opus analysis (default: true)

Note: Deep analysis uses Claude Code headless mode (`claude -p`), not the Anthropic API.
      This leverages your existing Claude Code authentication - no separate API key needed.

The hook runs silently unless issues are found, then outputs to stderr.
Failures are handled gracefully - they don't block Claude Code.
"""

import asyncio
import json
import os
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


async def run_monitoring() -> None:
    """Run memory system monitoring.

    Performs health check and deep analysis if issues found.
    Outputs results to stderr and stores analysis as memory.
    """
    try:
        # Check if monitoring is enabled
        enabled = os.environ.get("RECALL_MONITOR_ENABLED", "false").lower() == "true"
        if not enabled:
            # Silently skip if not enabled
            return

        # Import recall modules
        from recall.config import RecallSettings
        from recall.monitoring import Monitor
        from recall.storage.hybrid import HybridStore

        # Load settings
        settings = RecallSettings()

        # Determine namespace for storing results
        namespace = get_project_namespace()

        # Create store
        store = await HybridStore.create(
            sqlite_path=settings.get_sqlite_path(),
            chroma_path=settings.get_chroma_path(),
            collection_name=settings.collection_name,
            ollama_host=settings.ollama_host,
            ollama_model=settings.ollama_model,
        )

        try:
            # Create monitor (uses Claude CLI headless mode for deep analysis)
            monitor = Monitor(store, settings, use_claude_cli=True)

            # Run check on ALL namespaces (None = no filter)
            # Monitoring should check everything, not just current project
            deep_analysis = os.environ.get("RECALL_MONITOR_DEEP", "true").lower() == "true"
            result = await monitor.run_check(
                namespace=None,  # Check all namespaces
                deep_analysis=deep_analysis,
            )

            if not result.success:
                print(f"[recall-monitor] Check failed: {result.error}", file=sys.stderr)
                return

            # If no issues, exit silently
            if not result.issues:
                return

            # Issues found - output summary to stderr
            critical_count = sum(1 for i in result.issues if i.severity == "critical")
            warning_count = sum(1 for i in result.issues if i.severity == "warning")

            print("\n" + "="*60, file=sys.stderr)
            print("[recall-monitor] Memory System Health Report", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Found {len(result.issues)} issues:", file=sys.stderr)
            print(f"  - {critical_count} CRITICAL", file=sys.stderr)
            print(f"  - {warning_count} WARNINGS", file=sys.stderr)
            print("", file=sys.stderr)

            # Group issues by category
            by_category: dict[str, list] = {}
            for issue in result.issues:
                if issue.category not in by_category:
                    by_category[issue.category] = []
                by_category[issue.category].append(issue)

            # Output issues by category
            for category, issues in by_category.items():
                print(f"{category.upper()} ({len(issues)}):", file=sys.stderr)
                for issue in issues[:3]:  # Show first 3 of each category
                    severity_marker = "🔴" if issue.severity == "critical" else "🟡"
                    print(f"  {severity_marker} {issue.description}", file=sys.stderr)
                    print(f"     → {issue.recommendation}", file=sys.stderr)
                if len(issues) > 3:
                    print(f"  ... and {len(issues) - 3} more", file=sys.stderr)
                print("", file=sys.stderr)

            # If deep analysis was performed, output key recommendations
            if result.analysis:
                print("ANALYSIS:", file=sys.stderr)
                print(f"  {result.analysis.get('summary', 'No summary')}", file=sys.stderr)
                print("", file=sys.stderr)

                recommendations = result.analysis.get("recommendations", [])
                if recommendations:
                    print("TOP RECOMMENDATIONS:", file=sys.stderr)
                    for i, rec in enumerate(recommendations[:3], 1):
                        priority = rec.get("priority", "medium").upper()
                        action = rec.get("action", "")
                        print(f"  {i}. [{priority}] {action}", file=sys.stderr)
                    print("", file=sys.stderr)

                # Store analysis as memory
                try:
                    analysis_content = (
                        f"Memory system monitoring found {len(result.issues)} issues:\n"
                        f"Critical: {critical_count}, Warnings: {warning_count}\n\n"
                        f"Analysis: {result.analysis.get('summary', '')}\n\n"
                        f"Recommendations:\n"
                    )
                    for rec in recommendations:
                        analysis_content += f"- [{rec.get('priority', 'medium')}] {rec.get('action', '')}\n"

                    await store.add_memory(
                        content=analysis_content,
                        memory_type="session",
                        namespace=namespace,
                        importance=0.7 if critical_count > 0 else 0.5,
                        confidence=0.8,
                        metadata={
                            "source": "recall-monitor",
                            "issues_found": len(result.issues),
                            "critical_count": critical_count,
                            "warning_count": warning_count,
                        },
                    )
                except Exception as e:
                    print(f"[recall-monitor] Failed to store analysis: {e}", file=sys.stderr)

            print("="*60, file=sys.stderr)
            print("", file=sys.stderr)

        finally:
            await store.close()

    except ImportError as e:
        # Recall not installed or import failed
        print(f"[recall-monitor] Import error (recall may not be installed): {e}", file=sys.stderr)
    except Exception as e:
        # Silently fail - don't block Claude Code
        print(f"[recall-monitor] Error: {e}", file=sys.stderr)


def main():
    """Main hook entry point.

    Runs monitoring asynchronously and handles all errors gracefully.
    """
    try:
        asyncio.run(run_monitoring())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[recall-monitor] Fatal error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
