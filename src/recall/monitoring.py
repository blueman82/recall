"""Memory system monitoring with Claude models.

This module provides automated monitoring of the recall memory system using
Claude models for quick checks (Haiku) and deep analysis (Opus).

Key capabilities:
- Quick health checks using Haiku (~$0.001)
- Deep analysis using Opus (~$0.10) when issues detected
- Detection of contradictions, golden rule violations, stale patterns
- Automated issue resolution suggestions
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from recall.config import RecallSettings
from recall.storage.hybrid import HybridStore
from recall.validation import analyze_memory_health

logger = logging.getLogger(__name__)


@dataclass
class MonitorIssue:
    """A single issue detected by monitoring.

    Attributes:
        category: Issue category (contradictions, low_confidence, stale, golden_rule)
        severity: Severity level (critical, warning, info)
        memory_id: Primary memory ID involved
        related_id: Related memory ID (for contradictions)
        description: Human-readable description
        recommendation: Suggested action
        metadata: Additional context
    """

    category: str
    severity: str
    memory_id: str
    related_id: Optional[str] = None
    description: str = ""
    recommendation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorResult:
    """Result of a monitoring run.

    Attributes:
        success: Whether monitoring completed successfully
        issues: List of detected issues
        analysis: Deep analysis results (if performed)
        timestamp: When monitoring ran
        error: Error message if failed
    """

    success: bool
    issues: list[MonitorIssue] = field(default_factory=list)
    analysis: Optional[dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


class Monitor:
    """Memory system monitor using Claude models.

    Uses a two-tier approach:
    1. Haiku for quick, cheap checks (~$0.001)
    2. Opus for deep analysis when issues found (~$0.10)

    Args:
        store: HybridStore instance for memory operations
        settings: RecallSettings for configuration
        use_anthropic: If True, use Anthropic API; if False, use Ollama

    Example:
        >>> settings = RecallSettings(anthropic_api_key="sk-...")
        >>> async with HybridStore.create() as store:
        ...     monitor = Monitor(store, settings)
        ...     result = await monitor.run_check()
        ...     if result.issues:
        ...         analysis = await monitor.analyze_issues(result.issues)
    """

    def __init__(
        self,
        store: HybridStore,
        settings: RecallSettings,
        use_anthropic: bool = True,
    ):
        """Initialize Monitor with store and settings.

        Args:
            store: HybridStore instance
            settings: RecallSettings instance
            use_anthropic: Whether to use Anthropic API (vs Ollama)
        """
        self._store = store
        self._settings = settings
        self._use_anthropic = use_anthropic

    async def haiku_check(
        self, namespace: Optional[str] = None
    ) -> MonitorResult:
        """Run quick health check using analyze_memory_health.

        This is the first tier - a fast, cheap check for common issues:
        - Unresolved CONTRADICTS edges
        - Golden rule violations
        - Stale patterns (not validated in 30 days)
        - Low-confidence memories (< 0.15)

        Args:
            namespace: Limit check to specific namespace (optional)

        Returns:
            MonitorResult with detected issues

        Example:
            >>> result = await monitor.haiku_check()
            >>> print(f"Found {len(result.issues)} issues")
        """
        try:
            # Use the existing analyze_memory_health function
            health = await analyze_memory_health(
                self._store,
                namespace=namespace,
                include_contradictions=True,
                include_low_confidence=True,
                include_stale=True,
                stale_days=30,
            )

            # Convert health report to MonitorIssues
            issues: list[MonitorIssue] = []

            # Add contradictions (critical)
            for contradiction in health.get("contradictions", []):
                issues.append(
                    MonitorIssue(
                        category="contradictions",
                        severity="critical",
                        memory_id=contradiction["memory_a_id"],
                        related_id=contradiction["memory_b_id"],
                        description=f"Contradiction: {contradiction['memory_a_content'][:50]}... vs {contradiction['memory_b_content'][:50]}...",
                        recommendation=contradiction["recommendation"],
                        metadata={
                            "memory_a_content": contradiction["memory_a_content"],
                            "memory_b_content": contradiction["memory_b_content"],
                        },
                    )
                )

            # Add low confidence (warnings)
            for low_conf in health.get("low_confidence", []):
                issues.append(
                    MonitorIssue(
                        category="low_confidence",
                        severity="warning",
                        memory_id=low_conf["memory_id"],
                        description=f"Low confidence ({low_conf['confidence']:.2f}): {low_conf['content']}",
                        recommendation=low_conf["recommendation"],
                        metadata={"confidence": low_conf["confidence"]},
                    )
                )

            # Add stale memories (warnings)
            for stale in health.get("stale", []):
                issues.append(
                    MonitorIssue(
                        category="stale",
                        severity="warning",
                        memory_id=stale["memory_id"],
                        description=f"Stale ({stale['days_since_access']} days): {stale['content']}",
                        recommendation=stale["recommendation"],
                        metadata={"days_since_access": stale["days_since_access"]},
                    )
                )

            # Check for golden rule violations
            # Golden rules should have high confidence and specific types
            memories = self._store.list_memories(namespace=namespace, limit=1000)
            for memory in memories:
                mem_type = memory.get("type", "")
                confidence = memory.get("confidence", 0.3)

                # Check if marked as golden_rule but low confidence
                if mem_type == "golden_rule" and confidence < 0.9:
                    issues.append(
                        MonitorIssue(
                            category="golden_rule",
                            severity="critical",
                            memory_id=memory["id"],
                            description=f"Golden rule with low confidence ({confidence:.2f})",
                            recommendation="Review and validate or downgrade from golden_rule type",
                            metadata={
                                "confidence": confidence,
                                "content": memory.get("content", "")[:100],
                            },
                        )
                    )

            logger.info(
                f"Haiku check completed: {len(issues)} issues found "
                f"({health['summary']['critical']} critical, {health['summary']['warnings']} warnings)"
            )

            return MonitorResult(success=True, issues=issues)

        except Exception as e:
            logger.error(f"Haiku check failed: {e}")
            return MonitorResult(success=False, error=str(e))

    async def opus_analyze(
        self, issues: list[MonitorIssue]
    ) -> Optional[dict[str, Any]]:
        """Run deep analysis using Opus for detected issues.

        This is the second tier - expensive but thorough analysis:
        - Root cause analysis
        - Resolution suggestions
        - Cleanup recommendations
        - Pattern detection across issues

        Args:
            issues: List of MonitorIssue to analyze

        Returns:
            Analysis results dict with recommendations, or None if failed

        Example:
            >>> analysis = await monitor.opus_analyze(result.issues)
            >>> print(analysis['summary'])
            >>> for rec in analysis['recommendations']:
            ...     print(f"- {rec}")
        """
        if not issues:
            return {
                "summary": "No issues to analyze",
                "recommendations": [],
            }

        try:
            # Build analysis prompt
            prompt = self._build_analysis_prompt(issues)

            # Call Claude Opus (or Ollama fallback)
            if self._use_anthropic:
                response = await self._call_anthropic_opus(prompt)
            else:
                response = await self._call_ollama_analysis(prompt)

            if response:
                logger.info(f"Opus analysis completed: {len(response.get('recommendations', []))} recommendations")
                return response

            return None

        except Exception as e:
            logger.error(f"Opus analysis failed: {e}")
            return None

    def _build_analysis_prompt(self, issues: list[MonitorIssue]) -> str:
        """Build analysis prompt for Claude Opus.

        Args:
            issues: List of issues to analyze

        Returns:
            Formatted prompt string
        """
        # Categorize issues
        critical = [i for i in issues if i.severity == "critical"]
        warnings = [i for i in issues if i.severity == "warning"]

        prompt = f"""You are analyzing a memory system's health. Found {len(issues)} issues:

CRITICAL ISSUES ({len(critical)}):
"""
        for i, issue in enumerate(critical, 1):
            prompt += f"\n{i}. {issue.category.upper()}: {issue.description}"
            if issue.related_id:
                prompt += f" (related: {issue.related_id})"

        prompt += f"\n\nWARNINGS ({len(warnings)}):\n"
        for i, issue in enumerate(warnings, 1):
            prompt += f"\n{i}. {issue.category.upper()}: {issue.description}"

        prompt += """

Please provide a deep analysis with:
1. Root cause analysis - why are these issues occurring?
2. Priority recommendations - what should be fixed first?
3. Cleanup strategy - how to resolve each category?
4. Prevention - how to avoid these issues in the future?

Respond ONLY with a JSON object:
{
  "summary": "Brief overall assessment",
  "root_causes": ["cause 1", "cause 2", ...],
  "recommendations": [
    {"priority": "high|medium|low", "action": "specific action", "rationale": "why"}
  ],
  "cleanup_steps": ["step 1", "step 2", ...],
  "prevention": ["prevention strategy 1", ...]
}
"""
        return prompt

    async def _call_claude_cli(
        self, prompt: str, model: str = "opus"
    ) -> Optional[dict[str, Any]]:
        """Call Claude CLI in headless mode for analysis.

        Uses `claude -p` with structured JSON output for reliable parsing.
        This approach leverages existing Claude Code authentication and
        avoids needing to manage API keys separately.

        Args:
            prompt: Analysis prompt
            model: Model alias (opus, haiku, sonnet)

        Returns:
            Parsed JSON response or None if failed

        Example:
            >>> result = await monitor._call_claude_cli(prompt, model="haiku")
            >>> print(result["summary"])
        """
        import asyncio
        import shutil

        try:
            # Check if claude CLI is available
            claude_path = shutil.which("claude")
            if not claude_path:
                logger.warning("Claude CLI not found in PATH")
                return None

            # Define JSON schema for structured output validation
            json_schema = json.dumps({
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "root_causes": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "priority": {"type": "string"},
                                "action": {"type": "string"},
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                    "cleanup_steps": {"type": "array", "items": {"type": "string"}},
                    "prevention": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "recommendations"],
            })

            # Build command with optimal flags:
            # -p: Print mode (non-interactive)
            # --output-format json: Structured JSON output
            # --model: Specify model (opus for deep analysis)
            # --system-prompt: Override default to ensure JSON-only responses
            # --no-session-persistence: Don't save to history
            system_prompt = (
                "You are a JSON-only responder for memory system analysis. "
                "Output valid JSON matching the requested schema. No explanations or markdown."
            )

            cmd = [
                claude_path,
                "-p",
                "--output-format", "json",
                "--model", model,
                "--system-prompt", system_prompt,
                "--no-session-persistence",
                prompt,
            ]

            # Run subprocess asynchronously
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=120,  # 2 minute timeout for complex analysis
            )

            if proc.returncode != 0:
                logger.error(f"Claude CLI failed: {stderr.decode()}")
                return None

            # Parse JSON output from Claude CLI
            output = stdout.decode().strip()
            if not output:
                logger.warning("Claude CLI returned empty output")
                return None

            # The JSON output format wraps the result in metadata
            # Example: {"type":"result","result":"```json\n{...}\n```",...}
            wrapper = json.loads(output)

            # Extract the actual content from the wrapper
            if isinstance(wrapper, dict) and "result" in wrapper:
                content = wrapper["result"]

                if isinstance(content, str):
                    # Strip markdown code blocks if present
                    # Claude often wraps JSON in ```json ... ```
                    content = content.strip()
                    if content.startswith("```"):
                        # Remove opening ```json or ```
                        lines = content.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        # Remove closing ```
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        content = "\n".join(lines)

                    return cast(dict[str, Any], json.loads(content))
                elif isinstance(content, dict):
                    return cast(dict[str, Any], content)

            return None

        except asyncio.TimeoutError:
            logger.error("Claude CLI timed out")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude CLI response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Claude CLI: {e}")
            return None

    async def _call_anthropic_opus(self, prompt: str) -> Optional[dict[str, Any]]:
        """Call Claude for deep analysis using CLI.

        This is a convenience wrapper that calls _call_claude_cli with opus model.
        Kept for backwards compatibility.

        Args:
            prompt: Analysis prompt

        Returns:
            Parsed JSON response or None if failed
        """
        return await self._call_claude_cli(prompt, model="opus")

    async def _call_ollama_analysis(self, prompt: str) -> Optional[dict[str, Any]]:
        """Call Ollama for analysis (fallback).

        Args:
            prompt: Analysis prompt

        Returns:
            Parsed JSON response or None if failed
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self._settings.ollama_host}/api/generate",
                    json={
                        "model": self._settings.ollama_monitor_model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
                response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "")

                # Parse JSON response and explicitly cast to expected type
                parsed = json.loads(response_text)
                return cast(dict[str, Any], parsed)

        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            return None

    async def run_check(
        self, namespace: Optional[str] = None, deep_analysis: bool = True
    ) -> MonitorResult:
        """Run full monitoring check with optional deep analysis.

        Combines haiku_check() for quick issue detection with opus_analyze()
        for deep analysis when issues are found.

        Args:
            namespace: Limit check to specific namespace (optional)
            deep_analysis: Run Opus analysis if issues found (default: True)

        Returns:
            MonitorResult with issues and optional analysis

        Example:
            >>> result = await monitor.run_check()
            >>> if result.success and result.issues:
            ...     for issue in result.issues:
            ...         print(f"{issue.severity}: {issue.description}")
            ...     if result.analysis:
            ...         print("Analysis:", result.analysis['summary'])
        """
        # Run quick check
        result = await self.haiku_check(namespace)

        if not result.success:
            return result

        # If issues found and deep analysis enabled, run Opus
        if result.issues and deep_analysis:
            analysis = await self.opus_analyze(result.issues)
            result.analysis = analysis

        return result
