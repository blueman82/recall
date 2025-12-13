"""Validation and contradiction detection for the recall system.

This module provides functionality for:
- Detecting contradictions between memories
- Auto-supersede logic for replacing outdated memories
- Validation loop support (TRY → BREAK → ANALYZE → LEARN)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from recall.memory.types import RelationType
from recall.storage.hybrid import HybridStore

logger = logging.getLogger(__name__)

# Similarity threshold for considering memories as potentially contradictory
CONTRADICTION_SIMILARITY_THRESHOLD = 0.7

# Prompt template for LLM contradiction detection
CONTRADICTION_PROMPT_TEMPLATE = """You are analyzing two memories for potential contradictions.

Memory A: {memory_a}
Memory B: {memory_b}

Do these memories contradict each other? A contradiction means they make incompatible claims about the same thing, such as:
- Opposite preferences ("prefers X" vs "prefers Y" for the same thing)
- Conflicting decisions ("use library A" vs "use library B" for the same purpose)
- Mutually exclusive patterns or facts

Respond with ONLY a JSON object:
{{"contradicts": true/false, "reason": "brief explanation"}}"""


@dataclass
class ContradictionResult:
    """Result of contradiction detection for a memory.

    Attributes:
        memory_id: ID of the memory that was checked
        contradictions: List of memory IDs that contradict this memory
        edges_created: Number of CONTRADICTS edges created
        error: Error message if detection failed
    """
    memory_id: str
    contradictions: list[str] = field(default_factory=list)
    edges_created: int = 0
    error: Optional[str] = None


@dataclass
class SupersedeResult:
    """Result of auto-supersede check.

    Attributes:
        memory_id: ID of the newer memory
        superseded_id: ID of the memory that was superseded (if any)
        edge_created: Whether a SUPERSEDES edge was created
        reason: Explanation for the supersede decision
        error: Error message if check failed
    """
    memory_id: str
    superseded_id: Optional[str] = None
    edge_created: bool = False
    reason: Optional[str] = None
    error: Optional[str] = None


async def _call_ollama_llm(
    prompt: str,
    host: str = "http://localhost:11434",
    model: str = "llama3.2",
    timeout: int = 30,
) -> Optional[dict[str, Any]]:
    """Call Ollama LLM for contradiction analysis.

    Args:
        prompt: The prompt to send to the LLM
        host: Ollama server host URL
        model: Model to use for generation
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response or None if failed
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "")

            # Parse the JSON response
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    return json.loads(json_match.group())
                logger.warning(f"Failed to parse LLM response as JSON: {response_text}")
                return None

    except httpx.HTTPError as e:
        logger.warning(f"HTTP error calling Ollama LLM: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error calling Ollama LLM: {e}")
        return None


async def detect_contradictions(
    store: HybridStore,
    memory_id: str,
    similarity_threshold: float = CONTRADICTION_SIMILARITY_THRESHOLD,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llama3.2",
    create_edges: bool = True,
) -> ContradictionResult:
    """Detect memories that contradict a given memory.

    Uses a two-phase approach:
    1. Find semantically similar memories using ChromaDB (similarity > threshold)
    2. Use LLM reasoning to determine if they actually contradict

    Optionally creates CONTRADICTS edges between contradicting memories.

    Args:
        store: HybridStore instance for storage operations
        memory_id: ID of the memory to check for contradictions
        similarity_threshold: Minimum similarity for considering a contradiction (default: 0.7)
        ollama_host: Ollama server host for LLM calls
        ollama_model: Model to use for contradiction detection
        create_edges: Whether to create CONTRADICTS edges (default: True)

    Returns:
        ContradictionResult with list of contradicting memory IDs

    Example:
        >>> result = await detect_contradictions(store, "mem_123")
        >>> for conflict_id in result.contradictions:
        ...     print(f"Memory {memory_id} contradicts {conflict_id}")
    """
    # Get the source memory
    memory = await store.get_memory(memory_id)
    if memory is None:
        return ContradictionResult(
            memory_id=memory_id,
            error=f"Memory '{memory_id}' not found",
        )

    source_content = memory.get("content", "")
    source_namespace = memory.get("namespace", "global")

    # Search for semantically similar memories in the same namespace
    try:
        similar_memories = await store.search(
            query=source_content,
            n_results=20,  # Get top 20 similar memories
            namespace=source_namespace,
        )
    except Exception as e:
        return ContradictionResult(
            memory_id=memory_id,
            error=f"Failed to search for similar memories: {e}",
        )

    # Filter by similarity threshold and exclude self
    candidates = []
    for mem in similar_memories:
        if mem["id"] == memory_id:
            continue
        similarity = mem.get("similarity", 0.0)
        if similarity >= similarity_threshold:
            candidates.append(mem)

    if not candidates:
        return ContradictionResult(memory_id=memory_id)

    # Check each candidate for contradiction using LLM
    contradictions: list[str] = []
    edges_created = 0

    for candidate in candidates:
        candidate_content = candidate.get("content", "")
        candidate_id = candidate["id"]

        # Build prompt for LLM
        prompt = CONTRADICTION_PROMPT_TEMPLATE.format(
            memory_a=source_content,
            memory_b=candidate_content,
        )

        # Call LLM for contradiction analysis
        llm_result = await _call_ollama_llm(
            prompt=prompt,
            host=ollama_host,
            model=ollama_model,
        )

        if llm_result and llm_result.get("contradicts", False):
            contradictions.append(candidate_id)
            logger.info(
                f"Contradiction detected: {memory_id} <-> {candidate_id}: "
                f"{llm_result.get('reason', 'No reason given')}"
            )

            # Create CONTRADICTS edge if requested
            if create_edges:
                try:
                    # Check if edge already exists
                    existing_edges = store.get_edges(
                        memory_id,
                        direction="both",
                        edge_type="contradicts",
                    )
                    edge_exists = any(
                        (e["source_id"] == memory_id and e["target_id"] == candidate_id) or
                        (e["source_id"] == candidate_id and e["target_id"] == memory_id)
                        for e in existing_edges
                    )

                    if not edge_exists:
                        store.add_edge(
                            source_id=memory_id,
                            target_id=candidate_id,
                            edge_type=RelationType.CONTRADICTS.value,
                            weight=1.0,
                            metadata={"reason": llm_result.get("reason", "")},
                        )
                        edges_created += 1
                except Exception as e:
                    logger.warning(f"Failed to create CONTRADICTS edge: {e}")

    return ContradictionResult(
        memory_id=memory_id,
        contradictions=contradictions,
        edges_created=edges_created,
    )


async def check_supersedes(
    store: HybridStore,
    memory_id: str,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llama3.2",
    create_edge: bool = True,
) -> SupersedeResult:
    """Check if a memory should supersede another based on validation history.

    A newer memory supersedes an older one when:
    1. They are semantically similar (same topic)
    2. The newer memory consistently succeeds where the older one failed
    3. They have sufficient validation history to compare

    Args:
        store: HybridStore instance for storage operations
        memory_id: ID of the (potentially newer) memory to check
        ollama_host: Ollama server host for LLM calls
        ollama_model: Model to use for supersede analysis
        create_edge: Whether to create SUPERSEDES edge (default: True)

    Returns:
        SupersedeResult with superseded memory ID if applicable

    Example:
        >>> result = await check_supersedes(store, "mem_new")
        >>> if result.superseded_id:
        ...     print(f"{memory_id} supersedes {result.superseded_id}")
    """
    # Get the source memory
    memory = await store.get_memory(memory_id)
    if memory is None:
        return SupersedeResult(
            memory_id=memory_id,
            error=f"Memory '{memory_id}' not found",
        )

    source_content = memory.get("content", "")
    source_namespace = memory.get("namespace", "global")
    source_confidence = memory.get("confidence", 0.3)
    source_created = memory.get("created_at", 0)

    # Get validation history for source memory
    source_events = store.get_validation_events(memory_id)
    source_successes = sum(1 for e in source_events if e.get("event_type") == "succeeded")
    source_failures = sum(1 for e in source_events if e.get("event_type") == "failed")

    # Need at least 2 successful validations to consider superseding
    if source_successes < 2:
        return SupersedeResult(memory_id=memory_id)

    # Search for semantically similar older memories
    try:
        similar_memories = await store.search(
            query=source_content,
            n_results=10,
            namespace=source_namespace,
        )
    except Exception as e:
        return SupersedeResult(
            memory_id=memory_id,
            error=f"Failed to search for similar memories: {e}",
        )

    # Find older memories with worse performance
    for candidate in similar_memories:
        candidate_id = candidate["id"]

        # Skip self
        if candidate_id == memory_id:
            continue

        # Check if candidate is older
        candidate_created = candidate.get("created_at", float("inf"))
        if candidate_created >= source_created:
            continue  # Candidate is newer or same age

        # Check candidate's confidence (lower = worse performance)
        candidate_confidence = candidate.get("confidence", 0.3)
        if candidate_confidence >= source_confidence:
            continue  # Candidate is performing better or equally

        # Get candidate's validation history
        candidate_events = store.get_validation_events(candidate_id)
        candidate_failures = sum(1 for e in candidate_events if e.get("event_type") == "failed")

        # Candidate should have more failures than source
        if candidate_failures <= source_failures:
            continue

        # Use LLM to confirm they're about the same topic and newer is better
        prompt = f"""Compare these two memories about the same topic:

Older Memory (created first): {candidate.get('content', '')}
- Confidence: {candidate_confidence:.2f}
- Failures: {candidate_failures}

Newer Memory (created later): {source_content}
- Confidence: {source_confidence:.2f}
- Successes: {source_successes}

Does the newer memory represent an improved understanding or correction of the older one?
Respond with ONLY a JSON object:
{{"supersedes": true/false, "reason": "brief explanation"}}"""

        llm_result = await _call_ollama_llm(
            prompt=prompt,
            host=ollama_host,
            model=ollama_model,
        )

        if llm_result and llm_result.get("supersedes", False):
            reason = llm_result.get("reason", "Better performance in validation")

            # Create SUPERSEDES edge if requested
            if create_edge:
                try:
                    store.add_edge(
                        source_id=memory_id,
                        target_id=candidate_id,
                        edge_type=RelationType.SUPERSEDES.value,
                        weight=1.0,
                        metadata={"reason": reason},
                    )

                    # Reduce importance of superseded memory
                    current_importance = candidate.get("importance", 0.5)
                    await store.update_memory(
                        candidate_id,
                        importance=current_importance * 0.5,
                    )

                    return SupersedeResult(
                        memory_id=memory_id,
                        superseded_id=candidate_id,
                        edge_created=True,
                        reason=reason,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create SUPERSEDES edge: {e}")
                    return SupersedeResult(
                        memory_id=memory_id,
                        superseded_id=candidate_id,
                        edge_created=False,
                        reason=reason,
                        error=str(e),
                    )
            else:
                return SupersedeResult(
                    memory_id=memory_id,
                    superseded_id=candidate_id,
                    edge_created=False,
                    reason=reason,
                )

    return SupersedeResult(memory_id=memory_id)


async def analyze_memory_health(
    store: HybridStore,
    namespace: Optional[str] = None,
    include_contradictions: bool = True,
    include_low_confidence: bool = True,
    include_stale: bool = True,
    stale_days: int = 30,
) -> dict[str, Any]:
    """Analyze the health of memories in the system.

    Checks for:
    1. Unresolved contradictions
    2. Low-confidence memories (< 0.15, candidates for deletion)
    3. Stale memories (not validated in N days)

    Args:
        store: HybridStore instance for storage operations
        namespace: Limit analysis to specific namespace (optional)
        include_contradictions: Check for contradictions (default: True)
        include_low_confidence: Find low-confidence memories (default: True)
        include_stale: Find stale memories (default: True)
        stale_days: Days without validation to consider stale (default: 30)

    Returns:
        Dict with categorized issues and recommendations
    """
    import time

    issues: dict[str, Any] = {
        "contradictions": [],
        "low_confidence": [],
        "stale": [],
        "summary": {
            "total_issues": 0,
            "critical": 0,
            "warnings": 0,
        },
    }

    # Get all memories for analysis
    memories = store.list_memories(
        namespace=namespace,
        limit=1000,
    )

    now = time.time()
    stale_threshold = now - (stale_days * 24 * 60 * 60)

    for memory in memories:
        memory_id = memory["id"]
        confidence = memory.get("confidence", 0.3)
        accessed_at = memory.get("accessed_at", now)

        # Check for low confidence (candidate for deletion)
        if include_low_confidence and confidence < 0.15:
            issues["low_confidence"].append({
                "memory_id": memory_id,
                "content": memory.get("content", "")[:100],
                "confidence": confidence,
                "recommendation": "Consider deleting this memory",
            })
            issues["summary"]["warnings"] += 1
            issues["summary"]["total_issues"] += 1

        # Check for stale memories
        if include_stale and accessed_at < stale_threshold:
            # Only flag non-golden-rule memories as stale
            mem_type = memory.get("type", "session")
            if mem_type != "golden_rule" and confidence < 0.9:
                days_stale = (now - accessed_at) / (24 * 60 * 60)
                issues["stale"].append({
                    "memory_id": memory_id,
                    "content": memory.get("content", "")[:100],
                    "days_since_access": int(days_stale),
                    "recommendation": "Validate or delete this memory",
                })
                issues["summary"]["warnings"] += 1
                issues["summary"]["total_issues"] += 1

    # Check for contradictions (more expensive, so optional)
    if include_contradictions:
        # Look for existing CONTRADICTS edges
        for memory in memories:
            memory_id = memory["id"]
            edges = store.get_edges(memory_id, edge_type="contradicts")

            for edge in edges:
                # Only report each contradiction once (source side)
                if edge["source_id"] == memory_id:
                    target = await store.get_memory(edge["target_id"])
                    issues["contradictions"].append({
                        "memory_a_id": memory_id,
                        "memory_a_content": memory.get("content", "")[:100],
                        "memory_b_id": edge["target_id"],
                        "memory_b_content": target.get("content", "")[:100] if target else "Unknown",
                        "recommendation": "Resolve contradiction by validating or deleting one memory",
                    })
                    issues["summary"]["critical"] += 1
                    issues["summary"]["total_issues"] += 1

    return issues
