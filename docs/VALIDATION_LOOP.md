# Validation Loop - Building Confidence in Memories

The validation loop is Recall's mechanism for learning which memories are actually useful. Memories start with low confidence and gain or lose confidence based on real-world outcomes.

## Table of Contents

1. [Overview](#overview)
2. [The Loop](#the-loop)
3. [Confidence Scoring](#confidence-scoring)
4. [Golden Rules](#golden-rules)
5. [API Reference](#api-reference)
6. [Integration Examples](#integration-examples)

---

## Overview

### The Problem

Not all memories are equally reliable:
- Some are just notes that may or may not be accurate
- Some are outdated and no longer apply
- Some are proven through repeated use

### The Solution

Track whether memories actually work when applied:
- **Success** → Increase confidence (+0.1)
- **Failure** → Decrease confidence (-0.15)
- **High confidence** → Promote to Golden Rule

```
Low Confidence ────────────────────────► High Confidence
    0.3            0.5         0.7            0.9+
    │              │           │               │
  Untested    Somewhat     Reliable      GOLDEN RULE
              Validated
```

---

## The Loop

The validation loop follows the ELF (Experiential Learning Framework) pattern:

```
┌─────────────────────────────────────────────┐
│                                             │
│   TRY → BREAK → ANALYZE → LEARN            │
│    │                         │              │
│    └─────────────────────────┘              │
│                                             │
└─────────────────────────────────────────────┘
```

### 1. TRY - Apply a Memory

Record that you're about to use a memory:

```python
memory_apply_tool(
    memory_id="mem_123",
    context="Using for npm vs pnpm decision",
    session_id="session_456"
)
```

This creates an "applied" validation event and updates the access timestamp.

### 2. BREAK - Attempt to Use It

Actually use the memory's guidance:

```python
# Memory says: "Use pnpm, not npm"
result = run_command("pnpm install")
```

### 3. ANALYZE - Evaluate the Outcome

Did it work?

```python
if result.exit_code == 0:
    success = True
    error_msg = None
else:
    success = False
    error_msg = result.stderr
```

### 4. LEARN - Record the Outcome

Update the memory's confidence:

```python
outcome = memory_outcome_tool(
    memory_id="mem_123",
    success=success,
    error_msg=error_msg,
    session_id="session_456"
)

print(f"New confidence: {outcome['new_confidence']}")
if outcome['promoted']:
    print("Memory promoted to Golden Rule!")
```

---

## Confidence Scoring

### Default Starting Confidence

All new memories start at **0.3** (configurable via `RECALL_DEFAULT_CONFIDENCE`).

### Confidence Adjustments

| Outcome | Adjustment | Rationale |
|---------|------------|-----------|
| Success | +0.1 | Gradual increase rewards consistent success |
| Failure | -0.15 | Asymmetric penalty - failures are more informative |

### Confidence Thresholds

| Range | Meaning | Behavior |
|-------|---------|----------|
| 0.0 - 0.15 | Very low confidence | Candidate for deletion |
| 0.15 - 0.5 | Low confidence | Use with caution |
| 0.5 - 0.7 | Moderate confidence | Generally reliable |
| 0.7 - 0.9 | High confidence | Trustworthy |
| 0.9 - 1.0 | **Golden Rule** | Always show, protected from deletion |

### Math Example

Starting confidence: 0.3

```
Apply & Succeed: 0.3 + 0.1 = 0.4
Apply & Succeed: 0.4 + 0.1 = 0.5
Apply & Succeed: 0.5 + 0.1 = 0.6
Apply & Fail:    0.6 - 0.15 = 0.45
Apply & Succeed: 0.45 + 0.1 = 0.55
...
```

After **7 consecutive successes** from default: `0.3 + (7 × 0.1) = 1.0` → Golden Rule

---

## Golden Rules

### What Are Golden Rules?

Golden Rules are memories that have been validated through repeated successful use. They represent proven knowledge that should always be surfaced.

### Automatic Promotion

When a memory's confidence reaches **0.9 or higher**, it's automatically promoted:

```python
# Original memory
{
    "type": "preference",
    "content": "Use pnpm for package management",
    "confidence": 0.3
}

# After 6+ successful applications
{
    "type": "golden_rule",
    "content": "Use pnpm for package management",
    "confidence": 0.9,
    "metadata": {
        "promoted_from": "preference"
    }
}
```

### Golden Rule Properties

1. **Always visible** - Included in context regardless of token budget
2. **Protected** - Cannot be deleted without `force=True`
3. **Eligible types** - Only `preference`, `decision`, and `pattern` can become golden rules

### Viewing Golden Rules

```python
# List all golden rules
memory_list_tool(
    memory_type="golden_rule"
)
```

---

## API Reference

### memory_apply_tool

Record that a memory is being applied.

```python
memory_apply_tool(
    memory_id: str,       # Required: Memory to apply
    context: str = None,  # Optional: Why/how it's being used
    session_id: str = None  # Optional: Session identifier
) -> dict
```

**Returns:**
```python
{
    "success": True,
    "memory_id": "mem_123",
    "access_count": 5,       # Updated access count
    "accessed_at": "2024-..."  # Updated timestamp
}
```

### memory_outcome_tool

Record the outcome of applying a memory.

```python
memory_outcome_tool(
    memory_id: str,         # Required: Memory that was applied
    success: bool,          # Required: Did it work?
    error_msg: str = None,  # Optional: Error message if failed
    session_id: str = None  # Optional: Session identifier
) -> dict
```

**Returns:**
```python
{
    "success": True,
    "memory_id": "mem_123",
    "previous_confidence": 0.5,
    "new_confidence": 0.6,  # Or 0.35 if failed
    "promoted": False,      # True if now a golden rule
    "event_id": 42          # Validation event ID
}
```

### validation_history_tool

Get validation history for a memory.

```python
validation_history_tool(
    memory_id: str,
    limit: int = 20
) -> dict
```

**Returns:**
```python
{
    "success": True,
    "memory_id": "mem_123",
    "history": [
        {
            "event_type": "applied",
            "timestamp": "2024-...",
            "context": "Using for X"
        },
        {
            "event_type": "outcome",
            "timestamp": "2024-...",
            "success": True,
            "confidence_before": 0.5,
            "confidence_after": 0.6
        }
    ],
    "total_applied": 5,
    "total_succeeded": 4,
    "total_failed": 1,
    "success_rate": 0.8
}
```

---

## Integration Examples

### Example 1: Validating a Package Manager Preference

```python
# Memory: "Use pnpm, not npm"
memory_id = "mem_pkg_manager"

# TRY: Record application
memory_apply_tool(
    memory_id=memory_id,
    context="Installing dependencies for new project"
)

# BREAK: Use the memory
result = subprocess.run(["pnpm", "install"], capture_output=True)

# ANALYZE & LEARN: Record outcome
if result.returncode == 0:
    memory_outcome_tool(memory_id=memory_id, success=True)
else:
    memory_outcome_tool(
        memory_id=memory_id,
        success=False,
        error_msg=result.stderr.decode()
    )
```

### Example 2: Validating a Code Pattern

```python
# Memory: "Always use async/await with database calls"
memory_id = "mem_async_db"

# TRY
memory_apply_tool(memory_id=memory_id, context="Writing DB query function")

# BREAK: Write the code using the pattern
code = """
async def get_user(user_id: str) -> User:
    return await db.users.find_one({"_id": user_id})
"""

# ANALYZE: Run tests
test_result = run_tests()

# LEARN
memory_outcome_tool(
    memory_id=memory_id,
    success=test_result.passed,
    error_msg=test_result.error if not test_result.passed else None
)
```

### Example 3: Automatic Validation in Hooks

The `recall-capture.py` hook can automatically validate memories by analyzing session transcripts:

```python
# In recall-capture.py (conceptual)
for memory in session_memories_used:
    # Check if the session ended successfully
    if session_had_errors_related_to(memory):
        memory_outcome_tool(memory_id=memory.id, success=False)
    else:
        memory_outcome_tool(memory_id=memory.id, success=True)
```

### Example 4: Checking Success Rate Before Using

```python
# Before applying a memory, check its track record
history = validation_history_tool(memory_id="mem_123")

if history["success_rate"] < 0.5:
    print(f"Warning: This memory has only {history['success_rate']*100}% success rate")
    print(f"Consider alternatives or investigate failures")
```

---

## Best Practices

### 1. Always Apply Before Using

Even if you don't record the outcome, `memory_apply_tool` updates access tracking:

```python
# Good: Record that you're using the memory
memory_apply_tool(memory_id=id, context="For X")
# ... use memory ...

# Even better: Record the outcome too
memory_outcome_tool(memory_id=id, success=True)
```

### 2. Be Specific About Failures

Include error messages to help with debugging:

```python
# Good
memory_outcome_tool(
    memory_id=id,
    success=False,
    error_msg="TypeError: Cannot read property 'x' of undefined"
)

# Less useful
memory_outcome_tool(memory_id=id, success=False)
```

### 3. Use Session IDs for Correlation

Session IDs help correlate multiple validation events:

```python
session_id = "session_123"

memory_apply_tool(memory_id="A", session_id=session_id)
memory_apply_tool(memory_id="B", session_id=session_id)

# Later, can query: "What memories were used in this session?"
```

### 4. Trust Golden Rules

Golden rules have been validated through repeated use:

```python
if memory.type == "golden_rule":
    # This has been proven reliable
    apply_without_hesitation(memory)
else:
    # Consider with appropriate skepticism based on confidence
    if memory.confidence > 0.7:
        apply_confidently(memory)
    else:
        apply_cautiously(memory)
```

### 5. Clean Up Low-Confidence Memories

Periodically remove memories that consistently fail:

```python
# Find candidates for deletion
for memory in memories:
    if memory.confidence < 0.15:
        print(f"Consider deleting: {memory.content[:50]}...")
        # memory_forget_tool(memory_id=memory.id)
```

---

## Troubleshooting

### Memory Not Gaining Confidence

1. **Are you calling `memory_outcome_tool`?** Just applying isn't enough.
2. **Check session_id correlation** - Outcomes should match applications.
3. **View history**: `validation_history_tool(memory_id=id)`

### Memory Stuck at Low Confidence

If a memory keeps failing:
1. Check if it's still valid/accurate
2. Consider updating or replacing it
3. Look for a contradicting memory with higher confidence

### Golden Rule Not Appearing

Golden rules require:
1. Confidence >= 0.9
2. Original type must be `preference`, `decision`, or `pattern`
3. Promotion happens automatically on outcome recording
