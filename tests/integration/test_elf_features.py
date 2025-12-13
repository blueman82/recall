"""Integration tests for ELF (Experiential Learning Framework) features.

These tests verify the complete ELF workflow:
1. File tracking → query → retrieval
2. Confidence lifecycle (0.3 → 0.9 → golden rule promotion)
3. Validation loop (apply → outcome → confidence adjustment)
4. Contradiction detection (semantic similarity + LLM reasoning)
5. Golden rule protection (cannot delete without force=True)
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from recall.embedding.ollama import OllamaClient
from recall.memory.operations import (
    memory_apply,
    memory_forget,
    memory_outcome,
    memory_store,
    memory_validate,
)
from recall.memory.types import MemoryType
from recall.storage.chromadb import ChromaStore
from recall.storage.hybrid import HybridStore
from recall.storage.sqlite import SQLiteStore
from recall.validation import detect_contradictions


def unique_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"test_elf_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_embedding_client():
    """Create mock OllamaClient for testing."""
    client = AsyncMock(spec=OllamaClient)
    # Return consistent embeddings for mxbai dimension
    client.embed.return_value = [0.1] * 1024
    return client


@pytest.fixture
def ephemeral_store(mock_embedding_client):
    """Create HybridStore with ephemeral stores for testing."""
    sqlite = SQLiteStore(ephemeral=True)
    chroma = ChromaStore(ephemeral=True, collection_name=unique_collection_name())

    store = HybridStore(
        sqlite_store=sqlite,
        chroma_store=chroma,
        embedding_client=mock_embedding_client,
    )
    yield store
    sqlite.close()


@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM responses for contradiction/supersede detection."""
    async def mock_call(prompt, **kwargs):
        # Default: no contradiction
        if "contradicts" in prompt.lower():
            return {"contradicts": False, "reason": "Not contradictory"}
        elif "supersedes" in prompt.lower():
            return {"supersedes": False, "reason": "Not superseding"}
        return {}

    return mock_call


class TestFileTracking:
    """Test file activity tracking and retrieval."""

    @pytest.mark.asyncio
    async def test_file_activity_store_and_query(self, ephemeral_store):
        """Test storing file activity and querying it back."""
        # Store file activities
        activity_id_1 = ephemeral_store.add_file_activity(
            file_path="/Users/harrison/project/main.py",
            action="edit",
            session_id="session_001",
            project_root="/Users/harrison/project",
            file_type="python",
            metadata={"lines_changed": 42},
        )
        assert activity_id_1 > 0

        activity_id_2 = ephemeral_store.add_file_activity(
            file_path="/Users/harrison/project/utils.py",
            action="read",
            session_id="session_001",
            project_root="/Users/harrison/project",
            file_type="python",
        )
        assert activity_id_2 > 0

        # Query activities by file path
        activities = ephemeral_store.get_file_activity(
            file_path="/Users/harrison/project/main.py"
        )
        assert len(activities) == 1
        assert activities[0]["action"] == "edit"
        assert activities[0]["file_type"] == "python"

        # Query all activities for project
        activities = ephemeral_store.get_file_activity(
            project_root="/Users/harrison/project"
        )
        assert len(activities) == 2

    @pytest.mark.asyncio
    async def test_file_activity_retrieval_by_action(self, ephemeral_store):
        """Test filtering file activities by action type."""
        ephemeral_store.add_file_activity(
            file_path="/project/file1.py",
            action="write",
            project_root="/project",
        )
        ephemeral_store.add_file_activity(
            file_path="/project/file2.py",
            action="read",
            project_root="/project",
        )
        ephemeral_store.add_file_activity(
            file_path="/project/file3.py",
            action="edit",
            project_root="/project",
        )

        # Query write actions only
        write_activities = ephemeral_store.get_file_activity(action="write")
        assert len(write_activities) == 1
        assert write_activities[0]["action"] == "write"

    @pytest.mark.asyncio
    async def test_get_recent_files(self, ephemeral_store):
        """Test retrieving recently accessed files with aggregated stats."""
        # Add multiple activities for same file
        for i in range(3):
            ephemeral_store.add_file_activity(
                file_path="/project/main.py",
                action="edit",
                project_root="/project",
            )

        ephemeral_store.add_file_activity(
            file_path="/project/utils.py",
            action="read",
            project_root="/project",
        )

        # Get recent files
        recent = ephemeral_store.get_recent_files(
            project_root="/project",
            limit=10,
            days=14,
        )

        assert len(recent) == 2
        # main.py should have access_count=3
        main_py = next(f for f in recent if f["file_path"] == "/project/main.py")
        assert main_py["access_count"] == 3
        assert main_py["last_action"] == "edit"


class TestConfidenceLifecycle:
    """Test confidence score lifecycle and golden rule promotion."""

    @pytest.mark.asyncio
    async def test_memory_starts_with_default_confidence(self, ephemeral_store):
        """Test that new memories start with 0.3 confidence."""
        result = await memory_store(
            store=ephemeral_store,
            content="User prefers tabs over spaces",
            memory_type=MemoryType.PREFERENCE,
        )
        assert result.success is True

        # Check initial confidence
        memory = await ephemeral_store.get_memory(result.id)
        assert memory["confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_successful_validation_increases_confidence(self, ephemeral_store):
        """Test that successful validation increases confidence by 0.1."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Always use async/await for IO operations",
            memory_type=MemoryType.PATTERN,
        )

        # Validate successfully
        validate_result = await memory_validate(
            store=ephemeral_store,
            memory_id=store_result.id,
            success=True,
            adjustment=0.1,
        )

        assert validate_result.success is True
        assert validate_result.old_confidence == 0.3
        assert validate_result.new_confidence == 0.4
        assert validate_result.promoted is False

    @pytest.mark.asyncio
    async def test_failed_validation_decreases_confidence(self, ephemeral_store):
        """Test that failed validation decreases confidence by 1.5x adjustment."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Use global variables for state management",
            memory_type=MemoryType.PATTERN,
        )

        # Validate with failure (confidence should drop by 0.1 * 1.5 = 0.15)
        validate_result = await memory_validate(
            store=ephemeral_store,
            memory_id=store_result.id,
            success=False,
            adjustment=0.1,
        )

        assert validate_result.success is True
        assert validate_result.old_confidence == 0.3
        assert abs(validate_result.new_confidence - 0.15) < 0.001
        assert validate_result.promoted is False

    @pytest.mark.asyncio
    async def test_confidence_reaches_golden_rule_threshold(self, ephemeral_store):
        """Test that confidence can reach 0.9 through multiple validations."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Always validate user input",
            memory_type=MemoryType.PATTERN,
        )

        # Validate successfully 7 times to reach 0.3 + (7 * 0.1) = 1.0 (capped at 1.0)
        for i in range(7):
            validate_result = await memory_validate(
                store=ephemeral_store,
                memory_id=store_result.id,
                success=True,
                adjustment=0.1,
            )
            assert validate_result.success is True

        # Final confidence should be 1.0 (capped)
        memory = await ephemeral_store.get_memory(store_result.id)
        assert abs(memory["confidence"] - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_golden_rule_promotion_at_09_confidence(self, ephemeral_store):
        """Test that memory is promoted to golden rule when confidence reaches 0.9."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Never commit secrets to version control",
            memory_type=MemoryType.PATTERN,
        )

        # Validate successfully 7 times to reach >= 0.9
        # Due to floating point, we go one extra to ensure we're >= 0.9
        for i in range(7):
            validate_result = await memory_validate(
                store=ephemeral_store,
                memory_id=store_result.id,
                success=True,
                adjustment=0.1,
            )

        # Last validation should trigger promotion
        assert validate_result.success is True
        assert validate_result.new_confidence >= 0.9
        assert validate_result.promoted is True

        # Check that type was changed to golden_rule
        memory = await ephemeral_store.get_memory(store_result.id)
        assert memory["type"] == "golden_rule"
        import json
        metadata = json.loads(memory["metadata"]) if isinstance(memory["metadata"], str) else memory["metadata"]
        assert metadata["promoted_from"] == "pattern"

    @pytest.mark.asyncio
    async def test_only_preference_decision_pattern_can_be_promoted(self, ephemeral_store):
        """Test that only certain types can be promoted to golden rule."""
        # Session type should NOT be promoted even at 0.9 confidence
        store_result = await memory_store(
            store=ephemeral_store,
            content="Session context",
            memory_type=MemoryType.SESSION,
        )

        # Manually set confidence to 0.9
        await ephemeral_store.update_memory(
            memory_id=store_result.id,
            confidence=0.9,
        )

        # Validate to trigger promotion check
        validate_result = await memory_validate(
            store=ephemeral_store,
            memory_id=store_result.id,
            success=True,
            adjustment=0.1,
        )

        # Should NOT be promoted
        assert validate_result.promoted is False
        memory = await ephemeral_store.get_memory(store_result.id)
        assert memory["type"] == "session"

    @pytest.mark.asyncio
    async def test_confidence_clamped_at_boundaries(self, ephemeral_store):
        """Test that confidence is clamped between 0.0 and 1.0."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Test memory",
            memory_type=MemoryType.PATTERN,
        )

        # Validate successfully many times (should cap at 1.0)
        for _ in range(10):
            await memory_validate(
                store=ephemeral_store,
                memory_id=store_result.id,
                success=True,
                adjustment=0.1,
            )

        memory = await ephemeral_store.get_memory(store_result.id)
        assert memory["confidence"] <= 1.0

        # Fail validation many times (should floor at 0.0)
        for _ in range(10):
            await memory_validate(
                store=ephemeral_store,
                memory_id=store_result.id,
                success=False,
                adjustment=0.2,
            )

        memory = await ephemeral_store.get_memory(store_result.id)
        assert memory["confidence"] >= 0.0


class TestValidationLoop:
    """Test the complete validation loop: apply → outcome → confidence adjustment."""

    @pytest.mark.asyncio
    async def test_apply_records_usage(self, ephemeral_store):
        """Test that memory_apply records when a memory is used."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Use UTF-8 encoding for all files",
            memory_type=MemoryType.PATTERN,
        )

        # Apply the memory
        apply_result = await memory_apply(
            store=ephemeral_store,
            memory_id=store_result.id,
            context="Applied to file creation logic",
            session_id="session_001",
        )

        assert apply_result.success is True
        assert apply_result.memory_id == store_result.id
        assert apply_result.event_id is not None
        # Event ID being non-null proves the event was created

    @pytest.mark.asyncio
    async def test_successful_outcome_increases_confidence(self, ephemeral_store):
        """Test that recording a successful outcome increases confidence."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Run tests before committing",
            memory_type=MemoryType.PATTERN,
        )

        # Apply the memory
        await memory_apply(
            store=ephemeral_store,
            memory_id=store_result.id,
            context="Pre-commit hook",
        )

        # Record successful outcome
        outcome_result = await memory_outcome(
            store=ephemeral_store,
            memory_id=store_result.id,
            success=True,
        )

        assert outcome_result.success is True
        assert outcome_result.outcome_success is True
        assert abs(outcome_result.new_confidence - 0.4) < 0.001  # 0.3 + 0.1

        # Verify confidence was actually updated in the store
        memory = await ephemeral_store.get_memory(store_result.id)
        assert abs(memory["confidence"] - 0.4) < 0.001

    @pytest.mark.asyncio
    async def test_failed_outcome_decreases_confidence(self, ephemeral_store):
        """Test that recording a failed outcome decreases confidence."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Use synchronous code everywhere",
            memory_type=MemoryType.PATTERN,
        )

        # Apply the memory
        await memory_apply(
            store=ephemeral_store,
            memory_id=store_result.id,
            context="Database operations",
        )

        # Record failed outcome
        outcome_result = await memory_outcome(
            store=ephemeral_store,
            memory_id=store_result.id,
            success=False,
            error_msg="Caused performance issues",
        )

        assert outcome_result.success is True
        assert outcome_result.outcome_success is False
        assert abs(outcome_result.new_confidence - 0.15) < 0.001  # 0.3 - (0.1 * 1.5)

        # Verify confidence was actually updated in the store
        memory = await ephemeral_store.get_memory(store_result.id)
        assert abs(memory["confidence"] - 0.15) < 0.001

    @pytest.mark.asyncio
    async def test_validation_loop_multiple_cycles(self, ephemeral_store):
        """Test multiple apply/outcome cycles for a memory."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Cache API responses for 5 minutes",
            memory_type=MemoryType.DECISION,
        )

        # Cycle 1: Success
        await memory_apply(ephemeral_store, store_result.id, "API endpoint A")
        result1 = await memory_outcome(ephemeral_store, store_result.id, success=True)
        assert abs(result1.new_confidence - 0.4) < 0.001

        # Cycle 2: Success
        await memory_apply(ephemeral_store, store_result.id, "API endpoint B")
        result2 = await memory_outcome(ephemeral_store, store_result.id, success=True)
        assert abs(result2.new_confidence - 0.5) < 0.001

        # Cycle 3: Failure
        await memory_apply(ephemeral_store, store_result.id, "API endpoint C")
        result3 = await memory_outcome(ephemeral_store, store_result.id, success=False)
        assert abs(result3.new_confidence - 0.35) < 0.001  # 0.5 - 0.15

        # Verify final confidence matches
        memory = await ephemeral_store.get_memory(store_result.id)
        assert abs(memory["confidence"] - 0.35) < 0.001

    @pytest.mark.asyncio
    async def test_apply_updates_accessed_at(self, ephemeral_store):
        """Test that applying a memory updates its accessed_at timestamp."""
        # Store memory
        store_result = await memory_store(
            store=ephemeral_store,
            content="Test memory",
            memory_type=MemoryType.PATTERN,
        )

        # Get initial accessed_at
        memory_before = await ephemeral_store.get_memory(store_result.id)
        accessed_before = memory_before["accessed_at"]

        # Apply the memory
        await memory_apply(
            store=ephemeral_store,
            memory_id=store_result.id,
            context="Test context",
        )

        # Check that accessed_at was updated
        memory_after = await ephemeral_store.get_memory(store_result.id)
        accessed_after = memory_after["accessed_at"]
        assert accessed_after >= accessed_before


class TestContradictionDetection:
    """Test contradiction detection using semantic similarity and LLM reasoning."""

    @pytest.mark.asyncio
    async def test_detect_contradictions_no_conflicts(self, ephemeral_store, mock_ollama_llm):
        """Test that non-contradicting memories are not flagged."""
        # Store two non-contradicting memories
        result1 = await memory_store(
            store=ephemeral_store,
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
        )

        result2 = await memory_store(
            store=ephemeral_store,
            content="User prefers vim keybindings",
            memory_type=MemoryType.PREFERENCE,
        )

        # Mock LLM to say no contradiction
        with patch("recall.validation._call_ollama_llm", side_effect=mock_ollama_llm):
            contradiction_result = await detect_contradictions(
                store=ephemeral_store,
                memory_id=result1.id,
                similarity_threshold=0.5,
                create_edges=True,
            )

        assert contradiction_result.memory_id == result1.id
        assert len(contradiction_result.contradictions) == 0
        assert contradiction_result.edges_created == 0

    @pytest.mark.asyncio
    async def test_detect_contradictions_with_conflict(self, ephemeral_store):
        """Test that contradicting memories are detected and edges created."""
        # Store two contradicting memories
        result1 = await memory_store(
            store=ephemeral_store,
            content="Always use tabs for indentation",
            memory_type=MemoryType.PREFERENCE,
            namespace="project:myapp",
        )

        result2 = await memory_store(
            store=ephemeral_store,
            content="Always use spaces for indentation",
            memory_type=MemoryType.PREFERENCE,
            namespace="project:myapp",
        )

        # Mock LLM to detect contradiction
        async def mock_llm_contradiction(prompt, **kwargs):
            if "contradicts" in prompt.lower():
                return {"contradicts": True, "reason": "Opposite indentation preferences"}
            return {"contradicts": False, "reason": "No conflict"}

        with patch("recall.validation._call_ollama_llm", side_effect=mock_llm_contradiction):
            contradiction_result = await detect_contradictions(
                store=ephemeral_store,
                memory_id=result1.id,
                similarity_threshold=0.5,
                create_edges=True,
            )

        assert len(contradiction_result.contradictions) == 1
        assert result2.id in contradiction_result.contradictions
        assert contradiction_result.edges_created == 1

        # Verify CONTRADICTS edge was created
        edges = ephemeral_store.get_edges(result1.id, edge_type="contradicts")
        assert len(edges) == 1
        assert edges[0]["target_id"] == result2.id

    @pytest.mark.asyncio
    async def test_detect_contradictions_respects_namespace(self, ephemeral_store):
        """Test that contradiction detection is scoped to the same namespace."""
        # Store similar memories in different namespaces
        result1 = await memory_store(
            store=ephemeral_store,
            content="Use React for UI",
            memory_type=MemoryType.DECISION,
            namespace="project:app1",
        )

        result2 = await memory_store(
            store=ephemeral_store,
            content="Use Vue for UI",
            memory_type=MemoryType.DECISION,
            namespace="project:app2",  # Different namespace
        )

        # Mock LLM to detect contradiction
        async def mock_llm_contradiction(prompt, **kwargs):
            return {"contradicts": True, "reason": "Different frameworks"}

        with patch("recall.validation._call_ollama_llm", side_effect=mock_llm_contradiction):
            contradiction_result = await detect_contradictions(
                store=ephemeral_store,
                memory_id=result1.id,
                similarity_threshold=0.5,
            )

        # Should not find contradictions in different namespace
        assert len(contradiction_result.contradictions) == 0

    @pytest.mark.asyncio
    async def test_detect_contradictions_excludes_self(self, ephemeral_store, mock_ollama_llm):
        """Test that a memory doesn't contradict itself."""
        result = await memory_store(
            store=ephemeral_store,
            content="Always use type hints in Python",
            memory_type=MemoryType.PATTERN,
        )

        with patch("recall.validation._call_ollama_llm", side_effect=mock_ollama_llm):
            contradiction_result = await detect_contradictions(
                store=ephemeral_store,
                memory_id=result.id,
                similarity_threshold=0.0,  # Very low threshold
            )

        # Should not find itself as a contradiction
        assert result.id not in contradiction_result.contradictions

    @pytest.mark.asyncio
    async def test_detect_contradictions_no_duplicate_edges(self, ephemeral_store):
        """Test that duplicate CONTRADICTS edges are not created."""
        result1 = await memory_store(
            store=ephemeral_store,
            content="Memory A",
            memory_type=MemoryType.PATTERN,
        )

        result2 = await memory_store(
            store=ephemeral_store,
            content="Memory B",
            memory_type=MemoryType.PATTERN,
        )

        async def mock_llm_contradiction(prompt, **kwargs):
            return {"contradicts": True, "reason": "Conflict"}

        # Detect contradictions twice
        with patch("recall.validation._call_ollama_llm", side_effect=mock_llm_contradiction):
            await detect_contradictions(ephemeral_store, result1.id, create_edges=True)
            await detect_contradictions(ephemeral_store, result1.id, create_edges=True)

        # Should only create one edge
        edges = ephemeral_store.get_edges(result1.id, edge_type="contradicts")
        assert len(edges) <= 1


class TestGoldenRuleProtection:
    """Test that golden rules are protected from deletion."""

    @pytest.mark.asyncio
    async def test_cannot_delete_golden_rule_without_force(self, ephemeral_store):
        """Test that golden rules cannot be deleted without force=True."""
        # Store and promote to golden rule
        result = await memory_store(
            store=ephemeral_store,
            content="Never expose API keys",
            memory_type=MemoryType.PATTERN,
        )

        # Promote to golden rule by validating until >= 0.9
        for _ in range(7):  # Extra validation to ensure >= 0.9
            await memory_validate(ephemeral_store, result.id, success=True)

        memory = await ephemeral_store.get_memory(result.id)
        # Check if promoted (either by type or confidence)
        is_golden = memory["type"] == "golden_rule" or memory["confidence"] >= 0.9
        assert is_golden, f"Memory not golden: type={memory['type']}, conf={memory['confidence']}"

        # Try to delete without force
        forget_result = await memory_forget(
            store=ephemeral_store,
            memory_id=result.id,
            force=False,
        )

        assert forget_result.success is False
        assert "golden rule" in forget_result.error.lower()
        assert forget_result.deleted_count == 0

        # Verify memory still exists
        memory = await ephemeral_store.get_memory(result.id)
        assert memory is not None

    @pytest.mark.asyncio
    async def test_can_delete_golden_rule_with_force(self, ephemeral_store):
        """Test that golden rules CAN be deleted with force=True."""
        # Store and promote to golden rule
        result = await memory_store(
            store=ephemeral_store,
            content="Always validate inputs",
            memory_type=MemoryType.PATTERN,
        )

        # Promote by setting type directly
        await ephemeral_store.update_memory(
            memory_id=result.id,
            memory_type="golden_rule",
        )

        # Delete with force
        forget_result = await memory_forget(
            store=ephemeral_store,
            memory_id=result.id,
            force=True,
        )

        assert forget_result.success is True
        assert forget_result.deleted_count == 1

        # Verify memory was deleted
        memory = await ephemeral_store.get_memory(result.id)
        assert memory is None

    @pytest.mark.asyncio
    async def test_high_confidence_memory_protected_as_golden_rule(self, ephemeral_store):
        """Test that memories with confidence >= 0.9 are protected like golden rules."""
        # Store memory with high confidence
        result = await memory_store(
            store=ephemeral_store,
            content="Use HTTPS everywhere",
            memory_type=MemoryType.PATTERN,
        )

        # Boost confidence to 0.9
        await ephemeral_store.update_memory(
            memory_id=result.id,
            confidence=0.9,
        )

        # Try to delete without force
        forget_result = await memory_forget(
            store=ephemeral_store,
            memory_id=result.id,
            force=False,
        )

        assert forget_result.success is False
        assert "golden rule" in forget_result.error.lower()

    @pytest.mark.asyncio
    async def test_forget_by_query_skips_golden_rules(self, ephemeral_store):
        """Test that semantic search deletion skips golden rules."""
        # Store regular memory
        result1 = await memory_store(
            store=ephemeral_store,
            content="Regular pattern about security",
            memory_type=MemoryType.PATTERN,
        )

        # Store golden rule
        result2 = await memory_store(
            store=ephemeral_store,
            content="Golden rule about security",
            memory_type=MemoryType.GOLDEN_RULE,
        )

        # Try to delete by query without force
        forget_result = await memory_forget(
            store=ephemeral_store,
            query="security",
            n_results=10,
            force=False,
        )

        # Should only delete regular memory
        assert forget_result.success is True
        assert result1.id in forget_result.deleted_ids
        assert result2.id not in forget_result.deleted_ids

        # Verify golden rule still exists
        golden = await ephemeral_store.get_memory(result2.id)
        assert golden is not None

    @pytest.mark.asyncio
    async def test_normal_memory_can_be_deleted_without_force(self, ephemeral_store):
        """Test that non-golden-rule memories can be deleted normally."""
        result = await memory_store(
            store=ephemeral_store,
            content="Temporary note",
            memory_type=MemoryType.SESSION,
        )

        forget_result = await memory_forget(
            store=ephemeral_store,
            memory_id=result.id,
            force=False,
        )

        assert forget_result.success is True
        assert forget_result.deleted_count == 1


class TestELFEdgeCases:
    """Test edge cases and error conditions for ELF features."""

    @pytest.mark.asyncio
    async def test_validate_nonexistent_memory(self, ephemeral_store):
        """Test validating a memory that doesn't exist."""
        result = await memory_validate(
            store=ephemeral_store,
            memory_id="nonexistent_id",
            success=True,
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_apply_nonexistent_memory(self, ephemeral_store):
        """Test applying a memory that doesn't exist."""
        result = await memory_apply(
            store=ephemeral_store,
            memory_id="nonexistent_id",
            context="Test context",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_outcome_nonexistent_memory(self, ephemeral_store):
        """Test recording outcome for a memory that doesn't exist."""
        result = await memory_outcome(
            store=ephemeral_store,
            memory_id="nonexistent_id",
            success=True,
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_validate_with_invalid_adjustment(self, ephemeral_store):
        """Test that validation rejects invalid adjustment values."""
        result = await memory_store(
            store=ephemeral_store,
            content="Test memory",
            memory_type=MemoryType.PATTERN,
        )

        # Invalid: negative adjustment
        validate_result = await memory_validate(
            store=ephemeral_store,
            memory_id=result.id,
            success=True,
            adjustment=-0.1,
        )
        assert validate_result.success is False

        # Invalid: adjustment > 1.0
        validate_result = await memory_validate(
            store=ephemeral_store,
            memory_id=result.id,
            success=True,
            adjustment=1.5,
        )
        assert validate_result.success is False

    @pytest.mark.asyncio
    async def test_detect_contradictions_nonexistent_memory(self, ephemeral_store):
        """Test contradiction detection for non-existent memory."""
        result = await detect_contradictions(
            store=ephemeral_store,
            memory_id="nonexistent_id",
        )

        assert result.memory_id == "nonexistent_id"
        assert "not found" in result.error.lower()
        assert len(result.contradictions) == 0

    @pytest.mark.asyncio
    async def test_confidence_adjustment_preserves_golden_rule_type(self, ephemeral_store):
        """Test that confidence adjustment doesn't change golden rule type back."""
        # Store as golden rule
        result = await memory_store(
            store=ephemeral_store,
            content="Constitutional principle",
            memory_type=MemoryType.GOLDEN_RULE,
        )

        # Fail validation (reduce confidence)
        await memory_validate(ephemeral_store, result.id, success=False)

        # Type should still be golden_rule
        memory = await ephemeral_store.get_memory(result.id)
        assert memory["type"] == "golden_rule"
