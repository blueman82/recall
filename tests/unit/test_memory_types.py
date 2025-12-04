"""Unit tests for memory types module."""

from datetime import datetime

import pytest

from recall.memory import (
    Edge,
    Memory,
    MemoryType,
    RecallResult,
    RelationType,
    StoreResult,
    validate_namespace,
)


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_type_values(self):
        """MemoryType should have correct string values."""
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.PATTERN.value == "pattern"
        assert MemoryType.SESSION.value == "session"

    def test_memory_type_from_string(self):
        """Should be able to create MemoryType from string value."""
        assert MemoryType("preference") == MemoryType.PREFERENCE
        assert MemoryType("decision") == MemoryType.DECISION
        assert MemoryType("pattern") == MemoryType.PATTERN
        assert MemoryType("session") == MemoryType.SESSION

    def test_memory_type_invalid_value_raises(self):
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError):
            MemoryType("invalid")


class TestRelationType:
    """Tests for RelationType enum."""

    def test_relation_type_values(self):
        """RelationType should have correct string values."""
        assert RelationType.RELATES_TO.value == "relates_to"
        assert RelationType.SUPERSEDES.value == "supersedes"
        assert RelationType.CAUSED_BY.value == "caused_by"
        assert RelationType.CONTRADICTS.value == "contradicts"

    def test_relation_type_from_string(self):
        """Should be able to create RelationType from string value."""
        assert RelationType("relates_to") == RelationType.RELATES_TO
        assert RelationType("supersedes") == RelationType.SUPERSEDES
        assert RelationType("caused_by") == RelationType.CAUSED_BY
        assert RelationType("contradicts") == RelationType.CONTRADICTS


class TestValidateNamespace:
    """Tests for namespace validation."""

    def test_global_namespace_is_valid(self):
        """'global' should be a valid namespace."""
        assert validate_namespace("global") is True

    def test_project_namespace_is_valid(self):
        """'project:{name}' should be valid."""
        assert validate_namespace("project:myproject") is True
        assert validate_namespace("project:my-project") is True
        assert validate_namespace("project:my_project") is True
        assert validate_namespace("project:Project123") is True

    def test_empty_namespace_is_invalid(self):
        """Empty string should be invalid."""
        assert validate_namespace("") is False

    def test_invalid_namespace_formats(self):
        """Various invalid formats should fail validation."""
        assert validate_namespace("local") is False
        assert validate_namespace("project:") is False
        assert validate_namespace("project") is False
        assert validate_namespace("project:my project") is False  # no spaces
        assert validate_namespace("project:my/project") is False  # no slashes
        assert validate_namespace("PROJECT:test") is False  # case sensitive


class TestMemory:
    """Tests for Memory dataclass."""

    def test_memory_creation_with_required_fields(self):
        """Should create Memory with all required fields."""
        memory = Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.PREFERENCE,
        )
        assert memory.id == "test-id"
        assert memory.content == "Test content"
        assert memory.content_hash == "abc123"
        assert memory.type == MemoryType.PREFERENCE

    def test_memory_default_values(self):
        """Should have correct default values."""
        memory = Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.DECISION,
        )
        assert memory.namespace == "global"
        assert memory.importance == 0.5
        assert memory.access_count == 0
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.accessed_at, datetime)

    def test_memory_with_custom_namespace(self):
        """Should accept valid custom namespace."""
        memory = Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.DECISION,
            namespace="project:myapp",
        )
        assert memory.namespace == "project:myapp"

    def test_memory_with_custom_importance(self):
        """Should accept custom importance value."""
        memory = Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.PREFERENCE,
            importance=0.9,
        )
        assert memory.importance == 0.9

    def test_memory_invalid_namespace_raises(self):
        """Invalid namespace should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid namespace"):
            Memory(
                id="test-id",
                content="Test content",
                content_hash="abc123",
                type=MemoryType.PATTERN,
                namespace="invalid",
            )

    def test_memory_importance_below_zero_raises(self):
        """Importance below 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Importance must be between"):
            Memory(
                id="test-id",
                content="Test content",
                content_hash="abc123",
                type=MemoryType.SESSION,
                importance=-0.1,
            )

    def test_memory_importance_above_one_raises(self):
        """Importance above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Importance must be between"):
            Memory(
                id="test-id",
                content="Test content",
                content_hash="abc123",
                type=MemoryType.PATTERN,
                importance=1.1,
            )

    def test_memory_importance_boundary_values(self):
        """Boundary values 0.0 and 1.0 should be valid."""
        memory_low = Memory(
            id="test-id-1",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.PREFERENCE,
            importance=0.0,
        )
        assert memory_low.importance == 0.0

        memory_high = Memory(
            id="test-id-2",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.DECISION,
            importance=1.0,
        )
        assert memory_high.importance == 1.0

    def test_memory_with_all_memory_types(self):
        """Should work with all MemoryType values."""
        for memory_type in MemoryType:
            memory = Memory(
                id=f"test-{memory_type.value}",
                content="Test content",
                content_hash="abc123",
                type=memory_type,
            )
            assert memory.type == memory_type


class TestEdge:
    """Tests for Edge dataclass."""

    def test_edge_creation(self):
        """Should create Edge with required fields."""
        edge = Edge(
            source_id="memory-1",
            target_id="memory-2",
            relation=RelationType.RELATES_TO,
        )
        assert edge.source_id == "memory-1"
        assert edge.target_id == "memory-2"
        assert edge.relation == RelationType.RELATES_TO

    def test_edge_default_values(self):
        """Should have correct default values."""
        edge = Edge(
            source_id="memory-1",
            target_id="memory-2",
            relation=RelationType.SUPERSEDES,
        )
        assert edge.weight == 1.0
        assert isinstance(edge.created_at, datetime)

    def test_edge_with_custom_weight(self):
        """Should accept custom weight value."""
        edge = Edge(
            source_id="memory-1",
            target_id="memory-2",
            relation=RelationType.CAUSED_BY,
            weight=0.5,
        )
        assert edge.weight == 0.5

    def test_edge_with_all_relation_types(self):
        """Should work with all RelationType values."""
        for relation_type in RelationType:
            edge = Edge(
                source_id="memory-1",
                target_id="memory-2",
                relation=relation_type,
            )
            assert edge.relation == relation_type


class TestStoreResult:
    """Tests for StoreResult dataclass."""

    def test_store_result_success(self):
        """Successful store result should have id."""
        result = StoreResult(success=True, id="stored-id")
        assert result.success is True
        assert result.id == "stored-id"
        assert result.error is None

    def test_store_result_failure(self):
        """Failed store result should have error."""
        result = StoreResult(success=False, error="Storage failed")
        assert result.success is False
        assert result.id is None
        assert result.error == "Storage failed"

    def test_store_result_default_values(self):
        """Should have None defaults for optional fields."""
        result = StoreResult(success=True)
        assert result.id is None
        assert result.error is None


class TestRecallResult:
    """Tests for RecallResult dataclass."""

    def test_recall_result_empty(self):
        """Empty recall result should have defaults."""
        result = RecallResult()
        assert result.memories == []
        assert result.total == 0
        assert result.score is None

    def test_recall_result_with_memories(self):
        """Should hold list of Memory objects."""
        memory = Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.PREFERENCE,
        )
        result = RecallResult(memories=[memory], total=1)
        assert len(result.memories) == 1
        assert result.memories[0] == memory
        assert result.total == 1

    def test_recall_result_with_score(self):
        """Should hold score for search results."""
        memory = Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.DECISION,
        )
        result = RecallResult(memories=[memory], total=1, score=0.95)
        assert result.score == 0.95

    def test_recall_result_total_can_exceed_memories_count(self):
        """Total can be larger than memories list (for pagination)."""
        memory = Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.PATTERN,
        )
        result = RecallResult(memories=[memory], total=100, score=0.8)
        assert len(result.memories) == 1
        assert result.total == 100
        assert result.score == 0.8
