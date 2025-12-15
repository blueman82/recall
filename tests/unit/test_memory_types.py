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
from recall.memory.types import ExpandedMemory, GraphExpansionConfig


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


class TestExpandedMemory:
    """Tests for ExpandedMemory dataclass."""

    @pytest.fixture
    def sample_memory(self):
        """Create a sample Memory for testing."""
        return Memory(
            id="test-id",
            content="Test content",
            content_hash="abc123",
            type=MemoryType.PREFERENCE,
        )

    def test_expanded_memory_creation(self, sample_memory):
        """Should create ExpandedMemory with required fields."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=0.85,
            hop_distance=1,
        )
        assert expanded.memory == sample_memory
        assert expanded.relevance_score == 0.85
        assert expanded.hop_distance == 1

    def test_expanded_memory_default_values(self, sample_memory):
        """Should have correct default values."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=0.75,
            hop_distance=2,
        )
        assert expanded.path == []
        assert expanded.edge_weight_product == 1.0
        assert expanded.explanation == ""

    def test_expanded_memory_with_path(self, sample_memory):
        """Should accept path list of edge types."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=0.6,
            hop_distance=2,
            path=["supersedes", "relates_to"],
        )
        assert expanded.path == ["supersedes", "relates_to"]
        assert len(expanded.path) == 2

    def test_expanded_memory_with_edge_weight_product(self, sample_memory):
        """Should accept edge weight product."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=0.5,
            hop_distance=3,
            edge_weight_product=0.49,  # 0.7 * 0.7
        )
        assert expanded.edge_weight_product == 0.49

    def test_expanded_memory_with_explanation(self, sample_memory):
        """Should accept explanation string."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=0.7,
            hop_distance=1,
            explanation="1 hop via supersedes, combined weight 0.70",
        )
        assert expanded.explanation == "1 hop via supersedes, combined weight 0.70"

    def test_expanded_memory_full_construction(self, sample_memory):
        """Should handle all fields including explanation."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=0.42,
            hop_distance=2,
            path=["supersedes", "relates_to"],
            edge_weight_product=0.6,
            explanation="2 hops via supersedes → relates_to, combined weight 0.42",
        )
        assert expanded.memory.id == "test-id"
        assert expanded.relevance_score == 0.42
        assert expanded.hop_distance == 2
        assert expanded.path == ["supersedes", "relates_to"]
        assert expanded.edge_weight_product == 0.6
        assert "2 hops" in expanded.explanation

    def test_expanded_memory_zero_hop_distance(self, sample_memory):
        """Should allow hop_distance=0 for direct matches."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=1.0,
            hop_distance=0,
        )
        assert expanded.hop_distance == 0

    def test_expanded_memory_field_access(self, sample_memory):
        """Should provide direct access to underlying memory fields."""
        expanded = ExpandedMemory(
            memory=sample_memory,
            relevance_score=0.8,
            hop_distance=1,
        )
        # Access nested memory properties
        assert expanded.memory.content == "Test content"
        assert expanded.memory.type == MemoryType.PREFERENCE
        assert expanded.memory.namespace == "global"


class TestGraphExpansionConfig:
    """Tests for GraphExpansionConfig dataclass."""

    def test_graph_expansion_config_default_values(self):
        """Should have correct default values."""
        config = GraphExpansionConfig()
        assert config.max_depth == 1
        assert config.decay_factor == 0.7
        assert config.include_edge_types is None
        assert config.exclude_edge_types is None
        assert config.max_expanded == 20

    def test_graph_expansion_config_safety_guards_defaults(self):
        """Should have safety guard defaults set by __post_init__."""
        config = GraphExpansionConfig()
        assert config.max_nodes_visited == 200
        assert config.max_edges_per_node == 10

    def test_graph_expansion_config_default_edge_type_weights(self):
        """Should set default edge type weights in __post_init__."""
        config = GraphExpansionConfig()
        assert config.edge_type_weights["supersedes"] == 1.0
        assert config.edge_type_weights["caused_by"] == 0.9
        assert config.edge_type_weights["relates_to"] == 0.7
        assert config.edge_type_weights["contradicts"] == 0.5

    def test_graph_expansion_config_custom_max_depth(self):
        """Should accept custom max_depth."""
        config = GraphExpansionConfig(max_depth=3)
        assert config.max_depth == 3

    def test_graph_expansion_config_custom_decay_factor(self):
        """Should accept custom decay_factor."""
        config = GraphExpansionConfig(decay_factor=0.8)
        assert config.decay_factor == 0.8

    def test_graph_expansion_config_custom_edge_type_weights(self):
        """Should merge custom weights with defaults."""
        custom_weights = {"supersedes": 0.5, "custom_type": 0.3}
        config = GraphExpansionConfig(edge_type_weights=custom_weights)

        # Custom values should override defaults
        assert config.edge_type_weights["supersedes"] == 0.5
        assert config.edge_type_weights["custom_type"] == 0.3
        # Defaults should still be present for other types
        assert config.edge_type_weights["caused_by"] == 0.9
        assert config.edge_type_weights["relates_to"] == 0.7
        assert config.edge_type_weights["contradicts"] == 0.5

    def test_graph_expansion_config_include_edge_types(self):
        """Should accept include_edge_types set."""
        config = GraphExpansionConfig(include_edge_types={"supersedes", "relates_to"})
        assert config.include_edge_types == {"supersedes", "relates_to"}
        assert "supersedes" in config.include_edge_types
        assert "contradicts" not in config.include_edge_types

    def test_graph_expansion_config_exclude_edge_types(self):
        """Should accept exclude_edge_types set."""
        config = GraphExpansionConfig(exclude_edge_types={"contradicts"})
        assert config.exclude_edge_types == {"contradicts"}
        assert "contradicts" in config.exclude_edge_types

    def test_graph_expansion_config_custom_max_expanded(self):
        """Should accept custom max_expanded."""
        config = GraphExpansionConfig(max_expanded=50)
        assert config.max_expanded == 50

    def test_graph_expansion_config_custom_safety_guards(self):
        """Should accept custom safety guard values."""
        config = GraphExpansionConfig(
            max_nodes_visited=500,
            max_edges_per_node=25,
        )
        assert config.max_nodes_visited == 500
        assert config.max_edges_per_node == 25

    def test_graph_expansion_config_full_construction(self):
        """Should handle all configuration options."""
        config = GraphExpansionConfig(
            max_depth=2,
            decay_factor=0.8,
            edge_type_weights={"supersedes": 1.0},
            include_edge_types={"supersedes", "caused_by"},
            exclude_edge_types=None,
            max_expanded=30,
            max_nodes_visited=300,
            max_edges_per_node=15,
        )
        assert config.max_depth == 2
        assert config.decay_factor == 0.8
        assert config.edge_type_weights["supersedes"] == 1.0
        assert config.include_edge_types == {"supersedes", "caused_by"}
        assert config.exclude_edge_types is None
        assert config.max_expanded == 30
        assert config.max_nodes_visited == 300
        assert config.max_edges_per_node == 15

    def test_graph_expansion_config_edge_weights_not_mutated(self):
        """Creating config should not mutate passed dict."""
        original_weights = {"supersedes": 0.5}
        original_copy = dict(original_weights)

        config = GraphExpansionConfig(edge_type_weights=original_weights)

        # Config should have merged weights
        assert "caused_by" in config.edge_type_weights
        # Original dict should be unchanged (due to copy in merge)
        # Note: The current implementation mutates the merged dict,
        # but original is not affected because we create a new dict
        assert original_weights == original_copy or "caused_by" not in original_weights
