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


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    @pytest.fixture
    def sample_memory(self):
        """Create a sample Memory for testing."""
        return Memory(
            id="test-node-id",
            content="Test node content for graph visualization",
            content_hash="abc123",
            type=MemoryType.PREFERENCE,
            importance=0.7,
            confidence=0.8,
        )

    def test_graph_node_creation(self):
        """Should create GraphNode with required fields."""
        from recall.memory.types import GraphNode

        node = GraphNode(
            id="node-1",
            content_preview="Short content",
            memory_type="preference",
            confidence=0.8,
            importance=0.7,
        )
        assert node.id == "node-1"
        assert node.content_preview == "Short content"
        assert node.memory_type == "preference"
        assert node.confidence == 0.8
        assert node.importance == 0.7

    def test_graph_node_from_memory(self, sample_memory):
        """Should create GraphNode from Memory object."""
        from recall.memory.types import GraphNode

        node = GraphNode.from_memory(sample_memory)
        assert node.id == "test-node-id"
        assert node.content_preview == "Test node content for graph visualization"
        assert node.memory_type == "preference"
        assert node.confidence == 0.8
        assert node.importance == 0.7

    def test_graph_node_content_truncation(self):
        """Content longer than 100 chars should be truncated with ellipsis."""
        from recall.memory.types import GraphNode

        # Create memory with 150-char content
        long_content = "A" * 150
        memory = Memory(
            id="long-content-id",
            content=long_content,
            content_hash="longcontent123",
            type=MemoryType.DECISION,
            importance=0.5,
            confidence=0.3,
        )

        node = GraphNode.from_memory(memory)
        assert len(node.content_preview) == 100  # 97 chars + "..."
        assert node.content_preview.endswith("...")
        assert node.content_preview[:97] == long_content[:97]

    def test_graph_node_content_exactly_100(self):
        """Content exactly 100 chars should not be truncated."""
        from recall.memory.types import GraphNode

        exact_content = "B" * 100
        memory = Memory(
            id="exact-id",
            content=exact_content,
            content_hash="exact123",
            type=MemoryType.PATTERN,
            importance=0.6,
            confidence=0.5,
        )

        node = GraphNode.from_memory(memory)
        assert node.content_preview == exact_content
        assert len(node.content_preview) == 100

    def test_graph_node_short_content_unchanged(self, sample_memory):
        """Content shorter than 100 chars should remain unchanged."""
        from recall.memory.types import GraphNode

        node = GraphNode.from_memory(sample_memory)
        assert node.content_preview == sample_memory.content


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_graph_edge_creation(self):
        """Should create GraphEdge with required fields."""
        from recall.memory.types import GraphEdge

        edge = GraphEdge(
            id=1,
            source_id="node-a",
            target_id="node-b",
            edge_type="relates_to",
            weight=0.8,
        )
        assert edge.id == 1
        assert edge.source_id == "node-a"
        assert edge.target_id == "node-b"
        assert edge.edge_type == "relates_to"
        assert edge.weight == 0.8

    def test_graph_edge_from_edge(self):
        """Should create GraphEdge from Edge object."""
        from recall.memory.types import GraphEdge

        edge = Edge(
            source_id="mem-1",
            target_id="mem-2",
            relation=RelationType.SUPERSEDES,
            weight=0.9,
        )

        graph_edge = GraphEdge.from_edge(edge, edge_id=42)
        assert graph_edge.id == 42
        assert graph_edge.source_id == "mem-1"
        assert graph_edge.target_id == "mem-2"
        assert graph_edge.edge_type == "supersedes"
        assert graph_edge.weight == 0.9

    def test_graph_edge_from_edge_all_relation_types(self):
        """Should work with all RelationType values."""
        from recall.memory.types import GraphEdge

        for idx, rel_type in enumerate(RelationType):
            edge = Edge(
                source_id="src",
                target_id="tgt",
                relation=rel_type,
            )
            graph_edge = GraphEdge.from_edge(edge, edge_id=idx)
            assert graph_edge.edge_type == rel_type.value


class TestGraphPath:
    """Tests for GraphPath dataclass."""

    def test_graph_path_creation(self):
        """Should create GraphPath with required fields."""
        from recall.memory.types import GraphPath

        path = GraphPath(
            node_ids=["node-a", "node-b", "node-c"],
            edge_types=["relates_to", "supersedes"],
            total_weight=0.72,
            relevance_score=0.65,
        )
        assert path.node_ids == ["node-a", "node-b", "node-c"]
        assert path.edge_types == ["relates_to", "supersedes"]
        assert path.total_weight == 0.72
        assert path.relevance_score == 0.65

    def test_graph_path_empty_lists(self):
        """Should handle empty lists for origin-only paths."""
        from recall.memory.types import GraphPath

        path = GraphPath(
            node_ids=["origin"],
            edge_types=[],
            total_weight=1.0,
            relevance_score=1.0,
        )
        assert len(path.node_ids) == 1
        assert len(path.edge_types) == 0

    def test_graph_path_single_hop(self):
        """Should represent single hop correctly."""
        from recall.memory.types import GraphPath

        path = GraphPath(
            node_ids=["origin", "target"],
            edge_types=["caused_by"],
            total_weight=1.0,
            relevance_score=0.7,
        )
        assert len(path.node_ids) == 2
        assert len(path.edge_types) == 1


class TestGraphStats:
    """Tests for GraphStats dataclass."""

    def test_graph_stats_creation(self):
        """Should create GraphStats with required fields."""
        from recall.memory.types import GraphStats

        stats = GraphStats(
            node_count=10,
            edge_count=15,
            max_depth_reached=2,
            origin_id="mem-origin",
        )
        assert stats.node_count == 10
        assert stats.edge_count == 15
        assert stats.max_depth_reached == 2
        assert stats.origin_id == "mem-origin"

    def test_graph_stats_zero_values(self):
        """Should handle zero values for empty graph."""
        from recall.memory.types import GraphStats

        stats = GraphStats(
            node_count=1,
            edge_count=0,
            max_depth_reached=0,
            origin_id="lonely-node",
        )
        assert stats.node_count == 1
        assert stats.edge_count == 0
        assert stats.max_depth_reached == 0


class TestGraphInspectionResult:
    """Tests for GraphInspectionResult dataclass."""

    def test_graph_inspection_result_success(self):
        """Should create successful GraphInspectionResult."""
        from recall.memory.types import GraphInspectionResult, GraphStats

        stats = GraphStats(
            node_count=3,
            edge_count=2,
            max_depth_reached=1,
            origin_id="origin-id",
        )
        result = GraphInspectionResult(
            success=True,
            origin_id="origin-id",
            nodes=[],
            edges=[],
            paths=[],
            stats=stats,
        )
        assert result.success is True
        assert result.origin_id == "origin-id"
        assert result.error is None

    def test_graph_inspection_result_failure(self):
        """Should create failed GraphInspectionResult with error."""
        from recall.memory.types import GraphInspectionResult

        result = GraphInspectionResult(
            success=False,
            error="Memory not found",
        )
        assert result.success is False
        assert result.error == "Memory not found"

    def test_graph_inspection_result_default_values(self):
        """Should have correct default values."""
        from recall.memory.types import GraphInspectionResult

        result = GraphInspectionResult(success=True)
        assert result.origin_id == ""
        assert result.nodes == []
        assert result.edges == []
        assert result.paths == []
        assert result.stats is None
        assert result.error is None

    def test_graph_inspection_result_to_mermaid_empty(self):
        """to_mermaid should handle empty nodes."""
        from recall.memory.types import GraphInspectionResult

        result = GraphInspectionResult(success=True, nodes=[])
        mermaid = result.to_mermaid()
        assert "flowchart TD" in mermaid
        assert "No nodes found" in mermaid

    def test_graph_inspection_result_to_mermaid_basic(self):
        """to_mermaid should generate valid Mermaid syntax."""
        from recall.memory.types import GraphInspectionResult, GraphNode, GraphEdge

        node1 = GraphNode(
            id="node-1",
            content_preview="First node content",
            memory_type="preference",
            confidence=0.8,
            importance=0.7,
        )
        node2 = GraphNode(
            id="node-2",
            content_preview="Second node content",
            memory_type="decision",
            confidence=0.6,
            importance=0.5,
        )
        edge = GraphEdge(
            id=1,
            source_id="node-1",
            target_id="node-2",
            edge_type="relates_to",
            weight=1.0,
        )

        result = GraphInspectionResult(
            success=True,
            origin_id="node-1",
            nodes=[node1, node2],
            edges=[edge],
        )

        mermaid = result.to_mermaid()

        # Verify Mermaid structure
        assert mermaid.startswith("flowchart TD")
        assert 'node-1["First node content"]' in mermaid
        assert 'node-2["Second node content"]' in mermaid
        assert "node-1 -->|relates_to| node-2" in mermaid

    def test_graph_inspection_result_to_mermaid_escapes_quotes(self):
        """to_mermaid should escape quotes in content."""
        from recall.memory.types import GraphInspectionResult, GraphNode

        node = GraphNode(
            id="quote-node",
            content_preview='Content with "quotes" inside',
            memory_type="pattern",
            confidence=0.5,
            importance=0.5,
        )

        result = GraphInspectionResult(
            success=True,
            origin_id="quote-node",
            nodes=[node],
        )

        mermaid = result.to_mermaid()
        # Quotes should be replaced with single quotes
        assert '"quotes"' not in mermaid
        assert "'quotes'" in mermaid

    def test_graph_inspection_result_to_mermaid_truncates_long_labels(self):
        """to_mermaid should truncate labels longer than 50 chars."""
        from recall.memory.types import GraphInspectionResult, GraphNode

        # Content is 60 chars
        long_content = "A" * 60
        node = GraphNode(
            id="long-node",
            content_preview=long_content,
            memory_type="session",
            confidence=0.4,
            importance=0.3,
        )

        result = GraphInspectionResult(
            success=True,
            origin_id="long-node",
            nodes=[node],
        )

        mermaid = result.to_mermaid()
        # Should have truncated to 47 chars + "..."
        assert "A" * 47 + "..." in mermaid
