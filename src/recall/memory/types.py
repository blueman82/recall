"""Core data types for the memory system.

This module defines the data structures used throughout the recall system:
- Memory: Main dataclass for storing memory entries
- MemoryType: Enum for categorizing memories
- RelationType: Enum for defining relationships between memories
- Edge: Dataclass for representing relationships between memories
- StoreResult: Result of a memory store operation
- RecallResult: Result of a memory recall/query operation
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MemoryType(Enum):
    """Types of memories that can be stored.

    Categorizes memories by their semantic purpose:
    - PREFERENCE: User preferences or settings
    - DECISION: Design or implementation decisions
    - PATTERN: Recognized patterns or recurring behaviors
    - SESSION: Session-related information or conversation context
    - FILE_CONTEXT: File activity tracking (what files were touched)
    - GOLDEN_RULE: High-confidence memories that are constitutional principles
    """
    PREFERENCE = "preference"
    DECISION = "decision"
    PATTERN = "pattern"
    SESSION = "session"
    FILE_CONTEXT = "file_context"
    GOLDEN_RULE = "golden_rule"


class RelationType(Enum):
    """Types of relationships between memories.

    Defines how memories can be linked together:
    - RELATES_TO: General relationship between memories
    - SUPERSEDES: One memory replaces another
    - CAUSED_BY: One memory was caused by another
    - CONTRADICTS: Memories contain conflicting information
    """
    RELATES_TO = "relates_to"
    SUPERSEDES = "supersedes"
    CAUSED_BY = "caused_by"
    CONTRADICTS = "contradicts"


# Pattern for validating project namespaces
_PROJECT_NAMESPACE_PATTERN = re.compile(r"^project:[a-zA-Z0-9_-]+$")


def validate_namespace(namespace: str) -> bool:
    """Validate that a namespace is either 'global' or matches 'project:{name}'.

    Args:
        namespace: The namespace string to validate

    Returns:
        True if valid, False otherwise
    """
    if namespace == "global":
        return True
    return bool(_PROJECT_NAMESPACE_PATTERN.match(namespace))


@dataclass
class Memory:
    """A memory entry in the recall system.

    Represents a single piece of stored information with metadata
    for categorization, organization, and retrieval tracking.

    Attributes:
        id: Unique identifier for the memory
        content: The actual content/text of the memory
        content_hash: Hash of content for deduplication
        type: Category of the memory (MemoryType enum)
        namespace: Scope of the memory ('global' or 'project:{name}')
        importance: Importance score from 0.0 to 1.0
        confidence: Confidence score from 0.0 to 1.0 (validated through usage)
        created_at: When the memory was created
        accessed_at: When the memory was last accessed
        access_count: Number of times the memory has been accessed
        metadata: Optional additional metadata

    Raises:
        ValueError: If namespace format is invalid or scores are out of range
    """
    id: str
    content: str
    content_hash: str
    type: MemoryType
    namespace: str = "global"
    importance: float = 0.5
    confidence: float = 0.3
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate memory fields after initialization."""
        # Validate namespace format
        if not validate_namespace(self.namespace):
            raise ValueError(
                f"Invalid namespace '{self.namespace}'. "
                "Must be 'global' or match 'project:{{name}}' pattern."
            )

        # Validate importance range
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(
                f"Importance must be between 0.0 and 1.0, got {self.importance}"
            )

        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def is_golden_rule(self) -> bool:
        """Check if this memory qualifies as a golden rule.

        Returns:
            True if confidence >= 0.9 or type is GOLDEN_RULE
        """
        return self.confidence >= 0.9 or self.type == MemoryType.GOLDEN_RULE

    def can_be_promoted(self) -> bool:
        """Check if this memory can be promoted to golden rule.

        Only PREFERENCE, DECISION, and PATTERN types can be promoted.

        Returns:
            True if eligible for golden rule promotion
        """
        promotable_types = {MemoryType.PREFERENCE, MemoryType.DECISION, MemoryType.PATTERN}
        return self.type in promotable_types and self.confidence >= 0.9


@dataclass
class Edge:
    """Represents a relationship between two memories.

    Defines a directed edge in the memory graph connecting two memories
    with a specific relationship type and weight.

    Attributes:
        source_id: ID of the source memory
        target_id: ID of the target memory
        relation: Type of relationship (RelationType enum)
        weight: Strength of the relationship from 0.0 to 1.0
        created_at: When the edge was created
    """
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StoreResult:
    """Result of a memory store operation.

    Indicates whether storage succeeded and provides the stored memory's ID
    or an error message if it failed.

    Attributes:
        success: Whether the operation succeeded
        id: ID of the stored memory (if successful)
        content_hash: SHA-256 hash of content for deduplication (if successful)
        error: Error message (if failed)
    """
    success: bool
    id: Optional[str] = None
    content_hash: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExpandedMemory:
    """A memory with graph expansion metadata.

    Wraps a Memory object with additional information about how it was
    discovered through graph traversal, including relevance scoring and
    the path taken to reach it.

    Attributes:
        memory: The underlying Memory object
        relevance_score: Combined relevance score (0.0 to 1.0) based on
            semantic similarity and graph distance
        hop_distance: Number of edges traversed to reach this memory (0 = direct match)
        path: List of edge types (RelationType values) in traversal order
        edge_weight_product: Product of all edge weights along the path
        explanation: Human-readable explanation of why this memory is relevant
    """
    memory: Memory
    relevance_score: float
    hop_distance: int
    path: list[str] = field(default_factory=list)
    edge_weight_product: float = 1.0
    explanation: str = ""


@dataclass
class GraphExpansionConfig:
    """Configuration for graph expansion behavior during memory recall.

    Controls how the graph is traversed when expanding from seed memories,
    including edge type weighting, depth limits, and safety guards.

    Attributes:
        max_depth: Maximum number of edges to traverse from seed memories (default: 1)
        decay_factor: Factor by which relevance decays per hop (default: 0.7)
        edge_type_weights: Weights for different edge types (0.0 to 1.0).
            Higher weights mean stronger relevance propagation.
            Defaults: supersedes=1.0, caused_by=0.9, relates_to=0.7, contradicts=0.5
        include_edge_types: Optional set of edge types to include (None means all)
        exclude_edge_types: Optional set of edge types to exclude (None means none)
        max_expanded: Maximum number of expanded memories to return
        max_nodes_visited: Safety guard - maximum nodes to visit during traversal
        max_edges_per_node: Safety guard - maximum edges to follow per node
    """
    max_depth: int = 1
    decay_factor: float = 0.7
    edge_type_weights: dict[str, float] = field(default_factory=dict)
    include_edge_types: Optional[set[str]] = None
    exclude_edge_types: Optional[set[str]] = None
    max_expanded: int = 20
    max_nodes_visited: int = 200
    max_edges_per_node: int = 10

    def __post_init__(self) -> None:
        """Merge default edge type weights with caller-provided values."""
        default_weights = {
            "supersedes": 1.0,
            "caused_by": 0.9,
            "relates_to": 0.7,
            "contradicts": 0.5,
        }
        # Merge defaults with caller-provided weights (caller takes precedence)
        merged = {**default_weights, **self.edge_type_weights}
        self.edge_type_weights = merged


@dataclass
class GraphNode:
    """Summarized memory info for graph visualization.

    Provides a lightweight representation of a memory node for display
    in graph inspection results, with truncated content for efficiency.

    Attributes:
        id: Memory ID
        content_preview: Truncated content (max 100 chars, ellipsis if truncated)
        memory_type: Memory type as string
        confidence: Confidence score
        importance: Importance score (0.0 to 1.0)
    """
    id: str
    content_preview: str
    memory_type: str
    confidence: float
    importance: float

    @classmethod
    def from_memory(cls, memory: Memory, hop_distance: int = 0) -> "GraphNode":
        """Create a GraphNode from a Memory object.

        Args:
            memory: The source Memory object
            hop_distance: Number of hops from origin (unused, kept for compatibility)

        Returns:
            GraphNode with truncated content preview
        """
        content = memory.content
        if len(content) > 100:
            content_preview = content[:97] + "..."
        else:
            content_preview = content
        return cls(
            id=memory.id,
            content_preview=content_preview,
            memory_type=memory.type.value,
            confidence=memory.confidence,
            importance=memory.importance,
        )


@dataclass
class GraphEdge:
    """Relationship info for graph visualization.

    Represents an edge in the memory graph for inspection results.

    Attributes:
        id: Edge identifier (integer)
        source_id: Source memory ID
        target_id: Target memory ID
        edge_type: Relationship type as string
        weight: Edge weight (0.0 to 1.0)
    """
    id: int
    source_id: str
    target_id: str
    edge_type: str
    weight: float

    @classmethod
    def from_edge(cls, edge: Edge, edge_id: int = 0) -> "GraphEdge":
        """Create a GraphEdge from an Edge object.

        Args:
            edge: The source Edge object
            edge_id: Integer identifier for this edge

        Returns:
            GraphEdge for visualization
        """
        return cls(
            id=edge_id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            edge_type=edge.relation.value,
            weight=edge.weight,
        )


@dataclass
class GraphPath:
    """Path from origin with scoring.

    Records the traversal path taken to reach a node from the origin,
    including computed relevance scores.

    Attributes:
        node_ids: List of memory IDs in traversal order (origin first)
        edge_types: List of edge types traversed
        total_weight: Product of all edge weights along path
        relevance_score: Combined relevance score for this path
    """
    node_ids: list[str]
    edge_types: list[str]
    total_weight: float
    relevance_score: float


@dataclass
class GraphStats:
    """Summary statistics for graph inspection.

    Provides aggregate information about the inspected graph region.

    Attributes:
        node_count: Total number of nodes discovered
        edge_count: Total number of edges discovered
        max_depth_reached: Maximum hop distance actually traversed
        origin_id: ID of the origin memory node
    """
    node_count: int
    edge_count: int
    max_depth_reached: int
    origin_id: str


@dataclass
class GraphInspectionResult:
    """Combined result for graph inspection tool.

    Contains all information needed to visualize and understand
    the graph structure around a memory.

    Attributes:
        success: Whether the inspection succeeded
        origin_id: ID of the origin memory node
        nodes: List of GraphNode objects discovered
        edges: List of GraphEdge objects discovered
        paths: List of GraphPath objects showing traversal paths
        stats: GraphStats summary
        error: Error message (if failed)
    """
    success: bool
    origin_id: str = ""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    stats: Optional[GraphStats] = None
    error: Optional[str] = None

    def to_mermaid(self) -> str:
        """Generate Mermaid flowchart syntax for the graph.

        Returns:
            Mermaid flowchart TD syntax string with nodes as rounded
            rectangles and edges labeled with relationship types.
        """
        if not self.nodes:
            return "flowchart TD\n    empty[No nodes found]"

        lines = ["flowchart TD"]

        # Add nodes as rounded rectangles with content preview
        for node in self.nodes:
            # Escape special characters in content preview
            label = node.content_preview.replace('"', "'").replace("\n", " ")
            # Truncate label further for mermaid display
            if len(label) > 50:
                label = label[:47] + "..."
            lines.append(f'    {node.id}["{label}"]')

        # Add edges with relationship type labels
        for edge in self.edges:
            lines.append(f"    {edge.source_id} -->|{edge.edge_type}| {edge.target_id}")

        return "\n".join(lines)


@dataclass
class RecallResult:
    """Result of a memory recall/query operation.

    Contains the list of memories matching a query, the total count
    of matches (which may be larger than the returned list if limited),
    and an optional relevance score.

    Attributes:
        memories: List of Memory objects matching the query
        total: Total number of matching memories
        score: Relevance score for the search results (0.0 to 1.0)
        expanded_memories: List of ExpandedMemory objects discovered through
            graph traversal (populated when include_related=True)
    """
    memories: list[Memory] = field(default_factory=list)
    total: int = 0
    score: Optional[float] = None
    expanded_memories: list[ExpandedMemory] = field(default_factory=list)


@dataclass
class ValidateResult:
    """Result of a memory validation operation.

    Records the outcome of validating (confirming or refuting) a memory.

    Attributes:
        success: Whether the validation was recorded
        memory_id: ID of the validated memory
        old_confidence: Confidence before validation
        new_confidence: Confidence after validation
        promoted: Whether memory was promoted to golden rule
        error: Error message (if failed)
    """
    success: bool
    memory_id: Optional[str] = None
    old_confidence: Optional[float] = None
    new_confidence: Optional[float] = None
    promoted: bool = False
    error: Optional[str] = None


@dataclass
class ApplyResult:
    """Result of applying a memory (recording its use).

    Attributes:
        success: Whether the apply was recorded
        memory_id: ID of the applied memory
        event_id: ID of the validation event created
        error: Error message (if failed)
    """
    success: bool
    memory_id: Optional[str] = None
    event_id: Optional[int] = None
    error: Optional[str] = None


@dataclass
class OutcomeResult:
    """Result of recording an outcome for a memory application.

    Attributes:
        success: Whether the outcome was recorded
        memory_id: ID of the memory
        outcome_success: Whether the memory application succeeded
        new_confidence: Updated confidence score
        promoted: Whether memory was promoted to golden rule
        error: Error message (if failed)
    """
    success: bool
    memory_id: Optional[str] = None
    outcome_success: Optional[bool] = None
    new_confidence: Optional[float] = None
    promoted: bool = False
    error: Optional[str] = None
