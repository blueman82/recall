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
    """
    PREFERENCE = "preference"
    DECISION = "decision"
    PATTERN = "pattern"
    SESSION = "session"


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
        created_at: When the memory was created
        accessed_at: When the memory was last accessed
        access_count: Number of times the memory has been accessed

    Raises:
        ValueError: If namespace format is invalid or importance is out of range
    """
    id: str
    content: str
    content_hash: str
    type: MemoryType
    namespace: str = "global"
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

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
class RecallResult:
    """Result of a memory recall/query operation.

    Contains the list of memories matching a query, the total count
    of matches (which may be larger than the returned list if limited),
    and an optional relevance score.

    Attributes:
        memories: List of Memory objects matching the query
        total: Total number of matching memories
        score: Relevance score for the search results (0.0 to 1.0)
    """
    memories: list[Memory] = field(default_factory=list)
    total: int = 0
    score: Optional[float] = None
