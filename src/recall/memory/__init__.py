"""Memory module for the recall system.

This module provides the core data types, structures, and operations for memory
storage and retrieval.
"""

from recall.memory.operations import (
    ForgetResult,
    StoreResultWithRelations,
    memory_context,
    memory_forget,
    memory_recall,
    memory_relate,
    memory_store,
)
from recall.memory.types import (
    Edge,
    Memory,
    MemoryType,
    RecallResult,
    RelationType,
    StoreResult,
    validate_namespace,
)

__all__ = [
    "Edge",
    "ForgetResult",
    "Memory",
    "MemoryType",
    "RecallResult",
    "RelationType",
    "StoreResult",
    "StoreResultWithRelations",
    "memory_context",
    "memory_forget",
    "memory_recall",
    "memory_relate",
    "memory_store",
    "validate_namespace",
]
