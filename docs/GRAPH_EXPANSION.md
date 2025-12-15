# Graph Expansion in Recall

Recall's graph expansion feature allows you to discover related memories by traversing relationship edges. This document explains how it works and how to use it effectively.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Edge Types](#edge-types)
3. [BFS Traversal](#bfs-traversal)
4. [Relevance Scoring](#relevance-scoring)
5. [Edge Type Filtering](#edge-type-filtering)
6. [Safety Guards](#safety-guards)
7. [API Reference](#api-reference)
8. [Examples](#examples)

---

## Core Concepts

### Nodes and Edges

Recall stores memories as **nodes** in a graph. Relationships between memories are **edges** connecting those nodes.

```
┌─────────────────┐         ┌─────────────────┐
│  Memory A       │         │  Memory B       │
│  (node)         │─────────│  (node)         │
└─────────────────┘  edge   └─────────────────┘
```

**Without graph expansion:** Semantic search finds memories based on text similarity alone.

**With graph expansion:** After finding primary results, Recall follows edges to discover related memories that might not match the query text but are semantically connected.

---

## Edge Types

Recall supports four relationship types:

| Edge Type | Weight | Description | Use Case |
|-----------|--------|-------------|----------|
| `relates_to` | 0.7 | General association | Linking related concepts |
| `supersedes` | 1.0 | Newer replaces older | Version updates, corrections |
| `caused_by` | 0.9 | Causal relationship | Root cause analysis |
| `contradicts` | 0.5 | Conflicting information | Identifying conflicts |

### Creating Relationships

Use `memory_relate_tool` to create edges:

```python
memory_relate_tool(
    source_id="mem_123",
    target_id="mem_456",
    relation="supersedes",  # or: relates_to, caused_by, contradicts
    weight=0.9              # optional, 0.0-1.0, default 1.0
)
```

### Automatic Relationships

Recall can automatically create edges:

- **`contradicts`**: Created by `detect_contradictions()` when semantically similar memories conflict
- **`supersedes`**: Created by `check_supersedes()` when validation shows one memory outperforms another

---

## BFS Traversal

Graph expansion uses **Breadth-First Search (BFS)** to explore relationships in waves:

```
                    Search Result
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         Memory A    Memory B    Memory C     ← Hop 1 (depth=1)
            │            │
       ┌────┴────┐       │
       ▼         ▼       ▼
    Memory D  Memory E  Memory F              ← Hop 2 (depth=2)
```

**Why BFS?** Closer memories (fewer hops) are discovered first, aligning with the intuition that directly related memories are more relevant.

### Controlling Traversal Depth

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 1 | Maximum hops from primary results |
| `max_expanded` | 20 | Maximum expanded memories to return |

```python
# Only direct neighbors
memory_recall_tool(query="...", include_related=True, max_depth=1)

# Neighbors and neighbors-of-neighbors
memory_recall_tool(query="...", include_related=True, max_depth=2)
```

---

## Relevance Scoring

Expanded memories are ranked by a **relevance score** (0.0 to 1.0):

```
relevance = decay_factor^hop_distance × path_weight × type_weight
```

### Components

#### 1. Decay Factor

Distance penalty - further memories score lower:

| Hops | decay=0.7 | decay=0.8 |
|------|-----------|-----------|
| 1 | 0.70 | 0.80 |
| 2 | 0.49 | 0.64 |
| 3 | 0.34 | 0.51 |

#### 2. Path Weight

Product of all edge weights along the path. If edges have weights 0.9 and 0.8:
```
path_weight = 0.9 × 0.8 = 0.72
```

#### 3. Type Weight

Geometric mean of edge type weights along the path (see [Edge Types](#edge-types) for defaults).

### Example Calculation

Finding a memory 2 hops away via `supersedes → relates_to`:

```
decay_factor = 0.7
hop_distance = 2
edge_weights = 1.0, 0.8
type_weights = 1.0 (supersedes), 0.7 (relates_to)

relevance = 0.7^2 × (1.0 × 0.8) × √(1.0 × 0.7)
          = 0.49 × 0.8 × 0.837
          = 0.328
```

---

## Edge Type Filtering

Control which relationship types to traverse:

### Include Filter (Whitelist)

Only follow specified edge types:

```python
memory_recall_tool(
    query="...",
    include_related=True,
    include_edge_types=["supersedes", "caused_by"]
)
```

### Exclude Filter (Blacklist)

Follow all edge types except specified ones:

```python
memory_recall_tool(
    query="...",
    include_related=True,
    exclude_edge_types=["contradicts"]
)
```

**Note:** If both are specified, `include_edge_types` takes precedence.

---

## Safety Guards

Prevent runaway traversal in densely connected graphs:

| Guard | Default | Description |
|-------|---------|-------------|
| `max_expanded` | 20 | Stop after collecting this many expanded memories |
| `max_nodes_visited` | 200 | Stop after visiting this many nodes total |
| `max_edges_per_node` | 10 | Maximum edges to follow from any single node |

These limits ensure predictable performance regardless of graph density.

---

## API Reference

### memory_recall_tool

```python
memory_recall_tool(
    query: str,                           # Search query text
    n_results: int = 5,                   # Max primary results
    namespace: str = None,                # Filter by namespace
    memory_type: str = None,              # Filter by memory type
    min_importance: float = None,         # Minimum importance score
    include_related: bool = False,        # Enable graph expansion
    max_depth: int = 1,                   # Max hops for expansion
    max_expanded: int = 20,               # Max expanded memories
    decay_factor: float = 0.7,            # Relevance decay per hop
    include_edge_types: list[str] = None, # Edge type whitelist
    exclude_edge_types: list[str] = None, # Edge type blacklist
) -> dict
```

### Response Structure

```python
{
    "success": True,
    "memories": [                # Primary search results
        {
            "id": "...",
            "content": "...",
            "type": "preference",
            # ... other memory fields
        }
    ],
    "total": 5,
    "score": 0.85,              # Average similarity score
    "expanded": [               # Graph-expanded memories
        {
            "id": "...",
            "content": "...",
            "type": "pattern",
            "relevance_score": 0.80,
            "hop_distance": 1,
            "path": ["supersedes"],
            "explanation": "1 hop via supersedes, combined weight 0.80"
        }
    ]
}
```

### memory_relate_tool

```python
memory_relate_tool(
    source_id: str,             # Source memory ID
    target_id: str,             # Target memory ID
    relation: str,              # Edge type (relates_to, supersedes, etc.)
    weight: float = 1.0,        # Edge weight (0.0 to 1.0)
) -> dict
```

---

## Examples

### Basic Graph Expansion

```python
# Find memories about "git workflow" and related memories
result = memory_recall_tool(
    query="git workflow",
    n_results=3,
    include_related=True
)

print(f"Primary: {len(result['memories'])}")
print(f"Expanded: {len(result['expanded'])}")
```

### Finding Superseding Memories

```python
# Find outdated info and what replaced it
result = memory_recall_tool(
    query="old API endpoint",
    include_related=True,
    include_edge_types=["supersedes"]
)

for exp in result['expanded']:
    print(f"Replaced by: {exp['content'][:50]}...")
```

### Multi-Hop Traversal

```python
# Explore 2 levels of relationships
result = memory_recall_tool(
    query="authentication",
    include_related=True,
    max_depth=2,
    decay_factor=0.8  # Keep distant memories more relevant
)

for exp in result['expanded']:
    print(f"[{exp['hop_distance']} hops] {exp['explanation']}")
```

### Creating a Knowledge Graph

```python
# Store related memories
mem1 = memory_store_tool(content="Use JWT for auth", memory_type="decision")
mem2 = memory_store_tool(content="JWT tokens expire in 1 hour", memory_type="decision")
mem3 = memory_store_tool(content="Refresh tokens last 7 days", memory_type="decision")

# Connect them
memory_relate_tool(mem1['id'], mem2['id'], "relates_to")
memory_relate_tool(mem1['id'], mem3['id'], "relates_to")
memory_relate_tool(mem3['id'], mem2['id'], "caused_by")  # refresh tokens caused by JWT expiry

# Now querying for "JWT" will also surface token expiry and refresh info
```

---

## Best Practices

1. **Start with `max_depth=1`** - Direct relationships are usually most relevant
2. **Use `supersedes` for updates** - Automatically surfaces newer information
3. **Filter with `include_edge_types`** when you know what you want
4. **Tune `decay_factor`** - Use 0.8 for shallow graphs, 0.7 for deep traversal
5. **Check `explanation` field** - Great for debugging why a memory was included

---

## Data Types

### GraphExpansionConfig

```python
@dataclass
class GraphExpansionConfig:
    max_depth: int = 1
    decay_factor: float = 0.7
    edge_type_weights: dict[str, float]  # defaults: supersedes=1.0, caused_by=0.9, relates_to=0.7, contradicts=0.5
    include_edge_types: Optional[set[str]] = None
    exclude_edge_types: Optional[set[str]] = None
    max_expanded: int = 20
    max_nodes_visited: int = 200
    max_edges_per_node: int = 10
```

### ExpandedMemory

```python
@dataclass
class ExpandedMemory:
    memory: Memory              # The discovered memory
    relevance_score: float      # Combined relevance (0.0 to 1.0)
    hop_distance: int           # Number of edges traversed
    path: list[str]             # Edge types in traversal order
    edge_weight_product: float  # Product of edge weights
    explanation: str            # Human-readable explanation
```
