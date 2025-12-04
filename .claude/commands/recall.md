# Search Memories

Search stored memories for: **$ARGUMENTS**

## Instructions

1. Call `memory_recall_tool` via the recall MCP server with:
   - `query`: "$ARGUMENTS"
   - `n_results`: 10
   - `include_related`: true

2. Format results as a concise markdown list:

```
## Memories matching "$ARGUMENTS"

- [preference] User prefers dark mode (importance: 0.8, global)
- [decision] Using FastAPI for backend (importance: 0.9, project:myapp)
```

3. If no results, say "No memories found matching '$ARGUMENTS'"

4. Keep output terse - no verbose explanations.
