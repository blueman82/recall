# List Memories

Browse and inspect stored memories.

## Instructions

1. Call `memory_recall_tool` via recall MCP with a broad query:
   - `query`: "$ARGUMENTS" (or "all memories preferences decisions patterns" if empty)
   - `n_results`: 20
   - `include_related`: true

2. Group results by type and format as:

```
## Stored Memories

### Preferences (3)
- User prefers 2-space indent [global, importance: 0.8]
- Dark mode in all applications [global, importance: 0.7]
- TypeScript over JavaScript [project:webapp, importance: 0.6]

### Decisions (2)
- Use FastAPI for backend [project:api, importance: 0.9]
- PostgreSQL for database [project:api, importance: 0.8]

### Patterns (1)
- Always run tests before committing [global, importance: 0.5]

### Session (0)
(none)

---
Total: 6 memories | Global: 3 | Project-scoped: 3
```

3. If $ARGUMENTS provided, filter/search within that scope.

4. Keep output scannable - this is for quick inspection.
