# Store Memory

Store the following as a memory in Recall:

**Content:** $ARGUMENTS

## Instructions

1. Determine the appropriate memory type:
   - `preference` - User preferences or settings (e.g., "use 2-space indent", "prefer dark mode")
   - `decision` - Technical decisions (e.g., "use FastAPI for backend", "chose PostgreSQL over MySQL")
   - `pattern` - Recurring behaviors or conventions (e.g., "always run tests before commit")
   - `session` - Session-specific context (default if unclear)

2. Determine importance (0.0-1.0):
   - 0.8-1.0: Critical preferences/decisions that should always apply
   - 0.5-0.7: Important but context-dependent
   - 0.3-0.5: Nice to know, lower priority

3. Determine namespace:
   - `global` - Applies across all projects
   - `project:{name}` - Specific to current project (auto-detect from cwd)

4. Call the `memory_store_tool` via the recall MCP server with appropriate parameters.

5. Confirm storage with a brief message showing: type, namespace, and importance assigned.

Keep the response concise - just confirm what was stored.
