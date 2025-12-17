# Delete Memories

Delete memories matching: **$ARGUMENTS**

## Instructions

1. **Detect input type**: Check if "$ARGUMENTS" is a memory ID or search query.

   **Memory ID patterns**:
   - UUID format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (e.g., `550e8400-e29b-41d4-a716-446655440000`)
   - Timestamped format: `mem_{digits}_{hex8}` (e.g., `mem_1702783200000000_abc12def`)

2. **If it looks like a memory ID**:
   - Call `memory_forget_tool` directly with:
     - `memory_id`: "$ARGUMENTS"
     - `confirm`: true
   - Report the result:
   ```
   Deleted memory: $ARGUMENTS
   ```
   - If the memory wasn't found, report the error.

3. **If it looks like a search query**:
   - First, search for memories using `memory_recall_tool`:
     - `query`: "$ARGUMENTS"
     - `n_results`: 5

   - Show the user what will be deleted:
   ```
   ## Memories to delete:
   - [id: mem_xxx] [preference] User prefers tabs over spaces
   - [id: mem_yyy] [decision] Use React for frontend
   ```

   - Ask for confirmation before deleting.

   - Call `memory_forget_tool` with:
     - `query`: "$ARGUMENTS"
     - `confirm`: true

   - Report what was deleted:
   ```
   Deleted 2 memories matching "$ARGUMENTS"
   ```

Keep responses concise.
