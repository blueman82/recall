# Delete Memories

Delete memories matching: **$ARGUMENTS**

## Instructions

1. First, search for memories matching the query using `memory_recall_tool`:
   - `query`: "$ARGUMENTS"
   - `n_results`: 5

2. Show the user what will be deleted:
```
## Memories to delete:
- [preference] User prefers tabs over spaces
- [decision] Use React for frontend
```

3. Ask for confirmation before deleting (unless query is very specific like an ID).

4. Call `memory_forget_tool` with:
   - `query`: "$ARGUMENTS"
   - `confirm`: true

5. Report what was deleted:
```
Deleted 2 memories matching "$ARGUMENTS"
```

Keep responses concise.
