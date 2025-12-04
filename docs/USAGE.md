# Recall Usage Guide

## Three Ways to Use Recall

| Method | Who Controls | When It Runs |
|--------|--------------|--------------|
| **MCP Tools** | Claude decides | When Claude thinks it's relevant |
| **Hooks** | Automatic | Session start/end |
| **Slash Commands** | You decide | When you invoke them |

---

## Slash Commands (Explicit Control)

| Command | Purpose |
|---------|---------|
| `/remember <text>` | Store a memory you want kept |
| `/recall <query>` | Search your memories |
| `/memories` | List all stored memories |
| `/forget <query>` | Delete memories |

**Example:**
```
/remember always use pytest, never unittest
/recall testing preferences
/memories
/forget outdated preference
```

---

## Hooks (Automatic)

| Hook | What It Does |
|------|--------------|
| **SessionStart** | Loads relevant memories as context when you start Claude Code |
| **SessionEnd** | Summarizes session, stores important decisions/preferences automatically |

You don't invoke these - they run automatically.

---

## MCP Tools (Claude's Choice)

Claude can call these when it thinks they're useful:

| Tool | Purpose |
|------|---------|
| `memory_store_tool` | Store a memory |
| `memory_recall_tool` | Search memories |
| `memory_relate_tool` | Link memories together |
| `memory_context_tool` | Get formatted context |
| `memory_forget_tool` | Delete memories |

**When Claude uses them:** If you mention preferences, make decisions, or ask about past context - Claude may proactively store/recall.

---

## When to Use What

| Situation | Use |
|-----------|-----|
| "I want this remembered for sure" | `/remember` |
| "What do I have stored?" | `/memories` |
| "Find my preferences on X" | `/recall X` |
| "Delete old/wrong memories" | `/forget` |
| "Just working normally" | Let hooks + MCP handle it |

---

## Memory Types

| Type | Example |
|------|---------|
| `preference` | "Use 2-space indent", "Prefer dark mode" |
| `decision` | "Chose FastAPI for backend", "Using PostgreSQL" |
| `pattern` | "Always run tests before commit" |
| `session` | Temporary context from conversations |

---

## Namespaces

- `global` - Applies everywhere
- `project:{name}` - Scoped to specific project

Memories are auto-scoped based on your working directory.
