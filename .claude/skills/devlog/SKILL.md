---
name: devlog
description: Add a timestamped entry to DEVLOG.md for tracking progress, decisions, findings, and changes
user-invocable: true
---

# Devlog Entry

Add a structured entry to `DEVLOG.md` to maintain a paper trail of project progress.

## Usage

`/devlog <type> <message>`

Types: `decision`, `finding`, `milestone`, `fix`, `note`

## Behavior

1. Read the current `DEVLOG.md` (create if missing)
2. Get the current date, git branch, and latest commit hash
3. Append a new entry at the TOP of the log (below the header), formatted as:

```markdown
### YYYY-MM-DD HH:MM — [TYPE] Summary
- **Branch:** `branch-name` | **Commit:** `abc1234`
- Detail line 1
- Detail line 2
```

4. If the user provides arguments, use them as the entry content
5. If no arguments, summarize recent work from the conversation context

## Entry Types

| Type | When to use |
|------|-------------|
| `decision` | Architectural choice, resolved open question, tech decision |
| `finding` | Data exploration result, validation outcome, benchmark result |
| `milestone` | Feature complete, tests passing, phase gate met |
| `fix` | Bug fix, test fix, config correction |
| `note` | General progress note, context for future sessions |

## Rules

- Always prepend (newest first)
- Keep entries concise (2-5 bullet points)
- Include relevant file paths when applicable
- Never remove existing entries
