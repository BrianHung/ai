---
'@ai-sdk/harness-claude-code': patch
---

feat(harness-claude-code): add `experimental_persistentSession` to keep one Claude Code process alive across turns, so background shells and subagents keep running after the turn that started them, and `experimental_onBackgroundPart` to receive what they report between turns
