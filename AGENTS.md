# AGENTS.md - LlamaHerd Project Rules

## Design Philosophy

**Agents first.** All features, APIs, CLIs, and docs must be designed for autonomous agents (Hermes, Claude Code, Codex, etc.) as the primary consumer — not just humans reading terminals. This means:

- **CLI commands output JSON by default** — parseable by agents. Add `--format table` for humans.
- **APIs return structured JSON** with consistent schemas, not human-readable prose.
- **Exit codes matter** — 0 for success, non-zero for failure. Errors go to stderr, data to stdout.
- **Ids over names** — accept both, but always return stable identifiers.
- **No interactive prompts** — every operation must be doable in a single invocation with flags/args.
- **Docs for agents** — README sections should cover programmatic usage (curl examples, CLI flags) before GUI instructions.

## Code Conventions

- Single-file modules are fine until they're not. Split when a file exceeds ~3000 lines.
- All DB schema changes must be backward-compatible (ALTER TABLE, not DROP+CREATE).
- Rate limits and quotas should be optional with sensible defaults (NULL = unlimited).
- The proxy should never crash due to bad config — log warnings and degrade gracefully.

## Branching

- `main` is the release branch. Push directly for small changes.
- Feature branches for large changes: `feat/rate-limits`, `feat/cli`, etc.