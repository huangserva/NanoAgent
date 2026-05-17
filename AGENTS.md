# nanobot - Codex Execution Contract

This file complements `CLAUDE.md`.

- `CLAUDE.md` remains the source of truth for repo architecture, commands, testing norms, and safety boundaries.
- `AGENTS.md` defines the Codex-facing execution contract for implementation work in this repository.
- If the two overlap, prefer the repo facts in `CLAUDE.md` and the actual filesystem.

## Exact Execution Flag

When launching Codex for execution work in this repository, pass the exact flag:

```bash
--dangerously-bypass-approvals-and-sandbox
```

Rules:

- Do not replace the exact flag with natural-language phrases like “maximum permissions” or “full permissions”.
- Prefer explicit commands such as `codex exec --dangerously-bypass-approvals-and-sandbox ...` when handing implementation work to Codex.
- If execution is routed through the local `/codex:*` companion flow, the underlying execution path must still preserve this exact flag.

## Routing Contract

Choose the lightest route that matches the work:

- **Read-only / explanation**: inspect, summarize, compare, or answer directly
- **Small implementation**: one narrow Codex execution task with explicit scope
- **Medium implementation**: define scope, `do_not_change`, done criteria, and verification before execution
- **High-risk implementation**: plan first, execute second, review third

High-risk work includes:

- public interfaces or shared types
- config schema or persistence changes
- auth, payment, permissions, CI/CD, or dependency changes
- wide refactors or hard-to-reverse operations

## Task Handoff Format

When delegating execution to Codex, prefer a narrow contract with:

- `task`
- `scope`
- `do_not_change`
- `definition_of_done`
- `verification`

Do not hand Codex a vague large request when a smaller first step can be verified independently.

## Guardrails

Unless the user explicitly asks for it, do not introduce:

- new dependencies or package managers
- migrations or schema churn unrelated to the task
- CI/CD changes
- new services, daemons, or global state
- credential files or secret-handling changes
- broad refactors outside the requested outcome

Keep changes minimal. Do not expand scope just because a broader cleanup looks attractive.

## Verification and Review

Do not report completion without evidence.

- Run the smallest relevant tests or build checks for the changed surface.
- If verification is incomplete, say what was not verified.
- Do not invent command output, test results, or repository state.

For risky or shared-surface changes:

1. Claude review checks direction, scope, and goal alignment.
2. Codex review checks implementation correctness.
3. Use adversarial review when the blast radius is high.

## Runtime Ownership

The tmux-backed local `/codex:*` control plane is the authoritative execution and monitoring layer for this repository.

- Background execution, status, result, and cancel flows belong to the local Codex companion runtime.
- Do not introduce a second competing execution control plane through external workflow tools.
- Policy and routing may evolve, but execution and monitoring should continue to flow through the existing tmux-backed runtime.
