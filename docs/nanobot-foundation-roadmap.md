# Nanobot Foundation Roadmap

## Goal

Use nanobot as the runtime foundation for a future digital avatar / digital butler.
Do not aim for immediate Hermes parity. First verify controllability, speed, skill invocation, and IM integration.

## Stage 1: Minimal foundation validation (1-2 weeks)

- Keep the core path: `nanobot/agent/loop.py`, `nanobot/agent/runner.py`, `nanobot/channels/`, `nanobot/agent/skills.py`
- Connect **1 IM + 1 provider + 1 user skill**
- Do not migrate large Hermes upper-layer capabilities yet

### Acceptance

- Stable basic conversation
- Reliable user skill invocation
- Acceptable response speed
- Stable IM message flow

## Stage 2: External memory integration (1-2 weeks)

- Pre-turn / post-turn integration points have landed in `AgentLoop` through the
  optional `ExternalMemoryBridge`.
- `nanobot/memory_service/*` now provides a local SQLite + FTS5 memory service
  for turn events, simple typed memories, and reserved job records.
- Still to complete: configuration schema, default/runtime enablement, typed
  memory API coverage, operational UX, and productized behavior around recall
  and writeback.
- Keep nanobot focused on runtime rather than heavy memory ownership.

### Acceptance

- Bridge-level cross-session recall/writeback can be verified when explicitly
  enabled or injected.
- Product-level preferences, tasks, and user profile recall still need
  configuration and UX work before being treated as a default capability.
- Behavior remains controllable and debuggable.

## Stage 3: Digital butler capabilities (2-4 weeks)

- Add proactive tasks, reminders, workflow orchestration, and multi-skill routing
- Add permission boundaries, failure recovery, and observability
- Port only high-value Hermes upper-layer capabilities

### Acceptance

- Support 2-3 real daily scenarios end-to-end

## Rules

- Treat nanobot as a kernel, not a Hermes replacement
- Keep memory and self-learning external
- Validate one IM first, then extend
- Close real scenarios before broad feature migration

## Stage 1 likely touch points

- `nanobot/agent/loop.py`
- `nanobot/agent/runner.py`
- `nanobot/channels/manager.py`
- The selected IM channel
- `nanobot/agent/skills.py`
- `nanobot/providers/factory.py`
- `nanobot/session/manager.py`
