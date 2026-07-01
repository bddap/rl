## Mandate (owner, 2026-07-01, verbatim intent)

> I'd like a system for implementing multiple policy architectures — multiple crabs in the same GCR instance running different policy architectures. I'm aware this goes against our "don't keep multiple versions of the same thing" rule, but I'd like to iterate on multiple architectures concurrently. I trust you can find a principled approach. We'll cull inferior architectures as we go — just want to be able to run multiple different brains in the same binary.

Progress on this system is a standing item in the daily status emails.

## The principled resolution of the one-impl tension

The one-implementation rule bans two *seams* for the same thing drifting apart — not N implementations behind one seam. So: **exactly one polymorphic policy interface** (the existing obs→act seam, typed per rl#43), with N architecture implementations registered behind it as leaves. Shared once, used by every brain: rollout plumbing, trainer loop, checkpoint I/O, eval/metrics, GCR binding. An architecture is then *data + one leaf impl file*; **culling an inferior architecture = deleting its leaf**, and nothing else moves. What stays illegal: a second trainer, a second obs/act seam, a per-arch fork of any shared path.

## Design constraints (increment 0 refines these)

- **Checkpoints carry architecture identity.** A checkpoint must be un-loadable into the wrong architecture by construction (tagged format; strong types, not a filename convention).
- **Per-crab brain binding in GCR.** Each crab in one instance binds a (architecture, checkpoint) pair; SP=MP single code path unaffected. The arch must be *visible* — label crabs by brain in demo/HUD so a playtest can tell who's who.
- **Trainer strategy is an increment-0 decision:** one run per arch on shared infra with namespaced checkpoints/metrics vs. one run training a population. Training is GPU and bigger NNs are the coming bottleneck — the design must state its GPU cost model.
- **Per-arch eval is the culling instrument.** Same reward, same curriculum, per-arch curves (daily-email chart gains per-arch series). Culls are evidence-based, owner-informed.
- **Registry, not config sprawl:** adding an architecture = one leaf module + one registry entry.

## Increments

- [x] **0 — Design pass (reviewer-looped):** done — [design doc](https://github.com/bddap/rl/issues/200#issuecomment-4860327262) (seam audit, checkpoint envelope, per-arch-runs decision + GPU cost model, GCR binding/labeling, eval/culling procedure, fleet migration). Boxes below refined per that doc; each box's detail lives there.
- [x] 1 — Seam + registry: `AnyBrain` enum-Module + `ArchId` (no trait — see design §1); `CrabBrain` → `bot/arch/mlp256.rs` as `Mlp256`; per-row log_std behind a typed `GaussianHead` shared floor/clamp layer; brain I/O round-trips the LEAF record. Acceptance: bit-identical actions on a fixed obs vector + a current-main `brain.bin` loads unchanged (golden file).
- [ ] 2 — Tagged checkpoints, format: `CheckpointEnvelope {kind, version, arch}` on all four artifacts (replaces `OptimizerCheckpoint`; arch as validated string, never a bincode enum — design §2); loader dispatch on arch, per-artifact refusal policy, dir-coherence check, `RigFit` arch arm; `shape.txt` deleted; `migrate-checkpoint` tool (sole legacy parser).
- [ ] 3 — Tagged checkpoints, fleet: migrate live dirs; release store + TV + decks redeployed via the normal pipeline; `rl-release-build` `CHECKPOINT_FILES` updated (drop `curriculum.bin`); migration tool deleted when the fleet is done.
- [ ] 4 — Per-arch trainer runs (bothouse): `rl-target@<run>` templated unit, per-run dirs, explicit `--workers` partition, `--arch` flag / resume-tag-authoritative selection, OTEL run tag; eval-monitor + daily-email per-arch series.
- [ ] 5 — Second architecture lands as a leaf (within the v1 feedforward/Gaussian contract); ≥2 seeds per arch, equal-env-step curves on the daily email; first evidence-based cull decision (owner-informed, executed as a leaf delete).
- [ ] 6 — GCR multi-brain, sim/wire: `Sim.crab`/`CrabPose`/snapshot/articulation go per-crab; per-env `Policy` bindings with host fail-loud validation; stale all-peers weights-digest gate deleted (asset gate kept).
- [ ] 7 — GCR multi-brain, visible: shared world-space brain labels (`arch @shortdigest` + failure attribution), demo proof first then GCR; a playtest can tell who's who.

## Non-goals

No second trainer/seam/plumbing per arch (that's the drift the rule exists for). No silent fallback between brains — a crab that can't load its assigned brain fails loud ([[real-sally-definition]]). Culling stays a deliberate delete, not an automatic kill switch.


