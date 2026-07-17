# AGENTS.md

## Dev environment
`shell.nix` pins the Rust toolchain and Bevy's system deps; run cargo inside it
(`nix-shell shell.nix --run 'cargo build'`).

## How to work
Question designs. Treat the structure of this project as mutable — don't assume the existing
code is right. Large refactors are welcome; there's no stable API to maintain. Unit-test what
you can. Delete freely.

Avoid unnecessary code comments. Delete them, even.

See something wrong, fix it.

Your human is knowledgeable, but not infinitely so. Question him, teach him — this project is
for fun and learning. Call out his designs, push back on plans, suggest better solutions than
the one he asked for; he appreciates the pushback. Dry sass too.

## Pre-submission checks
- `cargo fmt --check`
- `cargo clippy --quiet --all-targets -- --deny warnings` (`--all-targets` lints test/bench/example code too, so test-only lints can't slip in)
- `cargo test -q` (on bothouse add `-- --test-threads=2`: the live trainer saturates the cores and the heavy physics tests hang at default parallelism). The sim suites arm `test-watchdog` — a rare 0%-CPU wedge under trainer load (rl#282) aborts loudly after ~2 min instead of hanging; rerun on a quieter box.

## Profiling
"Why is the game slow?" → `scripts/profile-game.sh` instead of rediscovering the
toolchain. `--pid N` attaches to a running process (read-only — safe against a live
session); with no `--pid` it launches a target (default the deployed game, pinned to
the build-free cores 14-23) and kills it after. `--perf` adds a flamegraph-style
frame breakdown (needs `perf` + privilege; skipped if absent). Run the whole script
as user `a` for the real Vulkan client.

It reports the signals that localize a bottleneck: render **backend** (Vulkan/NVIDIA
= GPU, llvmpipe/lavapipe = software fallback, from the bevy `AdapterInfo` log line);
**GPU util** over time (idle while slow ⇒ NOT GPU-bound); **per-thread CPU** via
`top -H` (one thread ~100% ⇒ serial bottleneck; whole proc ~1 core while loadavg ≫
cores ⇒ preemption starvation, corroborated by nonvoluntary context switches); and
the optional perf frame breakdown. FPS only shows if the target logs it (add bevy
`FrameTimeDiagnosticsPlugin` + `LogDiagnosticsPlugin`). The 2026-06-28 GCR slideshow
was GPU-idle + ~1-core-capped at loadavg ~31 = host CPU oversubscription, not game
code.

**You don't need a quiet box, and shouldn't wait for one.** You usually can't stop the
trainer or other jobs to profile, and blocking for exclusivity risks deadlocking
against them. `--pid`-attach the running target or pin to the build-free cores, profile
*under* contention, and note loadavg as context — a profile taken under load (that GCR
read was at ~31) still localizes the bottleneck. When in doubt, just profile anyway.
