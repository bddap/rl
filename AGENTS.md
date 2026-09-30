# Working on rl

Edit by subtraction: resolve a problem by deleting code; a tactical patch over a symptom is not accepted. One implementation per thing, never two alive.

Question designs and propose better ones. Large refactors are welcome; there is no
stable API to preserve. Fix problems you find and unit-test behavior where practical.
This is a learning project; explain disagreements directly, with dry humor welcome.

Delete code comments; keep only a why the code cannot show.

## Checks

Run Cargo inside [shell.nix](shell.nix). Before submitting code:

- `cargo fmt --check`
- `cargo clippy --quiet --all-targets -- --deny warnings`
- `cargo test -q -- --test-threads=2`

[test-map.json](test-map.json) selects covering suites. A test runs in the map or is
deleted; do not use `#[ignore]`. Simulation tests use `test-watchdog` to abort hangs.
If contention trips it, rerun with fewer threads or on a less loaded machine.

## Probes and profiling

Use the release store's live checkpoint pointer for probes and screenshots. Keep
the tagged envelope and terrain provenance; do not make ad-hoc checkpoint copies
that can outlive format migrations.

For slow frames, use [scripts/profile-game.sh](scripts/profile-game.sh).
`--pid N` observes a running process; without it the script launches and later kills
a target. `--perf` adds a breakdown when available. Run as the graphical-session
user for the real Vulkan client. Profile under contention without stopping other
jobs or waiting for exclusivity, and record load average. Inspect the backend,
GPU utilization and per-thread CPU before attributing a bottleneck.

## Boundaries

Keep this project independent. Reference other projects only as declared, versioned
dependencies, exposing names and versions rather than internals. Give shared services
neutral project-owned names. Exclude deployment-specific paths, addresses, service
or queue names, credentials, camera frames and private renders. Before landing,
inspect the diff for undeclared project references and deployment details.
