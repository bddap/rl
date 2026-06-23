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
- `cargo test -q`
