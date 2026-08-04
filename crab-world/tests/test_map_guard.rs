//! test-map.json content guard (bothouse#240): the manifest is repo-owned data
//! the botq acceptance gate trusts at landed HEAD, and `botq test-map --check`
//! lints completeness, not content — so without this pin, silently dropping a
//! landing command from the manifest would fail nothing anywhere. Each pinned
//! command maps to a real incident (see the manifest rule's why). Raw substring
//! match, deliberately: the commands contain no JSON escapes, and a parser
//! dependency buys nothing over "the command text is still present".

#[test]
fn manifest_carries_the_full_landing_matrix() {
    // Resolved at RUNTIME from the test cwd (cargo runs tests in the package dir),
    // not via CARGO_MANIFEST_DIR baked at compile time: kache restores test
    // binaries across worktrees, and a baked absolute path would read the
    // BUILDING worktree's manifest, not the one under test.
    let manifest =
        std::fs::read_to_string("../test-map.json").expect("test-map.json at the repo root");
    for cmd in [
        "cargo fmt --check",
        "cargo clippy --quiet --all-targets -- --deny warnings",
        "cargo build --release -p rl-train",
        "cargo build --release -p rl-demo -p game -p rl-update-ui",
        "cargo test -q -- --test-threads=2",
        // The eval seam's end-to-end physical run (rl#341 S2-5) — the one executed
        // test of run_eval's real worlds; dropping it re-opens the untested
        // measurement path.
        "cargo test -q -p crab-world --release --lib rest_pose_has_zero_torque_and_no_progress",
    ] {
        assert!(
            manifest.contains(cmd),
            "test-map.json no longer carries `{cmd}` — a landing config silently dropped"
        );
    }
}
