//! test-map.json content guard (bothouse#240): the manifest is repo-owned data
//! the botq acceptance gate trusts at landed HEAD, and `botq test-map --check`
//! lints completeness, not content — so without this pin, silently dropping a
//! landing command from the manifest would fail nothing anywhere. Each pinned
//! command maps to a real incident (see the manifest rule's why). Raw substring
//! match, deliberately: the commands contain no JSON escapes, and a parser
//! dependency buys nothing over "the command text is still present".

#[test]
fn manifest_carries_the_full_landing_matrix() {
    let manifest =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../test-map.json"))
            .expect("test-map.json at the repo root");
    for cmd in [
        "cargo fmt --check",
        "cargo clippy --quiet --all-targets -- --deny warnings",
        "cargo build --release -p rl-train",
        "cargo build --release -p rl-demo -p game -p rl-update-ui",
        "cargo test -q -- --test-threads=2",
    ] {
        assert!(
            manifest.contains(cmd),
            "test-map.json no longer carries `{cmd}` — a landing config silently dropped"
        );
    }
}
