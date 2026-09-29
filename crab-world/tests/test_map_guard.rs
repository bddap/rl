//! test-map.json guards: the manifest is the landing gate's only source of which
//! commands run, and a resolver lints its completeness, not its content — so without
//! the pin below, silently dropping a landing command would fail nothing. Raw
//! substring match: the commands contain no JSON escapes.

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
        // The rl#411 platform-freedom check: game-web compiles the whole web leaf
        // (net + crab-world, render on) for wasm32; dropping it re-opens silent
        // web-build rot from dep-side edits.
        "cargo check --target wasm32-unknown-unknown -p net-proto -p net-link -p game-web",
    ] {
        assert!(
            manifest.contains(cmd),
            "test-map.json no longer carries `{cmd}` — a landing config silently dropped"
        );
    }
}

#[test]
fn no_test_is_ignored() {
    fn visit(dir: &std::path::Path, hits: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("readable repo dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let kind = entry.file_type().expect("file type");
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    visit(&path, hits);
                }
            } else if name.ends_with(".rs") {
                let text = std::fs::read_to_string(&path).expect("utf-8 source");
                for (i, line) in text.lines().enumerate() {
                    let line = line.trim_start();
                    if line.starts_with("#[ignore")
                        || (line.starts_with("#[cfg_attr") && line.contains("ignore"))
                    {
                        hits.push(format!("{}:{}", path.display(), i + 1));
                    }
                }
            }
        }
    }
    let mut hits = Vec::new();
    visit(std::path::Path::new(".."), &mut hits);
    assert!(
        hits.is_empty(),
        "ignored tests are not allowed — wire each into a test-map.json rule or delete it: {hits:?}"
    );
}
