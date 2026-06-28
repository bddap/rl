//! Bakes build provenance — the commit sha + UTC build time — into crab-world so the
//! rendering binaries can show a subtle always-on corner stamp (see src/build_info.rs).
//! A binary that failed to redeploy then shows an OLD sha/date, making a stale deploy
//! obvious at a glance — the whole reason the stamp exists.

use std::path::Path;
use std::process::Command;

fn main() {
    let sha = run("git", &["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // A trailing `+` flags an uncommitted working tree, so a hand-built dev binary is never
    // mistaken for the pristine commit it sits on.
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false);
    let sha = if dirty { format!("{sha}+") } else { sha };
    println!("cargo:rustc-env=RL_BUILD_SHA={sha}");

    // Format the build time here (std has no date formatting); coreutils `date` is always
    // on the build PATH. Fall back to the raw epoch if it somehow isn't.
    let date = run("date", &["-u", "+%Y-%m-%d %H:%M UTC"]).unwrap_or_else(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!("epoch {secs} UTC")
    });
    println!("cargo:rustc-env=RL_BUILD_DATE={date}");

    // Re-run ONLY when the checked-out commit moves — not every build. Keying on the git
    // ref files (which a commit / `reset --hard` rewrites) keeps the stamp tracking the
    // built commit no matter which crate the commit touched, without forcing a recompile
    // of crab-world on every incremental build (which a build-time timestamp otherwise would).
    for p in git_ref_paths() {
        println!("cargo:rerun-if-changed={p}");
    }
}

/// The git ref files whose change means "the commit moved": HEAD, the loose ref it points
/// at (if any), and packed-refs (covers a packed branch with no loose file). Only paths
/// that exist are emitted — a missing rerun-if-changed path would force an always-rerun.
fn git_ref_paths() -> Vec<String> {
    let mut paths = Vec::new();
    let mut push_if_exists = |p: Option<String>| {
        if let Some(p) = p.filter(|p| Path::new(p).exists()) {
            paths.push(p);
        }
    };
    push_if_exists(run("git", &["rev-parse", "--git-path", "HEAD"]));
    if let Some(reff) = run("git", &["symbolic-ref", "-q", "HEAD"]) {
        push_if_exists(run("git", &["rev-parse", "--git-path", &reff]));
    }
    push_if_exists(run("git", &["rev-parse", "--git-path", "packed-refs"]));
    paths
}

/// Run a command, returning its trimmed stdout, or `None` on spawn failure / nonzero exit /
/// empty output (e.g. building outside a git checkout — the stamp then reads "unknown").
fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}
