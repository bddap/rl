//! rl#116 belt-and-braces: fail `cargo test` if any module outside the physics
//! side queries `&mut Transform` on a rapier crab body part. The runtime pose
//! sentinel catches such a write in any world it runs in; this static gate
//! catches it at test time even for systems only registered in the windowed
//! app that no test steps.
//!
//! Known gaps (the sentinel covers these at runtime): `world.get_mut::<Transform>`,
//! `Mut<Transform>` system params, queries hidden behind type aliases, queries
//! reading a marker as data (`(&mut Transform, &CrabCarapace)` with no `With<>`),
//! and a comment containing `>` inside a multi-line generic span (it truncates the
//! span early). The scan sees only the OPENERS' generic spans (path-qualified names
//! inside a span are normalized to their last segment, so `bevy::prelude::Transform`
//! still matches). The scanner itself is self-tested below — a gate whose detector
//! rots passes vacuously forever.

use std::path::{Path, PathBuf};

const BODY_MARKERS: [&str; 4] = ["CrabBodyPart", "CrabCarapace", "CrabClawTip", "CrabJoint"];

/// Paths (workspace-relative prefixes) allowed to take `&mut Transform` on a body part:
/// - `crab-world/src/bot/`: physics owns the bodies (spawn, rescue, reset tests) —
///   EXCEPT `bot/skin/` (see `DENIED`).
/// - `crab-world/src/training/`: test-only reset teleports (all under `#[cfg(test)]`,
///   headless `Visuals(false)` worlds; the scan can't see cfg, so the dir is listed).
/// - `net/src/external_crab.rs`: the GCR bridge. Its rl#240 recenter is a sanctioned
///   physics teleport riding the `PoseSentinelSet` lane (ordered after the sentinel,
///   consumed by the same tick's SyncBackend).
/// - `crab-world/src/eval.rs`: the pace probe's `pace_recenter` (rl#280) — the same
///   rl#240 recenter teleport in the eval's headless `Visuals(false)` worlds,
///   consumed by the same tick's SyncBackend.
///
/// Rendering has NO exception: since rl#274 every render consumer reads the sampled
/// `CrabRenderPose` overlay — the remote-adopt puppet-write carve-out this list used to
/// hold for `net/src/render/articulation.rs` is gone.
const ALLOWED: [&str; 4] = [
    "crab-world/src/bot/",
    "crab-world/src/training/",
    "net/src/external_crab.rs",
    "crab-world/src/eval.rs",
];

/// Carve-outs from `ALLOWED`, checked first: `bot/skin/` is the render-side cosmetic
/// path — the 931936a9 offender's home — and must never write body-part Transforms
/// even though it lives under the physics-owned `bot/` prefix.
const DENIED: [&str; 1] = ["crab-world/src/bot/skin/"];

#[test]
fn only_physics_takes_mut_transform_on_crab_body_parts() {
    let root = workspace_root();
    let mut violations = Vec::new();
    for krate in workspace_members(&root) {
        for file in rs_files(&root.join(&krate).join("src")) {
            let rel = file
                .strip_prefix(&root)
                .expect("scanned file under root")
                .to_string_lossy()
                .replace('\\', "/");
            if !DENIED.iter().any(|d| rel.starts_with(d))
                && ALLOWED.iter().any(|a| rel.starts_with(a))
            {
                continue;
            }
            let src = std::fs::read_to_string(&file).expect("read source");
            for span in scan_source(&src) {
                violations.push(format!("{rel}: {span}"));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "rl#116: `&mut Transform` on a rapier-driven crab body part outside the physics \
         side. bevy_rapier syncs changed Transforms back into the body — write render-only \
         proxies (skin bones / CrabSkinRepose) instead, or extend the documented allowlist \
         in this test if the write genuinely belongs to physics.\n{}",
        violations.join("\n")
    );
}

/// Violating spans in one source file: a query-shaped generic span taking
/// `&mut Transform` filtered `With<a body marker>`.
fn scan_source(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for span in query_generic_spans(src) {
        let span = strip_path_qualifiers(&span);
        if span.contains("mut Transform")
            && with_filters(&span)
                .iter()
                .any(|w| BODY_MARKERS.iter().any(|m| w.contains(m)))
        {
            out.push(condense(&span));
        }
    }
    out
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("game/ sits in the workspace root")
        .to_path_buf()
}

/// Every workspace member with a `src/` dir, from the root Cargo.toml — a new crate
/// is scanned by default instead of silently skipped by a hardcoded list.
fn workspace_members(root: &Path) -> Vec<String> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("root Cargo.toml");
    let at = manifest.find("members").expect("members key");
    let list_end = manifest[at..].find(']').expect("members list closed") + at;
    let members: Vec<String> = manifest[at..list_end]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect();
    assert!(
        members.len() >= 5,
        "workspace member parse broke — got {members:?}"
    );
    members
}

fn rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(rs_files(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out
}

/// The generic argument span of every query-shaped opener, covering the `Query`
/// system param and its turbofish, the world query methods, and bevy's
/// single-result params.
const OPENERS: [&str; 6] = [
    "Query<",
    "Query::<",
    "query::<",
    "query_filtered::<",
    "Single<",
    "Populated<",
];

fn query_generic_spans(src: &str) -> Vec<String> {
    let mut spans = Vec::new();
    for opener in OPENERS {
        let mut from = 0;
        while let Some(at) = src[from..].find(opener) {
            let start = from + at + opener.len();
            if let Some(span) = balanced_angle_span(&src[start..]) {
                spans.push(span);
            }
            from = start;
        }
    }
    spans
}

fn balanced_angle_span(rest: &str) -> Option<String> {
    let mut depth = 1usize;
    for (i, c) in rest.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// The contents of each `With<…>` inside a query span (`Without<…>` does not match).
fn with_filters(span: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = span[from..].find("With<") {
        let start = from + at;
        let preceded_by_ident = start > 0
            && span[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let inner_start = start + "With<".len();
        if !preceded_by_ident && let Some(inner) = balanced_angle_span(&span[inner_start..]) {
            out.push(inner);
        }
        from = inner_start;
    }
    out
}

fn condense(span: &str) -> String {
    span.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `bevy::prelude::Transform` → `Transform`: drop every `ident::` prefix so
/// path-qualified spellings can't slip past the substring matches.
fn strip_path_qualifiers(span: &str) -> String {
    let mut out = String::with_capacity(span.len());
    let mut token_start = out.len();
    for c in span.chars() {
        if c.is_alphanumeric() || c == '_' {
            out.push(c);
        } else if c == ':' && out.ends_with(':') {
            out.truncate(token_start);
        } else {
            if c == ':' {
                out.push(c);
                continue;
            }
            out.push(c);
            token_start = out.len();
        }
    }
    out
}

/// The detector detects: without these, an under-matching regression makes the
/// gate above pass vacuously forever (the starving-gate failure mode).
#[test]
fn scanner_catches_known_violation_shapes() {
    let caught = [
        // The incident's literal shape.
        "fn bad(mut q: Query<&mut Transform, With<CrabBodyPart>>) {}",
        // Fully path-qualified spelling.
        "fn bad(mut q: bevy::prelude::Query<&mut bevy::prelude::Transform, \
         bevy::prelude::With<crab_world::bot::body::CrabCarapace>>) {}",
        // Direct world query.
        "let mut q = world.query_filtered::<&mut Transform, With<CrabClawTip>>();",
        // Compound filter, extra data, multi-line.
        "fn bad(q: Query<\n    (&mut Transform, &CrabEnvId),\n    (With<CrabJoint>, \
         Without<BoneDrive>),\n>) {}",
        // bevy single-result system param.
        "fn bad(t: Single<&mut Transform, With<CrabCarapace>>) {}",
    ];
    for src in caught {
        assert!(
            !scan_source(src).is_empty(),
            "scanner missed a violation it must catch: {src}"
        );
    }
}

#[test]
fn scanner_ignores_legitimate_shapes() {
    let clean = [
        // Read-only on a body part is fine.
        "fn ok(q: Query<&Transform, With<CrabCarapace>>) {}",
        // Mutating a non-body entity, body marker only excluded.
        "fn ok(q: Query<&mut Transform, (With<TargetBall>, Without<CrabClawTip>)>) {}",
        // GlobalTransform is not Transform.
        "fn ok(q: Query<&mut GlobalTransform, With<CrabBodyPart>>) {}",
        // Body marker in another query in the same fn, mut Transform elsewhere.
        "fn ok(a: Query<&mut Transform, With<Camera3d>>, b: Query<&Velocity, \
         With<CrabBodyPart>>) {}",
    ];
    for src in clean {
        assert!(
            scan_source(src).is_empty(),
            "scanner false-positived on legitimate code: {src}"
        );
    }
}
