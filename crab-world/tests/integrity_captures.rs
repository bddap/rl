//! rl#351: every rl#343 blow-up onset committed under
//! `docs/evidence/chase-archaeology/integrity-*/`, replayed one tick on the shipped
//! plant, stays below the near-miss line.

use crab_world::physics::SHIPPED_SOLVER;
use crab_world::physics::snapshot::{
    ContactsOff, FreeLimits, PlantSnapshot, ReplayConfig, ShapeVariant, Vec3,
};

/// `rl-train repro`'s near-miss line, on the rl#343 bound's `lin.max(ang/3)`.
const NEAR_MISS_M_S: f32 = 30.0;

/// Bumped by hand with each onset committed, so a lost capture fails loudly.
const ONSETS: usize = 145;

fn onsets(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let entries = |dir: &std::path::Path| {
        std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|e| e.expect("readable evidence dir").path())
            .collect::<Vec<_>>()
    };
    let mut out: Vec<_> = entries(root)
        .into_iter()
        .filter(|d| {
            d.is_dir()
                && d.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("integrity-"))
        })
        .flat_map(|d| entries(&d))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    out.sort();
    out
}

#[test]
fn integrity_captures_replay_quiet() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/evidence/chase-archaeology");
    let captures = onsets(&root);
    assert_eq!(captures.len(), ONSETS, "onsets under {}", root.display());
    let cfg = ReplayConfig {
        drive_scale: 1.0,
        solver: SHIPPED_SOLVER,
        limit_softness: None,
        free_limits: FreeLimits::None,
        contacts_off: ContactsOff::None,
        shape: ShapeVariant::AsIs,
    };
    let mut loud = Vec::new();
    for path in &captures {
        let name = path.strip_prefix(&root).unwrap_or(path).display();
        let snap = PlantSnapshot::load(path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let original = snap
            .expected
            .iter()
            .map(|(v, w)| Vec3::from(*v).length().max(Vec3::from(*w).length() / 3.0))
            .fold(0.0, f32::max);
        assert!(
            original > NEAR_MISS_M_S,
            "{name}: not a blow-up onset ({original:.1} m/s in the original run)"
        );
        let out = snap.replay(&cfg);
        let speed = out.max_speed_after.max(out.max_angvel_after / 3.0);
        println!("{name}: {original:.1} -> {speed:.2} m/s");
        if speed.is_nan() || speed >= NEAR_MISS_M_S {
            loud.push(format!("{name}: {speed:.1} m/s"));
        }
    }
    assert!(
        loud.is_empty(),
        "trip onsets still blow up on the shipped plant ({SHIPPED_SOLVER}): {loud:?}"
    );
}
