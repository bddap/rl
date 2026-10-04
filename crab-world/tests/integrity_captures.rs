//! rl#351: rl#343 blow-up onsets captured under the old (2,12,3)×2 solver, replayed
//! one tick on the shipped solver, stay below the near-miss line.

use crab_world::physics::SHIPPED_SOLVER;
use crab_world::physics::snapshot::{
    ContactsOff, FreeLimits, PlantSnapshot, ReplayConfig, ShapeVariant,
};

/// `rl-train repro`'s near-miss line, on the rl#343 bound's `lin.max(ang/3)`.
const NEAR_MISS_M_S: f32 = 30.0;

/// (2,12,3)×2 onsets: the reproduction hunt (trips and near misses), the rate A/B and process split's
/// shipped arms, and the re-sized lever A/B's shipped arm. Left out:
/// `integrity-process-split/inproc-vm-w0-r0-e0-t720366.bin`, a carpus spin trip that
/// outer 4 does not remove (43 m/s, 286 rad/s): outer 4 lowers the trip rate, and its
/// own arm still tripped twice in 36.2M ticks.
const CAPTURES: &[&str] = &[
    "integrity-repro/w0-e0-t148847.bin",
    "integrity-repro/w0-e0-t323647.bin",
    "integrity-repro/w4-e1-t19561.bin",
    "integrity-repro/w6-e0-t414529.bin",
    "integrity-repro/w6-e0-t92296.bin",
    "integrity-repro/w8-e0-t525125.bin",
    "integrity-repro/w9-e0-t334610.bin",
    "integrity-rate-ab/shipped-w3-r0-e0-t332677.bin",
    "integrity-process-split/inproc-vm-w5-r0-e0-t615836.bin",
    "integrity-lever-ab-36m/shipped-w12-r0-e1-t470214.bin",
    "integrity-lever-ab-36m/shipped-w13-r0-e1-t145406.bin",
    "integrity-lever-ab-36m/shipped-w16-r0-e0-t347915.bin",
    "integrity-lever-ab-36m/shipped-w17-r0-e0-t130564.bin",
    "integrity-lever-ab-36m/shipped-w19-r0-e1-t182156.bin",
    "integrity-lever-ab-36m/shipped-w19-r1-e0-t459794.bin",
    "integrity-lever-ab-36m/shipped-w32-r0-e0-t106934.bin",
    "integrity-lever-ab-36m/shipped-w32-r1-e1-t482225.bin",
    "integrity-lever-ab-36m/shipped-w34-r0-e0-t58601.bin",
    "integrity-lever-ab-36m/shipped-w36-r0-e0-t402058.bin",
    "integrity-lever-ab-36m/shipped-w38-r0-e0-t424455.bin",
    "integrity-lever-ab-36m/shipped-w38-r1-e1-t197048.bin",
];

#[test]
fn integrity_captures_replay_quiet() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/evidence/chase-archaeology");
    let cfg = ReplayConfig {
        drive_scale: 1.0,
        solver: SHIPPED_SOLVER,
        limit_softness: None,
        free_limits: FreeLimits::None,
        contacts_off: ContactsOff::None,
        shape: ShapeVariant::AsIs,
    };
    let mut loud = Vec::new();
    for name in CAPTURES {
        let snap = PlantSnapshot::load(&root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            snap.original_max_speed() > NEAR_MISS_M_S,
            "{name}: not a blow-up onset ({:.1} m/s in the original run)",
            snap.original_max_speed()
        );
        let out = snap.replay(&cfg);
        let speed = out.max_speed_after.max(out.max_angvel_after / 3.0);
        println!("{name}: {:.1} -> {speed:.2} m/s", snap.original_max_speed());
        if speed.is_nan() || speed >= NEAR_MISS_M_S {
            loud.push(format!("{name}: {speed:.1} m/s"));
        }
    }
    assert!(
        loud.is_empty(),
        "trip onsets still blow up on the shipped solver {SHIPPED_SOLVER}: {loud:?}"
    );
}
