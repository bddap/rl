//! rl#116 armed-render smoke: step the GCR client's armed NN-crab world with
//! `Visuals(true)` — skin, repose publisher, and pose sentinel live — entirely
//! headless, and require the crab to stay finite and near its spawn.
//!
//! This is the test the play-day crash proved was missing: the render-side
//! cosmetic mutation only fired with `Visuals` on, and every headless test ran
//! `Visuals(false)`. A reintroduced render-writes-physics bug fails here twice
//! over: the pose sentinel panics naming the write, and the blowup would break
//! the bounds below. Lives in `game` because this crate always arms the
//! `render` feature, so plain workspace `cargo test` runs it.

use crab_world::Visuals;
use crab_world::policy::Policy;
use net::external_crab::run_headless_probe;

#[test]
fn armed_visual_crab_stays_finite_and_grounded() {
    // Rest-pose policy on purpose: the guard is about the render/physics seam,
    // not the brain.
    let ticks = 256;
    let samples = run_headless_probe(Policy::rest(), 0x116, ticks, 1, Visuals(true));
    // One sample per FIXED step; the first app.update() has a zero delta and
    // fires no FixedUpdate, so N updates yield N-1 steps.
    assert!(
        samples.len() as u64 >= ticks - 1,
        "expected ~one sample per tick, got {} of {ticks}",
        samples.len()
    );

    // The probe reports (0,0,0) when no env-0 carapace exists, which every bound
    // below accepts — so first prove the crab is actually there: a spawned, settled
    // carapace rests visibly above the ground plane.
    assert!(
        samples
            .iter()
            .filter(|s| s.tick >= 64)
            .all(|s| s.carapace_y > 0.05),
        "no settled carapace above ground — the armed crab never spawned (or fell \
         through the world), so the smoke test would otherwise pass vacuously"
    );

    for s in &samples {
        assert!(
            s.carapace_arena_x.is_finite()
                && s.carapace_y.is_finite()
                && s.carapace_arena_z.is_finite(),
            "tick {}: carapace went non-finite — the rl#116 failure shape",
            s.tick
        );
        assert!(
            s.carapace_arena_x.abs() < 20.0
                && s.carapace_arena_z.abs() < 20.0
                && s.carapace_y > -2.0
                && s.carapace_y < 5.0,
            "tick {}: carapace at ({}, {}, {}) — a rest-pose crab teleporting away from \
             spawn means something is writing rapier-driven Transforms (rl#116)",
            s.tick,
            s.carapace_arena_x,
            s.carapace_y,
            s.carapace_arena_z,
        );
    }
}
