//! rl#116 armed-render smoke: step the GCR client's armed NN-crab world with
//! `Visuals(true)` — skin and pose sentinel live — entirely headless, and require
//! the crab to settle, fall asleep, and stay asleep.
//!
//! This is the test the play-day crash proved was missing: the render-side
//! cosmetic mutation only fired with `Visuals` on, and every headless test ran
//! `Visuals(false)`. A reintroduced render-writes-physics bug fails here twice
//! over: the pose sentinel panics naming the write, and the woken/teleported
//! body breaks the rest bounds below. Lives in `game` because this crate always
//! arms the `render` feature, so plain workspace `cargo test` runs it.
//!
//! FLAT ground on purpose (rl#406): on the canonical GCR bake this seed's spawn
//! locale is a ~35° mountainside where a zero-drive crab legitimately slides —
//! ~12-18 m in 256 ticks, with the exact trajectory chaos-sensitive to any float
//! perturbation (sally.glb presence flipped the verdict by spawning the skin's
//! visual entities). Terrain sliding is physics, not a render-seam signal; on a
//! plane the rest contract is exact — settle, sleep (rl#392's bit-exact rest),
//! near-zero drift — so every bound below has real margin in every environment,
//! skin asset present or not. This is also the one test that pins sleep
//! engagement in the ARMED-RENDER feature graph (the render build realizes
//! louder rest noise than headless-only — see `CRAB_SETTLE_EXTRA_ITERATIONS`).

use crab_world::Visuals;
use crab_world::policy::Policy;
use crab_world::terrain::TerrainGrid;
use net::probe::run_headless_probe;

// rl#282: this probe (~3 s normally) has wedged indefinitely (0% CPU,
// futex_wait, 45+ min) under trainer saturation — NOT a wgpu device request:
// `headless_stack` passes `backends: None`, so bevy skips renderer init
// entirely. The stall watchdog bounds it with a loud abort.
test_watchdog::arm!();

#[test]
fn armed_visual_crab_stays_finite_and_grounded() {
    // Rest-pose policy on purpose: the guard is about the render/physics seam,
    // not the brain.
    let ticks = 512;
    // The rl#305 randomized spawn locale can land anywhere on the canonical
    // tile, so the flat fixture spans the same footprint — the heightfield
    // collider must exist under whatever (x, z) this seed draws.
    let gcr = TerrainGrid::gcr();
    let grid = std::sync::Arc::new(TerrainGrid::flat(gcr.extent_x().max(gcr.extent_z()) / 2.0));
    let samples = run_headless_probe(Policy::rest(), 0x116, ticks, 1, Visuals(true), grid);
    // The probe pumps exactly one fixed step per sim tick (parked auto-pump).
    assert!(
        samples.len() as u64 >= ticks - 1,
        "expected ~one sample per tick, got {} of {ticks}",
        samples.len()
    );

    // The probe reports zeros when no env-0 carapace exists, which every bound
    // below accepts — so first prove the crab is actually there: a spawned, settled
    // carapace rests visibly above the ground plane.
    assert!(
        samples
            .iter()
            .filter(|s| s.tick >= 64)
            .all(|s| s.carapace_above_ground > 0.05),
        "no settled carapace above ground — the armed crab never spawned (or fell \
         through the world), so the smoke test would otherwise pass vacuously"
    );

    // Drift is measured from the crab's OWN sim spawn: the run layout randomizes
    // the spawn locale per seed (rl#305). Same seed ⇒ the same draw the probe's
    // sim made. Measured rest behavior on the plane: total drift < 0.4 m,
    // above-ground 0.4-0.65 m — the 2 m caps are severalfold margin, not tuning.
    let (spawn_x, spawn_z) = {
        use net::sim::{PlayerId, Sim};
        Sim::new(0x116, &[PlayerId(0)]).crabs()[0].pos().to_meters()
    };
    for s in &samples {
        assert!(
            s.carapace_x.is_finite()
                && s.carapace_above_ground.is_finite()
                && s.carapace_z.is_finite(),
            "tick {}: carapace went non-finite — the rl#116 failure shape",
            s.tick
        );
        assert!(
            (s.carapace_x - spawn_x).abs() < 2.0
                && (s.carapace_z - spawn_z).abs() < 2.0
                && s.carapace_above_ground > -2.0
                && s.carapace_above_ground < 2.0,
            "tick {}: carapace at ({}, {} above ground, {}) vs spawn ({spawn_x}, \
             {spawn_z}) — a rest-pose crab on flat ground leaving its spawn means \
             something is writing rapier-driven Transforms (rl#116)",
            s.tick,
            s.carapace_x,
            s.carapace_above_ground,
            s.carapace_z,
        );
    }

    // The rl#392 rest contract in the armed-render graph: a zero-drive crab on a
    // plane must retire into rapier sleep, and nothing may wake it — a wake with
    // no drive is exactly a foreign poke at the physics body (rl#116). Both
    // asset states measured asleep by tick 237/274; 448 is the margin, not a fit.
    let first_asleep = samples
        .iter()
        .find(|s| s.crab_asleep)
        .map(|s| s.tick)
        .expect(
            "the rest crab never fell asleep — sleep is not engaging in the armed-render graph",
        );
    assert!(
        first_asleep <= 448,
        "rest crab still awake at tick {first_asleep} — sleep engagement regressed \
         in the armed-render graph (rl#392 gates, CRAB_SETTLE_EXTRA_ITERATIONS)"
    );
    for s in samples.iter().filter(|s| s.tick >= first_asleep) {
        assert!(
            s.crab_asleep,
            "tick {}: crab woke after sleeping at {first_asleep} with zero drive — \
             something poked the physics body (rl#116)",
            s.tick
        );
    }
}
