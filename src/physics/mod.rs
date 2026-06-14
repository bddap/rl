pub mod world;

pub use world::PhysicsWorldPlugin;

use bevy_rapier3d::prelude::TimestepMode;

/// One Bevy tick advances physics by exactly this, split into [`PHYSICS_SUBSTEPS`]
/// solver sub-steps. Shared by production AND every headless test via
/// [`fixed_timestep`], so the three can never drift — a test that ran different
/// physics than the demo/training loop would prove nothing.
pub const PHYSICS_DT: f32 = 1.0 / 64.0;

/// Solver sub-steps per tick. 4, not 1: at one sub-step the solver can't converge
/// the floppy legs resting against their joint limits under the zero policy, so a
/// crab at rest buzzes and bounces (issue #18). More sub-steps shrink the per-solve
/// dt and the rest-twitch falls off sharply (carapace angular speed ~2.3 -> ~1.0
/// rad/s, bounce ~7.5 -> ~3 cm) while the legs still crumple. 4 is the knee — past
/// it the gains shrink and the cost (linear in sub-steps) stops paying off.
pub const PHYSICS_SUBSTEPS: usize = 4;

/// The fixed physics timestep. Use everywhere (main + headless tests) so the
/// simulation is identical and reproducible across all three.
pub fn fixed_timestep() -> TimestepMode {
    TimestepMode::Fixed {
        dt: PHYSICS_DT,
        substeps: PHYSICS_SUBSTEPS,
    }
}
