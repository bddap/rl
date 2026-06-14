pub mod world;

pub use world::PhysicsWorldPlugin;

use bevy_rapier3d::prelude::TimestepMode;

/// One Bevy tick advances physics by exactly this, split into [`PHYSICS_SUBSTEPS`]
/// solver sub-steps. Shared by production AND every headless test via
/// [`fixed_timestep`], so the three can never drift — a test that ran different
/// physics than the demo/training loop would prove nothing.
pub const PHYSICS_DT: f32 = 1.0 / 64.0;

/// Solver sub-steps per tick. At one sub-step the solver can't converge the floppy
/// legs + soft joint limits resting under the zero policy, so the crab buzzes
/// (issue #18); more sub-steps shrink the per-solve dt and the twitch falls off.
/// But sub-steps cost ~linearly and 4 tanked the realtime demo framerate, so 2 is
/// the balance: airborne angular momentum is fully conserved (1.0×, same as 4) and
/// rest clears the quiet bar, at half the solver cost. The residual rest bounce
/// (~3.5 cm) is the floppy multibody not fully settling in 2 sub-steps — reducing
/// it further is a joint/contact-softness tuning problem, NOT more sub-steps.
pub const PHYSICS_SUBSTEPS: usize = 2;

/// The fixed physics timestep. Use everywhere (main + headless tests) so the
/// simulation is identical and reproducible across all three.
pub fn fixed_timestep() -> TimestepMode {
    TimestepMode::Fixed {
        dt: PHYSICS_DT,
        substeps: PHYSICS_SUBSTEPS,
    }
}
