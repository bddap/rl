//! The RL training loop, integrated with the Bevy game loop as ECS systems, split three ways:
//! [`state`] owns [`TrainingState`] and the per-horizon protocol; [`step`] runs the per-tick
//! Sense→Think→Act ([`brain_step`]); [`lifecycle`] holds the per-env episode machine and the
//! reward/terminal finalize plus the [`reset_crab`]/[`save_on_exit`] systems.

mod lifecycle;
mod state;
mod step;

/// Re-exported for `reward`'s calibration tests and for `rl-train`'s eval `--ticks`
/// default (one episode horizon); `finalize_transitions` uses the [`lifecycle`]-local
/// constant directly.
pub use lifecycle::MAX_EPISODE_TICKS;
pub(crate) use lifecycle::{reset_crab, save_on_exit};
pub use state::STEPS_PER_ROLLOUT;
pub(crate) use state::{HorizonOutput, HorizonRequest, TrainingState};
pub(crate) use step::brain_step;
