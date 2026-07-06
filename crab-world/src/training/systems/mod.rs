
mod lifecycle;
mod state;
mod step;

/// Re-exported for `reward`'s calibration tests and for `rl-train`'s eval `--ticks`
/// default (one episode horizon); `finalize_transitions` uses the [`lifecycle`]-local
/// constant directly.
pub use lifecycle::MAX_EPISODE_TICKS;
pub(crate) use lifecycle::reset_crab;
pub use state::STEPS_PER_ROLLOUT;
pub(crate) use state::{HorizonOutput, HorizonRequest, TrainingState};
pub(crate) use step::brain_step;
