mod lifecycle;
mod state;
mod step;
mod trace;

/// Re-exported for `reward`'s calibration tests and `rl-train`'s eval horizon (one
/// episode).
pub(crate) use lifecycle::INTEGRITY_VIOLATION;
pub use lifecycle::MAX_EPISODE_TICKS;
pub(crate) use lifecycle::reset_crab;
pub use state::STEPS_PER_ROLLOUT;
pub(crate) use state::{HorizonOutput, HorizonRequest, LearnerState, StepTelemetry, WorkerState};
pub(crate) use step::brain_step;
