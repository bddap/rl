//! Headless training-SUCCESS eval — the TRUE measure of the trained crab, distinct from the
//! training reward. The reward is a *proxy* the optimizer maximizes and can diverge from the
//! behaviour we actually want; this eval measures that behaviour DIRECTLY, so it (not a reward
//! curve) is what a human/daemon watches to judge a run.
//!
//! What it does, all HEADLESS (physics + policy rollout, no bevy window — fast + CI-able):
//! - Reuses the demo/train crab+ball scenario unchanged: the shared windowless physics+bot
//!   stack ([`crate::bot::headless::headless_stack`]) spawns the one rig-derived crab at the
//!   env-0 origin exactly as training and the demo do — no reinvented crab/ball/physics.
//! - Places the ball at a FIXED, deterministic FAR distance from the crab (default the far edge
//!   of the training band, [`TARGET_ARENA_HALF`]) via [`CrabTargets`], the SAME single source of
//!   truth the policy observes — and, unlike the demo, never relocates it, so the target is the
//!   same every run.
//! - Drives the loaded policy DETERMINISTICALLY (the policy MEAN, no exploration noise / OU /
//!   σ-widening) for `active_ticks` ticks — the same [`Policy::act`] the demo runs. It measures
//!   the settled policy, not the explorer.
//!
//! Two HONEST numbers, both read off the REAL physics — never a reward term, a curriculum-clamped
//! value, or a finish-time artifact (the gait work has misread proxy metrics repeatedly):
//! 1. **progress_m** — real metres the carapace CLOSED toward the ball: initial carapace→ball
//!    distance minus the CLOSEST it ever got, in true 3D euclidean metres ([`dist_3d`], the same
//!    `d` the grab/reach event is defined on). "Does the crab make progress toward the ball."
//! 2. **total_torque** — the actual applied joint torque summed over the rollout: each active
//!    tick, Σ over joints |clamp(action)|·`drive_torque_ceiling`, i.e. the exact muscle command
//!    [`apply_actions`](crate::bot::actuator::apply_actions) puts on the body. "Done using minimal
//!    torques" — lower is better.

use std::path::Path;

use bevy::prelude::*;

use crate::bot::RESET_GRACE_TICKS;
use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{CrabCarapace, CrabEnvId, CrabJoint};
use crate::bot::headless::{
    HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
};
use crate::bot::sensor::{CrabObservation, CrabTargets};
use crate::bot::{BotSet, CrabSpawns};
use crate::policy::Policy;
use crate::training::curriculum::{
    CURRICULUM_REACH_RADIUS, TARGET_ARENA_HALF, TARGET_Y_MAX, TARGET_Y_MIN,
};
use crate::training::reward::dist_3d;

/// The default FAR target distance (planar metres from the crab's spawn): the far edge of the
/// FIXED training band ([`TARGET_ARENA_HALF`]) — the hardest target still IN the distribution the
/// policy trained on, so it is challenging but reachable and never poses a goal training never
/// saw. Derived from the arena, not a bare literal, so an arena change moves it in lock-step.
pub const DEFAULT_TARGET_DISTANCE_M: f32 = TARGET_ARENA_HALF;

/// The fixed claw-height (world Y) the eval places the ball at: the midpoint of the training
/// target-Y band, so the ball sits at a genuine reach height like the ones training sampled —
/// but fixed (not the training random draw), which is what makes the eval deterministic.
const TARGET_Y: f32 = (TARGET_Y_MIN + TARGET_Y_MAX) / 2.0;

/// The two honest numbers (plus the context to read them) from one deterministic rollout of a
/// loaded policy against a fixed far ball. `progress_m` and `total_torque` are the headline pair;
/// the rest let a human trust them at face value (what the initial/closest/final distance was,
/// whether a real checkpoint drove the crab, how many ticks ran).
#[derive(Debug, Clone, Copy)]
pub struct EvalReport {
    /// PRIMARY — real metres of progress the carapace made toward the ball:
    /// `initial_distance_m − closest_distance_m` (never negative). Higher is better.
    pub progress_m: f32,
    /// SECONDARY — total applied joint torque over the active rollout (N·m summed across joints
    /// and ticks): Σ_ticks Σ_joints |clamp(action)|·ceiling. Lower is better.
    pub total_torque: f32,
    /// Mean applied joint torque per active tick (`total_torque / active_ticks_run`) — the same
    /// quantity as `total_torque`, per-tick, so it stays comparable across different tick counts.
    pub mean_torque_per_tick: f32,
    /// Carapace→ball 3D distance (m) at the start of the active rollout (after the settle drop).
    pub initial_distance_m: f32,
    /// Closest carapace→ball 3D distance (m) reached at any active tick — the best approach.
    pub closest_distance_m: f32,
    /// Carapace→ball 3D distance (m) at the final active tick.
    pub final_distance_m: f32,
    /// The fixed target distance the ball was placed at (planar m from spawn).
    pub target_distance_m: f32,
    /// Whether the crab got within the canonical reach radius ([`CURRICULUM_REACH_RADIUS`]) of the
    /// ball at any point — the same "reached it" event the demo/training use.
    pub reached: bool,
    /// Active ticks actually rolled (excludes the settle window).
    pub active_ticks: u64,
    /// Whether a real checkpoint drove the crab (vs the zero-action rest pose when no/`--` bad
    /// checkpoint loaded) — a `false` here makes the numbers a rest-pose baseline, not a policy.
    pub policy_loaded: bool,
}

/// Fixed knobs for one eval run: where to put the ball and how long to settle before measuring.
#[derive(Resource, Clone, Copy)]
struct EvalConfig {
    target_distance: f32,
    /// Ticks to hold zero actions after spawn so the crab drops and settles onto the ground
    /// before the initial distance is captured — the shared post-respawn settle so the baseline
    /// is a resting crab, not one mid-air. See [`RESET_GRACE_TICKS`].
    settle_ticks: u64,
}

/// Running accumulators for one eval run, read out after the loop finishes.
#[derive(Resource, Default)]
struct EvalState {
    tick: u64,
    target_set: bool,
    /// Carapace→ball distance at the end of settle (the active-rollout baseline).
    initial_dist: f32,
    /// Closest carapace→ball distance seen during the active rollout.
    closest_dist: f32,
    /// Most recent carapace→ball distance (the final distance once the run ends).
    last_dist: f32,
    /// Σ applied joint torque over active ticks (f64 so a long rollout doesn't lose precision).
    torque_sum: f64,
    /// Active ticks that contributed a torque sample.
    torque_ticks: u64,
}

/// Run one deterministic headless eval of the policy in `checkpoint_dir` against a ball fixed at
/// `target_distance` planar metres, for `active_ticks` ticks after a settle window, and return the
/// two honest numbers (+ context). Deterministic: fixed spawn, fixed target, mean-action policy,
/// single-threaded pools + serial schedules, so the same checkpoint yields the same report.
pub fn run_eval(checkpoint_dir: &Path, active_ticks: u64, target_distance: f32) -> EvalReport {
    // Pin every process-global pool to one thread BEFORE building the App, so the sim + the tiny
    // inference matmul run in one fixed float-op order and the report reproduces run-to-run (the
    // same recipe the trainer uses; see `pin_single_thread_pools`).
    pin_single_thread_pools();

    let policy = Policy::load(checkpoint_dir);
    let policy_loaded = policy.is_loaded();

    // The shared windowless physics+bot stack — one standalone env — spawns the rig-derived crab
    // at the env-0 origin exactly as training/demo do (no reinvented scenario).
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
    });
    app.insert_resource(EvalConfig {
        target_distance,
        settle_ticks: RESET_GRACE_TICKS as u64,
    })
    .init_resource::<EvalState>()
    .insert_non_send_resource(policy)
    .add_systems(FixedUpdate, eval_step.in_set(BotSet::Think));

    // Fix ECS system order too, so nothing about the report depends on thread scheduling.
    force_serial_schedules(&mut app);
    app.finish();
    app.cleanup();

    // One physics tick per update: settle first (zero actions, baseline capture), then the active
    // policy rollout. `eval_step` gates settle-vs-active on the tick counter it owns and only
    // increments `torque_ticks` on active ticks — so drive updates until it has recorded exactly
    // `active_ticks` active ticks, rather than a fixed update count: the first update(s) run
    // Startup with no FixedUpdate tick, which would otherwise under-count the rollout by one.
    // The cap (a generous warm-up margin) makes a wiring bug that stalls `eval_step` fail as a
    // short report instead of an infinite loop.
    let settle_ticks = RESET_GRACE_TICKS as u64;
    let max_updates = settle_ticks + active_ticks + 64;
    let mut updates = 0u64;
    while active_torque_ticks(&app) < active_ticks && updates < max_updates {
        app.update();
        updates += 1;
    }

    let state = app
        .world()
        .get_resource::<EvalState>()
        .expect("eval state present");
    let progress_m = (state.initial_dist - state.closest_dist).max(0.0);
    let mean_torque_per_tick = if state.torque_ticks > 0 {
        (state.torque_sum / state.torque_ticks as f64) as f32
    } else {
        0.0
    };
    EvalReport {
        progress_m,
        total_torque: state.torque_sum as f32,
        mean_torque_per_tick,
        initial_distance_m: state.initial_dist,
        closest_distance_m: state.closest_dist,
        final_distance_m: state.last_dist,
        target_distance_m: target_distance,
        reached: state.closest_dist <= CURRICULUM_REACH_RADIUS,
        active_ticks: state.torque_ticks,
        policy_loaded,
    }
}

/// Active ticks `eval_step` has measured so far — the loop drives updates until this reaches the
/// requested `active_ticks`, so warm-up updates that run no FixedUpdate tick don't under-count.
fn active_torque_ticks(app: &App) -> u64 {
    app.world()
        .get_resource::<EvalState>()
        .map(|s| s.torque_ticks)
        .unwrap_or(0)
}

/// The eval's per-tick driver + meter (BotSet::Think, after Sense writes the observation and
/// before Act applies the actions): seed the fixed far target once, drive the policy (zeros while
/// settling), and record the carapace→ball distance + the applied joint torque. One system so the
/// torque it records is exactly the action it just wrote — the same the actuator then applies.
#[allow(clippy::too_many_arguments)]
fn eval_step(
    policy: NonSend<Policy>,
    cfg: Res<EvalConfig>,
    mut state: ResMut<EvalState>,
    spawns: Res<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    joints: Query<(&CrabJoint, &CrabEnvId)>,
) {
    // Seed the FIXED far ball once, before any measurement: a deterministic point at
    // `target_distance` along +x from the env-0 spawn, at the fixed claw height. Written into
    // `CrabTargets` — the single source the observation reads — and never moved again (unlike the
    // demo, which teleports on reach). Seeded here (not at Startup) to dodge the same
    // `spawn_initial_crabs` resize race the demo's `target_ball` seeder avoids.
    // Mark seeded ONLY once the slot is actually written — otherwise a tick where the env-0 target
    // slot isn't sized yet would flip `target_set` with no ball placed, and every later tick would
    // silently measure the distance to the origin (target `None` → ZERO) instead of the ball. In
    // practice `spawn_initial_crabs` (Startup) sizes the slot before this FixedUpdate runs, but
    // gating on the write makes the honest metric impossible to corrupt if that ordering ever moves.
    if !state.target_set
        && let Some(slot) = targets.envs.first_mut()
    {
        let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
        *slot = Some(Vec3::new(origin.x + cfg.target_distance, TARGET_Y, origin.z));
        state.target_set = true;
    }
    // Until the ball is seeded there is nothing meaningful to measure; skip the tick rather than
    // record a bogus distance-to-origin. (Only the very first update(s), before the slot exists.)
    let Some(target) = targets.get(0) else {
        state.tick += 1;
        return;
    };

    let settling = state.tick < cfg.settle_ticks;

    // Drive: hold zeros through the settle drop, then the deterministic policy mean.
    if let Some(a) = actions.envs.first_mut() {
        *a = if settling {
            [0.0; ACTION_SIZE]
        } else {
            obs.envs
                .first()
                .map(|o| policy.act(o))
                .unwrap_or([0.0; ACTION_SIZE])
        };
    }

    // Carapace→ball distance this tick (env 0), the REAL 3D euclidean metres.
    if let Some(cpos) = carapace_q
        .iter()
        .find(|(e, _)| e.0 == 0)
        .map(|(_, t)| t.translation)
        .filter(|p| p.is_finite())
    {
        let d = dist_3d(cpos, target);
        if settling {
            // Track the settling position; the last settle tick leaves the resting baseline.
            state.initial_dist = d;
            state.closest_dist = d;
        } else {
            state.closest_dist = state.closest_dist.min(d);
        }
        state.last_dist = d;
    }

    // Applied joint torque this active tick: exactly what `apply_actions` will put on the body —
    // Σ over env-0 joints of |clamp(action)|·drive_torque_ceiling. Measured on the action just
    // written, reusing the actuator's clamp + per-joint ceiling so it can't drift from what's
    // applied. Settle ticks (zero actions) don't count.
    if !settling && let Some(a) = actions.envs.first() {
        let mut tick_torque = 0.0f32;
        for (joint, env) in joints.iter() {
            if env.0 != 0 {
                continue;
            }
            let raw = a[joint.id.index()];
            let clamped = if raw.is_finite() {
                raw.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            tick_torque += clamped.abs() * joint.id.drive_torque_ceiling();
        }
        state.torque_sum += tick_torque as f64;
        state.torque_ticks += 1;
    }

    state.tick += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default far distance is the far edge of the training band — one source (the arena),
    /// so it can't drift from the distribution the policy trained on.
    #[test]
    fn default_far_distance_is_the_training_band_edge() {
        assert_eq!(DEFAULT_TARGET_DISTANCE_M, TARGET_ARENA_HALF);
        // Both operands are `const`, so pin these as compile-time checks (a `const` block) —
        // the invariant can never regress into a run-only failure.
        const {
            assert!(
                DEFAULT_TARGET_DISTANCE_M > CURRICULUM_REACH_RADIUS,
                "the ball must start FAR — well outside the reach radius"
            );
            // The fixed claw height sits inside the training target-Y band (a genuine reach).
            assert!(TARGET_Y > TARGET_Y_MIN && TARGET_Y < TARGET_Y_MAX);
        }
    }

    /// End-to-end harness invariant on the honest metrics, with NO checkpoint: the zero-action
    /// rest pose applies exactly zero joint torque, so `total_torque` must be EXACTLY 0 — a proxy
    /// or off-by-one that leaked settle/idle torque in would fail here. The crab only settles in
    /// place (no policy driving it), so it makes no real progress toward the far ball. Also pins
    /// that the ball is placed at the requested far distance and the report is well-formed.
    ///
    /// Builds a headless bevy+rapier App, so it is `#[ignore]` by default: run with
    /// `cargo test --release -p crab-world -- --ignored eval`.
    #[test]
    #[ignore = "builds a headless bevy+rapier App; run with --ignored"]
    fn rest_pose_has_zero_torque_and_no_progress() {
        let dir = std::env::temp_dir().join(format!("rl-eval-restpose-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let r = run_eval(&dir, 200, DEFAULT_TARGET_DISTANCE_M);

        assert!(!r.policy_loaded, "an empty dir loads no policy (rest pose)");
        // Zero-action rest pose ⇒ every commanded torque is 0 ⇒ total is EXACTLY 0.
        assert_eq!(
            r.total_torque, 0.0,
            "the rest pose applies no joint torque, so total_torque must be exactly 0"
        );
        assert_eq!(r.mean_torque_per_tick, 0.0);
        assert_eq!(r.active_ticks, 200, "all active ticks are measured");
        assert!(
            r.initial_distance_m.is_finite() && r.closest_distance_m.is_finite(),
            "distances are real finite metres"
        );
        // The ball is far: the crab spawns ~target_distance away and, undriven, can't reach it.
        assert!(
            r.initial_distance_m > CURRICULUM_REACH_RADIUS,
            "the ball starts far outside reach ({} m)",
            r.initial_distance_m
        );
        assert!(!r.reached, "a rest-pose crab never reaches a far ball");
        // progress = initial − closest ≥ 0 by construction, and an undriven crab that only
        // settles in place closes at most a small amount (no locomotion).
        assert!(
            (0.0..1.0).contains(&r.progress_m),
            "rest-pose progress should be ~0, got {} m",
            r.progress_m
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
