
use std::path::Path;

use bevy::prelude::*;

use crate::bot::RESET_GRACE_TICKS;
use crate::bot::actuator::{ACTION_SIZE, CrabActions, applied_torque};
use crate::bot::body::{CrabCarapace, CrabEnvId, CrabJoint};
use crate::bot::headless::{
    HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
};
use crate::bot::sensor::{CrabObservation, CrabTargets};
use crate::bot::{BotSet, CrabSpawns};
use crate::policy::Policy;
use crate::training::reward::dist_3d;
use crate::training::targets::{REACH_RADIUS, TARGET_ARENA_HALF, TARGET_Y_MAX, TARGET_Y_MIN};

pub const DEFAULT_TARGET_DISTANCE_M: f32 = TARGET_ARENA_HALF;

/// The default eval episode length — one training episode horizon (~23 s of crab time
/// at 64 Hz), enough for a working gait to traverse a far target. The (ticks, distance)
/// PAIR defines "the chase eval"; every judge — the CLI (release gate, monitor) and the
/// trainer's keep-best gate — must take its defaults from HERE or the metric forks.
pub const DEFAULT_EVAL_TICKS: u64 = crate::training::systems::MAX_EPISODE_TICKS as u64;

const TARGET_Y: f32 = (TARGET_Y_MIN + TARGET_Y_MAX) / 2.0;

#[derive(Debug, Clone, Copy)]
pub struct EvalReport {
    pub progress_m: f32,
    pub total_torque: f32,
    pub mean_torque_per_tick: f32,
    pub initial_distance_m: f32,
    pub closest_distance_m: f32,
    pub final_distance_m: f32,
    pub target_distance_m: f32,
    pub reached: bool,
    pub active_ticks: u64,
    pub policy_loaded: bool,
}

#[derive(Resource, Clone, Copy)]
struct EvalConfig {
    target_distance: f32,
    settle_ticks: u64,
}

#[derive(Resource, Default)]
struct EvalState {
    tick: u64,
    target_set: bool,
    initial_dist: f32,
    closest_dist: f32,
    last_dist: f32,
    torque_sum: f64,
    torque_ticks: u64,
}

pub fn run_eval(
    _body_gate: crate::mesh_fallback::BodyGate,
    checkpoint_dir: &Path,
    active_ticks: u64,
    target_distance: f32,
) -> Result<EvalReport, String> {
    pin_single_thread_pools();

    // Classify BEFORE judging (the same one classifier every arming surface uses):
    // an eval of a checkpoint the runtime would refuse to arm must be a refusal, not a
    // rest-pose baseline quietly printed as the run's training progress.
    match crate::policy::checkpoint_fits_rig(checkpoint_dir) {
        crate::policy::RigFit::Ok | crate::policy::RigFit::Missing => {}
        crate::policy::RigFit::Refused(why) => {
            return Err(format!("checkpoint at {} refused: {why}", checkpoint_dir.display()));
        }
        crate::policy::RigFit::Mismatch(dims) => {
            return Err(format!(
                "checkpoint at {} was built for a different rig ({}/{} obs/act)",
                checkpoint_dir.display(),
                dims.obs,
                dims.action,
            ));
        }
    }

    let policy = Policy::load(checkpoint_dir);
    let policy_loaded = policy.is_loaded();

    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        arena: crate::physics::Arena::WalledBox,
    });
    app.insert_resource(EvalConfig {
        target_distance,
        settle_ticks: RESET_GRACE_TICKS as u64,
    })
    .init_resource::<EvalState>()
    .insert_non_send_resource(policy)
    .add_systems(FixedUpdate, eval_step.in_set(BotSet::Think));

    force_serial_schedules(&mut app);
    app.finish();
    app.cleanup();

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
    Ok(EvalReport {
        progress_m,
        total_torque: state.torque_sum as f32,
        mean_torque_per_tick,
        initial_distance_m: state.initial_dist,
        closest_distance_m: state.closest_dist,
        final_distance_m: state.last_dist,
        target_distance_m: target_distance,
        reached: state.closest_dist <= REACH_RADIUS,
        active_ticks: state.torque_ticks,
        policy_loaded,
    })
}

fn active_torque_ticks(app: &App) -> u64 {
    app.world()
        .get_resource::<EvalState>()
        .map(|s| s.torque_ticks)
        .unwrap_or(0)
}

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
    if !state.target_set
        && let Some(slot) = targets.envs.first_mut()
    {
        let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
        *slot = Some(Vec3::new(
            origin.x + cfg.target_distance,
            TARGET_Y,
            origin.z,
        ));
        state.target_set = true;
    }
    let Some(target) = targets.get(0) else {
        state.tick += 1;
        return;
    };

    let settling = state.tick < cfg.settle_ticks;

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

    if let Some(cpos) = carapace_q
        .iter()
        .find(|(e, _)| e.0 == 0)
        .map(|(_, t)| t.translation)
        .filter(|p| p.is_finite())
    {
        let d = dist_3d(cpos, target);
        if settling {
            state.initial_dist = d;
            state.closest_dist = d;
        } else {
            state.closest_dist = state.closest_dist.min(d);
        }
        state.last_dist = d;
    }

    if !settling && let Some(a) = actions.envs.first() {
        let mut tick_torque = 0.0f32;
        for (joint, env) in joints.iter() {
            if env.0 != 0 {
                continue;
            }
            tick_torque += applied_torque(joint.id, a[joint.id.index()]).abs();
        }
        state.torque_sum += tick_torque as f64;
        state.torque_ticks += 1;
    }

    state.tick += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_far_distance_is_the_training_band_edge() {
        assert_eq!(DEFAULT_TARGET_DISTANCE_M, TARGET_ARENA_HALF);
        const {
            assert!(
                DEFAULT_TARGET_DISTANCE_M > REACH_RADIUS,
                "the ball must start FAR — well outside the reach radius"
            );
            assert!(TARGET_Y > TARGET_Y_MIN && TARGET_Y < TARGET_Y_MAX);
        }
    }

    #[test]
    #[ignore = "builds a headless bevy+rapier App; run with --ignored"]
    fn rest_pose_has_zero_torque_and_no_progress() {
        let dir = std::env::temp_dir().join(format!("rl-eval-restpose-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Explicit, greppable test-only opt-in: this eval deliberately runs whatever
        // body the test env constructs (usually the fallback — no sally.glb in CI).
        let r = run_eval(
            crate::mesh_fallback::BodyGate::FallbackAllowed,
            &dir,
            200,
            DEFAULT_TARGET_DISTANCE_M,
        )
        .expect("an absent checkpoint is the legitimate baseline, never a refusal");

        assert!(!r.policy_loaded, "an empty dir loads no policy (rest pose)");
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
        assert!(
            r.initial_distance_m > REACH_RADIUS,
            "the ball starts far outside reach ({} m)",
            r.initial_distance_m
        );
        assert!(!r.reached, "a rest-pose crab never reaches a far ball");
        assert!(
            (0.0..1.0).contains(&r.progress_m),
            "rest-pose progress should be ~0, got {} m",
            r.progress_m
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
