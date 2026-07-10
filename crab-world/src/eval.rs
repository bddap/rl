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
use crate::training::targets::{
    REACH_RADIUS, TARGET_ARENA_HALF, TARGET_Y_MAX, TARGET_Y_MIN, polar_target,
};

pub const DEFAULT_TARGET_DISTANCE_M: f32 = TARGET_ARENA_HALF;

/// The default eval episode length PER BEARING — one training episode horizon (~23 s of
/// crab time at 64 Hz), enough for a working gait to traverse a far target. The (ticks,
/// distance, compass) TRIPLE defines "the chase eval"; every judge — the CLI (release
/// gate, monitor) and the trainer's keep-best gate — must take its defaults from HERE
/// or the metric forks.
pub const DEFAULT_EVAL_TICKS: u64 = crate::training::systems::MAX_EPISODE_TICKS as u64;

/// The fixed compass of target bearings every eval sweeps, uniform in [0, 2π) with
/// bearing 0 = +X (the single bearing the pre-rl#239 eval judged, kept so bearing-0
/// numbers stay comparable to the historical curve). Training samples bearing uniformly
/// (`targets::sample_target`), so a brain competent at one bearing only is a training
/// pathology the eval must EXPOSE: the headline score is the MIN over these bearings —
/// a mean would let seven striding bearings hide one dead one, which is exactly the
/// blindness that scored an 8.93 m (+X) brain as "can chase" while it shuffled in
/// place at the bearing GCR players face (rl#239).
pub const EVAL_BEARINGS: usize = 8;

const TARGET_Y: f32 = (TARGET_Y_MIN + TARGET_Y_MAX) / 2.0;

/// One bearing's episode — the same measurements the pre-compass eval reported for its
/// single +X episode.
#[derive(Debug, Clone, Copy)]
pub struct BearingReport {
    pub bearing_rad: f32,
    pub progress_m: f32,
    pub total_torque: f32,
    pub mean_torque_per_tick: f32,
    pub initial_distance_m: f32,
    pub closest_distance_m: f32,
    pub final_distance_m: f32,
    pub reached: bool,
    pub active_ticks: u64,
}

/// The full compass of bearing episodes. The headline the gates key off is
/// [`Self::worst`] — DERIVED, never stored, so it cannot drift from `per_bearing`.
#[derive(Debug, Clone, Copy)]
pub struct EvalReport {
    pub target_distance_m: f32,
    pub policy_loaded: bool,
    pub per_bearing: [BearingReport; EVAL_BEARINGS],
}

impl EvalReport {
    /// The WORST (min-progress) bearing's episode — one coherent real episode, so
    /// `reached`/torque/distances all describe the same run. MIN over bearings is the
    /// anti-gaming headline: a mean would let seven striding bearings hide one dead
    /// one. `total_cmp` only for a deterministic pick; `progress_m` is never NaN
    /// (`(a-b).max(0.0)` scrubs it).
    pub fn worst(&self) -> &BearingReport {
        self.per_bearing
            .iter()
            .min_by(|a, b| a.progress_m.total_cmp(&b.progress_m))
            .expect("compass is non-empty")
    }

    /// Min-over-bearings chase progress — THE headline scalar every gate compares.
    pub fn progress_m(&self) -> f32 {
        self.worst().progress_m
    }

    /// Whether the crab reached the ball at its WORST bearing.
    pub fn reached(&self) -> bool {
        self.worst().reached
    }
}

#[derive(Resource, Clone, Copy)]
struct EvalConfig {
    target_distance: f32,
    bearing_rad: f32,
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

    // ONE read arms-or-refuses (rl#241 — a classify-then-load pair could straddle a
    // checkpoint swap): a checkpoint the runtime would refuse to arm must refuse the
    // eval too, not become a rest-pose baseline quietly printed as the run's training
    // progress. Missing is the legitimate no-brain-yet case — judge the explicit rest
    // baseline. One load also means every bearing judges the SAME weights: the CLI is
    // pointed at a LIVE checkpoint dir (rl-eval-monitor, the release gate), and a
    // per-bearing reload across the ~2 min sweep could min over a composite of
    // adjacent brains that no single brain achieved.
    let policy = match crate::policy::load_armed(checkpoint_dir) {
        Ok(policy) => policy,
        Err(crate::policy::CheckpointUnusable::Missing) => Policy::rest(),
        Err(crate::policy::CheckpointUnusable::Refused(why)) => {
            return Err(format!(
                "checkpoint at {} refused: {why}",
                checkpoint_dir.display()
            ));
        }
        Err(crate::policy::CheckpointUnusable::Mismatch(dims)) => {
            return Err(format!(
                "checkpoint at {} was built for a different rig ({}/{} obs/act)",
                checkpoint_dir.display(),
                dims.obs,
                dims.action,
            ));
        }
    };
    let policy_loaded = policy.is_loaded();
    let policy = std::rc::Rc::new(policy);

    let mut per_bearing = [None; EVAL_BEARINGS];
    for (slot, bearing_rad) in per_bearing.iter_mut().zip(eval_bearings()) {
        *slot = Some(run_bearing(
            policy.clone(),
            active_ticks,
            target_distance,
            bearing_rad,
        ));
    }
    let per_bearing = per_bearing.map(|r| r.expect("compass and slots are the same length"));

    Ok(EvalReport {
        target_distance_m: target_distance,
        policy_loaded,
        per_bearing,
    })
}

/// The compass bearings in sweep order: bearing i = i·2π/[`EVAL_BEARINGS`].
pub fn eval_bearings() -> impl Iterator<Item = f32> {
    (0..EVAL_BEARINGS).map(|i| i as f32 * std::f32::consts::TAU / EVAL_BEARINGS as f32)
}

/// One episode at one bearing — a fresh world per bearing, so each episode is exactly
/// the pre-compass eval (deterministic per brain) with the target rotated.
fn run_bearing(
    policy: std::rc::Rc<Policy>,
    active_ticks: u64,
    target_distance: f32,
    bearing_rad: f32,
) -> BearingReport {
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        arena: crate::physics::Arena::WalledBox,
        visuals: crate::Visuals(false),
    });
    app.insert_resource(EvalConfig {
        target_distance,
        bearing_rad,
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
    BearingReport {
        bearing_rad,
        progress_m,
        total_torque: state.torque_sum as f32,
        mean_torque_per_tick,
        initial_distance_m: state.initial_dist,
        closest_distance_m: state.closest_dist,
        final_distance_m: state.last_dist,
        reached: state.closest_dist <= REACH_RADIUS,
        active_ticks: state.torque_ticks,
    }
}

fn active_torque_ticks(app: &App) -> u64 {
    app.world()
        .get_resource::<EvalState>()
        .map(|s| s.torque_ticks)
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn eval_step(
    policy: NonSend<std::rc::Rc<Policy>>,
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
        let origin = spawns.origin(0);
        *slot = Some(polar_target(
            origin,
            cfg.bearing_rad,
            cfg.target_distance,
            TARGET_Y,
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
    fn compass_covers_bearings_and_keeps_the_historical_first() {
        let origin = Vec3::new(2.0, 0.0, -3.0);
        let d = DEFAULT_TARGET_DISTANCE_M;
        let targets: Vec<Vec3> = eval_bearings()
            .map(|b| polar_target(origin, b, d, TARGET_Y))
            .collect();
        assert_eq!(targets.len(), EVAL_BEARINGS);

        // Bearing 0 IS the pre-compass eval's +X pose — the historical curve stays
        // comparable at that bearing.
        let first = targets[0];
        assert!((first.x - (origin.x + d)).abs() < 1e-4 && (first.z - origin.z).abs() < 1e-4);

        for (i, t) in targets.iter().enumerate() {
            let planar = Vec2::new(t.x - origin.x, t.z - origin.z);
            assert!(
                (planar.length() - d).abs() < 1e-3,
                "bearing {i} target sits at the eval distance"
            );
            assert_eq!(t.y, TARGET_Y);
            for (j, u) in targets.iter().enumerate().skip(i + 1) {
                assert!(
                    (*t - *u).length() > 1.0,
                    "bearings {i} and {j} must pose distinct targets"
                );
            }
        }
    }

    #[test]
    #[ignore = "builds a headless bevy+rapier App per bearing; run with --ignored"]
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
        assert!(!r.reached(), "a rest-pose crab never reaches a far ball");
        assert!(
            (0.0..1.0).contains(&r.progress_m()),
            "rest-pose progress should be ~0, got {} m",
            r.progress_m()
        );
        for b in &r.per_bearing {
            assert_eq!(
                b.total_torque, 0.0,
                "the rest pose applies no joint torque, so total_torque must be exactly 0"
            );
            assert_eq!(b.mean_torque_per_tick, 0.0);
            assert_eq!(b.active_ticks, 200, "all active ticks are measured");
            assert!(
                b.initial_distance_m.is_finite() && b.closest_distance_m.is_finite(),
                "distances are real finite metres"
            );
            assert!(
                b.initial_distance_m > REACH_RADIUS,
                "the ball starts far outside reach ({} m) at bearing {:.0}°",
                b.initial_distance_m,
                b.bearing_rad.to_degrees()
            );
            assert!(
                (0.0..1.0).contains(&b.progress_m),
                "rest pose shuffles nowhere at bearing {:.0}°, got {} m",
                b.bearing_rad.to_degrees(),
                b.progress_m
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
