//! rl#351: frozen-checkpoint integrity reproduction. Rolls a saved checkpoint
//! through the trainer's own rollout worlds and horizon loop — envs, horizon,
//! sampling and exploration-σ floor at the checkpoint's tick — with no update, until
//! a finite tick budget is spent or an rl#343 violation trips. A statistical
//! reproduction, not a replay of the training trajectory: each worker's seed stream
//! starts fresh. A worker's stream is not reproducible across processes, so the
//! evidence is captured in-run: each env's previous-tick [`PlantSnapshot`] is saved
//! the tick a near miss shows, for `sally-replay`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy::prelude::With;
use bevy_rapier3d::prelude::RapierRigidBodyHandle;
use bevy_rapier3d::rapier::dynamics::RigidBodyHandle;

use crate::TrainConfig;
use crate::bot::actuator::CrabActions;
use crate::bot::arch::ArchId;
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};
use crate::fnv::Fnv;
use crate::physics::snapshot::PlantSnapshot;
use crate::training::checkpoint::TICK_WATERMARK_FILENAME;
use crate::training::systems::{INTEGRITY_VIOLATION, LearnerState, WorkerState};

use super::{
    ANNEAL_EPOCH_FILENAME, RollOutcome, RollRequest, build_rollout_app, roll_one_horizon,
    snapshot_brain_bytes, warm_up_app,
};

/// A part past this (in the rl#343 bound's units, `lin.max(ang / 3)`) is logged as a
/// near miss: the 100 m/s trip's mechanism at an amplitude a finite budget can see.
const NEAR_MISS_SPEED: f32 = 30.0;

pub struct WorkerOutcome {
    pub worker: usize,
    pub ticks: u64,
    pub violation: Option<String>,
}

fn read_u64(dir: &Path, name: &str) -> Result<u64, String> {
    let path = dir.join(name);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.trim()
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Rolls `workers` (default `0..k`) for `ticks_per_worker` each. Refuses anything but a
/// warm, plant-matched checkpoint: a reproduction on a cold or foreign brain is noise.
pub fn run_repro(
    config: &TrainConfig,
    k: usize,
    horizon: u64,
    ticks_per_worker: u64,
    capture_dir: Option<PathBuf>,
) -> Result<Vec<WorkerOutcome>, String> {
    if horizon == 0 || ticks_per_worker == 0 {
        return Err("--horizon and --ticks must be positive".to_string());
    }
    let config = &TrainConfig {
        seed: Some(config.seed.unwrap_or_else(rand::random)),
        ..config.clone()
    };
    let dir = &config.checkpoint.checkpoint_dir;
    let watermark = read_u64(dir, TICK_WATERMARK_FILENAME)?;
    let epoch = read_u64(dir, ANNEAL_EPOCH_FILENAME)?;
    crate::bot::body::record_plant(dir).map_err(|e| format!("plant refused: {e}"))?;
    let state = LearnerState::new(config, None);
    if state.resumed_set_key().is_none() {
        return Err(format!("no coherent checkpoint set in {}", dir.display()));
    }
    let arch: ArchId = state.brain().arch();
    let log_std_floor = state
        .ppo_config()
        .log_std_floor(watermark.saturating_sub(epoch));
    eprintln!(
        "[repro] checkpoint {} @ {watermark} ticks, arch {arch:?}, log_std floor {log_std_floor:.3}, \
         simulation={:016x}, terrain {:?}, band ≤{} m, {} env(s)/worker, horizon {horizon}, \
         {ticks_per_worker} ticks/worker, seed {}",
        dir.display(),
        crate::simulation::simulation_identity(),
        config.terrain,
        config.band_max_m,
        config.num_envs(),
        config.seed.expect("resolved above"),
    );
    let request = RollRequest {
        brain_bytes: Arc::new(snapshot_brain_bytes(state.brain())),
        normalizer: Arc::new(state.normalizer_snapshot()),
        log_std_floor,
    };
    let capture_dir = capture_dir.map(Arc::new);
    super::init_process_pools();
    let worker_config = TrainConfig {
        ticks: 0,
        ..config.clone()
    };
    let handles: Vec<_> = (0..k)
        .map(|id| {
            let (config, request, capture_dir) =
                (worker_config.clone(), request.clone(), capture_dir.clone());
            let handle = std::thread::Builder::new()
                .name(format!("rollout-{id}"))
                .spawn(move || {
                    roll_worker(
                        id,
                        &config,
                        arch,
                        horizon,
                        &request,
                        ticks_per_worker,
                        capture_dir,
                    )
                })
                .expect("spawn repro worker");
            (id, handle)
        })
        .collect();
    handles
        .into_iter()
        .map(|(worker, h)| match h.join() {
            Ok(ticks) => Ok(WorkerOutcome {
                worker,
                ticks,
                violation: None,
            }),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "non-string panic".to_string());
                if msg.starts_with(INTEGRITY_VIOLATION) {
                    Ok(WorkerOutcome {
                        worker,
                        ticks: 0,
                        violation: Some(msg),
                    })
                } else {
                    Err(format!("repro worker {worker} died: {msg}"))
                }
            }
        })
        .collect()
}

fn roll_worker(
    id: usize,
    config: &TrainConfig,
    arch: ArchId,
    horizon: u64,
    request: &RollRequest,
    budget: u64,
    capture_dir: Option<Arc<PathBuf>>,
) -> u64 {
    let mut app = build_rollout_app(id, config, arch);
    warm_up_app(&mut app);
    let envs = config.num_envs();
    let mut last: Vec<Option<PlantSnapshot>> = vec![None; envs];
    let digest = std::cell::Cell::new(Fnv::new());
    let mut parts = app.world_mut().query_filtered::<(
        &CrabEnvId,
        &bevy_rapier3d::prelude::Velocity,
        Option<&CrabJoint>,
    ), With<CrabBodyPart>>();
    let mut carapaces = app
        .world_mut()
        .query_filtered::<(&CrabEnvId, &RapierRigidBodyHandle), With<CrabCarapace>>();
    let mut before_tick = |app: &mut bevy::app::App| {
        let tick = app
            .world()
            .get_non_send::<WorkerState>()
            .expect("rollout WorkerState")
            .total_steps();
        let mut d = digest.get();
        for row in app.world().resource::<CrabActions>().rows() {
            for v in row {
                d.write(&v.to_bits().to_le_bytes());
            }
        }
        digest.set(d);
        let mut flagged = vec![false; envs];
        for (env, vel, joint) in parts.iter(app.world()) {
            let (lin, ang) = (vel.linear.length(), vel.angular.length());
            if !lin.is_finite() || !ang.is_finite() || lin.max(ang / 3.0) > NEAR_MISS_SPEED {
                eprintln!(
                    "[repro] near-miss worker {id} env {} tick {tick}: {:?} lin {lin:.1} ang {ang:.1}",
                    env.0,
                    joint.map(|j| j.id),
                );
                if let Some(f) = flagged.get_mut(env.0) {
                    *f = true;
                }
            }
        }
        let Some(dir) = capture_dir.as_deref() else {
            return;
        };
        let alive: Vec<(usize, RigidBodyHandle)> = carapaces
            .iter(app.world())
            .map(|(env, h)| (env.0, h.0))
            .collect();
        for e in (0..envs).filter(|&e| flagged[e]) {
            let Some(mut snap) = last[e].take() else {
                continue;
            };
            if snap.tick + 1 != tick || !alive.contains(&(e, snap.parts[0])) {
                continue;
            }
            snap.finish(app.world_mut());
            let path = dir.join(format!("w{id}-e{e}-t{}.bin", snap.tick));
            snap.save(&path)
                .unwrap_or_else(|err| panic!("save {}: {err}", path.display()));
            eprintln!("[repro] captured {}", path.display());
        }
        for (e, slot) in last.iter_mut().enumerate() {
            *slot = alive
                .iter()
                .any(|(env, _)| *env == e)
                .then(|| PlantSnapshot::capture(app.world_mut(), tick, e));
        }
    };
    let mut rolled = 0u64;
    let mut reached = (0u64, 0u64);
    while rolled < budget {
        match roll_one_horizon(&mut app, request, horizon, &mut before_tick) {
            RollOutcome::Rolled { output, ticks } => {
                rolled += ticks;
                reached.0 += output.telemetry.reach_reached;
                reached.1 += output.telemetry.reach_finished;
            }
            RollOutcome::SnapshotLoadFailed => panic!("repro worker {id}: snapshot load failed"),
        }
        if rolled % (horizon * 8) < horizon {
            eprintln!(
                "[repro] worker {id}: {rolled} ticks, reach {}/{} episodes, action digest {:016x}",
                reached.0,
                reached.1,
                digest.get().finish()
            );
        }
    }
    rolled
}
