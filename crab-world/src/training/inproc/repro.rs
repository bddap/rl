//! rl#351: frozen-checkpoint integrity reproduction. Rolls a saved checkpoint
//! through the trainer's own rollout worlds and horizon loop — same seeds, envs,
//! horizon, exploration-σ floor at the checkpoint's tick — with no update, until a
//! finite tick budget is spent or an rl#343 violation trips. Each worker is an
//! independent seeded stream, so a hit replays by rerunning that one worker, this
//! time saving a [`PlantSnapshot`] per tick before the trip for `sally-replay`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy::prelude::With;

use crate::TrainConfig;
use crate::bot::body::{CrabBodyPart, CrabEnvId, CrabJoint};
use crate::bot::arch::ArchId;
use crate::physics::snapshot::PlantSnapshot;
use crate::training::checkpoint::TICK_WATERMARK_FILENAME;
use crate::training::systems::{LearnerState, WorkerState};

use super::{
    ANNEAL_EPOCH_FILENAME, RollOutcome, RollRequest, build_rollout_app, roll_one_horizon,
    snapshot_brain_bytes, warm_up_app,
};

/// A part past this (in the rl#343 bound's units, `lin.max(ang / 3)`) is logged as a
/// near miss: the 100 m/s trip's mechanism at an amplitude a finite budget can see.
const NEAR_MISS_SPEED: f32 = 30.0;

pub struct Capture {
    pub env: usize,
    pub from_tick: u64,
    pub dir: PathBuf,
}

pub struct WorkerOutcome {
    pub worker: usize,
    pub ticks: u64,
    /// The worker's panic message: the rl#343 violation with its flight recorder.
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
    only_worker: Option<usize>,
    horizon: u64,
    ticks_per_worker: u64,
    capture: Option<Capture>,
) -> Result<Vec<WorkerOutcome>, String> {
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
         {ticks_per_worker} ticks/worker",
        dir.display(),
        crate::simulation::simulation_identity(),
        config.terrain,
        config.band_max_m,
        config.num_envs(),
    );
    let request = RollRequest {
        brain_bytes: Arc::new(snapshot_brain_bytes(state.brain())),
        normalizer: Arc::new(state.normalizer_snapshot()),
        log_std_floor,
    };
    let ids: Vec<usize> = only_worker.map_or_else(|| (0..k).collect(), |w| vec![w]);
    if capture.is_some() && ids.len() != 1 {
        return Err("--capture-* needs --only-worker".to_string());
    }
    let capture = capture.map(Arc::new);
    super::init_process_pools();
    let worker_config = TrainConfig {
        ticks: 0,
        ..config.clone()
    };
    let handles: Vec<_> = ids
        .iter()
        .map(|&id| {
            let (config, request, capture) =
                (worker_config.clone(), request.clone(), capture.clone());
            let handle = std::thread::Builder::new()
                .name(format!("rollout-{id}"))
                .spawn(move || {
                    roll_worker(id, &config, arch, horizon, &request, ticks_per_worker, capture)
                })
                .expect("spawn repro worker");
            (id, handle)
        })
        .collect();
    Ok(handles
        .into_iter()
        .map(|(worker, h)| match h.join() {
            Ok(ticks) => WorkerOutcome {
                worker,
                ticks,
                violation: None,
            },
            Err(payload) => WorkerOutcome {
                worker,
                ticks: 0,
                violation: Some(
                    payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "non-string panic".to_string()),
                ),
            },
        })
        .collect())
}

fn roll_worker(
    id: usize,
    config: &TrainConfig,
    arch: ArchId,
    horizon: u64,
    request: &RollRequest,
    budget: u64,
    capture: Option<Arc<Capture>>,
) -> u64 {
    let mut app = build_rollout_app(id, config, arch);
    warm_up_app(&mut app);
    let mut pending: Option<PlantSnapshot> = None;
    let mut parts = app.world_mut().query_filtered::<(
        &CrabEnvId,
        &bevy_rapier3d::prelude::Velocity,
        Option<&CrabJoint>,
    ), With<CrabBodyPart>>();
    let mut before_tick = |app: &mut bevy::app::App| {
        let tick = app
            .world()
            .get_non_send::<WorkerState>()
            .expect("rollout WorkerState")
            .total_steps();
        for (env, vel, joint) in parts.iter(app.world()) {
            let (lin, ang) = (vel.linear.length(), vel.angular.length());
            let speed = lin.max(ang / 3.0);
            if !(speed <= NEAR_MISS_SPEED) {
                eprintln!(
                    "[repro] near-miss worker {id} env {} tick {tick}: {:?} lin {lin:.1} ang {ang:.1}",
                    env.0,
                    joint.map(|j| j.id),
                );
            }
        }
        let Some(c) = capture.as_deref() else {
            return;
        };
        if tick < c.from_tick {
            return;
        }
        if let Some(mut snap) = pending.take() {
            snap.finish(app.world_mut());
            let path = c.dir.join(format!("w{id}-e{}-t{}.bin", c.env, snap.tick));
            snap.save(&path)
                .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
        }
        pending = Some(PlantSnapshot::capture(app.world_mut(), tick, c.env));
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
        if rolled % (horizon * 256) < horizon {
            eprintln!(
                "[repro] worker {id}: {rolled} ticks, reach {}/{} episodes",
                reached.0, reached.1
            );
        }
    }
    rolled
}
