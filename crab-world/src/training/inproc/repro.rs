//! rl#351: frozen-checkpoint integrity reproduction. Rolls a saved checkpoint
//! through the trainer's own rollout worlds and horizon loop — envs, horizon,
//! sampling and exploration-σ floor at the checkpoint's tick — with no update, for a
//! finite tick budget, counting rl#343 violations. A statistical
//! reproduction, not a replay of the training trajectory: each worker's seed stream
//! starts fresh. A worker's stream is not reproducible across processes, so the
//! evidence is captured in-run: each env's previous-tick [`PlantSnapshot`] is saved
//! the tick a near miss shows, for `sally-replay`.
//!
//! A trip ends that worker's world, not its budget: the worker rebuilds a fresh world
//! on a new seed and rolls on, so every worker spends its full budget and the trip
//! rate is trips over the ticks actually rolled. `--solver` swaps the solver counts
//! after warm-up, for a matched rate A/B of solver configurations; a zero-drive crab
//! still adds its settle iterations to the outer count, as in the shipped plant.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy::prelude::{App, With};
use bevy_rapier3d::plugin::context::RapierContextSimulation;
use bevy_rapier3d::prelude::{RapierRigidBodyHandle, Velocity};
use bevy_rapier3d::rapier::dynamics::RigidBodyHandle;

use crate::TrainConfig;
use crate::bot::actuator::CrabActions;
use crate::bot::arch::ArchId;
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};
use crate::fnv::Fnv;
use crate::physics::snapshot::PlantSnapshot;
use crate::physics::{SolverCounts, solver_timestep};
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
    /// Ticks rolled, including each tripping tick.
    pub ticks: u64,
    /// rl#343 panic messages, in order.
    pub trips: Vec<String>,
}

pub struct ReproReport {
    pub workers: Vec<WorkerOutcome>,
    /// The whole process's CPU time: the contention-proof price of the ticks, policy
    /// inference and world rebuilds included.
    pub cpu_secs: f64,
}

fn read_u64(dir: &Path, name: &str) -> Result<u64, String> {
    let path = dir.join(name);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.trim()
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Rolls `workers` (ids) for `ticks_per_worker` each. Refuses anything but a
/// warm, plant-matched checkpoint: a reproduction on a cold or foreign brain is noise.
pub fn run_repro(
    config: &TrainConfig,
    workers: std::ops::Range<usize>,
    horizon: u64,
    ticks_per_worker: u64,
    capture_dir: Option<PathBuf>,
    solver: SolverCounts,
) -> Result<ReproReport, String> {
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
         simulation={:016x}, solver {solver}, terrain {:?}, band ≤{} m, {} env(s)/worker, \
         horizon {horizon}, {ticks_per_worker} ticks/worker, seed {}",
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
    let handles: Vec<_> = workers
        .map(|id| {
            let (config, request, capture_dir) =
                (worker_config.clone(), request.clone(), capture_dir.clone());
            std::thread::Builder::new()
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
                        solver,
                    )
                })
                .expect("spawn repro worker")
        })
        .collect();
    let workers = handles
        .into_iter()
        .map(|h| h.join().map_err(|p| panic_text(p.as_ref()))?)
        .collect::<Result<_, _>>()?;
    Ok(ReproReport {
        workers,
        cpu_secs: process_cpu_secs(),
    })
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "non-string panic".to_string())
}

fn process_cpu_secs() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid out-pointer for the duration of the call.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_PROCESS_CPUTIME_ID)");
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// World 0 keeps the run's seed; later worlds hash the world index in, because the
/// per-worker role seed XORs multiples of one constant and a linear world offset
/// would hand one worker's restart another worker's stream.
fn world_seed(base: u64, world: u64) -> u64 {
    if world == 0 {
        return base;
    }
    let mut f = Fnv::new();
    f.write(&base.to_le_bytes());
    f.write(&world.to_le_bytes());
    f.finish()
}

/// Spends worker `id`'s budget across as many worlds as its trips take, each on its
/// own seed.
#[allow(clippy::too_many_arguments)]
fn roll_worker(
    id: usize,
    config: &TrainConfig,
    arch: ArchId,
    horizon: u64,
    request: &RollRequest,
    budget: u64,
    capture_dir: Option<Arc<PathBuf>>,
    solver: SolverCounts,
) -> Result<WorkerOutcome, String> {
    let base_seed = config.seed.expect("run_repro resolves the seed");
    let mut out = WorkerOutcome {
        worker: id,
        ticks: 0,
        trips: Vec::new(),
    };
    for world in 0u64.. {
        if out.ticks >= budget {
            break;
        }
        let config = TrainConfig {
            seed: Some(world_seed(base_seed, world)),
            ..config.clone()
        };
        let rolled = std::cell::Cell::new(0u64);
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            roll_world(
                id,
                &config,
                arch,
                horizon,
                request,
                budget - out.ticks,
                capture_dir.as_deref(),
                solver,
                world,
                &rolled,
            )
        }));
        out.ticks += rolled.get();
        if let Err(payload) = run {
            if rolled.get() == 0 {
                return Err(format!(
                    "repro worker {id} world {world} died before its first tick: {}",
                    panic_text(payload.as_ref())
                ));
            }
            let msg = panic_text(payload.as_ref());
            if !msg.starts_with(INTEGRITY_VIOLATION) {
                return Err(format!("repro worker {id} died: {msg}"));
            }
            eprintln!(
                "[repro] TRIP worker {id} world {world} at worker tick {}: {}",
                out.ticks,
                msg.lines().next().unwrap_or_default()
            );
            out.trips.push(msg);
        }
    }
    Ok(out)
}

/// The shipped counts are already in place, so warm-up and the plant guards run on
/// the shipped solver; only the rolled ticks see `solver`.
fn set_solver(app: &mut App, solver: SolverCounts) {
    app.insert_resource(solver_timestep(solver));
    let world = app.world_mut();
    let mut sims = world.query::<&mut RapierContextSimulation>();
    for mut sim in sims.iter_mut(world) {
        let p = &mut sim.integration_parameters;
        (
            p.num_solver_iterations,
            p.num_internal_pgs_iterations,
            p.num_internal_stabilization_iterations,
        ) = solver.iterations;
    }
}

#[allow(clippy::too_many_arguments)]
fn roll_world(
    id: usize,
    config: &TrainConfig,
    arch: ArchId,
    horizon: u64,
    request: &RollRequest,
    budget: u64,
    capture_dir: Option<&PathBuf>,
    solver: SolverCounts,
    world: u64,
    rolled: &std::cell::Cell<u64>,
) {
    let mut app = build_rollout_app(id, config, arch);
    warm_up_app(&mut app);
    set_solver(&mut app, solver);
    let envs = config.num_envs();
    let mut last: Vec<Option<PlantSnapshot>> = vec![None; envs];
    let digest = std::cell::Cell::new(Fnv::new());
    let mut parts = app
        .world_mut()
        .query_filtered::<(&CrabEnvId, &Velocity, Option<&CrabJoint>), With<CrabBodyPart>>();
    let mut carapaces = app
        .world_mut()
        .query_filtered::<(&CrabEnvId, &RapierRigidBodyHandle), With<CrabCarapace>>();
    let mut before_tick = |app: &mut App| {
        rolled.set(rolled.get() + 1);
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
                    "[repro] near-miss worker {id} world {world} env {} tick {tick}: {:?} lin {lin:.1} ang {ang:.1}",
                    env.0,
                    joint.map(|j| j.id),
                );
                if let Some(f) = flagged.get_mut(env.0) {
                    *f = true;
                }
            }
        }
        let Some(dir) = capture_dir else {
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
            let path = dir.join(format!("w{id}-r{world}-e{e}-t{}.bin", snap.tick));
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
    let mut reached = (0u64, 0u64);
    while rolled.get() < budget {
        match roll_one_horizon(&mut app, request, horizon, &mut before_tick) {
            RollOutcome::Rolled { output, .. } => {
                reached.0 += output.telemetry.reach_reached;
                reached.1 += output.telemetry.reach_finished;
            }
            RollOutcome::SnapshotLoadFailed => panic!("repro worker {id}: snapshot load failed"),
        }
        if rolled.get() % (horizon * 8) < horizon {
            eprintln!(
                "[repro] worker {id} world {world}: {} ticks, reach {}/{} episodes, action digest {:016x}",
                rolled.get(),
                reached.0,
                reached.1,
                digest.get().finish()
            );
        }
    }
}
