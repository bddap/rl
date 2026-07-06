
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use bevy::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings};

use crate::TrainConfig;
use crate::bot::arch::{AnyBrain, ArchId};

use super::TrainBackend;
use super::algorithm::RolloutBuffer;
use super::checkpoint::{CheckpointDir, TICK_WATERMARK_FILENAME};
use super::normalizer::NormalizerSnapshot;
use super::systems::{HorizonOutput, HorizonRequest, TrainingState};

type SnapshotRecorder = BinBytesRecorder<FullPrecisionSettings>;

fn apply_nice(nice: i32) {
    let nice = nice.clamp(0, 19);
    if nice == 0 {
        return;
    }
    unsafe {
        *libc::__errno_location() = 0;
        let rc = libc::setpriority(libc::PRIO_PROCESS, 0, nice);
        if rc == -1 && *libc::__errno_location() != 0 {
            let err = std::io::Error::last_os_error();
            eprintln!("[nice] setpriority({nice}) failed: {err} — running at normal priority");
        }
    }
}

pub fn default_workers(explicit: Option<usize>) -> usize {
    let k = explicit.unwrap_or_else(|| usable_cores().saturating_sub(2).max(1));
    k.clamp(1, 64)
}

/// At most the cores this process can actually run on: the PHYSICAL core count
/// capped by `available_parallelism()`.
///
/// Physical, not logical, as the base: each rollout thread saturates a core with
/// rapier + a burn forward pass, and two such threads sharing one physical core
/// via hyperthreading contend for the same FPU/cache and net well under 2× — so
/// a count keyed off logical CPUs alone would oversubscribe ~2× and thrash. On
/// Linux the physical count is the distinct (physical id, core id) pairs in
/// `/proc/cpuinfo`, which collapses hyperthreads onto their shared core.
///
/// Capped by `available_parallelism()` because `/proc/cpuinfo` is host-wide
/// while `available_parallelism()` honors cgroup CPU quotas and affinity masks:
/// under a capped cgroup (CI, botq workers) the host may show 12 physical cores
/// while the scheduler grants 8 — planning threads for cores the cgroup denies
/// just adds contention. Falls back to `available_parallelism()` alone if
/// `/proc/cpuinfo` is unavailable or yields nothing.
fn usable_cores() -> usize {
    let granted = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        let mut pairs = std::collections::HashSet::new();
        let (mut phys, mut core) = (None, None);
        for line in info.lines() {
            if let Some(v) = line.strip_prefix("physical id") {
                phys = v
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.trim().parse::<u32>().ok());
            } else if let Some(v) = line.strip_prefix("core id") {
                core = v
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.trim().parse::<u32>().ok());
            } else if line.trim().is_empty() {
                if let (Some(p), Some(c)) = (phys.take(), core.take()) {
                    pairs.insert((p, c));
                }
            }
        }
        if let (Some(p), Some(c)) = (phys, core) {
            pairs.insert((p, c));
        }
        if !pairs.is_empty() {
            return pairs.len().min(granted);
        }
    }
    granted
}

fn init_process_pools() {
    crate::bot::headless::pin_single_thread_pools();
}

fn read_tick_watermark(dir: &Path) -> u64 {
    std::fs::read_to_string(dir.join(TICK_WATERMARK_FILENAME))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Persist the tick odometer via the crate-wide [`super::atomic_write`] (temp + fsync +
/// rename), so neither a process crash nor a power loss can leave a torn count a restart
/// would misread; a write failure is logged, not fatal.
fn write_tick_watermark(dir: &Path, ticks: u64) {
    let path = dir.join(TICK_WATERMARK_FILENAME);
    if let Err(e) = super::atomic_write(&path, ticks.to_string().as_bytes()) {
        eprintln!("[learner] failed to persist tick watermark to {path:?}: {e}");
    }
}

const ANNEAL_EPOCH_FILENAME: &str = "log_std_anneal_epoch.txt";

fn read_or_init_anneal_epoch(dir: &Path, total_ticks: u64) -> u64 {
    let path = dir.join(ANNEAL_EPOCH_FILENAME);
    let stored = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    match stored {
        Some(epoch) if epoch <= total_ticks => epoch,
        _ => {
            if let Err(e) = super::atomic_write(&path, total_ticks.to_string().as_bytes()) {
                eprintln!("[learner] failed to persist σ-anneal epoch to {path:?}: {e}");
            }
            total_ticks
        }
    }
}


#[derive(Clone)]
struct RollRequest {
    brain_bytes: Arc<Vec<u8>>,
    normalizer: Arc<NormalizerSnapshot>,
    log_std_floor: f32,
}

enum RollOutcome {
    Rolled {
        output: HorizonOutput,
        ticks: u64,
    },
    Panicked,
    SnapshotLoadFailed,
}

struct RolloutThread {
    request_tx: Sender<RollRequest>,
    result_rx: Receiver<RollOutcome>,
    handle: Option<JoinHandle<()>>,
}

impl RolloutThread {
    fn spawn(id: usize, config: TrainConfig, arch: ArchId, horizon: u64) -> Self {
        let (request_tx, request_rx) = channel::<RollRequest>();
        let (result_tx, result_rx) = channel::<RollOutcome>();
        let handle = std::thread::Builder::new()
            .name(format!("rollout-{id}"))
            .spawn(move || rollout_thread_main(id, config, arch, horizon, request_rx, result_tx))
            .expect("spawn rollout thread");
        Self {
            request_tx,
            result_rx,
            handle: Some(handle),
        }
    }
}

impl Drop for RolloutThread {
    fn drop(&mut self) {
        let (dead_tx, _) = channel::<RollRequest>();
        self.request_tx = dead_tx;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn rollout_thread_main(
    id: usize,
    config: TrainConfig,
    arch: ArchId,
    horizon: u64,
    request_rx: Receiver<RollRequest>,
    result_tx: Sender<RollOutcome>,
) {
    let num_envs = config.envs.max(1) as usize;
    let mut app = build_rollout_app(id, &config, arch, num_envs);
    warm_up_app(&mut app);

    while let Ok(req) = request_rx.recv() {
        let result = roll_with_recovery(
            &mut app,
            |a| roll_one_horizon(a, &req, horizon),
            || {
                eprintln!(
                    "[rollout-{id}] env panicked mid-roll (likely a solver NaN); \
                     rebuilding this thread's world, run continues"
                );
                let mut fresh = build_rollout_app(id, &config, arch, num_envs);
                warm_up_app(&mut fresh);
                fresh
            },
        );
        if result_tx.send(result).is_err() {
            break;
        }
    }
}

fn roll_with_recovery(
    app: &mut App,
    roll: impl FnOnce(&mut App) -> RollOutcome,
    rebuild: impl FnOnce() -> App,
) -> RollOutcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| roll(app))) {
        Ok(r) => r,
        Err(_) => {
            *app = rebuild();
            RollOutcome::Panicked
        }
    }
}

fn roll_one_horizon(app: &mut App, req: &RollRequest, horizon: u64) -> RollOutcome {
    {
        let mut st = app
            .world_mut()
            .get_non_send_resource_mut::<TrainingState>()
            .expect("rollout TrainingState");
        let opened = st.begin_horizon(HorizonRequest {
            brain_bytes: &req.brain_bytes,
            normalizer: (*req.normalizer).clone(),
            log_std_floor: req.log_std_floor,
        });
        if !opened {
            return RollOutcome::SnapshotLoadFailed;
        }
    }

    let start = horizon_tick(app);
    while horizon_tick(app) - start < horizon {
        app.update();
    }
    let rolled = horizon_tick(app) - start;

    let mut st = app
        .world_mut()
        .get_non_send_resource_mut::<TrainingState>()
        .expect("rollout TrainingState");
    RollOutcome::Rolled {
        output: st.end_horizon(),
        ticks: rolled,
    }
}

fn horizon_tick(app: &mut App) -> u64 {
    app.world()
        .get_non_send_resource::<TrainingState>()
        .map(|st| st.total_steps)
        .unwrap_or(0)
}

fn warm_up_app(app: &mut App) {
    for _ in 0..2 {
        app.update();
    }
    let mut st = app
        .world_mut()
        .get_non_send_resource_mut::<TrainingState>()
        .expect("rollout TrainingState");
    let _ = st.end_horizon();
}

fn build_rollout_app(id: usize, config: &TrainConfig, arch: ArchId, num_envs: usize) -> App {
    use crate::bot::headless::{HeadlessStack, WorldRole, headless_stack};
    use crate::training::systems;
    use crate::training::systems::{brain_step, reset_crab};

    let mut app = headless_stack(HeadlessStack {
        num_envs,
        role: WorldRole::RolloutWorker,
        arena: crate::physics::Arena::WalledBox,
    });

    let state = systems::TrainingState::new_worker(config, id, arch);
    app.insert_non_send_resource(state)
        .add_systems(
            FixedUpdate,
            (brain_step, reset_crab)
                .chain()
                .in_set(crate::bot::BotSet::Think),
        );

    crate::bot::headless::force_serial_schedules(&mut app);

    app.finish();
    app.cleanup();
    app
}

/// The learner loop ({snapshot → roll all → merge → GPU update}) — a child module so the
/// `wgpu` gate is written ONCE here instead of stamped on each of its items (#123).
#[cfg(feature = "wgpu")]
mod learner;
#[cfg(feature = "wgpu")]
pub use learner::run_learner;

fn snapshot_brain_bytes(brain: &AnyBrain<TrainBackend>) -> Vec<u8> {
    brain
        .record_leaf(&SnapshotRecorder::default(), ())
        .expect("serialize brain snapshot")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::sensor::OBS_SIZE;
    use crate::training::checkpoint::crab_optimizer;
    use crate::training::normalizer::NormalizerIncrement;
    use crate::training::update::ppo_update_core;

    pub(super) fn empty_normalizer_increment() -> NormalizerIncrement {
        crate::training::normalizer::IncrementAccumulator::new().increment()
    }

    #[test]
    fn default_workers_leaves_two_usable_cores_and_honors_override() {
        let usable = usable_cores();
        let granted = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(
            (1..=granted).contains(&usable),
            "usable {usable} must be in 1..=granted {granted}"
        );

        let k = default_workers(None);
        assert!(k >= 1, "thread count must be at least 1, got {k}");
        if usable > 2 {
            assert_eq!(
                k,
                usable - 2,
                "default must leave exactly 2 usable cores free"
            );
        }

        assert_eq!(default_workers(Some(3)), 3, "explicit count must win");
        assert_eq!(default_workers(Some(0)), 1, "0 clamps up to 1");
        assert_eq!(default_workers(Some(999)), 64, "huge clamps down to 64");
    }

    #[test]
    fn tick_watermark_round_trips_and_defaults_to_zero() {
        let dir = std::env::temp_dir().join(format!("rl_test_tick_wm_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(read_tick_watermark(&dir), 0, "absent watermark reads as 0");
        write_tick_watermark(&dir, 1_234_567);
        assert_eq!(read_tick_watermark(&dir), 1_234_567, "persisted reads back");
        std::fs::write(dir.join(TICK_WATERMARK_FILENAME), b"not a number").unwrap();
        assert_eq!(read_tick_watermark(&dir), 0, "unparsable reads as 0");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn brain_snapshot_round_trips_in_memory() {
        use crate::bot::arch::ArchId;
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::Tensor;

        let device = NdArrayDevice::Cpu;
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
        let bytes = snapshot_brain_bytes(&brain);

        let reloaded = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device)
            .load_leaf_record(&SnapshotRecorder::default(), bytes, &device)
            .expect("load snapshot");

        let obs = Tensor::<TrainBackend, 2>::zeros([1, OBS_SIZE], &device);
        let (m0, s0) = brain.policy(obs.clone());
        let (m1, s1) = reloaded.policy(obs);
        let (m0, m1): (Vec<f32>, Vec<f32>) = (
            m0.to_data().to_vec().unwrap(),
            m1.to_data().to_vec().unwrap(),
        );
        for (i, (a, b)) in m0.iter().zip(m1.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "policy mean[{i}] diverged: {a} vs {b}"
            );
        }
        let (s0, s1): (Vec<f32>, Vec<f32>) = (
            s0.to_data().to_vec().unwrap(),
            s1.to_data().to_vec().unwrap(),
        );
        for (i, (a, b)) in s0.iter().zip(s1.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "log_std[{i}] diverged: {a} vs {b}");
        }
    }

    pub(super) fn scratch_config(tag: &str, m: u64) -> (TrainConfig, std::path::PathBuf) {
        use clap::Parser;
        let dir = std::env::temp_dir().join(format!("rl_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = TrainConfig::try_parse_from([
            "rl",
            "--checkpoint-dir",
            dir.to_str().unwrap(),
            "--envs",
            &m.to_string(),
        ])
        .expect("parse scratch TrainConfig");
        (config, dir)
    }

    #[test]
    fn panicking_roll_is_isolated_and_run_continues() {
        let mut app = App::new();
        let rebuilt = std::cell::Cell::new(false);

        let r1 = roll_with_recovery(
            &mut app,
            |_app| panic!("simulated env blowup mid-roll"),
            || {
                rebuilt.set(true);
                App::new()
            },
        );
        assert!(
            matches!(r1, RollOutcome::Panicked),
            "a panicking roll must report Panicked (no samples are even representable)"
        );
        assert!(
            rebuilt.get(),
            "the App must have been rebuilt after the panic"
        );

        let r2 = roll_with_recovery(
            &mut app,
            |_app| RollOutcome::Rolled {
                output: HorizonOutput {
                    envs: vec![RolloutBuffer::new()],
                    increment: empty_normalizer_increment(),
                    rewards: vec![1.5],
                    drift: (0.0, 0),
                    reach: (0, 0),
                    glitch_drops: 0,
                    nonfinite_obs: 0,
                },
                ticks: 64,
            },
            || panic!("rebuild must NOT run on a successful roll"),
        );
        let RollOutcome::Rolled { output, ticks } = r2 else {
            panic!("the recovered roll must succeed (Rolled), got Panicked");
        };
        assert_eq!(ticks, 64, "the recovered roll's result must pass through");
        assert_eq!(output.rewards, vec![1.5]);
    }

    #[test]
    fn empty_increment_merge_is_a_noop() {
        use crate::training::normalizer::{NORMALIZER_CLIP, ObsNormalizer};

        let mut master = ObsNormalizer::new(NORMALIZER_CLIP);
        for i in 0..30 {
            let mut o = [0.0f32; OBS_SIZE];
            o[0] = i as f32;
            o[1] = (i as f32) * 0.3 - 4.0;
            master.normalize(&o);
        }
        let before = bincode::serialize(&master.snapshot()).unwrap();

        master.merge(&empty_normalizer_increment());
        let after = bincode::serialize(&master.snapshot()).unwrap();

        assert_eq!(
            before, after,
            "merging an empty (panicked-thread) increment must leave the master \
             normalizer byte-unchanged"
        );
    }

    #[test]
    #[ignore = "builds a bevy+rapier App; run with --ignored"]
    fn rollout_thread_collects_per_env_buffers_and_learns() {
        let m = 2u64;
        let horizon = 96u64;
        let (config, dir) = scratch_config("parity_thread", m);

        let mut state = TrainingState::new(&config, None);
        let before = snapshot_brain_bytes(state.brain());

        let thread = RolloutThread::spawn(0, config.clone(), state.brain().arch(), horizon);
        thread
            .request_tx
            .send(RollRequest {
                brain_bytes: Arc::new(snapshot_brain_bytes(state.brain())),
                normalizer: Arc::new(state.normalizer_snapshot()),
                log_std_floor: crate::bot::arch::LOG_STD_MIN,
            })
            .expect("send request");
        let RollOutcome::Rolled { output, .. } = thread.result_rx.recv().expect("recv result")
        else {
            panic!("the roll must not panic");
        };
        let envs = output.envs;
        assert_eq!(
            envs.len(),
            m as usize,
            "one buffer per env (GAE must never sweep across envs)"
        );
        let total: usize = envs.iter().map(|e| e.len()).sum();
        let max = (m * horizon) as usize;
        assert!(
            total > 0 && total <= max,
            "collected {total} transitions, expected (0, {max}]"
        );

        let rollouts: Vec<RolloutBuffer> = envs;
        let mut optimizer = crab_optimizer();
        let (brain, ppo_config, device, ret_norm, rng) = state.learner_parts();
        let metrics = ppo_update_core(
            brain,
            &mut optimizer,
            ppo_config,
            &rollouts,
            device,
            ret_norm,
            rng,
            crate::bot::arch::LOG_STD_MIN,
        );
        assert!(
            metrics.policy_loss.is_finite()
                && metrics.value_loss.is_finite()
                && metrics.entropy.is_finite(),
            "PPO metrics must be finite: {metrics:?}"
        );
        let after = snapshot_brain_bytes(state.brain());
        assert_ne!(
            before, after,
            "the PPO update must change the policy weights (learning)"
        );

        drop(thread);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "builds two bevy+rapier Apps; run with --ignored"]
    fn two_threads_each_collect_a_full_horizon() {
        let m = 1u64;
        let horizon = 96u64;
        let k = 2usize;
        let (config, dir) = scratch_config("parity_two_threads", m);

        let state = TrainingState::new(&config, None);
        let brain_bytes = Arc::new(snapshot_brain_bytes(state.brain()));
        let normalizer = Arc::new(state.normalizer_snapshot());

        let threads: Vec<RolloutThread> = (0..k)
            .map(|id| RolloutThread::spawn(id, config.clone(), state.brain().arch(), horizon))
            .collect();
        for t in &threads {
            t.request_tx
                .send(RollRequest {
                    brain_bytes: Arc::clone(&brain_bytes),
                    normalizer: Arc::clone(&normalizer),
                    log_std_floor: crate::bot::arch::LOG_STD_MIN,
                })
                .expect("send");
        }
        let mut buffers = 0usize;
        let mut total = 0usize;
        for t in &threads {
            let RollOutcome::Rolled { output, .. } = t.result_rx.recv().expect("recv") else {
                panic!("neither thread should panic");
            };
            let envs = output.envs;
            assert_eq!(envs.len(), m as usize, "each thread returns M buffers");
            buffers += envs.len();
            total += envs.iter().map(|e| e.len()).sum::<usize>();
        }
        assert_eq!(buffers, k * m as usize, "learner sees K·M buffers");
        let max = k * (m * horizon) as usize;
        assert!(
            total > 0 && total <= max,
            "collected {total} transitions across {k} threads, expected (0, {max}]"
        );

        drop(threads);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
