//! In-process, multi-threaded PPO: K rollout threads feed one learner, all in
//! ONE process — no worker processes, no IPC.
//!
//! # Why one process scales
//! A benchmark proved N independent crab+NN contexts run as N OS threads in one
//! process scale near-linearly with cores — burn-ndarray does not serialize the
//! tiny `[≤16,77]` inference, and rapier's per-world solve is independent. Several
//! process-global thread pools would otherwise make the K threads contend — or, for
//! matrixmultiply's gemm tree, deadlock — so this module pins them all to 1 before
//! any `App` is built (rayon + that gemm tree behind burn-ndarray's matmul, and
//! bevy's three task pools; see [`init_process_pools`] for which and why). Each
//! `App`'s schedule is also forced onto the single-threaded executor, so bevy's
//! multithreaded executor can't fan one `App`'s systems back onto a shared pool.
//!
//! # Design: synchronous actor-learner (on-policy PPO preserved)
//! Each iteration the learner snapshots the current policy, every thread rolls a
//! fixed horizon H in its own world acting with a LOCAL forward pass, ships its
//! transitions back over a channel, and the learner concatenates all K·M env
//! buffers, runs GAE per env, and does the PPO epochs via `ppo_update_core`.
//! Synchronous + one snapshot per update = every sample is drawn from a single
//! consistent snapshot of the policy being updated, so it stays on-policy.
//!
//! A thread's rollout is collected by the same `brain_step` code regardless of K,
//! and the learner runs one update over the merged buffers regardless of K — so a
//! K=1 run is just the K-thread algorithm with one rollout thread. That is the
//! parity the tests pin (K=1 collection == one-thread reference; merge == one
//! stream), which lets the K>1 path inherit the single-thread correctness.
//!
//! # Weight sharing — why a byte snapshot, not a shared `Arc<CrabBrain>`
//! The NdArray tensors a `CrabBrain` holds are `!Send`, so a live brain cannot be
//! shared across the thread boundary at all. Instead the learner serializes the
//! policy to bytes once per iteration with burn's in-memory `BinBytesRecorder`
//! (the exact `FullPrecisionSettings` bincode the checkpoint uses, so the weights
//! round-trip bit-identically), wraps them in an `Arc<Vec<u8>>`, and each thread
//! deserializes that into its OWN thread-local brain before rolling. Bytes are
//! `Send + Sync`; the tensors stay thread-local. Correctness: the snapshot is the
//! fully-updated post-previous-iteration weights, captured before any thread runs,
//! so every thread rolls with a consistent, complete net — never one caught
//! half-updated.
//!
//! # Normalizer
//! The learner owns the master `ObsNormalizer`. Each iteration it snapshots the
//! master, hands it to every thread (which loads it so its policy normalizes
//! against the full baseline), and each thread accumulates a per-horizon
//! INCREMENT over only the observations it saw this horizon. The learner merges
//! those increments — the parallel-Welford merge is exact only for streams the
//! master has not already counted, which the increment guarantees (the re-handed
//! baseline snapshot is never re-merged). No double-count.
//!
//! # Crash isolation
//! The one real advantage the multiprocess design had — a wedged env can't take
//! the run down — is preserved with `std::panic::catch_unwind` around each
//! thread's per-horizon roll. A non-finite/blow-up env is already reset in-world
//! by `rescue_nonfinite_crabs` without a panic; the catch_unwind is the backstop
//! for the rarer hard fault (a solver NaN that panics rapier within the step that
//! created it). On a caught panic the thread rebuilds ONLY its own App and
//! rejoins at the live policy next iteration; the learner sees that thread
//! contribute no samples for the iteration and the other threads are untouched.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use bevy::app::TaskPoolOptions;
use bevy::prelude::*;
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

use crate::bot::brain::CrabBrain;
use crate::{TrainConfig, UpdateDevice};

use super::algorithm::{RolloutBuffer, Transition};
use super::session::{
    CURRICULUM_FILENAME, Curriculum, CurriculumProgress, ObsNormalizerData, TrainBackend,
    TrainingState, load_curriculum, ppo_update_core, save_curriculum,
};

/// Recorder for the in-memory weight snapshot. The same precision settings the
/// on-disk checkpoint (`BinFileRecorder<FullPrecisionSettings>`) uses, so a brain
/// round-trips identically through either.
type SnapshotRecorder = BinBytesRecorder<FullPrecisionSettings>;

/// Lower the calling process's scheduling priority to `nice` (POSIX
/// `setpriority`). Positive niceness yields CPU to higher-priority work, so the
/// owner's foreground game preempts training. Clamped to [0, 19]: a negative nice
/// raises priority and needs privilege, which training must never take. 0 is a
/// no-op (the default inherited priority).
///
/// Best-effort: a failure is logged, not fatal — the trainer still runs, just at
/// normal priority, which is a missing nicety rather than a broken run.
fn apply_nice(nice: i32) {
    let nice = nice.clamp(0, 19);
    if nice == 0 {
        return;
    }
    // SAFETY: setpriority is a plain libc call with no Rust-side invariants; we
    // pass PRIO_PROCESS + who=0 (this process) and read errno on the documented
    // -1 return. (-1 is also a legal priority, so errno must be cleared first to
    // disambiguate per the man page.)
    unsafe {
        *libc::__errno_location() = 0;
        let rc = libc::setpriority(libc::PRIO_PROCESS, 0, nice);
        if rc == -1 && *libc::__errno_location() != 0 {
            let err = std::io::Error::last_os_error();
            eprintln!("[nice] setpriority({nice}) failed: {err} — running at normal priority");
        }
    }
}

/// Resolve the thread count: an explicit `--workers` wins; otherwise default to
/// PHYSICAL cores minus a couple for headroom (floor 1), clamped to [1, 64]. The
/// learner's PPO update runs on the main thread but is mostly idle during rollout
/// (it blocks on the threads), so it isn't counted against the 2 reserved cores.
///
/// Physical, not logical: each rollout thread saturates a core with rapier + a burn
/// forward pass, and two such threads sharing one physical core via hyperthreading
/// contend for the same FPU/cache and net well under 2× — so a default keyed off
/// `available_parallelism()` (which counts logical CPUs) would oversubscribe ~2×
/// and thrash. `physical_cores()` reads the real core count; `available_parallelism`
/// is the portable fallback if that can't be determined.
pub fn default_workers(explicit: Option<usize>) -> usize {
    let k = explicit.unwrap_or_else(|| physical_cores().saturating_sub(2).max(1));
    k.clamp(1, 64)
}

/// Physical CPU core count. On Linux, count the distinct (physical id, core id)
/// pairs in `/proc/cpuinfo` — that collapses hyperthreads onto their shared core.
/// Falls back to `available_parallelism()` (logical CPUs) if `/proc/cpuinfo` is
/// unavailable or yields nothing.
fn physical_cores() -> usize {
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
                // Blank line ends one processor's block; record its core if complete.
                if let (Some(p), Some(c)) = (phys.take(), core.take()) {
                    pairs.insert((p, c));
                }
            }
        }
        // The last block may not be followed by a blank line.
        if let (Some(p), Some(c)) = (phys, core) {
            pairs.insert((p, c));
        }
        if !pairs.is_empty() {
            return pairs.len();
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Pin every process-global thread pool to 1 thread BEFORE any `App` is built, the
/// first matmul runs, or burn's rayon pool is first touched. The learner calls it
/// once at startup. K concurrent rollout-thread gemms otherwise fight over the pools
/// the matmul stack shares process-wide:
///
///   - `RAYON_NUM_THREADS=1` — the outer batch loop (`run_par!`/`iter_range_par!`).
///     A safe perf knob (work-stealing pool, no capacity-1 hazard), so an owner's
///     export is honored; we only default it.
///   - `MATMUL_NUM_THREADS=1` — matrixmultiply's inner-gemm `thread-tree`. THIS is
///     the pool that wedges: the tree is built to be driven by one matmul at a time,
///     so K rollout threads racing it can deadlock — an intermittent race, so a run
///     may go far before it bites. Pinned to 1 the tree has no workers and every gemm
///     runs inline on its calling thread. >1 is a correctness hazard here, not a
///     tuning choice, so it is forced (with a warning) rather than honored.
///   - bevy's task pools — `with_num_threads(1)` pins all three (Io, AsyncCompute,
///     Compute); each is process-global and shared by the K worker Apps, so one
///     thread is the only size at which they can't contend across workers.
///
/// Cost is ~nil: the rollout forward pass is a tiny `[≤16,77]` matmul. Only the
/// learner's PPO-update gemms (`[64,256]×[256,256]`) are big enough to have threaded,
/// and that update runs while every rollout thread is idle — off the rollout critical
/// path.
fn init_process_pools() {
    // SAFETY: single-threaded at process start (no rollout threads spawned yet), so
    // these set_vars race nothing. They must land before the pools initialize — each
    // reads its env exactly once, lazily, on first use.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
    }
    // Unlike RAYON, MATMUL_NUM_THREADS>1 doesn't just slow K>1 down — it re-arms the
    // shared-tree deadlock — so force 1 rather than honor an export. Warn instead of
    // silently overriding, so a stale >1 in the env can't resurface as a rare hang
    // with no breadcrumb.
    if let Ok(prev) = std::env::var("MATMUL_NUM_THREADS")
        && prev != "1"
    {
        eprintln!(
            "[learner] MATMUL_NUM_THREADS={prev} re-enables the shared matmul tree \
             (deadlock risk with K>1 rollout threads); forcing 1"
        );
    }
    unsafe {
        std::env::set_var("MATMUL_NUM_THREADS", "1");
    }
    // with_num_threads(1) pins bevy's three global task pools to one worker each (see
    // the doc above). Building the App also forces every schedule onto the
    // single-threaded executor (see `build_rollout_app`), but the pools must be pinned
    // before any App grabs the defaults.
    TaskPoolOptions::with_num_threads(1).create_default_pools();
}

/// Filename of the tick-budget odometer, beside the checkpoint, so a restarted
/// learner resumes the `--ticks` budget rather than restarting it (the overnight
/// loop makes restarts the expected case).
const TICK_WATERMARK_FILENAME: &str = "ticks.txt";

/// Total physics ticks simulated so far, from the watermark, or 0 if absent or
/// unparsable (a fresh run, or a pre-watermark checkpoint — both start at 0).
fn read_tick_watermark(dir: &Path) -> u64 {
    std::fs::read_to_string(dir.join(TICK_WATERMARK_FILENAME))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Persist the tick odometer. Temp-then-rename so a crash mid-write can't leave a
/// torn count a restart would misread; a write failure is logged, not fatal.
fn write_tick_watermark(dir: &Path, ticks: u64) {
    let path = dir.join(TICK_WATERMARK_FILENAME);
    let tmp = path.with_extension("txt.tmp");
    if let Err(e) =
        std::fs::write(&tmp, ticks.to_string()).and_then(|()| std::fs::rename(&tmp, &path))
    {
        eprintln!("[learner] failed to persist tick watermark to {path:?}: {e}");
    }
}

// ---------------------------------------------------------------------------
// Rollout thread
// ---------------------------------------------------------------------------

/// What the learner hands a rollout thread for one horizon: the policy weight
/// snapshot (bincode bytes, shared read-only) and the master normalizer stats. An
/// `Arc` so K threads share one allocation per iteration rather than K copies.
struct RollRequest {
    brain_bytes: Arc<Vec<u8>>,
    normalizer: Arc<ObsNormalizerData>,
    /// The current curriculum band the thread samples this horizon's targets from. The
    /// learner owns advancement and ships the band down each horizon; the thread never
    /// advances it. `Copy` (a tiny band), so no `Arc` is warranted.
    curriculum: Curriculum,
}

/// What a rollout thread returns after one horizon.
///
/// `Panicked` is its own variant rather than a `panicked: bool` beside the data
/// fields because a panic and real samples are mutually exclusive: when the roll
/// unwinds, the partial buffers/increment/rewards die with the discarded App, so a
/// flag-plus-empty-fields encoding lets an illegal "panicked with samples" state be
/// constructed. The enum makes that unrepresentable — the learner matches `Rolled`
/// to get the data and treats `Panicked` as a no-op (trains on the other threads).
enum RollOutcome {
    Rolled {
        /// Per-env transition buffers (one per env; GAE never sweeps across envs).
        envs: Vec<Vec<Transition>>,
        /// Per-horizon normalizer INCREMENT — only the observations this horizon saw,
        /// so merging it into the master (which holds the baseline) never double-counts.
        increment: ObsNormalizerData,
        /// Rewards of episodes that finished during this horizon.
        rewards: Vec<f32>,
        /// Carapace planar drift-from-spawn this horizon as `(sum, count)` over
        /// recording-env ticks; the learner aggregates across threads into a mean (the
        /// walking diagnostic).
        drift: (f64, u64),
        /// This horizon's per-episode reach tally as `(reached, finished)`; the learner
        /// pools it across threads into the competence window that gates curriculum
        /// advancement.
        reach: (u64, u64),
        /// Physics ticks actually rolled this horizon.
        ticks: u64,
    },
    /// The roll unwound and the thread rebuilt its App; it contributes nothing this
    /// iteration.
    Panicked,
}

/// A live rollout thread: the channels to drive it (request in, result out) and
/// its join handle. Built once and reused every iteration — building a bevy +
/// rapier App costs seconds, so it is paid once, exactly as the old worker process
/// reused its App across horizons.
struct RolloutThread {
    request_tx: Sender<RollRequest>,
    result_rx: Receiver<RollOutcome>,
    handle: Option<JoinHandle<()>>,
}

impl RolloutThread {
    /// Spawn thread `id`: build its headless App once, then loop
    /// {recv request → load snapshot → roll H ticks → send result}. The loop ends
    /// when `request_tx` is dropped (learner shutdown), which closes the recv. `id`
    /// names the OS thread and tags its panic-recovery log line.
    fn spawn(id: usize, config: TrainConfig, horizon: u64) -> Self {
        let (request_tx, request_rx) = channel::<RollRequest>();
        let (result_tx, result_rx) = channel::<RollOutcome>();
        let handle = std::thread::Builder::new()
            .name(format!("rollout-{id}"))
            .spawn(move || rollout_thread_main(id, config, horizon, request_rx, result_tx))
            .expect("spawn rollout thread");
        Self {
            request_tx,
            result_rx,
            handle: Some(handle),
        }
    }
}

impl Drop for RolloutThread {
    /// Dropping the request sender closes the thread's recv loop; join so the
    /// process doesn't outlive its workers (and any in-flight App tears down
    /// cleanly). Best-effort: a thread that already panicked out of its loop joins
    /// immediately.
    fn drop(&mut self) {
        // Drop the sender first so the thread's `recv` returns Err and it exits.
        // Replace with a fresh dummy channel whose sender we immediately drop —
        // `Sender` has no `close`, so dropping our only clone is how we signal EOF.
        let (dead_tx, _) = channel::<RollRequest>();
        self.request_tx = dead_tx; // drops the real sender, closing the thread's rx
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// One rollout thread's body: build the App once, then serve roll requests until
/// the request channel closes. Each horizon's roll is wrapped in `catch_unwind` so
/// a hard solver panic rebuilds only this thread's App and the run continues.
fn rollout_thread_main(
    id: usize,
    config: TrainConfig,
    horizon: u64,
    request_rx: Receiver<RollRequest>,
    result_tx: Sender<RollOutcome>,
) {
    let num_envs = config.envs.max(1) as usize;
    let mut app = build_rollout_app(id, &config, num_envs);
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
                let mut fresh = build_rollout_app(id, &config, num_envs);
                warm_up_app(&mut fresh);
                fresh
            },
        );
        // A send error means the learner has shut down; just exit the loop.
        if result_tx.send(result).is_err() {
            break;
        }
    }
}

/// Run one horizon's roll, isolating a panic so one wedged env can't abort the run
/// (the crash-isolation the multiprocess design gave, kept in-process). `roll` does
/// the work; if it unwinds, `rebuild` produces a fresh App to replace the
/// possibly-poisoned one and `RollOutcome::Panicked` is returned so the learner
/// simply trains on the other threads' samples this iteration.
///
/// `&mut App` is not `UnwindSafe` (interior-mutable world state), but on a panic the
/// App is REPLACED wholesale — no possibly-inconsistent state is read after the
/// unwind — which is exactly what `AssertUnwindSafe` is for. The real common case
/// (a non-finite pose) never reaches here: `rescue_nonfinite_crabs` resets that env
/// in-world without panicking; this is the backstop for a hard solver NaN that
/// panics rapier within the step that created it.
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

/// Roll exactly `horizon` ticks with the request's snapshot, then drain. Loads the
/// snapshot weights + master normalizer into the thread's `TrainingState`, steps
/// the App one tick per `update()` (the App's clock advances exactly one fixed dt
/// per update), and reads the per-env buffers + increment + finished rewards out.
fn roll_one_horizon(app: &mut App, req: &RollRequest, horizon: u64) -> RollOutcome {
    {
        let mut st = app
            .world_mut()
            .get_non_send_resource_mut::<TrainingState>()
            .expect("rollout TrainingState");
        st.load_brain_bytes(&req.brain_bytes);
        st.set_normalizer((*req.normalizer).clone());
        st.set_curriculum(req.curriculum);
        st.reset_horizon_counter();
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
        envs: st.take_rollouts(),
        increment: st.normalizer_increment_snapshot(),
        rewards: st.drain_finished_episode_rewards(),
        drift: st.drain_drift(),
        reach: st.drain_reach(),
        ticks: rolled,
    }
}

/// Total physics ticks this thread has simulated (monotonic across horizons).
fn horizon_tick(app: &mut App) -> u64 {
    app.world()
        .get_non_send_resource::<TrainingState>()
        .map(|st| st.total_steps)
        .unwrap_or(0)
}

/// Scratch metrics dir for rollout thread `id` so K threads don't clobber one CSV
/// (the learner owns "tmp", the established curve location). Throwaway.
fn worker_metrics_dir(id: usize) -> PathBuf {
    std::env::temp_dir().join(format!("rl-rollout-{id}-metrics"))
}

/// Warm up a freshly built App: spawn the crabs and step a couple of updates so
/// the spawn systems run and grace begins to elapse, then discard anything those
/// updates recorded — otherwise the first horizon would carry pre-horizon
/// transitions and double-count samples.
fn warm_up_app(app: &mut App) {
    for _ in 0..2 {
        app.update();
    }
    let mut st = app
        .world_mut()
        .get_non_send_resource_mut::<TrainingState>()
        .expect("rollout TrainingState");
    let _ = st.take_rollouts();
    st.drain_finished_episode_rewards();
}

/// Build one rollout thread's headless App: the production headless training stack
/// (physics + bot + training) in worker mode (no local PPO update), driven by hand
/// via `app.update()`, one physics tick per update so a horizon is exactly H ticks.
///
/// The bevy `TaskPoolPlugin` is pinned to 1 thread and EVERY schedule is forced
/// onto the single-threaded executor — without the latter bevy's multithreaded
/// executor dispatches systems onto the global `ComputeTaskPool` and N apps
/// serialize on it (flat throughput). Both are unconditional here.
fn build_rollout_app(id: usize, config: &TrainConfig, num_envs: usize) -> App {
    use crate::bot::test_util::{HeadlessStack, WorldRole, headless_stack};
    use crate::training::session;
    use crate::training::session::{brain_step, reset_crab, save_on_exit};

    // Per-thread scratch CSV dir so K threads never write the same file.
    let metrics_dir = worker_metrics_dir(id);

    // The shared windowless physics+bot stack in rollout-worker mode: the 1-thread
    // task pool + ScheduleRunner loop (the K-world scaling fix — see
    // `WorldRole::RolloutWorker`), one physics tick per update so a horizon is EXACTLY
    // H ticks (reproducible sample counts).
    let mut app = headless_stack(HeadlessStack {
        num_envs,
        role: WorldRole::RolloutWorker,
    });

    // Worker-mode training state + the Sense→Think→Act systems. No PPO-update step
    // runs in this app: the driver thread reads the per-env buffers out each horizon
    // and the learner owns the update. save_on_exit stays harmless (no AppExit fires
    // here).
    let state = session::TrainingState::new_worker(config, &metrics_dir);
    app.insert_non_send_resource(state)
        .add_systems(
            FixedUpdate,
            (brain_step, reset_crab)
                .chain()
                .in_set(crate::bot::BotSet::Think),
        )
        .add_systems(Last, save_on_exit);

    // Force every schedule onto the single-threaded executor so ECS never dispatches
    // onto the global ComputeTaskPool (which would serialize the K threads).
    // Unconditional, and must run AFTER add_systems above — the schedules don't exist
    // until the systems are wired.
    {
        use bevy::ecs::schedule::{ExecutorKind, Schedules};
        let mut schedules = app.world_mut().resource_mut::<Schedules>();
        for (_label, schedule) in schedules.iter_mut() {
            schedule.set_executor_kind(ExecutorKind::SingleThreaded);
        }
    }

    app.finish();
    app.cleanup();
    app
}

// ---------------------------------------------------------------------------
// Learner (main thread)
// ---------------------------------------------------------------------------

/// Run as the learner: own the policy + optimizer + master normalizer, spawn K
/// rollout threads, and loop {snapshot weights → roll all → merge → PPO update}.
/// `k` threads × M envs each (`config.envs`) × `horizon` ticks = the per-iteration
/// sample count.
///
/// Stops at the first of: `iters` PPO iterations (0 = unbounded) or `config.ticks`
/// total physics ticks (0 = unbounded) — the latter is the production budget the
/// crab-train loop sets via `--ticks`, on hitting which the learner prints
/// "Tick budget reached" (verbatim, the loop's termination grep) and exits. The
/// policy is (re)loaded from `--checkpoint-dir` by `TrainingState::new`, so a
/// learner restarted by the service resumes from the latest checkpoint.
pub fn run_learner(
    config: &TrainConfig,
    k: usize,
    horizon: u64,
    iters: u64,
    nice: i32,
    update_device: UpdateDevice,
) {
    // Own nicing here (one place): lowers this whole process's priority before any
    // world is built, so a foreground game preempts training. The rollout threads
    // spawned below inherit it (POSIX priority is per-process).
    apply_nice(nice);
    init_process_pools();

    let m = config.envs as usize;
    let tick_budget = config.ticks;
    let checkpoint_dir = config.checkpoint_dir.clone();
    std::fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");

    // The learner hosts the policy through a normal TrainingState (brain +
    // optimizer + normalizer + config) but steps no world: it builds rollouts from
    // the threads' buffers and runs `ppo_update_core` over them. `new` loads any
    // existing checkpoint in checkpoint_dir — that is the resume.
    let mut state = TrainingState::new(config);

    // `--update-device gpu` (rl#49): bring up the GPU learner once (its GPU brain +
    // Adam optimizer persist across iters, like the CPU optimizer's moments). The
    // adapter probe + assertion happen here at construction, so a missing/software GPU
    // fails at boot before any rollout. (The first GPU update still pays a one-time
    // shader-compile warmup; that lands in iter 0's update time, excluded by
    // `warmup_iters` from the steady-state rate.) The GPU backend is only compiled
    // under `--features wgpu`; requesting `gpu` without it is a hard, clear failure —
    // never a silent CPU fallback (a mislabelled run is worse than no run).
    #[cfg(feature = "wgpu")]
    let mut gpu_learner = match update_device {
        UpdateDevice::Cpu => None,
        UpdateDevice::Gpu => Some(super::session::GpuLearner::new()),
    };
    #[cfg(not(feature = "wgpu"))]
    if matches!(update_device, UpdateDevice::Gpu) {
        eprintln!(
            "[learner] --update-device gpu requires the `wgpu` cargo feature \
             (build with `--features wgpu`)"
        );
        std::process::exit(2);
    }

    // Resume the tick odometer from the checkpoint, not from 0: the overnight loop
    // makes a learner restart the expected case, and without persistence each
    // restart would re-grant the full `--ticks` budget and over-simulate.
    let mut total_ticks = read_tick_watermark(&checkpoint_dir);

    // The distance curriculum: the learner owns the one advancing instance. Resume the
    // rung from the checkpoint so a warm restart CONTINUES the curriculum; a fresh run
    // or a pre-curriculum checkpoint loads rung 1 (see `load_curriculum`). The window is
    // transient and starts empty — competence is re-measured from live episodes, so the
    // next advance simply waits a full window after the restart.
    let mut progress =
        CurriculumProgress::new(load_curriculum(&checkpoint_dir.join(CURRICULUM_FILENAME)));

    let compute_threads = bevy::tasks::ComputeTaskPool::get().thread_num();
    eprintln!(
        "[learner] in-process: K={k} threads × M={m} envs × H={horizon} ticks/iter → {} transitions/update | budget {} ticks (0=∞), {iters} iters (0=∞) | nice {nice} | compute pool {compute_threads} thread(s), RAYON_NUM_THREADS={}",
        k as u64 * m as u64 * horizon,
        tick_budget,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
    );

    // Spawn the K rollout threads; each builds its App (seconds) before serving.
    let threads: Vec<RolloutThread> = (0..k)
        .map(|id| RolloutThread::spawn(id, config.clone(), horizon))
        .collect();
    eprintln!("[learner] {k} rollout thread(s) building worlds…");

    // Wall-clock + sample accounting for the samples/sec report. `warmup_iters` are
    // excluded from the headline rate so first-iteration build/JIT/page-in costs
    // don't drag the steady-state number down.
    let warmup_iters = 1u64;
    let mut timed_samples = 0u64;
    let mut timed_rollout_secs = 0f64;
    let mut timed_wall_secs = 0f64;
    let mut total_samples = 0u64;
    let mut budget_hit = false;

    let mut iter = 0u64;
    loop {
        if iters != 0 && iter >= iters {
            break;
        }
        let wall_start = Instant::now();

        // 1) Snapshot the current policy to bytes + the master normalizer, shared
        //    read-only across all threads via Arc. This is the consistent snapshot
        //    every thread rolls with — captured before any thread runs, so no
        //    thread can see a half-updated net. Also persist a checkpoint so a live
        //    demo / a restart picks up the latest weights — durability only, not a
        //    handoff to the threads (they get the in-memory snapshot above).
        let brain_bytes = Arc::new(snapshot_brain_bytes(&state.brain));
        let normalizer = Arc::new(state.normalizer_snapshot());
        // The band every thread samples this horizon — captured here so an advance from
        // last horizon's reach (applied after the roll, below) takes effect on THIS roll,
        // and all threads roll the same band. Persist it beside the weights so a restart
        // resumes the rung rather than the start band.
        let curriculum = progress.curriculum();
        state.save_checkpoint();
        save_curriculum(curriculum, &checkpoint_dir.join(CURRICULUM_FILENAME));

        // 2) Roll one synchronous horizon across all threads. Send every thread its
        //    request, then collect every result (the barrier: the update waits for
        //    the slowest). A thread that panicked mid-roll returns an empty result
        //    and has already rebuilt its world — the run continues on the rest.
        let rollout_start = Instant::now();
        for t in &threads {
            t.request_tx
                .send(RollRequest {
                    brain_bytes: Arc::clone(&brain_bytes),
                    normalizer: Arc::clone(&normalizer),
                    curriculum,
                })
                // A closed channel means the thread's OS thread itself died — not a
                // caught roll panic (those return a `panicked` result), but a fault
                // building/warming a world. Unrecoverable for training, so abort
                // loud; crab-train's restart loop resumes from the checkpoint.
                .expect(
                    "rollout thread died (could not rebuild its world); resume from checkpoint",
                );
        }
        let mut results: Vec<RollOutcome> = Vec::with_capacity(k);
        for t in &threads {
            results.push(t.result_rx.recv().expect(
                "rollout thread died (could not rebuild its world); resume from checkpoint",
            ));
        }
        let rollout_secs = rollout_start.elapsed().as_secs_f64();

        // 3) Merge each Rolled thread's normalizer increment + episode rewards and
        //    collect its env buffers. The increment is only this horizon's samples,
        //    so merging it into the master (which already holds the baseline) counts
        //    each sample exactly once. A Panicked thread contributes nothing — no
        //    merge, no buffers — so the master can't be corrupted by a wedged thread.
        let mut rollouts: Vec<RolloutBuffer> = Vec::new();
        let mut iter_samples = 0u64;
        let mut rolled_ticks = 0u64;
        let mut panics = 0u32;
        // Drift summed across threads this iter; divided for the mean logged below.
        let mut drift_sum = 0f64;
        let mut drift_count = 0u64;
        // Per-episode reach tally pooled across threads this iter, fed to the curriculum
        // after the merge so the competence window sees every thread's episodes.
        let mut reach_reached = 0u64;
        let mut reach_finished = 0u64;
        for r in results {
            match r {
                RollOutcome::Rolled {
                    envs,
                    increment,
                    rewards,
                    drift,
                    reach,
                    ticks,
                } => {
                    rolled_ticks += ticks;
                    state.merge_normalizer(&increment);
                    for reward in rewards {
                        state.record_episode_reward(reward);
                    }
                    drift_sum += drift.0;
                    drift_count += drift.1;
                    reach_reached += reach.0;
                    reach_finished += reach.1;
                    for env in envs {
                        iter_samples += env.len() as u64;
                        rollouts.push(RolloutBuffer { transitions: env });
                    }
                }
                RollOutcome::Panicked => panics += 1,
            }
        }
        // Feed this iter's finished episodes to the curriculum, which may advance the
        // band — taking effect on the NEXT horizon's `progress.curriculum()` snapshot.
        progress.record_episodes(reach_reached, reach_finished);

        // 4) PPO update — `ppo_update_core` over all K·M buffers. The trailing
        //    bootstrap per buffer is recomputed inside from the current brain (which
        //    IS the snapshot the threads just rolled with), so no per-env value
        //    crosses any boundary. On `--update-device gpu` the SAME update runs on the
        //    GPU (policy mirrored CPU→GPU, updated, mirrored back); the CPU path is the
        //    unchanged production update. `update_secs` spans the whole phase either
        //    way — including the GPU's host↔device copies — so the per-iter wall time
        //    is a fair CPU-vs-GPU comparison.
        let update_start = Instant::now();
        // `gpu_timing` carries the GPU path's load/update/store split for the log; it is
        // `None` on the CPU path (and always `None` when built without `wgpu`).
        #[cfg(feature = "wgpu")]
        let (metrics, gpu_timing) = match gpu_learner.as_mut() {
            Some(gpu) => {
                let (brain, ppo_config, ret_norm) = state.learner_parts_for_gpu();
                let (m, t) = gpu.update(brain, ppo_config, &rollouts, ret_norm);
                (m, Some(t))
            }
            None => {
                let (brain, optimizer, ppo_config, device, ret_norm) = state.learner_parts();
                let m = ppo_update_core(brain, optimizer, ppo_config, &rollouts, device, ret_norm);
                (m, None)
            }
        };
        #[cfg(not(feature = "wgpu"))]
        let (metrics, gpu_timing): (_, Option<super::session::GpuUpdateTiming>) = {
            let (brain, optimizer, ppo_config, device, ret_norm) = state.learner_parts();
            let m = ppo_update_core(brain, optimizer, ppo_config, &rollouts, device, ret_norm);
            (m, None)
        };
        let update_secs = update_start.elapsed().as_secs_f64();
        let wall_secs = wall_start.elapsed().as_secs_f64();

        total_samples += iter_samples;
        total_ticks += rolled_ticks;
        // Persist the odometer alongside the weights this update produced, so a
        // restart resumes the budget instead of resetting it.
        write_tick_watermark(&checkpoint_dir, total_ticks);
        if iter >= warmup_iters {
            timed_samples += iter_samples;
            timed_rollout_secs += rollout_secs;
            timed_wall_secs += wall_secs;
        }

        let avg_reward = state.avg_reward(20);
        // Mean carapace planar drift-from-spawn over recording-env ticks this iter — the
        // walking diagnostic (climbs from ~0). 0 when no recording tick was sampled.
        let drift = if drift_count > 0 {
            drift_sum / drift_count as f64
        } else {
            0.0
        };
        let sps_iter = iter_samples as f64 / rollout_secs.max(1e-9);
        let sps_rollout = if timed_rollout_secs > 0.0 {
            timed_samples as f64 / timed_rollout_secs
        } else {
            0.0
        };
        let panic_note = if panics > 0 {
            format!(" | {panics} thread(s) recovered from a panic this iter")
        } else {
            String::new()
        };
        // The curriculum band these rollouts used, plus this iter's reach-fraction over
        // finished episodes — so an advance is visible (band steps out) and its approach
        // is legible (reach climbing toward the threshold). `-` when no episode finished.
        let (band_min, band_max) = curriculum.band();
        let reach_note = if reach_finished > 0 {
            format!("{:.2}", reach_reached as f64 / reach_finished as f64)
        } else {
            "-".to_string()
        };
        // On the GPU update path, break the update phase into load(CPU→GPU)/compute/
        // store(GPU→CPU) ms so the host↔device copy cost is visible (the microbench
        // excluded it). Empty on the CPU path.
        let update_note = match gpu_timing {
            Some(t) => format!(
                " [gpu load {:.0}ms + compute {:.0}ms + store {:.0}ms]",
                t.load_ms, t.update_ms, t.store_ms
            ),
            None => String::new(),
        };
        eprintln!(
            "[learner] iter {iter} | {iter_samples} samples | rollout {rollout_secs:.3}s ({rolled_ticks} ticks) update {update_secs:.3}s{update_note} | sps(iter rollout) {sps_iter:.0} sps(steady rollout) {sps_rollout:.0} | total {total_samples} ({total_ticks} ticks) | reward(20) {avg_reward:.3} | drift {drift:.2}m | band {band_min:.1}-{band_max:.1}m (reach {reach_note}) | ploss {:.3} vloss {:.3} ent {:.3}{panic_note}",
            metrics.policy_loss, metrics.value_loss, metrics.entropy,
        );

        iter += 1;

        // Tick budget (`--ticks`): counted in physics ticks, so a run simulates a
        // fixed amount regardless of K or machine speed.
        if tick_budget != 0 && total_ticks >= tick_budget {
            budget_hit = true;
            break;
        }
    }

    // Final checkpoint so the last update's weights are on disk. The rollout threads
    // are torn down by their Drop (channel close + join) when `threads` drops. Persist
    // the curriculum alongside so a resume continues at the rung the run reached.
    state.save_checkpoint();
    save_curriculum(progress.curriculum(), &checkpoint_dir.join(CURRICULUM_FILENAME));
    if timed_samples > 0 {
        let rollout_sps = timed_samples as f64 / timed_rollout_secs.max(1e-9);
        let e2e_sps = timed_samples as f64 / timed_wall_secs.max(1e-9);
        eprintln!(
            "[learner] DONE: rollout {rollout_sps:.0} samples/sec | end-to-end {e2e_sps:.0} samples/sec | {timed_samples} samples over {timed_wall_secs:.1}s ({timed_rollout_secs:.1}s rollout) | K={k} M={m} H={horizon}"
        );
    }
    if budget_hit {
        // The crab-train overnight loop greps for this exact phrase to stop
        // resuming; keep the wording stable.
        eprintln!("[learner] Tick budget reached ({total_ticks} ticks) — stopping training.");
    }
    drop(threads);
}

/// Serialize a brain to the in-memory snapshot bytes the rollout threads load.
/// `FullPrecisionSettings` bincode — the same the on-disk checkpoint uses, so the
/// round-trip is exact.
fn snapshot_brain_bytes(brain: &CrabBrain<TrainBackend>) -> Vec<u8> {
    SnapshotRecorder::default()
        .record(brain.clone().into_record(), ())
        .expect("serialize brain snapshot")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::sensor::OBS_SIZE;

    /// A zero-count normalizer increment: a fresh normalizer's snapshot. Used to fill
    /// a `Rolled` outcome's `increment` in tests that don't exercise real stats, and
    /// to pin the no-op-merge property the normalizer relies on.
    fn empty_normalizer_increment() -> ObsNormalizerData {
        use crate::training::session::{NORMALIZER_CLIP, ObsNormalizer};
        ObsNormalizer::new(NORMALIZER_CLIP).snapshot()
    }

    /// The thread count cap the owner asked for: the default is PHYSICAL cores minus
    /// a couple (floor 1), and an explicit `--workers` still wins. Both clamp into
    /// [1, 64]. Keyed off physical cores so it never oversubscribes hyperthreads.
    #[test]
    fn default_workers_leaves_two_physical_cores_and_honors_override() {
        let physical = physical_cores();
        assert!(physical >= 1, "physical core count must be >= 1");
        // Physical cores must not exceed logical CPUs (hyperthreads only add logical).
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(
            physical <= logical,
            "physical {physical} must be <= logical {logical}"
        );

        let k = default_workers(None);
        assert!(k >= 1, "thread count must be at least 1, got {k}");
        if physical > 2 {
            assert_eq!(
                k,
                physical - 2,
                "default must leave exactly 2 physical cores free"
            );
        }

        assert_eq!(default_workers(Some(3)), 3, "explicit count must win");
        assert_eq!(default_workers(Some(0)), 1, "0 clamps up to 1");
        assert_eq!(default_workers(Some(999)), 64, "huge clamps down to 64");
    }

    /// The tick odometer must survive a learner restart: what `write_tick_watermark`
    /// persists, `read_tick_watermark` returns; an absent or torn file reads as 0.
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

    /// The in-memory weight snapshot must round-trip a brain bit-identically — it is
    /// how every sample-generating thread gets the policy, so a lossy round-trip
    /// would have the threads roll a different net than the learner updates,
    /// breaking the on-policy guarantee. Pins that a brain serialized with
    /// `snapshot_brain_bytes` and reloaded produces the same policy means/log_std.
    #[test]
    fn brain_snapshot_round_trips_in_memory() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::module::Module;
        use burn::tensor::Tensor;

        let device = NdArrayDevice::Cpu;
        let brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);
        let bytes = snapshot_brain_bytes(&brain);

        let record = SnapshotRecorder::default()
            .load(bytes, &device)
            .expect("load snapshot");
        let reloaded = CrabBrain::<TrainBackend>::new(&device).load_record(record);

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

    /// A `TrainConfig` pointing at an empty scratch dir (no checkpoint loads), with
    /// `m` envs per thread. Every other field keeps its default (tick budget 0).
    fn scratch_config(tag: &str, m: u64) -> (TrainConfig, std::path::PathBuf) {
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

    /// Crash isolation — the one real advantage the multiprocess design had, kept
    /// in-process. A roll that PANICS must not abort the run: `roll_with_recovery`
    /// must catch the unwind, rebuild the App, return `Panicked`, and a SUBSEQUENT
    /// roll on the same (rebuilt) App must succeed. This drives the exact production
    /// recovery path (the rollout thread loop calls this) with trivial `App`s and
    /// closures, so it stays fast and deterministic — no reliance on provoking a real
    /// solver NaN. A single wedged env is thereby proven not to take the other
    /// threads / the run down.
    #[test]
    fn panicking_roll_is_isolated_and_run_continues() {
        let mut app = App::new();
        let rebuilt = std::cell::Cell::new(false);

        // Iteration 1: the roll panics (a stand-in for a hard solver NaN).
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

        // Iteration 2: a normal roll on the SAME (rebuilt) App must succeed — the
        // thread survived the panic and keeps serving, the run continues.
        let r2 = roll_with_recovery(
            &mut app,
            |_app| RollOutcome::Rolled {
                envs: vec![Vec::new()],
                increment: empty_normalizer_increment(),
                rewards: vec![1.5],
                drift: (0.0, 0),
                reach: (0, 0),
                ticks: 64,
            },
            || panic!("rebuild must NOT run on a successful roll"),
        );
        let RollOutcome::Rolled { ticks, rewards, .. } = r2 else {
            panic!("the recovered roll must succeed (Rolled), got Panicked");
        };
        assert_eq!(ticks, 64, "the recovered roll's result must pass through");
        assert_eq!(rewards, vec![1.5]);
    }

    /// Merging a zero-count increment must leave the master normalizer byte-unchanged
    /// — the no-op-merge property that makes an empty increment safe. (A panicked
    /// thread now skips the merge entirely via `RollOutcome::Panicked`, but the same
    /// property still backs any zero-sample horizon.) Compared via the snapshot's
    /// bincode bytes, the same form that crosses the iteration.
    #[test]
    fn empty_increment_merge_is_a_noop() {
        use crate::training::session::{NORMALIZER_CLIP, ObsNormalizer};

        // A master with some real stats.
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

    /// Threaded-rollout shape: one rollout THREAD running M envs for a horizon must
    /// collect M independent per-env buffers totaling ~M·H transitions, and the
    /// learner's update over them must change the policy (learning happens). This is
    /// the structural invariant the K>1 path is built on: each thread runs the
    /// worker-mode `brain_step` collection and the learner concatenates the per-env
    /// buffers, GAE never sweeping across an env boundary. (The reward-vs-samples
    /// numeric parity within noise is shown live in the smoke test — stochastic
    /// action sampling makes two separate Apps' trajectories diverge tick-to-tick, so
    /// a unit test pins the deterministic structure instead.)
    ///
    /// Heavy (builds one headless bevy+rapier App), so it is `#[ignore]` by default
    /// and run explicitly in CI / by hand: `cargo test --release -- --ignored`.
    #[test]
    #[ignore = "builds a bevy+rapier App; run with --ignored"]
    fn rollout_thread_collects_per_env_buffers_and_learns() {
        // NB: not calling `init_process_pools` — the parallel test harness is not
        // single-threaded at start, so its `set_var` would race. The per-App
        // `TaskPoolPlugin{1}` + single-threaded executor (in `build_rollout_app`)
        // still apply; throughput scaling is validated by the live smoke test, not
        // here. This test only checks buffer collection + that the update learns.
        let m = 2u64;
        let horizon = 96u64;
        let (config, dir) = scratch_config("parity_thread", m);

        // The learner side: owns the policy; snapshot it before and after to prove
        // the update over the thread's buffers actually moved the weights.
        let mut state = TrainingState::new(&config);
        let before = snapshot_brain_bytes(&state.brain);

        let thread = RolloutThread::spawn(0, config.clone(), horizon);
        thread
            .request_tx
            .send(RollRequest {
                brain_bytes: Arc::new(snapshot_brain_bytes(&state.brain)),
                normalizer: Arc::new(state.normalizer_snapshot()),
                curriculum: Curriculum::start(),
            })
            .expect("send request");
        let RollOutcome::Rolled { envs, .. } = thread.result_rx.recv().expect("recv result") else {
            panic!("the roll must not panic");
        };
        assert_eq!(
            envs.len(),
            m as usize,
            "one buffer per env (GAE must never sweep across envs)"
        );
        let total: usize = envs.iter().map(|e| e.len()).sum();
        // Each env records one transition per tick except during the post-reset
        // settle grace, so the count is at most M·H and within a grace window of it.
        let max = (m * horizon) as usize;
        assert!(
            total > 0 && total <= max,
            "collected {total} transitions, expected (0, {max}]"
        );

        // The learner update over the thread's buffers must change the policy.
        let rollouts: Vec<RolloutBuffer> = envs
            .into_iter()
            .map(|transitions| RolloutBuffer { transitions })
            .collect();
        let (brain, optimizer, ppo_config, device, ret_norm) = state.learner_parts();
        let metrics = ppo_update_core(brain, optimizer, ppo_config, &rollouts, device, ret_norm);
        assert!(
            metrics.policy_loss.is_finite()
                && metrics.value_loss.is_finite()
                && metrics.entropy.is_finite(),
            "PPO metrics must be finite: {metrics:?}"
        );
        let after = snapshot_brain_bytes(&state.brain);
        assert_ne!(
            before, after,
            "the PPO update must change the policy weights (learning)"
        );

        drop(thread);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two rollout threads must each independently collect a full horizon's per-env
    /// buffers from one shared snapshot — the K>1 path. Proves the channel fan-out /
    /// fan-in and the shared `Arc` snapshot work: both threads return M buffers and
    /// the learner sees 2·M buffers totaling up to 2·M·H transitions — i.e. K threads
    /// yield K·M per-env buffers (K× one thread's per-iteration sample count).
    ///
    /// Heavy (builds two headless Apps), so `#[ignore]` by default.
    #[test]
    #[ignore = "builds two bevy+rapier Apps; run with --ignored"]
    fn two_threads_each_collect_a_full_horizon() {
        // See the sibling test: `init_process_pools` is skipped under the parallel
        // test harness (its `set_var` would race); per-App pinning still applies.
        let m = 1u64;
        let horizon = 96u64;
        let k = 2usize;
        let (config, dir) = scratch_config("parity_two_threads", m);

        let state = TrainingState::new(&config);
        let brain_bytes = Arc::new(snapshot_brain_bytes(&state.brain));
        let normalizer = Arc::new(state.normalizer_snapshot());

        let threads: Vec<RolloutThread> = (0..k)
            .map(|id| RolloutThread::spawn(id, config.clone(), horizon))
            .collect();
        for t in &threads {
            t.request_tx
                .send(RollRequest {
                    brain_bytes: Arc::clone(&brain_bytes),
                    normalizer: Arc::clone(&normalizer),
                    curriculum: Curriculum::start(),
                })
                .expect("send");
        }
        let mut buffers = 0usize;
        let mut total = 0usize;
        for t in &threads {
            let RollOutcome::Rolled { envs, .. } = t.result_rx.recv().expect("recv") else {
                panic!("neither thread should panic");
            };
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
