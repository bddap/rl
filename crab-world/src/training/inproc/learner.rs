//! The wgpu-gated learner — the main-thread `{snapshot → roll all → merge → GPU update}`
//! loop ([`run_learner`]) and its phase helpers. A child module of [`super`] (inproc) so
//! the `wgpu` gate is expressed ONCE, on the `mod` declaration, instead of stamped on
//! every item (#123): everything here is reachable only from `run_learner`, which needs
//! the GPU backend to exist (rl#49 — the PPO update runs ONLY on the GPU).

use super::*;

struct MergedRollout {
    rollouts: Vec<RolloutBuffer>,
    samples: u64,
    ticks: u64,
    panics: u32,
    /// Threads that REFUSED their horizon because the snapshot failed to load (bddap/rl#177),
    /// pooled this iter; surfaced on the log so a recurring load failure can't hide.
    snapshot_load_failures: u32,
    /// Carapace planar drift-from-spawn this iter as `(sum, count)` over recording-env
    /// ticks, pooled across threads; the log divides it for the mean.
    drift: (f64, u64),
    /// Per-episode reach tally `(reached, finished)` pooled across threads; feeds the
    /// best-keeper's solid-reach floor and the log's reach fraction.
    reach: (u64, u64),
    /// Progress-term drops (non-physical deltas the reward zeroed; bddap/rl#175) pooled across
    /// threads this iter; surfaced on the log so a silent reward dropout is visible. 0 normally.
    glitch_drops: u64,
    /// Non-finite obs elements skipped from the normalizer (bddap/rl#181) pooled across threads
    /// this iter; surfaced on the log so a NaN sensor/physics reading is visible. 0 normally.
    nonfinite_obs: u64,
}

/// Phase 1 — capture the consistent per-iteration view every thread rolls: the policy
/// weights and the master normalizer baseline (a snapshot, never an increment).
/// Captured before any thread runs, so none sees a half-updated net.
fn snapshot_policy(state: &TrainingState, log_std_floor: f32) -> RollRequest {
    RollRequest {
        brain_bytes: Arc::new(snapshot_brain_bytes(state.brain())),
        normalizer: Arc::new(state.normalizer_snapshot()),
        log_std_floor,
    }
}

/// Phase 1 (durability) — persist the checkpoint so a live demo / a restart picks up
/// the latest weights, normalizer, and Adam moments. Not a handoff to the threads (they
/// get the in-memory snapshot); the Adam state lives on the GPU learner, so it is saved
/// here beside the brain, stamped with the set's save stamp so a resume can verify the
/// moments belong to this brain (bddap/rl#215). If the set itself didn't complete, the
/// optimizer is skipped too — the previous save's optimizer stays paired with the
/// previous save's brain.
fn persist_checkpoint(
    state: &TrainingState,
    gpu_learner: &crate::training::gpu::GpuLearner,
    checkpoint_dir: &Path,
) {
    let paths = CheckpointDir::new(checkpoint_dir);
    if let Some(save_stamp) = state.save_checkpoint() {
        gpu_learner.save_adam_state(&paths.optimizer_path(), state.brain().arch(), save_stamp);
    }
}

/// Phase 2 — roll one synchronous horizon across all threads: send each its request,
/// then collect every result. This is the barrier — the update waits for the slowest.
/// A closed channel means a thread's OS thread died building/warming a world (not a
/// caught roll panic, which returns `Panicked`); that is unrecoverable, so abort loud
/// and let crab-train's restart loop resume from the checkpoint.
fn dispatch_horizon(threads: &[RolloutThread], request: &RollRequest) -> Vec<RollOutcome> {
    const DIED: &str = "rollout thread died (could not rebuild its world); resume from checkpoint";
    for t in threads {
        t.request_tx.send(request.clone()).expect(DIED);
    }
    threads
        .iter()
        .map(|t| t.result_rx.recv().expect(DIED))
        .collect()
}

/// Phase 3 — reduce the threads' outcomes into the learner's master state and one
/// [`MergedRollout`]: fold each `Rolled` thread's DISJOINT increment into the master
/// (counting each sample once, since the master holds the baseline), record its
/// rewards, and pool its buffers + aggregates. A `Panicked` thread contributes nothing
/// — no merge, no buffers — so a wedged thread can't corrupt the master.
fn merge_rollouts(state: &mut TrainingState, results: Vec<RollOutcome>) -> MergedRollout {
    let mut merged = MergedRollout {
        rollouts: Vec::new(),
        samples: 0,
        ticks: 0,
        panics: 0,
        snapshot_load_failures: 0,
        drift: (0.0, 0),
        reach: (0, 0),
        glitch_drops: 0,
        nonfinite_obs: 0,
    };
    for r in results {
        match r {
            RollOutcome::Rolled { output, ticks } => {
                merged.ticks += ticks;
                state.merge_normalizer(&output.increment);
                for reward in output.rewards {
                    state.record_episode_reward(reward);
                }
                merged.drift.0 += output.drift.0;
                merged.drift.1 += output.drift.1;
                merged.reach.0 += output.reach.0;
                merged.reach.1 += output.reach.1;
                merged.glitch_drops += output.glitch_drops;
                merged.nonfinite_obs += output.nonfinite_obs;
                for buf in output.envs {
                    merged.samples += buf.len() as u64;
                    merged.rollouts.push(buf);
                }
            }
            RollOutcome::Panicked => merged.panics += 1,
            RollOutcome::SnapshotLoadFailed => merged.snapshot_load_failures += 1,
        }
    }
    merged
}

/// The per-iteration values the learner log line reports, gathered so the formatting
/// lives in [`log_iteration`] rather than interleaved with the update phase.
struct IterReport<'a> {
    iter: u64,
    samples: u64,
    rollout_secs: f64,
    ticks: u64,
    update_secs: f64,
    gpu_timing: &'a crate::training::gpu::GpuUpdateTiming,
    /// This iter's rollout samples/sec (instantaneous) and the steady-state rate
    /// (excludes the warmup iters), respectively.
    sps_iter: f64,
    sps_rollout: f64,
    total_samples: u64,
    total_ticks: u64,
    avg_reward: f32,
    drift: (f64, u64),
    reach: (u64, u64),
    metrics: &'a crate::training::algorithm::PpoMetrics,
    panics: u32,
    /// Threads that refused their horizon on a snapshot load failure this iter (bddap/rl#177).
    snapshot_load_failures: u32,
    /// Progress-term drops this iter — non-physical deltas the reward zeroed (bddap/rl#175).
    glitch_drops: u64,
    /// Non-finite obs elements skipped from the normalizer this iter (bddap/rl#181).
    nonfinite_obs: u64,
}

/// Phase 4 (reporting) — emit the one steady-state learner log line. Derives the means
/// and notes (drift, reach fraction, GPU split, panic recoveries) from the raw counts
/// so the iteration body carries no formatting.
fn log_iteration(r: &IterReport) {
    let drift = if r.drift.1 > 0 {
        r.drift.0 / r.drift.1 as f64
    } else {
        0.0
    };
    let (reached, finished) = r.reach;
    let reach_note = if finished > 0 {
        format!("{:.2}", reached as f64 / finished as f64)
    } else {
        "-".to_string()
    };
    let panic_note = if r.panics > 0 {
        format!(" | {} thread(s) recovered from a panic this iter", r.panics)
    } else {
        String::new()
    };
    // Surfaced only when nonzero — both are 0 on a healthy iter, so the line stays quiet until a
    // fault actually occurs (rl#177 load refusals, rl#175 progress-zeroing).
    let load_fail_note = if r.snapshot_load_failures > 0 {
        format!(
            " | {} thread(s) REFUSED a horizon (snapshot load failed)",
            r.snapshot_load_failures
        )
    } else {
        String::new()
    };
    let glitch_note = if r.glitch_drops > 0 {
        format!(" | {} progress-glitch drop(s)", r.glitch_drops)
    } else {
        String::new()
    };
    let nonfinite_obs_note = if r.nonfinite_obs > 0 {
        format!(
            " | {} non-finite obs element(s) skipped (sensor/physics anomaly)",
            r.nonfinite_obs
        )
    } else {
        String::new()
    };
    let nonfinite_returns_note = if r.metrics.nonfinite_returns > 0 {
        format!(
            " | {} non-finite return(s) skipped (env diverged)",
            r.metrics.nonfinite_returns
        )
    } else {
        String::new()
    };
    let update_note = format!(
        " [gpu load {:.0}ms + compute {:.0}ms + store {:.0}ms]",
        r.gpu_timing.load_ms, r.gpu_timing.update_ms, r.gpu_timing.store_ms
    );
    let IterReport {
        iter,
        samples,
        rollout_secs,
        ticks,
        update_secs,
        sps_iter,
        sps_rollout,
        total_samples,
        total_ticks,
        avg_reward,
        metrics,
        ..
    } = *r;
    eprintln!(
        "[learner] iter {iter} | {samples} samples | rollout {rollout_secs:.3}s ({ticks} ticks) update {update_secs:.3}s{update_note} | sps(iter rollout) {sps_iter:.0} sps(steady rollout) {sps_rollout:.0} | total {total_samples} ({total_ticks} ticks) | reward(20) {avg_reward:.3} | drift {drift:.2}m | reach {reach_note} over {finished} ep | ploss {:.3} vloss {:.3} ent {:.3} kl {:.4} steps {} bdiv {:.3}{panic_note}{load_fail_note}{glitch_note}{nonfinite_returns_note}{nonfinite_obs_note}",
        metrics.policy_loss,
        metrics.value_loss,
        metrics.entropy,
        metrics.kl,
        metrics.steps,
        metrics.behavior_backend_div,
    );
}

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
///
/// Gated on `wgpu` (default-on for `rl-train`): the PPO update runs ONLY on the GPU
/// (rl#49), so the learner needs the GPU backend to exist. A `--no-default-features`
/// trainer drops this function, turning `main`'s call site into a compile error rather
/// than a learner with no update path. (crab-world builds without `wgpu` for the render
/// bins, which only do CPU inference and never call this.)
///
/// `_body_gate` does nothing at runtime — it is the PROOF the bddap/rl#214 body
/// preflight ran ([`crate::mesh_fallback::require_canonical_body`]), required here so a
/// new entry point can't build training worlds while silently on the fallback body.
pub fn run_learner(
    _body_gate: crate::mesh_fallback::BodyGate,
    config: &TrainConfig,
    requested_arch: Option<ArchId>,
    k: usize,
    horizon: u64,
    iters: u64,
    nice: i32,
) {
    // Own nicing here (one place): lowers this whole process's priority before any
    // world is built, so a foreground game preempts training. The rollout threads
    // spawned below inherit it (POSIX priority is per-process).
    apply_nice(nice);
    init_process_pools();

    // Resolve the run's master RNG seed ONCE here, so the same base seed reaches the host
    // and every rollout worker (each mixes in its index for an independent stream — see
    // `TrainingState::build`). A single logged seed then reproduces the whole run; left to
    // each `TrainingState` to resolve, an entropy-default run would draw K+1 unrelated
    // seeds and no one value could reproduce it.
    let mut config_owned = config.clone();
    if config_owned.seed.is_none() {
        config_owned.seed = Some(rand::random::<u64>());
    }
    let config = &config_owned;

    let m = config.envs as usize;
    let tick_budget = config.ticks;
    let checkpoint_dir = config.checkpoint.checkpoint_dir.clone();
    std::fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");

    // The learner hosts the policy through a normal TrainingState (brain on the CPU
    // backend + normalizer + config) but steps no world: it builds rollouts from the
    // threads' buffers and runs the PPO update over them. `new` loads any existing
    // checkpoint in checkpoint_dir — that is the resume. The CPU brain stays the source
    // of truth (rollout snapshots + checkpoints read it); the GPU learner only borrows
    // it each iter to update on the device.
    let mut state = TrainingState::new(config, requested_arch);
    // The run's RESOLVED architecture: the checkpoint tag on a resume, the `--arch`
    // request (default mlp512x3) on a fresh start. Everything downstream that must hold
    // the same variant — the GPU learner and every rollout worker — is built from this,
    // never from the flag.
    let arch = state.brain().arch();

    // The GPU learner (rl#49): the SOLE PPO-update path. Its GPU brain + Adam optimizer
    // persist across iters, like the CPU optimizer's moments. The adapter probe +
    // software-fallback assertion happen here at construction, so a missing/software GPU
    // fails at boot before any rollout. (The first GPU update still pays a one-time
    // shader-compile warmup; that lands in iter 0's update time, excluded by
    // `warmup_iters` from the steady-state rate.)
    let mut gpu_learner = crate::training::gpu::GpuLearner::new(arch);

    // Resume the optimizer's Adam moments + step from the checkpoint so the update
    // continues with warm momentum instead of the brief self-correcting transient a cold
    // optimizer costs (rl#60). An absent optimizer.bin resumes cold — backward compatible,
    // no error (see `load_optimizer`, which also refuses a wrong-arch/legacy/corrupt or
    // cross-save file to cold). Skipped entirely when the brain itself cold-started (a
    // fresh dir, or a DOF change, bddap/rl#31): the moments are per-parameter, so loading
    // old moments onto a fresh net would misalign them exactly as the old brain would —
    // hence the load is keyed on the resumed set's key, which only a warm start has.
    let ckpt = CheckpointDir::new(&checkpoint_dir);
    if let Some(key) = state.resumed_set_key() {
        gpu_learner.load_adam_state(&ckpt.optimizer_path(), key);
    } else {
        eprintln!("[learner] optimizer not warm-started: the brain cold-started — cold moments");
    }

    // Resume the tick odometer from the checkpoint, not from 0: the overnight loop
    // makes a learner restart the expected case, and without persistence each
    // restart would re-grant the full `--ticks` budget and over-simulate.
    let mut total_ticks = read_tick_watermark(&checkpoint_dir);

    // Anchor the exploration-σ anneal to THIS experiment's start (bddap/rl#161): the schedule
    // ramps from a wide floor down to the refine floor over `log_std_anneal_ticks`, measured
    // from `anneal_epoch`. Persisted beside the checkpoint so the anneal continues across the
    // overnight loop's restarts rather than re-widening on every relaunch.
    let anneal_epoch = read_or_init_anneal_epoch(&checkpoint_dir, total_ticks);
    eprintln!(
        "[learner] exploration-σ schedule: log_std floor {:.3} → {:.3} over {} ticks (epoch @ {} ticks)",
        state.config.log_std_floor_start,
        state.config.log_std_floor_end,
        state.config.log_std_anneal_ticks,
        anneal_epoch,
    );


    // Best-by-reach keeping (rl#157): mirror the checkpoint set into `<ckpt>/best/`
    // whenever the policy demonstrates a new high-water reach, so a collapse stays confined to
    // `<ckpt>/` (the trainer resumes from it) while the demo/release — which mirror
    // `best/` — hold the high-water-mark gait. Resumes the running best from the sidecar.
    let mut best_keeper = crate::training::best::BestKeeper::new(&checkpoint_dir);

    let compute_threads = bevy::tasks::ComputeTaskPool::get().thread_num();
    eprintln!(
        "[learner] in-process: K={k} threads × M={m} envs × H={horizon} ticks/iter → {} transitions/update | budget {} ticks (0=∞), {iters} iters (0=∞) | nice {nice} | compute pool {compute_threads} thread(s), RAYON_NUM_THREADS={}",
        k as u64 * m as u64 * horizon,
        tick_budget,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "<unset>".into()),
    );

    // Spawn the K rollout threads; each builds its App (seconds) before serving.
    let threads: Vec<RolloutThread> = (0..k)
        .map(|id| RolloutThread::spawn(id, config.clone(), arch, horizon))
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

        // This iteration's exploration-σ floor from the anneal schedule, keyed to ticks since
        // the experiment's epoch (bddap/rl#161). The SAME scalar drives the rollout sampling
        // (via the snapshot) and the PPO update's on-backend log-prob recompute, so behavior
        // and update policies share σ. `total_ticks` is the pre-iteration odometer here (it is
        // bumped after the update below), so rollout and update agree within the iteration.
        let log_std_floor = state
            .config
            .log_std_floor(total_ticks.saturating_sub(anneal_epoch));

        // 1) Capture the consistent per-iteration snapshot (weights + master normalizer,
        //    none half-updated) and persist a checkpoint.
        let request = snapshot_policy(&state, log_std_floor);
        persist_checkpoint(&state, &gpu_learner, &checkpoint_dir);

        // 2) Roll one synchronous horizon across all threads.
        let rollout_start = Instant::now();
        let results = dispatch_horizon(&threads, &request);
        let rollout_secs = rollout_start.elapsed().as_secs_f64();

        // 3) Reduce the threads' outcomes into the master + this iter's aggregates.
        let merged = merge_rollouts(&mut state, results);

        // 4) PPO update on the GPU — the SOLE update path (rl#49). The CPU policy is
        //    mirrored CPU→GPU, the one `ppo_update_core` runs on the device, and the
        //    result is mirrored back into the CPU brain (the source of truth the next
        //    rollout snapshot + the checkpoint read). The trailing bootstrap per buffer
        //    is recomputed inside from the current brain (which IS the snapshot the
        //    threads just rolled with), so no per-env value crosses any boundary.
        //    `update_secs` spans the whole phase — including the host↔device copies —
        //    so it is the honest per-iter update cost.
        let update_start = Instant::now();
        let (brain, ppo_config, ret_norm, rng) = state.learner_parts_for_gpu();
        let (metrics, gpu_timing) = gpu_learner.update(
            brain,
            ppo_config,
            &merged.rollouts,
            ret_norm,
            rng,
            log_std_floor,
        );
        let update_secs = update_start.elapsed().as_secs_f64();
        let wall_secs = wall_start.elapsed().as_secs_f64();

        total_samples += merged.samples;
        total_ticks += merged.ticks;
        // Persist the odometer alongside the weights this update produced, so a
        // restart resumes the budget instead of resetting it.
        write_tick_watermark(&checkpoint_dir, total_ticks);
        if iter >= warmup_iters {
            timed_samples += merged.samples;
            timed_rollout_secs += rollout_secs;
            timed_wall_secs += wall_secs;
        }

        let sps_rollout = if timed_rollout_secs > 0.0 {
            timed_samples as f64 / timed_rollout_secs
        } else {
            0.0
        };
        log_iteration(&IterReport {
            iter,
            samples: merged.samples,
            rollout_secs,
            ticks: merged.ticks,
            update_secs,
            gpu_timing: &gpu_timing,
            sps_iter: merged.samples as f64 / rollout_secs.max(1e-9),
            sps_rollout,
            total_samples,
            total_ticks,
            avg_reward: state.avg_reward(20),
            drift: merged.drift,
            reach: merged.reach,
            metrics: &metrics,
            panics: merged.panics,
            snapshot_load_failures: merged.snapshot_load_failures,
            glitch_drops: merged.glitch_drops,
            nonfinite_obs: merged.nonfinite_obs,
        });

        // Consider this iter's policy for `<ckpt>/best/`. The reach signal is over THIS
        // iter's finished episodes (None when none finished — the EMA holds). `<ckpt>/` on
        // disk still holds this iter's policy (persisted at the top, not rewritten until the
        // next iter), so a snapshot now captures the policy that earned the reach. See
        // `BestKeeper::observe`.
        let (reached, finished) = merged.reach;
        let reach_fraction = (finished > 0).then(|| reached as f32 / finished as f32);
        best_keeper.observe(reach_fraction);

        iter += 1;

        // Tick budget (`--ticks`): counted in physics ticks, so a run simulates a
        // fixed amount regardless of K or machine speed.
        if tick_budget != 0 && total_ticks >= tick_budget {
            budget_hit = true;
            break;
        }
    }

    // Final checkpoint so the last update's weights are on disk — through
    // `persist_checkpoint`, optimizer included, so the whole dir carries ONE save stamp
    // and a later resume warm-starts the moments instead of refusing a stale stamp
    // (bddap/rl#215). The rollout threads are torn down by their Drop (channel close +
    // join) when `threads` drops.
    persist_checkpoint(&state, &gpu_learner, &checkpoint_dir);
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

#[cfg(test)]
mod tests {
    use super::*;
    // Shared scratch helpers live beside the other inproc tests.
    use super::super::tests::{empty_normalizer_increment, scratch_config};

    /// The reduction must ATTRIBUTE each outcome correctly: a `SnapshotLoadFailed` refusal
    /// (bddap/rl#177) and a `Panicked` thread each contribute NO samples/buffers and are counted
    /// in their OWN tally, while a `Rolled` thread's aggregates — including the progress-glitch
    /// count (bddap/rl#175) — flow through. This is the property that keeps a refused/wedged
    /// horizon from masquerading as honest data. App-free: `merge_rollouts` over hand-built
    /// outcomes, no rollout thread or GPU.
    #[test]
    fn merge_rollouts_attributes_refusals_panics_and_glitches() {
        let (config, dir) = scratch_config("merge_attr", 2);
        let mut state = TrainingState::new(&config, None);

        let results = vec![
            RollOutcome::Rolled {
                output: HorizonOutput {
                    envs: vec![],                            // no env buffers → zero samples
                    increment: empty_normalizer_increment(), // a no-op merge
                    rewards: vec![1.0],
                    drift: (0.5, 2),
                    reach: (1, 3),
                    glitch_drops: 3,
                    nonfinite_obs: 7,
                },
                ticks: 64,
            },
            RollOutcome::SnapshotLoadFailed,
            RollOutcome::Panicked,
        ];
        let merged = merge_rollouts(&mut state, results);

        assert_eq!(
            merged.snapshot_load_failures, 1,
            "the refusal is counted, distinct from a panic"
        );
        assert_eq!(
            merged.panics, 1,
            "the panic is counted, distinct from a refusal"
        );
        assert_eq!(
            merged.samples, 0,
            "neither a refusal nor a panic contributes samples"
        );
        assert!(merged.rollouts.is_empty(), "neither contributes a buffer");
        assert_eq!(
            merged.glitch_drops, 3,
            "the Rolled thread's progress-glitch count flows through"
        );
        assert_eq!(
            merged.nonfinite_obs, 7,
            "the Rolled thread's non-finite obs count flows through"
        );
        assert_eq!(merged.ticks, 64, "only the Rolled thread's ticks count");
        assert_eq!(
            merged.drift,
            (0.5, 2),
            "the Rolled thread's drift flows through"
        );
        assert_eq!(
            merged.reach,
            (1, 3),
            "the Rolled thread's reach flows through"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
