//! The GPU-resident learner (rl#49) — the SOLE PPO-update path in production. Brings up
//! the wgpu/Vulkan backend on the discrete GPU (with a hard software-fallback guard),
//! mirrors the CPU policy onto the device each iteration, runs the one
//! [`super::update::ppo_update_core`] there, and mirrors the result back. The whole module
//! is gated on the `wgpu` feature (default-on for `rl-train`, off for the render bins).

use std::path::Path;

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use burn::tensor::backend::AutodiffBackend;
use rand::rngs::StdRng;
use tracing::info;

use super::TrainBackend;
use super::algorithm::{PpoConfig, PpoMetrics, ReturnNormalizer, RolloutBuffer};
use super::checkpoint::{CrabOpt, crab_optimizer, load_optimizer, save_optimizer};
use super::update::ppo_update_core;
use crate::bot::brain::CrabBrain;

/// The GPU training backend: `Autodiff<Wgpu>` over Vulkan. The live learner runs the
/// one generic [`ppo_update_core`] on this — the same update the CPU-backed parity test
/// exercises, just on the GPU device. WGSL→Vulkan (the SPIR-V
/// `burn/vulkan` path is a separate lever). Rollout inference stays on
/// [`TrainBackend`] (CPU): rollouts are many tiny per-step obs forwards across the K
/// worker threads, where GPU dispatch overhead would dominate; only the one batched
/// update moves to the GPU.
pub(crate) type GpuBackend = Autodiff<burn::backend::wgpu::Wgpu>;

/// Bring up the wgpu/Vulkan backend on the discrete GPU and PROVE the chosen adapter
/// is real hardware, returning the device to run on. The single source of the
/// adapter-selection + software-fallback guard for the live `learn` learner, so there is
/// one place that decides "is this actually the GPU".
///
/// The guard is the load-bearing part. The box's Vulkan ICD set includes lavapipe
/// (`lvp` — a CPU software rasteriser, `DeviceType::Cpu`); if wgpu silently fell back
/// to it, the "GPU" update would run on the CPU. So we (a) request `DiscreteGpu(0)`,
/// which cubecl filters by `device_type == DiscreteGpu` (lavapipe, a CPU device, is
/// excluded before selection — and cubecl panics "No Discrete GPU device found" rather
/// than fall back), and (b) read the chosen adapter, PRINT it, and PANIC if it is a
/// `Cpu`/`Other` device or a known software-rasteriser name. Pair with
/// `VK_ICD_FILENAMES` pointing at only `nvidia_icd.json` to make the NVIDIA card the
/// only Vulkan device at all.
pub(crate) fn init_gpu_backend() -> burn::backend::wgpu::WgpuDevice {
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::Vulkan, init_setup};

    // DiscreteGpu(0), not DefaultDevice, so cubecl's filter excludes lavapipe before
    // selection (see the guard rationale above).
    let device = WgpuDevice::DiscreteGpu(0);

    // init_setup::<Vulkan> forces the Vulkan API, registers this device, and hands back
    // the setup so we can inspect the real adapter.
    info!("[learner] initialising wgpu/Vulkan on {device:?} …");
    let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
    let adapter = setup.adapter.get_info();
    info!(
        "[learner] wgpu adapter: name={:?} backend={:?} device_type={:?} driver={:?} {:?}",
        adapter.name, adapter.backend, adapter.device_type, adapter.driver, adapter.driver_info,
    );

    // Hard gate: refuse a software adapter (a CPU run mislabelled as GPU is worse than no
    // result). Check the device_type via its Debug form (to avoid pinning the wgpu crate
    // version) AND the adapter name against the known software-rasteriser names.
    let name_lc = adapter.name.to_lowercase();
    let type_str = format!("{:?}", adapter.device_type).to_lowercase();
    let is_software = type_str.contains("cpu")
        // Some software ICDs report `DeviceType::Other` rather than `Cpu`; reject those
        // too. A real discrete GPU reports `DiscreteGpu`, so this never rejects hardware.
        || type_str.contains("other")
        || name_lc.contains("llvmpipe")
        || name_lc.contains("lavapipe")
        || name_lc.contains("software")
        || name_lc.contains("swiftshader");
    assert!(
        !is_software,
        "wgpu selected a SOFTWARE adapter (name={:?}, type={:?}) — refusing to run the update on \
         it. Set VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json to expose \
         only the NVIDIA card.",
        adapter.name, adapter.device_type,
    );
    info!(
        "[learner] adapter confirmed as hardware GPU ({}) — proceeding.",
        adapter.name
    );
    device
}

/// Wall-clock breakdown of one learner iteration's GPU update phase: the CPU→GPU weight
/// load, the GPU PPO update, and the GPU→CPU write-back, in milliseconds. The update pays
/// the two host↔device copies every iter, so the learner logs all three to show whether
/// the copies eat the GPU win — the make-or-break end-to-end number.
#[derive(Clone, Copy)]
pub(crate) struct GpuUpdateTiming {
    /// CPU brain → bytes → GPU brain (load the policy onto the device for this update).
    pub load_ms: f64,
    /// The GPU [`ppo_update_core`] itself (autodiff backward + Adam steps).
    pub update_ms: f64,
    /// Updated GPU brain → bytes → CPU brain (the next rollout reads the CPU copy).
    pub store_ms: f64,
}

/// The GPU-resident learner. Owns a GPU brain + GPU Adam optimizer that PERSIST across
/// iterations — the optimizer's moment estimates must
/// carry over across updates, so this is built once and reused, never per-iter.
///
/// The CPU `TrainingState.brain` stays the source of truth. Each iteration
/// [`Self::update`] mirrors the CPU policy onto the device, runs the one generic
/// [`ppo_update_core`] there, and mirrors the result back — no second update
/// implementation, only a device for the existing one. Weights cross the boundary as the
/// same `FullPrecisionSettings` bincode the snapshot/checkpoint uses (records are
/// backend-agnostic); no tensor is ever moved directly between backends.
///
/// Generic over the device backend `B` — the injected-device seam. Production uses the
/// default [`GpuBackend`] via [`GpuLearner::new`] (the real discrete GPU); a test injects a
/// CPU backend via [`GpuLearner::with_device`] to exercise this marshalling/timing path with
/// no GPU. The default type parameter keeps every production callsite (`GpuLearner`,
/// `GpuLearner::new()`) unchanged.
pub(crate) struct GpuLearner<B: AutodiffBackend = GpuBackend> {
    device: B::Device,
    brain: CrabBrain<B>,
    optimizer: CrabOpt<B>,
}

impl GpuLearner<GpuBackend> {
    /// Bring up the real discrete-GPU Vulkan backend and build the learner on it.
    ///
    /// # Panics
    /// Via [`init_gpu_backend`], if no real discrete-GPU Vulkan adapter is available (a
    /// software lavapipe/llvmpipe adapter, or none at all). Deliberate: it must fail
    /// loudly at boot, never silently run on the CPU.
    pub fn new() -> Self {
        Self::with_device(init_gpu_backend())
    }
}

impl<B: AutodiffBackend> GpuLearner<B> {
    /// Build the learner on an explicitly-provided device — the injected-device seam behind
    /// [`GpuLearner::new`]. The brain's initial weights are irrelevant: [`Self::update`] loads
    /// the CPU policy onto it before every update, so the first update trains the real policy,
    /// not this fresh net. Production injects the discrete GPU; a test injects a CPU device.
    /// `pub(crate)`, not `pub`: the only public door to a learner is [`GpuLearner::new`], which
    /// pins `B = GpuBackend` and runs the software-adapter gate, so the non-GPU constructor
    /// can't be reached from outside the crate to build a CPU "GPU learner" in production.
    pub(crate) fn with_device(device: B::Device) -> Self {
        let brain: CrabBrain<B> = CrabBrain::new(&device);
        let optimizer: CrabOpt<B> = crab_optimizer();
        Self {
            device,
            brain,
            optimizer,
        }
    }

    /// Run one PPO update on the device and mirror the result back to the CPU brain. Loads
    /// `cpu_brain`'s current weights onto the device (so it updates exactly the policy the
    /// threads rolled with), runs [`ppo_update_core`] on [`B`], then writes the result back
    /// into `cpu_brain`. `ret_norm` (backend-independent f32 stats) is advanced in place as
    /// the CPU path does. `rng` drives the update's minibatch shuffle (the learner owns it,
    /// seeded from the run's master seed). Returns the metrics + the load/update/store
    /// wall-clock split.
    #[tracing::instrument(skip_all)]
    pub fn update(
        &mut self,
        cpu_brain: &mut CrabBrain<TrainBackend>,
        config: &PpoConfig,
        rollouts: &[RolloutBuffer],
        ret_norm: &mut ReturnNormalizer,
        rng: &mut StdRng,
        // This iteration's exploration-σ floor — the SAME lower `log_std` clamp the rollout
        // sampled under, threaded through so the on-backend π_old/π_new recompute matches the
        // behavior policy (see [`ppo_update_core`]).
        log_std_floor: f32,
    ) -> (PpoMetrics, GpuUpdateTiming) {
        use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
        type Bridge = BinBytesRecorder<FullPrecisionSettings>;

        // CPU → GPU: serialize the CPU policy to bincode, then load it into the GPU brain.
        let t_load = std::time::Instant::now();
        let bytes = Bridge::default()
            .record(cpu_brain.clone().into_record(), ())
            .expect("serialize CPU brain for GPU update");
        let record = Bridge::default()
            .load(bytes, &self.device)
            .expect("load brain record onto GPU");
        self.brain = self.brain.clone().load_record(record);
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

        // GPU: the one production update, on the device. `ppo_update_core` reads each
        // minibatch's losses back via `into_scalar`, which forces that minibatch's
        // forward (and the prior step's backward+optimizer.step it depends on) to
        // complete — so almost all the work is genuinely synced inside this region. The
        // explicit `sync` after forces the FINAL minibatch's backward+step too, so
        // `update_ms` is the true compute time and `store_ms` below is purely the copy
        // (without it, the last step would leak into the store timing).
        let t_update = std::time::Instant::now();
        let metrics = ppo_update_core(
            &mut self.brain,
            &mut self.optimizer,
            config,
            rollouts,
            &self.device,
            ret_norm,
            rng,
            log_std_floor,
        );
        <B as burn::tensor::backend::Backend>::sync(&self.device)
            .expect("GPU sync after PPO update");
        let update_ms = t_update.elapsed().as_secs_f64() * 1000.0;

        // GPU → CPU: mirror the updated weights back so the next rollout snapshot +
        // the checkpoint (both off the CPU brain) carry this iteration's update.
        let t_store = std::time::Instant::now();
        let bytes = Bridge::default()
            .record(self.brain.clone().into_record(), ())
            .expect("serialize GPU brain back to CPU");
        let cpu_device = NdArrayDevice::Cpu;
        let record = Bridge::default()
            .load(bytes, &cpu_device)
            .expect("load updated brain record onto CPU");
        *cpu_brain = cpu_brain.clone().load_record(record);
        let store_ms = t_store.elapsed().as_secs_f64() * 1000.0;

        (
            metrics,
            GpuUpdateTiming {
                load_ms,
                update_ms,
                store_ms,
            },
        )
    }

    /// Persist the optimizer's Adam state (per-param m/v + step) to `path`, reading the
    /// moment tensors back off the GPU. Called each iteration beside the brain checkpoint so
    /// a resume warm-starts the optimizer. Best-effort (see [`save_optimizer`]).
    pub fn save_adam_state(&self, path: &Path) {
        save_optimizer(&self.optimizer, path);
    }

    /// Restore the optimizer's Adam state from `path`, uploading the moments back onto this
    /// learner's GPU device. A missing file (pre-rl#60 checkpoint) or an unrecognized version
    /// leaves the optimizer cold without error (see [`load_optimizer`]).
    pub fn load_adam_state(&mut self, path: &Path) {
        let cold = std::mem::replace(&mut self.optimizer, crab_optimizer());
        self.optimizer = load_optimizer(cold, path, &self.device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::sensor::OBS_SIZE;
    use burn::tensor::Tensor;
    use rand::SeedableRng;

    fn policy_means(brain: &CrabBrain<TrainBackend>, device: &NdArrayDevice) -> Vec<f32> {
        let obs = Tensor::<TrainBackend, 2>::zeros([1, OBS_SIZE], device);
        brain.policy(obs, crate::bot::brain::LOG_STD_MIN).0.to_data().to_vec().unwrap()
    }

    /// The injected-device seam lets the CPU↔device marshalling + timing path run with NO
    /// GPU: build the learner on a CPU backend ([`GpuLearner::with_device`]), round-trip a
    /// brain through [`GpuLearner::update`] with an empty rollout (a no-op PPO step), and
    /// confirm the weights survive the serialize→load→serialize→load bridge and the timing
    /// split is populated. This is the coverage the production `GpuLearner::new()` path (which
    /// requires a real discrete GPU) cannot give a unit test.
    #[test]
    fn cpu_device_seam_round_trips_brain_through_update() {
        let device = NdArrayDevice::Cpu;
        let mut learner = GpuLearner::<TrainBackend>::with_device(device);
        let mut cpu_brain: CrabBrain<TrainBackend> = CrabBrain::new(&device);
        let before = policy_means(&cpu_brain, &device);

        // Empty rollouts ⇒ ppo_update_core is a no-op, so update() exercises purely the
        // CPU→device load, the sync, and the device→CPU store — the marshalling path.
        let config = PpoConfig::default();
        let mut ret_norm = ReturnNormalizer::new();
        let mut rng = StdRng::seed_from_u64(0);
        let (_metrics, timing) = learner.update(
            &mut cpu_brain,
            &config,
            &[],
            &mut ret_norm,
            &mut rng,
            crate::bot::brain::LOG_STD_MIN,
        );

        let after = policy_means(&cpu_brain, &device);
        for (i, (a, b)) in before.iter().zip(after.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "weight[{i}] not preserved by the bridge round-trip: {a} vs {b}"
            );
        }
        assert!(timing.load_ms >= 0.0 && timing.update_ms >= 0.0 && timing.store_ms >= 0.0);
    }
}
