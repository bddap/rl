//! The GPU-resident learner (rl#49) — the SOLE PPO-update path in production. Brings up
//! the wgpu/Vulkan backend on the discrete GPU (with a hard software-fallback guard),
//! mirrors the CPU policy onto the device each iteration, runs the one
//! [`super::update::ppo_update_core`] there, and mirrors the result back. The whole module
//! is gated on the `wgpu` feature (default-on for `rl-train`, off for the render bins).

use std::path::Path;

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use rand::rngs::StdRng;

use super::TrainBackend;
use super::algorithm::{PpoConfig, PpoMetrics, ReturnNormalizer, RolloutBuffer};
use super::checkpoint::{CrabOpt, crab_optimizer, load_optimizer, save_optimizer};
use super::update::ppo_update_core;
use crate::bot::brain::CrabBrain;

/// The GPU training backend: `Autodiff<Wgpu>` over Vulkan. The live learner (the SOLE
/// update path) and the `bench-update --backend gpu` comparison both run the one generic
/// [`ppo_update_core`] on this — same update, GPU device. WGSL→Vulkan (the SPIR-V
/// `burn/vulkan` path is a separate lever). Rollout inference stays on
/// [`TrainBackend`] (CPU): rollouts are many tiny per-step obs forwards across the K
/// worker threads, where GPU dispatch overhead would dominate; only the one batched
/// update moves to the GPU.
pub(crate) type GpuBackend = Autodiff<burn::backend::wgpu::Wgpu>;

/// Bring up the wgpu/Vulkan backend on the discrete GPU and PROVE the chosen adapter
/// is real hardware, returning the device to run on. The single source of the
/// adapter-selection + software-fallback guard, shared by `bench-update --backend gpu`
/// and the live `learn` learner (the SOLE update path), so there is one place that
/// decides "is this actually the GPU".
///
/// The guard is the load-bearing part. The box's Vulkan ICD set includes lavapipe
/// (`lvp` — a CPU software rasteriser, `DeviceType::Cpu`); if wgpu silently fell back
/// to it, the "GPU" update would run on the CPU. So we (a) request `DiscreteGpu(0)`,
/// which cubecl filters by `device_type == DiscreteGpu` (lavapipe, a CPU device, is
/// excluded before selection — and cubecl panics "No Discrete GPU device found" rather
/// than fall back), and (b) read the chosen adapter, PRINT it, and PANIC if it is a
/// `Cpu`/`Other` device or a known software-rasteriser name. Pair with
/// `VK_ICD_FILENAMES` pointing at only `nvidia_icd.json` to make the NVIDIA card the
/// only Vulkan device at all. `tag` prefixes the log lines (e.g. `bench-update` /
/// `learner`).
pub(crate) fn init_gpu_backend(tag: &str) -> burn::backend::wgpu::WgpuDevice {
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::Vulkan, init_setup};

    // DiscreteGpu(0), not DefaultDevice, so cubecl's filter excludes lavapipe before
    // selection (see the guard rationale above).
    let device = WgpuDevice::DiscreteGpu(0);

    // init_setup::<Vulkan> forces the Vulkan API, registers this device, and hands back
    // the setup so we can inspect the real adapter.
    eprintln!("[{tag}] initialising wgpu/Vulkan on {device:?} …");
    let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
    let info = setup.adapter.get_info();
    eprintln!(
        "[{tag}] wgpu adapter: name={:?} backend={:?} device_type={:?} driver={:?} {:?}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info,
    );

    // Hard gate: refuse a software adapter (a CPU run mislabelled as GPU is worse than no
    // result). Check the device_type via its Debug form (to avoid pinning the wgpu crate
    // version) AND the adapter name against the known software-rasteriser names.
    let name_lc = info.name.to_lowercase();
    let type_str = format!("{:?}", info.device_type).to_lowercase();
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
        info.name, info.device_type,
    );
    eprintln!(
        "[{tag}] adapter confirmed as hardware GPU ({}) — proceeding.",
        info.name
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

/// The GPU-resident learner — the SOLE PPO-update path (rl#49). Owns a GPU brain + GPU
/// Adam optimizer that PERSIST across iterations — the optimizer's moment estimates must
/// carry over across updates, so this is built once and reused, never per-iter.
///
/// The CPU `TrainingState.brain` stays the source of truth. Each iteration
/// [`Self::update`] mirrors the CPU policy onto the GPU, runs the one generic
/// [`ppo_update_core`] there, and mirrors the result back — no second update
/// implementation, only a device for the existing one. Weights cross the boundary as the
/// same `FullPrecisionSettings` bincode the snapshot/checkpoint uses (records are
/// backend-agnostic); no tensor is ever moved directly between backends.
pub(crate) struct GpuLearner {
    device: burn::backend::wgpu::WgpuDevice,
    brain: CrabBrain<GpuBackend>,
    optimizer: CrabOpt<GpuBackend>,
}

impl GpuLearner {
    /// Bring up the GPU backend and build a GPU brain + the shared [`crab_optimizer`].
    /// The brain's initial weights are irrelevant: [`Self::update`] loads the CPU policy
    /// onto it before every update, so the first update trains the real policy, not this
    /// fresh net.
    ///
    /// # Panics
    /// Via [`init_gpu_backend`], if no real discrete-GPU Vulkan adapter is available (a
    /// software lavapipe/llvmpipe adapter, or none at all). Deliberate: the GPU is the
    /// only update path, so it must fail loudly at boot, never silently run on the CPU.
    pub fn new() -> Self {
        let device = init_gpu_backend("learner");
        let brain: CrabBrain<GpuBackend> = CrabBrain::new(&device);
        let optimizer: CrabOpt<GpuBackend> = crab_optimizer();
        Self {
            device,
            brain,
            optimizer,
        }
    }

    /// Run one PPO update on the GPU and mirror the result back to the CPU brain. Loads
    /// `cpu_brain`'s current weights onto the GPU (so the GPU updates exactly the policy
    /// the threads rolled with), runs [`ppo_update_core`] on [`GpuBackend`], then writes
    /// the result back into `cpu_brain`. `ret_norm` (backend-independent f32 stats) is
    /// advanced in place as the CPU path does. `rng` drives the update's minibatch shuffle
    /// (the learner owns it, seeded from the run's master seed). Returns the metrics + the
    /// load/update/store wall-clock split.
    pub fn update(
        &mut self,
        cpu_brain: &mut CrabBrain<TrainBackend>,
        config: &PpoConfig,
        rollouts: &[RolloutBuffer],
        ret_norm: &mut ReturnNormalizer,
        rng: &mut StdRng,
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
        );
        <GpuBackend as burn::tensor::backend::Backend>::sync(&self.device)
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
