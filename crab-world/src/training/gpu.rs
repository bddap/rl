use std::path::Path;

use burn::backend::Autodiff;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::backend::AutodiffBackend;
use rand::rngs::StdRng;
use tracing::info;

use super::TrainBackend;
use super::algorithm::{PpoConfig, PpoMetrics, ReturnNormalizer, RolloutBuffer};
use super::checkpoint::{CrabOpt, crab_optimizer, load_optimizer, save_optimizer};
use super::envelope::SetKey;
use super::update::ppo_update_core;
use crate::bot::arch::{AnyBrain, ArchId};

pub(crate) type GpuBackend = Autodiff<burn::backend::wgpu::Wgpu>;

pub(crate) fn init_gpu_backend() -> burn::backend::wgpu::WgpuDevice {
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::Vulkan, init_setup};

    let device = WgpuDevice::DiscreteGpu(0);

    info!("[learner] initialising wgpu/Vulkan on {device:?} …");
    let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
    let adapter = setup.adapter.get_info();
    info!(
        "[learner] wgpu adapter: name={:?} backend={:?} device_type={:?} driver={:?} {:?}",
        adapter.name, adapter.backend, adapter.device_type, adapter.driver, adapter.driver_info,
    );

    let name_lc = adapter.name.to_lowercase();
    let type_str = format!("{:?}", adapter.device_type).to_lowercase();
    let is_software = type_str.contains("cpu")
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

#[derive(Clone, Copy)]
pub(crate) struct GpuUpdateTiming {
    pub load_ms: f64,
    pub update_ms: f64,
    pub store_ms: f64,
}

pub(crate) struct GpuLearner<B: AutodiffBackend = GpuBackend> {
    device: B::Device,
    brain: AnyBrain<B>,
    optimizer: CrabOpt<B>,
}

impl GpuLearner<GpuBackend> {
    pub fn new(arch: ArchId) -> Self {
        Self::with_device(arch, init_gpu_backend())
    }
}

impl<B: AutodiffBackend> GpuLearner<B> {
    pub(crate) fn with_device(arch: ArchId, device: B::Device) -> Self {
        let brain: AnyBrain<B> = AnyBrain::init(arch, &device);
        let optimizer: CrabOpt<B> = crab_optimizer();
        Self {
            device,
            brain,
            optimizer,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn update(
        &mut self,
        cpu_brain: &mut AnyBrain<TrainBackend>,
        config: &PpoConfig,
        rollouts: &[RolloutBuffer],
        ret_norm: &mut ReturnNormalizer,
        rng: &mut StdRng,
        log_std_floor: f32,
    ) -> (PpoMetrics, GpuUpdateTiming) {
        use burn::record::{BinBytesRecorder, FullPrecisionSettings};
        type Bridge = BinBytesRecorder<FullPrecisionSettings>;

        let t_load = std::time::Instant::now();
        let bytes = cpu_brain
            .record_leaf(&Bridge::default(), ())
            .expect("serialize CPU brain for GPU update");
        self.brain = self
            .brain
            .clone()
            .load_leaf_record(&Bridge::default(), bytes, &self.device)
            .expect("load brain record onto GPU");
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

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

        let t_store = std::time::Instant::now();
        let bytes = self
            .brain
            .record_leaf(&Bridge::default(), ())
            .expect("serialize GPU brain back to CPU");
        let cpu_device = NdArrayDevice::Cpu;
        *cpu_brain = cpu_brain
            .clone()
            .load_leaf_record(&Bridge::default(), bytes, &cpu_device)
            .expect("load updated brain record onto CPU");
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

    pub fn save_adam_state(&self, path: &Path, arch: ArchId, save_stamp: u64) {
        save_optimizer(&self.optimizer, arch, path, save_stamp);
    }

    pub fn load_adam_state(&mut self, path: &Path, key: SetKey) {
        let cold = std::mem::replace(&mut self.optimizer, crab_optimizer());
        self.optimizer = load_optimizer(cold, path, &self.device, key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::sensor::OBS_SIZE;
    use burn::tensor::Tensor;
    use rand::SeedableRng;

    fn policy_means(brain: &AnyBrain<TrainBackend>, device: &NdArrayDevice) -> Vec<f32> {
        let obs = Tensor::<TrainBackend, 2>::zeros([1, OBS_SIZE], device);
        brain.policy(obs).0.to_data().to_vec().unwrap()
    }

    #[test]
    fn cpu_device_seam_round_trips_brain_through_update() {
        let device = NdArrayDevice::Cpu;
        let mut learner = GpuLearner::<TrainBackend>::with_device(ArchId::DEFAULT, device);
        let mut cpu_brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
        let before = policy_means(&cpu_brain, &device);

        let config = PpoConfig::default();
        let mut ret_norm = ReturnNormalizer::new();
        let mut rng = StdRng::seed_from_u64(0);
        let (_metrics, timing) = learner.update(
            &mut cpu_brain,
            &config,
            &[],
            &mut ret_norm,
            &mut rng,
            crate::bot::arch::LOG_STD_MIN,
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
