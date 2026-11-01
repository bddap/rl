//! Its own test binary: brain init draws from the backend's process-global RNG, which
//! any other test in the same process can reseed or draw from mid-comparison (rl#207).

use burn::backend::ndarray::NdArrayDevice;
use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use crab_world::bot::arch::{AnyBrain, ArchId};
use crab_world::bot::sensor::OBS_SIZE;
use crab_world::training::{InferBackend, TrainBackend};

fn init_with_seed(seed: u64) -> Vec<u32> {
    let device = NdArrayDevice::Cpu;
    TrainBackend::seed(&device, seed);
    let brain: AnyBrain<InferBackend> =
        AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device).valid();
    let obs = Tensor::<InferBackend, 2>::zeros([3, OBS_SIZE], &device);
    let (means, log_std) = brain.policy(obs.clone());
    [means, brain.value(obs)]
        .into_iter()
        .flat_map(|t| t.to_data().to_vec::<f32>().unwrap())
        .chain(log_std.to_data().to_vec::<f32>().unwrap())
        .map(f32::to_bits)
        .collect()
}

#[test]
fn brain_init_is_reproducible_from_the_backend_seed() {
    assert_eq!(
        init_with_seed(0x5EED),
        init_with_seed(0x5EED),
        "same backend seed gave different initial weights — init is not seed-deterministic, \
         so training can't replay a run from its logged --seed"
    );
    assert_ne!(
        init_with_seed(0x5EED),
        init_with_seed(0xC0FFEE),
        "different seeds must give different initial weights, else seeding does nothing"
    );
}
