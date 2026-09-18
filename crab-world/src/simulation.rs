use std::path::Path;

use serde::Serialize;
use tracing::{info, warn};

/// The resolved values that decide where a driven crab ends up, and nothing that
/// does not: a checkpoint trains on ONE binary and ships to three (the native
/// trainer, the deck/TV game, the wasm demo), so any build-side input — commit,
/// source hash, compiler, target — would refuse every checkpoint the pipeline
/// exists to ship. A mechanism change no value carries is tagged at its own
/// site inside [`crate::bot::body::constructed_plant_digest`].
#[derive(Clone, Serialize)]
pub(crate) struct SimulationComponents {
    pub rapier_pin: String,
    pub baked_rig: u64,
    pub physics: crate::physics::PhysicsParameters,
    pub joint_interface: u64,
    pub plant: u64,
}

impl SimulationComponents {
    pub(crate) fn current() -> Self {
        Self {
            rapier_pin: env!("RL_RAPIER_PIN").into(),
            baked_rig: crate::bot::rig::baked_body_digest(),
            physics: crate::physics::identity_parameters(),
            joint_interface: crate::bot::channel_layout_digest(),
            plant: crate::bot::body::constructed_plant_digest(),
        }
    }

    pub(crate) fn digest(&self) -> u64 {
        crate::fnv::fnv1a(&bincode::serialize(self).expect("simulation identity serializes"))
    }
}

pub fn simulation_identity() -> u64 {
    static IDENTITY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *IDENTITY.get_or_init(|| SimulationComponents::current().digest())
}

/// A stamp that differs from this build is a physics tweak the checkpoint predates,
/// which a warm start usually survives; the line makes the skew visible, nothing
/// refuses on it.
pub(crate) fn report_simulation_identity(path: &Path, checkpoint: Option<u64>) {
    let built = simulation_identity();
    let stamp = checkpoint.map_or("unstamped".to_string(), |s| format!("{s:016x}"));
    let line = format!(
        "{}: simulation identity: checkpoint {stamp}, this build {built:016x}",
        path.display()
    );
    if checkpoint == Some(built) {
        info!("{line}");
    } else {
        warn!("{line}");
    }
}

#[cfg(test)]
pub(crate) fn captured_logs<T>(f: impl FnOnce() -> T) -> (T, String) {
    #[derive(Clone, Default)]
    struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Sink {
        type Writer = Sink;
        fn make_writer(&'a self) -> Sink {
            self.clone()
        }
    }
    let sink = Sink::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(sink.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .finish();
    let out = tracing::subscriber::with_default(subscriber, f);
    let logs = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
    (out, logs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_simulation_component_changes_identity() {
        let current = SimulationComponents::current();
        let expected = current.digest();
        let changes: [fn(&mut SimulationComponents); 10] = [
            |s| s.rapier_pin.push('x'),
            |s| s.baked_rig ^= 1,
            |s| s.physics.substeps += 1,
            |s| s.physics.dt *= 2.0,
            |s| s.physics.integration.num_solver_iterations += 1,
            |s| s.physics.integration.contact_softness.natural_frequency += 1.0,
            |s| s.physics.gravity[1] += 1.0,
            |s| s.physics.length_unit *= 2.0,
            |s| s.joint_interface ^= 1,
            |s| s.plant ^= 1,
        ];
        for change in changes {
            let mut other = current.clone();
            change(&mut other);
            assert_ne!(other.digest(), expected);
        }
    }

    #[test]
    fn report_names_both_identities_at_every_stamp_state() {
        let built = simulation_identity();
        let path = Path::new("brain.bin");
        for (stamp, level, shown) in [
            (Some(built), "INFO", format!("{built:016x}")),
            (Some(built ^ 1), "WARN", format!("{:016x}", built ^ 1)),
            (None, "WARN", "unstamped".to_string()),
        ] {
            let ((), logs) = captured_logs(|| report_simulation_identity(path, stamp));
            let line = format!(
                "brain.bin: simulation identity: checkpoint {shown}, this build {built:016x}"
            );
            assert_eq!(logs.lines().count(), 1, "{logs}");
            assert!(logs.contains(level) && logs.contains(&line), "{logs}");
        }
    }
}
