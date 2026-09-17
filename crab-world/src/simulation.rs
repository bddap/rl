use serde::Serialize;

#[derive(Clone, Serialize)]
pub(crate) struct SimulationComponents {
    pub rl_commit: String,
    pub source_digest: String,
    pub build_digest: String,
    pub rapier_pin: String,
    pub baked_rig: u64,
    pub physics: crate::physics::PhysicsParameters,
    pub joint_interface: u64,
    pub plant: u64,
}

impl SimulationComponents {
    pub(crate) fn current() -> Self {
        Self {
            rl_commit: env!("RL_COMMIT").into(),
            source_digest: env!("RL_SOURCE_DIGEST").into(),
            build_digest: env!("RL_BUILD_DIGEST").into(),
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

pub(crate) fn check_simulation_identity(checkpoint: Option<u64>, built: u64) -> Result<(), String> {
    if checkpoint == Some(built) {
        Ok(())
    } else {
        Err(format!(
            "simulation identity mismatch: checkpoint {} but this build is {built:016x}; use the matching simulation build or a fresh checkpoint directory",
            checkpoint
                .map(|d| format!("{d:016x}"))
                .unwrap_or_else(|| "missing (unverified simulator)".into()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_simulation_component_changes_identity_and_refuses() {
        let current = SimulationComponents::current();
        let expected = current.digest();
        let changes: [fn(&mut SimulationComponents); 13] = [
            |s| s.rl_commit.push('x'),
            |s| s.source_digest.push('x'),
            |s| s.build_digest.push('x'),
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
            let refusal = check_simulation_identity(Some(other.digest()), expected).unwrap_err();
            assert!(refusal.contains("simulation identity"));
        }
        assert!(check_simulation_identity(None, expected).is_err());
        assert!(check_simulation_identity(Some(expected), expected).is_ok());
    }
}
