//! The trainer's rollout env IS the headless server world (rl#298 stage 4, the env
//! swap): [`build_rollout_app`] — the [`RolloutEnvBuilder`] `rl-train` injects into
//! the learner — composes crab-world's training wiring onto [`headless_server_world`],
//! the same world the crab-slot host harness drives through `advance → slot →
//! step_next`. One construction for the world she trains in and the world she is
//! served in, so their physics cannot drift and "training arena" is no longer a
//! separate thing.
//!
//! The ball target REPLACES the hunt feed here, it does not ride its plumbing: the
//! bridge's hunt path (`set_crab_walk_target`) poses prey at the one fixed claw
//! height, which cannot express training's surface-relative target band or the
//! under-carapace close disc the grab curriculum needs (rl#250) — and the bridge is
//! stage 5's deletion target besides. The training target source stays the ball
//! sampler (`training::targets`), whose band/bearing/surface conventions the hunt
//! poser already shares through the one `band_lure` implementation.
//!
//! [`RolloutEnvBuilder`]: crab_world::training::inproc::RolloutEnvBuilder

use std::sync::Arc;

use bevy::prelude::*;
use crab_world::bot::arch::ArchId;
use crab_world::bot::headless::{HeadlessStack, WorldRole, force_serial_schedules, headless_stack};
use crab_world::terrain::TerrainGrid;
use crab_world::training::inproc::wire_rollout_training;
use crab_world::vehicle::VehiclePlugin;
use crab_world::{TrainConfig, Visuals};

/// The headless server world: the physics core, the crab stack, and the vehicle layer
/// — crafts are world content that strikes Sally and gets pushed back (rl#298
/// stage 1) — on the given ground. The crab-slot host harness arms its bridge +
/// policy on top of this; [`build_rollout_app`] arms the training systems instead.
/// (The windowed host composes the same content plugins onto its windowed base in
/// `render::app`; headless consumers all come through here.)
pub(crate) fn headless_server_world(
    num_envs: usize,
    role: WorldRole,
    grid: Arc<TerrainGrid>,
) -> App {
    let mut app = headless_stack(HeadlessStack {
        num_envs,
        role,
        grid,
        visuals: Visuals(false),
    });
    app.add_plugins(VehiclePlugin);
    app
}

/// Build one rollout worker's env: the headless server world on the canonical ground
/// (the plant's world half, rl#293 — recorded in the checkpoint sidecar beside the
/// friction cap) driven by the training systems in the same `BotSet::Think` slot the
/// host's inference policy occupies.
pub fn build_rollout_app(id: usize, config: &TrainConfig, arch: ArchId, num_envs: usize) -> App {
    let mut app = headless_server_world(num_envs, WorldRole::RolloutWorker, TerrainGrid::gcr());
    wire_rollout_training(&mut app, config, id, arch);
    force_serial_schedules(&mut app);
    app.finish();
    app.cleanup();
    app
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use crab_world::bot::actuator::CrabActions;
    use crab_world::bot::body::{CrabCarapace, CrabEnvId};
    use crab_world::bot::sensor::CrabTargets;
    use crab_world::bot::{CrabSpawns, RESET_GRACE_TICKS};
    use crab_world::terrain::Terrain;
    use crab_world::training::targets::BAND_MAX_M;
    use crab_world::vehicle::VehicleControls;

    use super::*;

    /// The production rollout env end-to-end through its public surface: the server
    /// world armed (vehicle layer present), every env's crab spawned, the training
    /// driver sampling live actions in the Think slot, and the ball target seeded to
    /// its band + surface conventions — the stage-4 acceptance in one world.
    #[test]
    fn rollout_env_is_the_server_world_driven_by_the_training_systems() {
        let m = 2usize;
        let dir = std::env::temp_dir().join(format!("rl_rollout_env_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = TrainConfig::try_parse_from([
            "rl",
            "--checkpoint-dir",
            dir.to_str().unwrap(),
            "--envs",
            &m.to_string(),
            "--seed",
            "20905",
        ])
        .expect("parse rollout TrainConfig");

        let mut app = build_rollout_app(0, &config, ArchId::DEFAULT, m);
        // Past spawn + settle grace, into live recording.
        for _ in 0..(RESET_GRACE_TICKS + 24) {
            app.update();
        }

        assert!(
            app.world().contains_resource::<VehicleControls>(),
            "the rollout env must be the server world — vehicle layer armed"
        );

        let mut crabs = app
            .world_mut()
            .query_filtered::<&CrabEnvId, With<CrabCarapace>>();
        let mut envs: Vec<usize> = crabs.iter(app.world()).map(|e| e.0).collect();
        envs.sort_unstable();
        assert_eq!(envs, vec![0, 1], "one crab per env in the one world");

        let actions = app.world().resource::<CrabActions>();
        for e in 0..m {
            assert!(
                actions.rows()[e].iter().any(|v| *v != 0.0),
                "env {e}: the training driver must be sampling live actions post-settle"
            );
        }
        let first: Vec<_> = (0..m).map(|e| actions.rows()[e]).collect();
        app.update();
        let actions = app.world().resource::<CrabActions>();
        assert!(
            (0..m).any(|e| actions.rows()[e] != first[e]),
            "consecutive steps must draw fresh exploration samples — an inference \
             (deterministic) driver would repeat under a frozen obs"
        );

        let spawns = app.world().resource::<CrabSpawns>();
        let targets = app.world().resource::<CrabTargets>();
        let terrain = app.world().resource::<Terrain>();
        for e in 0..m {
            let t = targets.get(e).expect("the ball target must be seeded");
            let origin = spawns.origin(e);
            let planar = Vec2::new(t.x - origin.x, t.z - origin.z).length();
            assert!(
                planar <= BAND_MAX_M + 1e-3,
                "env {e}: ball at {planar} m must sit inside the trained band"
            );
            let above = t.y - terrain.height(t.x, t.z);
            assert!(
                (0.0..1.0).contains(&above),
                "env {e}: ball rides {above} m above the surface — the ball target's \
                 surface-relative convention, not the hunt feed's fixed claw height"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
