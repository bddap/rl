
use bevy::prelude::*;

use crate::sim::{Input, PlayerId, Pos, Sim};
use crab_world::bot::body::{CrabCarapace, CrabEnvId};
use crab_world::bot::sensor::CrabTargets;

use super::{
    ExternalCrabBridge, ExternalCrabPlugin, hash_crab_physics, integrate_crab, sync_external_crab,
};

#[derive(Clone, Copy, Debug)]
pub struct ProbeSample {
    pub tick: u64,
    pub crab_x_m: f32,
    pub crab_z_m: f32,
    pub dist_to_prey_m: f32,
    pub state_hash: u64,
    pub carapace_arena_x: f32,
    pub carapace_arena_z: f32,
    pub carapace_y: f32,
    pub min_claw_to_target_m: f32,
}

struct ProbeDriver {
    sim: Sim,
    samples: Vec<ProbeSample>,
    log_every: u64,
}

fn probe_step(
    mut driver: NonSendMut<ProbeDriver>,
    mut bridge: ResMut<ExternalCrabBridge>,
    targets: Res<CrabTargets>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    claw_q: Query<(&CrabEnvId, &Transform), With<crab_world::bot::body::CrabClawTip>>,
) {
    sync_external_crab(&mut driver.sim, &mut bridge);
    let prey = driver.sim.nearest_living_player_pos(0);

    let me = PlayerId(0);
    driver.sim.step(&std::collections::BTreeMap::from([(
        me,
        Input::from_axes(0.0, 0.0),
    )]));

    let tick = driver.sim.tick();
    if tick == 1 || tick.is_multiple_of(driver.log_every) {
        let crab = driver.sim.crabs()[0].pos();
        let (crab_x_m, crab_z_m) = crab.to_meters();
        let dist_to_prey_m = prey
            .map(|p| {
                // Integer delta first, then the one Pos→meters rule — bit-identical to
                // the old inline `/ UNIT` casts.
                let (dx, dz) = Pos {
                    x: p.x - crab.x,
                    z: p.z - crab.z,
                }
                .to_meters();
                (dx * dx + dz * dz).sqrt()
            })
            .unwrap_or(f32::NAN);
        let state_hash = driver.sim.state_hash();

        let (carapace_arena_x, carapace_y, carapace_arena_z) = carapace_q
            .iter()
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| (t.translation.x, t.translation.y, t.translation.z))
            .unwrap_or((0.0, 0.0, 0.0));

        let min_claw_to_target_m = targets
            .get(0)
            .map(|target| {
                claw_q
                    .iter()
                    .filter(|(env, tip)| env.0 == 0 && tip.translation.is_finite())
                    .map(|(_, tip)| tip.translation.distance(target))
                    .fold(f32::INFINITY, f32::min)
            })
            .unwrap_or(f32::NAN);

        driver.samples.push(ProbeSample {
            tick,
            crab_x_m,
            crab_z_m,
            dist_to_prey_m,
            state_hash,
            carapace_arena_x,
            carapace_arena_z,
            carapace_y,
            min_claw_to_target_m,
        });
    }
}

fn headless_nn_crab_app(checkpoint_dir: &std::path::Path, crab_spawn: Pos) -> bevy::app::App {
    use crab_world::bot::headless::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
    };

    pin_single_thread_pools();

    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        // The probe models the GCR client, so it steps the client's OPEN inference
        // field (rl#209), not the walled training box.
        arena: crab_world::physics::Arena::OpenField,
    });
    app.add_plugins(ExternalCrabPlugin {
        checkpoint_dirs: vec![checkpoint_dir.to_path_buf()],
        crab_spawns: vec![crab_spawn],
    });
    super::arm(app.world_mut());
    force_serial_schedules(&mut app);
    app
}

pub fn run_headless_probe(
    checkpoint_dir: &std::path::Path,
    seed: u64,
    ticks: u64,
    log_every: u64,
) -> Vec<ProbeSample> {
    let me = PlayerId(0);
    let mut sim = Sim::new(seed, &[me]);
    let crab_spawn = sim.crabs()[0].pos();
    let crab = sim.crabs()[0];
    sim.set_external_crab_pose(0, crab.pos(), crab.yaw(), 0);

    let mut app = headless_nn_crab_app(checkpoint_dir, crab_spawn);
    app.insert_non_send_resource(ProbeDriver {
        sim,
        samples: Vec::new(),
        log_every: log_every.max(1),
    });
    app.add_systems(
        FixedUpdate,
        probe_step.after(integrate_crab).after(hash_crab_physics),
    );

    for _ in 0..ticks {
        app.update();
    }
    app.world()
        .get_non_send_resource::<ProbeDriver>()
        .map(|d| d.samples.clone())
        .unwrap_or_default()
}


pub struct StabilityResult {
    pub samples: Vec<ProbeSample>,
    pub ram_tick: u64,
}

impl StabilityResult {
    pub fn carapace_stayed_finite(&self) -> bool {
        self.samples.iter().all(|s| {
            s.carapace_arena_x.is_finite()
                && s.carapace_y.is_finite()
                && s.carapace_arena_z.is_finite()
        })
    }
}

pub fn run_vehicle_stability_probe(
    checkpoint_dir: &std::path::Path,
    seed: u64,
    warmup: u64,
    post: u64,
) -> StabilityResult {
    use bevy_rapier3d::prelude::Velocity;
    use crab_world::vehicle::{VehicleKind, spawn_ram_vehicle};

    let me = PlayerId(0);
    let mut sim = Sim::new(seed, &[me]);
    let crab_spawn = sim.crabs()[0].pos();
    let crab = sim.crabs()[0];
    sim.set_external_crab_pose(0, crab.pos(), crab.yaw(), 0);

    let mut app = headless_nn_crab_app(checkpoint_dir, crab_spawn);
    app.insert_non_send_resource(ProbeDriver {
        sim,
        samples: Vec::new(),
        log_every: 1,
    });
    app.add_systems(
        FixedUpdate,
        probe_step.after(integrate_crab).after(hash_crab_physics),
    );

    for _ in 0..warmup {
        app.update();
    }
    let ram_tick = app
        .world()
        .get_non_send_resource::<ProbeDriver>()
        .map(|d| d.sim.tick())
        .unwrap_or(warmup);

    let carapace = {
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>();
        q.iter(app.world())
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| t.translation)
            .unwrap_or(Vec3::ZERO)
    };
    let spawn_at = Transform::from_translation(carapace + Vec3::new(1.2, -0.15, 0.0));
    let ram_velocity = Velocity {
        linear: Vec3::new(-10.0, 0.0, 0.0),
        angular: Vec3::ZERO,
    };
    spawn_ram_vehicle(app.world_mut(), VehicleKind::Plane, spawn_at, ram_velocity);

    for _ in 0..post {
        app.update();
    }

    let samples = app
        .world()
        .get_non_send_resource::<ProbeDriver>()
        .map(|d| d.samples.clone())
        .unwrap_or_default();
    StabilityResult { samples, ram_tick }
}
