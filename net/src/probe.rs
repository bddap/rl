//! Headless NN-crab probes: the host's crab slot pumping the one world (rl#298
//! stage 5) beside an authoritative [`Sim`], no renderer, sampling sim + world state
//! per tick. Consumed by `game nn-crab-probe` (behavior/
//! determinism A/B) and `game nn-crab-vehicle-stability` (the rl#137 ram test).
//!
//! The probe's sim is stepped 1:1 with the fixed schedule (one physics pass per sim
//! tick — the probe's historical cadence, kept so hash logs stay comparable), and the
//! claws are STRIPPED from the fed poses: the probe measures pursuit, and a downed
//! prey would end the very chase being measured.

use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;

use crate::crab_slot::{self, NnCrabPlugin, arm, park_fixed_auto_pump, restart_crabs_to_spawns};
use crate::sim::{Externals, Input, PlayerId, Pos, Sim};
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint};
use crab_world::bot::physics_digest::crab_state_digest;
use crab_world::bot::sensor::CrabTargets;

#[derive(Clone, Copy, Debug)]
pub struct ProbeSample {
    pub tick: u64,
    pub crab_x_m: f32,
    pub crab_z_m: f32,
    pub dist_to_prey_m: f32,
    pub state_hash: u64,
    pub carapace_x: f32,
    pub carapace_z: f32,
    pub carapace_y: f32,
    /// Carapace height above the local ground surface — the grounded-ness measure
    /// (absolute y is a mountainside elevation on terrain, meaningless as one).
    pub carapace_above_ground: f32,
    pub min_claw_to_target_m: f32,
}

struct Probe {
    app: App,
    sim: Sim,
    samples: Vec<ProbeSample>,
    log_every: u64,
}

impl Probe {
    fn new(policy: crab_world::policy::Policy, seed: u64, visuals: crab_world::Visuals) -> Self {
        use crab_world::bot::headless::{
            HeadlessStack, WorldRole, force_serial_schedules, headless_stack,
            pin_single_thread_pools,
        };

        pin_single_thread_pools();

        let me = PlayerId(0);
        let sim = Sim::new(seed, &[me]);
        let spawns: Vec<Pos> = sim.crabs().iter().map(|c| c.pos()).collect();

        // The BARE crab stack, not `headless_server_world`: the stability probe
        // hand-spawns its ram craft, which the vehicle layer's `manage_vehicles`
        // would despawn as pilotless.
        let mut app = headless_stack(HeadlessStack {
            num_envs: spawns.len(),
            role: WorldRole::Standalone,
            // The probe models the GCR host, so it steps the host's ground — the
            // canonical terrain bake (rl#209, rl#293).
            grid: crab_world::terrain::TerrainGrid::gcr(),
            visuals,
        });
        app.add_plugins(NnCrabPlugin::new(vec![policy], spawns.clone()));
        arm(app.world_mut());
        park_fixed_auto_pump(&mut app);
        restart_crabs_to_spawns(app.world_mut(), &spawns);
        force_serial_schedules(&mut app);

        Self {
            app,
            sim,
            samples: Vec::new(),
            log_every: 1,
        }
    }

    /// One probe tick through the host seam: `Update` wrap, one fixed pump (the
    /// probe's 1:1 cadence), poses off the world into the sim step, next tick's hunt
    /// fed back — the same pump→collect→feed seam the hosts run ([`pump_slot_steps`]).
    fn tick(&mut self) {
        let inputs = crab_slot::slot_inputs(&self.sim);
        self.app.update();
        let mut poses = crab_slot::pump_slot_steps(self.app.world_mut(), 1, &inputs);
        for p in &mut poses {
            // Pursuit probe: no downs (module doc) — the pose crosses, the claws don't.
            p.claws.clear();
        }
        let prey = self.sim.nearest_living_player_pos(0);

        let me = PlayerId(0);
        self.sim.step(
            &std::collections::BTreeMap::from([(me, Input::from_axes(0.0, 0.0))]),
            Externals::crabs_only(&poses),
        );

        let tick = self.sim.tick();
        if tick == 1 || tick.is_multiple_of(self.log_every) {
            self.sample(tick, prey);
        }
    }

    fn sample(&mut self, tick: u64, prey: Option<Pos>) {
        let world = self.app.world_mut();
        let terrain = world.resource::<crab_world::terrain::Terrain>().clone();

        let crab = self.sim.crabs()[0].pos();
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

        // The probe's hash-log contract: the sim hash PLUS the full-body crab digest, so a
        // run-vs-run diff catches float divergence anywhere in the crab's rapier state. This
        // fold is probe-only — the runtime sim hash stopped carrying it in rl#223 (clients
        // adopt host state and never step, so a cross-peer digest compared host values with
        // themselves).
        let state_hash = self.sim.state_hash()
            ^ crab_state_digest(
                world
                    .query_filtered::<(
                        &Transform,
                        &Velocity,
                        Option<&CrabJoint>,
                        Option<&CrabCarapace>,
                    ), With<CrabBodyPart>>()
                    .iter(world),
            );

        let (carapace_x, carapace_y, carapace_z, carapace_above_ground) = world
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>()
            .iter(world)
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| {
                let p = t.translation;
                (p.x, p.y, p.z, p.y - terrain.height(p.x, p.z))
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0));

        let target = world.resource::<CrabTargets>().get(0);
        let min_claw_to_target_m = target
            .map(|target| {
                world
                    .query_filtered::<(&CrabEnvId, &Transform), With<CrabClawTip>>()
                    .iter(world)
                    .filter(|(env, tip)| env.0 == 0 && tip.translation.is_finite())
                    .map(|(_, tip)| tip.translation.distance(target))
                    .fold(f32::INFINITY, f32::min)
            })
            .unwrap_or(f32::NAN);

        self.samples.push(ProbeSample {
            tick,
            crab_x_m,
            crab_z_m,
            dist_to_prey_m,
            state_hash,
            carapace_x,
            carapace_z,
            carapace_y,
            carapace_above_ground,
            min_claw_to_target_m,
        });
    }

    fn carapace(&mut self) -> Vec3 {
        let world = self.app.world_mut();
        world
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>()
            .iter(world)
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| t.translation)
            .unwrap_or(Vec3::ZERO)
    }
}

/// `visuals`: `Visuals(true)` steps the ARMED-RENDER configuration headless — the
/// skin and the rl#116 pose sentinel all live — which is the
/// exact configuration the GCR play-day crash showed no headless test covered. The
/// determinism/behavior probes pass `Visuals(false)`, matching what they hash.
pub fn run_headless_probe(
    policy: crab_world::policy::Policy,
    seed: u64,
    ticks: u64,
    log_every: u64,
    visuals: crab_world::Visuals,
) -> Vec<ProbeSample> {
    let mut probe = Probe::new(policy, seed, visuals);
    probe.log_every = log_every.max(1);
    for _ in 0..ticks {
        probe.tick();
    }
    probe.samples
}

pub struct StabilityResult {
    pub samples: Vec<ProbeSample>,
    pub ram_tick: u64,
}

impl StabilityResult {
    pub fn carapace_stayed_finite(&self) -> bool {
        self.samples.iter().all(|s| {
            s.carapace_x.is_finite() && s.carapace_y.is_finite() && s.carapace_z.is_finite()
        })
    }
}

pub fn run_vehicle_stability_probe(
    policy: crab_world::policy::Policy,
    seed: u64,
    warmup: u64,
    post: u64,
) -> StabilityResult {
    use crab_world::vehicle::{VehicleKind, spawn_ram_vehicle};

    let mut probe = Probe::new(policy, seed, crab_world::Visuals(false));
    for _ in 0..warmup {
        probe.tick();
    }
    let ram_tick = probe.sim.tick();

    let carapace = probe.carapace();
    let spawn_at = Transform::from_translation(carapace + Vec3::new(1.2, -0.15, 0.0));
    let ram_velocity = Velocity {
        linear: Vec3::new(-10.0, 0.0, 0.0),
        angular: Vec3::ZERO,
    };
    spawn_ram_vehicle(
        probe.app.world_mut(),
        VehicleKind::Plane,
        spawn_at,
        ram_velocity,
    );

    for _ in 0..post {
        probe.tick();
    }

    StabilityResult {
        samples: probe.samples,
        ram_tick,
    }
}
