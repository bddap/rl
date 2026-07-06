
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::sim::Pos;
use crab_world::Visuals;
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};
use crab_world::bot::sensor::CrabTargets;
use crab_world::bot::{BotSet, CrabSpawns};
use crab_world::crab_view::CrabBrainLabels;
use crab_world::play::Policy;

const CLAW_TARGET_Y: f32 = 0.3;

pub struct CrabPolicies(pub Vec<Policy>);

#[derive(Resource)]
pub struct ExternalCrabBridge {
    crabs: Vec<CrabBridge>,
}

struct CrabBridge {
    world_pos_m: Vec2,
    last_carapace_m: Option<Vec2>,
    yaw_turns: i32,
    settle: u32,
    hunt_target_m: Option<Vec2>,
    phys_digest: u64,
}

fn pos_to_m(p: Pos) -> Vec2 {
    let (x, z) = p.to_meters();
    Vec2::new(x, z)
}

struct CrabPlacement {
    shift: Vec2,
    /// The crab's game-world ground point (XZ).
    game_spot: Vec2,
}

impl CrabBridge {
    fn new(spawn: Pos) -> Self {
        Self {
            world_pos_m: pos_to_m(spawn),
            last_carapace_m: None,
            yaw_turns: 0,
            hunt_target_m: None,
            settle: crab_world::bot::RESET_GRACE_TICKS,
            phys_digest: 0,
        }
    }

    fn world_pos(&self) -> Pos {
        debug_assert!(
            self.world_pos_m.is_finite()
                && self.world_pos_m.x.abs() <= 100_000.0
                && self.world_pos_m.y.abs() <= 100_000.0,
            "external crab world_pos_m out of live bounds: {:?}",
            self.world_pos_m
        );
        Pos::from_meters(self.world_pos_m.x, self.world_pos_m.y)
    }

    fn render_placement_m(&self) -> Option<CrabPlacement> {
        self.last_carapace_m.map(|c| CrabPlacement {
            shift: self.world_pos_m - c,
            game_spot: self.world_pos_m,
        })
    }

    fn restart_to_spawn(&mut self, spawn: Pos) {
        self.world_pos_m = pos_to_m(spawn);
        self.last_carapace_m = None;
        self.settle = crab_world::bot::RESET_GRACE_TICKS;
    }
}

impl ExternalCrabBridge {
    pub fn new(spawns: &[Pos]) -> Self {
        Self {
            crabs: spawns.iter().map(|&s| CrabBridge::new(s)).collect(),
        }
    }

    pub fn crab_count(&self) -> usize {
        self.crabs.len()
    }

    pub fn crab_poses(&self) -> Vec<crate::server::CrabPose> {
        self.crabs
            .iter()
            .map(|c| crate::server::CrabPose {
                pos: c.world_pos(),
                yaw: c.yaw_turns,
                digest: c.phys_digest,
            })
            .collect()
    }

    pub fn set_hunt_target(&mut self, idx: usize, prey: Option<Pos>) {
        self.crabs[idx].hunt_target_m = prey.map(pos_to_m);
    }

    pub fn restart_to_spawns(&mut self, spawns: &[Pos]) {
        assert_eq!(
            spawns.len(),
            self.crabs.len(),
            "restart spawns must cover every bridged crab"
        );
        for (crab, &spawn) in self.crabs.iter_mut().zip(spawns) {
            crab.restart_to_spawn(spawn);
        }
    }
}

pub(crate) fn cold_respawn_armed_crab(world: &mut World) {
    use bevy::ecs::world::CommandQueue;

    let mut by_env: std::collections::BTreeMap<usize, Vec<Entity>> = Default::default();
    for (e, env) in world
        .query_filtered::<(Entity, &CrabEnvId), With<CrabBodyPart>>()
        .iter(world)
    {
        by_env.entry(env.0).or_default().push(e);
    }
    if by_env.is_empty() {
        return;
    }
    let origins = world.resource::<CrabSpawns>().0.clone();
    world.resource_scope(|world, assets: Mut<crab_world::bot::body::CrabAssets>| {
        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, world);
        for (env, parts) in by_env {
            let origin = origins.get(env).copied().unwrap_or(Vec3::ZERO);
            crab_world::bot::respawn_crab(&mut commands, &assets, parts.into_iter(), origin, env);
        }
        queue.apply(world);
    });
}

pub fn sync_external_crab(sim: &mut crate::sim::Sim, bridge: &mut ExternalCrabBridge) {
    for (idx, pose) in bridge.crab_poses().into_iter().enumerate() {
        sim.set_external_crab_pose(idx, pose.pos, pose.yaw, pose.digest);
    }
    for idx in 0..bridge.crab_count() {
        let prey = sim.nearest_living_player_pos(idx);
        bridge.set_hunt_target(idx, prey);
    }
}

#[allow(clippy::type_complexity)]
fn hash_crab_physics(
    policies: NonSend<CrabPolicies>,
    mut bridge: ResMut<ExternalCrabBridge>,
    bodies: Query<
        (
            &CrabEnvId,
            &Transform,
            &Velocity,
            Option<&CrabJoint>,
            Option<&CrabCarapace>,
        ),
        With<CrabBodyPart>,
    >,
) {
    assert_eq!(
        policies.0.len(),
        bridge.crabs.len(),
        "one policy per bridged crab"
    );
    for (idx, (policy, crab)) in policies.0.iter().zip(bridge.crabs.iter_mut()).enumerate() {
        let phys = crab_world::bot::physics_digest::crab_state_digest(
            bodies
                .iter()
                .filter(|(env, ..)| env.0 == idx)
                .map(|(_, t, v, j, c)| (t, v, j, c)),
        );
        crab.phys_digest = phys ^ policy.weights_digest().map_or(0, std::num::NonZeroU64::get);
    }
}

#[derive(Resource)]
pub struct ExternalCrabArmed;

fn external_crab_armed(active: Option<Res<ExternalCrabArmed>>) -> bool {
    active.is_some()
}

pub fn arm(world: &mut World) {
    world.insert_resource(ExternalCrabArmed);
    world.insert_resource(crab_world::bot::CrabRescueIsFault);
}

pub struct ExternalCrabPlugin {
    pub checkpoint_dirs: Vec<std::path::PathBuf>,
    pub crab_spawns: Vec<Pos>,
}

impl Plugin for ExternalCrabPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            !self.checkpoint_dirs.is_empty(),
            "a round runs at least one brain binding (rl#114)"
        );
        assert_eq!(
            self.checkpoint_dirs.len(),
            self.crab_spawns.len(),
            "one crab spawn per brain binding — the sim's crab count must match the bindings"
        );
        let policies: Vec<Policy> = self
            .checkpoint_dirs
            .iter()
            .enumerate()
            .map(|(idx, dir)| {
                let policy = Policy::load(dir);
                if !policy.is_loaded() {
                    warn!(
                        "external_crab: no usable checkpoint for crab {idx} at {} — that NN \
                         crab holds rest pose",
                        dir.display()
                    );
                }
                policy
            })
            .collect();
        app.insert_non_send_resource(CrabPolicies(policies));
        app.insert_resource(ExternalCrabBridge::new(&self.crab_spawns));

        app.add_systems(
            Update,
            ensure_crab_env
                .run_if(external_crab_armed)
                .before(crab_world::bot::spawn_initial_crabs),
        );
        app.add_systems(
            Update,
            crab_world::bot::spawn_initial_crabs
                .run_if(external_crab_armed)
                .run_if(crab_not_yet_spawned),
        );

        app.add_systems(
            FixedUpdate,
            (
                set_crab_walk_target.before(BotSet::Sense),
                run_crab_policy.in_set(BotSet::Think),
            )
                .run_if(external_crab_armed),
        );
        // Publish each binding's on-screen brain label (rl#200 increment 7). In FixedUpdate
        // deliberately: only the physics-pumping peer (solo/host) advances FixedUpdate
        // (the wall-clock auto-pump is PARKED — to a 86400s timestep, so "never" really
        // means "not within a day's uptime" — and `pump_fixed_steps` is lockstep-driven),
        // so this is host-only by construction; on a remote-adopt client the articulation
        // `apply` is the sole label writer and the two can't fight over the resource.
        app.init_resource::<CrabBrainLabels>();
        app.add_systems(
            FixedUpdate,
            publish_brain_labels.run_if(external_crab_armed),
        );
        app.add_systems(
            FixedUpdate,
            integrate_crab
                .after(PhysicsSet::Writeback)
                .run_if(external_crab_armed),
        );
        app.add_systems(
            FixedUpdate,
            hash_crab_physics
                .after(integrate_crab)
                .after(PhysicsSet::Writeback)
                .run_if(external_crab_armed),
        );

        if app.world().get_resource::<Visuals>().is_some_and(|v| v.0) {
            app.add_systems(
                FixedUpdate,
                publish_skin_repose
                    .after(integrate_crab)
                    .run_if(external_crab_armed),
            );
        }
    }
}

fn ensure_crab_env(
    bridge: Res<ExternalCrabBridge>,
    mut num_envs: ResMut<crab_world::bot::NumEnvs>,
) {
    let want = bridge.crab_count();
    if num_envs.0 < want {
        num_envs.0 = want;
    }
}

/// Keep [`CrabBrainLabels`] current with the bindings — one label per env, formatted by the
/// ONE formatter (`Policy::brain_label`), write-on-change. GCR policies never hot-reload, so
/// this settles to one write per arm; it stays a system (not an arm-time one-shot) so the
/// labels can never go stale against whatever drives the crabs. Teardown clears the resource
/// ([`crate::render`]'s `teardown_round`), un-labeling the crab bodies that outlive the round.
fn publish_brain_labels(policies: NonSend<CrabPolicies>, mut labels: ResMut<CrabBrainLabels>) {
    let want: Vec<String> = policies.0.iter().map(|p| p.brain_label()).collect();
    if labels.0 != want {
        labels.0 = want;
    }
}

fn crab_not_yet_spawned(crabs: Query<(), With<CrabCarapace>>) -> bool {
    crabs.is_empty()
}

fn set_crab_walk_target(
    bridge: Res<ExternalCrabBridge>,
    spawns: Res<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
) {
    for (idx, crab) in bridge.crabs.iter().enumerate() {
        let Some(slot) = targets.envs.get_mut(idx) else {
            return;
        };
        let Some(hunt) = crab.hunt_target_m else {
            *slot = None;
            continue;
        };
        let to_prey = hunt - crab.world_pos_m;
        if to_prey.length_squared() < 1e-6 {
            continue;
        }

        let origin = spawns.0.get(idx).copied().unwrap_or(Vec3::ZERO);
        let carapace = carapace_q
            .iter()
            .find(|(env, _)| env.0 == idx)
            .map(|(_, t)| t.translation)
            .unwrap_or(origin);

        let target = Vec3::new(
            carapace.x + to_prey.x,
            CLAW_TARGET_Y,
            carapace.z + to_prey.y,
        );
        *slot = Some(target);
    }
}

fn run_crab_policy(
    policies: NonSend<CrabPolicies>,
    mut bridge: ResMut<ExternalCrabBridge>,
    obs: Res<crab_world::bot::sensor::CrabObservation>,
    mut actions: ResMut<crab_world::bot::actuator::CrabActions>,
) {
    assert_eq!(
        policies.0.len(),
        bridge.crabs.len(),
        "one policy per bridged crab"
    );
    for (idx, (policy, crab)) in policies.0.iter().zip(bridge.crabs.iter_mut()).enumerate() {
        if crab.settle > 0 {
            crab.settle = crab_world::bot::settle_countdown(crab.settle);
            if let Some(a) = actions.envs.get_mut(idx) {
                *a = [0.0; crab_world::bot::actuator::ACTION_SIZE];
            }
            continue;
        }
        if let (Some(o), Some(a)) = (obs.envs.get(idx), actions.envs.get_mut(idx)) {
            *a = policy.act(o);
        }
    }
}

fn integrate_crab(
    mut bridge: ResMut<ExternalCrabBridge>,
    mut rescued: MessageReader<crab_world::bot::CrabRescued>,
    carapace_q: Query<(&CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
) {
    let rescued_envs: std::collections::BTreeSet<usize> = rescued.read().map(|m| m.env).collect();
    for (idx, crab) in bridge.crabs.iter_mut().enumerate() {
        if rescued_envs.contains(&idx) {
            crab.last_carapace_m = None;
            crab.settle = crab_world::bot::RESET_GRACE_TICKS;
        }

        let Some((_, t, vel)) = carapace_q.iter().find(|(env, _, _)| env.0 == idx) else {
            continue;
        };
        if !t.translation.is_finite() {
            continue;
        }
        let here = Vec2::new(t.translation.x, t.translation.z);
        if crab.settle == 0
            && let Some(prev) = crab.last_carapace_m
        {
            crab.world_pos_m += here - prev;
        }
        crab.last_carapace_m = Some(here);

        let v = Vec2::new(vel.linear.x, vel.linear.z);
        if v.length_squared() > 1e-4 {
            let radians = v.x.atan2(v.y);
            crab.yaw_turns = crate::sim::trig_client::radians_to_turns(radians);
        }
    }
}

fn publish_skin_repose(
    bridge: Res<ExternalCrabBridge>,
    repose_out: Option<ResMut<crab_world::bot::skin::CrabSkinRepose>>,
) {
    let Some(mut out) = repose_out else {
        return;
    };
    let rs = crate::render::world_render_scale();
    out.0 = bridge
        .crabs
        .iter()
        .enumerate()
        .filter_map(|(idx, crab)| {
            crab.render_placement_m().map(|r| {
                let s = r.game_spot * rs - (r.game_spot - r.shift);
                (
                    idx,
                    crab_world::bot::skin::SkinRepose {
                        shift: Vec3::new(s.x, 0.0, s.y),
                    },
                )
            })
        })
        .collect();
}

mod probe;

pub use probe::{ProbeSample, StabilityResult, run_headless_probe, run_vehicle_stability_probe};
