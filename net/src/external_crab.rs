//! Drives NN "Sally" crabs in GCR: each crab runs its trained policy in a local
//! physics arena; the bridge poses its hunt target from the sim and integrates
//! carapace deltas back into world position.
//!
//! The posed hunt target is clamped to the training band's far edge
//! ([`TARGET_ARENA_HALF`], ~9 m) along the true crab→player bearing, re-posed each
//! tick as she closes — heading stays honest, and the target_local obs channel
//! stays in-distribution. Without the clamp Sally always starts OOD: players spawn
//! beyond `sim::MIN_CRAB_SPAWN_DISTANCE` (well past the band) and roam farther mid-game, and
//! measured (rl#144) a healthy brain closes ~7× slower on a ~21 m target than the
//! chase eval's in-band rate — the ±5σ obs clamp alone does NOT yield a full-tilt
//! walk. The clamp tracks the same constant training draws targets from, so it
//! stays correct if the band is later extended.
//!
//! Necessary, not sufficient: closure is also bearing-dependent (the brain strides
//! +X but shuffles at other bearings, and the chase eval only poses +X — rl#239),
//! and the spawn-relative body.pos obs channel still drifts OOD on a long
//! open-field chase (rl#240) — neither is fixable at the posing layer. For the
//! latter, [`bound_body_pos_drift`] measures the drift every tick and carries the
//! fix — recenter the local arena by teleporting the drifted crab back onto its
//! spawn origin — behind [`ARM_BODY_POS_RECENTER`] (default off, see its doc for
//! the sequencing gate).

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::sim::Pos;
use crab_world::Visuals;
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId};
use crab_world::bot::sensor::CrabTargets;
use crab_world::bot::{BotSet, CrabSpawns};
use crab_world::crab_view::CrabBrainLabels;
use crab_world::play::Policy;
use crab_world::training::targets::TARGET_ARENA_HALF;

const CLAW_TARGET_Y: f32 = 0.3;

/// rl#240 flip: recenter the local arena (teleport the drifted crab back onto its spawn
/// origin) whenever the spawn-relative body.pos obs channel leaves the training band.
/// GATED OFF: flipping changes what the net observes mid-chase — a training-relevant
/// change — and is sequenced behind the rl#239 honest-bearing eval deploy and the σ-floor
/// experiment's conclusion (one live training change at a time). Until then
/// [`bound_body_pos_drift`] only measures. Flip = set `true`; behavior is otherwise
/// bit-identical for existing brains. Flip precondition beyond the sequencing gate:
/// vehicles share the crab's arena frame (rl#235 rams, and the render anchor derives
/// from it), so the teleport must carry co-arena vehicles too — see rl#240.
const ARM_BODY_POS_RECENTER: bool = false;

/// Arms [`bound_body_pos_drift`]'s recenter teleport. Inserted by the plugin iff
/// [`ARM_BODY_POS_RECENTER`], which stays the sole authority; private so nothing else
/// can arm it (tests live in this module).
#[derive(Resource)]
struct BodyPosRecenter;

/// Each further [`DRIFT_LOG_STEP_M`] of unrecentered peak drift earns one more log line.
const DRIFT_LOG_STEP_M: f32 = 5.0;

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
    /// Log cursor for [`bound_body_pos_drift`]'s measurement: the next spawn-relative
    /// drift (m) worth a log line. Starts at the in-distribution edge, then advances by
    /// [`DRIFT_LOG_STEP_M`] per line so a long unrecentered chase is quantified without
    /// flooding.
    next_drift_log_m: f32,
    /// Recenter teleports this round (rl#240) — stays 0 while the flip is gated off.
    recenters: u32,
    /// Consecutive post-settle ticks [`integrate_crab`] found no carapace for this env —
    /// the crab's world pos freezes at its last integrate (still lethal there) while the
    /// sim looks healthy, so the miss is counted and reported like drift, never skipped
    /// silently (rl#241). Resets when the carapace comes back.
    missed_carapace_ticks: u64,
    /// Log cursor for the miss counter, [`next_drift_log_m`]'s sibling: the next miss
    /// count worth a log line (doubles per line, so a persistent miss is quantified
    /// without flooding).
    next_miss_log_ticks: u64,
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
            next_drift_log_m: TARGET_ARENA_HALF,
            recenters: 0,
            missed_carapace_ticks: 0,
            next_miss_log_ticks: 1,
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
        self.next_drift_log_m = TARGET_ARENA_HALF;
        self.recenters = 0;
        self.missed_carapace_ticks = 0;
        self.next_miss_log_ticks = 1;
    }

    /// One post-settle tick with no carapace to integrate: count it, and report at
    /// doubling thresholds — loud but bounded, the drift guard's reporting shape (rl#241).
    fn note_missed_carapace(&mut self, idx: usize) {
        self.missed_carapace_ticks += 1;
        if self.missed_carapace_ticks >= self.next_miss_log_ticks {
            error!(
                "external_crab: env {idx} has no carapace to integrate — world pos frozen \
                 for {} ticks (the crab still kills players at its stale position); a \
                 despawn/wiring bug, not a legitimate state (rl#241)",
                self.missed_carapace_ticks
            );
            self.next_miss_log_ticks = self.missed_carapace_ticks * 2;
        }
    }

    fn carapace_found(&mut self) {
        self.missed_carapace_ticks = 0;
        self.next_miss_log_ticks = 1;
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
        sim.set_external_crab_pose(idx, pose.pos, pose.yaw);
    }
    for idx in 0..bridge.crab_count() {
        let prey = sim.nearest_living_player_pos(idx);
        bridge.set_hunt_target(idx, prey);
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
    /// The policies the launch gate loaded and vetted — the plugin never touches disk, so
    /// what the gate armed IS what drives (rl#241: the old plugin-side re-load could race
    /// a checkpoint swap and warn-and-arm a rest-pose statue the gate never saw). `Mutex<
    /// Option<…>>` only because `Plugin::build` takes `&self`; `build` moves them out.
    policies: std::sync::Mutex<Option<Vec<Policy>>>,
    crab_spawns: Vec<Pos>,
}

impl ExternalCrabPlugin {
    pub fn new(policies: Vec<Policy>, crab_spawns: Vec<Pos>) -> Self {
        Self {
            policies: std::sync::Mutex::new(Some(policies)),
            crab_spawns,
        }
    }
}

impl Plugin for ExternalCrabPlugin {
    fn build(&self, app: &mut App) {
        let policies = self
            .policies
            .lock()
            .unwrap()
            .take()
            .expect("ExternalCrabPlugin is built once");
        assert!(
            !policies.is_empty(),
            "a round runs at least one brain binding (rl#114)"
        );
        assert_eq!(
            policies.len(),
            self.crab_spawns.len(),
            "one crab spawn per brain binding — the sim's crab count must match the bindings"
        );
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

        if ARM_BODY_POS_RECENTER {
            app.insert_resource(BodyPosRecenter);
        }
        app.add_systems(
            FixedUpdate,
            (
                // After rescue: both write crab Transforms before Sense; the edge makes the
                // interleaving deterministic (a rescued env respawns at origin, so the guard
                // then sees ~0 drift instead of racing the respawn).
                bound_body_pos_drift
                    .after(crab_world::bot::rescue_nonfinite_crabs)
                    .before(set_crab_walk_target),
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

/// rl#240 guard for the spawn-relative body.pos obs channel: training bounds it to the
/// walled box, OpenField doesn't, so a long chase walks it arbitrarily OOD. Always
/// MEASURES (rate-limited warn lines quantify the drift); when [`BodyPosRecenter`] is
/// armed it also FIXES it — teleport every part of the drifted env back onto its spawn
/// origin in one tick (a uniform Transform shift, a clean multibody teleport through
/// rapier — pinned by crab-world's `uniform_part_shift_teleports_the_multibody_cleanly`)
/// and shift the world-pos integrator's `prev` by the same delta so the teleport never
/// counts as motion. body.pos snaps to ~0, exactly the every-episode spawn distribution.
///
/// Ordering: before [`set_crab_walk_target`] (so the target is posed from the
/// post-teleport carapace) and hence before Sense and rapier's SyncBackend.
fn bound_body_pos_drift(
    mut bridge: ResMut<ExternalCrabBridge>,
    spawns: Res<CrabSpawns>,
    armed: Option<Res<BodyPosRecenter>>,
    mut targets: ResMut<CrabTargets>,
    mut parts: Query<(&CrabEnvId, &mut Transform, Option<&CrabCarapace>), With<CrabBodyPart>>,
) {
    for (idx, crab) in bridge.crabs.iter_mut().enumerate() {
        let Some(carapace) = parts
            .iter()
            .find(|(env, _, cara)| env.0 == idx && cara.is_some())
            .map(|(_, t, _)| t.translation)
        else {
            // Same absence [`integrate_crab`] counts and reports this tick (rl#241) —
            // one counter, not two log streams for one missing entity.
            continue;
        };
        if !carapace.is_finite() {
            continue; // the rescue path owns non-finite crabs
        }
        let origin = spawns.0.get(idx).copied().unwrap_or(Vec3::ZERO);
        let drift = Vec2::new(carapace.x - origin.x, carapace.z - origin.z);
        let drift_m = drift.length();
        if drift_m <= TARGET_ARENA_HALF {
            continue;
        }

        if armed.is_some() {
            let delta = Vec3::new(-drift.x, 0.0, -drift.y);
            for (env, mut t, _) in parts.iter_mut() {
                if env.0 == idx {
                    t.translation += delta;
                }
            }
            if let Some(prev) = crab.last_carapace_m.as_mut() {
                *prev += Vec2::new(delta.x, delta.z);
            }
            // Carry the posed target into the new frame too: set_crab_walk_target normally
            // re-poses it right after, but its prey-on-top-of-crab early-out (`to_prey ≈ 0`)
            // keeps the previous slot — which would be a whole `-delta` stale after this
            // teleport, exactly the OOD spike this system exists to prevent.
            if let Some(t) = targets.envs.get_mut(idx).and_then(|s| s.as_mut()) {
                *t += delta;
            }
            crab.recenters += 1;
            info!(
                "external_crab: recentered env {idx}'s local arena by {drift_m:.1} m \
                 (recenter #{} this round, rl#240)",
                crab.recenters
            );
        } else if drift_m >= crab.next_drift_log_m {
            warn!(
                "external_crab: env {idx} body.pos drifted {drift_m:.1} m from spawn — outside \
                 the {TARGET_ARENA_HALF} m in-distribution radius, obs OOD (rl#240; recenter \
                 gated off)"
            );
            crab.next_drift_log_m = drift_m + DRIFT_LOG_STEP_M;
        }
    }
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
        // In-distribution guard (rl#144, module header): pose the target at most one
        // band-edge away along the true bearing.
        let to_prey = to_prey.clamp_length_max(TARGET_ARENA_HALF);

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
    let crab_count = bridge.crabs.len();
    assert_eq!(policies.0.len(), crab_count, "one policy per bridged crab");
    for (idx, (policy, crab)) in policies.0.iter().zip(bridge.crabs.iter_mut()).enumerate() {
        if crab.settle > 0 {
            crab.settle = crab_world::bot::settle_countdown(crab.settle);
            // Pre-spawn the env slots don't exist yet (`spawn_initial_crabs` sizes them on
            // the first armed Update, and FixedUpdate can tick first) — that window lives
            // entirely inside the settle grace, which is why the slot check below is a
            // hard assert and not a skip.
            if let Some(a) = actions.envs.get_mut(idx) {
                *a = [0.0; crab_world::bot::actuator::ACTION_SIZE];
            }
            continue;
        }
        // Post-settle, a missing slot means this crab would hold rest pose in a live
        // round — a wiring bug, never a condition to skip past silently (rl#241).
        let (Some(o), Some(a)) = (obs.envs.get(idx), actions.envs.get_mut(idx)) else {
            panic!(
                "external_crab: crab {idx} has no env slot ({} obs / {} action slots sized \
                 for {} crabs) — it would silently hold rest pose in a live round (rl#241)",
                obs.envs.len(),
                actions.envs.len(),
                crab_count,
            );
        };
        *a = policy.act(o);
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
            // The rescue respawned it at the origin — re-arm the drift measurement.
            crab.next_drift_log_m = TARGET_ARENA_HALF;
        }

        let Some((_, t, vel)) = carapace_q.iter().find(|(env, _, _)| env.0 == idx) else {
            // Pre-spawn (inside the settle grace) the carapace legitimately doesn't
            // exist yet; past it, a miss freezes this crab's world pos — count and
            // report it (rl#241).
            if crab.settle == 0 {
                crab.note_missed_carapace(idx);
            }
            continue;
        };
        crab.carapace_found();
        if !t.translation.is_finite() {
            continue; // the rescue path owns non-finite crabs (a Fault when armed)
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

#[cfg(test)]
mod recenter_tests {
    use crab_world::bot::body::CrabBodyPart;
    use crab_world::bot::headless::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
    };
    use crab_world::bot::sensor::BODY_POS_SLOT;

    use super::*;

    /// The GCR client's stack minus the sim: OpenField arena, one bridged crab. The
    /// explicit rest-pose policy drives nothing, so the crab just stands — drift is
    /// injected by hand.
    fn gcr_like_app() -> App {
        pin_single_thread_pools();
        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            arena: crab_world::physics::Arena::OpenField,
        });
        app.add_plugins(ExternalCrabPlugin::new(
            vec![Policy::rest()],
            vec![Pos::from_meters(0.0, 0.0)],
        ));
        arm(app.world_mut());
        force_serial_schedules(&mut app);
        // Past bridge settle (RESET_GRACE_TICKS) so the world-pos integrator is live.
        for _ in 0..64 {
            app.update();
        }
        app
    }

    /// Emulate a long chase's walk: move the whole crab without touching the bridge.
    fn shift_parts(app: &mut App, delta: Vec3) {
        let mut q = app
            .world_mut()
            .query_filtered::<&mut Transform, With<CrabBodyPart>>();
        for mut t in q.iter_mut(app.world_mut()) {
            t.translation += delta;
        }
    }

    fn carapace_xz(app: &mut App) -> Vec2 {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        let t = q.single(app.world()).expect("carapace").translation;
        Vec2::new(t.x, t.z)
    }

    #[test]
    fn recenter_bounds_local_drift_and_keeps_world_pos_honest() {
        let mut app = gcr_like_app();
        app.insert_resource(BodyPosRecenter);
        let w0 = app.world().resource::<ExternalCrabBridge>().crabs[0].world_pos_m;

        shift_parts(&mut app, Vec3::new(20.0, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        let crab = &app.world().resource::<ExternalCrabBridge>().crabs[0];
        assert_eq!(crab.recenters, 1, "a 20 m drift must trigger one recenter");
        // The walk folds into world_pos_m exactly once; the teleport back never counts.
        let walked = crab.world_pos_m - w0;
        assert!(
            (walked - Vec2::new(20.0, 0.0)).length() < 0.5,
            "world_pos_m must gain the walk and nothing else, gained {walked:?}"
        );
        assert!(
            carapace_xz(&mut app).length() < 1.0,
            "carapace must be back on its spawn origin"
        );
        let obs = app.world().resource::<crab_world::bot::sensor::CrabObservation>();
        let pos_xz = Vec2::new(obs.envs[0][BODY_POS_SLOT], obs.envs[0][BODY_POS_SLOT + 2]);
        assert!(
            pos_xz.length() < 1.0,
            "body.pos obs channel must be back in distribution, got {pos_xz:?}"
        );
    }

    #[test]
    fn gated_off_only_measures_and_never_teleports() {
        let mut app = gcr_like_app();

        shift_parts(&mut app, Vec3::new(20.0, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        let crab = &app.world().resource::<ExternalCrabBridge>().crabs[0];
        assert_eq!(crab.recenters, 0, "unarmed: never teleport");
        assert!(
            crab.next_drift_log_m > TARGET_ARENA_HALF + DRIFT_LOG_STEP_M,
            "the drift crossing must advance the log cursor, got {}",
            crab.next_drift_log_m
        );
        assert!(
            carapace_xz(&mut app).x > 15.0,
            "unarmed: the crab stays where it walked"
        );
    }
}
