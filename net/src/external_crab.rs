//! Drives NN "Sally" crabs in GCR: each crab runs its trained policy in a local
//! physics arena; the bridge poses its hunt target from the sim and integrates
//! carapace deltas back into world position.
//!
//! ONE scale since rl#256: the world runs at the rig's own meters, so the local
//! arena and the sim differ only by a per-round TRANSLATION (the spawn-anchored
//! [`ArenaAnchor`]) — deltas cross freely, absolute positions add the anchor. The
//! deleted 35× scale seam is the bug class rl#253/rl#254 came from: an un-scaled
//! carapace integration once made her sim position creep at 1/35 of her trained
//! stride while her true-size render treadmilled in place.
//!
//! The posed hunt target is the prey offset, re-posed each tick as she closes,
//! clamped to the training band's far edge ([`TARGET_ARENA_HALF`], ~9 m) along the
//! true crab→player bearing. On the current map every prey offset lands well inside
//! the band (~5.3 m at round start — the spawn clearance), in the close-range
//! regime the rl#252 probe measures and the rl#250 curriculum trains — the clamp
//! is a dormant guard that keeps a future larger map in-distribution.
//!
//! The spawn-relative body.pos obs channel would drift OOD on a long open-field
//! chase (rl#240) — not fixable at the posing layer. [`bound_body_pos_drift`]
//! measures the drift every tick and bounds it ([`ARM_BODY_POS_RECENTER`]): recenter
//! the local arena by teleporting the drifted env — crab parts, every co-arena craft,
//! pending boardings — back onto its spawn origin in one uniform shift, advancing the
//! arena↔world anchor in lockstep so nothing observable moves (to f32 rounding).

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::sim::Pos;
use crab_world::Visuals;
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId};
use crab_world::vehicle::{Vehicle, VehicleControls};
use crab_world::bot::sensor::CrabTargets;
use crab_world::bot::{BotSet, CrabSpawns};
use crab_world::crab_view::CrabBrainLabels;
use crab_world::policy::Policy;
use crab_world::training::targets::TARGET_ARENA_HALF;

const CLAW_TARGET_Y: f32 = 0.3;

/// rl#240 flip: recenter the local arena (teleport the drifted crab back onto its spawn
/// origin) whenever the spawn-relative body.pos obs channel leaves the training band.
/// ARMED 2026-07-16: the sequencing gates cleared — the rl#239 honest-bearing eval is
/// deployed and the σ-floor experiment (rl#182) concluded. The recenter is a uniform
/// arena-content teleport: the env's parts, every craft, every pending boarding pose,
/// and the posed hunt target shift by one delta while [`ArenaAnchor`]'s world end
/// advances by the opposite one — world = arena + anchor is invariant, so rendered
/// poses, the pilot-follow feed, world_pos_m, and imminent contacts (rl#235 rams) all
/// survive the teleport to f32 rounding (interpolation windows see a >5 m jump, reset,
/// and hold the arrival pose); only body.pos snaps back to the spawn distribution.
/// SINGLE-CRAB rounds only: per-env deltas conflict on the one shared craft set and the
/// one crab-0-pinned anchor, so multi-crab rounds stay measure-only until crafts anchor
/// per-env (see rl#240). Flip back = set `false`.
const ARM_BODY_POS_RECENTER: bool = true;

/// Arms [`bound_body_pos_drift`]'s recenter teleport. Inserted by the plugin iff
/// [`ARM_BODY_POS_RECENTER`] and the round is single-crab — the build-time decision is
/// the sole authority, so an armed-but-restricted state is unrepresentable at runtime;
/// private so nothing else can arm it (tests live in this module).
#[derive(Resource)]
struct BodyPosRecenter;

/// Each further [`DRIFT_LOG_STEP_M`] of unrecentered peak drift earns one more log line.
const DRIFT_LOG_STEP_M: f32 = 5.0;

/// Pursuit-heartbeat cadence: one range line per this many hunt-fed ticks (~5 s at the
/// 60 Hz fixed step).
const HUNT_LOG_EVERY_TICKS: u64 = 300;

pub struct CrabPolicies(pub Vec<Policy>);

#[derive(Resource)]
pub struct ExternalCrabBridge {
    crabs: Vec<CrabBridge>,
    /// The game-world point corresponding to crab 0's arena spawn origin — the world end
    /// of the ONE arena↔world correspondence [`publish_arena_anchor`] anchors on
    /// (rl#224); on the bridge, not per-crab, because there IS exactly one anchor.
    /// Spawn-pinned per round, then advanced in lockstep by each rl#240 recenter —
    /// static BETWEEN recenters, never tracking her walk.
    anchor_world_m: Vec2,
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
    /// Recenter teleports this round (rl#240).
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
    /// Ticks with a hunt target fed, driving the pursuit heartbeat in
    /// [`ExternalCrabBridge::set_hunt_target`].
    hunt_log_ticks: u64,
    /// Her claw pincers' REAL physics capsules as of the last physics tick (rl#249), in
    /// arena meters: XZ relative to the same-tick carapace ground point (so the offsets
    /// survive arena↔world drift and recenter teleports), y absolute (both grounds are
    /// y = 0).
    claws: Vec<ArenaClaw>,
}

/// One captured claw capsule — see [`CrabBridge::claws`] for the coordinate convention.
struct ArenaClaw {
    a: Vec3,
    b: Vec3,
    radius: f32,
}

fn pos_to_m(p: Pos) -> Vec2 {
    let (x, z) = p.to_meters();
    Vec2::new(x, z)
}

struct CrabPlacement {
    /// The crab's arena carapace ground point (XZ, arena frame).
    carapace_m: Vec2,
    /// The crab's game-world ground point (XZ, world frame).
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
            hunt_log_ticks: 0,
            claws: Vec::new(),
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

    /// Her captured claw capsules in sim space (rl#249): carapace-relative offsets about
    /// her sim ground point (XZ) and the shared ground plane (y) — one scale, rl#256.
    fn claw_poses(&self) -> Vec<crate::sim::ClawPose> {
        let xz = |v: Vec3| {
            crate::sim::Pos::from_meters(self.world_pos_m.x + v.x, self.world_pos_m.y + v.z)
        };
        let y = |v: Vec3| crate::sim::meters_to_grid(v.y);
        self.claws
            .iter()
            .map(|c| crate::sim::ClawPose {
                a: xz(c.a),
                b: xz(c.b),
                a_y: y(c.a),
                b_y: y(c.b),
                radius: crate::sim::meters_to_grid(c.radius),
            })
            .collect()
    }

    fn render_placement_m(&self) -> Option<CrabPlacement> {
        self.last_carapace_m.map(|c| CrabPlacement {
            carapace_m: c,
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
        self.hunt_log_ticks = 0;
        self.claws.clear();
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

    fn note_carapace_found(&mut self) {
        self.missed_carapace_ticks = 0;
        self.next_miss_log_ticks = 1;
    }
}

impl ExternalCrabBridge {
    pub fn new(spawns: &[Pos]) -> Self {
        Self {
            crabs: spawns.iter().map(|&s| CrabBridge::new(s)).collect(),
            anchor_world_m: spawns.first().copied().map(pos_to_m).unwrap_or_default(),
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
                claws: c.claw_poses(),
            })
            .collect()
    }

    pub fn set_hunt_target(&mut self, idx: usize, prey: Option<Pos>) {
        let crab = &mut self.crabs[idx];
        crab.hunt_target_m = prey.map(pos_to_m);
        // Pursuit heartbeat: the crab→prey range (m) every ~5 s. The prey feed is
        // wherever the sim says the player IS — a craft pose while piloting (rl#258) —
        // so a "she stopped giving chase" report (rl#265) is diagnosable from any run's
        // log, live deck telemetry included, without a repro.
        crab.hunt_log_ticks += 1;
        if crab.hunt_log_ticks.is_multiple_of(HUNT_LOG_EVERY_TICKS)
            && let Some(prey_m) = crab.hunt_target_m
        {
            info!(
                "external_crab: env {idx} prey {:.1} m away",
                (prey_m - crab.world_pos_m).length()
            );
        }
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
        if let Some(&spawn) = spawns.first() {
            self.anchor_world_m = pos_to_m(spawn);
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
    let origins = world.resource::<CrabSpawns>().clone();
    world.resource_scope(|world, assets: Mut<crab_world::bot::body::CrabAssets>| {
        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, world);
        for (env, parts) in by_env {
            let origin = origins.origin(env);
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
            if self.crab_spawns.len() == 1 {
                app.insert_resource(BodyPosRecenter);
            } else {
                warn!(
                    "external_crab: {} crabs — rl#240 recenter disarmed for this round \
                     (per-env deltas conflict on the shared craft set and the crab-0 \
                     anchor); body.pos drift is measured only",
                    self.crab_spawns.len()
                );
            }
        }
        // The recenter's boarding carry reads VehicleControls unconditionally; probe and
        // test stacks legitimately run without VehiclePlugin, so guarantee the (empty)
        // resource here rather than making the carry an optional silent skip.
        app.init_resource::<VehicleControls>();
        app.add_systems(
            FixedUpdate,
            (
                // After rescue: both write crab Transforms before Sense; the edge makes the
                // interleaving deterministic (a rescued env respawns at origin, so the guard
                // then sees ~0 drift instead of racing the respawn). After the pose sentinel:
                // the recenter is a sanctioned physics teleport (rl#116) — ordered there, the
                // same tick's SyncBackend consumes it before the sentinel ever sees it.
                // Before the boarding spawn edge: a pending boarding must be carried into
                // the recentered frame BEFORE it is consumed, or the craft materialises a
                // full recenter distance from its walker.
                bound_body_pos_drift
                    .after(crab_world::bot::PoseSentinelSet)
                    .after(crab_world::bot::rescue_nonfinite_crabs)
                    .before(set_crab_walk_target)
                    .before(crab_world::vehicle::VehicleManageSet),
                set_crab_walk_target.before(BotSet::Sense),
                run_crab_policy.in_set(BotSet::Think),
            )
                .run_if(external_crab_armed),
        );
        // Publish each binding's on-screen brain label (rl#200 increment 7). In FixedUpdate
        // deliberately: only the physics-pumping peer (solo/host) advances FixedUpdate
        // (the wall-clock auto-pump is PARKED — to a 86400s timestep, so "never" really
        // means "not within a day's uptime" — and `pump_fixed_steps` is driven by the client-sim tick drain),
        // so this is host-only by construction; on a remote-adopt client the articulation
        // `apply` is the sole label writer and the two can't fight over the resource.
        app.init_resource::<CrabBrainLabels>();
        app.init_resource::<ArenaAnchor>();
        app.add_systems(
            FixedUpdate,
            // The anchor publisher runs after the recenter so a teleport's carried crafts
            // and its advanced anchor always land on the same tick (and hence the same
            // articulation message) — split across ticks they'd pop on screen once.
            (
                publish_brain_labels,
                publish_arena_anchor.after(bound_body_pos_drift),
            )
                .run_if(external_crab_armed),
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
/// Crafts share the arena frame, so the shift carries them (and every pending boarding
/// pose) too, and [`ExternalCrabBridge::anchor_world_m`] advances by the opposite delta
/// — world = arena + anchor stays invariant (see [`ARM_BODY_POS_RECENTER`]).
///
/// Ordering: before [`set_crab_walk_target`] (so the target is posed from the
/// post-teleport carapace) and hence before Sense and rapier's SyncBackend; before
/// the boarding spawn edge (`VehicleManageSet` — a pending boarding is carried before
/// it is consumed); before [`publish_arena_anchor`] (so the carried crafts and the
/// advanced anchor land on the wire in the same tick — a stale anchor would pop every
/// craft by the full delta).
fn bound_body_pos_drift(
    mut bridge: ResMut<ExternalCrabBridge>,
    spawns: Res<CrabSpawns>,
    armed: Option<Res<BodyPosRecenter>>,
    mut targets: ResMut<CrabTargets>,
    mut parts: Query<(&CrabEnvId, &mut Transform, Option<&CrabCarapace>), With<CrabBodyPart>>,
    mut crafts: Query<&mut Transform, (With<Vehicle>, Without<CrabBodyPart>)>,
    mut controls: ResMut<VehicleControls>,
) {
    let ExternalCrabBridge {
        crabs,
        anchor_world_m,
    } = bridge.as_mut();
    for (idx, crab) in crabs.iter_mut().enumerate() {
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
        let origin = spawns.origin(idx);
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
            // Crafts live in the same arena frame (rl#235 rams are real contacts;
            // boardings pose in arena m): carry every body and every pending boarding by
            // the same delta, and advance the anchor's world end by the opposite one —
            // rendered poses, the pilot-follow feed, and imminent contacts all cancel
            // out. Boardings too because a command authored pre-recenter is consumed at
            // the spawn edge AFTER this system (the VehicleManageSet edge); uncarried,
            // that craft would materialise a full recenter distance from its walker.
            for mut t in crafts.iter_mut() {
                t.translation += delta;
            }
            for cmd in controls.0.values_mut() {
                cmd.boarding.pos += delta;
            }
            *anchor_world_m -= Vec2::new(delta.x, delta.z);
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
                 disarmed)"
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
        // Pre-spawn the slots don't exist (settle grace); post-settle a miss is the
        // slot-desync class `run_crab_policy` panics on the same tick — skip THIS crab
        // only, never abort the whole loop.
        let Some(slot) = targets.envs.get_mut(idx) else {
            continue;
        };
        let Some(hunt) = crab.hunt_target_m else {
            *slot = None;
            continue;
        };
        // Prey offset, then the in-distribution guard (module header): pose the target
        // at most one band-edge away along the true bearing. Current-map offsets sit
        // well inside the band, so the clamp is dormant.
        let to_prey = hunt - crab.world_pos_m;
        if to_prey.length_squared() < 1e-6 {
            continue;
        }
        let to_prey = to_prey.clamp_length_max(TARGET_ARENA_HALF);

        let origin = spawns.origin(idx);
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
            let _ = actions.rest(idx); // deliberate skip pre-spawn (see comment above)
            continue;
        }
        // Post-settle, a missing slot means this crab would hold rest pose in a live
        // round — a wiring bug, never a condition to skip past silently (rl#241).
        let landed = obs
            .rows()
            .get(idx)
            .is_some_and(|o| actions.set_row(idx, policy.act(o)));
        assert!(
            landed,
            "external_crab: crab {idx} has no env slot ({} obs / {} action slots sized \
             for {} crabs) — it would silently hold rest pose in a live round (rl#241)",
            obs.rows().len(),
            actions.len(),
            crab_count,
        );
    }
}

fn integrate_crab(
    mut bridge: ResMut<ExternalCrabBridge>,
    mut rescued: MessageReader<crab_world::bot::CrabRescued>,
    carapace_q: Query<(&CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
    claw_q: Query<(&CrabEnvId, &Transform, &Collider), With<CrabClawTip>>,
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
        crab.note_carapace_found();
        if !t.translation.is_finite() {
            continue; // the rescue path owns non-finite crabs (a Fault when armed)
        }
        let here = Vec2::new(t.translation.x, t.translation.z);
        if crab.settle == 0
            && let Some(prev) = crab.last_carapace_m
        {
            // One scale (rl#256): her trained stride IS her world stride.
            crab.world_pos_m += here - prev;
        }
        crab.last_carapace_m = Some(here);

        let v = Vec2::new(vel.linear.x, vel.linear.z);
        if v.length_squared() > 1e-4 {
            let radians = v.x.atan2(v.y);
            crab.yaw_turns = crate::sim::trig_client::radians_to_turns(radians);
        }

        // Capture her claw pincers' REAL colliders against the same-tick carapace point
        // (rl#249) — the sim decides claw-touch downs against these, so there is no
        // second hitbox to drift from the physics claw.
        crab.claws = claw_q
            .iter()
            .filter(|(env, ..)| env.0 == idx)
            .filter_map(|(_, t, col)| {
                let cap = col.as_capsule()?;
                let rel = |p: Vec3| {
                    let w = t.transform_point(p);
                    Vec3::new(w.x - here.x, w.y, w.z - here.y)
                };
                Some(ArenaClaw {
                    a: rel(cap.segment().a()),
                    b: rel(cap.segment().b()),
                    radius: cap.radius(),
                })
            })
            .collect();
    }
}

/// Where the shared physics arena sits in the render world — a per-round translate pinned
/// by crab 0's spawn correspondence (game spawn ↔ arena spawn origin), static between
/// rl#240 recenters (each recenter advances it in lockstep with the carried arena
/// content, so nothing rendered through it moves).
/// Vehicles and the cockpit camera render through THIS, never through the per-crab skin
/// repose: the repose re-tracks the live carapace every tick to pin Sally's skin to her sim
/// spot, so borrowing it as the arena transform dragged ~(1−rs) of her every movement into
/// every craft's rendered pose — the ship visibly danced whenever she wiggled (rl#224).
/// Since rl#254 the skin and arena frames advance identically as she walks (the trade the
/// pre-fix (1−rs)-per-step drift posed is gone); a pilot's aim at her SKIN differs from her
/// collider only by her settle-window motion, a small per-round constant. For crabs beyond 0 the
/// skin-vs-collider offset is nonzero from tick 0 (inter-crab spawn deltas render ·rs via the
/// sim but ·1 in the arena) — pre-existing with the old borrowed-repose anchor too.
///
/// Host-authored ([`publish_arena_anchor`], FixedUpdate ⇒ host-only like the brain labels:
/// only the physics-pumping ServerAuth peer advances FixedUpdate — if client-side FixedUpdate
/// pumping ever lands, this publisher would fight `apply`'s adopted value and must gain a
/// role gate) and shipped on the articulation wire; a client adopts it verbatim in `apply`.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct ArenaAnchor(pub Vec3);

fn publish_arena_anchor(
    bridge: Res<ExternalCrabBridge>,
    spawns: Res<CrabSpawns>,
    mut out: ResMut<ArenaAnchor>,
) {
    if bridge.crabs.is_empty() {
        return;
    }
    if spawns.is_empty() {
        return; // pre-spawn frame — `spawn_initial_crabs` hasn't rebuilt the origins yet
    }
    let origin = spawns.origin(0);
    let w = bridge.anchor_world_m;
    let want = ArenaAnchor(Vec3::new(w.x - origin.x, 0.0, w.y - origin.z));
    if *out != want {
        *out = want;
    }
}

fn publish_skin_repose(
    bridge: Res<ExternalCrabBridge>,
    repose_out: Option<ResMut<crab_world::bot::skin::CrabSkinRepose>>,
) {
    let Some(mut out) = repose_out else {
        return;
    };
    out.0 = bridge
        .crabs
        .iter()
        .enumerate()
        .filter_map(|(idx, crab)| {
            crab.render_placement_m().map(|r| {
                let s = r.game_spot - r.carapace_m;
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

/// rl#224 gates: a wiggling (even violently flailing) Sally must not move a ship she isn't
/// touching — neither its arena body (physics) nor its RENDERED pose (arena pose + the
/// [`ArenaAnchor`] anchor, which is static by construction). Before the fix the anchor
/// tracked her live carapace, so her 9.6 m flail-walk dragged the parked ship's rendered
/// pose 9.3 m; and the boarding spawn at 0.5 m altitude materialised the craft inside her
/// body, so contact batted it ~8 m.
#[cfg(test)]
mod ship_wiggle_tests {
    use crab_world::bot::actuator::CrabActions;
    use crab_world::bot::headless::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
    };
    use crab_world::vehicle::{
        PilotCommand, PilotId, Vehicle, VehicleControls, VehicleKind, VehiclePlugin,
    };

    use super::*;

    #[derive(Resource, Default)]
    struct Wiggle(f32);

    /// Overwrite the (unloaded ⇒ zero-action) policy's output with a full-scale square wave
    /// — a flail far more violent than any idle wiggle, so the gates bound the worst case.
    fn drive_wiggle(w: Res<Wiggle>, mut actions: ResMut<CrabActions>) {
        // Every-tick system: pre-spawn skips are fine, later ticks land the flail.
        let _ = actions.fill(0, w.0);
    }

    fn gcr_like_app_with_vehicles() -> App {
        pin_single_thread_pools();
        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            arena: crab_world::physics::Arena::OpenField,
            visuals: crab_world::Visuals(false),
        });
        app.add_plugins(VehiclePlugin);
        app.add_plugins(ExternalCrabPlugin::new(
            vec![Policy::rest()],
            // A nonzero GAME spawn (the arena spawn stays at the grid origin): the arena
            // anchor is then nonzero, so the static-anchor assertions can't pass vacuously.
            vec![Pos::from_meters(30.0, -14.0)],
        ));
        app.init_resource::<Wiggle>();
        app.add_systems(
            FixedUpdate,
            drive_wiggle
                .after(run_crab_policy)
                .in_set(crab_world::bot::BotSet::Think),
        );
        arm(app.world_mut());
        force_serial_schedules(&mut app);
        app
    }

    fn ship_pos(app: &mut App) -> Vec3 {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<Vehicle>>();
        q.single(app.world()).expect("one ship").translation
    }

    /// A boarding pose at `(x, z)`, at rest — clear of the origin-standing Sally, like a
    /// real walker's spot would be (rl#258: crafts materialise where the player is).
    fn boarding_at(x: f32, z: f32) -> crab_world::vehicle::Boarding {
        crab_world::vehicle::Boarding {
            pos: Vec3::new(x, 0.0, z),
            yaw: 0.0,
            velocity: Vec3::ZERO,
        }
    }

    fn flail(app: &mut App, ticks: u32) {
        for t in 0..ticks {
            app.world_mut().resource_mut::<Wiggle>().0 = if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
            app.update();
        }
    }

    /// The ship's on-screen point — arena pose + anchor, the frame every craft renders
    /// through. The recenter-armed invariant is about THIS sum: a recenter may shift
    /// both ends, never what the player sees.
    fn ship_render_pos(app: &mut App) -> Vec3 {
        let anchor = app.world().resource::<ArenaAnchor>().0;
        ship_pos(app) + anchor
    }

    fn recenters(app: &App) -> u32 {
        app.world().resource::<ExternalCrabBridge>().crabs[0].recenters
    }

    #[test]
    fn parked_ship_stays_put_on_screen_while_sally_flails() {
        let mut app = gcr_like_app_with_vehicles();
        crab_world::vehicle::spawn_ram_vehicle(
            app.world_mut(),
            VehicleKind::Ship,
            Transform::from_xyz(5.0, 0.5, 5.0),
            bevy_rapier3d::prelude::Velocity::default(),
        );
        app.world_mut().resource_mut::<VehicleControls>().0.insert(
            PilotId(0),
            PilotCommand::new(VehicleKind::Ship, boarding_at(5.0, 5.0)),
        );
        for _ in 0..64 {
            app.update();
        }
        let render0 = ship_render_pos(&mut app);
        let mut prev_anchor = *app.world().resource::<ArenaAnchor>();
        assert_ne!(prev_anchor.0, Vec3::ZERO, "the armed round published an anchor");
        let mut prev_recenters = recenters(&app);

        let mut max_ship_d = 0.0f32;
        for t in 0..400u32 {
            app.world_mut().resource_mut::<Wiggle>().0 = if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
            app.update();
            max_ship_d = max_ship_d.max(ship_render_pos(&mut app).distance(render0));
            // The anchor moves ONLY on a recenter epoch — an anchor tracking her
            // per-tick wiggle is exactly the rendered-ship-follows-Sally bug (rl#224).
            let anchor = *app.world().resource::<ArenaAnchor>();
            let recs = recenters(&app);
            if anchor != prev_anchor {
                assert!(
                    recs > prev_recenters,
                    "the arena anchor moved without a recenter — the \
                     rendered-ship-follows-Sally bug"
                );
            }
            prev_anchor = anchor;
            prev_recenters = recs;
        }
        assert!(
            recenters(&app) >= 1,
            "the 400-tick flail-walk (9.6 m pre-fix) must cross the band and recenter — \
             if physics changed enough that it no longer does, the epoch assertions \
             above went vacuous; re-pin this test"
        );
        assert!(
            max_ship_d < 1e-3,
            "an untouched parked ship must not move ON SCREEN while Sally flails \
             (recenter teleports must cancel against the anchor), moved {max_ship_d} m"
        );
    }

    #[test]
    fn boarded_ship_spawns_clear_of_a_flailing_sally() {
        let mut app = gcr_like_app_with_vehicles();
        app.world_mut().resource_mut::<VehicleControls>().0.insert(
            PilotId(0),
            PilotCommand::new(VehicleKind::Ship, boarding_at(5.0, 5.0)),
        );
        for _ in 0..64 {
            app.update();
        }
        let render0 = ship_render_pos(&mut app);
        flail(&mut app, 200);
        let moved = ship_render_pos(&mut app).distance(render0);
        assert!(
            moved < 0.05,
            "a freshly boarded craft must materialise clear of the crab's body — \
             contact shoved it {moved} m"
        );
    }

    /// rl#240 flip pins: a recenter is a uniform arena-content teleport — the parked
    /// craft's ARENA pose and the anchor both move by the (opposite) recenter delta, its
    /// RENDERED pose doesn't move at all, and a boarding authored before the teleport
    /// still materialises its craft at the walker's world spot (the pending-command
    /// carry; uncarried it would spawn a full recenter distance away).
    #[test]
    fn recenter_carries_crafts_and_boardings_and_advances_the_anchor() {
        use crab_world::bot::body::CrabBodyPart;

        let mut app = gcr_like_app_with_vehicles();
        crab_world::vehicle::spawn_ram_vehicle(
            app.world_mut(),
            VehicleKind::Ship,
            Transform::from_xyz(5.0, 0.5, 5.0),
            bevy_rapier3d::prelude::Velocity::default(),
        );
        app.world_mut().resource_mut::<VehicleControls>().0.insert(
            PilotId(0),
            PilotCommand::new(VehicleKind::Ship, boarding_at(5.0, 5.0)),
        );
        for _ in 0..64 {
            app.update();
        }
        let ship0 = ship_pos(&mut app);
        let anchor0 = app.world().resource::<ArenaAnchor>().0;
        let render0 = ship0 + anchor0;

        // A second pilot boards right as the drift crosses the band: its command's
        // boarding pose must ride the teleport into the new frame.
        let walker_world = Vec3::new(7.0, 0.0, -3.0) + anchor0;
        app.world_mut().resource_mut::<VehicleControls>().0.insert(
            PilotId(1),
            PilotCommand::new(VehicleKind::Ship, boarding_at(7.0, -3.0)),
        );
        // The command is still PENDING here (its craft spawns on the next tick — the
        // same tick the recenter fires), so this pins the boarding-carry path, not the
        // entity carry.
        let crafts_now = {
            let mut q = app.world_mut().query_filtered::<(), With<Vehicle>>();
            q.iter(app.world()).count()
        };
        assert_eq!(crafts_now, 1, "pilot 1's craft must not exist pre-recenter");

        // Emulate a long chase: walk the whole crab 20 m out in one step.
        let mut q = app
            .world_mut()
            .query_filtered::<&mut Transform, With<CrabBodyPart>>();
        for mut t in q.iter_mut(app.world_mut()) {
            t.translation += Vec3::new(20.0, 0.0, 0.0);
        }
        for _ in 0..2 {
            app.update();
        }

        assert_eq!(recenters(&app), 1, "a 20 m drift must trigger one recenter");
        let anchor1 = app.world().resource::<ArenaAnchor>().0;
        let advanced = anchor1 - anchor0;
        assert!(
            (advanced - Vec3::new(20.0, 0.0, 0.0)).length() < 0.5,
            "the anchor's world end must advance by the recenter delta, got {advanced:?}"
        );
        let mut ships: Vec<(PilotId, Vec3)> = {
            let mut q = app.world_mut().query::<(&Vehicle, &Transform)>();
            q.iter(app.world())
                .map(|(v, t)| (v.pilot, t.translation))
                .collect()
        };
        ships.sort_by_key(|(p, _)| p.0);
        let [(_, parked), (_, boarded)] = ships[..] else {
            panic!("both pilots' crafts exist, got {}", ships.len());
        };
        assert!(
            (parked - (ship0 + Vec3::new(-20.0, 0.0, 0.0))).length() < 0.5,
            "the parked craft's arena pose must be carried by the recenter delta"
        );
        assert!(
            (parked + anchor1 - render0).length() < 0.05,
            "the parked craft must not move ON SCREEN through a recenter, moved {:?}",
            parked + anchor1 - render0
        );
        let boarded_render = boarded + anchor1;
        let miss = Vec2::new(
            boarded_render.x - walker_world.x,
            boarded_render.z - walker_world.z,
        )
        .length();
        assert!(
            miss < 1.0,
            "a boarding authored pre-recenter must still materialise at its walker's \
             world spot, landed {miss} m off"
        );
    }
}

#[cfg(test)]
mod gcr_crab_tests {
    use crab_world::bot::body::CrabBodyPart;
    use crab_world::bot::headless::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
    };

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
            visuals: crab_world::Visuals(false),
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
        // The walk folds into world_pos_m exactly once, 1:1 (rl#256); the teleport
        // back never counts.
        let walked = crab.world_pos_m - w0;
        let want = Vec2::new(20.0, 0.0);
        assert!(
            (walked - want).length() < 0.5,
            "world_pos_m must gain the walk and nothing else, \
             gained {walked:?}, want {want:?}"
        );
        assert!(
            carapace_xz(&mut app).length() < 1.0,
            "carapace must be back on its spawn origin"
        );
        let obs = app
            .world()
            .resource::<crab_world::bot::sensor::CrabObservation>();
        let body_pos = obs.env(0).expect("env 0 sized").body_pos();
        let pos_xz = Vec2::new(body_pos.x, body_pos.z);
        assert!(
            pos_xz.length() < 1.0,
            "body.pos obs channel must be back in distribution, got {pos_xz:?}"
        );
    }

    #[test]
    fn unarmed_only_measures_and_never_teleports() {
        let mut app = gcr_like_app();
        // The flip default arms recenter via the plugin — disarm to pin the
        // measure-only path (the flip-back configuration).
        app.world_mut().remove_resource::<BodyPosRecenter>();

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

    /// rl#240's multi-crab restriction: per-env recenter deltas conflict on the shared
    /// craft set and the crab-0 anchor, so a multi-crab round measures but never
    /// teleports — even armed.
    #[test]
    fn multi_crab_rounds_measure_but_never_teleport() {
        pin_single_thread_pools();
        let mut app = headless_stack(HeadlessStack {
            num_envs: 2,
            role: WorldRole::Standalone,
            arena: crab_world::physics::Arena::OpenField,
            visuals: crab_world::Visuals(false),
        });
        app.add_plugins(ExternalCrabPlugin::new(
            vec![Policy::rest(), Policy::rest()],
            vec![Pos::from_meters(0.0, 0.0), Pos::from_meters(8.0, 0.0)],
        ));
        arm(app.world_mut());
        force_serial_schedules(&mut app);
        for _ in 0..64 {
            app.update();
        }

        // Walk only env 0's crab out of the band.
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &mut Transform), With<CrabBodyPart>>();
        for (env, mut t) in q.iter_mut(app.world_mut()) {
            if env.0 == 0 {
                t.translation += Vec3::new(20.0, 0.0, 0.0);
            }
        }
        for _ in 0..2 {
            app.update();
        }

        let crab = &app.world().resource::<ExternalCrabBridge>().crabs[0];
        assert_eq!(crab.recenters, 0, "multi-crab: never teleport");
        assert!(
            crab.next_drift_log_m > TARGET_ARENA_HALF + DRIFT_LOG_STEP_M,
            "multi-crab: the drift is still measured, cursor {}",
            crab.next_drift_log_m
        );
        let origin0 = app.world().resource::<CrabSpawns>().origin(0);
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>();
        let carapace0 = q
            .iter(app.world())
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| t.translation)
            .expect("env 0 carapace");
        assert!(
            carapace0.x - origin0.x > 15.0,
            "multi-crab: the crab stays where it walked"
        );
    }

    /// rl#254 pin, rl#256 form: carapace deltas integrate into the world position 1:1
    /// — there is no scale left to miss. (Pre-rl#256 a missed 35× conversion crept the
    /// sim position at 1/35 of her stride; one frame makes that bug unrepresentable.)
    #[test]
    fn world_pos_integrates_carapace_deltas_one_to_one() {
        let mut app = gcr_like_app();
        let w0 = app.world().resource::<ExternalCrabBridge>().crabs[0].world_pos_m;

        shift_parts(&mut app, Vec3::new(0.5, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        let walked = app.world().resource::<ExternalCrabBridge>().crabs[0].world_pos_m - w0;
        let want = Vec2::new(0.5, 0.0);
        assert!(
            (walked - want).length() < 0.05,
            "a 0.5 m walk must integrate as {want:?}, got {walked:?}"
        );
    }

    /// rl#254 pin, render half: the skin repose shift is CONSTANT while she walks —
    /// `d(game_spot) = d(carapace)` exactly (one frame, rl#256). Under the un-scaled integration the
    /// shift grew every step, so her true-size skin treadmilled in place while her
    /// legs strode (the owner's "ground wiggles with her movements").
    #[test]
    fn skin_repose_shift_is_constant_while_walking() {
        use bevy::ecs::system::RunSystemOnce;
        // Visuals(false) skips the publisher's registration — drive it directly.
        let mut app = gcr_like_app();
        app.init_resource::<crab_world::bot::skin::CrabSkinRepose>();
        let read_shift = |app: &mut App| {
            app.world_mut()
                .run_system_once(publish_skin_repose)
                .expect("publish_skin_repose runs");
            app.world()
                .resource::<crab_world::bot::skin::CrabSkinRepose>()
                .0[&0]
                .shift
        };
        let shift0 = read_shift(&mut app);

        shift_parts(&mut app, Vec3::new(0.5, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        let shift = read_shift(&mut app);
        assert!(
            (shift - shift0).length() < 0.02,
            "the repose shift must not track her walk (got {shift0:?} → {shift:?}) — \
             a growing shift is the rl#254 treadmill"
        );
    }

    /// rl#254 pin, rl#256 form: the posed walk target is the prey offset, 1:1, clamped
    /// to the training band's far edge. (Pre-rl#256 a missed 35× conversion posed
    /// in-band prey ~35× too far and the clamp turned every chase into a fixed
    /// 9-m carrot.)
    #[test]
    fn walk_target_poses_prey_offset_clamped_to_band() {
        let mut app = gcr_like_app();
        let cara = carapace_xz(&mut app);

        // In-band prey, 5 m from her origin spawn — must pose unclamped.
        let hunt = Pos::from_meters(5.0, 0.0);
        app.world_mut()
            .resource_mut::<ExternalCrabBridge>()
            .set_hunt_target(0, Some(hunt));
        app.update();
        let posed = app.world().resource::<CrabTargets>().envs[0].expect("target posed");
        let want = cara + Vec2::new(5.0, 0.0);
        let got = Vec2::new(posed.x, posed.z);
        assert!(
            (got - want).length() < 0.05,
            "in-band prey must pose at the converted offset {want:?}, got {got:?}"
        );

        // Beyond-band prey (only reachable on a future larger map): clamps to the edge.
        let hunt = Pos::from_meters(50.0, 0.0);
        app.world_mut()
            .resource_mut::<ExternalCrabBridge>()
            .set_hunt_target(0, Some(hunt));
        app.update();
        let posed = app.world().resource::<CrabTargets>().envs[0].expect("target posed");
        let d = (Vec2::new(posed.x, posed.z) - carapace_xz(&mut app)).length();
        assert!(
            (d - TARGET_ARENA_HALF).abs() < 0.05,
            "beyond-band prey must clamp to the {TARGET_ARENA_HALF} m edge, posed {d}"
        );
    }

    /// rl#225 pin, component dimension: a body part not matched by the cage pass's query
    /// silently vanishes from the collider render modes. Count coverage with the pass's OWN
    /// query aliases ([`crab_world::crab_view::CrabCagePartData`]/`CrabCagePartFilter` — one
    /// source, so this pin can't drift from the system) through both paths that create GCR
    /// crab bodies: the initial armed spawn and the round-boundary cold respawn (the rescue
    /// path funnels into the same `respawn_crab`). The other invisibility dimension — a
    /// collider SHAPE the drawer can't trace — is loud by construction instead
    /// (`error_once` in `draw_collider_wireframe`).
    #[test]
    fn every_body_part_is_visible_to_the_collider_wireframe_query() {
        use crab_world::crab_view::{CrabCagePartData, CrabCagePartFilter};

        fn assert_cage_covers_all(app: &mut App, ctx: &str) {
            let all = app
                .world_mut()
                .query_filtered::<Entity, With<CrabBodyPart>>()
                .iter(app.world())
                .count();
            let caged = app
                .world_mut()
                .query_filtered::<CrabCagePartData, CrabCagePartFilter>()
                .iter(app.world())
                .count();
            assert!(all > 0, "{ctx}: no crab body parts spawned");
            assert_eq!(
                caged,
                all,
                "{ctx}: {} of {all} body parts are invisible to the collider wireframe \
                 query (rl#225)",
                all - caged
            );
        }

        let mut app = gcr_like_app();
        assert_cage_covers_all(&mut app, "initial armed spawn");

        cold_respawn_armed_crab(app.world_mut());
        assert_cage_covers_all(&mut app, "after a round-boundary cold respawn");
    }
}
