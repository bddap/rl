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
//! measures the drift every tick and bounds it ([`ARM_BODY_POS_RECENTER`]): rebase
//! the drifted env's spawn origin to the crab's own ground point, advancing
//! [`ExternalCrabBridge::anchor_world_m`] in lockstep so the published
//! [`ArenaAnchor`] never moves. No Transform is touched, so the terrain stays glued
//! under every foot and craft (rl#281 stage 6).

use bevy::prelude::*;
use bevy_rapier3d::geometry::ColliderView;
use bevy_rapier3d::prelude::*;

use crate::sim::Pos;
use crab_world::Visuals;
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId};
use crab_world::bot::sensor::CrabTargets;
use crab_world::bot::{BotSet, CrabSpawns};
use crab_world::crab_view::CrabBrainLabels;
use crab_world::policy::Policy;
use crab_world::training::targets::{TARGET_ARENA_HALF, band_lure, recenter_delta};
use crab_world::vehicle::{Vehicle, VehicleControls};

/// The posed hunt target's height ABOVE THE LOCAL SURFACE at the target's own xz —
/// training's convention (`sample_target` bands height-above-surface), so `target:y`
/// stays in-distribution on terrain; flat grids reduce to the old absolute 0.3.
const CLAW_TARGET_Y: f32 = 0.3;

/// rl#240 flip: recenter the spawn-relative body.pos obs channel whenever a long chase
/// walks it out of the training band. ARMED 2026-07-16 (the rl#239 honest-bearing eval
/// deployed, the rl#182 σ-floor experiment concluded). Since rl#281 stage 6 the
/// recenter REBASES the env's spawn origin to the crab
/// ([`CrabSpawns::rebase_origin_to`]) instead of teleporting the crab to the origin:
/// body.pos snaps back to the spawn distribution identically, but no Transform moves —
/// a teleport would swap the physics terrain locale beneath the crab's feet while the
/// rendered mountain stayed put, exactly the render≠physics seam rl#281 forbids. With
/// nothing physical to carry, the old single-crab restriction (per-env teleport deltas
/// conflicting on the shared craft set) is gone: origins are per-env state, so every
/// round recenters. Flip back = set `false` (measure-only).
const ARM_BODY_POS_RECENTER: bool = true;

/// Arms [`bound_body_pos_drift`]'s origin rebase. Inserted by the plugin iff
/// [`ARM_BODY_POS_RECENTER`] — the build-time decision is the sole authority; private
/// so nothing else can arm it. With the flip const-true, its runtime absence is now
/// solely the tests' disarm lever (the measure-only pin).
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
    /// Spawn-pinned per round, then advanced by each rl#240 origin rebase in lockstep
    /// with the arena end, so the published [`ArenaAnchor`] difference never moves
    /// within a round — this never tracks her walk, only her recenter epochs.
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
    /// survive arena↔world drift), y the height above the terrain surface under the
    /// claw — the sim's claw check spans the player's own surface-relative height band
    /// ([`crate::sim::ClawPose`]); on the flat grids surface ≡ 0 and this is
    /// bit-identical to absolute y.
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
    /// her sim ground point (XZ) and height above the local surface (y) — one scale,
    /// rl#256.
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

    /// Private: every restart must go through [`restart_bridge_to_spawns`], which also
    /// carries surviving crafts by the anchor-reset delta — a bare bridge reset after a
    /// recenter would teleport them on screen.
    fn restart_to_spawns(&mut self, spawns: &[Pos]) {
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

/// The rl#204 RESTART for the arena↔world correspondence: reset the bridge to the new
/// spawns and carry every surviving craft (and pending boarding pose) by the anchor's
/// world-frame reset delta. Crafts persist through a RESTART (only round teardown
/// despawns them), so an anchor snapping back from its recenter-advanced value — or to
/// a different spawn — must not move a still-flying craft on screen: the same
/// world = arena + anchor invariance the recenter itself keeps ([`ARM_BODY_POS_RECENTER`]).
/// With no recenters and unchanged spawns the delta is zero and this is exactly the old
/// bare `restart_to_spawns`.
pub(crate) fn restart_bridge_to_spawns(world: &mut World, spawns: &[Pos]) {
    let old_w = {
        let mut bridge = world.resource_mut::<ExternalCrabBridge>();
        let old_w = bridge.anchor_world_m;
        bridge.restart_to_spawns(spawns);
        old_w
    };
    // rl#289: the bridge reset re-pins only env 0's arena end (the anchor); every other
    // env's origin still carries last round's rebased wander while its sim end respawns
    // fresh, so env≠0's arena↔sim offset would differ from the shared anchor by the
    // accumulated differential wander — its skin (repose-pinned to the sim spot) would
    // render horizontally off its arena body (the ram target, rendered through the
    // anchor) and off the local ground height (the repose shift is planar, exact only
    // where it equals the y = 0 anchor). Re-pin every origin to the sim spawn LAYOUT
    // about env 0's kept origin, so world = arena + anchor holds per-env at respawn —
    // the caller cold-respawns every crab onto these origins in the same call. Skipped
    // while the origins aren't built yet (fresh-app install: `spawn_initial_crabs` runs
    // after arm and rebuilds the grid layout; that first round's tick-0 offset is the
    // pre-existing [`ArenaAnchor`] note, not restart wander).
    if let Some(&w0) = spawns.first() {
        let terrain = world.resource::<crab_world::terrain::Terrain>().clone();
        let mut origins = world.resource_mut::<CrabSpawns>();
        if !origins.is_empty() {
            let offsets: Vec<Vec2> = spawns.iter().map(|&s| pos_to_m(s) - pos_to_m(w0)).collect();
            origins.repin_layout(&offsets, &terrain);
        }
    }
    let carry = old_w - world.resource::<ExternalCrabBridge>().anchor_world_m;
    if carry != Vec2::ZERO {
        let delta = Vec3::new(carry.x, 0.0, carry.y);
        let terrain = world.resource::<crab_world::terrain::Terrain>().clone();
        let mut q = world.query_filtered::<&mut Transform, With<Vehicle>>();
        for mut t in q.iter_mut(world) {
            // The carry preserves the craft's WORLD pose, which shifts its arena xz
            // onto terrain the craft never measured — floor it clear of the new
            // locale's ground. Zero clearance: a settled craft must not pop upward
            // (flat grids: a no-op to contact slop).
            t.translation =
                crab_world::vehicle::clear_of_ground(t.translation + delta, 0.0, &terrain);
        }
        if let Some(mut controls) = world.get_resource_mut::<VehicleControls>() {
            for cmd in controls.0.values_mut() {
                // Pending boardings get the same floor at their spawn edge
                // (`spawn_vehicle`), so the raw carry suffices here.
                cmd.boarding.pos += delta;
            }
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
    let terrain = world.resource::<crab_world::terrain::Terrain>().clone();
    world.resource_scope(|world, assets: Mut<crab_world::bot::body::CrabAssets>| {
        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, world);
        for (env, parts) in by_env {
            let origin = origins.origin(env);
            crab_world::bot::respawn_crab(
                &mut commands,
                &assets,
                &terrain,
                parts.into_iter(),
                origin,
                env,
            );
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
                // After rescue: a rescued env respawns at its origin, so the guard sees
                // ~0 drift there instead of racing the respawn. Before the walk target
                // (and hence Sense): the posed target and the obs both read a rebased
                // origin the same tick it moves.
                bound_body_pos_drift
                    .after(crab_world::bot::rescue_lost_crabs)
                    .before(set_crab_walk_target),
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
            // The anchor publisher runs after the recenter: a rebase moves BOTH ends of
            // the correspondence (origin and anchor_world_m) in one system, and ordered
            // after it the publisher can never observe the half-updated pair.
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
/// walled box, the open grids don't, so a long chase walks it arbitrarily OOD. Always
/// MEASURES (rate-limited warn lines quantify the drift); when [`BodyPosRecenter`] is
/// armed it also FIXES it — rebase the env's spawn origin to the carapace's own ground
/// point ([`CrabSpawns::rebase_origin_to`]). body.pos snaps to ~0, exactly the
/// every-episode spawn distribution, and NOTHING physical moves: no teleport, no craft
/// or boarding carry, and on terrain the world stays glued to the physics under every
/// foot (see [`ARM_BODY_POS_RECENTER`]). For crab 0 the anchor's world end
/// ([`ExternalCrabBridge::anchor_world_m`]) advances by the same planar delta, so the
/// published [`ArenaAnchor`] difference is unchanged — world = arena + anchor holds
/// with both ends still.
///
/// Trigger and drift band are the shared rl#240 formula
/// ([`crab_world::training::targets::recenter_delta`]), the same one the eval's pace
/// probe recenters by (rl#280) — the trainer/eval side still applies it as a teleport,
/// which is sound there because nothing rendered is watching mid-episode.
///
/// Ordering: before [`set_crab_walk_target`] and hence Sense, so the posed target and
/// the obs read a rebased origin the same tick it moves.
fn bound_body_pos_drift(
    mut bridge: ResMut<ExternalCrabBridge>,
    mut spawns: ResMut<CrabSpawns>,
    terrain: Res<crab_world::terrain::Terrain>,
    armed: Option<Res<BodyPosRecenter>>,
    carapaces: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
) {
    let ExternalCrabBridge {
        crabs,
        anchor_world_m,
    } = bridge.as_mut();
    for (idx, crab) in crabs.iter_mut().enumerate() {
        let Some(carapace) = carapaces
            .iter()
            .find(|(env, _)| env.0 == idx)
            .map(|(_, t)| t.translation)
        else {
            // Same absence [`integrate_crab`] counts and reports this tick (rl#241) —
            // one counter, not two log streams for one missing entity.
            continue;
        };
        if !carapace.is_finite() {
            continue; // the rescue path owns non-finite crabs
        }
        let origin = spawns.origin(idx);
        let Some(delta) = recenter_delta(origin, carapace, &terrain) else {
            continue;
        };
        let drift_m = Vec2::new(delta.x, delta.z).length();

        if armed.is_some() {
            let rebased = spawns.rebase_origin_to(idx, carapace, &terrain);
            if idx == 0 {
                *anchor_world_m += Vec2::new(rebased.x - origin.x, rebased.z - origin.z);
            }
            crab.recenters += 1;
            info!(
                "external_crab: recentered env {idx} — origin rebased {drift_m:.1} m to \
                 her ground point (recenter #{} this round, rl#240)",
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
    terrain: Res<crab_world::terrain::Terrain>,
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
        // at most one band-edge away along the true bearing — the shared rl#240 lure
        // (crab_world::training::targets), the same clamp the eval's pace probe
        // measures under (rl#280). Current-map offsets sit well inside the band, so
        // the clamp is dormant.
        let to_prey = hunt - crab.world_pos_m;
        if to_prey.length_squared() < 1e-6 {
            continue;
        }

        let origin = spawns.origin(idx);
        let carapace = carapace_q
            .iter()
            .find(|(env, _)| env.0 == idx)
            .map(|(_, t)| t.translation)
            .unwrap_or(origin);

        // The lure's y rides the surface at the LURE's xz ([`CLAW_TARGET_Y`]) — an
        // absolute y here would feed `target:y` an obs off by the local elevation,
        // unmeasured by the rl#240 drift guard (which watches only body.pos).
        let lure = band_lure(carapace, to_prey, 0.0);
        *slot = Some(terrain.place(Vec2::new(lure.x, lure.z), CLAW_TARGET_Y));
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
    terrain: Res<crab_world::terrain::Terrain>,
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
                let Some((a, b, radius)) = claw_tip_capsule(col.as_typed_shape(), Mat4::IDENTITY)
                else {
                    // A claw the sim can't model is a code defect, never a skippable
                    // row: this claw would stop touching players in MP with no other
                    // symptom (rl#288). ERROR so it surfaces through fleet telemetry.
                    error_once!(
                        "claw capture: claw-tip collider is not capsule-readable — \
                         this claw is INVISIBLE to MP claw-touch (rl#288)"
                    );
                    return None;
                };
                let rel = |p: Vec3| {
                    let w = t.transform_point(p);
                    // y is height above the surface under the claw point, so the sim's
                    // surface-relative player span applies on any grid (flat: surface ≡ 0).
                    Vec3::new(w.x - here.x, w.y - terrain.height(w.x, w.z), w.z - here.y)
                };
                Some(ArenaClaw {
                    a: rel(a),
                    b: rel(b),
                    radius,
                })
            })
            .collect();
    }
}

/// Resolve a claw-tip collider to its capsule — segment endpoints in entity-local
/// space plus radius. `spawn_crab` wraps every placed shape as a ONE-shape compound
/// (today only cuboid links; the read-through is future-proofing so the pincer stays
/// readable if capsules ever ship wrapped the same way), so readable = a bare capsule,
/// or a one-shape compound resolving to one. `None` = anything else — a multi-shape
/// compound is a shape our claw model can't honestly reduce, not a wrapper — and the
/// caller screams: a bare `as_capsule` here silently dropped the claw from MP
/// claw-touch (rl#288).
fn claw_tip_capsule(view: ColliderView<'_>, local: Mat4) -> Option<(Vec3, Vec3, f32)> {
    match view {
        ColliderView::Capsule(c) => {
            let seg = c.segment();
            Some((
                local.transform_point3(seg.a()),
                local.transform_point3(seg.b()),
                c.radius(),
            ))
        }
        ColliderView::Compound(c) => {
            let mut shapes = c.shapes();
            match (shapes.next(), shapes.next()) {
                (Some((pos, rot, sub)), None) => {
                    claw_tip_capsule(sub, local * Mat4::from_rotation_translation(rot, pos))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Where the shared physics arena sits in the render world — a per-round translate pinned
/// by crab 0's spawn correspondence (game spawn ↔ arena spawn origin), static for the
/// whole round: an rl#240 recenter moves both ends of the correspondence together
/// (origin rebase + `anchor_world_m` advance), so the published difference never
/// changes; only a round RESTART re-pins it. Its y is 0 by construction — the world's
/// y datum IS the arena's (the terrain surface renders untranslated in y and sim
/// heights are surface-relative), so world = arena + anchor holds componentwise on the
/// baked tile with no vertical leg to compensate (rl#281 stage 6).
/// Vehicles and the cockpit camera render through THIS, never through the per-crab skin
/// repose: the repose re-tracks the live carapace every tick to pin Sally's skin to her sim
/// spot, so borrowing it as the arena transform dragged ~(1−rs) of her every movement into
/// every craft's rendered pose — the ship visibly danced whenever she wiggled (rl#224).
/// Since rl#254 the skin and arena frames advance identically as she walks (the trade the
/// pre-fix (1−rs)-per-step drift posed is gone); a pilot's aim at her SKIN differs from her
/// collider only by her settle-window motion, a small per-round constant. For crabs beyond 0
/// the same holds from every round RESTART on — [`restart_bridge_to_spawns`] re-pins the whole
/// origin layout to the sim spawns (rl#289) — but the fresh-app FIRST round keeps a nonzero
/// tick-0 offset (`spawn_initial_crabs` lays origins on its own grid, not the sim spawn
/// layout), pre-existing with the old borrowed-repose anchor too.
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

#[cfg(test)]
mod claw_tip_capsule_tests {
    use super::*;

    #[test]
    fn reads_bare_and_compound_wrapped_capsules() {
        let a = Vec3::new(0.0, -0.05, 0.0);
        let b = Vec3::new(0.0, 0.05, 0.0);
        let bare = Collider::capsule(a, b, 0.02);
        let (ba, bb, br) = claw_tip_capsule(bare.as_typed_shape(), Mat4::IDENTITY).unwrap();
        assert!((ba - a).length() < 1e-6 && (bb - b).length() < 1e-6 && (br - 0.02).abs() < 1e-6);

        // A one-shape-compound-wrapped capsule (spawn_crab's wrapping convention)
        // must read through with the sub-shape placement folded in.
        let off = Vec3::new(0.1, 0.2, 0.3);
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let wrapped = Collider::compound(vec![(off, rot, Collider::capsule(a, b, 0.02))]);
        let (wa, wb, wr) = claw_tip_capsule(wrapped.as_typed_shape(), Mat4::IDENTITY).unwrap();
        assert!((wa - (off + rot * a)).length() < 1e-5);
        assert!((wb - (off + rot * b)).length() < 1e-5);
        assert!((wr - 0.02).abs() < 1e-6);
    }

    #[test]
    fn refuses_non_capsule_shapes() {
        let boxy = Collider::compound(vec![(
            Vec3::ZERO,
            Quat::IDENTITY,
            Collider::cuboid(0.1, 0.1, 0.1),
        )]);
        assert!(claw_tip_capsule(boxy.as_typed_shape(), Mat4::IDENTITY).is_none());
        let bare_box = Collider::cuboid(0.1, 0.1, 0.1);
        assert!(claw_tip_capsule(bare_box.as_typed_shape(), Mat4::IDENTITY).is_none());
        // A multi-shape compound is not a wrapper — even one containing a capsule
        // cannot be honestly reduced to the claw model, so it must refuse.
        let multi = Collider::compound(vec![
            (Vec3::ZERO, Quat::IDENTITY, Collider::capsule_y(0.05, 0.02)),
            (Vec3::X, Quat::IDENTITY, Collider::cuboid(0.1, 0.1, 0.1)),
        ]);
        assert!(claw_tip_capsule(multi.as_typed_shape(), Mat4::IDENTITY).is_none());
    }
}

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
        assert_ne!(
            prev_anchor.0,
            Vec3::ZERO,
            "the armed round published an anchor"
        );
        let mut prev_recenters = recenters(&app);

        let mut max_ship_d = 0.0f32;
        for t in 0..1200u32 {
            app.world_mut().resource_mut::<Wiggle>().0 = if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
            // The pre-rl#20 flail marched 9.6 m/400 ticks and crossed the band on its
            // own; the baked-collider-table crab (rl#20 phase 2) only shuffles 3-7 m,
            // so chaos no longer arms the recenter this test exists to exercise.
            // Teleport her past the band once instead — the same drift trigger
            // production sees, minus the chaotic-gait pin that keeps rotting.
            if t == 600 {
                let mut cq = app
                    .world_mut()
                    .query_filtered::<&Transform, With<CrabCarapace>>();
                let carapace_x = cq.single(app.world()).expect("carapace").translation.x;
                // Absolute, not relative: her chaotic shuffle may sit anywhere inside
                // the band, so land the CARAPACE at band + 2 m from the origin.
                let dx = (TARGET_ARENA_HALF + 2.0) - carapace_x;
                super::gcr_crab_tests::shift_parts(&mut app, Vec3::new(dx, 0.0, 0.0));
            }
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
            "the t=600 past-the-band teleport must arm a recenter — if it no longer \
             does, the recenter machinery itself changed and the epoch assertions \
             above went vacuous"
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

    /// rl#240 flip pins, stage-6 form: a recenter is an origin REBASE — the anchor and
    /// every Transform (crab, parked craft, a boarding pending across the recenter
    /// tick) are untouched, so nothing can move on screen by construction; only the
    /// spawn origin (and with it the body.pos obs channel) snaps.
    #[test]
    fn recenter_rebases_origin_and_touches_no_transform_or_anchor() {
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

        // A second pilot boards right as the drift crosses the band: pending across the
        // recenter tick, its craft must still materialise at the authored arena spot —
        // trivially now, since the frame no longer shifts underneath it.
        app.world_mut().resource_mut::<VehicleControls>().0.insert(
            PilotId(1),
            PilotCommand::new(VehicleKind::Ship, boarding_at(7.0, -3.0)),
        );
        let crafts_now = {
            let mut q = app.world_mut().query_filtered::<(), With<Vehicle>>();
            q.iter(app.world()).count()
        };
        assert_eq!(crafts_now, 1, "pilot 1's craft must not exist pre-recenter");

        // Emulate a long chase: walk the whole crab 20 m out in one step.
        super::gcr_crab_tests::shift_parts(&mut app, Vec3::new(20.0, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        assert_eq!(recenters(&app), 1, "a 20 m drift must trigger one recenter");
        let anchor1 = app.world().resource::<ArenaAnchor>().0;
        assert!(
            (anchor1 - anchor0).length() < 1e-4,
            "an origin rebase must not move the published anchor, moved {:?}",
            anchor1 - anchor0
        );
        let origin = app.world().resource::<CrabSpawns>().origin(0);
        let carapace = {
            let mut q = app
                .world_mut()
                .query_filtered::<&Transform, With<CrabCarapace>>();
            q.single(app.world()).expect("carapace").translation
        };
        assert!(
            Vec2::new(origin.x - carapace.x, origin.z - carapace.z).length() < 1.0,
            "the rebased origin must sit at her ground point, origin {origin:?} vs \
             carapace {carapace:?}"
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
            (parked - ship0).length() < 0.05,
            "the parked craft's arena pose must be untouched by a rebase, moved {:?}",
            parked - ship0
        );
        let miss = Vec2::new(boarded.x - 7.0, boarded.z - (-3.0)).length();
        assert!(
            miss < 1.0,
            "a boarding pending across the recenter tick must materialise at its \
             authored arena spot, landed {miss} m off"
        );
    }

    /// rl#204 RESTART after a recenter: the bridge's anchor snaps back to spawn, so the
    /// surviving craft must be carried by that reset delta — uncarried, the anchor drop
    /// would teleport a still-flying craft (and its pilot-follow feed) by the full
    /// accumulated recenter distance on screen.
    #[test]
    fn restart_after_recenter_keeps_surviving_craft_on_screen() {
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

        super::gcr_crab_tests::shift_parts(&mut app, Vec3::new(20.0, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }
        assert_eq!(
            recenters(&app),
            1,
            "the drift must recenter before the restart"
        );

        let spawn = Pos::from_meters(30.0, -14.0);
        restart_bridge_to_spawns(app.world_mut(), &[spawn]);
        cold_respawn_armed_crab(app.world_mut());
        app.update();

        let anchor = app.world().resource::<ArenaAnchor>().0;
        let moved = ship_render_pos(&mut app).distance(render0);
        assert!(
            moved < 0.05,
            "a surviving craft must not move ON SCREEN through a round restart \
             (anchor reset must carry it), moved {moved} m"
        );
        // And the correspondence really did reset: the anchor is the spawn-pinned value
        // again, not the recenter-advanced one.
        let origin = app.world().resource::<CrabSpawns>().origin(0);
        let want = Vec3::new(30.0 - origin.x, 0.0, -14.0 - origin.z);
        assert!(
            (anchor - want).length() < 1e-3,
            "the restart must re-pin the anchor to the spawn correspondence, \
             got {anchor:?}, want {want:?}"
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
    pub(super) fn shift_parts(app: &mut App, delta: Vec3) {
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
        let w0 = app.world().resource::<ExternalCrabBridge>().crabs[0].world_pos_m;

        shift_parts(&mut app, Vec3::new(20.0, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        let crab = &app.world().resource::<ExternalCrabBridge>().crabs[0];
        assert_eq!(crab.recenters, 1, "a 20 m drift must trigger one recenter");
        // The walk folds into world_pos_m exactly once, 1:1 (rl#256).
        let walked = crab.world_pos_m - w0;
        let want = Vec2::new(20.0, 0.0);
        assert!(
            (walked - want).length() < 0.5,
            "world_pos_m must gain the walk and nothing else, \
             gained {walked:?}, want {want:?}"
        );
        // The rebase moves the ORIGIN to her, never her to the origin.
        assert!(
            carapace_xz(&mut app).x > 15.0,
            "the crab must stay where it walked — a rebase teleports nothing"
        );
        let origin = app.world().resource::<CrabSpawns>().origin(0);
        assert!(
            (Vec2::new(origin.x, origin.z) - carapace_xz(&mut app)).length() < 1.0,
            "the origin must have rebased to her ground point, got {origin:?}"
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
    fn unarmed_only_measures_and_never_recenters() {
        let mut app = gcr_like_app();
        // The flip default arms recenter via the plugin — disarm to pin the
        // measure-only path (the flip-back configuration).
        app.world_mut().remove_resource::<BodyPosRecenter>();
        let origin0 = app.world().resource::<CrabSpawns>().origin(0);

        shift_parts(&mut app, Vec3::new(20.0, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        let crab = &app.world().resource::<ExternalCrabBridge>().crabs[0];
        assert_eq!(crab.recenters, 0, "unarmed: never recenter");
        assert!(
            crab.next_drift_log_m > TARGET_ARENA_HALF + DRIFT_LOG_STEP_M,
            "the drift crossing must advance the log cursor, got {}",
            crab.next_drift_log_m
        );
        assert_eq!(
            app.world().resource::<CrabSpawns>().origin(0),
            origin0,
            "unarmed: the origin never rebases"
        );
        assert!(
            carapace_xz(&mut app).x > 15.0,
            "unarmed: the crab stays where it walked"
        );
    }

    /// The old single-crab restriction is gone with the teleport (rl#281 stage 6):
    /// origins are per-env state, so a multi-crab round recenters exactly the drifted
    /// env — and the crab-0-pinned anchor stays put when a non-zero env rebases too.
    #[test]
    fn multi_crab_rounds_recenter_only_the_drifted_env() {
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
        let anchor0 = *app.world().resource::<ArenaAnchor>();
        let origin1_before = app.world().resource::<CrabSpawns>().origin(1);

        // Walk only env 1's crab out of the band: the non-anchor env must rebase its
        // own origin without touching env 0's or the shared anchor.
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &mut Transform), With<CrabBodyPart>>();
        for (env, mut t) in q.iter_mut(app.world_mut()) {
            if env.0 == 1 {
                t.translation += Vec3::new(20.0, 0.0, 0.0);
            }
        }
        for _ in 0..2 {
            app.update();
        }

        let bridge = app.world().resource::<ExternalCrabBridge>();
        assert_eq!(bridge.crabs[1].recenters, 1, "the drifted env recenters");
        assert_eq!(bridge.crabs[0].recenters, 0, "the standing env does not");
        let origin1 = app.world().resource::<CrabSpawns>().origin(1);
        assert!(
            origin1.x - origin1_before.x > 15.0,
            "env 1's origin must have rebased to its walked-out crab, got {origin1:?}"
        );
        assert_eq!(
            *app.world().resource::<ArenaAnchor>(),
            anchor0,
            "a non-anchor env's rebase must not move the crab-0-pinned anchor"
        );
    }

    /// rl#289: a round RESTART re-pins EVERY env's origin to the fresh sim spawn
    /// layout, not just env 0's anchor end. Without it, an env≠0 origin keeps last
    /// round's rebased wander while its sim end respawns fresh, so its arena↔sim
    /// offset diverges from the shared anchor by the accumulated differential wander —
    /// the crab's skin renders horizontally off its ram collider and off the local
    /// ground height.
    #[test]
    fn restart_repins_every_env_origin_to_the_sim_spawn_layout() {
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

        // Wander only env 1 out of the band: its origin rebases — the differential
        // wander a bare bridge restart used to leave in the layout.
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &mut Transform), With<CrabBodyPart>>();
        for (env, mut t) in q.iter_mut(app.world_mut()) {
            if env.0 == 1 {
                t.translation += Vec3::new(20.0, 0.0, 0.0);
            }
        }
        for _ in 0..2 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<ExternalCrabBridge>().crabs[1].recenters,
            1,
            "env 1 must carry a rebased origin into the restart"
        );

        let spawns = [Pos::from_meters(3.0, -2.0), Pos::from_meters(9.0, 4.0)];
        restart_bridge_to_spawns(app.world_mut(), &spawns);
        cold_respawn_armed_crab(app.world_mut());
        app.update();

        let (o0, o1) = {
            let origins = app.world().resource::<CrabSpawns>();
            (origins.origin(0), origins.origin(1))
        };
        let got = Vec2::new(o1.x - o0.x, o1.z - o0.z);
        let want = pos_to_m(spawns[1]) - pos_to_m(spawns[0]);
        assert!(
            (got - want).length() < 1e-3,
            "the restart must re-pin env 1's origin to the sim spawn layout about \
             env 0's kept origin, inter-origin delta {got:?}, want {want:?}"
        );
        // And the cold respawn landed env 1's body ON its re-pinned origin — origin and
        // body agree in the same call (rl#242), so its repose shift equals the anchor.
        let carapace1 = {
            let mut q = app
                .world_mut()
                .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>();
            q.iter(app.world())
                .find(|(env, _)| env.0 == 1)
                .expect("env 1's carapace respawned")
                .1
                .translation
        };
        assert!(
            Vec2::new(carapace1.x - o1.x, carapace1.z - o1.z).length() < 1.0,
            "env 1's crab must respawn on its re-pinned origin, carapace {carapace1:?} \
             vs origin {o1:?}"
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

    /// rl#281 stage-6 review catch: the posed hunt target's y must ride the SURFACE at
    /// the lure's own xz. An absolute y feeds `target:y` an obs off by the full local
    /// elevation — and nothing measures it: the rl#240 drift guard watches only
    /// body.pos, which the origin rebase keeps in-band.
    #[test]
    fn walk_target_y_rides_the_surface_on_terrain() {
        pin_single_thread_pools();
        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            arena: crab_world::physics::Arena::Terrain,
            visuals: crab_world::Visuals(false),
        });
        app.add_plugins(ExternalCrabPlugin::new(
            vec![Policy::rest()],
            vec![Pos::from_meters(0.0, 0.0)],
        ));
        arm(app.world_mut());
        force_serial_schedules(&mut app);
        for _ in 0..64 {
            app.update();
        }

        // Walk her to unambiguous elevation (the tile center's datum is ~0, which
        // couldn't distinguish surface-relative from absolute): scan outward for a
        // ≥20 m spot, teleport onto its surface, and let the recenter rebase there.
        let terrain = app
            .world()
            .resource::<crab_world::terrain::Terrain>()
            .clone();
        let cara0 = carapace_xz(&mut app);
        let spot = (1..40)
            .flat_map(|r| [(r as f32 * 100.0, 0.0), (0.0, r as f32 * 100.0)])
            .map(|(x, z)| Vec2::new(x, z))
            .find(|p| terrain.height(p.x, p.y).abs() > 20.0)
            .expect("the seed-281 tile has >20 m relief within 4 km of center");
        let dh = terrain.height(spot.x, spot.y) - terrain.height(cara0.x, cara0.y);
        shift_parts(&mut app, Vec3::new(spot.x - cara0.x, dh, spot.y - cara0.y));
        for _ in 0..2 {
            app.update();
        }

        let cara = carapace_xz(&mut app);
        let prey = cara + Vec2::new(50.0, 0.0); // beyond band ⇒ clamps to the 9 m edge
        app.world_mut()
            .resource_mut::<ExternalCrabBridge>()
            .set_hunt_target(0, Some(Pos::from_meters(prey.x, prey.y)));
        app.update();
        let posed = app.world().resource::<CrabTargets>().envs[0].expect("target posed");
        let surface = terrain.height(posed.x, posed.z);
        assert!(
            surface.abs() > 1.0,
            "non-vacuous: the lure must sit on real elevation, surface {surface}"
        );
        assert!(
            (posed.y - (surface + CLAW_TARGET_Y)).abs() < 0.05,
            "target y must be {CLAW_TARGET_Y} above the surface at ITS xz \
             (surface {surface}), got {}",
            posed.y
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
