//! The host's crab slot: the seam between [`Server::advance`] and
//! [`Server::step_next`] where the tick's crab physics — including each crab's policy
//! forward (`run_crab_policy` in `BotSet::Think`) — runs in the host's ONE world, and
//! the resulting world poses are handed to `step_next`. An INTERNAL seam: only the
//! server-auth arm calls [`pump_crab_slot`] — the windowed driver and the renderless
//! [`HeadlessHostWorld`] go through this one function, so a host with and without a
//! renderer cannot drift. Remote-adopt clients never enter it: their `FixedUpdate` is
//! parked ([`park_fixed_auto_pump`]) and nothing pumps it, so the policy is host-side
//! by construction; clients only consume the resulting `CoreSnapshot` +
//! `CrabArticulation` streams.
//!
//! ONE frame since rl#298 stage 5: the world's coordinates ARE the sim's meters. Sally
//! is world content — her sim pose is READ off her carapace every tick
//! ([`collect_crab_poses`]), her claws are her real physics capsules in world frame,
//! and the deleted `external_crab` bridge's local-arena translation (per-round
//! `ArenaAnchor`, carapace-delta integration into `world_pos_m`, drift/anchor
//! lockstep) has nothing left to translate. The bridge's band clamp on the posed hunt
//! target came back world-frame (rl#301, [`set_crab_walk_target`]): piloted prey can
//! out-range the training band even with one frame.
//!
//! The obs half of the seam (rl#298 stage 3): each pumped step builds the policy's
//! observation from THIS world's ECS (`build_observation` in `BotSet::Sense`), and the
//! in-game action period is one physics step — training's control period, by shared
//! construction: the trainer's `brain_step` and the host's `run_crab_policy` both run
//! in `BotSet::Think` of the same `FixedUpdate` schedule, pumped once per
//! `PHYSICS_DT` (`headless_stack`'s manual time drive there, [`pump_fixed_steps`]
//! here). The stage-3 tests below pin both halves.
//!
//! [`Server::advance`]: crate::server::Server::advance
//! [`Server::step_next`]: crate::server::Server::step_next

use bevy::prelude::*;
use bevy_rapier3d::geometry::ColliderView;
use bevy_rapier3d::prelude::*;

use crate::sim::{ClawPose, CrabPose, Pos, Sim};
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId};
use crab_world::bot::sensor::CrabTargets;
use crab_world::bot::{BotSet, CrabSpawns, recenter_drifted_origins};
#[cfg(feature = "render")]
use crab_world::crab_view::CrabBrainLabels;
use crab_world::policy::Policy;
use crab_world::training::targets::band_lure;

/// The posed hunt target's height ABOVE THE LOCAL SURFACE at the target's own xz —
/// training's convention (the ball sampler bands height-above-surface), so `target:y`
/// stays in-distribution on terrain; flat grids reduce to the old absolute 0.3.
pub(crate) const CLAW_TARGET_Y: f32 = 0.3;

/// Pursuit-heartbeat cadence: one range line per this many hunt-fed ticks (~5 s at the
/// 60 Hz fixed step) — so a "she stopped giving chase" report (rl#265) is diagnosable
/// from any run's log, live deck telemetry included, without a repro.
const HUNT_LOG_EVERY_TICKS: u64 = 300;

pub struct CrabPolicies(pub Vec<Policy>);

/// The armed marker: the NN crabs drive (policy forward, hunt posing, origin
/// recenter). Absent = the stack is installed but the round is not live.
#[derive(Resource)]
pub struct NnCrabsArmed;

fn crabs_armed(active: Option<Res<NnCrabsArmed>>) -> bool {
    active.is_some()
}

pub fn arm(world: &mut World) {
    world.insert_resource(NnCrabsArmed);
    world.insert_resource(crab_world::bot::CrabRescueIsFault);
}

fn pos_to_m(p: Pos) -> Vec2 {
    let (x, z) = p.to_meters();
    Vec2::new(x, z)
}

/// Per-env counter of consecutive post-settle ticks [`collect_crab_poses`] found no
/// carapace to read — the crab's sim pose holds at its last adopted spot (still lethal
/// there) while the world looks healthy, so the miss is counted and reported like
/// drift, never skipped silently (rl#241). Resets when the carapace comes back; the
/// settle window and the rescue-owned non-finite state are legitimate, not misses.
#[derive(Default)]
struct Miss {
    ticks: u64,
    /// Log cursor: the next miss count worth a log line (doubles per line, so a
    /// persistent miss is quantified without flooding).
    next_log: u64,
}

#[derive(Resource)]
struct SlotMisses(Vec<Miss>);

/// Post-settle rest-hold countdown per env, re-armed by a respawn (round restart,
/// rescue): the freshly-placed ragdoll settles before the policy drives it — and the
/// pre-spawn window (obs/action slots not yet sized) lives entirely inside it, which
/// is why `run_crab_policy`'s slot check is a hard assert, not a skip.
#[derive(Resource)]
struct CrabSettle(Vec<u32>);

/// The hunt targets the slot fed after the last pump — consumed by
/// [`set_crab_walk_target`] on the NEXT pump, exactly the ordering the bridge had:
/// this tick's pump walks on LAST tick's targets.
#[derive(Resource, Default)]
struct CrabHunt {
    targets: Vec<Option<Pos>>,
    fed_ticks: u64,
}

pub struct NnCrabPlugin {
    /// The policies the launch gate loaded and vetted — the plugin never touches disk, so
    /// what the gate armed IS what drives (rl#241: a plugin-side re-load could race
    /// a checkpoint swap and warn-and-arm a rest-pose statue the gate never saw). `Mutex<
    /// Option<…>>` only because `Plugin::build` takes `&self`; `build` moves them out.
    policies: std::sync::Mutex<Option<Vec<Policy>>>,
    crab_spawns: Vec<Pos>,
}

impl NnCrabPlugin {
    pub fn new(policies: Vec<Policy>, crab_spawns: Vec<Pos>) -> Self {
        Self {
            policies: std::sync::Mutex::new(Some(policies)),
            crab_spawns,
        }
    }
}

impl Plugin for NnCrabPlugin {
    fn build(&self, app: &mut App) {
        let policies = self
            .policies
            .lock()
            .unwrap()
            .take()
            .expect("NnCrabPlugin is built once");
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
        app.insert_resource(SlotMisses(
            self.crab_spawns.iter().map(|_| Miss::default()).collect(),
        ));
        app.insert_resource(CrabSettle(vec![
            crab_world::bot::RESET_GRACE_TICKS;
            self.crab_spawns.len()
        ]));
        app.init_resource::<CrabHunt>();

        app.add_systems(
            Update,
            ensure_crab_env
                .run_if(crabs_armed)
                .before(crab_world::bot::spawn_initial_crabs),
        );
        app.add_systems(
            Update,
            crab_world::bot::spawn_initial_crabs
                .run_if(crabs_armed)
                .run_if(crab_not_yet_spawned),
        );

        app.add_systems(
            FixedUpdate,
            (
                // After rescue: a rescued env respawns at its origin, so the recenter
                // sees ~0 drift there instead of racing the respawn.
                recenter_drifted_origins
                    .after(crab_world::bot::rescue_lost_crabs)
                    .before(BotSet::Sense),
                set_crab_walk_target.before(BotSet::Sense),
                run_crab_policy.in_set(BotSet::Think),
            )
                .run_if(crabs_armed),
        );
        // Publish each binding's on-screen brain label (rl#200 increment 7). In FixedUpdate
        // deliberately: only the physics-pumping peer (solo/host) advances FixedUpdate
        // (the wall-clock auto-pump is PARKED — `park_fixed_auto_pump`, exact since
        // rl#298 stage 4 — and `pump_fixed_steps` is driven by the client-sim tick drain),
        // so this is host-only by construction; on a remote-adopt client the articulation
        // `apply` is the sole label writer and the two can't fight over the resource.
        // Render-gated: labels feed UI a renderless host doesn't have — the seam itself
        // is render-free (rl#298 stage 4).
        #[cfg(feature = "render")]
        {
            app.init_resource::<CrabBrainLabels>();
            app.add_systems(FixedUpdate, publish_brain_labels.run_if(crabs_armed));
        }
    }
}

fn ensure_crab_env(settle: Res<CrabSettle>, mut num_envs: ResMut<crab_world::bot::NumEnvs>) {
    let want = settle.0.len();
    if num_envs.0 < want {
        num_envs.0 = want;
    }
}

fn crab_not_yet_spawned(crabs: Query<(), With<CrabCarapace>>) -> bool {
    crabs.is_empty()
}

/// Keep [`CrabBrainLabels`] current with the bindings — one label per env, formatted by the
/// ONE formatter (`Policy::brain_label`), write-on-change. GCR policies never hot-reload, so
/// this settles to one write per arm; it stays a system (not an arm-time one-shot) so the
/// labels can never go stale against whatever drives the crabs. Teardown clears the resource
/// (`crate::render`'s `teardown_round`), un-labeling the crab bodies that outlive the round.
#[cfg(feature = "render")]
fn publish_brain_labels(policies: NonSend<CrabPolicies>, mut labels: ResMut<CrabBrainLabels>) {
    let want: Vec<String> = policies.0.iter().map(|p| p.brain_label()).collect();
    if labels.0 != want {
        labels.0 = want;
    }
}

/// Pose each env's walk target at its hunt prey, lured to at most one band edge from
/// the carapace ([`band_lure`], rl#301): a walking prey always sits far inside the
/// training band — she out-charges any walker — but a PILOTED craft's shadow
/// (`nearest_living_player_pos` carries the plane's planar pos) can leave
/// `BAND_MAX_M` entirely, and the sensor's `target_local` is unclamped, so an
/// unlured far flight would feed the target obs ~4× outside training support
/// (worst case the rl#137 non-finite → rescue class). y rides the surface at the
/// LURED point's xz ([`CLAW_TARGET_Y`]), training's convention, so `target:y` stays
/// in-distribution on terrain.
fn set_crab_walk_target(
    hunt: Res<CrabHunt>,
    terrain: Res<crab_world::terrain::Terrain>,
    carapaces: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    mut targets: ResMut<CrabTargets>,
) {
    let mut lure_from: Vec<Option<Vec2>> = vec![None; hunt.targets.len()];
    for (env, t) in &carapaces {
        if let Some(slot) = lure_from.get_mut(env.0)
            && t.translation.is_finite()
        {
            *slot = Some(Vec2::new(t.translation.x, t.translation.z));
        }
    }
    for (idx, prey) in hunt.targets.iter().enumerate() {
        // Pre-spawn the slots don't exist (settle grace); post-settle a miss is the
        // slot-desync class `run_crab_policy` panics on the same tick — skip THIS crab
        // only, never abort the whole loop.
        let Some(slot) = targets.envs.get_mut(idx) else {
            continue;
        };
        *slot = prey.map(|p| {
            let m = pos_to_m(p);
            // No finite carapace ⇒ nothing to lure FROM — pose at the prey raw. The
            // settle-grace and rescue-owned non-finite windows hold the policy off
            // the obs; a post-settle missing carapace is the rl#241 miss class,
            // counted and reported by `collect_crab_poses`.
            let m = match lure_from[idx] {
                Some(cxz) => band_lure(cxz, m - cxz),
                None => m,
            };
            terrain.place(m, CLAW_TARGET_Y)
        });
    }
}

fn run_crab_policy(
    policies: NonSend<CrabPolicies>,
    mut settle: ResMut<CrabSettle>,
    mut rescued: MessageReader<crab_world::bot::CrabRescued>,
    obs: Res<crab_world::bot::sensor::CrabObservation>,
    mut actions: ResMut<crab_world::bot::actuator::CrabActions>,
) {
    for m in rescued.read() {
        // A rescue respawned the env at its origin — give the fresh ragdoll the same
        // settle grace a restart does.
        if let Some(s) = settle.0.get_mut(m.env) {
            *s = crab_world::bot::RESET_GRACE_TICKS;
        }
    }
    let crab_count = settle.0.len();
    assert_eq!(policies.0.len(), crab_count, "one policy per slot crab");
    for (idx, (policy, grace)) in policies.0.iter().zip(settle.0.iter_mut()).enumerate() {
        if *grace > 0 {
            *grace = crab_world::bot::settle_countdown(*grace);
            let _ = actions.rest(idx); // deliberate skip pre-spawn (see CrabSettle)
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
            "crab slot: crab {idx} has no env slot ({} obs / {} action slots sized \
             for {} crabs) — it would silently hold rest pose in a live round (rl#241)",
            obs.rows().len(),
            actions.len(),
            crab_count,
        );
    }
}

/// One env's carapace read before the [`CrabPose`] materializes — `yaw` is `None`
/// when she stands still (no planar velocity to derive a facing from): the merge
/// holds the sim's current facing there instead of inventing one.
struct ReadPose {
    pos: Pos,
    yaw: Option<i32>,
    claws: Vec<ClawPose>,
}

/// Read every crab's world pose + claw capsules off its body — the sim feed
/// ([`crate::sim::Externals::crabs`]). One frame: the carapace's translation IS her
/// sim position, her claw capsules cross in world coordinates with surface-relative
/// heights ([`ClawPose`]'s convention). Yaw derives from the carapace's planar
/// velocity. `fallback` is each crab's current sim pose ([`SlotInputs::fallback`]):
/// where the world has nothing readable, the sim keeps acting at her last adopted
/// spot, clawless — a claw capture must never outlive the tick that measured it
/// (rl#294), so a stale-lethal window is unrepresentable. A missing carapace past the
/// settle window is counted loudly (rl#241); non-finite is the rescue path's.
pub(crate) fn collect_crab_poses(world: &mut World, fallback: &[CrabPose]) -> Vec<CrabPose> {
    let terrain = world.resource::<crab_world::terrain::Terrain>().clone();
    let mut read: std::collections::BTreeMap<usize, ReadPose> = Default::default();
    let mut rescue_owned: std::collections::BTreeSet<usize> = Default::default();
    {
        let mut carapace_q =
            world.query_filtered::<(&CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>();
        for (env, t, vel) in carapace_q.iter(world) {
            if !t.translation.is_finite() {
                // The rescue path owns non-finite crabs (a Fault when armed) — present,
                // not a miss.
                rescue_owned.insert(env.0);
                continue;
            }
            let here = t.translation;
            let v = Vec2::new(vel.linear.x, vel.linear.z);
            read.entry(env.0).or_insert(ReadPose {
                pos: Pos::from_meters(here.x, here.z),
                yaw: (v.length_squared() > 1e-4)
                    .then(|| crate::sim::trig_client::radians_to_turns(v.x.atan2(v.y))),
                claws: Vec::new(),
            });
        }
    }
    {
        // Capture her claw pincers' REAL colliders (rl#249) — the sim decides
        // claw-touch downs against these, so there is no second hitbox to drift from
        // the physics claw.
        let mut claw_q =
            world.query_filtered::<(&CrabEnvId, &Transform, &Collider), With<CrabClawTip>>();
        let mut claws: Vec<(usize, ClawPose)> = Vec::new();
        for (env, t, col) in claw_q.iter(world) {
            if !read.contains_key(&env.0) {
                continue;
            }
            let Some((a, b, radius)) = claw_tip_capsule(col.as_typed_shape(), Mat4::IDENTITY)
            else {
                // A claw the sim can't model is a code defect, never a skippable
                // row: this claw would stop touching players in MP with no other
                // symptom (rl#288). ERROR so it surfaces through fleet telemetry.
                error_once!(
                    "claw capture: claw-tip collider is not capsule-readable — \
                     this claw is INVISIBLE to MP claw-touch (rl#288)"
                );
                continue;
            };
            // y is height above the surface under the claw point, so the sim's
            // surface-relative player span applies on any grid (flat: surface ≡ 0).
            let world_pt = |p: Vec3| {
                let w = t.transform_point(p);
                (
                    Pos::from_meters(w.x, w.z),
                    crate::sim::meters_to_grid(w.y - terrain.height(w.x, w.z)),
                )
            };
            let (pa, ay) = world_pt(a);
            let (pb, by) = world_pt(b);
            claws.push((
                env.0,
                ClawPose {
                    a: pa,
                    b: pb,
                    a_y: ay,
                    b_y: by,
                    radius: crate::sim::meters_to_grid(radius),
                },
            ));
        }
        for (env, claw) in claws {
            read.get_mut(&env).expect("gated above").claws.push(claw);
        }
    }

    let settle = world.resource::<CrabSettle>().0.clone();
    let mut misses = world.resource_mut::<SlotMisses>();
    assert_eq!(
        fallback.len(),
        misses.0.len(),
        "one fallback pose per slot crab — the sim's crab count must match the slot"
    );
    fallback
        .iter()
        .enumerate()
        .map(|(idx, held)| match read.remove(&idx) {
            Some(pose) => {
                let miss = &mut misses.0[idx];
                miss.ticks = 0;
                miss.next_log = 0;
                CrabPose {
                    pos: pose.pos,
                    // Standing still: hold the sim's current facing.
                    yaw: pose.yaw.unwrap_or(held.yaw),
                    claws: pose.claws,
                }
            }
            None => {
                // Pre-spawn (inside the settle grace) and rescue-owned non-finite
                // windows legitimately have nothing to read; past those, a miss is a
                // despawn/wiring bug — count and report it (rl#241).
                if settle.get(idx).is_none_or(|&s| s == 0) && !rescue_owned.contains(&idx) {
                    let miss = &mut misses.0[idx];
                    miss.ticks += 1;
                    if miss.ticks >= miss.next_log {
                        error!(
                            "crab slot: env {idx} has no carapace to read — sim pose held \
                             at her last adopted spot for {} ticks; a despawn/wiring bug, \
                             not a legitimate state (rl#241)",
                            miss.ticks
                        );
                        miss.next_log = miss.ticks * 2;
                    }
                }
                held.clone()
            }
        })
        .collect()
}

/// Feed the NEXT pump's hunt targets (and the ~5 s pursuit heartbeat, rl#265).
/// `poses` are the crab poses just collected — the heartbeat's range source.
pub(crate) fn feed_hunt(world: &mut World, hunt: &[Option<Pos>], poses: &[CrabPose]) {
    let mut state = world.resource_mut::<CrabHunt>();
    state.targets = hunt.to_vec();
    state.fed_ticks += 1;
    if state.fed_ticks.is_multiple_of(HUNT_LOG_EVERY_TICKS) {
        for (idx, prey) in hunt.iter().enumerate() {
            if let (Some(p), Some(pose)) = (prey, poses.get(idx)) {
                let range = (pos_to_m(*p) - pos_to_m(pose.pos)).length();
                info!("crab slot: env {idx} prey {range:.1} m away");
            }
        }
    }
}

/// The caller's half of the slot contract, computed from the authoritative sim BEFORE
/// the pump. One implementation for every caller of [`pump_crab_slot`], so the
/// pre-read cannot drift between the windowed driver, the renderless host, and the
/// probes.
pub(crate) struct SlotInputs {
    /// The tick being stepped INTO — sets the 64:30 step count.
    pub(crate) stepping_into: u64,
    /// Each crab's hunt target (the sim's nearest living player). The pump never
    /// mutates the sim, so a pre-pump read equals a post-pump one; it feeds AFTER the
    /// pump deliberately — this tick's pump walks on LAST tick's targets, exactly the
    /// bridge-era ordering.
    pub(crate) hunt: Vec<Option<Pos>>,
    /// Each crab's current sim pose, clawless — [`collect_crab_poses`]'s feed for an
    /// env with no readable carapace: the sim holds its own last adopted pose, so the
    /// slot needs no second pose-holder.
    pub(crate) fallback: Vec<CrabPose>,
}

pub(crate) fn slot_inputs(sim: &Sim) -> SlotInputs {
    SlotInputs {
        stepping_into: sim.tick() + 1,
        hunt: (0..sim.crabs().len())
            .map(|idx| sim.nearest_living_player_pos(idx))
            .collect(),
        fallback: sim
            .crabs()
            .iter()
            .map(|c| CrabPose {
                pos: c.pos(),
                yaw: c.yaw(),
                claws: Vec::new(),
            })
            .collect(),
    }
}

/// Run the crab slot for the tick being stepped INTO: pump the owed fixed steps
/// (sensing, policy forward, actuation, physics), read the crabs'
/// world poses + claws for [`Server::step_next`](crate::server::Server::step_next),
/// and feed the NEXT tick's hunt targets. [`pump_slot_steps`] is the same seam at an
/// explicit step count (the probes' 1:1 cadence) — pump→collect→feed ordering has one
/// owner.
pub(crate) fn pump_crab_slot(world: &mut World, inputs: &SlotInputs) -> Vec<CrabPose> {
    pump_slot_steps(
        world,
        crate::cadence::steps_for_tick(inputs.stepping_into),
        inputs,
    )
}

pub(crate) fn pump_slot_steps(world: &mut World, steps: u32, inputs: &SlotInputs) -> Vec<CrabPose> {
    pump_fixed_steps(world, steps);
    let poses = collect_crab_poses(world, &inputs.fallback);
    feed_hunt(world, &inputs.hunt, &poses);
    poses
}

/// The rl#204 RESTART: re-pin every env's spawn origin to the fresh sim spawn layout —
/// AT the sim's own coordinates, one frame (rl#298 stage 5) — cold-respawn every crab
/// onto its origin (same call, so origin and body never disagree, rl#242), and reset
/// the slot's bookkeeping. Crafts are untouched: with no bridge frame, a restart moves
/// no coordinate system under them.
pub(crate) fn restart_crabs_to_spawns(world: &mut World, spawns: &[Pos]) {
    let layout: Vec<Vec2> = spawns.iter().map(|&s| pos_to_m(s)).collect();
    let base = *layout
        .first()
        .expect("NnCrabPlugin guarantees at least one crab (rl#114)");
    if world.resource::<CrabSpawns>().is_empty() {
        // Fresh app: `spawn_initial_crabs` hasn't laid the origins yet — hand it
        // the layout so the FIRST origins are the sim spawns too, never the
        // training grid (rl#290).
        world.insert_resource(crab_world::bot::InitialCrabLayout {
            base_xz: base,
            spawns_m: layout,
        });
    } else {
        let terrain = world.resource::<crab_world::terrain::Terrain>().clone();
        world
            .resource_mut::<CrabSpawns>()
            .repin_layout(base, &layout, &terrain);
    }
    {
        let mut misses = world.resource_mut::<SlotMisses>();
        assert_eq!(
            spawns.len(),
            misses.0.len(),
            "restart spawns must cover every slot crab"
        );
        misses.0 = spawns.iter().map(|_| Miss::default()).collect();
    }
    world
        .resource_mut::<CrabSettle>()
        .0
        .fill(crab_world::bot::RESET_GRACE_TICKS);
    *world.resource_mut::<CrabHunt>() = CrabHunt::default();
    cold_respawn_armed_crab(world);
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

/// Manually drive `steps` fixed-schedule passes — the host's physics pump. The
/// wall-clock auto-pump is parked ([`park_fixed_auto_pump`]), so these calls are the
/// ONLY thing that advances `FixedMain`; the per-tick count comes from
/// [`crate::cadence::steps_for_tick`], the one source for the 64:30 staircase.
pub(crate) fn pump_fixed_steps(world: &mut World, steps: u32) {
    use bevy::app::FixedMain;
    use bevy::time::{Fixed, Time, Virtual};

    if steps == 0 {
        return;
    }
    for _ in 0..steps {
        let fixed = world.resource::<Time<Fixed>>().as_generic();
        *world.resource_mut::<Time>() = fixed;
        world.run_schedule(FixedMain);
    }
    let virt = world.resource::<Time<Virtual>>().as_generic();
    *world.resource_mut::<Time>() = virt;
}

/// Park the wall-clock fixed-timestep auto-pump — EXACTLY: every peer's `FixedUpdate`
/// then advances only through [`pump_fixed_steps`], which only the server-auth arm
/// calls. Two layers because neither alone is airtight: the overstep discard runs
/// every frame BEFORE bevy's accrual+expend, so the accumulator entering the expend
/// holds at most one frame's virtual delta — which the huge timestep can never expend
/// (a bare huge timestep still fired one rogue un-pumped action+physics step per
/// ~24 h of accumulated uptime, rl#298 stage 4).
pub(crate) fn park_fixed_auto_pump(app: &mut App) {
    use bevy::app::RunFixedMainLoop;

    app.world_mut()
        .resource_mut::<bevy::time::Time<bevy::time::Fixed>>()
        .set_timestep(std::time::Duration::from_secs(86_400));
    app.add_systems(
        RunFixedMainLoop,
        discard_parked_overstep.in_set(bevy::app::RunFixedMainLoopSystems::BeforeFixedMainLoop),
    );
}

/// [`park_fixed_auto_pump`]'s per-frame half: zero the fixed accumulator before the
/// frame's accrual. [`pump_fixed_steps`] is untouched — it runs `FixedMain` directly
/// and never consults the accumulator.
fn discard_parked_overstep(mut fixed: ResMut<bevy::time::Time<bevy::time::Fixed>>) {
    let overstep = fixed.overstep();
    fixed.discard_overstep(overstep);
}

/// A renderless host's crab world — the "headless HOST driver" (rl#298 stage 5):
/// the same server world the windowed host pumps ([`headless_server_world`] — the one
/// constructor the trainer's rollout env builds on too), armed and pumped through the
/// same [`pump_crab_slot`] seam. This is what makes crab poses mandatory everywhere:
/// the headless `game net` / solo harnesses serve REAL world-read poses (a rest-pose
/// brain when no checkpoint is bound) instead of the deleted inert-crab escape.
///
/// [`headless_server_world`]: crab_world::bot::headless::headless_server_world
pub struct HeadlessHostWorld {
    app: App,
}

impl HeadlessHostWorld {
    pub fn new(policies: Vec<Policy>, sim: &Sim) -> Self {
        use crab_world::bot::headless::{WorldRole, headless_server_world};

        let spawns: Vec<Pos> = sim.crabs().iter().map(|c| c.pos()).collect();
        let mut app = headless_server_world(
            spawns.len(),
            WorldRole::Standalone,
            // The canonical bake (rl#209, rl#293): the ground every production peer runs.
            crab_world::terrain::TerrainGrid::gcr(),
        );
        app.add_plugins(NnCrabPlugin::new(policies, spawns.clone()));
        arm(app.world_mut());
        park_fixed_auto_pump(&mut app);
        restart_crabs_to_spawns(app.world_mut(), &spawns);
        Self { app }
    }

    /// Drain every assembled-and-ready tick through the real host path
    /// (`advance` happened caller-side): pump the slot, step the server, return the
    /// stepped ticks in order.
    pub fn step_ready_ticks(
        &mut self,
        server: &mut crate::server::Server,
    ) -> Vec<crate::server::SteppedTick> {
        self.app.update();
        let mut out = Vec::new();
        while server.next_tick_ready() {
            let inputs = slot_inputs(server.sim());
            let poses = pump_crab_slot(self.app.world_mut(), &inputs);
            let stepped = server.step_next(&poses, Default::default());
            if stepped.restarted {
                // Mirror the windowed driver: an rl#204 RESTART re-rolled the sim's
                // crabs, so the world's crabs re-pin and respawn with it — without
                // this, the next pump adopts the stale world pose and silently undoes
                // the restart for the crab.
                let spawns: Vec<Pos> = server.sim().crabs().iter().map(|c| c.pos()).collect();
                restart_crabs_to_spawns(self.app.world_mut(), &spawns);
            }
            out.push(stepped);
        }
        out
    }
}

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

/// One-frame pins (rl#298 stage 5): the world's coordinates ARE the sim's meters, so
/// the slot's pose read, hunt posing, origin bookkeeping, and restarts all work in
/// sim coordinates directly. Flat fixture grid where the geometry is hand-computed;
/// the terrain leg keeps its own gcr-grid test.
#[cfg(test)]
mod one_world_tests {
    use crab_world::bot::headless::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
    };

    use super::*;

    fn flat_field_grid() -> std::sync::Arc<crab_world::terrain::TerrainGrid> {
        std::sync::Arc::new(crab_world::terrain::TerrainGrid::flat(16_384.0))
    }

    /// The GCR host's crab stack minus the sim — one crab, rest-pose brain, spawned at
    /// a NONZERO sim spawn so absolute-frame assertions can't pass vacuously.
    fn one_world_app(spawns: &[Pos]) -> App {
        pin_single_thread_pools();
        let mut app = headless_stack(HeadlessStack {
            num_envs: spawns.len(),
            role: WorldRole::Standalone,
            grid: flat_field_grid(),
            visuals: crab_world::Visuals(false),
        });
        app.add_plugins(NnCrabPlugin::new(
            spawns.iter().map(|_| Policy::rest()).collect(),
            spawns.to_vec(),
        ));
        arm(app.world_mut());
        restart_crabs_to_spawns(app.world_mut(), spawns);
        force_serial_schedules(&mut app);
        // Past settle (RESET_GRACE_TICKS) so the policy and pose reads are live.
        for _ in 0..64 {
            app.update();
        }
        app
    }

    /// Emulate a long chase's walk: move the whole crab without touching the slot.
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

    /// rl#290, one-frame form: a fresh app's FIRST origins land AT the sim spawns —
    /// absolute coordinates, no gauge — and the crab body spawns on them.
    #[test]
    fn fresh_app_lays_origins_on_the_sim_spawns_absolutely() {
        let spawn = Pos::from_meters(30.0, -14.0);
        let mut app = one_world_app(&[spawn]);
        let origin = app.world().resource::<CrabSpawns>().origin(0);
        assert!(
            (Vec2::new(origin.x, origin.z) - pos_to_m(spawn)).length() < 1e-3,
            "the origin must sit AT the sim spawn (one frame), got {origin:?}"
        );
        assert!(
            (carapace_xz(&mut app) - pos_to_m(spawn)).length() < 1.0,
            "the crab body must spawn on its sim spawn"
        );
    }

    /// The slot reads her sim pose OFF the body — and her claw capsules ride along
    /// (rl#249: the sim's down check uses her real physics claws, no second hitbox).
    #[test]
    fn collected_pose_is_the_carapace_and_carries_real_claws() {
        let spawn = Pos::from_meters(30.0, -14.0);
        let mut app = one_world_app(&[spawn]);
        let held = vec![CrabPose {
            pos: spawn,
            yaw: 0,
            claws: Vec::new(),
        }];
        let poses = collect_crab_poses(app.world_mut(), &held);
        let cara = carapace_xz(&mut app);
        assert_eq!(poses.len(), 1);
        assert!(
            (pos_to_m(poses[0].pos) - cara).length() < 0.02,
            "the collected sim pose must be the carapace's world point, got {:?} vs {cara:?}",
            pos_to_m(poses[0].pos)
        );
        assert!(
            !poses[0].claws.is_empty(),
            "the collected pose must carry her claw capsules"
        );
        for claw in &poses[0].claws {
            let (ax, az) = claw.a.to_meters();
            assert!(
                (Vec2::new(ax, az) - cara).length() < 2.0,
                "claw capsules cross in world frame, near her body"
            );
        }
    }

    /// rl#240, one-frame form: a past-the-band walk rebases the env's ORIGIN to her
    /// ground point and NOTHING physical moves (no teleport; the crab stays where it
    /// walked). Sim-state assertions only — since rl#311 the obs carries no position
    /// channel for the rebase to show up in.
    #[test]
    fn recenter_rebases_origin_and_teleports_nothing() {
        let spawn = Pos::from_meters(0.0, 0.0);
        let mut app = one_world_app(&[spawn]);

        shift_parts(&mut app, Vec3::new(20.0, 0.0, 0.0));
        for _ in 0..2 {
            app.update();
        }

        assert!(
            carapace_xz(&mut app).x > 15.0,
            "the crab must stay where it walked — a rebase teleports nothing"
        );
        let origin = app.world().resource::<CrabSpawns>().origin(0);
        assert!(
            (Vec2::new(origin.x, origin.z) - carapace_xz(&mut app)).length() < 1.0,
            "the origin must have rebased to her ground point, got {origin:?}"
        );
    }

    /// Multi-crab: only the drifted env's origin rebases (origins are per-env state).
    #[test]
    fn multi_crab_rounds_recenter_only_the_drifted_env() {
        let spawns = [Pos::from_meters(0.0, 0.0), Pos::from_meters(8.0, 0.0)];
        let mut app = one_world_app(&spawns);
        let origin0_before = app.world().resource::<CrabSpawns>().origin(0);
        let origin1_before = app.world().resource::<CrabSpawns>().origin(1);

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

        let origin1 = app.world().resource::<CrabSpawns>().origin(1);
        assert!(
            origin1.x - origin1_before.x > 15.0,
            "env 1's origin must have rebased to its walked-out crab, got {origin1:?}"
        );
        assert_eq!(
            app.world().resource::<CrabSpawns>().origin(0),
            origin0_before,
            "the standing env's origin must not move"
        );
    }

    /// rl#289, one-frame form: a round RESTART re-pins EVERY env's origin AT the fresh
    /// sim spawns (absolute) and cold-respawns each crab onto its origin (rl#242).
    #[test]
    fn restart_repins_origins_to_the_new_sim_spawns() {
        let spawns = [Pos::from_meters(0.0, 0.0), Pos::from_meters(8.0, 0.0)];
        let mut app = one_world_app(&spawns);

        let fresh = [Pos::from_meters(3.0, -2.0), Pos::from_meters(9.0, 4.0)];
        restart_crabs_to_spawns(app.world_mut(), &fresh);
        app.update();

        for (env, want) in fresh.iter().enumerate() {
            let origin = app.world().resource::<CrabSpawns>().origin(env);
            assert!(
                (Vec2::new(origin.x, origin.z) - pos_to_m(*want)).length() < 1e-3,
                "env {env}'s origin must re-pin AT its new sim spawn, got {origin:?}"
            );
        }
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>();
        let carapaces: Vec<(usize, Vec3)> = q
            .iter(app.world())
            .map(|(env, t)| (env.0, t.translation))
            .collect();
        assert_eq!(carapaces.len(), 2);
        for (env, c) in carapaces {
            let want = pos_to_m(fresh[env]);
            assert!(
                (Vec2::new(c.x, c.z) - want).length() < 1.0,
                "env {env}'s crab must respawn on its re-pinned origin, carapace {c:?}"
            );
        }
    }

    /// A prey inside the training band poses WHERE IT IS ([`band_lure`] presents an
    /// in-band target unmoved), lifted to [`CLAW_TARGET_Y`] above the local surface.
    #[test]
    fn walk_target_poses_at_the_prey() {
        let mut app = one_world_app(&[Pos::from_meters(0.0, 0.0)]);
        let prey = Pos::from_meters(5.0, -3.0);
        feed_hunt(app.world_mut(), &[Some(prey)], &[]);
        app.update();
        let posed = app.world().resource::<CrabTargets>().envs[0].expect("target posed");
        let want = pos_to_m(prey);
        assert!(
            (Vec2::new(posed.x, posed.z) - want).length() < 1e-3,
            "the target must pose AT the prey {want:?}, got {posed:?}"
        );
        assert!(
            (posed.y - CLAW_TARGET_Y).abs() < 1e-3,
            "flat grid: target y is CLAW_TARGET_Y above 0, got {}",
            posed.y
        );

        feed_hunt(app.world_mut(), &[None], &[]);
        app.update();
        assert!(
            app.world().resource::<CrabTargets>().envs[0].is_none(),
            "no prey ⇒ no posed target"
        );
    }

    /// rl#301: a piloted prey's shadow can leave the training band — the posed
    /// target is lured to at most one band edge from the carapace along the true
    /// bearing, never presented outside training support.
    #[test]
    fn walk_target_lures_far_prey_to_the_band_edge() {
        use crab_world::training::targets::BAND_MAX_M;
        let mut app = one_world_app(&[Pos::from_meters(0.0, 0.0)]);
        let prey = Pos::from_meters(400.0, -300.0); // 500 m out — far past the band
        feed_hunt(app.world_mut(), &[Some(prey)], &[]);
        app.update();
        let posed = app.world().resource::<CrabTargets>().envs[0].expect("target posed");
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        let carapace = q.iter(app.world()).next().expect("carapace").translation;
        let to_posed = Vec2::new(posed.x - carapace.x, posed.z - carapace.z);
        let to_prey = pos_to_m(prey) - Vec2::new(carapace.x, carapace.z);
        assert!(
            (to_posed.length() - BAND_MAX_M).abs() < 1e-2,
            "far prey must pose AT the band edge, got {} m from the carapace",
            to_posed.length()
        );
        assert!(
            to_posed.normalize().dot(to_prey.normalize()) > 0.9999,
            "the lure must hold the true bearing to the prey"
        );
        assert!(
            (posed.y - CLAW_TARGET_Y).abs() < 1e-3,
            "flat grid: the lured point's y is CLAW_TARGET_Y above 0, got {}",
            posed.y
        );
    }

    /// rl#281 stage-6 catch, kept through stage 5: the posed target's y must ride the
    /// SURFACE at the target's own xz — an absolute y feeds `target:y` an obs off by
    /// the full local elevation, and nothing else measures it.
    #[test]
    fn walk_target_y_rides_the_surface_on_terrain() {
        pin_single_thread_pools();
        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            grid: crab_world::terrain::TerrainGrid::gcr(),
            visuals: crab_world::Visuals(false),
        });
        let spawn = Pos::from_meters(0.0, 0.0);
        app.add_plugins(NnCrabPlugin::new(vec![Policy::rest()], vec![spawn]));
        arm(app.world_mut());
        restart_crabs_to_spawns(app.world_mut(), &[spawn]);
        force_serial_schedules(&mut app);
        for _ in 0..4 {
            app.update();
        }

        // A prey spot with unambiguous elevation (the tile center's datum is ~0,
        // which couldn't distinguish surface-relative from absolute).
        let terrain = app
            .world()
            .resource::<crab_world::terrain::Terrain>()
            .clone();
        let spot = (1..40)
            .flat_map(|r| [(r as f32 * 100.0, 0.0), (0.0, r as f32 * 100.0)])
            .map(|(x, z)| Vec2::new(x, z))
            .find(|p| terrain.height(p.x, p.y).abs() > 20.0)
            .expect("the seed-281 tile has >20 m relief within 4 km of center");
        feed_hunt(
            app.world_mut(),
            &[Some(Pos::from_meters(spot.x, spot.y))],
            &[],
        );
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
    /// query aliases through both paths that create GCR crab bodies: the initial armed
    /// spawn and the round-boundary cold respawn.
    #[cfg(feature = "render")]
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

        let mut app = one_world_app(&[Pos::from_meters(0.0, 0.0)]);
        assert_cage_covers_all(&mut app, "initial armed spawn");

        cold_respawn_armed_crab(app.world_mut());
        assert_cage_covers_all(&mut app, "after a round-boundary cold respawn");
    }
}

#[cfg(test)]
mod tests {
    use crab_world::bot::actuator::{ACTION_SIZE, CrabActions};
    use crab_world::bot::body::CrabJoint;
    use crab_world::bot::headless::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_server_world, headless_stack,
        pin_single_thread_pools,
    };
    use crab_world::bot::physics_digest::crab_state_digest;
    use crab_world::bot::sensor::CrabObservation;
    use crab_world::policy::RestFallback;

    use super::*;
    use crate::client::TickMsg;
    use crate::server::Server;
    use crate::sim::{Input, PlayerId};

    /// The park is EXACT (rl#298 stage 4): a marathon host accumulates unbounded
    /// wall-clock into `Time<Fixed>`'s overstep, and past the 86400 s timestep the
    /// auto-pump fires one rogue un-pumped action+physics step. The discard half must
    /// keep the accumulator from ever reaching the timestep; the control app (the old
    /// timestep-only park) proves this test would catch the rogue step.
    #[test]
    fn parked_auto_pump_never_fires_even_past_a_day_of_uptime() {
        use std::time::Duration;

        use bevy::time::{Fixed, Time, TimeUpdateStrategy, Virtual};

        #[derive(Resource, Default)]
        struct FixedRuns(u32);

        let build = |exact_park: bool| {
            let mut app = App::new();
            app.add_plugins(MinimalPlugins);
            app.init_resource::<FixedRuns>()
                .add_systems(FixedUpdate, |mut runs: ResMut<FixedRuns>| runs.0 += 1);
            // Uncapped virtual time so 25 h of uptime is a few updates, not 350k
            // real-delta frames under the 250 ms clamp.
            app.world_mut()
                .resource_mut::<Time<Virtual>>()
                .set_max_delta(Duration::MAX);
            if exact_park {
                park_fixed_auto_pump(&mut app);
            } else {
                // The pre-stage-4 park: timestep only, accumulator left running.
                app.world_mut()
                    .resource_mut::<Time<Fixed>>()
                    .set_timestep(Duration::from_secs(86_400));
            }
            app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs(
                30_000,
            )));
            app
        };

        let mut control = build(false);
        for _ in 0..5 {
            control.update();
        }
        assert!(
            control.world().resource::<FixedRuns>().0 > 0,
            "teeth: 120000 s of uptime must overflow the bare 86400 s park — if it \
             doesn't, this test can no longer detect the rogue step"
        );

        let mut parked = build(true);
        for _ in 0..5 {
            parked.update();
        }
        assert_eq!(
            parked.world().resource::<FixedRuns>().0,
            0,
            "the exact park must never let the auto-pump fire"
        );
        let overstep = parked.world().resource::<Time<Fixed>>().overstep();
        assert!(
            overstep <= Duration::from_secs(30_000),
            "post-frame residual must be at most one frame's delta, got {overstep:?}"
        );
    }

    #[test]
    fn manual_pump_matches_auto_pump_step_for_step() {
        let build = || {
            headless_stack(HeadlessStack {
                num_envs: 1,
                role: WorldRole::Standalone,
                // Models the GCR client's world, so it steps the client's ground — the
                // canonical terrain bake (rl#209, rl#293).
                grid: crab_world::terrain::TerrainGrid::gcr(),
                visuals: crab_world::Visuals(false),
            })
        };
        let mut auto = build();
        let mut manual = build();
        auto.update();
        manual.update();
        park_fixed_auto_pump(&mut manual);

        let digest = |app: &mut App| -> u64 {
            let mut q = app.world_mut().query_filtered::<(
                &Transform,
                &Velocity,
                Option<&CrabJoint>,
                Option<&CrabCarapace>,
            ), With<CrabBodyPart>>();
            crab_state_digest(q.iter(app.world()))
        };
        let set_torque = |app: &mut App, a: [f32; ACTION_SIZE]| {
            assert!(app.world_mut().resource_mut::<CrabActions>().set_row(0, a));
        };

        let mut lcg: u64 = 0x1234_5678_9abc_def0;
        for t in 0..120u32 {
            let mut act = [0.0f32; ACTION_SIZE];
            for slot in act.iter_mut() {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *slot = ((lcg >> 40) as u32 as f32 / (1u32 << 24) as f32) * 1.6 - 0.8;
            }
            set_torque(&mut auto, act);
            set_torque(&mut manual, act);
            auto.update();
            pump_fixed_steps(manual.world_mut(), 1);
            assert_eq!(
                digest(&mut auto),
                digest(&mut manual),
                "manual pump diverged from auto-pump at tick {t}"
            );
        }
    }

    /// A headless host in the LIVE configuration: the server world
    /// ([`headless_server_world`] — the same constructor the trainer's rollout env
    /// builds on, rl#298 stage 4) with the crab stack armed, the wall-clock auto-pump
    /// parked from birth (like the windowed app), so physics advances only through
    /// the slot — no renderer anywhere. The flat grid keeps the motion assertions
    /// attributable to the drive, not to sliding down a GCR slope — spanning the GCR
    /// tile, so wherever the sim's random spawn frame (rl#305) lands the crab, she
    /// lands ON the collider instead of falling off a 512 m test slab.
    fn headless_host_app(policy: Policy, crab_spawn: Pos) -> App {
        pin_single_thread_pools();
        let gcr = crab_world::terrain::TerrainGrid::gcr();
        let half = gcr.extent_x().max(gcr.extent_z()) / 2.0;
        let mut app = headless_server_world(
            1,
            WorldRole::Standalone,
            std::sync::Arc::new(crab_world::terrain::TerrainGrid::flat(half)),
        );
        app.add_plugins(NnCrabPlugin::new(vec![policy], vec![crab_spawn]));
        arm(app.world_mut());
        park_fixed_auto_pump(&mut app);
        restart_crabs_to_spawns(app.world_mut(), &[crab_spawn]);
        force_serial_schedules(&mut app);
        app
    }

    /// One authoritative server tick through the REAL host path: `advance` → the crab
    /// slot → `step_next`. `app.update()` first, as on the live host, where the
    /// `Update` schedule (crab spawn, bookkeeping) wraps the slot every frame.
    fn server_tick(
        app: &mut App,
        server: &mut Server,
        issue_tick: u64,
    ) -> crate::snapshot::CoreSnapshot {
        app.update();
        server.advance(TickMsg {
            issue_tick,
            input: Input::from_axes(0.0, 0.0),
            pilot: None,
        });
        let mut last = None;
        while server.next_tick_ready() {
            let inputs = slot_inputs(server.sim());
            let poses = pump_crab_slot(app.world_mut(), &inputs);
            let stepped = server.step_next(&poses, Default::default());
            last = Some(
                crate::snapshot::CoreSnapshot::from_bytes(&stepped.snapshot)
                    .expect("the authoritative server's snapshot must decode"),
            );
        }
        last.expect("one assembled tick per advance on the solo host")
    }

    fn solo_server() -> Server {
        let me = PlayerId(0);
        Server::new(me, &[me], Sim::new(0x5A11, &[me]))
    }

    /// A solo authoritative server plus the headless host world armed with `policy`,
    /// crab at the sim's own spawn — one frame, so the world MUST spawn her there.
    fn solo_host(policy: Policy) -> (Server, App) {
        let server = solo_server();
        let spawn = server.sim().crabs()[0].pos();
        let app = headless_host_app(policy, spawn);
        (server, app)
    }

    /// rl#298 stage 2, link 1: the policy FORWARD runs inside the server's crab slot.
    /// After a slot pump, the actions driving the in-world body's motors are exactly
    /// the loaded brain's forward pass over the obs the sensor built that same pump —
    /// pinned by recomputing `act` on the observed row, so a Rest shortcut or a
    /// stale/overwritten row cannot pass.
    #[test]
    fn host_slot_runs_the_policy_forward_between_advance_and_step_next() {
        // The enveloped golden brain (crab-world's format-drift fixture): a real
        // checkpoint the loader arms, so `act` runs a real NN forward.
        let golden = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crab-world/tests/data/golden-mlp512x3-env");
        let policy = Policy::load(&golden, RestFallback::Rest);
        assert!(
            policy.is_loaded(),
            "the golden checkpoint must load — a Rest fallback would make this test vacuous"
        );

        let (mut server, mut app) = solo_host(policy);

        // Past spawn + settle grace (32 physics ticks), into live sensing/acting.
        for t in 0..60u64 {
            server_tick(&mut app, &mut server, t);
        }

        let obs = app.world().resource::<CrabObservation>().rows()[0];
        assert!(
            obs.iter().any(|v| *v != 0.0),
            "the sensor must have built a LIVE obs this pump — on the defaulted all-zero \
             row the forward-pass pin below would hold vacuously"
        );
        let expected = app.world().non_send_resource::<CrabPolicies>().0[0].act(&obs);
        let got = app.world().resource::<CrabActions>().rows()[0];
        assert_eq!(
            got, expected,
            "the motors' action row must be the policy forward over this pump's obs"
        );
    }

    /// rl#298 stage 2, link 2: motion the slot pumps into the in-world body lands in
    /// the authoritative snapshot clients decode. A full-scale square wave (the rl#224
    /// flail, driven AFTER the policy so the Rest brain doesn't zero it) shuffles the
    /// crab; the moved position must come back out of `step_next`'s `CoreSnapshot`.
    #[test]
    fn slot_pumped_crab_motion_reaches_the_snapshot_clients_decode() {
        #[derive(Resource, Default)]
        struct Wave(f32);
        fn drive_wave(w: Res<Wave>, mut actions: ResMut<CrabActions>) {
            // Every-tick system: pre-spawn skips are fine, later ticks land the flail.
            let _ = actions.fill(0, w.0);
        }

        let (mut server, mut app) = solo_host(Policy::rest());
        let spawn = server.sim().crabs()[0].pos();
        app.init_resource::<Wave>();
        // After Think (the policy's set), before Act (apply_actions): the wave is the
        // last CrabActions writer of the pump, same guarantee as ordering directly
        // against the private run_crab_policy.
        app.add_systems(
            FixedUpdate,
            drive_wave.after(BotSet::Think).before(BotSet::Act),
        );

        let mut tick = 0u64;
        let mut snap = None;
        for _ in 0..100u64 {
            snap = Some(server_tick(&mut app, &mut server, tick));
            tick += 1;
        }
        for t in 0..300u64 {
            app.world_mut().resource_mut::<Wave>().0 = if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
            snap = Some(server_tick(&mut app, &mut server, tick));
            tick += 1;
        }

        let snap = snap.expect("ticks ran");
        let end = snap.crabs[0].pos();
        let (dx, dz) = Pos {
            x: end.x - spawn.x,
            z: end.z - spawn.z,
        }
        .to_meters();
        let moved = (dx * dx + dz * dz).sqrt();
        assert!(
            moved > 0.5,
            "the flailing in-world crab must have moved in the snapshot clients decode \
             (moved {moved:.3} m)"
        );
    }

    /// rl#298 stage 3, obs half — stage-5 (one-frame) form: the row the policy
    /// consumes IS the one world's same-tick state. A snapshot captured between Sense
    /// and Think pins the target channel against the posed lure in the live body frame
    /// — and the lure re-derives from the SIM's own prey DIRECTLY (world = sim frame;
    /// the bridge correspondence this test used to thread through is deleted). The
    /// crab is kept in motion (the rl#224 flail) so the body moves between passes:
    /// `target_local` folds in the live carapace pose, so a row stale by even one step
    /// exceeds its tolerance, and a defaulted or wrong-frame row misses by meters.
    #[test]
    fn obs_seam_reads_the_one_worlds_state_with_the_sims_prey_as_target() {
        #[derive(Resource, Default)]
        struct SenseSnap {
            carapace: Option<(Vec3, Quat)>,
            prev_carapace: Option<Vec3>,
            target: Option<Vec3>,
        }
        fn snap_sense(
            mut snap: ResMut<SenseSnap>,
            targets: Res<CrabTargets>,
            carapace_q: Query<&Transform, With<CrabCarapace>>,
        ) {
            let Some(t) = carapace_q.iter().next() else {
                return;
            };
            snap.prev_carapace = snap.carapace.map(|(p, _)| p);
            snap.carapace = Some((t.translation, t.rotation));
            snap.target = targets.get(0);
        }
        #[derive(Resource, Default)]
        struct Wave(f32);
        fn drive_wave(w: Res<Wave>, mut actions: ResMut<CrabActions>) {
            let _ = actions.fill(0, w.0);
        }

        let (mut server, mut app) = solo_host(Policy::rest());
        app.init_resource::<SenseSnap>();
        app.init_resource::<Wave>();
        app.add_systems(
            FixedUpdate,
            (
                snap_sense.after(BotSet::Sense).before(BotSet::Think),
                drive_wave.after(BotSet::Think).before(BotSet::Act),
            ),
        );

        // Past spawn + settle grace, into live sensing with a fed hunt target. The
        // flail starts only post-settle so the settle ragdoll isn't perturbed.
        for t in 0..60u64 {
            if t >= 20 {
                app.world_mut().resource_mut::<Wave>().0 =
                    if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
            }
            server_tick(&mut app, &mut server, t);
        }

        let (pos, rot, prev, target) = {
            let snap = app.world().resource::<SenseSnap>();
            let (pos, rot) = snap.carapace.expect("carapace snapshotted post-settle");
            let target = snap
                .target
                .expect("the sim's living player must have posed a hunt target");
            let prev = snap.prev_carapace.expect("two sensed passes ran");
            (pos, rot, prev, target)
        };
        // The flailing body really moves between passes — what gives the body pin
        // below its anti-staleness teeth (inter-pass motion ≫ its tolerance).
        assert!(
            (pos - prev).length() > 1e-4,
            "the flail must move the carapace between sensed passes \
             (moved {:.6} m)",
            (pos - prev).length()
        );
        let obs = app.world().resource::<CrabObservation>();
        let view = obs.env(0).expect("env 0 sized");

        let target_local = view.target_local();
        let expected_local = rot.inverse() * (target - pos);
        assert!(
            (target_local - expected_local).length() < 1e-4,
            "obs target_local {target_local:?} must be the posed target in the live \
             body frame ({expected_local:?})"
        );

        // The seam crossing, one frame: the posed target IS the sim's prey position,
        // surface-lifted — no correspondence to thread through.
        let prey = server
            .sim()
            .nearest_living_player_pos(0)
            .expect("player 0 is alive and stationary");
        let (px, pz) = prey.to_meters();
        let terrain = app.world().resource::<crab_world::terrain::Terrain>();
        let expected_target = terrain.place(bevy::math::Vec2::new(px, pz), CLAW_TARGET_Y);
        assert!(
            (target - expected_target).length() < 1e-3,
            "the posed target {target:?} must be the sim's prey, surface-lifted \
             ({expected_target:?})"
        );
    }

    /// rl#298 stage 3, control-rate half: equality with training's control period is
    /// by shared construction (module doc); this pins the host half — across a real
    /// server run, every physics step the 64:30 cadence owes writes the crab's
    /// actions exactly once (no decimation, no extra passes), so the action period
    /// is one physics step (`PHYSICS_DT`), training's control period. Counted by
    /// change-detection on `CrabActions` in the window where `run_crab_policy` is
    /// its only writer — a disarmed or decimated policy system fails here where a
    /// bare pass counter would stay green. If the in-game step rate ever exceeds
    /// the control rate, action repeat must restore this equality (the epic's
    /// parenthetical).
    #[test]
    fn in_game_action_period_equals_trainings_control_period() {
        #[derive(Resource, Default)]
        struct ActionWrites(u64);
        fn count_action_writes(actions: Res<CrabActions>, mut writes: ResMut<ActionWrites>) {
            writes.0 += u64::from(actions.is_changed());
        }

        let (mut server, mut app) = solo_host(Policy::rest());
        app.init_resource::<ActionWrites>();
        app.add_systems(
            FixedUpdate,
            count_action_writes.after(BotSet::Think).before(BotSet::Act),
        );

        for t in 0..90u64 {
            server_tick(&mut app, &mut server, t);
        }

        let ticks = server.sim().tick();
        assert_eq!(
            ticks, 90,
            "one authoritative tick per advance on the solo host"
        );
        let writes = app.world().resource::<ActionWrites>().0;
        assert_eq!(
            writes,
            crate::cadence::cumulative_steps(ticks),
            "every pumped physics step must write the crab's actions exactly once — \
             the action period is one physics step (PHYSICS_DT), training's control \
             period"
        );
    }
}
