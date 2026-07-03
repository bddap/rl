//! A real rapier-simulated, NN-driven giant crab — the trained RL body ("real Sally"), the ONLY
//! crab the game has (no integer point-pursuer to fall back to). Armed by the
//! [`ExternalCrabArmed`] runtime gate on a SOLO round (a Host-alone Start, or scripted `--host`
//! that found no peer) and on a NETWORKED round with synced weights.
//!
//! # When it drives the crab (and why it's safe)
//! The game's [`crate::sim`] is a bit-deterministic INTEGER lockstep sim so two peers
//! evolve identically. A rapier crab is FLOAT physics, so it can drive a *networked* crab
//! without desyncing ONLY when every peer (a) loaded the same brain (the weights-digest
//! handshake) and (b) steps the body the identical number of physics steps per lockstep tick
//! (the deterministic [`crate::cadence::PhysicsCadence`], pumped by
//! `net::render::drive_lockstep`). [`crate::may_arm_external_crab`] is the gate that
//! enforces both. On a solo round there's no peer to desync against, so it always arms. A
//! networked-UNSYNCED round CANNOT arm and — with no integer fallback — REFUSES LOUDLY (the
//! windowed client fails with an actionable peer-mismatch message; see [`crate::render`])
//! rather than silently substituting a fake crab for Sally.
//!
//! # How it works
//! The trained policy ([`crab_world::play::Policy`], loaded from a checkpoint) is a
//! *locomotion+reach* brain: its observation includes the touch target as a vector in
//! the carapace's local frame, and it has learned to WALK the body toward that vector
//! (the `ckpt-best.locomotion` weights walk + reach). We exploit that directly:
//!
//! - The crab lives in its OWN rapier world (the shared [`crab_world::bot`] /
//!   [`crab_world::physics`] stack) on the OPEN inference field
//!   ([`crab_world::physics::Arena::OpenField`], rl#209): an unbounded ground with no
//!   walls, so its per-round travel is unlimited. Training keeps its ±10 m walled box;
//!   the ground contact dynamics are identical (same y=0 surface, same material).
//! - Each control step we place its target at the nearest living game player's ACTUAL
//!   position, expressed in the crab's arena frame (`carapace + (player − crab)`), at any
//!   distance — no lead, no treadmill. The observation re-expresses that true offset in
//!   carapace-local axes, so the policy sees the real player offset and the WEIGHTS supply
//!   the approach (the bitter lesson — the approach logic is learned, not hand-coded).
//!   Training samples targets across the arena band (`[1.5, ~9] m`); a player farther
//!   than that (they spawn ≥12 m away — `sim::MIN_CRAB_SPAWN_DISTANCE`) is OUT of the trained
//!   band, but the obs normalizer clamps `target_local` to ±5σ, so a far offset saturates to
//!   "player is that way, far" and the crab walks full-tilt toward it — honest hunt heading,
//!   without resolving distance past the band edge — and with the open field it keeps
//!   walking until the gap re-enters the band. (The observation's spawn-relative `body.pos`
//!   slots saturate the same clamp once the crab is far from its arena spawn — a regime
//!   training never sampled; heading + gait slots stay in-distribution, and probe runs
//!   ~2× past the old wall still close the gap, but very long chases are less charted.)
//! - We integrate the carapace's horizontal displacement each game tick and add it to
//!   the game-world crab position, which we write back into the sim with
//!   [`Sim::set_external_crab_pose`]. Grabs / extraction / win-loss then resolve against the
//!   REAL crab body, exactly as the prompt asks.
//! - For rendering, the rig is carried to the game-world crab spot by a RENDER-ONLY repose
//!   published into [`crab_world::bot::skin::CrabSkinRepose`] each game tick and applied to the
//!   render BONES only; physics `Transform`s are never touched, and the giant FEEL comes from
//!   the R-shrunk human world — see [`publish_skin_repose`].

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::sim::Pos;
use crab_world::Visuals;
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};
use crab_world::bot::sensor::CrabTargets;
use crab_world::bot::{BotSet, CrabSpawns};
use crab_world::play::Policy;

/// Height (m) of the walk target placed at the player's position. A low ground-ish Y inside
/// the policy's trained target-height band (`TARGET_Y_MIN..MAX` ≈ 0.15..0.7 in training) so
/// the crab reads it as a walk-up-and-reach target. The hunt target is a planar (XZ) player
/// position with no carried height, so we pin a single ground-level grab height here.
const CLAW_TARGET_Y: f32 = 0.3;

/// Resource: the live bridge state between the rapier NN crab and the integer game sim.
/// Non-trivial state (a float world position + the last carapace sample) that must
/// persist across ticks, so it can't live on the `Copy` integer `Sim`.
#[derive(Resource)]
pub struct ExternalCrabBridge {
    /// The crab's accumulated game-world ground position, in metres (float; converted to
    /// the sim's [`UNIT`] fixed-point when written back). Seeded from the sim's integer
    /// crab spawn so the NN crab starts where the round placed it.
    world_pos_m: Vec2,
    /// Carapace horizontal position (arena frame, metres) sampled last game tick, to
    /// difference against this tick for the displacement we add to `world_pos_m`. `None`
    /// until the first sample (the crab spawns a frame after Startup).
    last_carapace_m: Option<Vec2>,
    /// Current facing yaw in the sim's [`crate::sim::trig`] turn units, from the
    /// crab's walk direction — for the rendered crab + any facing-dependent logic.
    yaw_turns: i32,
    /// Ticks left in the spawn SETTLE window: while > 0 the policy is held off and the
    /// crab holds zero torque so it DROPS from its spawn height and plants on its legs
    /// first — the same grace the demo/training give a fresh crab ([`RESET_GRACE_TICKS`]
    /// via [`settle_countdown`]). Driving the policy before the body has stood produces a
    /// flailing/locked pose that never walks (observed). Counts down to 0 = policy in
    /// control.
    settle: u32,
    /// Game-world point (metres) the crab is currently hunting — the nearest living
    /// player, refreshed each tick by [`drive_external_crab`]. `None` when every player is
    /// down (the crab then holds position). Read by [`set_crab_walk_target`] to aim the
    /// policy.
    hunt_target_m: Option<Vec2>,
    /// This tick's digest of the crab's full rapier physics state XORed with the
    /// policy-weights digest — recomputed each step by [`hash_crab_physics`] (see its doc).
    /// `0` until the first post-step hash.
    phys_digest: u64,
}

/// The sim's fixed-point [`Pos`] (XZ) as game-world metres on the bridge's `Vec2` frame
/// (sim `x`→`Vec2.x`, sim `z`→`Vec2.y`) — just [`Pos::to_meters`] (the one conversion
/// rule) repackaged onto `Vec2`.
fn pos_to_m(p: Pos) -> Vec2 {
    let (x, z) = p.to_meters();
    Vec2::new(x, z)
}

/// Where the crab rig sits in the game world — the 2D (XZ) precursor that [`publish_skin_repose`]
/// combines with the render-frame scale ([`crate::render::world_render_scale`]) to build the full
/// 3D [`crab_world::bot::skin::SkinRepose`]. A named pair (not a bare tuple) because
/// the two `Vec2`s are different KINDS — `shift` is an XZ delta, `pivot` an XZ world point — so
/// they can't be swapped at the use site. See [`ExternalCrabBridge::render_placement_m`].
struct CrabPlacement {
    /// Arena→game-world XZ translation for each part (`world_pos_m - last_carapace_m`).
    shift: Vec2,
    /// The crab's game-world ground point (XZ).
    pivot: Vec2,
}

impl ExternalCrabBridge {
    /// Seed the bridge at the sim's crab spawn (so the NN crab begins where the
    /// round placed the giant crab, MIN_CRAB_SPAWN_DISTANCE from the players).
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

    /// The crab's game-world position as the sim's fixed-point [`Pos`].
    pub fn world_pos(&self) -> Pos {
        // `world_pos_m` accumulates bounded per-tick steps, so any real session stays well
        // inside these conservative game-world bounds. A non-finite or wildly out-of-range
        // value means the integrator smeared a NaN / blow-up in upstream, and the
        // `as f32 → as i64` cast below would saturate silently into a bogus pose. Catch that
        // invariant break in debug; release keeps the bare cast unchanged (no clamp), so
        // behaviour is byte-identical.
        debug_assert!(
            self.world_pos_m.is_finite()
                && self.world_pos_m.x.abs() <= 100_000.0
                && self.world_pos_m.y.abs() <= 100_000.0,
            "external crab world_pos_m out of live bounds: {:?}",
            self.world_pos_m
        );
        Pos::from_meters(self.world_pos_m.x, self.world_pos_m.y)
    }

    /// Where the crab rig sits in the game world, as a `(shift, pivot)` pair in metres on the
    /// XZ `Vec2` frame:
    /// - `shift` translates each part's raw arena `Transform` to the game-world crab spot
    ///   (`world_pos_m - last_carapace_m`), and
    /// - `pivot` is the crab's game-world ground point (`world_pos_m`).
    ///
    /// Feeds the render-only repose — see [`publish_skin_repose`] for why placement never
    /// touches physics. `None` until the crab has been sampled once (so we never place against
    /// a stale/zero carapace).
    fn render_placement_m(&self) -> Option<CrabPlacement> {
        self.last_carapace_m.map(|c| CrabPlacement {
            shift: self.world_pos_m - c,
            pivot: self.world_pos_m,
        })
    }

    pub fn yaw_turns(&self) -> i32 {
        self.yaw_turns
    }

    /// This tick's full rapier physics digest (every actuated body's pose+velocity bits XORed with
    /// the policy weights — see [`hash_crab_physics`]), folded into the sim's `external_crab_digest`.
    /// The authoritative server reads it here (alongside [`world_pos`](Self::world_pos) /
    /// [`yaw_turns`](Self::yaw_turns)) to build the tick's [`crate::server::CrabPose`].
    pub fn phys_digest(&self) -> u64 {
        self.phys_digest
    }

    /// Set the game-world point (the nearest living player's [`Pos`]) the crab hunts, or
    /// `None` when every player is down. Called each game tick by the render loop, which
    /// owns the sim and so knows the prey; [`set_crab_walk_target`] reads it to aim the
    /// policy.
    pub fn set_hunt_target(&mut self, prey: Option<Pos>) {
        self.hunt_target_m = prey.map(pos_to_m);
    }

    /// Re-seed the bridge to the round's spawn after a deterministic sim RESTART
    /// ([`buttons::RESTART`]). The sim's [`reset`](crate::sim::Sim::reset) rebuilds the
    /// crab back AT spawn, but the float body keeps walking and the bridge keeps its
    /// accumulated `world_pos_m`; without this the next pose push would snap the
    /// freshly-restarted crab onto the still-walking body's old position — mid-gait at the
    /// wrong place, not at the computed spawn. So move the game-world position back to the
    /// integer spawn and forget the pre-restart carapace sample (re-seeded from the fresh pose
    /// next frame, exactly as the [`CrabRescued`](crab_world::bot::CrabRescued) path does — without
    /// it the first post-restart accumulation would difference the spawn against the old pose
    /// and inject a multi-metre false step). Re-settle too, so the round opens with the spawn
    /// drop/plant grace.
    ///
    /// DETERMINISM: this fires off the sim's restart EDGE (returned by the authoritative
    /// step — a restart never rewinds the tick, rl#204) — the same edge the cadence reset
    /// hangs off in [`crate::render`]'s `drive_lockstep`. Under host-authority only the
    /// server-auth peer steps (and so observes the edge); a remote client renders the host's
    /// articulation — its own armed body is never pumped, so there is nothing to re-seed.
    /// `spawn` is read
    /// back from the post-restart sim, which is itself deterministic. Unlike
    /// `CrabRescued` (a float-body teleport that leaves the game position put) this DOES move
    /// `world_pos_m`, because a restart moves the crab back to spawn.
    pub fn restart_to_spawn(&mut self, spawn: Pos) {
        self.world_pos_m = pos_to_m(spawn);
        self.last_carapace_m = None;
        self.settle = crab_world::bot::RESET_GRACE_TICKS;
    }
}

/// Cold-respawn the armed crab's rapier body at the round-boundary RESTART edge — despawn the
/// env-0 crab entity tree and rebuild it fresh from spawn, via the SAME
/// [`crab_world::bot::respawn_crab`] path the non-finite rescue uses (one respawn implementation,
/// no parallel reset to drift). [`ExternalCrabBridge::restart_to_spawn`] resets only the bridge's
/// bookkeeping; the rapier solver keeps WARM contact/warm-start state across a bare restart, so a
/// mid-game JOIN would put a warm incumbent body beside a cold joiner body in the same round — a
/// divergence folded LOUDLY into `state_hash` via the external-crab digest. Dropping the tree and
/// rebuilding gives every peer (incumbents AND the joiner) the IDENTICAL cold solver state, so the
/// round stays in lockstep. Called from the one restart edge in [`crate::render`]'s
/// `drive_lockstep`, so join and plain-restart share it.
///
/// Exclusive-world (the driver edge holds `&mut World`): collect the env-0 parts, lift `CrabAssets`
/// out with `resource_scope` so a temporary [`Commands`] can borrow the world, and apply the
/// despawn+spawn immediately — before the next `pump_fixed_steps`, so the first post-restart physics
/// step lands on the fresh body. The spawn origin is the env's recorded [`CrabSpawns`] entry (the
/// same point the initial spawn used), keeping the cold body bit-identical to a from-boot spawn.
pub(crate) fn cold_respawn_armed_crab(world: &mut World) {
    use bevy::ecs::world::CommandQueue;

    let parts: Vec<Entity> = world
        .query_filtered::<(Entity, &CrabEnvId), With<CrabBodyPart>>()
        .iter(world)
        .filter(|(_, env)| env.0 == 0)
        .map(|(e, _)| e)
        .collect();
    // Nothing to rebuild before the body exists (e.g. a restart in the first frames) — the guarded
    // initial spawn will place it; respawning an absent crab would be a no-op despawn + a duplicate.
    if parts.is_empty() {
        return;
    }
    let origin = world
        .resource::<CrabSpawns>()
        .0
        .first()
        .copied()
        .unwrap_or(Vec3::ZERO);
    world.resource_scope(|world, assets: Mut<crab_world::bot::body::CrabAssets>| {
        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, world);
        crab_world::bot::respawn_crab(&mut commands, &assets, parts.into_iter(), origin, 0);
        queue.apply(world);
    });
}

/// The per-tick bridge↔sim handshake for a sim driven DIRECTLY (the headless probe): push the
/// real crab body's game position + facing into the sim (so this tick's grab/extraction
/// resolve against it), then refresh the player the crab hunts (nearest living). The windowed
/// driver makes the same push through the authoritative server instead
/// ([`crate::server::CrabPose`] into `Server::step_next`).
pub fn sync_external_crab(sim: &mut crate::sim::Sim, bridge: &mut ExternalCrabBridge) {
    sim.set_external_crab_pose(bridge.world_pos(), bridge.yaw_turns(), bridge.phys_digest);
    bridge.set_hunt_target(sim.nearest_living_player_pos());
}

/// Recompute env 0's full rapier physics digest (every actuated body's pose+velocity bits,
/// via the shared [`crab_world::bot::physics_digest`]) XORed with the loaded policy's weights
/// digest, and store it on the bridge for the next pose push into the sim's state hash
/// ([`sync_external_crab`] / [`crate::server::CrabPose`]).
/// Runs each step AFTER the physics writeback + [`integrate_crab`], so it captures
/// this tick's settled state. Folding in the weights digest makes two peers running different
/// brains desync on the first tick, and folding in the full articulated state makes a float
/// divergence a detected desync, not just a 2D-pose one.
#[allow(clippy::type_complexity)]
fn hash_crab_physics(
    policy: NonSend<Policy>,
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
    let phys = crab_world::bot::physics_digest::crab_state_digest(
        bodies
            .iter()
            .filter(|(env, ..)| env.0 == 0)
            .map(|(_, t, v, j, c)| (t, v, j, c)),
    );
    // Fold the weight identity into the lockstep hash: `None` (rest pose / diagnostic brain,
    // no shared checkpoint) contributes 0 — a no-op XOR — exactly as the old `0` sentinel did.
    bridge.phys_digest = phys ^ policy.weights_digest().map_or(0, std::num::NonZeroU64::get);
}

/// Runtime gate for the external NN crab, as a presence marker: the resource exists iff the NN
/// crab is armed. The boot-menu path adds the whole NN stack at app build (plugins can't be
/// added later) but does NOT know yet how the round resolves — so every NN system is gated on
/// this, present ONLY once the round arms the crab ([`crate::may_arm_external_crab`]: solo
/// always, networked only with synced weights). While absent (only behind the boot menu, before
/// a round resolves — NumEnvs 0) the crab never spawns and `drive_lockstep` never pumps physics
/// or calls [`sync_external_crab`]. Once a round resolves it is ARMED, or — if it can't be
/// (networked-unsynced) — the round is REFUSED loudly (no integer fallback; see
/// [`crate::render`]'s `ensure_round_installed`). The scripted `Boot::Round` path inserts it
/// armed at build; there is ONE gate for all paths. The run-condition and render's reads gate on
/// `Option<Res<ExternalCrabArmed>>::is_some()`.
#[derive(Resource)]
pub struct ExternalCrabArmed;

/// Run condition: the external NN crab is armed. Gates every [`ExternalCrabPlugin`] system so they
/// idle (zero cost beyond the check) until the round arms the crab.
fn external_crab_armed(active: Option<Res<ExternalCrabArmed>>) -> bool {
    active.is_some()
}

/// Arm the one giant NN crab: insert the [`ExternalCrabArmed`] gate. Every arm site — the
/// windowed `Boot::Round` + menu-transition clients, the screenshot path, the headless probe —
/// goes through this one path. The walk target is the player's actual position (no per-peer
/// tunable steering it), so there is nothing solo-vs-networked to reconcile here.
pub fn arm(world: &mut World) {
    world.insert_resource(ExternalCrabArmed);
    // The armed crab IS the one trained Sally: a `rescue_nonfinite_crabs` fire is a
    // physics-correctness FAULT to surface LOUDLY (error log + aggregated telemetry, hard panic
    // in debug/test), never a silent catch-and-respawn. Marking it HERE — the one arm
    // path every site funnels through — means no caller can arm Sally and forget to make her
    // rescue loud; training (which never calls `arm`) keeps the quiet routine-terminator rescue.
    world.insert_resource(crab_world::bot::CrabRescueIsFault);
}

/// Plugin: the external NN crab. Adds the loaded policy + the bridge, and the systems that
/// (a) spawn the crab once armed, (b) aim + drive the policy, (c) integrate the body's walk
/// into the game crab. The caller must ALSO have added the bot/physics stack
/// (`RapierPhysicsPlugin` + `PhysicsWorldPlugin` + `BotPlugin`) to the same app — see
/// [`crate::render`]'s `add_external_nn_crab` — so the rig actually exists and steps. Every
/// system is gated on [`ExternalCrabArmed`]: the scripted `Boot::Round` path arms it at build;
/// the boot menu arms it only once the round resolves armable.
pub struct ExternalCrabPlugin {
    /// Directory the brain (`brain.bin`) + normalizer (`normalizer.bin`) load from.
    /// Configurable so deploy can point it at the chosen checkpoint.
    pub checkpoint_dir: std::path::PathBuf,
    /// The sim's crab spawn, so the bridge starts the NN crab there.
    pub crab_spawn: Pos,
}

impl Plugin for ExternalCrabPlugin {
    fn build(&self, app: &mut App) {
        let policy = Policy::load(&self.checkpoint_dir);
        // The NN crab only moves when a fitting checkpoint armed it; otherwise it stands
        // inert. `Policy::load` already logged WHY (a loud `error!` for a refused wrong-rig
        // checkpoint) — and every game entry point hard-fails on a bad checkpoint before this
        // plugin is even built, so a non-arming here is the legitimate "no brain yet" rest pose.
        if !policy.is_loaded() {
            warn!(
                "external_crab: no usable checkpoint at {} — NN crab holds rest pose",
                self.checkpoint_dir.display()
            );
        }
        app.insert_non_send_resource(policy);
        app.insert_resource(ExternalCrabBridge::new(self.crab_spawn));

        // Spawn the crab the first frame the gate is active. On the scripted solo
        // path the gate is true from build, so this spawns on frame 1 exactly as before; on
        // the menu path it spawns only once the solo round is chosen. Reuses the bot stack's
        // own [`spawn_initial_crabs`] (one spawn path, no drift) after bumping NumEnvs to 1.
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

        // Aim + drive the policy in the Sense→Think→Act chain. The walk target MUST be
        // placed BEFORE Sense: `build_observation` (in BotSet::Sense) reads `CrabTargets`
        // to build the target-relative observation the policy steers by, so a target set
        // after Sense would only reach the policy a tick late (and tick 0 not at all). Then
        // run the policy in Think; the actuator (Act) + rapier step follow from the bot
        // stack the caller added. Gated on the armed flag so the policy is inert until armed.
        app.add_systems(
            FixedUpdate,
            (
                set_crab_walk_target.before(BotSet::Sense),
                run_crab_policy.in_set(BotSet::Think),
            )
                .run_if(external_crab_armed),
        );
        // After physics has moved the body this fixed step, fold its displacement into
        // the game-world crab position. Runs in FixedUpdate (one physics step per run) so
        // every bit of walk is captured, not just the frames the render samples.
        app.add_systems(
            FixedUpdate,
            integrate_crab
                .after(PhysicsSet::Writeback)
                .run_if(external_crab_armed),
        );
        // Digest this step's settled rapier state (+ the policy weights) for the lockstep
        // desync check. After the writeback AND `integrate_crab` so it reads the final
        // post-step pose, and before the driver pushes the bridge into the sim
        // (`sync_external_crab`) each tick. Gated on the active flag so it never runs on a
        // round without the NN crab.
        app.add_systems(
            FixedUpdate,
            hash_crab_physics
                .after(integrate_crab)
                .after(PhysicsSet::Writeback)
                .run_if(external_crab_armed),
        );

        // Render-only: publish the giant-crab repose for the skin to apply to its render bones
        // ([`crab_world::bot::skin::drive_bones`]). This NEVER mutates the rapier-driven crab
        // `Transform`s — doing so teleported the body and crashed the client (see
        // [`crab_world::bot::skin::CrabSkinRepose`]). After `integrate_crab`, which finalises this
        // step's game-world position the repose is built from. Gated on Visuals — headless never
        // renders a skin, so it has nothing to repose.
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

/// Bump `NumEnvs` to 1 the first time the crab is armed, so the shared
/// [`crab_world::bot::spawn_initial_crabs`] (run right after, gated on the same armed flag) spawns
/// exactly one crab. This does NOT spawn — it sizes the env so the shared spawner does. Kept to
/// env 0 / a single crab — a round has one giant crab. A no-op once the crab exists (`NumEnvs`
/// already 1); the spawn-once guard is [`crab_not_yet_spawned`].
fn ensure_crab_env(mut num_envs: ResMut<crab_world::bot::NumEnvs>) {
    if num_envs.0 == 0 {
        num_envs.0 = 1;
    }
}

/// Run condition: no crab body has spawned yet — so [`crab_world::bot::spawn_initial_crabs`]
/// (which appends) fires exactly once on the menu solo path rather than every frame the
/// gate is active. The scripted solo path's crab also spawns through here, so there is one
/// spawn trigger, not two.
fn crab_not_yet_spawned(crabs: Query<(), With<CrabCarapace>>) -> bool {
    crabs.is_empty()
}

/// Aim the policy at the player's ACTUAL position, in the crab's ARENA frame (the frame the
/// body + the observation live in). The target is the carapace plus the true game-world
/// offset to the hunted player (`carapace + (player − crab)`) — no lead, no treadmill — so the
/// observation ([`crab_world::bot::sensor::build_observation`]) re-expresses the REAL player
/// offset in carapace-local axes and the policy's WEIGHTS supply the approach (the bitter
/// lesson). A player past the trained band (`[1.5, ~9] m`) saturates the obs normalizer clamp,
/// so the crab reads "player far, that way" and walks toward it without resolving exact
/// distance; on the open inference field (rl#209) it keeps walking until the gap re-enters
/// the band — see the module header. With no living target (all players down) the target is
/// dropped and the crab holds.
fn set_crab_walk_target(
    bridge: Res<ExternalCrabBridge>,
    spawns: Res<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
) {
    let Some(slot) = targets.envs.first_mut() else {
        return;
    };
    // The hunted player's offset from the crab, in GAME-world XZ. `None` ⇒ nobody to chase.
    let Some(hunt) = bridge.hunt_target_m else {
        *slot = None; // nobody to chase → no target → policy holds (rest pose)
        return;
    };
    let to_prey = hunt - bridge.world_pos_m;
    if to_prey.length_squared() < 1e-6 {
        return; // already on the prey; leave the last target so it doesn't flicker
    }

    // Carapace position in the arena frame (env 0). Fall back to the env's spawn origin
    // before the body exists.
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
    let carapace = carapace_q
        .iter()
        .find(|(env, _)| env.0 == 0)
        .map(|(_, t)| t.translation)
        .unwrap_or(origin);

    // Plant the target at the player's real position in the arena frame: the carapace plus the
    // full game-world offset to the player, at any distance. `CLAW_TARGET_Y` is a ground-level
    // grab height (the planar hunt target carries no height of its own).
    let target = Vec3::new(
        carapace.x + to_prey.x,
        CLAW_TARGET_Y,
        carapace.z + to_prey.y,
    );
    *slot = Some(target);
}

/// Run the loaded policy on env 0's observation and write env 0's action — the same
/// deterministic mean action the demo's `policy_step` uses ([`Policy::act`]), so the
/// external crab and the demo share one inference path. The actuator (bot stack) turns the
/// action into joint torques next in the chain.
fn run_crab_policy(
    policy: NonSend<Policy>,
    mut bridge: ResMut<ExternalCrabBridge>,
    obs: Res<crab_world::bot::sensor::CrabObservation>,
    mut actions: ResMut<crab_world::bot::actuator::CrabActions>,
) {
    // Spawn settle: hold zero torque while the fresh crab drops onto its legs (see the
    // `settle` field doc), then the policy takes over.
    if bridge.settle > 0 {
        bridge.settle = crab_world::bot::settle_countdown(bridge.settle);
        if let Some(a) = actions.envs.first_mut() {
            *a = [0.0; crab_world::bot::actuator::ACTION_SIZE];
        }
        return;
    }
    if let (Some(o), Some(a)) = (obs.envs.first(), actions.envs.first_mut()) {
        *a = policy.act(o);
    }
}

/// After a physics step, fold the carapace's horizontal walk into the game-world crab
/// position (the displacement since last step, 1:1) and the walk heading into the yaw,
/// storing both on the bridge. The render-loop owner ([`crate::render`]) then pushes
/// them into the sim each game tick via [`sync_external_crab`].
fn integrate_crab(
    mut bridge: ResMut<ExternalCrabBridge>,
    mut rescued: MessageReader<crab_world::bot::CrabRescued>,
    carapace_q: Query<(&CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
) {
    // A force-respawn (BotPlugin's `rescue_nonfinite_crabs`) teleported env 0's crab back
    // to the arena spawn. Forget the pre-blowup carapace so we DON'T difference the new
    // spawn against it (that would add a multi-metre false step to the game crab); the
    // next frame re-seeds `last_carapace_m` from the fresh pose. Re-settle too, so the
    // rebuilt body gets its drop/plant grace before we trust its motion again.
    let was_rescued = rescued.read().any(|m| m.env == 0);
    if was_rescued {
        bridge.last_carapace_m = None;
        bridge.settle = crab_world::bot::RESET_GRACE_TICKS;
    }

    let Some((_, t, vel)) = carapace_q.iter().find(|(env, _, _)| env.0 == 0) else {
        return;
    };
    if !t.translation.is_finite() {
        return; // a blown-up body (rescued next tick) — don't smear NaN into the position
    }
    let here = Vec2::new(t.translation.x, t.translation.z);
    // Accumulate the walk only when the body is under the policy (settle spent) AND we have
    // a prior sample to difference. The carapace barely translates during the settle DROP
    // (it's planting, not striding), so always tracking `last_carapace_m` — even through
    // settle — means the first post-settle accumulation differences against a pose ~at rest:
    // sub-millimetre of settle motion leaks in, which is negligible and (verified) does not
    // perturb the emergent walk, whereas deferring the seed a tick noticeably changed the
    // chaotic gait. `last_carapace_m` is None only right after spawn/rescue (re-seeded here).
    if bridge.settle == 0
        && let Some(prev) = bridge.last_carapace_m
    {
        bridge.world_pos_m += here - prev;
    }
    bridge.last_carapace_m = Some(here);

    // Facing from the horizontal walk velocity (when actually moving), so the rendered
    // crab + the sim yaw point where it's going. atan2(x, z) matches the sim's
    // turn-unit convention (see trig::atan2_turns: +Z forward, +X right).
    let v = Vec2::new(vel.linear.x, vel.linear.z);
    if v.length_squared() > 1e-4 {
        let radians = v.x.atan2(v.y); // (x, z) → heading
        bridge.yaw_turns = crate::sim::trig_client::radians_to_turns(radians);
    }
}

/// Render-only: publish the crab repose into [`crab_world::bot::skin::CrabSkinRepose`] so the skin
/// draws the arena rig at the crab's game spot. The skin's [`crab_world::bot::skin::drive_bones`]
/// applies this to the render BONES only; this system NEVER mutates the rapier-driven `CrabBodyPart`
/// `Transform`s, because bevy_rapier reads a changed body `Transform` back into the physics body —
/// a "cosmetic" link shift teleports the body and explodes the solver. So the render placement is
/// decoupled from physics by construction: the policy, the colliders, and the lockstep hash all
/// keep reading the raw ~1 m pose regardless of `Visuals`, so a windowed and a headless peer stay
/// bit-identical.
///
/// The repose is a pure rigid translate (NO scale): the crab renders at its TRUE physics size so a
/// collider wireframe overlays the mesh (render==physics). The giant FEEL comes from the human world
/// rendering R× smaller around it ([`crate::render::world_render_scale`]), so the crab still towers
/// without the inflation that would desync the wireframe and force retraining Sally bigger. `None`
/// until the crab is sampled once (no stale/zero carapace).
fn publish_skin_repose(
    bridge: Res<ExternalCrabBridge>,
    repose_out: Option<ResMut<crab_world::bot::skin::CrabSkinRepose>>,
) {
    // No skin loaded (no `sally.glb`) ⇒ no resource ⇒ nothing to repose (the static giant
    // silhouette is the visible crab there — `spawn_world` keeps it shown when no model loads).
    let Some(mut out) = repose_out else {
        return;
    };
    // Rigid translate carrying the arena rig to the crab's render-frame game spot:
    // `shift = game_spot·R − arena_carapace` (`render_placement_m` gives `pivot` = the game spot
    // and `pivot − shift` = the arena carapace). See the doc above for why no scale.
    let rs = crate::render::world_render_scale();
    out.0 = bridge.render_placement_m().map(|r| {
        let game = r.pivot; // crab's game-world XZ (m)
        let arena = r.pivot - r.shift; // the arena carapace XZ the rig sits at
        let s = game * rs - arena;
        crab_world::bot::skin::SkinRepose {
            shift: Vec3::new(s.x, 0.0, s.y),
            // Unused at scale 1 (the repose matrix is a pure translate); kept for the field.
            pivot: Vec3::ZERO,
            scale: 1.0,
        }
    });
}

mod probe;

// The single-peer NN-crab determinism + walk/stability verification harness lives in `probe`
// (out of this production bridge so the shipping real-Sally MP path stays lean). Re-exported
// here so its public entry points keep their `external_crab::…` paths.
pub use probe::{ProbeSample, StabilityResult, run_headless_probe, run_vehicle_stability_probe};
