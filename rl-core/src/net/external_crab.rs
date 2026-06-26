//! A real rapier-simulated, NN-driven giant crab, replacing the integer point-pursuer with the
//! actual trained RL body. Armed by the [`ExternalCrabArmed`] runtime gate on a SOLO round (a
//! Host-alone Start, or scripted `--host` that found no peer), and — since the GCR fold (rl#82)
//! — on a NETWORKED round with synced weights.
//!
//! # When it drives the crab (and why it's safe)
//! The game's [`crate::net::sim`] is a bit-deterministic INTEGER lockstep sim so two peers
//! evolve identically. A rapier crab is FLOAT physics, so it can drive a *networked* crab
//! without desyncing ONLY when every peer (a) loaded the same brain (the weights-digest
//! handshake) and (b) steps the body the identical number of physics steps per lockstep tick
//! (the deterministic [`crate::net::cadence::PhysicsCadence`], pumped by
//! `net::render::drive_lockstep`). [`crate::net::may_arm_external_crab`] is the gate that
//! enforces both; a networked-UNSYNCED round stays on the integer pursuit
//! ([`crate::net::sim::Sim::step`]), the cross-peer-safe fallback. On a solo round there's no
//! peer to desync against, so it always arms. The integer pursuit remains the FALLBACK, not a
//! parallel implementation — exactly one of the two drives any given round.
//!
//! # How it works
//! The trained policy ([`crate::play::Policy`], loaded from a checkpoint) is a
//! *locomotion+reach* brain: its observation includes the touch target as a vector in
//! the carapace's local frame, and it has learned to WALK the body toward that vector
//! (the `ckpt-best.locomotion` weights walk + reach). We exploit that directly:
//!
//! - The crab lives in its OWN small rapier arena (the shared [`crate::bot`] /
//!   [`crate::physics`] world, a ±10 m walled box), centred near the origin.
//! - Each control step we place its target a fixed lead distance AHEAD of the carapace
//!   in the direction of the nearest living game player (mapped into the crab's frame),
//!   so the policy always walks "toward the player". The target rides with the crab — a
//!   treadmill — so the body never has to physically cross the (large, unbounded) game
//!   map inside its little arena.
//! - We integrate the carapace's horizontal displacement each game tick and add it to
//!   the game-world crab position, which we write back into the sim with
//!   [`Sim::set_external_crab_pose`]. Grabs / extraction / win-loss then resolve against the
//!   REAL crab body, exactly as the prompt asks.
//! - For rendering, every crab body part's bevy `Transform` is shifted by a render-only
//!   offset each frame so the rig APPEARS at the game-world crab position (its physics
//!   stays in the small arena). The sally skin, if present, rides the same parts; with
//!   no model the rl#5 procedural fallback rig gives a Rapier debug-wireframe body.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::Visuals;
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};
use crate::bot::sensor::CrabTargets;
use crate::bot::{BotSet, CrabSpawns};
use crate::net::sim::{Pos, UNIT};
use crate::play::Policy;

/// How far ahead of the carapace (metres) to plant the target each step, toward the
/// player. Set to the MIDDLE of the policy's trained reach band (~1.5–2 m): the
/// `ckpt-best.locomotion` checkpoint was trained on a claw-tip-to-target proximity reward
/// (no base-locomotion term — see this module's report), so it LEANS/REACHES toward a
/// near target rather than walking to a far one. A lead inside its reach band makes it
/// lean + shuffle toward the player (the most pursuit this policy produces); a far lead it
/// just ignores (verified). `RL_CRAB_LEAD` overrides it. A future WALKING checkpoint
/// (locomotion-rewarded) would want a larger lead — drop it in via the configurable
/// checkpoint path and tune this.
const TARGET_LEAD_M: f32 = 1.6;

/// Height (m) of the walk-target lead point. A low ground-ish Y inside the policy's trained
/// reach band (`TARGET_Y_MIN..MAX` ≈ 0.15..0.7 in training) so the crab reads it as a
/// reach-toward-and-walk target, not an overhead one.
const TARGET_LEAD_Y: f32 = 0.3;

/// Default game-world metres the crab advances per metre its carapace actually walks in its
/// arena. 1.0 = the game crab moves exactly as far as the body does. The current policy
/// locomotes slowly, so `RL_CRAB_WORLD_GAIN` (resolved once onto the bridge) lets the owner
/// scale the showcase crab's map speed up WITHOUT the policy walking any harder — the
/// giant crab already reads as huge via [`crate::net::sim::CRAB_SCALE`] rendering, so a >1
/// gain just covers the big map faster. A real FEEL KNOB, env-tunable like `RL_CRAB_LEAD`.
const WORLD_GAIN_DEFAULT: f32 = 1.0;

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
    /// Current facing yaw in the sim's [`crate::net::sim::trig`] turn units, from the
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
    /// How far ahead of the carapace (m) to plant the walk target — resolved ONCE at plugin
    /// build (default [`TARGET_LEAD_M`], `RL_CRAB_LEAD` override) rather than re-reading the
    /// env var every tick.
    lead_m: f32,
    /// Game-world metres advanced per metre walked (see [`WORLD_GAIN_DEFAULT`]) — resolved
    /// once at plugin build (`RL_CRAB_WORLD_GAIN` override).
    world_gain: f32,
    /// This tick's peer-comparable digest of the crab's full rapier physics state XORed with
    /// the policy-weights digest — recomputed each step by [`hash_crab_physics`] and pushed
    /// into the sim by [`sync_external_crab`], so the lockstep desync check covers the
    /// articulated float body and rejects a peer running different weights (rl#82, GCR). `0`
    /// until the first post-step hash; the integer sim ignores it unless external control is
    /// armed.
    phys_digest: u64,
}

/// The sim's fixed-point [`Pos`] (XZ) as game-world metres on the bridge's `Vec2` frame
/// (sim `x`→`Vec2.x`, sim `z`→`Vec2.y`). One definition for every Pos→metres conversion in
/// this module, so the `/ UNIT` cast can't drift between the spawn seed, the restart re-seed,
/// and the hunt target (the manual's "one source, derive the rest").
fn pos_to_m(p: Pos) -> Vec2 {
    Vec2::new(p.x as f32 / UNIT as f32, p.z as f32 / UNIT as f32)
}

impl ExternalCrabBridge {
    /// Seed the bridge at the sim's integer crab spawn (so the NN crab begins where the
    /// round placed the giant crab, MIN_CRAB_SPAWN_DISTANCE from the players). `lead_m` /
    /// `world_gain` are the per-tick walk-target lead and the world-speed gain, resolved
    /// once by the plugin.
    fn new(spawn: Pos, lead_m: f32, world_gain: f32) -> Self {
        Self {
            world_pos_m: pos_to_m(spawn),
            lead_m,
            world_gain,
            last_carapace_m: None,
            yaw_turns: 0,
            hunt_target_m: None,
            settle: crate::training::systems::RESET_GRACE_TICKS,
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
        Pos {
            x: (self.world_pos_m.x * UNIT as f32) as i64,
            z: (self.world_pos_m.y * UNIT as f32) as i64,
        }
    }

    /// The render-only offset (metres, XZ) from the crab's arena frame to its game-world
    /// position: add it to every crab part's bevy `Transform` so the rig appears at the
    /// game crab spot while its physics stays in the small arena. `None` until the crab
    /// has been sampled once (so we never offset by a stale/zero carapace).
    fn render_offset_m(&self) -> Option<Vec2> {
        self.last_carapace_m.map(|c| self.world_pos_m - c)
    }

    pub fn yaw_turns(&self) -> i32 {
        self.yaw_turns
    }

    /// Set the game-world point (the nearest living player's [`Pos`]) the crab hunts, or
    /// `None` when every player is down. Called each game tick by the render loop, which
    /// owns the sim and so knows the prey; [`set_crab_walk_target`] reads it to aim the
    /// policy.
    pub fn set_hunt_target(&mut self, prey: Option<Pos>) {
        self.hunt_target_m = prey.map(pos_to_m);
    }

    /// Pin the feel knobs (walk-target lead, world-speed gain) to their canonical defaults,
    /// dropping any `RL_CRAB_LEAD` / `RL_CRAB_WORLD_GAIN` env override. MUST be called when
    /// arming on a NETWORKED round: those knobs are a per-PROCESS solo-tuning convenience, and
    /// they both feed the HASHED crab pose (`world_gain` scales the per-tick displacement;
    /// `lead_m` steers the policy's trajectory), so a peer that set them differently would walk
    /// the crab to a different pose and desync. The weights handshake covers only the brain
    /// bytes, not these — so networked play uses the canonical feel on every peer.
    pub fn pin_default_knobs(&mut self) {
        self.lead_m = TARGET_LEAD_M;
        self.world_gain = WORLD_GAIN_DEFAULT;
    }

    /// Re-seed the bridge to the round's spawn after a deterministic sim RESTART
    /// ([`buttons::RESTART`]). The sim's [`reset`](crate::net::sim::Sim::reset) rebuilds the
    /// integer crab back AT spawn, but the float body keeps walking and the bridge keeps its
    /// accumulated `world_pos_m`; without this the next [`sync_external_crab`] would snap the
    /// freshly-restarted crab onto the still-walking body's old position — mid-gait at the
    /// wrong place, not at the computed spawn. So move the game-world position back to the
    /// integer spawn and forget the pre-restart carapace sample (re-seeded from the fresh pose
    /// next frame, exactly as the [`CrabRescued`](crate::bot::CrabRescued) path does — without
    /// it the first post-restart accumulation would difference the spawn against the old pose
    /// and inject a multi-metre false step). Re-settle too, so the round opens with the spawn
    /// drop/plant grace.
    ///
    /// DETERMINISM: this fires off the sim's restart EDGE — the same edge the cadence reset
    /// hangs off in [`crate::net::render`]'s `drive_lockstep`, observed identically on every
    /// peer (the RESTART rides the shared lockstep input stream, so `advance_one` rewinds the
    /// sim on the SAME applied tick on both peers). `spawn` is read back from the post-restart
    /// sim, which is itself deterministic — so the re-seed is bit-identical cross-peer. Unlike
    /// `CrabRescued` (a float-body teleport that leaves the game position put) this DOES move
    /// `world_pos_m`, because a restart moves the integer crab.
    pub fn restart_to_spawn(&mut self, spawn: Pos) {
        self.world_pos_m = pos_to_m(spawn);
        self.last_carapace_m = None;
        self.settle = crate::training::systems::RESET_GRACE_TICKS;
    }
}

/// The per-tick bridge↔sim handshake, run BEFORE each [`Lockstep::try_advance`]: push the
/// real crab body's game position + facing into the sim (so this tick's grab/extraction
/// resolve against it), then refresh the player the crab hunts (nearest living — the same
/// prey the integer pursuit picks). ONE definition so the windowed driver
/// ([`crate::net::render`]'s `drive_lockstep`) and the headless probe can't drift on the
/// contract (the manual's "one implementation per thing").
pub fn sync_external_crab(ls: &mut crate::net::lockstep::Lockstep, bridge: &mut ExternalCrabBridge) {
    ls.set_external_crab_pose(bridge.world_pos(), bridge.yaw_turns(), bridge.phys_digest);
    bridge.set_hunt_target(ls.sim().nearest_living_player_pos());
}

/// Recompute env 0's full rapier physics digest (every actuated body's pose+velocity bits,
/// via the shared [`crate::bot::physics_digest`]) XORed with the loaded policy's weights
/// digest, and store it on the bridge for [`sync_external_crab`] to push into the lockstep
/// hash. Runs each step AFTER the physics writeback + [`integrate_crab`], so it captures
/// this tick's settled state. Folding in the weights digest makes two peers running different
/// brains desync on the first tick (the GCR shared-checkpoint guard), and folding in the full
/// articulated state makes a float divergence a detected desync, not just a 2D-pose one.
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
    let phys = crate::bot::physics_digest::crab_state_digest(
        bodies
            .iter()
            .filter(|(env, ..)| env.0 == 0)
            .map(|(_, t, v, j, c)| (t, v, j, c)),
    );
    bridge.phys_digest = phys ^ policy.weights_digest();
}

/// Runtime gate for the external NN crab (rl#58 + GCR rl#82), as a presence marker: the
/// resource exists iff the NN crab is armed. The boot-menu path adds the whole NN stack at app
/// build (plugins can't be added later) but does NOT know yet how the round resolves — so every
/// NN system is gated on this, present ONLY once the round arms the crab
/// ([`crate::net::may_arm_external_crab`]: solo always, networked only with synced weights).
/// While absent the crab never spawns and no policy/physics drives it, and crucially
/// [`crate::net::render`]'s `drive_lockstep` never pumps physics or calls [`sync_external_crab`]
/// — so a networked-UNSYNCED round with the stack present stays byte-identical to the
/// integer-crab path. The scripted `Boot::Round` path inserts it at build; there is ONE gate for
/// all paths, not a second always-on code path. The run-condition and render's reads gate on
/// `Option<Res<ExternalCrabArmed>>::is_some()`.
#[derive(Resource)]
pub struct ExternalCrabArmed;

/// Run condition: the external NN crab is armed. Gates every [`ExternalCrabPlugin`] system so they
/// idle (zero cost beyond the check) until the round arms the crab.
fn external_crab_armed(active: Option<Res<ExternalCrabArmed>>) -> bool {
    active.is_some()
}

/// Plugin: the external NN crab. Adds the loaded policy + the bridge, and the systems that
/// (a) spawn the crab once armed, (b) aim + drive the policy, (c) integrate the body's walk
/// into the game crab. The caller must ALSO have added the bot/physics stack
/// (`RapierPhysicsPlugin` + `PhysicsWorldPlugin` + `BotPlugin`) to the same app — see
/// [`crate::net::render`]'s `add_external_nn_crab` — so the rig actually exists and steps. Every
/// system is gated on [`ExternalCrabArmed`]: the scripted `Boot::Round` path arms it at build;
/// the boot menu arms it only once the round resolves armable (rl#58 + GCR).
pub struct ExternalCrabPlugin {
    /// Directory the brain (`brain.bin`) + normalizer (`normalizer.bin`) load from.
    /// Configurable so deploy can point it at the chosen checkpoint.
    pub checkpoint_dir: std::path::PathBuf,
    /// The sim's integer crab spawn, so the bridge starts the NN crab there.
    pub crab_spawn: Pos,
}

impl Plugin for ExternalCrabPlugin {
    fn build(&self, app: &mut App) {
        let policy = Policy::load(&self.checkpoint_dir);
        // Fail loud when there's no usable checkpoint: the policy returns the zero-action rest
        // pose, so the showcase crab stands inert. Name the surface + dir so the operator knows
        // WHY the demo crab isn't moving.
        if !policy.is_loaded() {
            warn!(
                "external_crab: no usable checkpoint at {} — NN crab holds rest pose",
                self.checkpoint_dir.display()
            );
        }
        app.insert_non_send_resource(policy);
        // Resolve the env-tunable knobs ONCE here (override → default), not per tick.
        let env_f32 = |key: &str, default: f32| {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(default)
        };
        let lead_m = env_f32("RL_CRAB_LEAD", TARGET_LEAD_M);
        let world_gain = env_f32("RL_CRAB_WORLD_GAIN", WORLD_GAIN_DEFAULT);
        app.insert_resource(ExternalCrabBridge::new(self.crab_spawn, lead_m, world_gain));

        // Spawn the crab the first frame the gate is active (rl#58). On the scripted solo
        // path the gate is true from build, so this spawns on frame 1 exactly as before; on
        // the menu path it spawns only once the solo round is chosen. Reuses the bot stack's
        // own [`spawn_initial_crabs`] (one spawn path, no drift) after bumping NumEnvs to 1.
        app.add_systems(
            Update,
            ensure_crab_env
                .run_if(external_crab_armed)
                .before(crate::bot::spawn_initial_crabs),
        );
        app.add_systems(
            Update,
            crate::bot::spawn_initial_crabs
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

        // Render-only: shift the rendered rig to the game-world crab position. MUST run in
        // FixedUpdate right after the physics writeback (where rapier sets each part's raw
        // arena-frame Transform) and after `integrate_crab` (which finalises the
        // offset for this step) — exactly ONCE per writeback. It adds the offset to the
        // freshly-written raw pose, so it must NOT run in Update: Update can run many times
        // between fixed steps, and `+=`-ing the offset each of those frames would
        // accumulate and fling the rig away. Gated on Visuals — headless has no rendered
        // transforms to place.
        if app.world().get_resource::<Visuals>().is_some_and(|v| v.0) {
            app.add_systems(
                FixedUpdate,
                offset_rendered_crab
                    .after(integrate_crab)
                    // MUST run after the determinism hash: this render-only shift mutates the
                    // crab parts' `Transform` and is gated on `Visuals`, so hashing the
                    // POST-shift pose would make a windowed peer and a headless peer disagree
                    // (the headless one never applies the shift). Hash the raw arena pose
                    // first, then cosmetically shift it.
                    .after(hash_crab_physics)
                    .after(PhysicsSet::Writeback)
                    .run_if(external_crab_armed),
            );
        }
    }
}

/// Bump `NumEnvs` to 1 the first time the crab is armed, so the shared
/// [`crate::bot::spawn_initial_crabs`] (run right after, gated on the same armed flag) spawns
/// exactly one crab. This does NOT spawn — it sizes the env so the shared spawner does. Kept to
/// env 0 / a single crab — a round has one giant crab. A no-op once the crab exists (`NumEnvs`
/// already 1); the spawn-once guard is [`crab_not_yet_spawned`].
fn ensure_crab_env(mut num_envs: ResMut<crate::bot::NumEnvs>) {
    if num_envs.0 == 0 {
        num_envs.0 = 1;
    }
}

/// Run condition: no crab body has spawned yet — so [`crate::bot::spawn_initial_crabs`]
/// (which appends) fires exactly once on the menu solo path rather than every frame the
/// gate is active. The scripted solo path's crab also spawns through here, so there is one
/// spawn trigger, not two.
fn crab_not_yet_spawned(crabs: Query<(), With<CrabCarapace>>) -> bool {
    crabs.is_empty()
}

/// Aim the policy: plant env 0's touch target a fixed lead ahead of the carapace toward
/// the nearest living player, in the crab's ARENA frame (the frame the body + the
/// observation live in). The observation ([`crate::bot::sensor::build_observation`])
/// rotates this into the carapace-local frame, so the policy sees "target is over there"
/// and walks for it. With no living target (all players down) the target is dropped and
/// the crab holds.
fn set_crab_walk_target(
    bridge: Res<ExternalCrabBridge>,
    spawns: Res<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
) {
    let Some(slot) = targets.envs.first_mut() else {
        return;
    };
    // Direction to the hunted player, in GAME-world XZ, reduced to a unit heading.
    let Some(hunt) = bridge.hunt_target_m else {
        *slot = None; // nobody to chase → no target → policy holds (rest pose)
        return;
    };
    let to_prey = hunt - bridge.world_pos_m;
    if to_prey.length_squared() < 1e-6 {
        return; // already on the prey; leave the last target so it doesn't flicker
    }
    let heading = to_prey.normalize();

    // Carapace position in the arena frame (env 0). Fall back to the env's spawn origin
    // before the body exists.
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
    let carapace = carapace_q
        .iter()
        .find(|(env, _)| env.0 == 0)
        .map(|(_, t)| t.translation)
        .unwrap_or(origin);

    // Lead point AHEAD of the carapace along the heading (lead resolved once on the bridge).
    // The target rides with the crab, so the policy keeps walking that way (treadmill). Its
    // height [`TARGET_LEAD_Y`] is a low ground-ish Y inside the trained reach band — the crab
    // learned to walk to reach a low target with a claw, so a ground-level lead reads as "go
    // there", and the walking that gets the claw near it is the locomotion we want.
    let lead = bridge.lead_m;
    let target = Vec3::new(
        carapace.x + heading.x * lead,
        TARGET_LEAD_Y,
        carapace.z + heading.y * lead,
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
    obs: Res<crate::bot::sensor::CrabObservation>,
    mut actions: ResMut<crate::bot::actuator::CrabActions>,
) {
    // Spawn settle: hold zero torque while the fresh crab drops onto its legs, exactly as
    // the demo/training do, so the policy doesn't drive a still-falling body into a locked
    // pose that never walks. Counts down to 0, then the policy takes over.
    if bridge.settle > 0 {
        bridge.settle = crate::training::systems::settle_countdown(bridge.settle);
        if let Some(a) = actions.envs.first_mut() {
            *a = [0.0; crate::bot::actuator::ACTION_SIZE];
        }
        return;
    }
    if let (Some(o), Some(a)) = (obs.envs.first(), actions.envs.first_mut()) {
        *a = policy.act(o);
    }
}

/// After a physics step, fold the carapace's horizontal walk into the game-world crab
/// position (the displacement since last step × the world gain) and the walk heading into
/// the yaw, storing both on the bridge. The render-loop owner ([`crate::net::render`]) then
/// pushes them into the sim each game tick via [`sync_external_crab`].
fn integrate_crab(
    mut bridge: ResMut<ExternalCrabBridge>,
    mut rescued: MessageReader<crate::bot::CrabRescued>,
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
        bridge.settle = crate::training::systems::RESET_GRACE_TICKS;
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
        let step = (here - prev) * bridge.world_gain;
        bridge.world_pos_m += step;
    }
    bridge.last_carapace_m = Some(here);

    // Facing from the horizontal walk velocity (when actually moving), so the rendered
    // crab + the sim yaw point where it's going. atan2(x, z) matches the sim's
    // turn-unit convention (see trig::atan2_turns: +Z forward, +X right).
    let v = Vec2::new(vel.linear.x, vel.linear.z);
    if v.length_squared() > 1e-4 {
        let radians = v.x.atan2(v.y); // (x, z) → heading
        let turn = crate::net::sim::trig::TURN as f32;
        let mut t = (radians / std::f32::consts::TAU * turn).round() as i32;
        t = t.rem_euclid(crate::net::sim::trig::TURN);
        bridge.yaw_turns = t;
    }
}

/// Render-only: shift every crab body part's bevy `Transform` by the bridge's arena→game
/// offset, so the rendered rig (and its skin) appears at the game-world crab position while
/// its rapier bodies stay in the small arena. Pure cosmetics — it never touches rapier, so
/// physics/determinism are untouched.
///
/// Scheduled in FixedUpdate right after rapier's `Writeback` (and `integrate_crab`),
/// so it runs EXACTLY ONCE per writeback — each writeback first overwrites every part's
/// `Transform` with the raw arena pose, then this adds the offset once. That is why the
/// add is safe: it must NOT run in `Update`, which can run several times between fixed
/// steps with no intervening writeback to reset the pose — there each `+=` would stack and
/// fling the rig away by the full (≥12 m) offset. Crab parts are always under motor torque
/// so they never sleep out of the writeback, keeping the one-add-per-step invariant.
fn offset_rendered_crab(
    bridge: Res<ExternalCrabBridge>,
    mut parts_q: Query<(&CrabEnvId, &mut Transform), With<CrabBodyPart>>,
) {
    let Some(off) = bridge.render_offset_m() else {
        return;
    };
    let shift = Vec3::new(off.x, 0.0, off.y);
    for (env, mut t) in parts_q.iter_mut() {
        if env.0 == 0 {
            t.translation += shift;
        }
    }
}

// ---------------------------------------------------------------------------
// Headless verification probe (no window / GPU / display)
// ---------------------------------------------------------------------------

use crate::net::lockstep::Lockstep;
use crate::net::sim::{Input, PlayerId};

/// One logged sample of the headless probe: tick, the NN crab's game position (m), and
/// its distance to the hunted player (m). The shrinking distance over a run is the
/// evidence the policy actually WALKS the crab toward the player.
#[derive(Clone, Copy, Debug)]
pub struct ProbeSample {
    pub tick: u64,
    pub crab_x_m: f32,
    pub crab_z_m: f32,
    pub dist_to_prey_m: f32,
    pub state_hash: u64,
    /// DIAG: carapace position in its ARENA frame (m) — to see whether the policy is
    /// actually locomoting the body at all (vs holding a pose).
    pub carapace_arena_x: f32,
    pub carapace_arena_z: f32,
    pub carapace_y: f32,
    /// DIAG: closest claw-tip→target distance (m) this sample. The training reward is a
    /// claw-tip-to-target proximity (no base-locomotion term), so a SHRINKING value here
    /// confirms the policy works-as-trained (it reaches), even when the base barely walks.
    pub min_claw_to_target_m: f32,
}

/// Probe driver state: the lockstep sim driven by hand (outside Bevy's schedules) so the
/// harness can step it once per `app.update()`, in step with the one physics tick each
/// update runs. Non-send because [`Lockstep`] owns a [`Sim`] whose hasher etc. need not
/// be `Sync` here, and only the main thread drives it.
struct ProbeDriver {
    ls: Lockstep,
    samples: Vec<ProbeSample>,
    /// Log a sample every this-many sim ticks (keeps the output skimmable).
    log_every: u64,
}

/// Headless probe system (FixedUpdate, AFTER `integrate_crab`): take the freshly
/// integrated NN-crab position from the bridge, feed it into the sim, advance one sim
/// tick with the local player holding still, and log periodically. Mirrors what
/// `render::drive_lockstep` does for the windowed app, minus input/telemetry/interp — a
/// purpose-built verification driver, not a second production loop.
fn probe_step(
    mut driver: NonSendMut<ProbeDriver>,
    mut bridge: ResMut<ExternalCrabBridge>,
    // Diagnostics live HERE (the probe), not on the production bridge: the shipping game
    // never needs the claw-reach signal or carapace height, so it shouldn't compute them.
    targets: Res<CrabTargets>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    claw_q: Query<(&CrabEnvId, &Transform), With<crate::bot::body::CrabClawTip>>,
) {
    // Push the crab body into the sim + refresh the hunted player — the SAME handshake the
    // windowed driver runs (one shared definition, no drift).
    sync_external_crab(&mut driver.ls, &mut bridge);
    let prey = driver.ls.sim().nearest_living_player_pos();

    // Local player holds still (neutral input) so the test isolates the CRAB's motion:
    // the crab should close the gap on a stationary player. Single peer → its own input
    // completes every tick.
    driver.ls.submit_local_input(Input::from_axes(0.0, 0.0));
    let _ = driver.ls.try_advance();

    // Log periodically (and always at tick 1 so the start point is recorded).
    let tick = driver.ls.sim().tick();
    if tick == 1 || tick.is_multiple_of(driver.log_every) {
        let crab = driver.ls.sim().crab().pos();
        let crab_x_m = crab.x as f32 / UNIT as f32;
        let crab_z_m = crab.z as f32 / UNIT as f32;
        let dist_to_prey_m = prey
            .map(|p| {
                let dx = (p.x - crab.x) as f32 / UNIT as f32;
                let dz = (p.z - crab.z) as f32 / UNIT as f32;
                (dx * dx + dz * dz).sqrt()
            })
            .unwrap_or(f32::NAN);
        let state_hash = driver.ls.sim().state_hash();

        // Carapace arena pose (env 0) for the "is it walking?" diagnostic.
        let (carapace_arena_x, carapace_y, carapace_arena_z) = carapace_q
            .iter()
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| (t.translation.x, t.translation.y, t.translation.z))
            .unwrap_or((0.0, 0.0, 0.0));

        // Closest claw-tip→target distance — the reach signal (the actual training reward),
        // showing the policy reaches even when the base barely walks.
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

/// Run the NN crab headlessly for `ticks` sim steps and return the logged samples. Builds
/// the SAME windowless bot+physics world the training/tests use ([`crate::bot::test_util::headless_stack`])
/// plus [`ExternalCrabPlugin`] and a hand-driven lockstep — so the crab steps the exact
/// dynamics the policy trained under, with no GPU/display. `checkpoint_dir` is the trained
/// policy; `seed` seeds the round (same seed twice ⇒ identical samples, the determinism
/// check). The local player holds still so a shrinking `dist_to_prey_m` proves the crab
/// walks toward it under the policy.
pub fn run_headless_probe(
    checkpoint_dir: &std::path::Path,
    seed: u64,
    ticks: u64,
    log_every: u64,
) -> Vec<ProbeSample> {
    use crate::bot::test_util::{HeadlessStack, WorldRole, headless_stack};

    let me = PlayerId(0);
    let ls = Lockstep::new(seed, &[me], me);
    let crab_spawn = ls.sim().crab().pos();
    // The crab is externally driven for the whole probe (we own its position). Arm + seed the
    // pose atomically with the crab's CURRENT spawn pose/yaw — writing back what's already
    // there, so this is a no-op on sim state, just with the ordering footgun removed.
    let mut ls = ls;
    let crab = ls.sim().crab();
    // Seed with a zero digest; the first post-step `hash_crab_physics` fills it before the
    // first `sync_external_crab` push, so the seeded value is never the one cross-checked.
    ls.initialize_external_crab(crab.pos(), crab.yaw(), 0);

    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
    });
    app.add_plugins(ExternalCrabPlugin {
        checkpoint_dir: checkpoint_dir.to_path_buf(),
        crab_spawn,
    });
    // The probe always arms the crab — insert the gate so the policy/integration systems run.
    // The crab already spawned via `headless_stack`'s `num_envs: 1`, so the plugin's own gated
    // spawn is a no-op (the not-yet-spawned guard sees it present).
    app.insert_resource(ExternalCrabArmed);
    app.insert_non_send_resource(ProbeDriver {
        ls,
        samples: Vec::new(),
        log_every: log_every.max(1),
    });
    // Drive the sim AFTER the bridge integrates this step's walk AND hashes the physics
    // state, so each sim tick reads the up-to-date crab position + digest.
    app.add_systems(
        FixedUpdate,
        probe_step.after(integrate_crab).after(hash_crab_physics),
    );

    // One physics tick + one sim tick per update.
    for _ in 0..ticks {
        app.update();
    }
    app.world()
        .get_non_send_resource::<ProbeDriver>()
        .map(|d| d.samples.clone())
        .unwrap_or_default()
}
