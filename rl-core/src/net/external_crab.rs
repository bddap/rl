//! A real rapier-simulated, NN-driven giant crab — the trained RL body ("real Sally"), the ONLY
//! crab the game has (rl#114: there is no integer point-pursuer to fall back to). Armed by the
//! [`ExternalCrabArmed`] runtime gate on a SOLO round (a Host-alone Start, or scripted `--host`
//! that found no peer), and — since the GCR fold (rl#82) — on a NETWORKED round with synced
//! weights.
//!
//! # When it drives the crab (and why it's safe)
//! The game's [`crate::net::sim`] is a bit-deterministic INTEGER lockstep sim so two peers
//! evolve identically. A rapier crab is FLOAT physics, so it can drive a *networked* crab
//! without desyncing ONLY when every peer (a) loaded the same brain (the weights-digest
//! handshake) and (b) steps the body the identical number of physics steps per lockstep tick
//! (the deterministic [`crate::net::cadence::PhysicsCadence`], pumped by
//! `net::render::drive_lockstep`). [`crate::net::may_arm_external_crab`] is the gate that
//! enforces both. On a solo round there's no peer to desync against, so it always arms. A
//! networked-UNSYNCED round CANNOT arm and — with no integer fallback — REFUSES LOUDLY (the
//! windowed client fails with an actionable peer-mismatch message; see [`crate::net::render`])
//! rather than silently substituting a fake crab for Sally.
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
//! - For rendering, the giant blow-up is a RENDER-ONLY repose published into
//!   [`crate::bot::skin::CrabSkinRepose`] each game tick ([`publish_skin_repose`]) and applied
//!   by the skin to its render BONES only ([`crate::bot::skin::drive_bones`]) — shifting the rig
//!   to the game-world crab spot and scaling it to the giant. The physics `Transform`s are NEVER
//!   touched: bevy_rapier syncs a changed body `Transform` back into the physics body, so a
//!   "cosmetic" link shift teleported the body and crashed the solver. With no `sally.glb` the
//!   rl#5 procedural fallback rig shows as a Rapier debug-wireframe body (at the ~1 m arena frame).

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

/// Where the crab rig sits in the game world before the giant SCALE is applied — the 2D (XZ)
/// precursor that [`publish_skin_repose`] combines with [`crate::net::render::crab_render_scale`]
/// to build the full 3D [`crate::bot::skin::SkinRepose`]. A named pair (not a bare tuple) because
/// the two `Vec2`s are different KINDS — `shift` is an XZ delta, `pivot` an XZ world point — so
/// they can't be swapped at the use site. See [`ExternalCrabBridge::render_placement_m`].
struct CrabPlacement {
    /// Arena→game-world XZ translation for each part (`world_pos_m - last_carapace_m`).
    shift: Vec2,
    /// The crab's game-world ground point (XZ), about which the rig is scaled up.
    pivot: Vec2,
}

impl ExternalCrabBridge {
    /// Seed the bridge at the sim's crab spawn (so the NN crab begins where the
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

    /// Where the crab rig sits in the game world before SCALE, as a `(shift, pivot)` pair in
    /// metres on the XZ `Vec2` frame:
    /// - `shift` translates each part's raw arena `Transform` to the game-world crab spot
    ///   (`world_pos_m - last_carapace_m`), and
    /// - `pivot` is the crab's game-world ground point (`world_pos_m`), about which the rig is
    ///   scaled up to the giant (see [`publish_skin_repose`]).
    ///
    /// WHY the giant is render-only and built here, not baked into the body: the trained policy +
    /// rapier run at the ~1 m physics scale the policy learned under ([`crate::bot::body`]), and
    /// the lockstep determinism hash reads that raw pose — so the giant is a RENDER-ONLY blow-up
    /// applied to the skin AFTER the hash. With no integer silhouette to stand in (rl#114) this
    /// rig is the ONLY crab, so without the matching blow-up it renders as a ~1 m speck in a world
    /// framed for the [`crate::net::render::CRAB_RENDER_HEIGHT`] monster — the play-day "no crab
    /// visible" bug. `None` until the crab has been sampled once (so we never place against a
    /// stale/zero carapace).
    fn render_placement_m(&self) -> Option<CrabPlacement> {
        self.last_carapace_m.map(|c| CrabPlacement {
            shift: self.world_pos_m - c,
            pivot: self.world_pos_m,
        })
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
    /// crab back AT spawn, but the float body keeps walking and the bridge keeps its
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
    /// `world_pos_m`, because a restart moves the crab back to spawn.
    pub fn restart_to_spawn(&mut self, spawn: Pos) {
        self.world_pos_m = pos_to_m(spawn);
        self.last_carapace_m = None;
        self.settle = crate::training::systems::RESET_GRACE_TICKS;
    }
}

/// The per-tick bridge↔sim handshake, run BEFORE each [`Lockstep::try_advance`]: push the
/// real crab body's game position + facing into the sim (so this tick's grab/extraction
/// resolve against it), then refresh the player the crab hunts (nearest living). ONE definition
/// so the windowed driver
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
/// While absent (only behind the boot menu, before a round resolves — NumEnvs 0) the crab never
/// spawns and `drive_lockstep` never pumps physics or calls [`sync_external_crab`]. Once a round
/// resolves it is ARMED, or — if it can't be (networked-unsynced) — the round is REFUSED loudly
/// (rl#114, no integer fallback; see [`crate::net::render`]'s `ensure_round_installed`). The
/// scripted `Boot::Round` path inserts it armed at build; there is ONE gate for all paths. The
/// run-condition and render's reads gate on `Option<Res<ExternalCrabArmed>>::is_some()`.
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
    /// The sim's crab spawn, so the bridge starts the NN crab there.
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

        // Render-only: publish the giant-crab repose for the skin to apply to its render bones
        // ([`crate::bot::skin::drive_bones`]). This NEVER mutates the rapier-driven crab
        // `Transform`s — doing so teleported the body and crashed the client (see
        // [`crate::bot::skin::CrabSkinRepose`]). After `integrate_crab`, which finalises this
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

/// Render-only: publish the giant-crab repose into [`crate::bot::skin::CrabSkinRepose`] so the
/// skin draws the small ~1 m arena rig as the game-world GIANT — shifted to the crab's game spot
/// and scaled up about its ground point. The skin's [`crate::bot::skin::drive_bones`] applies this
/// to the render BONES only; this system NEVER mutates the rapier-driven `CrabBodyPart`
/// `Transform`s, because bevy_rapier reads a changed body `Transform` back into the physics body
/// — a "cosmetic" link shift teleported the body ~12 m/step and exploded the solver into a
/// NaN-motor crash (the play-day "there is no game to play"). So the blow-up is decoupled from
/// physics by construction: the policy, the colliders, and the lockstep hash all keep reading the
/// raw ~1 m pose regardless of `Visuals`, so a windowed and a headless peer stay bit-identical.
///
/// The scale is what makes the armed NN crab visible: with no integer silhouette to stand in
/// (rl#114) the skinned rig is the ONLY crab, and the world (spawn distance, camera framing, the
/// extraction pillar) is dimensioned for the giant — without the matching blow-up the crab would
/// render as a ~1 m speck ≥12 m away and read as "no crab". `None` until the crab is sampled once
/// (no stale/zero carapace) or for a degenerate recipe (no scale) — identity leaves the rig at 1×.
fn publish_skin_repose(
    bridge: Res<ExternalCrabBridge>,
    repose_out: Option<ResMut<crate::bot::skin::CrabSkinRepose>>,
) {
    // No skin loaded (no `sally.glb`) ⇒ no resource ⇒ nothing to repose (the rl#5 procedural rig
    // shows as the Rapier debug wireframe instead).
    let Some(mut out) = repose_out else {
        return;
    };
    out.0 = bridge.render_placement_m().and_then(|r| {
        // The SAME target-height fit the integer silhouette uses (one source, no drift): the
        // body's natural standing height is ~0.6 m, so a bare `CRAB_SCALE` multiply would render
        // the crab several× too small.
        crate::net::render::crab_render_scale().map(|scale| crate::bot::skin::SkinRepose {
            shift: Vec3::new(r.shift.x, 0.0, r.shift.y),
            // The ground pivot: the crab's game-world XZ at floor level (y=0). Scaling about it
            // grows the rig up and outward in place — the carapace rises to the giant height, the
            // feet (near y=0 in the arena rig) stay on the floor.
            pivot: Vec3::new(r.pivot.x, 0.0, r.pivot.y),
            scale,
        })
    });
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

/// Build the windowless bot+physics world the headless NN-crab probes step: the SAME stack
/// the training/tests use ([`crate::bot::test_util::headless_stack`], one crab in env 0) plus
/// [`ExternalCrabPlugin`] (the policy + arena↔game bridge) with the crab ARMED. Shared by the
/// single-peer [`run_headless_probe`] and the two-peer [`run_cross_peer_probe`] so both step the
/// identical dynamics the policy trained under, with no GPU/display — one app-construction, no
/// drift between the two harnesses (the manual's "one implementation per thing"). The caller owns
/// the [`Lockstep`] driving and seeding; this only stands up the rapier NN body.
fn headless_nn_crab_app(checkpoint_dir: &std::path::Path, crab_spawn: Pos) -> bevy::app::App {
    use crate::bot::test_util::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
    };

    // GCR#82: pin every parallel-reduction pool to one thread BEFORE building the app, so the
    // rapier physics AND the burn matmul inference run in a single fixed float-op order — the
    // precondition for the crab to evolve bit-identically across processes. Today the unpinned
    // path happens to match cross-process too (rapier is serial — `parallel` off — and the
    // `[≤16,77]` NN matmul stays under matrixmultiply's threading threshold), but that's
    // INCIDENTAL: a larger NN (the stated next direction) would parallelize the matmul and a
    // multi-threaded ECS executor could reorder accumulation, silently reintroducing divergence.
    // Pinning makes determinism hold BY CONSTRUCTION. Same recipe the trainer uses (shared
    // `pin_single_thread_pools`). Idempotent across the two in-process peers.
    pin_single_thread_pools();

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
    // Force the ECS executor serial now that the plugin systems are wired — fixes the system run
    // ORDER, the second half of the determinism guarantee alongside the pinned pools. A system the
    // single-peer probe adds later (`probe_step`) lands in the already-serial schedule and inherits
    // it, so this one call covers both probe drivers.
    force_serial_schedules(&mut app);
    app
}

/// Run the NN crab headlessly for `ticks` sim steps and return the logged samples, via
/// [`headless_nn_crab_app`] + a hand-driven lockstep — so the crab steps the exact dynamics the
/// policy trained under, with no GPU/display. `checkpoint_dir` is the trained policy; `seed` seeds
/// the round (same seed twice ⇒ identical samples, the determinism check). The local player holds
/// still so a shrinking `dist_to_prey_m` proves the crab walks toward it under the policy.
///
/// NOTE: this single-peer probe steps the body one physics step per `app.update()` (a
/// walking/reproducibility sanity check, where the absolute gait speed doesn't matter). The
/// cross-peer determinism GATE [`run_cross_peer_probe`] instead steps at the PRODUCTION
/// [`crate::net::cadence::PhysicsCadence`] (2–3 steps/tick), matching what networked peers run.
pub fn run_headless_probe(
    checkpoint_dir: &std::path::Path,
    seed: u64,
    ticks: u64,
    log_every: u64,
) -> Vec<ProbeSample> {
    let me = PlayerId(0);
    let ls = Lockstep::new(seed, &[me], me);
    let crab_spawn = ls.sim().crab().pos();
    // The crab is externally driven for the whole probe (we own its position). Seed the pose with
    // the crab's CURRENT spawn pose/yaw — writing back what's already there, so this is a no-op on
    // sim state. Seed with a zero digest; the first post-step `hash_crab_physics` fills it before
    // the first `sync_external_crab` push, so the seeded value is never the one cross-checked.
    let mut ls = ls;
    let crab = ls.sim().crab();
    ls.set_external_crab_pose(crab.pos(), crab.yaw(), 0);

    let mut app = headless_nn_crab_app(checkpoint_dir, crab_spawn);
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

// ---------------------------------------------------------------------------
// Cross-peer NN-crab determinism harness (the decisive GCR #82 gate)
// ---------------------------------------------------------------------------

/// One applied tick of the two-peer probe: the tick number and the closing `state_hash` each
/// peer's lockstep computed for it. The two peers stayed deterministic iff `hash_a == hash_b`
/// for every tick — the float rapier NN crab evolved bit-identically on both.
#[derive(Clone, Copy, Debug)]
pub struct XPeerTick {
    pub tick: u64,
    pub hash_a: u64,
    pub hash_b: u64,
}

/// Result of [`run_cross_peer_probe`]: the per-tick hash pair plus the count of lockstep
/// desync FAULT EVENTS the peers' OWN cross-check (peer-advertised hashes in each [`TickMsg`])
/// raised. A pass is `faults == 0` AND every `hash_a == hash_b` — two independent checks of the
/// same property, from outside (the hash diff) and inside (the lockstep desync check). `faults`
/// counts events, not distinct diverged ticks: one divergence can surface in both arrival-order
/// halves of the cross-check, so it's a belt-and-suspenders signal — the per-tick hash diff is
/// the authoritative check.
pub struct XPeerResult {
    pub ticks: Vec<XPeerTick>,
    pub faults: usize,
}

impl XPeerResult {
    /// First tick whose two hashes disagree, if any — the point determinism broke.
    pub fn first_divergence(&self) -> Option<XPeerTick> {
        self.ticks.iter().copied().find(|t| t.hash_a != t.hash_b)
    }

    /// Both checks clean: no per-tick hash disagreed and the lockstep raised no desync fault.
    pub fn is_deterministic(&self) -> bool {
        self.faults == 0 && self.first_divergence().is_none()
    }
}

/// Push a headless peer's freshly-integrated NN-crab pose + physics digest into its OWN
/// lockstep and refresh the player it hunts — the SAME per-tick handshake
/// [`crate::net::render`]'s `drive_lockstep` runs for the windowed peer ([`sync_external_crab`]),
/// reaching into the app's [`ExternalCrabBridge`] resource. Called once per applied tick.
fn sync_peer(app: &mut bevy::app::App, ls: &mut Lockstep) {
    let mut bridge = app.world_mut().resource_mut::<ExternalCrabBridge>();
    sync_external_crab(ls, &mut bridge);
}

/// Run the real rapier-NN crab as the giant crab on TWO independent in-process peers and
/// return their per-tick hash pair — the decisive cross-peer determinism gate for GCR #82.
///
/// Each peer is a SEPARATE headless bot+physics world ([`headless_nn_crab_app`]) stepping its
/// OWN float rapier crab under the trained policy, plus its OWN integer [`Lockstep`] with the
/// crab handed to external control. Per applied tick the harness mirrors
/// [`crate::net::render`]'s `drive_lockstep` for both peers: pump the body the deterministic
/// [`PhysicsCadence`] number of physics steps for this tick ([`pump_fixed_steps`]), push that
/// peer's crab pose + weights-folded physics digest into its lockstep, then EXCHANGE the two
/// peers' inputs (each records the other's exact [`TickMsg`]) and advance one tick on each. Using
/// the SAME cadence path as production is what makes this a faithful proxy — the body is stepped
/// at the real 64:30 ratio (2–3 steps/tick), not some probe-only rate, so the hashed pose is the
/// one networked peers actually compute. The two peers move their PLAYERS differently (divergent
/// but faithfully-exchanged input), so the test exercises a real two-player round — yet their
/// giant crabs must reach byte-identical poses.
///
/// If every `hash_a == hash_b` and the lockstep raises no desync fault, the float NN crab is the
/// deterministic multiplayer crab on this hardware (same-arch, `enhanced-determinism` on; the
/// cross-ARCH case stays untested here — peers must run the same-arch binary deploy carries, since
/// there is no integer fallback any more, rl#114). A single diverging tick is the netcode-rethink
/// trigger. Same `(checkpoint, seed, ticks)` ⇒ identical result (the
/// inputs are a deterministic function of the tick index).
pub fn run_cross_peer_probe(checkpoint_dir: &std::path::Path, seed: u64, ticks: u64) -> XPeerResult {
    use crate::net::cadence::PhysicsCadence;
    use crate::net::render::{park_fixed_auto_pump, pump_fixed_steps};

    let p0 = PlayerId(0);
    let p1 = PlayerId(1);
    let peers = [p0, p1];

    // Both peers start from the SAME seed → identical integer sim → identical crab spawn, so
    // their float crabs begin at the same game-world pose.
    let crab_spawn = {
        let ls = Lockstep::new(seed, &peers, p0);
        ls.sim().crab().pos()
    };

    let mut app_a = headless_nn_crab_app(checkpoint_dir, crab_spawn);
    let mut app_b = headless_nn_crab_app(checkpoint_dir, crab_spawn);
    // Park the wall-clock auto-pump on both, then one update to run Startup (spawn the crab) with
    // ZERO physics steps — from here only `pump_fixed_steps` advances the body, at the cadence,
    // exactly as `add_external_nn_crab` + `drive_lockstep` do in production.
    park_fixed_auto_pump(app_a.world_mut());
    park_fixed_auto_pump(app_b.world_mut());
    app_a.update();
    app_b.update();

    // Each peer drives its OWN lockstep (its own `me`), with the crab armed + seeded at spawn.
    // Seed a zero digest; the first post-step `hash_crab_physics` fills the real one before the
    // first push, so the seeded value is never cross-checked.
    let mut ls_a = Lockstep::new(seed, &peers, p0);
    let mut ls_b = Lockstep::new(seed, &peers, p1);
    {
        let crab = ls_a.sim().crab();
        ls_a.set_external_crab_pose(crab.pos(), crab.yaw(), 0);
    }
    {
        let crab = ls_b.sim().crab();
        ls_b.set_external_crab_pose(crab.pos(), crab.yaw(), 0);
    }
    // The physics-step cadence per peer — `Default`-started and advanced once per applied tick on
    // each, so both peers run the identical step count for every tick (the GCR fold's core
    // invariant). Mirrors `drive_lockstep`'s `Local<PhysicsCadence>`.
    let mut cad_a = PhysicsCadence::default();
    let mut cad_b = PhysicsCadence::default();

    let mut out = Vec::new();
    let mut faults = 0usize;
    let mut issue = 0u64;
    // Step until BOTH peers have applied `ticks` ticks. Each iteration applies exactly one sim
    // tick per peer (the apply cursor leads by INPUT_DELAY, so the tick is always ready — warmup
    // or both inputs exchanged this iteration), and pumps that tick's cadence physics steps first.
    while ls_a.sim().tick() < ticks || ls_b.sim().tick() < ticks {
        // 1. Pump each peer's body the cadence steps for this tick (uses the hunt target each
        //    bridge set last iteration). One `pump_fixed_steps` call = `steps` `PHYSICS_DT` steps.
        pump_fixed_steps(app_a.world_mut(), cad_a.steps_for_next_tick());
        pump_fixed_steps(app_b.world_mut(), cad_b.steps_for_next_tick());

        // 2. Push each peer's freshly stepped crab pose + digest into its own lockstep.
        sync_peer(&mut app_a, &mut ls_a);
        sync_peer(&mut app_b, &mut ls_b);

        // 3. Each peer issues a DETERMINISTIC but distinct input, then they EXCHANGE — peer A
        //    records B's exact message and vice versa, exactly as the wire transport delivers it.
        //    Divergent player motion makes the round real; the exchange keeps both sims fed the
        //    identical {A,B} input set so any hash difference is the CRAB's float physics, not the
        //    players'.
        let t = issue as f32 * 0.1;
        issue += 1;
        let msg_a = ls_a.submit_local_input(Input::from_axes(t.cos(), t.sin()));
        let msg_b = ls_b.submit_local_input(Input::from_axes(-t.sin(), t.cos()));
        if ls_a.record_remote(p1, msg_b).is_some() {
            faults += 1;
        }
        if ls_b.record_remote(p0, msg_a).is_some() {
            faults += 1;
        }

        // 4. Advance one tick on each. Count any desync the lockstep's own cross-check raises.
        let tick_a = ls_a.advance_one().map(|f| (ls_a.last_applied(), f));
        let tick_b = ls_b.advance_one().map(|f| (ls_b.last_applied(), f));
        if let (Some((Some(ca), fa)), Some((Some(cb), fb))) = (tick_a, tick_b) {
            faults += fa.len() + fb.len();
            // Both peers advanced one tick this iteration, so they're on the same tick; enforce
            // it rather than trust it, then pair the two peers' hashes for that tick.
            debug_assert_eq!(ca.tick, cb.tick, "peers advanced out of lockstep");
            out.push(XPeerTick {
                tick: ca.tick,
                hash_a: ca.hash,
                hash_b: cb.hash,
            });
        }
    }

    XPeerResult { ticks: out, faults }
}
