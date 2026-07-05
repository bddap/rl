//! Real rapier-simulated, NN-driven giant crabs — the trained RL bodies ("real Sally"), the ONLY
//! crabs the game has (no integer point-pursuer to fall back to). A round runs one crab per
//! BRAIN BINDING (rl#200 multi-architecture: several brains, one GCR instance), each in its own
//! crab-world env — crab index == env id == the sim's crab index, one identity end to end.
//! Armed by the [`ExternalCrabArmed`] runtime gate on a SOLO round (a Host-alone Start, or
//! scripted `--host` that found no peer) and on a NETWORKED round with synced crab assets.
//!
//! # When it drives the crabs (and why it's safe)
//! Host-authoritative MP ([[mp-minecraft-model]]): only the HOST pumps these float bodies; a
//! remote client adopts [`crate::snapshot::CoreSnapshot`]s and renders the host's broadcast
//! articulation, simulating nothing. So per-crab brains never touch client-side simulation —
//! what must still agree across peers is the crab MODEL asset (colliders/rig are derived from
//! it on every peer), which [`crate::may_arm_external_crab`]'s asset gate enforces. A
//! networked round that can't agree REFUSES LOUDLY (an actionable peer-mismatch message)
//! rather than silently substituting a fake crab. The HOST self-validates every binding
//! fail-loud at launch (`game`'s checkpoint resolution): a checkpoint that is missing, refused
//! by its envelope, or built for the wrong rig aborts the launch naming the crab, dir, and
//! reason — an inert rest-pose Sally is never shipped silently.
//!
//! # How it works
//! Each trained policy ([`crab_world::play::Policy`], loaded from its binding's checkpoint) is a
//! *locomotion+reach* brain: its observation includes the touch target as a vector in
//! the carapace's local frame, and it has learned to WALK the body toward that vector
//! (the `ckpt-best.locomotion` weights walk + reach). We exploit that directly, per crab:
//!
//! - Every crab lives in its own env of ONE shared rapier world (the [`crab_world::bot`] /
//!   [`crab_world::physics`] stack) on the OPEN inference field
//!   ([`crab_world::physics::Arena::OpenField`], rl#209): an unbounded ground with no
//!   walls, so per-round travel is unlimited. Training keeps its ±10 m walled box;
//!   the ground contact dynamics are identical (same y=0 surface, same material).
//! - Each control step we place each crab's target at its nearest living game player's ACTUAL
//!   position, expressed in that crab's arena frame (`carapace + (player − crab)`), at any
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
//! - We integrate each carapace's horizontal displacement each game tick and add it to
//!   that crab's game-world position, which we write back into the sim with
//!   [`Sim::set_external_crab_pose`]. Grabs / extraction / win-loss then resolve against the
//!   REAL crab bodies, exactly as the prompt asks.
//! - For rendering, each rig is carried to its crab's game-world spot by a RENDER-ONLY repose
//!   published into [`crab_world::bot::skin::CrabSkinRepose`] (per env) each game tick and
//!   applied to the render BONES only; physics `Transform`s are never touched, and the giant
//!   FEEL comes from the R-shrunk human world — see [`publish_skin_repose`].

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::sim::Pos;
use crab_world::Visuals;
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};
use crab_world::bot::sensor::CrabTargets;
use crab_world::bot::{BotSet, CrabSpawns};
use crab_world::crab_view::CrabBrainLabels;
use crab_world::play::Policy;

/// Height (m) of the walk target placed at the player's position. A low ground-ish Y inside
/// the policy's trained target-height band (`TARGET_Y_MIN..MAX` ≈ 0.15..0.7 in training) so
/// a crab reads it as a walk-up-and-reach target. The hunt target is a planar (XZ) player
/// position with no carried height, so we pin a single ground-level grab height here.
const CLAW_TARGET_Y: f32 = 0.3;

/// The per-env brain bindings, in crab-index order — `policies.0[i]` drives env `i`'s crab
/// (rl#200). NonSend (the burn brain isn't `Send`), inserted by [`ExternalCrabPlugin`] from its
/// validated checkpoint dirs; ONE list, indexed exactly like `CrabActions.envs`, so a
/// binding/env mismatch is impossible by construction rather than reconciled at runtime.
pub struct CrabPolicies(pub Vec<Policy>);

/// Resource: the live bridge state between the rapier NN crabs and the integer game sim — one
/// [`CrabBridge`] per binding, in crab-index order (index == env id == sim crab index).
/// Non-trivial state (float world positions + last carapace samples) that must persist across
/// ticks, so it can't live on the `Copy` integer `Sim`.
#[derive(Resource)]
pub struct ExternalCrabBridge {
    crabs: Vec<CrabBridge>,
}

/// One crab's bridge state (see [`ExternalCrabBridge`]).
struct CrabBridge {
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
    /// Game-world point (metres) this crab is currently hunting — its nearest living
    /// player, refreshed each tick by the driver. `None` when every player is
    /// down (the crab then holds position). Read by [`set_crab_walk_target`] to aim the
    /// policy.
    hunt_target_m: Option<Vec2>,
    /// This tick's digest of the crab's full rapier physics state XORed with its
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

/// Where a crab rig sits in the game world — the 2D (XZ) precursor that [`publish_skin_repose`]
/// combines with the render-frame scale ([`crate::render::world_render_scale`]) to build the full
/// 3D [`crab_world::bot::skin::SkinRepose`]. A named pair (not a bare tuple) because
/// the two `Vec2`s are different KINDS — `shift` is an XZ delta, `game_spot` an XZ world point —
/// so they can't be swapped at the use site. See [`CrabBridge::render_placement_m`].
struct CrabPlacement {
    /// Arena→game-world XZ translation for each part (`world_pos_m - last_carapace_m`).
    shift: Vec2,
    /// The crab's game-world ground point (XZ).
    game_spot: Vec2,
}

impl CrabBridge {
    /// Seed one crab's bridge at its sim spawn (so the NN crab begins where the
    /// round placed it, MIN_CRAB_SPAWN_DISTANCE from the players).
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
    fn world_pos(&self) -> Pos {
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

    /// Where the crab rig sits in the game world, as a `(shift, game_spot)` pair in metres on
    /// the XZ `Vec2` frame:
    /// - `shift` translates each part's raw arena `Transform` to the game-world crab spot
    ///   (`world_pos_m - last_carapace_m`), and
    /// - `game_spot` is the crab's game-world ground point (`world_pos_m`).
    ///
    /// Feeds the render-only repose — see [`publish_skin_repose`] for why placement never
    /// touches physics. `None` until the crab has been sampled once (so we never place against
    /// a stale/zero carapace).
    fn render_placement_m(&self) -> Option<CrabPlacement> {
        self.last_carapace_m.map(|c| CrabPlacement {
            shift: self.world_pos_m - c,
            game_spot: self.world_pos_m,
        })
    }

    /// Re-seed this crab's bridge to the round's spawn after a deterministic sim RESTART
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
    fn restart_to_spawn(&mut self, spawn: Pos) {
        self.world_pos_m = pos_to_m(spawn);
        self.last_carapace_m = None;
        self.settle = crab_world::bot::RESET_GRACE_TICKS;
    }
}

impl ExternalCrabBridge {
    /// Seed one bridge per crab at the sim's crab spawns, in crab-index order.
    pub fn new(spawns: &[Pos]) -> Self {
        Self {
            crabs: spawns.iter().map(|&s| CrabBridge::new(s)).collect(),
        }
    }

    /// How many crabs this bridge drives — the binding count, one per env.
    pub fn crab_count(&self) -> usize {
        self.crabs.len()
    }

    /// Every crab's freshly-integrated pose + physics digest, in crab-index order — what the
    /// authoritative server injects into the tick ([`crate::server::Server::step_next`]).
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

    /// Set the game-world point (crab `idx`'s nearest living player's [`Pos`]) that crab hunts,
    /// or `None` when every player is down. Called each game tick by the driver, which
    /// owns the sim and so knows the prey; [`set_crab_walk_target`] reads it to aim the policy.
    pub fn set_hunt_target(&mut self, idx: usize, prey: Option<Pos>) {
        self.crabs[idx].hunt_target_m = prey.map(pos_to_m);
    }

    /// Re-seed every crab's bridge to its round spawn after a deterministic sim RESTART — see
    /// [`CrabBridge::restart_to_spawn`]. `spawns` comes from the post-restart sim's crab set,
    /// so it is index-aligned by construction; a count mismatch is a bridge/sim drift bug and
    /// panics rather than re-seeding a subset.
    ///
    /// DETERMINISM: this fires off the sim's restart EDGE (returned by the authoritative
    /// step — a restart never rewinds the tick, rl#204) — the same edge the cadence reset
    /// hangs off in [`crate::render`]'s `drive_lockstep`. Under host-authority only the
    /// server-auth peer steps (and so observes the edge); a remote client renders the host's
    /// articulation — its own armed bodies are never pumped, so there is nothing to re-seed.
    /// Unlike `CrabRescued` (a float-body teleport that leaves the game position put) this DOES
    /// move `world_pos_m`, because a restart moves the crabs back to spawn.
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

/// Cold-respawn every armed crab's rapier body at the round-boundary RESTART edge — despawn each
/// env's crab entity tree and rebuild it fresh from its spawn, via the SAME
/// [`crab_world::bot::respawn_crab`] path the non-finite rescue uses (one respawn implementation,
/// no parallel reset to drift). [`ExternalCrabBridge::restart_to_spawns`] resets only the bridge's
/// bookkeeping; the rapier solver keeps WARM contact/warm-start state across a bare restart, so
/// re-seeding alone would make the new round's physics path-dependent on the round before it (and
/// visible in the per-tick external-crab digests folded into `state_hash`). Dropping the trees and
/// rebuilding starts every round from the IDENTICAL from-boot solver state — a respawn, not a
/// teleport of the old bodies. Two callers, both round starts: the RESTART edge in
/// [`crate::render`]'s `drive_lockstep`, and the round install after a disconnect return
/// (rl#203), where the previous round's warm bodies persist across the menu.
///
/// Exclusive-world (the driver edge holds `&mut World`): collect each env's parts, lift `CrabAssets`
/// out with `resource_scope` so a temporary [`Commands`] can borrow the world, and apply the
/// despawn+spawn immediately — before the next `pump_fixed_steps`, so the first post-restart physics
/// step lands on the fresh bodies. Each spawn origin is the env's recorded [`CrabSpawns`] entry (the
/// same point the initial spawn used), keeping the cold bodies bit-identical to a from-boot spawn.
pub(crate) fn cold_respawn_armed_crab(world: &mut World) {
    use bevy::ecs::world::CommandQueue;

    let mut by_env: std::collections::BTreeMap<usize, Vec<Entity>> = Default::default();
    for (e, env) in world
        .query_filtered::<(Entity, &CrabEnvId), With<CrabBodyPart>>()
        .iter(world)
    {
        by_env.entry(env.0).or_default().push(e);
    }
    // Nothing to rebuild before any body exists (e.g. a restart in the first frames) — the guarded
    // initial spawn will place them; respawning absent crabs would be a no-op despawn + duplicates.
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

/// The per-tick bridge↔sim handshake for a sim driven DIRECTLY (the headless probe): push each
/// real crab body's game position + facing into the sim (so this tick's grab/extraction
/// resolve against them), then refresh the player each crab hunts (its nearest living). The
/// windowed driver makes the same push through the authoritative server instead
/// ([`crate::server::CrabPose`] into `Server::step_next`).
pub fn sync_external_crab(sim: &mut crate::sim::Sim, bridge: &mut ExternalCrabBridge) {
    for (idx, pose) in bridge.crab_poses().into_iter().enumerate() {
        sim.set_external_crab_pose(idx, pose.pos, pose.yaw, pose.digest);
    }
    for idx in 0..bridge.crab_count() {
        let prey = sim.nearest_living_player_pos(idx);
        bridge.set_hunt_target(idx, prey);
    }
}

/// Recompute each env's full rapier physics digest (every actuated body's pose+velocity bits,
/// via the shared [`crab_world::bot::physics_digest`]) XORed with that env's loaded policy's
/// weights digest, and store it on the bridge for the next pose push into the sim's state hash
/// ([`sync_external_crab`] / [`crate::server::CrabPose`]).
/// Runs each step AFTER the physics writeback + [`integrate_crab`], so it captures
/// this tick's settled state. Folding in the weights digest makes two hosts running different
/// brains desync on the first tick, and folding in the full articulated state makes a float
/// divergence a detected desync, not just a 2D-pose one.
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
    // The zip below would silently 0-digest tail crabs on a length drift — the exact class
    // these digests exist to catch — so refuse it instead (both vecs are built one-per-binding
    // by [`ExternalCrabPlugin`]; a mismatch is a wiring bug).
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
        // Fold the weight identity into the state hash: `None` (rest pose / diagnostic brain,
        // no checkpoint) contributes 0 — a no-op XOR — exactly as the old `0` sentinel did.
        crab.phys_digest = phys ^ policy.weights_digest().map_or(0, std::num::NonZeroU64::get);
    }
}

/// Runtime gate for the external NN crabs, as a presence marker: the resource exists iff they are
/// armed. The boot-menu path adds the whole NN stack at app build (plugins can't be
/// added later) but does NOT know yet how the round resolves — so every NN system is gated on
/// this, present ONLY once the round arms the crabs ([`crate::may_arm_external_crab`]: solo
/// always, networked only with synced crab assets). While absent (only behind the boot menu,
/// before a round resolves — NumEnvs 0) no crab spawns and `drive_lockstep` never pumps physics
/// or injects poses. Once a round resolves it is ARMED, or — if it can't be
/// (networked-unsynced) — the round is REFUSED loudly (no integer fallback; see
/// [`crate::render`]'s `ensure_round_installed`). The scripted `Boot::Round` path inserts it
/// armed at build; there is ONE gate for all paths. The run-condition and render's reads gate on
/// `Option<Res<ExternalCrabArmed>>::is_some()`.
#[derive(Resource)]
pub struct ExternalCrabArmed;

/// Run condition: the external NN crabs are armed. Gates every [`ExternalCrabPlugin`] system so
/// they idle (zero cost beyond the check) until the round arms them.
fn external_crab_armed(active: Option<Res<ExternalCrabArmed>>) -> bool {
    active.is_some()
}

/// Arm the giant NN crabs: insert the [`ExternalCrabArmed`] gate. Every arm site — the
/// windowed `Boot::Round` + menu-transition clients, the screenshot path, the headless probe —
/// goes through this one path. Each walk target is a player's actual position (no per-peer
/// tunable steering it), so there is nothing solo-vs-networked to reconcile here.
pub fn arm(world: &mut World) {
    world.insert_resource(ExternalCrabArmed);
    // The armed crabs ARE the trained Sallys: a `rescue_nonfinite_crabs` fire is a
    // physics-correctness FAULT to surface LOUDLY (error log + aggregated telemetry, hard panic
    // in debug/test), never a silent catch-and-respawn. Marking it HERE — the one arm
    // path every site funnels through — means no caller can arm Sally and forget to make her
    // rescue loud; training (which never calls `arm`) keeps the quiet routine-terminator rescue.
    world.insert_resource(crab_world::bot::CrabRescueIsFault);
}

/// Plugin: the external NN crabs. Adds the loaded per-env policies + the bridge, and the systems
/// that (a) spawn the crabs once armed, (b) aim + drive each policy, (c) integrate each body's
/// walk into its game crab. The caller must ALSO have added the bot/physics stack
/// (`RapierPhysicsPlugin` + `PhysicsWorldPlugin` + `BotPlugin`) to the same app — see
/// [`crate::render`]'s `add_external_nn_crab` — so the rigs actually exist and step. Every
/// system is gated on [`ExternalCrabArmed`]: the scripted `Boot::Round` path arms it at build;
/// the boot menu arms it only once the round resolves armable.
pub struct ExternalCrabPlugin {
    /// The brain bindings, in crab-index order: each entry is a directory one crab's brain
    /// (`brain.bin`) + normalizer (`normalizer.bin`) load from — env `i` runs `dirs[i]`
    /// (rl#200). Configurable so deploy can point each crab at its chosen checkpoint. At least
    /// one; the game entry points validate every dir fail-loud BEFORE building this plugin.
    pub checkpoint_dirs: Vec<std::path::PathBuf>,
    /// The sim's crab spawns, index-aligned with `checkpoint_dirs`, so each bridge starts its
    /// NN crab where the round placed it.
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
                // A crab only moves when a fitting checkpoint armed it; otherwise it stands
                // inert. `Policy::load` already logged WHY (a loud `error!` for a refused
                // wrong-rig checkpoint) — and every game entry point hard-fails on a bad
                // checkpoint before this plugin is even built, so a non-arming here is the
                // legitimate "no brain yet" rest pose.
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

        // Spawn the crabs the first frame the gate is active. On the scripted solo
        // path the gate is true from build, so this spawns on frame 1 exactly as before; on
        // the menu path it spawns only once the solo round is chosen. Reuses the bot stack's
        // own [`spawn_initial_crabs`] (one spawn path, no drift) after sizing NumEnvs to the
        // binding count.
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

        // Aim + drive the policies in the Sense→Think→Act chain. The walk targets MUST be
        // placed BEFORE Sense: `build_observation` (in BotSet::Sense) reads `CrabTargets`
        // to build the target-relative observation each policy steers by, so a target set
        // after Sense would only reach the policy a tick late (and tick 0 not at all). Then
        // run the policies in Think; the actuator (Act) + rapier step follow from the bot
        // stack the caller added. Gated on the armed flag so the policies are inert until armed.
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
        // After physics has moved the bodies this fixed step, fold their displacement into
        // the game-world crab positions. Runs in FixedUpdate (one physics step per run) so
        // every bit of walk is captured, not just the frames the render samples.
        app.add_systems(
            FixedUpdate,
            integrate_crab
                .after(PhysicsSet::Writeback)
                .run_if(external_crab_armed),
        );
        // Digest each step's settled rapier state (+ each policy's weights) for the
        // desync check. After the writeback AND `integrate_crab` so it reads the final
        // post-step poses, and before the driver pushes the bridge into the sim
        // (`sync_external_crab`) each tick. Gated on the active flag so it never runs on a
        // round without the NN crabs.
        app.add_systems(
            FixedUpdate,
            hash_crab_physics
                .after(integrate_crab)
                .after(PhysicsSet::Writeback)
                .run_if(external_crab_armed),
        );

        // Render-only: publish each giant-crab repose for the skin to apply to its render bones
        // ([`crab_world::bot::skin::drive_bones`]). This NEVER mutates the rapier-driven crab
        // `Transform`s — doing so teleported the body and crashed the client (see
        // [`crab_world::bot::skin::CrabSkinRepose`]). After `integrate_crab`, which finalises this
        // step's game-world positions the reposes are built from. Gated on Visuals — headless never
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

/// Size `NumEnvs` to the binding count the first time the crabs are armed, so the shared
/// [`crab_world::bot::spawn_initial_crabs`] (run right after, gated on the same armed flag) spawns
/// exactly one crab per binding. This does NOT spawn — it sizes the env set so the shared spawner
/// does. Never shrinks (the headless probes pre-size their own env). The spawn-once guard is
/// [`crab_not_yet_spawned`].
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

/// Run condition: no crab body has spawned yet — so [`crab_world::bot::spawn_initial_crabs`]
/// (which appends) fires exactly once on the menu solo path rather than every frame the
/// gate is active. The scripted solo path's crabs also spawn through here, so there is one
/// spawn trigger, not two.
fn crab_not_yet_spawned(crabs: Query<(), With<CrabCarapace>>) -> bool {
    crabs.is_empty()
}

/// Aim each policy at its player's ACTUAL position, in that crab's ARENA frame (the frame the
/// body + the observation live in). The target is the carapace plus the true game-world
/// offset to the hunted player (`carapace + (player − crab)`) — no lead, no treadmill — so the
/// observation ([`crab_world::bot::sensor::build_observation`]) re-expresses the REAL player
/// offset in carapace-local axes and the policy's WEIGHTS supply the approach (the bitter
/// lesson). A player past the trained band (`[1.5, ~9] m`) saturates the obs normalizer clamp,
/// so the crab reads "player far, that way" and walks toward it without resolving exact
/// distance; on the open inference field (rl#209) it keeps walking until the gap re-enters
/// the band — see the module header. With no living target (all players down) the target is
/// dropped and that crab holds.
fn set_crab_walk_target(
    bridge: Res<ExternalCrabBridge>,
    spawns: Res<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
) {
    for (idx, crab) in bridge.crabs.iter().enumerate() {
        let Some(slot) = targets.envs.get_mut(idx) else {
            return; // envs not sized yet (pre-spawn frame) — nothing to aim
        };
        // The hunted player's offset from this crab, in GAME-world XZ. `None` ⇒ nobody to chase.
        let Some(hunt) = crab.hunt_target_m else {
            *slot = None; // nobody to chase → no target → policy holds (rest pose)
            continue;
        };
        let to_prey = hunt - crab.world_pos_m;
        if to_prey.length_squared() < 1e-6 {
            continue; // already on the prey; leave the last target so it doesn't flicker
        }

        // Carapace position in this crab's arena frame. Fall back to the env's spawn origin
        // before the body exists.
        let origin = spawns.0.get(idx).copied().unwrap_or(Vec3::ZERO);
        let carapace = carapace_q
            .iter()
            .find(|(env, _)| env.0 == idx)
            .map(|(_, t)| t.translation)
            .unwrap_or(origin);

        // Plant the target at the player's real position in the arena frame: the carapace plus
        // the full game-world offset to the player, at any distance. `CLAW_TARGET_Y` is a
        // ground-level grab height (the planar hunt target carries no height of its own).
        let target = Vec3::new(
            carapace.x + to_prey.x,
            CLAW_TARGET_Y,
            carapace.z + to_prey.y,
        );
        *slot = Some(target);
    }
}

/// Run each loaded policy on its env's observation and write that env's action — the same
/// deterministic mean action the demo's `policy_step` uses ([`Policy::act`]), so the
/// external crabs and the demo share one inference path. The actuator (bot stack) turns the
/// actions into joint torques next in the chain.
fn run_crab_policy(
    policies: NonSend<CrabPolicies>,
    mut bridge: ResMut<ExternalCrabBridge>,
    obs: Res<crab_world::bot::sensor::CrabObservation>,
    mut actions: ResMut<crab_world::bot::actuator::CrabActions>,
) {
    // Same length guard as `hash_crab_physics`: a drift would silently freeze tail crabs.
    assert_eq!(
        policies.0.len(),
        bridge.crabs.len(),
        "one policy per bridged crab"
    );
    for (idx, (policy, crab)) in policies.0.iter().zip(bridge.crabs.iter_mut()).enumerate() {
        // Spawn settle: hold zero torque while a fresh crab drops onto its legs (see the
        // `settle` field doc), then the policy takes over.
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

/// After a physics step, fold each carapace's horizontal walk into its game-world crab
/// position (the displacement since last step, 1:1) and the walk heading into the yaw,
/// storing both on the bridge. The render-loop owner ([`crate::render`]) then pushes
/// them into the sim each game tick via [`sync_external_crab`] / the server's crab poses.
fn integrate_crab(
    mut bridge: ResMut<ExternalCrabBridge>,
    mut rescued: MessageReader<crab_world::bot::CrabRescued>,
    carapace_q: Query<(&CrabEnvId, &Transform, &Velocity), With<CrabCarapace>>,
) {
    // A force-respawn (BotPlugin's `rescue_nonfinite_crabs`) teleported some env's crab back
    // to its arena spawn. Forget that crab's pre-blowup carapace so we DON'T difference the new
    // spawn against it (that would add a multi-metre false step to the game crab); the
    // next frame re-seeds `last_carapace_m` from the fresh pose. Re-settle too, so the
    // rebuilt body gets its drop/plant grace before we trust its motion again.
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
            continue; // a blown-up body (rescued next tick) — don't smear NaN into the position
        }
        let here = Vec2::new(t.translation.x, t.translation.z);
        // Accumulate the walk only when the body is under the policy (settle spent) AND we have
        // a prior sample to difference. The carapace barely translates during the settle DROP
        // (it's planting, not striding), so always tracking `last_carapace_m` — even through
        // settle — means the first post-settle accumulation differences against a pose ~at rest:
        // sub-millimetre of settle motion leaks in, which is negligible and (verified) does not
        // perturb the emergent walk, whereas deferring the seed a tick noticeably changed the
        // chaotic gait. `last_carapace_m` is None only right after spawn/rescue (re-seeded here).
        if crab.settle == 0
            && let Some(prev) = crab.last_carapace_m
        {
            crab.world_pos_m += here - prev;
        }
        crab.last_carapace_m = Some(here);

        // Facing from the horizontal walk velocity (when actually moving), so the rendered
        // crab + the sim yaw point where it's going. atan2(x, z) matches the sim's
        // turn-unit convention (see trig::atan2_turns: +Z forward, +X right).
        let v = Vec2::new(vel.linear.x, vel.linear.z);
        if v.length_squared() > 1e-4 {
            let radians = v.x.atan2(v.y); // (x, z) → heading
            crab.yaw_turns = crate::sim::trig_client::radians_to_turns(radians);
        }
    }
}

/// Render-only: publish each crab's repose into [`crab_world::bot::skin::CrabSkinRepose`] (keyed
/// by env) so the skin draws each arena rig at its crab's game spot. The skin's
/// [`crab_world::bot::skin::drive_bones`] applies these to the render BONES only; this system
/// NEVER mutates the rapier-driven `CrabBodyPart` `Transform`s, because bevy_rapier reads a
/// changed body `Transform` back into the physics body — a "cosmetic" link shift teleports the
/// body and explodes the solver. So the render placement is decoupled from physics by
/// construction: the policies, the colliders, and the state hash all keep reading the raw ~1 m
/// poses regardless of `Visuals`, so a windowed and a headless peer stay bit-identical.
///
/// Each repose is a pure rigid translate (NO scale): a crab renders at its TRUE physics size so a
/// collider wireframe overlays the mesh (render==physics). The giant FEEL comes from the human
/// world rendering R× smaller around it ([`crate::render::world_render_scale`]), so the crabs
/// still tower without the inflation that would desync the wireframe and force retraining Sally
/// bigger. An env is absent from the map until its crab is sampled once (no stale/zero carapace).
fn publish_skin_repose(
    bridge: Res<ExternalCrabBridge>,
    repose_out: Option<ResMut<crab_world::bot::skin::CrabSkinRepose>>,
) {
    // No skin loaded (no `sally.glb`) ⇒ no resource ⇒ nothing to repose (the static giant
    // silhouette is the visible crab there — `spawn_world` keeps it shown when no model loads).
    let Some(mut out) = repose_out else {
        return;
    };
    // Rigid translate carrying each arena rig to its crab's render-frame game spot:
    // `shift = game_spot·R − arena_carapace` (`game_spot − shift` = the arena carapace the rig
    // sits at). See the doc above for why no scale.
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

// The single-peer NN-crab determinism + walk/stability verification harness lives in `probe`
// (out of this production bridge so the shipping real-Sally MP path stays lean). Re-exported
// here so its public entry points keep their `external_crab::…` paths.
pub use probe::{ProbeSample, StabilityResult, run_headless_probe, run_vehicle_stability_probe};
