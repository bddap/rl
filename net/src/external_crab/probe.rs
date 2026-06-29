//! Headless verification harness for the external NN crab (no window / GPU / display) — the
//! GCR #82 determinism + walk probes, kept OUT of the production bridge ([`super`]) so the
//! shipping real-Sally MP path stays minimal. Two drivers, both stepping the SAME windowless
//! bot+physics stack the production crab runs under ([`headless_nn_crab_app`]): a single-peer
//! walk/reproducibility probe ([`run_headless_probe`]) and the decisive two-peer cross-peer
//! determinism gate ([`run_cross_peer_probe`]). Nothing in the production bridge depends on this
//! module — the arrow points one way (harness → bridge), via the parent's [`sync_external_crab`]
//! / [`integrate_crab`] / [`hash_crab_physics`] and the [`ExternalCrabPlugin`] it arms.

use bevy::prelude::*;

use crate::lockstep::Lockstep;
use crate::sim::{Input, PlayerId, Pos, UNIT};
use crab_world::bot::body::{CrabCarapace, CrabEnvId};
use crab_world::bot::sensor::CrabTargets;

use super::{
    ExternalCrabBridge, ExternalCrabPlugin, hash_crab_physics, integrate_crab, sync_external_crab,
};


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
    claw_q: Query<(&CrabEnvId, &Transform), With<crab_world::bot::body::CrabClawTip>>,
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
/// the training/tests use ([`crab_world::bot::headless::headless_stack`], one crab in env 0) plus
/// [`ExternalCrabPlugin`] (the policy + arena↔game bridge) with the crab ARMED. Shared by the
/// single-peer [`run_headless_probe`] and the two-peer [`run_cross_peer_probe`] so both step the
/// identical dynamics the policy trained under, with no GPU/display — one app-construction, no
/// drift between the two harnesses (the manual's "one implementation per thing"). The caller owns
/// the [`Lockstep`] driving and seeding; this only stands up the rapier NN body.
fn headless_nn_crab_app(checkpoint_dir: &std::path::Path, crab_spawn: Pos) -> bevy::app::App {
    use crab_world::bot::headless::{
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
    // spawn is a no-op (the not-yet-spawned guard sees it present). Solo arm (no lead-pin): both
    // probes share one process env, so there's no per-peer lead to diverge on. One arm path.
    super::arm(app.world_mut(), false);
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
/// [`crate::cadence::PhysicsCadence`] (2–3 steps/tick), matching what networked peers run.
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
// Crab-policy-stability gate (the vehicle migration's DONE bar #2)
// ---------------------------------------------------------------------------

/// Result of [`run_vehicle_stability_probe`]: the per-tick samples plus the tick a ram vehicle was
/// dropped onto Sally. The caller (the `nn-crab-vehicle-stability` subcommand) gates on three facts
/// read off this: the policy WALKED before the hit, every post-hit carapace pose stayed FINITE (no
/// NaN/explosion), and the policy RESUMED walking after the hit (the trained gait recovered).
pub struct StabilityResult {
    pub samples: Vec<ProbeSample>,
    /// The tick at which the ram vehicle was spawned on the crab.
    pub ram_tick: u64,
}

impl StabilityResult {
    /// Every sampled carapace pose (before AND after the hit) is finite — the crab never exploded
    /// into NaN/inf under the collision. The hard floor of the stability gate; the caller adds the
    /// bounded-height + stood-back-up + still-reaching checks on the [`samples`](Self::samples).
    pub fn carapace_stayed_finite(&self) -> bool {
        self.samples.iter().all(|s| {
            s.carapace_arena_x.is_finite()
                && s.carapace_y.is_finite()
                && s.carapace_arena_z.is_finite()
        })
    }
}

/// Headless crab-policy-stability gate: run the trained NN crab, drop a real vehicle rigidbody onto
/// it mid-walk (the same collider/mass/groups boarding spawns — [`crab_world::vehicle`]), and keep
/// stepping. Proves the migration's headline (owner 703) without a GPU: the vehicle↔crab contact is
/// real, it shoves the crab by mass, and the trained walking RECOVERS — no NaN/explosion. `warmup`
/// ticks let the crab settle + start walking before the hit; `post` ticks watch it recover.
///
/// The ram is a PURE BALLISTIC body (no `VehiclePlugin`, so no force system drives it): it carries
/// momentum into Sally's legs and bounces, isolating the COLLISION + the policy's response (the
/// flight force model is covered by `crab_world::vehicle`'s own unit tests).
pub fn run_vehicle_stability_probe(
    checkpoint_dir: &std::path::Path,
    seed: u64,
    warmup: u64,
    post: u64,
) -> StabilityResult {
    use bevy_rapier3d::prelude::Velocity;
    use crab_world::vehicle::{VehicleKind, spawn_ram_vehicle};

    let me = PlayerId(0);
    let mut ls = Lockstep::new(seed, &[me], me);
    let crab_spawn = ls.sim().crab().pos();
    let crab = ls.sim().crab();
    ls.set_external_crab_pose(crab.pos(), crab.yaw(), 0);

    let mut app = headless_nn_crab_app(checkpoint_dir, crab_spawn);
    app.insert_non_send_resource(ProbeDriver {
        ls,
        samples: Vec::new(),
        log_every: 1,
    });
    app.add_systems(
        FixedUpdate,
        probe_step.after(integrate_crab).after(hash_crab_physics),
    );

    // Warm up: let the crab settle and begin walking toward the (still) player.
    for _ in 0..warmup {
        app.update();
    }
    let ram_tick = app
        .world()
        .get_non_send_resource::<ProbeDriver>()
        .map(|d| d.ls.sim().tick())
        .unwrap_or(warmup);

    // Drop a ram vehicle beside the crab at leg height, moving INTO it — a lateral shove of the
    // legs by mass. The carapace's arena pose (env 0) is where to aim; fall back to the origin.
    let carapace = {
        let mut q = app
            .world_mut()
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>();
        q.iter(app.world())
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| t.translation)
            .unwrap_or(Vec3::ZERO)
    };
    // Beside the crab (+X, 1.2 m out), just below carapace height (≈ leg level), aimed back at it.
    let spawn_at = Transform::from_translation(carapace + Vec3::new(1.2, -0.15, 0.0));
    let ram_velocity = Velocity {
        linear: Vec3::new(-10.0, 0.0, 0.0),
        angular: Vec3::ZERO,
    };
    spawn_ram_vehicle(
        app.world_mut(),
        VehicleKind::Plane,
        spawn_at,
        ram_velocity,
    );

    // Watch the crab take the hit and recover.
    for _ in 0..post {
        app.update();
    }

    let samples = app
        .world()
        .get_non_send_resource::<ProbeDriver>()
        .map(|d| d.samples.clone())
        .unwrap_or_default();
    StabilityResult { samples, ram_tick }
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
/// [`crate::render`]'s `drive_lockstep` runs for the windowed peer ([`sync_external_crab`]),
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
/// [`crate::render`]'s `drive_lockstep` for both peers: pump the body the deterministic
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
    use crate::cadence::PhysicsCadence;
    use crate::render::{park_fixed_auto_pump, pump_fixed_steps};

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

// ---------------------------------------------------------------------------
// Cross-peer NN-crab MID-GAME-JOIN determinism harness (the GCR MP Stage 3 gate, rl#151)
// ---------------------------------------------------------------------------

/// One applied post-setup tick of the join probe: the tick number and every PRESENT peer's
/// closing `state_hash` for it (in `PlayerId` order). A tick is in lockstep iff all present
/// peers' hashes are equal. Before the join only the incumbent is present; from the
/// effective tick the joiner is too, and their hashes for the SAME tick must match — the proof
/// that the warm incumbent's cold-respawned rapier crab is bit-identical to the joiner's
/// from-fresh one (job 412's existential risk, exercised with the real armed Sally).
#[derive(Clone, Debug)]
pub struct XJoinTick {
    pub tick: u64,
    /// `(player, state_hash)` for every peer that applied this tick, sorted by player.
    pub hashes: Vec<(PlayerId, u64)>,
}

impl XJoinTick {
    /// All present peers agreed on this tick's hash (vacuously true for a single-peer tick).
    fn agrees(&self) -> bool {
        self.hashes
            .first()
            .is_none_or(|(_, h0)| self.hashes.iter().all(|(_, h)| h == h0))
    }
}

/// Result of [`run_cross_peer_join_probe`]: the per-tick hashes across peers, the lockstep's own
/// desync-fault count, and the tick the joiner became effective. A PASS is `faults == 0`, every
/// tick's present hashes agree, AND at least one post-join tick actually had BOTH peers present
/// (so the join genuinely happened and was compared) — the same belt-and-suspenders shape as
/// [`XPeerResult`], extended over a changing roster.
pub struct XJoinResult {
    pub ticks: Vec<XJoinTick>,
    pub faults: usize,
    pub effective_tick: u64,
}

impl XJoinResult {
    /// First tick whose present peers disagree, if any — where the join broke lockstep.
    pub fn first_divergence(&self) -> Option<&XJoinTick> {
        self.ticks.iter().find(|t| !t.agrees())
    }

    /// Ticks at/after the join that BOTH peers applied (so the join was actually exercised).
    pub fn compared_post_join_ticks(&self) -> usize {
        self.ticks
            .iter()
            .filter(|t| t.tick >= self.effective_tick && t.hashes.len() >= 2)
            .count()
    }

    /// All clean: no desync fault, every tick's hashes agree, and the join was genuinely
    /// compared on at least one tick with both peers present.
    pub fn is_deterministic(&self) -> bool {
        self.faults == 0 && self.first_divergence().is_none() && self.compared_post_join_ticks() > 0
    }
}

/// One peer of the join probe: its own armed headless rapier-NN crab world + its own lockstep +
/// physics cadence. The incumbent starts at round boot; the joiner is constructed (cold) at the
/// admission moment via [`Lockstep::join_at`]. A deterministic per-peer input counter keeps the
/// run reproducible while the two peers drive their players DIFFERENTLY (a real two-player round).
struct JoinPeer {
    me: PlayerId,
    app: bevy::app::App,
    ls: Lockstep,
    cad: crate::cadence::PhysicsCadence,
    issued: u64,
}

impl JoinPeer {
    /// Build a peer from a ready [`Lockstep`], standing up its armed crab world at `crab_spawn`
    /// (the round's rebuilt crab pos, identical on every peer) with ZERO physics steps run — only
    /// [`pump_fixed_steps`] advances the body from here, exactly as production does.
    fn new(checkpoint_dir: &std::path::Path, ls: Lockstep, crab_spawn: Pos) -> Self {
        use crate::render::park_fixed_auto_pump;
        let mut app = headless_nn_crab_app(checkpoint_dir, crab_spawn);
        park_fixed_auto_pump(app.world_mut());
        app.update(); // run Startup (spawn the crab) with no physics steps
        let mut ls = ls;
        // Seed the lockstep's external-crab pose at spawn (digest 0); the first post-step sync
        // fills the real digest before it is ever cross-checked.
        ls.set_external_crab_pose(crab_spawn, 0, 0);
        Self { me: ls.me(), app, ls, cad: crate::cadence::PhysicsCadence::default(), issued: 0 }
    }

    /// This peer's deterministic-but-distinct input for its next issuing tick. A per-peer phase
    /// offset makes the two players move differently (so the round is real) while staying a pure
    /// function of `(me, issued)` — same probe args ⇒ identical run.
    fn next_input(&mut self) -> Input {
        let phase = self.issued as f32 * 0.1 + self.me.0 as f32 * 1.7;
        self.issued += 1;
        Input::from_axes(phase.cos(), phase.sin())
    }

    /// Advance every ready tick, mirroring `render::drive_lockstep`'s drain loop EXACTLY: pump the
    /// body the cadence's steps for the tick, push its pose+digest, advance one tick, and on the
    /// round-boundary RESTART edge (a roster-change rebuild — `sim().tick()` rewinds) reset the
    /// cadence, re-seed the bridge to spawn, and COLD-RESPAWN the rapier body. Records each applied
    /// tick's `(tick, hash)`. Returns the lockstep faults observed.
    fn drain(&mut self, target: u64, out: &mut std::collections::BTreeMap<u64, Vec<(PlayerId, u64)>>) -> usize {
        use crate::render::pump_fixed_steps;
        let mut faults = 0;
        while self.ls.next_tick_ready() && self.ls.next_tick() <= target {
            // Mirror `render::drive_lockstep`'s drain loop EXACTLY (one behaviour, faithfully
            // reproduced so the probe MEASURES production): pump the body the cadence's steps for
            // this tick, push its pose+digest, advance one tick, and on the RESTART edge (a
            // roster-change rebuild OR the plain-restart button — `sim().tick()` rewinds) reset the
            // cadence, re-seed the bridge to spawn, and cold-respawn the rapier body.
            let steps = self.cad.steps_for_next_tick();
            pump_fixed_steps(self.app.world_mut(), steps);
            sync_peer(&mut self.app, &mut self.ls);
            let before = self.ls.sim().tick();
            let tick_faults = self.ls.advance_one().expect("next_tick_ready was true");
            faults += tick_faults.len();
            let restarted = self.ls.sim().tick() < before;
            if restarted {
                self.cad = crate::cadence::PhysicsCadence::default();
                let spawn = self.ls.sim().crab().pos();
                self.app
                    .world_mut()
                    .resource_mut::<ExternalCrabBridge>()
                    .restart_to_spawn(spawn);
                super::cold_respawn_armed_crab(self.app.world_mut());
            }
            if let Some(c) = self.ls.last_applied() {
                out.entry(c.tick).or_default().push((self.me, c.hash));
            }
        }
        faults
    }
}

/// The GCR MP Stage 3 mid-game-join armed-Sally determinism PROBE (rl#151): run the real armed
/// rapier-NN crab on an INCUMBENT that hosts a live round, then have a fresh peer JOIN mid-game over
/// the round-boundary mechanism, and measure whether every peer computes the byte-identical
/// `state_hash` for every tick the joiner participates in.
///
/// This is the measurement the pure-core join determinism check (increment A) structurally CANNOT
/// do: that one folds `external_crab_digest = 0`, so it proves the INTEGER sim rebuilds
/// bit-identically but never exercises the rapier crab. Here the crab is the real trained body,
/// stepped at the production [`crate::cadence::PhysicsCadence`] with its full physics digest folded
/// into the hash, and the drain loop MIRRORS `render::drive_lockstep` exactly — so the result
/// reflects production. The measured FINDING is that the warm incumbent (cold-respawned at the
/// boundary) DIVERGES from the joiner's from-fresh body at the join tick: `cold_respawn_armed_crab`
/// respawns the crab entities but not the incumbent's warm `RapierContext` (solver/contact caches +
/// handle-arena free-list), so job 412's restored-vs-live divergence surfaces at the join. See the
/// `game nn-crab-join-xpeer` module doc for the full finding and why it confirms the state-resync
/// direction over bit-exact lockstep.
///
/// The incumbent starts SOLO (roster `{P0}`) and runs `pre_join_ticks`; then a roster change to
/// `{P0,P1}` is scheduled `JOIN_LEAD` ticks ahead (the server's real lead) and the joiner `P1` is
/// built via [`Lockstep::join_at`] at that tick. Both then run until each has applied through the
/// effective tick + `post_join_ticks`. Same `(checkpoint, seed, …)` ⇒ identical result.
pub fn run_cross_peer_join_probe(
    checkpoint_dir: &std::path::Path,
    seed: u64,
    pre_join_ticks: u64,
    post_join_ticks: u64,
) -> XJoinResult {
    use crate::server::JOIN_LEAD;

    let p0 = PlayerId(0);
    let p1 = PlayerId(1);

    // The incumbent hosts a solo round; its crab spawn is the {P0} round's spawn.
    let solo_ls = Lockstep::new(seed, &[p0], p0);
    let incumbent_spawn = solo_ls.sim().crab().pos();
    let mut incumbent = JoinPeer::new(checkpoint_dir, solo_ls, incumbent_spawn);

    // Per-tick hashes keyed by tick, each carrying every peer's hash for it.
    let mut hashes: std::collections::BTreeMap<u64, Vec<(PlayerId, u64)>> = std::collections::BTreeMap::new();

    // 1. Run the incumbent solo for `pre_join_ticks` applied ticks. Solo ⇒ its own input
    //    completes every tick, so no exchange is needed yet.
    let mut faults = 0usize;
    while incumbent.ls.next_tick() < pre_join_ticks {
        let input = incumbent.next_input();
        let _ = incumbent.ls.submit_local_input(input);
        faults += incumbent.drain(pre_join_ticks - 1, &mut hashes);
    }

    // 2. Admit the joiner: schedule the roster change JOIN_LEAD ticks ahead (the real server
    //    lead), exactly as `Server::admit`/`admit_joiners` would, and build the joiner's session
    //    at that effective tick over the new roster. The joiner's crab world is built COLD (fresh,
    //    zero steps) — the asymmetry under test against the incumbent's warm-then-respawned body.
    let effective_tick = incumbent.ls.next_tick() + JOIN_LEAD;
    let new_roster = [p0, p1];
    incumbent.ls.schedule_roster_change(effective_tick, &new_roster);
    let join_ls = Lockstep::join_at(seed, &new_roster, p1, effective_tick);
    let joiner_spawn = join_ls.sim().crab().pos();
    let mut joiner = JoinPeer::new(checkpoint_dir, join_ls, joiner_spawn);

    // 3. Run both until each has applied through the effective tick + the post-join window.
    let target = effective_tick + post_join_ticks;
    while incumbent.ls.next_tick() <= target || joiner.ls.next_tick() <= target {
        // Each peer submits one input for its next issuing tick, then they EXCHANGE — each
        // records the other's exact message (filed by the tick it carries), the in-process
        // analogue of the wire transport. The joiner issues from the effective tick; the
        // incumbent leads by INPUT_DELAY, so the joiner's apply is gated on the incumbent's
        // inputs arriving (natural lockstep backpressure) and never pumps a stalled tick.
        let inc_input = incumbent.next_input();
        let join_input = joiner.next_input();
        let m_inc = incumbent.ls.submit_local_input(inc_input);
        let m_join = joiner.ls.submit_local_input(join_input);
        if incumbent.ls.record_remote(p1, m_join).is_some() {
            faults += 1;
        }
        if joiner.ls.record_remote(p0, m_inc).is_some() {
            faults += 1;
        }
        faults += incumbent.drain(target, &mut hashes);
        faults += joiner.drain(target, &mut hashes);
    }

    let ticks = hashes
        .into_iter()
        .map(|(tick, mut hs)| {
            hs.sort_by_key(|(p, _)| *p);
            XJoinTick { tick, hashes: hs }
        })
        .collect();
    XJoinResult { ticks, faults, effective_tick }
}
