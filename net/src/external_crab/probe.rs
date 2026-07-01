//! Headless verification harness for the external NN crab (no window / GPU / display) — the
//! single-peer determinism + walk/stability probes, kept OUT of the production bridge ([`super`]) so
//! the shipping real-Sally MP path stays minimal. Both drivers step the SAME windowless bot+physics
//! stack the production crab runs under ([`headless_nn_crab_app`]): a walk/reproducibility probe
//! ([`run_headless_probe`]) and a vehicle-stability probe ([`run_vehicle_stability_probe`]). Nothing
//! in the production bridge depends on this module — the arrow points one way (harness → bridge), via
//! the parent's [`sync_external_crab`] / [`integrate_crab`] / [`hash_crab_physics`] and the
//! [`ExternalCrabPlugin`] it arms.

use bevy::prelude::*;

use crate::sim::{Input, PlayerId, Pos, Sim, UNIT};
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

/// Probe driver state: the [`Sim`] driven by hand (outside Bevy's schedules) so the harness can step
/// it once per `app.update()`, in step with the one physics tick each update runs. Non-send because
/// [`Sim`]'s hasher etc. need not be `Sync` here, and only the main thread drives it.
struct ProbeDriver {
    sim: Sim,
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
    sync_external_crab(&mut driver.sim, &mut bridge);
    let prey = driver.sim.nearest_living_player_pos();

    // Local player holds still (neutral input) so the test isolates the CRAB's motion: the crab
    // should close the gap on a stationary player. Single peer, so a complete neutral map advances
    // the sim one tick directly (no coordination — this is a determinism/walk probe, not a match).
    let me = PlayerId(0);
    driver.sim.step(&std::collections::BTreeMap::from([(me, Input::from_axes(0.0, 0.0))]));

    // Log periodically (and always at tick 1 so the start point is recorded).
    let tick = driver.sim.tick();
    if tick == 1 || tick.is_multiple_of(driver.log_every) {
        let crab = driver.sim.crab().pos();
        let crab_x_m = crab.x as f32 / UNIT as f32;
        let crab_z_m = crab.z as f32 / UNIT as f32;
        let dist_to_prey_m = prey
            .map(|p| {
                let dx = (p.x - crab.x) as f32 / UNIT as f32;
                let dz = (p.z - crab.z) as f32 / UNIT as f32;
                (dx * dx + dz * dz).sqrt()
            })
            .unwrap_or(f32::NAN);
        let state_hash = driver.sim.state_hash();

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
/// [`ExternalCrabPlugin`] (the policy + arena↔game bridge) with the crab ARMED. Shared by the walk
/// and vehicle-stability probes so both step the identical dynamics the policy trained under, with no
/// GPU/display — one app-construction, no drift (the manual's "one implementation per thing"). The
/// caller owns the [`Sim`] driving and seeding; this only stands up the rapier NN body.
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
    // spawn is a no-op (the not-yet-spawned guard sees it present). One arm path.
    super::arm(app.world_mut());
    // Force the ECS executor serial now that the plugin systems are wired — fixes the system run
    // ORDER, the second half of the determinism guarantee alongside the pinned pools. A system the
    // single-peer probe adds later (`probe_step`) lands in the already-serial schedule and inherits
    // it, so this one call covers both probe drivers.
    force_serial_schedules(&mut app);
    app
}

/// Run the NN crab headlessly for `ticks` sim steps and return the logged samples, via
/// [`headless_nn_crab_app`] + a hand-driven [`Sim`] — so the crab steps the exact dynamics the policy
/// trained under, with no GPU/display. `checkpoint_dir` is the trained policy; `seed` seeds the round
/// (same seed twice ⇒ identical samples, the determinism check). The local player holds still so a
/// shrinking `dist_to_prey_m` proves the crab walks toward it under the policy.
///
/// NOTE: this probe steps the body one physics step per `app.update()` — a walking/reproducibility
/// sanity check, where the absolute gait speed doesn't matter.
pub fn run_headless_probe(
    checkpoint_dir: &std::path::Path,
    seed: u64,
    ticks: u64,
    log_every: u64,
) -> Vec<ProbeSample> {
    let me = PlayerId(0);
    let mut sim = Sim::new(seed, &[me]);
    let crab_spawn = sim.crab().pos();
    // The crab is externally driven for the whole probe (we own its position). Seed the pose with
    // the crab's CURRENT spawn pose/yaw — writing back what's already there, so this is a no-op on
    // sim state. Seed with a zero digest; the first post-step `hash_crab_physics` fills it before
    // the first `sync_external_crab` push, so the seeded value is never the one used.
    let crab = sim.crab();
    sim.set_external_crab_pose(crab.pos(), crab.yaw(), 0);

    let mut app = headless_nn_crab_app(checkpoint_dir, crab_spawn);
    app.insert_non_send_resource(ProbeDriver {
        sim,
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
    let mut sim = Sim::new(seed, &[me]);
    let crab_spawn = sim.crab().pos();
    let crab = sim.crab();
    sim.set_external_crab_pose(crab.pos(), crab.yaw(), 0);

    let mut app = headless_nn_crab_app(checkpoint_dir, crab_spawn);
    app.insert_non_send_resource(ProbeDriver {
        sim,
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
        .map(|d| d.sim.tick())
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
