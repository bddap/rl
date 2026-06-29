pub mod actuator;
pub mod body;
pub mod brain;
pub mod collider_check;
pub mod meshfit;
pub mod physics_digest;
#[cfg(test)]
mod reset_test;
// Snapshot/restore of the entire crab physics state (MP snapshot/join + replay/debug).
// Gated on serde-serialize: it pulls serde on the rapier sets, which the trainer never needs.
#[cfg(feature = "serde-serialize")]
pub mod snapshot;
pub mod rig;
pub mod sensor;
#[cfg(test)]
mod sim_truth_test;
// Render-only: the skinned-mesh setup drives bevy's renderer (SkinnedMesh, materials,
// mesh assets), absent in the headless trainer's bevy build. Headless training spawns
// the colliders only — no skin — so the trainer loses nothing by gating this out.
#[cfg(feature = "render")]
pub mod skin;
// Not test-only: the `--check-rest-colliders` diagnostic ships in the binary and
// reuses `headless_app`/`tick` to build and settle the same windowless physics
// world the sim tests run, so one app builder serves both.
pub mod headless;

use bevy::prelude::*;
use bevy_rapier3d::plugin::PhysicsSet;

/// System sets that enforce Sense → Think → Act ordering across plugins.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum BotSet {
    /// Build observations from physics state.
    Sense,
    /// Neural network forward pass and RL bookkeeping.
    Think,
    /// Apply motor commands to joints.
    Act,
}

/// How many crab environments to run in this world (training parallelism).
/// Demo/screenshot always use 1. Set from `--envs` before BotPlugin builds.
#[derive(Resource, Clone, Copy)]
pub struct NumEnvs(pub usize);

/// Ground-plane spawn origin of each env's crab, indexed by env id. Resets
/// teleport back here, and observations report carapace position relative to
/// it so the policy can't read its env identity (or absolute drift) from x/z.
#[derive(Resource, Default)]
pub struct CrabSpawns(pub Vec<Vec3>);

/// Crabs spaced on a grid: 4 m apart, centered on the origin, so the whole grid
/// fits the ±10 m arena up to N=16. With far targets the crabs WALK and their paths
/// cross, but each env owns a private collision bit (see [`body::crab_collision`]),
/// so neighbours pass through one another — the grid is just the start layout, not a
/// no-touch guarantee.
fn grid_offset(env: usize, n: usize) -> Vec3 {
    let cols = (n as f32).sqrt().ceil() as usize;
    let spacing = 4.0;
    let col = env % cols;
    let row = env / cols;
    let rows = n.div_ceil(cols);
    Vec3::new(
        (col as f32 - (cols as f32 - 1.0) / 2.0) * spacing,
        0.0,
        (row as f32 - (rows as f32 - 1.0) / 2.0) * spacing,
    )
}

/// Plugin that manages bot spawning and per-frame sensor/actuator updates.
pub struct BotPlugin;

impl Plugin for BotPlugin {
    fn build(&self, app: &mut App) {
        // Enforce ordering: Sense → Think → Act, and run the whole loop BEFORE the
        // physics step (Rapier now lives in FixedUpdate too). So each tick observes
        // last step's state, picks an action, writes motor targets, THEN physics
        // integrates — one clean RL step per physics step.
        app.configure_sets(
            FixedUpdate,
            (BotSet::Sense, BotSet::Think, BotSet::Act)
                .chain()
                .before(PhysicsSet::SyncBackend),
        );

        app.init_resource::<actuator::CrabActions>()
            .init_resource::<sensor::CrabObservation>()
            // Always present so `build_observation` (shared with the demo) can read
            // it; the training plugin populates a real target per env, the demo
            // leaves it empty (a zero target vector in the obs).
            .init_resource::<sensor::CrabTargets>()
            .init_resource::<CrabSpawns>()
            // Resolve [`body::CrabModelPath`] (which crab this app shows) BEFORE `CrabAssets`,
            // whose `FromWorld` reads it. A surface that pre-inserted it keeps its choice; everyone
            // else gets the live `meshfit::model_path()` default. See `CrabModelPath` for why.
            .init_resource::<body::CrabModelPath>()
            .init_resource::<body::CrabAssets>()
            .init_resource::<RescueStats>()
            .add_message::<CrabRescued>()
            .add_systems(Startup, spawn_initial_crabs)
            .add_systems(FixedUpdate, rescue_nonfinite_crabs.before(BotSet::Sense))
            .add_systems(FixedUpdate, sensor::build_observation.in_set(BotSet::Sense))
            .add_systems(FixedUpdate, actuator::apply_actions.in_set(BotSet::Act));

        // The optional skinned model only renders with visuals on, and `skin` is
        // render-gated out of the headless trainer entirely — so the whole block is
        // render-only. (Headless training has no AssetServer to load the model with.)
        #[cfg(feature = "render")]
        if app
            .world()
            .get_resource::<crate::Visuals>()
            .is_some_and(|v| v.0)
        {
            skin::register(app);
        }
    }
}

/// Which crab body part [`rescue_nonfinite_crabs`] found non-finite first, named for the
/// loud log + the aggregated hub telemetry. `Carapace` is the multibody root; `Joint`
/// carries the actuated joint id (so `LegBasis(Left, 2)` pinpoints the offender). `Unknown`
/// is a defensive fallback that should be unreachable — every spawned [`body::CrabBodyPart`]
/// is either the carapace or an actuated joint (locked links like the eye-stalks aren't
/// spawned as parts), so it only fires if a future body part is added without either marker.
#[derive(Clone, Copy, Debug)]
pub enum RescueBody {
    Carapace,
    Joint(body::CrabJointId),
    Unknown,
}

impl std::fmt::Display for RescueBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RescueBody::Carapace => write!(f, "carapace"),
            RescueBody::Joint(id) => write!(f, "{id:?}"),
            RescueBody::Unknown => write!(f, "unknown"),
        }
    }
}

/// Present iff this world runs the ONE trained "Sally" as the single armed crab — solo or
/// networked play and the headless NN-crab probes — where a stable trained policy must NEVER
/// drive the body non-finite. There a [`rescue_nonfinite_crabs`] fire is a physics-correctness
/// FAULT to surface LOUDLY (and hard-fail in dev/debug/test builds), never a silent
/// catch-and-respawn (rl#137 — the silent rescue hid a frame-by-frame blowup for weeks).
/// Deliberately ABSENT in TRAINING, where a fresh/random policy plus the randomized-start
/// curriculum drives crabs non-finite constantly BY DESIGN and the rescue is the routine
/// per-episode reward terminator — making every training rescue a loud error or a panic would
/// drown the signal and kill the run. Inserted by [`net::external_crab::arm`] — the ONE arm
/// path every armed-Sally site (windowed play, screenshot, the headless probes) funnels through.
#[derive(Resource)]
pub struct CrabRescueIsFault;

/// Running tally of [`rescue_nonfinite_crabs`] fires, so a rescue is OBSERVABLE rather than
/// silent (rl#137). `total` is monotonic (a stable solo round leaves it at 0 — the verification
/// bar); `since_report` is drained+zeroed by the windowed driver each telemetry window to emit
/// ONE aggregated hub event per window instead of a per-step flood; `last_body` names the most
/// recent offender for that event.
#[derive(Resource, Default)]
pub struct RescueStats {
    pub total: u64,
    pub since_report: u32,
    pub last_body: Option<RescueBody>,
}

/// Sent when [`rescue_nonfinite_crabs`] had to rebuild an env's crab.
/// Training must treat it as an episode terminator: the replacement crab is
/// back at spawn, and letting the episode run on would smear that teleport
/// into the reward stream. `body` names the offending part for the loud surface.
#[derive(Message)]
pub struct CrabRescued {
    pub env: usize,
    pub body: RescueBody,
}

/// System: rebuilds any crab whose pose has gone non-finite (solver blowup,
/// tunneling). Runs ahead of Sense — so observations never see NaN — and
/// ahead of the physics sync, so the poisoned multibody is removed before
/// rapier's solver touches it again: NaN joint coordinates make the motor
/// constraint clamp on NaN bounds and panic the app. This catches corruption
/// on the tick after it happens; a NaN that panics the solver within the very
/// step that created it is still fatal, and the outer auto-resume loop is the
/// backstop for that rarer case.
///
/// LOUD by construction (rl#137): for the single armed Sally (the [`CrabRescueIsFault`]
/// marker present) every fire is a physics-correctness fault — an `error!` naming the
/// offending body + its non-finite value, a drained [`RescueStats`] tally the driver turns
/// into an aggregated hub telemetry event, and a HARD PANIC in dev/debug/test builds so a
/// regression can't hide. The deployed playtest is a release build (`debug_assertions` off):
/// it logs + counts + respawns as a VISIBLE last resort rather than crashing the family's
/// game, but never silently. In training (no marker) the fire stays quiet — it is the routine
/// episode terminator, not a fault.
pub fn rescue_nonfinite_crabs(
    mut commands: Commands,
    assets: Res<body::CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts: Query<
        (
            Entity,
            &body::CrabEnvId,
            &Transform,
            Option<&body::CrabCarapace>,
            Option<&body::CrabJoint>,
        ),
        With<body::CrabBodyPart>,
    >,
    mut rescued: MessageWriter<CrabRescued>,
    mut stats: ResMut<RescueStats>,
    // Present only for the single armed Sally (see [`CrabRescueIsFault`]); absent in training.
    is_fault: Option<Res<CrabRescueIsFault>>,
    time: Res<Time>,
    // Throttle the LOUD log to ~1/s (fixed clock) so an every-step blowup names the offender
    // without flooding the journal — the aggregated count rides the telemetry window instead.
    mut last_log_secs: Local<Option<f64>>,
) {
    // First non-finite part per sick env, captured WITH its offending value so the loud
    // surface can NAME the body (carapace root, else the actuated joint) and the number.
    let mut sick: Vec<(usize, RescueBody, Vec3, Quat)> = Vec::new();
    for (_, id, t, carapace, joint) in parts.iter() {
        if (!t.translation.is_finite() || !t.rotation.is_finite())
            && !sick.iter().any(|(e, ..)| *e == id.0)
        {
            let body = if carapace.is_some() {
                RescueBody::Carapace
            } else if let Some(j) = joint {
                RescueBody::Joint(j.id)
            } else {
                RescueBody::Unknown
            };
            sick.push((id.0, body, t.translation, t.rotation));
        }
    }
    if sick.is_empty() {
        return;
    }

    let fault = is_fault.is_some();
    for &(env, body, translation, rotation) in &sick {
        stats.total += 1;
        stats.since_report = stats.since_report.saturating_add(1);
        stats.last_body = Some(body);

        if fault {
            let now = time.elapsed_secs_f64();
            if last_log_secs.is_none_or(|t| now - t >= 1.0) {
                error!(
                    "rescue_nonfinite_crabs: armed crab (env {env}) went NON-FINITE at `{body}` \
                     translation={translation:?} rotation={rotation:?} — respawning as a VISIBLE \
                     last resort (physics-correctness fault, rl#137; rescue total={}). A stable \
                     trained Sally must never trip this.",
                    stats.total
                );
                *last_log_secs = Some(now);
            }
            // Dev/training/debug + tests hard-fail at the step boundary, surfaced at the source,
            // so a regression can't hide. The DEPLOYED playtest (release, debug_assertions off)
            // must NOT crash the family's game — it falls through to the loud respawn below.
            #[cfg(debug_assertions)]
            panic!(
                "rescue_nonfinite_crabs: armed crab (env {env}) went NON-FINITE at `{body}` \
                 translation={translation:?} rotation={rotation:?} (rl#137 — a stable trained \
                 Sally must never go non-finite; this is a physics-correctness regression)"
            );
        }
    }

    for &(env, body, ..) in &sick {
        let origin = spawns.0.get(env).copied().unwrap_or(Vec3::ZERO);
        respawn_crab(
            &mut commands,
            &assets,
            parts
                .iter()
                .filter(|(_, id, ..)| id.0 == env)
                .map(|(e, ..)| e),
            origin,
            env,
        );
        rescued.write(CrabRescued { env, body });
    }
}

/// Tears down every entity of one env's crab and spawns a fresh one at
/// `origin`. This is the only reliable reset: rapier 0.32 cannot rewrite
/// multibody joint coordinates in place, so a crab whose multibody state has
/// gone non-finite (e.g. tunneled through the floor at speed) is
/// unrecoverable by teleport-and-zero — the poisoned joint state survives any
/// amount of Transform/Velocity writing and every "reset" reproduces the same
/// wedged pose. Dropping the whole tree and rebuilding always works.
pub fn respawn_crab(
    commands: &mut Commands,
    assets: &body::CrabAssets,
    parts: impl Iterator<Item = Entity>,
    origin: Vec3,
    env: usize,
) {
    respawn_crab_rotated(commands, assets, parts, origin, env, Quat::IDENTITY);
}

/// Like [`respawn_crab`] but spawns the fresh crab rigidly rotated by `init_rotation`
/// (training's randomized-start curriculum — see `reset_crab`). `Quat::IDENTITY`
/// reproduces the upright respawn exactly.
pub fn respawn_crab_rotated(
    commands: &mut Commands,
    assets: &body::CrabAssets,
    parts: impl Iterator<Item = Entity>,
    origin: Vec3,
    env: usize,
    init_rotation: Quat,
) {
    for e in parts {
        commands.entity(e).despawn();
    }
    body::spawn_crab(commands, assets, origin, env, init_rotation);
}

/// Spawn `NumEnvs` crabs and size the per-env buffers to match — the [`BotPlugin`] Startup
/// system, also reusable as a deferred spawn. The boot-menu solo NN crab (rl#58) leaves
/// `NumEnvs` at 0 through the menu (so no crab spawns behind it), then bumps it to 1 and
/// runs THIS at the solo round transition, so the one crab-spawn path is shared (no second
/// spawn routine to drift from training/demo). Idempotent only via the caller's gating —
/// it appends, so run it exactly once per intended spawn.
pub fn spawn_initial_crabs(
    mut commands: Commands,
    assets: Res<body::CrabAssets>,
    num_envs: Res<NumEnvs>,
    mut spawns: ResMut<CrabSpawns>,
    mut actions: ResMut<actuator::CrabActions>,
    mut obs: ResMut<sensor::CrabObservation>,
    mut targets: ResMut<sensor::CrabTargets>,
) {
    let n = num_envs.0;
    actions.resize(n);
    obs.resize(n);
    targets.resize(n);
    for env in 0..n {
        let origin = grid_offset(env, n);
        spawns.0.push(origin);
        // Initial spawn stays upright; the randomized-start curriculum kicks in on the
        // first reset (see `reset_crab`). A clean first frame also keeps any non-training
        // caller of this system (the demo) upright.
        body::spawn_crab(&mut commands, &assets, origin, env, Quat::IDENTITY);
    }
}
