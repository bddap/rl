pub mod actuator;
pub mod arch;
pub mod body;
pub mod collider_check;
pub mod headless;
pub mod meshfit;
pub mod physics_digest;
pub mod pose_sentinel;
#[cfg(test)]
mod reset_test;
pub mod rig;
pub mod rig_audit;
pub mod sensor;
#[cfg(test)]
mod sim_truth_test;
#[cfg(feature = "render")]
pub mod skin;

use bevy::prelude::*;
use bevy_rapier3d::plugin::PhysicsSet;

/// One rig audit's outcome: it RAN and judged (the report explains the verdict on
/// stdout). "Couldn't run at all" is the audits' `Err(String)` instead — the two were
/// indistinguishable when the audits returned bare exit-code i32s (rl#270).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditVerdict {
    Pass,
    Fail,
}

impl AuditVerdict {
    pub fn failed(fail: bool) -> Self {
        if fail { Self::Fail } else { Self::Pass }
    }
}

/// Identity of this build's obs/action CHANNEL LAYOUT (bddap/rl#271): an FNV-1a fold of
/// one label per channel — action channels in [`body::CrabJointId::all`] order, then obs
/// slots in serialize order. Dims-only gates pass a same-count reorder, which would load
/// a trained checkpoint clean and drive the wrong joints; this digest is stamped into
/// brain envelopes so a layout change REFUSES stale brains instead of silently remapping
/// them (the body-digest pattern, bddap/rl#214).
pub fn channel_layout_digest() -> u64 {
    let mut h = crate::fnv::Fnv::new();
    for label in actuator::action_channel_labels()
        .iter()
        .chain(sensor::obs_channel_labels().iter())
    {
        h.write(label.as_bytes());
        h.write(b"\n");
    }
    h.finish()
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum BotSet {
    Sense,
    Think,
    Act,
}

/// The rl#116 pose sentinel's slot, ordered before `BotSet::Sense` and
/// `PhysicsSet::SyncBackend` (NOT guaranteed first in `FixedUpdate` — an unordered
/// system can still sneak in ahead of it). A legitimate physics-side teleport (e.g.
/// the rl#240 recenter) MUST order `.after(PoseSentinelSet)` — the same tick's
/// `SyncBackend` then consumes it without the sentinel ever seeing it. Foreign
/// writes from other schedules (where render systems live) always land between
/// fixed ticks, before this set, and are caught.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PoseSentinelSet;

#[derive(Resource, Clone, Copy)]
pub struct NumEnvs(pub usize);

#[derive(Resource, Default, Clone)]
pub struct CrabSpawns(Vec<Vec3>);

impl CrabSpawns {
    /// Explicit whole-set construction for tests and bootstrap — there is deliberately
    /// no way to append (rl#242: an append across two `spawn_initial_crabs` runs left
    /// stale origins live at current indices).
    pub fn from_origins(origins: Vec<Vec3>) -> Self {
        Self(origins)
    }

    fn rebuild(&mut self, n: usize) {
        self.0 = (0..n).map(|env| grid_offset(env, n)).collect();
    }

    /// Not yet rebuilt by `spawn_initial_crabs` — the pre-spawn frames a FixedUpdate
    /// consumer (net's arena-anchor publisher, rl#224) must sit out rather than hit
    /// [`Self::origin`]'s wiring-bug panic.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The spawn origin of a live env. Infallible by construction: every crab entity's
    /// env index comes from `spawn_initial_crabs`, which rebuilds this resource and
    /// sizes obs/targets/actions from the same n. A miss is a wiring bug, and
    /// substituting a finite default here silently corrupts everything downstream —
    /// the spawn-relative obs channel, respawn placement, reward drift (rl#242) — so
    /// it panics instead.
    pub fn origin(&self, env: usize) -> Vec3 {
        *self
            .0
            .get(env)
            .expect("CrabSpawns has no origin for a live env — spawn wiring bug (rl#242)")
    }
}

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

pub struct BotPlugin;

impl Plugin for BotPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            FixedUpdate,
            (BotSet::Sense, BotSet::Think, BotSet::Act)
                .chain()
                .before(PhysicsSet::SyncBackend),
        );

        app.init_resource::<actuator::CrabActions>()
            .init_resource::<sensor::CrabObservation>()
            .init_resource::<sensor::CrabTargets>()
            .init_resource::<CrabSpawns>()
            .init_resource::<body::CrabModelPath>()
            .init_resource::<body::CrabAssets>()
            .init_resource::<RescueStats>()
            .add_message::<CrabRescued>()
            .add_systems(Startup, spawn_initial_crabs)
            .add_systems(FixedUpdate, rescue_nonfinite_crabs.before(BotSet::Sense))
            .configure_sets(
                FixedUpdate,
                PoseSentinelSet
                    .before(BotSet::Sense)
                    .before(PhysicsSet::SyncBackend),
            )
            .add_systems(
                FixedUpdate,
                pose_sentinel::assert_body_transforms_rapier_owned
                    .in_set(PoseSentinelSet)
                    .run_if(pose_sentinel::visuals_on),
            )
            .add_systems(FixedUpdate, sensor::build_observation.in_set(BotSet::Sense))
            .add_systems(FixedUpdate, actuator::apply_actions.in_set(BotSet::Act));

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

#[derive(Resource)]
pub struct CrabRescueIsFault;

#[derive(Resource, Default)]
pub struct RescueStats {
    pub total: u64,
    pub since_report: u32,
    pub last_body: Option<RescueBody>,
}

#[derive(Message)]
pub struct CrabRescued {
    pub env: usize,
    pub body: RescueBody,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
    is_fault: Option<Res<CrabRescueIsFault>>,
    time: Res<Time>,
    mut last_log_secs: Local<Option<f64>>,
) {
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
            #[cfg(debug_assertions)]
            panic!(
                "rescue_nonfinite_crabs: armed crab (env {env}) went NON-FINITE at `{body}` \
                 translation={translation:?} rotation={rotation:?} (rl#137 — a stable trained \
                 Sally must never go non-finite; this is a physics-correctness regression)"
            );
        }
    }

    for &(env, body, ..) in &sick {
        let origin = spawns.origin(env);
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

pub fn respawn_crab(
    commands: &mut Commands,
    assets: &body::CrabAssets,
    parts: impl Iterator<Item = Entity>,
    origin: Vec3,
    env: usize,
) {
    respawn_crab_rotated(commands, assets, parts, origin, env, Quat::IDENTITY);
}

pub const RESET_GRACE_TICKS: u32 = 32;

pub fn settle_countdown(grace: u32) -> u32 {
    grace.saturating_sub(1)
}

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
    spawns.rebuild(n);
    for env in 0..n {
        body::spawn_crab(
            &mut commands,
            &assets,
            spawns.origin(env),
            env,
            Quat::IDENTITY,
        );
    }
}

#[cfg(test)]
mod layout_digest_tests {
    /// Golden pin of the channel-layout digest (bddap/rl#271). Red means you changed the
    /// action/obs channel order or slot map — every trained checkpoint stops loading on
    /// the new layout (by design: loading would silently remap channels). If the change
    /// is deliberate, update this constant and plan fresh checkpoint dirs. Labels are
    /// `{id:?}` Debug names, so a pure VARIANT RENAME also trips this — a false
    /// invalidation, the price of making reorders detectable; re-pinning after a rename
    /// (no semantic layout change) is safe and torches nothing.
    #[test]
    fn channel_layout_digest_is_pinned() {
        assert_eq!(super::channel_layout_digest(), GOLDEN, "see doc comment");
    }

    const GOLDEN: u64 = 0xd3a0_7749_ac5d_4870;
}
