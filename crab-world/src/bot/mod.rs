pub mod actuator;
pub mod arch;
pub mod body;
pub mod collider_check;
pub mod headless;
pub mod physics_digest;
pub mod pose_sentinel;
#[cfg(test)]
mod reset_test;
pub mod rig;
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

/// One verdict→exit-code mapping for every audit CLI (rl-train and the offline
/// `meshfit` tool), so the two can't diverge on what FAIL exits with.
impl From<AuditVerdict> for std::process::ExitCode {
    fn from(v: AuditVerdict) -> Self {
        match v {
            AuditVerdict::Pass => std::process::ExitCode::SUCCESS,
            AuditVerdict::Fail => std::process::ExitCode::FAILURE,
        }
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

/// The planar spawn layout (meters) the FIRST `spawn_initial_crabs` must lay the
/// origins on, instead of the training grid. Inserted by net's GCR restart — the
/// holder of the arena↔sim spawn correspondence — whenever it runs while the origins
/// are still unlaid, and consumed (removed) by the spawn, so a later respawn can never
/// read a stale round's layout. Training never inserts it and keeps the grid. Closes
/// the fresh-app first-round exception of rl#290: origins used to start on the grid
/// and only a round RESTART re-pinned them ([`CrabSpawns::repin_layout`], rl#289).
///
/// `base_xz` is the caller's base choice (`CrabSpawns::lay_layout`): env 0's spawn
/// lands surface-placed AT the base. Since rl#298 stage 5 every caller passes an
/// ABSOLUTE spot on the one shared tile — net the sim's own first spawn (one frame:
/// world coordinates ARE sim meters), the eval its terrain locale (rl#293).
#[derive(Resource)]
pub struct InitialCrabLayout {
    pub base_xz: Vec2,
    pub spawns_m: Vec<Vec2>,
}

#[derive(Resource, Default, Clone)]
pub struct CrabSpawns(Vec<Vec3>);

impl CrabSpawns {
    /// Explicit whole-set construction for tests and bootstrap — there is deliberately
    /// no way to append (rl#242: an append across two `spawn_initial_crabs` runs left
    /// stale origins live at current indices).
    pub fn from_origins(origins: Vec<Vec3>) -> Self {
        Self(origins)
    }

    /// Rebuild the spawn grid ON the terrain surface (rl#281): each origin's y is the
    /// ground height at its (x, z), so every consumer — initial spawn, episode respawn,
    /// rescue, the spawn-relative obs channel — is surface-relative through this one seam.
    fn rebuild(&mut self, n: usize, terrain: &crate::terrain::TerrainGrid) {
        self.0 = (0..n)
            .map(|env| {
                let p = grid_offset(env, n);
                terrain.place(Vec2::new(p.x, p.z), 0.0)
            })
            .collect();
    }

    /// THE layout→origins formula (rl#289/rl#290, one owner): env i's origin lands
    /// surface-placed at `base_xz + (sᵢ − s₀)` — only the layout's SHAPE is adopted;
    /// the base is the caller's gauge choice.
    fn lay_layout(
        &mut self,
        base_xz: Vec2,
        spawns_m: &[Vec2],
        terrain: &crate::terrain::TerrainGrid,
    ) {
        let Some(&s0) = spawns_m.first() else {
            self.0.clear();
            return;
        };
        self.0 = spawns_m
            .iter()
            .map(|&s| terrain.place(base_xz + (s - s0), 0.0))
            .collect();
    }

    /// [`Self::rebuild`]'s layout twin ([`InitialCrabLayout`], rl#290): lay the initial
    /// origins on the given planar spawn layout instead of the grid, env 0 at the
    /// layout's base — an ABSOLUTE spot since rl#298 stage 5 (net passes the sim's
    /// first spawn, the eval its terrain locale, rl#293).
    fn rebuild_from_layout(
        &mut self,
        layout: &InitialCrabLayout,
        terrain: &crate::terrain::TerrainGrid,
    ) {
        self.lay_layout(layout.base_xz, &layout.spawns_m, terrain);
    }

    /// Re-place a LIVE env's origin with a fresh locale — terrain training draws one
    /// per episode (rl#281 stage 4, `reset_crab`), the demo's reset re-rolls one per
    /// press (rl#300, `demo_respawn`) — and every caller respawns the crab onto it in
    /// the same tick, so origin and body never disagree. `pub(crate)`
    /// because that safety argument is the caller's, not this method's — exposing it
    /// wider reopens the rl#242 stale-origin bug class. Update-only: an out-of-range
    /// env is the same wiring bug [`Self::origin`] panics on, and appending is still
    /// impossible.
    pub(crate) fn set_origin(&mut self, env: usize, origin: Vec3) {
        *self
            .0
            .get_mut(env)
            .expect("CrabSpawns has no origin for a live env — spawn wiring bug (rl#242)") = origin;
    }

    /// The rl#240 recenter for a world that must stay glued to its terrain (net/GCR,
    /// rl#281 stage 6): move the ORIGIN to the crab instead of teleporting the crab to
    /// the origin. The origin becomes the carapace's own ground point, so the
    /// spawn-relative obs channel snaps back into the spawn distribution while every
    /// Transform — the crab's, the crafts', the terrain the player sees — stays
    /// untouched. (The trainer/eval teleport is fine mid-episode where nothing rendered
    /// is watching; under a rendered world it would swap the terrain locale beneath the
    /// crab's feet while the drawn mountain stayed put.) Same rl#242 invariant as
    /// [`Self::set_origin`], reached from the other end: origin and body agree in the
    /// same tick because the origin is DERIVED from the body. Returns the new origin.
    pub fn rebase_origin_to(
        &mut self,
        env: usize,
        carapace: Vec3,
        terrain: &crate::terrain::TerrainGrid,
    ) -> Vec3 {
        let origin = terrain.place(Vec2::new(carapace.x, carapace.z), 0.0);
        self.set_origin(env, origin);
        origin
    }

    /// [`Self::rebase_origin_to`]'s whole-set sibling, for a round RESTART (rl#289):
    /// re-place every live origin on the given planar spawn layout (meters), env 0 at
    /// `base_xz`. Since rl#298 stage 5 the caller (net's one-frame restart) passes the
    /// layout's own first spawn as the base — origins land AT the sim spawns
    /// absolutely, there being no arena gauge left to preserve. Same rl#242 invariant
    /// as [`Self::set_origin`], same argument: the sole caller respawns every crab
    /// onto the new origins in the same call, so origin and body never disagree.
    pub fn repin_layout(
        &mut self,
        base_xz: Vec2,
        spawns_m: &[Vec2],
        terrain: &crate::terrain::TerrainGrid,
    ) {
        assert_eq!(
            spawns_m.len(),
            self.0.len(),
            "a restart layout must cover every live env's origin (rl#242)"
        );
        self.lay_layout(base_xz, spawns_m, terrain);
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
            .add_systems(FixedUpdate, rescue_lost_crabs.before(BotSet::Sense))
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
    /// Rescues since the telemetry drain last cleared, tallied PER REASON: the
    /// fault/warn split must survive aggregation, or a legitimate tunneling rescue
    /// surfaces on the hub feed as an rl#137 non-finite fault — a false alarm.
    pub since_nonfinite: u32,
    pub since_below_terrain: u32,
    pub last_body: Option<RescueBody>,
}

#[derive(Message)]
pub struct CrabRescued {
    pub env: usize,
    pub body: RescueBody,
}

/// How far below the local terrain surface a part must sink before the y-floor rescue
/// fires (rl#283). The heightfield is a zero-thickness sheet — a body knocked through it
/// (vertical tunneling starts around 1.5-4 m/s for cm-scale feet at this tick rate)
/// falls forever, where the old halfspace made that impossible. Soft contacts rest with
/// ~cm penetration, so metres below the surface is unambiguous.
const BELOW_TERRAIN_RESCUE_M: f32 = 2.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RescueReason {
    /// NaN/inf pose — a physics-correctness fault (rl#137): error + debug-panic when
    /// armed. Wins over [`Self::BelowTerrain`] when one env shows both.
    NonFinite,
    /// Tunneled through the heightfield sheet (rl#283) — reachable by legitimate hard
    /// hits, so it warns when armed instead of faulting. Also catches a body fallen
    /// past the tile EDGE (the sampler clamps flat there but the collider ends): the
    /// interim backstop until rl#281's real world-bounds policy.
    BelowTerrain,
}

impl RescueReason {
    /// Per-reason rate-limit slot — shared limiting would let a warn-tier tunneling
    /// stream starve the rl#137 error line, which in release is the only fault signal.
    fn log_slot(self) -> usize {
        match self {
            RescueReason::NonFinite => 0,
            RescueReason::BelowTerrain => 1,
        }
    }
}

/// Respawn crabs physics can no longer recover: non-finite poses (rl#137) and bodies
/// fallen through the terrain sheet (rl#283 — the y-floor that restores the old
/// halfspace's no-fall-forever guarantee, keyed to [`TerrainGrid::height`] so it tracks
/// mountains and valleys alike).
///
/// [`TerrainGrid::height`]: crate::terrain::TerrainGrid::height
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn rescue_lost_crabs(
    mut commands: Commands,
    assets: Res<body::CrabAssets>,
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
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
    mut last_log_secs: Local<[Option<f64>; 2]>,
) {
    // Worst reason wins per env: a blowup plausibly leaves some parts NaN and others
    // flung finite-but-deep-below in the same tick, and query order is arbitrary — the
    // env must still register as the rl#137 fault, not a mere tunneling warn.
    let mut sick: std::collections::BTreeMap<usize, (RescueBody, Vec3, Quat, RescueReason)> =
        Default::default();
    for (_, id, t, carapace, joint) in parts.iter() {
        let reason = if !t.translation.is_finite() || !t.rotation.is_finite() {
            RescueReason::NonFinite
        } else if t.translation.y
            < terrain.height(t.translation.x, t.translation.z) - BELOW_TERRAIN_RESCUE_M
        {
            RescueReason::BelowTerrain
        } else {
            continue;
        };
        match sick.get(&id.0) {
            Some(&(.., RescueReason::NonFinite)) => continue,
            Some(_) if reason == RescueReason::BelowTerrain => continue,
            _ => {}
        }
        let body = if carapace.is_some() {
            RescueBody::Carapace
        } else if let Some(j) = joint {
            RescueBody::Joint(j.id)
        } else {
            RescueBody::Unknown
        };
        sick.insert(id.0, (body, t.translation, t.rotation, reason));
    }
    if sick.is_empty() {
        return;
    }

    let fault = is_fault.is_some();
    for (&env, &(body, translation, rotation, reason)) in &sick {
        stats.total += 1;
        match reason {
            RescueReason::NonFinite => {
                stats.since_nonfinite = stats.since_nonfinite.saturating_add(1);
            }
            RescueReason::BelowTerrain => {
                stats.since_below_terrain = stats.since_below_terrain.saturating_add(1);
            }
        }
        stats.last_body = Some(body);

        if !fault {
            continue;
        }
        let now = time.elapsed_secs_f64();
        let log_ok = last_log_secs[reason.log_slot()].is_none_or(|t| now - t >= 1.0);
        if log_ok {
            last_log_secs[reason.log_slot()] = Some(now);
        }
        match reason {
            RescueReason::NonFinite => {
                if log_ok {
                    error!(
                        "rescue_lost_crabs: armed crab (env {env}) went NON-FINITE at `{body}` \
                         translation={translation:?} rotation={rotation:?} — respawning as a \
                         VISIBLE last resort (physics-correctness fault, rl#137; rescue \
                         total={}). A stable trained Sally must never trip this.",
                        stats.total
                    );
                }
                #[cfg(debug_assertions)]
                panic!(
                    "rescue_lost_crabs: armed crab (env {env}) went NON-FINITE at `{body}` \
                     translation={translation:?} rotation={rotation:?} (rl#137 — a stable \
                     trained Sally must never go non-finite; this is a physics-correctness \
                     regression)"
                );
            }
            RescueReason::BelowTerrain => {
                if log_ok {
                    warn!(
                        "rescue_lost_crabs: armed crab (env {env}) fell through the terrain at \
                         `{body}` translation={translation:?} — respawning (rl#283 y-floor; \
                         rescue total={})",
                        stats.total
                    );
                }
            }
        }
    }

    for (&env, &(body, ..)) in &sick {
        let origin = spawns.origin(env);
        respawn_crab(
            &mut commands,
            &assets,
            &terrain,
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
    terrain: &crate::terrain::TerrainGrid,
    parts: impl Iterator<Item = Entity>,
    origin: Vec3,
    env: usize,
) {
    respawn_crab_rotated(
        commands,
        assets,
        terrain,
        parts,
        origin,
        env,
        Quat::IDENTITY,
    );
}

pub const RESET_GRACE_TICKS: u32 = 32;

pub fn settle_countdown(grace: u32) -> u32 {
    grace.saturating_sub(1)
}

pub fn respawn_crab_rotated(
    commands: &mut Commands,
    assets: &body::CrabAssets,
    terrain: &crate::terrain::TerrainGrid,
    parts: impl Iterator<Item = Entity>,
    origin: Vec3,
    env: usize,
    init_rotation: Quat,
) {
    for e in parts {
        commands.entity(e).despawn();
    }
    body::spawn_crab(commands, assets, terrain, origin, env, init_rotation);
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_initial_crabs(
    mut commands: Commands,
    assets: Res<body::CrabAssets>,
    num_envs: Res<NumEnvs>,
    terrain: Res<crate::terrain::Terrain>,
    mut spawns: ResMut<CrabSpawns>,
    layout: Option<Res<InitialCrabLayout>>,
    mut actions: ResMut<actuator::CrabActions>,
    mut obs: ResMut<sensor::CrabObservation>,
    mut targets: ResMut<sensor::CrabTargets>,
) {
    let n = num_envs.0;
    actions.resize(n);
    obs.resize(n);
    targets.resize(n);
    match &layout {
        Some(l) => {
            assert_eq!(
                l.spawns_m.len(),
                n,
                "the installed spawn layout must cover every env (rl#290)"
            );
            spawns.rebuild_from_layout(l, &terrain);
            commands.remove_resource::<InitialCrabLayout>();
        }
        None => spawns.rebuild(n, &terrain),
    }
    for env in 0..n {
        body::spawn_crab(
            &mut commands,
            &assets,
            &terrain,
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

    const GOLDEN: u64 = 0xba5b_bb62_cc3d_7657;
}
