//! Tripwire making "a render system mutates a rapier-driven `Transform`" loud and
//! harmless BEFORE it corrupts physics (rl#116).
//!
//! bevy_rapier syncs any CHANGED body `Transform` back into the rigid body at
//! `PhysicsSet::SyncBackend`, so a "cosmetic" write to a [`CrabBodyPart`]
//! teleports the body every step and blows up the multibody solver (the GCR
//! play-day NaN crash, fixed in 931936a9). Cosmetic placement must ride the
//! render-only proxies instead: [`super::skin`]'s bones / `CrabRenderPose` /
//! the sampled `CrabRenderPose` overlay (rl#274).
//!
//! Enforcement: between two `SyncBackend`s the only legitimate writer of a live
//! body part's `Transform` is rapier's own writeback, which mirrors the rigid
//! body's pose. So immediately before `SyncBackend` would consume a change, a
//! `Transform` that diverged from its rigid body's pose is a foreign write.
//! Scale is the strict case: a rigid body pose has no scale at all, so a
//! part `Transform.scale != ONE` is foreign with NO tolerance band — and extra
//! harmful, since bevy_rapier's `apply_scale` rebuilds the collider from it
//! (non-uniform → capsule degraded to a `ConvexPolyhedron`; the rl#314
//! TV wireframe-gap incident).
//! Failure tier mirrors `rescue_lost_crabs` (rl#137): debug panics naming
//! the write; release logs the same message (rate-limited) and snaps the
//! `Transform` back to the body's pose — a visible, loud self-heal that keeps a
//! play session alive with correct physics instead of crashing it.
//!
//! Coverage: any writer in another schedule (`Update`/`PostUpdate`, where render
//! systems live — the incident class) is caught at the next fixed step. A
//! `FixedUpdate` system ordered after [`super::PoseSentinelSet`] and before
//! `SyncBackend` is invisible to the check — that ordering is the SANCTIONED
//! lane for deliberate physics teleports (the rl#240 recenter); anything else
//! of that shape is covered by the static transform-ownership gate
//! (`game/tests/transform_ownership_gate.rs`).
//!
//! Armed only in visual worlds (`Visuals(true)`) — the configuration where
//! render systems exist and which no headless test used to cover. Headless
//! training/test worlds keep the write-`Transform`-to-teleport idiom. Since
//! rl#274 no render system writes a body-part `Transform` at all — GCR renders
//! every part through the sampled `CrabRenderPose` overlay on both arms — so in
//! a correct build this sentinel is a pure tripwire for regressions.
//!
//! Non-finite poses are skipped: NaN is [`super::rescue_lost_crabs`]'s
//! case (a visible respawn), and a foreign write is caught at its first finite
//! divergence, before NaN develops.
//!
//! Assumptions to revisit: divergence == foreign write requires
//! `TimestepMode::Fixed` (rapier's `TimestepMode::Interpolated` writeback would
//! legitimately diverge from the body pose every moving tick); and this whole
//! runtime layer exists because bevy_rapier 0.34 offers no way to DISABLE the
//! changed-Transform consume in SyncBackend — if upstream grows that config, the
//! by-construction fix is to turn consumption off and delete the sentinel.

use bevy::prelude::*;
use bevy_rapier3d::plugin::context::RapierRigidBodySet;
use bevy_rapier3d::prelude::RapierRigidBodyHandle;

use super::RescueBody;
use super::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};

pub fn visuals_on(v: Option<Res<crate::Visuals>>) -> bool {
    v.is_some_and(|v| v.0)
}

/// Rapier's writeback round-trip is angle-wise float-noisy at worst (quat NORM
/// drift is unbounded but excluded by construction — see [`pose_diverges`],
/// rl#297); a cosmetic edit is macroscopic. Deliberate blind band: a writer nudging under these thresholds
/// per fixed step stays invisible (SyncBackend itself consumes ANY change), but
/// the incident class — placing a crab somewhere — is orders of magnitude above
/// them. Shared with `headless::assert_transforms_match_rapier`.
pub const TRANSLATION_EPS: f32 = 1e-3;
pub const ROTATION_DOT_EPS: f32 = 1e-4;

pub fn pose_diverges(t: &Transform, body_pos: Vec3, body_rot: Quat) -> bool {
    // dot(a, b) = |a||b|·cos(θ/2), and rapier's pose integration never renormalizes
    // (rapier3d `RigidBodyVelocity::integrate` has renormalize_fast commented out), so
    // |body_rot| drifts off unit over thousands of steps. A raw dot conflates that norm
    // drift with angular divergence: once |q|² ≤ 1 - ROTATION_DOT_EPS the sentinel fired
    // on bitwise-EQUAL poses every step — and the snap-back rewrote the same drifted
    // quat, so it re-fired forever (rl#297). Scale by the norms to measure ANGLE only.
    (t.translation - body_pos).length() >= TRANSLATION_EPS
        || t.rotation.dot(body_rot).abs()
            <= (1.0 - ROTATION_DOT_EPS) * t.rotation.length() * body_rot.length()
}

/// Log the 1st, 2nd, 4th, 8th… snap-back. Wall/fixed clocks are unusable here:
/// GCR pins `Time` at 0 (the parked pump copies a never-advancing `Time<Fixed>`),
/// so a time-based limiter would log once per process and then heal silently.
fn should_log(snaps: u64) -> bool {
    snaps.is_power_of_two()
}

#[allow(clippy::type_complexity)]
pub fn assert_body_transforms_rapier_owned(
    set_q: Query<&RapierRigidBodySet>,
    mut parts: Query<
        (
            &mut Transform,
            &RapierRigidBodyHandle,
            &CrabEnvId,
            Option<&CrabCarapace>,
            Option<&CrabJoint>,
        ),
        With<CrabBodyPart>,
    >,
    mut snaps: Local<u64>,
    mut warned_no_set: Local<bool>,
) {
    let Ok(set) = set_q.single() else {
        // A visual world with crab parts but no (single) rapier body set means the
        // sentinel is NOT guarding — say so once instead of silently standing down.
        if !parts.is_empty() && !*warned_no_set {
            error!(
                "rl#116 pose sentinel: expected exactly one RapierRigidBodySet, found {} — \
                 body-part Transforms are UNGUARDED in this world",
                set_q.iter().count()
            );
            *warned_no_set = true;
        }
        return;
    };
    for (mut t, h, env, carapace, joint) in parts.iter_mut() {
        let Some(body) = set.bodies.get(h.0) else {
            continue;
        };
        // Scale first, and unconditionally: a rigid body pose has no scale, parts spawn
        // at ONE, and no sanctioned lane (teleport, recenter, rescue) ever writes one —
        // so ANY deviation is a foreign write. It is also the nastiest kind: bevy_rapier's
        // `apply_scale` rebuilds the collider from `GlobalTransform` scale, and past its
        // 1e-4 snap a non-uniform scale silently converts a capsule into a subdivided
        // `ConvexPolyhedron` — different contacts, different cost, and (pre-tracer) an
        // invisible cage part on the TV (the fleet-error this guard came from). Snapping
        // back to ONE makes `apply_scale` rebuild the true shape on the same frame's sync.
        if t.scale != Vec3::ONE {
            let msg = format!(
                "rl#116: a non-physics system wrote SCALE onto a rapier-driven Transform — \
                 crab env {} scale {:?}. bevy_rapier would rebuild the collider at that \
                 scale (a non-uniform one degrades a capsule into a ConvexPolyhedron). \
                 Nothing may scale a CrabBodyPart.",
                env.0, t.scale,
            );
            *snaps += 1;
            if should_log(*snaps) {
                error!(
                    "{msg} Snapping scale back to ONE (VISIBLE self-heal, occurrence {}).",
                    *snaps
                );
            }
            t.scale = Vec3::ONE;
            #[cfg(debug_assertions)]
            panic!("{msg}");
        }
        let iso = body.position();
        let body_pos: Vec3 = iso.translation;
        let body_rot: Quat = iso.rotation;
        if !t.translation.is_finite()
            || !t.rotation.is_finite()
            || !body_pos.is_finite()
            || !body_rot.is_finite()
        {
            continue;
        }
        if !pose_diverges(&t, body_pos, body_rot) {
            continue;
        }
        let part = if carapace.is_some() {
            RescueBody::Carapace
        } else if let Some(j) = joint {
            RescueBody::Joint(j.id)
        } else {
            RescueBody::Unknown
        };
        let msg = format!(
            "rl#116: a non-physics system wrote a rapier-driven Transform — crab env {} \
             `{part}` Transform is {:?}/{:?} but its rigid body is at {:?}/{:?}. SyncBackend \
             would teleport the body into this pose and blow up the solver. Cosmetic/render \
             placement must use the render-only proxies (skin bones / CrabRenderPose), never \
             a CrabBodyPart Transform.",
            env.0, t.translation, t.rotation, body_pos, body_rot,
        );
        *snaps += 1;
        if should_log(*snaps) {
            error!(
                "{msg} Snapping the Transform back to the body pose (VISIBLE self-heal, \
                 occurrence {}).",
                *snaps
            );
        }
        t.translation = body_pos;
        t.rotation = body_rot;
        #[cfg(debug_assertions)]
        panic!("{msg}");
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use super::pose_diverges;
    use crate::bot::body::CrabCarapace;
    use crate::bot::headless::{HeadlessStack, WorldRole, headless_stack, tick};

    /// A body quat after rapier's non-renormalizing integration drifted its norm to
    /// |q|² ≈ 0.9999 — captured verbatim from the rl#297 lavapipe repro log.
    const DRIFTED: Quat = Quat::from_xyzw(-0.64817595, -0.2963903, 0.46590018, 0.52426875);

    #[test]
    fn equal_poses_with_drifted_norm_do_not_diverge() {
        // The captured quat sits 0 ulps under the old firing threshold (it was logged
        // at the first step the inclusive `<=` fired); nudge it a hair further under
        // so the guard doesn't hinge on glam's exact summation order.
        let drifted = DRIFTED * 0.99997;
        assert!(
            drifted.length_squared() <= 1.0 - super::ROTATION_DOT_EPS,
            "fixture must reproduce the drift that made the raw dot fire on equality"
        );
        let t = Transform::from_translation(Vec3::new(70.4351, -235.33363, -305.16962))
            .with_rotation(drifted);
        assert!(!pose_diverges(&t, t.translation, drifted));
    }

    #[test]
    fn real_divergence_still_fires_despite_drifted_norm() {
        let pos = Vec3::new(70.4351, -235.33363, -305.16962);
        let rotated =
            Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(0.1) * DRIFTED);
        assert!(
            pose_diverges(&rotated, pos, DRIFTED),
            "a ~5.7° foreign rotation"
        );
        let moved = Transform::from_translation(pos + Vec3::X * 0.01).with_rotation(DRIFTED);
        assert!(
            pose_diverges(&moved, pos, DRIFTED),
            "a 1cm foreign translation"
        );
    }

    fn visual_app() -> App {
        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            grid: crate::terrain::TerrainGrid::gcr(),
            visuals: crate::Visuals(true),
        });
        tick(&mut app, 64);
        app
    }

    #[test]
    fn sentinel_stays_quiet_when_only_rapier_writes() {
        let mut app = visual_app();
        tick(&mut app, 64);
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        let t = q.single(app.world()).expect("carapace");
        assert!(t.translation.is_finite(), "settled crab is finite");
    }

    // Debug-only: release builds take the snap-back path instead of panicking.
    #[cfg(debug_assertions)]
    #[test]
    fn sentinel_panics_on_a_cosmetic_body_write() {
        let mut app = visual_app();
        {
            // The play-day incident in miniature: a "cosmetic" world-placement shift
            // written straight onto the rapier-driven carapace.
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.translation.x += 12.0;
        }
        // The harness captures this test's output, so the expected panic prints nothing
        // on success; no panic-hook suppression (it is process-global and would swallow
        // a concurrent test's message).
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| app.update()));
        let payload = result.expect_err("the sentinel must panic before SyncBackend");
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("rl#116"),
            "panic must name the invariant, got: {msg}"
        );
    }

    // Debug-only for the same reason; release takes the snap-to-ONE self-heal.
    #[cfg(debug_assertions)]
    #[test]
    fn sentinel_panics_on_a_foreign_scale_write() {
        let mut app = visual_app();
        {
            // The rl#314 incident in miniature: a non-uniform scale on a rapier-driven
            // part, which `apply_scale` would bake into the collider shape.
            let mut q = app
                .world_mut()
                .query_filtered::<&mut Transform, With<CrabCarapace>>();
            let mut t = q.single_mut(app.world_mut()).expect("carapace");
            t.scale = Vec3::new(1.0, 1.2, 1.0);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| app.update()));
        let payload = result.expect_err("the sentinel must panic on a foreign scale");
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("SCALE"),
            "panic must name the scale write, got: {msg}"
        );
    }
}
