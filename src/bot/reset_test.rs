//! Pins the one reliable crab-reset mechanism: full despawn + respawn
//! ([`respawn_crab`]).
//!
//! Teleport-and-zero resets cannot recover a crab whose multibody state has
//! gone non-finite — rapier 0.32 has no API to rewrite multibody joint
//! coordinates, so poisoned joint state survives any amount of Transform and
//! Velocity writing, and every such "reset" reproduces the same wedged pose
//! (one overnight run burned 12 h that way: every episode 1 step long, eyes
//! frozen 90 m under the floor). Note rapier's joint-removal bug (upstream
//! #927) is about removing joints from a live multibody; dropping the whole
//! tree at once must work, and the first test pins exactly that.

#![cfg(test)]

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::body::{CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId};
use super::test_util::{assert_transforms_match_rapier, headless_app, tick};
use super::{CrabSpawns, respawn_crab};

fn part_translations(app: &mut App) -> Vec<Vec3> {
    let mut q = app
        .world_mut()
        .query_filtered::<&Transform, With<CrabBodyPart>>();
    q.iter(app.world()).map(|t| t.translation).collect()
}

fn carapace_height(app: &mut App) -> f32 {
    let mut q = app
        .world_mut()
        .query_filtered::<&Transform, With<CrabCarapace>>();
    q.single(app.world()).expect("carapace").translation.y
}

fn respawn_env0(app: &mut App) {
    app.world_mut()
        .run_system_once(
            |mut commands: Commands,
             assets: Res<CrabAssets>,
             spawns: Res<CrabSpawns>,
             parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>| {
                let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
                respawn_crab(
                    &mut commands,
                    &assets,
                    parts.iter().filter(|(_, id)| id.0 == 0).map(|(e, _)| e),
                    origin,
                    0,
                );
            },
        )
        .expect("respawn system");
}

/// A healthy, settled crab: every part finite and near the spawn point, body
/// standing at roughly rest height.
fn assert_crab_sane(app: &mut App, n_parts: usize, context: &str) {
    let translations = part_translations(app);
    assert_eq!(translations.len(), n_parts, "{context}: part count");
    for t in &translations {
        assert!(t.is_finite(), "{context}: non-finite part at {t:?}");
        assert!(
            t.length() < 3.0,
            "{context}: part {t:?} far from spawn"
        );
    }
    let h = carapace_height(app);
    assert!(
        (0.3..0.9).contains(&h),
        "{context}: carapace height {h} not a stand"
    );
    assert_transforms_match_rapier(app);
}

#[test]
fn despawn_respawn_survives_rapier_and_lands_sane() {
    let mut app = headless_app();
    tick(&mut app, 192); // settle into the motor-held stance

    let n_parts = part_translations(&mut app).len();
    assert!(n_parts > 10, "expected a whole crab, got {n_parts} parts");

    respawn_env0(&mut app);
    // Commands apply, rapier digests the removal + insertion, physics steps.
    // A panic anywhere here is rapier failing the teardown.
    tick(&mut app, 8);
    assert_eq!(
        part_translations(&mut app).len(),
        n_parts,
        "respawn must rebuild the full crab"
    );

    tick(&mut app, 184); // settle the fresh crab
    assert_crab_sane(&mut app, n_parts, "after respawn");
}

#[test]
fn external_force_shoves_a_multibody_root() {
    // Velocity writes to a multibody root are silently ignored — its velocity
    // lives in the multibody's generalized coordinates, which the component
    // writeback never touches (the demo's poke was a no-op for exactly that
    // reason, issue #14). Per-link external FORCES, by contrast, are mapped
    // through the body Jacobians into generalized accelerations. Pin the
    // working channel: a sideways force burst on the settled crab must
    // actually move it.
    use bevy_rapier3d::prelude::ExternalForce;

    let mut app = headless_app();
    tick(&mut app, 192);

    let x0 = {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        q.single(app.world()).expect("carapace").translation.x
    };

    // 70 N for 8 ticks (0.125 s), the demo poke's burst.
    {
        let mut q = app
            .world_mut()
            .query_filtered::<&mut ExternalForce, With<CrabCarapace>>();
        let mut f = q.single_mut(app.world_mut()).expect("carapace force");
        f.force = Vec3::new(70.0, 0.0, 0.0);
    }
    tick(&mut app, 8);
    {
        let mut q = app
            .world_mut()
            .query_filtered::<&mut ExternalForce, With<CrabCarapace>>();
        let mut f = q.single_mut(app.world_mut()).expect("carapace force");
        f.force = Vec3::ZERO;
    }
    tick(&mut app, 16);

    let x1 = {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        q.single(app.world()).expect("carapace").translation.x
    };
    assert!(
        x1 - x0 > 0.05,
        "a 70 N / 0.125 s shove must visibly move the crab: x {x0:+.3} -> {x1:+.3}"
    );
}

#[test]
fn rescue_system_recovers_a_nan_poisoned_crab() {
    let mut app = headless_app();
    tick(&mut app, 192);
    let n_parts = part_translations(&mut app).len();

    // Poison the sim the way a tunneling blowup does: a non-finite root pose.
    // (Velocity writes don't reach a multibody root — its velocity lives in
    // the multibody's generalized coordinates — but pose writes do, the same
    // path episode-reset teleports used.) Without rescue_nonfinite_crabs
    // running ahead of the physics sync, the next solver step panics on NaN
    // motor-clamp bounds; with it, the poisoned crab must be torn down and
    // rebuilt before the solver ever sees it.
    {
        let mut q = app
            .world_mut()
            .query_filtered::<&mut Transform, With<CrabCarapace>>();
        let mut transform = q.single_mut(app.world_mut()).expect("carapace");
        transform.translation = Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    }

    tick(&mut app, 192);
    assert_crab_sane(&mut app, n_parts, "after rescue from NaN");

    // And the recovered world keeps working — no relapse into the wedge.
    tick(&mut app, 64);
    assert_crab_sane(&mut app, n_parts, "64 ticks later");
}
