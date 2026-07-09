use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use super::body::{CrabAssets, CrabBodyPart, CrabCarapace, CrabEnvId};
use super::headless::{assert_transforms_match_rapier, headless_app, tick};
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

fn assert_crab_sane(app: &mut App, n_parts: usize, context: &str) {
    let translations = part_translations(app);
    assert_eq!(translations.len(), n_parts, "{context}: part count");
    for t in &translations {
        assert!(t.is_finite(), "{context}: non-finite part at {t:?}");
        assert!(t.length() < 3.0, "{context}: part {t:?} far from spawn");
    }
    let h = carapace_height(app);
    assert!(
        (-0.2..1.5).contains(&h),
        "{context}: carapace height {h} not grounded (tunneled or launched)"
    );
    assert_transforms_match_rapier(app);
}

#[test]
fn despawn_respawn_survives_rapier_and_lands_sane() {
    let mut app = headless_app();
    tick(&mut app, 192);

    let n_parts = part_translations(&mut app).len();
    assert!(n_parts > 10, "expected a whole crab, got {n_parts} parts");

    respawn_env0(&mut app);
    tick(&mut app, 8);
    assert_eq!(
        part_translations(&mut app).len(),
        n_parts,
        "respawn must rebuild the full crab"
    );

    tick(&mut app, 184);
    assert_crab_sane(&mut app, n_parts, "after respawn");
}

/// GCR's rl#240 recenter guard teleports a drifted crab by writing every part's
/// Transform in one tick. This pins the mechanism it rides on: bevy_rapier treats a
/// uniform Transform shift as a clean multibody teleport — joints intact, nothing
/// scattered or snapped back.
#[test]
fn uniform_part_shift_teleports_the_multibody_cleanly() {
    use super::headless::{HeadlessStack, WorldRole, headless_stack};

    // OpenField: the landing spot must still have ground — the real teleport happens
    // far outside the walled box.
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        arena: crate::physics::Arena::OpenField,
    });
    tick(&mut app, 192);

    let before = part_translations(&mut app);
    let delta = Vec3::new(12.0, 0.0, -7.0);
    {
        let mut q = app
            .world_mut()
            .query_filtered::<&mut Transform, With<CrabBodyPart>>();
        for mut t in q.iter_mut(app.world_mut()) {
            t.translation += delta;
        }
    }
    tick(&mut app, 64);

    let after = part_translations(&mut app);
    assert_eq!(after.len(), before.len(), "no part lost in the teleport");
    // No structural change between the two reads, so query order is stable and the
    // lists pair up. Loose tolerance: the standing crab keeps micro-settling.
    for (b, a) in before.iter().zip(&after) {
        assert!(a.is_finite(), "non-finite part after teleport: {a:?}");
        assert!(
            (*a - (*b + delta)).length() < 0.5,
            "part scattered by teleport: {b:?} + {delta:?} -> {a:?}"
        );
    }
    assert_transforms_match_rapier(&mut app);
}

#[derive(Resource, Default)]
struct TestShove(Vec3);

fn apply_test_shove(
    shove: Res<TestShove>,
    mut q: Query<&mut bevy_rapier3d::prelude::ExternalForce, With<CrabCarapace>>,
) {
    if let Ok(mut f) = q.single_mut() {
        f.force += shove.0;
    }
}

#[test]
fn external_force_shoves_a_multibody_root() {
    use bevy_rapier3d::plugin::PhysicsSet;

    let mut app = headless_app();
    app.init_resource::<TestShove>();
    app.add_systems(
        FixedUpdate,
        apply_test_shove
            .after(crate::bot::BotSet::Act)
            .before(PhysicsSet::SyncBackend),
    );
    tick(&mut app, 192);

    let carapace_x = |app: &mut App| {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        q.single(app.world()).expect("carapace").translation.x
    };
    let x0 = carapace_x(&mut app);

    app.world_mut().resource_mut::<TestShove>().0 = Vec3::new(70.0, 0.0, 0.0);
    tick(&mut app, 8);
    app.world_mut().resource_mut::<TestShove>().0 = Vec3::ZERO;
    tick(&mut app, 16);

    let x1 = carapace_x(&mut app);
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

    {
        let mut q = app
            .world_mut()
            .query_filtered::<&mut Transform, With<CrabCarapace>>();
        let mut transform = q.single_mut(app.world_mut()).expect("carapace");
        transform.translation = Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    }

    tick(&mut app, 192);
    assert_crab_sane(&mut app, n_parts, "after rescue from NaN");

    tick(&mut app, 64);
    assert_crab_sane(&mut app, n_parts, "64 ticks later");
}
