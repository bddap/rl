use bevy::prelude::*;

use crate::bot::CrabSpawns;
use crate::bot::body::{self, CrabClawTip};
use crate::bot::sensor::CrabTargets;
use crate::training::targets::{closest_tip_dist, sample_target, tip_touch};

#[derive(Component)]
pub(super) struct TargetBall;

/// A pinned target position (rl-demo `--target-ball-at x,y,z`): the ball holds here
/// instead of sampling, until a claw touch re-rolls it. `None` = sample normally.
#[derive(Resource, Default, Clone, Copy)]
pub(super) struct TargetBallAt(pub(super) Option<Vec3>);

/// The demo ball mirrors the trained distribution, close disc included (rl#292:
/// ball-under is canonical, not a curriculum flag — and a ball she grabs at her
/// feet is the skill on display). One source with training's mix on purpose; a
/// demo pinned to the band would show a distribution she no longer trains.
const DEMO_CLOSE_FRAC: f32 = crate::training::targets::CLOSE_FRAC;

const TARGET_BALL_RADIUS: f32 = 0.08;

pub(super) fn spawn_target_ball(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(TARGET_BALL_RADIUS))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.05, 0.05),
            emissive: LinearRgba::new(1.6, 0.0, 0.0, 1.0),
            ..default()
        })),
        Transform::from_translation(Vec3::ZERO),
        TargetBall,
    ));
}

pub(super) fn target_ball(
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
    mut targets: ResMut<CrabTargets>,
    claw_tips_q: Query<(&body::CrabEnvId, &Transform), With<CrabClawTip>>,
    mut ball_q: Query<&mut Transform, (With<TargetBall>, Without<CrabClawTip>)>,
    mut rng: ResMut<super::DemoRng>,
    pinned: Res<TargetBallAt>,
) {
    let origin = spawns.origin(0);

    let mut target = match targets.get(0) {
        Some(t) => t,
        None => pinned
            .0
            .unwrap_or_else(|| sample_target(origin, DEMO_CLOSE_FRAC, &mut rng.0, &terrain)),
    };

    if closest_tip_dist(0, target, &claw_tips_q).is_some_and(tip_touch) {
        target = sample_target(origin, DEMO_CLOSE_FRAC, &mut rng.0, &terrain);
    }

    if let Some(slot) = targets.envs.first_mut() {
        *slot = Some(target);
    }
    if let Ok(mut ball) = ball_q.single_mut() {
        ball.translation = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::BotSet;
    use crate::bot::body::CrabCarapace;
    use crate::bot::headless::{flat_headless_app, tick};
    use crate::bot::sensor::CrabObservation;

    /// Env 0's carapace transform, read straight off the world.
    fn carapace(app: &mut App) -> Transform {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        *q.single(app.world()).expect("carapace")
    }

    /// The target slots of env 0's serialized observation — exactly the floats
    /// `policy_step` hands the network.
    fn obs_target(app: &App) -> Vec3 {
        let obs = app.world().resource::<CrabObservation>();
        obs.env(0).expect("env 0 sized").target_local()
    }

    /// Pins the demo's target WIRE end to end: [`target_ball`] seeds/holds
    /// [`CrabTargets`], Sense rotates the LIVE ball position into the carapace frame,
    /// and the slots the policy reads TRACK a moved ball. This is the wire the rl#228
    /// playtest put in question ("ignores ball location") — with it pinned, a
    /// no-pursuit demo is attributable to the brain, never to a silently dead or
    /// constant target input.
    #[test]
    fn demo_target_obs_tracks_moved_ball() {
        let mut app = flat_headless_app();
        app.init_resource::<super::super::DemoRng>();
        app.init_resource::<TargetBallAt>();
        // The demo/render-video schedule for this system, minus rendering: after Sense,
        // so it reads the post-physics state the observation consumed.
        app.add_systems(FixedUpdate, target_ball.after(BotSet::Sense));

        // Let target_ball seed a target and the un-driven crab settle to rest, so the
        // carapace barely moves between a tick's Sense and its end-of-tick transform
        // (the comparison below reads the transform after the tick).
        tick(&mut app, 240);
        let seeded = app
            .world()
            .resource::<CrabTargets>()
            .get(0)
            .expect("target_ball seeds env 0's target");
        assert!(
            seeded != Vec3::ZERO && obs_target(&app) != Vec3::ZERO,
            "seeded target must reach the observation"
        );

        // Move the ball to two distinct world points through the same slot the demo
        // writes; the obs the policy reads must follow each move, carapace-local.
        let mut seen = Vec::new();
        for p in [Vec3::new(6.0, 1.0, 0.0), Vec3::new(-4.0, 1.0, 5.0)] {
            app.world_mut().resource_mut::<CrabTargets>().envs[0] = Some(p);
            tick(&mut app, 1);
            let c = carapace(&mut app);
            let expected = c.rotation.inverse() * (p - c.translation);
            let got = obs_target(&app);
            assert!(
                (got - expected).length() < 0.05,
                "obs target {got} != carapace-local ball {expected} for world ball {p}"
            );
            seen.push(got);
        }
        assert!(
            (seen[0] - seen[1]).length() > 1.0,
            "obs must vary when the ball moves: {} vs {}",
            seen[0],
            seen[1]
        );
    }
}
