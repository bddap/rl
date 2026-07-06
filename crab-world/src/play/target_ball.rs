
use bevy::prelude::*;

use crate::bot::CrabSpawns;
use crate::bot::body::{self, CrabClawTip};
use crate::bot::sensor::CrabTargets;
use crate::training::reward::dist_3d;
use crate::training::targets::sample_target;

#[derive(Component)]
pub(super) struct TargetBall;

const DEMO_REACH_RADIUS: f32 = crate::training::targets::REACH_RADIUS;

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
    mut targets: ResMut<CrabTargets>,
    claw_tips_q: Query<(&body::CrabEnvId, &Transform), With<CrabClawTip>>,
    mut ball_q: Query<&mut Transform, (With<TargetBall>, Without<CrabClawTip>)>,
    mut rng: ResMut<super::DemoRng>,
) {
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);

    let mut target = match targets.get(0) {
        Some(t) => t,
        None => target_ball_at_from_env().unwrap_or_else(|| sample_target(origin, &mut rng.0)),
    };

    let mut min_dist = f32::INFINITY;
    for (env, tip) in claw_tips_q.iter() {
        if env.0 == 0 && tip.translation.is_finite() {
            min_dist = min_dist.min(dist_3d(tip.translation, target));
        }
    }

    if min_dist <= DEMO_REACH_RADIUS {
        target = sample_target(origin, &mut rng.0);
    }

    if let Some(slot) = targets.envs.first_mut() {
        *slot = Some(target);
    }
    if let Ok(mut ball) = ball_q.single_mut() {
        ball.translation = target;
    }
}

fn target_ball_at_from_env() -> Option<Vec3> {
    let raw = std::env::var("RL_TARGET_BALL_AT").ok()?;
    let parts: Vec<f32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    match parts.as_slice() {
        [x, y, z] => Some(Vec3::new(*x, *y, *z)),
        _ => None,
    }
}
