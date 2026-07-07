use bevy::prelude::*;

use super::components::CrabBodyPart;

#[derive(Default, Reflect, GizmoConfigGroup)]
#[reflect(Default)]
pub struct PivotGizmos;

const PIVOT_MARKER_RADIUS: f32 = 0.02;

fn draw_pivot_markers(
    mode: Option<Res<crate::crab_view::RenderMode>>,
    parts: Query<&GlobalTransform, With<CrabBodyPart>>,
    mut gizmos: Gizmos<PivotGizmos>,
) {
    if !mode.map(|m| m.shows_colliders()).unwrap_or(true) {
        return;
    }
    let color = Color::srgb(1.0, 0.0, 1.0);
    for gt in &parts {
        gizmos.sphere(
            Isometry3d::from_translation(gt.translation()),
            PIVOT_MARKER_RADIUS,
            color,
        );
    }
}

pub fn register_pivot_markers(app: &mut App) {
    app.insert_gizmo_config(
        PivotGizmos,
        GizmoConfig {
            depth_bias: -1.0,
            ..default()
        },
    );
    app.add_systems(Update, draw_pivot_markers);
}
