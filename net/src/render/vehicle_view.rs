//! Simple models for OTHER pilots' craft (rl#260): every remote craft renders as its kind's
//! cuboid silhouette ([`VehicleKind::silhouette`]) instead of a bare collider wireframe.
//! Kind and pose come off the same wire data on both arms ([`RemoteVehicle`] — the host from
//! its capture, a client from its adopt), so host and client render through ONE path. The
//! local pilot's own craft stays unrendered: the cockpit camera flies from it. The wireframe
//! pass remains the colliders-mode view ([`render_mode`]).

use crab_world::crab_view::RenderMode;
use crab_world::vehicle::VehicleKind;

use super::articulation::RemoteVehicle;
use super::*;

#[derive(Resource)]
struct CraftAssets {
    plane: CraftKindAssets,
    ship: CraftKindAssets,
}

struct CraftKindAssets {
    material: Handle<StandardMaterial>,
    /// One mesh per silhouette part (dims baked in), with its body-frame offset.
    parts: Vec<(Handle<Mesh>, Vec3)>,
}

impl CraftAssets {
    fn of(&self, kind: VehicleKind) -> &CraftKindAssets {
        match kind {
            VehicleKind::Plane => &self.plane,
            VehicleKind::Ship => &self.ship,
        }
    }
}

/// Marks one remote craft's model root, keyed by the wire identity that spawned it.
#[derive(Component)]
struct CraftModel {
    pilot: u8,
    kind: VehicleKind,
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<RemoteVehicle>();
    app.add_systems(Startup, build_assets);
    app.add_systems(
        Update,
        reconcile_craft_models.run_if(in_state(AppPhase::Playing)),
    );
}

fn build_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut build = |kind: VehicleKind, color: Color| CraftKindAssets {
        material: materials.add(StandardMaterial {
            base_color: color,
            ..default()
        }),
        parts: kind
            .silhouette()
            .iter()
            .map(|p| (meshes.add(Cuboid::from_size(p.half * 2.0)), p.offset))
            .collect(),
    };
    commands.insert_resource(CraftAssets {
        plane: build(VehicleKind::Plane, Color::srgb(0.85, 0.86, 0.9)),
        ship: build(VehicleKind::Ship, Color::srgb(0.35, 0.5, 0.75)),
    });
}

/// Keep one model per remote craft: spawn on a pilot's first wire pose (or a kind cycle —
/// the stale model despawns and a fresh one spawns in the same pass), track the pose while
/// it flies, despawn on step-out. Poses place through the STATIC arena→render frame
/// (rl#224), exactly like the wireframe pass.
fn reconcile_craft_models(
    mut commands: Commands,
    assets: Res<CraftAssets>,
    remote: Res<RemoteVehicle>,
    anchor: Res<crate::external_crab::ArenaAnchor>,
    mode: Res<RenderMode>,
    mut models: Query<(Entity, &CraftModel, &mut Transform, &mut Visibility)>,
) {
    let want_vis = if mode.shows_mesh() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    let placed = |v: &crate::articulation::VehiclePoseWire| {
        Transform::from_translation(anchor.0 + Vec3::from_array(v.pos))
            .with_rotation(Quat::from_array(v.rot))
    };
    let mut matched = std::collections::BTreeSet::new();
    for (entity, model, mut tf, mut vis) in &mut models {
        match remote.0.iter().find(|v| v.pilot == model.pilot) {
            Some(v) if v.kind == model.kind => {
                matched.insert(model.pilot);
                *tf = placed(v);
                if *vis != want_vis {
                    *vis = want_vis;
                }
            }
            _ => commands.entity(entity).despawn(),
        }
    }
    for v in &remote.0 {
        if matched.contains(&v.pilot) {
            continue;
        }
        let k = assets.of(v.kind);
        commands
            .spawn((
                DespawnOnExit(AppPhase::Playing),
                CraftModel {
                    pilot: v.pilot,
                    kind: v.kind,
                },
                placed(v),
                want_vis,
            ))
            .with_children(|parts| {
                for (mesh, offset) in &k.parts {
                    parts.spawn((
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(k.material.clone()),
                        Transform::from_translation(*offset),
                    ));
                }
            });
    }
}
