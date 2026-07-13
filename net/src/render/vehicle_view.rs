//! Simple models for OTHER pilots' craft (rl#260): every remote craft renders as its kind's
//! cuboid silhouette ([`VehicleKind::silhouette`]) instead of a bare collider wireframe.
//! Kind and pose come off the same wire data on both arms ([`RemoteVehicle`] — the host from
//! its capture, a client from its adopt), so host and client render through ONE path. The
//! local pilot's own craft stays unrendered: the cockpit camera flies from it. The wireframe
//! pass remains the colliders-mode view ([`render_mode`]).

use crab_world::crab_view::RenderMode;
use crab_world::vehicle::{PilotId, VehicleKind};

use super::articulation::{RemoteVehicle, SampledCraft};
use super::*;

#[derive(Resource)]
struct VehicleAssets {
    plane: KindAssets,
    ship: KindAssets,
}

struct KindAssets {
    material: Handle<StandardMaterial>,
    /// One mesh per silhouette part (dims baked in), with its body-frame offset.
    parts: Vec<(Handle<Mesh>, Vec3)>,
}

impl VehicleAssets {
    fn of(&self, kind: VehicleKind) -> &KindAssets {
        match kind {
            VehicleKind::Plane => &self.plane,
            VehicleKind::Ship => &self.ship,
        }
    }
}

/// Marks one remote craft's model root, keyed by the wire identity that spawned it.
#[derive(Component)]
struct VehicleModel {
    pilot: PilotId,
    kind: VehicleKind,
}

pub(super) fn register(app: &mut App) {
    // RemoteVehicle is round state — `install_round` inserts it on every path into Playing.
    app.add_systems(Startup, build_assets);
    // PostUpdate, after the Update chain has published this frame's RemoteVehicle (an
    // unordered Update slot could run first and leave every model a tick stale, visibly
    // shearing off the fresh wireframe in mesh+colliders mode), before propagation so the
    // root pose lands in this frame's GlobalTransforms.
    app.add_systems(
        PostUpdate,
        reconcile_vehicle_models
            .before(TransformSystems::Propagate)
            .run_if(in_state(AppPhase::Playing)),
    );
}

fn build_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut build = |kind: VehicleKind, color: Color| KindAssets {
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
    commands.insert_resource(VehicleAssets {
        plane: build(VehicleKind::Plane, Color::srgb(0.85, 0.86, 0.9)),
        ship: build(VehicleKind::Ship, Color::srgb(0.35, 0.5, 0.75)),
    });
}

/// Keep one model per remote craft: spawn on a pilot's first wire pose (or a kind cycle —
/// the stale model despawns and a fresh one spawns in the same pass), track the pose while
/// it flies, despawn on step-out. Poses sample per frame on the uniform physics-step
/// clock (rl#267) — the same [`super::pose::PoseWindow`] law as the cockpit, so a watched
/// craft doesn't step at raw tick cadence — and place through the STATIC arena→render
/// frame (rl#224), exactly like the wireframe pass.
fn reconcile_vehicle_models(
    mut commands: Commands,
    assets: Res<VehicleAssets>,
    remote: Res<RemoteVehicle>,
    clock: Res<super::driver::RenderClock>,
    anchor: Res<crate::external_crab::ArenaAnchor>,
    mode: Res<RenderMode>,
    mut models: Query<(Entity, &VehicleModel, &mut Transform, &mut Visibility)>,
) {
    let sampled = remote.sample(clock.tick, clock.frac);
    let want_vis = mode.mesh_visibility();
    let placed = |c: &SampledCraft| {
        Transform::from_translation(anchor.0 + c.pose.pos).with_rotation(c.pose.orient)
    };
    let mut matched = std::collections::BTreeSet::new();
    for (entity, model, mut tf, mut vis) in &mut models {
        match sampled.iter().find(|v| v.pilot == model.pilot) {
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
    for v in &sampled {
        if matched.contains(&v.pilot) {
            continue;
        }
        let k = assets.of(v.kind);
        commands
            .spawn((
                DespawnOnExit(AppPhase::Playing),
                VehicleModel {
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

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::articulation::VehiclePoseWire;

    /// [`VehicleAssets`] with default (empty) handles but the REAL per-kind part counts, so
    /// the reconcile logic runs without any render stack.
    fn stub_assets() -> VehicleAssets {
        let stub = |kind: VehicleKind| KindAssets {
            material: Handle::default(),
            parts: kind
                .silhouette()
                .iter()
                .map(|p| (Handle::default(), p.offset))
                .collect(),
        };
        VehicleAssets {
            plane: stub(VehicleKind::Plane),
            ship: stub(VehicleKind::Ship),
        }
    }

    const ANCHOR: Vec3 = Vec3::new(3.0, 0.0, -7.0);

    fn world_with(remote: Vec<VehiclePoseWire>, mode: RenderMode) -> World {
        let mut w = World::new();
        w.insert_resource(stub_assets());
        let mut rv = RemoteVehicle::default();
        rv.adopt(1, &remote);
        w.insert_resource(rv);
        // A 1-deep window samples its newest pose raw, so these reconcile tests see the
        // exact wire poses they fed; the interpolation law itself is pinned in pose.rs
        // and articulation.rs.
        w.insert_resource(super::super::driver::RenderClock { tick: 1, frac: 0.0 });
        w.insert_resource(crate::external_crab::ArenaAnchor(ANCHOR));
        w.insert_resource(mode);
        w
    }

    /// Adopt a fresh wire set at the next tick and move the render clock with it.
    fn readopt(w: &mut World, tick: u64, remote: Vec<VehiclePoseWire>) {
        w.resource_mut::<RemoteVehicle>().adopt(tick, &remote);
        w.insert_resource(super::super::driver::RenderClock { tick, frac: 0.0 });
    }

    fn wire(pilot: u8, kind: VehicleKind, pos: [f32; 3]) -> VehiclePoseWire {
        VehiclePoseWire {
            pilot,
            kind,
            pos,
            rot: [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn models(w: &mut World) -> Vec<(PilotId, VehicleKind, Vec3, Visibility, usize)> {
        let mut out: Vec<_> = w
            .query::<(&VehicleModel, &Transform, &Visibility, &Children)>()
            .iter(w)
            .map(|(m, t, v, c)| (m.pilot, m.kind, t.translation, *v, c.len()))
            .collect();
        out.sort_by_key(|e| e.0);
        out
    }

    #[test]
    fn spawns_tracks_and_despawns_remote_craft_models() {
        let mut w = world_with(
            vec![wire(1, VehicleKind::Plane, [2.0, 5.0, -1.0])],
            RenderMode::Mesh,
        );
        w.run_system_once(reconcile_vehicle_models).unwrap();
        let got = models(&mut w);
        assert_eq!(got.len(), 1, "one model per remote craft");
        let (pilot, kind, at, vis, n_parts) = got[0];
        assert_eq!((pilot, kind), (PilotId(1), VehicleKind::Plane));
        assert_eq!(
            at,
            ANCHOR + Vec3::new(2.0, 5.0, -1.0),
            "pose places through the arena anchor, like the wireframe pass"
        );
        assert_eq!(vis, Visibility::Visible);
        assert_eq!(
            n_parts,
            VehicleKind::Plane.silhouette().len(),
            "one child per silhouette part"
        );

        // The craft moves: the SAME entity tracks (still exactly one model).
        readopt(
            &mut w,
            2,
            vec![wire(1, VehicleKind::Plane, [4.0, 6.0, 0.0])],
        );
        w.run_system_once(reconcile_vehicle_models).unwrap();
        let got = models(&mut w);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].2, ANCHOR + Vec3::new(4.0, 6.0, 0.0));

        // Pilot steps out: the model despawns, children included.
        readopt(&mut w, 3, Vec::new());
        w.run_system_once(reconcile_vehicle_models).unwrap();
        assert!(models(&mut w).is_empty(), "step-out despawns the model");
        assert_eq!(
            w.query::<&Mesh3d>().iter(&w).count(),
            0,
            "no orphaned part meshes"
        );
    }

    #[test]
    fn kind_cycle_swaps_the_model_in_one_pass() {
        let mut w = world_with(
            vec![wire(2, VehicleKind::Plane, [0.0, 2.0, 0.0])],
            RenderMode::Mesh,
        );
        w.run_system_once(reconcile_vehicle_models).unwrap();
        readopt(&mut w, 2, vec![wire(2, VehicleKind::Ship, [0.0, 2.0, 0.0])]);
        w.run_system_once(reconcile_vehicle_models).unwrap();
        let got = models(&mut w);
        assert_eq!(got.len(), 1, "the stale kind's model is gone");
        assert_eq!(got[0].1, VehicleKind::Ship);
        assert_eq!(got[0].4, VehicleKind::Ship.silhouette().len());
    }

    #[test]
    fn colliders_mode_hides_the_models() {
        let mut w = world_with(
            vec![wire(1, VehicleKind::Ship, [0.0, 2.0, 0.0])],
            RenderMode::Colliders,
        );
        w.run_system_once(reconcile_vehicle_models).unwrap();
        assert_eq!(
            models(&mut w)[0].3,
            Visibility::Hidden,
            "colliders mode shows the wireframe, not the model"
        );
    }
}
