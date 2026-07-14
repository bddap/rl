use bevy::prelude::*;
use bevy::ui::{IsDefaultUiCamera, UiScale};
use bevy_rapier3d::geometry::ColliderView;
use bevy_rapier3d::prelude::Collider;

use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId};
use crate::bot::skin::{CrabRenderPose, CrabSkinRepose};

pub const COLLIDER_WIREFRAME_COLOR: Color = Color::srgb(0.2, 1.0, 0.4);

/// The ONE HUD-text green, shared by the corner render-mode label and the floating brain
/// labels — one source, so the overlay can't drift into two near-greens.
const HUD_TEXT_COLOR: Color = Color::srgb(0.4, 1.0, 0.55);

/// Which view of the crab a render surface shows. A [`clap::ValueEnum`] because it IS a CLI
/// value ([`crate::RenderArgs`]): clap owns the string→mode mapping, so an unrecognized value
/// is a parse error at t=0 rather than anything this code has to decide.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Default, Debug, clap::ValueEnum)]
pub enum RenderMode {
    #[default]
    Mesh,
    #[value(name = "mesh+colliders")]
    MeshColliders,
    Colliders,
}

impl RenderMode {
    pub fn next(self) -> Self {
        match self {
            RenderMode::Mesh => RenderMode::MeshColliders,
            RenderMode::MeshColliders => RenderMode::Colliders,
            RenderMode::Colliders => RenderMode::Mesh,
        }
    }

    pub fn shows_mesh(self) -> bool {
        !matches!(self, RenderMode::Colliders)
    }

    /// The ONE mode→mesh-visibility mapping, shared by every entity-visibility toggle (crab
    /// silhouette, craft models) so they can't drift.
    pub fn mesh_visibility(self) -> Visibility {
        if self.shows_mesh() {
            Visibility::Visible
        } else {
            Visibility::Hidden
        }
    }

    pub fn shows_colliders(self) -> bool {
        matches!(self, RenderMode::MeshColliders | RenderMode::Colliders)
    }

    pub fn label(self) -> &'static str {
        match self {
            RenderMode::Mesh => "mesh",
            RenderMode::MeshColliders => "mesh + colliders",
            RenderMode::Colliders => "colliders",
        }
    }
}

#[derive(Component)]
struct RenderModeLabel;

pub fn register<M>(app: &mut App, initial: RenderMode, cage_gate: impl SystemCondition<M>) {
    app.insert_resource(initial);
    app.init_resource::<CrabSkinRepose>();
    app.init_resource::<CrabRenderPose>();
    app.add_systems(Startup, spawn_render_mode_label);
    app.add_systems(Update, update_render_mode_label);
    // The per-crab brain labels (rl#200 increment 7). Visibility follows the DATA, not a
    // phase gate: nodes exist iff `CrabBrainLabels` has entries, so each binary controls the
    // labels by publishing/clearing the resource (the demo republishes write-on-change and
    // never clears; GCR publishes from its bindings and clears at round teardown — no stale
    // label can float over a menu the way an ungated gizmo cage did, rl#211).
    app.init_resource::<CrabBrainLabels>();
    app.add_systems(Update, sync_brain_label_nodes);
    // After transform propagation for the same reason as the cage: project THIS frame's
    // camera + carapace poses, not last frame's.
    app.add_systems(
        PostUpdate,
        position_brain_labels.after(TransformSystems::Propagate),
    );
    app.add_systems(
        PostUpdate,
        draw_crab_collider_wireframe
            .after(TransformSystems::Propagate)
            .run_if(cage_gate),
    );
}

fn spawn_render_mode_label(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(HUD_TEXT_COLOR),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(14.0),
            right: Val::Px(14.0),
            ..default()
        },
        RenderModeLabel,
    ));
}

fn update_render_mode_label(
    mode: Res<RenderMode>,
    mut label: Query<&mut Text, With<RenderModeLabel>>,
) {
    if !mode.is_changed() {
        return;
    }
    if let Ok(mut text) = label.single_mut() {
        **text = format!("Render: {}", mode.label());
    }
}

/// Per-crab brain labels, index-aligned with [`CrabEnvId`]: `labels.0[i]` is the finished
/// display string for env `i`'s crab — `arch @shortdigest`, or its attributed failure state
/// ("REFUSED: …", "no brain (rest pose)"). The ONE world-space label system for both
/// binaries (rl#200 increment 7): each publisher formats through `Policy::brain_label`
/// (the demo and the GCR host directly; GCR clients receive the host's strings over the
/// articulation wire), and this module renders whatever is here. Empty (the default) means
/// no labels — publishing and clearing IS the visibility control.
#[derive(Resource, Default, Clone, PartialEq, Eq, Debug)]
pub struct CrabBrainLabels(pub Vec<String>);

/// How far above the carapace center a crab's brain label floats, in world meters
/// (render==physics: the crab stands ~0.5 m, so this clears the raised claws in both
/// binaries — the GCR "giant" feel comes from shrinking the world, not scaling the crab).
const BRAIN_LABEL_LIFT: f32 = 0.75;

/// One floating brain-label text node, tagged with the crab index it follows.
#[derive(Component)]
struct BrainLabelNode(usize);

/// Reconcile the label UI nodes with [`CrabBrainLabels`]: one `Text` node per entry, text
/// kept current, extras despawned. Spawned hidden — [`position_brain_labels`] reveals a node
/// only once it has projected a real on-screen position for it (no one-frame corner flash).
fn sync_brain_label_nodes(
    labels: Res<CrabBrainLabels>,
    mut nodes: Query<(Entity, &BrainLabelNode, &mut Text)>,
    mut commands: Commands,
) {
    if !labels.is_changed() {
        return;
    }
    let mut have = vec![false; labels.0.len()];
    for (entity, node, mut text) in &mut nodes {
        match labels.0.get(node.0) {
            Some(want) => {
                have[node.0] = true;
                if text.as_str() != want {
                    **text = want.clone();
                }
            }
            None => commands.entity(entity).despawn(),
        }
    }
    for (i, label) in labels.0.iter().enumerate() {
        if have[i] {
            continue;
        }
        commands.spawn((
            Text::new(label.clone()),
            TextFont {
                font_size: 16.0,
                ..default()
            },
            TextColor(HUD_TEXT_COLOR),
            Node {
                position_type: PositionType::Absolute,
                ..default()
            },
            Visibility::Hidden,
            BrainLabelNode(i),
        ));
    }
}

/// Project each label to the viewport point above its crab's RENDERED carapace. The world
/// anchor is `repose · carapace_translation + lift` — the same placement the cage reuses, so
/// a label can't drift from the crab it names. Projects through the UI's own camera (the
/// `IsDefaultUiCamera` — the demo's offscreen screenshot/video target — else the one active
/// 3D camera), and hides a label whose crab is missing, behind the camera, or off-screen.
fn position_brain_labels(
    repose: Option<Res<CrabSkinRepose>>,
    sampled: Option<Res<CrabRenderPose>>,
    ui_scale: Res<UiScale>,
    carapaces: Query<(Entity, &GlobalTransform, &CrabEnvId), With<CrabCarapace>>,
    cameras: Query<(&Camera, &GlobalTransform, Has<IsDefaultUiCamera>), With<Camera3d>>,
    mut nodes: Query<(&BrainLabelNode, &mut Node, &mut Visibility, &ComputedNode)>,
) {
    let camera = cameras
        .iter()
        .filter(|(cam, ..)| cam.is_active)
        .max_by_key(|(.., is_ui)| *is_ui)
        .map(|(cam, gt, _)| (cam, gt));
    for (node, mut ui, mut vis, computed) in &mut nodes {
        // Layout hasn't measured this node yet (it spawned this frame — `ComputedNode`
        // is still the size-zero default): keep it hidden rather than reveal it
        // un-centered at a stale spot. Next frame's layout has the size. An EMPTY label
        // (the wire's pre-publish "" filler) measures zero forever and so never shows.
        if computed.size() == Vec2::ZERO {
            *vis = Visibility::Hidden;
            continue;
        }
        let anchor =
            carapaces
                .iter()
                .find(|(.., env)| env.0 == node.0)
                .map(|(entity, carapace, env)| {
                    let placement = repose
                        .as_deref()
                        .and_then(|r| r.0.get(&env.0))
                        .map(|s| s.matrix())
                        .unwrap_or(Mat4::IDENTITY);
                    // The RENDERED carapace is the sampled pose where a stream feeds one
                    // (rl#274) — anchoring to the raw physics pose would let the label lead
                    // the skin by the sampling window's latency.
                    let carapace = sampled.as_deref().map_or_else(
                        || carapace.compute_transform(),
                        |s| s.rendered(entity, carapace.compute_transform()),
                    );
                    placement.transform_point3(carapace.translation) + Vec3::Y * BRAIN_LABEL_LIFT
                });
        let projected = camera
            .zip(anchor)
            .and_then(|((cam, cam_gt), anchor)| cam.world_to_viewport(cam_gt, anchor).ok());
        match projected {
            Some(vp) => {
                // `world_to_viewport` is in viewport-logical pixels but `Val::Px` is in
                // UI-logical pixels — they differ by `UiScale` (kept on the workspace
                // rule by `crate::app_boot::ui_scale_for`), so divide it out or every
                // label drifts off its crab on any non-reference window size.
                let vp = vp / ui_scale.0;
                // X-centered on the anchor, text bottom sitting AT the anchor.
                let size = computed.size() * computed.inverse_scale_factor();
                ui.left = Val::Px(vp.x - size.x * 0.5);
                ui.top = Val::Px(vp.y - size.y);
                *vis = Visibility::Visible;
            }
            None => *vis = Visibility::Hidden,
        }
    }
}

/// The crab-cage pass's exact view of a body part — the ONE definition of what the collider
/// wireframe draws. Tests pin coverage against these same aliases, so the pin can't drift
/// from the system (rl#225).
pub type CrabCagePartData<'a> = (Entity, &'a GlobalTransform, &'a Collider, &'a CrabEnvId);
pub type CrabCagePartFilter = With<CrabBodyPart>;

fn draw_crab_collider_wireframe(
    mode: Res<RenderMode>,
    repose: Option<Res<CrabSkinRepose>>,
    sampled: Option<Res<CrabRenderPose>>,
    parts: Query<CrabCagePartData, CrabCagePartFilter>,
    mut gizmos: Gizmos,
) {
    if !mode.shows_colliders() {
        return;
    }
    for (entity, gt, collider, env) in &parts {
        let placement = repose
            .as_deref()
            .and_then(|r| r.0.get(&env.0))
            .map(|s| s.matrix())
            .unwrap_or(Mat4::IDENTITY);
        // The cage cages the crab you SEE — it already rides the render-side repose, so
        // it draws at the sampled render pose too where a stream feeds one (rl#274).
        let part = sampled.as_deref().map_or_else(
            || gt.compute_transform(),
            |s| s.rendered(entity, gt.compute_transform()),
        );
        let world = placement * part.to_matrix();
        draw_collider_wireframe(
            &mut gizmos,
            collider.as_typed_shape(),
            world,
            COLLIDER_WIREFRAME_COLOR,
        );
    }
}

pub fn draw_collider_wireframe(
    gizmos: &mut Gizmos,
    view: ColliderView<'_>,
    world: Mat4,
    color: Color,
) {
    match view {
        ColliderView::Cuboid(c) => draw_cuboid(gizmos, world, c.half_extents(), color),
        ColliderView::Capsule(c) => {
            let seg = c.segment();
            draw_capsule(gizmos, world, seg.a(), seg.b(), c.radius(), color);
        }
        ColliderView::Compound(c) => {
            for (pos, rot, sub) in c.shapes() {
                let sub_world = world * Mat4::from_rotation_translation(rot, pos);
                draw_collider_wireframe(gizmos, sub, sub_world, color);
            }
        }
        // A shape this drawer can't trace would vanish from the collider view and read as
        // "colliders missing" (rl#225) — so say so instead of silently dropping it. ERROR,
        // not warn: only ERROR-level lines surface through the fleet telemetry, and an
        // untraceable shape is a code defect (we built a collider our own drawer can't render).
        other => {
            error_once!(
                "collider wireframe: no tracer for {other:?} — this collider is INVISIBLE in \
                 the collider render modes"
            );
        }
    }
}

fn draw_cuboid(gizmos: &mut Gizmos, world: Mat4, half: Vec3, color: Color) {
    let corner = |sx: f32, sy: f32, sz: f32| {
        world.transform_point3(Vec3::new(sx * half.x, sy * half.y, sz * half.z))
    };
    let c = [
        corner(-1.0, -1.0, -1.0),
        corner(1.0, -1.0, -1.0),
        corner(1.0, -1.0, 1.0),
        corner(-1.0, -1.0, 1.0),
        corner(-1.0, 1.0, -1.0),
        corner(1.0, 1.0, -1.0),
        corner(1.0, 1.0, 1.0),
        corner(-1.0, 1.0, 1.0),
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for (a, b) in edges {
        gizmos.line(c[a], c[b], color);
    }
}

fn draw_capsule(gizmos: &mut Gizmos, world: Mat4, a: Vec3, b: Vec3, radius: f32, color: Color) {
    let pa = world.transform_point3(a);
    let pb = world.transform_point3(b);
    let seg = pb - pa;
    let len = seg.length();
    let rot = if len > 1e-6 {
        Quat::from_rotation_arc(Vec3::Y, seg / len)
    } else {
        Quat::IDENTITY
    };
    gizmos.primitive_3d(
        &Capsule3d::new(radius, len),
        Isometry3d::new((pa + pb) * 0.5, rot),
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ValueEnum;

    /// The CLI spellings are a contract with the deploy scripts and the operator's muscle
    /// memory, and they are no longer written down anywhere else (the hand-rolled `parse` that
    /// used to hold them is gone, rl#275). Pin them.
    #[test]
    fn value_enum_spellings_are_what_the_cli_promises() {
        let spelling = |m: RenderMode| m.to_possible_value().unwrap().get_name().to_string();
        assert_eq!(spelling(RenderMode::Mesh), "mesh");
        assert_eq!(spelling(RenderMode::MeshColliders), "mesh+colliders");
        assert_eq!(spelling(RenderMode::Colliders), "colliders");
    }

    /// Cycling the render mode must visit every mode and return home — the in-app E-cycle and
    /// the `--render-mode` values are the same three states.
    #[test]
    fn next_cycles_through_every_mode() {
        let mut seen = vec![RenderMode::Mesh];
        let mut m = RenderMode::Mesh;
        for _ in 0..2 {
            m = m.next();
            seen.push(m);
        }
        assert_eq!(
            seen,
            vec![
                RenderMode::Mesh,
                RenderMode::MeshColliders,
                RenderMode::Colliders
            ]
        );
        assert_eq!(m.next(), RenderMode::Mesh, "the cycle must close");
    }
}
