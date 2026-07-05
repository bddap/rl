//! The crab render-mode cycle + the ONE collider-wireframe implementation, shared by GCR
//! (`net::render`) and the rl-demo. Both render the crab at its TRUE physics size now
//! (render==physics — the giant feel comes from shrinking the human world, not inflating the
//! crab), so a collider wireframe overlays the mesh NATIVELY and one cage serves both binaries.
//!
//! [`RenderMode`] cycles `mesh → mesh+colliders → colliders`. A player-facing view toggle
//! (not a hidden debug flag): the binding + label come from each binary's controls-display
//! source ([`crate::controls`]), so the on-screen HUD can't drift from what the button does.
//! The MISSING-GLB fallback boots straight into [`RenderMode::Colliders`]: with no Sally mesh
//! the honest physics view IS the crab the player sees (never a placeholder box).
//!
//! The cage is drawn with gizmos (not Rapier's `RapierDebugRenderPlugin`) because GCR's crab
//! body lives at the ~1 m arena origin while the rendered crab is translated to its game spot:
//! [`crate::bot::skin::CrabSkinRepose`] carries that rigid shift, so reusing it (identity in
//! the rl-demo, where body == rendered crab) puts the cage exactly on the mesh in BOTH. No
//! scale anywhere — render==physics, so the cage is the colliders at true size.

use bevy::prelude::*;
use bevy::ui::{IsDefaultUiCamera, UiScale};
use bevy_rapier3d::geometry::ColliderView;
use bevy_rapier3d::prelude::Collider;

use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId};
use crate::bot::skin::CrabSkinRepose;

/// The ONE colour every physics collider wireframe draws in — the crab cage here, and GCR's piloted
/// vehicle (the `net` render-mode glue reuses [`draw_collider_wireframe`]). One source so the
/// honest-physics view reads uniform across both bodies.
pub const COLLIDER_WIREFRAME_COLOR: Color = Color::srgb(0.2, 1.0, 0.4);

/// The ONE HUD-text green, shared by the corner render-mode label and the floating brain
/// labels — one source, so the overlay can't drift into two near-greens.
const HUD_TEXT_COLOR: Color = Color::srgb(0.4, 1.0, 0.55);

/// What the crab render shows. A 3-state cycle, driven by one controller button (+ key) wired
/// through each binary's controls source. `Mesh` is the default player-facing view; the two
/// collider views overlay (or replace) it with the honest physics cage.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum RenderMode {
    /// The skinned Sally mesh only (or, with no glb, the physics-bones silhouette) — the
    /// player-facing default.
    #[default]
    Mesh,
    /// The mesh WITH the collider wireframe drawn on top — "does my hitbox match what I see".
    /// Since render==physics, the cage sits exactly on the mesh.
    MeshColliders,
    /// The collider wireframe ALONE (mesh hidden) — the honest physics view, and the
    /// missing-glb fallback.
    Colliders,
}

impl RenderMode {
    /// Cycle `Mesh → MeshColliders → Colliders → Mesh`, so one button walks every view.
    pub fn next(self) -> Self {
        match self {
            RenderMode::Mesh => RenderMode::MeshColliders,
            RenderMode::MeshColliders => RenderMode::Colliders,
            RenderMode::Colliders => RenderMode::Mesh,
        }
    }

    /// Whether the crab MESH (skin / silhouette) is shown in this mode — `false` only for the
    /// colliders-only view, which hides the mesh so the cage reads clean.
    pub fn shows_mesh(self) -> bool {
        !matches!(self, RenderMode::Colliders)
    }

    /// Whether the collider wireframe is drawn in this mode.
    pub fn shows_colliders(self) -> bool {
        matches!(self, RenderMode::MeshColliders | RenderMode::Colliders)
    }

    /// The short HUD name of this mode — what the on-screen label reads. Stays in sync with the
    /// cycle because both the label and the input read this one enum.
    pub fn label(self) -> &'static str {
        match self {
            RenderMode::Mesh => "mesh",
            RenderMode::MeshColliders => "mesh + colliders",
            RenderMode::Colliders => "colliders",
        }
    }

    /// Parse a render-mode token (a CLI flag / env value). `None` for an unknown token (the
    /// caller reports it); the three modes map by name, with `mesh-colliders` as an alias for
    /// the `+` form that's awkward on a shell.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mesh" => Some(RenderMode::Mesh),
            "mesh+colliders" | "mesh-colliders" => Some(RenderMode::MeshColliders),
            "colliders" => Some(RenderMode::Colliders),
            _ => None,
        }
    }

    /// The mode the environment asks for, `None` when it doesn't: `RL_RENDER_MODE=<mode>`,
    /// else the legacy idiom where `RL_DEBUG_COLLIDERS` (any value) means the colliders-only
    /// view. THE one reader of these two env vars — the precedence over flags and the
    /// unusable-mesh fallback lives in [`crate::mesh_fallback::initial_render_mode`], which
    /// treats `None` as "env didn't decide". An unparsable `RL_RENDER_MODE` warns and returns
    /// `None` (it decides nothing), so it can't suppress the honest broken-mesh fallback.
    pub fn env_mode() -> Option<Self> {
        if let Ok(v) = std::env::var("RL_RENDER_MODE") {
            let parsed = RenderMode::parse(&v);
            if parsed.is_none() {
                warn!("RL_RENDER_MODE={v:?} not one of mesh|mesh+colliders|colliders; ignoring it");
            }
            return parsed;
        }
        std::env::var_os("RL_DEBUG_COLLIDERS")
            .is_some()
            .then_some(RenderMode::Colliders)
    }
}

/// The corner text node naming the active render mode — the shared HUD label both binaries get.
#[derive(Component)]
struct RenderModeLabel;

/// Wire the shared render-mode cage into a render `App`, booting in `initial` (the missing-glb
/// fallback passes [`RenderMode::Colliders`]). Inserts the [`RenderMode`] resource, the gizmo
/// cage, and the corner label naming the active mode (so the HUD can't drift from the mode — one
/// source for both GCR and the rl-demo). The mesh-visibility half lives where each mesh does —
/// the skin in [`crate::bot::skin`] (shared, reads this resource), GCR's silhouette in
/// `net::render`. Call once; registration order vs the bot/physics plugins doesn't matter (Bevy
/// orders systems by set-label, and `init_resource` is order-independent).
///
/// `cage_gate` gates the cage DRAW: gizmos render through ANY camera — including a menu's
/// Camera2d — and GCR's crab body deliberately survives round teardown, so without a gate the
/// cage would draw over the post-disconnect menu (rl#211). This module can't know the caller's
/// phase type, so the caller supplies the condition; the rl-demo (no phases) passes `|| true`.
pub fn register<M>(app: &mut App, initial: RenderMode, cage_gate: impl SystemCondition<M>) {
    app.insert_resource(initial);
    // The crab's render placement (GCR's rigid shift to the game spot) is published by the
    // external-crab bridge into this resource EVERY armed frame — but `skin::register` only inits
    // it WHEN a Sally model loads. The colliders-only fallback has no skin, so init it here too
    // (idempotent) — else the cage would fall back to identity and draw at the ~1 m arena origin
    // instead of overlaying the giant crab's game spot. `None` (the default) until the bridge
    // publishes; the rl-demo, with no bridge, leaves it `None` ⇒ identity, which is correct there
    // (the demo's crab body IS the rendered crab).
    app.init_resource::<CrabSkinRepose>();
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
    // Draw AFTER transform propagation so each part's `GlobalTransform` holds this frame's
    // physics pose; otherwise the cage lags a frame.
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
        // Bottom-right: clear of the status HUD (top) and the hold-to-reveal controls hint
        // (bottom-left), so the mode line never overlaps them.
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
    ui_scale: Res<UiScale>,
    carapaces: Query<(&GlobalTransform, &CrabEnvId), With<CrabCarapace>>,
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
        let anchor = carapaces.iter().find(|(_, env)| env.0 == node.0).map(
            |(carapace, env)| {
                let placement = repose
                    .as_deref()
                    .and_then(|r| r.0.get(&env.0))
                    .map(|s| s.matrix())
                    .unwrap_or(Mat4::IDENTITY);
                placement.transform_point3(carapace.translation()) + Vec3::Y * BRAIN_LABEL_LIFT
            },
        );
        let projected = camera
            .zip(anchor)
            .and_then(|((cam, cam_gt), anchor)| cam.world_to_viewport(cam_gt, anchor).ok());
        match projected {
            Some(vp) => {
                // `world_to_viewport` is in viewport-logical pixels but `Val::Px` is in
                // UI-logical pixels — they differ by `UiScale` (the demo scales its HUD
                // by window height), so divide it out or every label drifts off its crab
                // on any non-reference window size.
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

/// Draw every crab's live colliders as a gizmo wireframe at the crab's RENDERED pose. Active in
/// any mode that [`RenderMode::shows_colliders`]. The transform is `repose · part_global`: each
/// part's `GlobalTransform` is its raw physics pose, and [`crate::bot::skin::SkinRepose::matrix`]
/// is the rigid shift the skin applies to relocate that env's arena rig to its game spot
/// (identity for an env with no repose entry — the rl-demo, where the body IS the rendered
/// crab). Reusing it — not a re-derived factor — is why a cage can't drift from its rendered
/// crab. There is no scale: render==physics. Per-env (rl#200: a GCR round renders one crab per
/// brain binding).
fn draw_crab_collider_wireframe(
    mode: Res<RenderMode>,
    repose: Option<Res<CrabSkinRepose>>,
    parts: Query<(&GlobalTransform, &Collider, &CrabEnvId), With<CrabBodyPart>>,
    mut gizmos: Gizmos,
) {
    if !mode.shows_colliders() {
        return;
    }
    for (gt, collider, env) in &parts {
        // The crab's render placement: the skin's rigid shift in GCR, identity in the rl-demo
        // (no bridge ⇒ no repose entry ⇒ the body is already the rendered crab).
        // render==physics, so there is no scale to apply to the cage.
        let placement = repose
            .as_deref()
            .and_then(|r| r.0.get(&env.0))
            .map(|s| s.matrix())
            .unwrap_or(Mat4::IDENTITY);
        let world = placement * gt.to_matrix();
        draw_collider_wireframe(
            &mut gizmos,
            collider.as_typed_shape(),
            world,
            COLLIDER_WIREFRAME_COLOR,
        );
    }
}

/// Draw one collider view as gizmo lines under world transform `world`. Handles the shapes the
/// crab body and the piloted vehicle actually use — the carapace compound-of-cuboid, the per-link
/// capsules, and the vehicle cuboid; other shapes are skipped (neither body has them). render==physics,
/// so `world` carries no scale and the shapes draw at true collider size. `pub` so GCR's render-mode
/// glue draws the piloted craft's collider through this ONE drawer (no second wireframe impl).
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
        // The crab uses only the above; anything else (a future shape) is skipped rather than
        // mis-drawn.
        _ => {}
    }
}

/// Wireframe box: the 12 edges of the cuboid `±half`, each corner pushed through `world`.
/// bevy 0.18's gizmos have no cuboid primitive, so draw the edges directly.
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

/// Wireframe capsule between the link-local segment endpoints `a`,`b` pushed through `world`.
/// The `Capsule3d` gizmo is Y-aligned, so rotate +Y onto the segment direction. render==physics
/// ⇒ `radius` is the true collider radius (no scale).
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
