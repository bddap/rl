//! Interactive collider-placement editor (`--edit-colliders OUT.ron`).
//!
//! Renders the glTF crab in its **bind pose** and overlays the fitted collider
//! table as wireframes, one bone at a time, so each collider can be hand-placed
//! against the mesh where the auto-fit's point-cloud orientation falls short. The
//! owner steps bone-by-bone; the active bone is highlighted (its vertex cloud +
//! bone segment drawn) so it's obvious which part is being placed and where.
//!
//! Authoring is in **raw glTF coordinates at scale 1.0** — the exact frame the fit
//! runs in (`meshfit`). The physics-skin's `CRAB_MODEL_SCALE` (1.2, which seats the
//! visual on the trained hand-coded body) is deliberately NOT applied here: the
//! editor is not the physics crab, it is the mesh the colliders are fit to, so mesh
//! and colliders share one frame and the overlay aligns exactly with no scale
//! reconciliation.
//!
//! Frames: a [`Placement`] is stored in the link-local frame (relative to the
//! proximal bone's bind basis `B` and pivot `p`), because that is what `body.rs`
//! consumes at spawn. The editor works in bind-pose **world** instead — collider
//! world pose `= (p + B·center, B·rotation)` — which is where the mesh actually is,
//! and converts back to link-local only when saving. So the file format never
//! changes (one [`FittedBody`] RON, no second representation); the editor is purely
//! a world-space front-end over it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use crate::bot::body::part_densities;
use crate::bot::meshfit::{
    BoneSpan, FittedBody, FittedPart, LoadedModel, PartId, Placement, Primitive, fit_part,
    model_path,
};

/// One editable part: its identity + bone span + vertex cloud (all in bind-pose
/// glTF world), and the collider being authored, held in WORLD space (position +
/// XYZ-euler degrees) so the egui fields and the 3D view never disagree and a quat
/// round-trip can't introduce gimbal jitter while dragging. The link-local
/// [`Placement`] is derived from these only at save (see [`EditPart::to_fitted`]).
struct EditPart {
    part: PartId,
    label: String,
    /// Proximal bone pivot (bind world) — the joint origin the placement is relative to.
    pivot: Vec3,
    /// Proximal bone bind-world basis `B` — the link-local frame mapper.
    basis: Quat,
    /// Distal bone origin (bind world) — draws the bone segment for the guide.
    distal: Vec3,
    /// The bone's skinned vertex cloud (bind world) — drawn to show the part's extent.
    cloud: Vec<Vec3>,
    density: f32,
    primitive: Primitive,
    world_pos: Vec3,
    world_euler_deg: Vec3,
}

impl EditPart {
    /// Seed the world-space working state from a link-local [`FittedPart`].
    fn seed(span: &BoneSpan, cloud: Vec<Vec3>, density: f32, fitted: FittedPart) -> Self {
        let world_pos = span.proximal + span.proximal_basis * fitted.placement.center;
        let world_rot = span.proximal_basis * fitted.placement.rotation;
        let (x, y, z) = world_rot.to_euler(EulerRot::XYZ);
        EditPart {
            part: fitted.part,
            label: format!("{:?}", fitted.part),
            pivot: span.proximal,
            basis: span.proximal_basis,
            distal: span.distal,
            cloud,
            density,
            primitive: fitted.primitive,
            world_pos,
            world_euler_deg: Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees()),
        }
    }

    /// World rotation from the edited euler angles.
    fn world_rot(&self) -> Quat {
        let e = self.world_euler_deg;
        Quat::from_euler(
            EulerRot::XYZ,
            e.x.to_radians(),
            e.y.to_radians(),
            e.z.to_radians(),
        )
    }

    /// The collider's world transform (for drawing).
    fn world_transform(&self) -> Transform {
        Transform::from_translation(self.world_pos).with_rotation(self.world_rot())
    }

    /// Convert the world-space working state back into a link-local [`FittedPart`]
    /// — the inverse of [`Self::seed`], so a seed→save round-trip is the identity.
    fn to_fitted(&self) -> FittedPart {
        let inv = self.basis.inverse();
        FittedPart {
            part: self.part,
            primitive: self.primitive,
            placement: Placement {
                center: inv * (self.world_pos - self.pivot),
                rotation: inv * self.world_rot(),
            },
            density: self.density,
        }
    }
}

/// The editor's whole state: every part, the one being edited, and where to save.
#[derive(Resource)]
struct Editor {
    parts: Vec<EditPart>,
    cur: usize,
    out_path: PathBuf,
    status: String,
}

impl Editor {
    fn active(&self) -> &EditPart {
        &self.parts[self.cur]
    }
    fn active_mut(&mut self) -> &mut EditPart {
        &mut self.parts[self.cur]
    }
}

/// Orbit camera state, driven by the mouse (drag to rotate, wheel to zoom).
#[derive(Resource)]
struct OrbitCam {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        // Front-3/4 view framing the bind-pose crab (feet at y=0, facing +z).
        OrbitCam {
            target: Vec3::new(0.0, 0.35, 0.0),
            yaw: 0.6,
            pitch: 0.35,
            radius: 3.2,
        }
    }
}

/// Build and run the editor app. Loads the model + seeds the collider table, then
/// opens a window; returns when the window is closed.
pub fn run(out_path: &Path) {
    let Some(mp) = model_path() else {
        eprintln!(
            "edit-colliders: no model — set CRAB_MODEL_PATH or place sally.glb at the dev path"
        );
        std::process::exit(1);
    };
    let model = match LoadedModel::load(&mp) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("edit-colliders: load {mp:?}: {e}");
            std::process::exit(1);
        }
    };
    let parts = seed_parts(&model, out_path);
    if parts.is_empty() {
        eprintln!("edit-colliders: no parts to edit (model has no recognised bones?)");
        std::process::exit(1);
    }

    // The rendered scene asset is resolved by the asset server relative to
    // BEVY_ASSET_ROOT; skin.rs uses the same CRAB_MODEL_PATH env for it.
    let scene_asset = std::env::var("CRAB_MODEL_PATH").unwrap_or_else(|_| "sally.glb".to_string());

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Crab Collider Editor".into(),
            ..default()
        }),
        ..default()
    }));
    if !app.is_plugin_added::<EguiPlugin>() {
        app.add_plugins(EguiPlugin::default());
    }
    // RL_EDITOR_BONE=<index> opens on that bone (else the first) — a dev aid for
    // grabbing a specific bone in a self-screenshot.
    let cur = std::env::var("RL_EDITOR_BONE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
        .min(parts.len() - 1);
    app.insert_resource(Editor {
        parts,
        cur,
        out_path: out_path.to_path_buf(),
        status: "seeded from auto-fit; step bones (N/P), drag in the panel, S to save".into(),
    })
    .insert_resource(OrbitCam::default())
    .insert_resource(SceneAsset(scene_asset))
    .add_systems(Startup, setup)
    .add_systems(Update, (orbit_camera, draw_overlay, handle_keys))
    .add_systems(EguiPrimaryContextPass, editor_ui);
    // RL_EDITOR_SHOT=path: snapshot the window after it settles, then exit — a
    // headless self-check (runs under Xvfb, no display) that the scene + overlay
    // actually render, and a handy way to grab the editor state for a bug report.
    if let Ok(shot) = std::env::var("RL_EDITOR_SHOT") {
        app.insert_resource(ShotPath(shot.into()));
        app.add_systems(Update, editor_screenshot);
    }
    app.run();
}

/// Output path for the `RL_EDITOR_SHOT` self-screenshot.
#[derive(Resource)]
struct ShotPath(PathBuf);

/// Capture the primary window once it has rendered a few frames, then exit.
fn editor_screenshot(
    mut commands: Commands,
    path: Res<ShotPath>,
    mut frame: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    *frame += 1;
    // ~120 render frames lets the GPU pipeline warm and the glTF finish loading.
    if *frame == 120 {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.0.clone()));
    }
    if *frame == 140 {
        exit.write(AppExit::Success);
    }
}

/// The glTF scene asset path (relative to the asset root), carried into `setup`.
#[derive(Resource)]
struct SceneAsset(String);

/// Seed every part from the live auto-fit, then overlay any entries already saved
/// in `out_path` (so re-opening resumes the owner's hand-placed work). Always
/// yields the full ordered part list, so stepping covers the whole body.
fn seed_parts(model: &LoadedModel, out_path: &Path) -> Vec<EditPart> {
    let by_part = model.vertices_by_part();
    // Resume from a saved table if present. A file that exists but won't parse is
    // reported loudly, not silently dropped — otherwise the editor would open on
    // the auto-fit as if it were the saved work and the user wouldn't know their
    // file was rejected. The bad file is left untouched until an explicit save.
    let saved: HashMap<PartId, FittedPart> = match std::fs::read_to_string(out_path) {
        Ok(ron) => match FittedBody::from_ron(&ron) {
            Ok(b) => b.parts.iter().map(|p| (p.part, *p)).collect(),
            Err(e) => {
                eprintln!(
                    "edit-colliders: {out_path:?} exists but did not load ({e}); seeding from \
                     the auto-fit. Your file is NOT overwritten until you save."
                );
                HashMap::new()
            }
        },
        Err(_) => HashMap::new(),
    };

    let mut parts = Vec::new();
    for (part, density) in part_densities() {
        let (Some((cloud, _)), Some(span)) = (by_part.get(&part), model.bone_span(part)) else {
            continue;
        };
        let fitted = saved.get(&part).copied().unwrap_or_else(|| {
            // No saved entry: seed from the canonical auto-fit (same `fit_part` the
            // bake uses, so the editor's seed can't drift from the baked table).
            let (primitive, placement) = fit_part(part, cloud, &span);
            FittedPart {
                part,
                primitive,
                placement,
                density,
            }
        });
        parts.push(EditPart::seed(&span, cloud.clone(), density, fitted));
    }
    parts
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, scene: Res<SceneAsset>) {
    // AmbientLight is a per-view component in bevy 0.18 (not a resource); on the
    // camera it lifts the mesh's shadowed underside so colliders read there too.
    commands.spawn((
        Camera3d::default(),
        Transform::default(),
        AmbientLight {
            brightness: 400.0,
            ..default()
        },
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Raw glTF scene at scale 1.0 (NOT the skin's 1.2) — no animation drives the
    // bones, so it renders in bind pose, the frame the colliders are authored in.
    commands.spawn((
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(scene.0.clone()))),
        Transform::IDENTITY,
    ));
}

/// Mouse orbit: left-drag rotates, wheel zooms. Writes the camera transform.
fn orbit_camera(
    mut cam: ResMut<OrbitCam>,
    mut cam_q: Query<&mut Transform, With<Camera3d>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut contexts: EguiContexts,
) {
    // Don't orbit while the cursor is over the egui panel (let the panel take it).
    let over_ui = contexts
        .ctx_mut()
        .map(|c| c.wants_pointer_input())
        .unwrap_or(false);

    let mut delta = Vec2::ZERO;
    if buttons.pressed(MouseButton::Left) && !over_ui {
        for m in motion.read() {
            delta += m.delta;
        }
    } else {
        motion.clear();
    }
    cam.yaw -= delta.x * 0.005;
    cam.pitch = (cam.pitch - delta.y * 0.005).clamp(-1.5, 1.5);

    let mut scroll = 0.0;
    if !over_ui {
        for w in wheel.read() {
            scroll += w.y;
        }
    } else {
        wheel.clear();
    }
    cam.radius = (cam.radius * (1.0 - scroll * 0.1)).clamp(0.4, 20.0);

    let dir = Vec3::new(
        cam.yaw.cos() * cam.pitch.cos(),
        cam.pitch.sin(),
        cam.yaw.sin() * cam.pitch.cos(),
    );
    if let Ok(mut t) = cam_q.single_mut() {
        *t = Transform::from_translation(cam.target + dir * cam.radius)
            .looking_at(cam.target, Vec3::Y);
    }
}

/// Draw every collider wireframe + the active bone's guide (cloud + segment).
fn draw_overlay(editor: Res<Editor>, mut gizmos: Gizmos) {
    for (i, p) in editor.parts.iter().enumerate() {
        let active = i == editor.cur;
        let color = if active {
            Color::srgb(1.0, 0.95, 0.2) // bright yellow
        } else {
            Color::srgb(0.25, 0.65, 0.75) // dim cyan
        };
        draw_primitive(&mut gizmos, &p.primitive, &p.world_transform(), color);
    }

    // Active-bone guide: bone segment + pivot + a sample of the vertex cloud, so
    // it's unambiguous which part is being placed and where it sits on the mesh.
    let a = editor.active();
    gizmos.line(a.pivot, a.distal, Color::WHITE);
    // Joint (parent) end green, far (child/tip) end red — so the bone's direction
    // is unambiguous and a collider can't be oriented/placed the wrong way round.
    gizmos.sphere(
        Isometry3d::from_translation(a.pivot),
        0.025,
        Color::srgb(0.2, 1.0, 0.3),
    );
    gizmos.sphere(
        Isometry3d::from_translation(a.distal),
        0.018,
        Color::srgb(1.0, 0.2, 0.2),
    );
    // Magenta, not a warm tone: the Sally model is orange/red, so an orange cloud
    // would vanish into the mesh — magenta is the one bright hue that contrasts
    // both the crab and the cyan/yellow collider wireframes.
    let cloud_color = Color::srgb(1.0, 0.1, 0.9);
    let step = (a.cloud.len() / 80).max(1);
    for v in a.cloud.iter().step_by(step) {
        gizmos.sphere(Isometry3d::from_translation(*v), 0.005, cloud_color);
    }
}

/// Draw one collider primitive as a wireframe at `t`. Uses only line/sphere/cuboid
/// gizmos; a capsule is two end-spheres joined by longitudinal lines (a faithful
/// enough silhouette without depending on a capsule-gizmo primitive).
fn draw_primitive(gizmos: &mut Gizmos, prim: &Primitive, t: &Transform, color: Color) {
    match *prim {
        Primitive::Cuboid { half_extents } => draw_box(gizmos, t, half_extents, color),
        Primitive::Ball { radius } => {
            gizmos.sphere(Isometry3d::new(t.translation, t.rotation), radius, color);
        }
        Primitive::Capsule {
            half_height,
            radius,
        } => {
            let axis = t.rotation * Vec3::Y;
            let a = t.translation + axis * half_height;
            let b = t.translation - axis * half_height;
            gizmos.sphere(Isometry3d::new(a, t.rotation), radius, color);
            gizmos.sphere(Isometry3d::new(b, t.rotation), radius, color);
            // Four longitudinal lines around the shaft.
            let u = (t.rotation * Vec3::X) * radius;
            let w = (t.rotation * Vec3::Z) * radius;
            for off in [u, -u, w, -w] {
                gizmos.line(a + off, b + off, color);
            }
        }
    }
}

/// Draw an oriented box as its 12 edges (no `Gizmos::cuboid` in this bevy). The 8
/// corners index as bits `x<<2 | y<<1 | z`; two corners share an edge iff they
/// differ in exactly one axis, i.e. their indices differ by a single bit (1/2/4).
fn draw_box(gizmos: &mut Gizmos, t: &Transform, half: Vec3, color: Color) {
    let mut v = [Vec3::ZERO; 8];
    for (i, slot) in v.iter_mut().enumerate() {
        let s = |bit: usize| if i & bit != 0 { 1.0 } else { -1.0 };
        *slot = t.translation + t.rotation * Vec3::new(s(4) * half.x, s(2) * half.y, s(1) * half.z);
    }
    for i in 0..8 {
        for j in (i + 1)..8 {
            if matches!(i ^ j, 1 | 2 | 4) {
                gizmos.line(v[i], v[j], color);
            }
        }
    }
}

/// Step to a different bone, clamping into range and refreshing the status line.
fn select(editor: &mut Editor, idx: usize) {
    editor.cur = idx.min(editor.parts.len() - 1);
    let a = editor.active();
    editor.status = format!(
        "editing {} ({}/{})",
        a.label,
        editor.cur + 1,
        editor.parts.len()
    );
}

/// Re-seed the active bone from the auto-fit, discarding hand edits to it (its
/// stored cloud + bone span are all the fit needs, so no model reload). The other
/// bones are untouched.
fn reset_active(editor: &mut Editor) {
    let cur = editor.cur;
    let (pivot, basis, distal, density, cloud, part) = {
        let a = &editor.parts[cur];
        (
            a.pivot,
            a.basis,
            a.distal,
            a.density,
            a.cloud.clone(),
            a.part,
        )
    };
    let span = BoneSpan {
        proximal: pivot,
        proximal_basis: basis,
        distal,
    };
    let (primitive, placement) = fit_part(part, &cloud, &span);
    let fitted = FittedPart {
        part,
        primitive,
        placement,
        density,
    };
    editor.parts[cur] = EditPart::seed(&span, cloud, density, fitted);
    editor.status = format!("reset {} to auto-fit", editor.parts[cur].label);
}

/// Keyboard: N/P step bones, S save, arrows nudge position, R reset active to fit.
fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut editor: ResMut<Editor>,
    mut contexts: EguiContexts,
) {
    // Ignore keys while egui has keyboard focus (typing in a field).
    if contexts
        .ctx_mut()
        .map(|c| c.wants_keyboard_input())
        .unwrap_or(false)
    {
        return;
    }
    let n = editor.parts.len();
    if keys.just_pressed(KeyCode::KeyN) {
        let next = (editor.cur + 1) % n;
        select(&mut editor, next);
    }
    if keys.just_pressed(KeyCode::KeyP) {
        let prev = (editor.cur + n - 1) % n;
        select(&mut editor, prev);
    }
    if keys.just_pressed(KeyCode::KeyS) {
        save(&mut editor);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        reset_active(&mut editor);
    }
    // Arrow keys nudge the active collider in world X/Z by 5 mm; PageUp/Down in Y.
    let mut d = Vec3::ZERO;
    if keys.just_pressed(KeyCode::ArrowLeft) {
        d.x -= 0.005;
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        d.x += 0.005;
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        d.z -= 0.005;
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        d.z += 0.005;
    }
    if keys.just_pressed(KeyCode::PageUp) {
        d.y += 0.005;
    }
    if keys.just_pressed(KeyCode::PageDown) {
        d.y -= 0.005;
    }
    if d != Vec3::ZERO {
        editor.active_mut().world_pos += d;
    }
}

/// The control panel: bone stepper, primitive + size, world transform, save.
fn editor_ui(mut contexts: EguiContexts, mut editor: ResMut<Editor>) -> Result {
    let n = editor.parts.len();
    let mut want_select: Option<usize> = None;
    let mut want_save = false;
    let mut want_reset = false;

    let ctx = contexts.ctx_mut()?;
    egui::SidePanel::left("editor")
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.heading("Collider editor");
            ui.label(&editor.status);
            ui.separator();

            let cur = editor.cur;
            ui.horizontal(|ui| {
                if ui.button("◀ prev (P)").clicked() {
                    want_select = Some((cur + n - 1) % n);
                }
                if ui.button("next (N) ▶").clicked() {
                    want_select = Some((cur + 1) % n);
                }
            });
            // Jump straight to any bone.
            egui::ComboBox::from_label("bone")
                .selected_text(editor.parts[cur].label.clone())
                .show_ui(ui, |ui| {
                    for (i, p) in editor.parts.iter().enumerate() {
                        if ui.selectable_label(i == cur, &p.label).clicked() {
                            want_select = Some(i);
                        }
                    }
                });
            ui.separator();

            let p = &mut editor.parts[cur];

            // Primitive kind + dimensions.
            ui.label("primitive");
            let mut kind = prim_kind(&p.primitive);
            let prev_kind = kind;
            ui.horizontal(|ui| {
                ui.selectable_value(&mut kind, 0u8, "box");
                ui.selectable_value(&mut kind, 1u8, "capsule");
                ui.selectable_value(&mut kind, 2u8, "ball");
            });
            if kind != prev_kind {
                p.primitive = reshape(&p.primitive, kind);
            }
            // Dimension floors are a positive epsilon, not 0: a zero extent/radius
            // is non-physical and `FittedBody::from_ron` would reject the whole
            // table, so the UI must not let one be authored (`save` re-checks too).
            match &mut p.primitive {
                Primitive::Cuboid { half_extents } => {
                    drag3(ui, "half-extents", half_extents, 0.001, 0.001..=2.0);
                }
                Primitive::Capsule {
                    half_height,
                    radius,
                } => {
                    drag1(ui, "half-height", half_height, 0.001, 0.001..=2.0);
                    drag1(ui, "radius", radius, 0.001, 0.001..=1.0);
                }
                Primitive::Ball { radius } => {
                    drag1(ui, "radius", radius, 0.001, 0.001..=1.0);
                }
            }
            ui.separator();

            ui.label("position (world)");
            drag3(ui, "pos", &mut p.world_pos, 0.001, -2.0..=2.0);
            ui.label("rotation (° world, XYZ)");
            drag3(ui, "rot", &mut p.world_euler_deg, 0.5, -180.0..=180.0);
            ui.separator();

            if ui.button("reset this bone to auto-fit (R)").clicked() {
                want_reset = true;
            }
            if ui.button("💾 SAVE (S)").clicked() {
                want_save = true;
            }
        });

    if let Some(i) = want_select {
        select(&mut editor, i);
    }
    if want_reset {
        reset_active(&mut editor);
    }
    if want_save {
        save(&mut editor);
    }
    Ok(())
}

/// Primitive → kind tag (0 box, 1 capsule, 2 ball) for the radio group.
fn prim_kind(p: &Primitive) -> u8 {
    match p {
        Primitive::Cuboid { .. } => 0,
        Primitive::Capsule { .. } => 1,
        Primitive::Ball { .. } => 2,
    }
}

/// Convert a primitive to a different kind, carrying its rough scale across so the
/// switch doesn't snap to a degenerate size. Box half-extents → capsule
/// (radius = mean of the two minor extents, half-height = longest less one cap) →
/// ball (radius = mean extent), and back, all via a common characteristic size.
fn reshape(p: &Primitive, kind: u8) -> Primitive {
    let e = match *p {
        Primitive::Cuboid { half_extents } => half_extents,
        Primitive::Capsule {
            half_height,
            radius,
        } => Vec3::new(radius, half_height, radius),
        Primitive::Ball { radius } => Vec3::splat(radius),
    };
    let mut sorted = e.to_array();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let long = sorted[0].max(1e-3);
    let minor = (0.5 * (sorted[1] + sorted[2])).max(1e-3);
    match kind {
        0 => Primitive::Cuboid { half_extents: e },
        // half_height is the cylinder half-length (caps excluded), so subtract one
        // cap radius to keep the capsule's overall extent ≈ the source's long axis.
        1 => Primitive::Capsule {
            half_height: (long - minor).max(1e-3),
            radius: minor,
        },
        _ => Primitive::Ball { radius: minor },
    }
}

fn drag1(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f32,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(v).speed(speed).range(range));
    });
}

fn drag3(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut Vec3,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::DragValue::new(&mut v.x)
                .speed(speed)
                .range(range.clone())
                .prefix("x "),
        );
        ui.add(
            egui::DragValue::new(&mut v.y)
                .speed(speed)
                .range(range.clone())
                .prefix("y "),
        );
        ui.add(
            egui::DragValue::new(&mut v.z)
                .speed(speed)
                .range(range)
                .prefix("z "),
        );
    });
}

/// Serialize the current placements to the output RON. The output IS the session's
/// accumulated work, so two guards protect it: (1) the table is re-parsed with the
/// loader's own validation before anything is written — a save that `--body fitted`
/// or a re-open couldn't load back is silently-lost work, so it's blocked, not
/// written; (2) the write is atomic (temp + rename) so a crash mid-write can't
/// truncate the existing file. Mirrors `training::session`'s checkpoint writes.
fn save(editor: &mut Editor) {
    let parts: Vec<FittedPart> = editor.parts.iter().map(EditPart::to_fitted).collect();
    let body = FittedBody {
        version: FittedBody::VERSION,
        parts,
    };
    let ron = match body.to_ron() {
        Ok(ron) => ron,
        Err(e) => {
            editor.status = format!("SAVE FAILED: {e}");
            return;
        }
    };
    if let Err(e) = FittedBody::from_ron(&ron) {
        editor.status = format!("SAVE BLOCKED (would not load back): {e}");
        return;
    }
    let tmp = editor.out_path.with_extension("tmp");
    if let Err(e) = std::fs::write(&tmp, &ron) {
        editor.status = format!("SAVE FAILED: write {tmp:?}: {e}");
        return;
    }
    match std::fs::rename(&tmp, &editor.out_path) {
        Ok(()) => {
            editor.status = format!("saved {} parts → {:?}", editor.parts.len(), editor.out_path)
        }
        Err(e) => editor.status = format!("SAVE FAILED: rename → {:?}: {e}", editor.out_path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seeding the editor from a placement and saving it back must reproduce the
    /// same placement exactly — the world↔link-local conversion is the one
    /// bug-prone seam.
    fn check_roundtrip(span: &BoneSpan, original: FittedPart) {
        let part = EditPart::seed(span, vec![Vec3::ZERO], original.density, original);
        let back = part.to_fitted();
        assert!(
            back.placement
                .center
                .abs_diff_eq(original.placement.center, 1e-5),
            "center {:?} != {:?}",
            back.placement.center,
            original.placement.center,
        );
        // Quats compare up to sign (q and −q are the same rotation).
        let dot = back
            .placement
            .rotation
            .dot(original.placement.rotation)
            .abs();
        assert!(dot > 1.0 - 1e-5, "rotation drifted (|dot| {dot})");
        assert_eq!(back.primitive, original.primitive);
        assert_eq!(back.density, original.density);
    }

    #[test]
    fn placement_world_roundtrip() {
        // A generic non-degenerate pose.
        check_roundtrip(
            &BoneSpan {
                proximal: Vec3::new(0.3, 1.1, -0.2),
                proximal_basis: Quat::from_euler(EulerRot::XYZ, 0.4, -0.9, 0.2),
                distal: Vec3::new(0.6, 1.0, -0.1),
            },
            FittedPart {
                part: PartId::Carapace,
                primitive: Primitive::Cuboid {
                    half_extents: Vec3::new(0.12, 0.05, 0.2),
                },
                placement: Placement {
                    center: Vec3::new(0.03, -0.01, 0.07),
                    rotation: Quat::from_euler(EulerRot::XYZ, 0.1, 0.5, -0.3),
                },
                density: 500.0,
            },
        );
        // Near gimbal lock: the editor stores XYZ-euler, whose middle (Y) angle here
        // sits at ~+90° where the decomposition degenerates. The round-trip must
        // still reproduce the rotation (from_euler∘to_euler is exact for the quat,
        // even though editing a single axis there would behave oddly).
        check_roundtrip(
            &BoneSpan {
                proximal: Vec3::ZERO,
                proximal_basis: Quat::IDENTITY,
                distal: Vec3::Z,
            },
            FittedPart {
                part: PartId::Carapace,
                primitive: Primitive::Capsule {
                    half_height: 0.1,
                    radius: 0.02,
                },
                placement: Placement {
                    center: Vec3::new(0.01, 0.02, -0.03),
                    rotation: Quat::from_euler(
                        EulerRot::XYZ,
                        0.2,
                        std::f32::consts::FRAC_PI_2 - 1e-4,
                        0.3,
                    ),
                },
                density: 300.0,
            },
        );
    }
}
