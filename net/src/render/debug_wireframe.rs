//! Debug-wireframe overlay for the GCR client (windowed `play` + the headless
//! screenshot path). A TOGGLE, OFF by default — a diagnostic, never player-facing chrome.
//!
//! The giant crab is a RENDER-ONLY blow-up: its rapier bodies live at the ~1 m arena
//! scale while the visible crab is `crab_render_scale`× bigger and shifted to its
//! game-world spot ([[gcr-crab-render-scale-architecture]]). A naive collider wireframe
//! therefore draws a tiny crab beside the giant. So this offers TWO views, cycled live:
//! - [`WireMode::Aligned`] — the live crab colliders scaled up by the SAME
//!   [`CrabSkinRepose`] the skin uses, so the cage sits ON the giant mesh (the "does my
//!   hitbox match what I see" view). Crab-only; the ~1 m players aren't reposed.
//! - [`WireMode::Raw`] — rapier's own [`RapierDebugRenderPlugin`]: EVERY collider
//!   (players, terrain, crab) at its TRUE physics transform. The crab reads tiny + offset
//!   beside the giant; that's the point — it reveals the physics-vs-render scale gap, so
//!   the on-screen label says RAW.
//!
//! Start in a mode with `--debug-wireframe <off|aligned|raw>` / `RL_DEBUG_WIREFRAME`
//! (and `RL_DEBUG_COLLIDERS` starts Raw, matching the rl-demo idiom); F3 cycles it live.

use super::*;
use bevy_rapier3d::geometry::ColliderView;
use bevy_rapier3d::prelude::{
    Collider, DebugRenderContext, DebugRenderMode, RapierDebugRenderPlugin,
};
use crab_world::bot::body::{CrabBodyPart, CrabEnvId};
use crab_world::bot::skin::CrabSkinRepose;

/// Which collider view the debug overlay draws. Off by default — the player-facing
/// showcase never shows wireframes unless the player asks (F3 / flag / env).
#[derive(Resource, Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum WireMode {
    /// No wireframes (the default, player-facing).
    #[default]
    Off,
    /// The crab's live colliders reposed to the giant render scale — overlays the mesh.
    Aligned,
    /// Rapier's raw debug-render: all colliders at true physics scale (crab tiny + offset).
    Raw,
}

impl WireMode {
    /// Cycle Off → Aligned → Raw → Off, so one key walks every view.
    fn next(self) -> Self {
        match self {
            WireMode::Off => WireMode::Aligned,
            WireMode::Aligned => WireMode::Raw,
            WireMode::Raw => WireMode::Off,
        }
    }

    /// The corner label for this mode — empty when off (no chrome), and explicit that
    /// Raw is physics-scale (NOT aligned to the giant) so the offset cage can't be
    /// misread as a bug.
    fn label(self) -> &'static str {
        match self {
            WireMode::Off => "",
            WireMode::Aligned => "DEBUG WIREFRAME — aligned (crab colliders at render scale)",
            WireMode::Raw => "DEBUG WIREFRAME — RAW physics scale (~1 m arena, NOT aligned)",
        }
    }

    /// Parse a `--debug-wireframe` flag value. `None` for an unknown token (the caller
    /// reports it); the three real modes map by name.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "off" => Some(WireMode::Off),
            "aligned" => Some(WireMode::Aligned),
            "raw" => Some(WireMode::Raw),
            _ => None,
        }
    }

    /// The initial mode from the environment, for callers with no explicit flag. Honors
    /// `RL_DEBUG_WIREFRAME=<mode>`, else falls back to the rl-demo idiom where
    /// `RL_DEBUG_COLLIDERS` (any value) starts the raw collider cage.
    pub fn from_env() -> Self {
        if let Ok(v) = std::env::var("RL_DEBUG_WIREFRAME") {
            return WireMode::parse(&v).unwrap_or_else(|| {
                warn!("RL_DEBUG_WIREFRAME={v:?} not one of off|aligned|raw; defaulting off");
                WireMode::Off
            });
        }
        if std::env::var_os("RL_DEBUG_COLLIDERS").is_some() {
            WireMode::Raw
        } else {
            WireMode::Off
        }
    }
}

/// The corner text node showing the active wireframe mode.
#[derive(Component)]
struct WireLabel;

/// Wire the debug-wireframe overlay into a render `App`, starting in `initial`. Adds
/// rapier's debug-render plugin (the Raw view) disabled-or-enabled to match, the live
/// F3 cycle, the corner label, and the aligned gizmo overlay. Call once, after the sim
/// systems are installed.
pub fn register(app: &mut App, initial: WireMode) {
    app.add_plugins(RapierDebugRenderPlugin {
        // Only the Raw mode wants rapier's cage; Aligned and Off keep it off. F3 flips
        // `DebugRenderContext.enabled` live (see `cycle_wire_mode`).
        enabled: initial == WireMode::Raw,
        // Collider shapes only — the default also draws per-body axes + joint markers,
        // an unreadable tangle on a 31-part body (same call the rl-demo makes).
        mode: DebugRenderMode::COLLIDER_SHAPES,
        ..default()
    });
    app.insert_resource(initial);
    app.add_systems(Startup, spawn_wire_label);
    app.add_systems(Update, (cycle_wire_mode, update_wire_label));
    // Draw the aligned cage AFTER transform propagation so each part's `GlobalTransform`
    // already holds this frame's physics pose; otherwise the cage lags a frame.
    app.add_systems(
        PostUpdate,
        draw_aligned_wireframe.after(TransformSystems::Propagate),
    );
}

fn spawn_wire_label(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.92, 0.2)),
        // Bottom-right: clear of the status HUD (top, full-width line) and the
        // hold-to-reveal controls hint (bottom-left), so the mode line never overlaps them.
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(14.0),
            right: Val::Px(14.0),
            ..default()
        },
        WireLabel,
    ));
}

/// F3 cycles the mode and syncs rapier's raw cage to it. Keyboard-only: the gamepad
/// buttons are all gameplay-bound, and the debug toggle is a developer affordance.
fn cycle_wire_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<WireMode>,
    mut rapier_debug: Option<ResMut<DebugRenderContext>>,
) {
    if !keys.just_pressed(KeyCode::F3) {
        return;
    }
    *mode = mode.next();
    // The raw rapier cage is on iff we're in Raw; Aligned uses our own gizmos instead.
    if let Some(dbg) = rapier_debug.as_mut() {
        dbg.enabled = *mode == WireMode::Raw;
    }
    info!("debug wireframe: {:?}", *mode);
}

fn update_wire_label(mode: Res<WireMode>, mut label: Query<&mut Text, With<WireLabel>>) {
    if !mode.is_changed() {
        return;
    }
    if let Ok(mut text) = label.single_mut() {
        **text = mode.label().to_string();
    }
}

/// Draw the live crab colliders as a wireframe scaled + shifted to the giant render
/// pose, so the cage overlays the visible mesh. Active only in [`WireMode::Aligned`].
///
/// The transform is `repose · part_global`: each part's `GlobalTransform` is its raw
/// ~1 m arena pose, and [`SkinRepose::matrix`] is the EXACT shift-to-game-spot +
/// scale-about-the-ground-pivot the skin applies — reusing it (not a re-derived factor)
/// is why the cage can't drift from the rendered crab. Crab env 0 only (GCR runs one
/// env); the human players keep rapier's raw cage in [`WireMode::Raw`].
fn draw_aligned_wireframe(
    mode: Res<WireMode>,
    repose: Option<Res<CrabSkinRepose>>,
    parts: Query<(&GlobalTransform, &Collider, &CrabEnvId), With<CrabBodyPart>>,
    mut gizmos: Gizmos,
) {
    if *mode != WireMode::Aligned {
        return;
    }
    // No repose yet (skin not loaded, or the crab unsampled this frame) ⇒ nothing to
    // align against; the raw mode still shows the ~1 m colliders.
    let Some(Some(rep)) = repose.as_deref().map(|r| r.0) else {
        return;
    };
    let repose_mat = rep.matrix();
    let color = Color::srgb(0.2, 1.0, 0.4);
    for (gt, collider, env) in &parts {
        if env.0 != 0 {
            continue;
        }
        let world = repose_mat * gt.to_matrix();
        draw_collider_view(
            &mut gizmos,
            collider.as_typed_shape(),
            world,
            rep.scale,
            color,
        );
    }
}

/// Draw one collider view as gizmo lines under world transform `world` (which carries
/// the giant's uniform `scale`). Handles the shapes the crab body actually uses — the
/// carapace compound-of-cuboid and the per-link capsules; other shapes are skipped
/// (the crab has none).
fn draw_collider_view(
    gizmos: &mut Gizmos,
    view: ColliderView<'_>,
    world: Mat4,
    scale: f32,
    color: Color,
) {
    match view {
        ColliderView::Cuboid(c) => draw_cuboid(gizmos, world, c.half_extents(), color),
        ColliderView::Capsule(c) => {
            let seg = c.segment();
            draw_capsule(gizmos, world, seg.a(), seg.b(), c.radius() * scale, color);
        }
        ColliderView::Compound(c) => {
            for (pos, rot, sub) in c.shapes() {
                let sub_world = world * Mat4::from_rotation_translation(rot, pos);
                draw_collider_view(gizmos, sub, sub_world, scale, color);
            }
        }
        // Crab colliders are only the above; anything else (a future shape) is skipped
        // rather than mis-drawn.
        _ => {}
    }
}

/// Wireframe box: the 12 edges of the cuboid `±half`, each corner pushed through
/// `world` (which includes the giant scale, so no separate scaling here). bevy 0.18's
/// gizmos have no cuboid primitive, so draw the edges directly.
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
    // bottom ring, top ring, verticals.
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

/// Wireframe capsule between the link-local segment endpoints `a`,`b` pushed through
/// `world`. `radius` is already giant-scaled by the caller (the segment endpoints pick
/// up the scale from `world`). The `Capsule3d` gizmo is Y-aligned, so rotate +Y onto the
/// segment direction.
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
