//! Togglable transparent joint telemetry overlay for the demo.
//!
//! Two scrolling line graphs drawn over the scene: every joint's **angle**
//! (the revolute coordinate, radians) on top and its **commanded torque** on
//! the bottom. The crab is torque-controlled, so the torque trace is the
//! policy's actual normalized output [-1, 1] — the real command, not a
//! reconstructed proxy. The angle is read off the links' orientations with the
//! same helper the observation uses, so the plot shows exactly what the policy
//! senses.
//!
//! Lines are drawn with gizmos through a dedicated 2D overlay camera on its own
//! render layer, so they composite on top of the 3D view without a second
//! opaque pass — the overlay is transparent by construction. Toggle with `G`
//! or the gamepad's North button. `RL_GRAPH=1` starts it visible;
//! `RL_GRAPH_SHOT=<path>` forces it on, captures the window after it settles,
//! and exits (headless self-check of the overlay).

use std::collections::VecDeque;
use std::path::PathBuf;

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy_rapier3d::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabEnvId, CrabJoint, CrabJointId, joint_angle};

/// Samples kept per joint trace (~4 s at 60 Hz).
const CAPACITY: usize = 240;
/// The overlay camera's render layer, isolated from the 3D scene (layer 0) so
/// its gizmos draw only here.
const OVERLAY_LAYER: usize = 1;
/// Fixed y-extent of the angle plot (radians); joint coords stay within ±π.
const ANGLE_RANGE: f32 = std::f32::consts::PI;

/// Per-joint scrolling history plus the visibility toggle.
#[derive(Resource)]
pub struct JointGraph {
    visible: bool,
    angle: Vec<VecDeque<f32>>,
    effort: Vec<VecDeque<f32>>,
}

impl Default for JointGraph {
    fn default() -> Self {
        Self {
            visible: std::env::var("RL_GRAPH").is_ok_and(|v| v == "1")
                || std::env::var("RL_GRAPH_SHOT").is_ok(),
            angle: vec![VecDeque::with_capacity(CAPACITY); CrabJointId::COUNT],
            effort: vec![VecDeque::with_capacity(CAPACITY); CrabJointId::COUNT],
        }
    }
}

/// Marks the labels/help text so visibility can follow the toggle.
#[derive(Component)]
struct GraphUi;

/// Drives the `RL_GRAPH_SHOT` self-check: capture the window once the scene has
/// settled, then exit. Absent when the env var is unset.
#[derive(Resource)]
struct GraphShot {
    path: PathBuf,
    frame: u32,
}

pub fn register(app: &mut App) {
    app.init_resource::<JointGraph>();
    if let Ok(path) = std::env::var("RL_GRAPH_SHOT") {
        app.insert_resource(GraphShot {
            path: path.into(),
            frame: 0,
        });
        app.add_systems(Update, graph_shot_capture);
    }
    app.add_systems(Startup, setup_overlay);
    app.add_systems(Update, (toggle_graph, draw_graph));
    app.add_systems(FixedUpdate, sample_graph);
}

fn setup_overlay(
    mut commands: Commands,
    mut configs: ResMut<GizmoConfigStore>,
    graph: Res<JointGraph>,
) {
    // Restrict gizmos to the overlay layer so they render through the 2D
    // camera only, never the 3D scene camera.
    let (config, _) = configs.config_mut::<DefaultGizmoConfigGroup>();
    config.render_layers = RenderLayers::layer(OVERLAY_LAYER);
    config.line.width = 1.5;

    commands.spawn((
        Camera2d,
        Camera {
            order: 10,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        RenderLayers::layer(OVERLAY_LAYER),
    ));

    let vis = if graph.visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    commands.spawn((
        GraphUi,
        vis,
        Text::new("joint telemetry (G)\ntop: angle (rad)   bottom: commanded torque"),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.85)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn toggle_graph(
    keys: Res<ButtonInput<KeyCode>>,
    pads: Query<&Gamepad>,
    mut graph: ResMut<JointGraph>,
    mut ui: Query<&mut Visibility, With<GraphUi>>,
) {
    let pad = pads.iter().any(|p| p.just_pressed(GamepadButton::North));
    if keys.just_pressed(KeyCode::KeyG) || pad {
        graph.visible = !graph.visible;
        for mut v in ui.iter_mut() {
            *v = if graph.visible {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
    }
}

/// Samples each joint's angle and commanded torque for env 0 every physics
/// step. Runs unconditionally so toggling the overlay on shows recent history.
fn sample_graph(
    actions: Res<CrabActions>,
    mut graph: ResMut<JointGraph>,
    joints: Query<(&CrabJoint, &CrabEnvId, &MultibodyJoint, &Transform)>,
    transforms: Query<&Transform>,
) {
    let Some(action) = actions.envs.first() else {
        return;
    };
    for (joint, env, mj, child_tf) in joints.iter() {
        if env.0 != 0 {
            continue;
        }
        let Ok(parent_tf) = transforms.get(mj.parent) else {
            continue;
        };
        let id = joint.id;
        let idx = id.index();

        let angle = joint_angle(id, parent_tf.rotation, child_tf.rotation);
        // Under torque control the NN output IS the signed torque (normalized),
        // so the "torque" plot is the real command, not a reconstructed proxy.
        let torque = action[idx].clamp(-1.0, 1.0);

        push(&mut graph.angle[idx], angle);
        push(&mut graph.effort[idx], torque);
    }
}

fn push(buf: &mut VecDeque<f32>, v: f32) {
    if buf.len() == CAPACITY {
        buf.pop_front();
    }
    buf.push_back(v);
}

fn draw_graph(graph: Res<JointGraph>, windows: Query<&Window>, mut gizmos: Gizmos) {
    if !graph.visible {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let (w, h) = (window.width(), window.height());
    let margin = 24.0;
    let plot_w = (w * 0.42).min(560.0);
    let plot_h = (h * 0.26).min(240.0);
    let left = -w / 2.0 + margin;
    let top = h / 2.0 - margin;

    // angle plot, then effort plot below it.
    draw_plot(
        &mut gizmos,
        &graph.angle,
        left,
        top,
        plot_w,
        plot_h,
        ANGLE_RANGE,
    );
    draw_plot(
        &mut gizmos,
        &graph.effort,
        left,
        top - plot_h - 40.0,
        plot_w,
        plot_h,
        1.0,
    );
}

/// Draws one plot: a faint frame, a zero line, and one colored trace per joint.
/// `top` is the plot's top edge (center-origin, +y up); values map symmetric
/// about the vertical midpoint with the given half-range.
fn draw_plot(
    gizmos: &mut Gizmos,
    series: &[VecDeque<f32>],
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    range: f32,
) {
    let bottom = top - height;
    let mid = (top + bottom) / 2.0;
    let frame = Color::srgba(1.0, 1.0, 1.0, 0.25);
    gizmos.rect_2d(
        Isometry2d::from_translation(Vec2::new(left + width / 2.0, mid)),
        Vec2::new(width, height),
        frame,
    );
    gizmos.line_2d(
        Vec2::new(left, mid),
        Vec2::new(left + width, mid),
        Color::srgba(1.0, 1.0, 1.0, 0.15),
    );

    for (j, buf) in series.iter().enumerate() {
        if buf.len() < 2 {
            continue;
        }
        let hue = (j as f32 / series.len() as f32) * 360.0;
        let color = Color::hsla(hue, 0.85, 0.6, 0.8);
        let pts = buf.iter().enumerate().map(|(i, &v)| {
            let x = left + (i as f32 / (CAPACITY - 1) as f32) * width;
            let y = mid + (v / range).clamp(-1.0, 1.0) * (height / 2.0);
            Vec2::new(x, y)
        });
        gizmos.linestrip_2d(pts, color);
    }
}

/// `RL_GRAPH_SHOT`: let the scene settle, capture the composited window
/// (overlay included), then exit — a human-free check that the overlay renders.
fn graph_shot_capture(
    mut commands: Commands,
    mut shot: ResMut<GraphShot>,
    mut exit: MessageWriter<AppExit>,
) {
    shot.frame += 1;
    if shot.frame == 150 {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(shot.path.clone()));
        info!(
            "graph self-check: capturing window to {}",
            shot.path.display()
        );
    }
    if shot.frame >= 156 {
        exit.write(AppExit::Success);
    }
}
