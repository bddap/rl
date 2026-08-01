//! Developer debug overlay (rl#326): OFF by default, F3 toggles it at runtime. Deliberately
//! outside the gameplay HUD/controls chrome — the clean-screen rule for normal play doesn't
//! apply to it, but it must render NOTHING until toggled. Widgets so far: FPS counter +
//! frame-time graph; new widgets are further children of the root column.

use std::collections::VecDeque;

use bevy::prelude::*;

/// 240 columns × 2 px ≈ 4 s of history at 60 fps.
const GRAPH_COLS: usize = 240;
const GRAPH_COL_PX: f32 = 2.0;
const GRAPH_HEIGHT_PX: f32 = 64.0;
/// A full-height bar is a 30 fps (33.3 ms) frame; slower frames clamp at the top.
const FULL_SCALE_MS: f32 = 1000.0 / 30.0;
/// Refresh cadence for the numeric readout — per-frame it flickers unreadably.
const FPS_TEXT_PERIOD_SECS: f32 = 0.25;
/// The counter and per-bar colors average/judge against nominal frame budgets.
const GOOD_MS: f32 = 17.0;
const OK_MS: f32 = 33.4;

pub struct DebugOverlayPlugin;

/// Toggle state as a resource (not just the root's `Visibility`) so surfaces without a
/// keyboard — the fp-screenshot evidence path — can boot with the overlay on by
/// overwriting it after app construction.
#[derive(Resource, Default)]
pub struct DebugOverlay {
    pub visible: bool,
}

/// Rolling wall-clock frame deltas; `Time<Real>` like [`crate::frame_telemetry`] — a
/// paused game still renders frames. Recorded even while hidden, so toggling on shows
/// the seconds that led up to the toggle instead of an empty graph.
#[derive(Resource, Default)]
struct FrameSamples(VecDeque<f32>);

#[derive(Component)]
struct OverlayRoot;

#[derive(Component)]
struct FpsText;

/// Column index into the graph, 0 = oldest (leftmost).
#[derive(Component)]
struct GraphBar(usize);

impl Plugin for DebugOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugOverlay>()
            .init_resource::<FrameSamples>()
            .add_systems(Startup, spawn_overlay)
            .add_systems(
                Update,
                (
                    toggle,
                    apply_visibility.run_if(resource_changed::<DebugOverlay>),
                    record_sample,
                    (update_fps_text, update_graph).run_if(|o: Res<DebugOverlay>| o.visible),
                )
                    .chain(),
            );
    }
}

fn toggle(keys: Res<ButtonInput<KeyCode>>, mut overlay: ResMut<DebugOverlay>) {
    if keys.just_pressed(KeyCode::F3) {
        overlay.visible = !overlay.visible;
    }
}

fn apply_visibility(
    overlay: Res<DebugOverlay>,
    mut root: Query<&mut Visibility, With<OverlayRoot>>,
) {
    let Ok(mut vis) = root.single_mut() else {
        return;
    };
    *vis = if overlay.visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
}

fn record_sample(time: Res<Time<Real>>, mut samples: ResMut<FrameSamples>) {
    if samples.0.len() == GRAPH_COLS {
        samples.0.pop_front();
    }
    samples.0.push_back(time.delta_secs());
}

fn spawn_overlay(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                padding: UiRect::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            Visibility::Hidden,
            OverlayRoot,
        ))
        .with_children(|root| {
            root.spawn((
                Text::new(""),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(0.95, 0.95, 0.95)),
                FpsText,
            ));
            root.spawn(Node {
                width: Val::Px(GRAPH_COLS as f32 * GRAPH_COL_PX),
                height: Val::Px(GRAPH_HEIGHT_PX),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::End,
                ..default()
            })
            .with_children(|graph| {
                for i in 0..GRAPH_COLS {
                    graph.spawn((
                        Node {
                            width: Val::Px(GRAPH_COL_PX),
                            height: Val::Px(0.0),
                            ..default()
                        },
                        BackgroundColor(Color::NONE),
                        GraphBar(i),
                    ));
                }
            });
        });
}

fn update_fps_text(
    samples: Res<FrameSamples>,
    time: Res<Time<Real>>,
    mut acc: Local<f32>,
    mut text: Query<&mut Text, With<FpsText>>,
) {
    *acc += time.delta_secs();
    if *acc < FPS_TEXT_PERIOD_SECS {
        return;
    }
    *acc = 0.0;
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    // Mean over the last ~second, not the last frame: one hitch shouldn't own the number
    // (the graph is where individual spikes show).
    let window: Vec<f32> = samples.0.iter().rev().take(60).copied().collect();
    let sum: f32 = window.iter().sum();
    if window.is_empty() || sum <= 0.0 {
        return;
    }
    let mean = sum / window.len() as f32;
    **text = format!("{:.0} fps  {:.1} ms", 1.0 / mean, mean * 1000.0);
}

fn update_graph(
    samples: Res<FrameSamples>,
    mut bars: Query<(&GraphBar, &mut Node, &mut BackgroundColor)>,
) {
    // Newest sample lands on the rightmost bar; a not-yet-full window leaves the left blank.
    let pad = GRAPH_COLS.saturating_sub(samples.0.len());
    for (bar, mut node, mut color) in &mut bars {
        let Some(&dt) = bar.0.checked_sub(pad).and_then(|i| samples.0.get(i)) else {
            node.height = Val::Px(0.0);
            *color = BackgroundColor(Color::NONE);
            continue;
        };
        let ms = dt * 1000.0;
        node.height = Val::Px((ms / FULL_SCALE_MS * GRAPH_HEIGHT_PX).clamp(1.0, GRAPH_HEIGHT_PX));
        *color = BackgroundColor(if ms <= GOOD_MS {
            Color::srgb(0.4, 1.0, 0.55)
        } else if ms <= OK_MS {
            Color::srgb(1.0, 0.8, 0.2)
        } else {
            Color::srgb(1.0, 0.3, 0.25)
        });
    }
}
