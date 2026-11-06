//! Developer debug overlay (rl#326): OFF by default, F3 (or a surface's chord) toggles it
//! at runtime. Deliberately outside the gameplay HUD/controls chrome — the clean-screen
//! rule for normal play doesn't apply to it, but it must render NOTHING until toggled.
//! Widgets so far: FPS counter + sim-time readout + frame-time graph; new widgets are
//! further children of the root column.

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
/// keyboard — the fp-screenshot evidence path, GCR's chord toggle — can flip it from
/// outside this module.
#[derive(Resource, Default)]
pub struct DebugOverlay {
    pub visible: bool,
}

/// This frame's sim cost, written by the surface's sim driver (GCR's `drive_client_sim`)
/// and CONSUMED by [`record_sample`] — a frame the driver skips (menu) records
/// zeros, never a stale value. `ticks` pinned at the driver's per-frame cap means the sim
/// can't keep up with wall time — the death-spiral signature.
#[derive(Resource, Default)]
pub struct SimFrameStats {
    pub ms: f32,
    pub ticks: u32,
}

/// One frame of the graph. The sim columns and `frame_ms` can be offset by one row:
/// the driver and overlay chains have no ordering edge.
#[derive(Clone, Copy)]
struct PerfSample {
    frame_ms: f32,
    sim_ms: f32,
    sim_ticks: u32,
}

/// Rolling wall-clock frame records; `Time<Real>` like [`crate::frame_telemetry`] — a
/// paused game still renders frames. Recorded even while hidden, so toggling on shows
/// the seconds that led up to the toggle.
#[derive(Resource, Default)]
struct PerfRing(VecDeque<PerfSample>);

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
            .init_resource::<PerfRing>()
            .init_resource::<SimFrameStats>()
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

fn record_sample(
    time: Res<Time<Real>>,
    mut sim: ResMut<SimFrameStats>,
    mut ring: ResMut<PerfRing>,
) {
    let sim = std::mem::take(&mut *sim);
    push_sample(
        &mut ring.0,
        PerfSample {
            frame_ms: time.delta_secs() * 1000.0,
            sim_ms: sim.ms,
            sim_ticks: sim.ticks,
        },
    );
}

fn push_sample(ring: &mut VecDeque<PerfSample>, sample: PerfSample) {
    if ring.len() == GRAPH_COLS {
        ring.pop_front();
    }
    ring.push_back(sample);
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
                    font_size: FontSize::Px(18.0),
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
    ring: Res<PerfRing>,
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
    let window: Vec<&PerfSample> = ring.0.iter().rev().take(60).collect();
    let frame_sum: f32 = window.iter().map(|s| s.frame_ms).sum();
    if window.is_empty() || frame_sum <= 0.0 {
        return;
    }
    let n = window.len() as f32;
    let frame_mean = frame_sum / n;
    let sim_mean: f32 = window.iter().map(|s| s.sim_ms).sum::<f32>() / n;
    let ticks = window.first().map_or(0, |s| s.sim_ticks);
    **text = format!(
        "{:.0} fps  {frame_mean:.1} ms | sim {sim_mean:.1} ms  {ticks} ticks",
        1000.0 / frame_mean
    );
}

fn update_graph(
    ring: Res<PerfRing>,
    mut bars: Query<(&GraphBar, &mut Node, &mut BackgroundColor)>,
) {
    // Newest sample lands on the rightmost bar; a not-yet-full window leaves the left
    // blank. The graph shows the ring's newest GRAPH_COLS samples.
    let skip = ring.0.len().saturating_sub(GRAPH_COLS);
    let pad = GRAPH_COLS.saturating_sub(ring.0.len());
    for (bar, mut node, mut color) in &mut bars {
        let Some(s) = bar.0.checked_sub(pad).and_then(|i| ring.0.get(skip + i)) else {
            node.height = Val::Px(0.0);
            *color = BackgroundColor(Color::NONE);
            continue;
        };
        let ms = s.frame_ms;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_capped() {
        let mut ring = VecDeque::new();
        for i in 0..(GRAPH_COLS + 10) {
            push_sample(
                &mut ring,
                PerfSample {
                    frame_ms: i as f32,
                    sim_ms: 1.0,
                    sim_ticks: 2,
                },
            );
        }
        assert_eq!(ring.len(), GRAPH_COLS);
        assert_eq!(ring.back().unwrap().frame_ms, (GRAPH_COLS + 9) as f32);
    }
}
