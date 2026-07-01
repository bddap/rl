//! Hands-on gamepad control: an alternative to the policy where a human feels the joint
//! dynamics by hand (a physics feel-test, not a learned driver). Toggled live with B/circle;
//! while active the D-pad picks a joint and the right stick drives its torque.

use bevy::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::body::CrabJointId;

/// Hands-on gamepad control state. `active` is toggled live with B/circle (and
/// seeded by `--manual-control`); `selected` is the joint the right stick drives,
/// `None` = all joints at zero torque until the D-pad picks one. Typed as a
/// [`CrabJointId`] (not a bare slot index) so the live joint is a valid joint by
/// construction — it names itself, with no runtime lookup that could miss.
#[derive(Resource)]
pub(super) struct ManualControl {
    pub(super) active: bool,
    pub(super) selected: Option<CrabJointId>,
}

/// Marker for the on-screen manual-control readout (the mode + the live joint).
#[derive(Component)]
pub(super) struct ManualHud;

/// System (BotSet::Think): hands-on gamepad control as an alternative to the policy.
/// B/circle toggles it; while active the D-pad up/down cycles which joint is live
/// and the right stick's Y drives THAT joint's torque (effort, not a target angle),
/// every other joint held at zero — a human feeling the joint dynamics by hand. When
/// inactive it hides its readout; `policy_step` drives. The "how to enter manual" prompt
/// lives in the data-driven controls overlay (hold Tab/View), not an always-on banner —
/// so this readout shows ONLY while manual is live (the joint being felt), nothing in the
/// default policy view.
pub(super) fn manual_control_step(
    gamepads: Query<&Gamepad>,
    mut manual: ResMut<ManualControl>,
    mut actions: ResMut<CrabActions>,
    mut hud: Query<(&mut Text, &mut Visibility), With<ManualHud>>,
) {
    let Some(gp) = gamepads.iter().next() else {
        return;
    };
    // East (B / circle), NOT North: North already toggles the joint-telemetry
    // graph (play::graph), so sharing it fired both on one press.
    if gp.just_pressed(GamepadButton::East) {
        manual.active = !manual.active;
        manual.selected = None;
    }

    let n = CrabJointId::COUNT;
    let mut line = String::new();
    if manual.active {
        // Cycle the live joint by slot, wrapping. `from_index` always yields a joint for a
        // value in 0..COUNT, so `selected` becomes `Some(joint)` on the first press.
        if gp.just_pressed(GamepadButton::DPadUp) {
            manual.selected = CrabJointId::from_index(manual.selected.map_or(0, |j| (j.index() + 1) % n));
        }
        if gp.just_pressed(GamepadButton::DPadDown) {
            manual.selected =
                CrabJointId::from_index(manual.selected.map_or(0, |j| (j.index() + n - 1) % n));
        }
        if let Some(a) = actions.envs.first_mut() {
            // Drive only the selected joint; hold everything else at zero torque.
            *a = [0.0; CrabJointId::COUNT];
            line = match manual.selected {
                Some(id) => {
                    let v = gp.right_stick().y.clamp(-1.0, 1.0);
                    let sel = id.index();
                    a[sel] = v;
                    // The typed joint names itself (`Debug`) — no entity lookup that could miss.
                    format!("MANUAL · {id:?} {}/{n} · torque {v:+.2}", sel + 1)
                }
                None => "MANUAL · pick a joint (D-pad)".to_string(),
            };
        }
    }
    if let Ok((mut text, mut vis)) = hud.single_mut() {
        // Touch each component only on an actual change so a hidden HUD in policy mode
        // isn't churning change-detection every frame. `line` is empty unless active, so
        // the text only updates while the readout is shown.
        let want = if manual.active {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
        if manual.active && **text != line {
            **text = line;
        }
    }
}

/// Top-right readout of the live manual joint, hidden in policy mode. Top-right because the
/// joint-telemetry graph owns the top-left corner; `manual_control_step` shows it only while
/// hands-on control is active.
pub(super) fn spawn_manual_hud(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.9, 0.4)),
        Visibility::Hidden,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            right: Val::Px(12.0),
            ..default()
        },
        ManualHud,
    ));
}
