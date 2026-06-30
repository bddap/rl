//! Hands-on gamepad control: an alternative to the policy where a human feels the joint
//! dynamics by hand (a physics feel-test, not a learned driver). Toggled live with B/circle;
//! while active the D-pad picks a joint and the right stick drives its torque.

use bevy::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::body::{CrabJoint, CrabJointId};

/// Hands-on gamepad control state. `active` is toggled live with Y/triangle (and
/// seeded by `--manual-control`); `selected` is the joint the right stick drives
/// ([`CrabJointId::index`]), None = all joints at zero torque until the D-pad picks one.
#[derive(Resource)]
pub(super) struct ManualControl {
    pub(super) active: bool,
    pub(super) selected: Option<usize>,
}

/// Marker for the on-screen manual-control readout (the mode + the live joint).
#[derive(Component)]
pub(super) struct ManualHud;

/// System (BotSet::Think): hands-on gamepad control as an alternative to the policy.
/// Y/triangle toggles it; while active the D-pad up/down cycles which joint is live
/// and the right stick's Y drives THAT joint's torque (effort, not a target angle),
/// every other joint held at zero — a human feeling the joint dynamics by hand. When
/// inactive it only refreshes the HUD; `policy_step` drives.
pub(super) fn manual_control_step(
    gamepads: Query<&Gamepad>,
    joint_ids: Query<&CrabJoint>,
    mut manual: ResMut<ManualControl>,
    mut actions: ResMut<CrabActions>,
    mut hud: Query<&mut Text, With<ManualHud>>,
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
    let mut line = "POLICY  (press B / circle for hands-on manual control)".to_string();
    if manual.active {
        if gp.just_pressed(GamepadButton::DPadUp) {
            manual.selected = Some(manual.selected.map_or(0, |i| (i + 1) % n));
        }
        if gp.just_pressed(GamepadButton::DPadDown) {
            manual.selected = Some(manual.selected.map_or(0, |i| (i + n - 1) % n));
        }
        if let Some(a) = actions.envs.first_mut() {
            // Drive only the selected joint; hold everything else at zero torque.
            *a = [0.0; CrabJointId::COUNT];
            line = match manual.selected {
                Some(sel) => {
                    let v = gp.right_stick().y.clamp(-1.0, 1.0);
                    a[sel] = v;
                    let name = joint_ids
                        .iter()
                        .find(|j| j.id.index() == sel)
                        .map(|j| format!("{:?}", j.id))
                        .unwrap_or_else(|| format!("#{sel}"));
                    format!(
                        "MANUAL  (B: exit   D-pad: pick joint   R-stick Y: torque)\n\
                         joint {sel}/{n}  {name}   torque {v:+.2}"
                    )
                }
                None => {
                    "MANUAL  (B: exit   D-pad up/down: pick a joint, then R-stick Y to actuate)"
                        .to_string()
                }
            };
        }
    }
    if let Ok(mut text) = hud.single_mut() {
        **text = line;
    }
}

/// Top-right readout of the active driver (policy vs manual) and the live joint.
/// Top-right because the joint-telemetry graph owns the top-left corner.
pub(super) fn spawn_manual_hud(mut commands: Commands) {
    commands.spawn((
        // Overwritten every frame by `manual_control_step`; seed it with the idle
        // (policy) line so the pre-first-update frame doesn't flash a stale label.
        Text::new("POLICY  (press B / circle for hands-on manual control)"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.9, 0.4)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            right: Val::Px(12.0),
            ..default()
        },
        ManualHud,
    ));
}
