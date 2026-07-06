
use bevy::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::body::CrabJointId;
use crate::controls::just_pressed;

use super::controls::{
    DemoAction, DemoControls, PICK_JOINT_NEXT_BUTTON, PICK_JOINT_PREV_BUTTON, torque_stick_y,
};

#[derive(Resource)]
pub(super) struct ManualControl {
    pub(super) active: bool,
    pub(super) selected: Option<CrabJointId>,
}

#[derive(Component)]
pub(super) struct ManualHud;

pub(super) fn manual_control_step(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut manual: ResMut<ManualControl>,
    mut actions: ResMut<CrabActions>,
    mut hud: Query<(&mut Text, &mut Visibility), With<ManualHud>>,
) {
    let Some(gp) = gamepads.iter().next() else {
        return;
    };
    // Dispatched from DEMO_BINDINGS like every tap verb (pad-only today — East, B/circle —
    // NOT North: North already toggles the joint-telemetry graph (play::graph), so sharing
    // it fired both on one press). The analog joint pick + torque below stay on the FIRST
    // pad; the toggle accepts any pad like the other verbs.
    if just_pressed::<DemoControls>(DemoAction::Manual, &keys, &gamepads) {
        manual.active = !manual.active;
        manual.selected = None;
    }

    let n = CrabJointId::COUNT;
    let mut line = String::new();
    if manual.active {
        if gp.just_pressed(PICK_JOINT_NEXT_BUTTON) {
            manual.selected =
                CrabJointId::from_index(manual.selected.map_or(0, |j| (j.index() + 1) % n));
        }
        if gp.just_pressed(PICK_JOINT_PREV_BUTTON) {
            manual.selected =
                CrabJointId::from_index(manual.selected.map_or(0, |j| (j.index() + n - 1) % n));
        }
        if let Some(a) = actions.envs.first_mut() {
            *a = [0.0; CrabJointId::COUNT];
            line = match manual.selected {
                Some(id) => {
                    let v = torque_stick_y(gp).clamp(-1.0, 1.0);
                    let sel = id.index();
                    a[sel] = v;
                    format!("MANUAL · {id:?} {}/{n} · torque {v:+.2}", sel + 1)
                }
                None => "MANUAL · pick a joint (D-pad)".to_string(),
            };
        }
    }
    if let Ok((mut text, mut vis)) = hud.single_mut() {
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
