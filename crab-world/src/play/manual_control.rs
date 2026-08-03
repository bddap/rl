use bevy::prelude::*;

use crate::bot::actuator::CrabActions;
use crate::bot::body::CrabJointId;
use crate::chord::Chords;

use super::controls::{
    DemoControls, PICK_JOINT_NEXT_BUTTON, PICK_JOINT_PREV_BUTTON, torque_stick_y,
};

#[derive(Resource)]
pub(super) struct ManualControl {
    pub(super) active: bool,
    pub(super) selected: Option<CrabJointId>,
}

#[derive(Component)]
pub(super) struct ManualHud;

pub(super) fn manual_control_step(
    gamepads: Query<&Gamepad>,
    chords: Res<Chords<DemoControls>>,
    mut manual: ResMut<ManualControl>,
    mut actions: ResMut<CrabActions>,
    mut hud: Query<(&mut Text, &mut Visibility), With<ManualHud>>,
) {
    let Some(gp) = gamepads.iter().next() else {
        return;
    };
    // The mode TOGGLE is a chord, dispatched in `demo_controls` (Update — the chord edge
    // is Update-rate). The analog joint pick + torque below stay on the FIRST pad. The
    // D-pad is also chord-code entry: while a capture is live (pad X held), the taps are
    // code, not joint picks.
    let n = CrabJointId::COUNT;
    let mut line = String::new();
    if manual.active {
        if !chords.capturing() {
            if gp.just_pressed(PICK_JOINT_NEXT_BUTTON) {
                manual.selected =
                    CrabJointId::from_index(manual.selected.map_or(0, |j| (j.index() + 1) % n));
            }
            if gp.just_pressed(PICK_JOINT_PREV_BUTTON) {
                manual.selected =
                    CrabJointId::from_index(manual.selected.map_or(0, |j| (j.index() + n - 1) % n));
            }
        }
        if actions.rest(0) {
            line = match manual.selected {
                Some(id) => {
                    let v = torque_stick_y(gp).clamp(-1.0, 1.0);
                    // The rest(0) above proved env 0 is sized.
                    let _ = actions.set_drive(0, id, v);
                    format!("MANUAL · {id:?} {}/{n} · torque {v:+.2}", id.index() + 1)
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
