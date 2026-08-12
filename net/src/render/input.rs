use super::driver::{FlightInput, PendingInput};
use super::*;

pub(super) struct PadAxes {
    pub(super) strafe: f32,
    pub(super) forward: f32,
    pub(super) d_yaw: f32,
    pub(super) d_pitch: f32,
}

pub(super) fn pad_stick_axes(left_stick: Vec2, right_stick: Vec2, dt: f32) -> PadAxes {
    let (mut strafe, mut forward) = (0.0, 0.0);
    if left_stick.length() > PAD_STICK_DEADZONE {
        strafe = left_stick.x;
        forward = left_stick.y;
    }
    let (mut d_yaw, mut d_pitch) = (0.0, 0.0);
    if right_stick.length() > PAD_STICK_DEADZONE {
        d_yaw = right_stick.x * PAD_LOOK_SPEED * dt;
        d_pitch = right_stick.y * PAD_LOOK_SPEED * dt;
    }
    PadAxes {
        strafe,
        forward,
        d_yaw,
        d_pitch,
    }
}

#[derive(Resource, Default)]
pub(super) struct CameraPitch(pub(super) f32);

#[derive(Resource, Default)]
pub(super) struct CameraYaw(pub(super) f32);

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    cursor: Query<&CursorOptions, With<PrimaryWindow>>,
    chords: Res<crab_world::chord::Chords<controls::GcrControls>>,
    voice: Res<super::voice::VoiceUx>,
    mut pending: ResMut<PendingInput>,
    mut flight: ResMut<FlightInput>,
    mut pitch: ResMut<CameraPitch>,
    mut yaw: ResMut<CameraYaw>,
) {
    let dt = time.delta_secs();

    // While a chord is being typed (rl#330) the WASD keys and d-pad ARE the code entry,
    // so their movement/flight readings go quiet for the span of the hold — including
    // the release frame (`typing`, not `capturing`: a release-frame tap joins the code
    // and must not also blip movement); the sticks stay live (analog is out of scope).
    let typing = chords.typing();

    let kc = controls::key_code_for;
    let held = |a| kc(a).is_some_and(|k| keys.pressed(k));

    let mut strafe = 0.0f32;
    let mut forward = 0.0f32;
    if !typing {
        forward += held(Action::MoveForward) as i32 as f32;
        forward -= held(Action::MoveBack) as i32 as f32;
        strafe += held(Action::StrafeRight) as i32 as f32;
        strafe -= held(Action::StrafeLeft) as i32 as f32;
    }

    let mut action = held(Action::Extract);
    // The on-foot movement stances (rl#355). Held-state like the axes — the sim owns
    // the semantics (sprint scales pace, jump lifts off grounded, slide skids on
    // momentum); a held jump auto-hops by design. `key_codes_for`, not `held`: sprint
    // binds BOTH shifts and the first-key shorthand would drop the right one.
    let held_any = |a| controls::key_codes_for(a).any(|k| keys.pressed(k));
    // While the voice-review modal is up (rl#378) the pad's South/East ARE
    // keep/discard, so their jump/slide/brake readings go quiet — same shape as the
    // chord-typing gate above. Keyboard stays live: confirm/discard ride Enter/Esc
    // there, which never collide with Space/C.
    let reviewing = voice.modal();
    let mut sprint = held_any(Action::Sprint);
    let mut jump = held_any(Action::Jump);
    let mut slide = held_any(Action::Slide);
    // Vehicle switching and restart are chords (rl#330): one code per vehicle plus an
    // exit code (X is de-overloaded — a bare tap does nothing), restart its own code —
    // no direct key or button remains for any of them.
    use super::driver::VehicleRequest;
    use crab_world::vehicle::VehicleKind;
    if chords.executed(Action::BoardPlane) {
        pending.vehicle = Some(VehicleRequest::Board(VehicleKind::Plane));
    }
    if chords.executed(Action::BoardShip) {
        pending.vehicle = Some(VehicleRequest::Board(VehicleKind::Ship));
    }
    if chords.executed(Action::ExitVehicle) {
        pending.vehicle = Some(VehicleRequest::Exit);
    }
    if chords.executed(Action::Restart) {
        pending.restart = true;
    }

    let mut d_yaw = 0.0f32;
    let mut d_pitch = 0.0f32;

    let grabbed = cursor
        .iter()
        .next()
        .is_some_and(|c| c.grab_mode != CursorGrabMode::None);
    if grabbed {
        let d = mouse_motion.delta;
        d_yaw += d.x * MOUSE_SENS;
        d_pitch -= d.y * MOUSE_SENS;
    }

    for gp in gamepads.iter() {
        let pad = pad_stick_axes(gp.left_stick(), gp.right_stick(), dt);
        strafe += pad.strafe;
        forward += pad.forward;
        d_yaw += pad.d_yaw;
        d_pitch += pad.d_pitch;
        if !typing {
            forward += (gp.pressed(GamepadButton::DPadUp) as i32
                - gp.pressed(GamepadButton::DPadDown) as i32) as f32;
            strafe += (gp.pressed(GamepadButton::DPadRight) as i32
                - gp.pressed(GamepadButton::DPadLeft) as i32) as f32;
        }
        action |= controls::gamepad_buttons_for(Action::Extract).any(|b| gp.pressed(b));
        sprint |= controls::gamepad_buttons_for(Action::Sprint).any(|b| gp.pressed(b));
        if !reviewing {
            jump |= controls::gamepad_buttons_for(Action::Jump).any(|b| gp.pressed(b));
            slide |= controls::gamepad_buttons_for(Action::Slide).any(|b| gp.pressed(b));
        }
    }
    if let Some(mb) = controls::MouseInput::Left.mouse_button() {
        action |= mouse_buttons.pressed(mb);
    }

    pending.strafe = (-strafe).clamp(-1.0, 1.0);
    pending.forward = forward.clamp(-1.0, 1.0);
    pending.yaw_delta -= d_yaw;
    pending.action |= action;
    pending.sprint |= sprint;
    pending.jump |= jump;
    pending.slide |= slide;

    pitch.0 = (pitch.0 + d_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    yaw.0 = (yaw.0 - d_yaw).rem_euclid(std::f32::consts::TAU);

    let nth_key = |a: Action, n: usize| {
        controls::key_codes_for(a)
            .nth(n)
            .is_some_and(|k| keys.pressed(k))
    };
    let nth_pad = |a: Action, n: usize| controls::gamepad_buttons_for(a).nth(n);
    let mut fi = FlightInput {
        wasd: if typing {
            Vec2::ZERO
        } else {
            Vec2::new(
                nth_key(Action::PlaneRudder, 1) as i32 as f32
                    - nth_key(Action::PlaneRudder, 0) as i32 as f32,
                nth_key(Action::PlaneThrottle, 0) as i32 as f32
                    - nth_key(Action::PlaneThrottle, 1) as i32 as f32,
            )
        },
        match_vel: nth_key(Action::MatchVelocity, 0),
        ..default()
    };
    if grabbed {
        let d = mouse_motion.delta;
        fi.mouse = Vec2::new(d.x, d.y) * FLIGHT_MOUSE_SENS;
    }
    for gp in gamepads.iter() {
        let l = gp.left_stick();
        if l.length() > PAD_STICK_DEADZONE {
            fi.left += l;
        }
        let r = gp.right_stick();
        if r.length() > PAD_STICK_DEADZONE {
            fi.right += r;
        }
        if let Some(b) = nth_pad(Action::PlaneThrottle, 0) {
            fi.rt += gp.get(b).unwrap_or(0.0);
        }
        if let Some(b) = nth_pad(Action::PlaneThrottle, 1) {
            fi.lt += gp.get(b).unwrap_or(0.0);
        }
        fi.lb |= nth_pad(Action::PlaneRudder, 0).is_some_and(|b| gp.pressed(b));
        fi.rb |= nth_pad(Action::PlaneRudder, 1).is_some_and(|b| gp.pressed(b));
        fi.match_vel |=
            !reviewing && nth_pad(Action::MatchVelocity, 0).is_some_and(|b| gp.pressed(b));
    }
    *flight = fi;
}

/// Quit is a chord (rl#330) — its code is ≥2 taps longer than every other (tested),
/// the chord system's replacement for the old timed pad hold: no stray tap after
/// another code can end the round, on both devices through the one path.
pub(super) fn quit_game(
    chords: Res<crab_world::chord::Chords<controls::GcrControls>>,
    mut exit: MessageWriter<AppExit>,
) {
    if chords.executed(Action::Quit) {
        exit.write(AppExit::Success);
    }
}

pub(super) fn grab_cursor(mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    if let Ok(mut c) = cursor.single_mut()
        && c.grab_mode != CursorGrabMode::Locked
    {
        c.grab_mode = CursorGrabMode::Locked;
        c.visible = false;
    }
}

pub(super) fn release_cursor(mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    if let Ok(mut c) = cursor.single_mut() {
        c.grab_mode = CursorGrabMode::None;
        c.visible = true;
    }
}
