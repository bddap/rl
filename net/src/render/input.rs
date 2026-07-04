//! Input gathering: WASD + mouse + gamepad -> [`PendingInput`] (+ client camera state).
//!
//! Samples local controls each render frame and integrates the client-side camera
//! pitch/yaw; produces only data destined for the next [`crate::sim::Input`] and never
//! touches the sim. The analog pad transform ([`pad_stick_axes`]) is split out so the
//! determinism test drives the exact client arithmetic.

use super::driver::{FlightInput, PendingInput};
use super::*;

/// One gamepad's contribution to this frame's control deltas: move axes from the left
/// stick, look deltas from the right. Raw f32 — like the keyboard/mouse contributions,
/// it crosses into the sim only after [`Input::new`] quantizes it (see [`gather_input`]).
pub(super) struct PadAxes {
    pub(super) strafe: f32,
    pub(super) forward: f32,
    /// Yaw-look this frame (radians), already scaled by [`PAD_LOOK_SPEED`] and dt.
    pub(super) d_yaw: f32,
    /// Pitch-look this frame (radians); client-only camera, never reaches the sim.
    pub(super) d_pitch: f32,
}

/// Map a gamepad's two sticks to this frame's move + look deltas. The pure analog core
/// of the pad branch, factored out of [`gather_input`] so the determinism test can drive
/// the SAME arithmetic the client runs (no copy to drift) — the sub-deadzone-clears-to-0,
/// the look = stick·speed·dt scaling. Buttons aren't here: they're plain bool reads with
/// no quantization concern, so they stay inline at the call site. Frame-local and f32;
/// the result is quantized downstream by [`Input::new`], so it never enters the sim raw.
pub(super) fn pad_stick_axes(left_stick: Vec2, right_stick: Vec2, dt: f32) -> PadAxes {
    // Deadzone on each stick's MAGNITUDE (not per-axis), so a resting stick's hardware
    // noise reads as exactly zero rather than creeping the avatar/view.
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

/// Client-side camera pitch (radians), integrated from look input. The sim models
/// only yaw (feet); pitch lives here and never reaches the sim, per the interface.
#[derive(Resource, Default)]
pub(super) struct CameraPitch(pub(super) f32);

/// Client-side camera YAW (radians), integrated from the same look input every frame.
/// While the local player is Alive the camera uses the AUTHORITATIVE sim yaw (so it
/// agrees with the avatar and peers) and this tracks it; once the player is downed or
/// extracted the sim freezes their yaw, so the camera falls back to this free value —
/// giving a spectator full free-look (yaw AND pitch), not pitch-only. Never reaches the
/// sim, so it adds no nondeterminism (a dead player's facing affects nothing).
#[derive(Resource, Default)]
pub(super) struct CameraYaw(pub(super) f32);

// ---------------------------------------------------------------------------
// Input: WASD + mouse + gamepad → PendingInput
// ---------------------------------------------------------------------------

/// Sample local controls each render frame into [`PendingInput`] and integrate the
/// client-side camera pitch. Produces ONLY data destined for the next [`Input`] — it
/// never touches the sim. The game is fully playable on keyboard+mouse OR a gamepad
/// alone, the two additive:
/// - move: WASD / left stick / d-pad
/// - look: mouse / right stick (yaw → sim, pitch → client-only)
/// - action (extract): Space / mouse-left / pad South / pad RT
/// - restart: R / pad Start (edge-triggered → [`buttons::RESTART`], lockstep)
/// - quit: Esc / hold pad North (handled in [`quit_game`])
///
/// Analog stick magnitudes are raw f32 here, but the ONLY path from this function to
/// the sim is via [`Input::new`] in [`drive_lockstep`], which quantizes every axis to
/// the fixed-point grid — the identical boundary keyboard/mouse cross. So no f32 ever
/// reaches the deterministic sim; the i16 [`Input`] each client ships up to the server
/// is the truth, and a pad input is bit-for-bit a keyboard input of the same magnitude.
#[allow(clippy::too_many_arguments)]
pub(super) fn gather_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    cursor: Query<&CursorOptions, With<PrimaryWindow>>,
    mut pending: ResMut<PendingInput>,
    mut flight: ResMut<FlightInput>,
    mut pitch: ResMut<CameraPitch>,
    mut yaw: ResMut<CameraYaw>,
) {
    let dt = time.delta_secs();

    // Every DISCRETE key/button below is looked up in the one binding table
    // (`controls::BINDINGS`) via these helpers, never hardcoded — so the keys the
    // client polls are exactly the keys the on-screen legend shows (no drift).
    // `kc(action)` is the keyboard key; `pad_pressed`/`pad_just_pressed` fold the pad's
    // primary+alternate buttons. The ANALOG channels (stick→axis math, mouse motion,
    // D-pad digital move) aren't rebindable bindings, so they stay inline here.
    let kc = controls::key_code_for;
    let held = |a| kc(a).is_some_and(|k| keys.pressed(k));

    // --- Move axes (last sample wins; re-sampled every frame) ---
    let mut strafe = 0.0f32;
    let mut forward = 0.0f32;
    forward += held(Action::MoveForward) as i32 as f32;
    forward -= held(Action::MoveBack) as i32 as f32;
    strafe += held(Action::StrafeRight) as i32 as f32;
    strafe -= held(Action::StrafeLeft) as i32 as f32;

    let mut action = held(Action::Extract);
    // Restart the round (R). Latched here, sent as buttons::RESTART, edge-triggered in
    // the sim — the server applies it and every client adopts the reset via the snapshot,
    // not a local-only reset (see `buttons::RESTART`).
    if kc(Action::Restart).is_some_and(|k| keys.just_pressed(k)) {
        pending.restart = true;
    }
    // Enter/exit a vehicle (E). A tap-toggle, edge-triggered like restart, but client-local —
    // `drive_lockstep` boards/steps-out on the server-authoritative arm (solo or host; see
    // `PeerRole::can_pilot`); it never reaches the sim.
    if kc(Action::EnterExit).is_some_and(|k| keys.just_pressed(k)) {
        pending.toggle_vehicle = true;
    }

    // --- Look (accumulated across frames) ---
    let mut d_yaw = 0.0f32;
    let mut d_pitch = 0.0f32;

    // Mouse look only when the cursor is grabbed (windowed play). In headless
    // screenshot mode there's no window/cursor, so this is simply skipped.
    let grabbed = cursor
        .iter()
        .next()
        .is_some_and(|c| c.grab_mode != CursorGrabMode::None);
    if grabbed {
        let d = mouse_motion.delta;
        d_yaw += d.x * MOUSE_SENS;
        d_pitch -= d.y * MOUSE_SENS;
    }

    // Gamepad (full pad-only play): left stick moves, right stick looks, South/RT =
    // extract, Start (tap) = restart. Quit (hold North) and reveal-controls (hold Back)
    // live in `quit_game` / the overlay system. Sticks have a deadzone so a resting stick
    // doesn't creep. Stick magnitudes are raw f32 here but cross into the sim ONLY through
    // `Input::new`'s fixed-point quantization (below) — the SAME boundary keyboard/mouse
    // pass — so the analog values never reach the deterministic sim.
    for gp in gamepads.iter() {
        // The analog stick → axis arithmetic (deadzone + look scaling) lives in the pure
        // `pad_stick_axes` so the determinism test can exercise the SAME transform the
        // client runs, with no copy to drift out of sync.
        let pad = pad_stick_axes(gp.left_stick(), gp.right_stick(), dt);
        strafe += pad.strafe;
        forward += pad.forward;
        d_yaw += pad.d_yaw;
        d_pitch += pad.d_pitch;
        // D-pad mirrors WASD as a digital move (kids reach for it instinctively, and it's
        // the obvious second way to walk): full ±1.0 on each axis, summed with the stick
        // and clamped at the funnel below — the SAME path WASD takes, so it quantizes
        // identically. Up=forward, Down=back, Right/Left=strafe (pre-negation downstream).
        forward += (gp.pressed(GamepadButton::DPadUp) as i32
            - gp.pressed(GamepadButton::DPadDown) as i32) as f32;
        strafe += (gp.pressed(GamepadButton::DPadRight) as i32
            - gp.pressed(GamepadButton::DPadLeft) as i32) as f32;
        action |= controls::gamepad_buttons_for(Action::Extract).any(|b| gp.pressed(b));
        // Restart on Start (tap), edge-triggered exactly like keyboard R: latched here,
        // sent as buttons::RESTART — the server applies it, clients adopt the reset via
        // the snapshot (see `buttons::RESTART`). Edge (just_pressed), not held. Quit is on its OWN
        // pad button (North, held), NOT Start — so beginning a quit can't fire this restart.
        if controls::gamepad_buttons_for(Action::Restart).any(|b| gp.just_pressed(b)) {
            pending.restart = true;
        }
        // Enter/exit vehicle on pad West (X), tap — same client-local toggle as keyboard E.
        if controls::gamepad_buttons_for(Action::EnterExit).any(|b| gp.just_pressed(b)) {
            pending.toggle_vehicle = true;
        }
    }
    // Mouse-left also fires action, for mouse-only play.
    if let Some(mb) = controls::MouseInput::Left.mouse_button() {
        action |= mouse_buttons.pressed(mb);
    }

    // Reconcile screen-right with the sim's X axis. The sim labels +X "strafe right"
    // and increasing yaw turns +Z toward +X — but a camera looking along +Z (yaw 0)
    // has its right axis at world −X, so world +X renders on the SCREEN-LEFT. Feeding
    // the player's "right" intents straight through would move the avatar and pan the
    // view the wrong way. Negating the two X-axis control intents (strafe and the
    // yaw-look delta) here — and only here — makes D / mouse-right / right-stick read
    // as screen-right, while the sim frame and the faithful world rendering stay
    // untouched (forward and pitch carry no X, so they're unaffected).
    pending.strafe = (-strafe).clamp(-1.0, 1.0);
    pending.forward = forward.clamp(-1.0, 1.0);
    pending.yaw_delta -= d_yaw;
    pending.action |= action;

    pitch.0 = (pitch.0 + d_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    // Integrate the client-side free-look yaw from the SAME (screen-corrected) delta the
    // sim yaw gets, so while alive it tracks the avatar and when dead it free-looks
    // seamlessly from the last facing. Wrap to keep it bounded over a long spectate.
    yaw.0 = (yaw.0 - d_yaw).rem_euclid(std::f32::consts::TAU);

    // --- Flight inputs (client-local vehicle), sampled fresh each frame and mapped per craft by
    // `flight_control` (plane = Ace Combat 6, ship = Outer Wilds). Read THROUGH the bindings — the
    // keys via `key_codes_for`, the pad buttons via `gamepad_buttons_for` — so the keys/buttons the
    // legend shows are exactly the ones polled, the same no-drift round-trip the foot controls get.
    // The ONLY raw reads are the two analog STICKS (Bevy's `left_stick`/`right_stick` API has no
    // discrete-button equivalent) — the documented analog exemption; their glyphs still come from the
    // one binding table. Reading these for the vehicle (not the sim's merged move/look) lets a craft
    // map a stick to a different DOF than the foot avatar. Never reaches the deterministic sim.
    //
    // The N-th bound key/button of an action; the ORDER is the canonical scheme order (e.g.
    // PlaneThrottle = [RT, LT], PlaneRudder = [LB, RB], pinned by a controls test), which the
    // direction signs below rely on.
    let nth_key = |a: Action, n: usize| {
        controls::key_codes_for(a)
            .nth(n)
            .is_some_and(|k| keys.pressed(k))
    };
    let nth_pad = |a: Action, n: usize| controls::gamepad_buttons_for(a).nth(n);
    let mut fi = FlightInput {
        // Keyboard flight axes through the FLIGHT bindings: throttle/forward = PlaneThrottle [W↑, S↓],
        // strafe/rudder = PlaneRudder [A←, D→] (the ship reuses the same physical keys via ShipThrust).
        wasd: Vec2::new(
            nth_key(Action::PlaneRudder, 1) as i32 as f32
                - nth_key(Action::PlaneRudder, 0) as i32 as f32,
            nth_key(Action::PlaneThrottle, 0) as i32 as f32
                - nth_key(Action::PlaneThrottle, 1) as i32 as f32,
        ),
        match_vel: nth_key(Action::MatchVelocity, 0),
        ..default()
    };
    if grabbed {
        let d = mouse_motion.delta;
        fi.mouse = Vec2::new(d.x, d.y) * FLIGHT_MOUSE_SENS;
    }
    for gp in gamepads.iter() {
        // Sticks: the analog exemption (Bevy's stick API), deadzoned on magnitude so a resting stick
        // reads zero. PlaneAttitude/ShipThrust bind LeftStick, ShipAim binds RightStick.
        let l = gp.left_stick();
        if l.length() > PAD_STICK_DEADZONE {
            fi.left += l;
        }
        let r = gp.right_stick();
        if r.length() > PAD_STICK_DEADZONE {
            fi.right += r;
        }
        // Triggers: the ANALOG value of the bound buttons (PlaneThrottle = [RT, LT]); bumpers + the
        // match-velocity face button: the bound buttons pressed (PlaneRudder = [LB, RB], MatchVelocity
        // = [South]). All via the bindings, so a rebind moves both the legend and the poll together.
        if let Some(b) = nth_pad(Action::PlaneThrottle, 0) {
            fi.rt += gp.get(b).unwrap_or(0.0);
        }
        if let Some(b) = nth_pad(Action::PlaneThrottle, 1) {
            fi.lt += gp.get(b).unwrap_or(0.0);
        }
        fi.lb |= nth_pad(Action::PlaneRudder, 0).is_some_and(|b| gp.pressed(b));
        fi.rb |= nth_pad(Action::PlaneRudder, 1).is_some_and(|b| gp.pressed(b));
        fi.match_vel |= nth_pad(Action::MatchVelocity, 0).is_some_and(|b| gp.pressed(b));
    }
    *flight = fi;
}

/// Quit the game (windowed play only): the keyboard Quit key (Esc), or HOLD the gamepad
/// Quit button (North/Y) for [`PAD_QUIT_HOLD_SECS`]. Both bindings come from the one
/// binding table ([`controls::BINDINGS`]), so this matches the legend. Purely client-local — sends
/// Bevy's [`AppExit`]; no sim/lockstep involvement, so it can't desync a peer (each client
/// just closes its own window) and the others play on. The pad Quit is a HOLD on its OWN
/// button (not Start, which restarts), so a stray press can't end the round for the couch.
pub(super) fn quit_game(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    mut quit_held: Local<f32>,
    mut exit: MessageWriter<AppExit>,
) {
    if controls::key_code_for(Action::Quit).is_some_and(|k| keys.just_pressed(k)) {
        exit.write(AppExit::Success);
        return;
    }
    // Accumulate held time while the pad Quit button is down on ANY pad; reset the instant
    // it's released, so only a sustained hold (not repeated taps) reaches the threshold.
    if pad_action_held(&gamepads, Action::Quit) {
        *quit_held += time.delta_secs();
        if *quit_held >= PAD_QUIT_HOLD_SECS {
            exit.write(AppExit::Success);
        }
    } else {
        *quit_held = 0.0;
    }
}

/// Grab + hide the cursor while a round runs, so mouse-look works and the pointer stays
/// captured. Level-triggered (grab whenever not yet locked), NOT a one-shot latch, so a round
/// entered AGAIN after a disconnect return (rl#203) re-grabs; the steady state is a read, so
/// no per-frame change-detection churn. (Grabbing AFTER the window is live, rather than via
/// the plugin's initial options, avoids a too-early lock failing on some platforms.)
pub(super) fn grab_cursor(mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    if let Ok(mut c) = cursor.single_mut()
        && c.grab_mode != CursorGrabMode::Locked
    {
        c.grab_mode = CursorGrabMode::Locked;
        c.visible = false;
    }
}

/// Release + show the cursor as the round ends — the OnExit(Playing) mirror of
/// [`grab_cursor`]. Without it the disconnect return (rl#203) lands on the menu's
/// "connection lost — rejoin?" prompt with a locked, invisible pointer, unclickable by mouse.
pub(super) fn release_cursor(mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>) {
    if let Ok(mut c) = cursor.single_mut() {
        c.grab_mode = CursorGrabMode::None;
        c.visible = true;
    }
}

/// Whether any connected gamepad currently HOLDS a button bound to `action`. The shared
/// read for the discrete pad buttons — folds the map's mapped buttons (via
/// [`controls::gamepad_buttons_for`]) across every pad. Factored out so `gather_input` and
/// `quit_game` don't each re-spell the nested any-any loop. (The overlay's own reveal-held
/// read lives in [`crab_world::controls`].)
fn pad_action_held(gamepads: &Query<&Gamepad>, action: Action) -> bool {
    gamepads
        .iter()
        .any(|gp| controls::gamepad_buttons_for(action).any(|b| gp.pressed(b)))
}
