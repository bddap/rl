//! Giant Crab Rescue's control scheme: GCR's action set + input vocabulary + glyph art + the
//! one [`BINDINGS`] table + the per-context row lists ([`FOOT_ROWS`]/[`PLANE_ROWS`]/[`SHIP_ROWS`]),
//! implemented as a [`ControlScheme`] for the reusable overlay framework in
//! [`crab_world::controls`]. The framework renders the context-sensitive legend + hold-to-reveal
//! overlay from this; the live input ([`crate::render`]'s `gather_input`/`quit_game`)
//! reads the SAME [`BINDINGS`] via [`key_code_for`]/[`gamepad_buttons_for`], so the keys the
//! client polls are exactly the keys the legend shows — no drift.
//!
//! ## One source of truth, two flight feels (botq#554)
//! Each [`Action`] binds one key/button (context-independent), and each context names the actions
//! it shows with the label correct THERE — the legend joins rows with bindings, so the displayed
//! GLYPH can't drift from the polled input. On foot the move keys walk the avatar; in the air the
//! two craft have DIFFERENT control schemes, so each gets its own flight actions + row list:
//! - **Plane = Ace Combat 6**: LEFT stick (or mouse) flies — pitch (AC6 flight-sim: pull back = nose
//!   up) + roll; RT/LT throttle/brake; LB/RB rudder. (Right stick is the camera — a free-look seam, not
//!   yet wired.) [`PlaneAttitude`](Action::PlaneAttitude)/[`PlaneThrottle`](Action::PlaneThrottle)/
//!   [`PlaneRudder`](Action::PlaneRudder).
//! - **Ship = Outer Wilds**: LEFT stick (or WASD) translates (forward/back + strafe); RIGHT stick
//!   (or mouse) AIMS (pitch + yaw); RT/LT thrust up/down; LB/RB roll; A (or Space) matches velocity
//!   (brake). 6-DOF thrust-and-coast.
//!
//! The flight actions are read for the CLIENT-LOCAL vehicle in `render`'s flight-input snapshot,
//! not through the sim's merged move/look axes — so a craft can map the same stick to a different
//! degree of freedom than the foot avatar does without the sim's axis-merge fighting it. Those reads
//! go THROUGH the bindings too: the keys via [`key_codes_for`], the pad buttons (triggers/bumpers/
//! face) via [`gamepad_buttons_for`] — so the legend's glyph and the polled input stay one source.
//! The label's MEANING (that "Throttle / brake" really is what RT/LT do on the plane) is a
//! hand-maintained description of the force model ([`crab_world::vehicle`]) + `drive_lockstep`'s
//! bridge, pinned by tests (see the plane-pitch-sign test in `vehicle`).
//!
//! Two caveats on the single-source guarantee:
//! - The ONE input read raw (not via a binding) is the two analog STICKS — Bevy's
//!   `left_stick`/`right_stick` API has no discrete-button equivalent to route through a binding. Their
//!   GLYPHS (LeftStick/RightStick) still come from the one table; only the magnitude is read directly.
//!   Everything else — keys, the analog triggers, the bumpers, the face buttons — polls through the
//!   bindings, so a rebind moves the legend and the poll together.
//! - Nothing here touches the deterministic sim. A binding only decides WHICH input feeds an
//!   [`Action`]; the foot value still funnels through `Input::new`'s fixed-point quantization
//!   downstream, and the vehicle is client-local crab-world state off the wire — so rebinding never
//!   changes a wire value and peers stay bit-identical (the determinism contract atop [`crate::sim`]).

use crab_world::controls::{Binding, ContextRow, ControlScheme, Glyph, KbBinding, PadBinding};

/// GCR's control scheme — the type parameter the framework is instantiated with. A
/// zero-size marker; all of GCR's control data hangs off its [`ControlScheme`] impl.
pub struct GcrControls;

/// The active control context in Giant Crab Rescue. On foot the move keys walk the avatar; in the
/// air each craft has its OWN control scheme (the plane flies AC6, the ship flies Outer Wilds), so
/// each is its own context with its own row list — the HUD names and labels the active one
/// automatically. `OnFoot` is the default (spawn) context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GcrContext {
    #[default]
    OnFoot,
    /// Flying the client-local plane (see [`crate::render`]'s `LocalVehicle`; off-wire, so solo or
    /// host — see `PeerRole::can_pilot`). Ace Combat 6 controls.
    Plane,
    /// Flying the client-local Outer-Wilds-style ship (the other [`LocalVehicle`] mode). The
    /// E/X enter-vehicle control CYCLES foot → plane → ship → foot.
    Ship,
}

/// A controllable action in Giant Crab Rescue. This enum is the row key of [`BINDINGS`]; the
/// per-app test forces every variant to have exactly one binding. Foot move axes are split into
/// four discrete directional actions (so WASD shows as four glyphs); the flight actions are
/// per-craft (the plane and ship assign the sticks/triggers to different degrees of freedom).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    MoveForward,
    MoveBack,
    StrafeLeft,
    StrafeRight,
    /// Aim the camera on foot (yaw → sim, pitch → client). Analog; no discrete key binding.
    Look,
    /// At the extraction pillar, confirm the pickup (the sim's `buttons::ACTION`).
    Extract,
    /// Restart the round for all peers (the sim's `buttons::RESTART`, edge-triggered).
    Restart,
    /// Quit the client (local `AppExit`; never touches the sim).
    Quit,
    /// CYCLE the client-local vehicle: foot → plane → ship → foot. A tap handled entirely in the
    /// windowed client's play layer ([`crate::render`]) — like [`Quit`](Action::Quit) it never
    /// crosses the wire or the deterministic sim, so the lockstep crab game is unaffected.
    EnterExit,
    /// CYCLE the crab render view: mesh → mesh+colliders → colliders. Pure client UI.
    CycleRenderMode,
    /// Hold to reveal the full control overlay; release to hide. Pure client UI.
    RevealControls,

    // --- Plane (Ace Combat 6) flight actions ---
    /// Plane attitude stick: pitch (AC6 flight-sim: pull back = nose up) + roll. Left stick / mouse.
    PlaneAttitude,
    /// Plane throttle: RT accelerate (afterburner feel), LT brake. RT+LT / W,S.
    PlaneThrottle,
    /// Plane rudder yaw: LB left, RB right. Bumpers / A,D.
    PlaneRudder,

    // --- Ship (Outer Wilds) flight actions ---
    /// Ship translational thrusters: forward/back + lateral strafe. Left stick / WASD.
    ShipThrust,
    /// Ship aim: pitch + yaw (rotate the ship). Right stick / mouse.
    ShipAim,
    /// Ship vertical thrusters: RT up, LT down. RT+LT.
    ShipLift,
    /// Ship roll: LB/RB. Bumpers.
    ShipRoll,
    /// Ship match-velocity brake (Outer Wilds): bleed relative velocity to rest. A / Space.
    MatchVelocity,
}

/// A keyboard key GCR binds. A closed set (only the keys GCR uses), so the bindings and glyph
/// table are exhaustive and a typo can't name a nonexistent key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    W,
    A,
    S,
    D,
    E,
    R,
    V,
    Tab,
    Space,
    Escape,
}

/// A mouse input GCR binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseInput {
    /// Pointer motion → look.
    Motion,
    /// Left button → extract (mouse-only play).
    Left,
}

/// A gamepad control GCR binds (Xbox-style names — the generic glyph set the MVP ships).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadButton {
    /// The "south" face button (Xbox A / PS Cross).
    South,
    /// The "north" face button (Xbox Y / PS Triangle).
    North,
    /// The "west" face button (Xbox X / PS Square).
    West,
    /// The "east" face button (Xbox B / PS Circle).
    East,
    /// Right analog trigger (RT) — alternate extract on foot; throttle/lift while flying.
    RightTrigger,
    /// Left analog trigger (LT) — brake (plane) / down-thrust (ship).
    LeftTrigger,
    /// Left bumper / shoulder (LB) — rudder-left (plane) / roll (ship).
    LeftBumper,
    /// Right bumper / shoulder (RB) — rudder-right (plane) / roll (ship).
    RightBumper,
    /// Start / Menu.
    Start,
    /// Back / Select / View.
    Back,
    LeftStick,
    RightStick,
    Dpad,
}

impl ControlScheme for GcrControls {
    type Action = Action;
    type Key = Key;
    type Pad = PadButton;
    type Mouse = MouseInput;
    type Context = GcrContext;

    fn bindings() -> &'static [Binding<Self>] {
        &BINDINGS
    }

    fn contexts() -> &'static [GcrContext] {
        &[GcrContext::OnFoot, GcrContext::Plane, GcrContext::Ship]
    }

    fn context_rows(ctx: GcrContext) -> &'static [ContextRow<Self>] {
        match ctx {
            GcrContext::OnFoot => &FOOT_ROWS,
            GcrContext::Plane => &PLANE_ROWS,
            GcrContext::Ship => &SHIP_ROWS,
        }
    }

    fn context_label(ctx: GcrContext) -> &'static str {
        match ctx {
            GcrContext::OnFoot => "On foot",
            GcrContext::Plane => "Piloting plane",
            GcrContext::Ship => "Piloting ship",
        }
    }

    fn context_id(ctx: GcrContext) -> &'static str {
        match ctx {
            GcrContext::OnFoot => "foot",
            GcrContext::Plane => "plane",
            GcrContext::Ship => "ship",
        }
    }

    fn context_from_id(id: &str) -> Option<GcrContext> {
        match id {
            "foot" => Some(GcrContext::OnFoot),
            "plane" => Some(GcrContext::Plane),
            "ship" => Some(GcrContext::Ship),
            _ => None,
        }
    }

    fn reveal_action() -> Action {
        Action::RevealControls
    }

    /// The asset path (under `controls/`) of the glyph for a key — the bundled Kenney Input
    /// Prompts (CC0) PNGs. One arm per [`Key`], so a binding can't reference a missing image.
    fn key_glyph(key: Key) -> Glyph {
        Glyph::Icon(match key {
            Key::W => "controls/keyboard_w.png",
            Key::A => "controls/keyboard_a.png",
            Key::S => "controls/keyboard_s.png",
            Key::D => "controls/keyboard_d.png",
            Key::E => "controls/keyboard_e.png",
            Key::R => "controls/keyboard_r.png",
            Key::V => "controls/keyboard_v.png",
            Key::Tab => "controls/keyboard_tab.png",
            Key::Space => "controls/keyboard_space.png",
            Key::Escape => "controls/keyboard_escape.png",
        })
    }

    fn pad_glyph(pad: PadButton) -> Glyph {
        match pad {
            PadButton::South => Glyph::Icon("controls/xbox_button_a.png"),
            PadButton::North => Glyph::Icon("controls/xbox_button_y.png"),
            PadButton::West => Glyph::Icon("controls/xbox_button_x.png"),
            PadButton::East => Glyph::Icon("controls/xbox_button_b.png"),
            PadButton::RightTrigger => Glyph::Icon("controls/xbox_rt.png"),
            // No bundled art for LT / LB / RB — the framework's text-chip glyph renders them (the
            // sanctioned art-less path), so the legend is correct before those PNGs ship.
            PadButton::LeftTrigger => Glyph::Label("LT"),
            PadButton::LeftBumper => Glyph::Label("LB"),
            PadButton::RightBumper => Glyph::Label("RB"),
            PadButton::Start => Glyph::Icon("controls/xbox_button_menu.png"),
            PadButton::Back => Glyph::Icon("controls/xbox_button_view.png"),
            PadButton::LeftStick => Glyph::Icon("controls/xbox_stick_l.png"),
            PadButton::RightStick => Glyph::Icon("controls/xbox_stick_r.png"),
            PadButton::Dpad => Glyph::Icon("controls/xbox_dpad.png"),
        }
    }

    fn mouse_glyph(mouse: MouseInput) -> Glyph {
        Glyph::Icon(match mouse {
            MouseInput::Motion => "controls/mouse_move.png",
            MouseInput::Left => "controls/mouse_left.png",
        })
    }
}

/// THE GCR binding table — one row per action, the key/button that triggers it (no labels;
/// those are per-context, see the `*_ROWS` lists). The deliberate choices:
/// - **Move (foot)**: WASD / left stick (+ D-pad shown once, on Forward), the FPS convention.
/// - **Look (foot)**: mouse / right stick.
/// - **Extract**: Space or left-click / pad South (A) or right trigger.
/// - **Restart**: R / pad Start (tap) — edge-triggered, rides the lockstep input stream.
/// - **Quit**: Esc / HOLD pad North (Y) ≥1s. A hold so a stray tap can't end the round; on
///   its OWN button (not Start) so quitting can't also fire the lockstep RESTART.
/// - **Enter/exit vehicle**: E / pad West (X) — tap to board / cycle / step out.
/// - **Reveal controls**: HOLD Tab / HOLD pad Back.
/// - **Flight (plane/ship)**: the sticks/triggers/bumpers, bound once here; each craft's context
///   relabels the ones it uses (a shared button like RT serves Extract on foot and throttle in the
///   air — the contexts are mutually exclusive, so the polled meaning is unambiguous).
pub const BINDINGS: [Binding<GcrControls>; 19] = [
    Binding {
        action: Action::MoveForward,
        keyboard: KbBinding::new(&[Key::W], &[]),
        // The D-pad is the alternate digital mover; surface its glyph once (here) rather
        // than on all four move rows.
        pad: PadBinding::new(&[PadButton::LeftStick, PadButton::Dpad]),
    },
    Binding {
        action: Action::MoveBack,
        keyboard: KbBinding::new(&[Key::S], &[]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    Binding {
        action: Action::StrafeLeft,
        keyboard: KbBinding::new(&[Key::A], &[]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    Binding {
        action: Action::StrafeRight,
        keyboard: KbBinding::new(&[Key::D], &[]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    Binding {
        action: Action::Look,
        keyboard: KbBinding::new(&[], &[MouseInput::Motion]),
        pad: PadBinding::new(&[PadButton::RightStick]),
    },
    Binding {
        action: Action::Extract,
        keyboard: KbBinding::new(&[Key::Space], &[MouseInput::Left]),
        pad: PadBinding::new(&[PadButton::South, PadButton::RightTrigger]),
    },
    Binding {
        action: Action::Restart,
        keyboard: KbBinding::new(&[Key::R], &[]),
        pad: PadBinding::new(&[PadButton::Start]),
    },
    Binding {
        action: Action::Quit,
        keyboard: KbBinding::hold(&[Key::Escape], &[]),
        pad: PadBinding::hold(&[PadButton::North]),
    },
    Binding {
        action: Action::EnterExit,
        keyboard: KbBinding::new(&[Key::E], &[]),
        pad: PadBinding::new(&[PadButton::West]),
    },
    Binding {
        action: Action::CycleRenderMode,
        keyboard: KbBinding::new(&[Key::V], &[]),
        pad: PadBinding::new(&[PadButton::East]),
    },
    Binding {
        action: Action::RevealControls,
        keyboard: KbBinding::hold(&[Key::Tab], &[]),
        pad: PadBinding::hold(&[PadButton::Back]),
    },
    // --- Plane (AC6) ---
    Binding {
        action: Action::PlaneAttitude,
        keyboard: KbBinding::new(&[], &[MouseInput::Motion]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    Binding {
        action: Action::PlaneThrottle,
        keyboard: KbBinding::new(&[Key::W, Key::S], &[]),
        pad: PadBinding::new(&[PadButton::RightTrigger, PadButton::LeftTrigger]),
    },
    Binding {
        action: Action::PlaneRudder,
        keyboard: KbBinding::new(&[Key::A, Key::D], &[]),
        pad: PadBinding::new(&[PadButton::LeftBumper, PadButton::RightBumper]),
    },
    // --- Ship (Outer Wilds) ---
    Binding {
        action: Action::ShipThrust,
        keyboard: KbBinding::new(&[Key::W, Key::A, Key::S, Key::D], &[]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    Binding {
        action: Action::ShipAim,
        keyboard: KbBinding::new(&[], &[MouseInput::Motion]),
        pad: PadBinding::new(&[PadButton::RightStick]),
    },
    Binding {
        action: Action::ShipLift,
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[PadButton::RightTrigger, PadButton::LeftTrigger]),
    },
    Binding {
        action: Action::ShipRoll,
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[PadButton::LeftBumper, PadButton::RightBumper]),
    },
    Binding {
        action: Action::MatchVelocity,
        keyboard: KbBinding::new(&[Key::Space], &[]),
        pad: PadBinding::new(&[PadButton::South]),
    },
];

/// The ON-FOOT context: the full ground control set, in legend order. The move keys walk the
/// avatar; `EnterExit` boards the plane.
pub const FOOT_ROWS: [ContextRow<GcrControls>; 11] = [
    ContextRow {
        action: Action::MoveForward,
        label: "Forward",
    },
    ContextRow {
        action: Action::MoveBack,
        label: "Back",
    },
    ContextRow {
        action: Action::StrafeLeft,
        label: "Strafe left",
    },
    ContextRow {
        action: Action::StrafeRight,
        label: "Strafe right",
    },
    ContextRow {
        action: Action::Look,
        label: "Look",
    },
    ContextRow {
        action: Action::Extract,
        label: "Extract",
    },
    ContextRow {
        action: Action::EnterExit,
        label: "Enter plane",
    },
    ContextRow {
        action: Action::CycleRenderMode,
        label: "Render view",
    },
    ContextRow {
        action: Action::Restart,
        label: "Restart round",
    },
    ContextRow {
        action: Action::Quit,
        label: "Quit",
    },
    ContextRow {
        action: Action::RevealControls,
        label: "Controls",
    },
];

/// The PILOTING-PLANE context (Ace Combat 6 layout). Left stick (or mouse) flies: pitch is the AC6
/// flight-sim convention the owner asked for (pull the stick BACK to raise the nose) and L/R banks
/// into a turn (screen-reconciled so stick-right banks right). RT/LT are the throttle/airbrake;
/// the bumpers are the rudder. `Extract` is omitted — the foot player feeds the sim neutral input
/// while piloting, so the pickup is inert aloft. The right stick is the camera (a free-look seam, not
/// yet wired — so no row for it).
///
/// The labels describe what `drive_lockstep`'s bridge does with each input (see
/// [`crab_world::vehicle`]); the plane-pitch-sign test pins the pitch direction + sensitivity.
pub const PLANE_ROWS: [ContextRow<GcrControls>; 8] = [
    ContextRow {
        action: Action::PlaneAttitude,
        label: "Pitch / roll",
    },
    ContextRow {
        action: Action::PlaneThrottle,
        label: "Throttle / brake",
    },
    ContextRow {
        action: Action::PlaneRudder,
        label: "Rudder (yaw)",
    },
    // E cycles foot → plane → ship → foot, so from the plane it boards the ship.
    ContextRow {
        action: Action::EnterExit,
        label: "Switch to ship",
    },
    ContextRow {
        action: Action::CycleRenderMode,
        label: "Render view",
    },
    ContextRow {
        action: Action::Restart,
        label: "Restart round",
    },
    ContextRow {
        action: Action::Quit,
        label: "Quit",
    },
    ContextRow {
        action: Action::RevealControls,
        label: "Controls",
    },
];

/// The PILOTING-SHIP context (Outer Wilds). Newtonian 6-DOF thrust-and-coast: the left stick (or
/// WASD) fires the translational thrusters (forward/back + strafe), the right stick (or mouse) AIMS
/// the ship (pitch + yaw), RT/LT thrust up/down, the bumpers roll, and A (or Space) matches velocity
/// — a brake that bleeds the drift to rest. `Extract` is omitted (neutral sim input while piloting).
/// The E-cycle reaches foot from here, so `EnterExit` reads "Exit to foot".
pub const SHIP_ROWS: [ContextRow<GcrControls>; 10] = [
    ContextRow {
        action: Action::ShipThrust,
        label: "Thrust: move / strafe",
    },
    ContextRow {
        action: Action::ShipAim,
        label: "Aim: pitch / yaw",
    },
    ContextRow {
        action: Action::ShipLift,
        label: "Thrust up / down",
    },
    ContextRow {
        action: Action::ShipRoll,
        label: "Roll",
    },
    ContextRow {
        action: Action::MatchVelocity,
        label: "Match velocity (brake)",
    },
    ContextRow {
        action: Action::EnterExit,
        label: "Exit to foot",
    },
    ContextRow {
        action: Action::CycleRenderMode,
        label: "Render view",
    },
    ContextRow {
        action: Action::Restart,
        label: "Restart round",
    },
    ContextRow {
        action: Action::Quit,
        label: "Quit",
    },
    ContextRow {
        action: Action::RevealControls,
        label: "Controls",
    },
];

// ---------------------------------------------------------------------------
// Bevy glue — the ONLY place GCR's typed inputs meet Bevy's input API. Render-only:
// Bevy's KeyCode/GamepadButton/MouseButton exist only under the `render` feature. The live
// predicates read BINDINGS via these, so the client polls exactly the keys the legend
// shows — no drift.
// ---------------------------------------------------------------------------

#[cfg(feature = "render")]
mod bevy_glue {
    use super::*;
    use bevy::prelude::{GamepadButton, KeyCode, MouseButton};
    use crab_world::controls::{ControlInput, binding};

    impl Key {
        fn key_code(self) -> KeyCode {
            match self {
                Key::W => KeyCode::KeyW,
                Key::A => KeyCode::KeyA,
                Key::S => KeyCode::KeyS,
                Key::D => KeyCode::KeyD,
                Key::E => KeyCode::KeyE,
                Key::R => KeyCode::KeyR,
                Key::V => KeyCode::KeyV,
                Key::Tab => KeyCode::Tab,
                Key::Space => KeyCode::Space,
                Key::Escape => KeyCode::Escape,
            }
        }
    }

    impl MouseInput {
        /// The Bevy `MouseButton` for a button input. `Motion` is pointer movement, not a
        /// button, so it has none — hence the `Option`.
        pub fn mouse_button(self) -> Option<MouseButton> {
            match self {
                MouseInput::Left => Some(MouseButton::Left),
                MouseInput::Motion => None,
            }
        }
    }

    impl PadButton {
        fn gamepad_button(self) -> Option<GamepadButton> {
            match self {
                PadButton::South => Some(GamepadButton::South),
                PadButton::North => Some(GamepadButton::North),
                PadButton::West => Some(GamepadButton::West),
                PadButton::East => Some(GamepadButton::East),
                // Bevy names the analog triggers `*Trigger2` and the shoulder bumpers `*Trigger`.
                PadButton::RightTrigger => Some(GamepadButton::RightTrigger2),
                PadButton::LeftTrigger => Some(GamepadButton::LeftTrigger2),
                PadButton::LeftBumper => Some(GamepadButton::LeftTrigger),
                PadButton::RightBumper => Some(GamepadButton::RightTrigger),
                PadButton::Start => Some(GamepadButton::Start),
                PadButton::Back => Some(GamepadButton::Select),
                // Sticks and the D-pad are read via their own axis/multi-button APIs.
                PadButton::LeftStick | PadButton::RightStick | PadButton::Dpad => None,
            }
        }
    }

    impl ControlInput for GcrControls {
        fn key_code(key: Key) -> Option<KeyCode> {
            Some(key.key_code())
        }
        fn gamepad_button(pad: PadButton) -> Option<GamepadButton> {
            pad.gamepad_button()
        }
    }

    /// The `KeyCode` bound to an action on the keyboard, if any. Reads [`BINDINGS`], so the
    /// client polls the mapped key — change the table, the poll follows. Context-independent:
    /// the key for an action is the same wherever it's shown (only its label changes), so the
    /// dispatch needs no context. Polls the FIRST bound key only while the legend renders
    /// every bound key: callers that need every key (the multi-key flight actions) use
    /// [`key_codes_for`].
    pub fn key_code_for(action: Action) -> Option<KeyCode> {
        binding::<GcrControls>(action)
            .and_then(|b| b.keyboard.keys.first().copied())
            .map(Key::key_code)
    }

    /// Every `KeyCode` bound to an action (for the multi-key flight actions, e.g. throttle = W,S).
    /// Reads [`BINDINGS`] so the polled keys are exactly the legend's.
    pub fn key_codes_for(action: Action) -> impl Iterator<Item = KeyCode> {
        binding::<GcrControls>(action)
            .into_iter()
            .flat_map(|b| b.keyboard.keys.iter().copied())
            .map(Key::key_code)
    }

    /// The pad `GamepadButton`(s) that trigger an action (every mapped button), for the
    /// discrete buttons. Sticks/D-pad yield nothing (read via their dedicated APIs). An
    /// iterator, not a `Vec`: callers just `.any(...)` each frame, so no heap alloc.
    pub fn gamepad_buttons_for(action: Action) -> impl Iterator<Item = GamepadButton> {
        binding::<GcrControls>(action)
            .into_iter()
            .flat_map(|b| b.pad.buttons.iter().copied())
            .filter_map(PadButton::gamepad_button)
    }
}

#[cfg(feature = "render")]
pub use bevy_glue::{gamepad_buttons_for, key_code_for, key_codes_for};

#[cfg(test)]
mod tests {
    use super::*;
    use crab_world::controls::{
        Device, Glyph, assert_scheme_well_formed, binding, legend, reveal_glyph,
    };

    const ALL_ACTIONS: [Action; 19] = [
        Action::MoveForward,
        Action::MoveBack,
        Action::StrafeLeft,
        Action::StrafeRight,
        Action::Look,
        Action::Extract,
        Action::Restart,
        Action::Quit,
        Action::EnterExit,
        Action::CycleRenderMode,
        Action::RevealControls,
        Action::PlaneAttitude,
        Action::PlaneThrottle,
        Action::PlaneRudder,
        Action::ShipThrust,
        Action::ShipAim,
        Action::ShipLift,
        Action::ShipRoll,
        Action::MatchVelocity,
    ];

    const ALL_CONTEXTS: [GcrContext; 3] = [GcrContext::OnFoot, GcrContext::Plane, GcrContext::Ship];

    /// Every [`Action`] / [`GcrContext`] is exhaustively classified (so a new variant can't be
    /// added without declaring it here), and the framework proves each action has exactly one
    /// binding, each context shows only bound actions + the reveal control, and the reveal
    /// control resolves to a hint glyph.
    #[test]
    fn scheme_is_well_formed() {
        fn action_classified(a: Action) -> bool {
            match a {
                Action::MoveForward
                | Action::MoveBack
                | Action::StrafeLeft
                | Action::StrafeRight
                | Action::Look
                | Action::Extract
                | Action::Restart
                | Action::Quit
                | Action::EnterExit
                | Action::CycleRenderMode
                | Action::RevealControls
                | Action::PlaneAttitude
                | Action::PlaneThrottle
                | Action::PlaneRudder
                | Action::ShipThrust
                | Action::ShipAim
                | Action::ShipLift
                | Action::ShipRoll
                | Action::MatchVelocity => true,
            }
        }
        fn ctx_classified(c: GcrContext) -> bool {
            match c {
                GcrContext::OnFoot | GcrContext::Plane | GcrContext::Ship => true,
            }
        }
        assert!(ALL_ACTIONS.iter().copied().all(action_classified));
        assert!(ALL_CONTEXTS.iter().copied().all(ctx_classified));
        assert_eq!(GcrControls::contexts(), &ALL_CONTEXTS);
        assert_scheme_well_formed::<GcrControls>(&ALL_ACTIONS, &ALL_CONTEXTS);
    }

    /// The no-drift property: each context's legend is built by joining its rows with the SAME
    /// binding table the live input reads, so every legend line's glyphs equal the bound
    /// action's `glyphs(device)` — for BOTH contexts and devices.
    #[test]
    fn legend_glyphs_come_from_the_bindings() {
        for ctx in ALL_CONTEXTS {
            for device in [Device::KeyboardMouse, Device::Gamepad] {
                let lines = legend::<GcrControls>(ctx, device);
                let rows = GcrControls::context_rows(ctx);
                // A row whose action isn't bound on this device is dropped from the legend, so
                // the line count is "rows bound on this device", not the raw row count.
                for line in &lines {
                    let row = rows.iter().find(|r| r.label == line.label).unwrap();
                    let b = binding::<GcrControls>(row.action).unwrap();
                    assert_eq!(line.glyphs, b.glyphs(device));
                    assert!(!line.glyphs.is_empty(), "{:?} shows no glyph", row.action);
                }
            }
        }
    }

    /// The context fix: foot and plane are DIFFERENT legends — the air drops Extract and shows the
    /// AC6 flight controls — so entering the plane visibly changes the HUD and names the craft.
    #[test]
    fn plane_context_relabels_and_differs_from_foot() {
        let foot = legend::<GcrControls>(GcrContext::OnFoot, Device::Gamepad);
        let plane = legend::<GcrControls>(GcrContext::Plane, Device::Gamepad);
        let labels = |ls: &[crab_world::controls::LegendLine]| {
            ls.iter().map(|l| l.label).collect::<Vec<_>>()
        };
        assert_ne!(
            labels(&foot),
            labels(&plane),
            "the legend must change per context"
        );
        // AC6 flight labels.
        assert!(plane.iter().any(|l| l.label == "Throttle / brake"));
        assert!(plane.iter().any(|l| l.label == "Pitch / roll"));
        assert!(plane.iter().any(|l| l.label == "Rudder (yaw)"));
        assert!(plane.iter().any(|l| l.label == "Switch to ship"));
        // The on-foot ground labels are gone in flight.
        assert!(!plane.iter().any(|l| l.label == "Strafe left"));
        assert!(
            !plane.iter().any(|l| l.label == "Extract"),
            "no Extract while piloting"
        );
        assert_eq!(
            GcrControls::context_label(GcrContext::Plane),
            "Piloting plane"
        );
    }

    /// The ship context is its OWN legend — distinct from foot AND plane — with the Outer Wilds
    /// 6-DOF controls, naming the active craft, riding the one binding table (no parallel system).
    #[test]
    fn ship_context_relabels_and_differs_from_foot_and_plane() {
        let foot = legend::<GcrControls>(GcrContext::OnFoot, Device::Gamepad);
        let plane = legend::<GcrControls>(GcrContext::Plane, Device::Gamepad);
        let ship = legend::<GcrControls>(GcrContext::Ship, Device::Gamepad);
        let labels = |ls: &[crab_world::controls::LegendLine]| {
            ls.iter().map(|l| l.label).collect::<Vec<_>>()
        };
        assert_ne!(
            labels(&ship),
            labels(&foot),
            "ship legend differs from foot"
        );
        assert_ne!(
            labels(&ship),
            labels(&plane),
            "ship legend differs from plane"
        );
        // Outer Wilds labels.
        assert!(ship.iter().any(|l| l.label == "Thrust: move / strafe"));
        assert!(ship.iter().any(|l| l.label == "Aim: pitch / yaw"));
        assert!(ship.iter().any(|l| l.label == "Roll"));
        assert!(ship.iter().any(|l| l.label == "Match velocity (brake)"));
        assert!(ship.iter().any(|l| l.label == "Exit to foot"));
        // No misleading ground / plane labels in the ship.
        assert!(!ship.iter().any(|l| l.label == "Throttle / brake"));
        assert!(
            !ship.iter().any(|l| l.label == "Extract"),
            "no Extract while piloting"
        );
        assert_eq!(
            GcrControls::context_label(GcrContext::Ship),
            "Piloting ship"
        );
    }

    /// The plane's flight controls ride the SAME physical pad inputs as foot/ship (one binding
    /// table, relabeled), not a parallel control system: the plane attitude is the left stick, the
    /// same physical stick foot moves with and the ship thrusts with.
    #[test]
    fn flight_rides_the_one_binding_table() {
        // Plane attitude + ship thrust both poll the LEFT stick — the same physical input the foot
        // move axes use, proving the relabel rides one binding table.
        assert!(
            binding::<GcrControls>(Action::PlaneAttitude)
                .unwrap()
                .pad
                .buttons
                .contains(&PadButton::LeftStick)
        );
        assert!(
            binding::<GcrControls>(Action::ShipThrust)
                .unwrap()
                .pad
                .buttons
                .contains(&PadButton::LeftStick)
        );
        assert!(
            binding::<GcrControls>(Action::MoveForward)
                .unwrap()
                .pad
                .buttons
                .contains(&PadButton::LeftStick)
        );
        // The throttle pad inputs are the analog triggers (RT then LT).
        assert_eq!(
            binding::<GcrControls>(Action::PlaneThrottle)
                .unwrap()
                .pad
                .buttons,
            &[PadButton::RightTrigger, PadButton::LeftTrigger]
        );
    }

    /// The flight bindings' ORDER is the contract the input bridge's `nth()` reads + its direction
    /// signs depend on (`render::input` polls "the 0th bound button" = RT/LB, "the 1st" = LT/RB; the
    /// keys W↑/S↓, A←/D→). Reorder a list and a control silently inverts — so pin the order here.
    #[test]
    fn flight_binding_order_is_canonical() {
        let pad = |a| binding::<GcrControls>(a).unwrap().pad.buttons;
        let keys = |a| binding::<GcrControls>(a).unwrap().keyboard.keys;
        // Triggers: accelerate/up FIRST (RT), brake/down SECOND (LT).
        assert_eq!(
            pad(Action::PlaneThrottle),
            &[PadButton::RightTrigger, PadButton::LeftTrigger]
        );
        assert_eq!(
            pad(Action::ShipLift),
            &[PadButton::RightTrigger, PadButton::LeftTrigger]
        );
        // Bumpers: left FIRST (LB), right SECOND (RB).
        assert_eq!(
            pad(Action::PlaneRudder),
            &[PadButton::LeftBumper, PadButton::RightBumper]
        );
        assert_eq!(
            pad(Action::ShipRoll),
            &[PadButton::LeftBumper, PadButton::RightBumper]
        );
        assert_eq!(pad(Action::MatchVelocity), &[PadButton::South]);
        // Keyboard: throttle/forward = W↑ then S↓; rudder/strafe = A← then D→.
        assert_eq!(keys(Action::PlaneThrottle), &[Key::W, Key::S]);
        assert_eq!(keys(Action::PlaneRudder), &[Key::A, Key::D]);
        assert_eq!(keys(Action::MatchVelocity), &[Key::Space]);
    }

    /// The screenshot context override round-trips the ids the evidence harness uses.
    #[test]
    fn context_from_id_round_trips() {
        assert_eq!(
            GcrControls::context_from_id("foot"),
            Some(GcrContext::OnFoot)
        );
        assert_eq!(
            GcrControls::context_from_id("plane"),
            Some(GcrContext::Plane)
        );
        assert_eq!(GcrControls::context_from_id("ship"), Some(GcrContext::Ship));
        assert_eq!(GcrControls::context_from_id("nope"), None);
    }

    /// Every ICON glyph the bindings surface is a bundled asset under `controls/`; the only text
    /// glyphs are the art-less trigger/bumper chips (LT/LB/RB), which the framework renders as
    /// keycaps. Catches a renamed/relocated PNG (which would 404 at runtime).
    #[test]
    fn glyphs_are_controls_icons_or_trigger_chips() {
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for b in &BINDINGS {
                for glyph in b.glyphs(device) {
                    match glyph {
                        Glyph::Icon(path) => assert!(
                            path.starts_with("controls/") && path.ends_with(".png"),
                            "glyph path {path:?} for {:?}/{device:?} is malformed",
                            b.action
                        ),
                        Glyph::Label(l) => assert!(
                            matches!(l, "LT" | "LB" | "RB"),
                            "unexpected text glyph {l:?} for {:?} (only the art-less triggers/bumpers)",
                            b.action
                        ),
                    }
                }
            }
        }
    }

    /// The reveal control differs by device and is the hint glyph: keyboard hold-Tab, pad
    /// hold-Back. Pins the hint to the bindings and that reveal reads as a HOLD on both
    /// devices, in every context.
    #[test]
    fn reveal_glyph_follows_the_bindings() {
        assert_eq!(
            reveal_glyph::<GcrControls>(Device::KeyboardMouse),
            Some(Glyph::Icon("controls/keyboard_tab.png"))
        );
        assert_eq!(
            reveal_glyph::<GcrControls>(Device::Gamepad),
            Some(Glyph::Icon("controls/xbox_button_view.png"))
        );
        for ctx in ALL_CONTEXTS {
            for device in [Device::KeyboardMouse, Device::Gamepad] {
                let reveal = legend::<GcrControls>(ctx, device)
                    .into_iter()
                    .find(|l| l.label == "Controls")
                    .unwrap();
                assert!(
                    reveal.hold,
                    "RevealControls must read as a hold in {ctx:?}/{device:?}"
                );
            }
        }
    }

    /// Extract is the one foot action with two glyphs on each device (key+mouse, A+RT) — pins
    /// that the alternate bindings actually surface, so a couch player sees both (on foot).
    #[test]
    fn extract_shows_both_bindings() {
        assert_eq!(
            binding::<GcrControls>(Action::Extract)
                .unwrap()
                .glyphs(Device::KeyboardMouse),
            vec![
                Glyph::Icon("controls/keyboard_space.png"),
                Glyph::Icon("controls/mouse_left.png")
            ]
        );
        assert_eq!(
            binding::<GcrControls>(Action::Extract)
                .unwrap()
                .glyphs(Device::Gamepad),
            vec![
                Glyph::Icon("controls/xbox_button_a.png"),
                Glyph::Icon("controls/xbox_rt.png")
            ]
        );
    }

    /// The shared-Start bug fix: pad Quit must be on its OWN button, NOT Start. Restart is
    /// Start (tap); if Quit shared Start, the press edge that begins a quit-hold would also
    /// fire the lockstep RESTART, restarting the round for every peer on the way out.
    #[test]
    fn pad_quit_is_a_hold_on_its_own_button_not_start() {
        let restart = binding::<GcrControls>(Action::Restart).unwrap();
        let quit = binding::<GcrControls>(Action::Quit).unwrap();
        assert_eq!(restart.pad.buttons, &[PadButton::Start]);
        assert!(!restart.pad.hold, "Restart is a tap");
        assert!(quit.pad.hold, "Quit is a hold");
        assert!(
            !quit.pad.buttons.contains(&PadButton::Start),
            "Quit must not share Start with Restart (a quit-hold would broadcast RESTART)"
        );
        let reveal = binding::<GcrControls>(Action::RevealControls).unwrap();
        assert_ne!(quit.pad.buttons, reveal.pad.buttons);
    }
}
