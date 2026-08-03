use crab_world::chord::{ChordDir, ChordEntry, ChordRegistry};
use crab_world::controls::{Binding, ContextRow, ControlScheme, Glyph, KbBinding, PadBinding};

pub struct GcrControls;

/// GCR's chord table (rl#330) — the ONE list of code → command assignments; the owner
/// tunes codes here in play. These commands have no direct bindings — one trigger
/// route per action, enforced by `assert_scheme_well_formed`. A bare modifier tap
/// (the empty code) is deliberately unassigned: the owner de-overloaded X — vehicle
/// switching is a code per vehicle, not a tap verb (the ↑-family = the sky craft,
/// ↓↓ = back to the ground). Quit's code is ≥2 taps longer than every other — one
/// stray tap after any registered code can never end the round (the chord replacement
/// for the old timed hold-to-quit guard; a couch kid mashing d-pad stays in the round).
pub const GCR_CHORDS: ChordRegistry<Action> = ChordRegistry::new(&[
    ChordEntry {
        code: &[ChordDir::Up, ChordDir::Left],
        action: Action::BoardPlane,
        label: "Enter plane",
    },
    ChordEntry {
        code: &[ChordDir::Up, ChordDir::Right],
        action: Action::BoardShip,
        label: "Enter ship",
    },
    ChordEntry {
        code: &[ChordDir::Down, ChordDir::Down],
        action: Action::ExitVehicle,
        label: "Exit to foot",
    },
    ChordEntry {
        code: &[ChordDir::Up],
        action: Action::CycleRenderMode,
        label: "Render mode",
    },
    ChordEntry {
        code: &[ChordDir::Down],
        action: Action::CycleGroundLook,
        label: "Ground look",
    },
    ChordEntry {
        code: &[ChordDir::Left],
        action: Action::SwapBrain,
        label: "Swap brain",
    },
    ChordEntry {
        code: &[ChordDir::Right],
        action: Action::Restart,
        label: "Restart round",
    },
    ChordEntry {
        code: &[ChordDir::Up, ChordDir::Up, ChordDir::Down, ChordDir::Down],
        action: Action::Quit,
        label: "Quit round",
    },
]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GcrContext {
    #[default]
    OnFoot,
    Plane,
    Ship,
    /// The boot menu + lobby (`AppPhase::Menu`/`Connecting`) — nav is egui-drawn but the
    /// KEYS/BUTTONS come from this table, same as every in-round context (rl#117).
    Menu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    MoveForward,
    MoveBack,
    StrafeLeft,
    StrafeRight,
    Look,
    Extract,
    Restart,
    Quit,
    BoardPlane,
    BoardShip,
    ExitVehicle,
    CycleRenderMode,
    CycleGroundLook,
    SwapBrain,
    RevealControls,

    PlaneAttitude,
    PlaneThrottle,
    PlaneRudder,

    ShipThrust,
    ShipAim,
    ShipLift,
    ShipRoll,
    MatchVelocity,

    MenuUp,
    MenuDown,
    MenuConfirm,
    MenuBack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    W,
    A,
    S,
    D,
    Tab,
    Space,
    Escape,
    ArrowUp,
    ArrowDown,
    Enter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseInput {
    Motion,
    Left,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadButton {
    South,
    East,
    RightTrigger,
    LeftTrigger,
    LeftBumper,
    RightBumper,
    Back,
    LeftStick,
    RightStick,
    Dpad,
    /// Single D-pad directions — unlike [`PadButton::Dpad`] (the whole cluster, an
    /// analog-style multi-input) these resolve to one `GamepadButton`, so menu nav can
    /// poll them straight from the binding table.
    DpadUp,
    DpadDown,
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

    fn chords() -> ChordRegistry<Action> {
        GCR_CHORDS
    }

    fn contexts() -> &'static [GcrContext] {
        &[
            GcrContext::OnFoot,
            GcrContext::Plane,
            GcrContext::Ship,
            GcrContext::Menu,
        ]
    }

    fn context_rows(ctx: GcrContext) -> &'static [ContextRow<Self>] {
        match ctx {
            GcrContext::OnFoot => &FOOT_ROWS,
            GcrContext::Plane => &PLANE_ROWS,
            GcrContext::Ship => &SHIP_ROWS,
            GcrContext::Menu => &MENU_ROWS,
        }
    }

    fn context_label(ctx: GcrContext) -> &'static str {
        match ctx {
            GcrContext::OnFoot => "On foot",
            GcrContext::Plane => "Piloting plane",
            GcrContext::Ship => "Piloting ship",
            GcrContext::Menu => "Menu",
        }
    }

    fn context_id(ctx: GcrContext) -> &'static str {
        match ctx {
            GcrContext::OnFoot => "foot",
            GcrContext::Plane => "plane",
            GcrContext::Ship => "ship",
            GcrContext::Menu => "menu",
        }
    }

    fn context_from_id(id: &str) -> Option<GcrContext> {
        match id {
            "foot" => Some(GcrContext::OnFoot),
            "plane" => Some(GcrContext::Plane),
            "ship" => Some(GcrContext::Ship),
            "menu" => Some(GcrContext::Menu),
            _ => None,
        }
    }

    fn reveal_action() -> Action {
        Action::RevealControls
    }

    // The capture is reset outside Playing (render/app.rs) and the d-pad is the nav
    // input there — the menu legend advertising dead codes would lie.
    fn context_shows_chords(ctx: GcrContext) -> bool {
        ctx != GcrContext::Menu
    }

    fn key_glyph(key: Key) -> Glyph {
        Glyph::Icon(match key {
            Key::W => "controls/keyboard_w.png",
            Key::A => "controls/keyboard_a.png",
            Key::S => "controls/keyboard_s.png",
            Key::D => "controls/keyboard_d.png",
            Key::Tab => "controls/keyboard_tab.png",
            Key::Space => "controls/keyboard_space.png",
            Key::Escape => "controls/keyboard_escape.png",
            // No key art for these — text chips, like the art-less triggers/bumpers.
            Key::ArrowUp => return Glyph::Label("Up"),
            Key::ArrowDown => return Glyph::Label("Down"),
            Key::Enter => return Glyph::Label("Enter"),
        })
    }

    fn pad_glyph(pad: PadButton) -> Glyph {
        match pad {
            PadButton::South => Glyph::Icon("controls/xbox_button_a.png"),
            PadButton::East => Glyph::Icon("controls/xbox_button_b.png"),
            PadButton::RightTrigger => Glyph::Icon("controls/xbox_rt.png"),
            PadButton::LeftTrigger => Glyph::Label("LT"),
            PadButton::LeftBumper => Glyph::Label("LB"),
            PadButton::RightBumper => Glyph::Label("RB"),
            PadButton::Back => Glyph::Icon("controls/xbox_button_view.png"),
            PadButton::LeftStick => Glyph::Icon("controls/xbox_stick_l.png"),
            PadButton::RightStick => Glyph::Icon("controls/xbox_stick_r.png"),
            PadButton::Dpad | PadButton::DpadUp | PadButton::DpadDown => {
                Glyph::Icon("controls/xbox_dpad.png")
            }
        }
    }

    fn mouse_glyph(mouse: MouseInput) -> Glyph {
        Glyph::Icon(match mouse {
            MouseInput::Motion => "controls/mouse_move.png",
            MouseInput::Left => "controls/mouse_left.png",
        })
    }
}

// The command verbs (the per-vehicle boards/exit, render/ground cycles, SwapBrain,
// Restart, Quit) have NO rows here: they are chord codes in [`GCR_CHORDS`] and a
// second trigger route is a
// well-formedness error. What remains is analog/held state plus menu nav and the
// hold-to-reveal legend — reads the chord system's execute-on-release can't express.
pub const BINDINGS: [Binding<GcrControls>; 19] = [
    Binding {
        action: Action::MoveForward,
        keyboard: KbBinding::new(&[Key::W], &[]),
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
        action: Action::RevealControls,
        keyboard: KbBinding::hold(&[Key::Tab], &[]),
        pad: PadBinding::hold(&[PadButton::Back]),
    },
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
    // Menu nav (rl#117). The analog stick also navigates (a latched axis read, app-side
    // like every stick) — LeftStick here is its legend token, DpadUp/DpadDown the
    // pollable buttons.
    Binding {
        action: Action::MenuUp,
        keyboard: KbBinding::new(&[Key::ArrowUp, Key::W], &[]),
        pad: PadBinding::new(&[PadButton::DpadUp, PadButton::LeftStick]),
    },
    Binding {
        action: Action::MenuDown,
        keyboard: KbBinding::new(&[Key::ArrowDown, Key::S], &[]),
        pad: PadBinding::new(&[PadButton::DpadDown, PadButton::LeftStick]),
    },
    Binding {
        action: Action::MenuConfirm,
        keyboard: KbBinding::new(&[Key::Enter, Key::Space], &[]),
        pad: PadBinding::new(&[PadButton::South]),
    },
    Binding {
        action: Action::MenuBack,
        keyboard: KbBinding::new(&[Key::Escape], &[]),
        pad: PadBinding::new(&[PadButton::East]),
    },
];

// The in-round row tables list only DIRECT-bound controls: the chorded command verbs'
// legend rows come from [`GCR_CHORDS`] itself (labels live there, appended by
// `crab_world::controls::legend`), so the two can't drift.
pub const FOOT_ROWS: [ContextRow<GcrControls>; 7] = [
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
        action: Action::RevealControls,
        label: "Controls",
    },
];

pub const PLANE_ROWS: [ContextRow<GcrControls>; 4] = [
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
    ContextRow {
        action: Action::RevealControls,
        label: "Controls",
    },
];

pub const SHIP_ROWS: [ContextRow<GcrControls>; 6] = [
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
        action: Action::RevealControls,
        label: "Controls",
    },
];

pub const MENU_ROWS: [ContextRow<GcrControls>; 5] = [
    ContextRow {
        action: Action::MenuUp,
        label: "Up",
    },
    ContextRow {
        action: Action::MenuDown,
        label: "Down",
    },
    ContextRow {
        action: Action::MenuConfirm,
        label: "Select",
    },
    ContextRow {
        action: Action::MenuBack,
        label: "Back",
    },
    ContextRow {
        action: Action::RevealControls,
        label: "Controls",
    },
];

#[cfg(feature = "render")]
mod bevy_glue {
    use super::*;
    use bevy::prelude::{GamepadButton, KeyCode, MouseButton};
    use crab_world::controls::ControlInput;

    impl Key {
        fn key_codes(self) -> &'static [KeyCode] {
            match self {
                Key::W => &[KeyCode::KeyW],
                Key::A => &[KeyCode::KeyA],
                Key::S => &[KeyCode::KeyS],
                Key::D => &[KeyCode::KeyD],
                Key::Tab => &[KeyCode::Tab],
                Key::Space => &[KeyCode::Space],
                Key::Escape => &[KeyCode::Escape],
                Key::ArrowUp => &[KeyCode::ArrowUp],
                Key::ArrowDown => &[KeyCode::ArrowDown],
                // One control to the player, two physical keys — full-size keyboards
                // confirm with either.
                Key::Enter => &[KeyCode::Enter, KeyCode::NumpadEnter],
            }
        }
    }

    impl MouseInput {
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
                PadButton::East => Some(GamepadButton::East),
                PadButton::RightTrigger => Some(GamepadButton::RightTrigger2),
                PadButton::LeftTrigger => Some(GamepadButton::LeftTrigger2),
                PadButton::LeftBumper => Some(GamepadButton::LeftTrigger),
                PadButton::RightBumper => Some(GamepadButton::RightTrigger),
                PadButton::Back => Some(GamepadButton::Select),
                PadButton::DpadUp => Some(GamepadButton::DPadUp),
                PadButton::DpadDown => Some(GamepadButton::DPadDown),
                PadButton::LeftStick | PadButton::RightStick | PadButton::Dpad => None,
            }
        }
    }

    impl ControlInput for GcrControls {
        fn key_codes(key: Key) -> &'static [KeyCode] {
            key.key_codes()
        }
        fn gamepad_button(pad: PadButton) -> Option<GamepadButton> {
            pad.gamepad_button()
        }
    }

    pub fn key_code_for(action: Action) -> Option<KeyCode> {
        key_codes_for(action).next()
    }

    pub fn key_codes_for(action: Action) -> impl Iterator<Item = KeyCode> {
        crab_world::controls::key_codes_for::<GcrControls>(action)
    }

    pub fn gamepad_buttons_for(action: Action) -> impl Iterator<Item = GamepadButton> {
        crab_world::controls::gamepad_buttons_for::<GcrControls>(action)
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

    const ALL_ACTIONS: [Action; 27] = [
        Action::MoveForward,
        Action::MoveBack,
        Action::StrafeLeft,
        Action::StrafeRight,
        Action::Look,
        Action::Extract,
        Action::Restart,
        Action::Quit,
        Action::BoardPlane,
        Action::BoardShip,
        Action::ExitVehicle,
        Action::CycleRenderMode,
        Action::CycleGroundLook,
        Action::SwapBrain,
        Action::RevealControls,
        Action::PlaneAttitude,
        Action::PlaneThrottle,
        Action::PlaneRudder,
        Action::ShipThrust,
        Action::ShipAim,
        Action::ShipLift,
        Action::ShipRoll,
        Action::MatchVelocity,
        Action::MenuUp,
        Action::MenuDown,
        Action::MenuConfirm,
        Action::MenuBack,
    ];

    const ALL_CONTEXTS: [GcrContext; 4] = [
        GcrContext::OnFoot,
        GcrContext::Plane,
        GcrContext::Ship,
        GcrContext::Menu,
    ];

    #[test]
    fn chord_table_carries_every_migrated_command() {
        GCR_CHORDS.assert_well_formed();
        // The bare tap is unassigned (X de-overloaded): vehicles are codes, one each.
        assert_eq!(GCR_CHORDS.lookup(&[]), None);
        assert_eq!(
            GCR_CHORDS.lookup(&[ChordDir::Up, ChordDir::Left]),
            Some(Action::BoardPlane)
        );
        assert_eq!(
            GCR_CHORDS.lookup(&[ChordDir::Up, ChordDir::Right]),
            Some(Action::BoardShip)
        );
        assert_eq!(
            GCR_CHORDS.lookup(&[ChordDir::Down, ChordDir::Down]),
            Some(Action::ExitVehicle)
        );
        assert_eq!(
            GCR_CHORDS.lookup(&[ChordDir::Up]),
            Some(Action::CycleRenderMode)
        );
        assert_eq!(
            GCR_CHORDS.lookup(&[ChordDir::Down]),
            Some(Action::CycleGroundLook)
        );
        assert_eq!(
            GCR_CHORDS.lookup(&[ChordDir::Left]),
            Some(Action::SwapBrain)
        );
        assert_eq!(GCR_CHORDS.lookup(&[ChordDir::Right]), Some(Action::Restart));
        assert_eq!(
            GCR_CHORDS.lookup(&[ChordDir::Up, ChordDir::Up, ChordDir::Down, ChordDir::Down]),
            Some(Action::Quit)
        );
    }

    /// Quit replaced the timed hold-to-quit guard, so it must stay the hardest command
    /// to fat-finger: a stray extra tap after ANY registered code must never quit —
    /// i.e. Quit's code is at least two taps longer than every other code. (Length,
    /// not just prefix-freedom: one accidental repeat is the realistic slip.)
    #[test]
    fn quit_code_survives_a_stray_tap_after_any_other_code() {
        let quit_len = GCR_CHORDS
            .entries()
            .iter()
            .find(|e| e.action == Action::Quit)
            .unwrap()
            .code
            .len();
        assert!(
            GCR_CHORDS
                .entries()
                .iter()
                .all(|e| e.action == Action::Quit || e.code.len() + 2 <= quit_len),
            "Quit's code must be ≥2 taps longer than every other code"
        );
    }

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
                | Action::BoardPlane
                | Action::BoardShip
                | Action::ExitVehicle
                | Action::CycleRenderMode
                | Action::CycleGroundLook
                | Action::SwapBrain
                | Action::RevealControls
                | Action::PlaneAttitude
                | Action::PlaneThrottle
                | Action::PlaneRudder
                | Action::ShipThrust
                | Action::ShipAim
                | Action::ShipLift
                | Action::ShipRoll
                | Action::MatchVelocity
                | Action::MenuUp
                | Action::MenuDown
                | Action::MenuConfirm
                | Action::MenuBack => true,
            }
        }
        fn ctx_classified(c: GcrContext) -> bool {
            match c {
                GcrContext::OnFoot | GcrContext::Plane | GcrContext::Ship | GcrContext::Menu => {
                    true
                }
            }
        }
        assert!(ALL_ACTIONS.iter().copied().all(action_classified));
        assert!(ALL_CONTEXTS.iter().copied().all(ctx_classified));
        assert_eq!(GcrControls::contexts(), &ALL_CONTEXTS);
        assert_scheme_well_formed::<GcrControls>(&ALL_ACTIONS, &ALL_CONTEXTS);
    }

    #[test]
    fn legend_glyphs_come_from_the_bindings() {
        for ctx in ALL_CONTEXTS {
            for device in [Device::KeyboardMouse, Device::Gamepad] {
                let lines = legend::<GcrControls>(ctx, device);
                let rows = GcrControls::context_rows(ctx);
                for line in &lines {
                    let Some(row) = rows.iter().find(|r| r.label == line.label) else {
                        // Not a binding row: must be a chord row (checked below).
                        assert!(
                            GCR_CHORDS.entries().iter().any(|e| e.label == line.label),
                            "legend line {:?} matches no row and no chord entry",
                            line.label
                        );
                        continue;
                    };
                    let b = binding::<GcrControls>(row.action).unwrap();
                    assert_eq!(line.glyphs, b.glyphs(device));
                    assert!(!line.glyphs.is_empty(), "{:?} shows no glyph", row.action);
                }
            }
        }
    }

    /// Stage 4 (rl#330): every in-round legend shows the chord verbs, rendered from
    /// [`GCR_CHORDS`] itself — modifier glyph then one chip per code step; the menu
    /// context shows none (the capture is reset there and the d-pad is nav).
    #[test]
    fn legend_renders_chord_rows_from_the_registry() {
        use crab_world::chord::{dir_text, modifier_glyph};
        for ctx in [GcrContext::OnFoot, GcrContext::Plane, GcrContext::Ship] {
            for device in [Device::KeyboardMouse, Device::Gamepad] {
                let lines = legend::<GcrControls>(ctx, device);
                for e in GCR_CHORDS.entries() {
                    let line = lines
                        .iter()
                        .find(|l| l.label == e.label)
                        .unwrap_or_else(|| panic!("{ctx:?}/{device:?} omits {:?}", e.label));
                    let mut want = vec![modifier_glyph(device)];
                    want.extend(e.code.iter().map(|&d| Glyph::Label(dir_text(d, device))));
                    assert_eq!(line.glyphs, want);
                    assert!(!line.hold, "chord rows carry no Hold prefix");
                }
            }
        }
        let menu = legend::<GcrControls>(GcrContext::Menu, Device::Gamepad);
        for e in GCR_CHORDS.entries() {
            assert!(
                !menu.iter().any(|l| l.label == e.label),
                "menu legend advertises dead code {:?}",
                e.label
            );
        }
    }

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
        assert!(plane.iter().any(|l| l.label == "Throttle / brake"));
        assert!(plane.iter().any(|l| l.label == "Pitch / roll"));
        assert!(plane.iter().any(|l| l.label == "Rudder (yaw)"));
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
        assert!(ship.iter().any(|l| l.label == "Thrust: move / strafe"));
        assert!(ship.iter().any(|l| l.label == "Aim: pitch / yaw"));
        assert!(ship.iter().any(|l| l.label == "Roll"));
        assert!(ship.iter().any(|l| l.label == "Match velocity (brake)"));
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

    /// The menu polls and the menu legend both come from the one table (rl#117): the
    /// bindings here are the ones `gather_menu_inputs` resolves at runtime.
    #[test]
    fn menu_context_rides_the_one_binding_table() {
        let keys = |a| binding::<GcrControls>(a).unwrap().keyboard.keys;
        let pad = |a| binding::<GcrControls>(a).unwrap().pad.buttons;
        assert_eq!(keys(Action::MenuUp), &[Key::ArrowUp, Key::W]);
        assert_eq!(keys(Action::MenuDown), &[Key::ArrowDown, Key::S]);
        assert_eq!(keys(Action::MenuConfirm), &[Key::Enter, Key::Space]);
        assert_eq!(keys(Action::MenuBack), &[Key::Escape]);
        assert_eq!(pad(Action::MenuConfirm), &[PadButton::South]);
        assert_eq!(pad(Action::MenuBack), &[PadButton::East]);
        assert!(pad(Action::MenuUp).contains(&PadButton::DpadUp));
        assert!(pad(Action::MenuDown).contains(&PadButton::DpadDown));

        let menu = legend::<GcrControls>(GcrContext::Menu, Device::Gamepad);
        for label in ["Up", "Down", "Select", "Back", "Controls"] {
            assert!(menu.iter().any(|l| l.label == label), "menu misses {label}");
        }
        assert!(
            !menu.iter().any(|l| l.label == "Extract"),
            "no round verbs in the menu"
        );
        assert_eq!(GcrControls::context_label(GcrContext::Menu), "Menu");
    }

    #[test]
    fn flight_rides_the_one_binding_table() {
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
        assert_eq!(
            binding::<GcrControls>(Action::PlaneThrottle)
                .unwrap()
                .pad
                .buttons,
            &[PadButton::RightTrigger, PadButton::LeftTrigger]
        );
    }

    #[test]
    fn flight_binding_order_is_canonical() {
        let pad = |a| binding::<GcrControls>(a).unwrap().pad.buttons;
        let keys = |a| binding::<GcrControls>(a).unwrap().keyboard.keys;
        assert_eq!(
            pad(Action::PlaneThrottle),
            &[PadButton::RightTrigger, PadButton::LeftTrigger]
        );
        assert_eq!(
            pad(Action::ShipLift),
            &[PadButton::RightTrigger, PadButton::LeftTrigger]
        );
        assert_eq!(
            pad(Action::PlaneRudder),
            &[PadButton::LeftBumper, PadButton::RightBumper]
        );
        assert_eq!(
            pad(Action::ShipRoll),
            &[PadButton::LeftBumper, PadButton::RightBumper]
        );
        assert_eq!(pad(Action::MatchVelocity), &[PadButton::South]);
        assert_eq!(keys(Action::PlaneThrottle), &[Key::W, Key::S]);
        assert_eq!(keys(Action::PlaneRudder), &[Key::A, Key::D]);
        assert_eq!(keys(Action::MatchVelocity), &[Key::Space]);
    }

    /// One legend token, several physical keys: the numpad's Enter confirms too, and it
    /// does so without printing a second chip in the legend (rl#117).
    #[cfg(feature = "render")]
    #[test]
    fn confirm_accepts_both_enter_keys_but_shows_one_chip() {
        use bevy::prelude::KeyCode;
        let codes: Vec<KeyCode> = key_codes_for(Action::MenuConfirm).collect();
        assert!(codes.contains(&KeyCode::Enter));
        assert!(codes.contains(&KeyCode::NumpadEnter));
        assert!(codes.contains(&KeyCode::Space));
        assert_eq!(
            binding::<GcrControls>(Action::MenuConfirm)
                .unwrap()
                .glyphs(Device::KeyboardMouse),
            vec![
                Glyph::Label("Enter"),
                Glyph::Icon("controls/keyboard_space.png")
            ]
        );
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
        assert_eq!(GcrControls::context_from_id("menu"), Some(GcrContext::Menu));
        assert_eq!(GcrControls::context_from_id("nope"), None);
    }

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
                            matches!(l, "LT" | "LB" | "RB" | "Up" | "Down" | "Enter"),
                            "unexpected text glyph {l:?} for {:?} (only the art-less \
                             triggers/bumpers and the menu keys)",
                            b.action
                        ),
                    }
                }
            }
        }
    }

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
}
