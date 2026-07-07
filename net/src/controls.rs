use crab_world::controls::{Binding, ContextRow, ControlScheme, Glyph, KbBinding, PadBinding};

pub struct GcrControls;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GcrContext {
    #[default]
    OnFoot,
    Plane,
    Ship,
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
    EnterExit,
    CycleRenderMode,
    RevealControls,

    PlaneAttitude,
    PlaneThrottle,
    PlaneRudder,

    ShipThrust,
    ShipAim,
    ShipLift,
    ShipRoll,
    MatchVelocity,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseInput {
    Motion,
    Left,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadButton {
    South,
    North,
    West,
    East,
    RightTrigger,
    LeftTrigger,
    LeftBumper,
    RightBumper,
    Start,
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
];

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

#[cfg(feature = "render")]
mod bevy_glue {
    use super::*;
    use bevy::prelude::{GamepadButton, KeyCode, MouseButton};
    use crab_world::controls::ControlInput;

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
                PadButton::RightTrigger => Some(GamepadButton::RightTrigger2),
                PadButton::LeftTrigger => Some(GamepadButton::LeftTrigger2),
                PadButton::LeftBumper => Some(GamepadButton::LeftTrigger),
                PadButton::RightBumper => Some(GamepadButton::RightTrigger),
                PadButton::Start => Some(GamepadButton::Start),
                PadButton::Back => Some(GamepadButton::Select),
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

    #[test]
    fn legend_glyphs_come_from_the_bindings() {
        for ctx in ALL_CONTEXTS {
            for device in [Device::KeyboardMouse, Device::Gamepad] {
                let lines = legend::<GcrControls>(ctx, device);
                let rows = GcrControls::context_rows(ctx);
                for line in &lines {
                    let row = rows.iter().find(|r| r.label == line.label).unwrap();
                    let b = binding::<GcrControls>(row.action).unwrap();
                    assert_eq!(line.glyphs, b.glyphs(device));
                    assert!(!line.glyphs.is_empty(), "{:?} shows no glyph", row.action);
                }
            }
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
        assert!(plane.iter().any(|l| l.label == "Switch to ship"));
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
        assert!(ship.iter().any(|l| l.label == "Exit to foot"));
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
