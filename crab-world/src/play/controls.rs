use bevy::prelude::*;

use crate::controls::{
    Binding, ContextRow, ControlInput, ControlScheme, Glyph, KbBinding, PadBinding,
};

pub(crate) struct DemoControls;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum DemoContext {
    #[default]
    Inspect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoAction {
    Orbit,
    Zoom,
    Rebuild,
    Poke,
    RenderView,
    JointGraph,
    Manual,
    PickJoint,
    Torque,
    Quit,
    RevealControls,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoKey {
    R,
    Space,
    Escape,
    Tab,
    Arrows,
    ZoomKeys,
    /// The render-view-cycle key (→ / right arrow).
    RenderViewKey,
    Graph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoMouse {
    /// Hold-and-drag to orbit (reuses the move icon; button: [`ORBIT_DRAG_BUTTON`]).
    Drag,
    Wheel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoPad {
    LeftStick,
    RightStick,
    RightTrigger,
    LeftTrigger,
    South,
    West,
    East,
    North,
    Start,
    /// D-pad Right (render-view cycle). Split from [`DemoPad::DpadUpDown`] so the discrete
    /// verb resolves to its one button; both show the same D-pad glyph.
    DpadRight,
    /// D-pad Up/Down (joint pick in manual mode) — a directional pair, read directly in
    /// `manual_control`, so it resolves to no single button.
    DpadUpDown,
    View,
}

impl ControlScheme for DemoControls {
    type Action = DemoAction;
    type Key = DemoKey;
    type Pad = DemoPad;
    type Mouse = DemoMouse;
    type Context = DemoContext;

    fn bindings() -> &'static [Binding<Self>] {
        &DEMO_BINDINGS
    }

    fn contexts() -> &'static [DemoContext] {
        &[DemoContext::Inspect]
    }

    fn context_rows(ctx: DemoContext) -> &'static [ContextRow<Self>] {
        match ctx {
            DemoContext::Inspect => &DEMO_ROWS,
        }
    }

    fn context_label(ctx: DemoContext) -> &'static str {
        match ctx {
            DemoContext::Inspect => "Crab demo",
        }
    }

    fn context_id(ctx: DemoContext) -> &'static str {
        match ctx {
            DemoContext::Inspect => "inspect",
        }
    }

    fn context_from_id(id: &str) -> Option<DemoContext> {
        match id {
            "inspect" => Some(DemoContext::Inspect),
            _ => None,
        }
    }

    fn reveal_action() -> DemoAction {
        DemoAction::RevealControls
    }

    fn key_glyph(key: DemoKey) -> Glyph {
        match key {
            DemoKey::R => Glyph::Icon("controls/keyboard_r.png"),
            DemoKey::Space => Glyph::Icon("controls/keyboard_space.png"),
            DemoKey::Escape => Glyph::Icon("controls/keyboard_escape.png"),
            DemoKey::Tab => Glyph::Icon("controls/keyboard_tab.png"),
            DemoKey::Arrows => Glyph::Label("Arrows"),
            DemoKey::ZoomKeys => Glyph::Label("- / ="),
            DemoKey::RenderViewKey => Glyph::Label("Right"),
            DemoKey::Graph => Glyph::Label("G"),
        }
    }

    fn pad_glyph(pad: DemoPad) -> Glyph {
        match pad {
            DemoPad::LeftStick => Glyph::Icon("controls/xbox_stick_l.png"),
            DemoPad::RightStick => Glyph::Icon("controls/xbox_stick_r.png"),
            DemoPad::RightTrigger => Glyph::Icon("controls/xbox_rt.png"),
            DemoPad::South => Glyph::Icon("controls/xbox_button_a.png"),
            DemoPad::North => Glyph::Icon("controls/xbox_button_y.png"),
            DemoPad::Start => Glyph::Icon("controls/xbox_button_menu.png"),
            DemoPad::DpadRight | DemoPad::DpadUpDown => Glyph::Icon("controls/xbox_dpad.png"),
            DemoPad::View => Glyph::Icon("controls/xbox_button_view.png"),
            DemoPad::LeftTrigger => Glyph::Label("LT"),
            DemoPad::West => Glyph::Label("X"),
            DemoPad::East => Glyph::Label("B"),
        }
    }

    fn mouse_glyph(mouse: DemoMouse) -> Glyph {
        match mouse {
            DemoMouse::Drag => Glyph::Icon("controls/mouse_move.png"),
            DemoMouse::Wheel => Glyph::Label("Wheel"),
        }
    }
}

impl ControlInput for DemoControls {
    fn key_code(key: DemoKey) -> Option<KeyCode> {
        match key {
            DemoKey::R => Some(KeyCode::KeyR),
            DemoKey::Space => Some(KeyCode::Space),
            DemoKey::Escape => Some(KeyCode::Escape),
            DemoKey::Tab => Some(KeyCode::Tab),
            DemoKey::RenderViewKey => Some(KeyCode::ArrowRight),
            DemoKey::Graph => Some(KeyCode::KeyG),
            // Multi-key analog tokens (orbit's four keys, the zoom pair) — read directly
            // in `cameras::orbit_camera`, so no single KeyCode.
            DemoKey::Arrows | DemoKey::ZoomKeys => None,
        }
    }

    fn gamepad_button(pad: DemoPad) -> Option<GamepadButton> {
        match pad {
            DemoPad::South => Some(GamepadButton::South),
            DemoPad::West => Some(GamepadButton::West),
            DemoPad::East => Some(GamepadButton::East),
            DemoPad::North => Some(GamepadButton::North),
            DemoPad::Start => Some(GamepadButton::Start),
            DemoPad::DpadRight => Some(GamepadButton::DPadRight),
            DemoPad::View => Some(GamepadButton::Select),
            // Analog/directional tokens (sticks, the zoom triggers, the joint-pick D-pad
            // pair) — read via their own axis/multi-button APIs in their systems.
            DemoPad::LeftStick
            | DemoPad::RightStick
            | DemoPad::RightTrigger
            | DemoPad::LeftTrigger
            | DemoPad::DpadUpDown => None,
        }
    }
}

/// THE demo binding table. The tap verbs DISPATCH from it (via
/// [`crate::controls::just_pressed`]); only the analog rows (Orbit/Zoom in
/// `cameras::orbit_camera`, PickJoint/Torque in `manual_control`) are read directly and
/// must match by hand — their concrete inputs are named right below the table.
/// Reveal binding: hold Tab /
/// hold pad View — both free in the demo's input set. Manual, PickJoint, and Torque are
/// gamepad-only (no keyboard binding), so the keyboard legend omits them. Labels live in
/// [`DEMO_ROWS`].
pub(crate) const DEMO_BINDINGS: [Binding<DemoControls>; 11] = [
    Binding {
        action: DemoAction::Orbit,
        keyboard: KbBinding::new(&[DemoKey::Arrows], &[DemoMouse::Drag]),
        pad: PadBinding::new(&[DemoPad::LeftStick]),
    },
    Binding {
        action: DemoAction::Zoom,
        keyboard: KbBinding::new(&[DemoKey::ZoomKeys], &[DemoMouse::Wheel]),
        pad: PadBinding::new(&[DemoPad::RightTrigger, DemoPad::LeftTrigger]),
    },
    Binding {
        action: DemoAction::Rebuild,
        keyboard: KbBinding::new(&[DemoKey::R], &[]),
        pad: PadBinding::new(&[DemoPad::South]),
    },
    Binding {
        action: DemoAction::Poke,
        keyboard: KbBinding::new(&[DemoKey::Space], &[]),
        pad: PadBinding::new(&[DemoPad::West]),
    },
    Binding {
        action: DemoAction::RenderView,
        keyboard: KbBinding::new(&[DemoKey::RenderViewKey], &[]),
        pad: PadBinding::new(&[DemoPad::DpadRight]),
    },
    Binding {
        action: DemoAction::JointGraph,
        keyboard: KbBinding::new(&[DemoKey::Graph], &[]),
        pad: PadBinding::new(&[DemoPad::North]),
    },
    Binding {
        action: DemoAction::Manual,
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::East]),
    },
    Binding {
        action: DemoAction::PickJoint,
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::DpadUpDown]),
    },
    Binding {
        action: DemoAction::Torque,
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::RightStick]),
    },
    Binding {
        action: DemoAction::Quit,
        keyboard: KbBinding::new(&[DemoKey::Escape], &[]),
        pad: PadBinding::new(&[DemoPad::Start]),
    },
    Binding {
        action: DemoAction::RevealControls,
        keyboard: KbBinding::hold(&[DemoKey::Tab], &[]),
        pad: PadBinding::hold(&[DemoPad::View]),
    },
];

// The analog rows' concrete inputs. These four rows (Orbit/Zoom/PickJoint/Torque) carry
// direction/sign semantics the table can't dispatch, so their systems read the device
// directly — these names keep that hand-matched surface HERE, with the table in view:
// rebinding an analog row means editing its const/accessor below AND its glyph token in
// [`DEMO_BINDINGS`] above, together.

/// Orbit keys, shown as the one [`DemoKey::Arrows`] glyph. Right-yaw is Comma, not
/// ArrowRight: the right arrow is the RenderView tap verb, and mouse drag already
/// covers free-look orbiting in every direction.
pub(super) const ORBIT_YAW_LEFT_KEY: KeyCode = KeyCode::ArrowLeft;
pub(super) const ORBIT_YAW_RIGHT_KEY: KeyCode = KeyCode::Comma;
pub(super) const ORBIT_PITCH_UP_KEY: KeyCode = KeyCode::ArrowUp;
pub(super) const ORBIT_PITCH_DOWN_KEY: KeyCode = KeyCode::ArrowDown;

/// Orbit mouse: hold-and-drag free-look ([`DemoMouse::Drag`]).
pub(super) const ORBIT_DRAG_BUTTON: MouseButton = MouseButton::Right;

/// Orbit stick ([`DemoPad::LeftStick`]) as (yaw, pitch) input. (The RIGHT stick is
/// reserved for manual-control joint torque; left-stick orbit is the convention anyway.)
pub(super) fn orbit_stick(gp: &Gamepad) -> Vec2 {
    gp.left_stick()
}

/// Zoom keys ([`DemoKey::ZoomKeys`], "- / =") and triggers ([`DemoPad::RightTrigger`] /
/// [`DemoPad::LeftTrigger`]). "Out" grows the orbit radius, "in" shrinks it.
pub(super) const ZOOM_OUT_KEY: KeyCode = KeyCode::Minus;
pub(super) const ZOOM_IN_KEY: KeyCode = KeyCode::Equal;
pub(super) const ZOOM_OUT_TRIGGER: GamepadButton = GamepadButton::RightTrigger2;
pub(super) const ZOOM_IN_TRIGGER: GamepadButton = GamepadButton::LeftTrigger2;

/// Manual-mode joint pick ([`DemoPad::DpadUpDown`]): up cycles forward, down back.
pub(super) const PICK_JOINT_NEXT_BUTTON: GamepadButton = GamepadButton::DPadUp;
pub(super) const PICK_JOINT_PREV_BUTTON: GamepadButton = GamepadButton::DPadDown;

/// Torque input ([`DemoPad::RightStick`]): the right stick's Y drives the picked joint.
pub(super) fn torque_stick_y(gp: &Gamepad) -> f32 {
    gp.right_stick().y
}

pub(crate) const DEMO_ROWS: [ContextRow<DemoControls>; 11] = [
    ContextRow {
        action: DemoAction::Orbit,
        label: "Orbit camera",
    },
    ContextRow {
        action: DemoAction::Zoom,
        label: "Zoom",
    },
    ContextRow {
        action: DemoAction::Rebuild,
        label: "Rebuild crab",
    },
    ContextRow {
        action: DemoAction::Poke,
        label: "Poke",
    },
    ContextRow {
        action: DemoAction::RenderView,
        label: "Render view (cycle)",
    },
    ContextRow {
        action: DemoAction::JointGraph,
        label: "Joint graph",
    },
    ContextRow {
        action: DemoAction::Manual,
        label: "Manual control",
    },
    ContextRow {
        action: DemoAction::PickJoint,
        label: "Pick joint (manual)",
    },
    ContextRow {
        action: DemoAction::Torque,
        label: "Joint torque (manual)",
    },
    ContextRow {
        action: DemoAction::Quit,
        label: "Quit",
    },
    ContextRow {
        action: DemoAction::RevealControls,
        label: "Controls",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_scheme_is_well_formed() {
        use crate::controls::assert_scheme_well_formed;
        const ALL: [DemoAction; 11] = [
            DemoAction::Orbit,
            DemoAction::Zoom,
            DemoAction::Rebuild,
            DemoAction::Poke,
            DemoAction::RenderView,
            DemoAction::JointGraph,
            DemoAction::Manual,
            DemoAction::PickJoint,
            DemoAction::Torque,
            DemoAction::Quit,
            DemoAction::RevealControls,
        ];
        fn classified(a: DemoAction) -> bool {
            match a {
                DemoAction::Orbit
                | DemoAction::Zoom
                | DemoAction::Rebuild
                | DemoAction::Poke
                | DemoAction::RenderView
                | DemoAction::JointGraph
                | DemoAction::Manual
                | DemoAction::PickJoint
                | DemoAction::Torque
                | DemoAction::Quit
                | DemoAction::RevealControls => true,
            }
        }
        assert!(ALL.iter().copied().all(classified));
        assert_scheme_well_formed::<DemoControls>(&ALL, &[DemoContext::Inspect]);
    }

    /// The tap verbs dispatch via [`crate::controls::just_pressed`], which drops binding
    /// tokens that don't resolve to a live Bevy input — so a rebind to an unresolvable
    /// token would show in the legend but never fire. Force every token on every row to
    /// resolve, except the explicitly-declared analog rows (read directly in their own
    /// systems). Fail-closed: a NEW action defaults into the check; declaring it analog is
    /// a conscious edit here.
    #[test]
    fn tap_verb_bindings_all_resolve_to_live_inputs() {
        use crate::controls::ControlInput;
        const ANALOG: [DemoAction; 4] = [
            DemoAction::Orbit,
            DemoAction::Zoom,
            DemoAction::PickJoint,
            DemoAction::Torque,
        ];
        for b in &DEMO_BINDINGS {
            if ANALOG.contains(&b.action) {
                continue;
            }
            for &k in b.keyboard.keys {
                assert!(
                    DemoControls::key_code(k).is_some(),
                    "{:?}: key token {k:?} resolves to no KeyCode — legend would show a dead key",
                    b.action
                );
            }
            for &p in b.pad.buttons {
                assert!(
                    DemoControls::gamepad_button(p).is_some(),
                    "{:?}: pad token {p:?} resolves to no button — legend would show a dead button",
                    b.action
                );
            }
        }
    }

    #[test]
    fn demo_icon_glyphs_are_under_controls_dir() {
        use crate::controls::{Device, Glyph};
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for b in &DEMO_BINDINGS {
                for glyph in b.glyphs(device) {
                    if let Glyph::Icon(path) = glyph {
                        assert!(
                            path.starts_with("controls/") && path.ends_with(".png"),
                            "glyph path {path:?} for {:?}/{device:?} is malformed",
                            b.action
                        );
                    }
                }
            }
        }
    }
}
