use bevy::prelude::*;

use crate::chord::{ChordDir, ChordEntry, ChordRegistry};
use crate::controls::{
    Binding, ContextRow, ControlInput, ControlScheme, Glyph, KbBinding, PadBinding,
};

pub struct DemoControls;

/// The demo's chord table (rl#330 stage 4) — every discrete verb is a code; the owner
/// tunes assignments here in play. Codes mirror [`crate::chord`]'s GCR conventions where
/// the verbs match (^ render view, < swap brain, > for the reset-flavored verb, ^^vv
/// quit — Quit stays ≥2 taps longer than every other code, so a stray tap after any
/// registered code can't kill the demo). The bare tap (empty code) is unassigned, same
/// as GCR's de-overloaded X.
pub(crate) const DEMO_CHORDS: ChordRegistry<DemoAction> = ChordRegistry::new(&[
    ChordEntry {
        code: &[ChordDir::Up],
        action: DemoAction::RenderView,
        label: "Render view",
    },
    ChordEntry {
        code: &[ChordDir::Down],
        action: DemoAction::Poke,
        label: "Poke",
    },
    ChordEntry {
        code: &[ChordDir::Left],
        action: DemoAction::SwapBrain,
        label: "Swap brain",
    },
    ChordEntry {
        code: &[ChordDir::Right],
        action: DemoAction::Rebuild,
        label: "Rebuild crab",
    },
    ChordEntry {
        code: &[ChordDir::Up, ChordDir::Down],
        action: DemoAction::JointGraph,
        label: "Joint graph",
    },
    ChordEntry {
        code: &[ChordDir::Down, ChordDir::Up],
        action: DemoAction::Manual,
        label: "Manual control",
    },
    ChordEntry {
        code: &[ChordDir::Up, ChordDir::Up, ChordDir::Down, ChordDir::Down],
        action: DemoAction::Quit,
        label: "Quit",
    },
]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemoContext {
    #[default]
    Inspect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoAction {
    Orbit,
    Zoom,
    Rebuild,
    Poke,
    RenderView,
    JointGraph,
    Manual,
    PickJoint,
    Torque,
    SwapBrain,
    Quit,
    RevealControls,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoKey {
    Tab,
    Arrows,
    ZoomKeys,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoMouse {
    /// Hold-and-drag to orbit (reuses the move icon; button: [`ORBIT_DRAG_BUTTON`]).
    Drag,
    Wheel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoPad {
    LeftStick,
    RightStick,
    RightTrigger,
    LeftTrigger,
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

    fn chords() -> ChordRegistry<DemoAction> {
        DEMO_CHORDS
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
            DemoKey::Tab => Glyph::Icon("controls/keyboard_tab.png"),
            DemoKey::Arrows => Glyph::Label("Arrows"),
            DemoKey::ZoomKeys => Glyph::Label("- / ="),
        }
    }

    fn pad_glyph(pad: DemoPad) -> Glyph {
        match pad {
            DemoPad::LeftStick => Glyph::Icon("controls/xbox_stick_l.png"),
            DemoPad::RightStick => Glyph::Icon("controls/xbox_stick_r.png"),
            DemoPad::RightTrigger => Glyph::Icon("controls/xbox_rt.png"),
            DemoPad::DpadUpDown => Glyph::Icon("controls/xbox_dpad.png"),
            DemoPad::View => Glyph::Icon("controls/xbox_button_view.png"),
            DemoPad::LeftTrigger => Glyph::Label("LT"),
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
    fn key_codes(key: DemoKey) -> &'static [KeyCode] {
        match key {
            DemoKey::Tab => &[KeyCode::Tab],
            // Multi-key analog tokens (orbit's four keys, the zoom pair) — read directly
            // in `cameras::orbit_camera`, so they bind no discrete key here.
            DemoKey::Arrows | DemoKey::ZoomKeys => &[],
        }
    }

    fn gamepad_button(pad: DemoPad) -> Option<GamepadButton> {
        match pad {
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

/// THE demo binding table — analog/held state only since the stage-4 migration
/// (rl#330): every discrete verb is a chord code in [`DEMO_CHORDS`], and a second
/// trigger route is a well-formedness error. The analog rows (Orbit/Zoom in
/// `cameras::orbit_camera`, PickJoint/Torque in `manual_control`) are read directly and
/// must match by hand — their concrete inputs are named right below the table. Reveal
/// binding: hold Tab / hold pad View — a hold-to-show read the chord system's
/// execute-on-release can't express. PickJoint and Torque are gamepad-only (no keyboard
/// binding), so the keyboard legend omits them. Labels live in [`DEMO_ROWS`].
pub(crate) const DEMO_BINDINGS: [Binding<DemoControls>; 5] = [
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

// Direct-bound rows only — the chorded verbs' legend rows come from [`DEMO_CHORDS`]
// itself (labels live there, appended by `crate::controls::legend`), so the two can't
// drift.
pub(crate) const DEMO_ROWS: [ContextRow<DemoControls>; 5] = [
    ContextRow {
        action: DemoAction::Orbit,
        label: "Orbit camera",
    },
    ContextRow {
        action: DemoAction::Zoom,
        label: "Zoom",
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
        const ALL: [DemoAction; 12] = [
            DemoAction::Orbit,
            DemoAction::Zoom,
            DemoAction::Rebuild,
            DemoAction::Poke,
            DemoAction::RenderView,
            DemoAction::JointGraph,
            DemoAction::Manual,
            DemoAction::PickJoint,
            DemoAction::Torque,
            DemoAction::SwapBrain,
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
                | DemoAction::SwapBrain
                | DemoAction::Quit
                | DemoAction::RevealControls => true,
            }
        }
        assert!(ALL.iter().copied().all(classified));
        assert_scheme_well_formed::<DemoControls>(&ALL, &[DemoContext::Inspect]);
    }

    #[test]
    fn chord_table_carries_every_migrated_command() {
        use crate::chord::ChordDir::*;
        DEMO_CHORDS.assert_well_formed();
        // The bare tap is unassigned, same as GCR's de-overloaded X.
        assert_eq!(DEMO_CHORDS.lookup(&[]), None);
        assert_eq!(DEMO_CHORDS.lookup(&[Up]), Some(DemoAction::RenderView));
        assert_eq!(DEMO_CHORDS.lookup(&[Down]), Some(DemoAction::Poke));
        assert_eq!(DEMO_CHORDS.lookup(&[Left]), Some(DemoAction::SwapBrain));
        assert_eq!(DEMO_CHORDS.lookup(&[Right]), Some(DemoAction::Rebuild));
        assert_eq!(
            DEMO_CHORDS.lookup(&[Up, Down]),
            Some(DemoAction::JointGraph)
        );
        assert_eq!(DEMO_CHORDS.lookup(&[Down, Up]), Some(DemoAction::Manual));
        assert_eq!(
            DEMO_CHORDS.lookup(&[Up, Up, Down, Down]),
            Some(DemoAction::Quit)
        );
    }

    /// Stage 4 (rl#330): the demo legend shows every chord verb, rendered from
    /// [`DEMO_CHORDS`] itself — modifier glyph then one chip per code step.
    #[test]
    fn legend_renders_chord_rows_from_the_registry() {
        use crate::chord::{dir_text, modifier_glyph};
        use crate::controls::{Device, Glyph, legend};
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            let lines = legend::<DemoControls>(DemoContext::Inspect, device);
            for e in DEMO_CHORDS.entries() {
                let line = lines
                    .iter()
                    .find(|l| l.label == e.label)
                    .unwrap_or_else(|| panic!("{device:?} legend omits {:?}", e.label));
                let mut want = vec![modifier_glyph(device)];
                want.extend(e.code.iter().map(|&d| Glyph::Label(dir_text(d, device))));
                assert_eq!(line.glyphs, want);
            }
        }
    }

    /// Same fat-finger bar as GCR's quit: a stray extra tap after ANY registered code
    /// must never kill the demo — Quit's code is ≥2 taps longer than every other.
    #[test]
    fn quit_code_survives_a_stray_tap_after_any_other_code() {
        let quit_len = DEMO_CHORDS
            .entries()
            .iter()
            .find(|e| e.action == DemoAction::Quit)
            .unwrap()
            .code
            .len();
        assert!(
            DEMO_CHORDS
                .entries()
                .iter()
                .all(|e| e.action == DemoAction::Quit || e.code.len() + 2 <= quit_len),
            "Quit's code must be ≥2 taps longer than every other code"
        );
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
                    !DemoControls::key_codes(k).is_empty(),
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
