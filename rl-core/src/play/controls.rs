//! The demo's control scheme — the demo's verbs for the reusable controls overlay
//! (`crate::controls`). This is the single source of the on-screen LEGEND (it replaces the
//! old hand-written HUD string, so the legend can't drift from the bindings). The demo's
//! live input is analog/multi-key and read directly in its own systems (`cameras::orbit_camera`,
//! `demo::demo_controls`, `manual_control::manual_control_step`, `toggle_graph`) — the map
//! drives only the overlay, at the granularity that reads well (orbit's four arrow keys show
//! as one glyph).

use bevy::prelude::*;

use crate::controls::{ControlEntry, ControlInput, ControlScheme, Glyph, KbBinding, PadBinding};

/// The demo's control scheme — a zero-size marker the overlay framework is instantiated
/// with. Disjoint from GCR's [`crate::net::controls::GcrControls`]: different verbs, own map.
pub(crate) struct DemoControls;

/// The demo's controllable verbs. The row key of [`DEMO_CONTROL_MAP`]; the per-app test
/// forces every variant to have exactly one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoAction {
    Orbit,
    Zoom,
    /// Rebuild the crab at a fresh random tilt.
    Rebuild,
    /// A random force/torque burst.
    Poke,
    Colliders,
    JointGraph,
    /// Hands-on manual gamepad control; the next two only apply while it's active.
    Manual,
    PickJoint,
    Torque,
    Quit,
    /// Hold to reveal the full control overlay.
    RevealControls,
}

/// The demo's keyboard glyph tokens. Some resolve to bundled icons (R/Space/Esc/Tab),
/// others to text keycaps for keys with no bundled icon (arrows/zoom/G) — see [`Glyph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoKey {
    R,
    Space,
    Escape,
    Tab,
    /// The orbit arrow keys (←/↑/↓ + comma), shown as one "Arrows" glyph.
    Arrows,
    /// The zoom keys (−/=), shown as one glyph.
    ZoomKeys,
    /// The collider-toggle key (→ / right arrow).
    ColliderKey,
    /// The joint-graph key (G).
    Graph,
}

/// The demo's mouse glyph tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DemoMouse {
    /// Right-drag to orbit (reuses the move icon).
    Drag,
    /// Wheel to zoom (text glyph; no bundled wheel icon).
    Wheel,
}

/// The demo's gamepad glyph tokens.
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
    /// One glyph for two D-pad roles: collider toggle on Right, joint-pick on Up/Down.
    Dpad,
    View,
}

impl ControlScheme for DemoControls {
    type Action = DemoAction;
    type Key = DemoKey;
    type Pad = DemoPad;
    type Mouse = DemoMouse;

    fn map() -> &'static [ControlEntry<Self>] {
        &DEMO_CONTROL_MAP
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
            DemoKey::ColliderKey => Glyph::Label("Right"),
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
            DemoPad::Dpad => Glyph::Icon("controls/xbox_dpad.png"),
            DemoPad::View => Glyph::Icon("controls/xbox_button_view.png"),
            // No bundled icon for these — text keycaps.
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
        // Only the reveal control (Tab) is polled by the overlay; the demo's other keys are
        // analog/multi-key and read directly in its own systems, so they map to no single
        // KeyCode here.
        match key {
            DemoKey::Tab => Some(KeyCode::Tab),
            _ => None,
        }
    }

    fn gamepad_button(pad: DemoPad) -> Option<GamepadButton> {
        // Likewise: only the reveal control (View) need resolve for the overlay's hold read.
        match pad {
            DemoPad::View => Some(GamepadButton::Select),
            _ => None,
        }
    }
}

/// THE demo control map — models the demo's CURRENT bindings (verified against
/// `cameras::orbit_camera`, `demo::demo_controls`, `manual_control::manual_control_step`,
/// and `player::graph::toggle_graph`). Reveal binding: hold Tab / hold pad View — both
/// free in the demo's input set. Manual, PickJoint, and Torque are gamepad-only (no
/// keyboard binding), so the keyboard legend omits them.
pub(crate) const DEMO_CONTROL_MAP: [ControlEntry<DemoControls>; 11] = [
    ControlEntry {
        action: DemoAction::Orbit,
        label: "Orbit camera",
        keyboard: KbBinding::new(&[DemoKey::Arrows], &[DemoMouse::Drag]),
        pad: PadBinding::new(&[DemoPad::LeftStick]),
    },
    ControlEntry {
        action: DemoAction::Zoom,
        label: "Zoom",
        keyboard: KbBinding::new(&[DemoKey::ZoomKeys], &[DemoMouse::Wheel]),
        pad: PadBinding::new(&[DemoPad::RightTrigger, DemoPad::LeftTrigger]),
    },
    ControlEntry {
        action: DemoAction::Rebuild,
        label: "Rebuild crab",
        keyboard: KbBinding::new(&[DemoKey::R], &[]),
        pad: PadBinding::new(&[DemoPad::South]),
    },
    ControlEntry {
        action: DemoAction::Poke,
        label: "Poke",
        keyboard: KbBinding::new(&[DemoKey::Space], &[]),
        pad: PadBinding::new(&[DemoPad::West]),
    },
    ControlEntry {
        action: DemoAction::Colliders,
        label: "Collider wireframes",
        keyboard: KbBinding::new(&[DemoKey::ColliderKey], &[]),
        pad: PadBinding::new(&[DemoPad::Dpad]),
    },
    ControlEntry {
        action: DemoAction::JointGraph,
        label: "Joint graph",
        keyboard: KbBinding::new(&[DemoKey::Graph], &[]),
        pad: PadBinding::new(&[DemoPad::North]),
    },
    ControlEntry {
        action: DemoAction::Manual,
        label: "Manual control",
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::East]),
    },
    ControlEntry {
        action: DemoAction::PickJoint,
        label: "Pick joint (manual)",
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::Dpad]),
    },
    ControlEntry {
        action: DemoAction::Torque,
        label: "Joint torque (manual)",
        keyboard: KbBinding::NONE,
        pad: PadBinding::new(&[DemoPad::RightStick]),
    },
    ControlEntry {
        action: DemoAction::Quit,
        label: "Quit",
        keyboard: KbBinding::new(&[DemoKey::Escape], &[]),
        pad: PadBinding::new(&[DemoPad::Start]),
    },
    ControlEntry {
        action: DemoAction::RevealControls,
        label: "Controls",
        keyboard: KbBinding::hold(&[DemoKey::Tab], &[]),
        pad: PadBinding::hold(&[DemoPad::View]),
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-app control-map invariant: every [`DemoAction`] has exactly one
    /// [`DEMO_CONTROL_MAP`] row, is bound on at least one device, and the reveal control
    /// resolves to a hint glyph. The exhaustive `match` forces a new variant to be added to
    /// `ALL` (so it can't slip in unmapped); the framework does the runtime checks. Manual /
    /// PickJoint / Torque are gamepad-only — legitimately unbound on keyboard.
    #[test]
    fn demo_map_is_well_formed() {
        use crate::controls::assert_map_well_formed;
        const ALL: [DemoAction; 11] = [
            DemoAction::Orbit,
            DemoAction::Zoom,
            DemoAction::Rebuild,
            DemoAction::Poke,
            DemoAction::Colliders,
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
                | DemoAction::Colliders
                | DemoAction::JointGraph
                | DemoAction::Manual
                | DemoAction::PickJoint
                | DemoAction::Torque
                | DemoAction::Quit
                | DemoAction::RevealControls => true,
            }
        }
        assert!(ALL.iter().copied().all(classified));
        assert_map_well_formed::<DemoControls>(&ALL);
    }

    /// Every icon glyph the demo map surfaces points under `controls/`; the text-keycap
    /// glyphs (arrows/zoom/G/LT/X/B/Wheel) are intentional and need no asset.
    #[test]
    fn demo_icon_glyphs_are_under_controls_dir() {
        use crate::controls::{Device, Glyph};
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for e in &DEMO_CONTROL_MAP {
                for glyph in e.glyphs(device) {
                    if let Glyph::Icon(path) = glyph {
                        assert!(
                            path.starts_with("controls/") && path.ends_with(".png"),
                            "glyph path {path:?} for {:?}/{device:?} is malformed",
                            e.action
                        );
                    }
                }
            }
        }
    }
}
