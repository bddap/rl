//! Giant Crab Rescue's control scheme: GCR's action set + input vocabulary + glyph art +
//! the one [`CONTROL_MAP`], implemented as a [`ControlScheme`] for the reusable overlay
//! framework in [`crate::controls`]. The framework renders the legend + hold-to-reveal
//! overlay from this; the live input ([`crate::net::render`]'s `gather_input`/`quit_game`)
//! reads the same map via [`key_code_for`]/[`gamepad_buttons_for`], so the keys the client
//! polls are exactly the keys the legend shows — no drift.
//!
//! Two caveats on the single-source guarantee:
//! - It covers the REBINDABLE discrete controls. The analog movers (sticks, mouse motion,
//!   D-pad digital move) are read directly in `gather_input` and only their GLYPHS appear
//!   in the map — fixed, not rebindable, so outside the key/glyph round-trip.
//! - Nothing here touches the sim. A binding only decides WHICH key feeds an [`Action`];
//!   the value still funnels through `Input::new`'s fixed-point quantization downstream, so
//!   rebinding never changes the wire value and peers on different bindings stay
//!   bit-identical (the determinism contract atop [`crate::net::sim`]).

use crate::controls::{ControlEntry, ControlScheme, Glyph, KbBinding, PadBinding};

/// GCR's control scheme — the type parameter the framework is instantiated with. A
/// zero-size marker; all of GCR's control data hangs off its [`ControlScheme`] impl.
pub struct GcrControls;

/// A controllable action in Giant Crab Rescue. The move axes are split into four discrete
/// directional actions (not one "move") so each has its own glyph in the legend (WASD shows
/// as four keys). [`Look`](Action::Look) is analog (mouse / right stick). This enum is the
/// row key of [`CONTROL_MAP`]; the per-app test forces every variant to have exactly one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    MoveForward,
    MoveBack,
    StrafeLeft,
    StrafeRight,
    /// Aim the camera (yaw → sim, pitch → client). Analog; no discrete key binding.
    Look,
    /// At the extraction pillar, confirm the pickup (the sim's `buttons::ACTION`).
    Extract,
    /// Restart the round for all peers (the sim's `buttons::RESTART`, edge-triggered).
    Restart,
    /// Quit the client (local `AppExit`; never touches the sim).
    Quit,
    /// Hold to reveal the full control overlay; release to hide. Pure client UI.
    RevealControls,
}

/// A keyboard key GCR binds. A closed set (only the keys GCR uses), so the map and glyph
/// table are exhaustive and a typo can't name a nonexistent key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    W,
    A,
    S,
    D,
    R,
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
    /// Right trigger (alternate extract).
    RightTrigger,
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

    fn map() -> &'static [ControlEntry<Self>] {
        &CONTROL_MAP
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
            Key::R => "controls/keyboard_r.png",
            Key::Tab => "controls/keyboard_tab.png",
            Key::Space => "controls/keyboard_space.png",
            Key::Escape => "controls/keyboard_escape.png",
        })
    }

    fn pad_glyph(pad: PadButton) -> Glyph {
        Glyph::Icon(match pad {
            PadButton::South => "controls/xbox_button_a.png",
            PadButton::North => "controls/xbox_button_y.png",
            PadButton::RightTrigger => "controls/xbox_rt.png",
            PadButton::Start => "controls/xbox_button_menu.png",
            PadButton::Back => "controls/xbox_button_view.png",
            PadButton::LeftStick => "controls/xbox_stick_l.png",
            PadButton::RightStick => "controls/xbox_stick_r.png",
            PadButton::Dpad => "controls/xbox_dpad.png",
        })
    }

    fn mouse_glyph(mouse: MouseInput) -> Glyph {
        Glyph::Icon(match mouse {
            MouseInput::Motion => "controls/mouse_move.png",
            MouseInput::Left => "controls/mouse_left.png",
        })
    }
}

/// THE GCR control map. One coherent scheme, chosen deliberately:
/// - **Move**: WASD / left stick (+ D-pad shown once, on Forward), the FPS convention.
/// - **Look**: mouse / right stick.
/// - **Extract**: Space or left-click / pad South (A) or right trigger.
/// - **Restart**: R / pad Start (tap) — edge-triggered, rides the lockstep input stream.
/// - **Quit**: Esc / HOLD pad North (Y) ≥1s. A hold so a stray tap can't end the round; on
///   its OWN button (not Start) so quitting can't also fire the lockstep RESTART a shared
///   Start tap would — Start is restart-only.
/// - **Reveal controls**: HOLD Tab / HOLD pad Back — show the overlay while held.
pub const CONTROL_MAP: [ControlEntry<GcrControls>; 9] = [
    ControlEntry {
        action: Action::MoveForward,
        label: "Forward",
        keyboard: KbBinding::new(&[Key::W], &[]),
        // The D-pad is the alternate digital mover; surface its glyph once (here) rather
        // than on all four move rows.
        pad: PadBinding::new(&[PadButton::LeftStick, PadButton::Dpad]),
    },
    ControlEntry {
        action: Action::MoveBack,
        label: "Back",
        keyboard: KbBinding::new(&[Key::S], &[]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    ControlEntry {
        action: Action::StrafeLeft,
        label: "Strafe left",
        keyboard: KbBinding::new(&[Key::A], &[]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    ControlEntry {
        action: Action::StrafeRight,
        label: "Strafe right",
        keyboard: KbBinding::new(&[Key::D], &[]),
        pad: PadBinding::new(&[PadButton::LeftStick]),
    },
    ControlEntry {
        action: Action::Look,
        label: "Look",
        keyboard: KbBinding::new(&[], &[MouseInput::Motion]),
        pad: PadBinding::new(&[PadButton::RightStick]),
    },
    ControlEntry {
        action: Action::Extract,
        label: "Extract",
        keyboard: KbBinding::new(&[Key::Space], &[MouseInput::Left]),
        pad: PadBinding::new(&[PadButton::South, PadButton::RightTrigger]),
    },
    ControlEntry {
        action: Action::Restart,
        label: "Restart round",
        keyboard: KbBinding::new(&[Key::R], &[]),
        pad: PadBinding::new(&[PadButton::Start]),
    },
    ControlEntry {
        action: Action::Quit,
        label: "Quit",
        keyboard: KbBinding::hold(&[Key::Escape], &[]),
        pad: PadBinding::hold(&[PadButton::North]),
    },
    ControlEntry {
        action: Action::RevealControls,
        label: "Controls",
        keyboard: KbBinding::hold(&[Key::Tab], &[]),
        pad: PadBinding::hold(&[PadButton::Back]),
    },
];

// ---------------------------------------------------------------------------
// Bevy glue — the ONLY place GCR's typed inputs meet Bevy's input API. Render-only:
// Bevy's KeyCode/GamepadButton/MouseButton exist only under the `render` feature. The live
// predicates read CONTROL_MAP via these, so the client polls exactly the keys the legend
// shows — no drift.
// ---------------------------------------------------------------------------

#[cfg(feature = "render")]
mod bevy_glue {
    use super::*;
    use crate::controls::{ControlInput, entry};
    use bevy::prelude::{GamepadButton, KeyCode, MouseButton};

    impl Key {
        fn key_code(self) -> KeyCode {
            match self {
                Key::W => KeyCode::KeyW,
                Key::A => KeyCode::KeyA,
                Key::S => KeyCode::KeyS,
                Key::D => KeyCode::KeyD,
                Key::R => KeyCode::KeyR,
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
                PadButton::RightTrigger => Some(GamepadButton::RightTrigger2),
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

    /// The `KeyCode` bound to an action on the keyboard, if any. Reads [`CONTROL_MAP`], so
    /// the client polls the mapped key — change the map, the poll follows. Polls the FIRST
    /// bound key only while the legend renders every bound key: identical today (every GCR
    /// row binds exactly one key), but a future multi-key action would poll one key yet show
    /// all — generalize this (and `gamepad_buttons_for`) before binding more than one.
    pub fn key_code_for(action: Action) -> Option<KeyCode> {
        entry::<GcrControls>(action)
            .and_then(|e| e.keyboard.keys.first().copied())
            .map(Key::key_code)
    }

    /// The pad `GamepadButton`(s) that trigger an action (every mapped button), for the
    /// discrete buttons. Sticks/D-pad yield nothing (read via their dedicated APIs). An
    /// iterator, not a `Vec`: callers just `.any(...)` each frame, so no heap alloc.
    pub fn gamepad_buttons_for(action: Action) -> impl Iterator<Item = GamepadButton> {
        entry::<GcrControls>(action)
            .into_iter()
            .flat_map(|e| e.pad.buttons.iter().copied())
            .filter_map(PadButton::gamepad_button)
    }
}

#[cfg(feature = "render")]
pub use bevy_glue::{gamepad_buttons_for, key_code_for};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{Device, Glyph, assert_map_well_formed, entry, legend, reveal_glyph};

    /// Every [`Action`] is exhaustively classified (so a new variant can't be added without
    /// declaring it here), and the framework proves each has exactly one map row, is bound on
    /// some device, and the reveal control resolves to a hint glyph. The compiler forces the
    /// `match` to cover every variant; `assert_map_well_formed` does the runtime checks.
    #[test]
    fn every_action_has_exactly_one_map_row() {
        const ALL: [Action; 9] = [
            Action::MoveForward,
            Action::MoveBack,
            Action::StrafeLeft,
            Action::StrafeRight,
            Action::Look,
            Action::Extract,
            Action::Restart,
            Action::Quit,
            Action::RevealControls,
        ];
        // Exhaustiveness guard: a new variant fails to compile until added to ALL above.
        fn classified(a: Action) -> bool {
            match a {
                Action::MoveForward
                | Action::MoveBack
                | Action::StrafeLeft
                | Action::StrafeRight
                | Action::Look
                | Action::Extract
                | Action::Restart
                | Action::Quit
                | Action::RevealControls => true,
            }
        }
        assert!(ALL.iter().copied().all(classified));
        assert_map_well_formed::<GcrControls>(&ALL);
    }

    /// The no-drift property: the legend is built by iterating the SAME table the live input
    /// reads, so every legend line's glyphs equal the row's `glyphs(device)`. GCR binds every
    /// action on both devices, so both legends cover all nine rows.
    #[test]
    fn legend_matches_the_map_for_both_devices() {
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            let lines = legend::<GcrControls>(device);
            assert_eq!(
                lines.len(),
                CONTROL_MAP.len(),
                "legend must cover every row"
            );
            for (line, e) in lines.iter().zip(CONTROL_MAP.iter()) {
                assert_eq!(line.label, e.label);
                assert_eq!(line.glyphs, e.glyphs(device));
                assert!(!line.glyphs.is_empty(), "{:?} shows no glyph", e.action);
            }
        }
    }

    /// Every glyph the map surfaces is a bundled icon under `controls/` (GCR ships no text
    /// labels). A renamed/relocated asset (or a typo) would 404 at runtime; this catches it.
    #[test]
    fn all_glyph_paths_are_under_controls_dir() {
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for e in &CONTROL_MAP {
                for glyph in e.glyphs(device) {
                    match glyph {
                        Glyph::Icon(path) => assert!(
                            path.starts_with("controls/") && path.ends_with(".png"),
                            "glyph path {path:?} for {:?}/{device:?} is malformed",
                            e.action
                        ),
                        Glyph::Label(l) => panic!("GCR uses only icons; got label {l:?}"),
                    }
                }
            }
        }
    }

    /// The reveal control differs by device and is the hint glyph: keyboard hold-Tab, pad
    /// hold-Back. Pins the hint to the map and that reveal reads as a HOLD on both devices.
    #[test]
    fn reveal_glyph_follows_the_map() {
        assert_eq!(
            reveal_glyph::<GcrControls>(Device::KeyboardMouse),
            Some(Glyph::Icon("controls/keyboard_tab.png"))
        );
        assert_eq!(
            reveal_glyph::<GcrControls>(Device::Gamepad),
            Some(Glyph::Icon("controls/xbox_button_view.png"))
        );
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            let reveal = legend::<GcrControls>(device)
                .into_iter()
                .find(|l| l.label == "Controls")
                .unwrap();
            assert!(
                reveal.hold,
                "RevealControls must read as a hold on {device:?}"
            );
        }
    }

    /// Extract is the one action with two glyphs on each device (key+mouse, A+RT) — pins
    /// that the alternate bindings actually surface, so a couch player sees both.
    #[test]
    fn extract_shows_both_bindings() {
        assert_eq!(
            entry::<GcrControls>(Action::Extract)
                .unwrap()
                .glyphs(Device::KeyboardMouse),
            vec![
                Glyph::Icon("controls/keyboard_space.png"),
                Glyph::Icon("controls/mouse_left.png")
            ]
        );
        assert_eq!(
            entry::<GcrControls>(Action::Extract)
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
        let restart = entry::<GcrControls>(Action::Restart).unwrap();
        let quit = entry::<GcrControls>(Action::Quit).unwrap();
        assert_eq!(restart.pad.buttons, &[PadButton::Start]);
        assert!(!restart.pad.hold, "Restart is a tap");
        assert!(quit.pad.hold, "Quit is a hold");
        assert!(
            !quit.pad.buttons.contains(&PadButton::Start),
            "Quit must not share Start with Restart (a quit-hold would broadcast RESTART)"
        );
        let reveal = entry::<GcrControls>(Action::RevealControls).unwrap();
        assert_ne!(quit.pad.buttons, reveal.pad.buttons);
    }
}
