//! Giant Crab Rescue's control scheme: GCR's action set + input vocabulary + glyph art + the
//! one [`BINDINGS`] table + the per-context row lists ([`FOOT_ROWS`]/[`PLANE_ROWS`]),
//! implemented as a [`ControlScheme`] for the reusable overlay framework in
//! [`crate::controls`]. The framework renders the context-sensitive legend + hold-to-reveal
//! overlay from this; the live input ([`crate::net::render`]'s `gather_input`/`quit_game`)
//! reads the SAME [`BINDINGS`] via [`key_code_for`]/[`gamepad_buttons_for`], so the keys the
//! client polls are exactly the keys the legend shows — no drift.
//!
//! Why two tables, not one. The KEY for an action is context-independent (W always feeds the
//! forward axis), so the binding lives once in [`BINDINGS`]. What CHANGES between contexts is
//! the LABEL and which actions are relevant: on foot W is "Forward"; piloting the plane the
//! same W is "Throttle up" (the plane sim reads `move_forward` as throttle — see
//! [`crate::net::sim`]'s `step_plane`). So each context is a row list naming the actions it
//! shows with the label correct THERE. The legend joins the rows with the bindings, so the
//! displayed KEY can't drift from the polled key; the label's meaning (e.g. "Throttle up" =
//! what `move_forward` does in flight) is a hand-maintained description, pinned where it's
//! non-obvious by a test (the plane-pitch-sign test).
//!
//! Two caveats on the single-source guarantee:
//! - It covers the REBINDABLE discrete controls. The analog movers (sticks, mouse motion,
//!   D-pad digital move) are read directly in `gather_input` and only their GLYPHS appear
//!   in the bindings — fixed, not rebindable, so outside the key/glyph round-trip.
//! - Nothing here touches the sim. A binding only decides WHICH key feeds an [`Action`];
//!   the value still funnels through `Input::new`'s fixed-point quantization downstream, so
//!   rebinding never changes the wire value and peers on different bindings stay
//!   bit-identical (the determinism contract atop [`crate::net::sim`]).

use crate::controls::{Binding, ContextRow, ControlScheme, Glyph, KbBinding, PadBinding};

/// GCR's control scheme — the type parameter the framework is instantiated with. A
/// zero-size marker; all of GCR's control data hangs off its [`ControlScheme`] impl.
pub struct GcrControls;

/// The active control context in Giant Crab Rescue. On foot the move keys walk the avatar;
/// piloting the plane the same keys fly it (throttle/pitch/yaw — see the module docs), so the
/// legend re-labels them. A new vehicle type is a new variant here + its own row list; the
/// HUD then names and labels it automatically. `OnFoot` is the default (spawn) context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GcrContext {
    #[default]
    OnFoot,
    /// Flying the single-player plane (client-local; see [`crate::net::render`]'s `LocalVehicle`).
    Plane,
}

/// A controllable action in Giant Crab Rescue. The move axes are split into four discrete
/// directional actions (not one "move") so each has its own glyph in the legend (WASD shows
/// as four keys). [`Look`](Action::Look) is analog (mouse / right stick). This enum is the
/// row key of [`BINDINGS`]; the per-app test forces every variant to have exactly one binding.
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
    /// Enter the vehicle/plane when on foot, or exit back to foot when piloting
    /// (single-player). A tap-toggle handled entirely in the windowed client's play
    /// layer ([`crate::net::render`]) — like [`Quit`](Action::Quit) it never crosses the
    /// wire or the deterministic sim, so the lockstep crab game is unaffected.
    EnterExit,
    /// Hold to reveal the full control overlay; release to hide. Pure client UI.
    RevealControls,
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
    type Context = GcrContext;

    fn bindings() -> &'static [Binding<Self>] {
        &BINDINGS
    }

    fn contexts() -> &'static [GcrContext] {
        &[GcrContext::OnFoot, GcrContext::Plane]
    }

    fn context_rows(ctx: GcrContext) -> &'static [ContextRow<Self>] {
        match ctx {
            GcrContext::OnFoot => &FOOT_ROWS,
            GcrContext::Plane => &PLANE_ROWS,
        }
    }

    fn context_label(ctx: GcrContext) -> &'static str {
        match ctx {
            GcrContext::OnFoot => "On foot",
            GcrContext::Plane => "Piloting plane",
        }
    }

    fn context_id(ctx: GcrContext) -> &'static str {
        match ctx {
            GcrContext::OnFoot => "foot",
            GcrContext::Plane => "plane",
        }
    }

    fn context_from_id(id: &str) -> Option<GcrContext> {
        match id {
            "foot" | "onfoot" => Some(GcrContext::OnFoot),
            "plane" => Some(GcrContext::Plane),
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
            Key::Tab => "controls/keyboard_tab.png",
            Key::Space => "controls/keyboard_space.png",
            Key::Escape => "controls/keyboard_escape.png",
        })
    }

    fn pad_glyph(pad: PadButton) -> Glyph {
        Glyph::Icon(match pad {
            PadButton::South => "controls/xbox_button_a.png",
            PadButton::North => "controls/xbox_button_y.png",
            PadButton::West => "controls/xbox_button_x.png",
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

/// THE GCR binding table — one row per action, the key/button that triggers it (no labels;
/// those are per-context, see [`FOOT_ROWS`]/[`PLANE_ROWS`]). The deliberate choices:
/// - **Move**: WASD / left stick (+ D-pad shown once, on Forward), the FPS convention.
/// - **Look**: mouse / right stick.
/// - **Extract**: Space or left-click / pad South (A) or right trigger.
/// - **Restart**: R / pad Start (tap) — edge-triggered, rides the lockstep input stream.
/// - **Quit**: Esc / HOLD pad North (Y) ≥1s. A hold so a stray tap can't end the round; on
///   its OWN button (not Start) so quitting can't also fire the lockstep RESTART a shared
///   Start tap would — Start is restart-only.
/// - **Enter/exit vehicle**: E / pad West (X) — tap to board a plane or step out
///   (single-player; a client-local toggle, never on the wire).
/// - **Reveal controls**: HOLD Tab / HOLD pad Back — show the overlay while held.
pub const BINDINGS: [Binding<GcrControls>; 10] = [
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
        action: Action::RevealControls,
        keyboard: KbBinding::hold(&[Key::Tab], &[]),
        pad: PadBinding::hold(&[PadButton::Back]),
    },
];

/// The ON-FOOT context: the full ground control set, in legend order. The move keys walk the
/// avatar; `EnterExit` boards the plane.
pub const FOOT_ROWS: [ContextRow<GcrControls>; 10] = [
    ContextRow { action: Action::MoveForward, label: "Forward" },
    ContextRow { action: Action::MoveBack, label: "Back" },
    ContextRow { action: Action::StrafeLeft, label: "Strafe left" },
    ContextRow { action: Action::StrafeRight, label: "Strafe right" },
    ContextRow { action: Action::Look, label: "Look" },
    ContextRow { action: Action::Extract, label: "Extract" },
    ContextRow { action: Action::EnterExit, label: "Enter plane" },
    ContextRow { action: Action::Restart, label: "Restart round" },
    ContextRow { action: Action::Quit, label: "Quit" },
    ContextRow { action: Action::RevealControls, label: "Controls" },
];

/// The PILOTING-PLANE context: the SAME move keys, re-labeled for flight to match what the
/// plane sim ACTUALLY does with them (`step_plane` in [`crate::net::sim`]): `move_forward` →
/// throttle, `move_strafe` → pitch, `look_yaw` → yaw. `Extract` is omitted — the foot player
/// feeds the sim neutral input while piloting, so the pickup button is inert in the air.
/// `EnterExit` now reads "Exit plane".
///
/// Pitch sign — the subtle part the labels MUST get right: `gather_input` negates the strafe
/// axis once (`render.rs`'s `pending.strafe = -strafe`, the screen-right↔sim-X reconcile), so
/// A (`StrafeLeft`) reaches the sim as POSITIVE `move_strafe` and D (`StrafeRight`) as
/// negative. `step_plane` makes positive `move_strafe` nose-UP. Net: **A climbs, D dives** —
/// so the labels ride those actions, not the screen-intuitive ones. (Pinned by the sim-side
/// `step_plane_positive_strafe_climbs` test + the render-side negation; reassigning the keys
/// for FPS feel is a separate taste call.)
pub const PLANE_ROWS: [ContextRow<GcrControls>; 9] = [
    ContextRow { action: Action::MoveForward, label: "Throttle up" },
    ContextRow { action: Action::MoveBack, label: "Throttle down" },
    ContextRow { action: Action::StrafeLeft, label: "Pitch up (climb)" },
    ContextRow { action: Action::StrafeRight, label: "Pitch down (dive)" },
    ContextRow { action: Action::Look, label: "Yaw / turn" },
    ContextRow { action: Action::EnterExit, label: "Exit plane" },
    ContextRow { action: Action::Restart, label: "Restart round" },
    ContextRow { action: Action::Quit, label: "Quit" },
    ContextRow { action: Action::RevealControls, label: "Controls" },
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
    use crate::controls::{ControlInput, binding};
    use bevy::prelude::{GamepadButton, KeyCode, MouseButton};

    impl Key {
        fn key_code(self) -> KeyCode {
            match self {
                Key::W => KeyCode::KeyW,
                Key::A => KeyCode::KeyA,
                Key::S => KeyCode::KeyS,
                Key::D => KeyCode::KeyD,
                Key::E => KeyCode::KeyE,
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
                PadButton::West => Some(GamepadButton::West),
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

    /// The `KeyCode` bound to an action on the keyboard, if any. Reads [`BINDINGS`], so the
    /// client polls the mapped key — change the table, the poll follows. Context-independent:
    /// the key for an action is the same wherever it's shown (only its label changes), so the
    /// dispatch needs no context. Polls the FIRST bound key only while the legend renders
    /// every bound key: identical today (every GCR action binds exactly one key), but a future
    /// multi-key action would poll one key yet show all — generalize this (and
    /// `gamepad_buttons_for`) before binding more than one.
    pub fn key_code_for(action: Action) -> Option<KeyCode> {
        binding::<GcrControls>(action)
            .and_then(|b| b.keyboard.keys.first().copied())
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
pub use bevy_glue::{gamepad_buttons_for, key_code_for};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{
        Device, Glyph, assert_scheme_well_formed, binding, legend, reveal_glyph,
    };

    const ALL_ACTIONS: [Action; 10] = [
        Action::MoveForward,
        Action::MoveBack,
        Action::StrafeLeft,
        Action::StrafeRight,
        Action::Look,
        Action::Extract,
        Action::Restart,
        Action::Quit,
        Action::EnterExit,
        Action::RevealControls,
    ];

    const ALL_CONTEXTS: [GcrContext; 2] = [GcrContext::OnFoot, GcrContext::Plane];

    /// Every [`Action`] / [`GcrContext`] is exhaustively classified (so a new variant can't be
    /// added without declaring it here), and the framework proves each action has exactly one
    /// binding, each context shows only bound actions + the reveal control, and the reveal
    /// control resolves to a hint glyph. The compiler forces the `match`es to cover every
    /// variant; `assert_scheme_well_formed` does the runtime checks.
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
                | Action::RevealControls => true,
            }
        }
        fn ctx_classified(c: GcrContext) -> bool {
            match c {
                GcrContext::OnFoot | GcrContext::Plane => true,
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
                assert_eq!(lines.len(), rows.len(), "{ctx:?}/{device:?}: every row shows");
                for (line, row) in lines.iter().zip(rows.iter()) {
                    let b = binding::<GcrControls>(row.action).unwrap();
                    assert_eq!(line.label, row.label);
                    assert_eq!(line.glyphs, b.glyphs(device));
                    assert!(!line.glyphs.is_empty(), "{:?} shows no glyph", row.action);
                }
            }
        }
    }

    /// The context fix itself: foot and plane are DIFFERENT legends — the move keys re-label
    /// for flight and the plane drops Extract — so entering the plane visibly changes the HUD.
    /// This is the bug the job fixes (a static legend showed "Strafe left/right" while flying).
    #[test]
    fn plane_context_relabels_and_differs_from_foot() {
        let foot = legend::<GcrControls>(GcrContext::OnFoot, Device::KeyboardMouse);
        let plane = legend::<GcrControls>(GcrContext::Plane, Device::KeyboardMouse);
        let labels = |ls: &[crate::controls::LegendLine]| {
            ls.iter().map(|l| l.label).collect::<Vec<_>>()
        };
        assert_ne!(labels(&foot), labels(&plane), "the legend must change per context");
        // Flight labels reflect what `step_plane` does with each key.
        assert!(plane.iter().any(|l| l.label == "Throttle up"));
        assert!(plane.iter().any(|l| l.label == "Pitch up (climb)"));
        assert!(plane.iter().any(|l| l.label == "Yaw / turn"));
        assert!(plane.iter().any(|l| l.label == "Exit plane"));
        // The on-foot ground labels are gone in flight (no misleading "Strafe"/"Forward").
        assert!(!plane.iter().any(|l| l.label == "Strafe left"));
        assert!(!plane.iter().any(|l| l.label == "Forward"));
        assert!(!plane.iter().any(|l| l.label == "Extract"), "no Extract while piloting");
        // The context labels name the active vehicle.
        assert_eq!(GcrControls::context_label(GcrContext::OnFoot), "On foot");
        assert_eq!(GcrControls::context_label(GcrContext::Plane), "Piloting plane");
    }

    /// The throttle/pitch keys in the plane context are the SAME physical keys as the foot
    /// move keys (W/S, D/A) — proof that the re-label rides one binding, not a parallel one.
    #[test]
    fn plane_reuses_the_foot_move_keys() {
        let throttle_up = binding::<GcrControls>(Action::MoveForward).unwrap();
        assert_eq!(throttle_up.keyboard.keys, &[Key::W]);
        let climb = binding::<GcrControls>(Action::StrafeRight).unwrap();
        assert_eq!(climb.keyboard.keys, &[Key::D]);
        // Both contexts reference these same actions (foot as Forward/Strafe, plane re-labeled).
        assert!(FOOT_ROWS.iter().any(|r| r.action == Action::MoveForward));
        assert!(PLANE_ROWS.iter().any(|r| r.action == Action::MoveForward));
    }

    /// The screenshot context override round-trips the ids the evidence harness uses.
    #[test]
    fn context_from_id_round_trips() {
        assert_eq!(GcrControls::context_from_id("foot"), Some(GcrContext::OnFoot));
        assert_eq!(GcrControls::context_from_id("plane"), Some(GcrContext::Plane));
        assert_eq!(GcrControls::context_from_id("nope"), None);
    }

    /// Every glyph the bindings surface is a bundled icon under `controls/` (GCR ships no text
    /// labels). A renamed/relocated asset (or a typo) would 404 at runtime; this catches it.
    #[test]
    fn all_glyph_paths_are_under_controls_dir() {
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for b in &BINDINGS {
                for glyph in b.glyphs(device) {
                    match glyph {
                        Glyph::Icon(path) => assert!(
                            path.starts_with("controls/") && path.ends_with(".png"),
                            "glyph path {path:?} for {:?}/{device:?} is malformed",
                            b.action
                        ),
                        Glyph::Label(l) => panic!("GCR uses only icons; got label {l:?}"),
                    }
                }
            }
        }
    }

    /// The reveal control differs by device and is the hint glyph: keyboard hold-Tab, pad
    /// hold-Back. Pins the hint to the bindings and that reveal reads as a HOLD on both
    /// devices, in every context (it's in each context's rows).
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
                assert!(reveal.hold, "RevealControls must read as a hold in {ctx:?}/{device:?}");
            }
        }
    }

    /// Extract is the one action with two glyphs on each device (key+mouse, A+RT) — pins
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
