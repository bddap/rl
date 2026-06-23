//! The one data-driven control map for Giant Crab Rescue: [`CONTROL_MAP`] is the single
//! typed source of truth that drives BOTH the input handling (`net::render::gather_input`
//! etc.) AND the on-screen control legend (`net::render`'s reveal overlay + corner hint).
//! The live key/button reads ([`key_code_for`]/[`gamepad_buttons_for`]) and the displayed
//! glyphs ([`glyphs_for`]/[`legend`]) both derive from this one table, so the legend can
//! never drift from the real bindings — rebind in the table and both move together. This
//! whole-module invariant is the WHY behind the design; the per-item docs below don't
//! repeat it.
//!
//! Two layers, split by the `render` feature:
//! - The **pure core** ([`Action`], [`KeyboardBinding`], [`PadBinding`], [`CONTROL_MAP`],
//!   [`legend`], glyph resolution) has NO Bevy dependency, so it compiles and unit-tests in
//!   the no-feature build like the determinism core. Physical inputs are this module's OWN
//!   small enums ([`Key`], [`PadButton`], [`MouseInput`]), not Bevy's — keeping the table
//!   the single source even where Bevy's input types (render-only) don't exist.
//! - The **Bevy glue** (the `#[cfg(feature = "render")]` block at the bottom) is the ONLY
//!   place mapping [`Key`]/[`PadButton`] to Bevy's input types.
//!
//! Two caveats on the single-source guarantee:
//! - It covers the REBINDABLE discrete controls. The analog movers (sticks, D-pad) are
//!   read directly in `gather_input` and only their GLYPHS appear in the table — fixed, not
//!   rebindable, so they're deliberately outside the map's key/glyph round-trip.
//! - Nothing here touches the sim. A binding only decides WHICH key feeds an [`Action`];
//!   the value still funnels through `Input::new`'s fixed-point quantization downstream, so
//!   rebinding never changes the wire value and peers on different bindings stay
//!   bit-identical (the determinism contract at the top of [`crate::net::sim`]).

/// A controllable action in Giant Crab Rescue. The verbs the game exposes — the move
/// axes are split into four discrete directional actions (not one "move" entry) so each
/// has its own glyph in the legend (WASD shows as four keys). [`Look`](Action::Look) is
/// analog (mouse / right stick) and has no discrete key. This enum is the row key of
/// [`CONTROL_MAP`]; adding a verb here without a map row fails to compile (the table is
/// exhaustively matched in tests), so the map can't silently omit an action.
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

/// The active input device, used to pick which column of the map the legend shows and
/// which glyph set to render. Detected from recent input (see
/// [`crate::net::render`]'s device tracker): the last device that produced input wins, so
/// a couch player who picks up the pad sees pad glyphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// The desktop default; the legend starts here until a pad is touched.
    #[default]
    KeyboardMouse,
    Gamepad,
}

// ---------------------------------------------------------------------------
// Physical inputs — this module's own typed vocabulary (no Bevy dependency)
// ---------------------------------------------------------------------------

/// A keyboard key this game binds. A closed set (only the keys GCR uses), so the map and
/// the glyph table are exhaustive and a typo can't name a nonexistent key. Mapped to
/// Bevy's `KeyCode` in the `render` glue below — the ONLY translation point.
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

/// A mouse input this game binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseInput {
    /// Pointer motion → look.
    Motion,
    /// Left button → extract (mouse-only play).
    Left,
}

/// A gamepad control this game binds (Xbox-style names — the generic glyph set the MVP
/// ships; per-brand glyphs are a follow-up). A closed set like [`Key`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadButton {
    /// The "south" face button (Xbox A / PS Cross / Switch B-position).
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

// ---------------------------------------------------------------------------
// Bindings — what physical inputs drive an action, per device
// ---------------------------------------------------------------------------

/// How an [`Action`] is triggered on keyboard/mouse. An action is driven by a key, a
/// mouse input, or both (e.g. Extract = Space OR left-click); [`Look`](Action::Look) is
/// mouse-motion only. Modeled as explicit `Option`s rather than a free list so "no
/// keyboard binding" and "no mouse binding" are distinct, representable states and the
/// glyph picker is total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyboardBinding {
    /// The primary key, if this action has one. `None` for purely analog/mouse actions.
    pub key: Option<Key>,
    /// An alternate mouse trigger, if any (Look's motion, Extract's left-click).
    pub mouse: Option<MouseInput>,
    /// True for an action triggered by a sustained hold rather than a tap (RevealControls'
    /// hold-Tab). Lets the legend annotate "Hold" straight from the table — same role as
    /// [`PadBinding::hold`], so hold-ness is data per device, not a special case in
    /// [`legend`].
    pub hold: bool,
}

impl KeyboardBinding {
    const fn key(k: Key) -> Self {
        Self {
            key: Some(k),
            mouse: None,
            hold: false,
        }
    }
    const fn hold_key(k: Key) -> Self {
        Self {
            key: Some(k),
            mouse: None,
            hold: true,
        }
    }
    const fn key_or_mouse(k: Key, m: MouseInput) -> Self {
        Self {
            key: Some(k),
            mouse: Some(m),
            hold: false,
        }
    }
    const fn mouse(m: MouseInput) -> Self {
        Self {
            key: None,
            mouse: Some(m),
            hold: false,
        }
    }
}

/// How an [`Action`] is triggered on a gamepad. `primary` is the canonical control; some
/// actions have an alternate (Extract = South OR RightTrigger). A `hold` flag marks the
/// two press-and-hold actions (Quit, RevealControls) so the legend can annotate "hold".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadBinding {
    pub primary: PadButton,
    pub alternate: Option<PadButton>,
    /// True for actions triggered by a sustained hold rather than a tap/press (Quit's
    /// safety hold, RevealControls). Drives the "hold" prefix in the legend.
    pub hold: bool,
}

impl PadBinding {
    const fn press(primary: PadButton) -> Self {
        Self {
            primary,
            alternate: None,
            hold: false,
        }
    }
    const fn press_or(primary: PadButton, alternate: PadButton) -> Self {
        Self {
            primary,
            alternate: Some(alternate),
            hold: false,
        }
    }
    const fn hold(primary: PadButton) -> Self {
        Self {
            primary,
            alternate: None,
            hold: true,
        }
    }
}

/// One row of the control map: an action, its label, and its binding on each device. The
/// label is the human verb shown beside the glyph in the legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlEntry {
    pub action: Action,
    /// Short human label for the legend (e.g. "Forward", "Extract").
    pub label: &'static str,
    pub keyboard: KeyboardBinding,
    pub pad: PadBinding,
}

/// THE control map (see the module doc for the single-source invariant). One coherent
/// scheme, chosen deliberately rather than accreted:
///
/// - **Move**: WASD / left stick (+ D-pad), the universal FPS convention.
/// - **Look**: mouse / right stick.
/// - **Extract**: Space or left-click / pad South (A) or right trigger.
/// - **Restart**: R / pad Start (tap) — edge-triggered, rides the lockstep input stream.
/// - **Quit**: Esc / HOLD pad North (Y) ≥1s. A hold so a stray tap can't end the round; on
///   its OWN button (not Start) so quitting can't also fire the lockstep RESTART that a
///   shared Start tap would — Start is restart-only.
/// - **Reveal controls**: HOLD Tab / HOLD pad Back (the conventional info/menu button) —
///   show the overlay while held.
///
/// `Look` (analog) is intentionally in the map for the legend even though it isn't a
/// discrete key; the glyph picker shows the mouse / right-stick icon for it.
pub const CONTROL_MAP: [ControlEntry; 9] = [
    ControlEntry {
        action: Action::MoveForward,
        label: "Forward",
        keyboard: KeyboardBinding::key(Key::W),
        pad: PadBinding::press(PadButton::LeftStick),
    },
    ControlEntry {
        action: Action::MoveBack,
        label: "Back",
        keyboard: KeyboardBinding::key(Key::S),
        pad: PadBinding::press(PadButton::LeftStick),
    },
    ControlEntry {
        action: Action::StrafeLeft,
        label: "Strafe left",
        keyboard: KeyboardBinding::key(Key::A),
        pad: PadBinding::press(PadButton::LeftStick),
    },
    ControlEntry {
        action: Action::StrafeRight,
        label: "Strafe right",
        keyboard: KeyboardBinding::key(Key::D),
        pad: PadBinding::press(PadButton::LeftStick),
    },
    ControlEntry {
        action: Action::Look,
        label: "Look",
        keyboard: KeyboardBinding::mouse(MouseInput::Motion),
        pad: PadBinding::press(PadButton::RightStick),
    },
    ControlEntry {
        action: Action::Extract,
        label: "Extract",
        keyboard: KeyboardBinding::key_or_mouse(Key::Space, MouseInput::Left),
        pad: PadBinding::press_or(PadButton::South, PadButton::RightTrigger),
    },
    ControlEntry {
        action: Action::Restart,
        label: "Restart round",
        keyboard: KeyboardBinding::key(Key::R),
        pad: PadBinding::press(PadButton::Start),
    },
    ControlEntry {
        action: Action::Quit,
        label: "Quit",
        keyboard: KeyboardBinding::key(Key::Escape),
        pad: PadBinding::hold(PadButton::North),
    },
    ControlEntry {
        action: Action::RevealControls,
        label: "Controls",
        keyboard: KeyboardBinding::hold_key(Key::Tab),
        pad: PadBinding::hold(PadButton::Back),
    },
];

/// Look up an action's entry in [`CONTROL_MAP`]. Total over [`Action`] — every variant has
/// a row (proven by [`tests::action_list_is_exhaustive`]), so this never returns `None` for
/// a real action; the `Option` is just the lookup's honest shape.
pub fn entry(action: Action) -> Option<&'static ControlEntry> {
    CONTROL_MAP.iter().find(|e| e.action == action)
}

/// The single glyph for the key/button to HOLD to reveal the overlay, for the corner hint.
/// RevealControls binds exactly one control per device (hold Tab / hold Back), so its
/// `glyphs_for` is always a single element — the `expect` documents that invariant rather
/// than inventing a fallback binding (pinned by [`tests::reveal_glyph_follows_the_map`]).
pub fn reveal_glyph(device: Device) -> &'static str {
    glyphs_for(Action::RevealControls, device)
        .first()
        .copied()
        .expect("RevealControls is always bound in CONTROL_MAP")
}

// ---------------------------------------------------------------------------
// Glyph resolution — the display-selection logic (pure, unit-tested)
// ---------------------------------------------------------------------------

/// The asset path (under the Bevy asset root's `controls/` dir) of the glyph for a given
/// [`Key`]. These are the bundled Kenney Input Prompts (CC0) keyboard PNGs, renamed flat
/// into `assets/controls/`. One arm per [`Key`] variant — the compiler forces a glyph for
/// every key the map can name, so a binding can't reference a missing image.
pub fn key_glyph(key: Key) -> &'static str {
    match key {
        Key::W => "controls/keyboard_w.png",
        Key::A => "controls/keyboard_a.png",
        Key::S => "controls/keyboard_s.png",
        Key::D => "controls/keyboard_d.png",
        Key::R => "controls/keyboard_r.png",
        Key::Tab => "controls/keyboard_tab.png",
        Key::Space => "controls/keyboard_space.png",
        Key::Escape => "controls/keyboard_escape.png",
    }
}

/// The glyph asset path for a mouse input.
pub fn mouse_glyph(m: MouseInput) -> &'static str {
    match m {
        MouseInput::Motion => "controls/mouse_move.png",
        MouseInput::Left => "controls/mouse_left.png",
    }
}

/// The glyph asset path for a gamepad control (the generic Xbox-style set the MVP ships).
pub fn pad_glyph(button: PadButton) -> &'static str {
    match button {
        PadButton::South => "controls/xbox_button_a.png",
        PadButton::North => "controls/xbox_button_y.png",
        PadButton::RightTrigger => "controls/xbox_rt.png",
        PadButton::Start => "controls/xbox_button_menu.png",
        PadButton::Back => "controls/xbox_button_view.png",
        PadButton::LeftStick => "controls/xbox_stick_l.png",
        PadButton::RightStick => "controls/xbox_stick_r.png",
        PadButton::Dpad => "controls/xbox_dpad.png",
    }
}

/// The glyph asset paths to display for `action` on `device`, in order — the core
/// display-selection function. Keyboard/mouse: the key glyph and/or the mouse glyph the
/// binding names; pad: the primary plus any alternate (Extract → A + RT). Empty only if an
/// action has no binding on that device — no such case in the current map.
pub fn glyphs_for(action: Action, device: Device) -> Vec<&'static str> {
    let Some(e) = entry(action) else {
        return Vec::new();
    };
    match device {
        Device::KeyboardMouse => {
            let mut g = Vec::new();
            if let Some(k) = e.keyboard.key {
                g.push(key_glyph(k));
            }
            if let Some(m) = e.keyboard.mouse {
                g.push(mouse_glyph(m));
            }
            g
        }
        Device::Gamepad => {
            let mut g = vec![pad_glyph(e.pad.primary)];
            if let Some(alt) = e.pad.alternate {
                g.push(pad_glyph(alt));
            }
            // All four move rows bind the left stick; surface the D-pad (the alternate
            // digital mover) just once, on Forward, so its glyph isn't repeated four times.
            if action == Action::MoveForward {
                g.push(pad_glyph(PadButton::Dpad));
            }
            g
        }
    }
}

/// One ready-to-render legend line: the action's label, whether it's a hold, and the glyph
/// asset paths to show (in order). What the overlay turns into an icon row + text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegendLine {
    pub label: &'static str,
    /// True if this action is a hold (Quit/RevealControls) — the overlay prefixes "Hold".
    pub hold: bool,
    pub glyphs: Vec<&'static str>,
}

/// Build the full legend for the active device: one [`LegendLine`] per [`CONTROL_MAP`] row,
/// in map order, each with the device-appropriate glyphs and hold-ness. Iterating the map
/// directly (not a separate action list) is what keeps it the only enumeration source, so
/// the legend can't drift from the live bindings.
pub fn legend(device: Device) -> Vec<LegendLine> {
    CONTROL_MAP
        .iter()
        .map(|e| LegendLine {
            label: e.label,
            hold: match device {
                Device::KeyboardMouse => e.keyboard.hold,
                Device::Gamepad => e.pad.hold,
            },
            glyphs: glyphs_for(e.action, device),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bevy glue — the ONLY place this module's typed inputs meet Bevy's input API.
// Render-only: Bevy's `KeyCode`/`GamepadButton`/`MouseButton` exist only under the
// `render` feature (the full bevy_input). The live-input predicates below read
// CONTROL_MAP, so the client polls exactly the keys the legend shows — no drift.
// ---------------------------------------------------------------------------

#[cfg(feature = "render")]
mod bevy_glue {
    use super::*;
    use bevy::prelude::{GamepadButton, KeyCode, MouseButton};

    impl Key {
        /// This game key as Bevy's `KeyCode`. The sole translation point; everything else
        /// names [`Key`].
        pub fn key_code(self) -> KeyCode {
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
        /// button, so it has no `MouseButton` — hence the `Option`.
        pub fn mouse_button(self) -> Option<MouseButton> {
            match self {
                MouseInput::Left => Some(MouseButton::Left),
                MouseInput::Motion => None,
            }
        }
    }

    impl PadButton {
        /// This game pad control as Bevy's `GamepadButton`, for the discrete buttons.
        /// Sticks and the D-pad are read via their own axis/multi-button APIs (left/right
        /// stick vectors, the four D-pad directions), so they have no single `GamepadButton`
        /// — `None`, and the input code handles them explicitly.
        pub fn gamepad_button(self) -> Option<GamepadButton> {
            match self {
                PadButton::South => Some(GamepadButton::South),
                PadButton::North => Some(GamepadButton::North),
                PadButton::RightTrigger => Some(GamepadButton::RightTrigger2),
                PadButton::Start => Some(GamepadButton::Start),
                PadButton::Back => Some(GamepadButton::Select),
                PadButton::LeftStick | PadButton::RightStick | PadButton::Dpad => None,
            }
        }
    }

    /// The `KeyCode` bound to an action on the keyboard, if any. Reads [`CONTROL_MAP`], so
    /// the client polls the mapped key — change the map, the poll follows.
    pub fn key_code_for(action: Action) -> Option<KeyCode> {
        entry(action)
            .and_then(|e| e.keyboard.key)
            .map(Key::key_code)
    }

    /// The pad `GamepadButton`(s) that trigger an action (primary + any alternate), for the
    /// discrete buttons. Sticks/D-pad yield nothing (read via their dedicated stick/axis
    /// APIs). An iterator, not a `Vec`: every caller (`gather_input`, `exit_on_esc`,
    /// `update_controls_ui`) just does `.any(...)` each frame, so there's no reason to heap-
    /// allocate in the input path.
    pub fn gamepad_buttons_for(action: Action) -> impl Iterator<Item = GamepadButton> {
        entry(action)
            .into_iter()
            .flat_map(|e| std::iter::once(e.pad.primary).chain(e.pad.alternate))
            .filter_map(PadButton::gamepad_button)
    }
}

#[cfg(feature = "render")]
pub use bevy_glue::{gamepad_buttons_for, key_code_for};

#[cfg(test)]
mod tests {
    use super::*;

    /// Every [`Action`] variant has exactly one [`CONTROL_MAP`] row — the map is the single
    /// source. The exhaustive `match` makes the compiler force a new variant to declare
    /// whether it's mapped, so an `Action` added without a table row can't slip through; the
    /// runtime half then proves the variants that SHOULD be mapped each have exactly one row
    /// (no dup, which would make [`entry`] ambiguous).
    #[test]
    fn every_action_has_exactly_one_map_row() {
        // `expect_mapped` is exhaustive over Action: adding a variant fails to compile until
        // it's classified here, and everything in the current map must be present + unique.
        fn expect_mapped(a: Action) -> bool {
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
        for e in &CONTROL_MAP {
            assert!(expect_mapped(e.action), "{:?} mapped but unclassified", e.action);
            let n = CONTROL_MAP.iter().filter(|x| x.action == e.action).count();
            assert_eq!(n, 1, "{:?} appears {n} times in the map; want exactly 1", e.action);
        }
    }

    /// The no-drift property: the legend is built by iterating the SAME table the live input
    /// reads, so every legend line's glyphs equal `glyphs_for(action, device)` and its label
    /// equals the row's label. A future refactor that built the legend from a separate
    /// hand-written list would diverge here.
    #[test]
    fn legend_matches_the_map_for_both_devices() {
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            let lines = legend(device);
            assert_eq!(lines.len(), CONTROL_MAP.len(), "legend must cover every map row");
            for (line, entry) in lines.iter().zip(CONTROL_MAP.iter()) {
                assert_eq!(line.label, entry.label);
                assert_eq!(
                    line.glyphs,
                    glyphs_for(entry.action, device),
                    "legend glyphs for {:?} on {device:?} drifted from glyphs_for",
                    entry.action
                );
                assert!(!line.glyphs.is_empty(), "{:?} on {device:?} shows no glyph", entry.action);
            }
        }
    }

    /// Every glyph path the map can ever surface must point under `controls/`. Cheap guard
    /// that a renamed/relocated asset (or a typo) is caught: the deploy ships exactly the
    /// files these paths name, so a path outside `controls/` would silently 404 at runtime.
    #[test]
    fn all_glyph_paths_are_under_controls_dir() {
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            for e in &CONTROL_MAP {
                for path in glyphs_for(e.action, device) {
                    assert!(
                        path.starts_with("controls/") && path.ends_with(".png"),
                        "glyph path {path:?} for {:?}/{device:?} is malformed",
                        e.action
                    );
                }
            }
        }
    }

    /// The reveal control differs by device and is the binding the corner hint shows:
    /// keyboard hold-Tab, pad hold-Back. Pins the hint to the map — if the RevealControls row
    /// changes, the hint glyph changes with it — and that the reveal reads as a HOLD on both
    /// devices (data-driven on each binding's `hold` flag).
    #[test]
    fn reveal_glyph_follows_the_map() {
        assert_eq!(reveal_glyph(Device::KeyboardMouse), "controls/keyboard_tab.png");
        assert_eq!(reveal_glyph(Device::Gamepad), "controls/xbox_button_view.png");
        for device in [Device::KeyboardMouse, Device::Gamepad] {
            let reveal = legend(device).into_iter().find(|l| l.label == "Controls").unwrap();
            assert!(reveal.hold, "RevealControls must read as a hold on {device:?}");
        }
    }

    /// Extract is the one action with two glyphs on each device (key+mouse, A+RT) — pins
    /// that the alternate bindings actually surface in the display, so a couch player sees
    /// both the A button and the trigger. Would fail if the alternate were dropped.
    #[test]
    fn extract_shows_both_bindings() {
        assert_eq!(
            glyphs_for(Action::Extract, Device::KeyboardMouse),
            vec!["controls/keyboard_space.png", "controls/mouse_left.png"]
        );
        assert_eq!(
            glyphs_for(Action::Extract, Device::Gamepad),
            vec!["controls/xbox_button_a.png", "controls/xbox_rt.png"]
        );
    }

    /// The fix for the shared-Start bug: pad Quit must be on its OWN button, NOT Start.
    /// Restart is Start (tap); if Quit shared Start, the press edge that begins a quit-hold
    /// would also fire the lockstep RESTART, restarting the round for every peer on the way
    /// out. Pinning Quit ≠ Start (and that it's a hold) guards that regression.
    #[test]
    fn pad_quit_is_a_hold_on_its_own_button_not_start() {
        let restart = entry(Action::Restart).unwrap();
        let quit = entry(Action::Quit).unwrap();
        assert_eq!(restart.pad.primary, PadButton::Start);
        assert!(!restart.pad.hold, "Restart is a tap");
        assert!(quit.pad.hold, "Quit is a hold");
        assert_ne!(
            quit.pad.primary,
            PadButton::Start,
            "Quit must not share Start with Restart (a quit-hold would broadcast RESTART)"
        );
        // And it doesn't collide with the reveal button either.
        assert_ne!(quit.pad.primary, entry(Action::RevealControls).unwrap().pad.primary);
    }
}
