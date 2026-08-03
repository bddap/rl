//! Chord-code command input (rl#330): hold the chord MODIFIER (pad X / right mouse
//! button), tap a CODE on the d-pad (pad) or WASD (kb/m), and the command executes when
//! the modifier is RELEASED. One data-driven registry (code → action) serves every
//! surface and both devices, so command space is unbounded without spending buttons.
//!
//! No timeout, by design: the held modifier itself delimits the capture, so the state
//! machine takes no clock at all — a code can be entered as slowly as the player likes,
//! and "no timeout" is structural rather than a tuning constant. The empty code (tap the
//! modifier, no directions) is a code like any other: the registry decides whether it
//! means something, so a bare modifier tap can keep a legacy tap-verb alive.

use std::fmt::Debug;

/// One step of a chord code: a d-pad direction on pad, WASD on keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordDir {
    Up,
    Down,
    Left,
    Right,
}

/// Longest registerable code. A capture that runs past this is POISONED — its release is
/// a no-op, never a truncation: truncating would execute a command the player didn't
/// type, and a no-op is the self-correcting error (release, re-enter the code).
pub const MAX_CHORD_LEN: usize = 8;

/// Quit's code, shared by every surface (GCR and the demo) so it stays one muscle
/// memory — and one constant, so a margin-driven change (each surface's tests require
/// Quit ≥2 taps longer than its longest other code) can't land on one table and not
/// the other.
pub const QUIT_CODE: &[ChordDir] = &[
    ChordDir::Up,
    ChordDir::Up,
    ChordDir::Down,
    ChordDir::Down,
    ChordDir::Left,
    ChordDir::Right,
];

pub struct ChordEntry<A: 'static> {
    pub code: &'static [ChordDir],
    pub action: A,
    /// The player-facing name — the held-modifier context menu and the controls
    /// legend render it, so it lives with the code in the one table.
    pub label: &'static str,
}

/// The one data table mapping chord codes to commands. Code assignments are gameplay
/// data the owner tunes — keep every entry in the surface's single registry const,
/// reached through [`crate::controls::ControlScheme::chords`] so the install sites and
/// the well-formedness checks all read the same table.
#[derive(Clone, Copy)]
pub struct ChordRegistry<A: 'static>(&'static [ChordEntry<A>]);

impl<A: Copy + PartialEq + Debug> ChordRegistry<A> {
    pub const fn new(entries: &'static [ChordEntry<A>]) -> Self {
        Self(entries)
    }

    /// The command a completed code executes; `None` (an unregistered code) is a no-op.
    pub fn lookup(&self, code: &[ChordDir]) -> Option<A> {
        self.0.iter().find(|e| e.code == code).map(|e| e.action)
    }

    /// Every entry, in table order — the stage-4 legend renders from this.
    pub fn entries(&self) -> &'static [ChordEntry<A>] {
        self.0
    }

    /// Panics on a duplicate code, one longer than [`MAX_CHORD_LEN`] (unreachable —
    /// the capture poisons first), or a blank/duplicate label (the menu and legend
    /// would render indistinguishable lines). Call from the surface's scheme test.
    pub fn assert_well_formed(&self) {
        for (i, e) in self.0.iter().enumerate() {
            assert!(
                e.code.len() <= MAX_CHORD_LEN,
                "code for {:?} is longer than MAX_CHORD_LEN ({MAX_CHORD_LEN}) — unenterable",
                e.action
            );
            assert!(!e.label.is_empty(), "{:?} has an empty label", e.action);
            for other in &self.0[i + 1..] {
                assert!(
                    e.code != other.code,
                    "code {:?} is registered twice: {:?} and {:?}",
                    e.code,
                    e.action,
                    other.action
                );
                assert!(
                    e.label != other.label,
                    "label {:?} is used twice: {:?} and {:?}",
                    e.label,
                    e.action,
                    other.action
                );
            }
        }
    }
}

/// The capture state machine, pure and frame-stepped: feed it each frame's modifier state
/// and the directions just pressed, get the completed code back on the release edge.
/// A newtype over a private enum so the illegal combinations (a buffer while idle, a
/// poison flag with no capture) are unrepresentable, not merely unreached.
#[derive(Default)]
pub struct ChordCapture(CaptureState);

#[derive(Default)]
enum CaptureState {
    #[default]
    Idle,
    Capturing(Vec<ChordDir>),
    /// Overflowed past [`MAX_CHORD_LEN`]; the release is a no-op.
    Poisoned,
}

impl ChordCapture {
    /// Advance one frame. `modifier_down` is whether the modifier is held THIS frame;
    /// `taps` are the directions just pressed this frame. Taps buffer while a capture is
    /// live — INCLUDING the release frame, since on a fast flick the last tap and the
    /// release routinely land together, and dropping the tap would execute the typed
    /// code's PREFIX, a command the player didn't enter. A stray tap outside a capture
    /// is not code entry. Returns the completed code exactly once, on the frame the
    /// modifier is released — possibly empty (a bare tap; the registry decides what that
    /// means). An overflow-poisoned capture returns `None` on release.
    pub fn step(
        &mut self,
        modifier_down: bool,
        taps: impl IntoIterator<Item = ChordDir>,
    ) -> Option<Vec<ChordDir>> {
        use CaptureState::*;
        if modifier_down && matches!(self.0, Idle) {
            self.0 = Capturing(Vec::new());
        }
        if !matches!(self.0, Idle) {
            for d in taps {
                match &mut self.0 {
                    Capturing(buf) if buf.len() < MAX_CHORD_LEN => buf.push(d),
                    Capturing(_) => self.0 = Poisoned,
                    Idle | Poisoned => {}
                }
            }
        }
        if modifier_down {
            return None;
        }
        match std::mem::take(&mut self.0) {
            Idle | Poisoned => None,
            Capturing(buf) => Some(buf),
        }
    }

    /// Whether a capture is live — input readers suppress WASD/d-pad MOVEMENT while the
    /// player is typing a code on those same inputs (sticks stay live; analog is out of
    /// chord scope).
    pub fn capturing(&self) -> bool {
        !matches!(self.0, CaptureState::Idle)
    }

    /// Drop any half-typed code — for state transitions (menu → round) that a held
    /// modifier could otherwise smuggle buffered taps across.
    pub fn reset(&mut self) {
        self.0 = CaptureState::Idle;
    }

    /// The code typed so far in the live capture — the context menu filters on it.
    /// `None` while idle AND while poisoned: an overflowed capture has no honest
    /// prefix to filter by (the menu shows its cancel hint instead).
    pub fn entered(&self) -> Option<&[ChordDir]> {
        match &self.0 {
            CaptureState::Capturing(buf) => Some(buf),
            CaptureState::Idle | CaptureState::Poisoned => None,
        }
    }
}

/// The one glyph text per code step — the context menu and the controls legend both
/// render codes through it, so they can't drift. ASCII stand-ins for the d-pad, not
/// ↑↓←→: bevy's default font is an ASCII subset and real arrows render as tofu boxes.
pub fn dir_text(d: ChordDir, device: crate::controls::Device) -> &'static str {
    match (device, d) {
        (crate::controls::Device::Gamepad, ChordDir::Up) => "^",
        (crate::controls::Device::Gamepad, ChordDir::Down) => "v",
        (crate::controls::Device::Gamepad, ChordDir::Left) => "<",
        (crate::controls::Device::Gamepad, ChordDir::Right) => ">",
        (crate::controls::Device::KeyboardMouse, ChordDir::Up) => "W",
        (crate::controls::Device::KeyboardMouse, ChordDir::Down) => "S",
        (crate::controls::Device::KeyboardMouse, ChordDir::Left) => "A",
        (crate::controls::Device::KeyboardMouse, ChordDir::Right) => "D",
    }
}

/// The chord MODIFIER's legend glyph — X on pad, right-mouse-hold on kb/m (no mouse
/// art in the asset set, so a text chip). One source for every scheme's chord rows.
pub fn modifier_glyph(device: crate::controls::Device) -> crate::controls::Glyph {
    match device {
        crate::controls::Device::Gamepad => {
            crate::controls::Glyph::Icon("controls/xbox_button_x.png")
        }
        crate::controls::Device::KeyboardMouse => crate::controls::Glyph::Label("RMB"),
    }
}

/// The held-modifier context menu's text (rl#330 stage 3): every registry entry whose
/// code starts with the typed prefix, one line each — the options filter down as the
/// code is entered. Pure so the filtering is unit-testable headless; the render glue
/// only decides visibility and pushes this into a Text node. `entered == None` means
/// the capture overflowed ([`MAX_CHORD_LEN`]) — nothing can match, say so.
pub fn chord_menu_text<A: Copy + PartialEq + Debug>(
    registry: ChordRegistry<A>,
    entered: Option<&[ChordDir]>,
    device: crate::controls::Device,
) -> String {
    let dir = |d: ChordDir| dir_text(d, device);
    let code_text = |code: &[ChordDir]| code.iter().copied().map(dir).collect::<String>();
    let Some(prefix) = entered else {
        return "code too long - release to cancel".to_string();
    };
    let mut out = if prefix.is_empty() {
        "enter code".to_string()
    } else {
        code_text(prefix)
    };
    let mut any = false;
    for e in registry
        .entries()
        .iter()
        .filter(|e| e.code.starts_with(prefix))
    {
        any = true;
        // * marks the entry the release would execute right now (ASCII, see above).
        let here = if e.code == prefix { " *" } else { "" };
        out.push_str(&format!("\n{}  {}{here}", code_text(e.code), e.label));
    }
    if !any {
        out.push_str("\nno match - release to cancel");
    }
    out
}

#[cfg(feature = "render")]
mod glue {
    use super::*;
    use crate::controls::ControlScheme;
    use bevy::prelude::*;

    /// The chord modifier, fixed fleet-wide by the rl#330 directive: X held on pad,
    /// right-click held on kb/m. One source — surfaces read these, never re-declare.
    pub const CHORD_MODIFIER_PAD: GamepadButton = GamepadButton::West;
    pub const CHORD_MODIFIER_MOUSE: MouseButton = MouseButton::Right;

    fn pad_dir(b: GamepadButton) -> Option<ChordDir> {
        match b {
            GamepadButton::DPadUp => Some(ChordDir::Up),
            GamepadButton::DPadDown => Some(ChordDir::Down),
            GamepadButton::DPadLeft => Some(ChordDir::Left),
            GamepadButton::DPadRight => Some(ChordDir::Right),
            _ => None,
        }
    }

    /// The kb code-entry key for a direction — THE WASD↔dir map ([`key_dir`] is its
    /// inverse); scripted evidence input synthesizes presses through it.
    pub fn dir_key(d: ChordDir) -> KeyCode {
        match d {
            ChordDir::Up => KeyCode::KeyW,
            ChordDir::Down => KeyCode::KeyS,
            ChordDir::Left => KeyCode::KeyA,
            ChordDir::Right => KeyCode::KeyD,
        }
    }

    fn key_dir(k: KeyCode) -> Option<ChordDir> {
        [
            ChordDir::Up,
            ChordDir::Down,
            ChordDir::Left,
            ChordDir::Right,
        ]
        .into_iter()
        .find(|&d| dir_key(d) == k)
    }

    /// A surface's live chord state: the capture plus the command the just-finished
    /// code executed THIS frame (cleared next frame). The registry itself lives on the
    /// scheme ([`ControlScheme::chords`]) — no copy here to drift. Dispatch systems
    /// read [`Chords::executed`]; they must be scheduled after [`capture_chords`],
    /// which [`install_chords`] guarantees by running the capture in `PreUpdate`.
    #[derive(Resource)]
    pub struct Chords<S: ControlScheme> {
        capture: ChordCapture,
        executed: Option<S::Action>,
        code_entry: bool,
    }

    impl<S: ControlScheme> Default for Chords<S> {
        fn default() -> Self {
            Self {
                capture: ChordCapture::default(),
                executed: None,
                code_entry: false,
            }
        }
    }

    impl<S: ControlScheme> Chords<S> {
        /// Did a chord execute `action` this frame?
        pub fn executed(&self, action: S::Action) -> bool {
            self.executed == Some(action)
        }

        /// See [`ChordCapture::capturing`].
        pub fn capturing(&self) -> bool {
            self.capture.capturing()
        }

        /// Whether code entry touched THIS frame: the live capture OR its release
        /// frame. Dispatchers sharing the code-entry inputs (WASD/d-pad movement,
        /// joint picks) suppress on THIS, not [`Chords::capturing`] — release-frame
        /// taps join the code (see [`ChordCapture::step`]) while `capturing` is
        /// already false, so a capturing-gated reader double-fires that last tap.
        pub fn typing(&self) -> bool {
            self.code_entry
        }

        /// See [`ChordCapture::entered`].
        pub fn entered(&self) -> Option<&[ChordDir]> {
            self.capture.entered()
        }
    }

    /// The held-modifier context menu (rl#330 stage 3): visible only while a capture
    /// is live, its body is [`chord_menu_text`] — the options filter down per typed
    /// prefix, on pad and kb/m alike (the capture is device-agnostic; only the glyphs
    /// follow [`ActiveDevice`]).
    #[derive(Component)]
    pub struct ChordMenuRoot<S: ControlScheme>(std::marker::PhantomData<fn() -> S>);

    fn spawn_chord_menu<S: ControlScheme>(mut commands: Commands) {
        commands
            .spawn((
                // Anchored high with a smallish font: the unfiltered list is the whole
                // registry (22 lines in GCR after the stage-5 variant codes), and the
                // LAST rows are the disruptive verbs (Restart/Quit) — clipping the
                // bottom hides exactly the wrong lines. 24% + 22×19.2px + padding
                // clears a 720p window with margin; 32%/20pt did not.
                Node {
                    position_type: PositionType::Absolute,
                    right: Val::Px(24.0),
                    top: Val::Percent(24.0),
                    padding: UiRect::all(Val::Px(14.0)),
                    display: Display::None,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.82)),
                ChordMenuRoot::<S>(std::marker::PhantomData),
            ))
            .with_children(|root| {
                root.spawn((
                    Text::new(""),
                    TextFont {
                        font_size: 16.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.95, 0.95, 0.95)),
                ));
            });
    }

    fn update_chord_menu<S: ControlScheme>(
        chords: Res<Chords<S>>,
        device: Option<Res<crate::controls::ActiveDevice>>,
        mut roots: Query<(&mut Node, &Children), With<ChordMenuRoot<S>>>,
        mut texts: Query<&mut Text>,
    ) {
        // `Option` on the device: a surface may install chords without the controls
        // overlay (which owns ActiveDevice); the menu then just keeps default glyphs.
        let device = device.map(|d| d.get()).unwrap_or_default();
        for (mut node, children) in &mut roots {
            if !chords.capturing() {
                if node.display != Display::None {
                    node.display = Display::None;
                }
                continue;
            }
            node.display = Display::Flex;
            let body = chord_menu_text(S::chords(), chords.entered(), device);
            for &child in children {
                if let Ok(mut text) = texts.get_mut(child)
                    && text.0 != body
                {
                    text.0 = body.clone();
                }
            }
        }
    }

    /// Level OR this-frame edge. Level-only sampling of the modifier drops a
    /// press-and-release that lands inside one frame (a hitch, a low-FPS stretch) —
    /// `pressed` reads false on both frames and the empty-code verb intermittently
    /// dies. Seen as just-pressed, the capture opens now and completes on the next
    /// frame's release. (The pad reads the same expression inline — `Gamepad` is a
    /// component, not a `ButtonInput`.)
    fn held_or_tapped<T: Copy + Eq + std::hash::Hash + Send + Sync + 'static>(
        input: &ButtonInput<T>,
        button: T,
    ) -> bool {
        input.pressed(button) || input.just_pressed(button)
    }

    /// Drive the capture from the real inputs, once per frame in `PreUpdate` (after
    /// bevy's input update, before every dispatch system in `Update`).
    pub fn capture_chords<S: ControlScheme>(
        keys: Res<ButtonInput<KeyCode>>,
        mouse: Res<ButtonInput<MouseButton>>,
        pads: Query<&Gamepad>,
        mut chords: ResMut<Chords<S>>,
    ) {
        let modifier = held_or_tapped(&mouse, CHORD_MODIFIER_MOUSE)
            || pads
                .iter()
                .any(|gp| gp.pressed(CHORD_MODIFIER_PAD) || gp.just_pressed(CHORD_MODIFIER_PAD));
        let taps = keys
            .get_just_pressed()
            .filter_map(|&k| key_dir(k))
            .chain(
                pads.iter()
                    .flat_map(|gp| gp.get_just_pressed().filter_map(|&b| pad_dir(b))),
            )
            .collect::<Vec<_>>();
        let was_live = chords.capture.capturing();
        let code = chords.capture.step(modifier, taps);
        chords.code_entry = was_live || chords.capture.capturing();
        chords.executed = code.and_then(|c| S::chords().lookup(&c));
    }

    /// See [`ChordCapture::reset`] — schedule on the surface's round-entry transition
    /// (e.g. `OnEnter(Playing)`) so a modifier held across it can't smuggle menu-time
    /// taps into the round.
    pub fn reset_chords<S: ControlScheme>(mut chords: ResMut<Chords<S>>) {
        chords.capture.reset();
        chords.executed = None;
        chords.code_entry = false;
    }

    /// The one wiring of chord input onto an app: the resource plus the capture system.
    /// The registry comes from the scheme itself ([`ControlScheme::chords`]), so every
    /// install site of a surface wires the same table by construction.
    pub fn install_chords<S: ControlScheme>(app: &mut App) {
        app.init_resource::<Chords<S>>()
            .add_systems(
                PreUpdate,
                capture_chords::<S>.after(bevy::input::InputSystems),
            )
            .add_systems(Startup, spawn_chord_menu::<S>)
            .add_systems(Update, update_chord_menu::<S>);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The menu's kb glyphs must be the keys the capture actually reads: both
        /// derive from [`dir_key`] (this test) or would silently drift from it.
        #[test]
        fn menu_kb_glyphs_match_the_code_entry_keys() {
            for d in [
                ChordDir::Up,
                ChordDir::Down,
                ChordDir::Left,
                ChordDir::Right,
            ] {
                // "KeyW" → "W": the header line of a menu over an empty registry is
                // exactly the typed prefix, one glyph here.
                let key_letter = format!("{:?}", dir_key(d)).replace("Key", "");
                let text = chord_menu_text(
                    ChordRegistry::<()>::new(&[]),
                    Some(&[d]),
                    crate::controls::Device::KeyboardMouse,
                );
                assert_eq!(text.lines().next(), Some(key_letter.as_str()));
            }
        }

        /// The modifier's INPUT constants and its legend GLYPH live apart (the glyph
        /// must stay outside this render-gated glue for the headless legend) — this
        /// is their drift alarm: retune the modifier and this fails until the glyph
        /// follows.
        #[test]
        fn modifier_glyph_matches_the_modifier_inputs() {
            use crate::controls::{Device, Glyph};
            assert_eq!(CHORD_MODIFIER_PAD, GamepadButton::West);
            assert_eq!(
                modifier_glyph(Device::Gamepad),
                Glyph::Icon("controls/xbox_button_x.png")
            );
            assert_eq!(CHORD_MODIFIER_MOUSE, MouseButton::Right);
            assert_eq!(modifier_glyph(Device::KeyboardMouse), Glyph::Label("RMB"));
        }

        /// Pins the fix for the sub-frame modifier tap: press+release inside one frame
        /// leaves `pressed` false while `just_pressed` is still set — a level-only
        /// sample would drop the tap and the empty-code verb would intermittently die.
        #[test]
        fn a_sub_frame_modifier_tap_still_reads_as_down() {
            let mut input = ButtonInput::<MouseButton>::default();
            input.press(CHORD_MODIFIER_MOUSE);
            input.release(CHORD_MODIFIER_MOUSE);
            assert!(!input.pressed(CHORD_MODIFIER_MOUSE));
            assert!(held_or_tapped(&input, CHORD_MODIFIER_MOUSE));
            input.clear();
            assert!(!held_or_tapped(&input, CHORD_MODIFIER_MOUSE));
        }
    }
}

#[cfg(feature = "render")]
pub use glue::{
    CHORD_MODIFIER_MOUSE, CHORD_MODIFIER_PAD, Chords, capture_chords, dir_key, install_chords,
    reset_chords,
};

#[cfg(test)]
mod tests {
    use super::ChordDir::*;
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Cmd {
        TapVerb,
        One,
        Two,
    }

    const REG: ChordRegistry<Cmd> = ChordRegistry::new(&[
        ChordEntry {
            code: &[],
            action: Cmd::TapVerb,
            label: "Tap verb",
        },
        ChordEntry {
            code: &[Up],
            action: Cmd::One,
            label: "One",
        },
        ChordEntry {
            code: &[Up, Down],
            action: Cmd::Two,
            label: "Two",
        },
    ]);

    #[test]
    fn code_buffers_while_held_and_executes_on_release() {
        let mut c = ChordCapture::default();
        assert_eq!(c.step(true, [Up]), None, "no execute while held");
        assert_eq!(c.step(true, [Down]), None);
        assert_eq!(c.step(false, []), Some(vec![Up, Down]));
        assert_eq!(REG.lookup(&[Up, Down]), Some(Cmd::Two));
    }

    #[test]
    fn release_frame_taps_join_the_code_not_its_prefix() {
        // A fast flick lands the last tap and the release on the same frame; the code
        // must be [Up], never the empty prefix (which is a REGISTERED command).
        let mut c = ChordCapture::default();
        assert_eq!(c.step(true, []), None);
        assert_eq!(c.step(false, [Up]), Some(vec![Up]));
    }

    #[test]
    fn reset_drops_a_half_typed_code() {
        let mut c = ChordCapture::default();
        c.step(true, [Up]);
        c.reset();
        assert!(!c.capturing());
        assert_eq!(c.step(false, []), None, "reset capture must not fire");
    }

    #[test]
    fn release_fires_exactly_once() {
        let mut c = ChordCapture::default();
        c.step(true, [Up]);
        assert!(c.step(false, []).is_some());
        assert_eq!(c.step(false, []), None, "no re-fire while idle");
    }

    #[test]
    fn bare_tap_yields_the_empty_code() {
        let mut c = ChordCapture::default();
        assert_eq!(c.step(true, []), None);
        assert_eq!(c.step(false, []), Some(vec![]));
        assert_eq!(REG.lookup(&[]), Some(Cmd::TapVerb));
    }

    #[test]
    fn directions_without_the_modifier_are_not_code_entry() {
        let mut c = ChordCapture::default();
        assert_eq!(c.step(false, [Up, Up]), None);
        c.step(true, [Down]);
        assert_eq!(
            c.step(false, []),
            Some(vec![Down]),
            "only modifier-held taps buffer"
        );
    }

    #[test]
    fn unregistered_code_is_a_no_op() {
        assert_eq!(REG.lookup(&[Down, Down, Left]), None);
    }

    #[test]
    fn overflow_poisons_the_capture_instead_of_truncating() {
        let mut c = ChordCapture::default();
        c.step(true, vec![Up; MAX_CHORD_LEN + 1]);
        assert_eq!(c.step(false, []), None, "poisoned release is a no-op");
        // The next capture is clean.
        c.step(true, [Up]);
        assert_eq!(c.step(false, []), Some(vec![Up]));
    }

    #[test]
    fn capture_has_no_timeout_only_release_completes_it() {
        // Structural: step() takes no clock. Idle held frames change nothing.
        let mut c = ChordCapture::default();
        c.step(true, [Up]);
        for _ in 0..10_000 {
            assert_eq!(c.step(true, []), None);
            assert!(c.capturing());
        }
        assert_eq!(c.step(false, []), Some(vec![Up]));
    }

    #[test]
    fn capturing_flags_the_held_span_for_movement_suppression() {
        let mut c = ChordCapture::default();
        assert!(!c.capturing());
        c.step(true, []);
        assert!(c.capturing());
        c.step(false, []);
        assert!(!c.capturing());
    }

    #[test]
    fn menu_filters_down_as_the_code_is_entered() {
        use crate::controls::Device;
        // Nothing typed yet: every option is on the table; the bare-tap verb is the
        // one the release would execute (*).
        let all = chord_menu_text(REG, Some(&[]), Device::Gamepad);
        assert_eq!(all, "enter code\n  Tap verb *\n^  One\n^v  Two");
        // One tap in: only the ^-family remains, One is now live.
        let up = chord_menu_text(REG, Some(&[Up]), Device::Gamepad);
        assert_eq!(up, "^\n^  One *\n^v  Two");
        // Keyboard glyphs mirror the same filtering (WASD is the kb code entry).
        let up_kb = chord_menu_text(REG, Some(&[Up]), Device::KeyboardMouse);
        assert_eq!(up_kb, "W\nW  One *\nWS  Two");
        // A dead-end prefix says so instead of listing nothing.
        let dead = chord_menu_text(REG, Some(&[Down]), Device::Gamepad);
        assert_eq!(dead, "v\nno match - release to cancel");
        // A poisoned capture (entered = None) has no honest prefix to filter by.
        let poisoned = chord_menu_text(REG, None, Device::Gamepad);
        assert_eq!(poisoned, "code too long - release to cancel");
    }

    #[test]
    fn well_formedness_rejects_duplicates() {
        REG.assert_well_formed();
        let dup: ChordRegistry<Cmd> = ChordRegistry::new(&[
            ChordEntry {
                code: &[Up],
                action: Cmd::One,
                label: "One",
            },
            ChordEntry {
                code: &[Up],
                action: Cmd::Two,
                label: "Two",
            },
        ]);
        assert!(std::panic::catch_unwind(|| dup.assert_well_formed()).is_err());
    }
}
