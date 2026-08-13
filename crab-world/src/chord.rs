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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChordDir {
    Up,
    Down,
    Left,
    Right,
}

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
    /// The player-facing name — the combo map (rl#358) renders it for discovered
    /// codes, so it lives with the code in the one table.
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
        self.entry(code).map(|e| e.action)
    }

    /// The full entry for a code — THE one code matcher; readers needing the label
    /// (the combo map, rl#358) go through this rather than re-scanning entries.
    pub fn entry(&self, code: &[ChordDir]) -> Option<&'static ChordEntry<A>> {
        self.0.iter().find(|e| e.code == code)
    }

    /// Every entry, in table order — the stage-4 legend renders from this.
    pub fn entries(&self) -> &'static [ChordEntry<A>] {
        self.0
    }

    /// Panics on a duplicate code or a blank/duplicate label (the combo map would
    /// render indistinguishable rooms). Call from the surface's scheme test. Code
    /// length is deliberately unbounded — deep codes are gameplay (rl#380). Prefix
    /// pairs (`L` ⊂ `LR`) are legal — release does an exact lookup — but a prefix
    /// code executes on an early release mid-entry, so registries must keep prefix
    /// codes benign; the destructive cases carry their own targeted guards (no code
    /// behind ExitVehicle's `vv`, Quit's ≥2-tap margin).
    pub fn assert_well_formed(&self) {
        for (i, e) in self.0.iter().enumerate() {
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
/// `Some(buffer)` is a live capture, `None` is idle; the newtype keeps the buffer
/// private (readers go through [`Self::entered`]). Code length is unbounded by design
/// (rl#380): deep codes are gameplay, and the buffer only grows by one press per
/// player input.
#[derive(Default)]
pub struct ChordCapture(Option<Vec<ChordDir>>);

impl ChordCapture {
    /// Advance one frame. `modifier_down` is whether the modifier is held THIS frame;
    /// `taps` are the directions just pressed this frame. Taps buffer while a capture is
    /// live — INCLUDING the release frame, since on a fast flick the last tap and the
    /// release routinely land together, and dropping the tap would execute the typed
    /// code's PREFIX, a command the player didn't enter. A stray tap outside a capture
    /// is not code entry. Returns the completed code exactly once, on the frame the
    /// modifier is released — possibly empty (a bare tap; the registry decides what that
    /// means).
    pub fn step(
        &mut self,
        modifier_down: bool,
        taps: impl IntoIterator<Item = ChordDir>,
    ) -> Option<Vec<ChordDir>> {
        if modifier_down && self.0.is_none() {
            self.0 = Some(Vec::new());
        }
        if let Some(buf) = &mut self.0 {
            buf.extend(taps);
        }
        if modifier_down {
            return None;
        }
        self.0.take()
    }

    /// Whether a capture is live — input readers suppress WASD/d-pad MOVEMENT while the
    /// player is typing a code on those same inputs (sticks stay live; analog is out of
    /// chord scope).
    pub fn capturing(&self) -> bool {
        self.0.is_some()
    }

    /// Drop any half-typed code — for state transitions (menu → round) that a held
    /// modifier could otherwise smuggle buffered taps across.
    pub fn reset(&mut self) {
        self.0 = None;
    }

    /// The code typed so far in the live capture — the combo map zooms to it.
    /// `None` while idle.
    pub fn entered(&self) -> Option<&[ChordDir]> {
        self.0.as_deref()
    }
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
        events: ChordEvents,
    }

    impl<S: ControlScheme> Default for Chords<S> {
        fn default() -> Self {
            Self {
                capture: ChordCapture::default(),
                executed: None,
                code_entry: false,
                events: ChordEvents::default(),
            }
        }
    }

    /// What the combo system CONSUMED this frame, for the d-pad instrument (rl#359):
    /// the code path including this frame's taps, how many of its tail arrived this
    /// frame, and whether a code completed (and was a registered one). Derived, not
    /// tracked: recomputed each frame from the capture's own transitions, so it can't
    /// drift from what the capture actually swallowed.
    #[derive(Default, PartialEq, Debug)]
    pub(super) struct ChordEvents {
        path: Vec<ChordDir>,
        fresh: usize,
        resolution: Option<bool>,
    }

    impl ChordEvents {
        /// `prev_depth`: the live capture's length BEFORE this frame's step.
        /// `completed`: the code the step returned (release frame) with its
        /// registry verdict. `live`: the capture's buffer after the step.
        pub(super) fn from_frame(
            prev_depth: usize,
            completed: Option<(Vec<ChordDir>, bool)>,
            live: Option<&[ChordDir]>,
        ) -> Self {
            let (path, resolution) = match (completed, live) {
                (Some((code, accepted)), _) => (code, Some(accepted)),
                (None, Some(buf)) => (buf.to_vec(), None),
                (None, None) => return Self::default(),
            };
            Self {
                fresh: path.len() - prev_depth,
                path,
                resolution,
            }
        }

        pub(super) fn presses(&self) -> impl Iterator<Item = (&[ChordDir], ChordDir)> {
            let first = self.path.len() - self.fresh;
            (first..self.path.len()).map(|i| (&self.path[..i], self.path[i]))
        }

        pub(super) fn resolution(&self) -> Option<(&[ChordDir], bool)> {
            self.resolution
                .map(|accepted| (self.path.as_slice(), accepted))
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

        /// The taps the combo system consumed THIS frame, each with the code prefix
        /// it landed after — the d-pad instrument (rl#359) maps every one to a note.
        /// Release-frame taps are included (they join the code; see
        /// [`ChordCapture::step`]).
        pub fn presses(&self) -> impl Iterator<Item = (&[ChordDir], ChordDir)> {
            self.events.presses()
        }

        /// The code that COMPLETED this frame, with whether the registry accepted
        /// it — the instrument's resolution sound. `None` on non-release frames.
        pub fn resolution(&self) -> Option<(&[ChordDir], bool)> {
            self.events.resolution()
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
        let prev_depth = chords.capture.entered().map_or(0, <[_]>::len);
        let code = chords.capture.step(modifier, taps);
        chords.code_entry = was_live || chords.capture.capturing();
        chords.executed = code.as_deref().and_then(|c| S::chords().lookup(c));
        // `accepted` IS `executed` — one verdict, not two lookups to drift apart.
        let accepted = chords.executed.is_some();
        chords.events = ChordEvents::from_frame(
            prev_depth,
            code.map(|c| (c, accepted)),
            chords.capture.entered(),
        );
    }

    /// See [`ChordCapture::reset`] — schedule on the surface's round-entry transition
    /// (e.g. `OnEnter(Playing)`) so a modifier held across it can't smuggle menu-time
    /// taps into the round.
    pub fn reset_chords<S: ControlScheme>(mut chords: ResMut<Chords<S>>) {
        chords.capture.reset();
        chords.executed = None;
        chords.code_entry = false;
        chords.events = ChordEvents::default();
    }

    /// The one wiring of chord input onto an app: the resource plus the capture system.
    /// The registry comes from the scheme itself ([`ControlScheme::chords`]), so every
    /// install site of a surface wires the same table by construction.
    pub fn install_chords<S: ControlScheme>(app: &mut App) {
        app.init_resource::<Chords<S>>().add_systems(
            PreUpdate,
            capture_chords::<S>.after(bevy::input::InputSystems),
        );
        // Code entry IS playing the d-pad instrument (rl#359) — every chord surface
        // sounds through the one install, so a new surface can't ship silent entry.
        crate::instrument::install::<S>(app);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The instrument's event accounting (rl#359): mid-capture taps come with the
        /// prefix each landed after; release-frame taps are included; the completed
        /// code carries the registry verdict; an idle frame yields nothing.
        #[test]
        fn chord_events_expose_consumed_taps_and_resolution() {
            use ChordDir::*;
            // Two taps arrived this frame onto a 1-deep capture.
            let live = ChordEvents::from_frame(1, None, Some(&[Up, Down, Left]));
            let presses: Vec<_> = live.presses().map(|(p, d)| (p.to_vec(), d)).collect();
            assert_eq!(presses, vec![(vec![Up], Down), (vec![Up, Down], Left)]);
            assert_eq!(live.resolution(), None);

            // Release frame with a same-frame final tap: the tap sounds AND the
            // accepted code resolves.
            let released = ChordEvents::from_frame(1, Some((vec![Up, Right], true)), None);
            let presses: Vec<_> = released.presses().map(|(p, d)| (p.to_vec(), d)).collect();
            assert_eq!(presses, vec![(vec![Up], Right)]);
            assert_eq!(released.resolution(), Some((&[Up, Right][..], true)));

            // Unknown code: resolution says so.
            let unknown = ChordEvents::from_frame(2, Some((vec![Down, Down], false)), None);
            assert_eq!(unknown.presses().count(), 0);
            assert_eq!(unknown.resolution(), Some((&[Down, Down][..], false)));

            // Idle frame: no path, no events.
            let idle = ChordEvents::from_frame(0, None, None);
            assert_eq!(idle.presses().count(), 0);
            assert_eq!(idle.resolution(), None);
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

    /// rl#380: deep codes are gameplay — the capture has NO length cap, so a long
    /// code survives entry intact.
    #[test]
    fn long_codes_are_captured_whole() {
        let mut c = ChordCapture::default();
        c.step(true, vec![Up; 40]);
        assert_eq!(c.step(false, []), Some(vec![Up; 40]));
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
