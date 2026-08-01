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

pub struct ChordEntry<A: 'static> {
    pub code: &'static [ChordDir],
    pub action: A,
}

/// The one data table mapping chord codes to commands. Code assignments are gameplay
/// data the owner tunes — keep every entry in the surface's single registry const.
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

    /// Panics on a duplicate code or one longer than [`MAX_CHORD_LEN`] (unreachable —
    /// the capture poisons first). Call from the surface's scheme test.
    pub fn assert_well_formed(&self) {
        for (i, e) in self.0.iter().enumerate() {
            assert!(
                e.code.len() <= MAX_CHORD_LEN,
                "code for {:?} is longer than MAX_CHORD_LEN ({MAX_CHORD_LEN}) — unenterable",
                e.action
            );
            for other in &self.0[i + 1..] {
                assert!(
                    e.code != other.code,
                    "code {:?} is registered twice: {:?} and {:?}",
                    e.code,
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

    fn key_dir(k: KeyCode) -> Option<ChordDir> {
        match k {
            KeyCode::KeyW => Some(ChordDir::Up),
            KeyCode::KeyS => Some(ChordDir::Down),
            KeyCode::KeyA => Some(ChordDir::Left),
            KeyCode::KeyD => Some(ChordDir::Right),
            _ => None,
        }
    }

    /// A surface's live chord state: its registry, the capture, and the command the
    /// just-finished code executed THIS frame (cleared next frame). Dispatch systems
    /// read [`Chords::executed`]; they must be scheduled after [`capture_chords`],
    /// which [`install_chords`] guarantees by running the capture in `PreUpdate`.
    #[derive(Resource)]
    pub struct Chords<S: ControlScheme> {
        registry: ChordRegistry<S::Action>,
        capture: ChordCapture,
        executed: Option<S::Action>,
    }

    impl<S: ControlScheme> Chords<S> {
        pub fn new(registry: ChordRegistry<S::Action>) -> Self {
            Self {
                registry,
                capture: ChordCapture::default(),
                executed: None,
            }
        }

        /// Did a chord execute `action` this frame?
        pub fn executed(&self, action: S::Action) -> bool {
            self.executed == Some(action)
        }

        /// See [`ChordCapture::capturing`].
        pub fn capturing(&self) -> bool {
            self.capture.capturing()
        }
    }

    /// Drive the capture from the real inputs, once per frame in `PreUpdate` (after
    /// bevy's input update, before every dispatch system in `Update`).
    pub fn capture_chords<S: ControlScheme>(
        keys: Res<ButtonInput<KeyCode>>,
        mouse: Res<ButtonInput<MouseButton>>,
        pads: Query<&Gamepad>,
        mut chords: ResMut<Chords<S>>,
    ) {
        // `just_pressed` too: a press-and-release inside one frame (a hitch, a low-FPS
        // stretch) reads `pressed == false` on both frames, and a level-only sample
        // would drop the tap outright — the empty-code verb would intermittently die.
        // Seen this frame as just-pressed, the capture opens now and completes on the
        // next frame's release.
        let modifier = mouse.pressed(CHORD_MODIFIER_MOUSE)
            || mouse.just_pressed(CHORD_MODIFIER_MOUSE)
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
        let code = chords.capture.step(modifier, taps);
        chords.executed = code.and_then(|c| chords.registry.lookup(&c));
    }

    /// See [`ChordCapture::reset`] — schedule on the surface's round-entry transition
    /// (e.g. `OnEnter(Playing)`) so a modifier held across it can't smuggle menu-time
    /// taps into the round.
    pub fn reset_chords<S: ControlScheme>(mut chords: ResMut<Chords<S>>) {
        chords.capture.reset();
        chords.executed = None;
    }

    /// The one wiring of chord input onto an app: the resource plus the capture system.
    pub fn install_chords<S: ControlScheme>(app: &mut App, registry: ChordRegistry<S::Action>) {
        app.insert_resource(Chords::<S>::new(registry)).add_systems(
            PreUpdate,
            capture_chords::<S>.after(bevy::input::InputSystems),
        );
    }
}

#[cfg(feature = "render")]
pub use glue::{
    CHORD_MODIFIER_MOUSE, CHORD_MODIFIER_PAD, Chords, capture_chords, install_chords, reset_chords,
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
        },
        ChordEntry {
            code: &[Up],
            action: Cmd::One,
        },
        ChordEntry {
            code: &[Up, Down],
            action: Cmd::Two,
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
    fn well_formedness_rejects_duplicates() {
        REG.assert_well_formed();
        let dup: ChordRegistry<Cmd> = ChordRegistry::new(&[
            ChordEntry {
                code: &[Up],
                action: Cmd::One,
            },
            ChordEntry {
                code: &[Up],
                action: Cmd::Two,
            },
        ]);
        assert!(std::panic::catch_unwind(|| dup.assert_well_formed()).is_err());
    }
}
