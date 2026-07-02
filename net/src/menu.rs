//! Boot menu for the windowed client: **Host / Join** (no separate Solo button —
//! Host-with-no-joiners IS solo, one codepath). Shown before any
//! round so the player picks a ROLE. Host opens a lobby and a **Start** button forms the
//! round NOW: alone → an instant solo round, peers present → a host-commanded networked
//! lockstep. Join sits in the lobby until the host Starts. There is no fragile discovery
//! timeout — the host decides when to begin.
//!
//! ## Determinism isolation — why the menu can't desync the sim
//!
//! This module is **client-side UI + connection setup ONLY**. It runs entirely BEFORE
//! the deterministic round exists and touches NEITHER the sim NOR the wire's lockstep
//! channel:
//! - The pre-round phases (`Menu`/`Connecting` — see [`crate::render::AppPhase`])
//!   hold no [`Lockstep`] and no sim. The sim is built only at the `Playing` transition,
//!   by [`net_loop::connect_and_form_lobby`]/[`crate::formation::solo_lockstep_for`]. The menu
//!   chooses *when* and *whether to network*; it contributes zero bytes to sim state.
//! - Networked formation runs on a background thread ([`spawn_formation`]); the menu
//!   only polls a channel for the finished [`net_loop::MatchResult`]. The barrier
//!   ([`crate::membership`]) is the determinism-critical code — every peer
//!   freezes the byte-identical sorted roster. The host-triggered start decides only
//!   the *moment* the barrier closes (the host's GO, not a timer); the roster a peer
//!   freezes is the membership core's same `live_set`, so it can't fork the agreed set.
//! - The only menu output that reaches the round is the [`StartChoice`] role, for Join the
//!   host's endpoint id to DIAL, and the host's Start/Cancel signals — addressing +
//!   commands, never sim data. Dialing only opens a QUIC link; the roster still comes from
//!   the barrier, so a typo'd code fails to form a match, it can't form a WRONG one.
//!
//! So the menu is a gate in front of the formation machinery, not a new path into
//! the sim.

use std::sync::mpsc;
use std::thread;

use anyhow::Result;
pub use iroh::EndpointId;

use crate::formation::{self, LobbyControl};
use crate::lockstep::Lockstep;
use crate::membership::Role;
use crate::net_loop::{self, MatchResult, NetDriver};

/// The role the player picked on the menu (just Host / Join — Host-with-no-joiners IS
/// solo). Selects which side of
/// the host-triggered lobby we run; NOT sim state — the resulting roster/seed is the
/// barrier's.
#[derive(Debug, Clone)]
pub enum StartChoice {
    /// Host a match: open a host-triggered lobby. Peers join by our code or mDNS;
    /// the host clicks **Start** to begin. Start with zero joiners is the SOLO round (the
    /// UI forms it locally and instantly — see [`solo_round`] — never depending on the
    /// barrier), and Start with peers present commands a synchronized networked start.
    Host,
    /// Join a host: dial this endpoint id first (mDNS resolves its LAN address), then sit
    /// in the lobby until the host Starts. `None` = discover on the LAN with no explicit
    /// dial (rely on mDNS alone).
    Join(Option<EndpointId>),
}

/// The participant floor for a menu-initiated NETWORKED match (the barrier's `expect`).
/// 2 — the host-commanded networked start only fires with at least one peer present; a
/// host that clicks Start ALONE takes the UI's local instant-solo path ([`solo_round`])
/// and never reaches the barrier, so this floor never blocks the solo case.
const NET_EXPECT: usize = 2;

/// A networked host-triggered formation running on a background thread, with the channels
/// its result, bound-id, live roster, and start/cancel commands flow over. Held by the menu
/// while [`AppPhase::Connecting`] so the render loop never blocks on the barrier. Dropping it
/// (e.g. on Cancel) drops the command/roster channels too — the barrier sees its `cancel_rx`
/// disconnect and tears its own session down promptly (no LAN phantom).
pub struct Formation {
    rx: mpsc::Receiver<Result<MatchResult>>,
    /// For a Join, the dialed host's id (shown as "dialing …"); `None` for a LAN-discover
    /// Join. For a Host this stays `None` — the host's OWN code arrives on `bound_rx`.
    dial_code: Option<EndpointId>,
    /// Our own endpoint id, reported by the worker once the session binds (Host shows it
    /// as the join code to share). Cached into `bound` on first read so the one-shot
    /// channel value isn't lost across frames.
    bound_rx: mpsc::Receiver<EndpointId>,
    bound: std::cell::Cell<Option<EndpointId>>,
    /// The host's Start command: one send commands the barrier's synchronized GO.
    /// `Some` only for a Host; a Join has no Start to give. `take`-n on first use so Start
    /// fires exactly once.
    start_tx: std::cell::Cell<Option<mpsc::Sender<()>>>,
    /// The Cancel command: a send (or simply dropping this [`Formation`]) tells the
    /// barrier to bail and shut the session down with no phantom.
    cancel_tx: mpsc::Sender<()>,
    /// Live roster feed from the barrier (us + peers, sorted), updated each beat. Drained
    /// into `roster` so the latest survives frame to frame for the lobby player list.
    roster_rx: mpsc::Receiver<Vec<EndpointId>>,
    roster: std::cell::RefCell<Vec<EndpointId>>,
    /// Whether this is a Host (vs Join) formation — for the lobby screen's copy + which
    /// code to display.
    pub hosting: bool,
}

impl Formation {
    /// Non-blocking poll: `Some` once the background barrier has finished (Ok with a
    /// match/alone/cancelled result, or Err on a formation failure), `None` while it's
    /// still forming. The render loop calls this each frame from the Connecting screen.
    pub fn poll(&self) -> Option<Result<MatchResult>> {
        match self.rx.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            // The worker dropped its sender without sending — only possible if it
            // panicked (every normal return sends first). Surface it as an error so the
            // menu shows a failure instead of hanging on a dead thread.
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::anyhow!("formation thread ended unexpectedly")))
            }
        }
    }

    /// The join code to show on the lobby screen: the host's own bound id when hosting
    /// (latched once the session binds), or the dialed host id when joining. `None` until
    /// a host's session has bound, or for a LAN-discover Join with no explicit code.
    pub fn display_code(&self) -> Option<EndpointId> {
        if self.hosting {
            // Latch the worker's one-shot bound-id report so it survives frame to frame.
            if self.bound.get().is_none()
                && let Ok(id) = self.bound_rx.try_recv()
            {
                self.bound.set(Some(id));
            }
            self.bound.get()
        } else {
            self.dial_code
        }
    }

    /// The current live lobby roster (us + every joined peer, sorted), drained from the
    /// barrier's feed. Drives the lobby player list. Empty until the session binds and the
    /// first beat lands; for a host alone it stays `[self]`.
    pub fn roster(&self) -> Vec<EndpointId> {
        // Drain to the newest snapshot (cheap; a couple per second), keeping the last.
        while let Ok(r) = self.roster_rx.try_recv() {
            *self.roster.borrow_mut() = r;
        }
        self.roster.borrow().clone()
    }

    /// How many players are currently in the lobby (us + peers). 1 = the host is still
    /// alone, so a Host Start now is the instant SOLO round; ≥2 = a networked start.
    pub fn lobby_len(&self) -> usize {
        self.roster().len()
    }

    /// Host: command the barrier's synchronized start. Fires the GO exactly once
    /// (the sender is taken on first call); a Join or a second call is a no-op. The caller
    /// uses this only when peers are present — a host alone takes the local instant-solo
    /// path instead.
    pub fn request_start(&self) {
        if let Some(tx) = self.start_tx.take() {
            let _ = tx.send(());
        }
    }

    /// Cancel the formation: tell the barrier to bail and shut its session down now,
    /// so leaving the lobby strands no ~12 s LAN phantom. Idempotent and also implied by
    /// simply dropping this [`Formation`] (the barrier's `cancel_rx` then disconnects).
    pub fn cancel(&self) {
        let _ = self.cancel_tx.send(());
    }
}

/// Start a host-triggered formation on a background thread and hand back the [`Formation`]
/// to poll + command. The barrier ([`net_loop::connect_and_form_lobby`]) builds its own
/// tokio runtime and blocks until the host starts / a peer cancels / it fails — so it MUST
/// run off the render thread, or the menu would freeze. The thread owns the runtime +
/// session for its lifetime and tears them down on return (Drop or an explicit shutdown on
/// Cancel).
///
/// `seed` is the shared match seed (the caller passes the one constant every peer uses,
/// so the menu can't introduce a per-peer seed and desync). `join` is the host id to dial
/// for [`StartChoice::Join`] (resolved via mDNS), or `None` to rely on mDNS discovery
/// alone (Host, or a Join that only browses the LAN).
fn spawn_formation(
    seed: u64,
    join: Option<EndpointId>,
    hosting: bool,
    telemetry: Option<EndpointId>,
    weights_digest: u64,
    asset_digest: u64,
) -> Formation {
    let (tx, rx) = mpsc::channel();
    // The worker reports our bound endpoint id (Host's join code) the instant the session
    // is up, the live roster each beat, and listens for Start (host GO) + Cancel commands.
    let (bound_tx, bound_rx) = mpsc::channel();
    let (roster_tx, roster_rx) = mpsc::channel();
    let (start_tx, start_rx) = mpsc::channel();
    let (cancel_tx, cancel_rx) = mpsc::channel();
    // The Role decides the barrier's close trigger (host commands the GO; joiner waits) and
    // is enforced in the membership type. A joiner also has no Start sender to give — it's
    // dropped here so the UI's `request_start` is a no-op — defense in depth atop the Role.
    let (role, host_start_tx) = if hosting {
        (Role::Host, Some(start_tx))
    } else {
        (Role::Joiner, None)
    };
    thread::spawn(move || {
        let result = net_loop::connect_and_form_lobby(
            seed,
            NET_EXPECT,
            join,
            telemetry,
            Some(bound_tx),
            LobbyControl {
                role,
                start_rx,
                cancel_rx,
                roster_tx,
            },
            weights_digest,
            asset_digest,
        );
        // Ignore a send error: it only means the menu moved on (receiver dropped), in
        // which case nobody is waiting and the session tears down on this fn's return.
        let _ = tx.send(result);
    });
    Formation {
        rx,
        dial_code: join,
        bound_rx,
        bound: std::cell::Cell::new(None),
        start_tx: std::cell::Cell::new(host_start_tx),
        cancel_tx,
        roster_rx,
        roster: std::cell::RefCell::new(Vec::new()),
        hosting,
    }
}

/// Kick off the host-triggered formation for a [`StartChoice`] (Host or Join). The single
/// place the menu turns a choice into a running lobby, so the Host vs Join parameterization
/// (who dials whom, who holds the Start command) lives in one spot. `weights_digest` is our
/// loaded NN-crab checkpoint's digest and `asset_digest` our crab-model digest,
/// `0` for none — both advertised in formation so peers can agree on a shared
/// brain AND a shared collider asset before arming the float crab.
pub fn begin(
    choice: &StartChoice,
    seed: u64,
    telemetry: Option<EndpointId>,
    weights_digest: u64,
    asset_digest: u64,
) -> Formation {
    match choice {
        StartChoice::Host => {
            spawn_formation(seed, None, true, telemetry, weights_digest, asset_digest)
        }
        StartChoice::Join(host) => {
            spawn_formation(seed, *host, false, telemetry, weights_digest, asset_digest)
        }
    }
}

/// The match a finished formation yielded, ready to drive a round: the agreed
/// [`Lockstep`] plus the [`NetDriver`] for its peers, or a solo lockstep when the barrier
/// fell back to alone — either way a playable round, so the menu and the headless `net`
/// driver treat "nobody showed" identically.
pub struct ReadyMatch {
    pub lockstep: Lockstep,
    pub net: Option<NetDriver>,
}

/// Turn a finished [`MatchResult`] into a [`ReadyMatch`], or `None` if the player cancelled
/// the lobby (no round to play — the UI returns to the menu). `Alone` becomes a
/// [`solo_round`] (the SAME deterministic solo the Host-alone Start uses — one definition,
/// no drift), so "Join, host never appeared" plays the identical offline round.
pub fn ready_from(result: MatchResult, seed: u64) -> Option<ReadyMatch> {
    match result {
        MatchResult::Joined(joined) => {
            let (lockstep, net) = *joined;
            Some(ReadyMatch {
                lockstep,
                net: Some(net),
            })
        }
        MatchResult::Alone => Some(solo_round(seed)),
        MatchResult::Cancelled => None,
    }
}

/// Build an OFFLINE round directly (no networking): the Host-alone Start (the UI forms it
/// the instant a host clicks Start with zero peers) and the barrier's Alone fallback both
/// use it, so the instant-solo path and the "nobody joined" path are the byte-identical
/// deterministic round from [`crate::formation::solo_lockstep_for`].
pub fn solo_round(seed: u64) -> ReadyMatch {
    ReadyMatch {
        lockstep: formation::solo_lockstep_for(seed),
        net: None,
    }
}

// ── Pure menu navigation state machine ──────────────────────────────────────
//
// The boot menu's focus + transitions as a Bevy-free, egui-free VALUE, so navigation and
// the Start transition are unit-testable without a live window. The render layer gathers
// raw input (keyboard, gamepad, egui clicks), reduces it to a [`MenuInput`], folds it
// through [`MenuNav::step`], and EXECUTES the returned [`MenuAction`] in one exhaustive
// `match`. So a focusable item with no wired action is a COMPILE error, not a silently
// dead button unreachable for a gamepad-only player. Like [`StartChoice`], the FSM never
// touches the sim/[`Lockstep`]/[`Formation`]; it only decides which abstract action a
// confirm means.

/// A focusable item on the Host / Join chooser. The join-code text field sits between the
/// two but is NOT in this ring — it's edited with mouse/keyboard, while a gamepad player
/// joins by LAN discovery (a blank code), so the navigable ring is the two actions only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChooserItem {
    Host,
    Join,
}

/// A focusable item in the HOST lobby (the only lobby with a focus ring). A joiner's lobby
/// has no choices — its only action is Cancel — so it carries no `LobbyItem` at all (see
/// [`MenuNav::JoinLobby`]), which is what makes "a joiner focused on Start" unrepresentable
/// rather than merely unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LobbyItem {
    Start,
    Cancel,
}

/// A focusable item on the "connection lost" prompt (rl#203) — rejoin the match we were
/// dropped from, or go back to the chooser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisconnectedItem {
    Rejoin,
    Leave,
}

/// One device-agnostic menu navigation event. Keyboard arrows/WASD, gamepad D-pad/stick,
/// and egui clicks all reduce to these (a click is `Confirm` after focusing the clicked
/// item). Left/Right collapse into Up/Down because every menu screen is a vertical list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuInput {
    Up,
    Down,
    Confirm,
    Back,
}

/// The side effect a [`MenuInput`] resolves to, executed by the render layer's single
/// dispatch. The whole point of the type: that `match` is exhaustive, so every menu action
/// is wired or the build fails. A pure focus move yields [`MenuAction::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    /// Focus moved (or a no-op confirm) — nothing for the render layer to do.
    None,
    /// Chooser: begin hosting a lobby.
    Host,
    /// Chooser: begin joining (the render layer reads the typed code field).
    Join,
    /// Lobby (host, peers present): command the synchronized networked start.
    StartNetworked,
    /// Lobby (host, alone): install the instant solo round and play now.
    StartSolo,
    /// Lobby: leave — cancel the formation and return to the chooser.
    Cancel,
    /// Disconnected prompt: re-dial the lost match's host and send a fresh
    /// [`crate::server::JoinRequest`] (the [`net_loop::connect_and_join`] path, rl#203).
    Rejoin,
}

/// The pure menu state: which screen, and the focus within it. Built by [`MenuNav::new`]
/// (the chooser, Host focused) and folded by [`MenuNav::step`]. The render layer mirrors its
/// AppPhase from the actions `step` returns, so the FSM screen and the Bevy phase move
/// together (no second source of truth to drift).
///
/// Host and joiner lobbies are SEPARATE variants, not one `Lobby { hosting: bool }`: only a
/// host has a focus ring (Start / Cancel), and splitting them makes "a joiner focused on
/// Start" unrepresentable instead of a runtime no-op — the make-illegal-states-
/// unrepresentable the menu's design aims for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuNav {
    /// The Host / Join chooser (AppPhase::Menu).
    Chooser { focus: ChooserItem },
    /// The host's lobby (AppPhase::Connecting): a Start / Cancel focus ring.
    HostLobby { focus: LobbyItem },
    /// A joiner's lobby (AppPhase::Connecting): no choices but Cancel, so no focus to hold.
    JoinLobby,
    /// The "connection lost — rejoin?" prompt (AppPhase::Menu, rl#203), shown when a live
    /// round's server link died: a Rejoin / Leave focus ring. Entered only by the disconnect
    /// return (never navigable-to), so a rejoinable host id always exists behind it.
    Disconnected { focus: DisconnectedItem },
    /// A rejoin dial is in flight (AppPhase::Connecting): like [`MenuNav::JoinLobby`], the only
    /// action is Cancel — the verdict (admitted / refused / unreachable) arrives on a poll.
    Rejoining,
}

impl Default for MenuNav {
    fn default() -> Self {
        Self::new()
    }
}

impl MenuNav {
    /// The starting state: the chooser with Host focused (the common Deck case).
    pub fn new() -> Self {
        MenuNav::Chooser {
            focus: ChooserItem::Host,
        }
    }

    /// Enter the lobby for a chosen role: the host's lobby (Start focused) or the joiner's
    /// choice-free lobby.
    fn lobby(hosting: bool) -> Self {
        if hosting {
            MenuNav::HostLobby {
                focus: LobbyItem::Start,
            }
        } else {
            MenuNav::JoinLobby
        }
    }

    /// Fold one input into the menu, returning the action for the render layer to execute.
    /// `lobby_len` is the live roster size (us + peers); it ONLY resolves a host's lobby
    /// Start into solo (≤1) vs networked (≥2), and is ignored on every other transition.
    pub fn step(&mut self, input: MenuInput, lobby_len: usize) -> MenuAction {
        match self {
            MenuNav::Chooser { focus } => match input {
                // A two-item vertical list: either direction just toggles.
                MenuInput::Up | MenuInput::Down => {
                    *focus = match focus {
                        ChooserItem::Host => ChooserItem::Join,
                        ChooserItem::Join => ChooserItem::Host,
                    };
                    MenuAction::None
                }
                MenuInput::Confirm => {
                    let hosting = matches!(focus, ChooserItem::Host);
                    *self = MenuNav::lobby(hosting);
                    if hosting {
                        MenuAction::Host
                    } else {
                        MenuAction::Join
                    }
                }
                // Already at the root — nowhere to back out to.
                MenuInput::Back => MenuAction::None,
            },
            MenuNav::HostLobby { focus } => match input {
                MenuInput::Up | MenuInput::Down => {
                    *focus = match focus {
                        LobbyItem::Start => LobbyItem::Cancel,
                        LobbyItem::Cancel => LobbyItem::Start,
                    };
                    MenuAction::None
                }
                MenuInput::Confirm => match focus {
                    LobbyItem::Start => {
                        // Resolve the host's Start against the LIVE roster (the single source
                        // for solo-vs-networked), then leave the lobby. Solo enters Playing
                        // now, so reset to a clean chooser; networked stays in the lobby
                        // waiting for the formed match to arrive.
                        if lobby_len <= 1 {
                            *self = MenuNav::new();
                            MenuAction::StartSolo
                        } else {
                            MenuAction::StartNetworked
                        }
                    }
                    LobbyItem::Cancel => {
                        *self = MenuNav::new();
                        MenuAction::Cancel
                    }
                },
                // Back out of the lobby == Cancel (tear the formation down, return to root).
                MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::Cancel
                }
            },
            // A joiner can only wait or leave: nav is inert, and the only action it can issue
            // (Confirm or Back) is Cancel. There is no Start to reach — by construction.
            MenuNav::JoinLobby => match input {
                MenuInput::Up | MenuInput::Down => MenuAction::None,
                MenuInput::Confirm | MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::Cancel
                }
            },
            // "Connection lost — rejoin?": a two-item vertical ring, like the chooser.
            // Back declines (the chooser, with the disconnect message still shown).
            MenuNav::Disconnected { focus } => match input {
                MenuInput::Up | MenuInput::Down => {
                    *focus = match focus {
                        DisconnectedItem::Rejoin => DisconnectedItem::Leave,
                        DisconnectedItem::Leave => DisconnectedItem::Rejoin,
                    };
                    MenuAction::None
                }
                MenuInput::Confirm => match focus {
                    DisconnectedItem::Rejoin => {
                        *self = MenuNav::Rejoining;
                        MenuAction::Rejoin
                    }
                    DisconnectedItem::Leave => {
                        *self = MenuNav::new();
                        MenuAction::None
                    }
                },
                MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::None
                }
            },
            // A rejoin in flight can only be waited out or abandoned — the same shape as the
            // joiner's lobby; the admitted/refused/unreachable verdict arrives via a poll.
            MenuNav::Rejoining => match input {
                MenuInput::Up | MenuInput::Down => MenuAction::None,
                MenuInput::Confirm | MenuInput::Back => {
                    *self = MenuNav::new();
                    MenuAction::Cancel
                }
            },
        }
    }

    /// Move focus to a specific chooser item (no-op off the chooser). Used when a mouse click
    /// lands on an item: the render layer focuses it then feeds `Confirm`, so a click takes
    /// the EXACT same path through `step` as a pad/keyboard confirm.
    pub fn focus_chooser(&mut self, item: ChooserItem) {
        if let MenuNav::Chooser { focus } = self {
            *focus = item;
        }
    }

    /// Move focus to a specific host-lobby item (no-op off the host lobby — a joiner has no
    /// focus). The click counterpart of [`Self::focus_chooser`].
    pub fn focus_lobby(&mut self, item: LobbyItem) {
        if let MenuNav::HostLobby { focus } = self {
            *focus = item;
        }
    }

    /// Enter the "connection lost — rejoin?" prompt (rl#203), Rejoin focused — the one
    /// disconnect-return entry, so the prompt always starts on the affirmative.
    pub fn disconnected() -> Self {
        MenuNav::Disconnected {
            focus: DisconnectedItem::Rejoin,
        }
    }

    /// Move focus to a specific disconnected-prompt item (no-op off the prompt). The click
    /// counterpart of [`Self::focus_chooser`].
    pub fn focus_disconnected(&mut self, item: DisconnectedItem) {
        if let MenuNav::Disconnected { focus } = self {
            *focus = item;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a Host formation hands back a usable Start command; a Join has no Start
    /// to give (it waits for the host's), so its `start_tx` is dropped at spawn. Pins that
    /// `request_start` is a no-op for a joiner — it can never command a start it isn't
    /// allowed to — and that `cancel`/`hosting` route per role. `#[ignore]` because `begin`
    /// spawns a real barrier thread that binds an iroh UDP endpoint (same reason as
    /// `net_loop`'s live formation test); run with `--ignored`. The pure host-vs-joiner
    /// protocol guarantee is covered socket-free by
    /// `membership::only_a_host_ever_advertises_the_start_go`.
    #[test]
    #[ignore = "binds a real iroh UDP endpoint via begin(); run explicitly with --ignored"]
    fn only_host_holds_the_start_command() {
        // A Host keeps its Start sender; calling request_start takes it (so a second call is
        // inert) — the once-only GO.
        let host = begin(&StartChoice::Host, 0, None, 0, 0);
        assert!(host.hosting, "Host formation is flagged hosting");
        host.request_start(); // consumes the sender
        // Cancel both so their barrier threads tear down promptly rather than lingering.
        host.cancel();

        let join = begin(&StartChoice::Join(None), 0, None, 0, 0);
        assert!(!join.hosting, "Join formation is not hosting");
        join.request_start(); // no-op: a joiner never had a Start sender
        join.cancel();
    }

    /// `ready_from(Alone)` yields a solo round (no NetDriver) seeded with the shared seed
    /// — the "nobody joined" path that makes Host → Start playable alone. Proves the menu
    /// maps Alone to the offline round rather than dropping the player out.
    #[test]
    fn alone_becomes_a_solo_round() {
        let seed = 0xABCD;
        let m = ready_from(MatchResult::Alone, seed).expect("Alone is a playable solo round");
        assert!(m.net.is_none(), "Alone is offline — no NetDriver");
        // The solo lockstep is built and ready (one player: us).
        assert_eq!(m.lockstep.me().0, 0, "solo player is id 0");
    }

    /// `ready_from(Cancelled)` is no round at all — the player backed out of the lobby, so
    /// the UI returns to the menu rather than installing a sim. Pins the type-honest `None`.
    #[test]
    fn cancelled_is_not_a_round() {
        assert!(
            ready_from(MatchResult::Cancelled, 0).is_none(),
            "a cancelled lobby yields no round"
        );
    }

    // ── MenuNav: the pure navigation FSM (no window needed) ──────────────────
    // Every navigation move and Start transition exercised as plain values, so the
    // menu's logic is proven without a live Bevy/egui window or a gamepad.

    /// The chooser starts on Host and Up/Down toggles between the two items (a two-item
    /// vertical list wraps trivially), with no side effect.
    #[test]
    fn chooser_navigates_between_host_and_join() {
        let mut nav = MenuNav::new();
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Host
            }
        );
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Join
            }
        );
        assert_eq!(nav.step(MenuInput::Up, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Host
            }
        );
        // Either direction toggles; Back at the root is inert.
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Join
            }
        );
        assert_eq!(nav.step(MenuInput::Back, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Chooser {
                focus: ChooserItem::Join
            }
        );
    }

    /// Confirming Host enters the lobby as a host (Start focused) and emits `Host`;
    /// confirming Join enters as a joiner (Cancel focused — a joiner can't Start) and emits
    /// `Join`. The chooser→lobby transition the menu depends on.
    #[test]
    fn confirm_host_or_join_enters_the_lobby_with_the_right_role() {
        let mut host = MenuNav::new();
        assert_eq!(host.step(MenuInput::Confirm, 0), MenuAction::Host);
        assert_eq!(
            host,
            MenuNav::HostLobby {
                focus: LobbyItem::Start
            }
        );

        let mut join = MenuNav::new();
        join.step(MenuInput::Down, 0); // focus Join
        assert_eq!(join.step(MenuInput::Confirm, 0), MenuAction::Join);
        assert_eq!(join, MenuNav::JoinLobby);
    }

    /// THE Start transition: a host confirming Start resolves against
    /// the live roster — alone (≤1) → an instant solo round, peers present (≥2) → the
    /// networked GO. Solo resets to a clean chooser (we're entering Playing); networked stays
    /// in the lobby waiting for the formed match.
    #[test]
    fn host_start_resolves_solo_vs_networked_by_roster() {
        let mut alone = MenuNav::lobby(true);
        assert_eq!(alone.step(MenuInput::Confirm, 1), MenuAction::StartSolo);
        assert_eq!(alone, MenuNav::new(), "solo Start resets to the chooser");

        // An empty roster (session not yet bound) is still the solo case.
        let mut empty = MenuNav::lobby(true);
        assert_eq!(empty.step(MenuInput::Confirm, 0), MenuAction::StartSolo);

        let mut networked = MenuNav::lobby(true);
        assert_eq!(
            networked.step(MenuInput::Confirm, 2),
            MenuAction::StartNetworked
        );
        assert_eq!(
            networked,
            MenuNav::HostLobby {
                focus: LobbyItem::Start
            },
            "networked Start stays in the lobby until the match forms"
        );
    }

    /// In the host lobby, Up/Down moves between Start and Cancel; confirming Cancel (or
    /// pressing Back) leaves the lobby and returns to the chooser, emitting `Cancel`.
    #[test]
    fn host_lobby_navigates_and_cancels() {
        let mut nav = MenuNav::lobby(true);
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::HostLobby {
                focus: LobbyItem::Cancel
            }
        );
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Cancel);
        assert_eq!(nav, MenuNav::new(), "Cancel returns to the chooser");

        // Back from anywhere in the lobby is the same as Cancel.
        let mut back = MenuNav::lobby(true);
        assert_eq!(back.step(MenuInput::Back, 5), MenuAction::Cancel);
        assert_eq!(back, MenuNav::new());
    }

    /// A joiner's lobby has no focus and no choices but Cancel: navigation is inert, and the
    /// only action it can ever issue (Confirm or Back) is `Cancel`. It can NEVER produce a
    /// Start — that state is unrepresentable (no `LobbyItem` on [`MenuNav::JoinLobby`]), a
    /// stronger guarantee than the host-only Start command in [`Formation`].
    #[test]
    fn joiner_lobby_can_only_cancel() {
        let mut nav = MenuNav::lobby(false);
        assert_eq!(nav, MenuNav::JoinLobby);
        // Navigation is inert — there's nothing to move to.
        assert_eq!(nav.step(MenuInput::Down, 9), MenuAction::None);
        assert_eq!(nav, MenuNav::JoinLobby);
        // Confirm can only ever cancel; even a populated roster yields no Start.
        assert_eq!(nav.step(MenuInput::Confirm, 9), MenuAction::Cancel);
        assert_eq!(nav, MenuNav::new());
    }

    /// THE disconnect flow (rl#203): the prompt starts on Rejoin, Up/Down toggles the two
    /// items, confirming Rejoin fires the rejoin dial and moves to Rejoining, and a rejoin in
    /// flight can only be cancelled — the same wait-or-leave shape as the joiner's lobby.
    #[test]
    fn disconnected_prompt_rejoins_or_leaves() {
        let mut nav = MenuNav::disconnected();
        assert_eq!(
            nav,
            MenuNav::Disconnected {
                focus: DisconnectedItem::Rejoin
            },
            "the prompt starts on the affirmative"
        );
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(
            nav,
            MenuNav::Disconnected {
                focus: DisconnectedItem::Leave
            }
        );
        assert_eq!(nav.step(MenuInput::Up, 0), MenuAction::None);
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Rejoin);
        assert_eq!(nav, MenuNav::Rejoining);

        // A rejoin in flight: navigation is inert; Confirm or Back abandons it.
        assert_eq!(nav.step(MenuInput::Down, 0), MenuAction::None);
        assert_eq!(nav, MenuNav::Rejoining);
        assert_eq!(nav.step(MenuInput::Back, 0), MenuAction::Cancel);
        assert_eq!(
            nav,
            MenuNav::new(),
            "an abandoned rejoin lands on the chooser"
        );

        // Declining the prompt (Leave, or Back) returns to the chooser with no side effect.
        let mut decline = MenuNav::disconnected();
        decline.step(MenuInput::Down, 0); // focus Leave
        assert_eq!(decline.step(MenuInput::Confirm, 0), MenuAction::None);
        assert_eq!(decline, MenuNav::new());
        let mut back = MenuNav::disconnected();
        assert_eq!(back.step(MenuInput::Back, 0), MenuAction::None);
        assert_eq!(back, MenuNav::new());
    }

    /// A click routes through the SAME `step` path as a pad/keyboard confirm: focus the
    /// clicked item, then `Confirm`. Clicking Join from a Host-focused chooser yields `Join`,
    /// proving click and controller can't diverge (the unified-dispatch guarantee).
    #[test]
    fn click_focuses_then_confirms_like_a_controller() {
        let mut nav = MenuNav::new(); // Host focused
        nav.focus_chooser(ChooserItem::Join);
        assert_eq!(nav.step(MenuInput::Confirm, 0), MenuAction::Join);

        // focus_lobby is a no-op off the lobby (and vice-versa) — it can't corrupt state.
        let mut chooser = MenuNav::new();
        chooser.focus_lobby(LobbyItem::Cancel);
        assert_eq!(
            chooser,
            MenuNav::new(),
            "focus_lobby is inert on the chooser"
        );
    }
}
