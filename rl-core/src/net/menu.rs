//! Boot menu for the windowed client: **Host / Join** (rl#58 — the old separate Solo
//! button is gone, because Host-with-no-joiners IS solo, one codepath). Shown before any
//! round so the player picks a ROLE. Host opens a lobby and a **Start** button forms the
//! round NOW: alone → an instant solo round, peers present → a host-commanded networked
//! lockstep. Join sits in the lobby until the host Starts. This supersedes the old
//! discover-or-fail boot (#55): there is no fragile discovery timeout — the host decides
//! when to begin.
//!
//! ## Determinism isolation — why the menu can't desync the sim
//!
//! This module is **client-side UI + connection setup ONLY**. It runs entirely BEFORE
//! the deterministic round exists and touches NEITHER the sim NOR the wire's lockstep
//! channel:
//! - The pre-round phases (`Menu`/`Connecting` — see [`crate::net::render::AppPhase`])
//!   hold no [`Lockstep`] and no sim. The sim is built only at the `Playing` transition,
//!   from the same [`net_loop::connect_and_form_lobby`]/[`net_loop::solo_lockstep_for`] the
//!   pre-menu boot used — so the agreed roster + seed form EXACTLY as before. The menu
//!   chooses *when* and *whether to network*; it contributes zero bytes to sim state.
//! - Networked formation runs on a background thread ([`spawn_formation`]); the menu
//!   only polls a channel for the finished [`net_loop::MatchResult`]. The barrier
//!   ([`crate::net::membership`]) is the determinism-critical code — every peer still
//!   freezes the byte-identical sorted roster. The host-triggered start (rl#58) moves only
//!   the *moment* the barrier closes (the host's GO instead of a timer); the roster a peer
//!   freezes is the membership core's same `live_set`, so it can't fork the agreed set.
//! - The only menu output that reaches the round is the [`StartChoice`] role, for Join the
//!   host's endpoint id to DIAL, and the host's Start/Cancel signals — addressing +
//!   commands, never sim data. Dialing only opens a QUIC link; the roster still comes from
//!   the barrier, so a typo'd code fails to form a match, it can't form a WRONG one.
//!
//! So two peers that reach a round via the menu are bit-identical to two that reached
//! it via the old auto-discover boot: the menu is a gate in front of the same machinery,
//! not a new path into the sim.

use std::sync::mpsc;
use std::thread;

use anyhow::Result;
pub use iroh::EndpointId;

use crate::net::lockstep::Lockstep;
use crate::net::membership::Role;
use crate::net::net_loop::{self, LobbyControl, MatchResult, NetDriver};

/// The role the player picked on the menu (rl#58: the menu is just Host / Join — the old
/// separate Solo button is gone, since Host-with-no-joiners IS solo). Selects which side of
/// the host-triggered lobby we run; NOT sim state — the resulting roster/seed is the
/// barrier's.
#[derive(Debug, Clone)]
pub enum StartChoice {
    /// Host a match: open a host-triggered lobby (rl#58). Peers join by our code or mDNS;
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
    /// The host's Start command (rl#58): one send commands the barrier's synchronized GO.
    /// `Some` only for a Host; a Join has no Start to give. `take`-n on first use so Start
    /// fires exactly once.
    start_tx: std::cell::Cell<Option<mpsc::Sender<()>>>,
    /// The Cancel command (rl#58): a send (or simply dropping this [`Formation`]) tells the
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

    /// Host: command the barrier's synchronized start (rl#58). Fires the GO exactly once
    /// (the sender is taken on first call); a Join or a second call is a no-op. The caller
    /// uses this only when peers are present — a host alone takes the local instant-solo
    /// path instead.
    pub fn request_start(&self) {
        if let Some(tx) = self.start_tx.take() {
            let _ = tx.send(());
        }
    }

    /// Cancel the formation (rl#58): tell the barrier to bail and shut its session down now,
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
/// (who dials whom, who holds the Start command) lives in one spot.
pub fn begin(choice: &StartChoice, seed: u64, telemetry: Option<EndpointId>) -> Formation {
    match choice {
        StartChoice::Host => spawn_formation(seed, None, true, telemetry),
        StartChoice::Join(host) => spawn_formation(seed, *host, false, telemetry),
    }
}

/// The match a finished formation yielded, ready to drive a round: the agreed
/// [`Lockstep`] plus the [`NetDriver`] for its peers, or a solo lockstep when the barrier
/// fell back to alone (rl#47) — either way a playable round, mirroring the old boot's
/// `MatchResult` handling so the menu and the headless `net` driver treat "nobody showed"
/// identically.
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
/// deterministic round from [`net_loop::solo_lockstep_for`].
pub fn solo_round(seed: u64) -> ReadyMatch {
    ReadyMatch {
        lockstep: net_loop::solo_lockstep_for(seed),
        net: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a Host formation hands back a usable Start command (rl#58); a Join has no Start
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
        let host = begin(&StartChoice::Host, 0, None);
        assert!(host.hosting, "Host formation is flagged hosting");
        host.request_start(); // consumes the sender
        // Cancel both so their barrier threads tear down promptly rather than lingering.
        host.cancel();

        let join = begin(&StartChoice::Join(None), 0, None);
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
}
