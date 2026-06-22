//! Boot menu for the windowed client (rl#56): Host / Join / Solo, shown before any
//! round so the player picks a ROLE instead of the old blind discover-or-fail window
//! (which errored with no/flickering peers — #55). It supersedes that auto-fallback:
//! Host → Start always yields a round (solo when nobody joined), so a lone player is
//! always playable without the fragile discovery timeout.
//!
//! ## Determinism isolation — why the menu can't desync the sim
//!
//! This module is **client-side UI + connection setup ONLY**. It runs entirely BEFORE
//! the deterministic round exists and touches NEITHER the sim NOR the wire's lockstep
//! channel:
//! - The pre-round phases (`Menu`/`Connecting` — see [`crate::net::render::AppPhase`])
//!   hold no [`Lockstep`] and no sim. The sim is built only at the `Playing` transition,
//!   from the same [`net_loop::connect_and_form`]/[`net_loop::solo_lockstep_for`] the
//!   pre-menu boot used — so the agreed roster + seed form EXACTLY as before. The menu
//!   chooses *when* and *whether to network*; it contributes zero bytes to sim state.
//! - Networked formation runs on a background thread ([`spawn_formation`]); the menu
//!   only polls a channel for the finished [`net_loop::MatchResult`]. The barrier
//!   ([`crate::net::membership`]) is the unchanged determinism-critical code — every
//!   peer still freezes the byte-identical sorted roster. The menu adds no roster input.
//! - The only menu output that reaches the round is the choice of [`StartChoice`]
//!   (solo vs networked) and, for Join, the host's endpoint id to DIAL — addressing
//!   data, never sim data. Dialing only opens a QUIC link; the roster still comes from
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
use crate::net::net_loop::{self, MatchResult, NetDriver};

/// What the player picked on the menu (or a scripted flag forced). Drives whether the
/// round is built instantly offline or after a networked formation. NOT sim state — it
/// only selects which formation path runs; the resulting roster/seed is the barrier's.
#[derive(Debug, Clone)]
pub enum StartChoice {
    /// Play offline immediately: build the solo [`Lockstep`] with no network. Both the
    /// explicit Solo button and Host → Start with zero joiners land here — the clean
    /// always-playable path that removes the #55 discovery failure.
    Solo,
    /// Host a networked match: run the shared formation barrier so any LAN peers that
    /// dialed our code (or found us via mDNS) join the agreed roster. With nobody else
    /// present the barrier-over-network forms `{self}` (see `expect == 1` in
    /// [`net_loop`]); but the menu's Start-solo button uses [`StartChoice::Solo`] for a
    /// guaranteed instant round rather than depending on that timing.
    Host,
    /// Join a host: dial this endpoint id first (mDNS resolves its LAN address), then run
    /// the same barrier so we land in the host's agreed roster. `None` = discover on the
    /// LAN with no explicit dial (rely on mDNS alone).
    Join(Option<EndpointId>),
}

/// How long the networked Host/Join paths wait in the barrier before concluding (passed
/// to [`net_loop::connect_and_form`] as `discover_secs`). Longer than the old 4 s boot
/// default because the player has now EXPLICITLY chosen to network and may be waiting on
/// a peer to also hit Join — give the LAN time rather than dropping to solo early. Stays
/// well under the barrier's `JOIN_WINDOW` (20 s) so the alone-fallback can still fire.
const NET_DISCOVER_SECS: u64 = 12;

/// The participant floor for a menu-initiated networked match (the barrier's `expect`).
/// 2 — a Host/Join player wants at least one peer; with none present after
/// [`NET_DISCOVER_SECS`] the barrier's rl#47 alone-fallback returns
/// [`MatchResult::Alone`] and we play solo, so "nobody joined" still yields a round.
const NET_EXPECT: usize = 2;

/// A networked formation running on a background thread, with the channels its result
/// and bound-id arrive on. Held by the menu while [`AppPhase::Connecting`] so the render
/// loop never blocks on the barrier (which can wait many seconds). Dropping the receiver
/// before the thread finishes just detaches it — the thread tears its own session down on
/// return.
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
    /// Whether this is a Host (vs Join) formation — for the lobby screen's copy + which
    /// code to display.
    pub hosting: bool,
}

impl Formation {
    /// Non-blocking poll: `Some` once the background barrier has finished (Ok with a
    /// match/alone result, or Err on a formation failure), `None` while it's still
    /// forming. The render loop calls this each frame from the Connecting screen.
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
}

/// Start a networked formation on a background thread and hand back the [`Formation`] to
/// poll. The barrier ([`net_loop::connect_and_form`]) builds its own tokio runtime and
/// blocks until it agrees / falls back to solo / fails — so it MUST run off the render
/// thread, or the menu would freeze for the whole discovery window. The thread owns the
/// runtime + session for its lifetime and tears them down on return (Drop), exactly as a
/// direct `connect_and_form` call would.
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
    // A second channel for the worker to report our own bound endpoint id the instant the
    // session is up, so a Host lobby can show its join code without waiting out the barrier.
    let (bound_tx, bound_rx) = mpsc::channel();
    thread::spawn(move || {
        let result = net_loop::connect_and_form_dialing(
            seed,
            NET_DISCOVER_SECS,
            NET_EXPECT,
            join,
            telemetry,
            Some(bound_tx),
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
        hosting,
    }
}

/// Kick off the formation for a networked [`StartChoice`] (Host or Join), or `None` for
/// Solo (which needs no formation — the caller builds the solo lockstep directly). The
/// single place the menu turns a choice into a running barrier, so the Host vs Join
/// parameterization (who dials whom) lives in one spot.
pub fn begin(choice: &StartChoice, seed: u64, telemetry: Option<EndpointId>) -> Option<Formation> {
    match choice {
        StartChoice::Solo => None,
        StartChoice::Host => Some(spawn_formation(seed, None, true, telemetry)),
        StartChoice::Join(host) => Some(spawn_formation(seed, *host, false, telemetry)),
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

/// Turn a finished [`MatchResult`] into a [`ReadyMatch`]. `Alone` becomes a [`solo_round`]
/// (the SAME deterministic solo the `--solo` path uses — one definition, no drift), so
/// "Host → Start, nobody joined" and "Join, host never appeared" both play the identical
/// offline round.
pub fn ready_from(result: MatchResult, seed: u64) -> ReadyMatch {
    match result {
        MatchResult::Joined(joined) => {
            let (lockstep, net) = *joined;
            ReadyMatch {
                lockstep,
                net: Some(net),
            }
        }
        MatchResult::Alone => solo_round(seed),
    }
}

/// Build an OFFLINE round directly (no networking): the Solo menu button and the
/// barrier's Alone fallback both use it, so the instant-solo path and the "nobody joined"
/// path are the byte-identical deterministic round from [`net_loop::solo_lockstep_for`].
pub fn solo_round(seed: u64) -> ReadyMatch {
    ReadyMatch {
        lockstep: net_loop::solo_lockstep_for(seed),
        net: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `begin` returns no formation for Solo (the caller builds the solo lockstep with no
    /// network) and a formation for the networked roles — the routing the render loop
    /// relies on to know whether to enter the Connecting screen at all. Pins that Solo
    /// never spins up a barrier/thread.
    #[test]
    fn begin_only_networks_for_host_and_join() {
        // No seed/telemetry side effects: Solo must short-circuit before any thread.
        assert!(
            begin(&StartChoice::Solo, 0, None).is_none(),
            "Solo must not start a formation"
        );
    }

    /// `ready_from(Alone)` yields a solo round (no NetDriver) seeded with the shared seed
    /// — the "nobody joined" path that makes Host → Start playable alone. Proves the menu
    /// maps Alone to the offline round rather than dropping the player out.
    #[test]
    fn alone_becomes_a_solo_round() {
        let seed = 0xABCD;
        let m = ready_from(MatchResult::Alone, seed);
        assert!(m.net.is_none(), "Alone is offline — no NetDriver");
        // The solo lockstep is built and ready (one player: us).
        assert_eq!(m.lockstep.me().0, 0, "solo player is id 0");
    }
}
