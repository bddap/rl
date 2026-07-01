//! Live telemetry side-channel: each game instance optionally streams structured
//! state to one collector so the round can be watched/debugged remotely.
//!
//! # Why this can't perturb the game
//!
//! The determinism of [`crate::sim`] + [`crate::lockstep`] is sacrosanct,
//! and the deployed peers pair over one specific lockstep ALPN
//! ([`crate::transport::ALPN`]). So telemetry is built to be provably unable to
//! touch either:
//!
//! - **Its own iroh endpoint + its own ALPN.** A [`TelemetrySender`] binds a SEPARATE
//!   [`iroh::Endpoint`] (not the game's) and dials the collector on
//!   [`TELEMETRY_ALPN`], which is distinct from the lockstep ALPN. The lockstep
//!   transport — its endpoint, its ALPN, its wire format, its pairing barrier — is
//!   byte-for-byte untouched; a telemetry endpoint cannot answer a lockstep dial or
//!   vice-versa (different ALPN ⇒ the QUIC handshake rejects a cross-dial).
//! - **Read-only on the sim.** Telemetry only READS already-computed state (tick,
//!   `state_hash`, outcome, …) to report it. Nothing here ever feeds back into the
//!   sim, so it adds zero nondeterminism — the same firewall the render client honors.
//! - **Best-effort, never blocking.** Events go through a bounded channel to a
//!   background task; if the channel is full or the send fails the event is DROPPED.
//!   A dead/absent collector, a slow link, a serialization hiccup — none can stall or
//!   crash the game. Losing telemetry is always preferable to perturbing the round.
//!
//! The collector ([`run_collector`]) binds an endpoint under a FIXED secret key so its
//! endpoint id is stable across restarts (senders can be configured with a constant
//! id), accepts many senders at once, and prints one human-readable line per event.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use iroh::endpoint::{Connection, RecvStream, SendStream, presets};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::{Endpoint, EndpointId, SecretKey};
use iroh_mdns_address_lookup::MdnsAddressLookup;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::sim::{Input, Outcome, Sim};

/// How many applied ticks between sampled telemetry snapshots. The high-frequency
/// [`Tick`](TelemetryEvent::Tick)/[`Input`](TelemetryEvent::Input) events are thinned to
/// one per this many ticks so a 30 Hz round streams a couple events/sec per deck instead
/// of flooding the collector. At [`TICK_HZ`](crate::sim::TICK_HZ)
/// = 30 that's ~one snapshot per simulated second, and both drivers (`game net` and the
/// windowed client) share this one constant so their feeds read the same.
pub const TELEMETRY_TICK_EVERY: u64 = 30;

/// serde shim for the foreign [`Outcome`] enum so it rides the telemetry wire by value
/// (the sim's own type, no presentation string). Kept here, not on [`Outcome`] itself, so
/// the deterministic sim module stays free of any telemetry/serde concern. Variants must
/// track [`Outcome`] exactly.
#[derive(Serialize, Deserialize)]
#[serde(remote = "Outcome")]
enum OutcomeWire {
    Ongoing,
    Extracted,
    Wiped,
}

/// ALPN for the telemetry side-channel. DISTINCT from
/// [`crate::transport::ALPN`] (the lockstep wire) on purpose: a telemetry dial
/// and a lockstep dial can never be confused, so wiring telemetry in cannot perturb
/// pairing or the deterministic transport. Bump the trailing version on any
/// incompatible change to the telemetry frame so a mismatched collector/sender refuse
/// to talk rather than mis-decode (telemetry-only — never affects the game wire).
pub const TELEMETRY_ALPN: &[u8] = b"bddap-rl-telemetry/0";

/// mDNS service name for telemetry discovery. CRUCIALLY DISTINCT from the game's
/// [`SERVICE_NAME`](crate::transport::SERVICE_NAME): if telemetry reused the game's
/// service, the game's discovery loop would see the collector + every telemetry endpoint
/// as candidate game peers and the extra endpoints would perturb match formation (a peer
/// could try to pair with a telemetry endpoint, or the churn delays the real peers
/// finding each other). A separate service name puts telemetry in its own discovery
/// namespace — the collector and senders find each other, and the game's pairing never
/// sees them.
pub const TELEMETRY_SERVICE_NAME: &str = "bddap-rl-telemetry";

/// Default path of the collector's persisted secret key. Kept OUT of the nix store and
/// the repo (it's a private key); generated on first run if absent so the collector's
/// endpoint id is stable across restarts without any manual key handling.
pub const DEFAULT_KEY_PATH: &str = "~/.config/rl-telemetry/collector.key";

/// One observed signal from a running game instance — the live-debug surface. Kept
/// compact: the high-frequency variants ([`Tick`](TelemetryEvent::Tick) /
/// [`Input`](TelemetryEvent::Input)) are SAMPLED by the sender (every Nth tick) so a
/// round can't flood the collector. Serialized with serde (bincode on the wire); the
/// format is internal to this side-channel, so it can evolve freely behind
/// [`TELEMETRY_ALPN`]'s version without any impact on the game wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    /// The match-formation barrier is still converging: `live` peers seen so far of
    /// `expect`. Emitted as the live count changes, so the collector shows each deck's
    /// roster filling up.
    RosterForming { live: usize, expect: usize },
    /// The barrier closed: the agreed roster (short endpoint ids, sorted) and its
    /// roster hash. Every peer that reports this for one match MUST carry the identical
    /// `roster_hash` — a mismatch across decks is a formation bug, visible at a glance.
    RosterAgreed {
        members: Vec<String>,
        roster_hash: u64,
        me: u8,
    },
    /// Formation failed (timed out without agreement) — the round never started.
    RosterFailed { reason: String },
    /// A periodic sim snapshot (sampled): the applied `tick`, its full `state_hash`,
    /// and the `roster` size (player count). Comparing the `(tick, state_hash)` across
    /// decks is the live cross-peer desync check. `roster` is the live match player count
    /// (us + peers — it grows on a mid-game join), reported identically by every driver
    /// so the field means one thing.
    Tick {
        tick: u64,
        state_hash: u64,
        roster: usize,
    },
    /// A sampled local input: the tick it applies at and the compact axes/buttons, so
    /// the operator can see what a given deck's player is doing (e.g. a stuck stick, or
    /// nobody moving). Compact ints, not the full struct.
    Input {
        tick: u64,
        strafe: i16,
        forward: i16,
        look: i16,
        buttons: u8,
    },
    /// The round resolved. Carries the deciding [`Outcome`] (never `Ongoing` — the
    /// constructor takes the sim's settled outcome) plus the final tick + state hash, so
    /// the collector records where each deck ended (and whether they agreed on the ending).
    RoundDecided {
        #[serde(with = "OutcomeWire")]
        outcome: Outcome,
        tick: u64,
        state_hash: u64,
    },
    /// A fault a deck detected (e.g. the armed crab going non-finite, rl#137) — a loud failure
    /// mirrored to the collector immediately so a remote operator sees it the instant it happens.
    Fault { msg: String },
}

impl TelemetryEvent {
    /// Build a [`Tick`](TelemetryEvent::Tick) by READING an already-stepped sim — never
    /// mutating it. `roster` is the live match player count (us + peers).
    pub fn tick(sim: &Sim, roster: usize) -> Self {
        TelemetryEvent::Tick {
            tick: sim.tick(),
            state_hash: sim.state_hash(),
            roster,
        }
    }

    /// Build an [`Input`](TelemetryEvent::Input) from the local input for `tick`.
    pub fn input(tick: u64, input: Input) -> Self {
        TelemetryEvent::Input {
            tick,
            strafe: input.move_strafe,
            forward: input.move_forward,
            look: input.look_yaw,
            buttons: input.buttons,
        }
    }

    /// Build a [`RoundDecided`](TelemetryEvent::RoundDecided) snapshot from a sim whose
    /// round has actually ended. Callers gate on `sim.outcome() != Outcome::Ongoing`; the
    /// debug assert keeps that invariant honest (a `RoundDecided` carrying `Ongoing` would
    /// be a contradiction on the feed).
    pub fn round_decided(sim: &Sim) -> Self {
        let outcome = sim.outcome();
        debug_assert_ne!(
            outcome,
            Outcome::Ongoing,
            "RoundDecided requires a decided round"
        );
        TelemetryEvent::RoundDecided {
            outcome,
            tick: sim.tick(),
            state_hash: sim.state_hash(),
        }
    }

    /// Whether this event may be dropped under queue pressure. Only the high-frequency
    /// SAMPLED variants ([`Tick`](TelemetryEvent::Tick)/[`Input`](TelemetryEvent::Input))
    /// are sheddable — losing one is just coarser sampling. Everything else (faults, round
    /// results, roster outcomes) is a CRITICAL one-shot signal that must never be silently
    /// dropped; [`TelemetrySender::send`] routes those on the unbounded lane (rl#125).
    fn is_sheddable(&self) -> bool {
        // New variants default to critical (the never-dropped lane) — the safe side: an
        // unclassified event grows memory loudly rather than vanishing silently.
        matches!(
            self,
            TelemetryEvent::Tick { .. } | TelemetryEvent::Input { .. }
        )
    }

    /// One human-readable line for the collector feed (the per-event suffix after the
    /// source tag + timestamp). Dense and scannable — the operator reads dozens of
    /// these streaming by.
    fn render(&self) -> String {
        match self {
            TelemetryEvent::RosterForming { live, expect } => {
                format!("forming roster: {live}/{expect} live")
            }
            TelemetryEvent::RosterAgreed {
                members,
                roster_hash,
                me,
            } => format!(
                "ROSTER AGREED: {} player(s) {:?}, me=P{me}, roster_hash={roster_hash:#018x}",
                members.len(),
                members
            ),
            TelemetryEvent::RosterFailed { reason } => format!("ROSTER FAILED: {reason}"),
            TelemetryEvent::Tick {
                tick,
                state_hash,
                roster,
            } => {
                format!("tick={tick:>6} hash={state_hash:#018x} roster={roster}")
            }
            TelemetryEvent::Input {
                tick,
                strafe,
                forward,
                look,
                buttons,
            } => format!(
                "input@{tick:>6} strafe={strafe:>5} fwd={forward:>5} look={look:>5} btn={buttons:#04x}"
            ),
            TelemetryEvent::RoundDecided {
                outcome,
                tick,
                state_hash,
            } => format!(
                "ROUND DECIDED: {} @tick {tick} hash={state_hash:#018x}",
                outcome_str(*outcome)
            ),
            TelemetryEvent::Fault { msg } => format!("FAULT: {msg}"),
        }
    }
}

/// Stable string for an [`Outcome`] in telemetry (decoupled from the sim's `Debug`).
fn outcome_str(o: Outcome) -> &'static str {
    match o {
        Outcome::Ongoing => "Ongoing",
        Outcome::Extracted => "Extracted (WON)",
        Outcome::Wiped => "Wiped (LOST)",
    }
}

/// What actually crosses the telemetry wire: an event tagged with WHO sent it (the
/// sender's game endpoint id, so the collector attributes each line to a deck even
/// though telemetry rides a different endpoint) and WHEN (wall-clock ms, the operator's
/// clock — telemetry is for humans, so a real timestamp beats a tick number here).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    /// The sender's GAME endpoint id bytes (32) — the identity the lockstep roster uses,
    /// NOT the telemetry endpoint's id, so the operator correlates a line with the deck
    /// it sees in `RosterAgreed`.
    pub game_id: [u8; 32],
    /// Wall-clock milliseconds since the Unix epoch at send time.
    pub wall_ms: u64,
    pub event: TelemetryEvent,
}

/// Upper bound on one encoded [`Envelope`], to reject a garbled length before
/// allocating. Events are tiny (a roster tops out at 256 short ids ≈ a few KiB); 64 KiB
/// is generous slack.
const MAX_FRAME_LEN: usize = 64 * 1024;

/// Bounded queue depth from the game thread to the telemetry I/O task. Small on
/// purpose: telemetry must never apply backpressure to the game, so when the link
/// can't keep up we DROP (see [`TelemetrySender::send`]) rather than grow unbounded or
/// block. A handful of events of slack absorbs normal jitter.
const QUEUE_DEPTH: usize = 256;

/// Handle the game holds to push telemetry. Cloneable and cheap; every method is
/// non-blocking and never stalls the game. Dropping all clones lets the background task
/// finish and the telemetry endpoint close.
///
/// Two priority lanes, so failure signals can't be lost behind chatter (rl#125):
/// - `shed_tx` — the high-frequency SAMPLED events ([`Tick`](TelemetryEvent::Tick) /
///   [`Input`](TelemetryEvent::Input)). Bounded; dropped under pressure is the
///   documented thinning.
/// - `crit_tx` — everything else (faults, round/roster outcomes). UNBOUNDED so a real
///   failure signal is NEVER silently dropped under queue pressure. Normal volume is
///   O(few) per round; the one way it grows is a sustained fault storm into a stalled
///   collector (bounded by the tick rate, tiny events) — losing-telemetry-is-fine still
///   holds, but never at the cost of a vanished fault, so unbounded is the right trade.
#[derive(Clone)]
pub struct TelemetrySender {
    shed_tx: mpsc::Sender<Envelope>,
    crit_tx: mpsc::UnboundedSender<Envelope>,
    game_id: [u8; 32],
}

impl TelemetrySender {
    /// Connect a telemetry stream to `collector` (its stable endpoint id) and return a
    /// handle to push events. Binds a SEPARATE LAN iroh endpoint (its own id, its own
    /// mDNS publish) and dials the collector on [`TELEMETRY_ALPN`] — the game's lockstep
    /// endpoint is never touched. `game_id` is THIS instance's game endpoint id, stamped
    /// on every event so the collector tags it.
    ///
    /// Best-effort end to end: if the collector can't be reached the background task
    /// logs once and keeps draining (so the game never blocks on it); events sent
    /// meanwhile are dropped. Returns an error only if the local telemetry endpoint
    /// can't be bound at all — the caller treats even that as "run without telemetry".
    pub async fn connect(collector: EndpointId, game_id: [u8; 32]) -> Result<Self> {
        let endpoint = bind_telemetry_endpoint()
            .await
            .context("binding telemetry endpoint")?;
        let (shed_tx, shed_rx) = mpsc::channel(QUEUE_DEPTH);
        let (crit_tx, crit_rx) = mpsc::unbounded_channel();
        tokio::spawn(sender_task(endpoint, collector, shed_rx, crit_rx));
        Ok(Self {
            shed_tx,
            crit_tx,
            game_id,
        })
    }

    /// Queue one event for delivery, stamped with our game id + the current wall clock.
    /// NON-BLOCKING and never stalls the game (safe from the hot sim loop), but priority
    /// depends on the event:
    /// - Sheddable ([`Tick`](TelemetryEvent::Tick)/[`Input`](TelemetryEvent::Input)) goes
    ///   on the bounded lane; a full queue silently drops it — that IS the sampling.
    /// - Critical (faults, round/roster outcomes) goes on the unbounded lane, so a real
    ///   failure signal is never dropped under pressure. It can only fail to enqueue if the
    ///   I/O task is already gone (all receivers dropped) — surfaced LOUDLY, never silent
    ///   (rl#125: a vanished fault/round result with zero trace is exactly what this bans).
    pub fn send(&self, event: TelemetryEvent) {
        let env = Envelope {
            game_id: self.game_id,
            wall_ms: now_ms(),
            event,
        };
        if env.event.is_sheddable() {
            // try_send (never .await): a full bounded queue or a closed receiver just drops
            // this sampled event — the side-channel is sacrificial relative to the game.
            let _ = self.shed_tx.try_send(env);
        } else if let Err(e) = self.crit_tx.send(env) {
            // Unbounded send fails only when the I/O task has stopped (receiver dropped),
            // so this is the telemetry stream having torn down, not back-pressure. Never
            // let a critical event vanish silently.
            tracing::error!(
                "telemetry: critical event lost (I/O task gone): {:?}",
                e.0.event
            );
        }
    }
}

/// Bind the telemetry sender's own LAN endpoint: same LAN-only shape as the game's
/// transport (relay disabled, mDNS the sole lookup, scoped to the TELEMETRY service —
/// [`TELEMETRY_SERVICE_NAME`], not the game's — so it can resolve the collector by id)
/// but a DISTINCT endpoint, so nothing here shares state with the lockstep endpoint. It
/// only ever dials out (never accepts), so it adds no game-visible surface to the LAN.
async fn bind_telemetry_endpoint() -> Result<Endpoint> {
    let endpoint = Endpoint::builder(presets::Minimal)
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .context("binding iroh endpoint")?;
    attach_lan_mdns(&endpoint).context("attaching mDNS to telemetry endpoint")?;
    Ok(endpoint)
}

/// The background I/O task: dial the collector once (retrying a few times for the case
/// where the game launches before the collector's mDNS record propagates), then drain
/// the queue, framing each [`Envelope`] onto the telemetry stream. Any send error tears
/// down and stops — the game keeps running regardless; telemetry just goes quiet. The
/// task ends when every [`TelemetrySender`] clone is dropped (the channel closes).
async fn sender_task(
    endpoint: Endpoint,
    collector: EndpointId,
    mut shed_rx: mpsc::Receiver<Envelope>,
    mut crit_rx: mpsc::UnboundedReceiver<Envelope>,
) {
    let conn = match dial_collector(&endpoint, collector).await {
        Ok(c) => c,
        Err(e) => {
            // One line, then drain-and-drop so the game's sends never block on a full
            // queue waiting for a connection that isn't coming.
            tracing::warn!(
                "telemetry: could not reach collector {collector}: {e:#} — running without telemetry"
            );
            drain(&mut shed_rx, &mut crit_rx).await;
            return;
        }
    };
    let mut send = match conn.open_uni().await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("telemetry: opening stream failed: {e:#}");
            drain(&mut shed_rx, &mut crit_rx).await;
            return;
        }
    };
    tracing::info!(
        "telemetry: streaming to collector {}",
        collector.fmt_short()
    );
    loop {
        // `biased`: always prefer the critical lane, so a fault/round result is written
        // ahead of the sampled chatter rather than waiting behind it. Critical events are
        // O(few) per round, so this can't starve the sampled lane.
        let env = tokio::select! {
            biased;
            Some(env) = crit_rx.recv() => env,
            Some(env) = shed_rx.recv() => env,
            else => break, // both lanes closed (all senders dropped)
        };
        if let Err(e) = write_frame(&mut send, &env).await {
            tracing::warn!("telemetry: send failed, stopping stream: {e:#}");
            break;
        }
    }
    // Best-effort flush; ignore errors (we're shutting the side-channel down anyway).
    let _ = send.finish();
    endpoint.close().await;
}

/// Drain and drop both lanes until every sender is gone — used when the collector is
/// unreachable, so the game's non-blocking sends never wedge on a full bounded queue.
async fn drain(
    shed_rx: &mut mpsc::Receiver<Envelope>,
    crit_rx: &mut mpsc::UnboundedReceiver<Envelope>,
) {
    loop {
        tokio::select! {
            Some(_) = shed_rx.recv() => {}
            Some(_) = crit_rx.recv() => {}
            else => break,
        }
    }
}

/// Dial the collector on [`TELEMETRY_ALPN`], retrying briefly so a game that starts
/// before the collector's mDNS address has propagated still connects. The id resolves
/// to an address via the shared mDNS service (same mechanism the lockstep uses for
/// peers); we just dial a different ALPN.
async fn dial_collector(endpoint: &Endpoint, collector: EndpointId) -> Result<Connection> {
    const ATTEMPTS: u32 = 10;
    const BACKOFF: Duration = Duration::from_millis(500);
    let mut last_err = None;
    for attempt in 0..ATTEMPTS {
        match endpoint.connect(collector, TELEMETRY_ALPN).await {
            Ok(conn) => return Ok(conn),
            Err(e) => {
                tracing::debug!(attempt, "telemetry dial failed: {e:#}");
                last_err = Some(anyhow::anyhow!("{e:#}"));
                tokio::time::sleep(BACKOFF).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no dial attempts")))
}

// ---------------------------------------------------------------------------
// Collector
// ---------------------------------------------------------------------------

/// Run the telemetry collector until interrupted (Ctrl-C). Binds an endpoint under the
/// FIXED secret key at `key_path` (generated on first use — see [`load_or_create_key`])
/// so its endpoint id is STABLE across restarts, prints that id (the value senders dial
/// with `--telemetry`), then accepts telemetry streams from any number of game
/// instances at once and prints a single merged, human-readable feed to stdout — one
/// line per event, prefixed with the short SOURCE game id so the daemon can `Monitor`
/// it and watch every deck live.
pub async fn run_collector(key_path: &Path) -> Result<()> {
    let secret = load_or_create_key(key_path)?;
    let endpoint = Endpoint::builder(presets::Minimal)
        .secret_key(secret)
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .context("binding collector endpoint")?;
    attach_lan_mdns(&endpoint).context("attaching mDNS to collector endpoint")?;
    let id = endpoint.id();

    // The id line is the operator-facing contract: it's what goes into `--telemetry` on
    // each game. Print it prominently and flush so a `Monitor`ing daemon sees it at once.
    println!("telemetry-collector endpoint id: {id}");
    println!("telemetry-collector short id:    {}", id.fmt_short());
    println!("(pass this id to a game as `--telemetry {id}`)");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let _router = Router::builder(endpoint.clone())
        .accept(TELEMETRY_ALPN, CollectorProto)
        .spawn();

    // Park until Ctrl-C; the router serves senders on its own tasks meanwhile.
    tokio::signal::ctrl_c()
        .await
        .context("waiting for shutdown signal")?;
    println!("telemetry-collector: shutting down");
    endpoint.close().await;
    Ok(())
}

/// Accept handler for inbound telemetry streams. Stateless: each connection is one
/// sender; spawn a reader that prints its framed events tagged by source.
#[derive(Clone, Debug)]
struct CollectorProto;

impl ProtocolHandler for CollectorProto {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let peer = connection.remote_id();
        // A sender opens ONE uni stream and pushes frames until it exits. The telemetry
        // endpoint id (`peer`) is logged once; the per-event source tag comes from the
        // envelope's GAME id, which is what the operator correlates with the roster.
        match connection.accept_uni().await {
            Ok(recv) => {
                tokio::spawn(async move {
                    if let Err(e) = collector_read_loop(recv).await {
                        tracing::debug!(%peer, "telemetry reader ended: {e:#}");
                    }
                });
            }
            Err(e) => tracing::debug!(%peer, "telemetry accept_uni failed: {e:#}"),
        }
        Ok(())
    }
}

/// Read length-framed [`Envelope`]s from one sender until its stream closes, printing
/// each as a human line tagged with the short SOURCE game id + a UTC-ish wall stamp.
/// A decode error closes just this sender's stream (logged at debug), never the
/// collector.
async fn collector_read_loop(mut recv: RecvStream) -> Result<()> {
    loop {
        let mut lenb = [0u8; 4];
        if recv.read_exact(&mut lenb).await.is_err() {
            return Ok(()); // clean EOF
        }
        let len = u32::from_le_bytes(lenb) as usize;
        anyhow::ensure!(len <= MAX_FRAME_LEN, "telemetry frame {len} exceeds cap");
        let mut buf = vec![0u8; len];
        recv.read_exact(&mut buf)
            .await
            .context("reading telemetry frame")?;
        let env: Envelope = bincode::deserialize(&buf).context("decoding telemetry envelope")?;
        print_event(&env);
    }
}

/// Print one event line: `[<short-game-id>] <hh:mm:ss.mmm> <event>`. Flushed each time
/// so a `tail -F`/`Monitor` sees events live, not buffered.
fn print_event(env: &Envelope) {
    let src = EndpointId::from_bytes(&env.game_id)
        .map(|id| id.fmt_short().to_string())
        .unwrap_or_else(|_| hex8(&env.game_id));
    println!("[{src}] {} {}", clock(env.wall_ms), env.event.render());
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

/// Format a wall-clock-ms as `HH:MM:SS.mmm` UTC for the feed. Pure integer arithmetic
/// (no chrono dep): seconds-of-day is all the operator needs to correlate events.
fn clock(wall_ms: u64) -> String {
    let secs = wall_ms / 1000;
    let ms = wall_ms % 1000;
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// First 8 bytes of an id as hex, the fallback source tag if an id won't parse back
/// into an [`EndpointId`] (should never happen — it was a valid id when sent).
fn hex8(id: &[u8; 32]) -> String {
    id[..4].iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Attach a TELEMETRY-scoped mDNS address lookup to `endpoint` and force an address
/// publish via the shared [`crate::transport::publish_lan_addr`], so a bare-id dial
/// (sender → collector) resolves on the LAN. Uses [`TELEMETRY_SERVICE_NAME`] — a SEPARATE
/// service from the game's — so these endpoints live in their own discovery namespace and
/// never perturb game pairing. Factored so sender + collector share it.
fn attach_lan_mdns(endpoint: &Endpoint) -> Result<()> {
    let mdns = MdnsAddressLookup::builder()
        .service_name(TELEMETRY_SERVICE_NAME)
        .build(endpoint.id())
        .context("starting mDNS discovery")?;
    endpoint
        .address_lookup()
        .context("endpoint has no address lookup registry")?
        .add(mdns);
    // Win the mDNS publish race (shared with the game transport — same first-publish vs.
    // service-loop-startup work-around), scoped to the telemetry namespace. Spawned
    // because attach is sync; the endpoint owns the lookup so the publish lands before any
    // dial completes in practice.
    let ep = endpoint.clone();
    tokio::spawn(async move {
        if let Err(e) = crate::transport::publish_lan_addr(&ep, TELEMETRY_SERVICE_NAME).await {
            tracing::debug!("telemetry mDNS publish: {e:#}");
        }
    });
    Ok(())
}

/// Write one `[len:u32 LE][bincode(Envelope)]` frame to the telemetry stream.
async fn write_frame(send: &mut SendStream, env: &Envelope) -> Result<()> {
    let body = bincode::serialize(env).context("encoding telemetry envelope")?;
    anyhow::ensure!(body.len() <= MAX_FRAME_LEN, "telemetry frame too large");
    let len = body.len() as u32;
    send.write_all(&len.to_le_bytes()).await?;
    send.write_all(&body).await?;
    Ok(())
}

/// Load the collector's persistent secret key from `path`, or generate + persist one on
/// first run so the endpoint id is STABLE across restarts (the whole point of a fixed
/// key — senders are configured with a constant id). Stored as 64 hex chars; the parent
/// dir is created and the file written 0600 (it's a private key).
pub fn load_or_create_key(path: &Path) -> Result<SecretKey> {
    let path = expand_tilde(path);
    if let Ok(text) = std::fs::read_to_string(&path) {
        let bytes = decode_key_hex(text.trim())
            .with_context(|| format!("parsing telemetry key at {}", path.display()))?;
        return Ok(SecretKey::from_bytes(&bytes));
    }
    // Absent (or unreadable) → generate a fresh key and persist it.
    let secret = SecretKey::generate();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let hex: String = secret
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    std::fs::write(&path, format!("{hex}\n"))
        .with_context(|| format!("writing telemetry key to {}", path.display()))?;
    set_key_permissions(&path);
    Ok(secret)
}

/// Decode 64 hex chars into a 32-byte key.
fn decode_key_hex(s: &str) -> Result<[u8; 32]> {
    anyhow::ensure!(s.len() == 64, "key must be 64 hex chars, got {}", s.len());
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("bad hex at byte {i}"))?;
    }
    Ok(out)
}

/// Restrict the key file to 0600 (owner read/write) — it's a private key. Unix-only;
/// a no-op elsewhere (the collector runs on bothouse/Linux).
#[cfg(unix)]
fn set_key_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}
#[cfg(not(unix))]
fn set_key_permissions(_path: &Path) {}

/// Expand a leading `~/` to `$HOME` so [`DEFAULT_KEY_PATH`] is usable as-is. Only the
/// leading `~/` form is handled (all we use); anything else is returned unchanged.
fn expand_tilde(path: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~")
        && let Some(home) = std::env::var_os("HOME")
    {
        return Path::new(&home).join(stripped);
    }
    path.to_path_buf()
}

/// Milliseconds since the Unix epoch (wall clock). Telemetry timestamps are for the
/// human operator, so a wall clock is right here (unlike the sim, which must never read
/// one). A pre-epoch clock (impossible in practice) reads as 0.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Shorten a list of endpoint ids to their `fmt_short` forms for a compact roster in
/// [`TelemetryEvent::RosterAgreed`]. Sorted by the caller already (the agreed roster is
/// sorted); we only shorten.
pub fn short_ids(ids: &[EndpointId]) -> Vec<String> {
    ids.iter().map(|id| id.fmt_short().to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sender wired to in-test receivers, bypassing the iroh endpoint that
    /// [`TelemetrySender::connect`] would bind — the priority routing in `send` is what's
    /// under test, not the transport.
    fn test_sender() -> (
        TelemetrySender,
        mpsc::Receiver<Envelope>,
        mpsc::UnboundedReceiver<Envelope>,
    ) {
        let (shed_tx, shed_rx) = mpsc::channel(QUEUE_DEPTH);
        let (crit_tx, crit_rx) = mpsc::unbounded_channel();
        (
            TelemetrySender {
                shed_tx,
                crit_tx,
                game_id: [0u8; 32],
            },
            shed_rx,
            crit_rx,
        )
    }

    fn a_tick(tick: u64) -> TelemetryEvent {
        TelemetryEvent::Tick {
            tick,
            state_hash: 0,
            roster: 1,
        }
    }

    /// rl#125: a Fault arriving while the sampled queue is saturated must still be
    /// delivered — the whole point of the side-channel is to surface faults. The old
    /// uniform `try_send` would have dropped it along with the ticks.
    #[test]
    fn critical_event_survives_sheddable_queue_pressure() {
        let (sender, _shed_rx, mut crit_rx) = test_sender();
        // Overfill the bounded sampled lane; these silently drop past QUEUE_DEPTH.
        for tick in 0..(QUEUE_DEPTH as u64 + 64) {
            sender.send(a_tick(tick));
        }
        // A fault under that exact pressure must still be queued on the critical lane.
        sender.send(TelemetryEvent::Fault {
            msg: "desync".into(),
        });
        match crit_rx.try_recv() {
            Ok(env) => assert!(
                matches!(env.event, TelemetryEvent::Fault { .. }),
                "expected the Fault, got {:?}",
                env.event
            ),
            other => panic!("critical Fault was dropped under queue pressure: {other:?}"),
        }
    }

    /// The sheddable/critical split: only the sampled high-frequency variants may be
    /// dropped; faults and outcomes never are.
    #[test]
    fn only_sampled_events_are_sheddable() {
        assert!(a_tick(0).is_sheddable());
        assert!(
            TelemetryEvent::Input {
                tick: 0,
                strafe: 0,
                forward: 0,
                look: 0,
                buttons: 0,
            }
            .is_sheddable()
        );
        for ev in [
            TelemetryEvent::Fault { msg: "x".into() },
            TelemetryEvent::RoundDecided {
                outcome: Outcome::Extracted,
                tick: 1,
                state_hash: 0,
            },
            TelemetryEvent::RosterFailed {
                reason: "timeout".into(),
            },
            TelemetryEvent::RosterForming { live: 1, expect: 2 },
            TelemetryEvent::RosterAgreed {
                members: vec![],
                roster_hash: 0,
                me: 0,
            },
        ] {
            assert!(!ev.is_sheddable(), "{ev:?} must be critical (never dropped)");
        }
    }
}
