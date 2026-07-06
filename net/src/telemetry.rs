
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

const TELEMETRY_TICK_EVERY: u64 = crate::sim::TICK_HZ;

/// The first sampling boundary strictly after `now` — the ONE cursor rule every driver
/// advances its telemetry watermark by (initialize with `next_sample_tick(0)`), so the
/// feeds sample on identical tick boundaries.
pub fn next_sample_tick(now: u64) -> u64 {
    (now / TELEMETRY_TICK_EVERY + 1) * TELEMETRY_TICK_EVERY
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Outcome")]
enum OutcomeWire {
    Ongoing,
    Extracted,
    Wiped,
}

pub const TELEMETRY_ALPN: &[u8] = b"bddap-rl-telemetry/1";

pub const TELEMETRY_SERVICE_NAME: &str = "bddap-rl-telemetry";

pub const DEFAULT_KEY_PATH: &str = "~/.config/rl-telemetry/collector.key";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    RosterForming { live: usize, expect: usize },
    RosterAgreed {
        members: Vec<String>,
        roster_hash: u64,
        me: u8,
    },
    /// A roster change that DIDN'T happen: formation failed (timed out without
    /// agreement — the round never started) or a mid-game join was refused.
    RosterFailed { reason: String },
    /// A rostered player DEPARTED mid-match — their link closed (clean exit, dead
    /// connection, or wedged-peer eviction) and the match continues without them
    /// (rl#198). `endpoint` is the short endpoint id.
    Departed { player: u8, endpoint: String },
    /// A mid-game joiner was ADMITTED as `player`; the roster change lands at
    /// `effective_tick`. A refused joiner reports as
    /// [`RosterFailed`](TelemetryEvent::RosterFailed) instead.
    Admitted {
        player: u8,
        endpoint: String,
        effective_tick: u64,
    },
    Tick {
        tick: u64,
        state_hash: u64,
        roster: usize,
    },
    Input {
        tick: u64,
        strafe: i16,
        forward: i16,
        look: i16,
        buttons: u8,
    },
    RoundDecided {
        #[serde(with = "OutcomeWire")]
        outcome: Outcome,
        tick: u64,
        state_hash: u64,
    },
    Fault { msg: String },
}

impl TelemetryEvent {
    pub fn tick(sim: &Sim, roster: usize) -> Self {
        TelemetryEvent::Tick {
            tick: sim.tick(),
            state_hash: sim.state_hash(),
            roster,
        }
    }

    pub fn input(tick: u64, input: Input) -> Self {
        TelemetryEvent::Input {
            tick,
            strafe: input.move_strafe,
            forward: input.move_forward,
            look: input.look_yaw,
            buttons: input.buttons,
        }
    }

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

    fn is_sheddable(&self) -> bool {
        matches!(
            self,
            TelemetryEvent::Tick { .. } | TelemetryEvent::Input { .. }
        )
    }

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
            TelemetryEvent::Departed { player, endpoint } => {
                format!("DEPARTED: P{player} ({endpoint}) — continuing without them")
            }
            TelemetryEvent::Admitted {
                player,
                endpoint,
                effective_tick,
            } => format!("ADMITTED: {endpoint} as P{player}, effective at tick {effective_tick}"),
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

fn outcome_str(o: Outcome) -> &'static str {
    match o {
        Outcome::Ongoing => "Ongoing",
        Outcome::Extracted => "Extracted (WON)",
        Outcome::Wiped => "Wiped (LOST)",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub game_id: [u8; 32],
    pub wall_ms: u64,
    pub event: TelemetryEvent,
}

const MAX_FRAME_LEN: usize = 64 * 1024;

const QUEUE_DEPTH: usize = 256;

#[derive(Clone)]
pub struct TelemetrySender {
    shed_tx: mpsc::Sender<Envelope>,
    crit_tx: mpsc::UnboundedSender<Envelope>,
    game_id: [u8; 32],
}

impl TelemetrySender {
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

    pub fn send(&self, event: TelemetryEvent) {
        let env = Envelope {
            game_id: self.game_id,
            wall_ms: now_ms(),
            event,
        };
        if env.event.is_sheddable() {
            let _ = self.shed_tx.try_send(env);
        } else if let Err(e) = self.crit_tx.send(env) {
            tracing::error!(
                "telemetry: critical event lost (I/O task gone): {:?}",
                e.0.event
            );
        }
    }
}

/// Drain + surface the server's chronic input-starvation reports (rl#213): one local `warn!`
/// per report, mirrored as a [`Fault`](TelemetryEvent::Fault) when a collector is wired. The
/// ONE drain policy every server driver (windowed + headless host) shares, so log-and-mirror
/// can't drift between them. Lives here, not on [`Server`], so server.rs stays free of
/// telemetry/log concerns; `server`/`tel` are `Option`s so a driver arm without one (a remote
/// client, a collector-less run) calls it the same way. Reports are rate-limited server-side
/// (≤1 per player per cooldown), so each one is worth a line.
pub fn surface_starvation(
    server: Option<&mut crate::server::Server>,
    tel: Option<&TelemetrySender>,
) {
    let Some(server) = server else { return };
    for r in server.take_starvation_reports() {
        tracing::warn!("{r}");
        if let Some(t) = tel {
            t.send(TelemetryEvent::Fault { msg: r.to_string() });
        }
    }
}

async fn bind_telemetry_endpoint() -> Result<Endpoint> {
    let endpoint = Endpoint::builder(presets::Minimal)
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .context("binding iroh endpoint")?;
    attach_lan_mdns(&endpoint).context("attaching mDNS to telemetry endpoint")?;
    Ok(endpoint)
}

async fn sender_task(
    endpoint: Endpoint,
    collector: EndpointId,
    mut shed_rx: mpsc::Receiver<Envelope>,
    mut crit_rx: mpsc::UnboundedReceiver<Envelope>,
) {
    let conn = match dial_collector(&endpoint, collector).await {
        Ok(c) => c,
        Err(e) => {
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
        let env = tokio::select! {
            biased;
            Some(env) = crit_rx.recv() => env,
            Some(env) = shed_rx.recv() => env,
            else => break,
        };
        if let Err(e) = write_frame(&mut send, &env).await {
            tracing::warn!("telemetry: send failed, stopping stream: {e:#}");
            break;
        }
    }
    let _ = send.finish();
    endpoint.close().await;
}

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

    println!("telemetry-collector endpoint id: {id}");
    println!("telemetry-collector short id:    {}", id.fmt_short());
    println!("(pass this id to a game as `--telemetry {id}`)");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let _router = Router::builder(endpoint.clone())
        .accept(TELEMETRY_ALPN, CollectorProto)
        .spawn();

    tokio::signal::ctrl_c()
        .await
        .context("waiting for shutdown signal")?;
    println!("telemetry-collector: shutting down");
    endpoint.close().await;
    Ok(())
}

#[derive(Clone, Debug)]
struct CollectorProto;

impl ProtocolHandler for CollectorProto {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let peer = connection.remote_id();
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

async fn collector_read_loop(mut recv: RecvStream) -> Result<()> {
    loop {
        let mut lenb = [0u8; 4];
        if recv.read_exact(&mut lenb).await.is_err() {
            return Ok(());
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

fn print_event(env: &Envelope) {
    let src = EndpointId::from_bytes(&env.game_id)
        .map(|id| id.fmt_short().to_string())
        .unwrap_or_else(|_| hex8(&env.game_id));
    println!("[{src}] {} {}", clock(env.wall_ms), env.event.render());
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

fn clock(wall_ms: u64) -> String {
    let secs = wall_ms / 1000;
    let ms = wall_ms % 1000;
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

fn hex8(id: &[u8; 32]) -> String {
    id[..4].iter().map(|b| format!("{b:02x}")).collect()
}


fn attach_lan_mdns(endpoint: &Endpoint) -> Result<()> {
    let mdns = MdnsAddressLookup::builder()
        .service_name(TELEMETRY_SERVICE_NAME)
        .build(endpoint.id())
        .context("starting mDNS discovery")?;
    endpoint
        .address_lookup()
        .context("endpoint has no address lookup registry")?
        .add(mdns);
    let ep = endpoint.clone();
    tokio::spawn(async move {
        if let Err(e) = crate::transport::publish_lan_addr(&ep, TELEMETRY_SERVICE_NAME).await {
            tracing::debug!("telemetry mDNS publish: {e:#}");
        }
    });
    Ok(())
}

async fn write_frame(send: &mut SendStream, env: &Envelope) -> Result<()> {
    let body = bincode::serialize(env).context("encoding telemetry envelope")?;
    anyhow::ensure!(body.len() <= MAX_FRAME_LEN, "telemetry frame too large");
    let len = body.len() as u32;
    send.write_all(&len.to_le_bytes()).await?;
    send.write_all(&body).await?;
    Ok(())
}

pub fn load_or_create_key(path: &Path) -> Result<SecretKey> {
    let path = expand_tilde(path);
    if let Ok(text) = std::fs::read_to_string(&path) {
        let bytes = decode_key_hex(text.trim())
            .with_context(|| format!("parsing telemetry key at {}", path.display()))?;
        return Ok(SecretKey::from_bytes(&bytes));
    }
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

fn decode_key_hex(s: &str) -> Result<[u8; 32]> {
    anyhow::ensure!(s.len() == 64, "key must be 64 hex chars, got {}", s.len());
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("bad hex at byte {i}"))?;
    }
    Ok(out)
}

#[cfg(unix)]
fn set_key_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}
#[cfg(not(unix))]
fn set_key_permissions(_path: &Path) {}

fn expand_tilde(path: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~")
        && let Some(home) = std::env::var_os("HOME")
    {
        return Path::new(&home).join(stripped);
    }
    path.to_path_buf()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn short_ids(ids: &[EndpointId]) -> Vec<String> {
    ids.iter().map(|id| id.fmt_short().to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A Fault arriving while the sampled queue is saturated must still be
    /// delivered — the whole point of the side-channel is to surface faults. A
    /// uniform bounded `try_send` would drop it along with the ticks.
    #[test]
    fn critical_event_survives_sheddable_queue_pressure() {
        let (sender, _shed_rx, mut crit_rx) = test_sender();
        for tick in 0..(QUEUE_DEPTH as u64 + 64) {
            sender.send(a_tick(tick));
        }
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
            TelemetryEvent::Departed {
                player: 1,
                endpoint: "abcd".into(),
            },
            TelemetryEvent::Admitted {
                player: 2,
                endpoint: "abcd".into(),
                effective_tick: 42,
            },
        ] {
            assert!(
                !ev.is_sheddable(),
                "{ev:?} must be critical (never dropped)"
            );
        }
    }
}
