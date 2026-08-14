//! Voice-note delivery (rl#378 stage 2): ship every confirmed take in the voice
//! dir to the assistant's inbox, deleting each local copy once the inbox acks
//! it. One code path for every platform: the game always dials
//! `$RL_VOICE_ENDPOINT` (default 127.0.0.1:4319) — on the TV kit that IS the
//! inbox receiver (machine-local), on a deck it is an `iroh-tunnel forward`
//! into bothouse, so authentication rides the established tunnel allowlist and
//! the transport split lives entirely in deployment config, not here.
//!
//! Wire format, one connection per note: `voice/1 <host> <len>\n` then exactly
//! `len` wav bytes; the receiver answers `ok\n` only after the file is durably
//! stored, and only then does the sender delete. A refused or failed send
//! keeps the wav for the next sweep (confirm or round start), so delivery is
//! at-least-once and a dead inbox never loses a take.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const DEFAULT_ENDPOINT: &str = "127.0.0.1:4319";

/// The recorder caps takes at 120 s ≈ 10.6 MB of 44.1 kHz mono; anything
/// bigger in the dir is not a voice note — leave it alone rather than stream
/// it at the inbox (which enforces the same bound).
const MAX_WAV_BYTES: u64 = 16 * 1024 * 1024;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// One sweep at a time process-wide: the round-start kick and a confirm-time
/// send could otherwise race the same file into a double delivery.
static SWEEP_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// Held for the duration of one sweep; dropping it frees the slot.
pub struct SweepSlot(());

impl Drop for SweepSlot {
    fn drop(&mut self) {
        SWEEP_IN_FLIGHT.store(false, Ordering::SeqCst);
    }
}

/// `None` while another sweep is still running — skip, the running sweep (or
/// the next one) covers the file.
pub fn try_begin_sweep() -> Option<SweepSlot> {
    SWEEP_IN_FLIGHT
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
        .then_some(SweepSlot(()))
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub sent: usize,
    /// Non-empty when a send failed; pending files stay on disk for retry.
    pub error: Option<String>,
}

/// The inbox to dial: `$RL_VOICE_ENDPOINT`, else the machine-local default.
pub fn endpoint() -> String {
    std::env::var("RL_VOICE_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.into())
}

/// Sender identity for the event line — informational, same sourcing as the
/// otel `host.name` resource attribute; authenticity is the transport's job.
pub fn host_tag() -> String {
    let raw = std::env::var("DECK_ID")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_default();
    sanitize_host(&raw)
}

/// The tag rides a whitespace-delimited protocol header and later a shell-side
/// filename, so it is clamped to a conservative alphabet.
fn sanitize_host(raw: &str) -> String {
    let clean: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(32)
        .collect();
    if clean.is_empty() {
        "unknown".into()
    } else {
        clean
    }
}

/// A transport failure (inbox unreachable, IO error) would fail every
/// remaining note identically, so the sweep stops. A per-note rejection must
/// NOT stop it: `continue` keeps one poison file from wedging the queue —
/// everything behind it still ships.
enum DeliverError {
    Transport(String),
    Rejected(String),
}

/// Deliver every pending take in `dir`, oldest first.
pub fn deliver_pending(dir: &Path, endpoint: &str, host: &str) -> Report {
    let mut pending = pending_takes(dir);
    pending.sort();
    let mut report = Report::default();
    for path in pending {
        match deliver_one(&path, endpoint, host) {
            Ok(()) => {
                if let Err(e) = std::fs::remove_file(&path) {
                    // Next sweep re-sends it; the inbox dedups nothing, but a
                    // duplicate note beats a lost one.
                    tracing::warn!("delivered voice note not deleted: {e}");
                }
                report.sent += 1;
            }
            Err(DeliverError::Rejected(e)) => {
                tracing::warn!("voice note rejected ({}): {e}", path.display());
                report.error = Some(e);
            }
            Err(DeliverError::Transport(e)) => {
                tracing::warn!("voice note delivery failed ({}): {e}", path.display());
                report.error = Some(e);
                break;
            }
        }
    }
    report
}

/// Confirmed takes only: `voice-*.wav`, never the writer's `.tmp` staging.
fn pending_takes(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| {
            p.extension().is_some_and(|x| x == "wav")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("voice-"))
        })
        .collect()
}

fn deliver_one(path: &Path, endpoint: &str, host: &str) -> Result<(), DeliverError> {
    use DeliverError::{Rejected, Transport};
    let len = std::fs::metadata(path)
        .map_err(|e| Rejected(e.to_string()))?
        .len();
    if len > MAX_WAV_BYTES {
        return Err(Rejected(format!(
            "{len} bytes exceeds the {MAX_WAV_BYTES} note cap"
        )));
    }
    let bytes = std::fs::read(path).map_err(|e| Rejected(e.to_string()))?;
    let addr = endpoint
        .to_socket_addrs()
        .map_err(|e| Transport(e.to_string()))?
        .next()
        .ok_or_else(|| Transport(format!("{endpoint} resolves to no address")))?;
    let stream =
        TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).map_err(|e| Transport(e.to_string()))?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .map_err(|e| Transport(e.to_string()))?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .map_err(|e| Transport(e.to_string()))?;
    let mut writer = &stream;
    writer
        .write_all(format!("voice/1 {host} {}\n", bytes.len()).as_bytes())
        .and_then(|()| writer.write_all(&bytes))
        .map_err(|e| Transport(e.to_string()))?;
    let mut reply = String::new();
    BufReader::new(&stream)
        .take(16)
        .read_line(&mut reply)
        .map_err(|e| Transport(e.to_string()))?;
    if reply.trim_end() == "ok" {
        Ok(())
    } else {
        Err(Rejected(format!("inbox refused the note: {reply:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    /// Accept one connection, return (header line, body bytes), reply `ok`.
    fn serve_one(listener: TcpListener) -> std::thread::JoinHandle<(String, Vec<u8>)> {
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&stream);
            let mut header = String::new();
            reader.read_line(&mut header).unwrap();
            let len: usize = header
                .trim_end()
                .rsplit(' ')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let mut body = vec![0u8; len];
            reader.read_exact(&mut body).unwrap();
            (&stream).write_all(b"ok\n").unwrap();
            (header, body)
        })
    }

    #[test]
    fn delivers_frames_and_deletes_on_ok() {
        let dir = std::env::temp_dir().join(format!("voice-deliver-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("voice-42.wav");
        std::fs::write(&wav, b"RIFFfake").unwrap();
        // Staging and foreign files must not ship.
        std::fs::write(dir.join("voice-43.tmp"), b"half").unwrap();
        std::fs::write(dir.join("notes.wav"), b"other").unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = serve_one(listener);

        let report = deliver_pending(&dir, &endpoint, "testhost");
        assert_eq!(
            report,
            Report {
                sent: 1,
                error: None
            }
        );
        assert!(!wav.exists(), "acked note must be deleted");
        assert!(dir.join("voice-43.tmp").exists());
        assert!(dir.join("notes.wav").exists());

        let (header, body) = server.join().unwrap();
        assert_eq!(header, "voice/1 testhost 8\n");
        assert_eq!(body, b"RIFFfake");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejected_note_does_not_wedge_the_queue() {
        let dir = std::env::temp_dir().join(format!("voice-deliver-rej-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("voice-1.wav");
        let good = dir.join("voice-2.wav");
        std::fs::write(&bad, b"poison").unwrap();
        std::fs::write(&good, b"RIFFfake").unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = std::thread::spawn(move || {
            // Refuse the first note, accept the second.
            for reply in [&b"no\n"[..], &b"ok\n"[..]] {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(&stream);
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                let len: usize = header
                    .trim_end()
                    .rsplit(' ')
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                std::io::copy(&mut reader.by_ref().take(len as u64), &mut std::io::sink()).unwrap();
                (&stream).write_all(reply).unwrap();
            }
        });

        let report = deliver_pending(&dir, &endpoint, "testhost");
        server.join().unwrap();
        assert_eq!(report.sent, 1, "the note behind the poison one must ship");
        assert!(report.error.is_some());
        assert!(bad.exists(), "rejected note stays for inspection");
        assert!(!good.exists(), "acked note deleted");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn keeps_wav_when_inbox_unreachable() {
        let dir = std::env::temp_dir().join(format!("voice-deliver-down-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("voice-7.wav");
        std::fs::write(&wav, b"RIFFfake").unwrap();
        // Bind then drop: the port is real but nothing listens.
        let endpoint = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().to_string()
        };
        let report = deliver_pending(&dir, &endpoint, "testhost");
        assert_eq!(report.sent, 0);
        assert!(report.error.is_some());
        assert!(wav.exists(), "unacked note must survive for retry");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn host_tag_clamps_to_header_safe_alphabet() {
        assert_eq!(sanitize_host("ablaised"), "ablaised");
        assert_eq!(sanitize_host("bad host\nname!"), "badhostname");
        assert_eq!(sanitize_host(""), "unknown");
        assert_eq!(sanitize_host(&"x".repeat(99)).len(), 32);
    }

    #[test]
    fn sweep_slot_is_exclusive_until_dropped() {
        let slot = try_begin_sweep().unwrap();
        assert!(try_begin_sweep().is_none());
        drop(slot);
        assert!(try_begin_sweep().is_some());
    }
}
