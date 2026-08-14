//! Voice-loop reply polling (rl#378 stage 3): the transport mirror of
//! [`crate::voice_delivery`]. The live session periodically asks the same
//! authenticated endpoint whether the assistant has queued a reply to a
//! playtest voice note; a reply arrives as text plus a bot-side-synthesized
//! read-aloud wav (TTS lives bot-side for the same reason STT does — the game
//! stays thin). Replies the session never polls up wait in the bot-side spool
//! for the next launch, so the queued-for-next-launch fallback needs no code
//! here at all.
//!
//! Wire format, one connection per poll: `voice-poll/1 <host>\n`; the inbox
//! answers `none\n` or `reply <text-len> <wav-len>\n` followed by exactly that
//! many text then wav bytes. The client sends `ack\n` only after it holds the
//! whole payload — the inbox deletes the reply on that ack, so a torn
//! connection re-serves it next poll (a duplicate reply beats a lost one).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// The inbox caps reply text at 2000 chars; 4× headroom keeps the bound about
/// resisting a rogue header, not about tracking the server's limit exactly.
const MAX_TEXT_BYTES: u64 = 8 * 1024;

/// Matches the note-upload cap — read-aloud TTS for a 2000-char reply is far
/// inside it.
const MAX_WAV_BYTES: u64 = 16 * 1024 * 1024;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// One assistant reply: what to put on screen and what to play.
#[derive(Debug, PartialEq, Eq)]
pub struct Reply {
    pub text: String,
    pub wav: Vec<u8>,
}

/// A received reply plus the live connection its ack rides on. The caller acks
/// only AFTER it has taken responsibility for the reply (handed it to the
/// display queue) — acking on receipt would let a teardown between receive and
/// hand-off delete the bundle bot-side with nothing shown (the reviewer-loop's
/// ack-then-drop window). Dropping this without [`Polled::split`]+ack simply
/// leaves the bundle spooled for the next poll.
pub struct Polled {
    reply: Reply,
    stream: TcpStream,
}

/// The deferred ack half of a [`Polled`].
pub struct ReplyAck(TcpStream);

impl Polled {
    pub fn split(self) -> (Reply, ReplyAck) {
        (self.reply, ReplyAck(self.stream))
    }
}

impl ReplyAck {
    /// Tell the inbox the reply is safely handed off; it deletes the bundle.
    pub fn ack(self) {
        // A failed ack write is the same as no ack: re-served next poll.
        let _ = (&self.0).write_all(b"ack\n");
    }
}

/// Ask the inbox for one queued reply. `Ok(None)` is the common quiet case; an
/// `Err` is a transport/protocol failure the caller should treat as "try again
/// next tick", never fatal — the reply (if any) is still spooled bot-side.
pub fn poll_one(endpoint: &str, host: &str) -> Result<Option<Polled>, String> {
    let addr = endpoint
        .to_socket_addrs()
        .map_err(|e| e.to_string())?
        .next()
        .ok_or_else(|| format!("{endpoint} resolves to no address"))?;
    let stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .map_err(|e| e.to_string())?;
    let mut writer = &stream;
    writer
        .write_all(format!("voice-poll/1 {host}\n").as_bytes())
        .map_err(|e| e.to_string())?;

    let mut reader = BufReader::new(&stream);
    let mut status = String::new();
    (&mut reader)
        .take(64)
        .read_line(&mut status)
        .map_err(|e| e.to_string())?;
    let status = status.trim_end();
    if status == "none" {
        return Ok(None);
    }
    let mut parts = status.split(' ');
    if parts.next() != Some("reply") {
        return Err(format!("unexpected poll response: {status:?}"));
    }
    let mut framed_len = |cap: u64, what: &str| -> Result<u64, String> {
        let n: u64 = parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("bad {what} length in {status:?}"))?;
        if n > cap {
            return Err(format!("{what} length {n} exceeds the {cap} cap"));
        }
        Ok(n)
    };
    let text_len = framed_len(MAX_TEXT_BYTES, "text")?;
    let wav_len = framed_len(MAX_WAV_BYTES, "wav")?;

    let mut text = vec![0u8; text_len as usize];
    reader.read_exact(&mut text).map_err(|e| e.to_string())?;
    let mut wav = vec![0u8; wav_len as usize];
    reader.read_exact(&mut wav).map_err(|e| e.to_string())?;
    let text = match String::from_utf8(text) {
        Ok(t) => t,
        Err(_) => {
            // Fully received but unusable: bot-authored garbage, not a torn
            // transport. Ack it away — leaving it would re-serve the same
            // poison bundle every poll and wedge everything behind it.
            let _ = writer.write_all(b"ack\n");
            return Err("reply text is not UTF-8 — acked away".into());
        }
    };
    Ok(Some(Polled {
        reply: Reply { text, wav },
        stream,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Serve one connection with a canned response; return (request line, got ack).
    fn serve_one(
        listener: TcpListener,
        response: &'static [u8],
    ) -> std::thread::JoinHandle<(String, bool)> {
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&stream);
            let mut req = String::new();
            reader.read_line(&mut req).unwrap();
            (&stream).write_all(response).unwrap();
            let mut ack = String::new();
            let got_ack = reader.read_line(&mut ack).is_ok() && ack == "ack\n";
            (req, got_ack)
        })
    }

    fn local(listener: &TcpListener) -> String {
        listener.local_addr().unwrap().to_string()
    }

    #[test]
    fn quiet_poll_returns_none() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let ep = local(&l);
        let server = serve_one(l, b"none\n");
        assert!(poll_one(&ep, "tv").unwrap().is_none());
        let (req, _) = server.join().unwrap();
        assert_eq!(req, "voice-poll/1 tv\n");
    }

    #[test]
    fn reply_is_framed_and_acked_only_on_demand() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let ep = local(&l);
        let server = serve_one(l, b"reply 5 8\nhelloRIFFfake");
        let (reply, ack) = poll_one(&ep, "deck-2").unwrap().unwrap().split();
        assert_eq!(reply.text, "hello");
        assert_eq!(reply.wav, b"RIFFfake");
        ack.ack();
        let (req, acked) = server.join().unwrap();
        assert_eq!(req, "voice-poll/1 deck-2\n");
        assert!(acked, "explicit ack must reach the server");
    }

    #[test]
    fn dropped_polled_reply_never_acks() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let ep = local(&l);
        let server = serve_one(l, b"reply 5 8\nhelloRIFFfake");
        drop(poll_one(&ep, "tv").unwrap().unwrap());
        let (_, acked) = server.join().unwrap();
        assert!(!acked, "an unhanded reply must stay spooled bot-side");
    }

    #[test]
    fn non_utf8_text_is_acked_away_not_wedged() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let ep = local(&l);
        let server = serve_one(l, b"reply 2 4\n\xff\xfeRIFF");
        assert!(poll_one(&ep, "tv").is_err());
        let (_, acked) = server.join().unwrap();
        assert!(
            acked,
            "fully-received poison must be acked so it can't wedge the queue"
        );
    }

    #[test]
    fn short_payload_is_an_error_not_a_hang() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let ep = local(&l);
        // Header promises more than the connection delivers, then EOF.
        let server = std::thread::spawn(move || {
            let (stream, _) = l.accept().unwrap();
            let mut reader = BufReader::new(&stream);
            let mut req = String::new();
            reader.read_line(&mut req).unwrap();
            (&stream).write_all(b"reply 5 100\nhello").unwrap();
        });
        assert!(poll_one(&ep, "tv").is_err());
        server.join().unwrap();
    }

    #[test]
    fn oversize_and_malformed_headers_refused() {
        for bad in [
            &b"reply 999999 4\nxxxx"[..],
            b"reply 4 99999999999\nxxxx",
            b"reply x y\n",
            b"garbage\n",
        ] {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let ep = local(&l);
            let server = std::thread::spawn(move || {
                let (stream, _) = l.accept().unwrap();
                let mut reader = BufReader::new(&stream);
                let mut req = String::new();
                reader.read_line(&mut req).unwrap();
                let _ = (&stream).write_all(bad);
            });
            assert!(poll_one(&ep, "tv").is_err(), "must refuse {bad:?}");
            server.join().unwrap();
        }
    }

    #[test]
    fn unreachable_inbox_is_a_soft_error() {
        let ep = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            local(&l)
        };
        assert!(poll_one(&ep, "tv").is_err());
    }
}
