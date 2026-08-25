//! Minimal strict WAV IO plus the one DSP rendering rate every procedural audio
//! source in the workspace runs at (net's wind/ambience, the d-pad instrument). The
//! fetch pipeline ships every sampled bed as 16-bit mono PCM at [`SAMPLE_RATE`]
//! (ffmpeg at asset-prep time is the one converter), so the reader REFUSES anything
//! else instead of resampling — a runtime resampler would be a second, silent
//! conversion path.

use std::path::Path;

/// The DSP rendering rate. Fixed rather than device-queried: rodio resamples to the
/// device, and every synth's filter/envelope coefficients derive from this number.
pub const SAMPLE_RATE: u32 = 44_100;

/// Read a whole WAV file as f32 samples in −1..1, requiring 16-bit mono PCM at
/// [`SAMPLE_RATE`].
pub fn read_mono_44k(path: &Path) -> Result<Vec<f32>, String> {
    parse_mono_44k(&std::fs::read(path).map_err(|e| e.to_string())?)
}

/// Same strict contract over in-memory bytes — for wavs that arrive on the
/// wire rather than from disk (the rl#378 assistant-reply TTS).
pub fn parse_mono_44k(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut off = 12;
    let mut fmt_seen = false;
    while off + 8 <= b.len() {
        let id: [u8; 4] = b[off..off + 4].try_into().unwrap();
        let sz = u32::from_le_bytes(b[off + 4..off + 8].try_into().unwrap()) as usize;
        let body = b.get(off + 8..off + 8 + sz).ok_or("truncated chunk")?;
        match &id {
            b"fmt " => {
                if sz < 16 {
                    return Err("fmt chunk too short".into());
                }
                let f = |i: usize| u16::from_le_bytes(body[i..i + 2].try_into().unwrap()) as u32;
                let (format, channels, bits) = (f(0), f(2), f(14));
                let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                if (format, channels, rate, bits) != (1, 1, SAMPLE_RATE, 16) {
                    return Err(format!(
                        "need 16-bit mono PCM at {SAMPLE_RATE} Hz, got format {format}, \
                         {channels} ch, {rate} Hz, {bits} bit"
                    ));
                }
                fmt_seen = true;
            }
            b"data" => {
                if !fmt_seen {
                    return Err("data chunk before fmt".into());
                }
                return Ok(body
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect());
            }
            _ => {}
        }
        // Chunks are word-aligned; an odd size carries a pad byte.
        off += 8 + sz + (sz & 1);
    }
    Err("no data chunk".into())
}

/// A decoded bed as f32 samples in −1..1 — [`parse_mono_44k`]'s strict contract as
/// a bevy asset, so sampled audio rides the ONE asset path (rl#411): whatever
/// reader the platform mounts (native fs, HTTP, embedded bytes) serves wavs too.
#[derive(bevy::asset::Asset, bevy::reflect::TypePath)]
pub struct MonoPcm(pub std::sync::Arc<[f32]>);

/// `.wav` [`bevy::asset::AssetLoader`] under the [`parse_mono_44k`] contract — a
/// wav in any other format is a load ERROR (the fetch pipeline ships 16-bit mono
/// PCM at [`SAMPLE_RATE`]; a runtime resampler would be a second conversion path).
#[derive(bevy::reflect::TypePath)]
pub struct WavLoader;

impl bevy::asset::AssetLoader for WavLoader {
    type Asset = MonoPcm;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        _settings: &(),
        _load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<MonoPcm, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        parse_mono_44k(&bytes)
            .map(|pcm| MonoPcm(pcm.into()))
            .map_err(std::io::Error::other)
    }

    fn extensions(&self) -> &[&str] {
        &["wav"]
    }
}

/// RIFF/WAVE writer for tests and evidence generators (net's wind/ambience clips,
/// the rl#359 instrument clips): 16-bit mono PCM at [`SAMPLE_RATE`] — the same
/// (only) format the reader accepts.
pub fn wav_bytes(pcm: &[i16]) -> Vec<u8> {
    let data_len = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend(b"RIFF");
    out.extend((36 + data_len).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes()); // PCM
    out.extend(1u16.to_le_bytes()); // mono
    out.extend(SAMPLE_RATE.to_le_bytes());
    out.extend((SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    out.extend(2u16.to_le_bytes()); // block align
    out.extend(16u16.to_le_bytes()); // bits
    out.extend(b"data");
    out.extend(data_len.to_le_bytes());
    out.extend(pcm.iter().flat_map(|s| s.to_le_bytes()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_writer() {
        let pcm: Vec<i16> = [0i16, 1, -1, i16::MAX, i16::MIN, 12345].into();
        let parsed = parse_mono_44k(&wav_bytes(&pcm)).unwrap();
        assert_eq!(parsed.len(), pcm.len());
        for (f, i) in parsed.iter().zip(&pcm) {
            assert!((f - *i as f32 / 32768.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_wrong_format() {
        let mut stereo = wav_bytes(&[0i16; 8]);
        stereo[22] = 2; // channels field
        assert!(parse_mono_44k(&stereo).unwrap_err().contains("2 ch"));

        let mut wrong_rate = wav_bytes(&[0i16; 8]);
        wrong_rate[24..28].copy_from_slice(&48_000u32.to_le_bytes());
        assert!(
            parse_mono_44k(&wrong_rate)
                .unwrap_err()
                .contains("48000 Hz")
        );

        assert!(parse_mono_44k(b"not audio at all").is_err());
    }
}
