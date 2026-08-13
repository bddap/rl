//! In-game voice notes (rl#378 stage 1): tap [`Action::VoiceRecord`] to start
//! recording (red ● indicator), tap again to stop into a review modal that plays
//! the take back; MenuConfirm keeps it (wav on disk), MenuBack discards, another
//! VoiceRecord tap replays. The wav directory is the seam to stage 2: confirm
//! (and each round start, for retries) fires a background
//! [`crate::voice_delivery`] sweep that ships pending takes to the assistant's
//! inbox and deletes them on ack.
//!
//! Capture must never block play, and cpal's `Stream` is `!Send`, so the whole
//! device dance (open, run, close) lives on one dedicated thread per take: the
//! game thread only flips this module's state machine and polls a channel. Mic
//! choice is `$RL_MIC` (case-insensitive substring of the cpal device name — the
//! per-platform knob, same env-var style as every other host knob) falling back to
//! the host default input. The mic's native rate/channels are downmixed and
//! linearly resampled to the workspace-wide 44.1 kHz mono of [`crab_world::wav`],
//! keeping that crate's strict single-format contract intact.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use bevy::audio::{AddAudioSource, Decodable, PlaybackSettings};
use bevy::prelude::*;

use super::app::AppPhase;
use crate::controls::{self, Action, GcrControls};
use crate::voice_delivery;
use crab_world::wav;

/// Memory/attention bound for one take; the thread stops itself at the cap even if
/// the stop tap never comes (e.g. the player forgot the recorder was hot).
const MAX_TAKE: Duration = Duration::from_secs(120);

/// A take shorter than this is a misfire (double-tap), not a message.
const MIN_TAKE_SAMPLES: usize = (wav::SAMPLE_RATE / 4) as usize;

/// How long transient status lines (saved / discarded / error) stay up.
const FLASH_SECS: f32 = 3.0;

pub(super) fn install(app: &mut App) {
    app.add_audio_source::<VoiceClip>();
    app.init_resource::<VoiceUx>();
    app.add_systems(Startup, spawn_voice_hud);
    app.add_systems(
        Update,
        // After gather_input: on the confirm/discard frame the gatherer must still
        // see the modal live so the same South/East press can't also jump/slide.
        (drive_voice, update_voice_hud)
            .chain()
            .after(super::input::gather_input)
            .run_if(in_state(AppPhase::Playing)),
    );
    app.add_systems(OnEnter(AppPhase::Playing), kick_pending_delivery);
    app.add_systems(OnExit(AppPhase::Playing), cancel_voice);
}

/// One finished take, already 44.1 kHz mono i16.
struct Take {
    samples: Vec<i16>,
}

enum Phase {
    Idle,
    /// The capture thread is live. Dropping `stop` tells it to wrap up; `done`
    /// yields the take (or the device error) once the thread finishes — also
    /// unprompted, when the [`MAX_TAKE`] cap trips. The Mutex is only for Sync
    /// (`Receiver` isn't; a `Resource` must be) — one system ever touches it.
    Recording {
        stop: Option<mpsc::Sender<()>>,
        done: std::sync::Mutex<mpsc::Receiver<Result<Take, String>>>,
        elapsed: f32,
    },
    Review {
        samples: Arc<Vec<i16>>,
    },
}

#[derive(Resource)]
pub(super) struct VoiceUx {
    phase: Phase,
    /// Transient status line + time left to show it.
    flash: Option<(String, f32)>,
    /// A running stage-2 delivery sweep's report channel; polled each frame so
    /// the sent/queued outcome can flash without ever blocking play.
    delivery: Option<std::sync::Mutex<mpsc::Receiver<voice_delivery::Report>>>,
    /// The modal just closed but a confirm/discard input is still physically held:
    /// jump/slide/brake are level-triggered `pressed()` reads, so without this the
    /// A that saved the note would hop the character on the very next frame.
    linger: bool,
}

impl Default for VoiceUx {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            flash: None,
            delivery: None,
            linger: false,
        }
    }
}

impl VoiceUx {
    /// While the review modal is up — or its closing press is still held — the
    /// confirm/discard inputs belong to the modal: the input gatherer consults this
    /// to keep them out of jump/slide/brake.
    pub(super) fn gameplay_gated(&self) -> bool {
        matches!(self.phase, Phase::Review { .. }) || self.linger
    }
}

/// A finished take as a bevy audio asset for review playback. Raw samples through
/// our own decoder — bevy's default feature set has no wav decoder, and the
/// procedural sources (wind/ambience/instrument) already established this path.
#[derive(Asset, TypePath)]
struct VoiceClip(Arc<Vec<i16>>);

impl Decodable for VoiceClip {
    type DecoderItem = i16;
    type Decoder = VoiceStream;
    fn decoder(&self) -> VoiceStream {
        VoiceStream {
            samples: self.0.clone(),
            pos: 0,
        }
    }
}

struct VoiceStream {
    samples: Arc<Vec<i16>>,
    pos: usize,
}

impl Iterator for VoiceStream {
    type Item = i16;
    fn next(&mut self) -> Option<i16> {
        let s = self.samples.get(self.pos).copied();
        self.pos += 1;
        s
    }
}

impl bevy::audio::Source for VoiceStream {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        1
    }
    fn sample_rate(&self) -> u32 {
        wav::SAMPLE_RATE
    }
    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            self.samples.len() as f64 / wav::SAMPLE_RATE as f64,
        ))
    }
}

/// Marks the review-playback entity so replay/teardown can find it.
#[derive(Component)]
struct VoicePlayback;

#[derive(Component)]
struct VoiceHud;

fn spawn_voice_hud(mut commands: Commands) {
    // Full-width transparent strip so the text centers itself; the strip ignores
    // clicks/layout by being absolute.
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: Val::Px(14.0),
            left: Val::Px(0.0),
            width: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            ..default()
        })
        .with_children(|p| {
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.35, 0.3)),
                Visibility::Hidden,
                VoiceHud,
            ));
        });
}

/// Short text name for an action's first bound gamepad button + keyboard key, for
/// the modal's prompt line. The button identity comes from the binding table; the
/// label text is local shorthand covering exactly the buttons the modal uses ("?"
/// would flag a rebind that outgrows it — extend the match then).
fn prompt_chip(action: Action) -> String {
    let pad = controls::gamepad_buttons_for(action)
        .next()
        .map(|b| match b {
            GamepadButton::South => "A",
            GamepadButton::East => "B",
            GamepadButton::West => "X",
            GamepadButton::North => "Y",
            _ => "?",
        });
    let key = controls::key_code_for(action).map(|k| match k {
        KeyCode::Enter => "Enter",
        KeyCode::Escape => "Esc",
        KeyCode::KeyV => "V",
        _ => "?",
    });
    match (pad, key) {
        (Some(p), Some(k)) => format!("{p}/{k}"),
        (Some(p), None) => p.into(),
        (None, Some(k)) => k.into(),
        (None, None) => "?".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_voice(
    mut voice: ResMut<VoiceUx>,
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    time: Res<Time>,
    mut commands: Commands,
    mut clips: ResMut<Assets<VoiceClip>>,
    playback: Query<Entity, With<VoicePlayback>>,
) {
    use crab_world::controls::{just_pressed, pressed};
    poll_delivery(&mut voice);
    let record = just_pressed::<GcrControls>(Action::VoiceRecord, &keys, &gamepads);
    let confirm = just_pressed::<GcrControls>(Action::MenuConfirm, &keys, &gamepads);
    let discard = just_pressed::<GcrControls>(Action::MenuBack, &keys, &gamepads);

    if voice.linger
        && !pressed::<GcrControls>(Action::MenuConfirm, &keys, &gamepads)
        && !pressed::<GcrControls>(Action::MenuBack, &keys, &gamepads)
    {
        // Cleared here (after gather_input) — the gate stays up through the release
        // frame, which only costs one extra suppressed frame. Action-level, so a
        // Space held since before a pad-closed modal keeps the gate up until full
        // release (known, accepted: it self-corrects on release + re-press, and
        // per-input tracking wouldn't pay for the frame it saves).
        voice.linger = false;
    }

    if let Some((_, left)) = &mut voice.flash {
        *left -= time.delta_secs();
        if *left <= 0.0 {
            voice.flash = None;
        }
    }

    match &mut voice.phase {
        Phase::Idle => {
            if record {
                let (stop_tx, stop_rx) = mpsc::channel();
                let (done_tx, done_rx) = mpsc::channel();
                std::thread::Builder::new()
                    .name("voice-capture".into())
                    .spawn(move || {
                        let _ = done_tx.send(capture_take(stop_rx));
                    })
                    .expect("spawn voice-capture thread");
                voice.phase = Phase::Recording {
                    stop: Some(stop_tx),
                    done: std::sync::Mutex::new(done_rx),
                    elapsed: 0.0,
                };
            }
        }
        Phase::Recording {
            stop,
            done,
            elapsed,
        } => {
            *elapsed += time.delta_secs();
            if record {
                // Dropping the sender is the stop signal; the thread then finishes
                // resampling and reports on `done` within a frame or two.
                *stop = None;
            }
            let polled = done.get_mut().unwrap().try_recv();
            match polled {
                Ok(Ok(take)) if take.samples.len() >= MIN_TAKE_SAMPLES => {
                    let samples = Arc::new(take.samples);
                    commands.spawn((
                        VoicePlayback,
                        AudioPlayer(clips.add(VoiceClip(samples.clone()))),
                        PlaybackSettings::DESPAWN,
                    ));
                    voice.phase = Phase::Review { samples };
                }
                Ok(Ok(_)) => {
                    voice.flash = Some(("voice note too short — discarded".into(), FLASH_SECS));
                    voice.phase = Phase::Idle;
                }
                Ok(Err(e)) => {
                    warn!("voice capture failed: {e}");
                    voice.flash = Some((format!("mic unavailable: {e}"), FLASH_SECS));
                    voice.phase = Phase::Idle;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    warn!("voice capture thread died");
                    voice.flash = Some(("mic capture failed".into(), FLASH_SECS));
                    voice.phase = Phase::Idle;
                }
            }
        }
        Phase::Review { samples } => {
            if confirm {
                let (msg, delivery) = match save_take(samples) {
                    Ok(path) => (
                        format!("voice note saved · {}", path.display()),
                        spawn_delivery_sweep(),
                    ),
                    Err(e) => {
                        warn!("voice note save failed: {e}");
                        (format!("save failed: {e}"), None)
                    }
                };
                stop_playback(&mut commands, &playback);
                voice.flash = Some((msg, FLASH_SECS));
                if delivery.is_some() {
                    voice.delivery = delivery;
                }
                voice.phase = Phase::Idle;
                voice.linger = true;
            } else if discard {
                stop_playback(&mut commands, &playback);
                voice.flash = Some(("voice note discarded".into(), FLASH_SECS));
                voice.phase = Phase::Idle;
                voice.linger = true;
            } else if record {
                // Replay the take from the top.
                stop_playback(&mut commands, &playback);
                commands.spawn((
                    VoicePlayback,
                    AudioPlayer(clips.add(VoiceClip(samples.clone()))),
                    PlaybackSettings::DESPAWN,
                ));
            }
        }
    }
}

/// Surface a finished delivery sweep's outcome. Empty sweeps (the round-start
/// kick found nothing pending) stay silent — no news is the common case.
fn poll_delivery(voice: &mut VoiceUx) {
    let Some(rx) = voice.delivery.take() else {
        return;
    };
    let polled = rx.lock().unwrap().try_recv();
    match polled {
        Ok(report) => {
            let flash = match (&report.error, report.sent) {
                (Some(_), _) => Some("voice note queued — inbox unreachable".into()),
                (None, 0) => None,
                (None, 1) => Some("voice note sent".into()),
                (None, n) => Some(format!("{n} voice notes sent")),
            };
            if let Some(f) = flash {
                voice.flash = Some((f, FLASH_SECS));
            }
        }
        Err(mpsc::TryRecvError::Empty) => voice.delivery = Some(rx),
        Err(mpsc::TryRecvError::Disconnected) => warn!("voice delivery thread died"),
    }
}

/// Fire one background delivery sweep over the voice dir; `None` when a sweep
/// is already running (its pass covers any file this one would have sent).
fn spawn_delivery_sweep() -> Option<std::sync::Mutex<mpsc::Receiver<voice_delivery::Report>>> {
    let slot = voice_delivery::try_begin_sweep()?;
    let dir = voice_dir()?;
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("voice-deliver".into())
        .spawn(move || {
            let _slot = slot;
            let report = voice_delivery::deliver_pending(
                &dir,
                &voice_delivery::endpoint(),
                &voice_delivery::host_tag(),
            );
            let _ = tx.send(report);
        })
        .ok()?;
    Some(std::sync::Mutex::new(rx))
}

/// Round start: retry anything a dead inbox left behind — the sender deletes
/// only on ack, so this is the at-least-once loop's other half.
fn kick_pending_delivery(mut voice: ResMut<VoiceUx>) {
    if voice.delivery.is_none() {
        voice.delivery = spawn_delivery_sweep();
    }
}

fn stop_playback(commands: &mut Commands, playback: &Query<Entity, With<VoicePlayback>>) {
    for e in playback.iter() {
        commands.entity(e).despawn();
    }
}

/// Round over mid-take: drop the capture thread's stop handle (it wraps up and its
/// result goes nowhere) and clear the modal — a hot mic must not outlive the round.
fn cancel_voice(
    mut voice: ResMut<VoiceUx>,
    mut commands: Commands,
    playback: Query<Entity, With<VoicePlayback>>,
    mut hud: Query<&mut Visibility, With<VoiceHud>>,
) {
    stop_playback(&mut commands, &playback);
    voice.phase = Phase::Idle;
    voice.flash = None;
    voice.linger = false;
    for mut vis in &mut hud {
        *vis = Visibility::Hidden;
    }
}

fn update_voice_hud(
    voice: Res<VoiceUx>,
    mut hud: Query<(&mut Text, &mut TextColor, &mut Visibility), With<VoiceHud>>,
) {
    let (line, color) = match &voice.phase {
        Phase::Recording { elapsed, .. } => (
            Some(format!(
                "● REC {}:{:02}  (tap {} to stop)",
                *elapsed as u32 / 60,
                *elapsed as u32 % 60,
                prompt_chip(Action::VoiceRecord),
            )),
            Color::srgb(1.0, 0.35, 0.3),
        ),
        Phase::Review { samples } => (
            Some(format!(
                "voice note {:.1}s — keep {} · discard {} · replay {}",
                samples.len() as f32 / wav::SAMPLE_RATE as f32,
                prompt_chip(Action::MenuConfirm),
                prompt_chip(Action::MenuBack),
                prompt_chip(Action::VoiceRecord),
            )),
            Color::srgb(1.0, 0.9, 0.4),
        ),
        Phase::Idle => (
            voice.flash.as_ref().map(|(m, _)| m.clone()),
            Color::srgb(1.0, 0.9, 0.4),
        ),
    };
    if let Ok((mut text, mut tc, mut vis)) = hud.single_mut() {
        let want = if line.is_some() {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
        if let Some(l) = line {
            if **text != l {
                **text = l;
            }
            if tc.0 != color {
                tc.0 = color;
            }
        }
    }
}

/// Where confirmed takes land: `$RL_VOICE_DIR`, else [`super::data_dir`] +
/// `voice`. `None` (no HOME at all) fails the save, not the boot.
fn voice_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("RL_VOICE_DIR") {
        return Some(PathBuf::from(d));
    }
    Some(super::data_dir()?.join("voice"))
}

fn save_take(samples: &[i16]) -> Result<PathBuf, String> {
    let dir = voice_dir().ok_or("no $HOME to save under")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    // Millis, not secs: rename() clobbers an existing name, and two quick takes can
    // land in the same second.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let path = dir.join(format!("voice-{stamp}.wav"));
    // Temp + rename so stage 2's pickup can never read a half-written wav.
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, wav::wav_bytes(samples)).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// The capture thread.
// ---------------------------------------------------------------------------

/// Everything cpal, one thread per take: open the mic, run until the stop handle
/// drops (or [`MAX_TAKE`]), return the take at 44.1 kHz mono.
fn capture_take(stop: mpsc::Receiver<()>) -> Result<Take, String> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = match std::env::var("RL_MIC") {
        Ok(want) => {
            let want_lc = want.to_lowercase();
            host.input_devices()
                .map_err(|e| e.to_string())?
                .find(|d| d.name().is_ok_and(|n| n.to_lowercase().contains(&want_lc)))
                .ok_or(format!("no input device matching RL_MIC={want}"))?
        }
        Err(_) => host
            .default_input_device()
            .ok_or("no default input device")?,
    };
    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let src_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    // The callback downmixes to mono f32 and appends; capacity for the whole cap
    // (~23 MB at 48 kHz) so a long take never reallocates inside the real-time
    // audio callback.
    let cap_samples = (src_rate as u64 * MAX_TAKE.as_secs()) as usize;
    let buf = Arc::new(std::sync::Mutex::new(Vec::<f32>::with_capacity(
        cap_samples,
    )));
    let err: Arc<std::sync::Mutex<Option<String>>> = Arc::default();

    macro_rules! stream_as {
        ($t:ty) => {{
            let buf = buf.clone();
            let err2 = err.clone();
            device
                .build_input_stream(
                    &config.clone().into(),
                    move |data: &[$t], _: &cpal::InputCallbackInfo| {
                        let mut b = buf.lock().unwrap();
                        if b.len() >= cap_samples {
                            return;
                        }
                        for frame in data.chunks(channels) {
                            let s: f32 = frame
                                .iter()
                                .map(|&v| cpal::Sample::to_sample::<f32>(v))
                                .sum::<f32>()
                                / channels as f32;
                            b.push(s);
                        }
                    },
                    move |e| {
                        *err2.lock().unwrap() = Some(e.to_string());
                    },
                    None,
                )
                .map_err(|e| e.to_string())?
        }};
    }
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => stream_as!(f32),
        cpal::SampleFormat::I16 => stream_as!(i16),
        cpal::SampleFormat::U16 => stream_as!(u16),
        other => return Err(format!("unsupported mic sample format {other:?}")),
    };
    stream.play().map_err(|e| e.to_string())?;

    // Ok(()) never arrives (nothing sends); Disconnected IS the stop tap.
    let _ = stop.recv_timeout(MAX_TAKE);
    drop(stream);

    if let Some(e) = err.lock().unwrap().take() {
        return Err(e);
    }
    let raw = std::mem::take(&mut *buf.lock().unwrap());
    Ok(Take {
        samples: resample_to_i16(&raw, src_rate),
    })
}

/// Linear resample from the mic's native rate to [`wav::SAMPLE_RATE`], f32 → i16.
/// Linear is plenty for speech review/transcription; anything fancier would be a
/// second DSP path to maintain.
fn resample_to_i16(src: &[f32], src_rate: u32) -> Vec<i16> {
    let to_i16 = |s: f32| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
    if src_rate == wav::SAMPLE_RATE {
        return src.iter().map(|&s| to_i16(s)).collect();
    }
    let n_out = (src.len() as u64 * wav::SAMPLE_RATE as u64 / src_rate as u64) as usize;
    let step = src_rate as f64 / wav::SAMPLE_RATE as f64;
    (0..n_out)
        .map(|i| {
            let pos = i as f64 * step;
            let j = pos as usize;
            let frac = (pos - j as f64) as f32;
            let a = src[j.min(src.len() - 1)];
            let b = src[(j + 1).min(src.len() - 1)];
            to_i16(a + (b - a) * frac)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 440 Hz tone through the 48 k → 44.1 k path keeps its pitch: zero crossings
    /// per second stay ~880 and the round-tripped wav parses under the strict reader.
    #[test]
    fn resample_preserves_pitch_and_wav_round_trips() {
        let src_rate = 48_000u32;
        let secs = 2.0f64;
        let src: Vec<f32> = (0..(src_rate as f64 * secs) as usize)
            .map(|i| (i as f64 / src_rate as f64 * 440.0 * std::f64::consts::TAU).sin() as f32)
            .collect();
        let out = resample_to_i16(&src, src_rate);
        let expect = (wav::SAMPLE_RATE as f64 * secs) as usize;
        assert!(
            (out.len() as i64 - expect as i64).abs() <= 2,
            "{}",
            out.len()
        );
        let crossings = out.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count() as f64;
        let hz = crossings / 2.0 / secs;
        assert!((hz - 440.0).abs() < 2.0, "pitch drifted to {hz} Hz");

        let dir = std::env::temp_dir().join(format!("voice-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        std::fs::write(&path, wav::wav_bytes(&out)).unwrap();
        let back = wav::read_mono_44k(&path).unwrap();
        assert_eq!(back.len(), out.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The identity path (already 44.1 k) is sample-exact apart from i16 rounding.
    #[test]
    fn resample_identity_is_lossless() {
        let src = [0.0f32, 0.5, -0.5, 1.0, -1.0];
        let out = resample_to_i16(&src, wav::SAMPLE_RATE);
        assert_eq!(out.len(), src.len());
        assert_eq!(out[3], i16::MAX);
    }

    /// Saved takes land where stage 2 will look, atomically named `voice-*.wav`.
    #[test]
    fn save_take_writes_strict_wav_under_rl_voice_dir() {
        let dir = std::env::temp_dir().join(format!("voice-save-{}", std::process::id()));
        // SAFETY: test-local env var; suites run with --test-threads=2 but no other
        // test reads RL_VOICE_DIR.
        unsafe { std::env::set_var("RL_VOICE_DIR", &dir) };
        let samples: Vec<i16> = (0..wav::SAMPLE_RATE).map(|i| (i % 100) as i16).collect();
        let path = save_take(&samples).unwrap();
        assert!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("voice-")
        );
        let back = wav::read_mono_44k(&path).unwrap();
        assert_eq!(back.len(), samples.len());
        unsafe { std::env::remove_var("RL_VOICE_DIR") };
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
