//! `rl-update-ui` — the windowed transfer-info UI for the Steam Deck "Update" shortcut.
//!
//! The "Update" shortcut pulls the latest rl build from the bothouse release store via
//! `rl-update` (deck-control). That puller is headless and prints a single result line;
//! in Game Mode there is no terminal to see it, and the owner wanted a real window:
//! live transfer progress, then hold until dismissed by an on-screen tap OR a physical
//! controller/keyboard button.
//!
//! This binary IS that window. It spawns `rl-update --json` — which emits one JSON
//! progress event per line (see deck-control rl-update's `Event`) — renders the stream
//! as a Bevy UI (overall bar, current file, bytes/MB), and on completion shows the
//! result and waits for ANY dismissal (gamepad button / key / mouse click / touch)
//! before closing with the puller's exit code. `rl-update` stays the SINGLE source of
//! the transfer (one implementation); this only renders what it reports.
//!
//! `--screenshot PATH` renders one representative frame (mid-transfer, or the done
//! screen with `--screenshot-done`) and exits — how the daemon surfaces the UI to the
//! owner. It renders through the SAME windowed path the deck uses (the proven UI render
//! route), so it needs a display: on bothouse, Xvfb + the lavapipe software-Vulkan ICD.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy::window::{MonitorSelection, WindowMode};
use clap::Parser;

// --- palette -----------------------------------------------------------------------
// Inline `Color::srgb` calls (it isn't const), wrapped in tiny fns so the few accents
// have one definition each.
fn accent() -> Color {
    Color::srgb(0.30, 0.69, 0.96)
}
fn dim() -> Color {
    Color::srgb(0.55, 0.57, 0.62)
}
fn ok_green() -> Color {
    Color::srgb(0.36, 0.83, 0.45)
}
fn err_red() -> Color {
    Color::srgb(0.94, 0.40, 0.38)
}

#[derive(Parser, Debug, Clone)]
#[command(about = "Windowed transfer-info UI for the rl Update shortcut (drives rl-update --json)")]
struct Args {
    /// The bothouse `rl-release-server`'s endpoint id — passed straight through to
    /// `rl-update`. (Mirrors rl-update's `--server`.)
    #[arg(long, env = "RL_RELEASE_SERVER")]
    server: String,
    /// The local install dir — passed through to `rl-update`.
    #[arg(long, env = "RL_INSTALL_DIR")]
    install_dir: PathBuf,
    /// The deck pull key — passed through to `rl-update`.
    #[arg(long, env = "RL_UPDATE_KEY")]
    key_file: PathBuf,
    /// Path to the `rl-update` binary to drive. Defaults to `./rl-update` (the Update
    /// shortcut `cd`s into the install dir, where the deck-owned copy sits beside it).
    #[arg(long, default_value = "./rl-update")]
    rl_update: PathBuf,
    /// Run in a window instead of borderless fullscreen (the Deck Game-Mode default).
    #[arg(long)]
    windowed: bool,
    /// Render one representative mid-transfer frame to PATH headless and exit — no
    /// window, software Vulkan. Used by the daemon to surface the UI to the owner.
    #[arg(long)]
    screenshot: Option<PathBuf>,
    /// Settle frames before the screenshot capture (GPU pipeline warm-up). Only used
    /// with `--screenshot`.
    #[arg(long, default_value_t = 120)]
    screenshot_settle: u32,
    /// With `--screenshot`, render the completed/dismiss screen instead of the default
    /// mid-transfer frame (so the daemon can surface both states from one binary).
    #[arg(long)]
    screenshot_done: bool,
}

impl Args {
    fn puller(&self) -> PullerArgs {
        PullerArgs {
            rl_update: self.rl_update.clone(),
            server: self.server.clone(),
            install_dir: self.install_dir.clone(),
            key_file: self.key_file.clone(),
        }
    }
}

/// The args `rl-update` needs, separated from the UI-only flags.
struct PullerArgs {
    rl_update: PathBuf,
    server: String,
    install_dir: PathBuf,
    key_file: PathBuf,
}

// --- shared transfer state ---------------------------------------------------------

/// Where the transfer is. `Connecting` until the first `file_start`, `Transferring`
/// during the pull, `Done` once the puller reports its terminal result. Dismissal is
/// only honored in `Done` (see [`handle_dismiss`]).
#[derive(Default, PartialEq, Clone, Copy)]
enum Phase {
    #[default]
    Connecting,
    Transferring,
    Done,
}

/// The terminal result the done-screen shows.
struct DoneInfo {
    ok: bool,
    message: String,
}

/// The transfer state the reader thread writes and the Bevy systems read each frame.
/// Behind one `Mutex`: a single coarse lock is plenty (events arrive at most a few
/// hundred times over a transfer; the UI reads it 60×/s).
#[derive(Default)]
struct State {
    phase: Phase,
    version: Option<String>,
    total: u64,
    done: u64,
    cur_idx: usize,
    cur_of: usize,
    cur_path: String,
    result: Option<DoneInfo>,
    /// `rl-update`'s exit code once it exits — propagated as this process's code so
    /// Steam/tooling sees pass/fail (the same contract the old shell wrapper had).
    child_code: Option<i32>,
}

#[derive(Resource, Clone)]
struct Ui(Arc<Mutex<State>>);

/// Lock the shared state, recovering from poisoning. A `State` left by a panicking
/// holder is still perfectly readable (it's plain data), so a poison must not cascade
/// into aborting the Bevy main thread or the reader thread — that would turn a
/// recoverable hiccup into a hung, non-dismissable window.
fn lock(m: &Mutex<State>) -> std::sync::MutexGuard<'_, State> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// --- puller subprocess + JSON event reader -----------------------------------------

/// Spawn `rl-update --json` and a thread that parses its event stream into `state`.
/// A spawn failure (binary missing) is itself a terminal failure shown in the window.
fn spawn_puller(args: &PullerArgs, state: Arc<Mutex<State>>) {
    let child = Command::new(&args.rl_update)
        .arg("--json")
        .arg("--server")
        .arg(&args.server)
        .arg("--install-dir")
        .arg(&args.install_dir)
        .arg("--key-file")
        .arg(&args.key_file)
        .stdout(Stdio::piped())
        // stderr is rl-update's tracing — let it flow to the shortcut's launch.log.
        .stderr(Stdio::inherit())
        .spawn();
    match child {
        Ok(child) => {
            thread::spawn(move || read_events(child, state));
        }
        Err(e) => {
            let mut s = lock(&state);
            s.phase = Phase::Done;
            s.result = Some(DoneInfo {
                ok: false,
                message: format!("could not launch {}: {e}", args.rl_update.display()),
            });
            // Exit 1 (a normal "failed" outcome the UI just displayed), NOT a crash code:
            // the window rendered the error and held for dismissal, so the shell wrapper
            // should trust it, not fall through to a headless retry that would re-run the
            // same unlaunchable `rl-update` and surface the identical error twice.
            s.child_code = Some(1);
        }
    }
}

/// Read the puller's JSON-lines stdout to EOF, applying each event, then reap the child
/// for its exit code. If the child dies WITHOUT a `done` event (a crash), synthesize a
/// failure so the window never hangs on a stale "transferring".
fn read_events(mut child: Child, state: Arc<Mutex<State>>) {
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // A non-JSON line (shouldn't happen in --json mode) is skipped, not fatal.
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                apply_event(&state, &v);
            }
        }
    }
    let code = child.wait().ok().and_then(|s| s.code());
    let mut s = lock(&state);
    s.child_code = code;
    if s.result.is_none() {
        s.phase = Phase::Done;
        let code_str = code
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".into());
        s.result = Some(DoneInfo {
            ok: false,
            message: format!("rl-update exited ({code_str}) with no result"),
        });
    }
}

/// Fold one parsed event into `state`. Field names match deck-control rl-update's
/// `Event` serde shape (pinned by its `event_json_wire_shape` test).
fn apply_event(state: &Arc<Mutex<State>>, v: &serde_json::Value) {
    let event = v.get("event").and_then(|e| e.as_str()).unwrap_or("");
    let u64f = |k: &str| v.get(k).and_then(|x| x.as_u64());
    let mut s = lock(state);
    if let Some(ver) = v.get("version").and_then(|x| x.as_str()) {
        s.version = Some(ver.to_string());
    }
    match event {
        "manifest" => {}
        "plan" => {
            if let Some(b) = u64f("bytes") {
                s.total = b;
            }
        }
        "file_start" => {
            s.phase = Phase::Transferring;
            s.cur_idx = u64f("idx").unwrap_or(0) as usize;
            s.cur_of = u64f("of").unwrap_or(0) as usize;
            s.cur_path = v
                .get("path")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
        }
        "progress" => {
            if let Some(d) = u64f("done") {
                s.done = d;
            }
            if let Some(t) = u64f("total")
                && t > 0
            {
                s.total = t;
            }
        }
        "done" => {
            let result = v.get("result").and_then(|x| x.as_str()).unwrap_or("failed");
            let message = v
                .get("message")
                .and_then(|x| x.as_str())
                .unwrap_or("(no message)")
                .to_string();
            s.phase = Phase::Done;
            let ok = result != "failed";
            // On a successful transfer fill the bar exactly (the last progress tick may
            // sit a hair short of total); leave it where it is for already/failed.
            if ok && result == "updated" && s.total > 0 {
                s.done = s.total;
            }
            s.result = Some(DoneInfo { ok, message });
        }
        _ => {}
    }
}

// --- UI ----------------------------------------------------------------------------

/// One text node's role, so a single query can update them all (distinct `With<>`
/// queries over `&mut Text` would alias; a label discriminant avoids that).
#[derive(Component, Clone, Copy)]
enum Slot {
    Status,
    Version,
    File,
    Bytes,
    Hint,
}

/// The progress-bar fill node (its width is set to the percent each frame).
#[derive(Component)]
struct BarFill;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000.0
}

/// Map the few non-ASCII glyphs in rl-update's result line to ASCII the bundled default
/// font actually has (it renders unknown glyphs as tofu boxes). Color already carries
/// pass/fail, so the leading status emoji is dropped rather than transliterated.
fn ascii_clean(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '✅' | '✓' | '✗' => ' ',
            '—' | '–' => '-',
            '·' => '-',
            '…' => '.', // a lone '.' is fine; the source rarely uses '…'
            other => other,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

fn setup_windowed(mut commands: Commands) {
    commands.spawn(Camera2d);
    build_ui(&mut commands);
}

/// Build the UI tree. With a single camera present, bevy_ui composites onto it as the
/// implicit default target (the same plain setup the demo's build-info overlay uses) —
/// no explicit camera tag needed.
fn build_ui(commands: &mut Commands) {
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                row_gap: Val::Px(16.0),
                padding: UiRect::all(Val::Px(40.0)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.06, 0.07, 0.09)),
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("rl Update"),
                TextFont {
                    font_size: 46.0,
                    ..default()
                },
                TextColor(Color::srgb(0.95, 0.95, 0.97)),
            ));
            p.spawn((
                Text::new("Checking for updates…"),
                TextFont {
                    font_size: 24.0,
                    ..default()
                },
                TextColor(accent()),
                Slot::Status,
            ));
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(dim()),
                Slot::Version,
            ));
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgb(0.80, 0.80, 0.86)),
                Slot::File,
            ));
            // Progress-bar track with an inner fill node.
            p.spawn((
                Node {
                    width: Val::Px(660.0),
                    height: Val::Px(28.0),
                    padding: UiRect::all(Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.15, 0.16, 0.20)),
            ))
            .with_children(|track| {
                track.spawn((
                    Node {
                        width: Val::Percent(0.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    BackgroundColor(accent()),
                    BarFill,
                ));
            });
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: 19.0,
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.85, 0.90)),
                Slot::Bytes,
            ));
            p.spawn((
                Text::new(""),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(dim()),
                Slot::Hint,
            ));
        });
}

/// Redraw the UI from the shared state each frame: the bar width, the status line +
/// color, the version/file/bytes labels, and the dismissal hint once done.
fn update_ui(
    ui: Res<Ui>,
    mut texts: Query<(&mut Text, &mut TextColor, &Slot)>,
    mut fill: Query<&mut Node, With<BarFill>>,
) {
    let s = lock(&ui.0);
    let pct = if s.total > 0 {
        (s.done as f64 / s.total as f64 * 100.0).clamp(0.0, 100.0)
    } else if s.phase == Phase::Done {
        100.0
    } else {
        0.0
    };
    if let Ok(mut node) = fill.single_mut() {
        node.width = Val::Percent(pct as f32);
    }
    for (mut text, mut color, slot) in &mut texts {
        match slot {
            Slot::Status => {
                let (msg, c) = match s.phase {
                    Phase::Connecting => ("Checking for updates...".to_string(), accent()),
                    Phase::Transferring => ("Transferring...".to_string(), accent()),
                    Phase::Done => match &s.result {
                        Some(d) if d.ok => (ascii_clean(&d.message), ok_green()),
                        Some(d) => (ascii_clean(&d.message), err_red()),
                        None => ("Done".to_string(), ok_green()),
                    },
                };
                text.0 = msg;
                color.0 = c;
            }
            Slot::Version => {
                text.0 = s
                    .version
                    .as_ref()
                    .map(|v| format!("version {v}"))
                    .unwrap_or_default();
            }
            Slot::File => {
                text.0 = if s.phase == Phase::Transferring && s.cur_of > 0 {
                    format!("file {}/{}   {}", s.cur_idx, s.cur_of, s.cur_path)
                } else {
                    String::new()
                };
            }
            Slot::Bytes => {
                text.0 = if s.total > 0 {
                    format!("{:.1} / {:.1} MB  ({pct:.0}%)", mb(s.done), mb(s.total))
                } else {
                    String::new()
                };
            }
            Slot::Hint => {
                text.0 = if s.phase == Phase::Done {
                    "Press A / any key / tap to close".to_string()
                } else {
                    String::new()
                };
            }
        }
    }
}

/// Close the window on ANY input — but ONLY once the transfer is done, so a stray
/// press mid-pull can't dismiss the result screen. Honors all the ways a Deck user
/// might press: a controller button, a keyboard key, a mouse click, or a touch/tap.
fn handle_dismiss(
    ui: Res<Ui>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    touches: Res<Touches>,
    gamepads: Query<&Gamepad>,
    mut exit: MessageWriter<AppExit>,
) {
    if lock(&ui.0).phase != Phase::Done {
        return;
    }
    let pressed = keys.get_just_pressed().next().is_some()
        || mouse.get_just_pressed().next().is_some()
        || touches.iter_just_pressed().next().is_some()
        || gamepads
            .iter()
            .any(|gp| gp.get_just_pressed().next().is_some());
    if pressed {
        exit.write(AppExit::Success);
    }
}

// --- entry points ------------------------------------------------------------------

fn main() {
    let args = Args::parse();
    let state = Arc::new(Mutex::new(State::default()));

    // `--screenshot` injects a representative snapshot and captures one frame (it must
    // run under a display — e.g. Xvfb + lavapipe on bothouse — since it renders through
    // the same windowed path the deck uses, the proven UI render route).
    if args.screenshot.is_some() {
        *lock(&state) = synthetic_state(args.screenshot_done);
    }

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "rl Update".into(),
            // Screenshot mode wants a fixed-size window to capture; the live shortcut
            // wants borderless fullscreen on the Deck (unless --windowed for dev).
            mode: if args.windowed || args.screenshot.is_some() {
                WindowMode::Windowed
            } else {
                WindowMode::BorderlessFullscreen(MonitorSelection::Primary)
            },
            ..default()
        }),
        ..default()
    }));
    app.insert_resource(Ui(state.clone()));
    app.add_systems(Startup, setup_windowed);
    if let Some(path) = args.screenshot.clone() {
        app.insert_resource(ScreenshotPath(path));
        app.insert_resource(ShotSettle(args.screenshot_settle));
        app.init_resource::<ShotFrames>();
        app.add_systems(Update, (update_ui, capture_when_settled));
    } else {
        // Spawn the puller from a STARTUP system — i.e. only AFTER DefaultPlugins
        // (winit + GPU) initialized successfully. If render init panics on the Deck
        // (no display, bad Vulkan), Startup never runs, so no orphan child is left
        // pulling headless behind a dead window — and update-and-play.sh's crash-code
        // fallback can safely run a fresh headless pull without racing a live puller.
        app.insert_resource(PullerCfg(args.puller()));
        app.add_systems(Startup, start_puller);
        app.add_systems(Update, (update_ui, handle_dismiss));
    }
    app.run();

    // Propagate the puller's exit code. `child_code` holds the reaped code once the
    // reader thread sees stdout EOF + `wait()`; on a fast dismissal (user closes the
    // result screen before the thread reaps) it may still be None, so reconstruct from
    // the terminal result — rl-update only ever exits 0 (updated/already) or 1 (failed),
    // so the reconstruction matches. A render-init panic never reaches here (exits 101);
    // the shell wrapper treats that as a crash and falls back.
    let code = {
        let s = lock(&state);
        s.child_code.unwrap_or_else(|| {
            if s.result.as_ref().is_some_and(|d| d.ok) {
                0
            } else {
                1
            }
        })
    };
    std::process::exit(code);
}

/// Config for [`start_puller`] (held as a resource so the Startup system can read it).
#[derive(Resource)]
struct PullerCfg(PullerArgs);

/// Startup system: launch `rl-update` once the window/GPU are up. See the spawn-timing
/// rationale in [`main`].
fn start_puller(cfg: Res<PullerCfg>, ui: Res<Ui>) {
    spawn_puller(&cfg.0, ui.0.clone());
}

// --- screenshot (windowed render path; run under a display) -------------------------

/// A plausible mid-transfer snapshot for `--screenshot`: 3 files, ~180 MB, ~40% through
/// file 2 — enough to show the bar, current file, version, and byte readout. With
/// `done`, render the completed/dismiss screen instead (so the daemon can surface both
/// states from the same binary).
fn synthetic_state(done: bool) -> State {
    if done {
        return State {
            phase: Phase::Done,
            version: Some("a1b2c3d-w1750800000".to_string()),
            total: 182_000_000,
            done: 182_000_000,
            result: Some(DoneInfo {
                ok: true,
                message: "✅ updated to a1b2c3d-w1750800000 — 182.0 MB in 24.3s = 7.5 MB/s"
                    .to_string(),
            }),
            ..State::default()
        };
    }
    State {
        phase: Phase::Transferring,
        version: Some("a1b2c3d-w1750800000".to_string()),
        total: 182_000_000,
        done: 73_000_000,
        cur_idx: 2,
        cur_of: 3,
        cur_path: "rl-demo".to_string(),
        result: None,
        child_code: None,
    }
}

#[derive(Resource)]
struct ScreenshotPath(PathBuf);

#[derive(Resource)]
struct ShotSettle(u32);

#[derive(Resource, Default)]
struct ShotFrames {
    n: u32,
    countdown: Option<u32>,
}

/// After the window + GPU pipeline warm (a few dozen frames), capture one PNG of the
/// primary window and run a short exit countdown so the async readback + encode finish
/// before the app exits.
fn capture_when_settled(
    mut commands: Commands,
    settle: Res<ShotSettle>,
    path: Res<ScreenshotPath>,
    mut frames: ResMut<ShotFrames>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Some(countdown) = &mut frames.countdown {
        *countdown = countdown.saturating_sub(1);
        if *countdown == 0 {
            exit.write(AppExit::Success);
        }
        return;
    }
    frames.n += 1;
    if frames.n < settle.0 {
        return;
    }
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path.0.clone()));
    info!("rl-update-ui screenshot: captured at frame {}", frames.n);
    frames.countdown = Some(30);
}
