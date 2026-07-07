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
    #[arg(long, env = "RL_RELEASE_SERVER")]
    server: String,
    #[arg(long, env = "RL_INSTALL_DIR")]
    install_dir: PathBuf,
    #[arg(long, env = "RL_UPDATE_KEY")]
    key_file: PathBuf,
    #[arg(long, default_value = "./rl-update")]
    rl_update: PathBuf,
    #[arg(long)]
    windowed: bool,
    #[arg(long)]
    screenshot: Option<PathBuf>,
    #[arg(long, default_value_t = 120)]
    screenshot_settle: u32,
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

struct PullerArgs {
    rl_update: PathBuf,
    server: String,
    install_dir: PathBuf,
    key_file: PathBuf,
}

#[derive(Default, PartialEq, Clone, Copy)]
enum Phase {
    #[default]
    Connecting,
    Transferring,
    Done,
}

struct DoneInfo {
    ok: bool,
    message: String,
}

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
    child_code: Option<i32>,
}

#[derive(Resource, Clone)]
struct Ui(Arc<Mutex<State>>);

fn lock(m: &Mutex<State>) -> std::sync::MutexGuard<'_, State> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

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
            s.child_code = Some(1);
        }
    }
}

fn read_events(mut child: Child, state: Arc<Mutex<State>>) {
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
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
            if ok && result == "updated" && s.total > 0 {
                s.done = s.total;
            }
            s.result = Some(DoneInfo { ok, message });
        }
        _ => {}
    }
}

#[derive(Component, Clone, Copy)]
enum Slot {
    Status,
    Version,
    File,
    Bytes,
    Hint,
}

#[derive(Component)]
struct BarFill;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000.0
}

fn ascii_clean(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '✅' | '✓' | '✗' => ' ',
            '—' | '–' => '-',
            '·' => '-',
            '…' => '.',
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

fn main() {
    let args = Args::parse();
    let state = Arc::new(Mutex::new(State::default()));

    if args.screenshot.is_some() {
        *lock(&state) = synthetic_state(args.screenshot_done);
    }

    let mut app = App::new();
    // Deliberately NOT `crab_world::app_boot::base_plugins`: the updater must stay
    // standalone (no crab-world/otel deps — it has to build and run when the main tree
    // is broken), it bundles no assets, and LogPlugin IS its subscriber (no otel::init).
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "rl Update".into(),
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
        app.insert_resource(PullerCfg(args.puller()));
        app.add_systems(Startup, start_puller);
        app.add_systems(Update, (update_ui, handle_dismiss));
    }
    app.run();

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

#[derive(Resource)]
struct PullerCfg(PullerArgs);

fn start_puller(cfg: Res<PullerCfg>, ui: Res<Ui>) {
    spawn_puller(&cfg.0, ui.0.clone());
}

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
