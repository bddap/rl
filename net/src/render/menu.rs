use std::sync::mpsc;

use bevy::prelude::*;
use bevy_egui::{
    EguiContextSettings, EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass,
    PrimaryEguiContext, egui,
};

use super::AppPhase;
use super::app::RoundOver;
use super::driver::PendingRound;
use crate::menu::{
    self, ChooserItem, DisconnectedItem, EndpointId, Formation, LobbyItem, MenuAction, MenuInput,
    MenuNav, StartChoice,
};
use crate::net_loop::{self, JoinResult};

pub struct MenuPlugin {
    pub seed: u64,
    pub telemetry: Option<EndpointId>,
    pub asset_digest: u64,
    pub crab_count: u8,
}

#[derive(Component)]
pub(super) struct MenuCamera;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
            // Registered with the egui bootstrap itself (not the menu systems below) so
            // any future egui surface inherits the workspace UI-scale rule for free.
            app.add_systems(Update, sync_egui_scale);
        }
        // The menu camera carries its own PrimaryEguiContext (spawn_menu_camera), so
        // auto-creation is off: bevy_egui's auto-create latches on the FIRST camera and
        // never fires again (Local<bool>, bevy_egui-0.39.1), which left the menu
        // context-less after a round-over respawn (rl#237). One creation path: ours —
        // flipped unconditionally (idempotent) so an EguiPlugin added elsewhere first
        // can't silently re-arm auto-create.
        app.world_mut()
            .resource_mut::<EguiGlobalSettings>()
            .auto_create_primary_context = false;
        app.insert_non_send_resource(MenuState::new(
            self.seed,
            self.telemetry,
            self.asset_digest,
            self.crab_count,
        ))
        .add_systems(
            OnEnter(AppPhase::Menu),
            (spawn_menu_camera, reset_menu_nav, consume_round_over).chain(),
        )
        // Tear it down as the round begins, before the FP Camera3d spawns, so the
        // two never coexist.
        .add_systems(OnEnter(AppPhase::Playing), despawn_menu_camera)
        // The menu + connecting poll draw in the egui pass (per render frame),
        // gated to the two pre-round phases so they vanish once Playing.
        .add_systems(
            EguiPrimaryContextPass,
            menu_screen.run_if(not(in_state(AppPhase::Playing))),
        );
    }
}

// egui renders at points × window scale factor × settings.scale_factor — the same
// composition bevy UI gives UiScale — so mirroring the already-synced UiScale (kept on
// the workspace rule by crab_world::app_boot) makes egui/bevy-UI divergence impossible
// (rl#227).
fn sync_egui_scale(ui_scale: Res<UiScale>, mut contexts: Query<&mut EguiContextSettings>) {
    for mut settings in &mut contexts {
        if (settings.scale_factor - ui_scale.0).abs() > 1e-3 {
            settings.scale_factor = ui_scale.0;
        }
    }
}

pub(super) fn spawn_menu_camera(mut commands: Commands, existing: Query<Entity, With<MenuCamera>>) {
    for e in existing.iter() {
        commands.entity(e).despawn();
    }
    // Explicit PrimaryEguiContext (it #[require]s EguiContext): the context must live
    // and die with THIS camera on every Menu entry, not depend on a one-shot global
    // auto-create that already fired on a previous life (rl#237).
    commands.spawn((Camera2d, MenuCamera, PrimaryEguiContext));
}

pub(super) fn despawn_menu_camera(mut commands: Commands, cams: Query<Entity, With<MenuCamera>>) {
    for e in cams.iter() {
        commands.entity(e).despawn();
    }
}

struct MenuState {
    seed: u64,
    telemetry: Option<EndpointId>,
    asset_digest: u64,
    crab_count: u8,
    nav: MenuNav,
    stick_latched: bool,
    code_input: String,
    forming: Option<Formation>,
    error: Option<String>,
    last_host: Option<EndpointId>,
    rejoining: Option<mpsc::Receiver<anyhow::Result<JoinResult>>>,
}

impl MenuState {
    fn new(seed: u64, telemetry: Option<EndpointId>, asset_digest: u64, crab_count: u8) -> Self {
        Self {
            seed,
            telemetry,
            asset_digest,
            crab_count,
            nav: MenuNav::new(),
            stick_latched: false,
            code_input: String::new(),
            forming: None,
            error: None,
            last_host: None,
            rejoining: None,
        }
    }
}

fn reset_menu_nav(mut state: NonSendMut<MenuState>) {
    state.nav = MenuNav::new();
    state.stick_latched = false;
}

fn consume_round_over(world: &mut World) {
    let Some(over) = world.remove_resource::<RoundOver>() else {
        return;
    };
    let mut state = world.non_send_resource_mut::<MenuState>();
    state.error = Some(over.message);
    state.last_host = Some(over.host);
    state.nav = MenuNav::disconnected();
}

fn menu_screen(
    mut contexts: EguiContexts,
    mut state: NonSendMut<MenuState>,
    mut pending: NonSendMut<PendingRound>,
    phase: Res<State<AppPhase>>,
    mut next: ResMut<NextState<AppPhase>>,
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    if matches!(phase.get(), AppPhase::Connecting)
        && (poll_formation(&mut state, &mut pending, &mut next)
            || poll_rejoin(&mut state, &mut pending, &mut next))
    {
        return Ok(());
    }

    let lobby_len = state.forming.as_ref().map(|f| f.lobby_len()).unwrap_or(0);

    let typing = ctx.wants_keyboard_input();

    let inputs = gather_menu_inputs(&keys, &gamepads, typing, &mut state.stick_latched);
    for input in inputs {
        let action = state.nav.step(input, lobby_len);
        if apply_action(action, &mut state, &mut pending, &mut next) {
            return Ok(());
        }
    }

    match phase.get() {
        AppPhase::Menu => {
            if matches!(state.nav, MenuNav::Disconnected { .. }) {
                if let Some(item) = draw_disconnected(ctx, &state) {
                    state.nav.focus_disconnected(item);
                    let action = state.nav.step(MenuInput::Confirm, lobby_len);
                    apply_action(action, &mut state, &mut pending, &mut next);
                }
            } else if let Some(item) = draw_chooser(ctx, &mut state) {
                state.nav.focus_chooser(item);
                let action = state.nav.step(MenuInput::Confirm, lobby_len);
                apply_action(action, &mut state, &mut pending, &mut next);
            }
        }
        AppPhase::Connecting => {
            if state.rejoining.is_some() {
                if draw_rejoining(ctx, &state) {
                    let action = state.nav.step(MenuInput::Confirm, lobby_len);
                    apply_action(action, &mut state, &mut pending, &mut next);
                }
            } else {
                let lobby = state
                    .forming
                    .as_ref()
                    .map(|f| f.roster())
                    .unwrap_or_default();
                if let Some(item) = draw_lobby(ctx, &state, &lobby) {
                    state.nav.focus_lobby(item);
                    let action = state.nav.step(MenuInput::Confirm, lobby_len);
                    apply_action(action, &mut state, &mut pending, &mut next);
                }
            }
        }
        AppPhase::Playing => {}
    }
    Ok(())
}

fn gather_menu_inputs(
    keys: &ButtonInput<KeyCode>,
    gamepads: &Query<&Gamepad>,
    typing: bool,
    stick_latched: &mut bool,
) -> Vec<MenuInput> {
    let mut out = Vec::new();

    if !typing {
        if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
            out.push(MenuInput::Up);
        }
        if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
            out.push(MenuInput::Down);
        }
        if keys.just_pressed(KeyCode::Enter)
            || keys.just_pressed(KeyCode::NumpadEnter)
            || keys.just_pressed(KeyCode::Space)
        {
            out.push(MenuInput::Confirm);
        }
        if keys.just_pressed(KeyCode::Escape) {
            out.push(MenuInput::Back);
        }
    }

    let mut up = false;
    let mut down = false;
    let mut confirm = false;
    let mut back = false;
    let mut stick_y = 0.0f32;
    for gp in gamepads.iter() {
        up |= gp.just_pressed(GamepadButton::DPadUp);
        down |= gp.just_pressed(GamepadButton::DPadDown);
        confirm |= gp.just_pressed(GamepadButton::South);
        back |= gp.just_pressed(GamepadButton::East);
        let y = gp.left_stick().y;
        if y.abs() > stick_y.abs() {
            stick_y = y;
        }
    }
    const NAV_THRESH: f32 = 0.6;
    if stick_y.abs() < NAV_THRESH * 0.5 {
        *stick_latched = false;
    } else if !*stick_latched && stick_y.abs() >= NAV_THRESH {
        *stick_latched = true;
        if stick_y > 0.0 {
            up = true;
        } else {
            down = true;
        }
    }

    if up {
        out.push(MenuInput::Up);
    }
    if down {
        out.push(MenuInput::Down);
    }
    if confirm {
        out.push(MenuInput::Confirm);
    }
    if back {
        out.push(MenuInput::Back);
    }
    out
}

fn apply_action(
    action: MenuAction,
    state: &mut MenuState,
    pending: &mut PendingRound,
    next: &mut NextState<AppPhase>,
) -> bool {
    match action {
        MenuAction::None => false,
        MenuAction::Host => {
            start_forming(state, &StartChoice::Host, next);
            true
        }
        MenuAction::Join => {
            let trimmed = state.code_input.trim();
            let host = if trimmed.is_empty() {
                None
            } else {
                match trimmed.parse::<EndpointId>() {
                    Ok(id) => Some(id),
                    Err(_) => {
                        state.error = Some("That join code isn't a valid endpoint id.".into());
                        state.nav = MenuNav::Chooser {
                            focus: ChooserItem::Join,
                        };
                        return false;
                    }
                }
            };
            start_forming(state, &StartChoice::Join(host), next);
            true
        }
        MenuAction::StartNetworked => {
            if let Some(f) = &state.forming {
                f.request_start();
            }
            false
        }
        MenuAction::StartSolo => {
            if let Some(f) = &state.forming {
                f.cancel();
            }
            state.forming = None;
            pending.0 = Some(
                super::app::arm_round(menu::solo_round(state.seed))
                    .expect("a solo round always arms (net is None — nothing to desync)"),
            );
            next.set(AppPhase::Playing);
            true
        }
        MenuAction::Cancel => {
            if let Some(f) = &state.forming {
                f.cancel();
            }
            state.forming = None;
            state.rejoining = None;
            next.set(AppPhase::Menu);
            true
        }
        MenuAction::Rejoin => {
            let Some(host) = state.last_host else {
                state.nav = MenuNav::new();
                return false;
            };
            state.error = None;
            let (tx, rx) = mpsc::channel();
            let (seed, telemetry, asset_digest, crab_count) = (
                state.seed,
                state.telemetry,
                state.asset_digest,
                state.crab_count,
            );
            std::thread::spawn(move || {
                let _ = tx.send(net_loop::connect_and_join(
                    seed,
                    host,
                    telemetry,
                    asset_digest,
                    crab_count,
                ));
            });
            state.rejoining = Some(rx);
            next.set(AppPhase::Connecting);
            true
        }
    }
}

fn poll_formation(
    state: &mut MenuState,
    pending: &mut PendingRound,
    next: &mut NextState<AppPhase>,
) -> bool {
    let Some(result) = state.forming.as_ref().and_then(|f| f.poll()) else {
        return false;
    };
    state.forming = None;
    match result {
        Ok(match_result) => match menu::ready_from(match_result, state.seed) {
            Some(ready) => arm_and_play(ready, state, pending, next),
            None => next.set(AppPhase::Menu),
        },
        Err(e) => {
            state.error = Some(format!("Couldn't form a match: {e:#}"));
            next.set(AppPhase::Menu);
        }
    }
    true
}

fn poll_rejoin(
    state: &mut MenuState,
    pending: &mut PendingRound,
    next: &mut NextState<AppPhase>,
) -> bool {
    let Some(rx) = &state.rejoining else {
        return false;
    };
    let result = match rx.try_recv() {
        Ok(r) => r,
        Err(mpsc::TryRecvError::Empty) => return false,
        Err(mpsc::TryRecvError::Disconnected) => {
            Err(anyhow::anyhow!("rejoin thread ended unexpectedly"))
        }
    };
    state.rejoining = None;
    state.nav = MenuNav::new();
    match result {
        Ok(JoinResult::Joined(joined)) => {
            let (client, net) = *joined;
            state.last_host = None;
            arm_and_play(
                menu::ReadyMatch {
                    client,
                    net: Some(net),
                },
                state,
                pending,
                next,
            );
        }
        Ok(JoinResult::Refused(reason)) => {
            state.error = Some(format!("The host refused our rejoin: {reason}"));
            state.last_host = None;
            next.set(AppPhase::Menu);
        }
        Ok(JoinResult::Unreachable) => {
            state.error = Some(
                "The host is unreachable — it may have quit. Host a new match, or join a \
                 fresh code."
                    .into(),
            );
            state.last_host = None;
            next.set(AppPhase::Menu);
        }
        Err(e) => {
            state.error = Some(format!("Rejoin failed: {e:#}"));
            next.set(AppPhase::Menu);
        }
    }
    true
}

fn arm_and_play(
    ready: menu::ReadyMatch,
    state: &mut MenuState,
    pending: &mut PendingRound,
    next: &mut NextState<AppPhase>,
) {
    match super::app::arm_round(ready) {
        Ok(armed) => {
            pending.0 = Some(armed);
            next.set(AppPhase::Playing);
        }
        Err(msg) => {
            state.error = Some(msg);
            next.set(AppPhase::Menu);
        }
    }
}

fn start_forming(state: &mut MenuState, choice: &StartChoice, next: &mut NextState<AppPhase>) {
    state.error = None;
    state.forming = Some(menu::begin(
        choice,
        state.seed,
        state.telemetry,
        state.asset_digest,
        state.crab_count,
    ));
    next.set(AppPhase::Connecting);
}

fn draw_chooser(ctx: &egui::Context, state: &mut MenuState) -> Option<ChooserItem> {
    let focus = match state.nav {
        MenuNav::Chooser { focus } => focus,
        _ => ChooserItem::Host,
    };
    let mut clicked = None;
    egui::Window::new("Giant Crab Rescue")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.heading("Giant Crab Rescue");
            ui.label("Rescue the giant crab. Reach the green pillar to extract.");
            ui.separator();

            if ui
                .selectable_label(
                    focus == ChooserItem::Host,
                    "Host (play alone or with others)",
                )
                .clicked()
            {
                clicked = Some(ChooserItem::Host);
            }

            ui.separator();
            ui.label("…or join someone on your LAN:");
            ui.horizontal(|ui| {
                ui.label("Join code:");
                ui.add(
                    egui::TextEdit::singleline(&mut state.code_input)
                        .desired_width(260.0)
                        .hint_text("paste host code (optional)"),
                );
            });
            if ui
                .selectable_label(focus == ChooserItem::Join, "Join a match")
                .clicked()
            {
                clicked = Some(ChooserItem::Join);
            }

            ui.separator();
            ui.label("Keyboard: arrows / WASD · Enter to select · Esc to back.");

            if let Some(err) = &state.error {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(230, 120, 120), err);
            }
        });
    clicked
}

fn draw_disconnected(ctx: &egui::Context, state: &MenuState) -> Option<DisconnectedItem> {
    let focus = match state.nav {
        MenuNav::Disconnected { focus } => focus,
        _ => DisconnectedItem::Rejoin,
    };
    let mut clicked = None;
    egui::Window::new("Connection lost")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.heading("Connection lost");
            if let Some(err) = &state.error {
                ui.colored_label(egui::Color32::from_rgb(230, 120, 120), err);
            }
            ui.separator();
            let rejoin_label = match state.last_host {
                Some(host) => format!("Rejoin the match (host {})", host.fmt_short()),
                None => "Rejoin the match".to_string(),
            };
            if ui
                .selectable_label(focus == DisconnectedItem::Rejoin, rejoin_label)
                .clicked()
            {
                clicked = Some(DisconnectedItem::Rejoin);
            }
            if ui
                .selectable_label(focus == DisconnectedItem::Leave, "Back to menu")
                .clicked()
            {
                clicked = Some(DisconnectedItem::Leave);
            }
            ui.separator();
            ui.label("Controller: A to select · B to back. Keyboard: Enter · Esc.");
        });
    clicked
}

fn draw_rejoining(ctx: &egui::Context, state: &MenuState) -> bool {
    let mut clicked = false;
    egui::Window::new("Rejoining")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.heading("Rejoining the match…");
            if let Some(host) = state.last_host {
                ui.label(format!("Dialing host {}…", host.fmt_short()));
            }
            ui.spinner();
            ui.separator();
            if ui.selectable_label(true, "Cancel").clicked() {
                clicked = true;
            }
            ui.separator();
            ui.label("Controller: A or B to cancel. Keyboard: Enter · Esc.");
        });
    clicked
}

fn draw_lobby(ctx: &egui::Context, state: &MenuState, lobby: &[EndpointId]) -> Option<LobbyItem> {
    let (hosting, focus) = match state.nav {
        MenuNav::HostLobby { focus } => (true, focus),
        MenuNav::JoinLobby => (false, LobbyItem::Cancel),
        _ => (
            state.forming.as_ref().is_some_and(|f| f.hosting),
            LobbyItem::Cancel,
        ),
    };
    let display_code = state.forming.as_ref().and_then(|f| f.display_code());
    let mut clicked = None;
    egui::Window::new("Lobby")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            if hosting {
                ui.heading("Hosting a match");
                ui.label("Share this join code (or others can find you on the LAN):");
                match display_code {
                    Some(code) => {
                        let mut code_str = code.to_string();
                        ui.add(egui::TextEdit::singleline(&mut code_str).desired_width(360.0));
                    }
                    None => {
                        ui.label("(starting host — code will appear shortly)");
                    }
                }
            } else {
                ui.heading("Joining a match…");
                match display_code {
                    Some(code) => {
                        ui.label(format!("Dialing host {}…", code.fmt_short()));
                    }
                    None => {
                        ui.label("Discovering a host on the LAN…");
                    }
                }
            }

            ui.separator();
            let me = if hosting { display_code } else { None };
            lobby_roster(ui, lobby, me);

            ui.separator();
            if hosting {
                let solo = lobby.len() <= 1;
                let label = if solo {
                    "Start (solo — nobody has joined)"
                } else {
                    "Start the match"
                };
                if ui
                    .selectable_label(focus == LobbyItem::Start, label)
                    .clicked()
                {
                    clicked = Some(LobbyItem::Start);
                }
            } else {
                ui.spinner();
                ui.label("Waiting for the host to start…");
            }
            if ui
                .selectable_label(focus == LobbyItem::Cancel, "Cancel")
                .clicked()
            {
                clicked = Some(LobbyItem::Cancel);
            }

            ui.separator();
            ui.label("Controller: A to select · B to cancel. Keyboard: Enter · Esc.");
        });
    clicked
}

fn lobby_roster(ui: &mut egui::Ui, roster: &[EndpointId], me: Option<EndpointId>) {
    if roster.is_empty() {
        ui.label("Players: (connecting…)");
        return;
    }
    ui.label(format!("Players in the lobby: {}", roster.len()));
    for id in roster {
        let tag = if Some(*id) == me { "  (you)" } else { "" };
        ui.label(format!("  • {}{}", id.fmt_short(), tag));
    }
}
