//! Boot menu: the client-side egui Host / Join UI + host-triggered lobby, gated to
//! the Menu/Connecting phases. Builds the round only at the Playing transition (via the
//! parent's [`super::driver::ensure_round_installed`]), so it never touches the sim. The
//! pure, Bevy-free connection orchestration lives in [`crate::menu`].

use std::sync::mpsc;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use super::AppPhase;
use super::app::RoundOver;
use super::driver::PendingRound;
use crate::menu::{
    self, ChooserItem, DisconnectedItem, EndpointId, Formation, LobbyItem, MenuAction, MenuInput,
    MenuNav, StartChoice,
};
use crate::net_loop::{self, JoinResult};

/// Wires the boot menu into the windowed app: the egui menu + connecting-poll pass.
/// The round install at `OnEnter(Playing)` is `ensure_round_installed` in the parent
/// module (always scheduled, chained ahead of the spawns) — the menu only *parks* its
/// chosen round in [`PendingRound`]. Carries the shared match seed + optional telemetry
/// collector so a networked Host/Join formation gets them.
pub struct MenuPlugin {
    pub seed: u64,
    pub telemetry: Option<EndpointId>,
    /// Our NN-crab checkpoint digest, `0` for none. Advertised in networked
    /// formation so peers can agree on a shared brain before arming the float crab.
    pub weights_digest: u64,
    /// Our crab-model-asset digest, `0` for none. Advertised alongside
    /// `weights_digest` so peers can agree on a shared collider asset before arming.
    pub asset_digest: u64,
}

/// The camera the menu/connecting screens render into. bevy_egui 0.39 is
/// camera-driven — it attaches its primary context to a [`Camera`] entity, so WITHOUT
/// a camera the egui pass is skipped and the menu never draws. The round spawns its own
/// `Camera3d` only at `OnEnter(Playing)`, so the menu needs this one of its own for the
/// pre-round phases; it's despawned the instant we enter Playing so it never coexists
/// with (or double-renders over) the FP camera.
#[derive(Component)]
struct MenuCamera;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        app.insert_non_send_resource(MenuState::new(
            self.seed,
            self.telemetry,
            self.weights_digest,
            self.asset_digest,
        ))
        // A 2D camera for the menu so bevy_egui has a context to render into.
        // Spawned on entering Menu (the default phase, so it fires at startup on the
        // menu boot; never on the scripted Boot::Round path, which supersedes Menu
        // with Playing before any transition). Re-entering Menu (Cancel/error from
        // Connecting) despawns any prior one first, so there's never a duplicate.
        // `consume_round_over` is chained AFTER the nav reset: a disconnect return (rl#203)
        // must land on the "connection lost — rejoin?" prompt, not the clean chooser the
        // reset produces.
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

/// Spawn the menu's 2D camera (despawning any leftover first, so re-entering Menu from
/// Connecting can't stack two). Without a camera bevy_egui renders nothing.
fn spawn_menu_camera(mut commands: Commands, existing: Query<Entity, With<MenuCamera>>) {
    for e in existing.iter() {
        commands.entity(e).despawn();
    }
    commands.spawn((Camera2d, MenuCamera));
}

/// Despawn the menu camera as the round starts, so it doesn't linger into Playing and
/// double-render over the FP `Camera3d`.
fn despawn_menu_camera(mut commands: Commands, cams: Query<Entity, With<MenuCamera>>) {
    for e in cams.iter() {
        commands.entity(e).despawn();
    }
}

/// The menu's working state (non-send: a started [`Formation`] holds a tokio runtime
/// via the `NetDriver`, which isn't `Send`). Holds the navigation FSM, the join-code text
/// field, the in-flight formation, and any error to show. The finished round is parked in
/// the parent's [`PendingRound`], not here. All pre-round UI bookkeeping — none of it is
/// sim state.
struct MenuState {
    seed: u64,
    telemetry: Option<EndpointId>,
    /// Our NN-crab checkpoint digest, `0` for none — handed to
    /// [`crate::menu::begin`] so networked formation advertises it.
    weights_digest: u64,
    /// Our crab-model-asset digest, `0` for none — handed to
    /// [`crate::menu::begin`] alongside `weights_digest`.
    asset_digest: u64,
    /// The pure navigation FSM ([`MenuNav`]) — focus + the chooser/lobby transition.
    /// Folded by controller/keyboard input AND egui clicks through one path, so every
    /// confirm (Start included) routes through the same tested dispatch.
    nav: MenuNav,
    /// Edge latch for left-stick menu nav: `true` while the stick is held past the nav
    /// threshold, cleared when it recenters, so a held stick steps one item, not many.
    stick_latched: bool,
    /// The join-code text the player is typing (an endpoint id), for Join-by-code.
    code_input: String,
    /// A networked Host/Join formation running on a background thread, while Connecting.
    forming: Option<Formation>,
    /// Last error to surface on the menu (bad code, formation failed), cleared when the
    /// player retries.
    error: Option<String>,
    /// The host endpoint of the match we were just disconnected from (rl#203) — the rejoin
    /// re-dial target behind the "connection lost — rejoin?" prompt. Set from [`RoundOver`],
    /// cleared once a rejoin verdict says it can't work (refused/unreachable).
    last_host: Option<EndpointId>,
    /// A rejoin dial ([`net_loop::connect_and_join`]) running on a background thread, while
    /// Connecting with [`MenuNav::Rejoining`]. Dropping it abandons the dial (the thread's
    /// send fails and its session tears down; the dial itself is bounded by the join timeout).
    rejoining: Option<mpsc::Receiver<anyhow::Result<JoinResult>>>,
}

impl MenuState {
    fn new(
        seed: u64,
        telemetry: Option<EndpointId>,
        weights_digest: u64,
        asset_digest: u64,
    ) -> Self {
        Self {
            seed,
            telemetry,
            weights_digest,
            asset_digest,
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

/// Reset the navigation FSM to the chooser whenever we (re)enter the Menu phase, so a
/// Cancel/error return from Connecting always lands on a clean Host-focused chooser.
fn reset_menu_nav(mut state: NonSendMut<MenuState>) {
    state.nav = MenuNav::new();
    state.stick_latched = false;
}

/// Consume a [`RoundOver`] left by `drive_lockstep` (rl#203: the live round's server link died
/// or the host refused us): surface its message and land on the "connection lost — rejoin?"
/// prompt. Chained after [`reset_menu_nav`], which it refines; a normal menu entry has no
/// `RoundOver` and this no-ops.
fn consume_round_over(world: &mut World) {
    let Some(over) = world.remove_resource::<RoundOver>() else {
        return;
    };
    let mut state = world.non_send_resource_mut::<MenuState>();
    state.error = Some(over.message);
    state.last_host = Some(over.host);
    state.nav = MenuNav::disconnected();
}

/// The single egui system for the boot flow: poll the formation, gather
/// controller/keyboard navigation, draw the current screen, and drive every transition
/// through the pure [`MenuNav`] FSM + one exhaustive [`apply_action`] dispatch.
///
/// Input unification: controller (D-pad/stick + A/B), keyboard (arrows/WASD +
/// Enter/Esc), and egui clicks ALL reduce to a [`MenuInput`] folded through `MenuNav`, so
/// every confirm — Start included — takes the same wired path and a gamepad-only player
/// (the Steam Deck case) can operate the whole menu.
///
/// Determinism: this only ever *selects/commands* a formation and reads its finished
/// result. The round it parks (in [`PendingRound`]) is built by [`menu::ready_from`] /
/// [`menu::solo_round`] from the unchanged barrier output — no sim state originates here.
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

    // Connecting: poll the background formation (or rejoin dial) FIRST, so a finished match
    // transitions this frame before we draw or take any input on a screen that's about to
    // vanish.
    if matches!(phase.get(), AppPhase::Connecting)
        && (poll_formation(&mut state, &mut pending, &mut next)
            || poll_rejoin(&mut state, &mut pending, &mut next))
    {
        return Ok(());
    }

    // The live roster size, the only context the FSM needs (host Start: solo vs networked).
    let lobby_len = state.forming.as_ref().map(|f| f.lobby_len()).unwrap_or(0);

    // Suppress KEYBOARD nav while egui has keyboard focus (the player is typing in the
    // join-code field) — otherwise Space/Enter/W/S double as nav and would, e.g., fire
    // Confirm mid-paste. The gamepad stays live (it never feeds the text field).
    let typing = ctx.wants_keyboard_input();

    // Controller / keyboard navigation: reduce raw input to FSM events, fold each, and
    // execute the action. A leaving action (Start/Cancel) makes the rest of the frame moot.
    let inputs = gather_menu_inputs(&keys, &gamepads, typing, &mut state.stick_latched);
    for input in inputs {
        let action = state.nav.step(input, lobby_len);
        if apply_action(action, &mut state, &mut pending, &mut next) {
            return Ok(());
        }
    }

    // Draw the current screen and route any click through the SAME FSM path (focus the
    // clicked item, then Confirm), so a click and a controller confirm can't diverge.
    match phase.get() {
        // The Menu phase draws the "connection lost — rejoin?" prompt when a disconnect
        // return landed us on it (rl#203), else the Host/Join chooser.
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
        // Connecting draws the rejoin dial when one is in flight, else the lobby.
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
        // Playing is gated out by the run condition; nothing to draw.
        AppPhase::Playing => {}
    }
    Ok(())
}

/// Reduce this frame's raw keyboard + gamepad state to [`MenuInput`] events (edge-
/// triggered, so a held control steps one item). Controller: D-pad + left stick to move,
/// South (A) to confirm, East (B) to back/cancel — folded across every connected pad.
/// Keyboard: arrows/WASD to move, Enter/Space to confirm, Esc to back. The thin, untested
/// input-gathering layer; all the navigation LOGIC lives in the unit-tested [`MenuNav`].
fn gather_menu_inputs(
    keys: &ButtonInput<KeyCode>,
    gamepads: &Query<&Gamepad>,
    typing: bool,
    stick_latched: &mut bool,
) -> Vec<MenuInput> {
    let mut out = Vec::new();

    // Keyboard, edge-triggered — skipped entirely while a text field has focus (`typing`)
    // so keystrokes meant for the join-code field don't also drive navigation.
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

    // Gamepad, folded across all pads. D-pad + face buttons are edge-triggered; the stick
    // is its own analog channel (handled below with a re-center latch).
    let mut up = false;
    let mut down = false;
    let mut confirm = false;
    let mut back = false;
    // Largest-magnitude left-stick Y across pads, so one player's stick drives the menu.
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
    // Stick → discrete nav with a re-center latch: emit ONE Up/Down on crossing the
    // threshold, then nothing until the stick falls back near center — so a held stick
    // moves one item, not a blur of them. Stick up (+Y) is Up.
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

/// Execute one [`MenuAction`]. The ONE place menu actions become side effects — the match
/// is exhaustive, so a new action variant can't be added without wiring it here (a dead
/// button is a compile error). Returns `true` if it changed the AppPhase (entered Playing,
/// or moved to/from the lobby), so the caller stops drawing a screen that's leaving this
/// frame — keeping the drawn screen consistent with the FSM's new state.
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
            // Parse the optional code: blank = discover on the LAN (no dial); a non-empty
            // field must parse to an endpoint id or it's a user error we surface. On a bad
            // code, revert the FSM to the chooser (it already advanced to the lobby on
            // Confirm) so the screen and AppPhase stay consistent.
            let trimmed = state.code_input.trim();
            let host = if trimmed.is_empty() {
                None
            } else {
                match trimmed.parse::<EndpointId>() {
                    Ok(id) => Some(id),
                    Err(_) => {
                        state.error = Some("That join code isn't a valid endpoint id.".into());
                        // The FSM advanced to the lobby on Confirm; revert it so the
                        // chooser (with the error) keeps drawing, consistent with the
                        // still-Menu AppPhase.
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
            // Peers present: command the barrier's synchronized GO. The formed networked
            // round arrives on a later poll, which then enters Playing.
            if let Some(f) = &state.forming {
                f.request_start();
            }
            false
        }
        MenuAction::StartSolo => {
            // Host-alone Start: abandon the wait (cancel the barrier so its session tears
            // down) and install the shared solo round INSTANTLY — the SAME deterministic
            // round the Alone fallback produces. No discovery dependency.
            if let Some(f) = &state.forming {
                f.cancel();
            }
            state.forming = None;
            // A solo round carries no formation verdict, so the gate structurally passes.
            pending.0 = Some(
                super::app::arm_round(menu::solo_round(state.seed))
                    .expect("a solo round always arms (net is None — nothing to desync)"),
            );
            next.set(AppPhase::Playing);
            true
        }
        MenuAction::Cancel => {
            // Tell the barrier to bail and tear its session down (no ~12s LAN phantom),
            // drop the handle, and return to the menu. An in-flight rejoin dial is dropped
            // the same way (its thread self-bounds on the join timeout and tears down).
            if let Some(f) = &state.forming {
                f.cancel();
            }
            state.forming = None;
            state.rejoining = None;
            next.set(AppPhase::Menu);
            true
        }
        MenuAction::Rejoin => {
            // Re-dial the lost match's host and JoinRequest back in (rl#203) — the SAME
            // [`net_loop::connect_and_join`] path `game net-join` scripts, run off-thread so
            // the menu never blocks on the dial. The verdict arrives in [`poll_rejoin`].
            let Some(host) = state.last_host else {
                // Unreachable by construction (the prompt is only entered with a host), but
                // degrade to the chooser rather than panic in UI code.
                state.nav = MenuNav::new();
                return false;
            };
            state.error = None;
            let (tx, rx) = mpsc::channel();
            let (seed, telemetry, asset_digest) = (state.seed, state.telemetry, state.asset_digest);
            std::thread::spawn(move || {
                let _ = tx.send(net_loop::connect_and_join(
                    seed,
                    host,
                    telemetry,
                    asset_digest,
                ));
            });
            state.rejoining = Some(rx);
            next.set(AppPhase::Connecting);
            true
        }
    }
}

/// Poll the background barrier; returns `true` if it had finished and we transitioned
/// (parked the round → Playing, or returned to Menu on cancel/error) this frame.
fn poll_formation(
    state: &mut MenuState,
    pending: &mut PendingRound,
    next: &mut NextState<AppPhase>,
) -> bool {
    let Some(result) = state.forming.as_ref().and_then(|f| f.poll()) else {
        return false;
    };
    // Done forming: drop the handle and act on the result.
    state.forming = None;
    match result {
        // A round formed (networked, or the Alone fallback): install it and play.
        // `ready_from` is `None` only for Cancelled, which the barrier reports after
        // tearing its session down — return to the menu, no phantom left behind.
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

/// Poll the background rejoin dial (rl#203); returns `true` if a verdict landed and we
/// transitioned this frame: admitted → park the round and enter Playing; refused /
/// unreachable / failed → back to the chooser with the verdict shown (and no further rejoin
/// offered — the verdict was terminal, so a re-dial can't change it).
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
        // The worker dropped its sender without sending — only a panic does that.
        Err(mpsc::TryRecvError::Disconnected) => {
            Err(anyhow::anyhow!("rejoin thread ended unexpectedly"))
        }
    };
    state.rejoining = None;
    state.nav = MenuNav::new();
    match result {
        // Admitted: the joiner's driver carries the gate-passed sync verdict, so the arm
        // gate structurally passes — but it stays the ONE gate every round crosses.
        Ok(JoinResult::Joined(joined)) => {
            let (lockstep, net) = *joined;
            state.last_host = None;
            arm_and_play(
                menu::ReadyMatch {
                    lockstep,
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

/// Gate a finished round through the ONE arm gate and act on the verdict: park the
/// [`ArmedRound`](super::app::ArmedRound) proof and enter Playing, or surface the actionable
/// refusal (peers disagree on the brain/colliders — run rl-update on every device) on the
/// chooser. Deliberately NO silent integer-crab swap — the round refuses, loud and visible.
/// The shared tail of [`poll_formation`] and [`poll_rejoin`], so the two can't drift.
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

/// Open the host-triggered lobby for a Host/Join choice and move to Connecting. Shared by
/// the Host and Join actions so the "begin lobby + clear error + switch phase" sequence
/// has one definition.
fn start_forming(state: &mut MenuState, choice: &StartChoice, next: &mut NextState<AppPhase>) {
    state.error = None;
    state.forming = Some(menu::begin(
        choice,
        state.seed,
        state.telemetry,
        state.weights_digest,
        state.asset_digest,
    ));
    next.set(AppPhase::Connecting);
}

/// Draw the Host / Join chooser (no separate Solo button; Host-alone IS solo).
/// The focused item (from [`MenuNav`]) is highlighted via `selectable_label`; returns the
/// item the player clicked, if any, for the caller to route through the FSM.
fn draw_chooser(ctx: &egui::Context, state: &mut MenuState) -> Option<ChooserItem> {
    let focus = match state.nav {
        MenuNav::Chooser { focus } => focus,
        // Off-screen (shouldn't happen in the Menu phase); default to Host highlight.
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

            // Host: open a lobby. Play alone (Start with nobody joined = an instant solo
            // round) or wait for others to Join by our code / the LAN, then Start.
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

/// Draw the "connection lost — rejoin?" prompt (rl#203): the disconnect message and the
/// focusable Rejoin / Leave pair. Returns the clicked item, for the caller to route through
/// the SAME FSM path as a controller confirm.
fn draw_disconnected(ctx: &egui::Context, state: &MenuState) -> Option<DisconnectedItem> {
    let focus = match state.nav {
        MenuNav::Disconnected { focus } => focus,
        // Only drawn on the prompt; a sane default keeps a stray frame rendering.
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

/// Draw the rejoin-in-flight screen: the dial status + Cancel (the only action — the FSM's
/// [`MenuNav::Rejoining`] mirrors the joiner's lobby). Returns `true` on a Cancel click, which
/// the caller routes through the FSM as a Confirm (Confirm on `Rejoining` IS Cancel).
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

/// Draw the lobby / connecting screen: the role, the join code (Host) or dial status
/// (Join), the live roster, and the focusable Start (host) + Cancel. Returns the clicked
/// item, if any. Polling already happened in `menu_screen`, so this only renders + reports.
fn draw_lobby(ctx: &egui::Context, state: &MenuState, lobby: &[EndpointId]) -> Option<LobbyItem> {
    let (hosting, focus) = match state.nav {
        MenuNav::HostLobby { focus } => (true, focus),
        MenuNav::JoinLobby => (false, LobbyItem::Cancel),
        // Off-screen default; the lobby only draws in Connecting where nav is a lobby
        // variant. Fall back to the formation's role so the frame still renders sanely.
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
                // The host's own code is its endpoint id, surfaced once the session binds.
                match display_code {
                    Some(code) => {
                        // A selectable, read-only field so the player can copy the code.
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

            // Live roster: the players currently in the lobby. Host alone shows
            // just itself, which is the cue that Start = a solo round. When hosting, the
            // host's own id is its join code (`display_code`), so mark it "you"; a joiner
            // doesn't know which id is its own here, so nothing is marked for it.
            ui.separator();
            let me = if hosting { display_code } else { None };
            lobby_roster(ui, lobby, me);

            ui.separator();
            if hosting {
                // The host commands the start. Alone → an instant solo round;
                // with peers → the synchronized networked start. The label reflects which,
                // read from the live roster so it's honest about what Start does.
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
                // A joiner can't start; it waits for the host's GO.
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

/// Draw the lobby's live player list: one line per player, `me` (if given)
/// marked. `roster` is the barrier's current `live_set` (sorted by id bytes), empty until
/// the session binds.
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
