use super::controls_sync::{sync_controls_context, sync_menu_controls_context};
use super::driver::{
    PendingRound, drive_client_sim, ensure_round_installed, insert_core, teardown_round,
};
use super::input::{gather_input, grab_cursor, quit_game, release_cursor};
use super::scene::{
    apply_transforms, place_extraction_pillar, reconcile_avatars, spawn_fp_camera, spawn_world,
    sync_ground_anchor,
};
use super::*;

pub enum Boot {
    Menu { seed: u64 },
    Round(Box<(ClientSim, Option<NetDriver>)>),
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppPhase {
    /// The boot menu: choose Host / Join. egui only.
    #[default]
    Menu,
    Connecting,
    /// The round is live: the FP client runs.
    Playing,
}

pub fn build_windowed_app(boot: Boot, view: crab_world::BootView) -> anyhow::Result<App> {
    let mut app = App::new();
    let window = Window {
        title: "Giant Crab Rescue".into(),
        // gamescope forces fullscreen on the Deck but a plain desktop leaves a Windowed app
        // windowed; BorderlessFullscreen makes the app own the policy on every target.
        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
        ..default()
    };
    // The browser's "fullscreen" is the canvas filling its parent — real fullscreen
    // needs a user gesture the boot path doesn't have. The page owns the layout; the
    // game binds to the one canvas the web entry ships (game-web/index.html).
    #[cfg(target_family = "wasm")]
    let window = Window {
        mode: WindowMode::Windowed,
        canvas: Some("#gcr-canvas".into()),
        fit_canvas_to_parent: true,
        ..window
    };
    // Web assets are baked (rl#411 stage 6): point the AssetServer at the pack's
    // in-memory tree instead of bevy's HTTP reader.
    #[cfg(target_family = "wasm")]
    crab_world::assets::register_web_asset_source(&mut app);
    app.add_plugins(crab_world::app_boot::base_plugins(Some(window)));
    app.add_plugins(crab_world::sky::NightSkyPlugin { moon: view.moon });
    super::audio::install(&mut app);
    super::ambience::install(&mut app);
    app.add_plugins(crab_world::physics::ArenaWorldPlugin {
        ground_look: view.ground_look,
    });
    install_craft_physics(&mut app);
    app.init_state::<AppPhase>();

    // The controls hint/overlay is app-global chrome (rl#117): the plugin owns its whole
    // lifecycle, and the per-phase sync systems below only retarget its ActiveContext
    // (Menu ↔ vehicles).
    app.add_plugins(crab_world::controls::ControlsOverlayPlugin::<GcrControls>::default());
    crab_world::chord::install_chords::<GcrControls>(&mut app);
    // The d-pad combo map (rl#358): discovered-only, persisted per save.
    super::chord_map::install(&mut app, super::chord_map::default_save_path());
    // In-game screenshot (rl#405): L3+R3 chord / F12, PNG + OTLP line.
    #[cfg(not(target_family = "wasm"))]
    super::live_screenshot::install(&mut app);
    // Outside Playing every chord dispatcher is gated off, so a capture there could
    // only pop the full-screen combo map and promise commands whose release does
    // nothing — keep the capture Idle instead (right after it would have opened, so
    // no Update reader ever sees a menu-phase capture).
    app.add_systems(
        PreUpdate,
        crab_world::chord::reset_chords::<GcrControls>
            .after(crab_world::chord::capture_chords::<GcrControls>)
            .run_if(not(in_state(AppPhase::Playing))),
    );

    app.init_non_send::<PendingRound>()
        .add_systems(
            OnEnter(AppPhase::Playing),
            (
                ensure_round_installed,
                spawn_world,
                spawn_fp_camera,
                super::audio::spawn_wind,
                // A modifier held across menu-confirm must not smuggle menu-time d-pad
                // taps into the round as a chord (rl#330).
                crab_world::chord::reset_chords::<GcrControls>,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                gather_input,
                drive_client_sim,
                // Directly after the sim step, before ANY consumer: the render
                // origin must be current when poses/transforms are written, or a
                // round's first frame renders against the previous locale (rl#354).
                sync_ground_anchor.before(crab_world::ground::apply_ground_anchor),
                reconcile_avatars,
                apply_transforms,
                // Bus writer before every bus reader, same frame: the code-entry
                // muffle (rl#359) lands on wind and beds as one mix move.
                super::audio::drive_muffle,
                super::audio::drive_wind,
                super::audio::trace_wind_speed,
                super::ambience::spawn_ambience,
                super::ambience::update_context,
                super::ambience::drive_layers,
                place_extraction_pillar,
            )
                .chain()
                .run_if(in_state(AppPhase::Playing)),
        )
        .add_systems(
            OnExit(AppPhase::Playing),
            (
                teardown_round,
                release_cursor,
                super::audio::despawn_wind,
                super::ambience::despawn_ambience,
            ),
        )
        .add_systems(
            Update,
            (grab_cursor, quit_game, toggle_debug_overlay).run_if(in_state(AppPhase::Playing)),
        )
        .add_systems(
            Update,
            (
                sync_controls_context.run_if(in_state(AppPhase::Playing)),
                sync_menu_controls_context.run_if(not(in_state(AppPhase::Playing))),
            )
                .before(update_controls_ui::<GcrControls>),
        );

    match boot {
        Boot::Round(round) => {
            let (client, net) = *round;
            let crate::menu::ReadyMatch { client, net } =
                start_round(crate::menu::ReadyMatch { client, net })
                    .map_err(|msg| anyhow::anyhow!(msg))?
                    .into_ready();
            let coord =
                super::driver::coordinator(net, client.peers(), client.me(), client.sim().clone());
            insert_core(&mut app, client, coord);
            app.world_mut()
                .resource_mut::<NextState<AppPhase>>()
                .set(AppPhase::Playing);
        }
        Boot::Menu { seed } => {
            app.insert_resource(BootedWithMenu);
            app.add_plugins(menu::MenuPlugin {
                seed,
                stamp: crate::SyncStamp::local(),
            });
        }
    }

    super::view_knobs::register(&mut app);

    Ok(app)
}

#[derive(Resource, Clone, Copy)]
pub(super) struct BootedWithMenu;

#[derive(Resource)]
pub(super) struct RoundOver {
    pub(super) message: String,
    pub(super) host: crate::menu::EndpointId,
}

/// A round whose peers passed the shared-world check — the only kind that may enter
/// Playing.
pub(super) struct SyncedRound(crate::menu::ReadyMatch);

impl SyncedRound {
    pub(super) fn into_ready(self) -> crate::menu::ReadyMatch {
        self.0
    }
}

pub(super) fn start_round(ready: crate::menu::ReadyMatch) -> Result<SyncedRound, String> {
    check_synced(ready.net.as_ref().map(NetDriver::synced)).map(|()| {
        // tracing, not println: the one round-entry marker that reaches every
        // platform's log sink (the web verify greps it from the browser console).
        tracing::info!(
            "ROUND_ARMED net={}",
            if ready.net.is_some() { "peers" } else { "solo" }
        );
        SyncedRound(ready)
    })
}

/// `None` is solo: no formation ran, nothing to disagree with.
pub(super) fn check_synced(synced: Option<bool>) -> Result<(), String> {
    if synced != Some(false) {
        return Ok(());
    }
    Err(
        "refusing to start the round — a peer stands on a different world (its binary \
         embeds a different terrain bake, so its ground would disagree with the poses it \
         adopts). Fix: run the identical build on every device, then re-form the match."
            .to_string(),
    )
}

/// The crafts' physics, pumped by hand on the host ([`super::physics_pump`]).
pub(super) fn install_craft_physics(app: &mut App) {
    app.add_plugins(crab_world::physics::WorldPhysicsPlugin)
        .add_plugins(crab_world::vehicle::VehiclePlugin);
    super::physics_pump::park_fixed_auto_pump(app);
}

/// The rl#326 F3 overlay, reachable from the controller: GCR runs on a TV where no
/// keyboard is at hand when the game turns slideshow (rl#331).
fn toggle_debug_overlay(
    chords: Res<crab_world::chord::Chords<GcrControls>>,
    mut overlay: ResMut<crab_world::debug_overlay::DebugOverlay>,
) {
    if chords.executed(controls::Action::ToggleDebugOverlay) {
        overlay.visible = !overlay.visible;
    }
}
