use super::driver::{
    PendingRound, drive_client_sim, ensure_round_installed, insert_core, park_fixed_auto_pump,
    teardown_round,
};
use super::hud::{spawn_hud, sync_controls_context, sync_menu_controls_context, update_hud};
use super::input::{gather_input, grab_cursor, quit_game, release_cursor};
use super::scene::{
    apply_transforms, place_extraction_pillar, reconcile_avatars, spawn_fp_camera, spawn_world,
    sync_arena_surface,
};
use super::*;

pub enum Boot {
    Menu {
        seed: u64,
        telemetry: Option<crate::menu::EndpointId>,
    },
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

pub fn build_windowed_app(
    boot: Boot,
    external_crab: Vec<crab_world::policy::Policy>,
    render_mode: super::RenderMode,
) -> anyhow::Result<App> {
    // NO determinism pin, on ANY boot (rl#199): only the solo/host peer steps the float NN
    // crab — a remote client adopts snapshots and steps nothing — so no runtime path compares
    // float state across peers (hash-log/telemetry hashes are offline diagnostics).
    // Single-thread pinning lives where reproducibility is actually consumed: the trainer,
    // eval, and the headless probe ([`crab_world::bot::headless::pin_single_thread_pools`]).
    let mut app = App::new();
    app.add_plugins(crab_world::app_boot::base_plugins(Some(Window {
        title: "Giant Crab Rescue".into(),
        // Fullscreen is the single source of truth for every GCR launch target. The Deck
        // shows fullscreen only because gamescope forces it; on a plain desktop/TV (bothouse)
        // a Windowed app stayed windowed. BorderlessFullscreen makes the app itself own the
        // policy, so bothouse matches the Deck with no separate per-host window-config path.
        mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
        ..default()
    })));
    app.add_plugins(crab_world::sky::NightSkyPlugin);
    app.init_state::<AppPhase>();

    // The controls hint/overlay is app-global chrome (rl#117): the plugin owns its whole
    // lifecycle, and the per-phase sync systems below only retarget its ActiveContext
    // (Menu ↔ vehicles). No force-knobs — the windowed client exposes none, so live state
    // drives the legend; it goes through the one installer anyway so there is no second
    // wiring of the overlay to drift.
    crab_world::controls::install_overlay::<GcrControls>(&mut app, &Default::default());

    app.init_non_send_resource::<PendingRound>()
        .add_systems(
            OnEnter(AppPhase::Playing),
            (
                ensure_round_installed,
                spawn_world,
                spawn_fp_camera,
                spawn_hud,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                gather_input,
                drive_client_sim,
                super::articulation::sample_crab_part_poses,
                reconcile_avatars,
                apply_transforms,
                sync_arena_surface,
                place_extraction_pillar,
                update_hud,
            )
                .chain()
                .run_if(in_state(AppPhase::Playing)),
        )
        .add_systems(OnExit(AppPhase::Playing), (teardown_round, release_cursor))
        .add_systems(
            Update,
            (grab_cursor, quit_game, super::brain_swap::swap_brain)
                .run_if(in_state(AppPhase::Playing)),
        )
        .add_systems(
            Update,
            (
                sync_controls_context.run_if(in_state(AppPhase::Playing)),
                sync_menu_controls_context.run_if(not(in_state(AppPhase::Playing))),
            )
                .before(update_controls_ui::<GcrControls>),
        );

    let body_digest = crab_world::mesh_fallback::constructed_body_digest();

    match boot {
        Boot::Round(round) => {
            let (client, net) = *round;
            let armed = arm_round(crate::menu::ReadyMatch { client, net })
                .map_err(|msg| anyhow::anyhow!(msg))?;
            let crate::menu::ReadyMatch { mut client, net } = armed.into_ready();
            let spawns = seed_round_crabs(&mut client, external_crab.len());
            let coord =
                super::driver::coordinator(net, client.peers(), client.me(), client.sim().clone());
            insert_core(&mut app, client, coord);
            install_armed_nn_crab(&mut app, external_crab, spawns);
            app.world_mut()
                .resource_mut::<NextState<AppPhase>>()
                .set(AppPhase::Playing);
        }
        Boot::Menu { seed, telemetry } => {
            app.insert_resource(BootedWithMenu);
            app.add_plugins(menu::MenuPlugin {
                seed,
                telemetry,
                body_digest,
                crab_count: external_crab.len() as u8,
            });
            {
                let policies = external_crab;
                let mut throwaway = crate::formation::solo_client_for(seed);
                throwaway.configure_crabs(policies.len());
                let crab_spawns: Vec<Pos> =
                    throwaway.sim().crabs().iter().map(|c| c.pos()).collect();
                add_external_nn_crab(&mut app, policies, crab_spawns);
                app.insert_resource(crab_world::bot::NumEnvs(0));
                app.insert_resource(ExternalCrabStackInstalled);
            }
        }
    }

    super::render_mode::register(&mut app, render_mode);

    Ok(app)
}

#[derive(Resource, Clone, Copy)]
pub(super) struct BootedWithMenu;

#[derive(Resource)]
pub(super) struct RoundOver {
    pub(super) message: String,
    pub(super) host: crate::menu::EndpointId,
}

#[derive(Resource, Clone, Copy)]
pub(super) struct ExternalCrabStackInstalled;

pub(super) struct ArmedRound(crate::menu::ReadyMatch);

impl ArmedRound {
    pub(super) fn into_ready(self) -> crate::menu::ReadyMatch {
        self.0
    }
}

pub(super) fn arm_round(ready: crate::menu::ReadyMatch) -> Result<ArmedRound, String> {
    check_armable(ready.net.as_ref().map(NetDriver::sync_verdict)).map(|()| ArmedRound(ready))
}

pub(super) fn check_armable(sync: Option<crate::SyncVerdict>) -> Result<(), String> {
    if crate::may_arm_external_crab(sync) {
        return Ok(());
    }
    let (cause, fix) = if !sync.is_some_and(|v| v.crabs) {
        (
            "the HOST's NN-crab count doesn't line up — it serves no crabs at all (a headless \
             driver hosting a rest-pose match) or a different count than this device renders",
            "host from a windowed device, and launch every device with the same \
             --nn-crab-checkpoint binding list",
        )
    } else {
        (
            "the crab body differs on a peer (a different sally.glb, baked collider table, \
             or binary version) — it would build and render a different crab",
            "run rl-update on every device so all peers share the same build + model",
        )
    };
    Err(format!(
        "rl#114: refusing to start the round — can't arm the trained NN crabs (\"Sally\") for \
         this multiplayer match because {cause}. Fix: {fix}, then re-form the match. (There is \
         deliberately no integer stand-in crab — an unarmable round refuses rather than \
         silently dropping Sally.)"
    ))
}

pub(super) fn seed_round_crabs(client: &mut ClientSim, crabs: usize) -> Vec<Pos> {
    client.configure_crabs(crabs);
    let spawns: Vec<Pos> = client.sim().crabs().iter().map(|c| c.pos()).collect();
    for (idx, crab) in client.sim().crabs().to_vec().into_iter().enumerate() {
        client.set_external_crab_pose(idx, crab.pos(), crab.yaw());
    }
    spawns
}

pub(super) fn add_external_nn_crab(
    app: &mut App,
    policies: Vec<crab_world::policy::Policy>,
    crab_spawns: Vec<Pos>,
) {
    // The world half of the plant rides the checkpoint (rl#281 stage 6): the launch gate
    // adopted the recorded arena before arming, so a terrain brain gets its baked tile —
    // GCR's world IS the arena, rendered through the anchor. The flat case unwalls: the
    // ±10 m training cage would cap the chase, and the open grid lets the crab pursue a
    // player (spawned beyond sim::MIN_CRAB_SPAWN_DISTANCE) clear across the map (rl#209).
    let arena = match crab_world::physics::train_arena() {
        crab_world::physics::TrainArena::WalledBox => crab_world::physics::Arena::OpenField,
        crab_world::physics::TrainArena::Terrain => crab_world::physics::Arena::Terrain,
    };
    app.insert_resource(crab_world::Visuals(true))
        .insert_resource(crab_world::bot::NumEnvs(policies.len()))
        .add_plugins(crab_world::physics::CrabPhysicsPlugin)
        .add_plugins(crab_world::physics::PhysicsWorldPlugin { arena })
        // The arena's own dressing — terrain mesh + biome tint, vista or flat lighting,
        // fog — replaces GCR's old bespoke checker quad (one ground path, rl#281):
        // [`sync_arena_surface`] keeps the drawn surface at the arena anchor so it renders
        // exactly where the physics stands.
        .add_plugins(crab_world::physics::ArenaVisualsPlugin)
        .add_plugins(crab_world::bot::BotPlugin)
        .add_plugins(crab_world::vehicle::VehiclePlugin)
        .add_plugins(crate::external_crab::ExternalCrabPlugin::new(
            policies,
            crab_spawns,
        ));

    park_fixed_auto_pump(app.world_mut());
}

pub(super) fn install_armed_nn_crab(
    app: &mut App,
    policies: Vec<crab_world::policy::Policy>,
    crab_spawns: Vec<Pos>,
) {
    add_external_nn_crab(app, policies, crab_spawns);
    crate::external_crab::arm(app.world_mut());
}
