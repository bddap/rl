use super::driver::{
    PendingRound, drive_lockstep, ensure_round_installed, insert_core, park_fixed_auto_pump,
    teardown_round,
};
use super::hud::{spawn_hud, sync_controls_context, update_hud};
use super::input::{gather_input, grab_cursor, quit_game, release_cursor};
use super::scene::{
    apply_transforms, follow_ground, reconcile_avatars, spawn_fp_camera, spawn_world,
};
use super::*;

pub enum Boot {
    Menu {
        seed: u64,
        telemetry: Option<crate::menu::EndpointId>,
    },
    Round(Box<(Lockstep, Option<NetDriver>)>),
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
    external_crab: Vec<std::path::PathBuf>,
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

    app.init_non_send_resource::<PendingRound>()
        .init_resource::<ActiveDevice>()
        .init_resource::<ActiveContext<GcrControls>>()
        .insert_resource(ForceRevealControls(false))
        .add_systems(
            OnEnter(AppPhase::Playing),
            (
                ensure_round_installed,
                spawn_world,
                spawn_fp_camera,
                spawn_hud,
                spawn_controls_ui::<GcrControls>,
                tag_controls_ui_for_round,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                gather_input,
                drive_lockstep,
                reconcile_avatars,
                apply_transforms,
                follow_ground,
                update_hud,
            )
                .chain()
                .run_if(in_state(AppPhase::Playing)),
        )
        .add_systems(OnExit(AppPhase::Playing), (teardown_round, release_cursor))
        .add_systems(
            Update,
            (
                grab_cursor,
                quit_game,
                (
                    track_active_device,
                    sync_controls_context,
                    update_controls_ui::<GcrControls>,
                )
                    .chain(),
            )
                .run_if(in_state(AppPhase::Playing)),
        );

    let asset_digest = crab_world::mesh_fallback::constructed_body_digest();

    match boot {
        Boot::Round(round) => {
            let (ls, net) = *round;
            let armed = arm_round(crate::menu::ReadyMatch { lockstep: ls, net })
                .map_err(|msg| anyhow::anyhow!(msg))?;
            let crate::menu::ReadyMatch {
                lockstep: mut ls,
                net,
            } = armed.into_ready();
            let spawns = seed_round_crabs(&mut ls, external_crab.len());
            let coord = super::driver::coordinator(net, ls.peers(), ls.me(), ls.sim().clone());
            insert_core(&mut app, ls, coord);
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
                asset_digest,
                crab_count: external_crab.len() as u8,
            });
            {
                let dirs = external_crab;
                let mut throwaway = crate::formation::solo_lockstep_for(seed);
                throwaway.configure_crabs(dirs.len());
                let crab_spawns: Vec<Pos> =
                    throwaway.sim().crabs().iter().map(|c| c.pos()).collect();
                add_external_nn_crab(&mut app, dirs, crab_spawns);
                app.insert_resource(crab_world::bot::NumEnvs(0));
                app.insert_resource(ExternalCrabStackInstalled);
            }
        }
    }

    super::render_mode::register(&mut app, render_mode);

    Ok(app)
}

#[allow(clippy::type_complexity)]
fn tag_controls_ui_for_round(
    mut commands: Commands,
    roots: Query<
        Entity,
        Or<(
            With<crab_world::controls::ControlsHintRoot>,
            With<crab_world::controls::ControlsOverlayRoot>,
        )>,
    >,
) {
    for e in roots.iter() {
        commands.entity(e).insert(DespawnOnExit(AppPhase::Playing));
    }
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
            "the crab colliders (the sally.glb model) differ on a peer — it would build and \
             render a different crab",
            "run rl-update on every device so all peers share the same crab model",
        )
    };
    Err(format!(
        "rl#114: refusing to start the round — can't arm the trained NN crabs (\"Sally\") for \
         this multiplayer match because {cause}. Fix: {fix}, then re-form the match. (There is \
         deliberately no integer stand-in crab — an unarmable round refuses rather than \
         silently dropping Sally.)"
    ))
}

pub(super) fn seed_round_crabs(ls: &mut Lockstep, crabs: usize) -> Vec<Pos> {
    ls.configure_crabs(crabs);
    let spawns: Vec<Pos> = ls.sim().crabs().iter().map(|c| c.pos()).collect();
    for (idx, crab) in ls.sim().crabs().to_vec().into_iter().enumerate() {
        ls.set_external_crab_pose(idx, crab.pos(), crab.yaw());
    }
    spawns
}

pub(super) fn add_external_nn_crab(
    app: &mut App,
    checkpoint_dirs: Vec<std::path::PathBuf>,
    crab_spawns: Vec<Pos>,
) {
    app.insert_resource(crab_world::Visuals(true))
        .insert_resource(crab_world::bot::NumEnvs(checkpoint_dirs.len()))
        .add_plugins(crab_world::physics::CrabPhysicsPlugin)
        // The OPEN inference field — unbounded ground, no walls — so the crab's per-round
        // travel isn't capped at the ±10 m training box and it can chase a player (spawned
        // ≥12 m out) clear across the map (rl#209). Training keeps the walled box.
        .add_plugins(crab_world::physics::PhysicsWorldPlugin {
            arena: crab_world::physics::Arena::OpenField,
        })
        .add_plugins(crab_world::bot::BotPlugin)
        .add_plugins(crab_world::vehicle::VehiclePlugin)
        .add_plugins(crate::external_crab::ExternalCrabPlugin {
            checkpoint_dirs,
            crab_spawns,
        });

    park_fixed_auto_pump(app.world_mut());
}

pub(super) fn install_armed_nn_crab(
    app: &mut App,
    checkpoint_dirs: Vec<std::path::PathBuf>,
    crab_spawns: Vec<Pos>,
) {
    add_external_nn_crab(app, checkpoint_dirs, crab_spawns);
    crate::external_crab::arm(app.world_mut());
}
