//! Windowed first-person client: app assembly + boot wiring.
//!
//! Builds the windowed `App` ([`build_windowed_app`]), defines the boot enum/phase
//! ([`Boot`]/[`AppPhase`]), and installs the real rapier-NN crab stack. The headless
//! screenshot builder lives in [`super::screenshot`]; the per-frame systems it wires
//! (`gather_input`, `drive_lockstep`, `apply_transforms`, `update_hud`) live in the
//! sibling submodules.

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

/// How the windowed client starts up: at the boot MENU (the interactive default — the
/// player picks Host/Join), or straight into a prebuilt ROUND (the scripted
/// `--host`/`--join` flags, which form the match up front so tests/scripts never depend on
/// clicking the menu). One enum, two boots, so "has a menu AND a prebuilt round" is
/// unrepresentable rather than two bool flags.
pub enum Boot {
    /// Show the boot menu first; the sim is built only once the player chooses and the
    /// host-triggered lobby resolves. `seed` is the shared match seed and `telemetry` the
    /// optional collector id — both threaded to whichever formation the menu kicks off.
    Menu {
        seed: u64,
        telemetry: Option<crate::menu::EndpointId>,
    },
    /// Skip the menu and play this already-formed round immediately. The scripted entry
    /// (`--host`/`--join` = the formed lockstep + its driver; a host-alone `--host` that
    /// found no peer = a solo lockstep + `None`). Boxed because the lockstep + driver are
    /// large and `Menu` is tiny — without the box every `Menu` would carry that dead weight
    /// (the same reason [`crate::net_loop::MatchResult::Joined`] boxes).
    Round(Box<(Lockstep, Option<NetDriver>)>),
}

/// The windowed client's top-level phase. The menu and lobby screens are PURE
/// client UI — no [`Lockstep`]/[`Sim`] exists until [`AppPhase::Playing`], which is entered
/// only after a choice (and, for networked roles, a host-commanded start). This is the
/// firewall that keeps the menu off the deterministic sim: the FP systems and the sim
/// resource are all gated to `Playing`, so menu state literally cannot reach the round
/// (it's built fresh on the transition from the formation machinery).
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppPhase {
    /// The boot menu: choose Host / Join. egui only.
    #[default]
    Menu,
    /// A host-triggered lobby is forming on a background thread; show the live roster +
    /// (Host) join code + Start, and poll for the result. A Host-alone Start skips straight
    /// to its instant solo round without lingering here.
    Connecting,
    /// The round is live: the FP client runs.
    Playing,
}

/// Build the windowed first-person client app. Starts at the boot menu or straight in a
/// round per [`Boot`]; owns the `Lockstep` + optional `NetDriver` as resources once
/// playing. Built here, not via a `Plugin` that holds the sim, because
/// `Plugin::build(&self)` can't move a non-`Clone` `Lockstep`/`NetDriver` out of
/// itself — inserting them as resources at the `Playing` transition is the clean path.
///
/// `external_crab` is the REQUIRED trained brain-binding list for the giant crabs — one
/// checkpoint dir per crab, in crab-index order (rl#200: a multi-brain round runs several REAL
/// rapier-simulated NN bodies, [`crate::external_crab`]). Every binding was validated fail-loud
/// by the CLI before this is called. There is NO integer fallback (rl#114): a SOLO round always
/// arms them; a NETWORKED round arms them once peers agree on the crab-model asset and the host
/// steps them at the deterministic cadence. A networked-UNSYNCED round CANNOT arm Sally and
/// FAILS LOUD — a clear peer-mismatch error naming the fix (run rl-update on every device)
/// rather than silently substituting a fake crab. The failure is GRACEFUL: the scripted
/// `Boot::Round` path returns it as an `Err` (clean CLI exit, no panic); the interactive menu
/// pre-gates it in `poll_formation` and returns to the chooser showing the message (no
/// mid-transition crash).
///
/// `render_mode` is the crab render view's starting mode (mesh in normal play; the player cycles
/// it live with the `CycleRenderMode` control, boots into a mode via `--render-mode`/
/// `RL_RENDER_MODE`, or — with no Sally glb — defaults to `Colliders`).
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
    // Night-sky skybox behind the first-person view (attaches to the FP camera when the
    // round spawns it). Shared with the rl-demo via `crab_world::sky`.
    app.add_plugins(crab_world::sky::NightSkyPlugin);
    app.init_state::<AppPhase>();

    // The FP round systems, gated to Playing. spawn_* runs at the Playing transition
    // (the sim doesn't exist until then); the per-frame systems run only while
    // playing so they never touch a not-yet-built GameState.
    //
    // `ensure_round_installed` is CHAINED ahead of the spawns: on the menu path it moves
    // the chosen round into GameState here (the sim must exist before spawn_world reads
    // it); on the scripted Boot::Round path GameState already exists, so it no-ops. The
    // chain is what guarantees the sim is live before the scene spawns — separate
    // OnEnter system sets have no ordering, which would race spawn_world ahead of the install.
    app.init_non_send_resource::<PendingRound>()
        .init_resource::<ActiveDevice>()
        // The overlay's live context starts on foot; `sync_controls_context` drives it from
        // `LocalVehicle` each frame (the windowed path; the screenshot path sets it from env).
        .init_resource::<ActiveContext<GcrControls>>()
        // The windowed client never forces the overlay open — it's hold-to-reveal. Inserting
        // the resource here (false) keeps `update_controls_ui` reading a plain `Res`, not an
        // `Option<Res>`; only the screenshot path sets it true.
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
        // The OnExit mirror of the OnEnter installs (rl#203: the disconnect return): round
        // ENTITIES despawn via `DespawnOnExit(Playing)` at their spawns; this removes the round
        // resources, disarms the crab, and hands the cursor back to the menu, so re-entering
        // Playing installs a fresh round.
        .add_systems(OnExit(AppPhase::Playing), (teardown_round, release_cursor))
        .add_systems(
            Update,
            (
                grab_cursor,
                quit_game,
                // chained so the glyph swap reflects THIS frame's device, and the legend +
                // context name reflect THIS frame's vehicle (sync before the overlay update).
                (
                    track_active_device,
                    sync_controls_context,
                    update_controls_ui::<GcrControls>,
                )
                    .chain(),
            )
                .run_if(in_state(AppPhase::Playing)),
        );

    // OUR crab-MODEL-asset digest: the giant crabs' rapier colliders are derived
    // from this asset ([`crab_world::mesh_fallback::constructed_body_digest`]), so peers must agree on it
    // before arming the float crabs — a different model builds and renders different crabs.
    // Computed unconditionally (it's a property of this peer's installed crab model,
    // independent of whether a checkpoint loaded); `0` for no model. (No weights digest is
    // advertised any more — rl#200 increment 6: clients run no inference, and each binding was
    // validated fail-loud at launch.)
    let asset_digest = crab_world::mesh_fallback::constructed_body_digest();

    match boot {
        // Scripted boot: insert the round now and jump straight to Playing (the menu
        // states are never entered). NextState applied before the first frame, so
        // OnEnter(Playing) fires and the world spawns on frame one — no menu flash. The
        // scripted `--host`/`--join` path tests/scripts use; a host-alone `--host` that
        // found no peer is a solo round here, so it gets the real NN crab.
        Boot::Round(round) => {
            let (ls, net) = *round;
            // Arm the one giant crab — the real NN body — through the SAME [`arm_round`] gate
            // the menu path uses ([`crate::may_arm_external_crab`], the determinism guard). A
            // networked round that can't agree on the crab colliders can't arm Sally; it
            // REFUSES rather than play a fake crab — but as a clean error bubbled to the CLI,
            // not a `panic!` process-abort (this is the scripted `--host`/`--join` path; the
            // interactive menu pre-gates in `poll_formation` and never reaches here).
            let armed = arm_round(crate::menu::ReadyMatch { lockstep: ls, net })
                .map_err(|msg| anyhow::anyhow!(msg))?;
            let crate::menu::ReadyMatch {
                lockstep: mut ls,
                net,
            } = armed.into_ready();
            // Size the crab set to the binding count, then capture the spawns + seed the
            // poses BEFORE `ls` moves into core.
            let spawns = seed_round_crabs(&mut ls, external_crab.len());
            let coord = super::driver::coordinator(net, ls.peers(), ls.me(), ls.sim().clone());
            insert_core(&mut app, ls, coord);
            // Known-armed at build: add the stack AND arm the gate now (one path,
            // [`install_armed_nn_crab`]), so the crabs spawn frame one at the players' actual
            // positions — nothing per-peer to reconcile.
            install_armed_nn_crab(&mut app, external_crab, spawns);
            app.world_mut()
                .resource_mut::<NextState<AppPhase>>()
                .set(AppPhase::Playing);
        }
        // Interactive boot: add the menu plugin (egui menu + lobby poll). The sim is built
        // later, at the Playing transition, from the choice the menu records.
        Boot::Menu { seed, telemetry } => {
            // A server-down verdict mid-round returns to this menu (rl#203); the scripted
            // Boot::Round path has no menu, so there it exits instead — this marker is how
            // `drive_lockstep` tells the two apart.
            app.insert_resource(BootedWithMenu);
            app.add_plugins(menu::MenuPlugin {
                seed,
                telemetry,
                asset_digest,
                crab_count: external_crab.len() as u8,
            });
            // NN crabs on the round: the menu can't know at BUILD time whether the
            // round will be solo, networked-synced, or networked-unsynced, so add the whole NN
            // stack now with the gate OFF and no crab spawned (NumEnvs 0), and arm it only if
            // the resolved round may ([`ensure_round_installed`] → [`crate::may_arm_external_crab`]:
            // solo always, networked only with synced assets). The crabs' arena spawns are a
            // pure function of the seed + binding count (a throwaway solo lockstep reads them),
            // so they're known here without the round existing yet. The checkpoints are
            // REQUIRED, so the stack is always installed; a networked-UNSYNCED round leaves the
            // gate off and `ensure_round_installed` FAILS LOUD rather than substituting a fake
            // crab.
            {
                let dirs = external_crab;
                let mut throwaway = crate::formation::solo_lockstep_for(seed);
                throwaway.configure_crabs(dirs.len());
                let crab_spawns: Vec<Pos> =
                    throwaway.sim().crabs().iter().map(|c| c.pos()).collect();
                add_external_nn_crab(&mut app, dirs, crab_spawns);
                // Gate OFF: leave `ExternalCrabArmed` ABSENT (presence is the state). The
                // transition (`ensure_round_installed`) inserts it iff the round resolves armable.
                app.insert_resource(crab_world::bot::NumEnvs(0)); // no crab spawns behind the menu
                app.insert_resource(ExternalCrabStackInstalled); // the transition may activate it
            }
        }
    }

    // The crab render-mode cycle (shared cage + skin/silhouette visibility + the live cycle).
    super::render_mode::register(&mut app, render_mode);

    Ok(app)
}

/// Tag the controls hint + overlay roots for round teardown. They're spawned by the SHARED
/// crab-world [`spawn_controls_ui`] (chained just before this), which can't know our
/// [`AppPhase`] — so the tag rides in here, keeping ALL round-entity cleanup on the one
/// `DespawnOnExit(Playing)` mechanism (rl#203: the disconnect return; without the tag the
/// round HUD would render over the menu and stack on re-entry).
#[allow(clippy::type_complexity)] // the Or-of-markers filter is the whole point of the query.
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

/// Presence marker: this app booted with the interactive menu ([`Boot::Menu`]), so a mid-round
/// server-down verdict (rl#203) has a menu to return to. Absent on the scripted [`Boot::Round`]
/// path, which exits loudly instead — a presence marker, not a bool, for the same reason as
/// [`ExternalCrabStackInstalled`].
#[derive(Resource, Clone, Copy)]
pub(super) struct BootedWithMenu;

/// Why the live round just ended out from under the player (rl#203: the server link died or the
/// host refused us) — inserted by `drive_lockstep` alongside the Playing→Menu transition, and
/// consumed by the menu's OnEnter to show the "connection lost — rejoin?" prompt. `host` is the
/// lost match's server endpoint, for the rejoin re-dial — non-optional, because a server-down
/// verdict only exists on a remote client, which always knows its server's endpoint.
#[derive(Resource)]
pub(super) struct RoundOver {
    pub(super) message: String,
    pub(super) host: crate::menu::EndpointId,
}

/// Presence marker: the boot-menu app installed the NN-crab stack at build. The checkpoint
/// is REQUIRED, so on the menu path this is always inserted; the Playing transition reads
/// its presence and arms the crab once the round resolves armable (solo always, networked only when
/// synced), FAILING LOUD otherwise — there is no integer crab to fall back to. A presence marker,
/// not a bool: "not installed" is simply the resource's absence (the scripted `Boot::Round` path),
/// so there's no degenerate `false` state to mishandle.
#[derive(Resource, Clone, Copy)]
pub(super) struct ExternalCrabStackInstalled;

/// PROOF that a round passed the arm gate for the one giant crab (the real NN body, "Sally") —
/// constructible ONLY by [`arm_round`], so holding one IS holding the verdict
/// (impossible-by-construction). `ensure_round_installed` consumes it and arms without
/// re-checking: a future path that parks an unvalidated round is a type error, not a slipped
/// runtime assert.
pub(super) struct ArmedRound(crate::menu::ReadyMatch);

impl ArmedRound {
    /// Surrender the proof and take the round to install. Consuming (not borrowing) so a
    /// round is installed at most once per gate pass.
    pub(super) fn into_ready(self) -> crate::menu::ReadyMatch {
        self.0
    }
}

/// The SINGLE arm gate for the one giant crab: pass and the round comes back as the
/// [`ArmedRound`] proof; fail and the `Err` names exactly why the networked round cannot arm
/// — peers disagree on the brain or the crab colliders, so a float crab would desync lockstep
/// — and how to fix it. With no integer fallback (rl#114) an unarmable round REFUSES rather
/// than silently substituting a fake crab. ONE source for both the gate and the
/// operator-facing text, used by the menu pre-gate (return to the chooser showing the `Err`,
/// no crash) and the scripted `Boot::Round` build (bubble it out as an error, no
/// panic). Solo always arms ([`crate::may_arm_external_crab`] on `None`), so the message only
/// ever describes a networked round.
pub(super) fn arm_round(ready: crate::menu::ReadyMatch) -> Result<ArmedRound, String> {
    check_armable(ready.net.as_ref().map(NetDriver::sync_verdict)).map(|()| ArmedRound(ready))
}

/// The pure core of [`arm_round`] — the arm decision + refusal message from the formation
/// verdict (`None` = solo), with no [`NetDriver`] so it's unit-testable headlessly (no
/// tokio/iroh).
pub(super) fn check_armable(sync: Option<crate::SyncVerdict>) -> Result<(), String> {
    if crate::may_arm_external_crab(sync) {
        return Ok(());
    }
    // Reached only on a networked round that can't arm (so `sync` is present). Name the
    // failing half: the collider asset, or the host-keyed crab count (a 0-count host is a
    // rest-pose match — the headless driver; a count mismatch renders the wrong number of
    // crabs). The host's own brains are validated fail-loud at launch — no brain cause here.
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

/// Size the round's crab set to the binding count (`crabs`), then seed each sim crab's spawn
/// pose into `ls` so every rapier-NN body begins where the round placed its giant crab, and
/// return those spawns for [`add_external_nn_crab`]. The ONE configure+seed every round
/// install funnels through — the windowed `Boot::Round` build, the menu path's
/// `ensure_round_installed`, and the headless screenshot builders — so they can't drift
/// (one implementation per thing). Writes back each crab's CURRENT pose, so sim state is
/// unchanged; digest 0 to seed (the bridge's first post-step `hash_crab_physics` fills the
/// real digests before any cross-check).
pub(super) fn seed_round_crabs(ls: &mut Lockstep, crabs: usize) -> Vec<Pos> {
    ls.configure_crabs(crabs);
    let spawns: Vec<Pos> = ls.sim().crabs().iter().map(|c| c.pos()).collect();
    for (idx, crab) in ls.sim().crabs().to_vec().into_iter().enumerate() {
        ls.set_external_crab_pose(idx, crab.pos(), crab.yaw(), 0);
    }
    spawns
}

/// Wire the real rapier-NN crabs into the windowed solo app: the bot/physics/brain stack
/// (the SAME plugins `rl --demo` runs, so the crabs step the exact dynamics the policies
/// trained under) plus the [`external_crab::ExternalCrabPlugin`] bridge that walks each toward
/// its player and feeds its body position back into the sim. With the model present the
/// cosmetic skins ride the bodies; with no `sally.glb` the visible crab is the static giant
/// silhouette (`spawn_world` keeps it shown when no skin loads).
pub(super) fn add_external_nn_crab(
    app: &mut App,
    checkpoint_dirs: Vec<std::path::PathBuf>,
    crab_spawns: Vec<Pos>,
) {
    app.insert_resource(crab_world::Visuals(true))
        .insert_resource(crab_world::bot::NumEnvs(checkpoint_dirs.len()))
        // Same fixed timestep + softened contact spring as training/demo, with Rapier in
        // FixedUpdate (lockstep with the Sense→Think→Act brain loop) — bundled in the one
        // order that applies the spring, so the solo crab's physics can't drift from what
        // the policy optimised under (see physics::CrabPhysicsPlugin).
        .add_plugins(crab_world::physics::CrabPhysicsPlugin)
        // The OPEN inference field — unbounded ground, no walls — so the crab's per-round
        // travel isn't capped at the ±10 m training box and it can chase a player (spawned
        // ≥12 m out) clear across the map (rl#209). Training keeps the walled box.
        .add_plugins(crab_world::physics::PhysicsWorldPlugin {
            arena: crab_world::physics::Arena::OpenField,
        })
        .add_plugins(crab_world::bot::BotPlugin)
        // The player's rapier flight vehicle — a rigidbody in this same crab world, so it collides
        // with Sally. Inert (no body, no systems firing on a spawned body) until the player boards
        // one; `drive_lockstep` files the pilot's `PilotCommand` into its `VehicleControls` each tick.
        .add_plugins(crab_world::vehicle::VehiclePlugin)
        .add_plugins(crate::external_crab::ExternalCrabPlugin {
            checkpoint_dirs,
            crab_spawns,
        });

    // Park the wall-clock FixedUpdate auto-pump; `drive_lockstep` pumps the body at the
    // deterministic [`PhysicsCadence`] instead (see [`park_fixed_auto_pump`]).
    park_fixed_auto_pump(app.world_mut());

    // The visible crab is the skin (or the silhouette `spawn_world` leaves shown when no model
    // loads), rendered at TRUE physics size ([`crate::render::world_render_scale`]). The
    // render-mode cycle (`super::render_mode`, the shared `crab_world::crab_view` cage) draws
    // the crab's live colliders translated to the same render spot, so the cage sits exactly
    // ON the mesh.
}

/// Add the rapier-NN crab stack AND arm the gate in one call — the known-armed-at-build pairing
/// every scripted/screenshot solo+net path funnels through (the windowed `Boot::Round`, the solo
/// and networked screenshot builders). Bundling the two so no site can install the stack and forget
/// to arm Sally (or arm with no stack behind it). The boot-MENU path is the deliberate exception: it
/// adds the stack UNarmed at build (gate off, `NumEnvs 0`) and arms only once the round resolves
/// armable, so it calls [`add_external_nn_crab`] directly.
pub(super) fn install_armed_nn_crab(
    app: &mut App,
    checkpoint_dirs: Vec<std::path::PathBuf>,
    crab_spawns: Vec<Pos>,
) {
    add_external_nn_crab(app, checkpoint_dirs, crab_spawns);
    crate::external_crab::arm(app.world_mut());
}
