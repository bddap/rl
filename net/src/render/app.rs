//! Windowed first-person client: app assembly + boot wiring.
//!
//! Builds the windowed `App` ([`build_windowed_app`]), defines the boot enum/phase
//! ([`Boot`]/[`AppPhase`]), and installs the real rapier-NN crab stack. The headless
//! screenshot builder lives in [`super::screenshot`]; the per-frame systems it wires
//! (`gather_input`, `drive_lockstep`, `apply_transforms`, `update_hud`) live in the
//! sibling submodules.

use super::*;
use super::driver::{
    PendingRound, drive_lockstep, ensure_round_installed, insert_core,
    park_fixed_auto_pump,
};
use super::hud::{spawn_hud, sync_controls_context, update_hud};
use super::input::{gather_input, grab_cursor_once, quit_game};
use super::scene::{apply_transforms, spawn_fp_camera, spawn_world};


/// How the windowed client starts up: at the boot MENU (the interactive default — the
/// player picks Host/Join, rl#58), or straight into a prebuilt ROUND (the scripted
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

/// The windowed client's top-level phase (rl#56). The menu and lobby screens are PURE
/// client UI — no [`Lockstep`]/[`Sim`] exists until [`AppPhase::Playing`], which is entered
/// only after a choice (and, for networked roles, a host-commanded start). This is the
/// firewall that keeps the menu off the deterministic sim: the FP systems and the sim
/// resource are all gated to `Playing`, so menu state literally cannot reach the round
/// (it's built fresh on the transition from the unchanged formation machinery).
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppPhase {
    /// The boot menu: choose Host / Join (rl#58). egui only.
    #[default]
    Menu,
    /// A host-triggered lobby is forming on a background thread; show the live roster +
    /// (Host) join code + Start, and poll for the result. A Host-alone Start skips straight
    /// to its instant solo round without lingering here.
    Connecting,
    /// The round is live: the FP client runs exactly as before rl#56.
    Playing,
}

/// Build the windowed first-person client app. Starts at the boot menu or straight in a
/// round per [`Boot`]; owns the `Lockstep` + optional `NetDriver` as resources once
/// playing. Built here, not via a `Plugin` that holds the sim, because
/// `Plugin::build(&self)` can't move a non-`Clone` `Lockstep`/`NetDriver` out of
/// itself — inserting them as resources at the `Playing` transition is the clean path.
///
/// `external_crab` is the REQUIRED trained checkpoint dir for the one giant crab — the REAL
/// rapier-simulated NN body ("real Sally", [`crate::external_crab`]). There is NO integer
/// fallback (rl#114): a SOLO round always arms it; a NETWORKED round arms it once peers agree on a
/// shared brain (the weights-digest handshake) and step it at the deterministic cadence (the GCR
/// fold, rl#82). A networked-UNSYNCED round CANNOT arm Sally and FAILS LOUD — a clear peer-mismatch
/// error naming the fix (run rl-update on every device) rather than silently substituting a fake
/// crab — the whole point of rl#114. The failure is GRACEFUL (rl#115): the scripted `Boot::Round`
/// path returns it as an `Err` (clean CLI exit, no panic); the interactive menu pre-gates it in
/// `poll_formation` and returns to the chooser showing the message (no mid-transition crash).
/// Pin every process-global task pool to a single worker, so the armed float rapier-NN crab
/// ("real Sally") evolves bit-identically across peers (GCR#113).
///
/// MUST be called before `App::new` and before any sim step / matmul, because the pools latch
/// their thread count on first use; see [`crab_world::bot::headless::pin_single_thread_pools`] for the
/// env-var/`OnceLock` mechanism. Same recipe the trainer and the #82 cross-peer probe run; exposed
/// as one thin `pub` entry so `game` (a separate crate) can call it without widening the
/// crate-internal helper.
///
/// The windowed client does NOT call this directly: [`build_windowed_app`] pins (this plus the
/// matching `force_serial_schedules` run-order pin) ONLY for a round with a remote peer, so the
/// common solo round runs multi-threaded (~60fps). This entry is for the headless screenshot path
/// (`game fp-screenshot`), which pins when its single armed-crab frame needs a stable solver.
pub fn pin_process_pools() {
    crab_world::bot::headless::pin_single_thread_pools();
}

/// `wire` is the debug-wireframe overlay's starting mode (off in normal play; the player
/// cycles it live with F3, or boots into a mode via `--debug-wireframe`/`RL_DEBUG_WIREFRAME`).
pub fn build_windowed_app(
    boot: Boot,
    external_crab: std::path::PathBuf,
    wire: super::WireMode,
) -> anyhow::Result<App> {
    // Pin the global task pools to ONE thread only for a round that actually has a remote peer.
    // The pin's sole purpose is bit-identical cross-peer float evolution (GCR#113), and it costs
    // ~30-60× frame time by serialising rapier's solver and Bevy's parallel render-prep onto a
    // single core. A SOLO round (a scripted `--host`/`--join` that found no peer → `None` driver)
    // has nobody to stay in sync with, so it skips the pin and runs multi-threaded (~60fps); a
    // NETWORKED round pins, exactly as before. The MENU boot builds its `App` — which latches the
    // pools — BEFORE the player picks Host/Join, so peer presence isn't known yet, and it pins
    // defensively so a round that later goes networked stays deterministic. This is the one path
    // parameterised by peer count, not an SP/MP mode branch (single-player IS multiplayer with
    // zero remote players, rl).
    //
    // MUST be decided here, ahead of `App::new()` latching Bevy's three task pools (first-writer
    // wins); the rayon/matmul env vars the pin sets are read lazily on the first sim step, also
    // after this point. See [`pin_process_pools`] / [`crab_world::bot::headless::pin_single_thread_pools`].
    let networked = match &boot {
        Boot::Round(round) => round.1.is_some(),
        Boot::Menu { .. } => true,
    };
    // The determinism pin has TWO halves gated on this one `networked` bool: the task-pool pin
    // here (MUST precede `App::new()` — Bevy's pools latch first-writer-wins) and the matching
    // `force_serial_schedules` near the end (MUST follow system wiring). They can't be one call
    // because of that ordering, so they're kept coupled by sharing this single condition — change
    // one gate and you must change the other, or networked peers desync.
    if networked {
        crab_world::bot::headless::pin_single_thread_pools();
        // Canary for the pin's one footgun: an external RAYON_NUM_THREADS override to anything but
        // "1" silently breaks cross-peer determinism. Only meaningful once we've pinned.
        debug_assert_eq!(
            std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
            Some("1"),
            "networked round must pin RAYON_NUM_THREADS=1 (and no external override)"
        );
    }
    let mut app = App::new();
    // Point bevy at the bundled `assets/` dir explicitly, so a fresh clone's `cargo run`
    // finds the committed control glyphs regardless of cwd or which workspace bin runs (the
    // default root is the running bin's crate dir, `game/assets`, which has no glyphs).
    // `BEVY_ASSET_ROOT` still overrides (deploy). See `crab_world::assets`.
    app.add_plugins(DefaultPlugins.set(AssetPlugin {
        file_path: crab_world::assets::bevy_asset_path().to_string_lossy().into_owned(),
        ..default()
    }).set(WindowPlugin {
        primary_window: Some(Window {
            title: "Giant Crab Rescue".into(),
            // Fullscreen is the single source of truth for every GCR launch target. The Deck
            // shows fullscreen only because gamescope forces it; on a plain desktop/TV (bothouse)
            // a Windowed app stayed windowed. BorderlessFullscreen makes the app itself own the
            // policy, so bothouse matches the Deck with no separate per-host window-config path.
            mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
            ..default()
        }),
        ..default()
    }));
    // Night-sky skybox behind the first-person view (attaches to the FP camera when the
    // round spawns it). Shared with the rl-demo via `crab_world::sky`.
    app.add_plugins(crab_world::sky::NightSkyPlugin);
    app.init_state::<AppPhase>();

    // The FP round systems, gated to Playing. spawn_* moved off Startup to the Playing
    // transition (the sim doesn't exist until then); the per-frame systems run only while
    // playing so they never touch a not-yet-built GameState. The set is IDENTICAL to the
    // pre-rl#56 wiring — only the schedule gating is new, so the round itself is unchanged.
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
                crab_world::build_info::spawn_build_info_overlay,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (gather_input, drive_lockstep, apply_transforms, update_hud)
                .chain()
                .run_if(in_state(AppPhase::Playing)),
        )
        .add_systems(
            Update,
            (
                grab_cursor_once,
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

    // OUR policy-weights digest (rl#82, GCR), advertised in networked formation so peers can
    // agree on a shared brain (see [`crate::may_arm_external_crab`]). `0` for no checkpoint.
    // MUST equal the digest the per-tick bridge folds into the lockstep hash — both come from
    // [`crab_world::play::checkpoint_digest`] (here from the path; on the bridge via
    // `Policy::weights_digest()`), so the cadence-fold follow-up that arms the crab must source
    // its folded digest from the SAME checkpoint, or a hot-reload could split the two.
    let weights_digest = crab_world::play::checkpoint_digest(&external_crab);
    // OUR crab-MODEL-asset digest (rl#100, GCR): the giant crab's rapier colliders are derived
    // from this asset ([`crab_world::bot::meshfit::crab_asset_digest`]), so peers must agree on it
    // before arming the float crab in lockstep — a different model builds different colliders
    // and desyncs even with identical brains. Computed unconditionally (it's a property of this
    // peer's installed crab model, independent of whether a checkpoint loaded); `0` for no model.
    let asset_digest = crab_world::bot::meshfit::crab_asset_digest();

    match boot {
        // Scripted boot: insert the round now and jump straight to Playing (the menu
        // states are never entered). NextState applied before the first frame, so
        // OnEnter(Playing) fires and the world spawns on frame one — no menu flash. The
        // scripted `--host`/`--join` path tests/scripts use; a host-alone `--host` that
        // found no peer is a solo round here, so it gets the real NN crab.
        Boot::Round(round) => {
            let (mut ls, net) = *round;
            // `networked` (== `net.is_some()`) was decided at the top of the fn to drive the pin.
            // The one giant crab is the real NN body (rl#114) — arm it now. A SOLO round always
            // may; a NETWORKED round may only with synced weights+assets
            // ([`crate::may_arm_external_crab`], the determinism guard). With no integer
            // fallback, a networked round that CAN'T arm FAILS LOUD here rather than playing a fake
            // crab. Capture the crab's spawn + seed the pose BEFORE `ls` moves into core.
            // A networked round that can't agree on the brain+colliders can't arm Sally. With no
            // integer fallback (rl#114) it REFUSES rather than play a fake crab — but as a clean
            // error bubbled to the CLI (rl#115), not a `panic!` process-abort. The interactive menu
            // never reaches here (it pre-gates in `poll_formation`); this is the scripted
            // `--host`/`--join` path, whose graceful failure is a non-zero exit with the message.
            if let Some(msg) = crab_arm_failure(&net) {
                anyhow::bail!(msg);
            }
            let spawn = seed_external_crab_solo(&mut ls);
            let source = super::driver::InputSource::coordinated(net, ls.peers());
            insert_core(&mut app, ls, source);
            // Known-armed at build: add the stack AND arm the gate now, so the crab spawns frame
            // one.
            add_external_nn_crab(&mut app, external_crab, spawn);
            // Arm the gate (and, networked, pin the lead so a per-peer env override can't desync
            // the hashed pose — solo keeps its tuning). One arm path, [`crate::external_crab::arm`].
            crate::external_crab::arm(app.world_mut(), networked);
            app.world_mut()
                .resource_mut::<NextState<AppPhase>>()
                .set(AppPhase::Playing);
        }
        // Interactive boot: add the menu plugin (egui menu + lobby poll). The sim is built
        // later, at the Playing transition, from the choice the menu records.
        Boot::Menu { seed, telemetry } => {
            app.add_plugins(menu::MenuPlugin {
                seed,
                telemetry,
                weights_digest,
                asset_digest,
            });
            // NN crab on the round (rl#58 + GCR): the menu can't know at BUILD time whether the
            // round will be solo, networked-synced, or networked-unsynced, so add the whole NN
            // stack now with the gate OFF and no crab spawned (NumEnvs 0), and arm it only if
            // the resolved round may ([`ensure_round_installed`] → [`crate::may_arm_external_crab`]:
            // solo always, networked only with synced weights). The crab's arena spawn is a pure
            // function of the seed (a throwaway solo lockstep reads it), so it's known here
            // without the round existing yet. The checkpoint is REQUIRED (rl#114), so the stack is
            // always installed; a networked-UNSYNCED round leaves the gate off and
            // `ensure_round_installed` FAILS LOUD rather than substituting a fake crab.
            {
                let dir = external_crab;
                let crab_spawn = crate::net_loop::solo_lockstep_for(seed)
                    .sim()
                    .crab()
                    .pos();
                add_external_nn_crab(&mut app, dir, crab_spawn);
                // Gate OFF: leave `ExternalCrabArmed` ABSENT (presence is the state). The
                // transition (`ensure_round_installed`) inserts it iff the round resolves armable.
                app.insert_resource(crab_world::bot::NumEnvs(0)); // no crab spawns behind the menu
                app.insert_resource(ExternalCrabStackInstalled); // the transition may activate it
            }
        }
    }

    // Pin ECS run order for a NETWORKED round: force every MAIN-world schedule onto the
    // single-threaded executor so systems never dispatch onto the global ComputeTaskPool, where
    // thread scheduling would reorder the float evolution that drives the armed NN crab and
    // desync peers (GCR#113, the same pin the trainer and #82 probe apply). The SECOND half of the
    // determinism pin — gated on the SAME `networked` bool as the task-pool pin near the top of the
    // fn (the two MUST stay coupled; see that block): a solo round keeps the parallel executor
    // (~60fps) — with no peer there is no cross-peer order to preserve. Must run AFTER all systems
    // are wired —
    // every plugin/`add_systems` above is in, the schedules now exist. This touches only the main
    // world's schedules (the sim); bevy's render sub-app keeps its own executor.
    // The debug-wireframe overlay (off by default). Adds its systems + rapier's debug-render
    // plugin; MUST precede `force_serial_schedules` so that pin covers its systems too.
    super::debug_wireframe::register(&mut app, wire);

    if networked {
        crab_world::bot::headless::force_serial_schedules(&mut app);
    }

    Ok(app)
}

/// Presence marker: the boot-menu app installed the NN-crab stack at build (rl#58). The checkpoint
/// is REQUIRED (rl#114), so on the menu path this is always inserted; the Playing transition reads
/// its presence and arms the crab once the round resolves armable (solo always, networked only when
/// synced), FAILING LOUD otherwise — there is no integer crab to fall back to. A presence marker,
/// not a bool: "not installed" is simply the resource's absence (the scripted `Boot::Round` path),
/// so there's no degenerate `false` state to mishandle.
#[derive(Resource, Clone, Copy)]
pub(super) struct ExternalCrabStackInstalled;

/// The SINGLE arm-decision-plus-message for the one giant crab (the real NN body, "Sally"):
/// `None` if the round MAY arm it ([`crate::may_arm_external_crab`]: solo always, networked only
/// with synced weights+assets), or `Some(message)` describing exactly why a networked round CANNOT
/// — peers disagree on the brain or the crab colliders, so a float crab would desync lockstep — and
/// how to fix it. With no integer fallback (rl#114) an unarmable round REFUSES rather than silently
/// substituting a fake crab. ONE source for both the gate and the operator-facing text, used by the
/// menu pre-gate (return to the chooser showing this, no crash — rl#115) and the scripted
/// `Boot::Round` build (bubble it out as an error, no panic). Solo always arms, so the message only
/// ever describes a networked round.
pub(super) fn crab_arm_failure(net: &Option<NetDriver>) -> Option<String> {
    crab_arm_failure_from(
        net.is_none(),
        net.as_ref().is_some_and(NetDriver::weights_synced),
        net.as_ref().is_some_and(NetDriver::assets_synced),
    )
}

/// The pure core of [`crab_arm_failure`] — the arm decision + message from the three synced flags,
/// with no [`NetDriver`] so it's unit-testable headlessly (no tokio/iroh). `None` = armable.
pub(super) fn crab_arm_failure_from(
    net_is_none: bool,
    weights_synced: bool,
    assets_synced: bool,
) -> Option<String> {
    if crate::may_arm_external_crab(net_is_none, weights_synced, assets_synced) {
        return None;
    }
    // Reached only on a networked round that can't arm; weights are checked first, so a synced
    // weights flag here means weights agree but the collider asset doesn't.
    let cause = if !weights_synced {
        "the trained brain (brain.bin) differs or is missing on a peer"
    } else {
        "the crab colliders (the sally.glb model) differ on a peer"
    };
    Some(format!(
        "rl#114: refusing to start the round — can't arm the trained NN crab (\"Sally\") for this \
         multiplayer match because {cause}. Every device MUST carry the IDENTICAL brain + crab \
         model, or lockstep would desync. Fix: run rl-update on every device so all peers share \
         the same weights and colliders, then re-form the match. (There is deliberately no integer \
         stand-in crab — a mismatched table refuses rather than silently dropping Sally.)"
    ))
}

/// Seed the sim crab's spawn pose into `ls` so the rapier-NN body begins where the round placed
/// the giant crab, and return that spawn for [`add_external_nn_crab`]. The ONE seed both the
/// windowed solo `Boot::Round` client and the headless screenshot use, so the evidence shot arms
/// the SAME way the player's client does (the manual's "one implementation per thing"). Writes
/// back the crab's CURRENT pose, so sim state is unchanged; digest 0 to seed (the bridge's first
/// post-step `hash_crab_physics` fills the real digest before any cross-check). Solo only — a
/// networked round arms through the digest handshake in `ensure_round_installed`, not here.
pub(super) fn seed_external_crab_solo(ls: &mut Lockstep) -> Pos {
    let crab = ls.sim().crab();
    let spawn = crab.pos();
    ls.set_external_crab_pose(spawn, crab.yaw(), 0);
    spawn
}

/// Wire the real rapier-NN crab into the windowed solo app: the bot/physics/brain stack
/// (the SAME plugins `rl --demo` runs, so the crab steps the exact dynamics the policy
/// trained under) plus the [`external_crab::ExternalCrabPlugin`] bridge that walks it toward the
/// player and feeds its body position back into the sim. With the model present the cosmetic
/// skin rides the body; with no `sally.glb` the visible crab is the static giant silhouette
/// (`spawn_world` keeps it shown when no skin loads).
pub(super) fn add_external_nn_crab(app: &mut App, checkpoint_dir: std::path::PathBuf, crab_spawn: Pos) {
    app.insert_resource(crab_world::Visuals(true))
        .insert_resource(crab_world::bot::NumEnvs(1))
        // Same fixed timestep + softened contact spring as training/demo, with Rapier in
        // FixedUpdate (lockstep with the Sense→Think→Act brain loop) — bundled in the one
        // order that applies the spring, so the solo crab's physics can't drift from what
        // the policy optimised under (see physics::CrabPhysicsPlugin).
        .add_plugins(crab_world::physics::CrabPhysicsPlugin)
        .add_plugins(crab_world::physics::PhysicsWorldPlugin)
        .add_plugins(crab_world::bot::BotPlugin)
        .add_plugins(crate::external_crab::ExternalCrabPlugin {
            checkpoint_dir,
            crab_spawn,
        });

    // Park the wall-clock FixedUpdate auto-pump; `drive_lockstep` pumps the body at the
    // deterministic [`PhysicsCadence`] instead (see [`park_fixed_auto_pump`]).
    park_fixed_auto_pump(app.world_mut());

    // The visible crab is the skin (or the silhouette `spawn_world` leaves shown when no model
    // loads), rendered at TRUE physics size; the giant feel comes from the R-shrunk human world
    // ([`crate::render::world_render_scale`]). The debug-wireframe overlay (`super::debug_wireframe`)
    // draws the crab's live colliders translated to the same render spot, so the cage sits exactly
    // ON the mesh — render==physics, no scale hack.
}
