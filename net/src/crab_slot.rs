//! The host's crab slot (rl#298 stage 2): the seam between [`Server::advance`] and
//! [`Server::step_next`] where the tick's crab physics — including each crab's policy
//! forward (`run_crab_policy` in `BotSet::Think`) — runs in the host's one world, and
//! the resulting world poses are handed to `step_next`. An INTERNAL seam: only the
//! server-auth arm calls [`pump_crab_slot`] — the windowed driver and the headless
//! server harness below go through this one function, so a host with and without a
//! renderer cannot drift. Remote-adopt clients never enter it: their `FixedUpdate` is
//! parked ([`park_fixed_auto_pump`]) and nothing pumps it, so the policy is host-side
//! by construction; clients only consume the resulting `CoreSnapshot` +
//! `CrabArticulation` streams.
//!
//! The obs half of the seam (rl#298 stage 3): each pumped step builds the policy's
//! observation from THIS world's ECS (`build_observation` in `BotSet::Sense`), and the
//! in-game action period is one physics step — training's control period, by shared
//! construction: the trainer's `brain_step` and the host's `run_crab_policy` both run
//! in `BotSet::Think` of the same `FixedUpdate` schedule, pumped once per
//! `PHYSICS_DT` (`headless_stack`'s manual time drive there, [`pump_fixed_steps`]
//! here). The stage-3 tests below pin both halves.
//!
//! [`Server::advance`]: crate::server::Server::advance
//! [`Server::step_next`]: crate::server::Server::step_next

use bevy::prelude::*;

use crate::external_crab::ExternalCrabBridge;
use crate::server::CrabPose;
use crate::sim::{Pos, Sim};

/// The caller's half of the slot contract, computed from the authoritative sim BEFORE
/// the pump: the tick being stepped INTO, and each crab's hunt target. One
/// implementation for every caller of [`pump_crab_slot`], so the pre-read (which tick,
/// which hunt source) cannot drift between the windowed driver and a renderless host.
// The seam compiles render-free (rl#298 stage 4) but its only production caller today
// is the windowed driver; the dead_code allowances fall away when a renderless
// consumer (the trainer's server-world env) arrives.
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(crate) fn slot_inputs(sim: &Sim) -> (u64, Vec<Option<Pos>>) {
    let hunt = (0..sim.crabs().len())
        .map(|idx| sim.nearest_living_player_pos(idx))
        .collect();
    (sim.tick() + 1, hunt)
}

/// Run the crab slot for the tick being stepped INTO: pump the owed fixed steps
/// (sensing, policy forward, actuation, physics, integration), collect the crabs'
/// world poses for [`Server::step_next`](crate::server::Server::step_next), and feed
/// the NEXT tick's hunt targets.
///
/// `hunt` is per sim crab, computed by the caller from the authoritative sim BEFORE
/// the pump — the pump never mutates the sim, so the values equal a post-pump read.
/// Feeding them after the pump is deliberate: this tick's pump walks on LAST tick's
/// targets, exactly the ordering the pre-extraction driver had.
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(crate) fn pump_crab_slot(
    world: &mut World,
    stepping_into: u64,
    hunt: &[Option<Pos>],
) -> Vec<CrabPose> {
    pump_fixed_steps(world, crate::cadence::steps_for_tick(stepping_into));
    world.resource_scope(|_, mut bridge: Mut<ExternalCrabBridge>| {
        assert_eq!(
            hunt.len(),
            bridge.crab_count(),
            "one hunt target per bridged crab — the sim's crab count must match the bridge"
        );
        let poses = bridge.crab_poses();
        for (idx, &prey) in hunt.iter().enumerate() {
            bridge.set_hunt_target(idx, prey);
        }
        poses
    })
}

/// Manually drive `steps` fixed-schedule passes — the host's physics pump. The
/// wall-clock auto-pump is parked ([`park_fixed_auto_pump`]), so these calls are the
/// ONLY thing that advances `FixedMain`; the per-tick count comes from
/// [`crate::cadence::steps_for_tick`], the one source for the 64:30 staircase.
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(crate) fn pump_fixed_steps(world: &mut World, steps: u32) {
    use bevy::app::FixedMain;
    use bevy::time::{Fixed, Time, Virtual};

    if steps == 0 {
        return;
    }
    for _ in 0..steps {
        let fixed = world.resource::<Time<Fixed>>().as_generic();
        *world.resource_mut::<Time>() = fixed;
        world.run_schedule(FixedMain);
    }
    let virt = world.resource::<Time<Virtual>>().as_generic();
    *world.resource_mut::<Time>() = virt;
}

/// Park the wall-clock fixed-timestep auto-pump (to an 86400 s timestep, so "never"
/// really means "not within a day's uptime"): every peer's `FixedUpdate` then advances
/// only through [`pump_fixed_steps`], which only the server-auth arm calls.
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(crate) fn park_fixed_auto_pump(world: &mut World) {
    world
        .resource_mut::<bevy::time::Time<bevy::time::Fixed>>()
        .set_timestep(std::time::Duration::from_secs(86_400));
}

#[cfg(test)]
mod tests {
    use bevy_rapier3d::prelude::Velocity;
    use crab_world::Visuals;
    use crab_world::bot::actuator::{ACTION_SIZE, CrabActions};
    use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint};
    use crab_world::bot::headless::{
        HeadlessStack, WorldRole, force_serial_schedules, headless_stack, pin_single_thread_pools,
    };
    use crab_world::bot::physics_digest::crab_state_digest;
    use crab_world::bot::sensor::CrabObservation;
    use crab_world::policy::{Policy, RestFallback};

    use super::*;
    use crate::client::TickMsg;
    use crate::external_crab::{CrabPolicies, ExternalCrabPlugin, arm};
    use crate::server::Server;
    use crate::sim::{Input, PlayerId, Sim};

    #[test]
    fn manual_pump_matches_auto_pump_step_for_step() {
        let build = || {
            headless_stack(HeadlessStack {
                num_envs: 1,
                role: WorldRole::Standalone,
                // Models the GCR client's world, so it steps the client's ground — the
                // canonical terrain bake (rl#209, rl#293).
                grid: crab_world::terrain::TerrainGrid::gcr(),
                visuals: crab_world::Visuals(false),
            })
        };
        let mut auto = build();
        let mut manual = build();
        auto.update();
        manual.update();
        park_fixed_auto_pump(manual.world_mut());

        let digest = |app: &mut App| -> u64 {
            let mut q = app.world_mut().query_filtered::<(
                &Transform,
                &Velocity,
                Option<&CrabJoint>,
                Option<&CrabCarapace>,
            ), With<CrabBodyPart>>();
            crab_state_digest(q.iter(app.world()))
        };
        let set_torque = |app: &mut App, a: [f32; ACTION_SIZE]| {
            assert!(app.world_mut().resource_mut::<CrabActions>().set_row(0, a));
        };

        let mut lcg: u64 = 0x1234_5678_9abc_def0;
        for t in 0..120u32 {
            let mut act = [0.0f32; ACTION_SIZE];
            for slot in act.iter_mut() {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *slot = ((lcg >> 40) as u32 as f32 / (1u32 << 24) as f32) * 1.6 - 0.8;
            }
            set_torque(&mut auto, act);
            set_torque(&mut manual, act);
            auto.update();
            pump_fixed_steps(manual.world_mut(), 1);
            assert_eq!(
                digest(&mut auto),
                digest(&mut manual),
                "manual pump diverged from auto-pump at tick {t}"
            );
        }
    }

    /// A headless host in the LIVE configuration: the crab stack armed in one world,
    /// the wall-clock auto-pump parked from birth (like the windowed app), so physics
    /// advances only through the slot — no renderer anywhere. The flat grid keeps the
    /// motion assertions attributable to the drive, not to sliding down the GCR
    /// origin slope.
    fn headless_host_app(policy: Policy, crab_spawn: Pos) -> App {
        pin_single_thread_pools();
        let mut app = headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            grid: std::sync::Arc::new(crab_world::terrain::TerrainGrid::flat(512.0)),
            visuals: Visuals(false),
        });
        app.add_plugins(ExternalCrabPlugin::new(vec![policy], vec![crab_spawn]));
        arm(app.world_mut());
        force_serial_schedules(&mut app);
        park_fixed_auto_pump(app.world_mut());
        app
    }

    /// One authoritative server tick through the REAL host path: `advance` → the crab
    /// slot → `step_next`. `app.update()` first, as on the live host, where the
    /// `Update` schedule (crab spawn, bookkeeping) wraps the slot every frame.
    fn server_tick(
        app: &mut App,
        server: &mut Server,
        issue_tick: u64,
    ) -> crate::snapshot::CoreSnapshot {
        app.update();
        server.advance(TickMsg {
            issue_tick,
            input: Input::from_axes(0.0, 0.0),
            pilot: None,
        });
        let mut last = None;
        while server.next_tick_ready() {
            let (stepping_into, hunt) = slot_inputs(server.sim());
            let poses = pump_crab_slot(app.world_mut(), stepping_into, &hunt);
            let stepped = server.step_next(&poses, Default::default());
            last = Some(
                crate::snapshot::CoreSnapshot::from_bytes(&stepped.snapshot)
                    .expect("the authoritative server's snapshot must decode"),
            );
        }
        last.expect("one assembled tick per advance on the solo host")
    }

    fn solo_server() -> Server {
        let me = PlayerId(0);
        Server::new(me, &[me], Sim::new(0x5A11, &[me]))
    }

    /// A solo authoritative server plus the headless host world armed with `policy`,
    /// crab at the sim's own spawn — the harness every slot test drives.
    fn solo_host(policy: Policy) -> (Server, App) {
        let server = solo_server();
        let spawn = server.sim().crabs()[0].pos();
        let app = headless_host_app(policy, spawn);
        (server, app)
    }

    /// rl#298 stage 2, link 1: the policy FORWARD runs inside the server's crab slot.
    /// After a slot pump, the actions driving the in-world body's motors are exactly
    /// the loaded brain's forward pass over the obs the sensor built that same pump —
    /// pinned by recomputing `act` on the observed row, so a Rest shortcut or a
    /// stale/overwritten row cannot pass.
    #[test]
    fn host_slot_runs_the_policy_forward_between_advance_and_step_next() {
        // The enveloped golden brain (crab-world's format-drift fixture): a real
        // checkpoint the loader arms, so `act` runs a real NN forward.
        let golden = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crab-world/tests/data/golden-mlp512x3-env");
        let policy = Policy::load(&golden, RestFallback::Rest);
        assert!(
            policy.is_loaded(),
            "the golden checkpoint must load — a Rest fallback would make this test vacuous"
        );

        let (mut server, mut app) = solo_host(policy);

        // Past spawn + settle grace (32 physics ticks), into live sensing/acting.
        for t in 0..60u64 {
            server_tick(&mut app, &mut server, t);
        }

        let obs = app.world().resource::<CrabObservation>().rows()[0];
        assert!(
            obs.iter().any(|v| *v != 0.0),
            "the sensor must have built a LIVE obs this pump — on the defaulted all-zero \
             row the forward-pass pin below would hold vacuously"
        );
        let expected = app.world().non_send_resource::<CrabPolicies>().0[0].act(&obs);
        let got = app.world().resource::<CrabActions>().rows()[0];
        assert_eq!(
            got, expected,
            "the motors' action row must be the policy forward over this pump's obs"
        );
    }

    /// rl#298 stage 2, link 2: motion the slot pumps into the in-world body lands in
    /// the authoritative snapshot clients decode. A full-scale square wave (the rl#224
    /// flail, driven AFTER the policy so the Rest brain doesn't zero it) shuffles the
    /// crab; the moved position must come back out of `step_next`'s `CoreSnapshot`.
    #[test]
    fn slot_pumped_crab_motion_reaches_the_snapshot_clients_decode() {
        #[derive(Resource, Default)]
        struct Wave(f32);
        fn drive_wave(w: Res<Wave>, mut actions: ResMut<CrabActions>) {
            // Every-tick system: pre-spawn skips are fine, later ticks land the flail.
            let _ = actions.fill(0, w.0);
        }

        let (mut server, mut app) = solo_host(Policy::rest());
        let spawn = server.sim().crabs()[0].pos();
        app.init_resource::<Wave>();
        // After Think (the policy's set), before Act (apply_actions): the wave is the
        // last CrabActions writer of the pump, same guarantee as ordering directly
        // against the private run_crab_policy.
        app.add_systems(
            FixedUpdate,
            drive_wave
                .after(crab_world::bot::BotSet::Think)
                .before(crab_world::bot::BotSet::Act),
        );

        let mut tick = 0u64;
        let mut snap = None;
        for _ in 0..100u64 {
            snap = Some(server_tick(&mut app, &mut server, tick));
            tick += 1;
        }
        for t in 0..300u64 {
            app.world_mut().resource_mut::<Wave>().0 = if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
            snap = Some(server_tick(&mut app, &mut server, tick));
            tick += 1;
        }

        let snap = snap.expect("ticks ran");
        let end = snap.crabs[0].pos();
        let (dx, dz) = Pos {
            x: end.x - spawn.x,
            z: end.z - spawn.z,
        }
        .to_meters();
        let moved = (dx * dx + dz * dz).sqrt();
        assert!(
            moved > 0.5,
            "the flailing in-world crab must have moved in the snapshot clients decode \
             (moved {moved:.3} m)"
        );
    }

    /// rl#298 stage 3, obs half: the row the policy consumes IS the one world's
    /// same-tick state. A snapshot captured between Sense and Think pins the body
    /// channel against the live carapace and the target channel against the posed
    /// lure — and the lure itself is re-derived from the SIM's own prey position
    /// through the one arena↔world correspondence (sim = arena + `ArenaAnchor`),
    /// independently of the bridge's integrated position. The crab is kept in motion
    /// (the rl#224 flail) so the body moves between passes: a row stale by even one
    /// step exceeds the body tolerance, and a defaulted or wrong-frame row misses by
    /// meters. Also pins the target conventions stage 4's trainer swap must
    /// reproduce (band lure along the true bearing, surface-relative lure height).
    #[test]
    fn obs_seam_reads_the_one_worlds_state_with_the_sims_prey_as_target() {
        #[derive(Resource, Default)]
        struct SenseSnap {
            carapace: Option<(Vec3, Quat)>,
            prev_carapace: Option<Vec3>,
            target: Option<Vec3>,
            origin: Vec3,
        }
        fn snap_sense(
            mut snap: ResMut<SenseSnap>,
            spawns: Res<crab_world::bot::CrabSpawns>,
            targets: Res<crab_world::bot::sensor::CrabTargets>,
            carapace_q: Query<&Transform, With<CrabCarapace>>,
        ) {
            if spawns.is_empty() {
                return; // pre-spawn pass — origins not laid yet
            }
            let Some(t) = carapace_q.iter().next() else {
                return;
            };
            snap.prev_carapace = snap.carapace.map(|(p, _)| p);
            snap.carapace = Some((t.translation, t.rotation));
            snap.target = targets.get(0);
            snap.origin = spawns.origin(0);
        }
        #[derive(Resource, Default)]
        struct Wave(f32);
        fn drive_wave(w: Res<Wave>, mut actions: ResMut<CrabActions>) {
            let _ = actions.fill(0, w.0);
        }

        let (mut server, mut app) = solo_host(Policy::rest());
        app.init_resource::<SenseSnap>();
        app.init_resource::<Wave>();
        app.add_systems(
            FixedUpdate,
            (
                snap_sense
                    .after(crab_world::bot::BotSet::Sense)
                    .before(crab_world::bot::BotSet::Think),
                drive_wave
                    .after(crab_world::bot::BotSet::Think)
                    .before(crab_world::bot::BotSet::Act),
            ),
        );

        // Past spawn + settle grace, into live sensing with a fed hunt target. The
        // flail starts only post-settle: pre-settle motion is never integrated into
        // her sim position (spawn-anchored until settle ends), so flailing there
        // would widen the settle slide the seam assertion's slop must absorb.
        for t in 0..60u64 {
            if t >= 20 {
                app.world_mut().resource_mut::<Wave>().0 =
                    if (t / 5) % 2 == 0 { 1.0 } else { -1.0 };
            }
            server_tick(&mut app, &mut server, t);
        }

        let (pos, rot, prev, target, origin) = {
            let snap = app.world().resource::<SenseSnap>();
            let (pos, rot) = snap.carapace.expect("carapace snapshotted post-settle");
            let target = snap
                .target
                .expect("the sim's living player must have posed a hunt target");
            let prev = snap.prev_carapace.expect("two sensed passes ran");
            (pos, rot, prev, target, snap.origin)
        };
        // The flailing body really moves between passes — what gives the body pin
        // below its anti-staleness teeth (inter-pass motion ≫ its tolerance).
        assert!(
            (pos - prev).length() > 1e-4,
            "the flail must move the carapace between sensed passes \
             (moved {:.6} m)",
            (pos - prev).length()
        );
        let obs = app.world().resource::<CrabObservation>();
        let view = obs.env(0).expect("env 0 sized");

        let body = view.body_pos();
        let expected_body = pos - origin;
        assert!(
            (body - expected_body).length() < 1e-5,
            "obs body.pos {body:?} must be the live carapace spawn-relative \
             ({expected_body:?})"
        );

        let target_local = view.target_local();
        let expected_local = rot.inverse() * (target - pos);
        assert!(
            (target_local - expected_local).length() < 1e-4,
            "obs target_local {target_local:?} must be the posed target in the live \
             body frame ({expected_local:?})"
        );

        // The seam crossing: the posed target re-derives from the sim's prey. Her
        // sim-frame position is carapace + anchor, up to the settle-grace slide the
        // bridge deliberately never integrates (spawn-anchored until settle ends,
        // ~cm) — hence the 0.1 m slop, an order under the ~5 m prey range and any
        // frame mix-up (meters).
        let prey = server
            .sim()
            .nearest_living_player_pos(0)
            .expect("player 0 is alive and stationary");
        let (px, pz) = prey.to_meters();
        let anchor = app
            .world()
            .resource::<crate::external_crab::ArenaAnchor>()
            .0;
        let to_prey = bevy::math::Vec2::new(px - (pos.x + anchor.x), pz - (pos.z + anchor.y));
        let lure = crab_world::training::targets::band_lure(pos, to_prey, 0.0);
        let terrain = app.world().resource::<crab_world::terrain::Terrain>();
        let expected_target = terrain.place(
            bevy::math::Vec2::new(lure.x, lure.z),
            crate::external_crab::CLAW_TARGET_Y,
        );
        assert!(
            (target - expected_target).length() < 0.1,
            "the posed target {target:?} must re-derive from the sim's prey through \
             the one arena↔world correspondence ({expected_target:?})"
        );
    }

    /// rl#298 stage 3, control-rate half: equality with training's control period is
    /// by shared construction (module doc); this pins the host half — across a real
    /// server run, every physics step the 64:30 cadence owes writes the crab's
    /// actions exactly once (no decimation, no extra passes), so the action period
    /// is one physics step (`PHYSICS_DT`), training's control period. Counted by
    /// change-detection on `CrabActions` in the window where `run_crab_policy` is
    /// its only writer — a disarmed or decimated policy system fails here where a
    /// bare pass counter would stay green. If the in-game step rate ever exceeds
    /// the control rate, action repeat must restore this equality (the epic's
    /// parenthetical).
    #[test]
    fn in_game_action_period_equals_trainings_control_period() {
        #[derive(Resource, Default)]
        struct ActionWrites(u64);
        fn count_action_writes(actions: Res<CrabActions>, mut writes: ResMut<ActionWrites>) {
            writes.0 += u64::from(actions.is_changed());
        }

        let (mut server, mut app) = solo_host(Policy::rest());
        app.init_resource::<ActionWrites>();
        app.add_systems(
            FixedUpdate,
            count_action_writes
                .after(crab_world::bot::BotSet::Think)
                .before(crab_world::bot::BotSet::Act),
        );

        for t in 0..90u64 {
            server_tick(&mut app, &mut server, t);
        }

        let ticks = server.sim().tick();
        assert_eq!(
            ticks, 90,
            "one authoritative tick per advance on the solo host"
        );
        let writes = app.world().resource::<ActionWrites>().0;
        assert_eq!(
            writes,
            crate::cadence::cumulative_steps(ticks),
            "every pumped physics step must write the crab's actions exactly once — \
             the action period is one physics step (PHYSICS_DT), training's control \
             period"
        );
    }
}
