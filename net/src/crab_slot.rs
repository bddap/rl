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
//! [`Server::advance`]: crate::server::Server::advance
//! [`Server::step_next`]: crate::server::Server::step_next

use bevy::prelude::*;

use crate::external_crab::ExternalCrabBridge;
use crate::server::CrabPose;
use crate::sim::Pos;

/// Run the crab slot for the tick being stepped INTO: pump the owed fixed steps
/// (sensing, policy forward, actuation, physics, integration), collect the crabs'
/// world poses for [`Server::step_next`](crate::server::Server::step_next), and feed
/// the NEXT tick's hunt targets.
///
/// `hunt` is per sim crab, computed by the caller from the authoritative sim BEFORE
/// the pump — the pump never mutates the sim, so the values equal a post-pump read.
/// Feeding them after the pump is deliberate: this tick's pump walks on LAST tick's
/// targets, exactly the ordering the pre-extraction driver had.
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
            let (stepping_into, hunt) = {
                let sim = server.sim();
                let hunt: Vec<Option<Pos>> = (0..sim.crabs().len())
                    .map(|idx| sim.nearest_living_player_pos(idx))
                    .collect();
                (sim.tick() + 1, hunt)
            };
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

        let mut server = solo_server();
        let spawn = server.sim().crabs()[0].pos();
        let mut app = headless_host_app(policy, spawn);

        // Past spawn + settle grace (32 physics ticks), into live sensing/acting.
        for t in 0..60u64 {
            server_tick(&mut app, &mut server, t);
        }

        let obs = app.world().resource::<CrabObservation>().rows()[0];
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

        let mut server = solo_server();
        let spawn = server.sim().crabs()[0].pos();
        let mut app = headless_host_app(Policy::rest(), spawn);
        app.init_resource::<Wave>();
        app.add_systems(
            FixedUpdate,
            drive_wave
                .after(crate::external_crab::run_crab_policy)
                .in_set(crab_world::bot::BotSet::Think),
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
}
