//! Headless NN-crab probes: the host's crab slot pumping the one world (rl#298
//! stage 5) beside an authoritative [`Sim`], no renderer, sampling sim + world state
//! per tick. Consumed by `game nn-crab-probe` (behavior/
//! determinism A/B) and `game nn-crab-vehicle-stability` (the rl#137 ram test).
//!
//! The probe's sim is stepped 1:1 with the fixed schedule (one physics pass per sim
//! tick — the probe's historical cadence, kept so hash logs stay comparable), and the
//! claws are STRIPPED from the fed poses: the probe measures pursuit, and a downed
//! prey would end the very chase being measured.

use bevy::prelude::*;
use bevy_rapier3d::prelude::Velocity;

use crate::crab_slot::{self, NnCrabPlugin, arm, park_fixed_auto_pump, restart_crabs_to_spawns};
use crate::sim::{Externals, Input, PlayerId, Pos, Sim};
use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint};
use crab_world::bot::physics_digest::crab_state_digest;
use crab_world::bot::sensor::CrabTargets;

#[derive(Clone, Copy, Debug)]
pub struct ProbeSample {
    pub tick: u64,
    pub crab_x_m: f32,
    pub crab_z_m: f32,
    pub dist_to_prey_m: f32,
    pub state_hash: u64,
    pub carapace_x: f32,
    pub carapace_z: f32,
    pub carapace_y: f32,
    /// Carapace height above the local ground surface — the grounded-ness measure
    /// (absolute y is a mountainside elevation on terrain, meaningless as one).
    pub carapace_above_ground: f32,
    pub min_claw_to_target_m: f32,
    /// Every dynamic body in the world asleep this tick — rest converged to
    /// rapier sleep (bit-exact zeros), the rl#392 rest contract. Only meaningful
    /// for zero-drive probes on restable ground: a driven crab is force-woken
    /// every tick and a passive crab on steep terrain may slide indefinitely.
    pub crab_asleep: bool,
}

struct Probe {
    app: App,
    sim: Sim,
    samples: Vec<ProbeSample>,
    log_every: u64,
}

impl Probe {
    /// `deterministic` pins single-thread pools + serial schedules for the hash-log
    /// probes; the rl#396 step profile passes `false` — it measures wall time, so it
    /// must run the production threading, and pinning is a process-global one-way
    /// door the profile process never opens.
    fn new(
        policy: crab_world::policy::Policy,
        seed: u64,
        visuals: crab_world::Visuals,
        deterministic: bool,
        grid: std::sync::Arc<crab_world::terrain::TerrainGrid>,
    ) -> Self {
        use crab_world::bot::headless::{
            HeadlessStack, WorldRole, force_serial_schedules, headless_stack,
            pin_single_thread_pools,
        };

        if deterministic {
            pin_single_thread_pools();
        }

        let me = PlayerId(0);
        let sim = Sim::new(seed, &[me]);
        let spawns: Vec<Pos> = sim.crabs().iter().map(|c| c.pos()).collect();

        // The BARE crab stack, not `headless_server_world`: the stability probe
        // hand-spawns its ram craft, which the vehicle layer's `manage_vehicles`
        // would despawn as pilotless.
        let mut app = headless_stack(HeadlessStack {
            num_envs: spawns.len(),
            role: WorldRole::Standalone,
            grid,
            visuals,
        });
        app.add_plugins(NnCrabPlugin::new(vec![policy], spawns.clone()));
        arm(app.world_mut());
        park_fixed_auto_pump(&mut app);
        restart_crabs_to_spawns(app.world_mut(), &spawns);
        if deterministic {
            force_serial_schedules(&mut app);
        }

        Self {
            app,
            sim,
            samples: Vec::new(),
            log_every: 1,
        }
    }

    /// One probe tick through the host seam: `Update` wrap, one fixed pump (the
    /// probe's 1:1 cadence), poses off the world into the sim step, next tick's hunt
    /// fed back — the same pump→collect→feed seam the hosts run ([`pump_slot_steps`]).
    fn tick(&mut self) {
        let inputs = crab_slot::slot_inputs(&self.sim);
        self.app.update();
        let mut poses = crab_slot::pump_slot_steps(self.app.world_mut(), 1, &inputs);
        for p in &mut poses {
            // Pursuit probe: no downs (module doc) — the pose crosses, the claws don't.
            p.claws.clear();
        }
        let prey = self.sim.nearest_living_player_pos(0);

        let me = PlayerId(0);
        self.sim.step(
            &std::collections::BTreeMap::from([(me, Input::from_axes(0.0, 0.0))]),
            Externals::crabs_only(&poses),
        );

        let tick = self.sim.tick();
        if tick == 1 || tick.is_multiple_of(self.log_every) {
            self.sample(tick, prey);
        }
    }

    fn sample(&mut self, tick: u64, prey: Option<Pos>) {
        let world = self.app.world_mut();
        let terrain = world.resource::<crab_world::terrain::Terrain>().clone();

        let crab = self.sim.crabs()[0].pos();
        let (crab_x_m, crab_z_m) = crab.to_meters();
        let dist_to_prey_m = prey
            .map(|p| {
                // Integer delta first, then the one Pos→meters rule — bit-identical to
                // the old inline `/ UNIT` casts.
                let (dx, dz) = Pos {
                    x: p.x - crab.x,
                    z: p.z - crab.z,
                }
                .to_meters();
                (dx * dx + dz * dz).sqrt()
            })
            .unwrap_or(f32::NAN);

        // The probe's hash-log contract: the sim hash PLUS the full-body crab digest, so a
        // run-vs-run diff catches float divergence anywhere in the crab's rapier state. This
        // fold is probe-only — the runtime sim hash stopped carrying it in rl#223 (clients
        // adopt host state and never step, so a cross-peer digest compared host values with
        // themselves).
        let state_hash = self.sim.state_hash()
            ^ crab_state_digest(
                world
                    .query_filtered::<(
                        &Transform,
                        &Velocity,
                        Option<&CrabJoint>,
                        Option<&CrabCarapace>,
                    ), With<CrabBodyPart>>()
                    .iter(world),
            );

        let (carapace_x, carapace_y, carapace_z, carapace_above_ground) = world
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>()
            .iter(world)
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| {
                let p = t.translation;
                (p.x, p.y, p.z, p.y - terrain.height(p.x, p.z))
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0));

        let crab_asleep = {
            use bevy_rapier3d::plugin::context::RapierRigidBodySet;
            let set = world
                .query::<&RapierRigidBodySet>()
                .single(world)
                .expect("the probe world has exactly one rapier context");
            let mut any_dynamic = false;
            let all_asleep = set
                .bodies
                .iter()
                .filter(|(_, rb)| rb.is_dynamic())
                .all(|(_, rb)| {
                    any_dynamic = true;
                    rb.is_sleeping()
                });
            any_dynamic && all_asleep
        };

        let target = world.resource::<CrabTargets>().get(0);
        let min_claw_to_target_m = target
            .map(|target| {
                world
                    .query_filtered::<(&CrabEnvId, &Transform), With<CrabClawTip>>()
                    .iter(world)
                    .filter(|(env, tip)| env.0 == 0 && tip.translation.is_finite())
                    .map(|(_, tip)| tip.translation.distance(target))
                    .fold(f32::INFINITY, f32::min)
            })
            .unwrap_or(f32::NAN);

        self.samples.push(ProbeSample {
            tick,
            crab_x_m,
            crab_z_m,
            dist_to_prey_m,
            state_hash,
            carapace_x,
            carapace_z,
            carapace_y,
            carapace_above_ground,
            min_claw_to_target_m,
            crab_asleep,
        });
    }

    fn carapace(&mut self) -> Vec3 {
        let world = self.app.world_mut();
        world
            .query_filtered::<(&CrabEnvId, &Transform), With<CrabCarapace>>()
            .iter(world)
            .find(|(env, _)| env.0 == 0)
            .map(|(_, t)| t.translation)
            .unwrap_or(Vec3::ZERO)
    }
}

/// `visuals`: `Visuals(true)` steps the ARMED-RENDER configuration headless — the
/// skin and the rl#116 pose sentinel all live — which is the
/// exact configuration the GCR play-day crash showed no headless test covered. The
/// determinism/behavior probes pass `Visuals(false)`, matching what they hash.
///
/// `grid`: the hash-log probes step [`TerrainGrid::gcr`] — the host's canonical
/// ground (rl#209, rl#293) — so their logs stay comparable run-to-run. Tests whose
/// promise needs RESTABLE ground (a zero-drive crab settling to sleep) pass a flat
/// grid instead: on the canonical bake a passive crab may legitimately slide down
/// a mountainside forever (rl#406), which is terrain physics, not a probe signal.
pub fn run_headless_probe(
    policy: crab_world::policy::Policy,
    seed: u64,
    ticks: u64,
    log_every: u64,
    visuals: crab_world::Visuals,
    grid: std::sync::Arc<crab_world::terrain::TerrainGrid>,
) -> Vec<ProbeSample> {
    let mut probe = Probe::new(policy, seed, visuals, true, grid);
    probe.log_every = log_every.max(1);
    for _ in 0..ticks {
        probe.tick();
    }
    probe.samples
}

pub struct StabilityResult {
    pub samples: Vec<ProbeSample>,
    pub ram_tick: u64,
}

impl StabilityResult {
    pub fn carapace_stayed_finite(&self) -> bool {
        self.samples.iter().all(|s| {
            s.carapace_x.is_finite() && s.carapace_y.is_finite() && s.carapace_z.is_finite()
        })
    }
}

pub fn run_vehicle_stability_probe(
    policy: crab_world::policy::Policy,
    seed: u64,
    warmup: u64,
    post: u64,
) -> StabilityResult {
    use crab_world::vehicle::{VehicleKind, spawn_ram_vehicle};

    let mut probe = Probe::new(
        policy,
        seed,
        crab_world::Visuals(false),
        true,
        crab_world::terrain::TerrainGrid::gcr(),
    );
    for _ in 0..warmup {
        probe.tick();
    }
    let ram_tick = probe.sim.tick();

    let carapace = probe.carapace();
    let spawn_at = Transform::from_translation(carapace + Vec3::new(1.2, -0.15, 0.0));
    let ram_velocity = Velocity {
        linear: Vec3::new(-10.0, 0.0, 0.0),
        angular: Vec3::ZERO,
    };
    spawn_ram_vehicle(
        probe.app.world_mut(),
        VehicleKind::Plane,
        spawn_at,
        ram_velocity,
    );

    for _ in 0..post {
        probe.tick();
    }

    StabilityResult {
        samples: probe.samples,
        ram_tick,
    }
}

/// rl#332 flight-soak thresholds. Sally's carapace rides well under one stature
/// (~0.61 m) off the ground in any gait the policy has ever produced; the rl#137
/// vehicle-ram gate already treats 5 m as "launched skyward". Flight is called at
/// 1.5 m (~2.5 statures) SUSTAINED — a hop or a terrain lip crosses briefly, a
/// launch stays up — or a single tick of vertical speed no gait reaches.
pub const FLIGHT_ALT_M: f32 = 1.5;
/// Consecutive ticks above [`FLIGHT_ALT_M`] before an event fires (2 s at the
/// probe's 64 Hz 1:1 cadence). Long, deliberately: a downhill ballistic sail on
/// steep GCR terrain — gravity-legit — measured ~1 s airborne at up to 3.5 m
/// above the receding slope; a real launch stays up.
pub const FLIGHT_SUSTAIN_TICKS: u32 = 128;
/// UPWARD vertical velocity that is instantly illegitimate, m/s. Downward is
/// gravity's and unbounded on a mountainside; upward has no legitimate source
/// past a stride's bounce (measured ≤ 2.25 m/s rebounding from a 13 m/s
/// downhill impact).
pub const FLIGHT_VY_M_S: f32 = 4.0;
/// Instant-trigger altitude, m above local ground — the rl#137 vehicle-ram
/// gate's "never launched her skyward" line.
pub const FLIGHT_ALT_INSTANT_M: f32 = 5.0;
/// Carapace displacement in one 1/64 s tick that can only be a rescue/teleport.
pub const TELEPORT_M: f32 = 5.0;

/// One tick of full-body state for the rolling evidence window.
struct SoakTick {
    tick: u64,
    cara_pos: Vec3,
    cara_linvel: Vec3,
    cara_angvel: Vec3,
    above_ground: f32,
    /// Touching contact points across the narrow phase this tick (soak world =
    /// Sally + ground + nothing else, so these are all hers).
    contacts: usize,
    /// Whole-body mechanical energy, J ([`crab_mech_energy`]) — mass-weighted
    /// incl. rotation, so a launch window can audit conversion (E
    /// non-increasing) vs injection (E steps up beyond actuator power) without
    /// per-part masses (rl#332 launch-geometry follow-up).
    energy: f32,
    /// (pos, linvel) per crab body part, query order (stable within a run).
    parts: Vec<(Vec3, Vec3)>,
}

impl SoakTick {
    fn jsonl(&self) -> String {
        let v3 = |v: Vec3| format!("[{},{},{}]", v.x, v.y, v.z);
        let parts: Vec<String> = self
            .parts
            .iter()
            .map(|(p, v)| format!("[{},{},{},{},{},{}]", p.x, p.y, p.z, v.x, v.y, v.z))
            .collect();
        format!(
            "{{\"tick\":{},\"cara\":{},\"linvel\":{},\"angvel\":{},\"above\":{},\"contacts\":{},\"energy\":{},\"parts\":[{}]}}",
            self.tick,
            v3(self.cara_pos),
            v3(self.cara_linvel),
            v3(self.cara_angvel),
            self.above_ground,
            self.contacts,
            self.energy,
            parts.join(",")
        )
    }
}

#[derive(Debug, Clone)]
pub struct SoakEvent {
    pub onset_tick: u64,
    pub kind: &'static str,
    pub peak_above_ground: f32,
    pub peak_vy: f32,
    pub evidence_path: std::path::PathBuf,
}

pub struct SoakReport {
    pub ticks_run: u64,
    pub events: Vec<SoakEvent>,
    /// Whole-run extrema — the censored-negative evidence when `events` is empty.
    pub max_above_ground: f32,
    pub max_abs_vy: f32,
    pub max_up_vy: f32,
    pub max_power_w: f32,
    pub teleports: u64,
    /// Zero-contact stretches ≥ 16 ticks (0.25 s): how often and how long she is
 /// genuinely airborne — the visible "flight" look even when every
    /// stretch is gravity-legit downhill ballistics.
    pub airborne_stretches: u64,
    pub longest_airborne_ticks: u64,
    /// Ticks where some part's speed multiplied >4× in one tick
    /// ([`crab_world::physics::snapshot::is_kick`]) — the rl#332 F3 shape.
    pub kicks: u64,
    /// (tick, part index, speed before, speed after) of the first kick.
    pub first_kick: Option<(u64, usize, f32, f32)>,
    /// Windows where ΔE exceeded Σ actuator power·dt + slack (`crab_world::physics::snapshot::LEDGER_SLACK_J`).
    pub ledger_breaches: u64,
    pub worst_breach_j: f32,
    pub first_breach_tick: Option<u64>,
    /// Ticks with a same-crab non-adjacent link pair interpenetrating >5 mm / >20 mm
    /// (geometric; those pairs raise no contact since rl#332).
    pub overlap_ticks_5mm: u64,
    pub overlap_ticks_20mm: u64,
    /// (tick, depth m, link a, link b) of the deepest same-crab overlap; its state
    /// is written to `<out>/overlap-worst.bin`.
    pub worst_overlap: Option<(u64, f32, String, String)>,
}

/// rl#332: soak the live-policy world for `ticks` and detect "flight" — sustained
/// carapace altitude or vertical speed beyond anything legitimate locomotion
/// produces. On detection, dumps the rolling pre-onset window plus a post-onset
/// tail (positions, velocities, contacts, per-part state) as JSONL under
/// `out_dir`, then keeps soaking after a cooldown so one run can catch several
/// events. Detection thresholds are the [`FLIGHT_ALT_M`]-family constants above.
/// After this soak tick every actuator drive is forced to zero (a FixedUpdate
/// system between `Think` and `Act`) — the rl#332 ablation lever: a passive
/// multibody that KEEPS accelerating indicts the solver; one that tumbles to
/// rest indicts actuator-sourced energy.
#[derive(bevy::prelude::Resource, Clone, Copy)]
pub struct ZeroDriveAfter(pub Option<u64>);

pub fn run_flight_soak(
    policy: crab_world::policy::Policy,
    seed: u64,
    ticks: u64,
    out_dir: &std::path::Path,
    progress_every: u64,
    zero_drive_after: Option<u64>,
    dump_state_at: &[u64],
) -> std::io::Result<SoakReport> {
    use bevy_rapier3d::plugin::context::RapierContextSimulation;
    use crab_world::physics::snapshot::{LEDGER_SLACK_J, LEDGER_WINDOW, PlantSnapshot, is_kick};
    use std::collections::VecDeque;
    use std::io::Write;

    const WINDOW: usize = 192; // 3 s of pre-onset state at 64 Hz
    const POST: u32 = 128; // 2 s post-onset tail
    const COOLDOWN: u64 = 512; // ticks after an event before re-arming

    std::fs::create_dir_all(out_dir)?;
    let mut probe = Probe::new(
        policy,
        seed,
        crab_world::Visuals(false),
        true,
        crab_world::terrain::TerrainGrid::gcr(),
    );
    probe.log_every = u64::MAX; // sampling below, not via Probe::sample
    probe.app.insert_resource(ZeroDriveAfter(None));
    probe.app.add_systems(
        bevy::app::FixedUpdate,
        (|gate: Res<ZeroDriveAfter>,
          mut actions: ResMut<crab_world::bot::actuator::CrabActions>| {
            if gate.0.is_some() {
                let _ = actions.fill(0, 0.0);
            }
        })
        .after(crab_world::bot::BotSet::Think)
        .before(crab_world::bot::BotSet::Act),
    );

    let mut window: VecDeque<SoakTick> = VecDeque::with_capacity(WINDOW + 1);
    let mut report = SoakReport {
        ticks_run: 0,
        events: Vec::new(),
        max_above_ground: f32::NEG_INFINITY,
        max_abs_vy: 0.0,
        max_up_vy: 0.0,
        max_power_w: 0.0,
        teleports: 0,
        airborne_stretches: 0,
        longest_airborne_ticks: 0,
        kicks: 0,
        first_kick: None,
        ledger_breaches: 0,
        worst_breach_j: 0.0,
        first_breach_tick: None,
        overlap_ticks_5mm: 0,
        overlap_ticks_20mm: 0,
        worst_overlap: None,
    };
    let mut worst_overlap_state: Option<PlantSnapshot> = None;
    let mut airborne_run: u64 = 0;
    let mut prev_parts: Vec<(Vec3, Vec3)> = Vec::new();
    let mut ledger: VecDeque<(f32, f32)> = VecDeque::with_capacity(LEDGER_WINDOW + 1);
    let mut pending_snapshot: Option<PlantSnapshot> = None;
    let mut high_streak: u32 = 0;
    let mut cooldown_until: u64 = 0;
    // (file, ticks left, event index) while dumping a post-onset tail.
    let mut tail: Option<(std::fs::File, u32, usize)> = None;
    let mut prev_pos: Option<Vec3> = None;

    let terrain = probe
        .app
        .world()
        .resource::<crab_world::terrain::Terrain>()
        .clone();

    for _ in 0..ticks {
        if zero_drive_after == Some(report.ticks_run) {
            probe.app.insert_resource(ZeroDriveAfter(zero_drive_after));
            probe
                .app
                .insert_resource(crab_world::bot::actuator::SettleExtraIterations(0));
            println!(
                "sally-soak: zero-drive engaged at tick {}",
                report.ticks_run
            );
        }
        probe.tick();
        report.ticks_run += 1;
        let tick = report.ticks_run;

        let world = probe.app.world_mut();
        let contacts = {
            let mut q = world.query::<&RapierContextSimulation>();
            q.single(&*world)
                .map(|sim| {
                    sim.narrow_phase
                        .contact_pairs()
                        .flat_map(|p| p.manifolds.iter())
                        .flat_map(|m| m.points.iter())
                        .filter(|pt| -pt.dist > 0.0)
                        .count()
                })
                .unwrap_or(0)
        };
        let mut cara: Option<(Vec3, Vec3, Vec3)> = None;
        let mut parts: Vec<(Vec3, Vec3)> = Vec::new();
        {
            let mut q = world.query_filtered::<(
                &CrabEnvId,
                &Transform,
                &Velocity,
                Option<&CrabCarapace>,
            ), With<CrabBodyPart>>();
            for (env, t, vel, carapace) in q.iter(&*world) {
                if env.0 != 0 {
                    continue;
                }
                parts.push((t.translation, vel.linear));
                if carapace.is_some() {
                    cara = Some((t.translation, vel.linear, vel.angular));
                }
            }
        }
        let Some((pos, linvel, angvel)) = cara else {
            continue; // carapace despawned mid-rescue; next tick has it back
        };
        let above = pos.y - terrain.height(pos.x, pos.z);
        let energy = crab_mech_energy(world);
        // Gross actuator power, exactly the eval accumulator's observable (rl#279):
        // commanded torque × the sensor's measured hinge rate, Σ over joints.
        let power_w = {
            let mut q = world.query::<(&crab_world::bot::body::CrabJoint, &CrabEnvId)>();
            let joints: Vec<crab_world::bot::body::CrabJointId> = q
                .iter(world)
                .filter(|(_, env)| env.0 == 0)
                .map(|(j, _)| j.id)
                .collect();
            let actions = world.resource::<crab_world::bot::actuator::CrabActions>();
            let obs = world.resource::<crab_world::bot::sensor::CrabObservation>();
            match obs.env(0) {
                Some(view) => joints
                    .iter()
                    .filter_map(|id| {
                        actions.drive(0, *id).map(|d| {
                            (crab_world::bot::actuator::applied_torque(*id, d)
                                * view.joint_rate(*id))
                            .abs()
                        })
                    })
                    .sum(),
                None => f32::NAN,
            }
        };
        report.max_power_w = report.max_power_w.max(power_w);

        if prev_parts.len() == parts.len() {
            for (i, ((_, v0), (_, v1))) in prev_parts.iter().zip(&parts).enumerate() {
                let (s0, s1) = (v0.length(), v1.length());
                if is_kick(s0, s1) {
                    report.kicks += 1;
                    if report.first_kick.is_none() {
                        report.first_kick = Some((tick, i, s0, s1));
                    }
                    if report.kicks <= 100 {
                        println!(
                            "sally-soak: KICK tick {tick} part {i} speed {s0:.2}→{s1:.2} m/s \
                             (carapace {:.2} m/s, above {above:.2} m, contacts {contacts}, E {energy:.0} J)",
                            linvel.length()
                        );
                    }
                }
            }
        }
        prev_parts.clone_from(&parts);
        let overlap = crab_world::bot::contact_audit::same_crab_overlap(world, 0.005);
        if overlap.depth > 0.005 {
            report.overlap_ticks_5mm += 1;
        }
        if overlap.depth > 0.02 {
            report.overlap_ticks_20mm += 1;
        }
        if report
            .worst_overlap
            .as_ref()
            .is_none_or(|w| overlap.depth > w.1)
        {
            report.worst_overlap = Some((tick, overlap.depth, overlap.a, overlap.b));
            worst_overlap_state = Some(PlantSnapshot::capture(world, tick));
        }
        if power_w.is_finite() {
            ledger.push_back((energy, power_w));
            if ledger.len() > LEDGER_WINDOW + 1 {
                ledger.pop_front();
            }
            if ledger.len() == LEDGER_WINDOW + 1 {
                let budget: f32 = ledger.iter().skip(1).map(|(_, p)| p).sum::<f32>()
                    / crab_world::physics::PHYSICS_HZ as f32
                    + LEDGER_SLACK_J;
                let de = energy - ledger.front().unwrap().0;
                if de > budget {
                    report.ledger_breaches += 1;
                    report.worst_breach_j = report.worst_breach_j.max(de - budget);
                    if report.first_breach_tick.is_none() {
                        report.first_breach_tick = Some(tick);
                    }
                    if report.ledger_breaches <= 20 {
                        println!(
                            "sally-soak: LEDGER breach tick {tick} ΔE {de:.0} J over 32 ticks vs budget {budget:.0} J \
                             (carapace {:.2} m/s, above {above:.2} m, contacts {contacts})",
                            linvel.length()
                        );
                    }
                }
            }
        } else {
            ledger.clear();
        }
        if dump_state_at.contains(&tick) {
            pending_snapshot = Some(PlantSnapshot::capture(world, tick));
        } else if let Some(mut snap) = pending_snapshot.take() {
            snap.finish(world);
            let path = out_dir.join(format!("state-{}.bin", snap.tick));
            snap.save(&path)?;
            println!(
                "sally-soak: state after tick {} + tick {} drives → {}",
                snap.tick,
                tick,
                path.display()
            );
        }

        report.max_above_ground = report.max_above_ground.max(above);
        report.max_abs_vy = report.max_abs_vy.max(linvel.y.abs());
        report.max_up_vy = report.max_up_vy.max(linvel.y);
        if contacts == 0 {
            airborne_run += 1;
        } else {
            if airborne_run >= 16 {
                report.airborne_stretches += 1;
                report.longest_airborne_ticks = report.longest_airborne_ticks.max(airborne_run);
            }
            airborne_run = 0;
        }
        let teleported = prev_pos.is_some_and(|p| (pos - p).length() > TELEPORT_M);
        if teleported {
            report.teleports += 1;
        }
        prev_pos = Some(pos);

        let state = SoakTick {
            tick,
            cara_pos: pos,
            cara_linvel: linvel,
            cara_angvel: angvel,
            above_ground: above,
            contacts,
            energy,
            parts,
        };

        // Finish an in-flight post-onset tail dump.
        if let Some((file, left, idx)) = &mut tail {
            writeln!(file, "{}", state.jsonl())?;
            *left -= 1;
            if *left == 0 {
                let done = report.events[*idx].clone();
                println!(
                    "sally-soak: EVENT {} at tick {} — kind={} peak_above={:.2} m peak_vy={:.2} m/s → {}",
                    idx,
                    done.onset_tick,
                    done.kind,
                    done.peak_above_ground,
                    done.peak_vy,
                    done.evidence_path.display()
                );
                tail = None;
            }
        } else {
            window.push_back(state);
            if window.len() > WINDOW {
                window.pop_front();
            }
        }

        // Track peaks for the event being tailed.
        if let Some((_, _, idx)) = &tail {
            let ev = &mut report.events[*idx];
            ev.peak_above_ground = ev.peak_above_ground.max(above);
            ev.peak_vy = ev.peak_vy.max(linvel.y.abs());
            continue;
        }

        high_streak = if above > FLIGHT_ALT_M {
            high_streak + 1
        } else {
            0
        };
        let kind = if !pos.is_finite() || !linvel.is_finite() {
            Some("non-finite")
        } else if above > FLIGHT_ALT_INSTANT_M {
            Some("skyward-altitude")
        } else if high_streak >= FLIGHT_SUSTAIN_TICKS {
            Some("sustained-altitude")
        } else if linvel.y > FLIGHT_VY_M_S {
            Some("upward-velocity")
        } else if teleported {
            Some("teleport-or-rescue")
        } else {
            None
        };

        if let Some(kind) = kind {
            if tick < cooldown_until {
                high_streak = 0;
                continue;
            }
            cooldown_until = tick + COOLDOWN;
            high_streak = 0;
            let idx = report.events.len();
            let path = out_dir.join(format!("event-{idx}-tick{tick}-{kind}.jsonl"));
            let mut file = std::fs::File::create(&path)?;
            writeln!(
                file,
                "{{\"event\":{idx},\"onset_tick\":{tick},\"kind\":\"{kind}\",\"seed\":{seed},\"pre_window\":{},\"post_window\":{POST}}}",
                window.len()
            )?;
            for s in &window {
                writeln!(file, "{}", s.jsonl())?;
            }
            window.clear();
            report.events.push(SoakEvent {
                onset_tick: tick,
                kind,
                peak_above_ground: above,
                peak_vy: linvel.y.abs(),
                evidence_path: path,
            });
            tail = Some((file, POST, idx));
        }

        if progress_every > 0 && tick.is_multiple_of(progress_every) {
            println!(
                "sally-soak: tick {tick}/{ticks} — above={above:.3} m speed={:.3} m/s \
                 E={energy:.1} J P={power_w:.0} W contacts={contacts} max_above={:.3} max|vy|={:.3} \
                 events={} teleports={}",
                linvel.length(),
                report.max_above_ground,
                report.max_abs_vy,
                report.events.len(),
                report.teleports
            );
        }
    }
    if let Some(snap) = worst_overlap_state {
        snap.save(&out_dir.join("overlap-worst.bin"))?;
    }
    Ok(report)
}

/// Total mechanical energy of env 0's crab from the rapier set (Σ ½m|v|² +
/// ½ω·I·ω + m·g·y) — the rl#332 discriminator: a PASSIVE body's total can only
/// fall (gravity is inside as PE), so growth is solver-injected; a driven body's
/// growth is bounded by actuator power.
fn crab_mech_energy(world: &mut World) -> f32 {
    use bevy_rapier3d::plugin::context::RapierRigidBodySet;
    use bevy_rapier3d::prelude::RapierRigidBodyHandle;
    use crab_world::bot::body::CrabBodyPart;

    let handles: Vec<bevy_rapier3d::rapier::dynamics::RigidBodyHandle> = {
        let mut q = world.query_filtered::<&RapierRigidBodyHandle, With<CrabBodyPart>>();
        q.iter(world).map(|h| h.0).collect()
    };
    let mut set_q = world.query::<&RapierRigidBodySet>();
    let Ok(set) = set_q.single(world) else {
        return f32::NAN;
    };
    crab_world::physics::snapshot::mech_energy(&set.bodies, &handles)
}

/// One pumped fixed step's cost (rl#396): `wall_ms` is the whole `FixedMain` pass —
/// sensing, policy forward, actuation, all substeps, writeback — i.e. what a
/// step-carrying render frame pays on top of its own render work. Headless on an
/// otherwise-idle box this is a FLOOR for the windowed host's per-step cost, not an
/// estimate of it: the deployed step shares cores with the pipelined render, and the
/// probe's FixedMain is lighter (no pose sentinel, no vehicle plugin), and the
/// tick-finalize frame additionally carries the driver's broadcast tail
/// (`server.step_next` + articulation capture) that only `finalize_ms`'s
/// collect+feed half models. A floor that busts the vsync budget is conclusive; a
/// floor that fits only fails to rule fitting out. The rapier
/// counter fields carry the LAST substep only: bevy_rapier loops
/// `PHYSICS_SUBSTEPS` inside one fixed pass and rapier resets its counters per
/// substep, so a whole-step physics share reads as ~`PHYSICS_SUBSTEPS × substep_ms`.
#[derive(Clone, Copy, Debug)]
pub struct StepSample {
    pub tick: u64,
    pub wall_ms: f64,
    pub substep_ms: f64,
    pub solver_ms: f64,
    pub collision_ms: f64,
    pub vel_res_ms: f64,
    pub vel_asm_ms: f64,
}

pub struct StepProfile {
    pub steps: Vec<StepSample>,
    /// Per-tick finalize (pose collect + hunt feed) — the pump tail the crossing
    /// frame pays once per tick on top of its step.
    pub finalize_ms: Vec<f64>,
    /// Per-tick `Update`-schedule wrap — headless here, so a floor for the windowed
    /// host's non-render per-frame work, not an estimate of it.
    pub update_ms: Vec<f64>,
    /// Sparse activity trace — the ruler discipline: a profile window must SHOW the
    /// crab hunting (distance changing, carapace walking), or it measured the
    /// settled scene the rl#396 stage-2 ruler was corrected for.
    pub activity: Vec<ProbeSample>,
    /// Isolated `Policy::act` mean — the NN forward alone, no world around it.
    pub policy_forward_ms: f64,
}

/// Profile the host's driven step in an ACTIVE-hunt scene: the prey kites away
/// whenever the crab closes inside `KITE_INSIDE_M`, so the chase never settles and
/// every measured step drives an awake, walking crab — the scene the TV actually
/// pays for (rl#396 stage-4 ruler correction), not the asleep-sim rest scene.
/// Cadence is the production 64:30 staircase (`steps_for_tick`), each owed step
/// pumped and timed individually — the render driver's spread-pump granularity.
pub fn run_step_profile(
    policy: crab_world::policy::Policy,
    seed: u64,
    warmup_ticks: u64,
    ticks: u64,
) -> StepProfile {
    use bevy_rapier3d::plugin::context::RapierContextSimulation;
    use std::time::Instant;

    // The prey holds the chase at the band the stage-4 TV windows measured
    // (prey 2.6–14 m, continuously moving).
    const KITE_INSIDE_M: f32 = 6.0;

    // NN forward alone, before the world takes the policy. A zero obs is fine:
    // the forward is dense matmuls, cost is data-independent.
    let policy_forward_ms = {
        let obs = [0.0f32; crab_world::bot::sensor::OBS_SIZE];
        for _ in 0..32 {
            std::hint::black_box(policy.act(std::hint::black_box(&obs)));
        }
        const N: u32 = 512;
        let t = Instant::now();
        for _ in 0..N {
            std::hint::black_box(policy.act(std::hint::black_box(&obs)));
        }
        t.elapsed().as_secs_f64() * 1e3 / N as f64
    };

    let mut probe = Probe::new(
        policy,
        seed,
        crab_world::Visuals(false),
        false,
        crab_world::terrain::TerrainGrid::gcr(),
    );
    probe.log_every = u64::MAX; // sampled below at the activity cadence, not per tick

    let mut out = StepProfile {
        steps: Vec::new(),
        finalize_ms: Vec::new(),
        update_ms: Vec::new(),
        activity: Vec::new(),
        policy_forward_ms,
    };

    let me = PlayerId(0);
    for i in 0..(warmup_ticks + ticks) {
        let measured = i >= warmup_ticks;

        // Kite input for this tick, rotated into the player's local frame (players
        // spawn with a round heading, so world axes are NOT walk axes).
        let crab = probe.sim.crabs()[0].pos();
        let prey = probe.sim.player(me);
        let input = prey
            .and_then(|p| {
                let (dx, dz) = Pos {
                    x: p.pos().x - crab.x,
                    z: p.pos().z - crab.z,
                }
                .to_meters();
                let d = (dx * dx + dz * dz).sqrt();
                (d > 0.1 && d < KITE_INSIDE_M).then(|| {
                    let (ux, uz) = (dx / d, dz / d);
                    let (sin, cos) = crate::cordic::trig::sin_cos(p.yaw());
                    let (sin, cos) = (
                        sin as f32 / crate::cordic::trig::ONE as f32,
                        cos as f32 / crate::cordic::trig::ONE as f32,
                    );
                    // advance_player: world v = (sin·f + cos·s, cos·f − sin·s).
                    Input::from_axes(cos * ux - sin * uz, sin * ux + cos * uz)
                })
            })
            .unwrap_or_else(|| Input::from_axes(0.0, 0.0));
        let prey = prey.map(|p| p.pos());

        let inputs = crab_slot::slot_inputs(&probe.sim);

        let t = Instant::now();
        probe.app.update();
        let update_ms = t.elapsed().as_secs_f64() * 1e3;

        if i == 0 {
            // The rapier context entity spawns on the app's first pass, so the
            // counters can only arm here, not at construction.
            let world = probe.app.world_mut();
            let mut q = world.query::<&mut RapierContextSimulation>();
            q.single_mut(world)
                .expect("headless_stack spawns exactly one rapier context")
                .pipeline
                .counters
                .enable();
        }

        for _ in 0..crate::cadence::steps_for_tick(inputs.stepping_into) {
            let t = Instant::now();
            crab_slot::pump_fixed_steps(probe.app.world_mut(), 1);
            let wall_ms = t.elapsed().as_secs_f64() * 1e3;
            if !measured {
                continue;
            }
            let world = probe.app.world_mut();
            let mut q = world.query::<&RapierContextSimulation>();
            let c = &q
                .single(world)
                .expect("headless_stack spawns exactly one rapier context")
                .pipeline
                .counters;
            out.steps.push(StepSample {
                tick: inputs.stepping_into,
                wall_ms,
                substep_ms: c.step_time.time_ms(),
                solver_ms: c.stages.solver_time.time_ms(),
                collision_ms: c.stages.collision_detection_time.time_ms(),
                vel_res_ms: c.solver.velocity_resolution_time.time_ms(),
                vel_asm_ms: c.solver.velocity_assembly_time.time_ms(),
            });
        }

        let t = Instant::now();
        // Steps already pumped above; a zero-step pump is finalize alone (pose
        // collect + hunt feed), the same split the render driver's spread pump runs.
        let mut poses = crab_slot::pump_slot_steps(probe.app.world_mut(), 0, &inputs);
        let finalize_ms = t.elapsed().as_secs_f64() * 1e3;
        for p in &mut poses {
            // Pursuit profile: no downs (module doc) — the pose crosses, the claws don't.
            p.claws.clear();
        }
        probe.sim.step(
            &std::collections::BTreeMap::from([(me, input)]),
            Externals::crabs_only(&poses),
        );

        if measured {
            out.finalize_ms.push(finalize_ms);
            out.update_ms.push(update_ms);
            let tick = probe.sim.tick();
            if out.activity.is_empty() || tick.is_multiple_of(64) {
                probe.sample(tick, prey);
                out.activity.push(*probe.samples.last().unwrap());
            }
        }
    }
    out
}
