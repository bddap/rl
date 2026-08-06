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
}

struct Probe {
    app: App,
    sim: Sim,
    samples: Vec<ProbeSample>,
    log_every: u64,
}

impl Probe {
    fn new(policy: crab_world::policy::Policy, seed: u64, visuals: crab_world::Visuals) -> Self {
        use crab_world::bot::headless::{
            HeadlessStack, WorldRole, force_serial_schedules, headless_stack,
            pin_single_thread_pools,
        };

        pin_single_thread_pools();

        let me = PlayerId(0);
        let sim = Sim::new(seed, &[me]);
        let spawns: Vec<Pos> = sim.crabs().iter().map(|c| c.pos()).collect();

        // The BARE crab stack, not `headless_server_world`: the stability probe
        // hand-spawns its ram craft, which the vehicle layer's `manage_vehicles`
        // would despawn as pilotless.
        let mut app = headless_stack(HeadlessStack {
            num_envs: spawns.len(),
            role: WorldRole::Standalone,
            // The probe models the GCR host, so it steps the host's ground — the
            // canonical terrain bake (rl#209, rl#293).
            grid: crab_world::terrain::TerrainGrid::gcr(),
            visuals,
        });
        app.add_plugins(NnCrabPlugin::new(vec![policy], spawns.clone()));
        arm(app.world_mut());
        park_fixed_auto_pump(&mut app);
        restart_crabs_to_spawns(app.world_mut(), &spawns);
        force_serial_schedules(&mut app);

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
pub fn run_headless_probe(
    policy: crab_world::policy::Policy,
    seed: u64,
    ticks: u64,
    log_every: u64,
    visuals: crab_world::Visuals,
) -> Vec<ProbeSample> {
    let mut probe = Probe::new(policy, seed, visuals);
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

    let mut probe = Probe::new(policy, seed, crab_world::Visuals(false));
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
    /// genuinely airborne — the owner-visible "flight" look even when every
    /// stretch is gravity-legit downhill ballistics.
    pub airborne_stretches: u64,
    pub longest_airborne_ticks: u64,
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
) -> std::io::Result<SoakReport> {
    use bevy_rapier3d::plugin::context::RapierContextSimulation;
    use std::collections::VecDeque;
    use std::io::Write;

    const WINDOW: usize = 192; // 3 s of pre-onset state at 64 Hz
    const POST: u32 = 128; // 2 s post-onset tail
    const COOLDOWN: u64 = 512; // ticks after an event before re-arming

    std::fs::create_dir_all(out_dir)?;
    let mut probe = Probe::new(policy, seed, crab_world::Visuals(false));
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
    };
    let mut airborne_run: u64 = 0;
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
    let g = 9.81f32;
    handles
        .iter()
        .filter_map(|h| set.bodies.get(*h))
        .map(|rb| {
            let m = rb.mass();
            let v: Vec3 = rb.linvel();
            let w: Vec3 = rb.angvel();
            let rmat = Mat3::from_quat(rb.position().rotation);
            let i_world = rmat
                * rb.mass_properties()
                    .local_mprops
                    .reconstruct_inertia_matrix()
                * rmat.transpose();
            0.5 * m * v.length_squared() + 0.5 * w.dot(i_world * w) + m * g * rb.center_of_mass().y
        })
        .sum()
}
