pub mod wire;

use std::path::Path;

use bevy::prelude::*;

use crate::bot::RESET_GRACE_TICKS;
use crate::bot::actuator::{CrabActions, applied_torque, total_drive_torque_ceiling};
use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint};
use crate::bot::headless::{
    WorldRole, force_serial_schedules, headless_server_world, pin_single_thread_pools,
};
use crate::bot::sensor::{CrabObservation, CrabTargets};
use crate::bot::{BotSet, CrabSpawns};
use crate::policy::Policy;
use crate::training::reward::dist_3d;
use crate::training::targets::{
    BAND_START_MIN, REACH_RADIUS, TARGET_Y_MAX, TARGET_Y_MIN, band_lure, closest_tip_dist,
    polar_target, recenter_delta, tip_touch,
};

/// The far sweep's ball distance — pinned from measured pace, NOT the band edge
/// (rl#292/rl#293): at the 128 m band edge an unreachable ball would make min-progress
/// measure the eval locale's worst obstruction (one cliffed bearing caps the min
/// forever and arrival vanishes from the headline). 24 m ≈ 65% of the pinned charge's
/// flat-ground horizon traverse ([`CRAB_CHARGE_SPEED_HEIGHTS_PER_S`] ≈ 1.6 m/s ×
/// [`DEFAULT_EVAL_TICKS`] ≈ 23 s ≈ 37 m), so arrival stays measurable for a competent
/// brain even on relief, and mid-band for the obs. NOTE: the far sweep runs NO
/// mid-episode recenter — matching the training regime, where `body.pos` runs out to
/// the per-episode traverse un-regauged; only the pace probe wires the
/// `pace_recenter` seam (the obs regime net's 9 m rebase produces). Re-pin from pace
/// evidence when the gait regime moves; numbers across a distance re-pin are
/// incomparable by design.
pub const DEFAULT_TARGET_DISTANCE_M: f32 = 24.0;

/// How many terrain locales the far sweep samples (rl#293): one compass per locale,
/// headline = MEDIAN over locales of min-over-bearings. The median keeps one
/// pathological locale (a cliff ring) from capping the headline forever the way a
/// min-over-locales would, while still refusing to let a single lucky meadow carry
/// it. Locales are drawn deterministically from the bake ([`eval_locales`]), so the
/// eval stays reproducible per brain; locale 0 is the tile center — the close and
/// pace probes run there, keeping the charge instrument's geometry fixed.
pub const EVAL_LOCALES: usize = 3;

/// The fixed eval locales: tile center plus [`EVAL_LOCALES`]−1 pinned interior
/// coordinates. LITERALS on purpose — deriving them from the training sampler (a
/// seeded `random_episode_origin`) would silently move the instrument whenever a
/// training knob (band, edge margin) or the RNG's stream changes, making headlines
/// incomparable with no signal. These move only when the bake or this constant does.
/// Both points sit kilometers inside the ±15 km tile with the full band clear of the
/// edge extension.
fn eval_locales() -> [Vec2; EVAL_LOCALES] {
    [
        Vec2::ZERO,
        Vec2::new(4800.0, -3600.0),
        Vec2::new(-6400.0, 5200.0),
    ]
}

/// The close-range probe distance (rl#252): the same compass swept with the ball just
/// outside the claw, so close-range reach has an instrument — without one, flipping the
/// rl#250 close-target curriculum shows only its downside (far-chase regression) and
/// one-change-at-a-time loses its readout. Pinned between [`REACH_RADIUS`] and
/// [`BAND_START_MIN`]: inside the close disc the rl#250 curriculum trains (which the
/// far band never samples), and outside the reach radius at spawn. The probe's floor
/// is zero under the tip-based touch (rl#253): Sally's zero-action rest pose passively
/// slumps up to ~0.55 m toward its ~270° side over a full episode, which drift-crossed
/// the old carapace-distance reach at ~3/8 bearings, but the slump carries her claw
/// tips AWAY from the ball (measured on the live body: min tip distance 0.51 m across
/// the compass, never near [`REACH_RADIUS`]) — so any reached bearing here is a real
/// strike, not drift. The probe is deterministic per brain; it landed BEFORE the
/// rl#250 flip so the flip is judged as a delta against a recorded baseline.
pub const CLOSE_PROBE_DISTANCE_M: f32 = 1.0;
const _: () = assert!(
    REACH_RADIUS < CLOSE_PROBE_DISTANCE_M && CLOSE_PROBE_DISTANCE_M < BAND_START_MIN,
    "the close probe must sit between the reach radius and the chase-band start"
);

/// The default eval episode length PER BEARING — one training episode horizon (~23 s of
/// crab time at 64 Hz), enough for a working gait to traverse a far target. The (ticks,
/// distance, compass) TRIPLE defines "the chase eval"; every judge — the CLI (release
/// gate, monitor) and the trainer's keep-best gate — must take its defaults from HERE
/// or the metric forks.
pub const DEFAULT_EVAL_TICKS: u64 = crate::training::systems::MAX_EPISODE_TICKS as u64;

/// The fixed compass of target bearings every eval sweeps, uniform in [0, 2π) with
/// bearing 0 = +X (the single bearing the pre-rl#239 eval judged, kept so bearing-0
/// numbers stay comparable to the historical curve). Training samples bearing uniformly
/// (`targets::sample_target`), so a brain competent at one bearing only is a training
/// pathology the eval must EXPOSE: the headline score is the MIN over these bearings —
/// a mean would let seven striding bearings hide one dead one, which is exactly the
/// blindness that scored an 8.93 m (+X) brain as "can chase" while it shuffled in
/// place at the bearing GCR players face (rl#239).
pub const EVAL_BEARINGS: usize = 8;

const TARGET_Y: f32 = (TARGET_Y_MIN + TARGET_Y_MAX) / 2.0;

/// Sally's sustained full-charge pace in BODY HEIGHTS per second — THE pinned measured
/// speed (rl#266), scale-free so each domain folds in its OWN sizing rule: net derives
/// the sim-side charge speed (× her `CRAB_SCALE` player-heights sim stature) that spawn
/// clearance and the pursuit-test driver build on, and this eval re-derives the arena
/// value (× the natural rig height) to re-measure her every run. Pinned FROM the eval's
/// own instrument ([`BearingReport::sustained_pace_m_per_s`], best pace-probe bearing)
/// so pin and measurement can never diverge in method — which is also why a change to
/// the instrument itself re-pins in the SAME change, from the new instrument's reading
/// of the then-live brain; [`run_eval`] flags when a retrain drifts the real gait
/// outside [`CHARGE_SPEED_DRIFT_TOL`] — re-measure and re-pin then.
///
/// Measured 2026-07-16 on the live mlp512x3-s2 brain — the third re-pin in five days
/// (1.74 → 2.31 → 2.61; the last is the rl#280 pace probe replacing the far-sweep
/// instrument, which had read 2.31 on the same brain that morning: the probe's lure
/// never lets her decelerate into an arrival, so it reads her pure charge a shade
/// higher — the instrument systematic had to move into the pin, not sit in every
/// future drift readout). Locomotion training is still accelerating her, which is how
/// the original hand-pinned 8.5 sim m/s rotted unnoticed while `progress_m` sat
/// saturated at the 9 m target (rl#266). Strictly her best 5-seconds-from-rest pace,
/// not a cruise speed: for a decaying speed profile the max prefix lands at the
/// [`PACE_WINDOW_MIN_S`] floor, so the opening lunge inflates it a little — the safe
/// direction for the spawn clearance derived from it. The instrument's saturation
/// ceiling is [`PACE_PROBE_DISTANCE_M`] / [`PACE_WINDOW_MIN_S`] ≈ 5.9 heights/s
/// (`charge_speed_guard_keeps_saturation_headroom`); when a re-pin approaches it, the
/// probe needs an even farther ball (rl#280).
pub const CRAB_CHARGE_SPEED_HEIGHTS_PER_S: f32 = 2.61;

/// Fractional drift band around [`CRAB_CHARGE_SPEED_HEIGHTS_PER_S`] before the eval
/// flags. Wide enough for brain-to-brain wobble in the measured pace and the residual
/// lunge-vs-sustained blur; a retrain that changes her locomotion regime lands well
/// outside it. At ±25% the spawn-grace guarantee (5 s of charge, net's
/// `SPAWN_GRACE_SECS`) stays within 4–6.7 s of truth — feel-tolerable; beyond that the
/// spawn-safety derivation is lying and the pin must be re-measured.
pub const CHARGE_SPEED_DRIFT_TOL: f32 = 0.25;

/// Prefix paces count only once this much active time has elapsed: her opening lunge
/// outpaces the sustained gait (rl#257) and a tiny elapsed divisor would let one
/// spawn-transient hop dominate. Five seconds is past the lunge and happens to be the
/// horizon the constant serves (spawn grace) — but it is a measurement floor, not that
/// feel knob. Ceiling to be aware of: a brain that reaches the ball inside this window
/// saturates the instrument at target_distance / PACE_WINDOW_MIN_S — the guard still
/// flags (saturation is itself far off any honest pin), but the measured number stops
/// tracking her true pace. That ceiling is why the charge metric reads from the
/// [`PACE_PROBE_DISTANCE_M`] probe, not the far sweep (rl#280).
const PACE_WINDOW_MIN_S: f32 = 5.0;

/// The pace probe's real ball distance (rl#280) — the charge-speed instrument's own
/// sweep, so its saturation ceiling (this / [`PACE_WINDOW_MIN_S`] ≈ 3.6 m/s ≈ 5.9
/// heights/s) keeps headroom over the pin's drift band. A LITERAL since rl#292
/// decoupled it from the band constant (it was 2× the old 9 m band edge): this is the
/// instrument's own geometry — the [`CRAB_CHARGE_SPEED_HEIGHTS_PER_S`] pin was
/// measured at exactly this distance, and moving it moves the saturation ceiling,
/// so it re-pins only WITH a charge re-measure, never by riding a band change.
/// The probe presents the ball through the [`band_lure`] clamp and the
/// [`pace_recenter`] body.pos recenter, matching the obs regime GCR play happens in
/// (whose recenter shares the [`recenter_delta`] trigger and whose hunt poser shares
/// the [`band_lure`] clamp, rl#301).
pub const PACE_PROBE_DISTANCE_M: f32 = 18.0;

/// The pace probe's episode length: twice the [`PACE_WINDOW_MIN_S`] floor, so the
/// prefix max scans [5 s, 10 s] — past where the old instrument's 9 m runs ever
/// reached (arrival ended them around 6.4 s at the then-pin) — at ~40% of what a
/// third full-length sweep at [`DEFAULT_EVAL_TICKS`] would cost (the keep-best gate
/// runs this eval inline on the learner thread). Paces up to the distance ceiling
/// stay measurable: only a run FASTER than [`PACE_PROBE_DISTANCE_M`] /
/// [`PACE_WINDOW_MIN_S`] can arrive before the window opens.
pub const PACE_PROBE_TICKS: u64 =
    (2.0 * PACE_WINDOW_MIN_S * crate::physics::PHYSICS_HZ as f32) as u64;

/// Net chase progress below which a bearing's J/m is not reported (rl#279): work over
/// a near-zero distance explodes toward ∞ without describing the gait — a parked or
/// dead bearing has a saturation number (always emitted) but no cost of transport.
pub const WORK_PER_M_MIN_PROGRESS_M: f32 = 0.5;

/// One bearing's episode — the same measurements the pre-compass eval reported for its
/// single +X episode.
#[derive(Debug, Clone, Copy)]
pub struct BearingReport {
    pub bearing_rad: f32,
    pub progress_m: f32,
    pub total_torque: f32,
    pub mean_torque_per_tick: f32,
    /// Mean commanded |torque| per tick as a fraction of the rig's total drive ceiling
    /// (rl#279) — in [0, 1] by construction (each joint's drive is clamped to its own
    /// ceiling). Effort observability only: never folded into the headline, never
    /// selected on.
    pub saturation: f32,
    /// Total mechanical work Σ|τ·ω|·Δt over the active ticks (rl#279), in joules of
    /// the sim's unit system. Commanded torque times the sensor's measured hinge rate
    /// — the same estimator either sign of back-driving counts as effort spent.
    pub work_j: f32,
    pub initial_distance_m: f32,
    pub closest_distance_m: f32,
    pub final_distance_m: f32,
    /// Closest any claw tip came to the target (infinite if no tip was seen) —
    /// the measure `reached` derives from. Distinct from `closest_distance_m`,
    /// which tracks the CARAPACE and stays the locomotion/progress measure:
    /// reaching is a claw strike, not a body-near pass (rl#253).
    pub closest_tip_distance_m: f32,
    pub reached: bool,
    pub active_ticks: u64,
    /// Best prefix pace toward the target (arena m/s): max over active ticks past
    /// [`PACE_WINDOW_MIN_S`] of progress-so-far / time-so-far. The max is taken right
    /// when she arrives at the ball, so parking there for the rest of the episode
    /// (which dilutes `progress_m / active_ticks`) can't understate her charge. The
    /// rl#266 speed guard reads the PACE sweep's values only (rl#280 — the far/close
    /// sweeps saturate too low to instrument her); theirs are diagnostic. 0.0 when
    /// she never progressed.
    pub sustained_pace_m_per_s: f32,
}

impl BearingReport {
    /// Cost of transport (rl#279): total mechanical work over net chase progress,
    /// `None` below [`WORK_PER_M_MIN_PROGRESS_M`] — a near-zero denominator would
    /// print an ∞-ish number that describes the guard, not the gait.
    pub fn j_per_m(&self) -> Option<f32> {
        (self.progress_m >= WORK_PER_M_MIN_PROGRESS_M).then(|| self.work_j / self.progress_m)
    }
}

/// One full compass of bearing episodes at one target distance — the distance travels
/// with the sweep's data so a printed line can never pair one sweep's numbers with the
/// other's distance.
#[derive(Debug, Clone, Copy)]
pub struct CompassSweep {
    pub target_distance_m: f32,
    pub per_bearing: [BearingReport; EVAL_BEARINGS],
}

impl CompassSweep {
    /// The WORST (min-progress) bearing's episode — one coherent real episode, so
    /// `reached`/torque/distances all describe the same run. MIN over bearings is the
    /// anti-gaming pick: a mean would let seven striding bearings hide one dead one.
    /// `total_cmp` only for a deterministic pick; `progress_m` is never NaN
    /// (`(a-b).max(0.0)` scrubs it).
    pub fn worst(&self) -> &BearingReport {
        self.per_bearing
            .iter()
            .min_by(|a, b| a.progress_m.total_cmp(&b.progress_m))
            .expect("compass is non-empty")
    }

    /// How many bearings reached the ball. For the close probe this is the rl#250
    /// emergence readout, diffed against the pre-flip baseline (zero for the rest
    /// pose under the tip-based touch — see [`CLOSE_PROBE_DISTANCE_M`]).
    pub fn reached_count(&self) -> usize {
        self.per_bearing.iter().filter(|b| b.reached).count()
    }

    /// Mean torque saturation over the compass (rl#279) — the sweep-level effort
    /// readout. A MEAN, unlike the min-progress headline: effort is observability,
    /// so the aggregate should describe the whole gait, not one adversarial bearing.
    pub fn mean_saturation(&self) -> f32 {
        let sum: f32 = self.per_bearing.iter().map(|b| b.saturation).sum();
        sum / EVAL_BEARINGS as f32
    }

    /// Mean cost of transport over the bearings that measured one (rl#279) — `None`
    /// when no bearing cleared [`WORK_PER_M_MIN_PROGRESS_M`].
    pub fn mean_j_per_m(&self) -> Option<f32> {
        let measured: Vec<f32> = self
            .per_bearing
            .iter()
            .filter_map(|b| b.j_per_m())
            .collect();
        (!measured.is_empty()).then(|| measured.iter().sum::<f32>() / measured.len() as f32)
    }

    /// Compact per-bearing progress readout for log lines (rl#276): the full vector
    /// the headline min collapses to one number, so a directional hole names its
    /// bearing in train.log at onset instead of after a forensic per-bearing job.
    pub fn progress_line(&self) -> String {
        self.per_bearing
            .iter()
            .map(|b| format!("{:.0}°:{:.2}", b.bearing_rad.to_degrees(), b.progress_m))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The BEST bearing's sustained pace (arena m/s) — her full charge. Max, not min:
    /// the speed guard asks how fast she really is (spawn safety cares about her top
    /// pace), while dead bearings are the headline min-progress gate's problem.
    fn best_sustained_pace_m_per_s(&self) -> f32 {
        self.per_bearing
            .iter()
            .map(|b| b.sustained_pace_m_per_s)
            .fold(0.0, f32::max)
    }
}

/// The sweeps of one eval, judging one load of one brain.
#[derive(Debug, Clone, Copy)]
pub struct EvalReport {
    pub policy_loaded: bool,
    /// The far sweeps, one compass per [`eval_locales`] locale — the chase eval
    /// proper, sole source of the headline (the MEDIAN locale's worst bearing).
    pub far: [CompassSweep; EVAL_LOCALES],
    /// The rl#252 close-range probe at [`CLOSE_PROBE_DISTANCE_M`], locale 0 only
    /// (fixed geometry — the emergence readout diffs against recorded baselines).
    /// SIDECAR ONLY — nothing here may feed [`Self::progress_m`]/[`Self::reached`],
    /// the headline every gate keys off: folding it in would be a training-adjacent
    /// metric change riding along with a measurement.
    pub close: CompassSweep,
    /// The rl#280 pace probe at [`PACE_PROBE_DISTANCE_M`], locale 0 only (the
    /// charge instrument's geometry stays fixed so the pin stays comparable).
    /// SIDECAR like the close probe: sole source of
    /// [`Self::measured_charge_heights_per_s`], never of the headline.
    pub pace: CompassSweep,
}

impl EvalReport {
    /// The MEDIAN locale's far sweep, by worst-bearing progress — the headline's one
    /// coherent compass (see [`EVAL_LOCALES`] for why median, not min, over locales).
    /// `total_cmp` for a deterministic pick.
    pub fn median_far(&self) -> &CompassSweep {
        let mut by_worst: [&CompassSweep; EVAL_LOCALES] = std::array::from_fn(|i| &self.far[i]);
        by_worst.sort_by(|a, b| a.worst().progress_m.total_cmp(&b.worst().progress_m));
        by_worst[EVAL_LOCALES / 2]
    }

    /// Median-over-locales of min-over-bearings FAR chase progress — THE headline
    /// scalar every gate compares.
    pub fn progress_m(&self) -> f32 {
        self.median_far().worst().progress_m
    }

    /// Whether the crab reached the far ball at the headline (median locale's worst)
    /// bearing.
    pub fn reached(&self) -> bool {
        self.median_far().worst().reached
    }

    /// Sally's measured charge speed in body-heights/s — the pace probe's best
    /// sustained pace over the natural rig height, the same height net's arena→sim
    /// seam divides by, so this number and the sim-side charge constant live on one
    /// scale. `None` when unmeasurable: no loaded policy (the rest pose measures the
    /// baseline, not her), no progress, a degenerate silhouette — or a pace sweep not
    /// at [`PACE_PROBE_DISTANCE_M`]: the pin was measured at the probe distance, and a
    /// different one moves the instrument's saturation ceiling (target /
    /// [`PACE_WINDOW_MIN_S`]), so comparing that against the pin would flag a geometry
    /// artifact as drift and invite a corrupting re-pin. (The far sweep's `--distance`
    /// no longer matters here — the probe always runs at its own distance, which is
    /// what decoupled the charge metric from the chase eval's geometry, rl#280.)
    pub fn measured_charge_heights_per_s(&self) -> Option<f32> {
        if !self.policy_loaded || self.pace.target_distance_m != PACE_PROBE_DISTANCE_M {
            return None;
        }
        let pace = self.pace.best_sustained_pace_m_per_s();
        let height = crate::mesh_fallback::natural_body_height()?;
        (pace > 0.0).then(|| pace / height)
    }

    /// Fractional drift of the measured charge speed from the pinned
    /// [`CRAB_CHARGE_SPEED_HEIGHTS_PER_S`] (+ = faster than pinned). Outside
    /// [`CHARGE_SPEED_DRIFT_TOL`] the spawn-safety derivation is stale — rl#266.
    pub fn charge_speed_drift(&self) -> Option<f32> {
        self.measured_charge_heights_per_s()
            .map(|m| m / CRAB_CHARGE_SPEED_HEIGHTS_PER_S - 1.0)
    }
}

#[derive(Resource, Clone, Copy)]
struct EvalConfig {
    target_distance: f32,
    bearing_rad: f32,
    settle_ticks: u64,
    /// Pace-probe mode (rl#280): the real ball sits past the training band edge, so
    /// the obs target is posed through [`probe_lure`] and body.pos drift through
    /// [`pace_recenter`] — the production seams — while every measurement stays
    /// against the real ball.
    pace_probe: bool,
}

#[derive(Resource, Default)]
struct EvalState {
    tick: u64,
    /// The REAL ball every distance/pace/tip measurement is taken against — the obs
    /// slot equals it except in pace-probe mode, where the slot holds the lure.
    /// `None` until the env slots exist (the pre-spawn window).
    real_target: Option<Vec3>,
    initial_dist: f32,
    closest_dist: f32,
    last_dist: f32,
    closest_tip_dist: Option<f32>,
    torque_sum: f64,
    work_sum: f64,
    torque_ticks: u64,
    best_pace_m_per_s: f32,
}

pub fn run_eval(
    _body_gate: crate::mesh_fallback::BodyGate,
    checkpoint_dir: &Path,
    active_ticks: u64,
    target_distance: f32,
) -> Result<EvalReport, String> {
    // Judge the policy on the plant it trained on (bddap/rl#268): a checkpoint's
    // recorded plant is adopted before any world spawns, so the invoker (the standing
    // rl-eval-monitor, the release gate, a hand eval) needs no plant knowledge — and a
    // conflicting env refuses rather than mismeasures. The provenance line prints
    // HERE, after adoption — printing it earlier would itself resolve the override
    // from the env and turn every sidecar adoption into a refusal. stdout, beside the
    // `EVAL_RESULT` lines: consumers filter by prefix (eval/wire.rs), so an artifact
    // that shows the numbers shows the plant they were measured on.
    crate::bot::body::adopt_recorded_plant(checkpoint_dir)?;
    println!("eval: plant: {}", crate::bot::body::plant_provenance());
    pin_single_thread_pools();

    // ONE read arms-or-refuses (rl#241 — a classify-then-load pair could straddle a
    // checkpoint swap): a checkpoint the runtime would refuse to arm must refuse the
    // eval too, not become a rest-pose baseline quietly printed as the run's training
    // progress. Missing is the legitimate no-brain-yet case — judge the explicit rest
    // baseline. One load also means every episode judges the SAME weights: the CLI is
    // pointed at a LIVE checkpoint dir (rl-eval-monitor, the release gate), and a
    // per-episode reload across the ~5 min far+close+pace sweeps could min over a
    // composite of adjacent brains that no single brain achieved.
    let policy = match crate::policy::load_armed(checkpoint_dir) {
        Ok(policy) => policy,
        Err(crate::policy::CheckpointUnusable::Missing) => Policy::rest(),
        Err(crate::policy::CheckpointUnusable::Refused(why)) => {
            return Err(format!(
                "checkpoint at {} refused: {why}",
                checkpoint_dir.display()
            ));
        }
        Err(crate::policy::CheckpointUnusable::Mismatch(dims)) => {
            return Err(format!(
                "checkpoint at {} was built for a different rig ({}/{} obs/act)",
                checkpoint_dir.display(),
                dims.obs,
                dims.action,
            ));
        }
    };
    let policy_loaded = policy.is_loaded();
    let policy = std::rc::Rc::new(policy);

    // The close and pace probes reuse the far sweep's episode definition (same
    // compass, same fresh-world episode) so their numbers read on the same scale; the
    // pace probe differs only where its geometry forces it to (rl#280). Only the far
    // sweep fans over locales (rl#293) — the probes are instruments with fixed
    // geometry at locale 0, the tile center.
    let locales = eval_locales();
    let report = EvalReport {
        policy_loaded,
        far: locales.map(|l| run_compass(&policy, active_ticks, target_distance, false, l)),
        close: run_compass(
            &policy,
            active_ticks,
            CLOSE_PROBE_DISTANCE_M,
            false,
            locales[0],
        ),
        pace: run_pace_probe(&policy, active_ticks, locales[0]),
    };
    // The rl#266 speed guard, HERE so every judge (CLI, keep-best gate) flags for free:
    // a flag, not a verdict — a drifted pin is a re-measure chore, not a bad brain.
    if let (Some(measured), Some(drift)) = (
        report.measured_charge_heights_per_s(),
        report.charge_speed_drift(),
    ) {
        if drift.abs() > CHARGE_SPEED_DRIFT_TOL {
            tracing::warn!(
                "charge speed drift (rl#266): measured {measured:.3} body-heights/s vs pinned \
                 {CRAB_CHARGE_SPEED_HEIGHTS_PER_S} ({:+.0}%) — re-measure and re-pin \
                 CRAB_CHARGE_SPEED_HEIGHTS_PER_S; spawn clearance and the pursuit-test pace \
                 derive from it",
                drift * 100.0,
            );
        }
    } else if policy_loaded && active_ticks >= PACE_PROBE_TICKS {
        // A loaded policy given the full probe budget and still unmeasurable means
        // zero pace on every probe bearing (or no rig height) — the charge keys just
        // VANISH from the wire and no consumer watches for absence, so the guard
        // would otherwise be defeated by silence (rl#280).
        tracing::warn!(
            "charge speed unmeasurable (rl#280): loaded policy, zero pace on every \
             pace-probe bearing (or no measurable rig height) — the wire's charge keys \
             are absent this eval; suspect a probe regression, not a slow brain"
        );
    }
    Ok(report)
}

fn run_compass(
    policy: &std::rc::Rc<Policy>,
    active_ticks: u64,
    target_distance: f32,
    pace_probe: bool,
    locale_xz: Vec2,
) -> CompassSweep {
    let mut per_bearing = [None; EVAL_BEARINGS];
    for (slot, bearing_rad) in per_bearing.iter_mut().zip(eval_bearings()) {
        *slot = Some(run_bearing(
            policy.clone(),
            active_ticks,
            target_distance,
            bearing_rad,
            pace_probe,
            locale_xz,
        ));
    }
    CompassSweep {
        target_distance_m: target_distance,
        per_bearing: per_bearing.map(|r| r.expect("compass and slots are the same length")),
    }
}

/// The rl#280 pace probe: [`run_compass`] with the probe's own geometry. Distance and
/// tick cap are owned HERE — the one place a probe sweep is assembled — so a probe at
/// the wrong distance (which [`EvalReport::measured_charge_heights_per_s`] would
/// silently refuse to read) is not constructible from `run_eval`. The `min` keeps a
/// caller's smaller tick budget (smoke tests) binding; a full-budget caller pays only
/// [`PACE_PROBE_TICKS`] per bearing — the pace prefix window is all the probe
/// measures, and this sweep rides every keep-best gate eval.
fn run_pace_probe(policy: &std::rc::Rc<Policy>, tick_budget: u64, locale_xz: Vec2) -> CompassSweep {
    run_compass(
        policy,
        tick_budget.min(PACE_PROBE_TICKS),
        PACE_PROBE_DISTANCE_M,
        true,
        locale_xz,
    )
}

/// The compass bearings in sweep order: bearing i = i·2π/[`EVAL_BEARINGS`].
/// `pub(crate)` so training-side per-bearing readouts (rl#276) label their bins from
/// THIS sequence rather than a second formula that could drift.
pub(crate) fn eval_bearings() -> impl Iterator<Item = f32> {
    (0..EVAL_BEARINGS).map(|i| i as f32 * std::f32::consts::TAU / EVAL_BEARINGS as f32)
}

/// Nearest compass bearing index for an arbitrary planar bearing in radians
/// (the [`polar_target`](crate::training::targets) convention: 0 = +X, toward +Z) —
/// how training episodes, whose bearings are sampled uniformly rather than swept,
/// bin onto [`eval_bearings`] for the rl#276 per-bearing reach tally. Lives beside
/// the compass definition so bin i and swept bearing i can never disagree.
pub(crate) fn bearing_bin(theta_rad: f32) -> usize {
    let step = std::f32::consts::TAU / EVAL_BEARINGS as f32;
    ((theta_rad / step).round().rem_euclid(EVAL_BEARINGS as f32)) as usize
}

/// One episode at one bearing — a fresh world per bearing, so each episode is exactly
/// the pre-compass eval (deterministic per brain) with the target rotated, spawned at
/// the sweep's locale (rl#293: the crab starts surface-placed at `locale_xz` via the
/// same [`InitialCrabLayout`](crate::bot::InitialCrabLayout) seam net's GCR restart
/// uses, so the eval locale rides the ONE spawn-layout path). The probe adds the
/// [`pace_recenter`] seam.
fn run_bearing(
    policy: std::rc::Rc<Policy>,
    active_ticks: u64,
    target_distance: f32,
    bearing_rad: f32,
    pace_probe: bool,
    locale_xz: Vec2,
) -> BearingReport {
    // The server world (rl#298 stage 6): the same constructor the trainer's rollout
    // env and net's headless host build from, vehicle layer included, so the eval
    // judges the env she trains in and is served in — no `VehicleControls` entries
    // means zero craft bodies, so today's sweeps measure the same world as before.
    // Every sweep — pace probe included — runs on the canonical terrain tile, the
    // only ground a loadable plant trains on (rl#293); targets, lure and recenter
    // are surface-aware.
    let mut app =
        headless_server_world(1, WorldRole::Standalone, crate::terrain::TerrainGrid::gcr());
    app.insert_resource(crate::bot::InitialCrabLayout {
        base_xz: locale_xz,
        spawns_m: vec![locale_xz],
    });
    app.insert_resource(EvalConfig {
        target_distance,
        bearing_rad,
        settle_ticks: RESET_GRACE_TICKS as u64,
        pace_probe,
    })
    .init_resource::<EvalState>()
    .insert_non_send_resource(policy)
    .add_systems(FixedUpdate, eval_step.in_set(BotSet::Think));
    if pace_probe {
        // Crab, real ball, and posed lure all carry by one delta, so measurement and
        // obs frame are both invariant across the shift; before(eval_step) only pins
        // WHICH tick re-poses the lure post-shift, for determinism.
        app.add_systems(
            FixedUpdate,
            pace_recenter.in_set(BotSet::Think).before(eval_step),
        );
    }

    force_serial_schedules(&mut app);
    app.finish();
    app.cleanup();

    let settle_ticks = RESET_GRACE_TICKS as u64;
    let max_updates = settle_ticks + active_ticks + 64;
    let mut updates = 0u64;
    while active_torque_ticks(&app) < active_ticks && updates < max_updates {
        app.update();
        updates += 1;
    }

    let state = app
        .world()
        .get_resource::<EvalState>()
        .expect("eval state present");
    let progress_m = (state.initial_dist - state.closest_dist).max(0.0);
    let mean_torque_per_tick = if state.torque_ticks > 0 {
        (state.torque_sum / state.torque_ticks as f64) as f32
    } else {
        0.0
    };
    // One binding feeds both fields, so `reached ⇔ tip_touch(closest_tip_distance_m)`
    // holds structurally (`tip_touch(INFINITY)` = false covers the no-tip-seen case).
    let closest_tip = state.closest_tip_dist.unwrap_or(f32::INFINITY);
    BearingReport {
        bearing_rad,
        progress_m,
        total_torque: state.torque_sum as f32,
        mean_torque_per_tick,
        saturation: mean_torque_per_tick / total_drive_torque_ceiling(),
        work_j: state.work_sum as f32,
        initial_distance_m: state.initial_dist,
        closest_distance_m: state.closest_dist,
        final_distance_m: state.last_dist,
        closest_tip_distance_m: closest_tip,
        reached: tip_touch(closest_tip),
        active_ticks: state.torque_ticks,
        sustained_pace_m_per_s: state.best_pace_m_per_s,
    }
}

fn active_torque_ticks(app: &App) -> u64 {
    app.world()
        .get_resource::<EvalState>()
        .map(|s| s.torque_ticks)
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn eval_step(
    policy: NonSend<std::rc::Rc<Policy>>,
    cfg: Res<EvalConfig>,
    mut state: ResMut<EvalState>,
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
    mut targets: ResMut<CrabTargets>,
    obs: Res<CrabObservation>,
    mut actions: ResMut<CrabActions>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    claw_tips_q: Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
    joints: Query<(&CrabJoint, &CrabEnvId)>,
) {
    if state.real_target.is_none()
        && let Some(slot) = targets.envs.first_mut()
    {
        let origin = spawns.origin(0);
        let p = polar_target(origin, cfg.bearing_rad, cfg.target_distance, TARGET_Y);
        let real = terrain.place(Vec2::new(p.x, p.z), TARGET_Y);
        state.real_target = Some(real);
        // In probe mode the crab is still at its origin, so the lure is exact here.
        *slot = Some(if cfg.pace_probe {
            probe_lure(origin, real, &terrain)
        } else {
            real
        });
    }
    let Some(target) = state.real_target else {
        state.tick += 1;
        return;
    };

    let settling = state.tick < cfg.settle_ticks;

    // Skips are deliberate: env 0 unsized = the pre-spawn window.
    if settling {
        let _ = actions.rest(0);
    } else if let Some(o) = obs.rows().first() {
        let _ = actions.set_row(0, policy.act(o));
    } else {
        let _ = actions.rest(0);
    }

    if let Some(cpos) = carapace_q
        .iter()
        .find(|(e, _)| e.0 == 0)
        .map(|(_, t)| t.translation)
        .filter(|p| p.is_finite())
    {
        // Probe mode re-poses the lure every tick from her CURRENT position — net's
        // `set_crab_walk_target` cadence — so the obs ball recedes ahead of her until
        // the real ball comes inside the band, then converges onto it.
        if cfg.pace_probe
            && let Some(slot) = targets.envs.first_mut()
        {
            *slot = Some(probe_lure(cpos, target, &terrain));
        }
        let d = dist_3d(cpos, target);
        if settling {
            state.initial_dist = d;
            state.closest_dist = d;
        } else {
            state.closest_dist = state.closest_dist.min(d);
            let elapsed_s = state.torque_ticks as f32 / crate::physics::PHYSICS_HZ as f32;
            if elapsed_s >= PACE_WINDOW_MIN_S {
                state.best_pace_m_per_s = state
                    .best_pace_m_per_s
                    .max((state.initial_dist - d) / elapsed_s);
            }
        }
        state.last_dist = d;
    }

    // Settle ticks are excluded so a spawn transient can't score a touch the
    // policy never earned.
    if !settling && let Some(d) = closest_tip_dist(0, target, &claw_tips_q) {
        state.closest_tip_dist = Some(state.closest_tip_dist.map_or(d, |cur| cur.min(d)));
    }

    if !settling && !actions.is_empty() {
        // Pure observation of what Sense/Think already computed (rl#279): commanded
        // torque × the sensor's measured hinge rate, so the accumulator can never
        // perturb the rollout it measures.
        let rates = obs.env(0);
        let mut tick_torque = 0.0f32;
        let mut tick_power = 0.0f32;
        for (joint, env) in joints.iter() {
            if env.0 != 0 {
                continue;
            }
            if let Some(drive) = actions.drive(0, joint.id) {
                let torque = applied_torque(joint.id, drive);
                tick_torque += torque.abs();
                if let Some(view) = &rates {
                    tick_power += (torque * view.joint_rate(joint.id)).abs();
                }
            }
        }
        state.torque_sum += tick_torque as f64;
        state.work_sum += (tick_power / crate::physics::PHYSICS_HZ as f32) as f64;
        state.torque_ticks += 1;
    }

    state.tick += 1;
}

/// The pace probe's posed obs target: [`band_lure`] of the real ball — beyond the
/// band the policy sees a ball a constant band edge ahead on the true bearing;
/// inside it (every probe ball, since rl#292 widened the band), the real ball
/// itself. The posed point is re-landed at the real
/// ball's height-above-surface over the LURE'S own ground: its absolute y is surface +
/// band at the distant xz, which on a slope can sit meters off the trained y band at
/// the lure's xz — an OOD obs target, the exact artifact the lure exists to prevent.
/// Flat grids (and an in-band real ball, same xz): bit-identical passthrough.
fn probe_lure(carapace: Vec3, real_target: Vec3, terrain: &crate::terrain::TerrainGrid) -> Vec3 {
    let cxz = Vec2::new(carapace.x, carapace.z);
    let to_real = Vec2::new(real_target.x - carapace.x, real_target.z - carapace.z);
    let posed = band_lure(cxz, to_real);
    terrain.place(
        posed,
        real_target.y - terrain.height(real_target.x, real_target.z),
    )
}

/// The rl#240 recenter, probe-side: past [`recenter_delta`]'s band edge, shift every
/// crab part back onto the spawn origin planar-wise — production's sanctioned
/// multibody teleport (rl#116) — and carry the real ball AND the posed lure by the
/// same delta, so both the remaining chase geometry (every distance eval_step
/// measures) and the obs frame are invariant across the shift. Without this the
/// probe's farther ball walks the spawn-relative body.pos obs channel out of the
/// training band, and the pace measured would be an OOD artifact, not her gait. On
/// the flat grids the shift is an exact ground symmetry; on a terrain plant the y
/// carry is carapace-exact only — the re-landed feet meet differently-sloped ground
/// and physics resolves the mismatch over the next ticks, an accepted approximation
/// (the alternative, no recenter, is the guaranteed OOD artifact).
///
/// Deliberately a TELEPORT, not net's origin rebase (rl#281 stage 6): the probe is a
/// MEASUREMENT — teleporting keeps the whole trek on the spawn's fixed locale, so
/// pace numbers are comparable across checkpoints and reproducible per seed, where a
/// rebase would let each run wander its own terrain path. Net inverts the trade: a
/// rendered world must stay glued under her feet, and nothing there is a measurement.
fn pace_recenter(
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
    mut state: ResMut<EvalState>,
    mut targets: ResMut<CrabTargets>,
    mut parts: Query<(&CrabEnvId, &mut Transform, Has<CrabCarapace>), With<CrabBodyPart>>,
) {
    // real_target set ⇒ env slots exist ⇒ origin(0) is wired (rl#242).
    let Some(real_target) = state.real_target else {
        return;
    };
    let Some(carapace) = parts
        .iter()
        .find(|(env, _, cara)| env.0 == 0 && *cara)
        .map(|(_, t, _)| t.translation)
        .filter(|p| p.is_finite())
    else {
        return;
    };
    let Some(delta) = recenter_delta(spawns.origin(0), carapace, &terrain) else {
        return;
    };
    for (env, mut t, _) in parts.iter_mut() {
        if env.0 == 0 {
            t.translation += delta;
        }
    }
    state.real_target = Some(real_target + delta);
    if let Some(slot) = targets.envs.first_mut().and_then(|s| s.as_mut()) {
        *slot += delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::targets::REACH_RADIUS;

    /// The rl#276 binning contract: every swept compass bearing bins to its own
    /// index — including off the compass (nearest wins, ties round away from zero)
    /// and for the negative-angle aliases `atan2` hands training episodes.
    #[test]
    fn bearing_bin_agrees_with_the_swept_compass() {
        let step = std::f32::consts::TAU / EVAL_BEARINGS as f32;
        for (i, bearing) in eval_bearings().enumerate() {
            assert_eq!(bearing_bin(bearing), i, "swept bearing {i} is its own bin");
            assert_eq!(
                bearing_bin(bearing + 0.4 * step),
                i,
                "nearest below midpoint"
            );
            assert_eq!(
                bearing_bin(bearing - 0.4 * step),
                i,
                "nearest above midpoint"
            );
            assert_eq!(
                bearing_bin(bearing - std::f32::consts::TAU),
                i,
                "the atan2 negative alias of bearing {i} bins identically"
            );
        }
        assert_eq!(
            bearing_bin(-step),
            EVAL_BEARINGS - 1,
            "-45° is the last bin"
        );
    }

    /// A canned report whose PACE sweep runs at `pace_distance_m` with ONE bearing
    /// pacing at `pace` (arena m/s) — the speed guard's inputs, nothing else non-zero.
    fn paced_report(policy_loaded: bool, pace_distance_m: f32, pace: f32) -> EvalReport {
        let bearing_at = |distance: f32| BearingReport {
            bearing_rad: 0.0,
            progress_m: 0.0,
            total_torque: 0.0,
            mean_torque_per_tick: 0.0,
            saturation: 0.0,
            work_j: 0.0,
            initial_distance_m: distance,
            closest_distance_m: distance,
            final_distance_m: distance,
            closest_tip_distance_m: f32::INFINITY,
            reached: false,
            active_ticks: DEFAULT_EVAL_TICKS,
            sustained_pace_m_per_s: 0.0,
        };
        let sweep_at = |distance: f32| CompassSweep {
            target_distance_m: distance,
            per_bearing: [bearing_at(distance); EVAL_BEARINGS],
        };
        let mut paced = bearing_at(pace_distance_m);
        paced.sustained_pace_m_per_s = pace;
        let mut pace_sweep = sweep_at(pace_distance_m);
        pace_sweep.per_bearing[3] = paced;
        EvalReport {
            policy_loaded,
            far: [sweep_at(DEFAULT_TARGET_DISTANCE_M); EVAL_LOCALES],
            close: sweep_at(CLOSE_PROBE_DISTANCE_M),
            pace: pace_sweep,
        }
    }

    /// Pins the rl#266 guard: drift is measured pace over the pin (best pace-probe
    /// bearing, in body heights), and every unmeasurable case is None — NOT a
    /// spurious drift.
    #[test]
    fn charge_speed_guard_measures_and_refuses() {
        let h = crate::mesh_fallback::natural_body_height().expect("rig height measures");
        let fast = CRAB_CHARGE_SPEED_HEIGHTS_PER_S * h * 1.5;

        let r = paced_report(true, PACE_PROBE_DISTANCE_M, fast);
        let drift = r.charge_speed_drift().expect("measurable");
        assert!((drift - 0.5).abs() < 1e-3, "drift {drift} should be +50%");
        assert!(drift > CHARGE_SPEED_DRIFT_TOL);

        // A pace sweep off the probe distance: the saturation ceiling moved — a
        // geometry artifact, not her gait; comparing it to the pin would invite a
        // corrupting re-pin.
        let short = paced_report(true, PACE_PROBE_DISTANCE_M - 1.0, fast);
        assert_eq!(short.measured_charge_heights_per_s(), None);

        // The rest-pose baseline measures the baseline, not her.
        let unloaded = paced_report(false, PACE_PROBE_DISTANCE_M, fast);
        assert_eq!(unloaded.measured_charge_heights_per_s(), None);

        // No progress at all: nothing to measure.
        let parked = paced_report(true, PACE_PROBE_DISTANCE_M, 0.0);
        assert_eq!(parked.measured_charge_heights_per_s(), None);
    }

    /// The instrument saturates at probe-distance/[`PACE_WINDOW_MIN_S`] (a brain
    /// arriving inside the pace window measures the ceiling, not its gait). If the
    /// pin's drift band ever reaches that ceiling, a fast retrain would read "within
    /// tolerance" and the guard would be silently defeated in exactly the direction
    /// (faster) history shows happens — so pin the headroom. rl#280 acceptance bar:
    /// the probe ball moves farther BEFORE this fails, never after.
    #[test]
    fn charge_speed_guard_keeps_saturation_headroom() {
        let h = crate::mesh_fallback::natural_body_height().expect("rig height measures");
        let ceiling_heights_per_s = PACE_PROBE_DISTANCE_M / PACE_WINDOW_MIN_S / h;
        assert!(
            ceiling_heights_per_s
                > CRAB_CHARGE_SPEED_HEIGHTS_PER_S * (1.0 + CHARGE_SPEED_DRIFT_TOL),
            "instrument ceiling {ceiling_heights_per_s} heights/s is inside the drift band — \
             the pace probe needs a farther ball before this pin can be trusted"
        );
    }

    /// The rl#280 probe's geometry: the episode covers the pace window with room for
    /// the prefix max to move (a probe shorter than the window would measure
    /// nothing, silently). The lure/recenter seams themselves are pinned where they
    /// live, `training::targets`.
    #[test]
    fn pace_probe_covers_the_window() {
        let window_ticks = PACE_WINDOW_MIN_S * crate::physics::PHYSICS_HZ as f32;
        assert!(PACE_PROBE_TICKS as f32 >= 2.0 * window_ticks);
    }

    /// Pins the rl#279 J/m guard: below the progress floor no cost of transport is
    /// reported (work over ~zero distance describes the guard, not the gait); at or
    /// above it the number is exactly work over net progress. And the compass mean
    /// averages only the bearings that measured one.
    #[test]
    fn j_per_m_is_guarded_by_the_progress_floor() {
        let mut r = paced_report(true, DEFAULT_TARGET_DISTANCE_M, 0.0);
        for b in &mut r.far[0].per_bearing {
            b.work_j = 10.0;
        }
        r.far[0].per_bearing[0].progress_m = WORK_PER_M_MIN_PROGRESS_M - 0.01;
        assert_eq!(r.far[0].per_bearing[0].j_per_m(), None);
        r.far[0].per_bearing[1].progress_m = WORK_PER_M_MIN_PROGRESS_M;
        assert_eq!(
            r.far[0].per_bearing[1].j_per_m(),
            Some(10.0 / WORK_PER_M_MIN_PROGRESS_M)
        );
        r.far[0].per_bearing[2].progress_m = 4.0;
        assert_eq!(r.far[0].per_bearing[2].j_per_m(), Some(2.5));
        // Mean over the two measured bearings only — the six guarded ones don't drag it.
        let mean = r.far[0].mean_j_per_m().expect("two bearings measured");
        assert!((mean - (20.0 + 2.5) / 2.0).abs() < 1e-4);

        // All bearings guarded → no sweep mean at all, never a 0/0.
        let parked = paced_report(true, DEFAULT_TARGET_DISTANCE_M, 0.0);
        assert_eq!(parked.far[0].mean_j_per_m(), None);
        assert_eq!(parked.far[0].mean_saturation(), 0.0);
    }

    #[test]
    fn default_far_distance_sits_inside_band_and_horizon() {
        use crate::training::targets::{BAND_MAX_M, DRIFT_REBASE_M};
        const {
            assert!(
                DEFAULT_TARGET_DISTANCE_M > REACH_RADIUS,
                "the ball must start FAR — well outside the reach radius"
            );
            // rl#292/rl#293: the far ball is pace-pinned and in-band (in-distribution
            // obs) — NOT the band edge: an unreachable ball would make min-progress
            // measure the locale's worst obstruction instead of her chase. Past the
            // drift radius only so the sweep covers body.pos states beyond net's 9 m
            // rebase window (the far sweep itself runs no recenter, like training).
            assert!(DEFAULT_TARGET_DISTANCE_M < BAND_MAX_M);
            assert!(DEFAULT_TARGET_DISTANCE_M > DRIFT_REBASE_M);
            assert!(TARGET_Y > TARGET_Y_MIN && TARGET_Y < TARGET_Y_MAX);
        }
        // Within-horizon arrival at the pinned pace (flat-ground bound): the far
        // ball must be reachable inside one episode with real margin.
        let h = crate::mesh_fallback::natural_body_height().expect("rig height measures");
        let flat_traverse = CRAB_CHARGE_SPEED_HEIGHTS_PER_S * h * (DEFAULT_EVAL_TICKS as f32)
            / crate::physics::PHYSICS_HZ as f32;
        assert!(
            DEFAULT_TARGET_DISTANCE_M < 0.75 * flat_traverse,
            "far ball {DEFAULT_TARGET_DISTANCE_M} m leaves no arrival margin against the \
             pinned-pace horizon traverse {flat_traverse} m — re-pin from pace evidence"
        );
    }

    #[test]
    fn compass_covers_bearings_and_keeps_the_historical_first() {
        let origin = Vec3::new(2.0, 0.0, -3.0);
        let d = DEFAULT_TARGET_DISTANCE_M;
        let targets: Vec<Vec3> = eval_bearings()
            .map(|b| polar_target(origin, b, d, TARGET_Y))
            .collect();
        assert_eq!(targets.len(), EVAL_BEARINGS);

        // Bearing 0 IS the pre-compass eval's +X pose — the historical curve stays
        // comparable at that bearing.
        let first = targets[0];
        assert!((first.x - (origin.x + d)).abs() < 1e-4 && (first.z - origin.z).abs() < 1e-4);

        for (i, t) in targets.iter().enumerate() {
            let planar = Vec2::new(t.x - origin.x, t.z - origin.z);
            assert!(
                (planar.length() - d).abs() < 1e-3,
                "bearing {i} target sits at the eval distance"
            );
            assert_eq!(t.y, TARGET_Y);
            for (j, u) in targets.iter().enumerate().skip(i + 1) {
                assert!(
                    (*t - *u).length() > 1.0,
                    "bearings {i} and {j} must pose distinct targets"
                );
            }
        }
    }

    #[test]
    #[ignore = "builds a headless bevy+rapier App per bearing; run with --ignored"]
    fn rest_pose_has_zero_torque_and_no_progress() {
        let dir = std::env::temp_dir().join(format!("rl-eval-restpose-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Explicit, greppable test-only opt-in: this eval deliberately runs whatever
        // body the test env constructs (usually the fallback — no sally.glb in CI).
        let r = run_eval(
            crate::mesh_fallback::BodyGate::FallbackAllowed,
            &dir,
            200,
            DEFAULT_TARGET_DISTANCE_M,
        )
        .expect("an absent checkpoint is the legitimate baseline, never a refusal");

        assert!(!r.policy_loaded, "an empty dir loads no policy (rest pose)");
        assert!(!r.reached(), "a rest-pose crab never reaches a far ball");
        // Zero floor: under the tip-based touch (rl#253) even the real Sally's
        // full-episode slump never brings a claw tip near the ball (see
        // CLOSE_PROBE_DISTANCE_M), so rest-pose reached_count is 0 on every body.
        assert_eq!(r.close.reached_count(), 0);
        assert_eq!(r.close.target_distance_m, CLOSE_PROBE_DISTANCE_M);
        for b in &r.close.per_bearing {
            assert_eq!(b.total_torque, 0.0);
            assert!(
                b.initial_distance_m > REACH_RADIUS,
                "close probe starts outside reach ({} m) at bearing {:.0}°",
                b.initial_distance_m,
                b.bearing_rad.to_degrees()
            );
            assert!(
                b.initial_distance_m < DEFAULT_TARGET_DISTANCE_M,
                "close probe is the CLOSE sweep, not another far one"
            );
        }
        assert!(
            (0.0..1.0).contains(&r.progress_m()),
            "rest-pose progress should be ~0, got {} m",
            r.progress_m()
        );
        for b in r.far.iter().flat_map(|s| &s.per_bearing) {
            assert_eq!(
                b.total_torque, 0.0,
                "the rest pose applies no joint torque, so total_torque must be exactly 0"
            );
            assert_eq!(b.mean_torque_per_tick, 0.0);
            assert_eq!(b.saturation, 0.0, "zero drive saturates nothing");
            assert_eq!(b.work_j, 0.0, "zero torque does zero mechanical work");
            // Passive slump CAN clear the J/m floor (~0.55 m on some bodies) —
            // measurable or not, zero work means a zero cost of transport.
            assert_eq!(b.j_per_m().unwrap_or(0.0), 0.0);
            assert_eq!(b.active_ticks, 200, "all active ticks are measured");
            assert!(
                b.initial_distance_m.is_finite() && b.closest_distance_m.is_finite(),
                "distances are real finite metres"
            );
            assert!(
                b.initial_distance_m > REACH_RADIUS,
                "the ball starts far outside reach ({} m) at bearing {:.0}°",
                b.initial_distance_m,
                b.bearing_rad.to_degrees()
            );
            assert!(
                (0.0..1.0).contains(&b.progress_m),
                "rest pose shuffles nowhere at bearing {:.0}°, got {} m",
                b.bearing_rad.to_degrees(),
                b.progress_m
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
