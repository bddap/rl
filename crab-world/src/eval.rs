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
use crate::bot::{BotSet, CrabRescued, CrabSpawns};
use crate::policy::Policy;
use crate::training::reward::dist_3d;
use crate::training::targets::{
    BAND_MAX_M, BAND_START_MIN, REACH_RADIUS, TARGET_Y_MAX, TARGET_Y_MIN, band_lure,
    closest_tip_dist, polar_target, recenter_delta, tip_touch,
};

/// The far sweep's ball distance — pinned from measured pace, NOT the band edge
/// (rl#292/rl#293): at the 128 m band edge an unreachable ball would make the pair's
/// progress measure the ray's worst obstruction instead of her chase. 24 m ≈ 65% of
/// the pinned charge's flat-ground horizon traverse
/// ([`CRAB_CHARGE_SPEED_HEIGHTS_PER_S`] ≈ 1.6 m/s × [`DEFAULT_EVAL_TICKS`] ≈ 23 s ≈
/// 37 m), so arrival stays measurable for a competent brain even on relief, and
/// mid-band for the obs. NOTE: the far sweep runs NO mid-episode recenter — matching
/// the training regime; only the pace probe wires the `pace_recenter` seam
/// (fixed-locale measurement, see its doc). Re-pin from pace evidence when the gait
/// regime moves; numbers across a distance re-pin are incomparable by design.
pub const DEFAULT_TARGET_DISTANCE_M: f32 = 24.0;

/// Max grade (|rise|/run, both directions) a scored far ray may show anywhere along
/// it — the rl#328 progressability bar, applied PER (heading, start) PAIR since the
/// mean-pair headline: every scored ray is geometrically walkable and not a free
/// slide (see [`bearing_progressable`] for why the bound is symmetric), so terrain
/// contributes neither forced zeros nor gravity-donated metres to the average — a
/// low mean is a policy verdict, never a terrain lottery. 0.36 ≈ 20° is deliberately
/// conservative — far below the 55° zero-input HOLD the contact fix guarantees
/// (`slope_hold_test`) and moderate for a competent walker. Rejecting more terrain
/// only flattens the arena; accepting an unclimbable ray hands the mean a
/// structural zero no policy can close, so err strict.
pub const MAX_PROGRESSABLE_GRADE: f32 = 0.36;

/// Ray-sampling stride for the progressability check — about one body length, dense
/// against the bake's 30 m cells (the sampler is triangle-exact, so features
/// narrower than a cell don't exist to miss).
const PROGRESSABLE_STEP_M: f32 = 1.0;

/// Whether a policy can make CONTROLLED progress along `bearing_rad` from `origin`:
/// no step of the ray to `distance_m` demands more than [`MAX_PROGRESSABLE_GRADE`]
/// in EITHER direction. Samples through [`polar_target`] — the SAME formula that
/// poses the ball — so the checked ray and the swept ray can never fork. The bound
/// is symmetric since the rl#341 mean-pair flip: uphill past the bar is unwalkable
/// (the rl#328 forced zero), and downhill past it is a luge run — a rest-pose body
/// SLIDES metres toward the ball for free (measured: 2.4 m at 200 ticks on a
/// per-ray-drawn start), which a MEAN monetizes directly where the retired min
/// ignored it. The all-8-bearings locale predicate this replaces bounded downhill
/// implicitly (the opposite bearing's uphill bound IS this bearing's downhill
/// bound); one scored ray must carry both bounds itself.
fn bearing_progressable(
    grid: &crate::terrain::TerrainGrid,
    origin: Vec2,
    bearing_rad: f32,
    distance_m: f32,
) -> bool {
    let origin3 = Vec3::new(origin.x, 0.0, origin.y);
    let steps = (distance_m / PROGRESSABLE_STEP_M).ceil().max(1.0) as usize;
    let step = distance_m / steps as f32;
    let mut prev = grid.height(origin.x, origin.y);
    for i in 1..=steps {
        let p = polar_target(origin3, bearing_rad, step * i as f32, 0.0);
        let h = grid.height(p.x, p.z);
        if (h - prev).abs() > MAX_PROGRESSABLE_GRADE * step {
            return false;
        }
        prev = h;
    }
    true
}

/// Far-sweep starts drawn per compass heading — [`EVAL_BEARINGS`] × this =
/// [`EVAL_PAIRS`] scored (heading, start) pairs. Four starts per heading keeps the
/// far sweep near the historical 3-locale × 8-bearing episode budget (32 vs 24
/// episodes) while sampling 32 DISTINCT terrain starts instead of 3.
pub const EVAL_STARTS_PER_HEADING: usize = 4;

/// The far sweep's (heading, start) pair count — the mean-progress headline's sample
/// size (owner 08-03: average progress over many headings over many deterministically
/// chosen starting positions).
pub const EVAL_PAIRS: usize = EVAL_BEARINGS * EVAL_STARTS_PER_HEADING;

/// How many deterministic start candidates [`eval_pairs`] may consume before it
/// REFUSES: a terrain (bake × amplitude) so rough that the sunflower exhausts this
/// many candidates without seating every heading is a broken instrument, and the
/// eval must fail loud rather than score rock (rl#328). The refusal is an `Err`, not
/// a panic — the CLI dies with a message and no `EVAL_RESULT`, and the keep-best
/// gate logs at error level and protects the incumbent.
const START_CANDIDATES_MAX: usize = 4096;

/// One far-sweep episode assignment: drive toward a ball `bearing_rad` off +X from a
/// surface spawn at `start_xz`.
#[derive(Debug, Clone, Copy)]
pub struct EvalPair {
    pub start_xz: Vec2,
    pub bearing_rad: f32,
}

/// The far sweep's (heading, start) pairs: headings cycle the fixed compass
/// ([`eval_bearings`]); starts come from a deterministic golden-angle sunflower over
/// the tile interior, each start taken by the FIRST candidate whose own ray is
/// progressable for that pair's heading ([`bearing_progressable`]). Pure function of
/// (grid, far distance): same bake and amplitude ⇒ same pairs, so the instrument
/// moves only when the ground does — which was already the re-baseline boundary.
/// Distinct starts by construction (each candidate is consumed at most once);
/// cross-env physics contamination is impossible regardless (envs are
/// collision-isolated by group, `bot::body::collision`). Errs when the candidate
/// budget exhausts — the loud high-amplitude refusal.
fn eval_pairs(
    grid: &crate::terrain::TerrainGrid,
    far_distance_m: f32,
) -> Result<Vec<EvalPair>, String> {
    // Keep every start AND its full band (targets, lure clamp) clear of the tile
    // edge, matching the training sampler's edge margin.
    let margin = 2.0 * BAND_MAX_M + far_distance_m.max(BAND_MAX_M);
    let radius = grid.extent_x().min(grid.extent_z()) / 2.0 - margin;
    if radius <= 0.0 {
        return Err(format!(
            "terrain tile ({:.0}×{:.0} m) is smaller than the eval's edge margin \
             ({margin:.0} m) — no interior to draw starts from",
            grid.extent_x(),
            grid.extent_z(),
        ));
    }
    // Golden-angle sunflower: uniform areal density over the interior disc, no two
    // candidates coincident, deterministic.
    let golden = std::f32::consts::TAU * (1.0 - 0.618_034);
    let bearings: Vec<f32> = eval_bearings().collect();
    let mut pairs = Vec::with_capacity(EVAL_PAIRS);
    let mut k = 0usize;
    for i in 0..EVAL_PAIRS {
        let bearing_rad = bearings[i % EVAL_BEARINGS];
        loop {
            k += 1;
            if k > START_CANDIDATES_MAX {
                return Err(format!(
                    "no progressable start for heading {:.0}° within {START_CANDIDATES_MAX} \
                     deterministic candidates (max uphill grade {MAX_PROGRESSABLE_GRADE}, \
                     distance {far_distance_m} m) — the terrain (bake or amplitude) is too \
                     rough for the instrument; refusing to score rock (rl#328)",
                    bearing_rad.to_degrees(),
                ));
            }
            let r = radius * (k as f32 / START_CANDIDATES_MAX as f32).sqrt();
            let start_xz = r * Vec2::from_angle(k as f32 * golden);
            if bearing_progressable(grid, start_xz, bearing_rad, far_distance_m) {
                pairs.push(EvalPair {
                    start_xz,
                    bearing_rad,
                });
                break;
            }
        }
    }
    Ok(pairs)
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

/// The default eval episode length PER PAIR — one training episode horizon (~23 s of
/// crab time at 64 Hz), enough for a working gait to traverse a far target. The eval's
/// instrument is the (ticks, distance, compass, starts, amplitude, settle) tuple —
/// `--ticks`/`--distance`/`--terrain-amplitude` move the first three knobs, the starts
/// derive from (bake × amplitude, distance), and the settle window borrows
/// [`RESET_GRACE_TICKS`] (a bot-module constant: touching it re-bases `initial_dist`
/// and every historical progress number). Every judge — the CLI (release gate,
/// monitor) and the trainer's keep-best gate — must take its defaults from HERE or the
/// metric forks.
pub const DEFAULT_EVAL_TICKS: u64 = crate::training::systems::MAX_EPISODE_TICKS as u64;

/// The fixed compass of target bearings every sweep cycles, uniform in [0, 2π) with
/// bearing 0 = +X (the single bearing the pre-rl#239 eval judged, kept so bearing-0
/// numbers stay comparable to the historical curve). Training samples bearing
/// uniformly (`targets::sample_target`), so a brain competent at one bearing only is
/// a training pathology the eval must EXPOSE: since the mean-pair headline (owner
/// 08-03) that exposure is the per-heading means + the min pair + the rescued-pair
/// count reported BESIDE the mean — a dead heading drags the average by its
/// [`EVAL_STARTS_PER_HEADING`] share and names itself in the tail statistics, where
/// the retired min-over-bearings headline flattened the whole learning phase to a
/// gradient-free 0.000 instead (the rl#239 blindness inverted).
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
/// [`pace_recenter`] recenter, matching the re-gauge regime GCR play happens in
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

/// One pair's episode — the same measurements the pre-compass eval reported for its
/// single +X episode, plus the pair's own start and its rescue tally.
#[derive(Debug, Clone, Copy)]
pub struct BearingReport {
    pub bearing_rad: f32,
    /// The pair's spawn locale (far sweep: the sunflower start; probes: tile center).
    pub start_xz: Vec2,
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
    /// How many times `rescue_lost_crabs` teleported this pair's crab back to its
    /// origin mid-episode (non-finite pose, through-terrain, buried). A rescue
    /// invalidates the trajectory as a chase, so the fold FORFEITS the pair's
    /// `progress_m` and pace to 0 (bddap/rl#341 S1-1: a crab that exploded at 12 m
    /// used to keep the 12 m and get promoted, silently) — the distances stay as
    /// measured for diagnosis, and the tally rides the wire.
    pub rescues: u32,
    /// Best prefix pace toward the target (arena m/s): max over active ticks past
    /// [`PACE_WINDOW_MIN_S`] of progress-so-far / time-so-far. The max is taken right
    /// when she arrives at the ball, so parking there for the rest of the episode
    /// (which dilutes `progress_m / active_ticks`) can't understate her charge. The
    /// rl#266 speed guard reads the PACE sweep's values only (rl#280 — the far/close
    /// sweeps saturate too low to instrument her); theirs are diagnostic. 0.0 when
    /// she never progressed (or was rescued).
    pub sustained_pace_m_per_s: f32,
}

impl BearingReport {
    /// Cost of transport (rl#279): total mechanical work over net chase progress,
    /// `None` below [`WORK_PER_M_MIN_PROGRESS_M`] — a near-zero denominator would
    /// print an ∞-ish number that describes the guard, not the gait.
    pub fn j_per_m(&self) -> Option<f32> {
        (self.progress_m >= WORK_PER_M_MIN_PROGRESS_M).then(|| self.work_j / self.progress_m)
    }

    /// SIGNED net chase progress (rl#310): initial minus FINAL distance, negative when
    /// the episode ends farther from the ball than it started. Diagnostic only —
    /// `progress_m` (best-approach over a running-min, zero-floored) stays the
    /// headline input and the keep-best bar, but it is ≥ 0 by construction, so a flat
    /// 0.0 can't distinguish a parked crab from one actively sliding away; this sign
    /// can. Derived, not stored: one source of truth with the two distances it
    /// reports on.
    pub fn net_progress_m(&self) -> f32 {
        self.initial_distance_m - self.final_distance_m
    }

    /// Whether this pair's episode was invalidated by a mid-episode rescue (S1-1).
    pub fn rescued(&self) -> bool {
        self.rescues > 0
    }
}

/// The far sweep: [`EVAL_PAIRS`] (heading, start) episodes at one target distance —
/// the distance travels with the sweep's data so a printed line can never pair one
/// sweep's numbers with another's distance.
#[derive(Debug, Clone)]
pub struct PairSweep {
    pub target_distance_m: f32,
    pub pairs: Vec<BearingReport>,
}

impl PairSweep {
    /// THE headline input: MEAN over pairs of zero-floored best-approach progress
    /// (owner 08-03). Every scored ray is geometrically progressable BY CONSTRUCTION
    /// ([`eval_pairs`]) and a rescued pair contributes a forfeited 0, so a low mean
    /// is a policy that didn't walk — never terrain that couldn't be walked, and
    /// never an explosion laundered into progress.
    pub fn mean_progress_m(&self) -> f32 {
        self.pairs.iter().map(|p| p.progress_m).sum::<f32>() / self.pairs.len().max(1) as f32
    }

    /// Mean SIGNED net progress (rl#310 in aggregate): unlike the zero-floored mean,
    /// this goes negative when pairs actively flee, so parked-vs-retreating stays
    /// decomposable after the fold.
    pub fn mean_net_progress_m(&self) -> f32 {
        self.pairs.iter().map(|p| p.net_progress_m()).sum::<f32>() / self.pairs.len().max(1) as f32
    }

    /// The WORST pair — the tail statistic reported beside the mean so a directional
    /// or locale-specific hole can't hide in the average (the anti-gaming property
    /// the retired min headline bought, kept as observability instead of as the
    /// score). `total_cmp` for a deterministic pick; `progress_m` is never NaN.
    pub fn min_pair(&self) -> &BearingReport {
        self.pairs
            .iter()
            .min_by(|a, b| a.progress_m.total_cmp(&b.progress_m))
            .expect("the far sweep is non-empty")
    }

    /// Pairs forfeited by a mid-episode rescue (S1-1) — nonzero means the physics
    /// faulted under this brain and the mean carries forfeited zeros.
    pub fn rescued_pairs(&self) -> usize {
        self.pairs.iter().filter(|p| p.rescued()).count()
    }

    /// How many pairs reached the ball (claw-tip touch).
    pub fn reached_count(&self) -> usize {
        self.pairs.iter().filter(|p| p.reached).count()
    }

    /// Mean torque saturation over the pairs (rl#279) — effort observability.
    pub fn mean_saturation(&self) -> f32 {
        self.pairs.iter().map(|p| p.saturation).sum::<f32>() / self.pairs.len().max(1) as f32
    }

    /// Mean total commanded |torque| over the pairs — the wire's `total_torque=`
    /// under the mean-pair schema (the monitor requires the key; a single worst
    /// episode's torque has no referent under an average).
    pub fn mean_total_torque(&self) -> f32 {
        self.pairs.iter().map(|p| p.total_torque).sum::<f32>() / self.pairs.len().max(1) as f32
    }

    /// Mean cost of transport over the pairs that measured one (rl#279) — `None`
    /// when no pair cleared [`WORK_PER_M_MIN_PROGRESS_M`].
    pub fn mean_j_per_m(&self) -> Option<f32> {
        let measured: Vec<f32> = self.pairs.iter().filter_map(|p| p.j_per_m()).collect();
        (!measured.is_empty()).then(|| measured.iter().sum::<f32>() / measured.len() as f32)
    }

    /// Compact per-HEADING mean-progress readout for log lines (rl#276): the
    /// directional profile the flat mean collapses, so a dead heading names itself
    /// in train.log at onset instead of after a forensic per-pair job.
    pub fn progress_line(&self) -> String {
        eval_bearings()
            .map(|b| {
                let (sum, n) = self
                    .pairs
                    .iter()
                    .filter(|p| bearing_bin(p.bearing_rad) == bearing_bin(b))
                    .fold((0.0f32, 0usize), |(s, n), p| (s + p.progress_m, n + 1));
                format!("{:.0}°:{:.2}", b.to_degrees(), sum / n.max(1) as f32)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// One full compass of bearing episodes at one target distance — the close and pace
/// probes' shape (fixed tile-center geometry, one episode per compass bearing).
#[derive(Debug, Clone, Copy)]
pub struct CompassSweep {
    pub target_distance_m: f32,
    pub per_bearing: [BearingReport; EVAL_BEARINGS],
}

impl CompassSweep {
    /// The WORST (min-progress) bearing's episode — one coherent real episode, so
    /// `reached`/torque/distances all describe the same run. `total_cmp` only for a
    /// deterministic pick; `progress_m` is never NaN (`(a-b).max(0.0)` scrubs it).
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

    /// The BEST bearing's sustained pace (arena m/s) — her full charge. Max, not min:
    /// the speed guard asks how fast she really is (spawn safety cares about her top
    /// pace), while dead bearings are the headline gate's problem.
    fn best_sustained_pace_m_per_s(&self) -> f32 {
        self.per_bearing
            .iter()
            .map(|b| b.sustained_pace_m_per_s)
            .fold(0.0, f32::max)
    }
}

/// The sweeps of one eval, judging one load of one brain.
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub policy_loaded: bool,
    /// The terrain amplitude the whole eval ran on (starts, episodes, probes — ONE
    /// grid, [`crate::terrain::TerrainGrid::gcr_with_amplitude`]). 1.0 = the
    /// canonical bake bit-identically.
    pub terrain_amplitude: f32,
    /// The far (heading, start) pairs — the chase eval proper, sole source of the
    /// mean-progress headline.
    pub far: PairSweep,
    /// The rl#252 close-range probe at [`CLOSE_PROBE_DISTANCE_M`], tile center only
    /// (fixed geometry — the emergence readout diffs against recorded baselines).
    /// SIDECAR ONLY — nothing here may feed [`Self::progress_m`], the headline every
    /// gate keys off: folding it in would be a training-adjacent metric change
    /// riding along with a measurement.
    pub close: CompassSweep,
    /// The rl#280 pace probe at [`PACE_PROBE_DISTANCE_M`], tile center only (the
    /// charge instrument's geometry stays fixed so the pin stays comparable).
    /// SIDECAR like the close probe: sole source of
    /// [`Self::measured_charge_heights_per_s`], never of the headline.
    pub pace: CompassSweep,
}

impl EvalReport {
    /// THE headline scalar every gate compares — one ruler, one convention: MEAN
    /// over [`EVAL_PAIRS`] progressable (heading, start) pairs of zero-floored
    /// best-approach FAR progress at [`DEFAULT_TARGET_DISTANCE_M`], rescued pairs
    /// forfeited to 0. Tail statistics ([`PairSweep::min_pair`],
    /// [`PairSweep::rescued_pairs`], the per-heading means) are REPORTED beside it,
    /// never folded in.
    pub fn progress_m(&self) -> f32 {
        self.far.mean_progress_m()
    }

    /// Sally's measured charge speed in body-heights/s — the pace probe's best
    /// sustained pace over the natural rig height, the same height net's arena→sim
    /// seam divides by, so this number and the sim-side charge constant live on one
    /// scale. `None` when unmeasurable: no loaded policy (the rest pose measures the
    /// baseline, not her), no progress, a degenerate silhouette — or an instrument
    /// off its pinned geometry: a pace sweep not at [`PACE_PROBE_DISTANCE_M`], or
    /// any terrain amplitude but the canonical 1.0 (the pin was measured on the
    /// canonical bake; comparing a scaled-relief probe against it would flag a
    /// geometry artifact as drift and invite a corrupting re-pin).
    pub fn measured_charge_heights_per_s(&self) -> Option<f32> {
        if !self.policy_loaded
            || self.pace.target_distance_m != PACE_PROBE_DISTANCE_M
            || self.terrain_amplitude != 1.0
        {
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

    /// True when the charge guard SHOULD have measured and could not: a loaded
    /// policy, the canonical instrument geometry (probe distance, amplitude 1), the
    /// full probe budget — and still no reading (zero pace on every probe bearing,
    /// or no rig height). The charge keys just VANISH from the wire otherwise and no
    /// consumer watches for absence, so this rides the wire as
    /// `charge_unmeasurable=true` and [`run_eval`] warns on it (rl#280, rl#341
    /// S2-4). Deliberately-off-instrument runs (short budget, non-1 amplitude) are
    /// NOT unmeasurable — they declined to measure.
    pub fn charge_unmeasurable(&self) -> bool {
        self.policy_loaded
            && self.terrain_amplitude == 1.0
            && self.pace.target_distance_m == PACE_PROBE_DISTANCE_M
            && self
                .pace
                .per_bearing
                .iter()
                .all(|b| b.active_ticks >= PACE_PROBE_TICKS)
            && self.measured_charge_heights_per_s().is_none()
    }
}

/// The keep-best sidecar's instrument fingerprint: everything the DEFAULT eval's
/// number depends on besides the brain — the bake bytes, the amplitude, the episode
/// geometry, the progressability bar, and the sample size. Stamped into the progress
/// sidecar beside the bar; a bar whose fingerprint no longer matches was earned on a
/// different instrument and is re-baselined instead of silently wedging promotion
/// (rl#341 S1-4: a re-bake or grade nudge used to freeze `beats` forever at warn
/// level).
pub fn instrument_fingerprint() -> String {
    format!(
        "bake={:016x} amplitude=1 distance_m={DEFAULT_TARGET_DISTANCE_M} \
         ticks={DEFAULT_EVAL_TICKS} grade={MAX_PROGRESSABLE_GRADE} pairs={EVAL_PAIRS} \
         settle={RESET_GRACE_TICKS}",
        crate::terrain::gcr_bake_digest(),
    )
}

/// CLI-facing validation for `--distance` (rl#341 S1-3): the far distance must be a
/// real positive length inside the training band — past [`BAND_MAX_M`] the obs
/// target is out-of-distribution and an unreachable ball measures obstruction, not
/// chase (rl#292); NaN/∞/negative used to panic in the locale bake, hang the start
/// search, or silently sweep the mirrored bearing.
pub fn validate_target_distance(d: f32) -> Result<f32, String> {
    if d.is_finite() && d > 0.0 && d <= BAND_MAX_M {
        Ok(d)
    } else {
        Err(format!(
            "target distance must be a finite length in (0, {BAND_MAX_M}] m, got {d}"
        ))
    }
}

#[derive(Resource)]
struct EvalConfig {
    target_distance: f32,
    /// Per-env target bearing — env i chases `bearings[i]` from its own start.
    bearings: Vec<f32>,
    settle_ticks: u64,
    /// Pace-probe mode (rl#280): the real ball sits past the training band edge, so
    /// the obs target is posed through [`probe_lure`] and planar drift through
    /// [`pace_recenter`] — the production seams — while every measurement stays
    /// against the real ball. Single-env by construction ([`run_pairs`] asserts).
    pace_probe: bool,
}

/// One pair's accumulators. The rl#310 sign pin and the walker test exercise the
/// same fold [`run_pairs`] ships by constructing this directly.
#[derive(Default, Clone)]
struct PairState {
    /// The REAL ball every distance/pace/tip measurement is taken against — the obs
    /// slot equals it except in pace-probe mode, where the slot holds the lure.
    real_target: Option<Vec3>,
    initial_dist: f32,
    closest_dist: f32,
    last_dist: f32,
    closest_tip_dist: Option<f32>,
    torque_sum: f64,
    work_sum: f64,
    best_pace_m_per_s: f32,
    rescues: u32,
}

#[derive(Resource)]
struct EvalState {
    tick: u64,
    /// Shared active-tick counter — every env steps in lockstep, so one counter is
    /// every pair's episode clock.
    torque_ticks: u64,
    pairs: Vec<PairState>,
}

impl EvalState {
    fn new(n: usize) -> Self {
        Self {
            tick: 0,
            torque_ticks: 0,
            pairs: vec![PairState::default(); n],
        }
    }
}

pub fn run_eval(
    _body_gate: crate::mesh_fallback::BodyGate,
    checkpoint_dir: &Path,
    active_ticks: u64,
    target_distance: f32,
    terrain_amplitude: f32,
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
    // per-episode reload across the ~5 min far+close+pace sweeps could average a
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

    // ONE grid for the WHOLE eval — start derivation, every episode world, both
    // probes — threaded, never re-resolved, so a scaled-amplitude run cannot fork
    // its arena from its measurement (rl#341 S4-A). Amplitude 1.0 IS the canonical
    // singleton, bit-identically.
    let grid = crate::terrain::TerrainGrid::gcr_with_amplitude(terrain_amplitude)?;
    let pairs = eval_pairs(&grid, target_distance)?;
    println!(
        "eval: {} far pairs (amplitude {terrain_amplitude}, max uphill grade \
         {MAX_PROGRESSABLE_GRADE}), starts: {}",
        pairs.len(),
        pairs
            .iter()
            .map(|p| format!("({:.0},{:.0})", p.start_xz.x, p.start_xz.y))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // The close and pace probes reuse the far sweep's episode definition (same
    // fresh-world episode, same fold) so their numbers read on the same scale; the
    // pace probe differs only where its geometry forces it to (rl#280). Only the far
    // pairs are progressability-drawn (rl#328): the probes keep their FIXED
    // tile-center geometry, because the charge pin and the close probe's recorded
    // baselines were measured there and moving either instrument re-pins/re-baselines
    // it. Both are SIDECARS (never the headline); the close probe's readout is
    // reached_count, and the pace probe reads its BEST bearing — neither is floored
    // by a dead one. The far pairs run batched, MAX_ENVS per world (envs are
    // collision-isolated by group); the probes stay single-env — their pinned
    // baselines were measured in single-env worlds and the pace seams are env-0.
    let mut far_reports = Vec::with_capacity(pairs.len());
    for chunk in pairs.chunks(crate::bot::body::MAX_ENVS) {
        far_reports.extend(run_pairs(
            &policy,
            active_ticks,
            target_distance,
            false,
            chunk,
            &grid,
        ));
    }
    let report = EvalReport {
        policy_loaded,
        terrain_amplitude,
        far: PairSweep {
            target_distance_m: target_distance,
            pairs: far_reports,
        },
        close: run_compass(
            &policy,
            active_ticks,
            CLOSE_PROBE_DISTANCE_M,
            false,
            Vec2::ZERO,
            &grid,
        ),
        pace: run_pace_probe(&policy, active_ticks, Vec2::ZERO, &grid),
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
    } else if report.charge_unmeasurable() {
        tracing::warn!(
            "charge speed unmeasurable (rl#280): loaded policy, zero pace on every \
             pace-probe bearing (or no measurable rig height) — suspect a probe \
             regression, not a slow brain; the wire carries charge_unmeasurable=true"
        );
    }
    if report.far.rescued_pairs() > 0 {
        tracing::warn!(
            "eval rescues (rl#341 S1-1): {} far pair(s) forfeited to a mid-episode \
             physics rescue — the mean carries forfeited zeros; see rescues= on the wire",
            report.far.rescued_pairs(),
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
    grid: &std::sync::Arc<crate::terrain::TerrainGrid>,
) -> CompassSweep {
    let mut per_bearing = [None; EVAL_BEARINGS];
    for (slot, bearing_rad) in per_bearing.iter_mut().zip(eval_bearings()) {
        let pair = EvalPair {
            start_xz: locale_xz,
            bearing_rad,
        };
        let mut reports = run_pairs(
            policy,
            active_ticks,
            target_distance,
            pace_probe,
            &[pair],
            grid,
        );
        *slot = Some(reports.remove(0));
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
fn run_pace_probe(
    policy: &std::rc::Rc<Policy>,
    tick_budget: u64,
    locale_xz: Vec2,
    grid: &std::sync::Arc<crate::terrain::TerrainGrid>,
) -> CompassSweep {
    run_compass(
        policy,
        tick_budget.min(PACE_PROBE_TICKS),
        PACE_PROBE_DISTANCE_M,
        true,
        locale_xz,
        grid,
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

/// One batched episode world: env i chases `pairs[i]`'s bearing from its own
/// surface-placed start (via the same
/// [`InitialCrabLayout`](crate::bot::InitialCrabLayout) seam net's GCR restart uses,
/// so eval starts ride the ONE spawn-layout path). All envs step in lockstep for the
/// same settle + active budget; envs are collision-isolated by group, so pairs can
/// never physically contaminate each other. Deterministic per brain — the worlds are
/// single-threaded and the pair list is a pure function of the grid.
fn run_pairs(
    policy: &std::rc::Rc<Policy>,
    active_ticks: u64,
    target_distance: f32,
    pace_probe: bool,
    pairs: &[EvalPair],
    grid: &std::sync::Arc<crate::terrain::TerrainGrid>,
) -> Vec<BearingReport> {
    assert!(
        !pairs.is_empty() && pairs.len() <= crate::bot::body::MAX_ENVS,
        "run_pairs takes 1..=MAX_ENVS pairs, got {}",
        pairs.len()
    );
    assert!(
        !pace_probe || pairs.len() == 1,
        "the pace probe is a single-env instrument (pace_recenter and the lure are env-0)"
    );
    // The server world (rl#298 stage 6): the same constructor the trainer's rollout
    // env and net's headless host build from, vehicle layer included, so the eval
    // judges the env she trains in and is served in — no `VehicleControls` entries
    // means zero craft bodies, so the sweeps measure the same world as before.
    let mut app = headless_server_world(pairs.len(), WorldRole::Standalone, grid.clone());
    app.insert_resource(crate::bot::InitialCrabLayout {
        base_xz: pairs[0].start_xz,
        spawns_m: pairs.iter().map(|p| p.start_xz).collect(),
    });
    // A rescue during a MEASUREMENT is an instrument fault, never routine (rl#341
    // S1-1): arming the fault flag turns on rescue_lost_crabs' loud logging, and
    // eval_step tallies the rescues so the fold can forfeit the pair.
    app.insert_resource(crate::bot::CrabRescueIsFault);
    app.insert_resource(EvalConfig {
        target_distance,
        bearings: pairs.iter().map(|p| p.bearing_rad).collect(),
        settle_ticks: RESET_GRACE_TICKS as u64,
        pace_probe,
    })
    .insert_resource(EvalState::new(pairs.len()))
    .insert_non_send_resource(policy.clone())
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
    // The pump must be 1:1 update:FixedUpdate (bot::headless pins Time<Fixed> to
    // PHYSICS_DT explicitly) — a broken ratio would silently truncate every episode
    // at the max_updates backstop instead (rl#341 S3-1), so refuse loud.
    assert!(
        active_torque_ticks(&app) == active_ticks,
        "episode ended at {}/{} active ticks after {updates} updates — the \
         update:FixedUpdate pump broke (rl#341 S3-1)",
        active_torque_ticks(&app),
        active_ticks,
    );

    let state = app
        .world()
        .get_resource::<EvalState>()
        .expect("eval state present");
    // A start the settle window never measured is a spawn wiring bug, not a slow
    // policy — it would fold to a policy-independent 0 (rl#341 S3-2).
    assert!(
        state.pairs.iter().all(|p| p.initial_dist > 0.0),
        "a pair saw no finite carapace during settling — spawn wiring bug (rl#341 S3-2)"
    );
    pairs
        .iter()
        .zip(state.pairs.iter())
        .map(|(pair, ps)| ps.report(pair.bearing_rad, pair.start_xz, state.torque_ticks))
        .collect()
}

impl PairState {
    /// Fold one finished episode into its report — pure, so the rl#310 sign pin and
    /// the rescue-forfeit pin can exercise the same construction [`run_pairs`]
    /// ships: `progress_m` zero-floored off the running-min `closest_dist`,
    /// [`BearingReport::net_progress_m`] signed off `last_dist`, and a rescued
    /// pair's progress and pace FORFEIT to 0 (S1-1 — the teleport back to origin
    /// makes the trajectory not-a-chase; the raw distances stay for diagnosis).
    fn report(&self, bearing_rad: f32, start_xz: Vec2, active_ticks: u64) -> BearingReport {
        let rescued = self.rescues > 0;
        let progress_m = if rescued {
            0.0
        } else {
            (self.initial_dist - self.closest_dist).max(0.0)
        };
        let mean_torque_per_tick = if active_ticks > 0 {
            (self.torque_sum / active_ticks as f64) as f32
        } else {
            0.0
        };
        // One binding feeds both fields, so `reached ⇔ tip_touch(closest_tip_distance_m)`
        // holds structurally (`tip_touch(INFINITY)` = false covers the no-tip-seen case).
        let closest_tip = self.closest_tip_dist.unwrap_or(f32::INFINITY);
        BearingReport {
            bearing_rad,
            start_xz,
            progress_m,
            total_torque: self.torque_sum as f32,
            mean_torque_per_tick,
            saturation: mean_torque_per_tick / total_drive_torque_ceiling(),
            work_j: self.work_sum as f32,
            initial_distance_m: self.initial_dist,
            closest_distance_m: self.closest_dist,
            final_distance_m: self.last_dist,
            closest_tip_distance_m: closest_tip,
            reached: !rescued && tip_touch(closest_tip),
            active_ticks,
            rescues: self.rescues,
            sustained_pace_m_per_s: if rescued { 0.0 } else { self.best_pace_m_per_s },
        }
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
    mut rescued: MessageReader<CrabRescued>,
    carapace_q: Query<(&CrabEnvId, &Transform), With<CrabCarapace>>,
    claw_tips_q: Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
    joints: Query<(&CrabJoint, &CrabEnvId)>,
) {
    let n = state.pairs.len();
    if state.pairs[0].real_target.is_none() && targets.envs.len() == n {
        for (i, slot) in targets.envs.iter_mut().enumerate() {
            let origin = spawns.origin(i);
            let p = polar_target(origin, cfg.bearings[i], cfg.target_distance, TARGET_Y);
            let real = terrain.place(Vec2::new(p.x, p.z), TARGET_Y);
            state.pairs[i].real_target = Some(real);
            // In probe mode the crab is still at its origin, so the lure is exact here.
            *slot = Some(if cfg.pace_probe {
                probe_lure(origin, real, &terrain)
            } else {
                real
            });
        }
    }
    if state.pairs[0].real_target.is_none() {
        state.tick += 1;
        return;
    }

    let settling = state.tick < cfg.settle_ticks;

    // Rescue tally (rl#341 S1-1): rescue_lost_crabs runs before Sense, so this
    // tick's teleports are visible here; the fold forfeits any rescued pair.
    for msg in rescued.read() {
        if let Some(p) = state.pairs.get_mut(msg.env) {
            p.rescues += 1;
        }
    }

    // Skips are deliberate: unsized envs = the pre-spawn window.
    if settling || obs.rows().len() != n {
        for i in 0..n {
            let _ = actions.rest(i);
        }
    } else {
        for (i, row) in obs.rows().iter().enumerate() {
            let _ = actions.set_row(i, policy.act(row));
        }
    }

    let elapsed_s = state.torque_ticks as f32 / crate::physics::PHYSICS_HZ as f32;
    for (env, t) in carapace_q.iter() {
        let i = env.0;
        let Some(cpos) = Some(t.translation).filter(|p| p.is_finite()) else {
            continue;
        };
        let Some(target) = state.pairs.get(i).and_then(|p| p.real_target) else {
            continue;
        };
        // Probe mode re-poses the lure every tick from her CURRENT position — net's
        // `set_crab_walk_target` cadence — so the obs ball recedes ahead of her until
        // the real ball comes inside the band, then converges onto it.
        if cfg.pace_probe
            && let Some(slot) = targets.envs.get_mut(i)
        {
            *slot = Some(probe_lure(cpos, target, &terrain));
        }
        let d = dist_3d(cpos, target);
        let p = &mut state.pairs[i];
        if settling {
            p.initial_dist = d;
            p.closest_dist = d;
        } else {
            p.closest_dist = p.closest_dist.min(d);
            if elapsed_s >= PACE_WINDOW_MIN_S {
                p.best_pace_m_per_s = p.best_pace_m_per_s.max((p.initial_dist - d) / elapsed_s);
            }
        }
        p.last_dist = d;
    }

    // Settle ticks are excluded so a spawn transient can't score a touch the
    // policy never earned.
    if !settling {
        for i in 0..n {
            let Some(target) = state.pairs[i].real_target else {
                continue;
            };
            if let Some(d) = closest_tip_dist(i, target, &claw_tips_q) {
                let p = &mut state.pairs[i];
                p.closest_tip_dist = Some(p.closest_tip_dist.map_or(d, |cur| cur.min(d)));
            }
        }
    }

    if !settling && !actions.is_empty() {
        // Pure observation of what Sense/Think already computed (rl#279): commanded
        // torque × the sensor's measured hinge rate, so the accumulator can never
        // perturb the rollout it measures.
        for (joint, env) in joints.iter() {
            let i = env.0;
            if i >= n {
                continue;
            }
            if let Some(drive) = actions.drive(i, joint.id) {
                let torque = applied_torque(joint.id, drive);
                let p = &mut state.pairs[i];
                p.torque_sum += torque.abs() as f64;
                if let Some(view) = obs.env(i) {
                    p.work_sum += ((torque * view.joint_rate(joint.id)).abs()
                        / crate::physics::PHYSICS_HZ as f32)
                        as f64;
                }
            }
        }
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
/// same delta, so the remaining chase geometry (every distance eval_step measures)
/// is invariant across the shift. Since rl#311 the obs carries no position channel,
/// so the shift is invisible to the policy by construction; what the teleport buys
/// is the fixed locale below. On
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
/// Env 0 only — the pace probe is single-env by construction ([`run_pairs`]).
fn pace_recenter(
    spawns: Res<CrabSpawns>,
    terrain: Res<crate::terrain::Terrain>,
    mut state: ResMut<EvalState>,
    mut targets: ResMut<CrabTargets>,
    mut parts: Query<(&CrabEnvId, &mut Transform, Has<CrabCarapace>), With<CrabBodyPart>>,
) {
    // real_target set ⇒ env slots exist ⇒ origin(0) is wired (rl#242).
    let Some(real_target) = state.pairs[0].real_target else {
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
    state.pairs[0].real_target = Some(real_target + delta);
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

    fn canned_bearing(distance: f32) -> BearingReport {
        BearingReport {
            bearing_rad: 0.0,
            start_xz: Vec2::ZERO,
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
            rescues: 0,
            sustained_pace_m_per_s: 0.0,
        }
    }

    /// A canned report whose PACE sweep runs at `pace_distance_m` with ONE bearing
    /// pacing at `pace` (arena m/s) — the speed guard's inputs, nothing else non-zero.
    fn paced_report(policy_loaded: bool, pace_distance_m: f32, pace: f32) -> EvalReport {
        let sweep_at = |distance: f32| CompassSweep {
            target_distance_m: distance,
            per_bearing: [canned_bearing(distance); EVAL_BEARINGS],
        };
        let mut paced = canned_bearing(pace_distance_m);
        paced.sustained_pace_m_per_s = pace;
        let mut pace_sweep = sweep_at(pace_distance_m);
        pace_sweep.per_bearing[3] = paced;
        EvalReport {
            policy_loaded,
            terrain_amplitude: 1.0,
            far: PairSweep {
                target_distance_m: DEFAULT_TARGET_DISTANCE_M,
                pairs: vec![canned_bearing(DEFAULT_TARGET_DISTANCE_M); EVAL_PAIRS],
            },
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
        assert!(
            !r.charge_unmeasurable(),
            "a measured charge is not 'unmeasurable'"
        );

        // A pace sweep off the probe distance: the saturation ceiling moved — a
        // geometry artifact, not her gait; comparing it to the pin would invite a
        // corrupting re-pin.
        let short = paced_report(true, PACE_PROBE_DISTANCE_M - 1.0, fast);
        assert_eq!(short.measured_charge_heights_per_s(), None);
        assert!(
            !short.charge_unmeasurable(),
            "off-pin geometry DECLINED to measure"
        );

        // A non-canonical amplitude scales the probe's relief — same class of
        // geometry artifact (rl#341/1a); the pin only reads at amplitude 1.
        let mut scaled = paced_report(true, PACE_PROBE_DISTANCE_M, fast);
        scaled.terrain_amplitude = 2.0;
        assert_eq!(scaled.measured_charge_heights_per_s(), None);
        assert!(!scaled.charge_unmeasurable());

        // The rest-pose baseline measures the baseline, not her.
        let unloaded = paced_report(false, PACE_PROBE_DISTANCE_M, fast);
        assert_eq!(unloaded.measured_charge_heights_per_s(), None);
        assert!(!unloaded.charge_unmeasurable());

        // No progress at all on the canonical instrument with the full budget:
        // nothing measured, and THAT is the loud unmeasurable case (S2-4).
        let parked = paced_report(true, PACE_PROBE_DISTANCE_M, 0.0);
        assert_eq!(parked.measured_charge_heights_per_s(), None);
        assert!(parked.charge_unmeasurable());

        // Short budget: the pace window never opened — declined, not unmeasurable.
        let mut short_budget = paced_report(true, PACE_PROBE_DISTANCE_M, 0.0);
        for b in &mut short_budget.pace.per_bearing {
            b.active_ticks = PACE_PROBE_TICKS - 1;
        }
        assert!(!short_budget.charge_unmeasurable());
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
    /// above it the number is exactly work over net progress. And the sweep mean
    /// averages only the pairs that measured one.
    #[test]
    fn j_per_m_is_guarded_by_the_progress_floor() {
        let mut r = paced_report(true, DEFAULT_TARGET_DISTANCE_M, 0.0);
        for b in &mut r.far.pairs {
            b.work_j = 10.0;
        }
        r.far.pairs[0].progress_m = WORK_PER_M_MIN_PROGRESS_M - 0.01;
        assert_eq!(r.far.pairs[0].j_per_m(), None);
        r.far.pairs[1].progress_m = WORK_PER_M_MIN_PROGRESS_M;
        assert_eq!(
            r.far.pairs[1].j_per_m(),
            Some(10.0 / WORK_PER_M_MIN_PROGRESS_M)
        );
        r.far.pairs[2].progress_m = 4.0;
        assert_eq!(r.far.pairs[2].j_per_m(), Some(2.5));
        // Mean over the two measured pairs only — the guarded ones don't drag it.
        let mean = r.far.mean_j_per_m().expect("two pairs measured");
        assert!((mean - (20.0 + 2.5) / 2.0).abs() < 1e-4);

        // All pairs guarded → no sweep mean at all, never a 0/0.
        let parked = paced_report(true, DEFAULT_TARGET_DISTANCE_M, 0.0);
        assert_eq!(parked.far.mean_j_per_m(), None);
        assert_eq!(parked.far.mean_saturation(), 0.0);
    }

    /// rl#310 pin: retreat is visible in the SIGNED net while the zero-floored
    /// best-approach headline input stays 0 — through the same fold `run_pairs`
    /// ships, so a flat `progress_m` decomposes into parked (net ≈ 0) vs retreating
    /// (net < 0).
    #[test]
    fn retreat_reports_negative_net_while_progress_stays_zero() {
        let state = PairState {
            initial_dist: 5.0,
            closest_dist: 5.0, // never got closer than the start…
            last_dist: 7.0,    // …and ended 2 m farther out
            ..Default::default()
        };
        let b = state.report(0.0, Vec2::ZERO, 100);
        assert_eq!(b.progress_m, 0.0);
        assert_eq!(b.net_progress_m(), -2.0);
    }

    /// rl#341 S1-1 pin: a pair whose crab was rescued mid-episode FORFEITS its
    /// progress and pace — an exploding lunge can no longer keep the meters it
    /// closed before the physics faulted (and be promoted on them) — while the raw
    /// distances survive for diagnosis and the tally is reported.
    #[test]
    fn rescued_pair_forfeits_progress_and_pace() {
        let state = PairState {
            initial_dist: 24.0,
            closest_dist: 12.0, // closed 12 m…
            last_dist: 24.0,
            best_pace_m_per_s: 3.0,
            rescues: 1, // …then exploded and was teleported home
            ..Default::default()
        };
        let b = state.report(0.0, Vec2::ZERO, 1500);
        assert!(b.rescued());
        assert_eq!(b.progress_m, 0.0, "rescued progress forfeits");
        assert_eq!(b.sustained_pace_m_per_s, 0.0, "rescued pace forfeits");
        assert!(!b.reached, "a rescued episode cannot count a strike");
        assert_eq!(
            b.closest_distance_m, 12.0,
            "raw distances stay for diagnosis"
        );

        let clean = PairState {
            rescues: 0,
            ..state
        };
        assert_eq!(clean.report(0.0, Vec2::ZERO, 1500).progress_m, 12.0);
    }

    /// The mean-pair fold (owner 08-03): headline = flat mean over pairs; the min
    /// pair and rescued count are the tail statistics beside it.
    #[test]
    fn mean_headline_with_tail_statistics() {
        let mut r = paced_report(true, PACE_PROBE_DISTANCE_M, 0.0);
        for (i, p) in r.far.pairs.iter_mut().enumerate() {
            p.progress_m = i as f32; // 0, 1, 2, …
            p.final_distance_m = p.initial_distance_m - i as f32;
        }
        let n = EVAL_PAIRS as f32;
        let expect_mean = (0..EVAL_PAIRS).sum::<usize>() as f32 / n;
        assert!((r.progress_m() - expect_mean).abs() < 1e-4);
        assert!((r.far.mean_net_progress_m() - expect_mean).abs() < 1e-4);
        assert_eq!(r.far.min_pair().progress_m, 0.0);
        assert_eq!(r.far.rescued_pairs(), 0);

        r.far.pairs[EVAL_PAIRS - 1].rescues = 2;
        assert_eq!(r.far.rescued_pairs(), 1);

        // The per-heading line has one entry per compass bearing.
        assert_eq!(r.far.progress_line().split(' ').count(), EVAL_BEARINGS);
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
            // obs) — NOT the band edge: an unreachable ball would make progress
            // measure the ray's worst obstruction instead of her chase. Past the
            // drift radius so the sweep covers treks longer than one rebase window
            // (the far sweep itself runs no recenter, like training).
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

    /// The rl#341 S1-3 CLI guard: only real in-band lengths pass.
    #[test]
    fn target_distance_validation_refuses_degenerates() {
        use crate::training::targets::BAND_MAX_M;
        assert!(validate_target_distance(DEFAULT_TARGET_DISTANCE_M).is_ok());
        assert!(validate_target_distance(1.0).is_ok());
        assert!(validate_target_distance(BAND_MAX_M).is_ok());
        for bad in [0.0, -5.0, f32::NAN, f32::INFINITY, BAND_MAX_M + 1.0] {
            assert!(validate_target_distance(bad).is_err(), "{bad} must refuse");
        }
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

    /// rl#328 progressability primitives: flat ground is progressable everywhere; a
    /// cliff across +X kills bearing 0 (pinning the 0 = +X ray convention against
    /// [`polar_target`]) and leaves 180° open.
    #[test]
    fn bearing_progressability_tracks_the_terrain() {
        let flat = crate::terrain::TerrainGrid::flat(512.0);
        for b in eval_bearings() {
            assert!(
                bearing_progressable(&flat, Vec2::ZERO, b, DEFAULT_TARGET_DISTANCE_M),
                "flat ground is progressable at {:.0}°",
                b.to_degrees()
            );
        }

        // 257² × 4 m cells (±512 m), a 100 m cliff wall over x > 10.
        let n = 257;
        let heights: Vec<i16> = (0..n * n)
            .map(|i| {
                let x = ((i % n) as f32 / (n - 1) as f32 - 0.5) * 1024.0;
                if x > 10.0 { 100 } else { 0 }
            })
            .collect();
        let walled = crate::terrain::TerrainGrid::test_grid(n, n, 4.0, 1.0, &heights);
        assert!(
            !bearing_progressable(&walled, Vec2::ZERO, 0.0, DEFAULT_TARGET_DISTANCE_M),
            "+X runs into the cliff"
        );
        assert!(
            bearing_progressable(
                &walled,
                Vec2::ZERO,
                std::f32::consts::PI,
                DEFAULT_TARGET_DISTANCE_M
            ),
            "−X is open ground"
        );
    }

    /// The pair derivation is deterministic, covers the compass evenly with
    /// [`EVAL_STARTS_PER_HEADING`] DISTINCT starts each, and every scored ray is
    /// progressable for its own heading (owner 08-03: many headings × many
    /// deterministically chosen starts).
    #[test]
    fn eval_pairs_are_deterministic_distinct_and_progressable() {
        let g = crate::terrain::TerrainGrid::gcr();
        let pairs = eval_pairs(&g, DEFAULT_TARGET_DISTANCE_M).expect("canonical bake derives");
        assert_eq!(pairs.len(), EVAL_PAIRS);
        for (i, p) in pairs.iter().enumerate() {
            assert_eq!(
                bearing_bin(p.bearing_rad),
                i % EVAL_BEARINGS,
                "headings cycle"
            );
            assert!(
                bearing_progressable(&g, p.start_xz, p.bearing_rad, DEFAULT_TARGET_DISTANCE_M),
                "pair {i} scored an un-progressable ray"
            );
            for (j, q) in pairs.iter().enumerate().skip(i + 1) {
                assert!(
                    p.start_xz.distance(q.start_xz) > 1.0,
                    "pairs {i} and {j} share a start"
                );
            }
        }
        let again = eval_pairs(&g, DEFAULT_TARGET_DISTANCE_M).expect("still derives");
        for (a, b) in pairs.iter().zip(again.iter()) {
            assert_eq!(a.start_xz, b.start_xz, "same bake ⇒ same pairs");
        }
    }

    /// A candidate whose ray is blocked is skipped, not scored: on a grid with a
    /// cliff wall across +X everywhere east of the origin, every bearing-0 start
    /// must sit where its own +X ray avoids the wall — and the derivation still
    /// seats every heading.
    #[test]
    fn eval_pairs_skip_unprogressable_candidates() {
        // ±512 m, a 100 m cliff at x ∈ (10, 14): +X rays crossing it are dead.
        let n = 257;
        let heights: Vec<i16> = (0..n * n)
            .map(|i| {
                let x = ((i % n) as f32 / (n - 1) as f32 - 0.5) * 1024.0;
                if (10.0..14.0).contains(&x) { 100 } else { 0 }
            })
            .collect();
        let grid = crate::terrain::TerrainGrid::test_grid(n, n, 4.0, 1.0, &heights);
        let d = 24.0;
        let pairs = eval_pairs(&grid, d).expect("open ground everywhere else");
        assert_eq!(pairs.len(), EVAL_PAIRS);
        for p in &pairs {
            assert!(bearing_progressable(&grid, p.start_xz, p.bearing_rad, d));
        }
    }

    /// The high-amplitude/rough-bake failure is a LOUD `Err`, never a swallowed
    /// panic (rl#341/1a): spike terrain no ray can cross exhausts the candidate
    /// budget and refuses with the rl#328 message.
    #[test]
    fn eval_pairs_refuse_unwalkable_terrain() {
        // ±512 m of 4 m-pitch 200 m spikes: every 1 m step of every ray demands a
        // grade far past the bar.
        let n = 257;
        let heights: Vec<i16> = (0..n * n)
            .map(|i| if (i % n + i / n) % 2 == 0 { 200 } else { 0 })
            .collect();
        let grid = crate::terrain::TerrainGrid::test_grid(n, n, 4.0, 1.0, &heights);
        let err = eval_pairs(&grid, 24.0).expect_err("spike terrain must refuse");
        assert!(err.contains("refusing to score rock"), "got: {err}");
    }

    /// On the committed bake, a scripted surface walker — no physics, no policy,
    /// just carapace-height steps along the ray — makes real progress on EVERY
    /// scored pair through the same fold `run_pairs` ships. Nonzero progress is
    /// achievable wherever the instrument scores, so a 0.000 mean once again
    /// measures the policy (rl#328's guarantee, carried to the pair metric).
    #[test]
    fn gcr_pairs_admit_a_scripted_walker() {
        let g = crate::terrain::TerrainGrid::gcr();
        for pair in eval_pairs(&g, DEFAULT_TARGET_DISTANCE_M).expect("canonical bake derives") {
            let origin3 = Vec3::new(pair.start_xz.x, 0.0, pair.start_xz.y);
            let tp = polar_target(
                origin3,
                pair.bearing_rad,
                DEFAULT_TARGET_DISTANCE_M,
                TARGET_Y,
            );
            let target = g.place(Vec2::new(tp.x, tp.z), TARGET_Y);
            let walk = |s: f32| {
                let p = polar_target(origin3, pair.bearing_rad, s, 0.0);
                dist_3d(g.place(Vec2::new(p.x, p.z), 0.5), target)
            };
            let initial = walk(0.0);
            let mut closest = initial;
            let mut s = 0.0;
            while s < DEFAULT_TARGET_DISTANCE_M {
                s += 0.5;
                closest = closest.min(walk(s));
            }
            let report = PairState {
                initial_dist: initial,
                closest_dist: closest,
                last_dist: closest,
                ..Default::default()
            }
            .report(pair.bearing_rad, pair.start_xz, 100);
            assert!(
                report.progress_m > DEFAULT_TARGET_DISTANCE_M / 2.0,
                "walker only made {:.2} m at start ({:.0},{:.0}) bearing {:.0}°",
                report.progress_m,
                pair.start_xz.x,
                pair.start_xz.y,
                pair.bearing_rad.to_degrees()
            );
        }
    }

    /// The end-to-end physical path (rl#341 S2-5): `run_eval` on an empty
    /// (rest-pose) checkpoint through real worlds — batched far pairs, close and
    /// pace probes — with the full fold and every guard live. The landing gate runs
    /// this whenever the eval seam is touched (test-map rule); it stays `#[ignore]`d
    /// for the plain workspace suite because it builds ~18 bevy+rapier worlds.
    #[test]
    #[ignore = "builds ~18 headless bevy+rapier worlds; the eval test-map rule runs it with --ignored"]
    fn rest_pose_has_zero_torque_and_no_progress() {
        let dir = std::env::temp_dir().join(format!("rl-eval-restpose-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Stamp the resolved plant: since the rl#293 strict adoption a recordless dir
        // REFUSES (unknown ground provenance), so the legitimate no-brain-yet
        // baseline is a plant-stamped dir with no brain — what a fresh run's
        // checkpoint dir looks like before the first save.
        crate::bot::body::record_plant(&dir).unwrap();

        // Explicit, greppable test-only opt-in: this eval deliberately runs whatever
        // body the test env constructs (usually the fallback — no sally.glb in CI).
        let r = run_eval(
            crate::mesh_fallback::BodyGate::FallbackAllowed,
            &dir,
            200,
            DEFAULT_TARGET_DISTANCE_M,
            1.0,
        )
        .expect("an absent checkpoint is the legitimate baseline, never a refusal");

        assert!(!r.policy_loaded, "an empty dir loads no policy (rest pose)");
        assert_eq!(r.far.pairs.len(), EVAL_PAIRS);
        assert_eq!(
            r.far.reached_count(),
            0,
            "a rest-pose crab never reaches a far ball"
        );
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
            "rest-pose mean progress should be ~0, got {} m",
            r.progress_m()
        );
        for b in &r.far.pairs {
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
            assert_eq!(b.rescues, 0, "a resting crab is never rescued");
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
                (0.0..1.5).contains(&b.progress_m),
                "rest pose shuffles nowhere at bearing {:.0}° start ({:.0},{:.0}), got {} m",
                b.bearing_rad.to_degrees(),
                b.start_xz.x,
                b.start_xz.y,
                b.progress_m
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
