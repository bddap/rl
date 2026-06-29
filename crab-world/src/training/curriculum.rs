//! The target-distance curriculum: the advancing distance band, the competence window
//! that widens it, target sampling from the current band, and the band's persistence.

use std::path::Path;

use bevy::prelude::Vec3;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::bot::CrabSpawns;
use crate::bot::sensor::CrabTargets;

use super::atomic_write;

/// The start band (rung 1): the planar (XZ) distance, in metres, at which a fresh
/// target spawns from the env's origin. NEAR on purpose. WHY a curriculum at all: a cold
/// policy cannot learn a FAR (3–6 m) target — that far out the reach term is both small
/// and flat (~0.115, slope ~0.05/m at 4.5 m), too weak for a stand to discover locomotion
/// from, so it stalls in the stand basin and never walks (verified: 150 iters pinned at
/// the stand floor, drift ~0.3 m). At the near band it is large and steep (~0.385, slope
/// ~0.13/m at 1.5 m), so the crab sets off immediately and a gait forms. The band then
/// WIDENS outward only as the policy masters the current rung (see [`Curriculum`]). Lower
/// bound clears the ~1.3 m reach shell so even the nearest target demands a step, not a lean.
pub(crate) const BAND_START_MIN: f32 = 1.5;
const BAND_START_MAX: f32 = 3.0;
/// How far the band slides outward per advancement (both bounds move, so the width is
/// invariant). 1 m is roughly one rung of reach-gradient difficulty: small enough that
/// the policy is already competent just inside the new far edge, large enough that the
/// curriculum reaches the arena cap in a handful of rungs rather than crawling.
const BAND_ADVANCE_STEP: f32 = 1.0;
/// Vertical band of the target (world Y). A modest claw-height span so a crab that
/// has walked up to the target still finishes with a real reach, not a foot-level
/// touch. Kept low and narrow — the reward is about getting THERE, so the target sits
/// just high enough to demand a genuine reach, no higher.
const TARGET_Y_MIN: f32 = 0.15;
const TARGET_Y_MAX: f32 = 0.7;
/// Half-extent the target's planar position is clamped within: a 1 m margin inside the
/// arena walls, DERIVED from the wall position so a wall move can't strand a far target
/// in or beyond a wall where the crab can't stand on it. The margin leaves room for the
/// crab's own body at the goal. It is also the curriculum's hard cap: the band's far
/// edge never advances past it, since a target the crab can't physically stand at is
/// not a rung worth training.
pub(crate) const TARGET_ARENA_HALF: f32 = crate::physics::world::ARENA_HALF_SIZE - 1.0;

/// Per-episode reach radius (m): the curriculum scores an episode "reached" if the
/// crab's claw tip came within this of the target at any tick. The CANONICAL reach
/// distance — the demo's ball-hop (`play::target_ball::DEMO_REACH_RADIUS`) derives from this one
/// constant, so "reached" means the same event a viewer sees the ball teleport on. Lives
/// in the always-compiled trainer rather than the
/// render-only demo, so the headless build owns the source. A touch looser than zero so a
/// near-miss the policy clearly solved still counts.
pub(crate) const CURRICULUM_REACH_RADIUS: f32 = 0.8;
/// Reach-fraction over the competence window at or above which the band advances. 0.6,
/// not ~1.0: the goal is "the policy reliably gets there", not "every episode is
/// perfect" — targets near the arena edge clamp short and some spawns are awkward, so
/// demanding unanimity would stall the curriculum on noise it has effectively mastered.
/// Reused by [`super::best`] as the solid-reach floor a checkpoint must clear to enter
/// `ckpt/best/` — the same bar that defines "the policy reliably gets there", so a
/// collapse (reach below it) can never become the best regardless of band.
pub(crate) const ADVANCE_REACH_FRACTION: f32 = 0.6;
/// Number of recent FINISHED episodes (pooled across all rollout threads) the
/// reach-fraction is measured over before an advance is considered. Wide enough that
/// one lucky streak can't trip an advance, narrow enough that the signal tracks the
/// CURRENT policy rather than ancient episodes from before the last advance. Episodes
/// from before an advance are dropped on advancing (see [`Curriculum::record_episode`])
/// so the window only ever judges the rung it currently sits on.
const COMPETENCE_WINDOW: usize = 200;

/// Serde mirror of [`Curriculum`] — the on-disk form. A plain `(min, max)` with no
/// invariant of its own; [`Curriculum::from_data`] re-validates on load, so a corrupt or
/// hand-edited file can never reconstitute an illegal band.
#[derive(Serialize, Deserialize)]
struct CurriculumData {
    min: f32,
    max: f32,
}

/// The target-distance curriculum: the single source of truth for the current planar
/// distance band, plus the competence window that decides when to widen it.
///
/// Invariant, upheld by construction (private fields, only [`Self::start`] and
/// [`Self::advanced`] build one): `BAND_START_MIN ≤ min < max ≤ TARGET_ARENA_HALF`, and
/// the width `max − min` is constant across rungs. So a `Curriculum` can never name an
/// empty or inverted band, nor one past the arena cap — illegal states are
/// unrepresentable rather than checked at every read.
///
/// The LEARNER owns the one instance: it pools every rollout thread's per-episode reach
/// outcomes into `window` and advances when the rung is mastered. Threads receive only
/// the band (`min`/`max`) for the horizon, sample targets from it, and ship reach counts
/// back — they never advance, so there is no second curriculum to drift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Curriculum {
    /// Near edge of the current band (m).
    min: f32,
    /// Far edge of the current band (m).
    max: f32,
}

impl Curriculum {
    /// Rung 1 — the near start band a cold policy can bootstrap from (see
    /// [`BAND_START_MIN`]). The only entry point for a fresh run or a checkpoint that
    /// predates the curriculum.
    pub(crate) const fn start() -> Self {
        Self {
            min: BAND_START_MIN,
            max: BAND_START_MAX,
        }
    }

    /// The current band `[min, max)` the thread samples a target distance from.
    pub(crate) fn band(self) -> (f32, f32) {
        (self.min, self.max)
    }

    /// The on-disk mirror (see [`save_curriculum`]).
    fn to_data(self) -> CurriculumData {
        CurriculumData {
            min: self.min,
            max: self.max,
        }
    }

    /// Reconstitute from the on-disk mirror, re-checking the invariant so a corrupt or
    /// hand-edited file cannot produce an illegal band: finite, `BAND_START_MIN ≤ min <
    /// max ≤ TARGET_ARENA_HALF`. `None` on any violation — the caller falls back to
    /// rung 1. (The width is NOT re-checked: only `start`/`advanced` build bands, both
    /// width-preserving, so an in-bounds persisted band is necessarily a real rung.)
    fn from_data(d: CurriculumData) -> Option<Self> {
        let ok = d.min.is_finite()
            && d.max.is_finite()
            && d.min >= BAND_START_MIN
            && d.min < d.max
            && d.max <= TARGET_ARENA_HALF;
        ok.then_some(Self {
            min: d.min,
            max: d.max,
        })
    }

    /// The next rung out: both edges slid by [`BAND_ADVANCE_STEP`] (width preserved),
    /// the far edge capped at [`TARGET_ARENA_HALF`]. `None` once the far edge is already
    /// at the cap — the curriculum is done.
    fn advanced(self) -> Option<Self> {
        if self.max >= TARGET_ARENA_HALF {
            return None;
        }
        let max = (self.max + BAND_ADVANCE_STEP).min(TARGET_ARENA_HALF);
        // Slide the near edge by the same amount the far edge actually moved (which the
        // cap may have shortened on the last rung), so the width stays exactly constant.
        let min = self.min + (max - self.max);
        Some(Self { min, max })
    }
}

/// The learner's competence tracker over the curriculum: the current rung plus a
/// sliding window of recent per-episode reach outcomes (pooled across rollout threads),
/// which gates advancement. Separate from [`Curriculum`] (the persisted band) because
/// the window is transient learner bookkeeping — it is rebuilt from live episodes after
/// a restart and deliberately NOT persisted (only the band itself survives a checkpoint).
pub(crate) struct CurriculumProgress {
    curriculum: Curriculum,
    /// `true` = that episode reached the target. Bounded to [`COMPETENCE_WINDOW`]
    /// (oldest dropped) so the fraction always reflects the current rung's recent
    /// performance, and cleared on every advance so a new rung starts judging fresh.
    window: std::collections::VecDeque<bool>,
}

impl CurriculumProgress {
    pub(crate) fn new(curriculum: Curriculum) -> Self {
        Self {
            curriculum,
            window: std::collections::VecDeque::with_capacity(COMPETENCE_WINDOW),
        }
    }

    pub(crate) fn curriculum(&self) -> Curriculum {
        self.curriculum
    }

    /// Fold one finished episode's reach outcome into the window and, if the rung is now
    /// mastered, advance the band. Mastery = a FULL window whose reach-fraction is at
    /// least [`ADVANCE_REACH_FRACTION`]. Requiring a full window stops a brand-new rung
    /// (or a fresh restart) from advancing on a handful of early episodes. On an advance
    /// the window is cleared so the next rung is judged only on episodes that actually
    /// faced it. Monotone: [`Curriculum::advanced`] only moves outward and returns `None`
    /// at the cap, so the band never regresses and never exceeds the arena. Returns `true`
    /// iff this episode triggered an advance, so a batch fold ([`Self::record_episodes`])
    /// can stop before seeding the cleared rung with episodes from the old band.
    pub(crate) fn record_episode(&mut self, reached: bool) -> bool {
        if self.window.len() == COMPETENCE_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(reached);

        if self.window.len() < COMPETENCE_WINDOW {
            return false;
        }
        let reached_count = self.window.iter().filter(|&&r| r).count();
        let fraction = reached_count as f32 / self.window.len() as f32;
        if fraction >= ADVANCE_REACH_FRACTION
            && let Some(next) = self.curriculum.advanced()
        {
            self.curriculum = next;
            self.window.clear();
            return true;
        }
        false
    }

    /// Fold a horizon's pooled reach tally (`reached` of `finished` episodes reached)
    /// into the window one episode at a time, so the window bound and the advance check
    /// run per episode exactly as if each had been recorded singly. Threads ship counts,
    /// not per-episode booleans, because the gate only uses the window's reach-fraction
    /// (order-free). STOP at an advance: the rest of this horizon's episodes were rolled
    /// against the now-superseded (nearer, easier) band, so folding them into the freshly
    /// cleared rung's window would bias the new rung optimistically — drop them; the next
    /// horizon faces the new band.
    pub(crate) fn record_episodes(&mut self, reached: u64, finished: u64) {
        let reached = reached.min(finished);
        for i in 0..finished {
            if self.record_episode(i < reached) {
                break;
            }
        }
    }
}

/// Sample a fresh target world position for a crab whose env spawns at `origin`, at a
/// planar distance drawn from the CURRENT curriculum band (see [`Curriculum`]). Picks a
/// uniform random heading and a distance in the band, places the target that far from
/// `origin` on the XZ plane, then CLAMPS it inside the arena (see [`TARGET_ARENA_HALF`])
/// so an edge spawn can't throw it into a wall. Y is an independent claw-height draw.
/// World-space (not carapace-relative) because the crab spawns at varied orientations
/// and walks: a point fixed in the world is an unambiguous goal the observation
/// re-expresses in body axes each tick. `pub(crate)` so the demo's red-ball marker
/// (`play::target_ball`) relocates its target through the very same rule training
/// samples — one sampling rule, so the demo can never pose a target training never saw.
pub(crate) fn sample_target(origin: Vec3, curriculum: Curriculum, rng: &mut impl rand::Rng) -> Vec3 {
    let (min, max) = curriculum.band();
    let theta = rng.gen_range(0.0..std::f32::consts::TAU);
    let dist = rng.gen_range(min..max);
    let x = (origin.x + dist * theta.cos()).clamp(-TARGET_ARENA_HALF, TARGET_ARENA_HALF);
    let z = (origin.z + dist * theta.sin()).clamp(-TARGET_ARENA_HALF, TARGET_ARENA_HALF);
    Vec3::new(x, rng.gen_range(TARGET_Y_MIN..TARGET_Y_MAX), z)
}

/// Install a fresh target for env `e`, sampled around its spawn slot from the current
/// curriculum `band` using the training run's seeded RNG. The one home for "a new target
/// is needed" — called to seed the first episode (envs start target-less) and to refresh
/// on every reset, so both callers sample it identically. (Training holds the target
/// fixed within an episode — no resample on reach; see the reach-hover note in
/// `brain_step`.)
pub(crate) fn seed_target(
    targets: &mut CrabTargets,
    spawns: &CrabSpawns,
    e: usize,
    curriculum: Curriculum,
    rng: &mut StdRng,
) {
    if let Some(slot) = targets.envs.get_mut(e) {
        let origin = spawns.0.get(e).copied().unwrap_or(Vec3::ZERO);
        *slot = Some(sample_target(origin, curriculum, rng));
    }
}

/// Persist the curriculum band beside the checkpoint (bincode, like the normalizers).
/// A write failure is logged, not fatal — the run continues; only a resume would lose
/// the rung and restart the curriculum from the start band.
pub(crate) fn save_curriculum(curriculum: Curriculum, path: &Path) {
    match bincode::serialize(&curriculum.to_data()) {
        Ok(bytes) => {
            if let Err(e) = atomic_write(path, &bytes) {
                warn!("Failed to write curriculum to {}: {e}", path.display());
            }
        }
        Err(e) => warn!("Failed to serialize curriculum: {e}"),
    }
}

/// Load the curriculum band from a checkpoint, defaulting to rung 1
/// ([`Curriculum::start`]) on ANY of: a missing file (fresh run, or a checkpoint that
/// predates the curriculum — the warm-continue case), a parse error, or a band that
/// fails the invariant (corrupt/hand-edited). Never returns an illegal band: the policy
/// simply resumes the curriculum from the start, which is safe because the start band is
/// learnable from any policy.
pub(crate) fn load_curriculum(path: &Path) -> Curriculum {
    let Ok(bytes) = std::fs::read(path) else {
        // Missing file is the EXPECTED case for a pre-curriculum checkpoint, so this is
        // info-level, not a warning — a warm-continue of an older policy is normal.
        info!(
            "No curriculum checkpoint at {} — starting the distance curriculum at rung 1",
            path.display()
        );
        return Curriculum::start();
    };
    match bincode::deserialize::<CurriculumData>(&bytes) {
        Ok(data) => Curriculum::from_data(data).unwrap_or_else(|| {
            warn!(
                "Curriculum checkpoint at {} is out of bounds — starting at rung 1",
                path.display()
            );
            Curriculum::start()
        }),
        Err(e) => {
            warn!(
                "Failed to deserialize curriculum from {} ({e}) — starting at rung 1",
                path.display()
            );
            Curriculum::start()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::checkpoint::CheckpointDir;
    use crate::training::reward::planar_dist;

    #[test]
    fn sampled_targets_lie_in_the_current_band_and_inside_the_arena() {
        // Every sampled target lies in the CURRENT curriculum band AND is clamped inside
        // the arena so a crab can always walk to and stand at it. The demo relocates its
        // target through this very `sample_target`, so the demo can never pose a goal
        // training never saw. Verified at BOTH the near start band and a far-advanced
        // band, since the curriculum moves the band outward over a run.
        let mut rng = rand::thread_rng();
        for curriculum in [Curriculum::start(), advanced_to_cap()] {
            let (min, max) = curriculum.band();
            // Worst-case CORNER origin (hard against two walls, where the clamp does the
            // most work — a target headed into the corner is pulled well inside the band).
            let origin = Vec3::new(8.0, 0.0, -8.0);
            let mut saw_clamped = false;
            for _ in 0..2000 {
                let t = sample_target(origin, curriculum, &mut rng);
                assert!(t.is_finite(), "a sampled target is always finite");
                // Inside the arena interior (the clamp guarantees this from any origin).
                assert!(
                    t.x.abs() <= TARGET_ARENA_HALF && t.z.abs() <= TARGET_ARENA_HALF,
                    "target {t:?} must stay inside ±{TARGET_ARENA_HALF} m"
                );
                assert!(t.y >= TARGET_Y_MIN && t.y <= TARGET_Y_MAX);
                // Pre-clamp distance is in the band; post-clamp can only shorten it. Track
                // that clamping actually engages at this edge origin (so the test
                // exercises the in-arena guarantee).
                let d = planar_dist(t, origin);
                if d + 1e-3 < min {
                    saw_clamped = true;
                }
            }
            // From a central origin, nothing is clamped and every target lies in the band.
            let center = Vec3::ZERO;
            for _ in 0..2000 {
                let t = sample_target(center, curriculum, &mut rng);
                let d = planar_dist(t, center);
                assert!(
                    (min..=max).contains(&d),
                    "from center, target distance {d} must lie in the current band \
                     [{min}, {max}]"
                );
            }
            assert!(
                saw_clamped,
                "an edge origin must sometimes clamp a target inward (in-arena guarantee active)"
            );
        }
    }

    /// A curriculum advanced repeatedly until it caps at the arena edge — the far end of
    /// the curriculum, used to verify sampling/advance behavior at the last rung.
    fn advanced_to_cap() -> Curriculum {
        let mut c = Curriculum::start();
        while let Some(next) = c.advanced() {
            c = next;
        }
        c
    }

    #[test]
    fn curriculum_starts_at_rung_one() {
        // A fresh curriculum is the near start band — the only band a cold policy can
        // bootstrap from.
        assert_eq!(Curriculum::start().band(), (BAND_START_MIN, BAND_START_MAX));
    }

    #[test]
    fn advances_one_step_when_competence_met() {
        // A full window at/above the threshold advances the band by exactly one STEP,
        // width preserved.
        let mut p = CurriculumProgress::new(Curriculum::start());
        feed(&mut p, COMPETENCE_WINDOW, ADVANCE_REACH_FRACTION);
        assert_eq!(
            p.curriculum().band(),
            (
                BAND_START_MIN + BAND_ADVANCE_STEP,
                BAND_START_MAX + BAND_ADVANCE_STEP
            ),
            "a mastered rung slides the whole band out by one STEP"
        );
    }

    #[test]
    fn does_not_advance_below_threshold_or_before_a_full_window() {
        // Below the reach threshold: no advance no matter how many episodes.
        let mut low = CurriculumProgress::new(Curriculum::start());
        feed(
            &mut low,
            COMPETENCE_WINDOW * 3,
            ADVANCE_REACH_FRACTION - 0.2,
        );
        assert_eq!(
            low.curriculum().band(),
            Curriculum::start().band(),
            "an under-competent policy never advances"
        );
        // At/above threshold but fewer than a full window: still no advance (a short
        // lucky streak must not trip it).
        let mut partial = CurriculumProgress::new(Curriculum::start());
        feed(&mut partial, COMPETENCE_WINDOW - 1, 1.0);
        assert_eq!(
            partial.curriculum().band(),
            Curriculum::start().band(),
            "a partial window cannot advance even at a perfect reach-fraction"
        );
    }

    #[test]
    fn never_regresses_and_caps_at_the_arena() {
        // Mastering rung after rung walks the band outward monotonically and stops at the
        // arena cap — the far edge never exceeds TARGET_ARENA_HALF, and once capped it
        // stays put no matter how much more competence arrives.
        let mut p = CurriculumProgress::new(Curriculum::start());
        let mut prev_min = BAND_START_MIN;
        // Far more competence than any finite number of rungs needs.
        for _ in 0..(2 * (TARGET_ARENA_HALF as usize) + 10) {
            feed(&mut p, COMPETENCE_WINDOW, 1.0);
            let (min, max) = p.curriculum().band();
            assert!(
                min >= prev_min,
                "the band must never slide inward (no regress)"
            );
            assert!(
                max <= TARGET_ARENA_HALF + 1e-6,
                "the far edge must never exceed the arena cap, got {max}"
            );
            prev_min = min;
        }
        let (min, max) = p.curriculum().band();
        assert!(
            (max - TARGET_ARENA_HALF).abs() < 1e-6,
            "enough mastery must drive the far edge to the arena cap, got {max}"
        );
        // Width is preserved across every advance (modulo the final cap clamp, which
        // shortens BOTH edges equally, so the width is identical to the start band's).
        assert!(
            (max - min - (BAND_START_MAX - BAND_START_MIN)).abs() < 1e-6,
            "the band width is invariant across rungs"
        );
        // Capped: `advanced()` yields nothing, so further mastery is a no-op.
        assert_eq!(
            p.curriculum().advanced(),
            None,
            "the capped band cannot advance"
        );
        feed(&mut p, COMPETENCE_WINDOW, 1.0);
        assert_eq!(
            p.curriculum().band(),
            (min, max),
            "a capped curriculum ignores further competence"
        );
    }

    #[test]
    fn advance_clears_the_window_so_the_new_rung_is_judged_fresh() {
        // After an advance the window is empty, so the next rung needs its own full
        // window before it can advance again — competence does not carry across rungs.
        let mut p = CurriculumProgress::new(Curriculum::start());
        feed(&mut p, COMPETENCE_WINDOW, 1.0); // advances to rung 2
        let rung2 = p.curriculum().band();
        // One episode short of a fresh full window on the new rung: must not advance.
        feed(&mut p, COMPETENCE_WINDOW - 1, 1.0);
        assert_eq!(
            p.curriculum().band(),
            rung2,
            "the new rung must accumulate its own full window before advancing"
        );
        // The episode that completes the fresh window advances again.
        feed(&mut p, 1, 1.0);
        assert_ne!(
            p.curriculum().band(),
            rung2,
            "completing a fresh full window at competence advances the new rung"
        );
    }

    #[test]
    fn record_episodes_matches_individual_records() {
        // The pooled-count path the learner uses (`record_episodes`) must advance
        // identically to recording episodes one at a time.
        let mut pooled = CurriculumProgress::new(Curriculum::start());
        pooled.record_episodes(COMPETENCE_WINDOW as u64, COMPETENCE_WINDOW as u64);
        let mut singly = CurriculumProgress::new(Curriculum::start());
        for _ in 0..COMPETENCE_WINDOW {
            singly.record_episode(true);
        }
        assert_eq!(
            pooled.curriculum().band(),
            singly.curriculum().band(),
            "pooled counts and individual records must advance the band identically"
        );
    }

    #[test]
    fn record_episodes_drops_leftovers_after_an_advance() {
        // A pooled batch that advances mid-fold must NOT seed the freshly cleared rung
        // with its remaining episodes — those were rolled against the old (nearer, easier)
        // band. With the window one short of full, the batch's first episode advances and
        // the other nine are leftovers that must be discarded, leaving the new rung empty.
        let mut p = CurriculumProgress::new(Curriculum::start());
        feed(&mut p, COMPETENCE_WINDOW - 1, 1.0);
        let rung1 = p.curriculum().band();
        p.record_episodes(10, 10);
        let rung2 = p.curriculum().band();
        assert_ne!(
            rung2, rung1,
            "the batch's first episode completes the window and advances"
        );
        // Had the nine leftovers seeded rung 2's window, a further WINDOW-1 reached
        // episodes would overfill it and advance again; dropped, WINDOW-1 leaves the new
        // window one short, so the band must stay at rung 2.
        feed(&mut p, COMPETENCE_WINDOW - 1, 1.0);
        assert_eq!(
            p.curriculum().band(),
            rung2,
            "leftover old-band episodes must not seed the freshly cleared rung's window"
        );
    }

    #[test]
    fn missing_or_corrupt_checkpoint_loads_rung_one() {
        let dir = std::env::temp_dir().join(format!("rl-curric-load-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = CheckpointDir::new(&dir).curriculum_path();

        // No file at all (fresh run OR a checkpoint predating the curriculum) → rung 1.
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            load_curriculum(&path).band(),
            Curriculum::start().band(),
            "a missing curriculum checkpoint must start at rung 1 (warm-continue safety)"
        );

        // Garbage bytes (corrupt file) → rung 1, not a panic or an illegal band.
        std::fs::write(&path, b"not a curriculum").expect("write garbage");
        assert_eq!(
            load_curriculum(&path).band(),
            Curriculum::start().band(),
            "a corrupt curriculum checkpoint must fall back to rung 1"
        );

        // An in-bounds advanced band round-trips exactly.
        let advanced = advanced_to_cap();
        save_curriculum(advanced, &path);
        assert_eq!(
            load_curriculum(&path).band(),
            advanced.band(),
            "a saved band must reload to the same rung (warm restart continues the curriculum)"
        );

        // A persisted band that violates the invariant (e.g. an out-of-arena far edge)
        // is rejected on load → rung 1.
        let bad = bincode::serialize(&CurriculumData {
            min: 1.5,
            max: TARGET_ARENA_HALF + 5.0,
        })
        .expect("serialize");
        std::fs::write(&path, &bad).expect("write bad band");
        assert_eq!(
            load_curriculum(&path).band(),
            Curriculum::start().band(),
            "an out-of-bounds persisted band must be rejected and fall back to rung 1"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Feed `n` finished episodes at a fixed reach-fraction into a progress tracker.
    /// `reached` of every `COMPETENCE_WINDOW`-sized chunk are reaches — but the helper
    /// just streams `n` episodes whose reach pattern hits `fraction`.
    fn feed(progress: &mut CurriculumProgress, episodes: usize, fraction: f32) {
        for i in 0..episodes {
            // Deterministic pattern that converges to `fraction` reached: reach iff the
            // running reached-count is below the target ratio. Over a full window this
            // lands within one episode of `fraction`.
            let reached = ((i as f32 + 1.0) * fraction).floor() > (i as f32 * fraction).floor();
            progress.record_episode(reached);
        }
    }
}
