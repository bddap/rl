use bevy::prelude::*;

use crate::bot::CrabSpawns;
use crate::bot::body::{CrabClawTip, CrabEnvId};
use crate::bot::sensor::CrabTargets;
use crate::terrain::TerrainGrid;
use crate::training::reward::dist_3d;

pub(crate) const BAND_START_MIN: f32 = 1.5;
/// Target height band, meters ABOVE the terrain surface at the target's own (x, z) —
/// the ball hugs the ground wherever it lands (rl#281).
pub(crate) const TARGET_Y_MIN: f32 = 0.15;
pub(crate) const TARGET_Y_MAX: f32 = 0.7;
/// Far edge of the trained chase band — the ONE source for "how far a target can
/// be" (rl#292: 0→100 m+ genuinely in-distribution; the old 9 m edge was the flat
/// box's wall margin, not a design choice). Since rl#298 stage 5 GCR poses its hunt
/// target AT the prey unclamped (on the live map prey never nears the edge); the
/// eval's pace probe is the surviving [`band_lure`] consumer.
pub const BAND_MAX_M: f32 = 128.0;

/// The spawn-relative `body.pos` obs-support radius — the recenter/rebase trigger
/// ([`recenter_delta`]), SPLIT from the target band (rl#292/rl#293): the band is how
/// far a ball may SPAWN, this is how far the obs channel may drift from the episode
/// origin before consumers re-gauge, bounded by per-episode traverse (~35 m at the
/// pinned pace over one horizon), NOT by the band. Kept at the historic 9 m the
/// deployed brains trained under; re-pin from pace evidence, never by tying it back
/// to [`BAND_MAX_M`] — at 128 m GCR would run `body.pos` ~4× past anything training
/// ever saw (the exact OOD class the recenter seam closes).
pub const DRIFT_REBASE_M: f32 = 9.0;

/// Fraction of target draws seeded in the close disc [0, [`BAND_START_MIN`]) —
/// under-carapace included — so claw-reach stays in-distribution beside the chase
/// (rl#250). A canonical constant since rl#292: ball-under is a standing directive,
/// not a per-run curriculum flag.
pub(crate) const CLOSE_FRAC: f32 = 0.1;

/// Margin kept off the terrain tile's edge when sampling spawns/targets — past the
/// last grid row the surface continues as a flat clamp-extension (see
/// [`TerrainGrid::height`]), real relief stops, and the rl#283 y-floor backstop is
/// the only netting; one chase band plus slack keeps every episode's whole geometry
/// on real terrain.
const TERRAIN_EDGE_MARGIN: f32 = 2.0 * BAND_MAX_M;

/// Absolute |x|,|z| bound for spawn/target sampling — the grid interior less the
/// edge margin, whatever the grid (rl#293: one formula, no flat fork). ONE source:
/// the per-episode spawn draw and [`sample_target`]'s in-arena check share it, so a
/// spawn can never be placed where its own targets don't fit. The assert makes a
/// too-small grid — a future undersized bake, or a flat TEST grid under
/// [`TERRAIN_EDGE_MARGIN`] + band — fail at first sample instead of silently
/// sampling nonsense (a negative clamp NaNs band distances and panics the origin
/// draw's `gen_range`).
pub(crate) fn sample_clamp_half(terrain: &TerrainGrid) -> f32 {
    let clamp = terrain.extent_x().min(terrain.extent_z()) / 2.0 - TERRAIN_EDGE_MARGIN;
    assert!(
        clamp > BAND_MAX_M,
        "grid half-span {} m leaves sampling clamp {clamp} m ≤ the {BAND_MAX_M} m band \
         — the whole chase geometry must fit on real ground",
        terrain.extent_x().min(terrain.extent_z()) / 2.0,
    );
    clamp
}

/// A fresh episode locale: uniform over the tile interior, on the surface. Training
/// re-draws this every respawn (`reset_crab`) so the policy sees the tile's whole
/// slope/relief distribution instead of the fixed spawn grid's one locale.
pub(crate) fn random_episode_origin(rng: &mut impl rand::Rng, terrain: &TerrainGrid) -> Vec3 {
    let clamp = sample_clamp_half(terrain);
    let xz = Vec2::new(rng.gen_range(-clamp..clamp), rng.gen_range(-clamp..clamp));
    terrain.place(xz, 0.0)
}

/// Claw-tip "touch" distance for [`tip_touch`]. Finer than the crab (rl#253):
/// inside the carapace footprint's inscribed radius, with the rest pose's claw
/// tips clear of the whole under-body disc (both pinned by
/// `reach_radius_is_finer_than_the_crab`), so a target seeded under the
/// carapace (rl#250) is only touchable by an actual claw strike — the skill
/// GCR's collider-contact downing (rl#249) needs. The floor is GCR's contact
/// scale, claw capsule radius + bridged `CLAW_DOWN_BUFFER` (`net::sim`) ≈
/// 0.1 m in these metres: 0.2 stays ~2× that, so exploration can plausibly
/// discover the grab.
pub(crate) const REACH_RADIUS: f32 = 0.2;

/// THE touch predicate — a claw tip within [`REACH_RADIUS`] of the target —
/// one source, so "touched" cannot drift between surfaces.
pub(crate) fn tip_touch(min_tip_dist: f32) -> bool {
    min_tip_dist < REACH_RADIUS
}

/// Closest claw tip to `target` for one env — the single-env form of the
/// measurement [`tip_touch`] judges (training's batched form is
/// `closest_tip_dists` in `systems::step`). `None` until a finite tip is seen.
pub(crate) fn closest_tip_dist(
    env: usize,
    target: Vec3,
    tips: &Query<(&CrabEnvId, &Transform), With<CrabClawTip>>,
) -> Option<f32> {
    let mut min: Option<f32> = None;
    for (e, tip) in tips.iter() {
        if e.0 == env && tip.translation.is_finite() {
            let d = dist_3d(tip.translation, target);
            min = Some(min.map_or(d, |cur| cur.min(d)));
        }
    }
    min
}

/// The ONE polar target-placement formula — `theta` from `origin` in the ground plane,
/// (cos, ·, sin) — shared by training sampling and the eval compass so the bearing
/// convention cannot drift between them. `y` is copied through verbatim; callers that
/// place on terrain re-lift the point with [`TerrainGrid::place`](crate::terrain::TerrainGrid::place).
pub(crate) fn polar_target(origin: Vec3, theta: f32, dist: f32, y: f32) -> Vec3 {
    Vec3::new(
        origin.x + dist * theta.cos(),
        y,
        origin.z + dist * theta.sin(),
    )
}

/// The rl#240 in-distribution guard's posed walk target: the real target re-posed at
/// most one band edge ([`BAND_MAX_M`]) from the crab along the true planar bearing,
/// `y` the caller's. The eval's pace probe (a distant ball, rl#280) is the surviving
/// consumer since rl#298 stage 5 deleted GCR's dormant clamp with the bridge. Since
/// rl#292 the edge is 128 m: a target inside it is presented WHERE IT IS, not lured
/// nearer.
pub fn band_lure(carapace: Vec3, planar_to_target: Vec2, y: f32) -> Vec3 {
    let to_target = planar_to_target.clamp_length_max(BAND_MAX_M);
    Vec3::new(carapace.x + to_target.x, y, carapace.z + to_target.y)
}

/// The rl#240 recenter's trigger and delta: once the carapace's planar drift from its
/// spawn origin leaves the obs-support radius ([`DRIFT_REBASE_M`] — NOT the target
/// band, rl#292), this is the exact shift back onto the origin; inside it, `None`. Consumers apply it two ways (rl#281 stage 6): the eval's
/// `pace_recenter` TELEPORTS the crab by the delta (fixed-locale measurement — see its
/// doc), while net's `recenter_drifted_origins` uses only the trigger and REBASES the
/// origin instead (`CrabSpawns::rebase_origin_to` — a rendered world must stay glued
/// under her feet). The y component carries the terrain's surface-height difference
/// across the teleport, so height-above-ground (and with it the spawn-relative
/// body:pos.y obs channel, which walks with elevation on terrain — rl#283's stage-4
/// audit item) is invariant; on the flat grids both heights are exactly 0 and the
/// legacy y=0 delta falls out bit-identical. ONE copy for the same reason as
/// [`band_lure`]: every consumer must agree on when the spawn-relative body.pos obs
/// channel is out of distribution.
pub fn recenter_delta(origin: Vec3, carapace: Vec3, terrain: &TerrainGrid) -> Option<Vec3> {
    let drift = Vec2::new(carapace.x - origin.x, carapace.z - origin.z);
    (drift.length() > DRIFT_REBASE_M).then(|| {
        Vec3::new(
            -drift.x,
            terrain.height(origin.x, origin.z) - terrain.height(carapace.x, carapace.z),
            -drift.y,
        )
    })
}

/// `close_frac` mixes close-disc [0, BAND_START_MIN) targets — under-carapace
/// included — into the chase band, so claw-reach can emerge from task variety
/// alone (rl#250): same y range, bearing, reward, and grab rule. Uniform-in-radius
/// on purpose: the density concentrates where the new skill lives, and targets the
/// rest pose already touches are re-seeded at episode start (`pre_touched_target`),
/// which carves the true no-op boundary better than any hand-drawn annulus could.
/// The band draw is LOG-UNIFORM over [BAND_START_MIN, BAND_MAX_M] (rl#292): equal
/// probability mass per distance octave, so ~40% of draws land inside the old 9 m
/// regime (near-field skill keeps its gradient) while 100 m+ treks carry real mass —
/// a uniform draw at this range would starve the near field to ~6%.
pub(crate) fn sample_target(
    origin: Vec3,
    close_frac: f32,
    rng: &mut impl rand::Rng,
    terrain: &TerrainGrid,
) -> Vec3 {
    let dist = if rng.gen_range(0.0..1.0) < close_frac {
        rng.gen_range(0.0..BAND_START_MIN)
    } else {
        let (min, max) = (BAND_START_MIN, BAND_MAX_M);
        let u: f32 = rng.gen_range(0.0..1.0);
        min * (max / min).powf(u)
    };
    let y = rng.gen_range(TARGET_Y_MIN..TARGET_Y_MAX);
    let at = |theta: f32| {
        let p = polar_target(origin, theta, dist, y);
        terrain.place(Vec2::new(p.x, p.z), y)
    };
    let clamp = sample_clamp_half(terrain);
    let in_arena = |p: &Vec3| p.x.abs() <= clamp && p.z.abs() <= clamp;

    for _ in 0..32 {
        let p = at(rng.gen_range(0.0..std::f32::consts::TAU));
        if in_arena(&p) {
            return p;
        }
    }
    let inward = Vec3::new(-origin.x, 0.0, -origin.z).normalize_or_zero();
    let theta = if inward == Vec3::ZERO {
        0.0
    } else {
        inward.z.atan2(inward.x)
    };
    at(theta)
}

pub(crate) fn seed_target(
    targets: &mut CrabTargets,
    spawns: &CrabSpawns,
    e: usize,
    rng: &mut rand::rngs::StdRng,
    terrain: &TerrainGrid,
) {
    if let Some(slot) = targets.envs.get_mut(e) {
        let origin = spawns.origin(e);
        *slot = Some(sample_target(origin, CLOSE_FRAC, rng, terrain));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::reward::planar_dist;

    /// A flat TEST grid big enough for the rl#292 band: clamp = half − edge margin
    /// must clear [`BAND_MAX_M`] plus the test origins (see [`sample_clamp_half`]).
    fn flat() -> TerrainGrid {
        TerrainGrid::flat(512.0)
    }

    /// The flat test grid's sampling clamp — the arena bound the in-arena asserts use.
    fn flat_clamp() -> f32 {
        sample_clamp_half(&flat())
    }

    /// rl#253's invariant, geometrically: the touch sphere is finer than the crab.
    /// If REACH_RADIUS out-reaches the carapace footprint or the rest pose's claw
    /// tips can already touch under-footprint points, the close-disc curriculum
    /// (rl#250) degenerates back to re-seed-everything and under-body touch is
    /// untrainable. Planar clearance bounds the 3D distance from below, so no
    /// settled-pose y is needed. Checks every recipe buildable on this host.
    /// Pins the bind pose only; tilted spawns and post-settle drift are
    /// backstopped at runtime by `pre_touched_target`'s re-seed.
    #[test]
    fn reach_radius_is_finer_than_the_crab() {
        use crate::bot::body::CrabJointId;
        use crate::bot::rig::{fallback_recipe, link_world_origins};

        let mut recipes = vec![("fallback", fallback_recipe())];
        if let Ok(u) = crate::mesh_fallback::usable_model() {
            recipes.push(("fitted", u.recipe.clone()));
        }
        for (name, recipe) in &recipes {
            let inscribed = recipe.carapace_half.x.min(recipe.carapace_half.z);
            assert!(
                REACH_RADIUS < inscribed,
                "{name}: REACH_RADIUS {REACH_RADIUS} m out-reaches the carapace's inscribed \
                 footprint radius {inscribed} m — touch is not finer than the crab"
            );

            let origins = link_world_origins(&recipe.links, recipe.hub_bind_world);
            let footprint_center = recipe.hub_bind_world + recipe.carapace_offset;
            let mut tips = 0usize;
            for (link, &tip) in recipe.links.iter().zip(&origins) {
                if !matches!(link.actuated, Some(CrabJointId::ClawPincer(_))) {
                    continue;
                }
                tips += 1;
                let dx = ((tip.x - footprint_center.x).abs() - recipe.carapace_half.x).max(0.0);
                let dz = ((tip.z - footprint_center.z).abs() - recipe.carapace_half.z).max(0.0);
                let clearance = dx.hypot(dz);
                assert!(
                    clearance > REACH_RADIUS,
                    "{name}: rest claw tip {tip:?} clears the carapace footprint by only \
                     {clearance} m ≤ REACH_RADIUS {REACH_RADIUS} m — the rest pose would \
                     auto-touch under-carapace targets"
                );
            }
            assert_eq!(tips, 2, "{name}: two claw-tip links measure the grab");
        }
    }

    #[test]
    fn sampled_targets_lie_at_the_band_distance_and_inside_the_arena() {
        let mut rng = rand::thread_rng();
        let (min, max) = (BAND_START_MIN, BAND_MAX_M);
        let clamp = flat_clamp();
        for origin in [
            Vec3::ZERO,
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(80.0, 0.0, -80.0),
        ] {
            for _ in 0..2000 {
                let t = sample_target(origin, 0.0, &mut rng, &flat());
                assert!(t.is_finite(), "a sampled target is always finite");
                assert!(
                    t.x.abs() <= clamp && t.z.abs() <= clamp,
                    "target {t:?} from {origin:?} must stay inside ±{clamp} m"
                );
                assert!(t.y >= TARGET_Y_MIN && t.y <= TARGET_Y_MAX);
                let d = planar_dist(t, origin);
                assert!(
                    d >= min - 1e-3 && d <= max + 1e-3,
                    "target from {origin:?} is at {d} m, outside the band [{min}, {max}]"
                );
            }
        }
    }

    // (No "band is the full arena" test: with TargetBand deleted the band IS the two
    // constants — a test would just restate them. The honest-distance test above pins
    // that samples span [BAND_START_MIN, BAND_MAX_M].)

    /// The rl#292 log-uniform draw: equal mass per distance octave — the old 9 m
    /// regime keeps ~40% of draws (near-field skill keeps its gradient), 100 m+
    /// carries real mass (a uniform draw would put ~94% of mass past 9 m and starve
    /// the near field; a near-biased power draw starves 100 m+ — this pins both
    /// tails at once).
    #[test]
    fn distance_draw_is_log_uniform_near_heavy_with_a_real_100m_tail() {
        let mut rng = rand::thread_rng();
        let origin = Vec3::ZERO;
        let n = 20_000u32;
        let (mut near9, mut far32, mut far100) = (0u32, 0u32, 0u32);
        for _ in 0..n {
            let d = planar_dist(sample_target(origin, 0.0, &mut rng, &flat()), origin);
            if d < 9.0 {
                near9 += 1;
            }
            if d > 32.0 {
                far32 += 1;
            }
            if d > 100.0 {
                far100 += 1;
            }
        }
        let (near9, far32, far100) = (
            near9 as f32 / n as f32,
            far32 as f32 / n as f32,
            far100 as f32 / n as f32,
        );
        assert!(
            near9 > 0.35,
            "old-band (<9 m) fraction {near9} must stay near-heavy (log-uniform ~0.40)"
        );
        assert!(
            far32 > 0.25,
            "far (>32 m) fraction {far32} must carry real mass (log-uniform ~0.31)"
        );
        assert!(
            far100 > 0.03,
            "100 m+ fraction {far100} must be genuinely in-distribution (log-uniform ~0.055), \
             not a vanishing tail"
        );
    }

    /// rl#292 owner constraint: NO target ever spawns below the terrain surface — at
    /// any bearing and any distance across the WHOLE distribution (under-carapace
    /// disc through the 100 m+ band), the one placement path samples the heightfield
    /// at the target's own (x, z) and lands the ball on the surface's y band. A fixed
    /// spawn height would fail this the moment a draw reaches a hilly cell.
    #[test]
    fn no_target_ever_spawns_below_the_terrain_surface() {
        let g = TerrainGrid::gcr();
        let mut rng = rand::thread_rng();
        for close_frac in [0.0, 1.0] {
            for _ in 0..32 {
                let origin = random_episode_origin(&mut rng, &g);
                for _ in 0..64 {
                    let t = sample_target(origin, close_frac, &mut rng, &g);
                    let above = t.y - g.height(t.x, t.z);
                    assert!(
                        (TARGET_Y_MIN - 1e-3..=TARGET_Y_MAX + 1e-3).contains(&above),
                        "target {t:?} (close_frac {close_frac}) rides {above} m above \
                         the surface — outside [{TARGET_Y_MIN}, {TARGET_Y_MAX}]; \
                         negative would be IN the ground"
                    );
                }
            }
        }
    }

    #[test]
    fn close_targets_cover_the_under_carapace_disc() {
        let mut rng = rand::thread_rng();
        for origin in [Vec3::ZERO, Vec3::new(80.0, 0.0, -80.0)] {
            let mut under_body = 0u32;
            for _ in 0..5000 {
                let t = sample_target(origin, 1.0, &mut rng, &flat());
                assert!(t.is_finite());
                assert!(t.x.abs() <= flat_clamp() && t.z.abs() <= flat_clamp());
                assert!(t.y >= TARGET_Y_MIN && t.y <= TARGET_Y_MAX);
                let d = planar_dist(t, origin);
                assert!(
                    d < BAND_START_MIN,
                    "close target from {origin:?} is at {d} m, outside [0, {BAND_START_MIN})"
                );
                if d < 0.5 {
                    under_body += 1;
                }
            }
            assert!(
                under_body > 500,
                "under-carapace (<0.5 m) draws {under_body}/5000 — the disc must reach under the body (uniform radius ~1/3)"
            );
        }
    }

    #[test]
    fn close_frac_mixes_and_zero_is_the_pure_band() {
        let mut rng = rand::thread_rng();
        let origin = Vec3::ZERO;
        let n = 20_000u32;
        let close = (0..n)
            .filter(|_| {
                planar_dist(sample_target(origin, 0.25, &mut rng, &flat()), origin) < BAND_START_MIN
            })
            .count() as f32
            / n as f32;
        assert!(
            (0.20..0.30).contains(&close),
            "close fraction {close} should track the requested 0.25"
        );
        for _ in 0..2000 {
            let d = planar_dist(sample_target(origin, 0.0, &mut rng, &flat()), origin);
            assert!(
                d >= BAND_START_MIN - 1e-3,
                "frac 0 must never sample close ({d})"
            );
        }
    }

    /// The rl#240 lure: beyond the band the posed ball rides exactly one band edge
    /// ahead along the true bearing; inside it, the posed ball IS the real ball.
    #[test]
    fn band_lure_clamps_to_the_band_edge_along_the_true_bearing() {
        let carapace = Vec3::new(1.0, 0.4, -2.0);
        let far = polar_target(carapace, 0.7, 2.0 * BAND_MAX_M, 0.6);
        let to_far = Vec2::new(far.x - carapace.x, far.z - carapace.z);

        let lure = band_lure(carapace, to_far, far.y);
        let planar = Vec2::new(lure.x - carapace.x, lure.z - carapace.z);
        assert!((planar.length() - BAND_MAX_M).abs() < 1e-2);
        assert!(
            planar.normalize().dot(to_far.normalize()) > 1.0 - 1e-6,
            "the lure sits on the real bearing"
        );
        assert_eq!(lure.y, far.y);

        // Inside the band nothing is clamped — the chase end is a real approach.
        let near = polar_target(carapace, 0.7, 3.0, 0.6);
        let to_near = Vec2::new(near.x - carapace.x, near.z - carapace.z);
        assert!((band_lure(carapace, to_near, near.y) - near).length() < 1e-5);
    }

    /// The rl#240 recenter formula: dormant inside the obs-support radius; one step
    /// outside it, the delta lands the carapace back on its origin planar-wise with
    /// y untouched on the flat grid.
    #[test]
    fn recenter_delta_snaps_planar_drift_back_onto_the_origin() {
        let origin = Vec3::new(2.0, 0.0, -3.0);

        let inside = origin + Vec3::new(DRIFT_REBASE_M - 0.1, 0.5, 0.0);
        assert_eq!(recenter_delta(origin, inside, &flat()), None);

        let out = origin + Vec3::new(DRIFT_REBASE_M * 0.8, 0.5, DRIFT_REBASE_M * 0.8);
        let delta = recenter_delta(origin, out, &flat()).expect("outside the band");
        assert_eq!(
            delta.y, 0.0,
            "flat grid: the legacy y=0 delta, bit-identical"
        );
        let back = out + delta;
        assert!((back.x - origin.x).abs() < 1e-5 && (back.z - origin.z).abs() < 1e-5);
        assert_eq!(back.y, out.y, "flat recenter never touches height");
    }

    /// On terrain the recenter carries the surface-height difference: after the
    /// shift, height-above-ground at the origin equals what it was at the drifted
    /// spot — the spawn-relative body:pos.y channel is invariant across the teleport.
    #[test]
    fn recenter_delta_preserves_height_above_terrain() {
        let g = TerrainGrid::gcr();
        // Two on-tile points a band-plus step apart with real relief between them.
        let origin = g.place(Vec2::new(500.0, -700.0), 0.0);
        let out_xz = Vec2::new(500.0 + DRIFT_REBASE_M * 1.5, -700.0);
        let clearance = 0.4;
        let out = g.place(out_xz, clearance);
        assert!(
            (g.height(origin.x, origin.z) - g.height(out.x, out.z)).abs() > 0.5,
            "a re-bake flattened this spot — pick points with real relief or the \
             test degenerates to the flat case"
        );
        let delta = recenter_delta(origin, out, &g).expect("outside the band");
        let back = out + delta;
        let height_above = back.y - g.height(back.x, back.z);
        assert!(
            (height_above - clearance).abs() < 1e-3,
            "height above surface must survive the recenter, got {height_above}"
        );
    }

    /// Terrain sampling (rl#281 stage 4): from a random on-tile origin, targets stay
    /// in the trained band around THAT origin, on the surface's y band, inside the
    /// tile interior; the one clamp formula serves flat grids too (rl#293).
    #[test]
    fn terrain_targets_band_around_the_origin_on_the_surface() {
        let g = TerrainGrid::gcr();
        assert_eq!(flat_clamp(), 512.0 - TERRAIN_EDGE_MARGIN);
        let clamp = sample_clamp_half(&g);
        assert!(
            clamp > 15_000.0,
            "the GCR tile interior is huge, got {clamp}"
        );

        let mut rng = rand::thread_rng();
        for _ in 0..64 {
            let origin = random_episode_origin(&mut rng, &g);
            assert!(origin.x.abs() <= clamp && origin.z.abs() <= clamp);
            assert!(
                (origin.y - g.height(origin.x, origin.z)).abs() < 1e-3,
                "episode origins sit on the surface"
            );
            for _ in 0..32 {
                let t = sample_target(origin, 0.0, &mut rng, &g);
                let d = planar_dist(t, origin);
                assert!(
                    (BAND_START_MIN - 1e-3..=BAND_MAX_M + 1e-3).contains(&d),
                    "target at {d} m is outside the band from its origin"
                );
                let above = t.y - g.height(t.x, t.z);
                assert!(
                    (TARGET_Y_MIN - 1e-3..=TARGET_Y_MAX + 1e-3).contains(&above),
                    "target rides {above} m above the surface, outside the y band"
                );
            }
        }
    }
}
