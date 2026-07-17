use bevy::prelude::*;

use crate::bot::CrabSpawns;
use crate::bot::body::{CrabClawTip, CrabEnvId};
use crate::bot::sensor::CrabTargets;
use crate::terrain::TerrainGrid;
use crate::training::reward::dist_3d;

pub(crate) const BAND_START_MIN: f32 = 1.5;
/// Target height band, meters ABOVE the terrain surface at the target's own (x, z) —
/// equal to absolute y on the flat arenas; on terrain (rl#281) the ball hugs the
/// ground wherever it lands.
pub(crate) const TARGET_Y_MIN: f32 = 0.15;
pub(crate) const TARGET_Y_MAX: f32 = 0.7;
/// Far edge of the trained chase band — the ONE source for "how far a target can
/// be". On the flat training box it doubles as the coordinate bound sampling clamps
/// targets to (|x|,|z| ≤ this), so extending the band past the wall margin needs a
/// bigger arena, not just a bigger constant. GCR's bridge (`net::external_crab`)
/// clamps its posed hunt target to this same constant; a copied literal there would
/// drift when the band moves.
pub const TARGET_ARENA_HALF: f32 = crate::physics::world::ARENA_HALF_SIZE - 1.0;

/// Margin kept off the terrain tile's edge when sampling spawns/targets — past the
/// last grid row the surface continues as a flat clamp-extension (see
/// [`TerrainGrid::height`]), real relief stops, and the rl#283 y-floor backstop is
/// the only netting; one chase band plus slack keeps every episode's whole geometry
/// on real terrain.
const TERRAIN_EDGE_MARGIN: f32 = 2.0 * TARGET_ARENA_HALF;

/// Absolute |x|,|z| bound for spawn/target sampling — the wall-derived clamp on the
/// flat training box, the tile interior on terrain (rl#281 stage 4). ONE source: the
/// per-episode spawn draw and [`sample_target`]'s in-arena check share it, so a spawn
/// can never be placed where its own targets don't fit.
pub(crate) fn sample_clamp_half(terrain: &TerrainGrid) -> f32 {
    if terrain.is_flat() {
        TARGET_ARENA_HALF
    } else {
        terrain.extent_x().min(terrain.extent_z()) / 2.0 - TERRAIN_EDGE_MARGIN
    }
}

/// A fresh episode locale on terrain: uniform over the tile interior, on the surface.
/// Terrain training re-draws this every respawn (`reset_crab`) so the policy sees the
/// tile's whole slope/relief distribution instead of the fixed spawn grid's one
/// locale; flat arenas never call it (the walled box IS one locale).
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

const NEAR_BIAS_EXP: f32 = 2.0;

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
/// most one band edge ([`TARGET_ARENA_HALF`]) from the crab along the true planar
/// bearing, `y` the caller's (net poses at its claw height, the eval at the real
/// ball's). THE one copy of the clamp both consumers share — net's
/// `set_crab_walk_target` (a distant GCR player) and the eval's pace probe (a distant
/// ball, rl#280) — so the band semantics cannot drift between the plant she is
/// deployed on and the instrument that measures her for it.
pub fn band_lure(carapace: Vec3, planar_to_target: Vec2, y: f32) -> Vec3 {
    let to_target = planar_to_target.clamp_length_max(TARGET_ARENA_HALF);
    Vec3::new(carapace.x + to_target.x, y, carapace.z + to_target.y)
}

/// The rl#240 recenter's trigger and delta: once the carapace's planar drift from its
/// spawn origin leaves the band, teleporting every crab part by exactly this delta
/// puts it back onto the origin planar-wise; inside the band, `None`. The y component
/// carries the terrain's surface-height difference across the shift, so
/// height-above-ground (and with it the spawn-relative body:pos.y obs channel, which
/// walks with elevation on terrain — rl#283's stage-4 audit item) is invariant; on the
/// flat grids both heights are exactly 0 and the legacy y=0 delta falls out
/// bit-identical. ONE copy for the same reason as [`band_lure`]: net's
/// `bound_body_pos_drift` and the eval's `pace_recenter` must agree on when the
/// spawn-relative body.pos obs channel is out of distribution.
pub fn recenter_delta(origin: Vec3, carapace: Vec3, terrain: &TerrainGrid) -> Option<Vec3> {
    let drift = Vec2::new(carapace.x - origin.x, carapace.z - origin.z);
    (drift.length() > TARGET_ARENA_HALF).then(|| {
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
pub(crate) fn sample_target(
    origin: Vec3,
    close_frac: f32,
    rng: &mut impl rand::Rng,
    terrain: &TerrainGrid,
) -> Vec3 {
    let dist = if rng.gen_range(0.0..1.0) < close_frac {
        rng.gen_range(0.0..BAND_START_MIN)
    } else {
        let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
        let u: f32 = rng.gen_range(0.0..1.0);
        min + (max - min) * u.powf(NEAR_BIAS_EXP)
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
    close_frac: f32,
    rng: &mut rand::rngs::StdRng,
    terrain: &TerrainGrid,
) {
    if let Some(slot) = targets.envs.get_mut(e) {
        let origin = spawns.origin(e);
        *slot = Some(sample_target(origin, close_frac, rng, terrain));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::reward::planar_dist;

    /// The training arena's flat grid — these tests pin the FLAT-arena band semantics.
    fn flat() -> TerrainGrid {
        TerrainGrid::flat(crate::physics::world::ARENA_HALF_SIZE)
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
        let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
        for origin in [
            Vec3::ZERO,
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(8.0, 0.0, -8.0),
        ] {
            for _ in 0..2000 {
                let t = sample_target(origin, 0.0, &mut rng, &flat());
                assert!(t.is_finite(), "a sampled target is always finite");
                assert!(
                    t.x.abs() <= TARGET_ARENA_HALF && t.z.abs() <= TARGET_ARENA_HALF,
                    "target {t:?} from {origin:?} must stay inside ±{TARGET_ARENA_HALF} m"
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
    // that samples span [BAND_START_MIN, TARGET_ARENA_HALF].)

    #[test]
    fn distance_draw_is_near_heavy_with_a_real_far_tail() {
        let mut rng = rand::thread_rng();
        let origin = Vec3::ZERO;
        let (min, max) = (BAND_START_MIN, TARGET_ARENA_HALF);
        let near_edge = 3.0;
        let far_edge = 6.0;
        let (mut near, mut far, n) = (0u32, 0u32, 20_000u32);
        for _ in 0..n {
            let d = planar_dist(sample_target(origin, 0.0, &mut rng, &flat()), origin);
            if d < near_edge {
                near += 1;
            }
            if d > far_edge {
                far += 1;
            }
        }
        let near_frac = near as f32 / n as f32;
        let far_frac = far as f32 / n as f32;
        assert!(
            near_frac > 0.40,
            "near ({min}-{near_edge} m) fraction {near_frac} should dominate (uniform would be ~0.2; EXP=2 ~0.45)"
        );
        assert!(
            far_frac > 0.15,
            "far (>{far_edge} m, up to {max} m) fraction {far_frac} must keep a FAT tail (EXP=2 ~0.22), not starve far"
        );
    }

    #[test]
    fn close_targets_cover_the_under_carapace_disc() {
        let mut rng = rand::thread_rng();
        for origin in [Vec3::ZERO, Vec3::new(8.0, 0.0, -8.0)] {
            let mut under_body = 0u32;
            for _ in 0..5000 {
                let t = sample_target(origin, 1.0, &mut rng, &flat());
                assert!(t.is_finite());
                assert!(t.x.abs() <= TARGET_ARENA_HALF && t.z.abs() <= TARGET_ARENA_HALF);
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
        let far = polar_target(carapace, 0.7, 2.0 * TARGET_ARENA_HALF, 0.6);
        let to_far = Vec2::new(far.x - carapace.x, far.z - carapace.z);

        let lure = band_lure(carapace, to_far, far.y);
        let planar = Vec2::new(lure.x - carapace.x, lure.z - carapace.z);
        assert!((planar.length() - TARGET_ARENA_HALF).abs() < 1e-4);
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

    /// The rl#240 recenter formula: dormant inside the band; one band-edge step
    /// outside it, the delta lands the carapace back on its origin planar-wise with
    /// y untouched on the flat grid.
    #[test]
    fn recenter_delta_snaps_planar_drift_back_onto_the_origin() {
        let origin = Vec3::new(2.0, 0.0, -3.0);

        let inside = origin + Vec3::new(TARGET_ARENA_HALF - 0.1, 0.5, 0.0);
        assert_eq!(recenter_delta(origin, inside, &flat()), None);

        let out = origin + Vec3::new(TARGET_ARENA_HALF * 0.8, 0.5, TARGET_ARENA_HALF * 0.8);
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
        let out_xz = Vec2::new(500.0 + TARGET_ARENA_HALF * 1.5, -700.0);
        let clearance = 0.4;
        let out = g.place(out_xz, clearance);
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
    /// tile interior; the flat clamp stays the wall-derived constant.
    #[test]
    fn terrain_targets_band_around_the_origin_on_the_surface() {
        let g = TerrainGrid::gcr();
        assert_eq!(sample_clamp_half(&flat()), TARGET_ARENA_HALF);
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
                    (BAND_START_MIN - 1e-3..=TARGET_ARENA_HALF + 1e-3).contains(&d),
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
