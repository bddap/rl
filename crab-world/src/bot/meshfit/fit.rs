//! Collider-fit primitives: fit a capsule to a limb's vertex cloud, and score a
//! live collider's signed surface agreement against the cloud it stands in for
//! (the `--verify-colliders` regression gate). All in the glTF's bind-pose world
//! frame, the frame [`super::LoadedModel`]'s clouds live in.

use bevy::prelude::*;

/// A capsule fitted to a vertex cloud, in the cloud's own (world) frame.
#[derive(Clone, Copy, Debug)]
pub struct FittedCapsule {
    /// Segment endpoints (capsule = swept sphere between these).
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
}

#[cfg(test)]
impl FittedCapsule {
    /// Test-only: production reads the endpoints (`a`/`b`) directly; this is the
    /// convenience the capsule-fit test uses.
    pub fn segment_len(&self) -> f32 {
        (self.b - self.a).length()
    }
}

/// Perpendicular-spread percentile that sets the capsule radius. A high percentile
/// (not the max) so a thin tail of skinning-bled vertices doesn't inflate the radius.
/// 0.92 rather than 0.95 because the leg-coxa clouds carry a fatter outlier tail than
/// 5%: from p90 to p95 the middle-leg (legs 1–2) coxa radius jumps ~15–20% on a bulk
/// (p50) thickness that's near-identical across all eight legs — a fit artifact, not
/// real flesh. That extra envelope made adjacent middle-leg coxae interpenetrate ~64mm
/// at the settled rest pose, a penetration the solver fights into a visible body sag
/// (`--check-rest-colliders`); trimming the tail deflates the over-fit capsules in
/// proportion to their own tail (well-fit legs barely move), cutting that overlap ~28%
/// and letting the body settle higher. 0.92 is the floor that stays self-consistent
/// with the `--verify-colliders` radius-vs-spread gate: that gate measures the live
/// radius against the cloud's p95 spread and flags it "starved" below 0.85×, so a fit
/// percentile under ~0.90 would make a capsule fail the very gate that sized it.
const RADIUS_PERCENTILE: f32 = 0.92;

/// Fit a capsule to a point cloud:
///   - axis = first principal component (largest-variance direction),
///   - radius = [`RADIUS_PERCENTILE`] of perpendicular distance to that axis
///     (percentile, not max, so a few skinning-bleed outliers don't inflate it),
///   - segment = the axial extent shrunk by `radius` at each end, so the caps
///     cover the tips instead of the capsule overhanging by a full radius.
///
/// Returns `None` for clouds too small to fit (< 4 points).
pub fn fit_capsule(points: &[Vec3]) -> Option<FittedCapsule> {
    if points.len() < 4 {
        return None;
    }
    // Principal axis = largest-variance eigenvector of the covariance. (When the
    // top two variances are close — a chunky near-isotropic cloud — this axis is
    // ill-defined, so a capsule is the wrong primitive there; the carapace takes a
    // box instead.)
    let (centroid, axes) = covariance_eigenframe(points);
    let axis = axes[0];

    // Project onto the axis for axial extent; perpendicular residual for radius.
    let mut tmin = f32::INFINITY;
    let mut tmax = f32::NEG_INFINITY;
    let mut perp: Vec<f32> = Vec::with_capacity(points.len());
    for &p in points {
        let d = p - centroid;
        let t = d.dot(axis);
        tmin = tmin.min(t);
        tmax = tmax.max(t);
        perp.push((d - axis * t).length());
    }
    // Radius from a high perpendicular-spread percentile: robust to a tail of bled
    // vertices (see RADIUS_PERCENTILE for why that exact percentile, not the max).
    perp.sort_by(f32::total_cmp);
    let radius = percentile(&perp, RADIUS_PERCENTILE).max(1e-4);

    // Pull the endpoints in by `radius` so the spherical caps land on the tips.
    // Clamp so a stubby cloud doesn't invert into a negative segment.
    let half_axial = (tmax - tmin) * 0.5;
    let half_seg = (half_axial - radius).max(0.0);
    let mid = centroid + axis * (tmin + tmax) * 0.5;
    Some(FittedCapsule {
        a: mid - axis * half_seg,
        b: mid + axis * half_seg,
        radius,
    })
}

/// Linear-interpolated percentile of a pre-sorted slice. `q` in 0..1.
fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = q * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// How well a *live* collider surface hugs the vertex cloud it stands in for, in
/// model units, in the cloud's own (bind-pose world) frame. It keeps the sign and
/// scores the geometry the body actually spawns with: positive
/// surface distance = vertex OUTSIDE the collider (mesh pokes out → physics
/// under-coverage), negative = inside (collider margin / bulge). The split lets
/// `--verify-colliders` tell "mesh escapes the collider" from "collider oversized",
/// and the axis/radius diagnostics say *why* a capsule misses its limb.
pub struct ColliderScore {
    pub n: usize,
    /// Fraction of vertices more than 1 mm outside the collider.
    pub frac_outside: f32,
    /// 95th-percentile and max depth a vertex pokes out past the surface.
    pub poke_out_p95: f32,
    pub poke_out_max: f32,
    /// 95th-percentile depth the surface sits past the mesh on the covered side.
    pub bulge_p95: f32,
    /// Axis/radius diagnostics — `Some` only when the collider is a capsule fitted to
    /// a cloud big enough to have a well-defined principal axis. Grouping them in one
    /// `Option` makes a box-with-axis-skew unrepresentable: a box ([`score_box`]) and a
    /// too-small capsule cloud both yield `None`, so a reader can't read a meaningless
    /// skew/ratio off them (the previous `NaN`-in-band could, and only a runtime
    /// `is_nan` check stood between that and silent garbage).
    pub capsule: Option<CapsuleDiagnostics>,
}

/// Why a capsule misses (or hugs) its limb — only meaningful for a capsule collider.
#[derive(Clone, Copy, Debug)]
pub struct CapsuleDiagnostics {
    /// Angle (deg) between the capsule axis and the cloud's principal axis — large
    /// = capsule pointed off the limb.
    pub axis_skew_deg: f32,
    /// Live radius ÷ the cloud's p95 perpendicular spread about the live axis;
    /// `>1` fat, `<1` starved.
    pub radius_ratio: f32,
}

impl ColliderScore {
    /// Aggregate a set of signed surface distances. Leaves [`Self::capsule`] `None`;
    /// [`score_capsule`] fills it in when the cloud supports the diagnostics.
    fn from_signed(sd: &[f32]) -> Self {
        let inv = 1.0 / sd.len().max(1) as f32;
        let mut outs: Vec<f32> = sd.iter().copied().filter(|&d| d > 0.0).collect();
        let mut ins: Vec<f32> = sd
            .iter()
            .copied()
            .filter(|&d| d < 0.0)
            .map(f32::abs)
            .collect();
        outs.sort_by(f32::total_cmp);
        ins.sort_by(f32::total_cmp);
        let pctl = |s: &[f32], q| if s.is_empty() { 0.0 } else { percentile(s, q) };
        ColliderScore {
            n: sd.len(),
            frac_outside: sd.iter().filter(|&&d| d > 0.001).count() as f32 * inv,
            poke_out_p95: pctl(&outs, 0.95),
            poke_out_max: outs.last().copied().unwrap_or(0.0),
            bulge_p95: pctl(&ins, 0.95),
            capsule: None,
        }
    }
}

/// Score a cloud against a capsule given by its segment endpoints + radius (world).
pub fn score_capsule(points: &[Vec3], a: Vec3, b: Vec3, radius: f32) -> ColliderScore {
    let seg = b - a;
    let seg_len2 = seg.length_squared().max(1e-12);
    let sd: Vec<f32> = points
        .iter()
        .map(|&p| {
            let t = ((p - a).dot(seg) / seg_len2).clamp(0.0, 1.0);
            (p - (a + seg * t)).length() - radius
        })
        .collect();
    let mut s = ColliderScore::from_signed(&sd);
    // Diagnostics: how skewed is the live axis vs the cloud's true long axis, and
    // is the radius fat/starved relative to the cloud's spread about that axis. Only
    // defined when the cloud is big enough to have a principal axis and the segment is
    // non-degenerate; otherwise the diagnostics stay `None`.
    let axis = seg.normalize_or_zero();
    if points.len() >= 4 && axis.length_squared() > 0.5 {
        let (_centroid, axes) = covariance_eigenframe(points);
        let axis_skew_deg = axis.dot(axes[0]).abs().clamp(0.0, 1.0).acos().to_degrees();
        let mut perp: Vec<f32> = points
            .iter()
            .map(|&p| {
                let d = p - a;
                (d - axis * d.dot(axis)).length()
            })
            .collect();
        perp.sort_by(f32::total_cmp);
        let radius_ratio = radius / percentile(&perp, 0.95).max(1e-4);
        s.capsule = Some(CapsuleDiagnostics {
            axis_skew_deg,
            radius_ratio,
        });
    }
    s
}

/// Score a cloud against a world-axis-aligned box (centre + half-extents).
pub fn score_box(points: &[Vec3], center: Vec3, half: Vec3) -> ColliderScore {
    let sd: Vec<f32> = points
        .iter()
        .map(|&p| {
            let local = (p - center).abs() - half;
            let outside = local.max(Vec3::ZERO).length();
            let inside = local.max_element().min(0.0);
            outside + inside
        })
        .collect();
    ColliderScore::from_signed(&sd)
}

/// Eigenframe (orthonormal eigenvectors, sorted by descending eigenvalue) of a
/// cloud's 3x3 covariance, via cyclic Jacobi rotations. Returns the cloud's centroid
/// and the three axes longest-first — centroid here so the one place that needs it
/// for the covariance is the one place that computes it. The eigenvalues only order
/// the axes; no caller needs their magnitudes, so they aren't returned.
/// Robust and dependency-free; 3x3 converges in a handful of sweeps.
// The Jacobi sweeps update two columns (p, q) of `a`/`v` per `k`; the explicit
// index keeps the rotation readable as matrix math, so range loops stay.
#[allow(clippy::needless_range_loop)]
fn covariance_eigenframe(points: &[Vec3]) -> (Vec3, [Vec3; 3]) {
    let centroid = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    // Symmetric covariance as a [[f64;3];3].
    let mut a = [[0.0f64; 3]; 3];
    for &p in points {
        let d = [
            (p.x - centroid.x) as f64,
            (p.y - centroid.y) as f64,
            (p.z - centroid.z) as f64,
        ];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += d[i] * d[j];
            }
        }
    }
    // V accumulates the eigenvectors (columns).
    let mut v = [[0.0f64; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..16 {
        // Largest off-diagonal magnitude.
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-14 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-18 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // Apply the Jacobi rotation J^T A J and accumulate into V.
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut cols: Vec<(f64, Vec3)> = (0..3)
        .map(|j| {
            (
                a[j][j],
                Vec3::new(v[0][j] as f32, v[1][j] as f32, v[2][j] as f32).normalize_or_zero(),
            )
        })
        .collect();
    cols.sort_by(|x, y| y.0.total_cmp(&x.0));
    (centroid, [cols[0].1, cols[1].1, cols[2].1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The capsule fitter recovers a known capsule: sample points on a capsule
    /// of known axis/radius/length and check the fit lands close.
    #[test]
    fn fit_recovers_synthetic_capsule() {
        // Capsule along +X, radius 0.05, segment half-length 0.2.
        let axis = Vec3::X;
        let r = 0.05f32;
        let hseg = 0.2f32;
        let mut pts = Vec::new();
        // Cylinder wall samples.
        for i in 0..40 {
            let t = -hseg + 2.0 * hseg * (i as f32 / 39.0);
            for k in 0..16 {
                let a = std::f32::consts::TAU * (k as f32 / 16.0);
                pts.push(Vec3::new(t, r * a.cos(), r * a.sin()));
            }
        }
        let fit = fit_capsule(&pts).expect("fit");
        // Axis recovered (allow sign flip).
        let dir = (fit.b - fit.a).normalize();
        assert!(dir.dot(axis).abs() > 0.999, "axis off: {dir:?}");
        assert!(
            (fit.radius - r).abs() < 0.01,
            "radius {} vs {r}",
            fit.radius
        );
        // Endpoints pulled in by radius → segment ≈ 2*hseg - 2*r.
        let expected_seg = 2.0 * hseg - 2.0 * r;
        assert!(
            (fit.segment_len() - expected_seg).abs() < 0.03,
            "seg {} vs {expected_seg}",
            fit.segment_len()
        );
    }

    /// `score_box` and a too-small capsule cloud carry no axis/radius diagnostics
    /// (the illegal "box with skew" state is now unrepresentable); a fittable capsule
    /// cloud does. Guards the `Option` modelling from regressing to an in-band sentinel.
    #[test]
    fn box_and_tiny_cloud_have_no_capsule_diagnostics() {
        let pts = [Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(
            score_box(&pts, Vec3::ZERO, Vec3::splat(1.0))
                .capsule
                .is_none(),
            "a box has no capsule diagnostics"
        );
        assert!(
            score_capsule(&pts, Vec3::ZERO, Vec3::X, 0.1)
                .capsule
                .is_none(),
            "a <4-point cloud has no capsule diagnostics"
        );
        // A cloud along +X with four points and a non-degenerate segment IS fittable.
        let line: Vec<Vec3> = (0..8)
            .map(|i| Vec3::new(i as f32 * 0.1, 0.0, 0.0))
            .collect();
        assert!(
            score_capsule(&line, Vec3::ZERO, Vec3::X, 0.05)
                .capsule
                .is_some(),
            "a fittable capsule cloud has diagnostics"
        );
    }
}
