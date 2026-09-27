use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct FittedCapsule {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
}

#[derive(Clone, Debug)]
pub enum FittedShape {
    Capsule(FittedCapsule),
    Cuboid { center: Vec3, rot: Quat, half: Vec3 },
    Hull(Vec<Vec3>),
}

/// Radius percentiles tried per capsule candidate. The historical fixed 0.92 leaves
/// ~8% of a surface-vertex cloud outside by construction; higher-coverage radii
/// compete and the score decides.
const RADIUS_QS: [f32; 2] = [0.92, 0.96];

/// Extent percentiles tried per cuboid candidate (per-axis lo/hi, so an asymmetric
/// cloud shifts the box center rather than fattening it). 1.0 = the exact extremes:
/// on a surface-vertex cloud every point is real mesh, so full cover is a candidate.
const BOX_QS: [f32; 3] = [0.98, 0.995, 1.0];

/// How much better than the best capsule a cuboid must score to displace it. The
/// corner-emptiness term already prices the box's dishonesty (capsules carry no
/// analogous hidden volume), so no extra bias: an added margin here re-created the
/// degenerate hip blob by starving the box that fixes it.
const BOX_MARGIN: f32 = 1.0;

/// Fit the best-scoring allowed primitive to a part's skinned-vertex cloud.
///
/// Candidates: capsules about the dominant PCA axis and (when given) the bone-chain
/// direction — the chain axis is what rescues an isotropic cloud whose PCA axis is
/// noise (the degenerate middle-coxa hips) — each with uniform and taper-aware end
/// insets over [`RADIUS_QS`]; plus PCA- and identity-frame cuboids over [`BOX_QS`].
/// One dimensionless loss ([`fit_loss`]) picks the winner.
pub fn fit_link_shape(points: &[Vec3], chain_dir: Option<Vec3>) -> Option<FittedShape> {
    if points.len() < 4 {
        return None;
    }
    let (centroid, axes) = pca_frame(points);
    let scale = cloud_thickness(points, centroid, axes[0]);

    let mut axis_cands = vec![axes[0]];
    if let Some(d) = chain_dir {
        let d = d.normalize_or_zero();
        if d.length_squared() > 0.5 && d.dot(axes[0]).abs() < 0.995 {
            axis_cands.push(d);
        }
    }

    let mut best_cap: Option<(f32, FittedCapsule)> = None;
    for axis in axis_cands {
        for cap in capsule_candidates(points, centroid, axis, &RADIUS_QS) {
            let s = score_capsule(points, cap.a, cap.b, cap.radius);
            let loss = fit_loss(&s, scale);
            if best_cap.is_none_or(|(l, _)| loss < l) {
                best_cap = Some((loss, cap));
            }
        }
    }
    let (cap_loss, cap) = best_cap?;
    let mut best = (cap_loss, FittedShape::Capsule(cap));

    // Two frames: the PCA frame fits elongated slabs; the identity frame keeps an
    // isotropic blob's box untilted, so no corner pokes past the flesh envelope on a
    // noise axis.
    let id_axes = [Vec3::X, Vec3::Y, Vec3::Z];
    for frame in [axes, id_axes] {
        for (center, rot, half) in box_candidates(points, centroid, frame) {
            let s = score_box(points, center, rot, half);
            let loss = fit_loss(&s, scale)
                + CORNER_W * corner_emptiness(points, center, rot, half) / half.length();
            if loss < BOX_MARGIN * cap_loss && loss < best.0 {
                best = (loss, FittedShape::Cuboid { center, rot, half });
            }
        }
    }
    Some(best.1)
}

/// The cloud's convex hull as its vertices; `None` when it spans no 3D hull.
pub fn fit_hull(points: &[Vec3]) -> Option<FittedShape> {
    bevy_rapier3d::parry::shape::ConvexPolyhedron::from_convex_hull(points)
        .map(|h| h.points().to_vec())
        .filter(|v| hull_min_half_width(v) > 1e-4)
        .map(FittedShape::Hull)
}

/// Weight of the corner-emptiness term relative to the box's half-diagonal.
const CORNER_W: f32 = 1.0;

/// Mean distance from the box's corners to the nearest cloud point. A cuboid on a
/// ROUNDED part carries large empty corners — collider where no flesh renders,
/// reaching past the true envelope (a wrist box corner dug 5 cm under the ground
/// plane) — so empty corners price the box out and rounded parts stay capsules.
fn corner_emptiness(points: &[Vec3], center: Vec3, rot: Quat, half: Vec3) -> f32 {
    let mut total = 0.0;
    for i in 0..8 {
        let s = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        let corner = center + rot * (s * half);
        total += points
            .iter()
            .map(|&p| (p - corner).length_squared())
            .fold(f32::INFINITY, f32::min)
            .sqrt();
    }
    total / 8.0
}

/// One dimensionless fit loss for every candidate shape. Poke-out (mesh the physics
/// doesn't back) and bulge (collider past the rendered surface — invisible walls)
/// are both dishonest, poke twice as much; the outside fraction trades ~1% of stray
/// verts against pk95 of ~scale/4.
fn fit_loss(s: &ColliderScore, scale: f32) -> f32 {
    4.0 * s.frac_outside + (s.poke_out_p95 + 0.5 * s.bulge_p95) / scale
}

/// Typical cloud thickness — p95 of perpendicular distance about the dominant axis —
/// used to normalise absolute poke/bulge distances in [`fit_loss`].
fn cloud_thickness(points: &[Vec3], centroid: Vec3, axis: Vec3) -> f32 {
    let mut perp: Vec<f32> = points
        .iter()
        .map(|&p| {
            let d = p - centroid;
            (d - axis * d.dot(axis)).length()
        })
        .collect();
    perp.sort_by(f32::total_cmp);
    percentile(&perp, 0.95).max(1e-4)
}

fn capsule_candidates(
    points: &[Vec3],
    centroid: Vec3,
    axis: Vec3,
    radius_qs: &[f32],
) -> Vec<FittedCapsule> {
    let mut tmin = f32::INFINITY;
    let mut tmax = f32::NEG_INFINITY;
    let mut proj: Vec<(f32, f32)> = Vec::with_capacity(points.len());
    for &p in points {
        let d = p - centroid;
        let t = d.dot(axis);
        tmin = tmin.min(t);
        tmax = tmax.max(t);
        proj.push((t, (d - axis * t).length()));
    }
    let span = (tmax - tmin).max(1e-6);
    let tmid = (tmin + tmax) * 0.5;
    let mut all: Vec<f32> = proj.iter().map(|&(_, r)| r).collect();
    let mut lo: Vec<f32> = proj
        .iter()
        .filter(|&&(t, _)| t < tmid)
        .map(|&(_, r)| r)
        .collect();
    let mut hi: Vec<f32> = proj
        .iter()
        .filter(|&&(t, _)| t >= tmid)
        .map(|&(_, r)| r)
        .collect();
    all.sort_by(f32::total_cmp);
    lo.sort_by(f32::total_cmp);
    hi.sort_by(f32::total_cmp);

    let mut cands = Vec::with_capacity(radius_qs.len() * 2);
    for &q in radius_qs {
        let r = percentile(&all, q).max(1e-4);
        // Uniform end insets — right for an untapered link.
        let half_seg = ((span * 0.5) - r).max(0.0);
        let mid = centroid + axis * tmid;
        cands.push(FittedCapsule {
            a: mid - axis * half_seg,
            b: mid + axis * half_seg,
            radius: r,
        });
        // Taper-aware end insets: each end sphere sits one LOCAL end-radius in from
        // its extreme, so a 2:1 tapered link doesn't get its thin end
        // shortened by the fat end's radius.
        if !lo.is_empty() && !hi.is_empty() {
            let r_lo = percentile(&lo, q).max(1e-4);
            let r_hi = percentile(&hi, q).max(1e-4);
            let a = centroid + axis * (tmin + r_lo.min(span * 0.5));
            let b = centroid + axis * (tmax - r_hi.min(span * 0.5));
            if (b - a).length_squared() > 1e-10 {
                cands.push(FittedCapsule { a, b, radius: r });
            }
        }
    }
    cands
}

fn box_candidates(points: &[Vec3], centroid: Vec3, axes: [Vec3; 3]) -> Vec<(Vec3, Quat, Vec3)> {
    let rot = Quat::from_mat3(&Mat3::from_cols(axes[0], axes[1], axes[2]));
    let mut local: [Vec<f32>; 3] = [
        Vec::with_capacity(points.len()),
        Vec::with_capacity(points.len()),
        Vec::with_capacity(points.len()),
    ];
    for &p in points {
        let d = p - centroid;
        for (k, ax) in axes.iter().enumerate() {
            local[k].push(d.dot(*ax));
        }
    }
    for l in &mut local {
        l.sort_by(f32::total_cmp);
    }
    BOX_QS
        .iter()
        .map(|&q| {
            let mut half = Vec3::ZERO;
            let mut mid = Vec3::ZERO;
            for k in 0..3 {
                let lo = percentile(&local[k], 1.0 - q);
                let hi = percentile(&local[k], q);
                half[k] = ((hi - lo) * 0.5).max(1e-4);
                mid[k] = (hi + lo) * 0.5;
            }
            (centroid + rot * mid, rot, half)
        })
        .collect()
}

/// Centroid + right-handed orthonormal PCA frame (descending variance).
fn pca_frame(points: &[Vec3]) -> (Vec3, [Vec3; 3]) {
    let (centroid, raw) = covariance_eigenframe(points);
    let e0 = if raw[0].length_squared() > 0.5 {
        raw[0]
    } else {
        Vec3::Y
    };
    let e1 = (raw[1] - e0 * e0.dot(raw[1])).normalize_or_zero();
    let e1 = if e1.length_squared() > 0.5 {
        e1
    } else {
        e0.any_orthonormal_vector()
    };
    (centroid, [e0, e1, e0.cross(e1)])
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

pub struct ColliderScore {
    pub n: usize,
    pub frac_outside: f32,
    pub poke_out_p95: f32,
    pub poke_out_max: f32,
    pub bulge_p95: f32,
    pub capsule: Option<CapsuleDiagnostics>,
}

#[derive(Clone, Copy, Debug)]
pub struct CapsuleDiagnostics {
    pub axis_skew_deg: f32,
    pub radius_ratio: f32,
}

impl ColliderScore {
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

pub fn score_box(points: &[Vec3], center: Vec3, rot: Quat, half: Vec3) -> ColliderScore {
    let inv_rot = rot.inverse();
    let sd: Vec<f32> = points
        .iter()
        .map(|&p| {
            let local = (inv_rot * (p - center)).abs() - half;
            let outside = local.max(Vec3::ZERO).length();
            let inside = local.max_element().min(0.0);
            outside + inside
        })
        .collect();
    ColliderScore::from_signed(&sd)
}

fn hull_faces(hull: &[Vec3]) -> Vec<(Vec3, [Vec3; 3])> {
    crab_world::bot::rig::hull_triangles(hull)
        .into_iter()
        .map(|t| ((t[1] - t[0]).cross(t[2] - t[0]).normalize(), t))
        .collect()
}

/// Half the hull's thinnest width over its face normals — the polytope analogue of a
/// cuboid's smallest half-extent.
pub fn hull_min_half_width(hull: &[Vec3]) -> f32 {
    hull_faces(hull)
        .iter()
        .map(|(n, _)| {
            let (lo, hi) = hull
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                    (lo.min(n.dot(*p)), hi.max(n.dot(*p)))
                });
            (hi - lo) * 0.5
        })
        .fold(f32::INFINITY, f32::min)
}

/// Exact signed distances: inside, the nearest face plane; outside, the nearest face.
/// Not parry's polyhedron point projection — it is GJK-based and misreads points on
/// the boundary.
pub fn score_hull(points: &[Vec3], hull: &[Vec3]) -> ColliderScore {
    use bevy_rapier3d::parry::query::PointQuery;
    use bevy_rapier3d::parry::shape::Triangle;

    let faces = hull_faces(hull);
    let sd: Vec<f32> = points
        .iter()
        .map(|&p| {
            let plane = faces
                .iter()
                .map(|(n, t)| n.dot(p - t[0]))
                .fold(f32::NEG_INFINITY, f32::max);
            if plane <= 0.0 {
                return plane;
            }
            faces
                .iter()
                .map(|(_, t)| Triangle::new(t[0], t[1], t[2]).distance_to_local_point(p, true))
                .fold(f32::INFINITY, f32::min)
        })
        .collect();
    ColliderScore::from_signed(&sd)
}

#[allow(clippy::needless_range_loop)]
fn covariance_eigenframe(points: &[Vec3]) -> (Vec3, [Vec3; 3]) {
    let centroid = points.iter().copied().sum::<Vec3>() / points.len() as f32;
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
    let mut v = [[0.0f64; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..16 {
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

    /// A true capsule SURFACE (cylinder plus hemispherical end caps) — an
    /// open-ended tube would leave its end rings poking 0.41r out of any fitted
    /// capsule and unfairly favour a box.
    fn synthetic_capsule_cloud(axis: Vec3, r: f32, hseg: f32) -> Vec<Vec3> {
        let mut pts = Vec::new();
        for i in 0..40 {
            let t = -hseg + 2.0 * hseg * (i as f32 / 39.0);
            for k in 0..16 {
                let a = std::f32::consts::TAU * (k as f32 / 16.0);
                pts.push(axis * t + Vec3::new(0.0, r * a.cos(), r * a.sin()));
            }
        }
        for (end, sign) in [(-hseg, -1.0f32), (hseg, 1.0)] {
            for i in 1..6 {
                let phi = std::f32::consts::FRAC_PI_2 * (i as f32 / 6.0);
                let (rc, out) = (r * phi.cos(), r * phi.sin());
                for k in 0..8 {
                    let a = std::f32::consts::TAU * (k as f32 / 8.0);
                    pts.push(
                        axis * (end + sign * out) + Vec3::new(0.0, rc * a.cos(), rc * a.sin()),
                    );
                }
            }
            pts.push(axis * (end + sign * r));
        }
        pts
    }

    #[test]
    fn fit_recovers_synthetic_capsule() {
        let r = 0.05f32;
        let hseg = 0.2f32;
        let pts = synthetic_capsule_cloud(Vec3::X, r, hseg);
        let Some(FittedShape::Capsule(fit)) = fit_link_shape(&pts, None) else {
            panic!("a genuinely cylindrical cloud must fit a capsule, not a box");
        };
        let dir = (fit.b - fit.a).normalize();
        assert!(dir.dot(Vec3::X).abs() > 0.999, "axis off: {dir:?}");
        assert!(
            (fit.radius - r).abs() < 0.01,
            "radius {} vs {r}",
            fit.radius
        );
        let seg = (fit.b - fit.a).length();
        let expected_seg = 2.0 * hseg;
        assert!(
            (seg - expected_seg).abs() < 0.03,
            "seg {seg} vs {expected_seg}"
        );
    }

    #[test]
    fn boxy_cloud_fits_a_cuboid() {
        // A flat slab: capsules waste most of their cross-section on it.
        let mut pts = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                for (k, s) in [(0.0f32, -1.0f32), (0.0, 1.0)] {
                    let x = -0.2 + 0.4 * (i as f32 / 11.0);
                    let z = -0.1 + 0.2 * (j as f32 / 11.0);
                    pts.push(Vec3::new(x, k + s * 0.02, z));
                }
            }
        }
        match fit_link_shape(&pts, None) {
            Some(FittedShape::Cuboid { half, .. }) => {
                assert!(
                    half.max_element() < 0.25 && half.min_element() < 0.05,
                    "slab box extents off: {half:?}"
                );
            }
            other => panic!("a slab must fit a cuboid, got {other:?}"),
        }
        let Some(FittedShape::Hull(hull)) = fit_hull(&pts) else {
            panic!("a 3D cloud has a hull");
        };
        assert_eq!(hull.len(), 8, "a slab's hull is its 8 corners");
        assert_eq!(score_hull(&pts, &hull).frac_outside, 0.0);
        let flat: Vec<Vec3> = pts.iter().map(|p| p.with_y(0.0)).collect();
        assert!(fit_hull(&flat).is_none(), "a coplanar cloud spans no hull");
    }

    #[test]
    fn degenerate_isotropic_cloud_uses_the_chain_axis() {
        // An isotropic-ish blob: PCA axis is noise, the chain hint must orient it.
        let mut pts = Vec::new();
        for i in 0..200 {
            let a = i as f32 * 2.399963; // golden-angle sphere cover
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / 200.0;
            let r = (1.0 - z * z).sqrt();
            pts.push(Vec3::new(r * a.cos(), r * a.sin(), z) * 0.08);
        }
        let chain = Vec3::new(1.0, 0.0, 0.0);
        let Some(FittedShape::Capsule(fit)) = fit_link_shape(&pts, Some(chain)) else {
            panic!("expected a capsule");
        };
        // On a sphere every axis scores alike; the fit must stay sane (radius ≈ 0.08).
        assert!(
            (fit.radius - 0.08).abs() < 0.02,
            "blob radius {} vs 0.08",
            fit.radius
        );
    }

    #[test]
    fn box_and_tiny_cloud_have_no_capsule_diagnostics() {
        let pts = [Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(
            score_box(&pts, Vec3::ZERO, Quat::IDENTITY, Vec3::splat(1.0))
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

    /// A hull vertex must read as ON the surface — the case a GJK projection misreads.
    #[test]
    fn score_hull_is_exact() {
        let half = Vec3::new(1.0, 0.5, 0.25);
        let hull: Vec<Vec3> =
            crab_world::bot::rig::cuboid_corners(Vec3::ZERO, Quat::IDENTITY, half).to_vec();
        let at = |p: Vec3| score_hull(&[p], &hull);
        assert!((at(Vec3::new(3.0, 0.0, 0.0)).poke_out_max - 2.0).abs() < 1e-5);
        assert!((at(Vec3::new(2.0, 1.5, 0.0)).poke_out_max - 2f32.sqrt()).abs() < 1e-5);
        assert!((at(Vec3::new(2.0, 1.5, 1.25)).poke_out_max - 3f32.sqrt()).abs() < 1e-5);
        assert!((at(Vec3::new(0.5, 0.0, 0.0)).bulge_p95 - 0.25).abs() < 1e-5);
        let corner = at(half);
        assert_eq!((corner.frac_outside, corner.poke_out_max), (0.0, 0.0));
        assert!((hull_min_half_width(&hull) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn oriented_score_box_matches_rotated_points() {
        let rot = Quat::from_rotation_z(0.7);
        let pts: Vec<Vec3> = [
            Vec3::new(0.3, 0.05, -0.1),
            Vec3::new(-0.3, -0.05, 0.1),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.35, 0.0, 0.0),
        ]
        .iter()
        .map(|&p| rot * p + Vec3::splat(2.0))
        .collect();
        let s = score_box(&pts, Vec3::splat(2.0), rot, Vec3::new(0.3, 0.05, 0.1));
        assert_eq!(s.n, 4);
        assert!(
            (s.poke_out_max - 0.05).abs() < 1e-3,
            "the one poking point sticks out 0.05, got {}",
            s.poke_out_max
        );
    }
}
