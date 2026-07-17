use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct FittedCapsule {
    pub a: Vec3,
    pub b: Vec3,
    pub radius: f32,
}

#[cfg(test)]
impl FittedCapsule {
    pub fn segment_len(&self) -> f32 {
        (self.b - self.a).length()
    }
}

const RADIUS_PERCENTILE: f32 = 0.92;

pub fn fit_capsule(points: &[Vec3]) -> Option<FittedCapsule> {
    if points.len() < 4 {
        return None;
    }
    let (centroid, axes) = covariance_eigenframe(points);
    let axis = axes[0];

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
    perp.sort_by(f32::total_cmp);
    let radius = percentile(&perp, RADIUS_PERCENTILE).max(1e-4);

    let half_axial = (tmax - tmin) * 0.5;
    let half_seg = (half_axial - radius).max(0.0);
    let mid = centroid + axis * (tmin + tmax) * 0.5;
    Some(FittedCapsule {
        a: mid - axis * half_seg,
        b: mid + axis * half_seg,
        radius,
    })
}

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

    #[test]
    fn fit_recovers_synthetic_capsule() {
        let axis = Vec3::X;
        let r = 0.05f32;
        let hseg = 0.2f32;
        let mut pts = Vec::new();
        for i in 0..40 {
            let t = -hseg + 2.0 * hseg * (i as f32 / 39.0);
            for k in 0..16 {
                let a = std::f32::consts::TAU * (k as f32 / 16.0);
                pts.push(Vec3::new(t, r * a.cos(), r * a.sin()));
            }
        }
        let fit = fit_capsule(&pts).expect("fit");
        let dir = (fit.b - fit.a).normalize();
        assert!(dir.dot(axis).abs() > 0.999, "axis off: {dir:?}");
        assert!(
            (fit.radius - r).abs() < 0.01,
            "radius {} vs {r}",
            fit.radius
        );
        let expected_seg = 2.0 * hseg - 2.0 * r;
        assert!(
            (fit.segment_len() - expected_seg).abs() < 0.03,
            "seg {} vs {expected_seg}",
            fit.segment_len()
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
