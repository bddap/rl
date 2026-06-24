//! Point-in-mesh containment: the single reporter behind `--verify-pivots` and the
//! skin-diag audit (both ask "does this pivot sit inside the surface?"). Pure
//! geometry on a triangle soup, no collider-fitting state. `--verify-colliders` is
//! deliberately NOT here — it scores collider-vs-cloud surface overlap, a different
//! question, and lives with the fit primitives in [`super::fit`].

use bevy::prelude::*;

/// One query point's verdict against a closed-ish triangle soup: the
/// orientation-normalised winding number, the surface distance signed so
/// `+` is OUTSIDE, and the inside/outside call. The signed-distance sign is
/// derived from `inside` (not an independent threshold), so the two can't
/// disagree at the boundary.
#[derive(Clone, Copy, Debug)]
pub struct Containment {
    /// Winding number normalised by the soup's orientation so an interior point
    /// reads ≈ +1 regardless of CW/CCW triangle winding.
    pub wn: f32,
    /// Nearest-surface distance, signed: `-` inside, `+` outside.
    pub signed_dist: f32,
    pub inside: bool,
}

/// A triangle soup prepared for point-in-mesh containment queries: the global
/// winding orientation is resolved once (from the signed volume) so every probe
/// shares it. This is the single mesh-containment path behind `--verify-pivots`,
/// the skin-diag settled audit, and its bind-pose reference — each is a borrow
/// of a different surface (bind mesh, live skinned soup) asking the same
/// inside/outside question, so the probe convention must not drift between them.
pub struct MeshContainment<'a> {
    positions: &'a [Vec3],
    triangles: &'a [[u32; 3]],
    signed_vol: f64,
    /// `+1` for CCW-outward triangles, `-1` for CW — the factor that makes an
    /// interior winding number read positive. Set from `signed_vol`'s sign, the
    /// orientation reference that survives the crab's cavity-dwelling centroid.
    orient: f32,
}

impl<'a> MeshContainment<'a> {
    pub fn new(positions: &'a [Vec3], triangles: &'a [[u32; 3]]) -> Self {
        let signed_vol = mesh_signed_volume(positions, triangles);
        let orient = if signed_vol < 0.0 { -1.0 } else { 1.0 };
        Self {
            positions,
            triangles,
            signed_vol,
            orient,
        }
    }

    /// Signed volume of the soup (its sign set `orient`); reported in the headers.
    pub fn signed_vol(&self) -> f64 {
        self.signed_vol
    }

    /// The winding orientation factor (`±1`); a `-1` means the triangles wind CW.
    pub fn orient(&self) -> f32 {
        self.orient
    }

    /// Test one point: orientation-normalised winding, signed nearest-surface
    /// distance (`+` OUTSIDE), and the inside call (`wn > 0.5`).
    pub fn probe(&self, p: Vec3) -> Containment {
        let wn = winding_number(p, self.positions, self.triangles) * self.orient;
        let d = nearest_surface_distance(p, self.positions, self.triangles);
        let inside = wn > 0.5;
        Containment {
            wn,
            signed_dist: if inside { -d } else { d },
            inside,
        }
    }

    /// Just the signed distance (`-` inside, `+` outside) for callers that don't
    /// need the winding number or the boolean.
    pub fn signed_dist(&self, p: Vec3) -> f32 {
        self.probe(p).signed_dist
    }
}

/// Axis-aligned bounds (min, max) of a point set; `(+∞, -∞)` for an empty set so
/// a later `size = hi - lo` reads as degenerate rather than silently zero.
pub fn aabb(pts: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &p in pts {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (lo, hi)
}

/// Generalized winding number of a triangle soup at `p`: `(1/4π)·Σ` of each
/// triangle's signed solid angle, via the Van Oosterom–Strackee `atan2` formula.
/// ≈1 (or ≈−1 for the opposite global winding) inside a closed surface, ≈0 outside,
/// and — unlike parity ray-casting — degrades gracefully on a non-watertight mesh
/// (fractional values reveal exactly how open it is). Sign depends on triangle
/// orientation; the caller normalises it via the soup's signed volume sign so an
/// interior point reads +1 regardless of CW/CCW winding.
fn winding_number(p: Vec3, positions: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    let mut acc = 0.0f64;
    for t in tris {
        let a = positions[t[0] as usize] - p;
        let b = positions[t[1] as usize] - p;
        let c = positions[t[2] as usize] - p;
        let (la, lb, lc) = (a.length() as f64, b.length() as f64, c.length() as f64);
        let num = a.dot(b.cross(c)) as f64;
        let den =
            la * lb * lc + (a.dot(b) as f64) * lc + (b.dot(c) as f64) * la + (c.dot(a) as f64) * lb;
        acc += 2.0 * num.atan2(den);
    }
    (acc / (4.0 * std::f64::consts::PI)) as f32
}

/// Signed volume of the triangle soup (∑ of per-triangle tetrahedron volumes
/// `v0·(v1×v2)/6`). Its SIGN fixes the global winding convention without needing an
/// interior reference point: CCW-outward triangles give a positive volume and make
/// interior winding numbers read +1; CW give negative and −1. Robust where a single
/// "is this point inside?" probe isn't — the crab's vertex centroid lands in a
/// cavity, so it can't be trusted to orient the sign.
fn mesh_signed_volume(positions: &[Vec3], tris: &[[u32; 3]]) -> f64 {
    let mut acc = 0.0f64;
    for t in tris {
        let v0 = positions[t[0] as usize];
        let v1 = positions[t[1] as usize];
        let v2 = positions[t[2] as usize];
        acc += v0.dot(v1.cross(v2)) as f64 / 6.0;
    }
    acc
}

/// Unsigned distance from `p` to a triangle (`v0`,`v1`,`v2`) — the standard
/// closest-point-on-triangle clamp. Used to report HOW FAR a query point sits from
/// the mesh surface (the winding number gives inside/outside; this gives the depth).
fn point_tri_distance(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    let ab = v1 - v0;
    let ac = v2 - v0;
    let ap = p - v0;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - v1;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let cp = p - v2;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (v0 + ab * v - p).length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (v0 + ac * w - p).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (v1 + (v2 - v1) * w - p).length();
    }
    // Interior of the face: project onto its plane via barycentric weights.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (v0 + ab * v + ac * w - p).length()
}

/// Nearest unsigned surface distance from `p` over the whole triangle soup.
fn nearest_surface_distance(p: Vec3, positions: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    let mut best = f32::INFINITY;
    for t in tris {
        let d = point_tri_distance(
            p,
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        if d < best {
            best = d;
        }
    }
    best
}
