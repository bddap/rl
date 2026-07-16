//! The rig-geometry audits behind `meshfit verify-colliders` / `verify-pivots` —
//! table-formatted reports over the canonical model, shaped like crab-world's
//! `collider_check::run` (an [`AuditVerdict`] when the audit ran, `Err` when it
//! couldn't) so the CLI stays thin dispatch. The pass/fail bars are pure predicates
//! over the fit scores, unit-tested here without a model file.
//!
//! The audits judge the CURRENT fit of the CURRENT asset (`bake::fitted_recipe`), so
//! a fitter change can be audited before it's baked; `bake::tests::baked_matches_refit`
//! is what binds that fit to the committed table the runtime actually runs.

use bevy::prelude::*;

use crab_world::bot::AuditVerdict;
use crab_world::bot::rig::{self, PartId, RestShape, RigRecipe};
use crab_world::mesh_fallback::model_path;

use crate::bake::fitted_recipe;
use crate::containment::{MeshContainment, aabb};
use crate::fit::{ColliderScore, score_box, score_capsule};
use crate::gltf_load::{LoadedModel, load_bind_mesh};

/// The capsule<->cloud agreement bar: too much of the cloud outside, a p95 poke past
/// 15% of the radius (5 mm floor), a bulge past half the radius, or a fitted
/// axis/radius far off the collider's.
fn capsule_fit_fails(s: &ColliderScore, radius: f32) -> bool {
    s.frac_outside > 0.05
        || s.poke_out_p95 > (0.15 * radius).max(0.005)
        || s.bulge_p95 > 0.5 * radius
        || s.capsule
            .is_some_and(|c| c.axis_skew_deg > 15.0 || !(0.85..=1.4).contains(&c.radius_ratio))
}

/// The carapace-slab bar: the trunk cloud mostly inside, at most a 2 cm p95 poke.
fn cuboid_fit_fails(s: &ColliderScore) -> bool {
    s.frac_outside > 0.03 || s.poke_out_p95 > 0.02
}

/// Winding numbers a closed mesh never produces: off-integer by more than 0.1.
fn fractional_windings(windings: &[f32]) -> usize {
    windings
        .iter()
        .filter(|&&w| (w - w.round()).abs() > 0.1)
        .count()
}

/// A watertight mesh reads ~+1 at a known-interior point, ~0 far outside, and
/// integer windings everywhere.
fn mesh_is_watertight(hub_wn: f32, far_wn: f32, fractional: usize) -> bool {
    hub_wn > 0.9 && far_wn.abs() < 0.1 && fractional == 0
}

/// The shared audit preamble: resolve, load, and rig the canonical model. The error
/// string is the reason the audit cannot run at all (distinct from a FAIL verdict).
fn load_rigged_model(audit: &str) -> Result<(std::path::PathBuf, LoadedModel, RigRecipe), String> {
    let Some(model_path) = model_path() else {
        return Err(format!(
            "{audit}: no model — set CRAB_MODEL_PATH or place sally.glb at the dev path"
        ));
    };
    let model =
        LoadedModel::load(&model_path).map_err(|e| format!("{audit}: load {model_path:?}: {e}"))?;
    let recipe =
        fitted_recipe(&model).ok_or_else(|| format!("{audit}: model built no rig recipe"))?;
    Ok((model_path, model, recipe))
}

/// Collider<->mesh agreement over every rest collider: does each capsule/cuboid sit
/// on the flesh its part cloud describes?
pub fn verify_colliders() -> Result<AuditVerdict, String> {
    let (_, model, recipe) = load_rigged_model("verify-colliders")?;
    let clouds = model.vertices_by_part();
    let trunk = model.vertices_for_bones(&rig::TRUNK_BONES);

    println!("collider<->mesh agreement (model units; +out = mesh pokes OUT of collider):");
    println!(
        "  {:<22} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>6}  {:>7}",
        "part", "n", "r", "fOut%", "pk95", "pkMax", "bulge", "skew", "rRat", "verdict"
    );

    let fmt = |x: Option<f32>| x.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"));
    let mut ranking: Vec<(String, f32, bool)> = Vec::new();
    let mut any_fail = false;

    for rc in rig::rest_colliders(&recipe) {
        let label = format!("{:?}", rc.part);
        let (score, rnorm, fail) = match rc.shape {
            RestShape::Capsule { a, b, radius } => {
                let pts = clouds.get(&rc.part).map(|p| p.as_slice()).unwrap_or(&[]);
                let s = score_capsule(pts, a, b, radius);
                let fail = capsule_fit_fails(&s, radius);
                (s, radius.max(1e-3), fail)
            }
            RestShape::Cuboid { center, half } => {
                // The carapace slab is fit against the trunk-bone cloud.
                let s = score_box(&trunk, center, half);
                let fail = cuboid_fit_fails(&s);
                (s, half.min_element().max(1e-3), fail)
            }
        };
        any_fail |= fail;
        ranking.push((label.clone(), score.poke_out_p95 / rnorm, fail));
        println!(
            "  {:<22} {:>5} {:>6.3} {:>6.1} {:>6.3} {:>6.3} {:>6.3} {:>5} {:>6}  {}",
            label,
            score.n,
            rnorm,
            score.frac_outside * 100.0,
            score.poke_out_p95,
            score.poke_out_max,
            score.bulge_p95,
            fmt(score.capsule.map(|c| c.axis_skew_deg)),
            fmt(score.capsule.map(|c| c.radius_ratio)),
            if fail { "FAIL" } else { "pass" },
        );
    }

    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst: Vec<String> = ranking
        .iter()
        .take(6)
        .map(|(l, s, f)| format!("{l} {:.2}{}", s, if *f { "!" } else { "" }))
        .collect();
    println!("worst (pk95/r): {}", worst.join(", "));
    println!(
        "{}",
        if any_fail {
            "VERDICT: FAIL — some colliders sit off the mesh"
        } else {
            "VERDICT: pass"
        }
    );
    Ok(AuditVerdict::failed(any_fail))
}

/// Joint-pivot containment: does every pivot (and capsule endpoint / box face
/// center) sit INSIDE the bind mesh?
pub fn verify_pivots() -> Result<AuditVerdict, String> {
    let (model_path, _model, recipe) = load_rigged_model("verify-pivots")?;
    let mesh = load_bind_mesh(&model_path)
        .map_err(|e| format!("verify-pivots: load mesh {model_path:?}: {e}"))?;
    let pos = &mesh.positions;
    let tris = &mesh.triangles;

    let (lo, hi) = aabb(pos);

    let soup = MeshContainment::new(pos, tris);
    let signed_vol = soup.signed_vol();
    let orient = soup.orient();
    let probe = |p: Vec3| {
        let c = soup.probe(p);
        (c.wn, c.signed_dist, c.inside)
    };

    let hub = rig::rest_colliders(&recipe)
        .iter()
        .find(|rc| rc.part == PartId::Carapace)
        .map(|rc| rc.pivot)
        .unwrap_or((lo + hi) * 0.5);
    let centroid = pos.iter().copied().sum::<Vec3>() / pos.len().max(1) as f32;
    let far = hi + (hi - lo).max(Vec3::splat(1.0)) + Vec3::splat(10.0);
    let (hub_wn, _, _) = probe(hub);
    let (cen_wn, _, _) = probe(centroid);
    let (far_wn, _, _) = probe(far);

    println!(
        "mesh: {} verts, {} triangles, bbox {:.3}..{:.3}, signed_vol={:.4}",
        pos.len(),
        tris.len(),
        lo,
        hi,
        signed_vol
    );
    println!(
        "self-check: hub(interior) wn={:+.3} (expect ~+1), vertex-centroid wn={:+.3} (in a cavity → ~0 ok), far-point wn={:+.3} (expect ~0){}",
        hub_wn,
        cen_wn,
        far_wn,
        if orient < 0.0 {
            "  [triangle winding is CW/flipped — normalised via signed volume]"
        } else {
            ""
        }
    );

    println!();
    println!("per-link containment (signed dist: + = OUTSIDE mesh, - = inside):");
    println!(
        "  {:<24} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4}",
        "link", "piv.wn", "piv.dist", "in?", "a.wn", "a.dist", "in?", "b.wn", "b.dist", "in?"
    );

    let mut pivots_out = 0usize;
    let mut endpoints_out = 0usize;
    let mut offenders: Vec<(String, f32)> = Vec::new();
    let mut windings: Vec<f32> = Vec::new();
    let yn = |b: bool| if b { "IN" } else { "OUT" };

    for rc in rig::rest_colliders(&recipe) {
        let label = format!("{:?}", rc.part);
        let (pwn, pdist, pin) = probe(rc.pivot);
        windings.push(pwn);
        if !pin {
            pivots_out += 1;
            offenders.push((format!("{label} pivot"), pdist));
        }
        match rc.shape {
            RestShape::Capsule { a, b, .. } => {
                let (awn, adist, ain) = probe(a);
                let (bwn, bdist, bin) = probe(b);
                windings.push(awn);
                windings.push(bwn);
                for (tag, inside, dist) in [("a", ain, adist), ("b", bin, bdist)] {
                    if !inside {
                        endpoints_out += 1;
                        offenders.push((format!("{label} {tag}"), dist));
                    }
                }
                println!(
                    "  {:<24} | {:>+7.3} {:>+8.4} {:>4} | {:>+7.3} {:>+8.4} {:>4} | {:>+7.3} {:>+8.4} {:>4}",
                    label,
                    pwn,
                    pdist,
                    yn(pin),
                    awn,
                    adist,
                    yn(ain),
                    bwn,
                    bdist,
                    yn(bin)
                );
            }
            RestShape::Cuboid { center, half } => {
                println!(
                    "  {:<24} | {:>+7.3} {:>+8.4} {:>4} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4}",
                    label,
                    pwn,
                    pdist,
                    yn(pin),
                    "(box)",
                    "corners",
                    "↓",
                    "",
                    "",
                    ""
                );
                for sx in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        for sz in [-1.0f32, 1.0] {
                            let corner = center + half * Vec3::new(sx, sy, sz);
                            let (cwn, cdist, cin) = probe(corner);
                            windings.push(cwn);
                            if !cin {
                                endpoints_out += 1;
                                offenders.push((
                                    format!("{label} corner({sx:+.0},{sy:+.0},{sz:+.0})"),
                                    cdist,
                                ));
                            }
                            println!(
                                "      corner ({:+.0},{:+.0},{:+.0})         | {:>+7.3} {:>+8.4} {:>4}",
                                sx,
                                sy,
                                sz,
                                cwn,
                                cdist,
                                yn(cin)
                            );
                        }
                    }
                }
                let (ccwn, ccdist, ccin) = probe(center);
                println!(
                    "      center                       | {:>+7.3} {:>+8.4} {:>4}",
                    ccwn,
                    ccdist,
                    yn(ccin)
                );
            }
        }
    }

    let fractional = fractional_windings(&windings);
    let clean = mesh_is_watertight(hub_wn, far_wn, fractional);

    offenders.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    println!();
    println!(
        "watertight: {} — {}/{} query windings are fractional (off integer by >0.1); interior wn={:+.3}, exterior wn={:+.3}",
        if clean { "CLEAN/closed" } else { "MESSY/open" },
        fractional,
        windings.len(),
        hub_wn,
        far_wn
    );
    println!(
        "SUMMARY: {pivots_out} pivot(s) OUTSIDE mesh, {endpoints_out} endpoint/corner(s) OUTSIDE mesh"
    );
    println!("worst offenders (model units outside the surface):");
    for (label, d) in offenders.iter().take(12) {
        println!("  {:<34} {:+.4}", label, d);
    }
    if offenders.is_empty() {
        println!("  (none — every query point is inside the mesh)");
    }

    let pass = pivots_out == 0;
    println!(
        "VERDICT: {}",
        if pass {
            "all joint pivots lie INSIDE the mesh"
        } else {
            "some joint pivots lie OUTSIDE the mesh — see ranking"
        }
    );
    Ok(AuditVerdict::failed(!pass))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::CapsuleDiagnostics;

    fn clean_score() -> ColliderScore {
        ColliderScore {
            n: 100,
            frac_outside: 0.0,
            poke_out_p95: 0.0,
            poke_out_max: 0.0,
            bulge_p95: 0.0,
            capsule: None,
        }
    }

    /// Each capsule bar trips alone.
    #[test]
    fn capsule_bars_trip_independently() {
        let r = 0.1;
        assert!(!capsule_fit_fails(&clean_score(), r));

        let mut s = clean_score();
        s.frac_outside = 0.06;
        assert!(capsule_fit_fails(&s, r));

        let mut s = clean_score();
        s.poke_out_p95 = 0.15 * r + 1e-4;
        assert!(capsule_fit_fails(&s, r));
        // A sub-5 mm poke never fails, even where 15% of the radius is tighter:
        // the absolute floor keeps hairline pokes on skinny links out of the verdict.
        s.poke_out_p95 = 0.004;
        assert!(!capsule_fit_fails(&s, 0.01));

        let mut s = clean_score();
        s.bulge_p95 = 0.5 * r + 1e-4;
        assert!(capsule_fit_fails(&s, r));

        let mut s = clean_score();
        s.capsule = Some(CapsuleDiagnostics {
            axis_skew_deg: 16.0,
            radius_ratio: 1.0,
        });
        assert!(capsule_fit_fails(&s, r), "axis skew past 15° fails");
        s.capsule = Some(CapsuleDiagnostics {
            axis_skew_deg: 0.0,
            radius_ratio: 1.5,
        });
        assert!(capsule_fit_fails(&s, r), "fitted radius far off fails");
        s.capsule = Some(CapsuleDiagnostics {
            axis_skew_deg: 14.0,
            radius_ratio: 1.0,
        });
        assert!(!capsule_fit_fails(&s, r), "in-band diagnostics pass");
    }

    /// The carapace-slab bars are absolute: 3% outside, 2 cm p95 poke.
    #[test]
    fn cuboid_bars_are_absolute() {
        assert!(!cuboid_fit_fails(&clean_score()));

        let mut s = clean_score();
        s.frac_outside = 0.04;
        assert!(cuboid_fit_fails(&s));

        let mut s = clean_score();
        s.poke_out_p95 = 0.021;
        assert!(cuboid_fit_fails(&s));
        s.poke_out_p95 = 0.019;
        assert!(!cuboid_fit_fails(&s));
    }

    /// The watertight verdict: interior ~+1, exterior ~0, integer windings.
    #[test]
    fn watertight_verdict() {
        let clean = [1.0, 0.98, 0.0, -0.02];
        assert_eq!(fractional_windings(&clean), 0);
        assert!(mesh_is_watertight(0.98, -0.02, 0));

        let open = [0.5, 1.0];
        assert_eq!(fractional_windings(&open), 1);
        assert!(!mesh_is_watertight(0.98, -0.02, 1));
        assert!(!mesh_is_watertight(0.5, 0.0, 0), "hub barely inside");
        assert!(!mesh_is_watertight(1.0, 0.3, 0), "exterior sees the mesh");
    }
}
