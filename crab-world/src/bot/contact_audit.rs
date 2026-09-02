use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::body::{CrabBodyPart, CrabCarapace, CrabJoint};

/// Sub-5 mm contact depth is arrival, not a fight: the crumple settles
/// asymptotically, so some pair at the touching frontier always shows sub-mm depth
/// with residual drift.
const FIGHT_MIN_DEPTH: f32 = 0.005;

/// The demo's `--contact-audit` flag: every 64th tick, log crab-vs-terrain
/// penetrations past [`FIGHT_MIN_DEPTH`] and the lowest collider clearance.
pub fn live_contact_audit(
    sim: Query<&RapierContextSimulation>,
    cols: Query<&RapierContextColliders>,
    parts: Query<(Option<&CrabJoint>, Has<CrabCarapace>), With<CrabBodyPart>>,
    terrain: Res<crate::terrain::Terrain>,
    mut tick: Local<u32>,
) {
    *tick += 1;
    if *tick % 64 != 2 {
        return;
    }
    let (Ok(sim), Ok(cols)) = (sim.single(), cols.single()) else {
        return;
    };
    let name = |p: (Option<&CrabJoint>, bool)| {
        p.0.map(|j| format!("{:?}", j.id))
            .unwrap_or_else(|| "Carapace".to_string())
    };
    let mut terr: Vec<(f32, String)> = Vec::new();
    for pair in sim.narrow_phase.contact_pairs() {
        let (Some(e1), Some(e2)) = (
            cols.collider_entity(pair.collider1),
            cols.collider_entity(pair.collider2),
        ) else {
            continue;
        };
        let mut depth = 0.0f32;
        for m in &pair.manifolds {
            for pt in &m.points {
                depth = depth.max(-pt.dist);
            }
        }
        if depth <= FIGHT_MIN_DEPTH {
            continue;
        }
        let p = match (parts.get(e1), parts.get(e2)) {
            (Ok(p), Err(_)) | (Err(_), Ok(p)) => p,
            _ => continue,
        };
        let ground = [pair.collider1, pair.collider2].into_iter().any(|h| {
            cols.colliders
                .get(h)
                .is_some_and(|c| c.shape().as_heightfield().is_some())
        });
        if ground {
            terr.push((depth, name(p)));
        }
    }
    terr.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Contacts only exist near the surface — a part that tunneled fully below the
    // heightfield has NO manifold. Catch those geometrically: lowest collider point
    // vs the sampled ground height under the part's center (exact on flat cells,
    // ~slope*extent error on inclines — fine for a >5mm audit).
    let mut clear: Vec<(f32, String)> = Vec::new();
    for (handle, co) in cols.colliders.iter() {
        let Some(e) = cols.collider_entity(handle) else {
            continue;
        };
        let Ok(p) = parts.get(e) else {
            continue;
        };
        let t = co.translation();
        clear.push((co.compute_aabb().mins.y - terrain.height(t.x, t.z), name(p)));
    }
    clear.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (min_clear, min_part) = clear
        .first()
        .map(|(c, p)| (*c, p.as_str()))
        .unwrap_or((f32::INFINITY, "-"));
    println!(
        "AUDIT tick {}: {} crab-terrain contacts >{:.0}mm; min-clearance {:.0}mm {}",
        *tick,
        terr.len(),
        FIGHT_MIN_DEPTH * 1000.0,
        min_clear * 1000.0,
        min_part,
    );
    for (d, p) in terr.iter().take(6) {
        println!("  {:>4.0}mm {p} vs terrain", d * 1000.0);
    }
}
