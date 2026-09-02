use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use bevy_rapier3d::rapier::geometry::ColliderHandle;

use super::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint};

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

/// Deepest geometric interpenetration between two links of env 0's crab this tick —
/// pairs that are not joint-adjacent and do not involve the carapace (nested coxae
/// sit inside it by design). Same-crab pairs raise no contact since rl#332, so this
/// is the only ruler left for leg–leg overlap.
pub struct OverlapScan {
    pub depth: f32,
    pub a: String,
    pub b: String,
    /// Pairs deeper than `min_depth`.
    pub pairs_over: usize,
}

pub fn same_crab_overlap(world: &mut World, min_depth: f32) -> OverlapScan {
    use bevy_rapier3d::rapier::parry::bounding_volume::BoundingVolume;
    use bevy_rapier3d::rapier::parry::query::contact;
    use std::collections::HashSet;

    let mut q = world.query_filtered::<(
        Entity,
        &CrabEnvId,
        &RapierColliderHandle,
        Option<&CrabJoint>,
        Option<&MultibodyJoint>,
        Has<CrabCarapace>,
    ), With<CrabBodyPart>>();
    let mut links: Vec<(Entity, ColliderHandle, String)> = Vec::new();
    let mut adjacent: HashSet<(Entity, Entity)> = HashSet::new();
    for (e, env, h, joint, mj, carapace) in q.iter(world) {
        if env.0 != 0 || carapace {
            continue;
        }
        let Some(joint) = joint else { continue };
        if let Some(mj) = mj {
            adjacent.insert((e, mj.parent));
            adjacent.insert((mj.parent, e));
        }
        links.push((e, h.0, format!("{:?}", joint.id)));
    }
    let cols = world
        .query::<&RapierContextColliders>()
        .single(world)
        .expect("one rapier context");
    let mut scan = OverlapScan {
        depth: 0.0,
        a: String::new(),
        b: String::new(),
        pairs_over: 0,
    };
    for (i, (ea, ha, na)) in links.iter().enumerate() {
        let Some(ca) = cols.colliders.get(*ha) else {
            continue;
        };
        for (eb, hb, nb) in links.iter().skip(i + 1) {
            if adjacent.contains(&(*ea, *eb)) {
                continue;
            }
            let Some(cb) = cols.colliders.get(*hb) else {
                continue;
            };
            if !ca.compute_aabb().intersects(&cb.compute_aabb()) {
                continue;
            }
            let Ok(Some(c)) = contact(ca.position(), ca.shape(), cb.position(), cb.shape(), 0.0)
            else {
                continue;
            };
            let depth = -c.dist;
            if depth > min_depth {
                scan.pairs_over += 1;
            }
            if depth > scan.depth {
                scan.depth = depth;
                scan.a = na.clone();
                scan.b = nb.clone();
            }
        }
    }
    scan
}
