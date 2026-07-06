
use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::geometry::ColliderHandle;
use bevy_rapier3d::rapier::parry::query::contact;

use super::body::{CrabBodyPart, CrabCarapace, CrabJoint};
use super::headless::{headless_app, tick};

const SETTLE_TICKS: u32 = 192;

const TOUCH_EPS: f32 = 1e-4;

struct Part {
    entity: Entity,
    name: String,
    handle: ColliderHandle,
}

struct Finding {
    a: String,
    b: String,
    depth: f32,
    active: bool,
    joints_apart: Option<u32>,
}

impl Finding {
    fn is_illegal(&self) -> bool {
        self.active && self.depth > TOUCH_EPS
    }

    fn adjacency(&self) -> String {
        match self.joints_apart {
            Some(1) => "jointed".to_string(),
            Some(n) => format!("{n} joints apart"),
            None => "separate limbs".to_string(),
        }
    }
}

pub fn run() -> i32 {
    let mut app = headless_app();
    tick(&mut app, SETTLE_TICKS);

    let parts = collect_parts(&mut app);
    if parts.len() < 10 {
        eprintln!(
            "check-rest-colliders: only {} crab colliders after settle — crab failed to spawn (no model?)",
            parts.len()
        );
        return 1;
    }
    let joints_apart = joint_distances(&mut app);

    let findings = intersecting_pairs(&mut app, &parts, &joints_apart);
    report(parts.len(), &findings)
}

fn joint_distances(app: &mut App) -> HashMap<(Entity, Entity), u32> {
    let mut adj: HashMap<Entity, Vec<Entity>> = HashMap::new();
    {
        let mut q = app
            .world_mut()
            .query_filtered::<(Entity, &MultibodyJoint), With<CrabBodyPart>>();
        for (child, joint) in q.iter(app.world()) {
            adj.entry(child).or_default().push(joint.parent);
            adj.entry(joint.parent).or_default().push(child);
        }
    }

    let mut dist: HashMap<(Entity, Entity), u32> = HashMap::new();
    let nodes: Vec<Entity> = adj.keys().copied().collect();
    for &src in &nodes {
        let mut seen: HashMap<Entity, u32> = HashMap::from([(src, 0)]);
        let mut queue = std::collections::VecDeque::from([src]);
        while let Some(n) = queue.pop_front() {
            let d = seen[&n];
            for &m in &adj[&n] {
                if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(m) {
                    e.insert(d + 1);
                    queue.push_back(m);
                }
            }
        }
        for (dst, d) in seen {
            if src != dst {
                dist.insert((src, dst), d);
            }
        }
    }
    dist
}

fn collect_parts(app: &mut App) -> Vec<Part> {
    let mut colliders_q = app.world_mut().query::<&RapierContextColliders>();
    let ctx = colliders_q.single(app.world()).expect("one rapier context");
    let handle_of: HashMap<Entity, ColliderHandle> = ctx
        .entity2collider()
        .iter()
        .map(|(&e, &h)| (e, h))
        .collect();

    let mut q = app
        .world_mut()
        .query_filtered::<(Entity, Option<&CrabJoint>, Has<CrabCarapace>), With<CrabBodyPart>>();
    q.iter(app.world())
        .filter_map(|(entity, joint, is_carapace)| {
            let handle = *handle_of.get(&entity)?;
            let name = match (joint, is_carapace) {
                (Some(j), _) => format!("{:?}", j.id),
                (None, true) => "Carapace".to_string(),
                (None, false) => format!("UnknownPart({})", entity.index()),
            };
            Some(Part {
                entity,
                name,
                handle,
            })
        })
        .collect()
}

fn intersecting_pairs(
    app: &mut App,
    parts: &[Part],
    joints_apart: &HashMap<(Entity, Entity), u32>,
) -> Vec<Finding> {
    let mut q = app
        .world_mut()
        .query::<(&RapierContextColliders, &RapierContextSimulation)>();
    let (cols, sim) = q.single(app.world()).expect("one rapier context");

    let mut findings = Vec::new();
    for (i, a) in parts.iter().enumerate() {
        let Some(ca) = cols.colliders.get(a.handle) else {
            continue;
        };
        for b in parts.iter().skip(i + 1) {
            let Some(cb) = cols.colliders.get(b.handle) else {
                continue;
            };
            let Ok(Some(c)) = contact(ca.position(), ca.shape(), cb.position(), cb.shape(), 0.0)
            else {
                continue;
            };
            let depth = -c.dist;
            if depth <= TOUCH_EPS {
                continue;
            }
            let active = sim
                .narrow_phase
                .contact_pair(a.handle, b.handle)
                .is_some_and(|p| p.has_any_active_contact());
            findings.push(Finding {
                a: a.name.clone(),
                b: b.name.clone(),
                depth,
                active,
                joints_apart: joints_apart.get(&(a.entity, b.entity)).copied(),
            });
        }
    }
    findings.sort_by(|x, y| {
        y.depth
            .partial_cmp(&x.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    findings
}

fn report(n_colliders: usize, findings: &[Finding]) -> i32 {
    let illegal: Vec<&Finding> = findings.iter().filter(|f| f.is_illegal()).collect();

    println!(
        "rest-pose collider intersection check: {} crab colliders, settled {} ticks\n",
        n_colliders, SETTLE_TICKS
    );

    if illegal.is_empty() {
        println!("VERDICT: CLEAN — no illegal rest-pose collider intersections");
    } else {
        println!(
            "VERDICT: {} ILLEGAL rest-pose collider intersection(s) (overlap the solver is fighting):",
            illegal.len()
        );
        for f in &illegal {
            println!(
                "  {:<26} <-> {:<26} {:>8.2} mm  ({})",
                f.a,
                f.b,
                f.depth * 1000.0,
                f.adjacency()
            );
        }
    }

    let expected: Vec<&Finding> = findings.iter().filter(|f| !f.is_illegal()).collect();
    if !expected.is_empty() {
        println!(
            "\nexpected overlaps ({} — jointed anchors + group-filtered nested links Rapier suppresses, not failures):",
            expected.len()
        );
        for f in &expected {
            println!(
                "  {:<26} <-> {:<26} {:>8.2} mm  ({})",
                f.a,
                f.b,
                f.depth * 1000.0,
                f.adjacency()
            );
        }
    }

    i32::from(!illegal.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(active: bool, depth: f32, joints_apart: Option<u32>) -> Finding {
        Finding {
            a: "A".into(),
            b: "B".into(),
            depth,
            active,
            joints_apart,
        }
    }

    #[test]
    fn illegal_iff_overlapping_and_active() {
        assert!(finding(true, 0.02, Some(2)).is_illegal());
        assert!(!finding(false, 0.02, Some(1)).is_illegal());
    }

    #[test]
    fn sub_epsilon_touch_is_not_illegal() {
        assert!(!finding(true, TOUCH_EPS * 0.5, None).is_illegal());
    }

    #[test]
    fn adjacency_label_reflects_joint_distance() {
        assert_eq!(finding(true, 0.01, Some(1)).adjacency(), "jointed");
        assert_eq!(finding(true, 0.01, Some(3)).adjacency(), "3 joints apart");
        assert_eq!(finding(true, 0.01, None).adjacency(), "separate limbs");
    }
}
