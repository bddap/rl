use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::geometry::ColliderHandle;
use bevy_rapier3d::rapier::parry::query::contact;

use super::body::{CrabBodyPart, CrabCarapace, CrabJoint, CrabJointId};
use super::headless::{headless_app, tick};
use super::meshfit::PartId;

const SETTLE_TICKS: u32 = 576;

const TOUCH_EPS: f32 = 1e-4;

/// A settled part is "quiet" below these mean speeds. The same numbers bound
/// `sim_truth_test`'s quiet-at-rest asserts (which aggregate worst-link
/// rather than per-part, so they are the stricter users of the ceilings).
///
/// Sally's rest pose carries standing load through deep contacts — carapace
/// box on the leg bases (~70mm), folded pincer on the claw shoulder — and
/// collision-group experiments that removed them made claw rest jitter 3-4x
/// worse (rl#109), so overlap alone cannot be the illegal test. Whether the
/// 70mm render/physics disagreement itself should shrink is an open
/// collider-fit question, tracked separately.
pub(crate) const QUIET_LIN_MPS: f32 = 0.2;
pub(crate) const QUIET_ANG_RADPS: f32 = 0.3;

/// Ticks over which per-part mean speeds are sampled after the settle.
const QUIET_WINDOW_TICKS: u32 = 192;

/// A non-quiet contact must also be at least this deep to count as fought:
/// the crumple settles asymptotically, so at any settle length some pair at
/// the touching frontier shows sub-mm depth with residual drift — arrival,
/// not fighting. Real fights observed to date: claw pincer-shoulder at ~7mm,
/// the pre-nesting coxa-coxa wedges at 46mm+.
const FIGHT_MIN_DEPTH: f32 = 0.005;

/// Rest contacts the crab is allowed to carry standing load through — the
/// structural patterns observed on the settled real model: shell skirt on
/// the thigh (basis) capsules, same-side adjacent thighs/knees stacking in
/// the crumple, and the folded claws resting on themselves and each other.
/// Quietness exempts a contact from the illegal verdict ONLY on these pairs;
/// a quiet deep wedge anywhere else (the historical coxa-coxa class) stays
/// illegal.
fn allowed_rest_contact(a: Option<PartId>, b: Option<PartId>) -> bool {
    use CrabJointId::*;
    let (Some(a), Some(b)) = (a, b) else {
        return false;
    };
    let adjacent_stack = |x: CrabJointId, y: CrabJointId| match (x, y) {
        (LegBasis(s, i), LegBasis(t, j)) | (LegMerus(s, i), LegMerus(t, j)) => {
            s == t && i.abs_diff(j) == 1
        }
        _ => false,
    };
    match (a, b) {
        (PartId::Carapace, PartId::Joint(j)) | (PartId::Joint(j), PartId::Carapace) => {
            matches!(j, LegBasis(..))
        }
        (PartId::Joint(x), PartId::Joint(y)) => match (x, y) {
            (ClawShoulder(s), ClawPincer(t)) | (ClawPincer(s), ClawShoulder(t)) => s == t,
            (ClawShoulder(_), ClawShoulder(_)) => true,
            _ => adjacent_stack(x, y),
        },
        _ => false,
    }
}

struct Part {
    entity: Entity,
    name: String,
    part: Option<PartId>,
    handle: ColliderHandle,
}

struct Finding {
    a: String,
    b: String,
    depth: f32,
    active: bool,
    quiet: bool,
    allowed: bool,
    joints_apart: Option<u32>,
}

impl Finding {
    fn is_illegal(&self) -> bool {
        self.active && self.depth > FIGHT_MIN_DEPTH && !(self.quiet && self.allowed)
    }

    fn is_load_bearing(&self) -> bool {
        self.active && self.depth > TOUCH_EPS && !self.is_illegal()
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
    let quiet = quiet_parts(&mut app);
    debug_assert!(
        parts.iter().all(|p| quiet.contains_key(&p.entity)),
        "a crab collider is missing from the quiet map — Velocity component lost?"
    );
    let joints_apart = joint_distances(&mut app);

    let findings = intersecting_pairs(&mut app, &parts, &joints_apart, &quiet);
    report(parts.len(), &findings)
}

/// Mean speeds per body part over a post-settle window, reduced to "is this
/// part quiet" — an active contact between two quiet parts is at equilibrium
/// (load-bearing), not being fought.
fn quiet_parts(app: &mut App) -> HashMap<Entity, bool> {
    let mut sums: HashMap<Entity, (f32, f32)> = HashMap::new();
    for _ in 0..QUIET_WINDOW_TICKS {
        tick(app, 1);
        let mut q = app
            .world_mut()
            .query_filtered::<(Entity, &Velocity), With<CrabBodyPart>>();
        for (e, v) in q.iter(app.world()) {
            let s = sums.entry(e).or_default();
            s.0 += v.linear.length();
            s.1 += v.angular.length();
        }
    }
    let inv = 1.0 / QUIET_WINDOW_TICKS as f32;
    sums.into_iter()
        .map(|(e, (lin, ang))| (e, lin * inv < QUIET_LIN_MPS && ang * inv < QUIET_ANG_RADPS))
        .collect()
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
            let (name, part) = match (joint, is_carapace) {
                (Some(j), _) => (format!("{:?}", j.id), Some(PartId::Joint(j.id))),
                (None, true) => ("Carapace".to_string(), Some(PartId::Carapace)),
                (None, false) => (format!("UnknownPart({})", entity.index()), None),
            };
            Some(Part {
                entity,
                name,
                part,
                handle,
            })
        })
        .collect()
}

fn intersecting_pairs(
    app: &mut App,
    parts: &[Part],
    joints_apart: &HashMap<(Entity, Entity), u32>,
    quiet: &HashMap<Entity, bool>,
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
                quiet: quiet.get(&a.entity).copied().unwrap_or(false)
                    && quiet.get(&b.entity).copied().unwrap_or(false),
                allowed: allowed_rest_contact(a.part, b.part),
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
        "rest-pose collider intersection check: {} crab colliders, {} settle + {} quiet-window ticks\n",
        n_colliders, SETTLE_TICKS, QUIET_WINDOW_TICKS
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

    let load_bearing: Vec<&Finding> = findings.iter().filter(|f| f.is_load_bearing()).collect();
    if !load_bearing.is_empty() {
        println!(
            "\nactive rest contacts, not fought ({} — quiet allowlisted load-bearers or sub-{:.0}mm settling grazes):",
            load_bearing.len(),
            FIGHT_MIN_DEPTH * 1000.0
        );
        for f in &load_bearing {
            println!(
                "  {:<26} <-> {:<26} {:>8.2} mm  ({})",
                f.a,
                f.b,
                f.depth * 1000.0,
                f.adjacency()
            );
        }
    }

    let expected: Vec<&Finding> = findings.iter().filter(|f| !f.active).collect();
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

    use crate::bot::body::Side;

    fn finding(active: bool, quiet: bool, allowed: bool, depth: f32) -> Finding {
        Finding {
            a: "A".into(),
            b: "B".into(),
            depth,
            active,
            quiet,
            allowed,
            joints_apart: Some(2),
        }
    }

    #[test]
    fn illegal_unless_quiet_on_an_allowed_pair() {
        assert!(finding(true, false, true, 0.046).is_illegal());
        assert!(
            finding(true, true, false, 0.046).is_illegal(),
            "a quiet deep wedge off the allowlist (the coxa-coxa class) must stay red"
        );
        assert!(!finding(true, true, true, 0.07).is_illegal());
        assert!(!finding(false, true, false, 0.046).is_illegal());
        assert!(
            !finding(true, false, false, 0.0003).is_illegal(),
            "a sub-floor settling graze with residual drift is not a fight"
        );
    }

    #[test]
    fn unfought_active_contacts_are_load_bearing() {
        assert!(finding(true, true, true, 0.07).is_load_bearing());
        assert!(finding(true, false, false, 0.0003).is_load_bearing());
        assert!(!finding(false, true, true, 0.07).is_load_bearing());
        assert!(!finding(true, false, true, 0.07).is_load_bearing());
    }

    #[test]
    fn sub_epsilon_touch_is_neither_illegal_nor_load_bearing() {
        assert!(!finding(true, false, false, TOUCH_EPS * 0.5).is_illegal());
        assert!(!finding(true, true, true, TOUCH_EPS * 0.5).is_load_bearing());
    }

    #[test]
    fn allowlist_matches_structural_patterns_only() {
        use CrabJointId::*;
        let j = |id| Some(PartId::Joint(id));
        let carapace = Some(PartId::Carapace);
        assert!(allowed_rest_contact(carapace, j(LegBasis(Side::Left, 1))));
        assert!(allowed_rest_contact(j(LegBasis(Side::Right, 0)), carapace));
        assert!(allowed_rest_contact(
            j(ClawShoulder(Side::Left)),
            j(ClawPincer(Side::Left))
        ));
        assert!(!allowed_rest_contact(
            j(ClawShoulder(Side::Left)),
            j(ClawPincer(Side::Right))
        ));
        assert!(allowed_rest_contact(
            j(ClawShoulder(Side::Left)),
            j(ClawShoulder(Side::Right))
        ));
        assert!(allowed_rest_contact(
            j(LegMerus(Side::Left, 1)),
            j(LegMerus(Side::Left, 2))
        ));
        assert!(!allowed_rest_contact(
            j(LegMerus(Side::Left, 1)),
            j(LegMerus(Side::Right, 1))
        ));
        assert!(!allowed_rest_contact(
            j(LegBasis(Side::Left, 0)),
            j(LegBasis(Side::Left, 2))
        ));
        assert!(
            !allowed_rest_contact(j(LegCoxa(Side::Left, 1)), j(LegCoxa(Side::Left, 2))),
            "the historical coxa-coxa wedge class is NOT an allowed rest contact"
        );
        assert!(!allowed_rest_contact(carapace, j(LegCoxa(Side::Left, 1))));
        assert!(!allowed_rest_contact(None, j(LegBasis(Side::Left, 1))));
    }

    #[test]
    fn adjacency_label_reflects_joint_distance() {
        let with_dist = |joints_apart| Finding {
            joints_apart,
            ..finding(true, true, true, 0.01)
        };
        assert_eq!(with_dist(Some(1)).adjacency(), "jointed");
        assert_eq!(with_dist(Some(3)).adjacency(), "3 joints apart");
        assert_eq!(with_dist(None).adjacency(), "separate limbs");
    }
}
