use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::geometry::ColliderHandle;
use bevy_rapier3d::rapier::parry::query::contact;

use super::body::{CrabBodyPart, CrabCarapace, CrabJoint, CrabJointId};
use super::headless::{headless_app, tick};
use super::rig::PartId;
#[cfg(test)]
use super::rig::parts_adjacent;

const SETTLE_TICKS: u32 = 576;

const TOUCH_EPS: f32 = 1e-4;

/// A settled part is "quiet" below these mean speeds. The same numbers bound
/// `sim_truth_test`'s quiet-at-rest asserts (which aggregate worst-link
/// rather than per-part, so they are the stricter users of the ceilings).
///
/// Sally's rest pose carries standing load through deep contacts — carapace
/// box on the leg bases (~70mm), folded pincer on the claw shoulder — and
/// collision-group experiments that removed them made claw rest jitter 3-4x
/// worse (rl#109), so overlap alone cannot be the illegal test. The 70mm is
/// internal-only and benign (rl#234): her legs root under the shell skirt, so
/// the mesh volumes overlap the same way, and no external object can observe
/// it — render/physics honesty binds collider-vs-mesh at the surface the
/// world touches, not collider-vs-collider inside the body.
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
/// the thigh (basis) colliders, same-side adjacent thighs/knees stacking in
/// the crumple, the folded claws resting on themselves and each other, and
/// the front thigh nesting against its claw shoulder (a contact the mesh
/// really makes; it only registered once rl#20 Phase 1 fit the flesh fully).
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
            (ClawShoulder(s), LegBasis(t, 0)) | (LegBasis(s, 0), ClawShoulder(t)) => s == t,
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
    group_filtered: bool,
    quiet: bool,
    allowed: bool,
    joints_apart: Option<u32>,
}

/// One geometrically-overlapping crab collider pair at the current pose — THE
/// raw measurement, scanned once here and judged by both consumers (one
/// scanner, no parallel checker, rl#312): the rest-pose audit wraps it into
/// [`Finding`]s, the actuator-load test judges it against [`designed_contact`].
struct Overlap {
    a_ent: Entity,
    b_ent: Entity,
    a: String,
    b: String,
    a_part: Option<PartId>,
    b_part: Option<PartId>,
    depth: f32,
    /// The narrow phase holds an ACTIVE contact for the pair — the solver
    /// sees and resolves it. Pairs Rapier suppresses (contacts-disabled joint
    /// anchors, group-filtered nestings) overlap by design and read inactive.
    active: bool,
    /// The pair's live collision groups can never activate — BOTH directions
    /// must name the other's membership (rl#235), so failing either direction
    /// means the overlap is the filtered nesting the groups encode.
    group_filtered: bool,
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

pub fn run() -> Result<super::AuditVerdict, String> {
    let mut app = headless_app();
    tick(&mut app, SETTLE_TICKS);

    let parts = collect_parts(&mut app);
    if parts.len() < 10 {
        return Err(format!(
            "check-rest-colliders: only {} crab colliders after settle — crab failed to spawn (no model?)",
            parts.len()
        ));
    }
    let quiet = quiet_parts(&mut app);
    debug_assert!(
        parts.iter().all(|p| quiet.contains_key(&p.entity)),
        "a crab collider is missing from the quiet map — Velocity component lost?"
    );
    let joints_apart = joint_distances(&mut app);

    let findings = intersecting_pairs(&mut app, &parts, &joints_apart, &quiet);
    Ok(report(parts.len(), &findings))
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

fn overlapping_pairs(app: &mut App, parts: &[Part]) -> Vec<Overlap> {
    let mut q = app
        .world_mut()
        .query::<(&RapierContextColliders, &RapierContextSimulation)>();
    let (cols, sim) = q.single(app.world()).expect("one rapier context");

    let mut overlaps = Vec::new();
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
            let (ga, gb) = (ca.collision_groups(), cb.collision_groups());
            overlaps.push(Overlap {
                a_ent: a.entity,
                b_ent: b.entity,
                a: a.name.clone(),
                b: b.name.clone(),
                a_part: a.part,
                b_part: b.part,
                depth,
                active: sim
                    .narrow_phase
                    .contact_pair(a.handle, b.handle)
                    .is_some_and(|p| p.has_any_active_contact()),
                group_filtered: !(ga.memberships.intersects(gb.filter)
                    && gb.memberships.intersects(ga.filter)),
            });
        }
    }
    overlaps.sort_by(|x, y| {
        y.depth
            .partial_cmp(&x.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    overlaps
}

fn intersecting_pairs(
    app: &mut App,
    parts: &[Part],
    joints_apart: &HashMap<(Entity, Entity), u32>,
    quiet: &HashMap<Entity, bool>,
) -> Vec<Finding> {
    overlapping_pairs(app, parts)
        .into_iter()
        .map(|o| Finding {
            a: o.a,
            b: o.b,
            depth: o.depth,
            active: o.active,
            group_filtered: o.group_filtered,
            quiet: quiet.get(&o.a_ent).copied().unwrap_or(false)
                && quiet.get(&o.b_ent).copied().unwrap_or(false),
            allowed: allowed_rest_contact(o.a_part, o.b_part),
            joints_apart: joints_apart.get(&(o.a_ent, o.b_ent)).copied(),
        })
        .collect()
}

fn report(n_colliders: usize, findings: &[Finding]) -> super::AuditVerdict {
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
                "  {:<26} <-> {:<26} {:>8.2} mm  ({}, {})",
                f.a,
                f.b,
                f.depth * 1000.0,
                f.adjacency(),
                // Quiet-but-unallowed vs actively fought decides the remedy:
                // allowlist a structural rest contact, refit a fought one.
                if f.quiet { "quiet" } else { "FOUGHT" }
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
                "  {:<26} <-> {:<26} {:>8.2} mm  ({}{})",
                f.a,
                f.b,
                f.depth * 1000.0,
                f.adjacency(),
                if f.group_filtered {
                    ", group-filtered"
                } else {
                    ""
                }
            );
        }
    }

    super::AuditVerdict::failed(!illegal.is_empty())
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
            group_filtered: false,
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
            j(LegBasis(Side::Left, 0)),
            j(ClawShoulder(Side::Left))
        ));
        assert!(
            !allowed_rest_contact(j(LegBasis(Side::Left, 0)), j(ClawShoulder(Side::Right))),
            "front-thigh-on-shoulder nesting is same-side only"
        );
        assert!(
            !allowed_rest_contact(j(LegBasis(Side::Left, 1)), j(ClawShoulder(Side::Left))),
            "only the FRONT leg nests against the claw shoulder"
        );
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

/// The rl#312 designed-contact allowlist — DERIVED three ways, never a
/// hand-copied pair list: the live collision groups (a pair they filter can
/// never activate; its overlap is the nesting the groups encode, e.g. the
/// coxae tucked under the shell), the rig's joint adjacency
/// ([`parts_adjacent`], derived from the joint specs — the jointed pairs
/// whose multibody joints carry `contacts_enabled(false)`), and the
/// structural rest contacts [`allowed_rest_contact`] already encodes (the
/// designed shell-on-thigh / folded-claw load paths, rl#109/rl#234).
#[cfg(test)]
fn designed_contact(o: &Overlap) -> bool {
    let joint_adjacent = match (o.a_part, o.b_part) {
        (Some(a), Some(b)) => parts_adjacent(a, b),
        _ => false,
    };
    o.group_filtered || joint_adjacent || allowed_rest_contact(o.a_part, o.b_part)
}

#[cfg(test)]
mod load_tests {
    //! rl#312: Sally's body primitives never interpenetrate outside the
    //! designed-contact allowlist — (a) at rest and (b) while every actuator
    //! is driven through aggressive deterministic sequences (max torque both
    //! directions, alternating, seeded-random). The rl#303 buried-carapace
    //! trap was legs pinning the shell: self-collision integrity is
    //! load-bearing, so a violation names the offending pair and tick.

    use super::*;

    use std::collections::BTreeMap;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::bot::actuator::{ACTION_SIZE, CrabActions};

    /// Sim time each drive phase runs — at 64 Hz, 240 ticks = 3.75 s, long
    /// enough for a full-torque convulsion to fold limbs across the body.
    const PHASE_TICKS: u32 = 240;

    /// Ticks one alternating/random command holds before the flip/re-draw —
    /// 8 ticks = 125 ms: fast enough to slam, slow enough to swing fully.
    const COMMAND_HOLD_TICKS: u32 = 8;

    /// Deterministic seed for the random phase — the test must reproduce.
    const RANDOM_SEED: u64 = 0x5A11_0312;

    struct Violation {
        phase: &'static str,
        tick: u32,
        a: String,
        b: String,
        depth: f32,
        /// The solver held an active contact for the pair while it
        /// interpenetrated — fought, not a pass-through the narrow phase
        /// never saw.
        active: bool,
    }

    /// A disallowed pair interpenetrating deeper than [`FIGHT_MIN_DEPTH`] —
    /// the same floor the audit uses to separate a real fight from a
    /// sub-mm solver-softness graze, here for pairs with NO designed-contact
    /// exemption at all.
    fn scan(
        app: &mut App,
        parts: &[Part],
        phase: &'static str,
        tick: u32,
        violations: &mut Vec<Violation>,
    ) {
        for o in overlapping_pairs(app, parts) {
            if o.depth > FIGHT_MIN_DEPTH && !designed_contact(&o) {
                violations.push(Violation {
                    phase,
                    tick,
                    a: o.a,
                    b: o.b,
                    depth: o.depth,
                    active: o.active,
                });
            }
        }
    }

    fn run_phase(
        app: &mut App,
        parts: &mut Vec<Part>,
        phase: &'static str,
        first_tick: u32,
        command: &mut dyn FnMut(u32) -> [f32; ACTION_SIZE],
        violations: &mut Vec<Violation>,
    ) {
        for t in 0..PHASE_TICKS {
            let row = command(t);
            assert!(
                app.world_mut()
                    .resource_mut::<CrabActions>()
                    .set_row(0, row)
            );
            tick(app, 1);
            if parts
                .iter()
                .any(|p| app.world().get::<CrabBodyPart>(p.entity).is_none())
            {
                // A rescue (rl#283/#303) respawned her mid-phase: the old
                // entities are dead, re-collect so the scan keeps measuring
                // the live body rather than skipping stale handles.
                *parts = collect_parts(app);
            }
            scan(app, parts, phase, first_tick + t, violations);
        }
    }

    fn assert_no_violations(violations: &[Violation]) {
        // One line per offending pair: worst depth, first sighting (phase +
        // absolute tick), how many ticks it was seen and whether the solver
        // ever fought it.
        struct Worst {
            phase: &'static str,
            tick: u32,
            depth: f32,
            ticks: u32,
            fought: bool,
        }
        let mut worst: BTreeMap<(String, String), Worst> = BTreeMap::new();
        for v in violations {
            worst
                .entry((v.a.clone(), v.b.clone()))
                .and_modify(|w| {
                    w.depth = w.depth.max(v.depth);
                    w.ticks += 1;
                    w.fought |= v.active;
                })
                .or_insert(Worst {
                    phase: v.phase,
                    tick: v.tick,
                    depth: v.depth,
                    ticks: 1,
                    fought: v.active,
                });
        }
        assert!(
            violations.is_empty(),
            "rl#312: {} disallowed self-interpenetrating pair(s) under actuator load \
             (pair, worst depth, first sighting, ticks seen):\n{}",
            worst.len(),
            worst
                .iter()
                .map(|((a, b), w)| format!(
                    "  {a:<26} <-> {b:<26} {:>8.2} mm  first: {} tick {}  ({} ticks{})",
                    w.depth * 1000.0,
                    w.phase,
                    w.tick,
                    w.ticks,
                    if w.fought { ", FOUGHT" } else { "" },
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// rl#312 (a): at rest, no disallowed pair interpenetrates. This half is
    /// CLEAN on the current body and gates every build.
    #[test]
    fn body_primitives_do_not_interpenetrate_at_rest() {
        let mut app = headless_app();
        tick(&mut app, SETTLE_TICKS);
        let parts = collect_parts(&mut app);
        assert!(
            parts.len() >= 10,
            "only {} crab colliders after settle — crab failed to spawn (no model?)",
            parts.len()
        );
        let mut violations = Vec::new();
        scan(&mut app, &parts, "rest", 0, &mut violations);
        assert_no_violations(&violations);
    }

    /// rl#312 (b): the same invariant while every actuator is driven through
    /// aggressive deterministic sequences (max torque both directions,
    /// alternating, seeded-random). Ignored on rl#315: this currently FINDS
    /// 45 fought limb-crossing pairs (the claws scissor through each other at
    /// ~120 mm under sustained max drive) — run with `--ignored` to reproduce
    /// that table; un-ignore when the contact physics holds.
    #[test]
    #[ignore = "rl#315: limbs interpenetrate under sustained aggressive drive (45 fought pairs); reproduces the finding table"]
    fn body_primitives_never_interpenetrate_under_actuator_load() {
        let mut app = headless_app();
        tick(&mut app, SETTLE_TICKS);
        let mut parts = collect_parts(&mut app);
        assert!(
            parts.len() >= 10,
            "only {} crab colliders after settle — crab failed to spawn (no model?)",
            parts.len()
        );

        let mut violations = Vec::new();

        // Pre-drive baseline: the rest invariant must hold before any drive,
        // or a drive-phase violation is misattributed.
        scan(&mut app, &parts, "rest", 0, &mut violations);

        // Full-torque holds in both directions, then a per-channel square
        // wave, then seeded-random rows re-drawn every hold window.
        let mut elapsed = 0u32;
        run_phase(
            &mut app,
            &mut parts,
            "max+",
            elapsed,
            &mut |_| [1.0; ACTION_SIZE],
            &mut violations,
        );
        elapsed += PHASE_TICKS;
        run_phase(
            &mut app,
            &mut parts,
            "max-",
            elapsed,
            &mut |_| [-1.0; ACTION_SIZE],
            &mut violations,
        );
        elapsed += PHASE_TICKS;
        run_phase(
            &mut app,
            &mut parts,
            "alternating",
            elapsed,
            &mut |t| {
                std::array::from_fn(|i| {
                    if (t / COMMAND_HOLD_TICKS) as usize % 2 == i % 2 {
                        1.0
                    } else {
                        -1.0
                    }
                })
            },
            &mut violations,
        );
        elapsed += PHASE_TICKS;
        let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
        let mut row = [0.0; ACTION_SIZE];
        run_phase(
            &mut app,
            &mut parts,
            "random",
            elapsed,
            &mut |t| {
                if t % COMMAND_HOLD_TICKS == 0 {
                    row = std::array::from_fn(|_| rng.gen_range(-1.0..=1.0));
                }
                row
            },
            &mut violations,
        );

        assert_no_violations(&violations);
    }
}
