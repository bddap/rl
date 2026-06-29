//! `--check-rest-colliders`: spawn the crab, settle it to rest, then flag any pair
//! of body colliders that interpenetrate at the settled pose AND that the solver is
//! actively fighting — an overlap between two links that shouldn't be colliding.
//!
//! What "shouldn't" means, and why we ask Rapier rather than re-deriving it: the body
//! deliberately lets some collider pairs overlap. A joint's two links overlap at
//! their shared anchor by construction, so the body disables their contacts
//! (`body::collision::no_adjacent_contacts`); and the carapace-nested proximal links
//! collide only with the arena ([`super::body::NESTED_COLLISION`]), so they may sit
//! buried in the shell. Rapier records exactly these exceptions — a suppressed or
//! group-filtered pair generates no solver contact — so its narrow phase already
//! knows which overlaps are intended. We therefore take "the solver has an ACTIVE
//! contact for this pair" ([`ContactPair::has_any_active_contact`]) as ground truth
//! for "a collision the rig did not mean to allow", instead of re-implementing the
//! group/joint filter ourselves and risking drift from it.
//!
//! Penetration DEPTH comes from parry's `contact` on the raw shapes, not from the
//! manifold: the manifold's points can under-report a deep overlap, whereas the
//! shape query gives the true signed distance. So the two signals split cleanly —
//! Rapier classifies (is this fought?), parry measures (how deep?).
//!
//! The joint-tree adjacency (how many joints apart the two links are) is reported as
//! a label only; the verdict turns on the active-contact test, which already
//! excludes the directly-jointed pairs Rapier suppresses.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::geometry::ColliderHandle;
use bevy_rapier3d::rapier::parry::query::contact;

use super::body::{CrabBodyPart, CrabCarapace, CrabJoint};
use super::headless::{headless_app, tick};

/// Ticks of physics to drop the unactuated crab from its spawn pose onto the
/// ground and let it settle — the same count the reset/skin sim tests settle for
/// (`super::reset_test`, `super::skin`), so this check inspects the identical rest
/// pose those tests assert is sane.
const SETTLE_TICKS: u32 = 192;

/// Penetration (metres) below which an overlap is treated as numerical noise rather
/// than a real intersection. Two colliders meeting exactly at a shared face read a
/// hair negative from solver settling and float error; this floor keeps that from
/// masquerading as an illegal overlap. Well under the 5 mm the existing
/// `contact_audit` already considers worth printing.
const TOUCH_EPS: f32 = 1e-4;

/// One settled collider: the entity (keys the joint-tree adjacency label), a human
/// name, and the live world-space collider handle (keys both the shape/pose lookup
/// and the narrow-phase active-contact lookup).
struct Part {
    entity: Entity,
    name: String,
    handle: ColliderHandle,
}

/// One intersecting collider pair at the settled pose. `active` is Rapier's verdict
/// (the solver has a real contact for it); `joints_apart` is the joint-tree distance
/// for the human-readable label.
struct Finding {
    a: String,
    b: String,
    /// Penetration depth in metres (positive; how far the shapes overlap), from parry.
    depth: f32,
    /// Rapier has an active solver contact for this pair — it's a collision the rig
    /// did not suppress, so an overlap here is illegal.
    active: bool,
    /// Number of joints between the two links along the rig tree, or `None` if they
    /// are on different limbs (no common path). 1 = directly jointed (parent↔child).
    joints_apart: Option<u32>,
}

impl Finding {
    /// Illegal = the shapes genuinely overlap and the solver is fighting that contact
    /// (Rapier didn't suppress it as a jointed-anchor or group-filtered pair).
    fn is_illegal(&self) -> bool {
        self.active && self.depth > TOUCH_EPS
    }

    /// Short adjacency label for reporting. Directly-jointed pairs never reach the
    /// illegal list (Rapier suppresses them), so this mostly annotates the expected
    /// overlaps and the limb-internal illegal ones.
    fn adjacency(&self) -> String {
        match self.joints_apart {
            Some(1) => "jointed".to_string(),
            Some(n) => format!("{n} joints apart"),
            None => "separate limbs".to_string(),
        }
    }
}

/// `--check-rest-colliders` entrypoint: build a windowless physics world, spawn and
/// settle the crab, then for every pair of body colliders measure the penetration
/// (parry) and ask Rapier whether it's an active contact, classify, print the verdict
/// and the per-pair table, and return a process exit code (zero when clean, nonzero
/// on any illegal intersection or a failed spawn). Mirrors the `--verify-*` checks:
/// a diagnostic that doubles as a regression gate on rig/collider changes.
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

/// Joint-tree distance between every pair of crab links, derived from the live
/// `MultibodyJoint` parent edges (the same tree the spawn wired). A BFS from each
/// link over the undirected parent graph; the carapace root (no joint) is the hub the
/// limb chains meet at, so two links on different limbs connect THROUGH it. Distance
/// 1 = directly jointed. Used only to LABEL findings — the verdict is Rapier's
/// active-contact call — so an absent entry (disconnected) just reads "separate limbs".
fn joint_distances(app: &mut App) -> HashMap<(Entity, Entity), u32> {
    // Undirected adjacency: every child↔parent edge from the multibody joints. The
    // carapace appears only as a parent (it carries no joint of its own), which is
    // correct — it's the root the chains hang off.
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

    // BFS from each node. The tree is ~35 nodes, so all-pairs BFS is trivial and
    // keeps the lookup a simple symmetric map the pair loop can index either way.
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

/// Every crab collider entity with its identity + live collider handle, read from
/// Rapier's collider set so the handle indexes the same shapes/poses the solver just
/// integrated. Names: the joint id for an actuated link, "Carapace" for the root.
/// (The cosmetic eye-stalks aren't spawned as physics bodies, so they never appear.)
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
            // Every spawned crab body is either an actuated link (carries a `CrabJoint`)
            // or the carapace root; the `(None, false)` fallback is unreachable.
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

/// Test every unordered pair of crab colliders at the settled pose: parry's `contact`
/// for the penetration depth, and Rapier's narrow phase for whether the pair is an
/// active (un-suppressed) contact. Returns only pairs that actually overlap (depth >
/// [`TOUCH_EPS`]). `contact`'s `dist` is the signed gap — negative when penetrating —
/// so `-dist` is the depth; the shape query bypasses every collision-group / contact
/// filter, so the geometry is measured even for pairs Rapier suppresses.
fn intersecting_pairs(
    app: &mut App,
    parts: &[Part],
    joints_apart: &HashMap<(Entity, Entity), u32>,
) -> Vec<Finding> {
    // Both contexts: colliders for shapes/poses, simulation for the narrow phase.
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
            // prediction 0.0: report only real overlaps/touches, not near-misses.
            let Ok(Some(c)) = contact(ca.position(), ca.shape(), cb.position(), cb.shape(), 0.0)
            else {
                continue;
            };
            let depth = -c.dist;
            if depth <= TOUCH_EPS {
                continue;
            }
            // Rapier's narrow phase clears (no solver contacts) any pair it suppresses
            // — a directly-jointed pair with contacts disabled, or a group-filtered
            // one — so an active solver contact means the rig did NOT intend this
            // overlap. This is the authoritative "is the solver fighting it" signal,
            // verified to match the suppression rules: at rest a jointed anchor pair
            // reads inactive while a 2-joints-apart sibling overlap reads active.
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

/// Print the verdict first (CLEAN, or the illegal pairs), then the expected overlaps
/// for context, and return the exit code. Only illegal pairs (genuine overlap the
/// solver is fighting) drive a nonzero exit; the suppressed/filtered overlaps are
/// shown so a human can eyeball a genuinely absurd fit but never fail the gate.
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

    // The expected overlaps, deepest first — the shared-anchor and nested-link
    // overlaps Rapier suppresses. Listed for eyeballing, not part of the verdict.
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

    /// A real overlap the solver is fighting is illegal; the same overlap that Rapier
    /// suppressed (no active contact) is not — the verdict turns purely on Rapier's
    /// active-contact call, never on the adjacency label.
    #[test]
    fn illegal_iff_overlapping_and_active() {
        assert!(finding(true, 0.02, Some(2)).is_illegal());
        assert!(!finding(false, 0.02, Some(1)).is_illegal());
    }

    /// A grazing touch under the noise floor is not an intersection even when active —
    /// it's two faces meeting, not an overlap to fix.
    #[test]
    fn sub_epsilon_touch_is_not_illegal() {
        assert!(!finding(true, TOUCH_EPS * 0.5, None).is_illegal());
    }

    /// The adjacency label reads off the joint-tree distance: 1 = jointed, n = n
    /// joints apart, none = different limbs.
    #[test]
    fn adjacency_label_reflects_joint_distance() {
        assert_eq!(finding(true, 0.01, Some(1)).adjacency(), "jointed");
        assert_eq!(finding(true, 0.01, Some(3)).adjacency(), "3 joints apart");
        assert_eq!(finding(true, 0.01, None).adjacency(), "separate limbs");
    }
}
