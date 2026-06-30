//! The policy-actuated joint set: [`CrabJointId`] (the obs/action slot key) and
//! [`Side`], the per-joint tuning each `CrabJointId` method consumes (torque
//! ceilings, travel limits, friction breakaway), and [`joint_angle`], the signed
//! joint coordinate the observation and telemetry both read. Stands alone — it
//! depends on no other body submodule.

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Torque ceilings — the magnitude an action of ±1 commands on each joint type.
// DIRECT-DRIVE (the policy's output IS the torque) and GRADED BY INERTIA: a joint's
// snappiness is torque/inertia, so a flat ceiling would make a feather-light distal
// link a hair-trigger while staying modest on the hip. Each ceiling is still several×
// the ~0.5 N·m a joint needs to bear its share of the body, so standing keeps ample
// authority; only the surplus that fed mid-air spin-up is cut.
// ---------------------------------------------------------------------------

const COXA_TORQUE_CEILING: f32 = 6.0;
const FEMUR_TORQUE_CEILING: f32 = 4.0;
const TIBIA_TORQUE_CEILING: f32 = 2.5;
// Cheliped joints, weighted by relative muscle mass: the closer (pincer) is the
// dominant muscle of the claw, the wrist the smallest. Grapsus is a grazer with
// modest symmetric claws, so these stay within the legs' range — not crusher-scale.
const CLAW_PINCER_TORQUE_CEILING: f32 = 7.0;
const CLAW_SHOULDER_TORQUE_CEILING: f32 = 4.0;
const CLAW_WRIST_TORQUE_CEILING: f32 = 3.0;

// Cheliped shoulder travel is ASYMMETRIC about the bind rest, and a single source for
// both sides (`CrabJointId::limits` matches `ClawShoulder(_)`). +θ swings the arm DOWN
// toward the ground, keeping the full reach the last-metre grab needs; −θ LIFTS it. The
// up-stop is bound by the CHELIPED CAPSULE FLESH (its highest point = collider center +
// radius), NOT the bare bone center: at −0.6 the palm bone center sat just under the
// shell top (≈0.61) so the old center-only guard passed, yet the fitted palm/forearm
// capsules — which carry a real radius — still topped out ≈0.69, ~0.09 above the shell
// and through the eye band (≈0.48–0.52), the owner-reported "arm intersects eye and
// carapace" (bddap/rl#41 refinement). −0.35 rad lifts the claw to about eye level
// (graceful defensive reach kept) while the whole cheliped capsule envelope stays ~2 cm
// BELOW the carapace top — clear of the shell and eye-stalks, with margin for soft-limit
// overshoot. Read off the bind-pose geometry; pinned by
// `rig::tests::shoulder_upswing_stays_below_carapace` (now a capsule-flesh, not bone-center, guard).
const CLAW_SHOULDER_UP_STOP: f32 = -0.35;
const CLAW_SHOULDER_DOWN_STOP: f32 = 1.0;

/// Breakaway torque of the leg joint-friction motor: the constant an external load
/// must exceed to back-drive the joint. Kept WELL below the ~0.5 N·m needed to hold a
/// leg (~1/5) so legs crumple under the lightest ground load instead of propping the
/// body up, yet nonzero because real joints have a little stiction. The motor's ramp
/// slope into this cap is `spawn::FRICTION_RAMP` (the joint is built there).
pub const LEG_FRICTION_CAP: f32 = 0.04;
const CLAW_FRICTION_CAP: f32 = 0.04;

/// Signed joint coordinate read off the two links' world orientations: the twist
/// of child-in-parent about the joint's free `axis_local` (the rig-derived axis
/// the actuator drives). Shared by the observation and the telemetry overlay so
/// both report the same angle the policy acts on.
pub fn joint_angle(axis_local: Vec3, parent_rot: Quat, child_rot: Quat) -> f32 {
    let q = (parent_rot.inverse() * child_rot).normalize();
    let v = Vec3::new(q.x, q.y, q.z);
    2.0 * v.dot(axis_local).atan2(q.w)
}

/// Every POLICY-ACTUATED joint — the locomotion-relevant subset of the crab's
/// articulation. The body also spawns the eye-stalks as locked links with no
/// `CrabJoint`; restoring finer articulation (bddap/rl#31) means adding a variant
/// here (which grows the observation/action vector and the net) plus a rig
/// [`JointSpec`](crate::bot::rig::recipe::JointSpec).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CrabJointId {
    // Legs (8 = side L/R × leg 0..3 front→back). The basal joint is 2-DOF: coxa
    // swings the leg fore/aft about a vertical axis at the body root, basis lifts it
    // up/down about a horizontal axis just distal (the levator/depressor DOF, the
    // biggest locomotion win — bddap/rl#31); merus and carpus are the two
    // load-bearing bends down the limb.
    LegCoxa(Side, u8),
    LegBasis(Side, u8),
    LegMerus(Side, u8),
    LegCarpus(Side, u8),
    // Claws (1 per side): shoulder lifts the arm, wrist bends the hand, pincer
    // opens/closes the movable finger.
    ClawShoulder(Side),
    ClawWrist(Side),
    ClawPincer(Side),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Side {
    Left,
    Right,
}

impl CrabJointId {
    /// Every policy-actuated joint, in observation/action slot order — the ONE
    /// source of truth from which [`index`](Self::index) and [`COUNT`](Self::COUNT)
    /// derive, so adding a variant means editing only this list (and a wrong array
    /// length is a compile error, not the runtime slot-aliasing the bijection test
    /// once had to catch). The order is load-bearing: PPO obs/action vectors and
    /// saved checkpoints are keyed by a joint's position here, so reordering would
    /// invalidate trained checkpoints. Legs first (front→back per side, both
    /// sides), then claws (per side) — left side ahead of right within each.
    const fn all() -> [CrabJointId; 38] {
        [
            CrabJointId::LegCoxa(Side::Left, 0),
            CrabJointId::LegBasis(Side::Left, 0),
            CrabJointId::LegMerus(Side::Left, 0),
            CrabJointId::LegCarpus(Side::Left, 0),
            CrabJointId::LegCoxa(Side::Left, 1),
            CrabJointId::LegBasis(Side::Left, 1),
            CrabJointId::LegMerus(Side::Left, 1),
            CrabJointId::LegCarpus(Side::Left, 1),
            CrabJointId::LegCoxa(Side::Left, 2),
            CrabJointId::LegBasis(Side::Left, 2),
            CrabJointId::LegMerus(Side::Left, 2),
            CrabJointId::LegCarpus(Side::Left, 2),
            CrabJointId::LegCoxa(Side::Left, 3),
            CrabJointId::LegBasis(Side::Left, 3),
            CrabJointId::LegMerus(Side::Left, 3),
            CrabJointId::LegCarpus(Side::Left, 3),
            CrabJointId::LegCoxa(Side::Right, 0),
            CrabJointId::LegBasis(Side::Right, 0),
            CrabJointId::LegMerus(Side::Right, 0),
            CrabJointId::LegCarpus(Side::Right, 0),
            CrabJointId::LegCoxa(Side::Right, 1),
            CrabJointId::LegBasis(Side::Right, 1),
            CrabJointId::LegMerus(Side::Right, 1),
            CrabJointId::LegCarpus(Side::Right, 1),
            CrabJointId::LegCoxa(Side::Right, 2),
            CrabJointId::LegBasis(Side::Right, 2),
            CrabJointId::LegMerus(Side::Right, 2),
            CrabJointId::LegCarpus(Side::Right, 2),
            CrabJointId::LegCoxa(Side::Right, 3),
            CrabJointId::LegBasis(Side::Right, 3),
            CrabJointId::LegMerus(Side::Right, 3),
            CrabJointId::LegCarpus(Side::Right, 3),
            CrabJointId::ClawShoulder(Side::Left),
            CrabJointId::ClawWrist(Side::Left),
            CrabJointId::ClawPincer(Side::Left),
            CrabJointId::ClawShoulder(Side::Right),
            CrabJointId::ClawWrist(Side::Right),
            CrabJointId::ClawPincer(Side::Right),
        ]
    }

    /// Total policy-actuated DOFs — sets the observation/action vector width and
    /// the net's input/output size. Locked rig joints are excluded (no `CrabJoint`).
    pub const COUNT: usize = Self::all().len();

    /// Flat observation/action slot for this joint (0..COUNT): its position in
    /// [`all`](Self::all). A bijection because `all` lists each joint once; the
    /// `index_is_a_bijection` test pins that no variant is duplicated there. O(COUNT)
    /// and never on a hot path.
    pub fn index(&self) -> usize {
        Self::all()
            .iter()
            .position(|j| j == self)
            .expect("every CrabJointId is listed in all()")
    }
}

impl CrabJointId {
    /// Peak DIRECT-DRIVE torque an action of ±1 commands on this joint — the
    /// magnitude the actuator applies via `ExternalForce`, graded by inertia
    /// (weakest on the lightest, most distal links). The policy's drive authority,
    /// distinct from the joint's friction motor ([`Self::friction_cap`]).
    pub fn drive_torque_ceiling(&self) -> f32 {
        match self {
            // The basis lift is hip-class — it bears the leg's weight as it raises
            // it — so it shares the coxa swing's ceiling.
            CrabJointId::LegCoxa(..) | CrabJointId::LegBasis(..) => COXA_TORQUE_CEILING,
            CrabJointId::LegMerus(..) => FEMUR_TORQUE_CEILING,
            CrabJointId::LegCarpus(..) => TIBIA_TORQUE_CEILING,
            CrabJointId::ClawShoulder(_) => CLAW_SHOULDER_TORQUE_CEILING,
            CrabJointId::ClawWrist(_) => CLAW_WRIST_TORQUE_CEILING,
            CrabJointId::ClawPincer(_) => CLAW_PINCER_TORQUE_CEILING,
        }
    }

    /// Breakaway torque of this joint's friction motor (the motor's ramp lives in
    /// `spawn::FRICTION_RAMP`): the constant an external load must beat to back-drive
    /// the joint, so legs crumple under ground contact and a modest command still actuates.
    pub fn friction_cap(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..)
            | CrabJointId::LegBasis(..)
            | CrabJointId::LegMerus(..)
            | CrabJointId::LegCarpus(..) => LEG_FRICTION_CAP,
            CrabJointId::ClawShoulder(_)
            | CrabJointId::ClawWrist(_)
            | CrabJointId::ClawPincer(_) => CLAW_FRICTION_CAP,
        }
    }

    /// Joint travel limits `[lo, hi]` in radians about the rig BIND POSE: the spawn
    /// bakes each link's bind orientation onto Rapier coordinate 0, so the range
    /// straddles 0 (0 = the bone's rest angle in the model). The policy commands
    /// torque, not position, so these are the hard stops the limb cannot pass, not
    /// a target. Per-family defaults, refined once the body stands.
    pub fn limits(&self) -> [f32; 2] {
        match self {
            CrabJointId::LegCoxa(..) => [-0.8, 0.8],
            // Levator/depressor sweep about the bind rest: enough to lift a foot clear
            // for a step and to push the body up, kept symmetric and modest so a fresh
            // policy can't fling the leg over the shell.
            CrabJointId::LegBasis(..) => [-0.6, 0.6],
            CrabJointId::LegMerus(..) => [-1.0, 1.0],
            CrabJointId::LegCarpus(..) => [-1.1, 1.1],
            CrabJointId::ClawShoulder(_) => [CLAW_SHOULDER_UP_STOP, CLAW_SHOULDER_DOWN_STOP],
            // Tuned wrist sweep, clamped tight: ±13.7° (±0.239110 rad) about the bind
            // rest — much narrower than the other joints because the wrist rotated too far.
            CrabJointId::ClawWrist(_) => [-0.239110, 0.239110],
            CrabJointId::ClawPincer(_) => [-0.5, 0.2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `index` must be a bijection onto `0..COUNT`. Slots key every action,
    /// observation, and per-joint array, so a duplicated variant in [`all`] would
    /// give two joints the same `position` — one never actuated, the other twice —
    /// and the model would still "train". `all` being the lone source now makes
    /// that a duplicate-list bug rather than three drifting copies, but pin it
    /// regardless: every joint maps to a distinct slot covering the whole range.
    #[test]
    fn index_is_a_bijection() {
        let mut seen = [false; CrabJointId::COUNT];
        let mut count = 0;
        for id in CrabJointId::all() {
            let i = id.index();
            assert!(i < CrabJointId::COUNT, "{id:?} index {i} out of range");
            assert!(!seen[i], "{id:?} aliases slot {i}");
            seen[i] = true;
            count += 1;
        }
        assert_eq!(
            count,
            CrabJointId::COUNT,
            "all() yielded the wrong joint count"
        );
        assert!(seen.iter().all(|&s| s), "index leaves a slot unfilled");
    }
}
