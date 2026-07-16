use bevy::prelude::*;

const COXA_TORQUE_CEILING: f32 = 6.0;
const FEMUR_TORQUE_CEILING: f32 = 4.0;
const TIBIA_TORQUE_CEILING: f32 = 2.5;
const CLAW_PINCER_TORQUE_CEILING: f32 = 7.0;
const CLAW_SHOULDER_TORQUE_CEILING: f32 = 4.0;
const CLAW_WRIST_TORQUE_CEILING: f32 = 3.0;

const CLAW_SHOULDER_UP_STOP: f32 = -0.35;
const CLAW_SHOULDER_DOWN_STOP: f32 = 1.0;

const LEG_FRICTION_CAP: f32 = 0.04;
const CLAW_FRICTION_CAP: f32 = 0.04;

/// Joint-damping experiment knob (bddap/rl#268): overrides the friction-motor torque
/// cap, in N·m, on EVERY crab joint. The 0.04 defaults give the plant ~1% of drive
/// authority in damping — flailing is free, so torque-saturating bang-bang is the
/// optimal policy; raising the cap toward drive authority prices oscillation.
///
/// An env var rather than a clap flag (the rl#272 run-knob front door) because it is
/// a PLANT property that must reach binaries with no CLI at all — game/rl-demo spawn
/// this same crab — as well as every sim inside one process: the learn rollout
/// threads, the in-process keep-best chase-eval, and the external honest-eval binary.
/// A damped policy measured on an undamped plant is a mismeasure, so evals of a
/// damped run must export this too; `learn` and `eval` both log the resolved value to
/// make the plant provable from their artifacts (checkpoints do NOT carry their
/// plant). Unset keeps the legacy constants bit-identical for every binary.
pub const JOINT_FRICTION_CAP_ENV: &str = "RL_JOINT_FRICTION_CAP";

/// The resolved per-run cap override, read once per process. A SET-but-invalid value
/// aborts instead of defaulting: silently training days on the wrong plant is the
/// failure mode this knob exists to prevent (same policy as the rl#272 run knobs).
pub fn friction_cap_override() -> Option<f32> {
    static OVERRIDE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| match std::env::var(JOINT_FRICTION_CAP_ENV) {
        Err(std::env::VarError::NotPresent) => None,
        Ok(raw) => Some(parse_friction_cap(&raw).unwrap_or_else(|e| {
            panic!("{JOINT_FRICTION_CAP_ENV}={raw}: {e} — refusing an ambiguous plant")
        })),
        Err(e @ std::env::VarError::NotUnicode(_)) => {
            panic!("{JOINT_FRICTION_CAP_ENV}: {e} — refusing an ambiguous plant")
        }
    })
}

/// One human-readable source for "which plant is this?" — logged by `learn` (into
/// train.log) and `eval` (beside the `EVAL_RESULT` lines) so run and measurement
/// artifacts both prove the friction cap they ran under.
pub fn friction_cap_provenance() -> String {
    match friction_cap_override() {
        Some(cap) => format!("joint friction cap {cap} N·m ({JOINT_FRICTION_CAP_ENV})"),
        None => format!(
            "joint friction cap leg {LEG_FRICTION_CAP} / claw {CLAW_FRICTION_CAP} N·m (default)"
        ),
    }
}

fn parse_friction_cap(raw: &str) -> Result<f32, String> {
    let v: f32 = raw
        .trim()
        .parse()
        .map_err(|e| format!("not a float: {e}"))?;
    // Negative is meaningless for a force cap; NaN/inf would poison the solver.
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(format!("{v} is not a finite non-negative torque cap"))
    }
}

pub fn joint_angle(axis_local: Vec3, parent_rot: Quat, child_rot: Quat) -> f32 {
    let q = (parent_rot.inverse() * child_rot).normalize();
    let v = Vec3::new(q.x, q.y, q.z);
    2.0 * v.dot(axis_local).atan2(q.w)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CrabJointId {
    LegCoxa(Side, u8),
    LegBasis(Side, u8),
    LegMerus(Side, u8),
    LegCarpus(Side, u8),
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
    /// The canonical joint enumeration — THE channel order of the action vector and the
    /// per-joint obs channels. Reordering it remaps a trained checkpoint's channels, so
    /// the order is folded into [`crate::bot::channel_layout_digest`] and stale brains
    /// refuse to load (bddap/rl#271).
    pub(crate) const fn all() -> [CrabJointId; 38] {
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

    pub const COUNT: usize = Self::all().len();

    pub fn index(&self) -> usize {
        Self::all()
            .iter()
            .position(|j| j == self)
            .expect("every CrabJointId is listed in all()")
    }

    pub fn from_index(i: usize) -> Option<CrabJointId> {
        Self::all().get(i).copied()
    }
}

impl CrabJointId {
    pub fn drive_torque_ceiling(&self) -> f32 {
        match self {
            CrabJointId::LegCoxa(..) | CrabJointId::LegBasis(..) => COXA_TORQUE_CEILING,
            CrabJointId::LegMerus(..) => FEMUR_TORQUE_CEILING,
            CrabJointId::LegCarpus(..) => TIBIA_TORQUE_CEILING,
            CrabJointId::ClawShoulder(_) => CLAW_SHOULDER_TORQUE_CEILING,
            CrabJointId::ClawWrist(_) => CLAW_WRIST_TORQUE_CEILING,
            CrabJointId::ClawPincer(_) => CLAW_PINCER_TORQUE_CEILING,
        }
    }

    pub fn friction_cap(&self) -> f32 {
        if let Some(cap) = friction_cap_override() {
            return cap;
        }
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

    pub fn limits(&self) -> [f32; 2] {
        match self {
            CrabJointId::LegCoxa(..) => [-0.8, 0.8],
            CrabJointId::LegBasis(..) => [-0.6, 0.6],
            CrabJointId::LegMerus(..) => [-1.0, 1.0],
            CrabJointId::LegCarpus(..) => [-1.1, 1.1],
            CrabJointId::ClawShoulder(_) => [CLAW_SHOULDER_UP_STOP, CLAW_SHOULDER_DOWN_STOP],
            CrabJointId::ClawWrist(_) => [-0.239110, 0.239110],
            CrabJointId::ClawPincer(_) => [-0.5, 0.2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Pure-parse tests only: the env read is process-global (OnceLock) and tests run
    // multi-threaded, so exercising the var itself would race sibling tests.
    #[test]
    fn friction_cap_parse_accepts_plausible_and_zero() {
        assert_eq!(parse_friction_cap("2.5"), Ok(2.5));
        assert_eq!(parse_friction_cap(" 0.04 "), Ok(0.04));
        assert_eq!(parse_friction_cap("0"), Ok(0.0));
    }

    #[test]
    fn friction_cap_parse_rejects_ambiguous_plants() {
        for bad in ["-0.5", "NaN", "inf", "2.5Nm", ""] {
            assert!(
                parse_friction_cap(bad).is_err(),
                "{bad:?} must not launch a run"
            );
        }
    }
}
