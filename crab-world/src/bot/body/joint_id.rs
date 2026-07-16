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
/// A damped policy measured on an undamped plant is a mismeasure, so checkpoints
/// carry their plant ([`PLANT_FILENAME`]) and `run_eval` adopts it; `learn` and
/// `eval` both log the resolved value to make the plant provable from their
/// artifacts. Unset keeps the legacy constants bit-identical for every binary.
pub const JOINT_FRICTION_CAP_ENV: &str = "RL_JOINT_FRICTION_CAP";

static OVERRIDE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();

/// The resolved per-run cap override, read once per process. A SET-but-invalid value
/// aborts instead of defaulting: silently training days on the wrong plant is the
/// failure mode this knob exists to prevent (same policy as the rl#272 run knobs).
pub fn friction_cap_override() -> Option<f32> {
    *OVERRIDE.get_or_init(env_override)
}

fn env_override() -> Option<f32> {
    match std::env::var(JOINT_FRICTION_CAP_ENV) {
        Err(std::env::VarError::NotPresent) => None,
        Ok(raw) => Some(parse_friction_cap(&raw).unwrap_or_else(|e| {
            panic!("{JOINT_FRICTION_CAP_ENV}={raw}: {e} — refusing an ambiguous plant")
        })),
        Err(e @ std::env::VarError::NotUnicode(_)) => {
            panic!("{JOINT_FRICTION_CAP_ENV}: {e} — refusing an ambiguous plant")
        }
    }
}

/// The plant sidecar: a checkpoint dir carries the non-default plant it trained on, so
/// measurement can never silently disagree with training (bddap/rl#268). The learner
/// writes it once at startup; `run_eval` ADOPTS it (or refuses a conflict), so an
/// eval invoker needs no plant knowledge at all — the standing rl-eval-monitor keeps
/// measuring every run correctly with zero configuration. Key-value lines so future
/// plant knobs (torque slew is the reserved next lever) extend it; an eval binary
/// that meets an unknown key REFUSES rather than mismeasure a plant it can't build.
pub const PLANT_FILENAME: &str = "plant.txt";
const PLANT_KEY: &str = "joint_friction_cap";

/// Learner-side: record the resolved plant into `<ckpt>/plant.txt`, or refuse a
/// launch whose plant disagrees with what the checkpoint trained on — resuming a run
/// on a different plant would silently poison the whole experiment. Absent knob +
/// absent sidecar writes nothing (legacy runs stay byte-identical on disk).
pub fn record_plant(ckpt_dir: &std::path::Path) -> Result<(), String> {
    let path = ckpt_dir.join(PLANT_FILENAME);
    let recorded = match std::fs::read_to_string(&path) {
        Ok(text) => Some(parse_plant(&text).map_err(|e| format!("{}: {e}", path.display()))?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    match (recorded, friction_cap_override()) {
        (Some(r), Some(v)) if r == v => Ok(()),
        (Some(r), resolved) => Err(format!(
            "{}: checkpoint trained with {PLANT_KEY} {r}, but this launch resolves {} — \
             fix the run's {JOINT_FRICTION_CAP_ENV}, never change a run's plant mid-flight",
            path.display(),
            resolved.map_or("the default plant".into(), |v| v.to_string()),
        )),
        (None, Some(v)) => {
            // A brain that already trained here has UNKNOWN plant provenance (pre-knob
            // or default); stamping the knob onto it would be a mid-run plant change.
            if ckpt_dir
                .join(crate::training::checkpoint::BRAIN_FILENAME)
                .exists()
            {
                return Err(format!(
                    "{}: {JOINT_FRICTION_CAP_ENV} is set but the existing checkpoint \
                     carries no plant record — start a fresh run for a new plant",
                    ckpt_dir.display()
                ));
            }
            std::fs::create_dir_all(ckpt_dir)
                .map_err(|e| format!("{}: {e}", ckpt_dir.display()))?;
            std::fs::write(&path, format!("{PLANT_KEY} {v}\n"))
                .map_err(|e| format!("{}: {e}", path.display()))
        }
        (None, None) => Ok(()),
    }
}

/// Eval-side: make the sim use the plant the checkpoint trained on. Sidecar present →
/// install it as the process override (or verify an already-resolved env agrees);
/// absent → env/default as-is (every pre-sidecar checkpoint is the default plant).
/// MUST run before the first crab spawn — the override is read-once.
pub fn adopt_recorded_plant(ckpt_dir: &std::path::Path) -> Result<(), String> {
    let path = ckpt_dir.join(PLANT_FILENAME);
    let recorded = match std::fs::read_to_string(&path) {
        Ok(text) => parse_plant(&text).map_err(|e| format!("{}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    if OVERRIDE.set(Some(recorded)).is_ok() {
        return Ok(());
    }
    // Already resolved (env set, or a prior spawn in this process): agreement or bust.
    match friction_cap_override() {
        Some(v) if v == recorded => Ok(()),
        resolved => Err(format!(
            "{}: checkpoint trained with {PLANT_KEY} {recorded}, but this process \
             already resolved {} — refusing to mismeasure",
            path.display(),
            resolved.map_or("the default plant".into(), |v| v.to_string()),
        )),
    }
}

fn parse_plant(text: &str) -> Result<f32, String> {
    let mut cap = None;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match line.split_once(' ') {
            Some((PLANT_KEY, value)) if cap.is_none() => {
                cap = Some(parse_friction_cap(value)?);
            }
            _ => return Err(format!("unknown or duplicate plant entry {line:?}")),
        }
    }
    cap.ok_or_else(|| "no plant entries".to_string())
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

    #[test]
    fn plant_sidecar_roundtrips_and_rejects_unknown_keys() {
        assert_eq!(parse_plant("joint_friction_cap 2.5\n"), Ok(2.5));
        assert_eq!(parse_plant("joint_friction_cap 0.04"), Ok(0.04));
        for bad in [
            "",
            "torque_slew 3.0\n", // future knob: old bin must refuse
            "joint_friction_cap 2.5\ntorque_slew 3.0\n", // ditto, alongside a known key
            "joint_friction_cap 2.5\njoint_friction_cap 2.5\n", // duplicate = ambiguous
            "joint_friction_cap NaN\n",
        ] {
            assert!(parse_plant(bad).is_err(), "{bad:?} must refuse");
        }
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
