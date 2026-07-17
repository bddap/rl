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
const FRICTION_KEY: &str = "joint_friction_cap";
const ARENA_KEY: &str = "arena";

/// The set-override half of the plant — exactly what the sidecar records. Absent key =
/// that knob's default; the file exists only when some knob is non-default, so legacy
/// runs stay byte-identical on disk.
#[derive(Clone, Copy, PartialEq)]
struct Plant {
    friction_cap: Option<f32>,
    arena: Option<crate::physics::TrainArena>,
}

impl Plant {
    fn resolved() -> Self {
        Plant {
            friction_cap: friction_cap_override(),
            arena: crate::physics::train_arena_override(),
        }
    }

    /// The default plant — no sidecar on disk. Every per-knob method below opens with
    /// an exhaustive destructure so the reserved next knob (torque slew) cannot be
    /// silently dropped from one of them: a slew-only plant missed by `is_default`
    /// would record NO sidecar, the exact mismeasure class this file exists to refuse.
    fn is_default(&self) -> bool {
        let Plant {
            friction_cap,
            arena,
        } = *self;
        friction_cap.is_none() && arena.is_none()
    }

    /// The sidecar file body — one `key value` line per set knob.
    fn render(&self) -> String {
        let Plant {
            friction_cap,
            arena,
        } = *self;
        let mut out = String::new();
        if let Some(cap) = friction_cap {
            out.push_str(&format!("{FRICTION_KEY} {cap}\n"));
        }
        if let Some(arena) = arena {
            out.push_str(&format!("{ARENA_KEY} {}\n", arena.key()));
        }
        out
    }
}

impl std::fmt::Display for Plant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Plant {
            friction_cap,
            arena,
        } = *self;
        if self.is_default() {
            return write!(f, "the default plant");
        }
        let mut sep = "";
        if let Some(cap) = friction_cap {
            write!(f, "{FRICTION_KEY} {cap}")?;
            sep = ", ";
        }
        if let Some(arena) = arena {
            write!(f, "{sep}{ARENA_KEY} {}", arena.key())?;
        }
        Ok(())
    }
}

/// Learner-side: record the resolved plant into `<ckpt>/plant.txt`, or refuse a
/// launch whose plant disagrees with what the checkpoint trained on — resuming a run
/// on a different plant would silently poison the whole experiment. Default plant +
/// absent sidecar writes nothing (legacy runs stay byte-identical on disk).
pub fn record_plant(ckpt_dir: &std::path::Path) -> Result<(), String> {
    let path = ckpt_dir.join(PLANT_FILENAME);
    let recorded = match std::fs::read_to_string(&path) {
        Ok(text) => Some(parse_plant(&text).map_err(|e| format!("{}: {e}", path.display()))?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let resolved = Plant::resolved();
    match recorded {
        Some(r) if r == resolved => Ok(()),
        Some(r) => Err(format!(
            "{}: checkpoint trained with [{r}], but this launch resolves [{resolved}] — \
             fix the run's {JOINT_FRICTION_CAP_ENV}/{}, never change a run's plant \
             mid-flight",
            path.display(),
            crate::physics::ARENA_ENV,
        )),
        None if resolved.is_default() => Ok(()),
        None => {
            // A brain that already trained here has UNKNOWN plant provenance (pre-knob
            // or default); stamping the knob onto it would be a mid-run plant change.
            if ckpt_dir
                .join(crate::training::checkpoint::BRAIN_FILENAME)
                .exists()
            {
                return Err(format!(
                    "{}: a non-default plant is set but the existing checkpoint \
                     carries no plant record — start a fresh run for a new plant",
                    ckpt_dir.display()
                ));
            }
            std::fs::create_dir_all(ckpt_dir)
                .map_err(|e| format!("{}: {e}", ckpt_dir.display()))?;
            std::fs::write(&path, resolved.render()).map_err(|e| format!("{}: {e}", path.display()))
        }
    }
}

/// Eval-side: make the sim use the plant the checkpoint trained on — the WHOLE plant,
/// absent keys included: an absent key (or absent file — pre-sidecar and default-plant
/// checkpoints alike) means that knob's DEFAULT, so a stray env override is pinned out
/// (or, if the env was already read, refused) rather than silently bending the
/// measurement. MUST run before the first world spawns — the overrides are read-once.
pub fn adopt_recorded_plant(ckpt_dir: &std::path::Path) -> Result<(), String> {
    let path = ckpt_dir.join(PLANT_FILENAME);
    let recorded = match std::fs::read_to_string(&path) {
        Ok(text) => parse_plant(&text).map_err(|e| format!("{}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Plant {
            friction_cap: None,
            arena: None,
        },
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    if OVERRIDE.set(recorded.friction_cap).is_err()
        && friction_cap_override() != recorded.friction_cap
    {
        // Already resolved (env set, or a prior spawn in this process): agreement or bust.
        return Err(format!(
            "{}: checkpoint plant has {FRICTION_KEY} {}, but this process already \
             resolved {} — refusing to mismeasure",
            path.display(),
            recorded
                .friction_cap
                .map_or("<default>".into(), |v| v.to_string()),
            friction_cap_override().map_or("<default>".into(), |v| v.to_string()),
        ));
    }
    crate::physics::world::adopt_train_arena(recorded.arena)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn parse_plant(text: &str) -> Result<Plant, String> {
    let mut plant = Plant {
        friction_cap: None,
        arena: None,
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match line.split_once(' ') {
            Some((FRICTION_KEY, value)) if plant.friction_cap.is_none() => {
                plant.friction_cap = Some(parse_friction_cap(value)?);
            }
            Some((ARENA_KEY, value)) if plant.arena.is_none() => {
                plant.arena = Some(crate::physics::TrainArena::parse(value)?);
            }
            _ => return Err(format!("unknown or duplicate plant entry {line:?}")),
        }
    }
    if plant.is_default() {
        return Err("no plant entries".to_string());
    }
    Ok(plant)
}

/// The effective-plant digest the MP membership handshake advertises (rl#286),
/// the world-identity sibling of [`crate::mesh_fallback::constructed_body_digest`]:
/// arena tag, the terrain bake's byte digest when that arena stands on one, and the
/// per-joint friction caps the solver will actually run. Peers whose digests differ
/// would simulate/render DIFFERENT WORLDS under identical body digests — a flat
/// client adopting terrain-height poses floats/buries every crab and craft — so the
/// sync verdict refuses to arm the round instead.
///
/// Digests the RESOLVED physics, not the override provenance: an env/sidecar cap set
/// to the default value hashes identically to no override, and a changed default
/// constant changes the digest. MUST run after [`adopt_recorded_plant`] (it reads the
/// same read-once overrides — reading first would latch the pre-adopt plant, which
/// adoption then refuses loudly rather than mismeasure).
pub fn constructed_plant_digest() -> u64 {
    let mut h = crate::fnv::Fnv::new();
    let arena = crate::physics::train_arena();
    h.write(arena.key().as_bytes());
    if arena == crate::physics::TrainArena::Terrain {
        h.write(&crate::terrain::gcr_bake_digest().to_le_bytes());
    }
    let cap = friction_cap_override();
    h.write(&cap.unwrap_or(LEG_FRICTION_CAP).to_bits().to_le_bytes());
    h.write(&cap.unwrap_or(CLAW_FRICTION_CAP).to_bits().to_le_bytes());
    h.finish()
}

/// One human-readable source for "which plant is this?" — logged by `learn` (into
/// train.log) and `eval` (beside the `EVAL_RESULT` lines) so run and measurement
/// artifacts both prove the friction cap and arena they ran under.
pub fn plant_provenance() -> String {
    let cap = match friction_cap_override() {
        Some(cap) => format!("joint friction cap {cap} N·m ({JOINT_FRICTION_CAP_ENV})"),
        None => format!(
            "joint friction cap leg {LEG_FRICTION_CAP} / claw {CLAW_FRICTION_CAP} N·m (default)"
        ),
    };
    let arena = match crate::physics::train_arena_override() {
        Some(a) => format!("arena {} ({})", a.key(), crate::physics::ARENA_ENV),
        None => format!("arena {} (default)", crate::physics::train_arena().key()),
    };
    format!("{cap}; {arena}")
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
        let cap = |text: &str| parse_plant(text).map(|p| p.friction_cap);
        assert_eq!(cap("joint_friction_cap 2.5\n"), Ok(Some(2.5)));
        assert_eq!(cap("joint_friction_cap 0.04"), Ok(Some(0.04)));
        let arena = |text: &str| parse_plant(text).map(|p| p.arena);
        assert_eq!(
            arena("arena terrain\n"),
            Ok(Some(crate::physics::TrainArena::Terrain))
        );
        let both = parse_plant("joint_friction_cap 2.5\narena terrain\n").expect("both keys");
        assert_eq!(
            (both.friction_cap, both.arena),
            (Some(2.5), Some(crate::physics::TrainArena::Terrain))
        );
        // render/parse round-trip: what record_plant writes, adopt reads back equal.
        assert!(parse_plant(&both.render()).expect("round-trip") == both);
        for bad in [
            "",
            "torque_slew 3.0\n", // future knob: old bin must refuse
            "joint_friction_cap 2.5\ntorque_slew 3.0\n", // ditto, alongside a known key
            "joint_friction_cap 2.5\njoint_friction_cap 2.5\n", // duplicate = ambiguous
            "joint_friction_cap NaN\n",
            "arena mars\n",                   // unknown arena
            "arena terrain\narena terrain\n", // duplicate = ambiguous
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
