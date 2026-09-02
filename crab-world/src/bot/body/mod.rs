mod collision;
mod components;
mod joint_id;
mod spawn;

#[cfg(feature = "render")]
mod debug_gizmos;

pub use collision::{ARENA_COLLISION, CRAB_COLLISION, VEHICLE_COLLISION};

pub use components::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint, CrabModelPath,
    CrabRestPose,
};

pub use joint_id::{
    CrabJointId, PLANT_FILENAME, Side, adopt_recorded_plant, adopt_recorded_plant_forcing_terrain,
    constructed_plant_digest, friction_cap_override, joint_angle, plant_provenance, record_plant,
};

pub(in crate::bot) use spawn::set_flail_damping;
pub(crate) use spawn::{
    CRAB_SETTLE_EXTRA_ITERATIONS, CRAB_SLEEP_NOISE_FLOOR, random_spawn_rotation,
};
pub use spawn::{LIMIT_SOFTNESS, SPAWN_HEIGHT, spawn_crab};

#[cfg(feature = "render")]
pub use debug_gizmos::{PivotGizmos, register_pivot_markers};
