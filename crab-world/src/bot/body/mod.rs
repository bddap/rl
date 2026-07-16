mod collision;
mod components;
mod joint_id;
mod spawn;

#[cfg(feature = "render")]
mod debug_gizmos;

pub use collision::{
    ARENA_COLLISION, MAX_ENVS, NESTED_COLLISION, VEHICLE_COLLISION, crab_collision,
};

pub use components::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint, CrabModelPath,
    CrabRestPose, render_recipe,
};

pub use joint_id::{CrabJointId, Side, friction_cap_provenance, joint_angle};

pub(crate) use spawn::random_spawn_rotation;
pub use spawn::{LIMIT_SOFTNESS, SPAWN_HEIGHT, spawn_crab};

#[cfg(feature = "render")]
pub use debug_gizmos::{PivotGizmos, register_pivot_markers};
