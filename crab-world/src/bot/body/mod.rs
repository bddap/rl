//! Crab body definition: the `CrabJointId` action/observation joint set, the
//! per-instance joint + marker components, and `spawn_crab`, which instantiates
//! the rig-derived body recipe ([`super::rig`] owns the geometry).
//!
//! The body is a Rapier multibody tree rooted at the carapace. Only the locomotion
//! joints are policy-actuated and carry a [`CrabJointId`]: each leg's
//! coxa/basis/merus/carpus and each claw's shoulder/wrist/pincer (all revolute). The
//! basal joint is 2-DOF — coxa swings the leg fore/aft, basis lifts it up/down —
//! split across two short links; the remaining proximal bones ride those links, so
//! there are no locked stubs. The cosmetic eye-stalks (locked, no `CrabJointId`) are
//! NOT given physics bodies at all — they aren't actuated, observed, or load-bearing,
//! and their skin bones ride the carapace ([`super::skin`]); they live only in the rig
//! for the cosmetic/debug view. The rest of the rig (shell, palpi, mouthparts) rides
//! the carapace. Add articulation by adding a `CrabJointId` variant and a
//! [`super::rig`] `JointSpec`.
//!
//! Split into cohesive units (this `mod.rs` just declares + re-exports them):
//! [`collision`] (groups + membership), [`joint_id`] (the `CrabJointId` set + its
//! tuning), [`components`] (the ECS markers + the `CrabAssets` recipe), [`spawn`]
//! (`spawn_crab` + helpers), and the render-only [`debug_gizmos`].

mod collision;
mod components;
mod joint_id;
mod spawn;

#[cfg(feature = "render")]
mod debug_gizmos;

pub use collision::{ARENA_COLLISION, MAX_ENVS, NESTED_COLLISION, crab_collision, vehicle_collision};

pub use components::{
    CrabAssets, CrabBodyPart, CrabCarapace, CrabClawTip, CrabEnvId, CrabJoint, CrabModelPath,
    CrabRestPose, render_recipe,
};

pub use joint_id::{CrabJointId, LEG_FRICTION_CAP, Side, joint_angle};

pub use spawn::{LIMIT_SOFTNESS, SPAWN_HEIGHT, spawn_crab};
pub(crate) use spawn::random_spawn_rotation;

#[cfg(feature = "render")]
pub use debug_gizmos::{PivotGizmos, register_pivot_markers};
