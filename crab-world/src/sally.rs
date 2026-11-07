//! Sally's skeleton as rapier bodies: one convex hull per bone and a limited hinge
//! per joint, read from the rigged `.glb` that `scripts/sally/sally.sh rig` bakes. The
//! hulls, hinge axes and limits are derived there from the purchased mesh and the
//! anatomy table in `scripts/sally/skeleton.md`; this side only instantiates them.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use serde::Deserialize;

/// Crab parts never collide with each other; only the world.
pub const CRAB_GROUP: Group = Group::GROUP_2;

#[derive(Component)]
pub struct SallyBone;

/// rapier gives every free angular joint axis 0.1 N·m·s/rad of damping whatever the
/// bodies' size; on gram-scale legs that freezes the joints, so Sally's are undamped.
pub(crate) fn undamp_hinges(
    added: Query<&RapierMultibodyJointHandle, (Added<RapierMultibodyJointHandle>, With<SallyBone>)>,
    mut joints: Query<&mut RapierContextJoints>,
) {
    for handle in &added {
        for mut set in &mut joints {
            if let Some((multibody, _)) = set.multibody_joints.get_mut(handle.0) {
                multibody.damping_mut().fill(0.0);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    /// Rest pose in model space (glTF's, Y up).
    pub rest: Transform,
    /// Bone-local points, meters.
    pub hull: Vec<Vec3>,
    pub density: f32,
    pub hinge: Option<Hinge>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hinge {
    /// Bone-local unit axis; positive is dorsal, anterior or opening, per
    /// `scripts/sally/skeleton.md`.
    pub axis: Vec3,
    /// Radians from the rest pose.
    pub limits: [f32; 2],
}

#[derive(Deserialize)]
struct Gltf {
    nodes: Vec<Node>,
    skins: Vec<Skin>,
}

#[derive(Deserialize)]
struct Skin {
    joints: Vec<usize>,
}

#[derive(Deserialize)]
struct Node {
    name: Option<String>,
    #[serde(default)]
    children: Vec<usize>,
    translation: Option<[f32; 3]>,
    rotation: Option<[f32; 4]>,
    scale: Option<[f32; 3]>,
    extras: Option<Extras>,
}

#[derive(Deserialize)]
struct Extras {
    rl: Option<BoneExtras>,
}

#[derive(Deserialize)]
struct BoneExtras {
    hull: Vec<f32>,
    density: f32,
    axis: Option<[f32; 3]>,
    limits_deg: Option<[f32; 2]>,
}

impl Node {
    fn local(&self) -> Transform {
        Transform {
            translation: self.translation.map_or(Vec3::ZERO, Vec3::from),
            rotation: self.rotation.map_or(Quat::IDENTITY, Quat::from_array),
            scale: self.scale.map_or(Vec3::ONE, Vec3::from),
        }
    }
}

fn json_chunk(glb: &[u8]) -> Result<&[u8], String> {
    let word = |at: usize| {
        glb.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or("truncated glb header")
    };
    if glb.get(0..4) != Some(b"glTF") {
        return Err("not a glb".into());
    }
    if glb.get(16..20) != Some(b"JSON") {
        return Err("glb's first chunk is not JSON".into());
    }
    let len = word(12)?;
    glb.get(20..20 + len)
        .ok_or_else(|| "truncated JSON chunk".into())
}

impl Skeleton {
    pub fn from_glb(glb: &[u8]) -> Result<Self, String> {
        let doc: Gltf = serde_json::from_slice(json_chunk(glb)?).map_err(|e| e.to_string())?;
        let [skin] = doc.skins.as_slice() else {
            return Err(format!("expected one skin, found {}", doc.skins.len()));
        };
        let mut parent_node = vec![None; doc.nodes.len()];
        for (i, node) in doc.nodes.iter().enumerate() {
            for &c in &node.children {
                parent_node[c] = Some(i);
            }
        }
        let model = |mut i: usize| {
            let mut t = doc.nodes[i].local();
            while let Some(p) = parent_node[i] {
                t = doc.nodes[p].local() * t;
                i = p;
            }
            t
        };
        let bone_of = |node: usize| skin.joints.iter().position(|&j| j == node);
        skin.joints
            .iter()
            .enumerate()
            .map(|(i, &j)| {
                let node = &doc.nodes[j];
                let name = node.name.clone().unwrap_or_default();
                let rl = node
                    .extras
                    .as_ref()
                    .and_then(|e| e.rl.as_ref())
                    .ok_or_else(|| format!("bone {name} carries no body"))?;
                let mut up = parent_node[j];
                let parent = loop {
                    match up {
                        None => break None,
                        Some(p) => match bone_of(p) {
                            Some(b) => break Some(b),
                            None => up = parent_node[p],
                        },
                    }
                };
                if parent.is_some_and(|p| p >= i) {
                    return Err(format!("bone {name} comes before its parent"));
                }
                let hinge = match (parent, rl.axis, rl.limits_deg) {
                    (None, _, _) => None,
                    (Some(_), Some(axis), Some([lo, hi])) => Some(Hinge {
                        axis: Vec3::from(axis).normalize(),
                        limits: [lo.to_radians(), hi.to_radians()],
                    }),
                    _ => return Err(format!("bone {name} has a parent but no hinge")),
                };
                Ok(Bone {
                    name,
                    parent,
                    rest: model(j),
                    hull: rl
                        .hull
                        .chunks_exact(3)
                        .map(|p| Vec3::new(p[0], p[1], p[2]))
                        .collect(),
                    density: rl.density,
                    hinge,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|bones| Self { bones })
    }

    /// One dynamic body per bone at `at * rest`, children jointed to parents at rest
    /// angle zero. Returns the entities in bone order.
    pub fn spawn(&self, commands: &mut Commands, at: Transform) -> Vec<Entity> {
        let mut entities: Vec<Entity> = Vec::with_capacity(self.bones.len());
        for bone in &self.bones {
            let pose = at * bone.rest;
            let collider = Collider::convex_hull(&bone.hull)
                .unwrap_or_else(|| panic!("bone {} has a degenerate hull", bone.name));
            let mut entity = commands.spawn((
                SallyBone,
                Name::new(bone.name.clone()),
                RigidBody::Dynamic,
                collider,
                ColliderMassProperties::Density(bone.density),
                CollisionGroups::new(CRAB_GROUP, Group::ALL ^ CRAB_GROUP),
                Transform::from_isometry(pose.to_isometry()),
            ));
            if let (Some(p), Some(hinge)) = (bone.parent, bone.hinge) {
                let parent = &self.bones[p];
                let rel = parent.rest.to_isometry().inverse() * bone.rest.to_isometry();
                let frame = Quat::from_rotation_arc(Vec3::X, hinge.axis);
                let joint = GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES)
                    .local_anchor1(rel.translation.into())
                    .local_basis1(rel.rotation * frame)
                    .local_anchor2(Vec3::ZERO)
                    .local_basis2(frame)
                    .limits(JointAxis::AngX, hinge.limits)
                    .contacts_enabled(false);
                // Reduced coordinates: as impulse joints, the light distal segments
                // overran their limits by up to 104° under the body's weight.
                entity.insert(MultibodyJoint::new(entities[p], joint));
            }
            entities.push(entity.id());
        }
        entities
    }

    /// Each hinge's angle from rest, radians, given every bone's world pose.
    pub fn hinge_angles(&self, poses: &[Transform]) -> Vec<Option<f32>> {
        self.bones
            .iter()
            .zip(poses)
            .map(|(bone, pose)| {
                let (p, hinge) = (bone.parent?, bone.hinge?);
                let rest = self.bones[p].rest.rotation.inverse() * bone.rest.rotation;
                let now = poses[p].rotation.inverse() * pose.rotation;
                let q = rest.inverse() * now;
                let q = if q.w < 0.0 { -q } else { q };
                Some(2.0 * q.xyz().dot(hinge.axis).atan2(q.w))
            })
            .collect()
    }

    /// The largest amount, radians, by which any hinge in `poses` passes its limits
    /// beyond the solver's slack.
    pub fn limit_excess(&self, poses: &[Transform]) -> f32 {
        const SLACK: f32 = 0.035;
        self.hinge_angles(poses)
            .iter()
            .zip(&self.bones)
            .filter_map(|(angle, bone)| Some((angle.as_ref()?, bone.hinge?.limits)))
            .map(|(&a, [lo, hi])| (lo - a).max(a - hi).max(0.0) - SLACK)
            .fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless::headless_world;
    use crate::terrain::TerrainGrid;
    use bevy::ecs::system::RunSystemOnce;
    use std::sync::Arc;

    fn cube(half: f32) -> Vec<f32> {
        (0..8)
            .flat_map(|i| {
                [
                    if i & 1 == 0 { -half } else { half },
                    if i & 2 == 0 { -half } else { half },
                    if i & 4 == 0 { -half } else { half },
                ]
            })
            .collect()
    }

    /// A body with one three-bone leg held out level: bones along local +Y, a node
    /// above the root bone carrying the Z-up to Y-up turn, and hinges on a horizontal
    /// axis so gravity loads them.
    fn glb(limits_deg: Option<[f32; 2]>) -> Vec<u8> {
        let bone = |name: &str, t: [f32; 3], children: Vec<usize>, joint: bool| {
            let mut rl = serde_json::json!({"hull": cube(if joint { 0.004 } else { 0.02 }), "density": 1100.0});
            if joint {
                rl["axis"] = serde_json::json!([1.0, 0.0, 0.0]);
                if let Some(limits) = limits_deg {
                    rl["limits_deg"] = serde_json::json!(limits);
                }
            }
            serde_json::json!({"name": name, "translation": t, "children": children,
                               "extras": {"rl": rl}})
        };
        let doc = serde_json::json!({
            "nodes": [
                {"name": "sally", "rotation": [-0.70710677, 0.0, 0.0, 0.70710677],
                 "children": [1, 5]},
                bone("body", [0.0, 0.0, 0.03], vec![2], false),
                bone("coxa", [0.025, 0.0, 0.0], vec![3], true),
                bone("merus", [0.0, 0.015, 0.0], vec![4], true),
                bone("dactylus", [0.0, 0.015, 0.0], vec![], true),
                {"name": "mesh"},
            ],
            "skins": [{"joints": [1, 2, 3, 4]}],
        });
        let mut json = serde_json::to_vec(&doc).unwrap();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut out = b"glTF".to_vec();
        out.extend(2u32.to_le_bytes());
        out.extend(((12 + 8 + json.len()) as u32).to_le_bytes());
        out.extend((json.len() as u32).to_le_bytes());
        out.extend(b"JSON");
        out.extend(json);
        out
    }

    #[test]
    fn reads_the_rig() {
        let skeleton = Skeleton::from_glb(&glb(Some([-30.0, 10.0]))).unwrap();
        let names: Vec<_> = skeleton.bones.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, ["body", "coxa", "merus", "dactylus"]);
        let parents: Vec<_> = skeleton.bones.iter().map(|b| b.parent).collect();
        assert_eq!(parents, [None, Some(0), Some(1), Some(2)]);
        assert!(skeleton.bones[0].hinge.is_none());
        assert_eq!(
            skeleton.bones[2].hinge.unwrap().limits,
            [(-30.0f32).to_radians(), 10.0f32.to_radians()]
        );
        assert!(
            skeleton.bones[0]
                .rest
                .translation
                .abs_diff_eq(Vec3::new(0.0, 0.03, 0.0), 1e-6)
        );
        assert!(
            skeleton.bones[1]
                .rest
                .translation
                .abs_diff_eq(Vec3::new(0.025, 0.03, 0.0), 1e-6)
        );
        assert_eq!(skeleton.bones[3].hull.len(), 8);
    }

    #[test]
    fn rejects_a_jointed_bone_without_a_hinge() {
        assert!(
            Skeleton::from_glb(&glb(None))
                .unwrap_err()
                .contains("no hinge")
        );
    }

    fn settle(skeleton: &Skeleton, lift: f32, ticks: usize) -> (App, Vec<Entity>) {
        let mut app = headless_world(Arc::new(TerrainGrid::flat(1.0)));
        let skeleton = skeleton.clone();
        let entities = app
            .world_mut()
            .run_system_once(move |mut commands: Commands| {
                skeleton.spawn(&mut commands, Transform::from_xyz(0.0, lift, 0.0))
            })
            .unwrap();
        for _ in 0..ticks {
            app.update();
        }
        (app, entities)
    }

    fn poses(app: &App, entities: &[Entity]) -> Vec<Transform> {
        entities
            .iter()
            .map(|&e| *app.world().get::<Transform>(e).unwrap())
            .collect()
    }

    #[test]
    fn spawns_at_rest_with_zero_hinge_angles() {
        let skeleton = Skeleton::from_glb(&glb(Some([-30.0, 10.0]))).unwrap();
        let (app, entities) = settle(&skeleton, 0.2, 1);
        for angle in skeleton
            .hinge_angles(&poses(&app, &entities))
            .into_iter()
            .flatten()
        {
            assert!(angle.abs() < 0.02, "{angle}");
        }
    }

    #[test]
    fn a_dropped_crab_keeps_its_hinges_in_range() {
        let skeleton = Skeleton::from_glb(&glb(Some([-30.0, 10.0]))).unwrap();
        let (app, entities) = settle(&skeleton, 0.05, 192);
        let poses = poses(&app, &entities);
        assert_eq!(skeleton.limit_excess(&poses), 0.0);
        let at_a_stop = skeleton
            .hinge_angles(&poses)
            .iter()
            .zip(&skeleton.bones)
            .filter_map(|(a, b)| Some((a.as_ref()?, b.hinge?.limits)))
            .any(|(&a, [lo, hi])| (a - lo).abs().min((a - hi).abs()) < 3f32.to_radians());
        assert!(
            at_a_stop,
            "gravity never loaded a hinge against its stop: {:?} {:?}",
            skeleton.hinge_angles(&poses),
            poses
        );
    }
}
