//! The crab policy `docs/policy.md` designs: one readout shared by every appendage,
//! fed that appendage's mirrored view of the crab, emitting its servo targets.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::eval::{Observation, Policy, model_left};
use crate::physics::PHYSICS_DT;
use crate::sally::{Hinge, Skeleton};

/// *Grapsus* stride frequency at its routine pace (Martinez et al. 1998, Table 1).
const CLOCK_HZ: f32 = 1.17;
const RATE_SCALE: f32 = 10.0;
const SPEED_SCALE: f32 = 0.1;
const HEIGHT_SCALE: f32 = 0.05;
const CARAPACE_INPUTS: usize = 10;
const WALK_INPUTS: usize = 3;
const STRIDE_INPUTS: usize = 7;
/// The share of its distance to the readout's targets the action covers each tick.
const CATCH_UP: f32 = 0.25;

/// The shared appendage readout, one appendage's view to its servo targets: each
/// target the tanh of a weighted sum of the view. `weights` is row-major, one row of
/// `inputs` per target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Actor {
    pub inputs: usize,
    pub weights: Vec<f32>,
}

impl Actor {
    /// All zeros: every target is the rest pose.
    pub fn new(anatomy: &Anatomy) -> Self {
        Self {
            inputs: anatomy.view_size(),
            weights: vec![0.0; anatomy.view_size() * anatomy.appendage_hinges()],
        }
    }

    fn check(&self, anatomy: &Anatomy) -> Result<(), String> {
        let (want_in, want_out) = (anatomy.view_size(), anatomy.appendage_hinges());
        if self.inputs != want_in || self.weights.len() != want_in * want_out {
            return Err(format!(
                "the policy holds {} weights over {} inputs; this rig needs {want_out} targets over {want_in}",
                self.weights.len(),
                self.inputs
            ));
        }
        Ok(())
    }

    fn targets<'a>(&'a self, view: &'a [f32]) -> impl Iterator<Item = f32> + 'a {
        assert_eq!(view.len(), self.inputs, "view size");
        self.weights
            .chunks_exact(self.inputs)
            .map(move |row| row.iter().zip(view).map(|(w, x)| w * x).sum::<f32>().tanh())
    }
}

/// One appendage: its hinges, proximal first, as indices into the per-hinge vectors.
#[derive(Debug, Clone)]
struct Appendage {
    hinges: Vec<usize>,
    /// A right appendage, seen through the sagittal mirror.
    mirrored: bool,
    kind: usize,
}

/// How the policy reads a [`Skeleton`]: its appendages, and its carapace's rest axes.
#[derive(Debug, Clone)]
pub struct Anatomy {
    appendages: Vec<Appendage>,
    kinds: usize,
    hinge_bones: Vec<usize>,
    parents: Vec<usize>,
    hinges: Vec<Hinge>,
    /// Model-space left, up and forward.
    basis: [Vec3; 3],
    carapace_rest: Quat,
}

impl Anatomy {
    pub fn new(skeleton: &Skeleton) -> Result<Self, String> {
        let hinge_bones = skeleton.hinge_bones();
        let mut groups: Vec<(String, bool, Vec<usize>)> = Vec::new();
        for (h, &b) in hinge_bones.iter().enumerate() {
            let name = &skeleton.bones[b].name;
            let (kind, side) = name
                .split_once('.')
                .zip(name.rsplit_once('.'))
                .map(|((kind, _), (_, side))| (kind, side))
                .ok_or_else(|| format!("bone {name} is not <appendage>.<segment>.<L|R>"))?;
            let mirrored = match side {
                "L" => false,
                "R" => true,
                _ => return Err(format!("bone {name} is on neither side")),
            };
            match groups
                .iter_mut()
                .find(|(k, m, _)| k == kind && *m == mirrored)
            {
                Some((_, _, hinges)) => hinges.push(h),
                None => groups.push((kind.to_owned(), mirrored, vec![h])),
            }
        }
        let mut kinds: Vec<&str> = groups.iter().map(|(k, _, _)| k.as_str()).collect();
        kinds.sort();
        kinds.dedup();
        let appendages: Vec<Appendage> = groups
            .iter()
            .map(|(kind, mirrored, hinges)| Appendage {
                hinges: hinges.clone(),
                mirrored: *mirrored,
                kind: kinds.iter().position(|k| k == kind).expect("kind listed"),
            })
            .collect();
        let Some(n) = appendages.first().map(|a| a.hinges.len()) else {
            return Err("the rig has no hinges".into());
        };
        if appendages.iter().any(|a| a.hinges.len() != n) {
            return Err("every appendage needs the same number of hinges".into());
        }
        let left = model_left(skeleton)?;
        Ok(Self {
            kinds: kinds.len(),
            parents: hinge_bones
                .iter()
                .map(|&b| skeleton.bones[b].parent.expect("hinge bone has a parent"))
                .collect(),
            hinges: hinge_bones
                .iter()
                .map(|&b| skeleton.bones[b].hinge.expect("hinge bone"))
                .collect(),
            hinge_bones,
            appendages,
            basis: [left, Vec3::Y, left.cross(Vec3::Y)],
            carapace_rest: skeleton.bones[0].rest.rotation,
        })
    }

    fn appendage_hinges(&self) -> usize {
        self.appendages[0].hinges.len()
    }

    fn view_size(&self) -> usize {
        3 * self.appendage_hinges() + CARAPACE_INPUTS + WALK_INPUTS + STRIDE_INPUTS * self.kinds
    }

    /// Which of a view's inputs are a leading or a trailing stride: zero on every
    /// tick of a task that does not walk, so no weight on one changes what the
    /// policy does there.
    pub fn strides(&self) -> Vec<bool> {
        let off_a_walk = stride(0.0, [1.0, 1.0]).map(|x| x == 0.0);
        let before = self.view_size() - STRIDE_INPUTS * self.kinds;
        (0..self.view_size())
            .map(|i| i >= before && off_a_walk[(i - before) % STRIDE_INPUTS])
            .collect()
    }

    /// Each hinge's angle in action units (0 rest, ±1 its limits) and its rate,
    /// scaled.
    fn joints(&self, obs: &Observation) -> (Vec<f32>, Vec<f32>) {
        let unit = obs
            .hinges
            .iter()
            .zip(&self.hinges)
            .map(|(&a, hinge)| hinge.action(a))
            .collect();
        let rate = self
            .hinge_bones
            .iter()
            .zip(&self.parents)
            .zip(&self.hinges)
            .map(|((&b, &p), hinge)| {
                let (child, parent) = (&obs.bones[b], &obs.bones[p]);
                (child.angvel - parent.angvel).dot(child.pose.rotation * hinge.axis) / RATE_SCALE
            })
            .collect();
        (unit, rate)
    }

    fn local(&self, obs: &Observation, v: Vec3) -> Vec3 {
        let frame = (obs.bones[0].pose.rotation * self.carapace_rest.inverse()).inverse();
        let m = frame * v;
        Vec3::new(
            m.dot(self.basis[0]),
            m.dot(self.basis[1]),
            m.dot(self.basis[2]),
        )
    }

    /// Gravity's direction, velocity and angular velocity in the carapace's frame,
    /// scaled.
    fn carapace(&self, obs: &Observation) -> [Vec3; 3] {
        let body = &obs.bones[0];
        [
            self.local(obs, Vec3::NEG_Y),
            self.local(obs, body.linvel) / SPEED_SCALE,
            self.local(obs, body.angvel) / RATE_SCALE,
        ]
    }

    /// Zero for a task that does not walk. A walk counts travel toward the side that
    /// faced left at spawn, and only the crab's turning since tells it where that is.
    fn walk(&self, obs: &Observation) -> Vec3 {
        self.local(obs, obs.task.walk_sign() * obs.spawn_left)
    }

    fn clock(obs: &Observation) -> [f32; 2] {
        let phase = std::f32::consts::TAU * CLOCK_HZ * obs.tick as f32 * PHYSICS_DT;
        [phase.sin(), phase.cos()]
    }

    fn height(obs: &Observation) -> f32 {
        obs.bones[0].pose.translation.y / HEIGHT_SCALE
    }

    /// Every appendage's view, concatenated in appendage order. `action` is the one
    /// last sent to the servos, in hinge order.
    fn views(&self, obs: &Observation, action: &[f32]) -> Vec<f32> {
        let (unit, rate) = self.joints(obs);
        let [gravity, vel, angvel] = self.carapace(obs);
        let walk = self.walk(obs);
        let clock = Self::clock(obs);
        let mut out = Vec::with_capacity(self.appendages.len() * self.view_size());
        for a in &self.appendages {
            let (vector, pseudo) = if a.mirrored {
                (Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, -1.0, -1.0))
            } else {
                (Vec3::ONE, Vec3::ONE)
            };
            out.extend(a.hinges.iter().map(|&h| unit[h]));
            out.extend(a.hinges.iter().map(|&h| rate[h]));
            out.extend(a.hinges.iter().map(|&h| action[h]));
            out.extend((gravity * vector).to_array());
            out.extend((vel * vector).to_array());
            out.extend((angvel * pseudo).to_array());
            out.push(Self::height(obs));
            let walk = walk * vector;
            out.extend(walk.to_array());
            let stride = stride(walk.x, clock);
            out.extend(
                (0..self.kinds).flat_map(|k| stride.map(|x| if k == a.kind { x } else { 0.0 })),
            );
        }
        out
    }

    /// `targets` are appendage-ordered, as the readout emits them; `action` is in
    /// hinge order.
    fn follow(&self, targets: &[f32], action: &mut [f32]) {
        let n = self.appendage_hinges();
        for (a, t) in self.appendages.iter().zip(targets.chunks_exact(n)) {
            for (&h, &t) in a.hinges.iter().zip(t) {
                action[h] += CATCH_UP * (t - action[h]);
            }
        }
    }
}

/// The stride an appendage is asked for when the walk points `side` of the way to its
/// own side: a constant, then a centre and a swing (the clock's sine and cosine) for
/// as far as the appendage leads, and another centre and swing for as far as it trails.
fn stride(side: f32, [sin, cos]: [f32; 2]) -> [f32; STRIDE_INPUTS] {
    let (lead, trail) = (side.max(0.0), (-side).max(0.0));
    [
        1.0,
        lead,
        lead * sin,
        lead * cos,
        trail,
        trail * sin,
        trail * cos,
    ]
}

/// An [`Actor`] driving a rig: each tick the action moves toward the targets the
/// readout gives every appendage.
#[derive(Clone)]
pub struct Driver {
    anatomy: Anatomy,
    actor: Actor,
}

impl Driver {
    pub fn new(skeleton: &Skeleton, actor: Actor) -> Result<Self, String> {
        let anatomy = Anatomy::new(skeleton)?;
        actor.check(&anatomy)?;
        Ok(Self { anatomy, actor })
    }
}

impl Policy for Driver {
    fn act(&mut self, obs: &Observation, action: &mut [f32]) {
        let targets: Vec<f32> = self
            .anatomy
            .views(obs, action)
            .chunks_exact(self.actor.inputs)
            .flat_map(|view| self.actor.targets(view))
            .collect();
        self.anatomy.follow(&targets, action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Env, Task};
    use crate::sally::test_rig::biped;

    /// The biped's left (`.L`) and right (`.R`) appendage indices.
    fn sides(anatomy: &Anatomy) -> (usize, usize) {
        let side = |m: bool| {
            anatomy
                .appendages
                .iter()
                .position(|a| a.mirrored == m)
                .unwrap()
        };
        (side(false), side(true))
    }

    /// A biped walking left, tilted, moving and bent unevenly.
    fn uneven(skeleton: &Skeleton) -> Observation {
        let mut obs = Env::new(skeleton, Task::WalkLeft).unwrap().observe();
        obs.tick = 37;
        for (i, bone) in obs.bones.iter_mut().enumerate() {
            let k = i as f32 + 1.0;
            bone.pose.rotation = Quat::from_euler(EulerRot::YXZ, 0.3 * k, -0.2 * k, 0.1 * k);
            bone.pose.translation += Vec3::new(0.01 * k, 0.002 * k, -0.003 * k);
            bone.linvel = Vec3::new(0.05 * k, -0.02, 0.03 * k);
            bone.angvel = Vec3::new(1.5 * k, -0.7 * k, 0.4);
        }
        obs.hinges = vec![0.3, -0.2, 0.5, -0.4];
        obs
    }

    /// `obs` reflected through the biped's sagittal plane (x = 0, its rest left being
    /// +x), its sides swapped.
    fn mirror(obs: &Observation, skeleton: &Skeleton) -> Observation {
        let swap = |i: usize| {
            let name = &skeleton.bones[i].name;
            let other = name
                .replace(".L", ".tmp")
                .replace(".R", ".L")
                .replace(".tmp", ".R");
            skeleton.bones.iter().position(|b| b.name == other).unwrap()
        };
        let reflect = |v: Vec3| Vec3::new(-v.x, v.y, v.z);
        let pseudo = |v: Vec3| Vec3::new(v.x, -v.y, -v.z);
        let mut out = obs.clone();
        out.task = Task::WalkRight;
        for (i, bone) in out.bones.iter_mut().enumerate() {
            let from = obs.bones[swap(i)];
            let q = from.pose.rotation;
            bone.pose.rotation = Quat::from_xyzw(q.x, -q.y, -q.z, q.w);
            bone.pose.translation = reflect(from.pose.translation);
            bone.linvel = reflect(from.linvel);
            bone.angvel = pseudo(from.angvel);
        }
        let hinge_bones = skeleton.hinge_bones();
        out.hinges = hinge_bones
            .iter()
            .map(|&b| obs.hinges[hinge_bones.iter().position(|&o| o == swap(b)).unwrap()])
            .collect();
        out
    }

    #[test]
    fn a_right_appendage_sees_the_left_one_s_mirror_image() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let obs = uneven(&skeleton);
        let mirrored = mirror(&obs, &skeleton);
        let (a, b) = (
            anatomy.views(&obs, &obs.hinges),
            anatomy.views(&mirrored, &mirrored.hinges),
        );
        let n = anatomy.view_size();
        let view = |v: &[f32], k: usize| v[k * n..(k + 1) * n].to_vec();
        let (left, right) = sides(&anatomy);
        for (x, y) in [(left, right), (right, left)] {
            let (x, y) = (view(&a, x), view(&b, y));
            assert!(
                x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 1e-5),
                "{x:?} vs {y:?}"
            );
        }
        assert!(
            view(&a, left)
                .iter()
                .zip(view(&a, right))
                .any(|(p, q)| (p - q).abs() > 0.1)
        );
    }

    #[test]
    fn views_read_the_carapace_in_its_own_frame() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let mut obs = Env::new(&skeleton, Task::Stand).unwrap().observe();
        obs.bones[0].pose.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        obs.bones[0].linvel = Vec3::new(0.0, 0.0, -SPEED_SCALE);
        let (left, _) = sides(&anatomy);
        let view = &anatomy.views(&obs, &[0.0; 4])[left * anatomy.view_size()..];
        let n = anatomy.appendage_hinges();
        let gravity = &view[3 * n..3 * n + 3];
        let velocity = &view[3 * n + 3..3 * n + 6];
        assert!(
            gravity
                .iter()
                .zip([0.0, -1.0, 0.0])
                .all(|(g, w)| (g - w).abs() < 1e-6),
            "{gravity:?}"
        );
        assert!(
            velocity
                .iter()
                .zip([1.0, 0.0, 0.0])
                .all(|(v, w)| (v - w).abs() < 1e-6),
            "{velocity:?}"
        );
    }

    #[test]
    fn each_appendage_sees_the_action_last_sent_to_its_own_hinges() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let obs = Env::new(&skeleton, Task::Stand).unwrap().observe();
        let sent = [0.1, 0.2, 0.3, 0.4];
        let views = anatomy.views(&obs, &sent);
        let n = anatomy.appendage_hinges();
        for (k, a) in anatomy.appendages.iter().enumerate() {
            let at = k * anatomy.view_size() + 2 * n;
            let want: Vec<f32> = a.hinges.iter().map(|&h| sent[h]).collect();
            assert_eq!(views[at..at + n], want);
        }
    }

    #[test]
    fn the_walk_is_seen_where_it_points_from_the_carapace() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let (left, right) = sides(&anatomy);
        let n = anatomy.appendage_hinges();
        let walk = |task: Task, yaw: f32, side: usize| {
            let mut obs = Env::new(&skeleton, task).unwrap().observe();
            obs.bones[0].pose.rotation = Quat::from_rotation_y(yaw) * obs.bones[0].pose.rotation;
            let at = side * anatomy.view_size() + 3 * n + CARAPACE_INPUTS;
            anatomy.views(&obs, &[0.0; 4])[at..at + WALK_INPUTS].to_vec()
        };
        let near = |got: Vec<f32>, want: [f32; 3]| {
            assert!(
                got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-5),
                "{got:?} vs {want:?}"
            );
        };
        near(walk(Task::WalkLeft, 0.0, left), [1.0, 0.0, 0.0]);
        near(walk(Task::WalkLeft, 0.0, right), [-1.0, 0.0, 0.0]);
        near(walk(Task::WalkRight, 0.0, right), [1.0, 0.0, 0.0]);
        near(walk(Task::Stand, 0.0, left), [0.0, 0.0, 0.0]);
        let turned = walk(Task::WalkLeft, std::f32::consts::FRAC_PI_2, left);
        assert!(
            turned[0].abs() < 1e-5 && turned[2].abs() > 0.99,
            "{turned:?}"
        );
        near(
            walk(Task::WalkLeft, std::f32::consts::FRAC_PI_2, right),
            [0.0, 0.0, turned[2]],
        );
    }

    #[test]
    fn an_appendage_reads_the_stride_through_its_own_kind_and_the_walk_s_side() {
        let (s, c) = (0.6, -0.8);
        assert_eq!(
            stride(0.5, [s, c]),
            [1.0, 0.5, 0.5 * s, 0.5 * c, 0.0, 0.0, 0.0]
        );
        assert_eq!(stride(-1.0, [s, c]), [1.0, 0.0, 0.0, 0.0, 1.0, s, c]);
        let standing = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(stride(0.0, [s, c]), standing);

        let skeleton = biped();
        let mut anatomy = Anatomy::new(&skeleton).unwrap();
        let (left, right) = sides(&anatomy);
        let strides = |anatomy: &Anatomy, task: Task, side: usize| {
            let mut obs = Env::new(&skeleton, task).unwrap().observe();
            obs.tick = 9;
            let end = (side + 1) * anatomy.view_size();
            let views = anatomy.views(&obs, &[0.0; 4]);
            let kinds = anatomy.kinds * STRIDE_INPUTS;
            (views[end - kinds..end].to_vec(), Anatomy::clock(&obs))
        };
        let (leads, [s, c]) = strides(&anatomy, Task::WalkLeft, left);
        let trails = [1.0, 0.0, 0.0, 0.0, 1.0, s, c];
        assert_eq!(leads, [1.0, 1.0, s, c, 0.0, 0.0, 0.0]);
        assert_eq!(strides(&anatomy, Task::WalkLeft, right).0, trails);
        assert_eq!(strides(&anatomy, Task::WalkRight, right).0, leads);
        assert_eq!(strides(&anatomy, Task::Stand, left).0, standing);
        assert_eq!(strides(&anatomy, Task::Righting, right).0, standing);

        anatomy.kinds = 3;
        anatomy.appendages[right].kind = 2;
        let off = [0.0; STRIDE_INPUTS];
        let first_kind = [leads.as_slice(), &off, &off].concat();
        let third_kind = [off.as_slice(), &off, &trails].concat();
        assert_eq!(strides(&anatomy, Task::WalkLeft, left).0, first_kind);
        assert_eq!(strides(&anatomy, Task::WalkLeft, right).0, third_kind);
    }

    #[test]
    fn a_walk_half_to_a_side_asks_for_half_the_stride() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let (left, right) = sides(&anatomy);
        let mut obs = Env::new(&skeleton, Task::WalkLeft).unwrap().observe();
        let turn = Quat::from_rotation_y(std::f32::consts::FRAC_PI_3);
        obs.bones[0].pose.rotation = turn * obs.bones[0].pose.rotation;
        let views = anatomy.views(&obs, &[0.0; 4]);
        let stride = |side: usize| {
            &views[(side + 1) * anatomy.view_size() - STRIDE_INPUTS..][..STRIDE_INPUTS]
        };
        let (leads, trails) = (stride(left), stride(right));
        assert!(
            (leads[1] - 0.5).abs() < 1e-5 && leads[4] == 0.0,
            "{leads:?}"
        );
        assert!(
            (trails[4] - 0.5).abs() < 1e-5 && trails[1] == 0.0,
            "{trails:?}"
        );
    }

    #[test]
    fn the_clock_turns_once_a_stride() {
        let mut obs = Env::new(&biped(), Task::Stand).unwrap().observe();
        assert_eq!(Anatomy::clock(&obs), [0.0, 1.0]);
        obs.tick = (0.25 / (CLOCK_HZ * PHYSICS_DT)).round() as usize;
        let [sin, cos] = Anatomy::clock(&obs);
        assert!(sin > 0.99 && cos.abs() < 0.1, "{sin} {cos}");
    }

    #[test]
    fn a_stride_is_zero_on_every_task_that_does_not_walk_and_read_on_those_that_do() {
        let skeleton = biped();
        let mut anatomy = Anatomy::new(&skeleton).unwrap();
        anatomy.kinds = 2;
        anatomy.appendages[1].kind = 1;
        let strides = anatomy.strides();
        assert_eq!(strides.len(), anatomy.view_size());
        assert_eq!(
            strides.iter().filter(|&&s| s).count(),
            2 * (STRIDE_INPUTS - 1)
        );
        let action = [0.3, -0.2, 0.1, 0.4];
        let mut seen = vec![false; strides.len()];
        for task in Task::ALL {
            let mut obs = uneven(&skeleton);
            obs.task = task;
            let walks = task.walk_sign() != 0.0;
            for view in anatomy.views(&obs, &action).chunks_exact(strides.len()) {
                for (i, (&x, &stride)) in view.iter().zip(&strides).enumerate() {
                    assert!(walks || !stride || x == 0.0, "{task:?}: input {i} is {x}");
                    seen[i] |= walks && stride && x != 0.0;
                }
            }
        }
        assert_eq!(seen, strides);
    }

    #[test]
    fn the_action_trails_its_targets() {
        let anatomy = Anatomy::new(&biped()).unwrap();
        let (first, second) = (&anatomy.appendages[0], &anatomy.appendages[1]);
        let targets = [1.0, -1.0, 0.0, 0.5];
        let mut action = vec![0.0; 4];
        anatomy.follow(&targets, &mut action);
        assert_eq!(action[first.hinges[0]], CATCH_UP);
        assert_eq!(action[first.hinges[1]], -CATCH_UP);
        anatomy.follow(&targets, &mut action);
        assert_eq!(
            action[first.hinges[0]],
            CATCH_UP + CATCH_UP * (1.0 - CATCH_UP)
        );
        for _ in 0..200 {
            anatomy.follow(&targets, &mut action);
        }
        assert!((action[first.hinges[0]] - 1.0).abs() < 1e-4);
        assert_eq!(action[second.hinges[0]], 0.0);
        assert!((action[second.hinges[1]] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn the_driver_moves_the_action_it_is_handed_toward_what_the_readout_makes_of_it() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let mut actor = Actor::new(&anatomy);
        for (i, w) in actor.weights.iter_mut().enumerate() {
            *w = 0.3 * (i as f32).sin();
        }
        let obs = Env::new(&skeleton, Task::WalkLeft).unwrap().observe();
        let after = |handed: &[f32]| {
            let targets: Vec<f32> = anatomy
                .views(&obs, handed)
                .chunks_exact(anatomy.view_size())
                .flat_map(|view| actor.targets(view))
                .collect();
            let mut action = handed.to_vec();
            anatomy.follow(&targets, &mut action);
            action
        };
        let mut driver = Driver::new(&skeleton, actor.clone()).unwrap();
        let mut action = vec![0.0; 4];
        driver.act(&obs, &mut action);
        let first = action.clone();
        assert_eq!(first, after(&[0.0; 4]));
        assert!(first.iter().any(|a| a.abs() > 0.01), "{first:?}");
        driver.act(&obs, &mut action);
        assert_eq!(action, after(&first));
        assert_ne!(action, first);
    }

    #[test]
    fn hinge_rates_follow_the_child_s_spin_about_its_axis() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let mut obs = uneven(&skeleton);
        let h = 0;
        let (b, p) = (anatomy.hinge_bones[h], anatomy.parents[h]);
        let axis = obs.bones[b].pose.rotation * anatomy.hinges[h].axis;
        obs.bones[b].angvel = obs.bones[p].angvel + 2.5 * axis;
        let (_, rate) = anatomy.joints(&obs);
        assert!((rate[h] - 2.5 / RATE_SCALE).abs() < 1e-5, "{}", rate[h]);
    }

    #[test]
    fn the_biped_is_two_one_leg_appendages_of_one_kind() {
        let anatomy = Anatomy::new(&biped()).unwrap();
        assert_eq!(
            (
                anatomy.appendages.len(),
                anatomy.appendage_hinges(),
                anatomy.kinds
            ),
            (2, 2, 1)
        );
        let mut action = vec![0.0; 4];
        anatomy.follow(&[0.1, 0.2, 0.3, 0.4], &mut action);
        let first = &anatomy.appendages[0];
        assert_eq!(action[first.hinges[0]], 0.1 * CATCH_UP);
        assert_eq!(action[first.hinges[1]], 0.2 * CATCH_UP);
        let mut seen = action.clone();
        seen.sort_by(f32::total_cmp);
        assert_eq!(seen, [0.1, 0.2, 0.3, 0.4].map(|t| t * CATCH_UP));
    }

    #[test]
    fn the_first_actor_holds_the_rest_pose() {
        let skeleton = biped();
        let actor = Actor::new(&Anatomy::new(&skeleton).unwrap());
        assert!(actor.weights.iter().all(|&w| w == 0.0));
        let mut driver = Driver::new(&skeleton, actor).unwrap();
        let mut action = vec![0.0; 4];
        driver.act(&uneven(&skeleton), &mut action);
        assert_eq!(action, [0.0; 4]);
    }

    #[test]
    fn a_policy_for_another_rig_is_refused() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let actor = Actor::new(&anatomy);
        assert!(Driver::new(&skeleton, actor.clone()).is_ok());
        let mut truncated = actor.clone();
        truncated.weights.pop();
        assert!(Driver::new(&skeleton, truncated).is_err());
        let mut wider = actor.clone();
        wider.inputs += 1;
        assert!(Driver::new(&skeleton, wider).is_err());
        let mut longer = actor.clone();
        longer.weights.extend(vec![0.0; longer.inputs]);
        assert!(Driver::new(&skeleton, longer).is_err());
    }

    #[test]
    fn a_target_is_tanh_of_its_row_s_weighted_view() {
        let actor = Actor {
            inputs: 2,
            weights: vec![1.0, 0.5, -2.0, 0.25],
        };
        let got: Vec<f32> = actor.targets(&[0.3, -0.2]).collect();
        let want = [(0.3f32 - 0.1).tanh(), (-0.6f32 - 0.05).tanh()];
        assert!(
            got.len() == 2 && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6),
            "{got:?}"
        );
    }
}
