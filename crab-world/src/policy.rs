//! The crab policy `docs/policy.md` designs: one network shared by every appendage,
//! fed that appendage's mirrored view of the crab, emitting its servo targets.

use bevy::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::RngCore;
use serde::{Deserialize, Serialize};

use crate::eval::{Observation, Policy, model_left};
use crate::physics::PHYSICS_DT;
use crate::sally::{Hinge, Skeleton};

/// *Grapsus* stride frequency at its routine pace (Martinez et al. 1998, Table 1).
const CLOCK_HZ: f32 = 1.17;
const RATE_SCALE: f32 = 10.0;
const SPEED_SCALE: f32 = 0.1;
const HEIGHT_SCALE: f32 = 0.05;
const HIDDEN: usize = 64;
const CARAPACE_INPUTS: usize = 10;
const WALK_INPUTS: usize = 3;
/// The share of its distance to the network's targets the action covers each tick.
const CATCH_UP: f32 = 0.25;
const LAST_LAYER_SCALE: f32 = 0.01;

/// The shared appendage network, one appendage's view to its servo targets: fully
/// connected, tanh after every layer. `params` holds each layer's row-major weights,
/// then its biases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actor {
    pub sizes: Vec<usize>,
    pub params: Vec<f32>,
}

impl Actor {
    pub fn new(anatomy: &Anatomy, rng: &mut ChaCha8Rng) -> Self {
        Self::sized(
            &[
                anatomy.view_size(),
                HIDDEN,
                HIDDEN,
                anatomy.appendage_hinges(),
            ],
            rng,
        )
    }

    fn sized(sizes: &[usize], rng: &mut ChaCha8Rng) -> Self {
        let mut params = Vec::new();
        let layers = sizes.len() - 1;
        for (l, w) in sizes.windows(2).enumerate() {
            let (n_in, n_out) = (w[0], w[1]);
            let mut bound = (6.0 / (n_in + n_out) as f32).sqrt();
            if l + 1 == layers {
                bound *= LAST_LAYER_SCALE;
            }
            params.extend((0..n_in * n_out).map(|_| (uniform(rng) * 2.0 - 1.0) * bound));
            params.extend(std::iter::repeat_n(0.0, n_out));
        }
        Self {
            sizes: sizes.to_vec(),
            params,
        }
    }

    fn layers(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.sizes.windows(2).scan(0, |at, w| {
            let start = *at;
            *at += w[0] * w[1] + w[1];
            Some((start, w[0], w[1]))
        })
    }

    fn check(&self, anatomy: &Anatomy) -> Result<(), String> {
        let held: usize = self
            .layers()
            .map(|(_, n_in, n_out)| n_in * n_out + n_out)
            .sum();
        if self.sizes.len() < 2 || self.params.len() != held {
            return Err("the policy's network is malformed".into());
        }
        let (want_in, want_out) = (anatomy.view_size(), anatomy.appendage_hinges());
        let (got_in, got_out) = (self.sizes[0], *self.sizes.last().unwrap());
        if (got_in, got_out) != (want_in, want_out) {
            return Err(format!(
                "the policy maps {got_in} inputs to {got_out} targets; this rig needs {want_in} to {want_out}"
            ));
        }
        Ok(())
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.sizes[0], "network input size");
        let mut out = x.to_vec();
        for (at, n_in, n_out) in self.layers() {
            let (w, b) = self.params[at..at + n_in * n_out + n_out].split_at(n_in * n_out);
            out = (0..n_out)
                .map(|o| {
                    let z: f32 = w[o * n_in..(o + 1) * n_in]
                        .iter()
                        .zip(&out)
                        .map(|(w, x)| w * x)
                        .sum();
                    (b[o] + z).tanh()
                })
                .collect();
        }
        out
    }

    pub fn last_layer_mut(&mut self) -> &mut [f32] {
        let (at, _, _) = self.layers().last().expect("a network has a layer");
        &mut self.params[at..]
    }
}

/// Uniform in [0, 1).
pub fn uniform(rng: &mut ChaCha8Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
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
        3 * self.appendage_hinges() + CARAPACE_INPUTS + WALK_INPUTS + 2 + self.kinds
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
            out.extend((walk * vector).to_array());
            out.extend(clock);
            out.extend((0..self.kinds).map(|k| (k == a.kind) as u8 as f32));
        }
        out
    }

    /// `targets` are appendage-ordered, as the network emits them; `action` is in
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

/// An [`Actor`] driving a rig: each tick the action moves toward the targets the
/// network gives every appendage.
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
            .chunks_exact(self.actor.sizes[0])
            .flat_map(|view| self.actor.forward(view))
            .collect();
        self.anatomy.follow(&targets, action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Env, Task};
    use crate::sally::test_rig::biped;
    use rand_chacha::rand_core::SeedableRng;

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
    fn the_driver_moves_the_action_it_is_handed_toward_what_the_network_sees_from_it() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let mut actor = Actor::new(&anatomy, &mut ChaCha8Rng::seed_from_u64(4));
        for p in &mut actor.params {
            *p *= 100.0;
        }
        let obs = Env::new(&skeleton, Task::WalkLeft).unwrap().observe();
        let after = |handed: &[f32]| {
            let targets: Vec<f32> = anatomy
                .views(&obs, handed)
                .chunks_exact(anatomy.view_size())
                .flat_map(|view| actor.forward(view))
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
    fn a_policy_for_another_rig_is_refused() {
        let skeleton = biped();
        let anatomy = Anatomy::new(&skeleton).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let actor = Actor::new(&anatomy, &mut rng);
        assert!(Driver::new(&skeleton, actor.clone()).is_ok());
        let mut truncated = actor.clone();
        truncated.params.pop();
        assert!(Driver::new(&skeleton, truncated).is_err());
        let wider = Actor::sized(&[anatomy.view_size() + 1, 4, 2], &mut rng);
        assert!(Driver::new(&skeleton, wider).is_err());
        let longer = Actor::sized(&[anatomy.view_size(), 4, 3], &mut rng);
        assert!(Driver::new(&skeleton, longer).is_err());
    }

    #[test]
    fn a_network_is_tanh_of_its_weighted_inputs_layer_by_layer() {
        let mut net = Actor::sized(&[2, 2, 1], &mut ChaCha8Rng::seed_from_u64(1));
        net.params = vec![1.0, 0.0, 0.5, -0.5, 0.1, 0.0, 2.0, -1.0, 0.25];
        let hidden = [(0.3f32 + 0.1).tanh(), (0.15f32 + 0.1).tanh()];
        let want = (2.0 * hidden[0] - hidden[1] + 0.25).tanh();
        let got = net.forward(&[0.3, -0.2]);
        assert!(got.len() == 1 && (got[0] - want).abs() < 1e-6, "{got:?}");
        assert_eq!(net.last_layer_mut(), [2.0, -1.0, 0.25]);
    }
}
