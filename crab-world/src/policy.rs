//! The crab policy `docs/policy.md` designs: one network shared by every appendage,
//! fed that appendage's mirrored view of the crab, emitting its servo targets.

use bevy::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::eval::{Observation, Policy, Task, model_left};
use crate::physics::PHYSICS_DT;
use crate::sally::{Hinge, Skeleton};

/// *Grapsus* stride frequency at its routine pace (Martinez et al. 1998, Table 1).
pub const CLOCK_HZ: f32 = 1.17;
const RATE_SCALE: f32 = 10.0;
const SPEED_SCALE: f32 = 0.1;
const HEIGHT_SCALE: f32 = 0.05;
const HIDDEN: usize = 64;
const CARAPACE_INPUTS: usize = 10;
const INITIAL_STD: f32 = 0.2;
const LAST_LAYER_SCALE: f32 = 0.01;

/// A fully connected network, tanh after every hidden layer and optionally after the
/// last. `params` holds each layer's row-major weights, then its biases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mlp {
    pub sizes: Vec<usize>,
    pub tanh_out: bool,
    pub params: Vec<f32>,
}

/// Every layer's input, and the network's output, from one forward pass.
pub struct Trace(Vec<Vec<f32>>);

impl Trace {
    pub fn output(&self) -> &[f32] {
        self.0.last().expect("a trace holds the output")
    }
}

impl Mlp {
    pub fn new(sizes: &[usize], tanh_out: bool, rng: &mut ChaCha8Rng) -> Self {
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
            tanh_out,
            params,
        }
    }

    /// Whether `params` holds exactly the layers `sizes` names.
    fn well_formed(&self) -> bool {
        self.sizes.len() >= 2
            && self.params.len()
                == self
                    .sizes
                    .windows(2)
                    .map(|w| w[0] * w[1] + w[1])
                    .sum::<usize>()
    }

    fn layers(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.sizes.windows(2).scan(0, |at, w| {
            let start = *at;
            *at += w[0] * w[1] + w[1];
            Some((start, w[0], w[1]))
        })
    }

    pub fn forward(&self, x: &[f32]) -> Trace {
        assert_eq!(x.len(), self.sizes[0], "network input size");
        let last = self.sizes.len() - 2;
        let mut trace = vec![x.to_vec()];
        for (l, (at, n_in, n_out)) in self.layers().enumerate() {
            let input = trace.last().expect("layer input");
            let (w, b) = self.params[at..at + n_in * n_out + n_out].split_at(n_in * n_out);
            let out = (0..n_out)
                .map(|o| {
                    let z = b[o]
                        + w[o * n_in..(o + 1) * n_in]
                            .iter()
                            .zip(input)
                            .map(|(w, x)| w * x)
                            .sum::<f32>();
                    if l < last || self.tanh_out {
                        z.tanh()
                    } else {
                        z
                    }
                })
                .collect();
            trace.push(out);
        }
        Trace(trace)
    }

    /// Adds into `grad`, shaped like `params`, the gradient of a loss whose gradient
    /// with respect to the output of `trace` is `d_out`.
    pub fn backward(&self, trace: &Trace, d_out: &[f32], grad: &mut [f32]) {
        let layers: Vec<_> = self.layers().collect();
        let last = layers.len() - 1;
        let mut d = d_out.to_vec();
        for (l, &(at, n_in, n_out)) in layers.iter().enumerate().rev() {
            if l < last || self.tanh_out {
                for (d, y) in d.iter_mut().zip(&trace.0[l + 1]) {
                    *d *= 1.0 - y * y;
                }
            }
            let input = &trace.0[l];
            let w = &self.params[at..at + n_in * n_out];
            let (gw, gb) = grad[at..at + n_in * n_out + n_out].split_at_mut(n_in * n_out);
            let mut d_in = vec![0.0; n_in];
            for o in 0..n_out {
                gb[o] += d[o];
                let row = o * n_in;
                for i in 0..n_in {
                    gw[row + i] += d[o] * input[i];
                    d_in[i] += d[o] * w[row + i];
                }
            }
            d = d_in;
        }
    }
}

/// Uniform in [0, 1).
pub fn uniform(rng: &mut ChaCha8Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    let u = 1.0 - uniform(rng);
    let v = uniform(rng);
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
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

    pub fn appendages(&self) -> usize {
        self.appendages.len()
    }

    pub fn appendage_hinges(&self) -> usize {
        self.appendages[0].hinges.len()
    }

    pub fn hinges(&self) -> usize {
        self.hinge_bones.len()
    }

    pub fn view_size(&self) -> usize {
        2 * self.appendage_hinges() + CARAPACE_INPUTS + 1 + 2 + self.kinds
    }

    pub fn critic_size(&self) -> usize {
        2 * self.hinges() + CARAPACE_INPUTS + Task::ALL.len() + 2 + 1
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

    /// Gravity's direction, velocity and angular velocity in the carapace's left, up,
    /// forward frame, and its height, scaled.
    fn carapace(&self, obs: &Observation) -> [Vec3; 3] {
        let body = &obs.bones[0];
        let frame = (body.pose.rotation * self.carapace_rest.inverse()).inverse();
        let local = |v: Vec3| {
            let m = frame * v;
            Vec3::new(
                m.dot(self.basis[0]),
                m.dot(self.basis[1]),
                m.dot(self.basis[2]),
            )
        };
        [
            local(Vec3::NEG_Y),
            local(body.linvel) / SPEED_SCALE,
            local(body.angvel) / RATE_SCALE,
        ]
    }

    fn clock(obs: &Observation) -> [f32; 2] {
        let phase = std::f32::consts::TAU * CLOCK_HZ * obs.tick as f32 * PHYSICS_DT;
        [phase.sin(), phase.cos()]
    }

    fn height(obs: &Observation) -> f32 {
        obs.bones[0].pose.translation.y / HEIGHT_SCALE
    }

    /// Every appendage's view, concatenated in appendage order.
    pub fn views(&self, obs: &Observation) -> Vec<f32> {
        let (unit, rate) = self.joints(obs);
        let [gravity, vel, angvel] = self.carapace(obs);
        let command = obs.task.walk_sign();
        let clock = Self::clock(obs);
        let mut out = Vec::with_capacity(self.appendages() * self.view_size());
        for a in &self.appendages {
            let (flip, vector, pseudo) = if a.mirrored {
                (-1.0, Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, -1.0, -1.0))
            } else {
                (1.0, Vec3::ONE, Vec3::ONE)
            };
            out.extend(a.hinges.iter().map(|&h| unit[h]));
            out.extend(a.hinges.iter().map(|&h| rate[h]));
            out.extend((gravity * vector).to_array());
            out.extend((vel * vector).to_array());
            out.extend((angvel * pseudo).to_array());
            out.push(Self::height(obs));
            out.push(command * flip);
            out.extend(clock);
            out.extend((0..self.kinds).map(|k| (k == a.kind) as u8 as f32));
        }
        out
    }

    pub fn critic_input(&self, obs: &Observation) -> Vec<f32> {
        let (unit, rate) = self.joints(obs);
        let mut out = Vec::with_capacity(self.critic_size());
        out.extend(unit);
        out.extend(rate);
        for v in self.carapace(obs) {
            out.extend(v.to_array());
        }
        out.push(Self::height(obs));
        out.extend(Task::ALL.iter().map(|&t| (t == obs.task) as u8 as f32));
        out.extend(Self::clock(obs));
        out.push(obs.tick as f32 / obs.task.ticks() as f32);
        out
    }

    /// Writes appendage-ordered `targets` into `action`, in hinge order.
    pub fn scatter(&self, targets: &[f32], action: &mut [f32]) {
        let n = self.appendage_hinges();
        for (a, t) in self.appendages.iter().zip(targets.chunks_exact(n)) {
            for (&h, &t) in a.hinges.iter().zip(t) {
                action[h] = t;
            }
        }
    }
}

/// The shared appendage network: the mean of a diagonal Gaussian over one
/// appendage's servo targets, with one log standard deviation per hinge position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actor {
    pub net: Mlp,
    pub log_std: Vec<f32>,
}

impl Actor {
    pub fn new(anatomy: &Anatomy, rng: &mut ChaCha8Rng) -> Self {
        Self {
            net: Mlp::new(
                &[
                    anatomy.view_size(),
                    HIDDEN,
                    HIDDEN,
                    anatomy.appendage_hinges(),
                ],
                true,
                rng,
            ),
            log_std: vec![INITIAL_STD.ln(); anatomy.appendage_hinges()],
        }
    }

    pub fn check(&self, anatomy: &Anatomy) -> Result<(), String> {
        if !self.net.well_formed() || !self.net.tanh_out {
            return Err("the policy's network is malformed".into());
        }
        let (want_in, want_out) = (anatomy.view_size(), anatomy.appendage_hinges());
        let (got_in, got_out) = (self.net.sizes[0], *self.net.sizes.last().unwrap());
        if (got_in, got_out, self.log_std.len()) != (want_in, want_out, want_out) {
            return Err(format!(
                "the policy maps {got_in} inputs to {got_out} targets; this rig needs {want_in} to {want_out}"
            ));
        }
        Ok(())
    }

    /// Every appendage's mean targets, appendage-ordered.
    pub fn means(&self, views: &[f32]) -> Vec<f32> {
        views
            .chunks_exact(self.net.sizes[0])
            .flat_map(|v| self.net.forward(v).output().to_vec())
            .collect()
    }

    /// Targets drawn from the Gaussian around `means`, as training explores.
    pub fn sample(&self, means: &[f32], rng: &mut ChaCha8Rng) -> Vec<f32> {
        let n = self.log_std.len();
        means
            .iter()
            .enumerate()
            .map(|(i, m)| m + self.log_std[i % n].exp() * gaussian(rng))
            .collect()
    }
}

pub fn critic(anatomy: &Anatomy, rng: &mut ChaCha8Rng) -> Mlp {
    Mlp::new(&[anatomy.critic_size(), HIDDEN, HIDDEN, 1], false, rng)
}

/// A trained [`Actor`] driving the crab with its mean action, or, given a noise
/// stream, with actions sampled around it as in training.
#[derive(Clone)]
pub struct Trained {
    pub anatomy: Anatomy,
    pub actor: Actor,
    pub noise: Option<ChaCha8Rng>,
}

impl Trained {
    pub fn new(skeleton: &Skeleton, actor: Actor) -> Result<Self, String> {
        let anatomy = Anatomy::new(skeleton)?;
        actor.check(&anatomy)?;
        Ok(Self {
            anatomy,
            actor,
            noise: None,
        })
    }

    pub fn sampled(self, seed: u64) -> Self {
        Self {
            noise: Some(ChaCha8Rng::seed_from_u64(seed)),
            ..self
        }
    }
}

impl Policy for Trained {
    fn act(&mut self, obs: &Observation, action: &mut [f32]) {
        let means = self.actor.means(&self.anatomy.views(obs));
        let targets = match &mut self.noise {
            Some(rng) => self.actor.sample(&means, rng),
            None => means,
        };
        self.anatomy.scatter(&targets, action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Env;
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
        let (a, b) = (anatomy.views(&obs), anatomy.views(&mirror(&obs, &skeleton)));
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
        let view = &anatomy.views(&obs)[left * anatomy.view_size()..];
        let n = anatomy.appendage_hinges();
        let gravity = &view[2 * n..2 * n + 3];
        let velocity = &view[2 * n + 3..2 * n + 6];
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
                anatomy.appendages(),
                anatomy.appendage_hinges(),
                anatomy.kinds
            ),
            (2, 2, 1)
        );
        let mut action = vec![0.0; 4];
        anatomy.scatter(&[1.0, 2.0, 3.0, 4.0], &mut action);
        let first = &anatomy.appendages[0];
        assert_eq!(action[first.hinges[0]], 1.0);
        assert_eq!(action[first.hinges[1]], 2.0);
        let mut seen = action.clone();
        seen.sort_by(f32::total_cmp);
        assert_eq!(seen, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn backward_accumulates() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let net = Mlp::new(&[2, 3, 1], false, &mut rng);
        let trace = net.forward(&[0.4, -0.1]);
        let mut once = vec![0.0; net.params.len()];
        net.backward(&trace, &[1.0], &mut once);
        let mut twice = vec![0.0; net.params.len()];
        net.backward(&trace, &[1.0], &mut twice);
        net.backward(&trace, &[1.0], &mut twice);
        assert!(
            once.iter()
                .zip(&twice)
                .all(|(a, b)| (2.0 * a - b).abs() < 1e-7)
        );
    }

    #[test]
    fn a_policy_for_another_rig_is_refused() {
        let skeleton = biped();
        let mut actor = Actor::new(
            &Anatomy::new(&skeleton).unwrap(),
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        assert!(Trained::new(&skeleton, actor.clone()).is_ok());
        let mut longer = actor.clone();
        longer.log_std.push(0.0);
        assert!(Trained::new(&skeleton, longer).is_err());
        let mut truncated = actor.clone();
        truncated.net.params.pop();
        assert!(Trained::new(&skeleton, truncated).is_err());
        actor.net.tanh_out = false;
        assert!(Trained::new(&skeleton, actor).is_err());
    }

    #[test]
    fn backward_matches_finite_differences() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for tanh_out in [false, true] {
            let mut net = Mlp::new(&[3, 5, 4, 2], tanh_out, &mut rng);
            for p in &mut net.params {
                *p = uniform(&mut rng) - 0.5;
            }
            let x = [0.3, -0.7, 0.2];
            let loss = |net: &Mlp| {
                let y = net.forward(&x);
                y.output()[0] * 2.0 - y.output()[1]
            };
            let mut grad = vec![0.0; net.params.len()];
            net.backward(&net.forward(&x), &[2.0, -1.0], &mut grad);
            for (i, &g) in grad.iter().enumerate() {
                let mut up = net.clone();
                up.params[i] += 1e-3;
                let mut down = net.clone();
                down.params[i] -= 1e-3;
                let numeric = (loss(&up) - loss(&down)) / 2e-3;
                assert!((numeric - g).abs() < 2e-3, "param {i}: {numeric} vs {g}");
            }
        }
    }
}
