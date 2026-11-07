//! The crab evals `docs/evals.md` designs: a policy drives Sally's hinge servos on
//! flat ground, headless and deterministic, and each task reduces the run to one score.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::dynamics::MultibodyJointHandle;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use serde::Serialize;

use crate::physics::{PHYSICS_DT, PHYSICS_GRAVITY, PHYSICS_HZ};
use crate::sally::{Hinge, Skeleton};

/// Lowest point of the crab at spawn, above the ground.
const DROP_M: f32 = 0.01;
/// A belly closer to the ground than this is resting on it: rapier's contacts
/// settle a hull a fraction of a millimetre into the ground.
const BELLY_CLEARANCE_M: f32 = 0.002;
const UPRIGHT_TILT_MAX: f32 = std::f32::consts::FRAC_PI_4;
/// Servo error at which a joint reaches its torque cap, radians.
const SERVO_SATURATION_RAD: f32 = 0.1;
/// Servo damping over stiffness, seconds.
const SERVO_DAMPING_S: f32 = 0.01;
const STAND_SCORED_S: f32 = 6.0;
const RIGHTED_HOLD_S: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Stand,
    WalkLeft,
    WalkRight,
    Righting,
}

impl Task {
    pub const ALL: [Task; 4] = [Task::Stand, Task::WalkLeft, Task::WalkRight, Task::Righting];

    pub fn seconds(self) -> f32 {
        match self {
            Task::Stand => 8.0,
            Task::WalkLeft | Task::WalkRight | Task::Righting => 10.0,
        }
    }

    /// Which way along [`Observation::spawn_left`] the task walks: +1, −1, or 0 for a
    /// task that does not walk.
    pub fn walk_sign(self) -> f32 {
        match self {
            Task::WalkLeft => 1.0,
            Task::WalkRight => -1.0,
            Task::Stand | Task::Righting => 0.0,
        }
    }

    pub fn ticks(self) -> usize {
        ticks(self.seconds())
    }
}

fn ticks(seconds: f32) -> usize {
    (seconds * PHYSICS_HZ as f32).round() as usize
}

#[derive(Debug, Clone, Copy)]
pub struct BoneState {
    pub pose: Transform,
    pub linvel: Vec3,
    pub angvel: Vec3,
}

/// What the harness shows a policy each tick. `bones` is in [`Skeleton`] order;
/// `hinges[i]` is the angle from rest, radians, of the hinge on bone
/// [`Skeleton::hinge_bones`]`[i]`.
#[derive(Debug, Clone)]
pub struct Observation {
    pub task: Task,
    pub tick: usize,
    pub bones: Vec<BoneState>,
    pub hinges: Vec<f32>,
    /// World-space horizontal unit vector the crab's left side faced at spawn; the
    /// walks are measured along it.
    pub spawn_left: Vec3,
}

/// `action[i]` drives the servo of hinge `i`, in [`Observation::hinges`] order: 0
/// holds the rest angle, +1 and −1 its upper and lower limits, linearly between.
/// The harness clamps to [−1, 1].
pub trait Policy {
    fn act(&mut self, obs: &Observation, action: &mut [f32]);
}

pub struct ZeroPolicy;

impl Policy for ZeroPolicy {
    fn act(&mut self, _: &Observation, action: &mut [f32]) {
        action.fill(0.0);
    }
}

/// Each servo target drawn uniform in [−1, 1], independently every tick.
pub struct RandomPolicy(ChaCha8Rng);

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self(ChaCha8Rng::seed_from_u64(seed))
    }
}

impl Policy for RandomPolicy {
    fn act(&mut self, _: &Observation, action: &mut [f32]) {
        for a in action {
            *a = (self.0.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
        }
    }
}

/// What one tick did, in the quantities the task metrics are built from.
#[derive(Debug, Clone, Copy)]
pub struct Tick {
    pub supported: bool,
    /// Carapace displacement this tick along [`Observation::spawn_left`], meters.
    pub left_m: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Episode {
    pub task: Task,
    /// The task's metric, per `docs/evals.md`; 0 for a run whose physics diverged.
    pub score: f32,
    pub finite: bool,
    pub supported_fraction: f32,
    pub mean_body_height_m: f32,
}

/// `skeleton`'s horizontal lateral axis in model space, from its `.R` bones to its
/// `.L` bones.
pub fn model_left(skeleton: &Skeleton) -> Result<Vec3, String> {
    let side = |suffix: &str| {
        let at: Vec<Vec3> = skeleton
            .bones
            .iter()
            .filter(|b| b.name.ends_with(suffix))
            .map(|b| b.rest.translation)
            .collect();
        (!at.is_empty()).then(|| at.iter().sum::<Vec3>() / at.len() as f32)
    };
    let (Some(l), Some(r)) = (side(".L"), side(".R")) else {
        return Err("the rig needs bones on both sides, named *.L and *.R".into());
    };
    (l - r)
        .with_y(0.0)
        .try_normalize()
        .ok_or_else(|| "the rig's left and right bones do not differ horizontally".into())
}

/// Every servo's torque cap, N·m: the crab's weight at a lever of its carapace width.
pub fn joint_torque_max(skeleton: &Skeleton) -> Result<f32, String> {
    let mass: f32 = skeleton
        .bones
        .iter()
        .map(|b| b.collider().raw.mass_properties(b.density).mass())
        .sum();
    let left = model_left(skeleton)?;
    let body = &skeleton.bones[0];
    let (lo, hi) = body
        .hull
        .iter()
        .map(|p| body.rest.transform_point(*p).dot(left))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), x| {
            (lo.min(x), hi.max(x))
        });
    Ok(mass * -PHYSICS_GRAVITY.y * (hi - lo))
}

/// Where `task` spawns the crab, its lowest point `DROP_M` above the ground, and the
/// world direction its left side faces there.
fn spawn_frame(skeleton: &Skeleton, task: Task) -> Result<(Transform, Vec3), String> {
    let left = model_left(skeleton)?;
    let flip = match task {
        Task::Righting => Quat::from_axis_angle(Vec3::Y.cross(left), std::f32::consts::PI),
        _ => Quat::IDENTITY,
    };
    let lowest = skeleton.lowest_point(&skeleton.rest_poses(Transform::from_rotation(flip)));
    let at = Transform::from_xyz(0.0, DROP_M - lowest, 0.0).with_rotation(flip);
    Ok((at, flip * left))
}

/// One episode of a task, stepped by its caller: [`Env::observe`], act, [`Env::step`]
/// until [`Env::done`], then [`Env::finish`].
pub struct Env<'a> {
    skeleton: &'a Skeleton,
    task: Task,
    app: App,
    entities: Vec<Entity>,
    hinge_bones: Vec<usize>,
    hinge_joints: Vec<MultibodyJointHandle>,
    spawn_left: Vec3,
    stiffness: f32,
    damping: f32,
    tick: usize,
    finite: bool,
    supported_ticks: usize,
    height_sum: f32,
    walked: f32,
    stood_late: usize,
    supported_run: usize,
    righted_at: Option<usize>,
}

impl<'a> Env<'a> {
    pub fn new(skeleton: &'a Skeleton, task: Task) -> Result<Self, String> {
        let (at, spawn_left) = spawn_frame(skeleton, task)?;
        let (mut app, entities) = skeleton.world(at);
        for &e in &entities {
            // A sleeping body ignores servo targets until something hits it.
            app.world_mut()
                .entity_mut(e)
                .insert((Velocity::zero(), Sleeping::disabled()));
        }
        // bevy_rapier creates the bodies on the first update and their joints on the second.
        app.update();
        app.update();

        let hinge_bones = skeleton.hinge_bones();
        let hinge_joints = hinge_bones
            .iter()
            .map(|&i| {
                app.world()
                    .get::<RapierMultibodyJointHandle>(entities[i])
                    .map(|h| h.0)
                    .ok_or_else(|| format!("bone {} has no joint", skeleton.bones[i].name))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let torque_max = joint_torque_max(skeleton)?;
        let stiffness = torque_max / SERVO_SATURATION_RAD;
        let mut env = Self {
            skeleton,
            task,
            app,
            entities,
            hinge_bones,
            hinge_joints,
            spawn_left,
            stiffness,
            damping: stiffness * SERVO_DAMPING_S,
            tick: 0,
            finite: true,
            supported_ticks: 0,
            height_sum: 0.0,
            walked: 0.0,
            stood_late: 0,
            supported_run: 0,
            righted_at: None,
        };
        env.for_each_servo(|data, _| {
            data.set_motor_model(JointAxis::AngX, MotorModel::ForceBased)
                .set_motor_max_force(JointAxis::AngX, torque_max);
        });
        Ok(env)
    }

    pub fn done(&self) -> bool {
        !self.finite || self.tick >= self.task.ticks()
    }

    fn poses(&self) -> Vec<Transform> {
        self.entities
            .iter()
            .map(|&e| {
                *self
                    .app
                    .world()
                    .get::<Transform>(e)
                    .expect("bone has a transform")
            })
            .collect()
    }

    pub fn observe(&self) -> Observation {
        let world = self.app.world();
        let bones = self
            .entities
            .iter()
            .map(|&e| {
                let v = world.get::<Velocity>(e).expect("bone has a velocity");
                BoneState {
                    pose: *world.get::<Transform>(e).expect("bone has a transform"),
                    linvel: v.linear,
                    angvel: v.angular,
                }
            })
            .collect::<Vec<_>>();
        let poses: Vec<Transform> = bones.iter().map(|b| b.pose).collect();
        Observation {
            task: self.task,
            tick: self.tick,
            bones,
            hinges: self
                .skeleton
                .hinge_angles(&poses)
                .into_iter()
                .flatten()
                .collect(),
            spawn_left: self.spawn_left,
        }
    }

    fn for_each_servo(
        &mut self,
        mut f: impl FnMut(&mut bevy_rapier3d::rapier::dynamics::GenericJoint, usize),
    ) {
        let world = self.app.world_mut();
        let mut joints = world.query::<&mut RapierContextJoints>();
        let mut joints = joints.single_mut(world).expect("one rapier context");
        for (i, &handle) in self.hinge_joints.iter().enumerate() {
            let (multibody, link) = joints
                .multibody_joints
                .get_mut(handle)
                .expect("hinge is in the multibody set");
            f(
                &mut multibody
                    .link_mut(link)
                    .expect("hinge link exists")
                    .joint
                    .data,
                i,
            );
        }
    }

    /// Sets every servo target from `action`, advances one tick and scores it.
    pub fn step(&mut self, action: &[f32]) -> Tick {
        assert_eq!(action.len(), self.hinge_bones.len(), "one action per hinge");
        assert!(!self.done(), "step after the episode ended");
        let (stiffness, damping) = (self.stiffness, self.damping);
        let hinges: Vec<Hinge> = self
            .hinge_bones
            .iter()
            .map(|&i| self.skeleton.bones[i].hinge.expect("hinge bone"))
            .collect();
        self.for_each_servo(|data, i| {
            let target = hinges[i].target(action[i].clamp(-1.0, 1.0));
            data.set_motor_position(JointAxis::AngX, target, stiffness, damping);
        });
        let before = self.poses()[0].translation;
        self.app.update();
        self.tick += 1;
        self.tally(before)
    }

    fn tally(&mut self, before: Vec3) -> Tick {
        let poses = self.poses();
        self.finite = poses
            .iter()
            .all(|t| t.translation.is_finite() && t.rotation.is_finite());
        if !self.finite {
            return Tick {
                supported: false,
                left_m: 0.0,
            };
        }
        let body = poses[0];
        let up = (body.rotation * self.skeleton.bones[0].rest.rotation.inverse()) * Vec3::Y;
        let belly = self.skeleton.bones[0]
            .hull
            .iter()
            .map(|p| body.transform_point(*p).y)
            .fold(f32::INFINITY, f32::min);
        let supported = up.angle_between(Vec3::Y) < UPRIGHT_TILT_MAX && belly > BELLY_CLEARANCE_M;
        self.supported_ticks += supported as usize;
        self.height_sum += body.translation.y;
        let left_m = (body.translation - before).dot(self.spawn_left);
        if supported {
            self.walked += left_m;
            if self.tick > self.task.ticks() - ticks(STAND_SCORED_S) {
                self.stood_late += 1;
            }
        }
        self.supported_run = if supported { self.supported_run + 1 } else { 0 };
        if self.righted_at.is_none() && self.supported_run >= ticks(RIGHTED_HOLD_S) {
            self.righted_at = Some(self.tick + 1 - self.supported_run);
        }
        Tick { supported, left_m }
    }

    pub fn finish(self) -> Episode {
        let ticks_total = self.task.ticks();
        let seconds = ticks_total as f32 * PHYSICS_DT;
        let score = if !self.finite {
            0.0
        } else {
            match self.task {
                Task::Stand => self.stood_late as f32 / ticks(STAND_SCORED_S) as f32,
                Task::WalkLeft | Task::WalkRight => self.task.walk_sign() * self.walked / seconds,
                Task::Righting => self
                    .righted_at
                    .map_or(0.0, |t| 1.0 - t as f32 / ticks_total as f32),
            }
        };
        let run = self.tick.max(1) as f32;
        Episode {
            task: self.task,
            score,
            finite: self.finite,
            supported_fraction: self.supported_ticks as f32 / run,
            mean_body_height_m: self.height_sum / run,
        }
    }
}

/// Runs one episode of `task` under `policy` and scores it.
pub fn run(skeleton: &Skeleton, task: Task, policy: &mut dyn Policy) -> Result<Episode, String> {
    let mut env = Env::new(skeleton, task)?;
    let mut action = vec![0.0; skeleton.hinge_bones().len()];
    while !env.done() {
        policy.act(&env.observe(), &mut action);
        env.step(&action);
    }
    Ok(env.finish())
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskScore {
    pub task: Task,
    pub episodes: Vec<Episode>,
    pub mean: f32,
}

/// Scores a policy family on every task: `make(seed)` builds a fresh policy per
/// episode, one episode per seed. Episodes run on parallel threads, each in its own
/// world, so the scores do not depend on the thread count.
pub fn score(
    skeleton: &Skeleton,
    seeds: &[u64],
    make: &(dyn Fn(u64) -> Box<dyn Policy> + Sync),
) -> Result<Vec<TaskScore>, String> {
    if seeds.is_empty() {
        return Err("no seeds to score".into());
    }
    let jobs: Vec<(Task, u64)> = Task::ALL
        .iter()
        .flat_map(|&task| seeds.iter().map(move |&seed| (task, seed)))
        .collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(jobs.len());
    let jobs = &jobs;
    let episodes: Vec<Episode> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|w| {
                scope.spawn(move || {
                    jobs.iter()
                        .enumerate()
                        .skip(w)
                        .step_by(threads)
                        .map(|(i, &(task, seed))| {
                            Ok((i, run(skeleton, task, make(seed).as_mut())?))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
            })
            .collect();
        let mut done = Vec::with_capacity(jobs.len());
        for worker in workers {
            done.extend(worker.join().expect("episode worker panicked")?);
        }
        done.sort_by_key(|&(i, _)| i);
        Ok::<_, String>(done.into_iter().map(|(_, e)| e).collect())
    })?;
    Ok(Task::ALL
        .iter()
        .zip(episodes.chunks(seeds.len()))
        .map(|(&task, episodes)| TaskScore {
            task,
            episodes: episodes.to_vec(),
            mean: episodes.iter().map(|e| e.score).sum::<f32>() / episodes.len() as f32,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sally::test_rig::biped as rig;

    /// Holds `first` on every servo for `seconds`, then `then`, to the episode's end.
    fn hold(task: Task, first: f32, seconds: f32, then: f32) -> (Episode, Vec<f32>) {
        let skeleton = rig();
        let mut env = Env::new(&skeleton, task).unwrap();
        while !env.done() {
            let a = if env.tick < ticks(seconds) {
                first
            } else {
                then
            };
            env.step(&[a; 4]);
        }
        let hinges = env.observe().hinges;
        (env.finish(), hinges)
    }

    #[test]
    fn the_crab_spawns_its_drop_height_above_the_ground() {
        let skeleton = rig();
        for task in [Task::Stand, Task::Righting] {
            let (start, _) = spawn_frame(&skeleton, task).unwrap();
            let lowest = skeleton.lowest_point(&skeleton.rest_poses(start));
            assert!((lowest - DROP_M).abs() < 1e-6, "{task:?}: {lowest}");
        }
    }

    #[test]
    fn left_points_from_the_right_legs_to_the_left() {
        assert!(model_left(&rig()).unwrap().abs_diff_eq(Vec3::X, 1e-6));
    }

    #[test]
    fn a_rig_without_sides_is_refused() {
        let mut skeleton = rig();
        for bone in &mut skeleton.bones {
            bone.name = bone.name.replace(".R", ".M");
        }
        assert!(Env::new(&skeleton, Task::Stand).is_err());
    }

    #[test]
    fn an_episode_replays_exactly() {
        let skeleton = rig();
        let a = run(&skeleton, Task::WalkLeft, &mut RandomPolicy::new(3)).unwrap();
        let b = run(&skeleton, Task::WalkLeft, &mut RandomPolicy::new(3)).unwrap();
        assert!(a.finite);
        assert_eq!(a.score.to_bits(), b.score.to_bits());
        assert_eq!(
            a.mean_body_height_m.to_bits(),
            b.mean_body_height_m.to_bits()
        );
    }

    #[test]
    fn parallel_scoring_matches_serial_episodes() {
        let skeleton = rig();
        let scores = score(&skeleton, &[1, 2], &|seed| {
            Box::new(RandomPolicy::new(seed))
        })
        .unwrap();
        for (task, scored) in Task::ALL.iter().zip(&scores) {
            assert_eq!(scored.task, *task);
            for (seed, episode) in [1, 2].iter().zip(&scored.episodes) {
                let serial = run(&skeleton, *task, &mut RandomPolicy::new(*seed)).unwrap();
                assert_eq!(episode.score.to_bits(), serial.score.to_bits());
            }
        }
    }

    #[test]
    fn servos_drive_hinges_toward_the_action() {
        for (a, want) in [(0.5f32, 30f32), (-0.5, -15.0)] {
            let hinges = hold(Task::Righting, a, 0.0, a).1;
            assert_eq!(hinges.len(), 4);
            for angle in hinges {
                assert!(
                    (angle.to_degrees() - want).abs() < 6.0,
                    "action {a}: {}°",
                    angle.to_degrees()
                );
            }
        }
    }

    #[test]
    fn actions_past_the_limits_are_clamped() {
        assert_eq!(
            hold(Task::Righting, 1.0, 0.0, 1.0).1,
            hold(Task::Righting, 5.0, 0.0, 5.0).1
        );
    }

    #[test]
    fn standing_counts_only_its_last_six_seconds() {
        let (held, _) = hold(Task::Stand, 0.0, 0.0, 0.0);
        assert_eq!(held.score, 1.0, "{held:?}");
        let (lifted, _) = hold(Task::Stand, 1.0, 8.0, 1.0);
        assert_eq!(lifted.score, 0.0, "{lifted:?}");
        let (late, _) = hold(Task::Stand, 1.0, 2.0, 0.0);
        assert!(late.supported_fraction < 0.9, "{late:?}");
        assert!(late.score > 0.9, "{late:?}");
    }

    #[test]
    fn an_overturned_crab_that_never_rights_scores_zero() {
        let episode = run(&rig(), Task::Righting, &mut ZeroPolicy).unwrap();
        assert!(episode.finite);
        assert_eq!(episode.score, 0.0);
    }

    #[test]
    fn walking_is_measured_toward_the_named_side() {
        let skeleton = rig();
        let left = run(&skeleton, Task::WalkLeft, &mut RandomPolicy::new(7)).unwrap();
        let right = run(&skeleton, Task::WalkRight, &mut RandomPolicy::new(7)).unwrap();
        assert_ne!(left.score, 0.0);
        assert_eq!(left.score, -right.score);
    }
}
