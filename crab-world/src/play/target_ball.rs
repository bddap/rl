//! The demo's red target ball — the visible stand-in for the target the policy reaches
//! for. [`crate::bot::sensor::CrabTargets`] is the single source of truth; this module
//! seeds/relocates it and snaps the ball to it, so the ball can never disagree with the
//! target the policy perceives.

use bevy::prelude::*;

use crate::bot::CrabSpawns;
use crate::bot::body::{self, CrabClawTip};
use crate::bot::sensor::CrabTargets;
use crate::training::curriculum::sample_target;
use crate::training::reward::dist_3d;

/// Marker on the demo's red target ball — the visible stand-in for the target the
/// policy is reaching for. Demo-only: training renders nothing and reads the target
/// straight from [`CrabTargets`].
#[derive(Component)]
pub(super) struct TargetBall;

/// 3D euclidean distance at which the DEMO counts a claw tip as having reached the
/// target and teleports it to a fresh target (see [`target_ball`]). Set at the edge
/// of the claw's reach. Because the reach `d` is 3D, this 0.8 m is a SPHERE about the
/// target, not a ground-plane cylinder: a tip standing under a raised ball is no longer
/// "reached" until it is within 0.8 m in 3D. The training reward is dense planar PROGRESS
/// plus a SPARSE grab terminal gated at exactly this radius (`reward::GRAB_REWARD`), so the
/// same radius defines the "reached it" event in three places that share one definition: the
/// grab terminal there, the demo's ball-hop here, and the training per-episode reach signal.
/// DERIVED from that curriculum constant
/// ([`crate::training::curriculum::CURRICULUM_REACH_RADIUS`]) — one source, in the
/// always-compiled trainer — so the demo and curriculum can't drift apart on the
/// radius. (0.8 m, as the doc above describes.)
const DEMO_REACH_RADIUS: f32 = crate::training::curriculum::CURRICULUM_REACH_RADIUS;

/// Radius (m) of the demo target ball. Bigger than [`DEMO_REACH_RADIUS`] so the
/// claw visibly reaches *into* the ball before it registers a reach and jumps —
/// a marker you can see from the orbit camera, not a pinprick.
const TARGET_BALL_RADIUS: f32 = 0.08;

/// Startup (demo only): spawn the red target ball. Its world position is driven
/// every tick by [`target_ball`] off [`CrabTargets`] — the same state the policy
/// observes and the reward scores — so it is a pure marker, never a second source of
/// truth. The target itself is seeded by `target_ball` (in FixedUpdate, after the
/// Startup that sizes [`CrabTargets`]), so the ball starts at the origin and snaps to
/// its target on the first tick.
pub(super) fn spawn_target_ball(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(TARGET_BALL_RADIUS))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.05, 0.05),
            // Emissive so the ball reads as a bright, self-lit dot regardless of
            // where the scene lighting falls — unmistakable against the crab/ground.
            emissive: LinearRgba::new(1.6, 0.0, 0.0, 1.0),
            ..default()
        })),
        Transform::from_translation(Vec3::ZERO),
        TargetBall,
    ));
}

/// FixedUpdate (demo only): drive the red ball off env 0's target, and TELEPORT the
/// target to a fresh point when a claw tip reaches it. [`CrabTargets`] is the single
/// source of truth — the same resource the observation reads; here we seed/relocate it
/// and snap the ball to it, so the ball can never disagree with the target the policy
/// perceives. Seeding and relocation both reuse [`sample_target`], the exact near-heavy
/// full-range rule training samples, so a demo target is always the same kind of walk-to
/// goal the policy was trained on (mostly near, with the same far tail).
///
/// **Intentional train/demo divergence.** Training uses ONE fixed target per episode: the
/// sparse grab terminal ENDS the episode on reach, so there is no per-tick stream to farm by
/// hovering (and dense progress telescopes to zero on a closed wobble — see `reward`). The
/// DEMO instead teleports the ball to a fresh target on reach, purely for watchability: it
/// keeps the crab walking continuously to new goals instead of parking
/// on one. This is safe because the policy learned "walk toward the current target
/// vector" and so generalizes to a target that moves — the demo just exercises that on
/// a livelier schedule than training. Reached is the closest 3D euclidean claw-tip-to-target
/// distance within [`DEMO_REACH_RADIUS`]. (The demo runs no training target system, so
/// this is the only writer of env 0's target; the initial seed happens here rather than
/// at Startup because `BotPlugin`'s Startup resize of [`CrabTargets`] would otherwise
/// race and clear it.)
pub(super) fn target_ball(
    spawns: Res<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    claw_tips_q: Query<(&body::CrabEnvId, &Transform), With<CrabClawTip>>,
    mut ball_q: Query<&mut Transform, (With<TargetBall>, Without<CrabClawTip>)>,
    mut rng: ResMut<super::DemoRng>,
) {
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);

    // Seed on first tick (target unset) so the demo always has a reach to show. An
    // explicit `RL_TARGET_BALL_AT` (screenshot evidence frames) pins the seed to a
    // chosen point; otherwise sample the reach box. Seeding here, not at Startup,
    // dodges a race with `BotPlugin`'s Startup resize of `CrabTargets`.
    // The demo samples from the same FIXED full-arena band the trainer uses, with the same
    // near-heavy weighting — so the streamed demo shows the crab chasing the real target at
    // the same distances the weights train on (mostly near, with a far tail): one
    // `sample_target` rule, so the demo can never pose a target training never saw.
    let mut target = match targets.get(0) {
        Some(t) => t,
        None => target_ball_at_from_env().unwrap_or_else(|| sample_target(origin, &mut rng.0)),
    };

    // Closest 3D euclidean distance from either claw tip to the target (the reward's
    // `d`, env 0) — 3D so the ball relocates at the same reached-moment training does
    // (the reward and this test MUST share one `d`; see `reward::dist_3d`).
    let mut min_dist = f32::INFINITY;
    for (env, tip) in claw_tips_q.iter() {
        if env.0 == 0 && tip.translation.is_finite() {
            min_dist = min_dist.min(dist_3d(tip.translation, target));
        }
    }

    // Reached → relocate to a fresh target (demo watchability only; see the fn
    // doc on why training does NOT do this). The ball follows the new target below.
    if min_dist <= DEMO_REACH_RADIUS {
        target = sample_target(origin, &mut rng.0);
    }

    // Write the one source of truth (seed or relocation), then snap the ball to it.
    if let Some(slot) = targets.envs.first_mut() {
        *slot = Some(target);
    }
    if let Ok(mut ball) = ball_q.single_mut() {
        ball.translation = target;
    }
}

/// Read `RL_TARGET_BALL_AT="x,y,z"` into an explicit world target for the screenshot
/// ball, so an evidence frame can place the red ball at a chosen point in the arena
/// (and a second frame at a moved point) deterministically, instead of a random
/// sample. `None` (the default) lets [`target_ball`] sample a fresh target.
fn target_ball_at_from_env() -> Option<Vec3> {
    let raw = std::env::var("RL_TARGET_BALL_AT").ok()?;
    let parts: Vec<f32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    match parts.as_slice() {
        [x, y, z] => Some(Vec3::new(*x, *y, *z)),
        _ => None,
    }
}
