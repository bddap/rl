//! Random external shoves during training (rl#298 stage 4): in the one-world endgame
//! anything can push Sally mid-chase — a craft ram, a falling body, terrain contact —
//! and a policy trained unshoved meets that as out-of-distribution input. Registered
//! only by the rollout worker (`wire_rollout_training`): eval measures the policy, so
//! it stays unshoved.
//!
//! Scale is anchored to measured world contacts, not invented: the rl#298 stage-1 ram
//! (a production craft at 8 m/s) measured ~0.2 m of carapace knockback (its pin
//! asserts >0.02 m), and 70 N over the same 8-tick burst visibly displaces a settled
//! crab (`external_force_shoves_a_multibody_root`). The force band brackets that
//! datum on both sides so the policy sees nudges through hits harder than a ram, at
//! ~2 shoves per max-length episode.

use bevy::prelude::*;
use bevy_rapier3d::prelude::ExternalForce;
use rand::Rng;

use super::TrainingState;
use super::lifecycle::EnvPhase;
use crate::bot::body::{CrabCarapace, CrabEnvId};

/// One burst = 8 physics ticks (0.125 s), the stage-1 ram-pin duration.
const SHOVE_TICKS: u32 = 8;
const SHOVE_FORCE_MIN_N: f32 = 40.0;
const SHOVE_FORCE_MAX_N: f32 = 160.0;
/// Per-tick start probability while recording — one shove per ~10 s per env.
const SHOVE_START_PROB: f32 = 1.0 / 640.0;

/// A live burst on one env's carapace. Horizontal by design: world contacts push
/// laterally; lift/drop dynamics already arise from terrain and the crab's own body.
#[derive(Clone, Copy, Default)]
pub(crate) struct ShoveState {
    remaining: u32,
    force: Vec3,
}

/// Draw, apply, and age each env's shove. Ordered after [`crate::bot::BotSet::Act`]
/// (whose `apply_actions` zeroes `ExternalForce.force` every tick) and before the
/// rapier sync, exactly the seam the stage-1 shove test proves out. Draws come from
/// the training RNG in env order, so a seeded run reproduces its shove schedule.
pub(crate) fn shove_crabs(
    mut training: NonSendMut<TrainingState>,
    mut carapaces: Query<(&CrabEnvId, &mut ExternalForce), With<CrabCarapace>>,
) {
    let TrainingState { envs, rng, .. } = &mut *training;
    for ep in envs.iter_mut() {
        // Starts are gated on Recording; clearing needs no code here because episode
        // end wholesale-replaces the `EnvEpisode` (`finalize_transitions`), so a burst
        // cannot outlive its episode — the respawn settle always starts unshoved.
        if matches!(ep.phase, EnvPhase::Recording)
            && ep.shove.remaining == 0
            && rng.gen_range(0.0..1.0) < SHOVE_START_PROB
        {
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let newtons = rng.gen_range(SHOVE_FORCE_MIN_N..SHOVE_FORCE_MAX_N);
            ep.shove = ShoveState {
                remaining: SHOVE_TICKS,
                force: Vec3::new(angle.cos(), 0.0, angle.sin()) * newtons,
            };
        }
    }
    for (env, mut force) in carapaces.iter_mut() {
        if let Some(ep) = envs.get(env.0)
            && ep.shove.remaining > 0
        {
            force.force += ep.shove.force;
        }
    }
    for ep in envs.iter_mut() {
        ep.shove.remaining = ep.shove.remaining.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The apply path end-to-end in a real training world: a hand-armed burst on env 0
    /// must survive `apply_actions`'s per-tick force zeroing and visibly displace the
    /// settled crab, and the burst must age out. The env is parked in `Settling` so
    /// the random draw gate stays cold — the assertions depend on the armed burst
    /// alone, not on the seeded RNG stream. (Draws are covered by
    /// `same_seed_reproduces_the_rollout_trajectory`, whose harness registers this
    /// system.)
    #[test]
    fn armed_shove_displaces_the_crab_and_expires() {
        use bevy_rapier3d::plugin::PhysicsSet;

        use crate::bot::BotSet;
        use crate::bot::headless::{flat_headless_app, tick};

        // Flat grid so displacement attributes to the shove, not the GCR origin
        // slope (the stage-1 shove test's finding). No brain/reset systems: the
        // shove seam only needs TrainingState's envs + rng beside the bot stack.
        let dir = std::env::temp_dir().join(format!("rl_test_shove_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = crate::TrainConfig::scratch(&dir, 1, 0x5140);
        let mut app = flat_headless_app();
        let mut state = TrainingState::new_worker(&config, 0, crate::bot::arch::ArchId::DEFAULT);
        // Park the env outside Recording for the whole test: the draw gate stays
        // cold, so no spontaneous burst can contaminate the measured displacement.
        state.envs[0].phase = EnvPhase::Settling { grace: u32::MAX };
        app.insert_non_send_resource(state);
        app.add_systems(
            FixedUpdate,
            shove_crabs
                .after(BotSet::Act)
                .before(PhysicsSet::SyncBackend),
        );
        // Settle fully before measuring so displacement attributes to the shove.
        tick(&mut app, 192);

        let carapace_xz = |app: &mut App| {
            let mut q = app
                .world_mut()
                .query_filtered::<&Transform, With<CrabCarapace>>();
            let t = q.single(app.world()).expect("carapace").translation;
            Vec2::new(t.x, t.z)
        };
        let p0 = carapace_xz(&mut app);

        {
            let mut st = app
                .world_mut()
                .get_non_send_resource_mut::<TrainingState>()
                .expect("training state");
            st.envs[0].shove = ShoveState {
                remaining: SHOVE_TICKS,
                force: Vec3::new(120.0, 0.0, 0.0),
            };
        }
        tick(&mut app, SHOVE_TICKS + 16);

        let moved = (carapace_xz(&mut app) - p0).length();
        assert!(
            moved > 0.05,
            "a 120 N / {SHOVE_TICKS}-tick shove must visibly move the crab, moved {moved:.3} m"
        );
        let st = app
            .world()
            .get_non_send_resource::<TrainingState>()
            .expect("training state");
        assert_eq!(st.envs[0].shove.remaining, 0, "the burst must age out");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
