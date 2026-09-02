//! Random external shoves during rollouts: in the one-world endgame anything can push
//! Sally mid-chase — a craft ram, a falling body, terrain contact — and a policy
//! trained unshoved meets that as out-of-distribution input. Registered only by the
//! rollout worker (`wire_rollout_training`); eval measures the policy unshoved.
//! `TrainConfig::shove_prob` (default 0) turns them on.

use bevy::prelude::*;
use bevy_rapier3d::prelude::ExternalForce;
use rand::Rng;

use super::lifecycle::EnvPhase;
use super::state::WorkerState;
use crate::bot::body::{CrabBodyMass, CrabCarapace, CrabEnvId};
use crate::physics::{PHYSICS_DT, PHYSICS_GRAVITY};

/// One burst = 8 physics ticks (0.125 s), the rl#298 stage-1 ram-pin duration.
const SHOVE_TICKS: u32 = 8;
/// Burst magnitude as a fraction of body weight, so the scale rides the rig's baked
/// mass: a fixed newton band was 2–8× the weight of the ~2 kg rig and launched her
/// ~2 m per burst, a recession the progress term then charged (rl#415).
const SHOVE_WEIGHT_FRAC: std::ops::Range<f32> = 0.1..0.4;
/// Hard cap on the velocity one burst can impart, whatever the rig weighs.
const SHOVE_DV_CAP_MPS: f32 = 0.5;

/// A live burst on one env's carapace. Horizontal by design: world contacts push
/// laterally; lift/drop dynamics already arise from terrain and the crab's own body.
#[derive(Clone, Copy, Default)]
pub(crate) struct ShoveState {
    remaining: u32,
    direction: Vec3,
    weight_frac: f32,
}

impl ShoveState {
    fn armed(direction: Vec3, weight_frac: f32) -> Self {
        Self {
            remaining: SHOVE_TICKS,
            direction,
            weight_frac,
        }
    }

    fn force(&self, body_mass: f32) -> Vec3 {
        let weight = body_mass * -PHYSICS_GRAVITY.y;
        let impulse_cap = SHOVE_DV_CAP_MPS * body_mass / (SHOVE_TICKS as f32 * PHYSICS_DT);
        self.direction * (self.weight_frac * weight).min(impulse_cap)
    }
}

/// Draw, apply, and age each env's shove. Ordered after [`crate::bot::BotSet::Act`]
/// (whose `apply_actions` zeroes `ExternalForce.force` every tick) and before the
/// rapier sync. Draws come from the training RNG in env order, so a seeded run
/// reproduces its shove schedule.
pub(crate) fn shove_crabs(
    mut training: NonSendMut<WorkerState>,
    mut carapaces: Query<(&CrabEnvId, &CrabBodyMass, &mut ExternalForce), With<CrabCarapace>>,
) {
    let WorkerState { rng, mode, .. } = &mut *training;
    let envs = &mut mode.envs;
    // Episode end wholesale-replaces the `EnvEpisode` (`finalize_transitions`), so a
    // burst cannot outlive its episode — the respawn settle always starts unshoved.
    for ep in envs.iter_mut() {
        if matches!(ep.phase, EnvPhase::Recording)
            && ep.shove.remaining == 0
            && rng.gen_range(0.0..1.0) < mode.shove_prob
        {
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            ep.shove = ShoveState::armed(
                Vec3::new(angle.cos(), 0.0, angle.sin()),
                rng.gen_range(SHOVE_WEIGHT_FRAC),
            );
        }
    }
    for (env, mass, mut force) in carapaces.iter_mut() {
        if let Some(ep) = envs.get(env.0)
            && ep.shove.remaining > 0
        {
            force.force += ep.shove.force(mass.0);
        }
    }
    for ep in envs.iter_mut() {
        ep.shove.remaining = ep.shove.remaining.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The apply path end-to-end in a real training world at the production scale: a
    /// hand-armed maximum burst on env 0 must survive `apply_actions`'s per-tick force
    /// zeroing and nudge the settled crab — but never launch or tip her (rl#415) —
    /// and the burst must age out. The env is parked in `Settling` so the random draw
    /// gate stays cold: the assertions depend on the armed burst alone.
    #[test]
    fn max_shove_nudges_the_crab_without_launching_or_tipping() {
        use bevy_rapier3d::plugin::PhysicsSet;

        use crate::bot::BotSet;
        use crate::bot::headless::{flat_headless_app, tick};

        let dir = std::env::temp_dir().join(format!("rl_test_shove_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = crate::TrainConfig::scratch(&dir, 1, 0x5140);
        let mut app = flat_headless_app();
        let mut state = WorkerState::new_worker(&config, 0, crate::bot::arch::ArchId::DEFAULT);
        state.mode.envs[0].phase = EnvPhase::Settling { grace: u32::MAX };
        app.insert_non_send(state);
        app.add_systems(
            FixedUpdate,
            shove_crabs
                .after(BotSet::Act)
                .before(PhysicsSet::SyncBackend),
        );
        tick(&mut app, 192);

        let pose = |app: &mut App| {
            let mut q = app
                .world_mut()
                .query_filtered::<(&Transform, &CrabBodyMass), With<CrabCarapace>>();
            let (t, m) = q.single(app.world()).expect("carapace");
            (
                Vec2::new(t.translation.x, t.translation.z),
                (t.rotation * Vec3::Y).y,
                m.0,
            )
        };
        let (p0, up0, mass) = pose(&mut app);
        let burst = ShoveState::armed(Vec3::X, SHOVE_WEIGHT_FRAC.end);
        let dv = burst.force(mass).length() * SHOVE_TICKS as f32 * PHYSICS_DT / mass;
        assert!(
            dv <= SHOVE_DV_CAP_MPS,
            "a max burst on the {mass:.2} kg rig imparts {dv:.2} m/s, over the cap"
        );
        app.world_mut()
            .get_non_send_mut::<WorkerState>()
            .expect("training state")
            .mode
            .envs[0]
            .shove = burst;

        let mut min_up = up0;
        let mut max_moved = 0.0f32;
        for _ in 0..SHOVE_TICKS + 256 {
            tick(&mut app, 1);
            let (p, up, _) = pose(&mut app);
            min_up = min_up.min(up);
            max_moved = max_moved.max((p - p0).length());
        }
        assert!(
            max_moved > 0.01,
            "a max burst must still nudge the crab, moved {max_moved:.3} m"
        );
        assert!(
            max_moved < 0.3,
            "a max burst must not launch the crab, moved {max_moved:.3} m"
        );
        assert!(
            min_up > up0 - 0.05,
            "a max burst must not tip the crab: up·Y fell {up0:.3} -> {min_up:.3}"
        );
        let st = app
            .world()
            .get_non_send::<WorkerState>()
            .expect("training state");
        assert_eq!(st.mode.envs[0].shove.remaining, 0, "the burst must age out");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
