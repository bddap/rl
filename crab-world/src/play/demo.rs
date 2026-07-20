use bevy::app::AppExit;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use rand::Rng;

use crate::bot::actuator::CrabActions;
use crate::bot::body::{self, CrabAssets, CrabBodyPart, CrabCarapace};
use crate::bot::sensor::CrabTargets;
use crate::bot::{CrabSpawns, RESET_GRACE_TICKS, respawn_crab_rotated, settle_countdown};
use crate::training::targets::{random_episode_origin, seed_target};

use crate::controls::just_pressed;

use super::controls::{DemoAction, DemoControls};

#[derive(Resource, Default)]
pub(super) struct DemoSettle(pub(super) u32);

#[derive(Resource, Default)]
pub(super) struct PokeBurst {
    ticks: u32,
    force: Vec3,
    torque: Vec3,
}

const POKE_TICKS: u32 = 8;
const POKE_FORCE: f32 = 70.0;
const POKE_TORQUE: f32 = 4.0;

pub(super) fn demo_poke(
    mut burst: ResMut<PokeBurst>,
    mut carapace_q: Query<&mut ExternalForce, With<CrabCarapace>>,
) {
    if burst.ticks == 0 {
        return;
    }
    burst.ticks -= 1;
    let Ok(mut f) = carapace_q.single_mut() else {
        return;
    };
    f.force += burst.force;
    f.torque += burst.torque;
}

/// Reset re-rolls the LOCALE (rl#300): Sally and the ball land at a fresh random
/// spot on the tile — the same per-episode draw training resets use — and a bad
/// roll is handled by pressing reset again, not by slope-aware spawn selection.
#[allow(clippy::too_many_arguments)]
fn demo_respawn(
    commands: &mut Commands,
    assets: &CrabAssets,
    spawns: &mut CrabSpawns,
    targets: &mut CrabTargets,
    terrain: &crate::terrain::TerrainGrid,
    parts: impl Iterator<Item = Entity>,
    settle: &mut DemoSettle,
    actions: &mut CrabActions,
    rng: &mut rand::rngs::StdRng,
) {
    let origin = random_episode_origin(rng, terrain);
    spawns.set_origin(0, origin);
    // The held target was banded around the OLD locale — re-seed from the new
    // origin (same argument as `reset_crab`'s re-seed).
    seed_target(targets, spawns, 0, rng, terrain);
    let init_rotation = body::random_spawn_rotation(rng);
    respawn_crab_rotated(commands, assets, terrain, parts, origin, 0, init_rotation);
    settle.0 = RESET_GRACE_TICKS;
    let _ = actions.rest(0); // deliberate skip pre-spawn
}

#[allow(clippy::too_many_arguments)]
pub(super) fn demo_controls(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
    assets: Res<CrabAssets>,
    mut spawns: ResMut<CrabSpawns>,
    mut targets: ResMut<CrabTargets>,
    terrain: Res<crate::terrain::Terrain>,
    parts_q: Query<Entity, With<CrabBodyPart>>,
    mut poke_burst: ResMut<PokeBurst>,
    mut actions: ResMut<CrabActions>,
    mut settle: ResMut<DemoSettle>,
    mut render_mode: ResMut<crate::crab_view::RenderMode>,
    mut rng: ResMut<super::DemoRng>,
    mut policy: NonSendMut<crate::policy::Policy>,
) {
    // The tap verbs dispatch from DEMO_BINDINGS, so they trigger on exactly the inputs the
    // legend shows. RenderView (→ / D-pad Right) CYCLES the render view (mesh →
    // mesh+colliders → colliders); the other arrow keys are the orbit camera — its
    // right-yaw moved to the comma key so → isn't double-bound (mouse right-drag still
    // orbits).
    let just = |a| just_pressed::<DemoControls>(a, &keys, &gamepads);
    let reset = just(DemoAction::Rebuild);
    let poke = just(DemoAction::Poke);
    let quit = just(DemoAction::Quit);
    let cycle_view = just(DemoAction::RenderView);

    if quit {
        exit.write(AppExit::Success);
    }
    if cycle_view {
        *render_mode = render_mode.next();
        info!("demo render mode: {:?}", *render_mode);
    }
    // The swapped label reaches the screen through the every-frame `publish_brain_label`;
    // a failed swap keeps the current brain (the one swap path already logs it loudly).
    if just(DemoAction::SwapBrain) {
        policy.cycle_brain();
    }
    if reset {
        demo_respawn(
            &mut commands,
            &assets,
            &mut spawns,
            &mut targets,
            &terrain,
            parts_q.iter(),
            &mut settle,
            &mut actions,
            &mut rng.0,
        );
    }
    if poke {
        let rng = &mut rng.0;
        let dir =
            Vec3::new(rng.gen_range(-1.0..1.0), 0.25, rng.gen_range(-1.0..1.0)).normalize_or_zero();
        *poke_burst = PokeBurst {
            ticks: POKE_TICKS,
            force: dir * POKE_FORCE,
            torque: Vec3::new(rng.gen_range(-1.0..1.0), 0.0, rng.gen_range(-1.0..1.0))
                * POKE_TORQUE,
        };
    }
}

pub(super) fn demo_settle(mut settle: ResMut<DemoSettle>, mut actions: ResMut<CrabActions>) {
    if settle.0 == 0 {
        return;
    }
    let _ = actions.rest(0); // deliberate skip pre-spawn
    settle.0 = settle_countdown(settle.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use rand::SeedableRng;

    use crate::bot::body::CrabEnvId;
    use crate::bot::headless::{flat_headless_app, tick};
    use crate::training::reward::planar_dist;
    use crate::training::targets::BAND_MAX_M;

    fn reset(app: &mut App) {
        app.world_mut()
            .run_system_once(
                |mut commands: Commands,
                 assets: Res<CrabAssets>,
                 mut spawns: ResMut<CrabSpawns>,
                 mut targets: ResMut<CrabTargets>,
                 terrain: Res<crate::terrain::Terrain>,
                 parts: Query<(Entity, &CrabEnvId), With<CrabBodyPart>>,
                 mut settle: ResMut<DemoSettle>,
                 mut actions: ResMut<CrabActions>,
                 mut rng: ResMut<super::super::DemoRng>| {
                    demo_respawn(
                        &mut commands,
                        &assets,
                        &mut spawns,
                        &mut targets,
                        &terrain,
                        parts.iter().filter(|(_, id)| id.0 == 0).map(|(e, _)| e),
                        &mut settle,
                        &mut actions,
                        &mut rng.0,
                    );
                },
            )
            .expect("demo reset system");
    }

    fn carapace(app: &mut App) -> Vec3 {
        let mut q = app
            .world_mut()
            .query_filtered::<&Transform, With<CrabCarapace>>();
        q.single(app.world()).expect("carapace").translation
    }

    /// rl#300: a reset re-rolls the locale — the origin leaves the boot spawn, the
    /// crab respawns ON the new origin, and the target re-seeds inside the trained
    /// band around it (the ball follows `CrabTargets`), press after press.
    #[test]
    fn reset_rerolls_crab_and_ball_locale() {
        let mut app = flat_headless_app();
        app.insert_resource(super::super::DemoRng(rand::rngs::StdRng::seed_from_u64(7)));
        app.init_resource::<DemoSettle>();
        tick(&mut app, 2);
        let boot = app.world().resource::<CrabSpawns>().origin(0);

        let mut origins = Vec::new();
        for press in 0..3 {
            reset(&mut app);
            tick(&mut app, 1);
            let origin = app.world().resource::<CrabSpawns>().origin(0);
            assert!(
                planar_dist(carapace(&mut app), origin) < 3.0,
                "press {press}: the crab must respawn on the re-rolled origin"
            );
            let target = app
                .world()
                .resource::<CrabTargets>()
                .get(0)
                .expect("press re-seeds the target");
            let d = planar_dist(target, origin);
            assert!(
                d <= BAND_MAX_M + 1e-3,
                "press {press}: target sits {d} m from the new origin — banded around \
                 the OLD locale"
            );
            origins.push(origin);
        }
        origins.push(boot);
        for (i, a) in origins.iter().enumerate() {
            for b in &origins[i + 1..] {
                assert!(
                    planar_dist(*a, *b) > 1e-3,
                    "each press draws a fresh locale (got {a:?} twice)"
                );
            }
        }
    }
}
