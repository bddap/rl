use bevy::app::AppExit;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use rand::Rng;

use crate::bot::actuator::CrabActions;
use crate::bot::body::{self, CrabAssets, CrabBodyPart, CrabCarapace};
use crate::bot::{CrabSpawns, RESET_GRACE_TICKS, respawn_crab_rotated, settle_countdown};

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

#[allow(clippy::too_many_arguments)]
fn demo_respawn(
    commands: &mut Commands,
    assets: &CrabAssets,
    spawns: &CrabSpawns,
    terrain: &crate::terrain::TerrainGrid,
    parts: impl Iterator<Item = Entity>,
    settle: &mut DemoSettle,
    actions: &mut CrabActions,
    init_rotation: Quat,
) {
    let origin = spawns.origin(0);
    respawn_crab_rotated(commands, assets, terrain, parts, origin, 0, init_rotation);
    settle.0 = RESET_GRACE_TICKS;
    let _ = actions.rest(0); // deliberate skip pre-spawn
}

fn random_demo_tilt(rng: &mut impl Rng) -> Quat {
    body::random_spawn_rotation(rng)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn demo_controls(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
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
            &spawns,
            &terrain,
            parts_q.iter(),
            &mut settle,
            &mut actions,
            random_demo_tilt(&mut rng.0),
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
