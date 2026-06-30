//! The demo crab's liveness loop: spawn/settle that holds the crab steady after a
//! respawn, the poke burst, and the keyboard/gamepad controls (respawn, poke, quit,
//! collider wireframes). Respawn is manual only — the A button (or R) re-tilts and
//! drops a fresh crab; there is no automatic timer or fall-triggered respawn.

use bevy::app::AppExit;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use rand::Rng;

use crate::bot::actuator::{ACTION_SIZE, CrabActions};
use crate::bot::body::{self, CrabAssets, CrabBodyPart, CrabCarapace};
use crate::bot::{CrabSpawns, respawn_crab_rotated};
use crate::training::systems::{RESET_GRACE_TICKS, settle_countdown};

/// Settle ticks remaining after a reset. The respawned crab starts in the
/// rest pose with the builder motors already holding it; the settle just
/// holds zero actions while it drops onto the ground and takes load. Seeded from
/// training's [`RESET_GRACE_TICKS`] and decremented via the shared
/// [`settle_countdown`] so the demo's drop window stays identical to the one the
/// policy was trained under (0 = settled, policy back in control).
#[derive(Resource, Default)]
pub(super) struct DemoSettle(pub(super) u32);

/// A poke is a short force burst, not a velocity write: a multibody link's
/// velocity lives in the multibody's generalized coordinates, which the
/// `Velocity` component writeback never touches (issue #14 — the old poke
/// was a silent no-op). Per-link external forces, by contrast, are mapped
/// through the body Jacobians into generalized accelerations, so force is
/// the one channel that actually reaches a multibody root. Rapier never
/// auto-clears user forces, hence the countdown that zeroes them.
#[derive(Resource, Default)]
pub(super) struct PokeBurst {
    ticks: u32,
    force: Vec3,
    torque: Vec3,
}

const POKE_TICKS: u32 = 8;
const POKE_FORCE: f32 = 70.0;
const POKE_TORQUE: f32 = 4.0;

/// System (FixedUpdate, after the actuator): adds the active poke burst on top
/// of the carapace's joint-reaction torques. The actuator overwrites every
/// link's `ExternalForce` each step, so the poke must run after it and *add*
/// rather than set — and it needs no cleanup, the actuator zeroes the baseline
/// next step.
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

/// Demo reset: rebuild the crab fresh at spawn — the only reset that
/// survives a corrupted multibody (see [`respawn_crab_rotated`]) — and hold zero
/// actions while it takes load. `init_rotation` is the spawn tilt: the demo feeds
/// a fresh [`body::random_spawn_rotation`] every time so the crab lands at a random
/// goofy angle and visibly tries to right itself, the "journey" the stream shows.
fn demo_respawn(
    commands: &mut Commands,
    assets: &CrabAssets,
    spawns: &CrabSpawns,
    parts: impl Iterator<Item = Entity>,
    settle: &mut DemoSettle,
    actions: &mut CrabActions,
    init_rotation: Quat,
) {
    let origin = spawns.0.first().copied().unwrap_or(Vec3::ZERO);
    respawn_crab_rotated(commands, assets, parts, origin, 0, init_rotation);
    settle.0 = RESET_GRACE_TICKS;
    if let Some(a) = actions.envs.first_mut() {
        *a = [0.0; ACTION_SIZE];
    }
}

/// A fresh random goofy spawn tilt for a demo respawn (see
/// [`body::random_spawn_rotation`]): mostly mild, sometimes fully inverted, random
/// yaw — so the demo crab keeps landing at new angles to right itself from.
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
    parts_q: Query<Entity, With<CrabBodyPart>>,
    mut poke_burst: ResMut<PokeBurst>,
    mut actions: ResMut<CrabActions>,
    mut settle: ResMut<DemoSettle>,
    // The SHARED render-mode cycle (the same `crab_view::RenderMode` GCR uses), so the demo and
    // GCR show the ONE collider wireframe. Replaces the old Rapier debug-render toggle.
    mut render_mode: ResMut<crate::crab_view::RenderMode>,
    // The demo's one seedable RNG — the respawn tilt and poke impulse draw from it (RL_DEMO_SEED
    // pins them), so the demo no longer reaches for an unseeded `thread_rng`.
    mut rng: ResMut<super::DemoRng>,
) {
    let mut reset = keys.just_pressed(KeyCode::KeyR);
    let mut poke = keys.just_pressed(KeyCode::Space);
    let mut quit = keys.just_pressed(KeyCode::Escape);
    // Right arrow / D-pad Right CYCLES the render view (mesh → mesh+colliders → colliders). The
    // arrow keys are otherwise the orbit camera, but its right-yaw moved to the comma key so this
    // single binding isn't double-bound (mouse right-drag still orbits).
    let mut cycle_view = keys.just_pressed(KeyCode::ArrowRight);
    for gp in gamepads.iter() {
        reset |= gp.just_pressed(GamepadButton::South);
        poke |= gp.just_pressed(GamepadButton::West);
        quit |= gp.just_pressed(GamepadButton::Start);
        cycle_view |= gp.just_pressed(GamepadButton::DPadRight);
    }

    if quit {
        exit.write(AppExit::Success);
    }
    if cycle_view {
        *render_mode = render_mode.next();
        info!("demo render mode: {:?}", *render_mode);
    }
    if reset {
        // A manual reset re-tilts too, so the owner can re-roll the righting attempt.
        demo_respawn(
            &mut commands,
            &assets,
            &spawns,
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

/// System (FixedUpdate, after Think): while a demo settle is active, hold
/// zero actions so the motors keep the rest pose while the fresh crab takes
/// load.
pub(super) fn demo_settle(mut settle: ResMut<DemoSettle>, mut actions: ResMut<CrabActions>) {
    if settle.0 == 0 {
        return;
    }
    if let Some(a) = actions.envs.first_mut() {
        *a = [0.0; ACTION_SIZE];
    }
    // Same countdown training's reset path runs (see `settle_countdown`); spent →
    // 0 (settled), which the demo treats as "policy back in control".
    settle.0 = settle_countdown(settle.0);
}
