//! The demo crab's liveness loop: spawn/settle/re-tilt/fall-rescue that keep the goofy
//! righting "journey" playing on a passive stream, the poke burst, and the keyboard/gamepad
//! controls (reset, poke, quit, collider wireframes).

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

/// Wall-clock since the last demo re-tilt. A passive stream needs the goofy
/// righting "journey" on a loop: a crab that lands on its feet (or never falls)
/// would otherwise just stand there, so [`demo_retilt`] re-tilts on this timer
/// regardless. See [`DEMO_RETILT_PERIOD_S`].
#[derive(Resource)]
pub(super) struct DemoRetilt {
    since: f32,
}

impl Default for DemoRetilt {
    fn default() -> Self {
        // Seed near the period so the first re-tilt fires a few seconds in — the
        // initial spawn is upright (shared spawn path, see `spawn_initial_crabs`),
        // so this is what makes the stream go goofy shortly after launch without
        // touching that path.
        Self {
            since: DEMO_RETILT_PERIOD_S - 3.0,
        }
    }
}

/// How often the demo re-tilts the crab to a fresh random orientation. Long
/// enough for a full righting attempt (succeed or, with current weights, flail and
/// fail) to play out and read clearly; short enough that the passive stream never
/// sits on a static pose.
const DEMO_RETILT_PERIOD_S: f32 = 9.0;

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
fn random_demo_tilt() -> Quat {
    body::random_spawn_rotation(&mut rand::thread_rng())
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
    // Always present in the demo: the Rapier debug-render plugin is added
    // unconditionally so this toggle works (RL_DEBUG_COLLIDERS only sets the
    // initial on/off — see main.rs).
    mut debug_render: ResMut<DebugRenderContext>,
) {
    let mut reset = keys.just_pressed(KeyCode::KeyR);
    let mut poke = keys.just_pressed(KeyCode::Space);
    let mut quit = keys.just_pressed(KeyCode::Escape);
    // Right arrow / D-pad Right toggles the collider wireframes live. The arrow
    // keys are otherwise the orbit camera, but its right-yaw moved to the comma
    // key so this single binding isn't double-bound (mouse right-drag still orbits).
    let mut toggle_colliders = keys.just_pressed(KeyCode::ArrowRight);
    for gp in gamepads.iter() {
        reset |= gp.just_pressed(GamepadButton::South);
        poke |= gp.just_pressed(GamepadButton::West);
        quit |= gp.just_pressed(GamepadButton::Start);
        toggle_colliders |= gp.just_pressed(GamepadButton::DPadRight);
    }

    if quit {
        exit.write(AppExit::Success);
    }
    if toggle_colliders {
        debug_render.enabled = !debug_render.enabled;
        info!("demo collider wireframes: {}", debug_render.enabled);
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
            random_demo_tilt(),
        );
    }
    if poke {
        let mut rng = rand::thread_rng();
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

/// System: the arena is finite and the policy can walk off it. A crab in
/// free fall under the world is lost to the viewer — rebuild it at spawn,
/// same as a training episode reset would.
pub(super) fn demo_fall_rescue(
    mut commands: Commands,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts_q: Query<Entity, With<CrabBodyPart>>,
    carapace_q: Query<&Transform, With<CrabCarapace>>,
    mut actions: ResMut<CrabActions>,
    mut settle: ResMut<DemoSettle>,
) {
    // A fresh respawn (this tick or a settling one) holds the crab near the origin,
    // not fallen — skipping while it settles also stops a same-tick double-respawn
    // racing `demo_retilt`, since despawns are deferred and the stale carapace would
    // still read as fallen in this query until the command flush.
    if settle.0 > 0 {
        return;
    }
    let Ok(t) = carapace_q.single() else { return };
    if t.translation.y > -2.0 {
        return;
    }
    demo_respawn(
        &mut commands,
        &assets,
        &spawns,
        parts_q.iter(),
        &mut settle,
        &mut actions,
        random_demo_tilt(),
    );
}

/// System: periodically re-tilt the demo crab to a fresh random orientation so the
/// passive stream always shows the goofy righting "journey" — even when the crab
/// lands on its feet or never falls (the fall-rescue alone wouldn't fire then). Held
/// off while a settle is in progress so a re-tilt can't interrupt the previous spawn
/// before it has landed and had its moment. See [`DemoRetilt`].
// A Bevy system: every parameter is a scheduler-injected `Res`/`ResMut`/`Query`, so the
// arity is the dependency list, not a refactor smell — bundling them into a SystemParam
// would only hide the wiring.
#[allow(clippy::too_many_arguments)]
pub(super) fn demo_retilt(
    time: Res<Time>,
    mut retilt: ResMut<DemoRetilt>,
    mut commands: Commands,
    assets: Res<CrabAssets>,
    spawns: Res<CrabSpawns>,
    parts_q: Query<Entity, With<CrabBodyPart>>,
    mut actions: ResMut<CrabActions>,
    mut settle: ResMut<DemoSettle>,
) {
    if settle.0 > 0 {
        // Don't advance the clock mid-settle: time the period from when the crab is
        // actually up and acting, not from the spawn it hasn't landed from yet.
        return;
    }
    retilt.since += time.delta_secs();
    if retilt.since < DEMO_RETILT_PERIOD_S {
        return;
    }
    retilt.since = 0.0;
    demo_respawn(
        &mut commands,
        &assets,
        &spawns,
        parts_q.iter(),
        &mut settle,
        &mut actions,
        random_demo_tilt(),
    );
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
