use super::app::ExternalCrabStackInstalled;
use super::driver::{
    FlightInput, GameState, PendingRound, VEHICLE_STICK_SENS, ensure_round_installed,
    flight_control, park_fixed_auto_pump, pump_fixed_steps,
};
use super::input::pad_stick_axes;
use super::scene::{lerp_pos, lerp_yaw, look_direction};
use super::*;
use crate::menu::ReadyMatch;
use crate::sim::{Sim, UNIT};
use crab_world::vehicle::VehicleKind;

#[test]
fn menu_handoff_installs_the_chosen_round() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(bevy::state::app::StatesPlugin)
        .init_state::<AppPhase>()
        .init_non_send_resource::<PendingRound>()
        .add_systems(OnEnter(AppPhase::Playing), ensure_round_installed);

    app.world_mut().insert_resource(ExternalCrabStackInstalled);

    let seed = 0x1234_5678;
    let armed = super::app::arm_round(ReadyMatch {
        client: crate::formation::solo_client_for(seed),
        net: None,
    })
    .expect("a solo round always arms");
    app.world_mut()
        .insert_non_send_resource(PendingRound(Some(armed)));
    app.world_mut()
        .resource_mut::<NextState<AppPhase>>()
        .set(AppPhase::Playing);

    app.update();

    assert_eq!(
        *app.world().resource::<State<AppPhase>>().get(),
        AppPhase::Playing,
        "the transition must have entered Playing"
    );
    let gs = app
        .world()
        .get_non_send_resource::<GameState>()
        .expect("ensure_round_installed must build GameState from the parked round");
    assert_eq!(gs.client.me(), crate::sim::PlayerId(0), "solo player id 0");
    assert!(
        matches!(
            *gs.coord,
            crate::net_loop::Coordinator::Server { net: None, .. }
        ),
        "a solo handoff installs a solo (internal-server) coordinator"
    );
    assert!(
        app.world()
            .get_non_send_resource::<PendingRound>()
            .is_some_and(|p| p.0.is_none()),
        "the chosen round must be taken out of PendingRound"
    );
}

#[test]
fn unarmable_round_refuses_with_actionable_message_not_a_crash() {
    use super::app::check_armable;
    use crate::SyncVerdict;
    let synced = |assets, crabs| Some(SyncVerdict { assets, crabs });
    assert!(
        check_armable(None).is_ok(),
        "solo (no net, no formation verdict) always arms"
    );
    assert!(
        check_armable(synced(true, true)).is_ok(),
        "a networked round with synced assets + crab count arms"
    );
    let count = check_armable(synced(true, false))
        .expect_err("a count-mismatched networked round must refuse, not arm the wrong crabs");
    assert!(
        count.contains("crab count") || count.contains("NN-crab count"),
        "names the count cause: {count}"
    );
    assert!(
        count.contains("binding list"),
        "tells the operator the fix: {count}"
    );
    let colliders = check_armable(synced(false, true))
        .expect_err("an unsynced-assets networked round must refuse, not arm a fake crab");
    assert!(
        colliders.contains("sally.glb"),
        "names the collider mismatch: {colliders}"
    );
    assert!(
        colliders.contains("rl-update"),
        "tells the operator how to fix it: {colliders}"
    );
    assert!(
        colliders.contains("refusing"),
        "the round REFUSES (no silent integer fallback): {colliders}"
    );
}

#[test]
fn manual_pump_matches_auto_pump_step_for_step() {
    use bevy_rapier3d::prelude::Velocity;
    use crab_world::bot::actuator::{ACTION_SIZE, CrabActions};
    use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint};
    use crab_world::bot::headless::{HeadlessStack, WorldRole, headless_stack};
    use crab_world::bot::physics_digest::crab_state_digest;

    let build = || {
        headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
            // Models the GCR client's world, so it steps the client's OPEN inference
            // field (rl#209), not the walled training box.
            arena: crab_world::physics::Arena::OpenField,
            visuals: crab_world::Visuals(false),
        })
    };
    let mut auto = build();
    let mut manual = build();
    auto.update();
    manual.update();
    park_fixed_auto_pump(manual.world_mut());

    let digest = |app: &mut App| -> u64 {
        let mut q = app.world_mut().query_filtered::<(
            &Transform,
            &Velocity,
            Option<&CrabJoint>,
            Option<&CrabCarapace>,
        ), With<CrabBodyPart>>();
        crab_state_digest(q.iter(app.world()))
    };
    let set_torque = |app: &mut App, a: [f32; ACTION_SIZE]| {
        app.world_mut().resource_mut::<CrabActions>().envs[0] = a;
    };

    let mut lcg: u64 = 0x1234_5678_9abc_def0;
    for t in 0..120u32 {
        let mut act = [0.0f32; ACTION_SIZE];
        for slot in act.iter_mut() {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = ((lcg >> 40) as u32 as f32 / (1u32 << 24) as f32) * 1.6 - 0.8;
        }
        set_torque(&mut auto, act);
        set_torque(&mut manual, act);
        auto.update();
        pump_fixed_steps(manual.world_mut(), 1);
        assert_eq!(
            digest(&mut auto),
            digest(&mut manual),
            "manual pump diverged from auto-pump at tick {t}"
        );
    }
}

#[test]
fn world_maps_sim_frame_directly() {
    let p = Pos {
        x: 2 * UNIT,
        z: 5 * UNIT,
    };
    let v = world(p, 1.6);
    assert_eq!(v, Vec3::new(2.0, 1.6, 5.0));
}

#[test]
fn camera_facing_matches_sim_yaw_convention() {
    let f0 = look_direction(0.0, 0.0);
    assert!(
        (f0 - Vec3::Z).length() < 1e-5,
        "yaw 0 should look +Z, got {f0:?}"
    );
    let fq = look_direction(std::f32::consts::FRAC_PI_2, 0.0);
    assert!(
        (fq - Vec3::X).length() < 1e-5,
        "quarter turn should look +X, got {fq:?}"
    );
}

#[test]
fn look_direction_pitches_without_flipping_heading() {
    let flat = look_direction(0.0, 0.0);
    assert!((flat - Vec3::Z).length() < 1e-5);
    let up = look_direction(0.0, 0.5);
    assert!(up.y > 0.0, "positive pitch looks up, got {up:?}");
    assert!(up.z > 0.0, "still facing +Z, got {up:?}");
}

#[test]
fn yaw_lerp_takes_short_path_across_wrap() {
    let a = trig::TURN - trig::TURN / 36;
    let b = trig::TURN / 36;
    let mid = lerp_yaw(a, b, 0.5);
    let mut n = mid % std::f32::consts::TAU;
    if n > std::f32::consts::PI {
        n -= std::f32::consts::TAU;
    }
    assert!(
        n.abs() < 0.2,
        "midpoint should be ~0 rad (short path), got {n}"
    );
}

#[test]
fn pos_lerp_midpoint() {
    let a = Pos { x: 0, z: 0 };
    let b = Pos { x: 1000, z: -400 };
    let mid = lerp_pos(a, b, 0.5);
    assert_eq!(mid, Pos { x: 500, z: -200 });
}

#[test]
fn full_look_axis_turns_one_tick_cap() {
    let mut sim = Sim::new(0, &[PlayerId(0)]);
    let before = sim.player(PlayerId(0)).unwrap().yaw();
    let look_axis = (MAX_YAW_PER_TICK_RADIANS / MAX_YAW_PER_TICK_RADIANS).clamp(-1.0, 1.0);
    assert_eq!(look_axis, 1.0, "a full-deflection look saturates the axis");
    let input = Input::new(0.0, 0.0, look_axis, 0);
    let mut inputs = BTreeMap::new();
    inputs.insert(PlayerId(0), input);
    sim.step(&inputs);
    let after = sim.player(PlayerId(0)).unwrap().yaw();
    let cap = trig::TURN / 24;
    assert_eq!(
        trig::wrap_turns(after - before),
        cap,
        "full look axis should turn exactly the sim's per-tick cap"
    );
}

#[test]
fn move_and_action_map_to_input() {
    let i = Input::new(1.0, 1.0, 0.0, buttons::ACTION);
    assert_eq!(i.move_strafe, Input::AXIS_SCALE, "full right → +AXIS_SCALE");
    assert_eq!(
        i.move_forward,
        Input::AXIS_SCALE,
        "full forward → +AXIS_SCALE"
    );
    assert!(i.pressed(buttons::ACTION), "action bit set when pressed");
    let n = Input::new(0.0, 0.0, 0.0, 0);
    assert!(!n.pressed(buttons::ACTION), "no action bit when unpressed");
}

#[test]
fn camera_right_is_negative_x_facing_plus_z() {
    let eye = Vec3::new(0.0, EYE_HEIGHT, 0.0);
    let cam = Transform::from_translation(eye).looking_at(eye + look_direction(0.0, 0.0), Vec3::Y);
    let right = cam.right().as_vec3();
    assert!(
        (right - Vec3::NEG_X).length() < 1e-5,
        "facing +Z, camera-right must be world −X (so sim +X is screen-left); got {right:?}"
    );
}

#[test]
fn pad_sub_deadzone_sticks_contribute_nothing() {
    let inside = PAD_STICK_DEADZONE * 0.9;
    let a = pad_stick_axes(Vec2::new(inside, 0.0), Vec2::new(0.0, inside), 1.0 / 60.0);
    assert_eq!(
        (a.strafe, a.forward),
        (0.0, 0.0),
        "sub-deadzone move is zero"
    );
    assert_eq!(
        (a.d_yaw, a.d_pitch),
        (0.0, 0.0),
        "sub-deadzone look is zero"
    );
}

#[test]
fn pad_above_deadzone_passes_move_and_scales_look_by_dt() {
    let dt = 1.0 / 60.0;
    let a = pad_stick_axes(Vec2::new(0.8, -0.6), Vec2::new(1.0, 0.0), dt);
    assert_eq!(a.strafe, 0.8, "left stick X → strafe, analog");
    assert_eq!(a.forward, -0.6, "left stick Y → forward, analog");
    assert!(
        (a.d_yaw - PAD_LOOK_SPEED * dt).abs() < 1e-6,
        "full right-stick-X look = PAD_LOOK_SPEED·dt, got {}",
        a.d_yaw
    );
    let b = pad_stick_axes(Vec2::ZERO, Vec2::new(1.0, 0.0), dt * 2.0);
    assert!(
        (b.d_yaw - 2.0 * a.d_yaw).abs() < 1e-6,
        "look is linear in dt"
    );
}

#[test]
fn pad_axes_are_not_pre_negated() {
    let a = pad_stick_axes(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0), 1.0 / 60.0);
    assert!(
        a.strafe > 0.0,
        "+stick X → +raw strafe (negation is downstream)"
    );
    assert!(
        a.d_yaw > 0.0,
        "+stick X → +raw yaw (negation is downstream)"
    );
}

#[test]
fn plane_flight_control_pitch_is_ac6_and_scaled() {
    let plane = |fi: FlightInput| flight_control(VehicleKind::Plane, &fi);
    assert!(
        plane(FlightInput {
            left: Vec2::new(0.0, -1.0),
            ..default()
        })
        .pitch
            > 0.0
    );
    assert!(
        plane(FlightInput {
            left: Vec2::new(0.0, 1.0),
            ..default()
        })
        .pitch
            < 0.0
    );
    // Mouse BACK (screen +y, drag down) also raises the nose (flight-sim).
    assert!(
        plane(FlightInput {
            mouse: Vec2::new(0.0, 1.0),
            ..default()
        })
        .pitch
            > 0.0
    );
    let full = plane(FlightInput {
        left: Vec2::new(0.0, -1.0),
        ..default()
    });
    assert!(
        (full.pitch - VEHICLE_STICK_SENS).abs() < 1e-6,
        "full-back stick → VEHICLE_STICK_SENS pitch"
    );
    let rolled = plane(FlightInput {
        left: Vec2::new(1.0, 0.0),
        ..default()
    });
    assert!(
        (rolled.roll + VEHICLE_STICK_SENS).abs() < 1e-6,
        "full-right stick → −VEHICLE_STICK_SENS roll (screen-reconciled)"
    );
    assert!(
        rolled.roll < 0.0 && rolled.yaw < 0.0,
        "right stick → bank right (−roll) + coordinated yaw"
    );
    assert!(
        plane(FlightInput {
            rt: 1.0,
            ..default()
        })
        .throttle_trim
            > 0.0
    );
    assert!(
        plane(FlightInput {
            lt: 1.0,
            ..default()
        })
        .throttle_trim
            < 0.0
    );
    assert!(
        plane(FlightInput {
            rb: true,
            ..default()
        })
        .yaw < 0.0
    );
    assert!(
        plane(FlightInput {
            lb: true,
            ..default()
        })
        .yaw > 0.0
    );
    let p = plane(FlightInput {
        left: Vec2::new(1.0, 1.0),
        rt: 1.0,
        ..default()
    });
    assert_eq!(p.thrust, Vec3::ZERO);
    assert!(!p.match_velocity);
}

#[test]
fn ship_flight_control_is_outer_wilds() {
    let ship = |fi: FlightInput| flight_control(VehicleKind::Ship, &fi);
    assert!(
        ship(FlightInput {
            left: Vec2::new(0.0, 1.0),
            ..default()
        })
        .thrust
        .z > 0.0
    );
    assert!(
        ship(FlightInput {
            left: Vec2::new(1.0, 0.0),
            ..default()
        })
        .thrust
        .x < 0.0
    );
    assert!(
        ship(FlightInput {
            rt: 1.0,
            ..default()
        })
        .thrust
        .y > 0.0
    );
    assert!(
        ship(FlightInput {
            lt: 1.0,
            ..default()
        })
        .thrust
        .y < 0.0
    );
    let aim_up = ship(FlightInput {
        right: Vec2::new(0.0, 1.0),
        ..default()
    });
    assert!(
        (aim_up.pitch - VEHICLE_STICK_SENS).abs() < 1e-6,
        "full-up aim stick → VEHICLE_STICK_SENS pitch"
    );
    assert!(
        ship(FlightInput {
            right: Vec2::new(1.0, 0.0),
            ..default()
        })
        .yaw < 0.0
    );
    assert_eq!(
        ship(FlightInput {
            left: Vec2::new(0.0, 1.0),
            ..default()
        })
        .thrust
        .z,
        1.0
    );
    assert!(
        ship(FlightInput {
            lb: true,
            ..default()
        })
        .roll
            > 0.0
    );
    assert!(
        ship(FlightInput {
            rb: true,
            ..default()
        })
        .roll
            < 0.0
    );
    assert!(
        ship(FlightInput {
            match_vel: true,
            ..default()
        })
        .match_velocity
    );
    assert_eq!(
        ship(FlightInput {
            rt: 1.0,
            ..default()
        })
        .throttle_trim,
        0.0
    );
}

/// The FP perspective's EFFECTIVE clip must sit at its configured `near`. Bevy 0.18
/// clips by `PerspectiveProjection::near_clip_plane` (an oblique portals/mirrors
/// plane), not `near` — its default is the stock 0.1 m plane, so a custom `near` with
/// a stale default plane still clips at 0.1 render-m ≈ 2 eye-heights: looking down
/// while standing saw straight through the floor (rl#196). For a straight-ahead
/// (non-oblique) plane matching `near`, the oblique adjustment is a no-op and the
/// infinite-reverse matrix carries `near` at w_axis.z — pin that so the pair can't
/// drift apart again on a bevy bump.
#[test]
fn fp_camera_effective_clip_is_the_scaled_near() {
    use bevy::camera::CameraProjection;
    let projection = scene::fp_perspective();
    let clip_from_view = projection.get_clip_from_view();
    assert_eq!(clip_from_view.w_axis.z, projection.near);
    assert!(
        projection.near < 0.01,
        "near plane must shrink with the render frame"
    );
}

/// The menu egui context must survive a full Menu → Playing → Menu cycle. bevy_egui's
/// auto-create fires ONCE (a `Local<bool>` latch): after `despawn_menu_camera` took the
/// first context down with its camera, a respawned menu camera stayed context-less and
/// `menu_screen` errored every frame (rl#237). The camera therefore carries an explicit
/// `PrimaryEguiContext` — pin that every (re)spawn has one.
#[test]
fn menu_camera_regains_egui_context_after_round_over() {
    use super::menu::{MenuCamera, despawn_menu_camera, spawn_menu_camera};
    use bevy_egui::PrimaryEguiContext;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(bevy::state::app::StatesPlugin)
        .init_state::<AppPhase>()
        .add_systems(OnEnter(AppPhase::Menu), spawn_menu_camera)
        .add_systems(OnEnter(AppPhase::Playing), despawn_menu_camera);

    let menu_cam_with_ctx = |app: &mut App| {
        let mut q = app
            .world_mut()
            .query_filtered::<Has<PrimaryEguiContext>, With<MenuCamera>>();
        q.iter(app.world()).collect::<Vec<_>>()
    };

    app.update();
    assert_eq!(
        menu_cam_with_ctx(&mut app),
        vec![true],
        "boot: the initial OnEnter(Menu) spawn carries the context"
    );

    app.world_mut()
        .resource_mut::<NextState<AppPhase>>()
        .set(AppPhase::Playing);
    app.update();
    assert!(
        menu_cam_with_ctx(&mut app).is_empty(),
        "Playing must tear the menu camera down"
    );

    app.world_mut()
        .resource_mut::<NextState<AppPhase>>()
        .set(AppPhase::Menu);
    app.update();
    assert_eq!(
        menu_cam_with_ctx(&mut app),
        vec![true],
        "the respawned menu camera must carry its own PrimaryEguiContext"
    );
}
