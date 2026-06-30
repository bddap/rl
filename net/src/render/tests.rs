//! Headless tests for the render client (no window/GPU): the menu->round handoff, the
//! manual-vs-auto crab pump determinism, and the client input/interpolation math.

use super::*;
use super::app::ExternalCrabStackInstalled;
use super::driver::{
    FlightInput, GameState, InputSource, PendingRound, VEHICLE_STICK_SENS, ensure_round_installed,
    flight_control,
};
use crab_world::vehicle::VehicleKind;
use super::input::pad_stick_axes;
use super::scene::{lerp_pos, lerp_yaw, look_direction};
use crate::menu::ReadyMatch;
use crate::net_loop;
use crate::sim::Sim;


/// The boot menu's handoff into the round (rl#56), exercised headlessly (no window):
/// park a chosen [`ReadyMatch`] in [`PendingRound`], request the Playing transition,
/// and prove `OnEnter(Playing)`'s [`ensure_round_installed`] builds a live
/// [`GameState`] from it — the determinism-critical link the menu depends on (the menu
/// only selects a round; this is where it actually becomes the sim). Uses
/// `MinimalPlugins` + the state plumbing only, so it needs no display/GPU and can run
/// on the headless box. (The egui UI + 2-peer formation still need on-device testing;
/// this pins the part that decides which sim the round runs.)
#[test]
fn menu_handoff_installs_the_chosen_round() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(bevy::state::app::StatesPlugin)
        .init_state::<AppPhase>()
        .init_non_send_resource::<PendingRound>()
        .add_systems(OnEnter(AppPhase::Playing), ensure_round_installed);

    // The menu app always installs the NN-crab stack at build (rl#114: the checkpoint is
    // required), and `ensure_round_installed` asserts its presence before arming. Mirror that
    // here so the handoff exercises the real menu path rather than tripping the build-wiring
    // assert.
    app.world_mut().insert_resource(ExternalCrabStackInstalled);

    // Park a solo round (the same one the Solo button / Alone fallback produce) and ask
    // to enter Playing, exactly as the menu does on a choice.
    let seed = 0x1234_5678;
    app.world_mut()
        .insert_non_send_resource(PendingRound(Some(ReadyMatch {
            lockstep: net_loop::solo_lockstep_for(seed),
            net: None,
        })));
    app.world_mut()
        .resource_mut::<NextState<AppPhase>>()
        .set(AppPhase::Playing);

    // One update applies the transition and runs OnEnter(Playing).
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
    // The installed sim is the chosen one: a single local player (solo), seeded as asked.
    assert_eq!(gs.ls.me(), crate::sim::PlayerId(0), "solo player id 0");
    assert!(
        matches!(&gs.input_source, InputSource::Coordinated(c) if c.is_solo()),
        "a solo handoff installs a solo (internal-server) coordinator"
    );
    // And the parked round was consumed (taken), not left to double-install.
    assert!(
        app.world()
            .get_non_send_resource::<PendingRound>()
            .is_some_and(|p| p.0.is_none()),
        "the chosen round must be taken out of PendingRound"
    );
}

/// An unarmable networked round (peers disagree on the brain/colliders) must drive the GRACEFUL
/// refusal, NOT a crash and NOT a silent integer-crab swap (rl#115 + rl#114). The arm decision +
/// operator message is the single [`super::app::crab_arm_failure_from`]; this pins that a solo or
/// fully-synced round arms (no message), while either mismatch REFUSES with an actionable message
/// naming the cause and the fix — the value the menu's `poll_formation` gate returns to the chooser
/// on instead of panicking. (The live 2-peer menu transition still needs on-device testing; the
/// `NetDriver` it carries owns a tokio/iroh session that won't stand up headlessly — this pins the
/// decision the gate is built on.)
#[test]
fn unarmable_round_refuses_with_actionable_message_not_a_crash() {
    use super::app::crab_arm_failure_from;
    // Armable: solo always arms; a fully-synced networked round arms. No refusal, no message.
    assert!(
        crab_arm_failure_from(true, false, false).is_none(),
        "solo (no net) always arms — the synced flags are irrelevant"
    );
    assert!(
        crab_arm_failure_from(false, true, true).is_none(),
        "a networked round with synced weights AND assets arms"
    );
    // A mismatched/absent brain refuses LOUD, naming the brain + the rl-update fix.
    let brain = crab_arm_failure_from(false, false, false)
        .expect("an unsynced-weights networked round must refuse, not arm a fake crab");
    assert!(brain.contains("brain.bin"), "names the brain mismatch: {brain}");
    assert!(
        brain.contains("rl-update"),
        "tells the operator how to fix it: {brain}"
    );
    assert!(
        brain.contains("refusing"),
        "the round REFUSES (no silent integer fallback): {brain}"
    );
    // Brain agrees but the colliders differ: refuse with the collider cause.
    let colliders = crab_arm_failure_from(false, true, false)
        .expect("an unsynced-assets networked round must refuse, not arm a fake crab");
    assert!(
        colliders.contains("sally.glb"),
        "names the collider mismatch: {colliders}"
    );
}

/// The GCR fold's manual fixed-step pump ([`pump_fixed_steps`]) must reproduce, bit-for-bit,
/// the physics Bevy's wall-clock auto-pump produces — the stepping `bot::determinism_probe`
/// proves deterministic. Build two identical headless crab worlds; step one with `app.update()`
/// (the auto-pump path) and the other with `pump_fixed_steps` after parking `Time<Fixed>` (the
/// windowed driver's path); drive both with the SAME scripted torque and assert their full
/// articulated crab digests agree every tick. If they do, the windowed crab inherits ALL of
/// the probe's determinism guarantees; if `pump_fixed_steps` ever double-stepped, skipped a
/// fixed sub-schedule, or fed the wrong clock, this diverges. (Render-only — it needs the real
/// rapier+bot stack — but headless: no window/GPU.)
#[test]
fn manual_pump_matches_auto_pump_step_for_step() {
    use crab_world::bot::actuator::{ACTION_SIZE, CrabActions};
    use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint};
    use crab_world::bot::physics_digest::crab_state_digest;
    use crab_world::bot::headless::{HeadlessStack, WorldRole, headless_stack};
    use bevy_rapier3d::prelude::Velocity;

    let build = || {
        headless_stack(HeadlessStack {
            num_envs: 1,
            role: WorldRole::Standalone,
        })
    };
    let mut auto = build();
    let mut manual = build();
    // One update each: run Startup (spawns the crab + sizes CrabActions) + one physics step,
    // so both worlds start from the identical post-spawn state.
    auto.update();
    manual.update();
    // Park the manual world's wall-clock auto-pump, exactly as `add_external_nn_crab` does, so
    // from here ONLY `pump_fixed_steps` advances its physics.
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

/// The frame conversion must match the sim's documented right-handed XZ layout:
/// +X right, +Z forward, Y up. A sim Pos maps straight through to Bevy XYZ with
/// the given height — no axis swap or sign flip.
#[test]
fn world_maps_sim_frame_directly() {
    let p = Pos {
        x: 2 * UNIT,
        z: 5 * UNIT,
    };
    // The sim XZ frame maps straight to Bevy's, then the whole point shrinks by the render-frame
    // scale so the human world renders small around the true-physics-size crab (render==physics).
    let rs = world_render_scale();
    let v = world(p, 1.6);
    assert_eq!(v, Vec3::new(2.0, 1.6, 5.0) * rs);
}

/// The camera's flat (zero-pitch) facing must match the sim's yaw convention:
/// yaw 0 looks +Z, a quarter turn looks +X — so what the player sees agrees with
/// where the sim says it faces.
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

/// Look direction at zero pitch is the flat facing; pitching up tilts +Y without
/// changing the horizontal heading sign.
#[test]
fn look_direction_pitches_without_flipping_heading() {
    let flat = look_direction(0.0, 0.0);
    assert!((flat - Vec3::Z).length() < 1e-5);
    let up = look_direction(0.0, 0.5);
    assert!(up.y > 0.0, "positive pitch looks up, got {up:?}");
    assert!(up.z > 0.0, "still facing +Z, got {up:?}");
}

/// Yaw interpolation takes the short way around the wrap: from just-below-a-full-
/// turn to just-above-zero tweens FORWARD through 0, not backward through ~2π.
#[test]
fn yaw_lerp_takes_short_path_across_wrap() {
    // a ≈ 350°, b ≈ 10° (in turn units). Halfway should land near 0° (=360°),
    // i.e. the short 20° arc, not 180°.
    let a = trig::TURN - trig::TURN / 36; // ~350°
    let b = trig::TURN / 36; // ~10°
    let mid = lerp_yaw(a, b, 0.5);
    // Normalize to [-π, π] around 0.
    let mut n = mid % std::f32::consts::TAU;
    if n > std::f32::consts::PI {
        n -= std::f32::consts::TAU;
    }
    assert!(
        n.abs() < 0.2,
        "midpoint should be ~0 rad (short path), got {n}"
    );
}

/// Position interpolation is the plain linear midpoint in fixed-point space.
#[test]
fn pos_lerp_midpoint() {
    let a = Pos { x: 0, z: 0 };
    let b = Pos { x: 1000, z: -400 };
    let mid = lerp_pos(a, b, 0.5);
    assert_eq!(mid, Pos { x: 500, z: -200 });
}

/// A full-deflection look this tick must map to EXACTLY the sim's per-tick yaw cap
/// — no more (the sim would clamp it and the camera would lag the avatar), no less
/// (the player couldn't turn as fast as the sim allows). This pins the client's
/// `look_yaw` normalization to the sim's `MAX_YAW_TURNS_PER_TICK`, the coupling
/// that keeps the FP camera and the authoritative yaw in agreement.
#[test]
fn full_look_axis_turns_one_tick_cap() {
    // Drive a fresh sim one tick with look_yaw at full deflection; the yaw delta
    // must equal the sim's documented per-tick cap (TURN/24).
    let mut sim = Sim::new(0, &[PlayerId(0)]);
    let before = sim.player(PlayerId(0)).unwrap().yaw();
    // The client builds this exact input for a +MAX_YAW_PER_TICK_RADIANS look:
    // yaw_delta / MAX_YAW_PER_TICK_RADIANS, saturating the axis at full deflection.
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

/// WASD-shaped move + the action button map to the expected fixed-point [`Input`]:
/// forward+right at full deflection quantize to +AXIS_SCALE, and pressing action
/// sets the ACTION bit. (Mirrors how `gather_input`/`drive_lockstep` build the
/// per-tick input from the accumulated controls.)
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

/// Pins the geometric fact that `gather_input`'s X-axis negation corrects: a camera
/// facing +Z (yaw 0) has its RIGHT axis at world −X, so the sim's "+X = strafe
/// right" renders on the SCREEN-LEFT. This is why the control layer negates strafe
/// and yaw-look — keeping the proof in a test so a future camera change can't
/// silently re-invert the controls.
#[test]
fn camera_right_is_negative_x_facing_plus_z() {
    let eye = Vec3::new(0.0, EYE_HEIGHT, 0.0);
    let cam =
        Transform::from_translation(eye).looking_at(eye + look_direction(0.0, 0.0), Vec3::Y);
    let right = cam.right().as_vec3();
    assert!(
        (right - Vec3::NEG_X).length() < 1e-5,
        "facing +Z, camera-right must be world −X (so sim +X is screen-left); got {right:?}"
    );
}

/// A stick resting inside the deadzone contributes exactly zero on every axis — the
/// guard that hardware idle-noise can't creep the avatar or drift the view. Tests the
/// REAL client transform (`pad_stick_axes`, which `gather_input` calls), so a future
/// edit that drops/weakens the deadzone fails here.
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

/// Past the deadzone, the left stick passes its raw magnitude straight to the move
/// axes (analog, not bang-bang) and the right stick's look scales with both deflection
/// and dt — pinning the frame-rate-independent look and the analog move feel.
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
    // Double the dt → double the per-frame look, the frame-rate independence that
    // keeps turn speed consistent across machines (the i16 it quantizes to is each
    // peer's own broadcast input, so this stays lockstep-safe — see net::desync_test).
    let b = pad_stick_axes(Vec2::ZERO, Vec2::new(1.0, 0.0), dt * 2.0);
    assert!(
        (b.d_yaw - 2.0 * a.d_yaw).abs() < 1e-6,
        "look is linear in dt"
    );
}

/// `pad_stick_axes` does NOT pre-negate any axis: the screen-relative X-negation is
/// applied once, downstream in `gather_input` (the `-strafe` / `yaw_delta -= d_yaw`
/// at the funnel), to BOTH keyboard and pad together. A positive stick X yields a
/// positive raw strafe/yaw here; if this fn negated too, the pad would invert. Pins
/// that the single negation site stays single (no double-negate, no pad-only flip).
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

/// The PLANE control bridge. Pins the directions the cockpit legend rides, above all the INTUITIVE
/// (de-inverted) pitch (the owner's complaint that the old AC6 inversion felt backwards): pushing the
/// left stick UP must raise the nose, matching the ship's aim and the on-foot look. Also pins the
/// VEHICLE_STICK_SENS scaling of the analog attitude stick (the "too sensitive" fix).
#[test]
fn plane_flight_control_pitch_is_intuitive_and_scaled() {
    let plane = |fi: FlightInput| flight_control(VehicleKind::Plane, &fi);
    // Intuitive pitch: stick UP (left.y > 0) → nose UP (pitch > 0); stick down → nose down.
    assert!(plane(FlightInput { left: Vec2::new(0.0, 1.0), ..default() }).pitch > 0.0);
    assert!(plane(FlightInput { left: Vec2::new(0.0, -1.0), ..default() }).pitch < 0.0);
    // Mouse UP (screen −y) also raises the nose (camera-style, matching the ship).
    assert!(plane(FlightInput { mouse: Vec2::new(0.0, -1.0), ..default() }).pitch > 0.0);
    // The analog attitude stick is scaled by VEHICLE_STICK_SENS, not raw: full deflection commands a
    // fraction of full authority (the controller "too sensitive" fix). Mouse keeps its own scale.
    let full = plane(FlightInput { left: Vec2::new(0.0, 1.0), ..default() });
    assert!((full.pitch - VEHICLE_STICK_SENS).abs() < 1e-6, "full-up stick → VEHICLE_STICK_SENS pitch");
    let rolled = plane(FlightInput { left: Vec2::new(1.0, 0.0), ..default() });
    assert!((rolled.roll - VEHICLE_STICK_SENS).abs() < 1e-6, "full-right stick → VEHICLE_STICK_SENS roll");
    // Roll: stick right → bank right (+roll), with a coordinating yaw the SAME sign (turns, not just
    // rolls).
    assert!(rolled.roll > 0.0 && rolled.yaw > 0.0, "right stick → bank right + coordinated yaw");
    // Throttle: RT accelerates (+), LT brakes (−). Rudder: RB right (+yaw), LB left (−yaw).
    assert!(plane(FlightInput { rt: 1.0, ..default() }).throttle_trim > 0.0);
    assert!(plane(FlightInput { lt: 1.0, ..default() }).throttle_trim < 0.0);
    assert!(plane(FlightInput { rb: true, ..default() }).yaw > 0.0);
    assert!(plane(FlightInput { lb: true, ..default() }).yaw < 0.0);
    // The plane thrusts through its lever, never the direct thrusters; it never match-velocities.
    let p = plane(FlightInput { left: Vec2::new(1.0, 1.0), rt: 1.0, ..default() });
    assert_eq!(p.thrust, Vec3::ZERO);
    assert!(!p.match_velocity);
}

/// The SHIP control bridge = Outer Wilds. Pins the 6-DOF thrust axes + the camera-style (NON-
/// inverted) aim, distinct from the plane.
#[test]
fn ship_flight_control_is_outer_wilds() {
    let ship = |fi: FlightInput| flight_control(VehicleKind::Ship, &fi);
    // Direct thrusters: left stick forward (+y) → +Z thrust; right (+x) → +X strafe; RT up / LT down.
    assert!(ship(FlightInput { left: Vec2::new(0.0, 1.0), ..default() }).thrust.z > 0.0);
    assert!(ship(FlightInput { left: Vec2::new(1.0, 0.0), ..default() }).thrust.x > 0.0);
    assert!(ship(FlightInput { rt: 1.0, ..default() }).thrust.y > 0.0);
    assert!(ship(FlightInput { lt: 1.0, ..default() }).thrust.y < 0.0);
    // Aim is camera-style, NOT inverted: right stick UP → nose UP (pitch > 0); right → yaw right.
    // The analog AIM stick is scaled by VEHICLE_STICK_SENS (the "too sensitive" fix), like the plane.
    let aim_up = ship(FlightInput { right: Vec2::new(0.0, 1.0), ..default() });
    assert!((aim_up.pitch - VEHICLE_STICK_SENS).abs() < 1e-6, "full-up aim stick → VEHICLE_STICK_SENS pitch");
    assert!(ship(FlightInput { right: Vec2::new(1.0, 0.0), ..default() }).yaw > 0.0);
    // Translational thrust keeps FULL authority — only rotation is desensitized.
    assert_eq!(ship(FlightInput { left: Vec2::new(0.0, 1.0), ..default() }).thrust.z, 1.0);
    // Roll on the bumpers; A/Space matches velocity. The ship has no throttle lever.
    assert!(ship(FlightInput { rb: true, ..default() }).roll > 0.0);
    assert!(ship(FlightInput { match_vel: true, ..default() }).match_velocity);
    assert_eq!(ship(FlightInput { rt: 1.0, ..default() }).throttle_trim, 0.0);
}
