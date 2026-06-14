//! TEMPORARY in-demo physics tuning panel (`--debug-sliders`).
//!
//! A throwaway debugging aid: drag sliders to feel out the physics live instead
//! of editing a constant and waiting on a release rebuild. It is wired ONLY into
//! the demo, behind the `--debug-sliders` flag, so it can never touch training,
//! headless, or a normal demo launch. Nothing here changes the static defaults in
//! [`crate::physics`] / [`crate::bot::body`] — the sliders are seeded FROM those
//! defaults and then override the live simulation on top of them, leaving the
//! source-of-truth constants free for separate tuning work.
//!
//! How the live override reaches rapier (all per-frame, in `FixedUpdate` ahead of
//! the solver step):
//! - Contact spring + `length_unit` are fields on the context's
//!   [`IntegrationParameters`]; we write them straight onto
//!   `RapierContextSimulation`.
//! - Substeps is the [`TimestepMode`] resource.
//! - Joint limit spring + friction motor live on each [`MultibodyJoint`]'s joint
//!   description. Mutating the bevy component trips bevy_rapier's
//!   `apply_joint_user_changes`, which re-syncs the description into the live
//!   multibody link — so a dragged slider reaches the running crab next step.
//! - Restitution is a [`Restitution`] component bevy_rapier reads on change; the
//!   crab/arena colliders spawn without one, so we INSERT it (and keep it updated)
//!   on every collider, giving a uniform coefficient (the `Average` combine rule
//!   then makes the effective restitution equal to the slider value).
//!
//! Each write is guarded by an equality check so a steady slider does not trip
//! change-detection every frame; this also makes the override sticky across a
//! demo respawn (a fresh crab's default-valued components differ from the slider
//! and get corrected once).

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::dynamics::SpringCoefficients;

use crate::bot::body::{self, CrabJoint, CrabJointId};
use crate::physics;

/// Live-tunable physics knobs, seeded from the current static defaults and then
/// overriding the running simulation. Ranges are picked to bracket each default
/// with room to explore both softer and stiffer; see [`Self::default`].
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct DebugPhysicsParams {
    /// Contact constraint spring (`IntegrationParameters::contact_softness`). The
    /// residual resting bounce lives here (see [`physics::CONTACT_SOFTNESS`]).
    pub contact_freq: f32,
    pub contact_damping: f32,
    /// Solver tolerance scale (`IntegrationParameters::length_unit`). Smaller =
    /// tighter for the cm-scale feet.
    pub length_unit: f32,
    /// Revolute joint limit spring ([`body::LIMIT_SOFTNESS`]) — the owner wants the
    /// damping slider mainly to *reduce* joint damping, hence the range floor < 2.0.
    pub limit_freq: f32,
    pub limit_damping: f32,
    /// Leg friction-motor breakaway cap in N·m ([`body::LEG_FRICTION_CAP`]); lower =
    /// floppier legs. Applies to leg joints only (claw/eye caps are left at spawn).
    pub leg_friction_cap: f32,
    /// Collider restitution coefficient (bounciness), 0 = no bounce.
    pub restitution: f32,
    /// Solver substeps per tick ([`TimestepMode`]) — also the realtime-framerate lever.
    pub substeps: usize,
}

impl Default for DebugPhysicsParams {
    fn default() -> Self {
        Self {
            contact_freq: physics::CONTACT_SOFTNESS.natural_frequency,
            contact_damping: physics::CONTACT_SOFTNESS.damping_ratio,
            length_unit: 1.0,
            limit_freq: body::LIMIT_SOFTNESS.natural_frequency,
            limit_damping: body::LIMIT_SOFTNESS.damping_ratio,
            leg_friction_cap: body::LEG_FRICTION_CAP,
            restitution: 0.0,
            substeps: physics::PHYSICS_SUBSTEPS,
        }
    }
}

/// Adds the debug slider panel to the demo. Construct it only behind
/// `--debug-sliders`; when absent, none of this is registered and the demo,
/// training, and headless paths are untouched.
pub struct DebugSlidersPlugin;

impl Plugin for DebugSlidersPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        app.init_resource::<DebugPhysicsParams>()
            // UI runs in the egui pass (per render frame); the apply system runs in
            // FixedUpdate ahead of the solver sync so a dragged value reaches the
            // step that follows.
            .add_systems(EguiPrimaryContextPass, debug_sliders_ui)
            .add_systems(
                FixedUpdate,
                apply_debug_params.before(bevy_rapier3d::plugin::PhysicsSet::SyncBackend),
            );
    }
}

/// Draw the slider panel and write edits back into [`DebugPhysicsParams`]. egui
/// edits mutate the resource in place; the apply system reads it next tick.
fn debug_sliders_ui(mut contexts: EguiContexts, mut params: ResMut<DebugPhysicsParams>) -> Result {
    let p = &mut *params;
    egui::Window::new("Physics debug (--debug-sliders)")
        .default_pos([12.0, 80.0])
        .show(contexts.ctx_mut()?, |ui| {
            ui.label("Live physics overrides. Defaults seeded from code.");
            ui.separator();

            ui.label("Contact spring (resting bounce)");
            ui.add(egui::Slider::new(&mut p.contact_freq, 5.0..=30.0).text("freq (Hz)"));
            ui.add(egui::Slider::new(&mut p.contact_damping, 0.5..=10.0).text("damping ratio"));
            ui.separator();

            ui.add(
                egui::Slider::new(&mut p.length_unit, 0.1..=1.0).text("length_unit (solver scale)"),
            );
            ui.separator();

            ui.label("Joint limit spring (drag damping DOWN to loosen)");
            ui.add(egui::Slider::new(&mut p.limit_freq, 30.0..=400.0).text("freq (Hz)"));
            ui.add(egui::Slider::new(&mut p.limit_damping, 0.1..=4.0).text("damping ratio"));
            ui.separator();

            ui.add(
                egui::Slider::new(&mut p.leg_friction_cap, 0.0..=0.5)
                    .text("leg friction cap (N·m)"),
            );
            ui.add(egui::Slider::new(&mut p.restitution, 0.0..=0.5).text("restitution"));
            ui.separator();

            ui.add(egui::Slider::new(&mut p.substeps, 1..=4).text("substeps (framerate lever)"));
            ui.label("Model SCALE is spawn-time only — respawn (R) needed, out of scope here.");
        });
    Ok(())
}

/// Whether a revolute joint's `softness` (limit spring) should track the limit
/// sliders. The prismatic pincer spawns without a limit spring, so leaving it out
/// keeps its behaviour identical to a no-flag run.
fn is_revolute(id: CrabJointId) -> bool {
    !matches!(id, CrabJointId::ClawPincer(_))
}

/// Whether a joint takes the leg friction-cap slider. Claw/eye motors keep their
/// own spawn caps (out of scope), matching the slider's "floppier legs" intent.
fn is_leg(id: CrabJointId) -> bool {
    matches!(
        id,
        CrabJointId::LegCoxa(..) | CrabJointId::LegFemur(..) | CrabJointId::LegTibia(..)
    )
}

/// Push [`DebugPhysicsParams`] through to the live simulation. Each write is
/// guarded by an equality check, so this is a no-op at a steady slider yet
/// re-applies the override after a respawn introduces default-valued components.
///
/// Runs every frame rather than gated on `Changed<DebugPhysicsParams>` precisely
/// because of respawn: a freshly built crab carries spawn-default joints/colliders
/// even when the resource has not changed, and only a per-frame reconciliation
/// catches them.
fn apply_debug_params(
    mut commands: Commands,
    params: Res<DebugPhysicsParams>,
    mut timestep: ResMut<TimestepMode>,
    mut sim: Query<&mut bevy_rapier3d::plugin::context::RapierContextSimulation>,
    mut joints: Query<(&mut MultibodyJoint, &CrabJoint)>,
    mut colliders: Query<(Entity, Option<&mut Restitution>), With<Collider>>,
) {
    // --- IntegrationParameters: contact spring + length_unit ---------------
    if let Ok(mut sim) = sim.single_mut() {
        let ip = &mut sim.integration_parameters;
        if ip.contact_softness.natural_frequency != params.contact_freq
            || ip.contact_softness.damping_ratio != params.contact_damping
        {
            ip.contact_softness = SpringCoefficients {
                natural_frequency: params.contact_freq,
                damping_ratio: params.contact_damping,
            };
        }
        if ip.length_unit != params.length_unit {
            ip.length_unit = params.length_unit;
        }
    }

    // --- Substeps (preserve dt) --------------------------------------------
    // Only Fixed is used by the demo/training/tests; leave any other mode alone.
    if let TimestepMode::Fixed { substeps, .. } = &mut *timestep
        && *substeps != params.substeps
    {
        *substeps = params.substeps;
    }

    // --- Joint limit spring + leg friction cap -----------------------------
    let limit = SpringCoefficients {
        natural_frequency: params.limit_freq,
        damping_ratio: params.limit_damping,
    };
    for (mut joint, crab_joint) in joints.iter_mut() {
        let id = crab_joint.id;
        let g = joint.data.as_ref();
        let softness_stale = is_revolute(id) && g.raw.softness != limit;
        // The revolute motor lives on the angular-X axis (RevoluteJointBuilder maps
        // the physical axis onto AngX); that is where the friction cap was set.
        let cap_stale = is_leg(id)
            && g.motor(JointAxis::AngX)
                .is_some_and(|m| m.max_force != params.leg_friction_cap);
        if !softness_stale && !cap_stale {
            continue; // nothing to change — don't trip change-detection
        }
        let g = joint.data.as_mut();
        if softness_stale {
            g.raw.softness = limit;
        }
        if cap_stale {
            g.set_motor_max_force(JointAxis::AngX, params.leg_friction_cap);
        }
    }

    // --- Restitution: inserted on every collider (crab + arena) ------------
    // The colliders spawn without a Restitution component, so a uniform value here
    // makes the Average combine rule yield exactly `params.restitution`.
    for (entity, restitution) in colliders.iter_mut() {
        match restitution {
            Some(mut r) => {
                if r.coefficient != params.restitution {
                    r.coefficient = params.restitution;
                }
            }
            None => {
                commands
                    .entity(entity)
                    .insert(Restitution::coefficient(params.restitution));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::test_util::{headless_app, tick};
    use bevy_rapier3d::plugin::context::{RapierContextJoints, RapierContextSimulation};

    /// Non-default values for every knob, each inside its slider range, so a test
    /// assertion can tell "the override took" from "still at the spawn default".
    fn overrides() -> DebugPhysicsParams {
        DebugPhysicsParams {
            contact_freq: 9.0,
            contact_damping: 3.0,
            length_unit: 0.3,
            limit_freq: 120.0,
            limit_damping: 0.5,
            leg_friction_cap: 0.42,
            restitution: 0.4,
            substeps: 3,
        }
    }

    /// Build the demo's apply path on top of the headless physics harness: a real
    /// crab + arena, the fixed timestep, and `apply_debug_params` (the UI half is
    /// GUI-only and excluded). One `app.update()` is one fixed tick.
    fn app_with_apply(params: DebugPhysicsParams) -> App {
        let mut app = headless_app();
        app.insert_resource(params).add_systems(
            FixedUpdate,
            apply_debug_params.before(bevy_rapier3d::plugin::PhysicsSet::SyncBackend),
        );
        app
    }

    /// Every slider must actually reach the live rapier state. This is the core
    /// guarantee of the panel: drag a value, the running sim changes. We assert at
    /// the rapier layer (integration parameters, the multibody link's joint, the
    /// collider) — not the bevy components — so the test proves the change crossed
    /// the bevy→rapier boundary, which is exactly what a GUI demo can't show.
    #[test]
    fn overrides_reach_the_live_simulation() {
        let mut app = app_with_apply(overrides());
        // A few ticks: tick 1 spawns the crab + inserts joints/colliders into rapier,
        // later ticks let apply + bevy_rapier's writeback propagate the overrides.
        tick(&mut app, 6);
        let p = overrides();

        // Contact spring + length_unit on the live integration parameters.
        let mut sim_q = app.world_mut().query::<&RapierContextSimulation>();
        let ip = &sim_q.single(app.world()).unwrap().integration_parameters;
        assert_eq!(ip.contact_softness.natural_frequency, p.contact_freq);
        assert_eq!(ip.contact_softness.damping_ratio, p.contact_damping);
        assert_eq!(ip.length_unit, p.length_unit);

        // Substeps on the timestep resource.
        match *app.world().resource::<TimestepMode>() {
            TimestepMode::Fixed { substeps, dt } => {
                assert_eq!(substeps, p.substeps);
                // dt is preserved — we only move substeps.
                assert_eq!(dt, physics::PHYSICS_DT);
            }
            other => panic!("expected Fixed timestep, got {other:?}"),
        }

        // Joint limit spring + leg friction cap, read back from the LIVE multibody
        // link (not the bevy component) to prove the writeback ran.
        let mut handle_q = app
            .world_mut()
            .query::<(&RapierMultibodyJointHandle, &CrabJoint)>();
        let joints: Vec<_> = handle_q
            .iter(app.world())
            .map(|(h, j)| (h.0, j.id))
            .collect();
        let mut ctx_q = app.world_mut().query::<&RapierContextJoints>();
        let ctx_joints = ctx_q.single(app.world()).unwrap();

        let mut checked_revolute = 0;
        let mut checked_leg = 0;
        for (handle, id) in joints {
            let (mb, link_id) = ctx_joints.multibody_joints.get(handle).expect("live joint");
            let data = &mb.link(link_id).unwrap().joint.data;
            if is_revolute(id) {
                assert_eq!(
                    data.softness.natural_frequency, p.limit_freq,
                    "{id:?} limit freq not applied"
                );
                assert_eq!(data.softness.damping_ratio, p.limit_damping);
                checked_revolute += 1;
            }
            if is_leg(id) {
                let motor = &data.motors[JointAxis::AngX as usize];
                assert_eq!(
                    motor.max_force, p.leg_friction_cap,
                    "{id:?} cap not applied"
                );
                checked_leg += 1;
            }
        }
        assert!(checked_revolute > 0, "no revolute joints checked");
        assert_eq!(checked_leg, 24, "expected 24 leg joints (8 legs x 3)");

        // Restitution inserted on the colliders (crab + arena).
        let mut rest_q = app.world_mut().query::<&Restitution>();
        let rests: Vec<f32> = rest_q.iter(app.world()).map(|r| r.coefficient).collect();
        assert!(!rests.is_empty(), "no Restitution components inserted");
        assert!(
            rests.iter().all(|&c| c == p.restitution),
            "a collider's restitution was not the override value"
        );
    }

    /// With params left at their defaults, the apply path must reproduce the
    /// hand-coded constants exactly — i.e. running with `--debug-sliders` and
    /// touching nothing is physically identical to running without it. This pins
    /// the "seeded from defaults, overrides on top" contract.
    #[test]
    fn defaults_match_the_static_constants() {
        let mut app = app_with_apply(DebugPhysicsParams::default());
        tick(&mut app, 6);

        let mut sim_q = app.world_mut().query::<&RapierContextSimulation>();
        let ip = &sim_q.single(app.world()).unwrap().integration_parameters;
        assert_eq!(
            ip.contact_softness.natural_frequency,
            physics::CONTACT_SOFTNESS.natural_frequency
        );
        assert_eq!(
            ip.contact_softness.damping_ratio,
            physics::CONTACT_SOFTNESS.damping_ratio
        );

        let mut handle_q = app
            .world_mut()
            .query::<(&RapierMultibodyJointHandle, &CrabJoint)>();
        let joints: Vec<_> = handle_q
            .iter(app.world())
            .map(|(h, j)| (h.0, j.id))
            .collect();
        let mut ctx_q = app.world_mut().query::<&RapierContextJoints>();
        let ctx_joints = ctx_q.single(app.world()).unwrap();
        for (handle, id) in joints {
            let (mb, link_id) = ctx_joints.multibody_joints.get(handle).expect("live joint");
            let data = &mb.link(link_id).unwrap().joint.data;
            if is_leg(id) {
                assert_eq!(
                    data.motors[JointAxis::AngX as usize].max_force,
                    body::LEG_FRICTION_CAP
                );
            }
            if is_revolute(id) {
                assert_eq!(
                    data.softness.natural_frequency,
                    body::LIMIT_SOFTNESS.natural_frequency
                );
            }
        }
    }
}
