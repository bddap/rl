pub mod world;

pub use world::PhysicsWorldPlugin;
#[cfg(feature = "render")]
pub use world::{ArenaVisualsPlugin, ArenaWorldPlugin};

use bevy_rapier3d::math::Vect;
use bevy_rapier3d::plugin::RapierContextInitialization;
use bevy_rapier3d::prelude::{RapierConfiguration, TimestepMode};
use bevy_rapier3d::rapier::dynamics::{IntegrationParameters, SpringCoefficients};

pub const PHYSICS_HZ: u64 = 64;

pub const PHYSICS_DT: f32 = 1.0 / PHYSICS_HZ as f32;

/// One-step momentum-cancel cap on a speed-proportional braking coefficient `c`
/// (brake force `F = −c·v`, held over one explicit tick at [`PHYSICS_HZ`]): at
/// `c = m·Hz` the tick's impulse exactly cancels the body's momentum, and past it
/// the "brake" overshoots zero and grows geometrically each step — one large
/// impulse becomes a ballistic escape (rl#339: the craft's quadratic drag past
/// ~440 m/s, Sally's carapace drag past ~1470 m/s). ONE source for the formula:
/// every explicit braking force caps its coefficient through here.
pub const fn brake_coeff_max(mass: f32) -> f32 {
    mass * PHYSICS_HZ as f32
}

/// Substeps + the solver iterations below are the DRIVEN-gait budget (rl#392):
/// this is what every awake tick pays, and at rl#340 stage 2's 4×(8/4/4) it was
/// ~3× too slow for the frame budget — the sim tick ran ~72 ms against 16.6 and
/// the game presented at 2 fps. Stage 2 had raised these globally because the
/// zero-drive mesh multibody chattered at rest (claw links 12.9 m/s of solver
/// noise); that rest regime is now retired from the solver entirely by SLEEP
/// instead (`bot::body`'s sleep gates + `CRAB_SETTLE_EXTRA_ITERATIONS`, the
/// rl#377 source-split), so the global counts only need to carry the actively
/// driven crab. Substeps dominate cost (~5 ms/substep fixed in the driven probe);
/// 2 is the pre-stage-2 value the 60 fps game shipped with.
///
/// Do NOT drop to 1 to buy frame budget (rl#396 stage 6): it does halve the
/// driven step (`game step-profile` p50 9.10→5.43 ms, same TV brain/scene), but
/// the doubled per-solve dt changes the PLANT, not just its convergence — the
/// rest-pose driven crab walks metres off spawn within ~3 s
/// (`armed_visual_crab_stays_finite_and_grounded`, red across the whole
/// iteration matrix (2,2,2)→(4,2,2)/(2,4,4) while the substeps=2 control stayed
/// green 3/3 in the same environment), and the rl#312 actuator-load
/// interpenetration residual busts its 25 mm cap (28.05 mm). Iteration raises —
/// the sanctioned compensation — don't move either wall, so the cost stays
/// paid at 2.
pub const PHYSICS_SUBSTEPS: usize = 2;

/// (outer, internal-PGS, internal-stabilization) solver iterations — the other
/// half of the [`PHYSICS_SUBSTEPS`] driven budget (rationale there). The
/// internal counts at 2 are load-bearing for DRIVEN momentum honesty, not rest
/// quiet: at 1/1 the airborne self-contact thrash leaks 0.31–0.58 m/s² of
/// phantom COM force across realization draws — straddling the rl#321 0.5
/// ceiling and approaching the pre-fix 0.64 scale — while 2/2 measures
/// 0.25–0.39 with real margin, for <1 ms/tick over 1/1 (4/4 buys nothing
/// further; 2/2 is the knee). Outer 2 (down from rapier's default 4, rl#396
/// stage 4): the driven-gait step is ~96% solver and outer count is its price —
/// the rapier-profiler probe measured 5.4→3.5 ms/substep (vel-resolution
/// 3.9→2.0 ms) for outer 4→2 on the GCR driven crab. The rl#396 stage-2 TV
/// ruler put the step-carrying frame right AT the 16.7 ms vsync budget
/// (~15 Hz of ~30 ms present pairs from just-missed frames); this shave is
/// the margin that moves it inside. Outer 3 is NOT a milder option: it
/// measured 4.5 vs 4.4 ms/substep against outer 4 — noise; the drop is at 2.
/// The rl#321 phantom residual stayed at its noise floor and chase-eval did
/// not regress (0.17→0.26 m mean pair progress, deployed release brain).
/// Stabilization at 3 (not 2) buys back the CONTACT-DEPTH convergence the
/// outer cut halved, and 3 is the whole corridor: at 2×(2/2/2) the rl#312
/// actuator-load interpenetration residual crossed its 25 mm/60-tick caps
/// (27.6 mm, 66 ticks — vs main's worst-observed 13.9 mm/8 ticks), while at
/// 2×(2/2/4) the extra positional projection makes the REST crab creep —
/// `armed_visual_crab_stays_finite_and_grounded` drifts 12 m off spawn,
/// deterministically. Stab sweeps are nearly free (3.51→3.54 ms/substep for
/// 2→4) because PGS+assembly own the solver's cost. A ZERO-DRIVE crab still
/// gets `CRAB_SETTLE_EXTRA_ITERATIONS` ADDED to the outer count, so its
/// settle total moved 16→14 — inside spawn.rs's measured bracket (awake at 8
/// total, the floor was set against 16); headless sleep engagement is
/// re-pinned green by `resting_crab_falls_asleep`, the render graph only by
/// the armed smoke's grounded/finite bound.
pub const SOLVER_ITERATIONS: (usize, usize, usize) = (2, 2, 3);

fn fixed_timestep() -> TimestepMode {
    TimestepMode::Fixed {
        dt: PHYSICS_DT,
        substeps: PHYSICS_SUBSTEPS,
    }
}

/// Held explicit (with the PostStartup assert) so a bevy_rapier plumbing change
/// can't silently swap the plant's contact stiffness. Frequency is rapier's 30 Hz
/// default — was 5 Hz, a 36× softer spring that rested weight-bearing limbs
/// 6–10 cm INSIDE the terrain (bddap/rl#299); at 30 Hz the same gait rests ≲1 cm.
/// Do NOT stiffen it to buy contact resolution at the cheap solver counts: 60 Hz
/// at 2×(4/1/1) injects energy under hard impacts — respawn drops launched the
/// carapace 50 m and the craft-ram momentum test blew up (rl#392); resolution
/// depth is the solver's job (`CRAB_SETTLE_EXTRA_ITERATIONS`), not the spring's.
/// Damping ζ20 (4× rapier's 5.0), raised in two measured steps that each bought
/// quiet without touching rest depth (frequency owns that): ζ5→ζ10 halved the
/// resting claw links' solve-noise (rl#340 stage 2); ζ10→ζ20 brings the rest
/// bounce and pre-sleep settle inside the `crab_settles_quietly_at_rest` bounds
/// at the cheap counts (0.045 m bounce at ζ10 vs the 0.024 bound, rl#392).
pub const CONTACT_SOFTNESS: SpringCoefficients<f32> = SpringCoefficients {
    natural_frequency: 30.0,
    damping_ratio: 20.0,
};

const LENGTH_UNIT: f32 = 1.0;

pub const PHYSICS_GRAVITY: Vect = Vect::new(0.0, -9.81, 0.0);

fn rapier_context_init() -> RapierContextInitialization {
    RapierContextInitialization::InitializeDefaultRapierContext {
        integration_parameters: IntegrationParameters {
            contact_softness: CONTACT_SOFTNESS,
            num_solver_iterations: SOLVER_ITERATIONS.0,
            num_internal_pgs_iterations: SOLVER_ITERATIONS.1,
            num_internal_stabilization_iterations: SOLVER_ITERATIONS.2,
            // rapier 0.35 split fixed-body contacts onto a new, stiffer default
            // spring (60 Hz, ζ 10) and moved friction out of the bias pass. The
            // terrain is a fixed body, so both silently replaced the contact
            // physics the plant was tuned under (rl#299 spring, rl#318 slope
            // hold): with either default, zero-input drift on a 55° ramp blows
            // through the slope-hold gate (14.2 m/tumble stock; 2.6 m with only
            // the spring pinned). Pin both to the pre-0.35 semantics.
            static_contact_softness: CONTACT_SOFTNESS,
            friction_in_bias_pass: true,
            // Third silent 0.35 default swap: a per-substep solver speed cap
            // (400 m/s · length_unit) that pre-0.35 rapier did not have. It
            // invisibly rewrites any fast body's velocity, which masks the very
            // energy-injection flings the rl#349 instruments hunt and voids the
            // rl#339 hypersonic drag-brake guarantees (both regression tests
            // launch above it and were clamped to exactly 400). The rl#339 drag
            // force caps own the speed bound; the solver must not have one.
            normalized_max_linear_velocity: f32::MAX,
            // Fourth silent 0.35 default swap: contact recycling keeps a
            // quasi-static pair's contact points stale (up to 5 cm of relative
            // pose drift) AND skips per-step joint-based contact filtering
            // until the pair moves. On the mesh multibody that props the
            // settling legs on phantom joint-adjacent contacts: the crumple
            // collapses from ~1.0 rad to 0.28 (`crab_settles_quietly_at_rest`
            // floppiness arm) — the rl#340-stage-7 "rest pose stiffened"
            // regression. The plant's rest behavior was tuned without it; off.
            // The zero-input slope-hold trade this uncovers is recorded on
            // rl#340 stage 7. The other 0.35 swaps (5 mm slop, 3 m/s corrective
            // cap, 2 cm speculative margin, manifold clustering) measured no
            // effect on any gated behavior and stay at their new defaults.
            contact_recycling: false,
            ..IntegrationParameters::default()
        },
        rapier_configuration: RapierConfiguration {
            gravity: PHYSICS_GRAVITY,
            ..RapierConfiguration::new(LENGTH_UNIT)
        },
    }
}

pub struct CrabPhysicsPlugin;

impl bevy::app::Plugin for CrabPhysicsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        use bevy::app::PostStartup;
        use bevy_rapier3d::plugin::{NoUserData, RapierPhysicsPlugin};
        app.insert_resource(fixed_timestep())
            .insert_resource(rapier_context_init())
            .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
            .add_systems(
                PostStartup,
                (assert_contact_spring_applied, assert_gravity_applied),
            );
    }
}

/// Runtime backstop that the spawned context carries [`CONTACT_SOFTNESS`] and
/// [`SOLVER_ITERATIONS`] — both live in the same `IntegrationParameters` and are
/// lost together by the same last-write-wins hazard.
fn assert_contact_spring_applied(
    ctx: bevy::ecs::system::Query<
        &bevy_rapier3d::plugin::context::RapierContextSimulation,
        bevy::ecs::query::With<bevy_rapier3d::plugin::context::DefaultRapierContext>,
    >,
) {
    let params = &ctx
        .single()
        .expect("CrabPhysicsPlugin: exactly one default Rapier context")
        .integration_parameters;
    // ONE source for the expected values: project them out of the same
    // initialization the plugin inserts, so a pin edited there can never
    // drift from this backstop (dt is excluded — the fixed-schedule plugin
    // rewrites it, so whole-struct equality would be wrong).
    let RapierContextInitialization::InitializeDefaultRapierContext {
        integration_parameters: expected,
        ..
    } = rapier_context_init()
    else {
        unreachable!("rapier_context_init always initializes a default context");
    };
    let project = |p: &IntegrationParameters| {
        (
            p.contact_softness.natural_frequency,
            p.contact_softness.damping_ratio,
            p.static_contact_softness.natural_frequency,
            p.static_contact_softness.damping_ratio,
            p.friction_in_bias_pass,
            p.normalized_max_linear_velocity,
            p.contact_recycling,
        )
    };
    assert_eq!(
        project(params),
        project(&expected),
        "CrabPhysicsPlugin: the spawned Rapier context lost the contact-semantics \
         pins — its RapierContextInitialization was overridden after the plugin \
         (last-write-wins). The contact physics is silently wrong; fix the init \
         ordering at the call site."
    );
    assert_eq!(
        (
            params.num_solver_iterations,
            params.num_internal_pgs_iterations,
            params.num_internal_stabilization_iterations,
        ),
        SOLVER_ITERATIONS,
        "CrabPhysicsPlugin: the spawned Rapier context lost SOLVER_ITERATIONS — its \
         RapierContextInitialization was overridden after the plugin (last-write-wins). \
         Rest contact silently reverts to chatter (rl#340 stage 2); fix the init \
         ordering at the call site."
    );
}

/// Runtime backstop that the spawned context's gravity is [`PHYSICS_GRAVITY`].
/// Gravity lives on `RapierConfiguration` (not the integration parameters), hence
/// its own query.
fn assert_gravity_applied(
    config: bevy::ecs::system::Query<
        &RapierConfiguration,
        bevy::ecs::query::With<bevy_rapier3d::plugin::context::DefaultRapierContext>,
    >,
) {
    let gravity = config
        .single()
        .expect("CrabPhysicsPlugin: exactly one default Rapier context")
        .gravity;
    assert_eq!(
        gravity, PHYSICS_GRAVITY,
        "CrabPhysicsPlugin: the spawned Rapier context's gravity is not PHYSICS_GRAVITY — \
         its RapierConfiguration was overridden after the plugin (last-write-wins). \
         Gravity is silently wrong; fix the init ordering at the call site."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::headless::headless_app;
    use bevy::prelude::With;
    use bevy_rapier3d::prelude::{DefaultRapierContext, RapierContextSimulation};

    #[test]
    fn contact_spring_is_applied() {
        let mut app = headless_app();
        app.update();
        let mut q = app
            .world_mut()
            .query_filtered::<&RapierContextSimulation, With<DefaultRapierContext>>();
        let ctx = q.single(app.world()).expect("one default rapier context");
        let spring = ctx.integration_parameters.contact_softness;
        assert_eq!(
            spring.natural_frequency, CONTACT_SOFTNESS.natural_frequency,
            "contact spring natural_frequency lost — init ordering broke"
        );
        assert_eq!(
            spring.damping_ratio, CONTACT_SOFTNESS.damping_ratio,
            "contact spring damping_ratio lost — init ordering broke"
        );
        assert_eq!(
            (
                ctx.integration_parameters.num_solver_iterations,
                ctx.integration_parameters.num_internal_pgs_iterations,
                ctx.integration_parameters
                    .num_internal_stabilization_iterations,
            ),
            SOLVER_ITERATIONS,
            "solver iteration counts lost — init ordering broke"
        );
    }

    #[test]
    fn gravity_matches_rapier_default() {
        assert_eq!(
            PHYSICS_GRAVITY,
            RapierConfiguration::new(LENGTH_UNIT).gravity,
            "PHYSICS_GRAVITY diverged from Rapier's default — verify the new value is \
             intentional and resume training; a silent change desyncs the checkpoint."
        );
    }

    #[test]
    fn gravity_is_applied() {
        let mut app = headless_app();
        app.update();
        let mut q = app
            .world_mut()
            .query_filtered::<&RapierConfiguration, With<DefaultRapierContext>>();
        let config = q.single(app.world()).expect("one default rapier context");
        assert_eq!(
            config.gravity, PHYSICS_GRAVITY,
            "active context gravity != PHYSICS_GRAVITY — init ordering broke"
        );
    }

    #[test]
    fn falling_body_is_deterministic() {
        use crate::bot::physics_digest::{DIGEST_SEED, body_bits, fold_bodies};
        use bevy::prelude::{Transform, Vec3};
        use bevy_rapier3d::prelude::{Collider, RigidBody, Velocity};

        const TICKS: usize = 32;
        const START: Vec3 = Vec3::new(1000.0, 500.0, 1000.0);

        fn run() -> (Vec<u64>, Transform, Velocity) {
            let mut app = headless_app();
            let body = app
                .world_mut()
                .spawn((
                    RigidBody::Dynamic,
                    Collider::ball(0.1),
                    Velocity::zero(),
                    Transform::from_translation(START),
                ))
                .id();
            let mut hashes = Vec::with_capacity(TICKS);
            for _ in 0..TICKS {
                app.update();
                let t = *app.world().entity(body).get::<Transform>().unwrap();
                let v = *app.world().entity(body).get::<Velocity>().unwrap();
                hashes.push(fold_bodies(DIGEST_SEED, vec![(0, body_bits(&t, &v))]));
            }
            let t = *app.world().entity(body).get::<Transform>().unwrap();
            let v = *app.world().entity(body).get::<Velocity>().unwrap();
            (hashes, t, v)
        }

        let (a_hashes, a_final, a_vel) = run();
        let (b_hashes, _, _) = run();

        assert_eq!(a_hashes, b_hashes, "free-fall trajectory not reproducible");
        assert!(
            a_final.translation.y < START.y - 1.0,
            "body did not fall: y {} -> {}",
            START.y,
            a_final.translation.y
        );
        assert!(
            a_vel.linear.y < -1.0,
            "downward velocity never built: {a_vel:?}"
        );
        let distinct = a_hashes.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(
            distinct.len(),
            TICKS,
            "state hash repeated across ticks — body wasn't actually moving"
        );
    }
}
