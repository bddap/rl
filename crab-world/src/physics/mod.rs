pub mod snapshot;
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
/// green 3/3 in the same environment — a reading on the pre-rl#406 mountainside
/// ruler, so treat the magnitude as indicative; the binding disqualifier is
/// next), and the rl#312 actuator-load
/// interpenetration residual busts its 25 mm cap (28.05 mm). Iteration raises —
/// the sanctioned compensation — don't move either wall, so the cost stays
/// paid at 2.
pub const PHYSICS_SUBSTEPS: usize = 2;

/// (outer, internal-PGS, internal-stabilization) solver iterations — the other
/// half of the [`PHYSICS_SUBSTEPS`] driven budget (rationale there).
///
/// Internal PGS at 12 (rl#332): the velocity rows of a stop-railed, saturated-drive
/// light link (drive + limit + contacts on a 2–50 g body under a 0.72 kg carapace)
/// do not converge in 2 PGS passes, and the residual leaves the solve as a
/// one-tick kick — a part going 0.2→8 m/s while the carapace walks at 0.3 m/s,
/// the rl#332 F3 onset shape. One-tick state replays (`game sally-replay`, same
/// state and drives, only the counts varied) put the kicked link at 8.1 m/s for
/// PGS 2, 1.9–4.8 for 8, 1.2–2.4 for 12–16, ~1 for 32 and for (32,8,8)×8 — the
/// converged answer; the 768-tick driven audit (`solver_variant_matrix`) counted
/// 14/4/1/0/0/0 kicks for PGS 2/4/6/8/12/16 at outer 2, while outer 3–4 at PGS 2
/// still kicked (11, 7) and softening the limit springs instead made it WORSE
/// (76 kicks, 0.16–0.74 rad stop sag). PGS passes are cheap: `game step-profile`
/// measured +0.1–0.2 ms/substep for 2→8 and a further ~+0.15 for 8→12 (p10,
/// vel-resolution), under 1 ms/step total against a 16.7 ms frame. 12 is where
/// every observed kick state converged below the detector; 8 left one at 4.8 m/s.
/// `driven_crab_energy_ledger_holds_on_shipped_solver` pins it.
///
/// The internal counts were at 2 for DRIVEN momentum honesty, not rest
/// quiet: at 1/1 the airborne self-contact thrash leaks 0.31–0.58 m/s² of
/// phantom COM force across realization draws — straddling the rl#321 0.5
/// ceiling and approaching the pre-fix 0.64 scale — while 2/2 measures
/// 0.25–0.39 with real margin, for <1 ms/tick over 1/1 (4/4 buys nothing
/// further; 2/2 is the knee). Outer 4 (rl#351): on the frozen hull checkpoint
/// the limit-saturated multibody solve (21–36 of 38 joints at a stop, on ground
/// contact) blows up in one tick at outer 2 — 13 rl#343 trips in 36.2M rollout
/// ticks against 2 at outer 4, matched seeds, one-sided p = 0.0037
/// (`docs/evidence/chase-archaeology/integrity-lever-ab-36m`), and every trip
/// hard-fails a training run. ×4 substeps also cut it to 2 trips, at 1.54× the
/// rollout CPU per tick against outer 4's 1.33×. The price is the frame budget
/// rl#396 stage 4 bought by dropping to outer 2 (5.4→3.5 ms/substep on the GCR
/// driven crab, which had put the step-carrying TV frame at the 16.7 ms vsync
/// line): the headless driven step p50 goes 6.8→10.4 ms on the training host.
/// Stabilization at 3 (not 2) was measured at outer 2, where it bought back the
/// CONTACT-DEPTH convergence the outer cut halved, and 3 was the whole corridor: at 2×(2/2/2) the rl#312
/// actuator-load interpenetration residual crossed its 25 mm/60-tick caps
/// (27.6 mm, 66 ticks — vs main's worst-observed 13.9 mm/8 ticks). The stab-4
/// disqualifier ("REST crab creeps 12 m") is RETRACTED (rl#406): that ruler was
/// the old mountainside armed smoke, where a zero-drive crab legitimately
/// slides 12-18 m at EVERY solver mix tried ((2,2,3) and (4,2,2) measured
/// overlapping bands) and the verdict flipped on float perturbations as small
/// as sally.glb's visual entities existing — so stab 3-vs-4 was never actually
/// discriminated; 3 stands on "sufficient for the rl#312 caps, extra buys
/// nothing measured". Stab sweeps are nearly free (3.51→3.54 ms/substep for
/// 2→4) because PGS+assembly own the solver's cost. A ZERO-DRIVE crab still
/// gets `CRAB_SETTLE_EXTRA_ITERATIONS` ADDED to the outer count, so its
/// settle total is 16 — the total spawn.rs's floor was set against; sleep engagement is pinned by
/// `resting_crab_falls_asleep` (headless graph) and the flat-ground armed
/// smoke's sleep bound (render graph, rl#406).
pub const SOLVER_ITERATIONS: (usize, usize, usize) = (4, 12, 3);

/// One physics step's solver budget: [`SOLVER_ITERATIONS`]-shaped counts per substep
/// and substeps per [`PHYSICS_DT`]. Diagnostics vary it (`sally-replay`, `rl-train
/// repro --solver`); the plant ships [`SHIPPED_SOLVER`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverCounts {
    pub iterations: (usize, usize, usize),
    pub substeps: usize,
}

pub const SHIPPED_SOLVER: SolverCounts = SolverCounts {
    iterations: SOLVER_ITERATIONS,
    substeps: PHYSICS_SUBSTEPS,
};

impl std::fmt::Display for SolverCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (o, p, st) = self.iterations;
        write!(f, "{o},{p},{st}x{}", self.substeps)
    }
}

/// `OUTER,PGS,STAB[xSUBSTEPS]`, substeps defaulting to shipped; `Display` writes the
/// full form.
impl std::str::FromStr for SolverCounts {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let bad = || format!("solver {s:?}: want OUTER,PGS,STAB[xSUBSTEPS], e.g. 4,12,3x2");
        let (counts, substeps) = match s.split_once('x') {
            Some((c, n)) => (c, n.parse().map_err(|_| bad())?),
            None => (s, PHYSICS_SUBSTEPS),
        };
        let n: Vec<usize> = counts
            .split(',')
            .map(|v| v.trim().parse().map_err(|_| bad()))
            .collect::<Result<_, _>>()?;
        match n[..] {
            [o, p, st] if o > 0 && substeps > 0 => Ok(Self {
                iterations: (o, p, st),
                substeps,
            }),
            _ => Err(bad()),
        }
    }
}

pub fn solver_timestep(solver: SolverCounts) -> TimestepMode {
    TimestepMode::Fixed {
        dt: PHYSICS_DT,
        substeps: solver.substeps,
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

#[derive(Clone, serde::Serialize)]
pub(crate) struct PhysicsParameters {
    pub dt: f32,
    pub substeps: u64,
    pub integration: IntegrationParameters,
    pub gravity: [f32; 3],
    pub length_unit: f32,
}

pub(crate) fn identity_parameters() -> PhysicsParameters {
    let RapierContextInitialization::InitializeDefaultRapierContext {
        integration_parameters,
        rapier_configuration,
    } = rapier_context_init()
    else {
        unreachable!()
    };
    PhysicsParameters {
        dt: PHYSICS_DT,
        substeps: PHYSICS_SUBSTEPS as u64,
        integration: integration_parameters,
        gravity: rapier_configuration.gravity.to_array(),
        length_unit: LENGTH_UNIT,
    }
}

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
        app.insert_resource(solver_timestep(SHIPPED_SOLVER))
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
    fn solver_counts_parse_both_forms() {
        assert_eq!(
            SHIPPED_SOLVER.to_string().parse::<SolverCounts>(),
            Ok(SHIPPED_SOLVER),
            "the CLI default is the Display form; it must parse back"
        );
        assert_eq!(
            "4,12,3x2".parse::<SolverCounts>(),
            Ok(SolverCounts {
                iterations: (4, 12, 3),
                substeps: 2
            })
        );
        assert_eq!(
            "8,4,4".parse::<SolverCounts>().map(|s| s.substeps),
            Ok(PHYSICS_SUBSTEPS)
        );
        for bad in ["", "2,12", "2,12,3,4", "0,12,3", "2,12,3x0", "a,b,c"] {
            assert!(bad.parse::<SolverCounts>().is_err(), "{bad:?} parsed");
        }
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
