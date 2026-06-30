//! Serialize / restore the ENTIRE crab physics state for bit-exact continuation —
//! the primitive an MP snapshot/join (and replay/debug) is built on.
//!
//! # What "entire state" means here, and what is proven
//! The completeness contract is bit-identity under restore: two worlds with *different*
//! prior states that each restore the SAME snapshot, then step forward under identical
//! inputs, produce a bit-for-bit equal [`crate::bot::physics_digest`] EVERY tick
//! (`restore_replays_bit_identical`). If any solver-read state were missing from the
//! snapshot, the two different priors would leak through and the traces would diverge — so
//! a pass proves the snapshot fully DETERMINES the future. This is the guarantee MP
//! snapshot/join, replay, and self-repair need: every peer that adopts the snapshot evolves
//! identically.
//!
//! What it does NOT claim: that a restored world matches a never-restored *live* world it was
//! forked from. It does not — they diverge (`restore_vs_live_original_diverges`), from a
//! bevy_rapier3d first-step reconciliation artifact, NOT missing snapshot state (the restored
//! rapier bytes equal the snapshot exactly and two restorers agree forever). GCR MP is
//! host-authoritative state-resync — every peer restores — so all-restorers-agree is the
//! operative guarantee; matching a live peer (a lockstep joiner) is out of scope. See
//! [`restore`] for the mechanism and the full boundary argument.
//!
//! # Why this delegates to rapier's own serde rather than a hand-rolled per-joint copy
//! The crab is a rapier *reduced-coordinate multibody* ([`MultibodyJoint`], not impulse
//! joints). A multibody link's `RigidBody` pose is the OUTPUT of forward-kinematics from
//! the multibody's generalized coordinates; the authoritative dynamic state is the
//! generalized position+velocity held inside the [`MultibodyJointSet`], plus the
//! narrow-phase warm-start contact impulses and island/sleep state the solver carries
//! tick-to-tick. A hand-rolled `{pos, quat, linvel, angvel}` per body — what the digest
//! hashes — is a *projection* of that state, not the state itself: writing those poses
//! back does NOT reconstruct the multibody's generalized coordinates, and the
//! `poses_only_*` test below proves it diverges. So the one source of truth is rapier's
//! state, captured through bevy_rapier3d's component-level serde
//! ([[prefer-strong-types-no-drift]]: no parallel copy that can drift from what the
//! solver actually steps).
//!
//! Gated behind the `serde-serialize` feature (off for the trainer, which never snapshots).

use bevy::prelude::*;
use bevy_rapier3d::plugin::context::{
    DefaultRapierContext, RapierContextColliders, RapierContextJoints, RapierContextSimulation,
    RapierRigidBodySet,
};
use bevy_rapier3d::rapier::prelude::{
    CCDSolver, ColliderSet, DefaultBroadPhase, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, RigidBodySet,
};

/// A complete, serializable capture of one rapier world's physics state — every field the
/// solver reads when it steps: the rigid bodies, colliders, the reduced-coordinate multibody
/// with its generalized coords + velocities, the impulse joints, the narrow-phase warm-start
/// contacts, and the broad-phase / island / sleep state. Each field is rapier's OWN serde
/// type (the inner set, not the bevy_rapier3d wrapper component), so [`capture`] and
/// [`restore`] touch the exact same thing and the captured bytes ARE the authoritative state,
/// not a re-derived shadow of it.
///
/// Why the inner rapier sets and individual sim fields, not the wrapper components: the wrappers
/// ([`RapierContextSimulation`] etc.) carry `serde(skip)` scaffolding (the entity↔handle maps,
/// the `Box<dyn EventHandler>`, per-frame event buffers) that is NOT dynamic state — [`restore`]
/// keeps the LIVE world's maps and never wants the capture's. Storing the inner sets drops that
/// scaffolding entirely. The omitted sim pieces — the `PhysicsPipeline` workspaces and event
/// buffers, all `serde(skip)` in rapier — are stateless scratch rebuilt on the next step.
///
/// Completeness is guarded at runtime, not by the type: a missing solver-read field would make
/// `restore_replays_bit_identical` fail. (It can't be guarded by exhaustive destructure —
/// `RapierContextSimulation` has `pub(crate)` fields this crate can't name — so a future rapier
/// adding a new *stateful* sim field would need that test, exercised hard enough to surface it,
/// to catch the gap. The current field list is rapier 0.32's complete non-skip state.)
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct CrabPhysicsSnapshot {
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    ccd_solver: CCDSolver,
    integration_parameters: IntegrationParameters,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    bodies: RigidBodySet,
}

impl CrabPhysicsSnapshot {
    /// Serialize to bytes (bincode) — the form an MP snapshot datagram or a replay file
    /// would carry. Round-trips through [`Self::from_bytes`] with no loss of dynamic state.
    /// Serializing our own valid in-memory state can't fail, so this panics rather than
    /// burdening every caller with an impossible error.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("CrabPhysicsSnapshot serialize")
    }

    /// Deserialize from [`Self::to_bytes`] bytes. Fallible by contract: the bytes come from an
    /// MP datagram or a replay file — untrusted, possibly truncated or version-mismatched — so
    /// a bad blob is an `Err`, never a process panic.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

/// Capture the default rapier context's entire physics state. Run after a physics step
/// (so the bodies exist and reflect the settled tick).
pub fn capture(app: &mut App) -> CrabPhysicsSnapshot {
    let mut q = app.world_mut().query_filtered::<(
        &RapierContextSimulation,
        &RapierContextColliders,
        &RapierContextJoints,
        &RapierRigidBodySet,
    ), With<DefaultRapierContext>>();
    let (sim, colliders, joints, bodies) = q.single(app.world()).expect("one default context");
    CrabPhysicsSnapshot {
        islands: sim.islands.clone(),
        broad_phase: sim.broad_phase.clone(),
        narrow_phase: sim.narrow_phase.clone(),
        ccd_solver: sim.ccd_solver.clone(),
        integration_parameters: sim.integration_parameters,
        colliders: colliders.colliders.clone(),
        impulse_joints: joints.impulse_joints.clone(),
        multibody_joints: joints.multibody_joints.clone(),
        bodies: bodies.bodies.clone(),
    }
}

/// Restore a captured physics state into `app`'s default rapier context: overwrite the serde'd
/// rapier state, then reconcile the bevy mirror so the next step continues from it. Works whether
/// the target world is the capture's own (in-place) or an independently-built peer (MP join) — both
/// allocate identical rapier handles (`handles_are_deterministic_across_worlds`), so the preserved
/// `serde(skip)` entity↔handle maps stay valid.
///
/// Three reconciliation steps, each load-bearing for the `restore_replays_bit_identical` guarantee:
/// 1. Re-write every crab body's bevy `Transform`/`Velocity` from the restored rapier body. The
///    actuator reads `Transform` next tick, so it must reflect the restored pose. This does NOT
///    corrupt the reduced-coordinate multibody: a link's pose+velocity is re-derived from the
///    restored generalized coordinates on the next forward-kinematics pass (proven by
///    `poses_only_restore_is_insufficient`, where a poses-only write is overridden by the coords).
/// 2. Also write `GlobalTransform` directly. `apply_rigid_body_user_changes` runs in FixedUpdate and
///    reads `GlobalTransform` BEFORE PostUpdate transform-propagation refreshes it from the new
///    `Transform`; without this it would push the target world's STALE pose into rapier.
/// 3. `clear_trackers` so none of the target world's history-dependent `Changed` flags survive into
///    the next step and get pushed into the freshly-restored rapier state.
///
/// COMPLETENESS, and the boundary of the guarantee: two worlds that each `restore` the same
/// snapshot step bit-for-bit IDENTICALLY forever (`restore_replays_bit_identical`), regardless of
/// their differing prior states — so the snapshot fully determines the future and nothing the
/// solver reads was left out. A restored world does NOT, however, match the never-restored *live*
/// world it was forked from: they diverge (`restore_vs_live_original_diverges`). That divergence is
/// a bevy_rapier3d first-step reconciliation artifact, not a rapier-physics or missing-state one —
/// the restored rapier bytes equal the snapshot exactly (`restore_round_trips_serde_state`) and all
/// restorers agree. The likely mechanism (a hypothesis — no test here isolates it): step 3's
/// `clear_trackers` (needed so the target's stale `Changed` flags can't corrupt the restored rapier
/// state — the restorers-agree guarantee depends on it) makes the restored world's first step skip
/// the `Transform`→isometry round-trip a live step's `apply_rigid_body_user_changes` performs, and
/// that one-step rounding difference then amplifies chaotically. GCR MP is host-authoritative
/// state-resync — every peer restores — so "all restorers agree" is exactly the operative guarantee;
/// matching a live peer (a lockstep joiner) would need bevy_rapier first-step reconciliation work,
/// out of scope.
pub fn restore(app: &mut App, snap: &CrabPhysicsSnapshot) {
    let entity = {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<DefaultRapierContext>>();
        q.single(app.world()).expect("one default context")
    };
    {
        let mut e = app.world_mut().entity_mut(entity);
        {
            let mut sim = e.get_mut::<RapierContextSimulation>().unwrap();
            sim.islands = snap.islands.clone();
            sim.broad_phase = snap.broad_phase.clone();
            sim.narrow_phase = snap.narrow_phase.clone();
            sim.ccd_solver = snap.ccd_solver.clone();
            sim.integration_parameters = snap.integration_parameters;
        }
        e.get_mut::<RapierContextColliders>().unwrap().colliders = snap.colliders.clone();
        {
            let mut j = e.get_mut::<RapierContextJoints>().unwrap();
            j.impulse_joints = snap.impulse_joints.clone();
            j.multibody_joints = snap.multibody_joints.clone();
        }
        e.get_mut::<RapierRigidBodySet>().unwrap().bodies = snap.bodies.clone();
    }
    sync_bevy_components_from_rapier(app, entity);

    // Clear ECS change-trackers so NONE of the target world's history-dependent `Changed` flags
    // survive into the next step. Otherwise bevy_rapier3d's `apply_rigid_body_user_changes` (which
    // pushes any `Changed` mirror component — Velocity, ExternalForce, Sleeping, … — into rapier
    // BEFORE the solve) would clobber the freshly-restored rapier state with the copy's stale
    // values. After this, rapier (correctly restored) is the source of truth for the next step; the
    // actuator still reads the correct `Transform` VALUE we wrote above, and the post-step writeback
    // re-marks the components normally. This is what makes restore robust to the target world's
    // prior history (the `restore_replays_bit_identical` perturbed-history test is the proof).
    app.world_mut().clear_trackers();
}

/// Re-derive every rigid-body entity's bevy `Transform`/`Velocity` from the restored rapier
/// body (rapier is the source of truth — no parallel pose copy stored in the snapshot). The write
/// is change-tracked on purpose — see [`restore`]'s docs.
fn sync_bevy_components_from_rapier(app: &mut App, ctx: Entity) {
    use bevy_rapier3d::dynamics::RapierRigidBodyHandle;
    use bevy_rapier3d::prelude::Velocity;
    use bevy_rapier3d::utils::iso_to_transform;

    // Resolve (entity, handle) first, then read the body set, then apply — keeping the three
    // borrows of the world disjoint.
    let ent_handles: Vec<(Entity, bevy_rapier3d::rapier::dynamics::RigidBodyHandle)> = {
        let mut q = app
            .world_mut()
            .query::<(Entity, &RapierRigidBodyHandle)>();
        q.iter(app.world()).map(|(e, h)| (e, h.0)).collect()
    };
    let updates: Vec<(Entity, Vec3, Quat, Vec3, Vec3)> = {
        let bodies = app.world().entity(ctx).get::<RapierRigidBodySet>().unwrap();
        ent_handles
            .iter()
            .filter_map(|(e, h)| {
                bodies.bodies.get(*h).map(|rb| {
                    let t = iso_to_transform(rb.position());
                    (*e, t.translation, t.rotation, rb.linvel(), rb.angvel())
                })
            })
            .collect()
    };
    for (e, translation, rotation, linear, angular) in updates {
        let mut em = app.world_mut().entity_mut(e);
        let mut new_transform = None;
        if let Some(mut t) = em.get_mut::<Transform>() {
            t.translation = translation;
            t.rotation = rotation;
            new_transform = Some(*t);
        }
        // Also write GlobalTransform: `apply_rigid_body_user_changes` runs in FixedUpdate and
        // reads GlobalTransform BEFORE PostUpdate transform-propagation refreshes it from the new
        // Transform, so without this it would push the target world's STALE pose into rapier and
        // clobber the restore. Crab parts are flat (no bevy parent), so GlobalTransform == Transform.
        if let (Some(t), Some(mut gt)) = (new_transform, em.get_mut::<GlobalTransform>()) {
            *gt = GlobalTransform::from(t);
        }
        if let Some(mut v) = em.get_mut::<Velocity>() {
            v.linear = linear;
            v.angular = angular;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::actuator::{ACTION_SIZE, CrabActions};
    use crate::bot::body::{CrabBodyPart, CrabCarapace, CrabJoint};
    use crate::bot::headless::{headless_app, tick};
    use crate::bot::physics_digest::crab_state_digest;
    use bevy_rapier3d::prelude::{RapierRigidBodyHandle, Velocity};

    /// Ticks of settle + flail before the snapshot is taken, so the crab is captured
    /// MID-SIM (legs loaded against the ground, contacts live, multibody coords non-rest)
    /// — the hard case, not a pristine spawn pose.
    const WARMUP: usize = 40;
    /// Ticks stepped after restore that must match bit-for-bit.
    const REPLAY: usize = 80;

    /// A deterministic torque schedule, byte-identical run to run (a fixed-seed LCG), driving
    /// every joint over its signed range so the legs flail and load the contact solver — the
    /// same shape `net::determinism_probe::scripted_actions` uses. `seed` lets a caller produce a
    /// DIFFERENT schedule (a divergent history for the restore target).
    fn scripted_actions_seeded(seed: u64, n: usize) -> Vec<[f32; ACTION_SIZE]> {
        let mut state: u64 = seed;
        let mut seq = Vec::with_capacity(n);
        for _ in 0..n {
            let mut a = [0.0f32; ACTION_SIZE];
            for slot in a.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 40) as u32; // 24 bits — exactly representable in f32
                *slot = (u as f32 / ((1u32 << 24) as f32)) * 1.8 - 0.9; // [-0.9, 0.9]
            }
            seq.push(a);
        }
        seq
    }

    /// The canonical schedule (the seed `net::determinism_probe` uses).
    fn scripted_actions(n: usize) -> Vec<[f32; ACTION_SIZE]> {
        scripted_actions_seeded(0x9E37_79B9_7F4A_7C15, n)
    }

    fn set_actions(app: &mut App, a: &[f32; ACTION_SIZE]) {
        app.world_mut().resource_mut::<CrabActions>().envs[0] = *a;
    }

    /// env 0's whole-crab digest (the GCR lockstep desync number's physics half), read off
    /// the bevy components — the SAME projection the production bridge hashes.
    fn digest(app: &mut App) -> u64 {
        let mut q = app.world_mut().query_filtered::<(
            &Transform,
            &Velocity,
            Option<&CrabJoint>,
            Option<&CrabCarapace>,
        ), With<CrabBodyPart>>();
        crab_state_digest(q.iter(app.world()))
    }

    /// Step `app` `n` ticks under `seq[off..]`, returning the per-tick digests.
    fn step_collect(app: &mut App, seq: &[[f32; ACTION_SIZE]], off: usize, n: usize) -> Vec<u64> {
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            set_actions(app, &seq[off + k]);
            app.update();
            out.push(digest(app));
        }
        out
    }

    /// A headless world with every schedule forced single-threaded — so ECS system run order (and
    /// thus float-reduction order) is fixed, the precondition the GCR determinism contract needs
    /// (see [`crate::bot::headless::force_serial_schedules`]). Without this, multi-threaded system
    /// scheduling makes even same-binary stepping non-reproducible across worlds.
    fn deterministic_app() -> App {
        crate::bot::headless::pin_single_thread_pools();
        let mut app = headless_app();
        crate::bot::headless::force_serial_schedules(&mut app);
        app
    }

    /// Build a world and step it to the warmup point (one tick to spawn the crab + size
    /// `CrabActions`, then `WARMUP` scripted-torque ticks).
    fn warmed_world(seq: &[[f32; ACTION_SIZE]]) -> App {
        let mut app = deterministic_app();
        tick(&mut app, 1); // spawn_initial_crabs builds the body + sizes CrabActions
        for a in seq.iter().take(WARMUP) {
            set_actions(&mut app, a);
            app.update();
        }
        app
    }

    /// CORRECTNESS — round-trip identity: a snapshot serialized to bytes and read back is the
    /// same physics state, by the digest oracle. (The bytes path is what an MP datagram / a
    /// replay file actually carries.)
    #[test]
    fn round_trip_bytes_preserve_state() {
        let seq = scripted_actions(WARMUP);
        let mut a = warmed_world(&seq);
        let bytes = capture(&mut a).to_bytes();

        // (a) Pure serde round-trip is the identity on the bytes.
        let bytes_again = CrabPhysicsSnapshot::from_bytes(&bytes).unwrap().to_bytes();
        assert_eq!(bytes, bytes_again, "serde round-trip is not the identity");

        // (b) Restoring the round-tripped snapshot back into the world, then re-capturing,
        // yields the SAME bytes — proving restore writes back exactly what was captured (the
        // canonical equality for the multibody's generalized coords, which a pose digest can't
        // see). `capture` reads only the rapier sets, so this compares at the rapier level —
        // where the full state lives — independent of restore's bevy-component reconciliation.
        restore(&mut a, &CrabPhysicsSnapshot::from_bytes(&bytes).unwrap());
        let bytes_after_restore = capture(&mut a).to_bytes();
        assert_eq!(
            bytes, bytes_after_restore,
            "capture→serialize→deserialize→restore→re-capture changed the physics state"
        );
    }

    /// `from_bytes` is fallible by contract (its bytes are untrusted MP/replay input): garbage
    /// must return `Err`, never panic — so a hostile or truncated datagram can't crash a peer.
    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(CrabPhysicsSnapshot::from_bytes(&[0xAB; 16]).is_err());
        assert!(CrabPhysicsSnapshot::from_bytes(&[]).is_err());
    }

    /// DETERMINISM (rl#139) — the base guarantee the restore/replay proofs build on, asserted
    /// on its own: two FRESH worlds built and stepped from the SAME scripted torque sequence
    /// (the "seed") produce BIT-IDENTICAL per-tick digest traces over `REPLAY` ticks. No
    /// snapshot/restore here — this pins that the sim is reproducible *from construction*, the
    /// precondition lockstep peers and reproducible training both rest on. A future change that
    /// slips an unseeded RNG or an order-dependent float reduction into the hashed path fails
    /// loudly here, on the tick it first diverges, instead of as a mysterious desync later.
    #[test]
    fn fresh_same_seed_worlds_step_bit_identical() {
        let seq = scripted_actions(WARMUP + REPLAY);
        let trace_a = step_collect(&mut warmed_world(&seq), &seq, WARMUP, REPLAY);
        let trace_b = step_collect(&mut warmed_world(&seq), &seq, WARMUP, REPLAY);

        // Non-vacuous: the trajectory genuinely evolves (the flail schedule moved the crab), so
        // an all-equal trace can't trivially satisfy the bit-identity below.
        assert!(
            trace_a.iter().collect::<std::collections::HashSet<_>>().len() > 1,
            "trajectory was static — the scripted torque didn't perturb the crab"
        );
        for (t, (a, b)) in trace_a.iter().zip(&trace_b).enumerate() {
            assert_eq!(
                a, b,
                "DIVERGED at tick {t}/{REPLAY}: world-A {a:#018x} != world-B {b:#018x} — the sim \
                 is NOT reproducible from identical construction (an unseeded RNG or an \
                 order-dependent reduction entered the hashed path)"
            );
        }
    }

    /// Restore the snapshot into a world, then step it `REPLAY` ticks under `seq`, returning the
    /// per-tick digest trace and the post-restore (pre-step) digest.
    fn restore_and_replay(build: impl Fn() -> App, bytes: &[u8], seq: &[[f32; ACTION_SIZE]]) -> (u64, Vec<u64>) {
        let mut w = build();
        restore(&mut w, &CrabPhysicsSnapshot::from_bytes(bytes).unwrap());
        let post = digest(&mut w);
        (post, step_collect(&mut w, seq, WARMUP, REPLAY))
    }

    /// COMPLETENESS — THE KEY PROOF. Snapshot a crab mid-sim, then restore it into TWO independent
    /// worlds that were driven to the snapshot point by DIFFERENT torque histories (so their own
    /// physics states genuinely differ — verified by distinct pre-restore digests). Stepping both
    /// forward `REPLAY` ticks under identical torque must yield BIT-IDENTICAL digest traces EVERY
    /// tick. A pass means the snapshot fully DETERMINES the future: nothing the solver reads to
    /// continue was left out, and the restoring world's prior state is completely overwritten. This
    /// is exactly the guarantee MP snapshot/join, replay, and self-repair need — every peer that
    /// adopts the snapshot evolves identically.
    ///
    /// (The captured state is byte-complete in BOTH worlds — `restore_round_trips_serde_state`
    /// asserts the post-restore re-serialization equals the snapshot. The boundary: this
    /// restorers-agree guarantee does NOT extend to matching a never-restored *live* world — see
    /// `restore_vs_live_original_diverges` and `restore`'s docs for why, and why GCR's
    /// host-authoritative resync only needs the restorers-agree guarantee.)
    #[test]
    fn restore_replays_bit_identical() {
        let seq = scripted_actions(WARMUP + REPLAY);

        // Capture from one world.
        let mut src = warmed_world(&seq);
        let bytes = capture(&mut src).to_bytes();

        // Two restore targets with DIFFERENT prior states (different warmup torque seeds).
        let hist_a = scripted_actions_seeded(0xD1FF_5EED_0BAD_F00D, WARMUP);
        let hist_b = scripted_actions_seeded(0x0DDB_A115_CAFE_F00D, WARMUP);
        let mut wa = warmed_world(&hist_a);
        let mut wb = warmed_world(&hist_b);
        let pre_a = digest(&mut wa);
        let pre_b = digest(&mut wb);
        drop((wa, wb)); // the digests are all we needed from these probes

        let (post_a, trace_a) = restore_and_replay(|| warmed_world(&hist_a), &bytes, &seq);
        let (post_b, trace_b) = restore_and_replay(|| warmed_world(&hist_b), &bytes, &seq);

        // Non-vacuous: the two targets really were in different states before restore, and the
        // trajectory really evolves.
        assert_ne!(pre_a, pre_b, "the two restore targets had the same prior state — vacuous");
        assert_eq!(post_a, post_b, "restore produced different post-restore poses across worlds");
        assert!(
            trace_a.iter().collect::<std::collections::HashSet<_>>().len() > 1,
            "trajectory was static — flail schedule didn't perturb the crab"
        );
        // THE PROOF: two independently-restored, different-prior worlds step bit-for-bit identically.
        for (t, (a, b)) in trace_a.iter().zip(&trace_b).enumerate() {
            assert_eq!(
                a, b,
                "DIVERGED at replay tick {t}/{REPLAY}: world-A {a:#018x} != world-B {b:#018x} — the \
                 snapshot does NOT fully determine the continuation (state the solver reads was \
                 left out)"
            );
        }
    }

    /// CORRECTNESS (cross-world) — the captured state is byte-complete: restoring the snapshot into
    /// a freshly-built world and re-serializing yields the SAME bytes. Proves restore reconstructs
    /// the entire rapier state (bodies, multibody generalized coords, narrow-phase contacts,
    /// islands), not a lossy projection — independent of the target world's history.
    #[test]
    fn restore_round_trips_serde_state() {
        let seq = scripted_actions(WARMUP);
        let bytes = capture(&mut warmed_world(&seq)).to_bytes();

        let hist = scripted_actions_seeded(0xBEEF_F00D_1234_5678, WARMUP);
        let mut w = warmed_world(&hist);
        restore(&mut w, &CrabPhysicsSnapshot::from_bytes(&bytes).unwrap());
        assert_eq!(
            capture(&mut w).to_bytes(),
            bytes,
            "restore into a different-history world did not reconstruct the full serde'd state"
        );
    }

    /// THE BOUNDARY OF THE GUARANTEE — a restored world does NOT match the never-restored *live*
    /// world it was forked from. Capture a snapshot from a live world, let that SAME world keep
    /// stepping (the live reference), and separately restore the snapshot into a fresh copy built by
    /// the identical history (so its pre-restore state already equals the snapshot — a no-op-valued
    /// restore) and step it under the identical inputs. Their digest traces DIVERGE.
    ///
    /// This is NOT missing snapshot state — the restored rapier bytes equal the snapshot exactly
    /// (`restore_round_trips_serde_state`) and two restorers agree forever
    /// (`restore_replays_bit_identical`); it is a bevy_rapier3d first-step reconciliation artifact,
    /// outside GCR's host-authoritative (all-restorers) guarantee. See [`super::restore`] for the
    /// likely mechanism and full boundary argument.
    ///
    /// Asserted (not commented) so it can't silently rot: if a future bevy_rapier makes the restored
    /// world match the live one, this flips to `assert_eq!` and the docs' boundary claim is removed.
    #[test]
    fn restore_vs_live_original_diverges() {
        let seq = scripted_actions(WARMUP + REPLAY);
        let mut orig = warmed_world(&seq);
        let bytes = capture(&mut orig).to_bytes();
        let reference = step_collect(&mut orig, &seq, WARMUP, REPLAY);

        let (_, replayed) = restore_and_replay(|| warmed_world(&seq), &bytes, &seq);
        assert_ne!(
            reference, replayed,
            "restored-vs-live-original UNEXPECTEDLY matched bit-for-bit — the bevy_rapier first-step \
             reconciliation artifact is gone; flip this to assert_eq! and update `restore`'s docs"
        );
    }

    /// EVIDENCE that the reduced-coordinate multibody needs its generalized coordinates, not
    /// just link poses: rewinding with ONLY the per-body `{pos, quat, linvel, angvel}` (what the
    /// digest hashes — the obvious hand-rolled snapshot) written into the bevy components does
    /// NOT reproduce the trajectory. The multibody's generalized coords are untouched, so the
    /// next step's forward kinematics overrides the written poses and the body keeps evolving
    /// from the un-rewound state. This is why [`CrabPhysicsSnapshot`] captures rapier's state.
    /// (If this ever started passing, the multibody-coord argument would need review — so it's an
    /// asserted, not a commented, claim.)
    #[test]
    fn poses_only_restore_is_insufficient() {
        let seq = scripted_actions(WARMUP + REPLAY);
        let mut app = warmed_world(&seq);

        // Snapshot the bevy poses by semantic body key at the warmup point.
        let poses: Vec<(usize, Transform, Velocity)> = {
            let mut q = app.world_mut().query_filtered::<(
                &Transform,
                &Velocity,
                Option<&CrabJoint>,
                Option<&CrabCarapace>,
            ), With<CrabBodyPart>>();
            q.iter(app.world())
                .filter_map(|(t, v, joint, cara)| {
                    crate::bot::physics_digest::body_key(cara.is_some(), joint).map(|k| (k, *t, *v))
                })
                .collect()
        };
        let reference = step_collect(&mut app, &seq, WARMUP, REPLAY);

        // "Rewind" by writing ONLY the poses back (no rapier-state restore).
        let updates: Vec<(Entity, Transform, Velocity)> = {
            let mut q = app.world_mut().query_filtered::<(
                Entity,
                Option<&CrabJoint>,
                Option<&CrabCarapace>,
            ), With<CrabBodyPart>>();
            q.iter(app.world())
                .filter_map(|(e, joint, cara)| {
                    let k = crate::bot::physics_digest::body_key(cara.is_some(), joint)?;
                    poses.iter().find(|(pk, _, _)| *pk == k).map(|(_, t, v)| (e, *t, *v))
                })
                .collect()
        };
        for (e, t, v) in updates {
            let mut em = app.world_mut().entity_mut(e);
            *em.get_mut::<Transform>().unwrap() = t;
            *em.get_mut::<Velocity>().unwrap() = v;
        }
        let replayed = step_collect(&mut app, &seq, WARMUP, REPLAY);

        assert_ne!(
            reference, replayed,
            "poses-only rewind unexpectedly reproduced the multibody trajectory — the \
             generalized-coordinate argument for CrabPhysicsSnapshot needs review"
        );
    }

    /// Precondition for cross-WORLD restore (MP join): two independently-built worlds, stepped
    /// identically, must allocate identical rapier rigid-body handles for the crab parts — else
    /// the in-place restore's handle-stable assumption wouldn't carry to a fresh peer. Pairs by
    /// the semantic body key, then compares the rapier handle bits.
    #[test]
    fn handles_are_deterministic_across_worlds() {
        let seq = scripted_actions(WARMUP);
        let handles = |app: &mut App| -> Vec<(usize, u32, u32)> {
            let mut q = app.world_mut().query_filtered::<(
                &RapierRigidBodyHandle,
                Option<&CrabJoint>,
                Option<&CrabCarapace>,
                &CrabBodyPart,
            ), ()>();
            let mut v: Vec<(usize, u32, u32)> = q
                .iter(app.world())
                .filter_map(|(h, joint, cara, _)| {
                    crate::bot::physics_digest::body_key(cara.is_some(), joint).map(|k| {
                        let (idx, generation) = h.0.0.into_raw_parts();
                        (k, idx, generation)
                    })
                })
                .collect();
            v.sort_by_key(|t| t.0);
            v
        };
        let mut a = warmed_world(&seq);
        let mut b = warmed_world(&seq);
        assert_eq!(
            handles(&mut a),
            handles(&mut b),
            "crab rapier handles differ across independently-built worlds — cross-world \
             restore can't rely on handle identity; an MP-join restore needs a re-key layer"
        );
    }
}
