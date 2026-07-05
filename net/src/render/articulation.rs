//! Render-side capture + apply for the crab pose extension frame — the render-only halves
//! of [`crate::articulation`].
//!
//! Under host-authority the windowed HOST runs every rapier Sally; a windowed remote CLIENT
//! renders the host's exact poses without simulating physics ([[silent-fallback-antipattern]]: no
//! second "run my own crab for visuals" path). So the host [`capture`]s each live crab's per-part
//! transforms + giant-blow-up placement each stepped tick — one [`CrabFrame`] per env, in env
//! order (rl#200: a multi-brain round runs several crabs) — and broadcasts them beside the
//! snapshot; the client [`apply`]s each frame onto its OWN matching env's crab entities — which
//! it spawns but never physics-steps, so the writes stick and
//! `crab_world::bot::skin::drive_bones` skins each mesh to the host's pose.
//!
//! Only the parts a skin bone actually follows are carried: those with a stable `PartId` (each
//! actuated joint link + the carapace). The cosmetic eye-stalk links carry no `PartId` and no bone
//! keys off them (their bones ride the carapace), so they need no sync — and couldn't be matched
//! across the wire without a stable key anyway.

use bevy::prelude::*;

use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId};
use crab_world::bot::skin::{CrabSkinRepose, SkinRepose};
use crab_world::crab_view::CrabBrainLabels;
use crab_world::vehicle::Vehicle;

use crate::articulation::{CrabArticulation, CrabFrame, PartTransform, ReposeWire, VehiclePoseWire};

/// (Client) The host's piloted craft's arena-frame pose off the wire, `None` while the host is on
/// foot. A remote client runs none of the vehicle's rapier (host-authoritative), so this mirror is
/// all it knows of the craft; `render_mode`'s wireframe drawer renders it (rl#192). Written by
/// [`apply`] each adopted tick; stays `None` on the host/solo, whose craft is the live body.
#[derive(Resource, Default)]
pub(super) struct RemoteVehicle(pub(super) Option<VehiclePoseWire>);

/// The part tag is `1 + joint.index()` in a `u8`, so the joint set must stay small enough that the
/// tag can't wrap — pin it at compile time (COUNT is 38 today; the wire format would need a rev long
/// before this fired, but the invariant is why the tag is a `u8`).
const _: () = assert!(CrabJointId::COUNT < u8::MAX as usize);

/// Wire tag for a body part's identity: `0` = the carapace, `1 + joint.index()` = an actuated
/// joint's link. Host and client compute it identically from the same rig, so a transform matches
/// its own part entity across the wire. `None` for an unkeyed part (an eye-stalk) — not a skin
/// drive target, so it is skipped rather than mis-matched.
fn part_tag(is_carapace: bool, joint: Option<&CrabJoint>) -> Option<u8> {
    match (is_carapace, joint) {
        (true, _) => Some(0),
        (_, Some(j)) => Some(1 + j.id.index() as u8),
        _ => None,
    }
}

/// (Host) Snapshot every crab's render pose for `tick`: per env, each keyed body part's
/// arena-frame transform (world-space — the parts are top-level, so `Transform` already is) plus
/// that crab's current giant-blow-up placement. Frames are emitted for envs `0..n_crabs`
/// contiguously (`n_crabs` = the bridge's binding count via [`CrabSkinRepose`]'s keys ∪ spawned
/// envs), so the frame index IS the crab index on the wire. Called right after
/// `Server::step_next`, so it is this tick's settled pose (`integrate_crab`/
/// `publish_skin_repose` ran during the physics pump). A frame's `repose` is `None` only before
/// the bridge has published one for that crab (transiently at spawn).
pub(super) fn capture(world: &mut World, tick: u64) -> CrabArticulation {
    // Group each env's keyed parts. BTreeMap so envs emit in index order.
    let mut by_env: std::collections::BTreeMap<usize, Vec<PartTransform>> = Default::default();
    let mut q = world.query_filtered::<(
        &Transform,
        &CrabEnvId,
        Option<&CrabJoint>,
        Option<&CrabCarapace>,
    ), With<CrabBodyPart>>();
    for (t, env, joint, carapace) in q.iter(world) {
        let Some(tag) = part_tag(carapace.is_some(), joint) else {
            continue;
        };
        by_env.entry(env.0).or_default().push(PartTransform {
            part: tag,
            pos: t.translation.to_array(),
            rot: t.rotation.to_array(),
        });
    }

    let reposes = world
        .get_resource::<CrabSkinRepose>()
        .map(|r| r.0.clone())
        .unwrap_or_default();

    // The host's own on-screen labels, published from its brain bindings
    // (`external_crab`'s `publish_brain_labels`) — shipped verbatim so every client renders
    // the host's exact who's-who strings (rl#200 increment 7).
    let labels = world
        .get_resource::<CrabBrainLabels>()
        .map(|l| l.0.clone())
        .unwrap_or_default();

    // One frame per env, contiguous from 0 — the wire's index IS the crab index. Spawned envs
    // are contiguous by construction (`spawn_initial_crabs` fills 0..NumEnvs), so this covers
    // exactly the armed crabs.
    let n_crabs = by_env.keys().last().map_or(0, |&max| max + 1);
    let crabs = (0..n_crabs)
        .map(|env| {
            let mut parts = by_env.remove(&env).unwrap_or_default();
            // Ascending tag order — a deterministic wire, and the client matches by tag anyway.
            parts.sort_by_key(|p| p.part);
            let repose = reposes.get(&env).map(|s| ReposeWire {
                shift: s.shift.to_array(),
                pivot: s.pivot.to_array(),
                scale: s.scale,
            });
            CrabFrame {
                parts,
                repose,
                brain_label: labels.get(env).cloned().unwrap_or_default(),
            }
        })
        .collect();

    // The HOST's own piloted craft, if it is flying one — pilot 0 by construction (the host holds
    // PlayerId(0)); despawned on foot, so the query itself is the presence signal. The wire still
    // carries just this one craft — per-pilot poses for remote pilots' crafts are the rl#191
    // articulation rev. Its `Transform` is arena-frame like the parts (a top-level rapier body).
    let vehicle = world
        .query::<(&Transform, &Vehicle)>()
        .iter(world)
        .find(|(_, v)| v.pilot == crab_world::vehicle::PilotId(0))
        .map(|(t, _)| VehiclePoseWire {
            pos: t.translation.to_array(),
            rot: t.rotation.to_array(),
        });

    CrabArticulation {
        tick,
        crabs,
        vehicle,
    }
}

/// (Client) Write a received crab pose onto each env's own crab render entities — overwriting
/// every keyed part's `Transform` and each crab's giant-blow-up placement, routing frame `i` to
/// env `i`. The client never pumps the crab physics, so these writes are not fought back by the
/// rapier solver and persist for `crab_world::bot::skin::drive_bones` (PostUpdate) to skin the
/// meshes from. A part in a frame with no matching local entity (or vice versa, including a
/// whole env the client hasn't spawned) is simply skipped — the rigs are identical on both
/// peers, so this only elides the transient pre-spawn frames.
pub(super) fn apply(world: &mut World, art: &CrabArticulation) {
    let by_env_tag: std::collections::HashMap<(usize, u8), &PartTransform> = art
        .crabs
        .iter()
        .enumerate()
        .flat_map(|(env, frame)| frame.parts.iter().map(move |p| ((env, p.part), p)))
        .collect();

    let mut q = world.query_filtered::<(
        &mut Transform,
        &CrabEnvId,
        Option<&CrabJoint>,
        Option<&CrabCarapace>,
    ), With<CrabBodyPart>>();
    for (mut t, env, joint, carapace) in q.iter_mut(world) {
        let Some(tag) = part_tag(carapace.is_some(), joint) else {
            continue;
        };
        if let Some(p) = by_env_tag.get(&(env.0, tag)) {
            t.translation = Vec3::from_array(p.pos);
            t.rotation = Quat::from_array(p.rot);
        }
    }

    if let Some(mut repose) = world.get_resource_mut::<CrabSkinRepose>() {
        // MERGE, never replace wholesale: a frame's `None` repose is "not published this
        // tick" (transient — spawn, or the rescue tick that clears the carapace sample), and
        // the documented contract is the client LEAVES its placement untouched then. Wiping
        // the entry would snap that crab to identity (arena scale at the arena origin) for a
        // frame — a visible glitch for an honest one-tick gap.
        for (env, frame) in art.crabs.iter().enumerate() {
            if let Some(r) = frame.repose {
                repose.0.insert(
                    env,
                    SkinRepose {
                        shift: Vec3::from_array(r.shift),
                        pivot: Vec3::from_array(r.pivot),
                        scale: r.scale,
                    },
                );
            }
        }
        // A crab the host no longer carries at all (a smaller adopted round) does drop out.
        repose.0.retain(|env, _| *env < art.crabs.len());
    }

    // Adopt the host's brain labels (write-on-change so the shared label UI only reconciles
    // when something actually changed). Like the parts, the client renders these verbatim —
    // it never re-derives who's who.
    let labels: Vec<String> = art.crabs.iter().map(|f| f.brain_label.clone()).collect();
    if world.get_resource::<CrabBrainLabels>().map(|l| &l.0) != Some(&labels) {
        world.insert_resource(CrabBrainLabels(labels));
    }

    // Mirror the host's piloted craft — including `None` (the host stepped out; a stale mirror
    // would freeze a ghost craft mid-air). Insert rather than get-mut so the mirror exists from
    // the first adopted frame.
    world.insert_resource(RemoteVehicle(art.vehicle));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab_world::bot::body::{CrabJointId, Side};

    /// Spawn one env's crab's keyed render parts — a carapace + one joint link — at the given
    /// transforms, the minimal rig [`capture`]/[`apply`] key off (no physics/GPU needed).
    fn spawn_parts(
        world: &mut World,
        env: usize,
        carapace: Transform,
        joint_id: CrabJointId,
        joint: Transform,
    ) -> (Entity, Entity) {
        let cara = world
            .spawn((CrabBodyPart, CrabCarapace, CrabEnvId(env), carapace))
            .id();
        let jnt = world
            .spawn((
                CrabBodyPart,
                CrabJoint {
                    id: joint_id,
                    axis_local: Vec3::X,
                },
                CrabEnvId(env),
                joint,
            ))
            .id();
        (cara, jnt)
    }

    #[test]
    fn capture_then_apply_reproduces_the_hosts_exact_pose_per_crab() {
        let joint_id = CrabJointId::ClawShoulder(Side::Left);
        let cara_t = Transform::from_xyz(1.0, 2.0, 3.0).with_rotation(Quat::from_rotation_y(0.5));
        let joint_t =
            Transform::from_xyz(-4.0, 0.25, 9.0).with_rotation(Quat::from_rotation_x(0.3));
        // A SECOND crab (rl#200) at distinct poses, so cross-env routing is actually pinned —
        // a capture/apply that collapsed envs would smear one crab onto the other.
        let cara_t1 = Transform::from_xyz(9.0, 1.0, -6.0).with_rotation(Quat::from_rotation_y(1.2));
        let joint_t1 =
            Transform::from_xyz(8.0, 0.5, -5.0).with_rotation(Quat::from_rotation_x(0.9));

        // HOST: known part poses per env + a published env-0 placement + a piloted craft.
        let mut host = World::new();
        spawn_parts(&mut host, 0, cara_t, joint_id, joint_t);
        spawn_parts(&mut host, 1, cara_t1, joint_id, joint_t1);
        host.insert_resource(CrabSkinRepose(
            [(
                0usize,
                SkinRepose {
                    shift: Vec3::new(10.0, 0.0, -20.0),
                    pivot: Vec3::Y,
                    scale: 8.0,
                },
            )]
            .into_iter()
            .collect(),
        ));
        // Per-crab brain labels, one a failure state — the who's-who channel crosses the
        // wire verbatim (rl#200 increment 7).
        host.insert_resource(CrabBrainLabels(vec![
            "mlp512x3 @cafef00d".to_string(),
            "REFUSED: wrong rig".to_string(),
        ]));
        let craft_t = Transform::from_xyz(2.0, 5.5, -1.0)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        crab_world::vehicle::spawn_ram_vehicle(
            &mut host,
            crab_world::vehicle::VehicleKind::Plane,
            craft_t,
            bevy_rapier3d::prelude::Velocity::default(),
        );

        // Capture and send it exactly as the transport does — through the wire codec.
        let art = capture(&mut host, 42);
        assert_eq!(art.tick, 42);
        assert_eq!(art.crabs.len(), 2, "one frame per env");
        assert_eq!(art.crabs[0].parts.len(), 2);
        assert_eq!(art.crabs[1].parts.len(), 2);
        assert!(art.crabs[0].repose.is_some());
        assert!(art.crabs[1].repose.is_none(), "env 1 has no placement yet");
        let art = crate::articulation::CrabArticulation::from_bytes(&art.to_bytes()).unwrap();

        // CLIENT: the SAME rigs at DIFFERENT poses with no placement yet (frozen just-spawned
        // crabs).
        let mut client = World::new();
        let (c_cara, c_joint) = spawn_parts(
            &mut client,
            0,
            Transform::from_xyz(-1.0, -1.0, -1.0),
            joint_id,
            Transform::from_xyz(5.0, 5.0, 5.0),
        );
        let (c_cara1, c_joint1) = spawn_parts(
            &mut client,
            1,
            Transform::from_xyz(-2.0, -2.0, -2.0),
            joint_id,
            Transform::from_xyz(6.0, 6.0, 6.0),
        );
        client.insert_resource(CrabSkinRepose::default());

        apply(&mut client, &art);

        // Each env's parts + placement now match the host's EXACTLY — the client renders the
        // host's poses, never its own physics, and never crab 1's pose on crab 0.
        let tf = |world: &mut World, e: Entity| *world.entity(e).get::<Transform>().unwrap();
        let got_cara = tf(&mut client, c_cara);
        let got_joint = tf(&mut client, c_joint);
        assert_eq!(got_cara.translation, cara_t.translation);
        assert_eq!(got_cara.rotation, cara_t.rotation);
        assert_eq!(got_joint.translation, joint_t.translation);
        assert_eq!(got_joint.rotation, joint_t.rotation);
        let got_cara1 = tf(&mut client, c_cara1);
        let got_joint1 = tf(&mut client, c_joint1);
        assert_eq!(got_cara1.translation, cara_t1.translation);
        assert_eq!(got_cara1.rotation, cara_t1.rotation);
        assert_eq!(got_joint1.translation, joint_t1.translation);
        assert_eq!(got_joint1.rotation, joint_t1.rotation);

        let reposes = client.resource::<CrabSkinRepose>().0.clone();
        let repose0 = reposes.get(&0).expect("env 0's repose applied");
        assert_eq!(repose0.shift, Vec3::new(10.0, 0.0, -20.0));
        assert_eq!(repose0.pivot, Vec3::Y);
        assert_eq!(repose0.scale, 8.0);
        assert!(
            !reposes.contains_key(&1),
            "an unpublished env stays at identity"
        );
        // The host's exact label strings crossed too — the client renders them verbatim,
        // never re-deriving who's who (and the failure attribution survives the wire).
        assert_eq!(
            client.resource::<CrabBrainLabels>().0,
            vec![
                "mlp512x3 @cafef00d".to_string(),
                "REFUSED: wrong rig".to_string()
            ]
        );
        // The host's piloted craft crossed too — the client's mirror holds its exact pose
        // (rl#192: this is what the second player's wireframe drawer renders).
        let craft = client
            .resource::<RemoteVehicle>()
            .0
            .expect("piloted craft applied");
        assert_eq!(Vec3::from_array(craft.pos), craft_t.translation);
        assert_eq!(Quat::from_array(craft.rot), craft_t.rotation);
    }
}
