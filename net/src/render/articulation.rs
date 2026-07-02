//! Render-side capture + apply for the crab pose extension frame (bddap/rl#151, increment 2
//! windowed) — the render-only halves of [`crate::articulation`].
//!
//! Under host-authority the windowed HOST runs the one rapier Sally; a windowed remote CLIENT
//! renders the host's exact pose without simulating physics ([[silent-fallback-antipattern]]: no
//! second "run my own crab for visuals" path). So the host [`capture`]s its live crab's per-part
//! transforms + giant-blow-up placement each stepped tick and broadcasts them beside the snapshot;
//! the client [`apply`]s them onto its OWN crab entities — which it spawns but never physics-steps,
//! so the writes stick and `crab_world::bot::skin::drive_bones` skins the mesh to the host's pose.
//!
//! Only the parts a skin bone actually follows are carried: those with a stable `PartId` (each
//! actuated joint link + the carapace). The cosmetic eye-stalk links carry no `PartId` and no bone
//! keys off them (their bones ride the carapace), so they need no sync — and couldn't be matched
//! across the wire without a stable key anyway.

use bevy::prelude::*;

use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId};
use crab_world::bot::skin::{CrabSkinRepose, SkinRepose};

use crate::articulation::{CrabArticulation, PartTransform, ReposeWire};

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

/// (Host) Snapshot env 0's crab render pose for `tick`: every keyed body part's arena-frame
/// transform (world-space — the parts are top-level, so `Transform` already is) plus the current
/// giant-blow-up placement. Called right after `Server::step_next`, so it is this tick's settled
/// pose (`integrate_crab`/`publish_skin_repose` ran during the physics pump). `repose` is `None`
/// only before the bridge has published one (transiently at spawn).
pub(super) fn capture(world: &mut World, tick: u64) -> CrabArticulation {
    let mut parts = Vec::new();
    let mut q = world.query_filtered::<(
        &Transform,
        &CrabEnvId,
        Option<&CrabJoint>,
        Option<&CrabCarapace>,
    ), With<CrabBodyPart>>();
    for (t, env, joint, carapace) in q.iter(world) {
        if env.0 != 0 {
            continue;
        }
        let Some(tag) = part_tag(carapace.is_some(), joint) else {
            continue;
        };
        parts.push(PartTransform {
            part: tag,
            pos: t.translation.to_array(),
            rot: t.rotation.to_array(),
        });
    }
    // Ascending tag order — a deterministic wire, and the client matches by tag regardless.
    parts.sort_by_key(|p| p.part);

    let repose = world
        .get_resource::<CrabSkinRepose>()
        .and_then(|r| r.0)
        .map(|s| ReposeWire {
            shift: s.shift.to_array(),
            pivot: s.pivot.to_array(),
            scale: s.scale,
        });

    CrabArticulation {
        tick,
        parts,
        repose,
    }
}

/// (Client) Write a received crab pose onto env 0's own crab render entities — overwriting each
/// keyed part's `Transform` and the giant-blow-up placement. The client never pumps the crab
/// physics, so these writes are not fought back by the rapier solver and persist for
/// `crab_world::bot::skin::drive_bones` (PostUpdate) to skin the mesh from. A part in the frame with
/// no matching local entity (or vice versa) is simply skipped — the rig is identical on both peers,
/// so this only elides the transient pre-spawn frames.
pub(super) fn apply(world: &mut World, art: &CrabArticulation) {
    let by_tag: std::collections::HashMap<u8, &PartTransform> =
        art.parts.iter().map(|p| (p.part, p)).collect();

    let mut q = world.query_filtered::<(
        &mut Transform,
        &CrabEnvId,
        Option<&CrabJoint>,
        Option<&CrabCarapace>,
    ), With<CrabBodyPart>>();
    for (mut t, env, joint, carapace) in q.iter_mut(world) {
        if env.0 != 0 {
            continue;
        }
        let Some(tag) = part_tag(carapace.is_some(), joint) else {
            continue;
        };
        if let Some(p) = by_tag.get(&tag) {
            t.translation = Vec3::from_array(p.pos);
            t.rotation = Quat::from_array(p.rot);
        }
    }

    if let Some(r) = &art.repose
        && let Some(mut repose) = world.get_resource_mut::<CrabSkinRepose>()
    {
        repose.0 = Some(SkinRepose {
            shift: Vec3::from_array(r.shift),
            pivot: Vec3::from_array(r.pivot),
            scale: r.scale,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab_world::bot::body::{CrabJointId, Side};

    /// Spawn one env-0 crab's keyed render parts — a carapace + one joint link — at the given
    /// transforms, the minimal rig [`capture`]/[`apply`] key off (no physics/GPU needed).
    fn spawn_parts(
        world: &mut World,
        carapace: Transform,
        joint_id: CrabJointId,
        joint: Transform,
    ) -> (Entity, Entity) {
        let cara = world
            .spawn((CrabBodyPart, CrabCarapace, CrabEnvId(0), carapace))
            .id();
        let jnt = world
            .spawn((
                CrabBodyPart,
                CrabJoint {
                    id: joint_id,
                    axis_local: Vec3::X,
                },
                CrabEnvId(0),
                joint,
            ))
            .id();
        (cara, jnt)
    }

    #[test]
    fn capture_then_apply_reproduces_the_hosts_exact_pose() {
        let joint_id = CrabJointId::ClawShoulder(Side::Left);
        let cara_t = Transform::from_xyz(1.0, 2.0, 3.0).with_rotation(Quat::from_rotation_y(0.5));
        let joint_t =
            Transform::from_xyz(-4.0, 0.25, 9.0).with_rotation(Quat::from_rotation_x(0.3));

        // HOST: known part poses + a published giant-blow-up placement.
        let mut host = World::new();
        spawn_parts(&mut host, cara_t, joint_id, joint_t);
        host.insert_resource(CrabSkinRepose(Some(SkinRepose {
            shift: Vec3::new(10.0, 0.0, -20.0),
            pivot: Vec3::Y,
            scale: 8.0,
        })));

        // Capture and send it exactly as the transport does — through the wire codec.
        let art = capture(&mut host, 42);
        assert_eq!(art.tick, 42);
        assert_eq!(art.parts.len(), 2);
        let art = crate::articulation::CrabArticulation::from_bytes(&art.to_bytes()).unwrap();

        // CLIENT: the SAME rig at DIFFERENT poses with no placement yet (a frozen just-spawned crab).
        let mut client = World::new();
        let (c_cara, c_joint) = spawn_parts(
            &mut client,
            Transform::from_xyz(-1.0, -1.0, -1.0),
            joint_id,
            Transform::from_xyz(5.0, 5.0, 5.0),
        );
        client.insert_resource(CrabSkinRepose(None));

        apply(&mut client, &art);

        // The client's parts + placement now match the host's EXACTLY — it renders the host's pose,
        // never its own physics.
        let got_cara = *client.entity(c_cara).get::<Transform>().unwrap();
        let got_joint = *client.entity(c_joint).get::<Transform>().unwrap();
        assert_eq!(got_cara.translation, cara_t.translation);
        assert_eq!(got_cara.rotation, cara_t.rotation);
        assert_eq!(got_joint.translation, joint_t.translation);
        assert_eq!(got_joint.rotation, joint_t.rotation);
        let repose = client
            .resource::<CrabSkinRepose>()
            .0
            .expect("repose applied");
        assert_eq!(repose.shift, Vec3::new(10.0, 0.0, -20.0));
        assert_eq!(repose.pivot, Vec3::Y);
        assert_eq!(repose.scale, 8.0);
    }
}
