use bevy::prelude::*;

use crab_world::bot::body::{CrabBodyPart, CrabCarapace, CrabEnvId, CrabJoint, CrabJointId};
use crab_world::bot::skin::{CrabSkinRepose, SkinRepose};
use crab_world::crab_view::CrabBrainLabels;
use crab_world::vehicle::{PilotId, Vehicle, VehicleKind};

use super::driver::RenderClock;
use super::pose::{Pose, PoseWindow};
use crate::articulation::{
    CrabArticulation, CrabFrame, PartTransform, ReposeWire, VehiclePoseWire,
};

/// Every NON-LOCAL pilot's craft kind + tick-stamped pose window (rl#191), feeding the
/// craft models (`vehicle_view`, rl#260) and the colliders-mode wireframe. The local
/// pilot's own craft is excluded: the cockpit camera flies from it instead. Fed per tick
/// from the same articulation on both arms — the host from its capture, a client from
/// its adopt — and sampled per frame on the uniform physics-step clock
/// ([`PoseWindow`]), so a watched craft moves as smoothly as the cockpit (rl#267).
#[derive(Resource, Default)]
pub(super) struct RemoteVehicle(std::collections::BTreeMap<PilotId, RemoteCraft>);

struct RemoteCraft {
    kind: VehicleKind,
    poses: PoseWindow,
}

/// One remote craft at render time — [`RemoteVehicle::sample`]'s output, so consumers
/// are agnostic to the interpolation behind it.
pub(super) struct SampledCraft {
    pub pilot: PilotId,
    pub kind: VehicleKind,
    pub pose: Pose,
}

impl RemoteVehicle {
    /// Adopt one tick's remote wire set: an absent pilot's craft drops (step-out — the
    /// model despawns this frame), a kind change restarts its window (the cycle swap
    /// teleports the new silhouette in; interpolating across it would smear one craft
    /// into the other).
    pub(super) fn adopt(&mut self, tick: u64, remote: &[VehiclePoseWire]) {
        self.0.retain(|pilot, craft| {
            remote
                .iter()
                .any(|v| PilotId(v.pilot) == *pilot && v.kind == craft.kind)
        });
        for v in remote {
            let craft = self
                .0
                .entry(PilotId(v.pilot))
                .or_insert_with(|| RemoteCraft {
                    kind: v.kind,
                    poses: PoseWindow::default(),
                });
            craft.poses.push(
                tick,
                Pose {
                    pos: Vec3::from_array(v.pos),
                    orient: Quat::from_array(v.rot),
                },
            );
        }
    }

    pub(super) fn contains(&self, pilot: PilotId) -> bool {
        self.0.contains_key(&pilot)
    }

    pub(super) fn sample(&self, now_tick: u64, frac: f32) -> Vec<SampledCraft> {
        self.0
            .iter()
            .filter_map(|(&pilot, craft)| {
                Some(SampledCraft {
                    pilot,
                    kind: craft.kind,
                    pose: craft.poses.sample(now_tick, frac)?,
                })
            })
            .collect()
    }
}

/// One log line per remote-craft EDGE — the pilot set changing (a boarding/exit seen from this
/// peer) and each craft's first real displacement (proof the other pilot's craft is moving here,
/// rl#191 increment 4) — instead of a per-tick pose flood.
#[derive(Resource, Default)]
pub(super) struct RemoteCraftWatch {
    pilots: Vec<u8>,
    first_pose: std::collections::BTreeMap<u8, Vec3>,
    moved: std::collections::BTreeSet<u8>,
}

const REMOTE_MOVED_LOG_METERS: f32 = 5.0;

pub(super) fn publish_remote_vehicles(
    world: &mut World,
    tick: u64,
    vehicles: &[VehiclePoseWire],
    me: PilotId,
) {
    let remote: Vec<VehiclePoseWire> = vehicles
        .iter()
        .filter(|v| v.pilot != me.0)
        .copied()
        .collect();
    let mut watch = world.get_resource_or_insert_with(RemoteCraftWatch::default);
    let pilots: Vec<u8> = remote.iter().map(|v| v.pilot).collect();
    if pilots != watch.pilots {
        info!(
            "remote crafts: pilots {:?} (was {:?})",
            pilots, watch.pilots
        );
        watch.pilots = pilots;
        watch
            .first_pose
            .retain(|p, _| remote.iter().any(|v| v.pilot == *p));
        watch.moved.retain(|p| remote.iter().any(|v| v.pilot == *p));
    }
    for v in &remote {
        let pos = Vec3::from_array(v.pos);
        let first = *watch.first_pose.entry(v.pilot).or_insert(pos);
        if !watch.moved.contains(&v.pilot) && first.distance(pos) > REMOTE_MOVED_LOG_METERS {
            watch.moved.insert(v.pilot);
            info!(
                "remote craft (pilot {}) has moved {:.1}m since it appeared",
                v.pilot,
                first.distance(pos)
            );
        }
    }
    // `install_round` owns creation — a missing resource here is a broken install, not
    // a case to paper over.
    world.resource_mut::<RemoteVehicle>().adopt(tick, &remote);
}

const _: () = assert!(CrabJointId::COUNT < u8::MAX as usize);

fn part_tag(is_carapace: bool, joint: Option<&CrabJoint>) -> Option<u8> {
    match (is_carapace, joint) {
        (true, _) => Some(0),
        (_, Some(j)) => Some(1 + j.id.index() as u8),
        _ => None,
    }
}

pub(super) fn capture(world: &mut World, tick: u64) -> CrabArticulation {
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

    let n_crabs = by_env.keys().last().map_or(0, |&max| max + 1);
    // Labels publish once the armed FixedUpdate first ticks — empty is the legitimate
    // pre-tick frame. A NON-empty mismatch is the rl#241 slot-desync class: it would
    // silently blank a crab's rl#200 who's-who attribution on every client. Latched
    // error!, not debug_assert: this must stay loud in the release builds the fleet
    // actually runs, without flooding a per-frame serializer.
    if !(labels.is_empty() || labels.len() == n_crabs) {
        static LABEL_DESYNC_REPORTED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !LABEL_DESYNC_REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            error!(
                "brain labels desynced from crab envs: {} labels for {n_crabs} crabs — \
                 who's-who attribution blanked on every client (rl#241/rl#200)",
                labels.len()
            );
        }
    }
    let crabs = (0..n_crabs)
        .map(|env| {
            let mut parts = by_env.remove(&env).unwrap_or_default();
            parts.sort_by_key(|p| p.part);
            let repose = reposes.get(&env).map(|s| ReposeWire {
                shift: s.shift.to_array(),
            });
            CrabFrame {
                parts,
                repose,
                brain_label: labels.get(env).cloned().unwrap_or_default(),
            }
        })
        .collect();

    let mut vehicles: Vec<VehiclePoseWire> = world
        .query::<(&Transform, &Vehicle)>()
        .iter(world)
        .map(|(t, v)| VehiclePoseWire {
            pilot: v.pilot.0,
            kind: v.kind,
            pos: t.translation.to_array(),
            rot: t.rotation.to_array(),
        })
        .collect();
    vehicles.sort_by_key(|v| v.pilot);

    let arena_anchor = world
        .get_resource::<crate::external_crab::ArenaAnchor>()
        .map(|a| a.0.to_array())
        .unwrap_or_default();

    CrabArticulation {
        tick,
        crabs,
        arena_anchor,
        vehicles,
    }
}

/// The adopted crab puppet's tick-stamped pose windows, one per (env, part-tag).
/// Fed by [`adopt`] each adopted tick, drained by [`sample_puppet_parts`] each frame:
/// the host steps the parts in physics, so applying its frames raw replays the 64:30
/// step bunching on every client — the same surge rl#264 fixed for the cockpit (rl#267).
/// Only the remote-adopt arm ever feeds this; on the host it stays empty and the
/// sampler writes nothing.
#[derive(Resource, Default)]
pub(super) struct PuppetWindows(std::collections::BTreeMap<(usize, u8), PoseWindow>);

/// Adopts one articulation tick: crab body parts land in [`PuppetWindows`] (rendered by
/// [`sample_puppet_parts`]); repose, arena anchor, and brain labels adopt directly —
/// they are discrete state, not stepped motion.
pub(super) fn adopt(world: &mut World, art: &CrabArticulation) {
    {
        // `install_round` owns creation, like `RemoteVehicle` above.
        let mut windows = world.resource_mut::<PuppetWindows>();
        // A shrunk crab set (a fresh round's re-adopt) drops the stale envs' windows.
        windows.0.retain(|(env, _), _| *env < art.crabs.len());
        for (env, frame) in art.crabs.iter().enumerate() {
            for p in &frame.parts {
                windows.0.entry((env, p.part)).or_default().push(
                    art.tick,
                    Pose {
                        pos: Vec3::from_array(p.pos),
                        orient: Quat::from_array(p.rot),
                    },
                );
            }
        }
    }

    if let Some(mut repose) = world.get_resource_mut::<CrabSkinRepose>() {
        for (env, frame) in art.crabs.iter().enumerate() {
            if let Some(r) = frame.repose {
                repose.0.insert(
                    env,
                    SkinRepose {
                        shift: Vec3::from_array(r.shift),
                    },
                );
            }
        }
        repose.0.retain(|env, _| *env < art.crabs.len());
    }

    // Adopt the host's arena anchor — the client-side write of [`ArenaAnchor`]; the
    // host-side publisher runs in FixedUpdate, so the two can't fight (see the resource doc).
    let anchor = crate::external_crab::ArenaAnchor(Vec3::from_array(art.arena_anchor));
    if world
        .get_resource::<crate::external_crab::ArenaAnchor>()
        .copied()
        != Some(anchor)
    {
        world.insert_resource(anchor);
    }

    // Adopt the host's brain labels (write-on-change so the shared label UI only reconciles
    // when something actually changed). Like the parts, the client renders these verbatim —
    // it never re-derives who's who.
    let labels: Vec<String> = art.crabs.iter().map(|f| f.brain_label.clone()).collect();
    if world.get_resource::<CrabBrainLabels>().map(|l| &l.0) != Some(&labels) {
        world.insert_resource(CrabBrainLabels(labels));
    }
}

type PuppetPartXf<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Transform,
        &'static CrabEnvId,
        Option<&'static CrabJoint>,
        Option<&'static CrabCarapace>,
    ),
    With<CrabBodyPart>,
>;

/// Writes the [`PuppetWindows`] samples onto the local `CrabBodyPart` `Transform`s each
/// frame. That is the one sanctioned mutation of body-part transforms outside physics
/// (rl#116): the windows fill only on a remote-adopt client, whose `FixedUpdate` is
/// parked — rapier never steps there, so these puppet writes never reach a solver (on
/// the host this is a no-op by construction). The pose sentinel and the
/// transform-ownership gate both allowlist exactly this path. Must run after the
/// frame's articulation [`adopt`] + `RenderClock` write, and before the skin's
/// PostUpdate `drive_bones`, so bones follow this frame's sampled parts.
pub(super) fn sample_puppet_parts(
    clock: Res<RenderClock>,
    windows: Res<PuppetWindows>,
    mut parts: PuppetPartXf,
) {
    if windows.0.is_empty() {
        return;
    }
    for (mut t, env, joint, carapace) in &mut parts {
        let Some(tag) = part_tag(carapace.is_some(), joint) else {
            continue;
        };
        let Some(p) = windows
            .0
            .get(&(env.0, tag))
            .and_then(|w| w.sample(clock.tick, clock.frac))
        else {
            continue;
        };
        t.translation = p.pos;
        t.rotation = p.orient;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crab_world::bot::body::{CrabJointId, Side};

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
        let cara_t1 = Transform::from_xyz(9.0, 1.0, -6.0).with_rotation(Quat::from_rotation_y(1.2));
        let joint_t1 =
            Transform::from_xyz(8.0, 0.5, -5.0).with_rotation(Quat::from_rotation_x(0.9));

        let mut host = World::new();
        spawn_parts(&mut host, 0, cara_t, joint_id, joint_t);
        spawn_parts(&mut host, 1, cara_t1, joint_id, joint_t1);
        host.insert_resource(CrabSkinRepose(
            [(
                0usize,
                SkinRepose {
                    shift: Vec3::new(10.0, 0.0, -20.0),
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
        host.insert_resource(crate::external_crab::ArenaAnchor(Vec3::new(
            3.5, 0.0, -7.25,
        )));
        let craft_t = Transform::from_xyz(2.0, 5.5, -1.0)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        crab_world::vehicle::spawn_ram_vehicle(
            &mut host,
            crab_world::vehicle::VehicleKind::Plane,
            craft_t,
            bevy_rapier3d::prelude::Velocity::default(),
        );

        let art = capture(&mut host, 42);
        assert_eq!(art.tick, 42);
        assert_eq!(art.crabs.len(), 2, "one frame per env");
        assert_eq!(art.crabs[0].parts.len(), 2);
        assert_eq!(art.crabs[1].parts.len(), 2);
        assert!(art.crabs[0].repose.is_some());
        assert!(art.crabs[1].repose.is_none(), "env 1 has no placement yet");
        let art = crate::articulation::CrabArticulation::from_bytes(&art.to_bytes()).unwrap();

        use bevy::ecs::system::RunSystemOnce;
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
        client.insert_resource(PuppetWindows::default());
        client.insert_resource(RemoteVehicle::default());

        adopt(&mut client, &art);
        // Adopted parts render through the per-frame sampler (rl#267): a 1-deep window
        // holds the newest pose, so sampling at the adopt tick reproduces the frame raw.
        client.insert_resource(RenderClock {
            tick: 42,
            frac: 0.0,
        });
        client
            .run_system_once(sample_puppet_parts)
            .expect("sampler runs on a bare world");
        // Viewed as pilot 7: pilot 0's craft is somebody else's ⇒ it lands in RemoteVehicle.
        publish_remote_vehicles(&mut client, 42, &art.vehicles, PilotId(7));

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
        // The host's arena anchor crossed verbatim — the client renders crafts through the
        // exact frame the host authored (rl#224), never a re-derived one.
        assert_eq!(
            client.resource::<crate::external_crab::ArenaAnchor>().0,
            Vec3::new(3.5, 0.0, -7.25)
        );
        let crafts = client.resource::<RemoteVehicle>().sample(42, 0.0);
        assert_eq!(crafts.len(), 1, "one piloted craft applied");
        assert_eq!(
            crafts[0].pilot,
            PilotId(0),
            "the ram helper spawns pilot 0's craft"
        );
        assert_eq!(crafts[0].pose.pos, craft_t.translation);
        assert_eq!(crafts[0].pose.orient, craft_t.rotation);

        // Viewed as pilot 0 the same craft is OURS — the cockpit, not a wireframe.
        publish_remote_vehicles(&mut client, 42, &art.vehicles, PilotId(0));
        assert!(
            client
                .resource::<RemoteVehicle>()
                .sample(42, 0.0)
                .is_empty(),
            "the local pilot's own craft never enters the remote wireframe set"
        );
    }

    /// The rl#267 pin for remote crafts: wire poses adopted per tick sample back out
    /// interpolated on the uniform physics-step clock — the same [`PoseWindow`] law the
    /// cockpit follows (its uniform-velocity sweep is pinned in `pose.rs`) — while a
    /// step-out drops the craft and a kind cycle restarts its window instead of
    /// smearing one silhouette into the other.
    #[test]
    fn remote_crafts_interpolate_and_reset_on_kind_cycle() {
        let wire = |pilot: u8, kind, x: f32| VehiclePoseWire {
            pilot,
            kind,
            pos: [x, 0.0, 0.0],
            rot: [0.0, 0.0, 0.0, 1.0],
        };
        let mut rv = RemoteVehicle::default();
        for tick in 1..=3u64 {
            rv.adopt(tick, &[wire(1, VehicleKind::Plane, tick as f32)]);
        }
        // Mid-frame between ticks 2 and 3: strictly between the two wire poses.
        let mid = rv.sample(3, 0.5);
        assert_eq!(mid.len(), 1);
        assert!(
            mid[0].pose.pos.x > 2.0 && mid[0].pose.pos.x < 3.0,
            "sampled x {} must interpolate, not snap to a raw tick pose",
            mid[0].pose.pos.x
        );
        assert!(rv.contains(PilotId(1)) && !rv.contains(PilotId(2)));

        // Kind cycle: the window restarts — the ship samples at its own first pose.
        rv.adopt(4, &[wire(1, VehicleKind::Ship, 10.0)]);
        let swapped = rv.sample(4, 0.5);
        assert_eq!(swapped[0].kind, VehicleKind::Ship);
        assert_eq!(
            swapped[0].pose.pos.x, 10.0,
            "a fresh window holds the newest pose; interpolating across the swap would \
             smear the plane into the ship"
        );

        // Step-out: absent from the wire set ⇒ dropped immediately.
        rv.adopt(5, &[]);
        assert!(rv.sample(5, 0.0).is_empty());
    }

    /// The rl#267 pin for the crab puppet: adopted part frames render through the same
    /// physics-step-clock sampling, not raw per tick.
    #[test]
    fn puppet_parts_interpolate_between_adopted_ticks() {
        use bevy::ecs::system::RunSystemOnce;
        let joint_id = CrabJointId::ClawShoulder(Side::Left);
        let mut client = World::new();
        let (cara, _) = spawn_parts(
            &mut client,
            0,
            Transform::IDENTITY,
            joint_id,
            Transform::IDENTITY,
        );
        client.insert_resource(CrabSkinRepose::default());
        client.insert_resource(PuppetWindows::default());
        for tick in 1..=3u64 {
            let art = CrabArticulation {
                tick,
                crabs: vec![CrabFrame {
                    parts: vec![
                        PartTransform {
                            part: 0,
                            pos: [tick as f32, 0.0, 0.0],
                            rot: [0.0, 0.0, 0.0, 1.0],
                        },
                        PartTransform {
                            part: 1 + joint_id.index() as u8,
                            pos: [0.0, tick as f32, 0.0],
                            rot: [0.0, 0.0, 0.0, 1.0],
                        },
                    ],
                    repose: None,
                    brain_label: String::new(),
                }],
                arena_anchor: [0.0; 3],
                vehicles: Vec::new(),
            };
            adopt(&mut client, &art);
        }
        client.insert_resource(RenderClock { tick: 3, frac: 0.5 });
        client
            .run_system_once(sample_puppet_parts)
            .expect("sampler runs on a bare world");
        let x = client
            .entity(cara)
            .get::<Transform>()
            .unwrap()
            .translation
            .x;
        assert!(
            x > 2.0 && x < 3.0,
            "carapace x {x} must interpolate between adopted ticks, not snap raw"
        );
    }
}
