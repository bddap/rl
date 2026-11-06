use bevy::prelude::*;

use crab_world::vehicle::{PilotId, Vehicle, VehicleKind};

use super::pose::{Pose, PoseWindow};
use net_proto::vehicles::{VehicleFrame, VehiclePoseWire};

/// Every NON-LOCAL pilot's craft kind + pose window (rl#191), feeding the craft models
/// (`vehicle_view`, rl#260). The local pilot's own
/// craft is excluded: the cockpit camera flies from it instead. Sampled through
/// [`PoseWindow`], so a watched craft moves as smoothly as the cockpit (rl#267).
#[derive(Resource, Default)]
pub(super) struct RemoteVehicle(std::collections::BTreeMap<PilotId, RemoteCraft>);

struct RemoteCraft {
    kind: VehicleKind,
    poses: PoseWindow,
    /// Newest wire thrust command, body-frame fractions (rl#308). Raw, not windowed:
    /// it drives plume intensity, where the flicker swallows any tick step.
    thrust: Vec3,
}

/// One remote craft at render time — [`RemoteVehicle::sample`]'s output, so consumers
/// are agnostic to the interpolation behind it.
pub(super) struct SampledCraft {
    pub pilot: PilotId,
    pub kind: VehicleKind,
    pub pose: Pose,
    /// Body-frame thrust command fractions — the exhaust-plume intensity (rl#308).
    pub thrust: Vec3,
}

impl RemoteVehicle {
    /// Adopt one remote wire set at physics step `step`: an absent pilot's craft drops
    /// (the pilot left — the model despawns this frame), a kind change restarts its window
    /// (the cycle swap teleports the new silhouette in; interpolating across it would
    /// smear one craft into the other).
    pub(super) fn adopt(&mut self, step: u64, remote: &[VehiclePoseWire]) {
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
                    thrust: Vec3::ZERO,
                });
            craft.thrust = v.thrust_fraction();
            craft.poses.push(
                step,
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
                    thrust: craft.thrust,
                })
            })
            .collect()
    }
}

pub(super) fn publish_remote_vehicles(
    world: &mut World,
    step: u64,
    vehicles: &[VehiclePoseWire],
    me: PilotId,
) {
    let remote: Vec<VehiclePoseWire> = vehicles
        .iter()
        .filter(|v| v.pilot != me.0)
        .copied()
        .collect();
    // `install_round` owns creation — a missing resource here is a broken install, not
    // a case to paper over.
    world.resource_mut::<RemoteVehicle>().adopt(step, &remote);
}

/// The host's craft poses at `tick`, for the [`VehicleFrame`] broadcast and the host's
/// own remote-craft view.
pub(super) fn capture(world: &mut World, tick: u64) -> VehicleFrame {
    // The pilots' live commands, for the wire's thrust fractions (rl#308).
    let controls = world
        .resource::<crab_world::vehicle::VehicleControls>()
        .0
        .clone();
    let mut vehicles: Vec<VehiclePoseWire> = world
        .query::<(&Transform, &Vehicle)>()
        .iter(world)
        .map(|(t, v)| VehiclePoseWire {
            pilot: v.pilot.0,
            kind: v.kind,
            pos: t.translation.to_array(),
            rot: t.rotation.to_array(),
            thrust: VehiclePoseWire::quantize_thrust(
                controls
                    .get(&v.pilot)
                    .map(|c| v.thrust_fraction(c))
                    .unwrap_or(Vec3::ZERO),
            ),
        })
        .collect();
    vehicles.sort_by_key(|v| v.pilot);
    VehicleFrame { tick, vehicles }
}

#[cfg(test)]
mod tests {
    use super::*;
    use net_proto::cadence::cumulative_steps;

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
            thrust: [0, 0, 0],
        };
        let mut rv = RemoteVehicle::default();
        for tick in 1..=3u64 {
            rv.adopt(
                cumulative_steps(tick),
                &[wire(1, VehicleKind::Plane, tick as f32)],
            );
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
        rv.adopt(cumulative_steps(4), &[wire(1, VehicleKind::Ship, 10.0)]);
        let swapped = rv.sample(4, 0.5);
        assert_eq!(swapped[0].kind, VehicleKind::Ship);
        assert_eq!(
            swapped[0].pose.pos.x, 10.0,
            "a fresh window holds the newest pose; interpolating across the swap would \
             smear the plane into the ship"
        );

        // Step-out: absent from the wire set ⇒ dropped immediately.
        rv.adopt(cumulative_steps(5), &[]);
        assert!(rv.sample(5, 0.0).is_empty());
    }
}
