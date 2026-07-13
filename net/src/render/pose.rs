//! Physics-step-time pose interpolation (rl#264/rl#267) — THE one mechanism for
//! rendering any tick-stamped pose stream. Craft and crab-part state advances a
//! VARIABLE number of physics steps per sim tick (the 64:30 staircase,
//! [`crate::cadence::cumulative_steps`]), so interpolating a pose pair by
//! tick-fraction surges rendered velocity ±50% ~4×/s. Every consumer — the local
//! cockpit ([`super::driver::LocalVehicle`]), remote pilots' craft models
//! ([`super::articulation::RemoteVehicle`]), and the crab body parts on both arms
//! ([`super::articulation::CrabPartWindows`], host and client alike since rl#274) —
//! samples through this window instead; a second interpolation mechanism is a bug
//! (rl#267).

use bevy::prelude::*;

#[derive(Clone, Copy)]
pub(super) struct Pose {
    pub pos: Vec3,
    pub orient: Quat,
}

/// The last three tick-stamped poses, a sliding window oldest→newest.
/// [`Self::sample`] walks a uniform physics-step clock held ONE step behind the
/// newest pose: the staircase never strays a full step from ideal (pinned by
/// `staircase_stays_within_one_step_of_ideal`), so the sample point always lands
/// inside this window and rendered motion is uniform by construction (cost: 1 step
/// ≈ 16 ms of latency). Feeders push one pose per tick — the host from its stepped
/// world, a remote client from its adopted articulation — and tick gaps (a lost
/// articulation datagram) interpolate across naturally.
#[derive(Clone, Copy, Default)]
pub(super) struct PoseWindow {
    buf: [Option<(u64, Pose)>; 3],
}

/// A per-tick pose jump no craft or crab part can cover by MOVING (plane terminal
/// ~4.5 m/s ⇒ ~0.15 m/tick) is a TELEPORT — a round-RESTART respawn, a non-finite
/// rescue (rl#137). Interpolating across one would smear the body over the window's
/// span on every observer, so the window restarts and holds the arrival pose — the
/// same motion-vs-teleport discrimination as `boarding_of`'s walk-speed guard.
const TELEPORT_RESET_METERS: f32 = 5.0;

impl PoseWindow {
    pub(super) fn push(&mut self, tick: u64, p: Pose) {
        // A non-advancing tick means the clock rewound under us (defensive; no
        // feeder currently produces one) — stale history would mis-scale, so drop it.
        if self.buf[2]
            .is_some_and(|(t, last)| tick <= t || last.pos.distance(p.pos) > TELEPORT_RESET_METERS)
        {
            *self = Self::default();
        }
        self.buf.rotate_left(1);
        self.buf[2] = Some((tick, p));
    }

    pub(super) fn sample(&self, now_tick: u64, tick_frac: f32) -> Option<Pose> {
        use crate::cadence::cumulative_steps;
        use crate::sim::TICK_HZ;
        use crab_world::physics::PHYSICS_HZ;

        let entries: [(u64, Pose); 3] = match self.buf {
            [Some(a), Some(b), Some(c)] => [a, b, c],
            // Engage grace: hold the newest pose until the window fills (1–2 ticks).
            _ => return self.buf.iter().flatten().last().map(|&(_, p)| p),
        };
        let r = PHYSICS_HZ as f64 / TICK_HZ as f64;
        // Render time in ticks is (now_tick − 1) + frac — a frame interpolates the
        // tick interval ENDING at the last stepped tick, same clock as scene.rs's
        // accumulator alpha. The trailing −1.0 is the one-step latency hold that
        // keeps the target inside the window (see the type doc).
        let ideal = r * (now_tick.saturating_sub(1) as f64 + tick_frac as f64) - 1.0;
        let steps = entries.map(|(t, _)| cumulative_steps(t) as f64);
        let target = ideal.clamp(steps[0], steps[2]);
        let (i0, i1) = if target <= steps[1] { (0, 1) } else { (1, 2) };
        let w = if steps[i1] > steps[i0] {
            ((target - steps[i0]) / (steps[i1] - steps[i0])) as f32
        } else {
            1.0
        };
        Some(Pose {
            pos: entries[i0].1.pos.lerp(entries[i1].1.pos, w),
            orient: entries[i0].1.orient.slerp(entries[i1].1.orient, w),
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.buf[2].is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_window_renders_uniform_velocity_through_the_staircase() {
        // The rl#264 pin: a body moving at CONSTANT velocity in physics time (0.1
        // units per step — craft-scale, so a multi-tick gap stays under the teleport
        // guard) must render at velocity UNIFORM IN RENDER TIME, even though the
        // 64:30 cadence bunches 2 vs 3 steps per tick — the old tick-fraction
        // interpolation surged ±50% on the 3-step ticks. When render time pauses and
        // jumps (a tick gap: both peers' clocks stall together on loss), the covered
        // distance must stay proportional to the jump — no surge, no shortfall.
        use crate::cadence::cumulative_steps;
        const STEP_METERS: f64 = 0.1;
        let r = STEP_METERS * crab_world::physics::PHYSICS_HZ as f64 / crate::sim::TICK_HZ as f64;
        let pose_at = |tick: u64| Pose {
            pos: Vec3::new(
                (cumulative_steps(tick) as f64 * STEP_METERS) as f32,
                0.0,
                0.0,
            ),
            orient: Quat::IDENTITY,
        };
        let mut window = PoseWindow::default();
        let mut last: Option<(f64, f32)> = None; // (render time in ticks, sampled x)
        let frames_per_tick = 4; // a 120 fps render against 30 Hz ticks
        let mut checked = 0u32;
        for tick in 1..200u64 {
            // Ticks 100/101 never land (lost datagrams; frames pace on adopted
            // ticks, so render time stalls with the window and jumps 3 ticks at 102).
            if tick == 100 || tick == 101 {
                continue;
            }
            window.push(tick, pose_at(tick));
            if tick < 4 {
                continue; // window fill
            }
            for f in 0..frames_per_tick {
                let frac = f as f32 / frames_per_tick as f32;
                let t_render = (tick - 1) as f64 + frac as f64;
                let x = window.sample(tick, frac).unwrap().pos.x;
                if let Some((t0, x0)) = last {
                    let expected = r * (t_render - t0);
                    assert!(
                        ((x - x0) as f64 - expected).abs() < 1e-3,
                        "tick {tick} frac {frac}: moved {} for {} render-ticks \
                         (expected {expected}) — the staircase leaked into rendered \
                         motion",
                        x - x0,
                        t_render - t0,
                    );
                    checked += 1;
                }
                last = Some((t_render, x));
            }
        }
        assert!(checked > 700, "the sweep must actually cover the run");
    }

    #[test]
    fn snapshot_stall_holds_still_then_resumes_forward() {
        // rl#273: during a snapshot stall the driver freezes its clock at
        // (last adopted tick, frac = 1.0) instead of letting frac wrap. Pin the
        // window's side of that contract: the frozen clock samples a CONSTANT pose,
        // and the resume sweep continues forward from exactly the held position.
        use crate::cadence::cumulative_steps;
        const STEP_METERS: f64 = 0.1;
        let pose_at = |tick: u64| Pose {
            pos: Vec3::new(
                (cumulative_steps(tick) as f64 * STEP_METERS) as f32,
                0.0,
                0.0,
            ),
            orient: Quat::IDENTITY,
        };
        let mut w = PoseWindow::default();
        for tick in 1..=9u64 {
            w.push(tick, pose_at(tick));
        }
        let held = w.sample(9, 1.0).unwrap().pos.x;
        for _ in 0..30 {
            assert_eq!(
                w.sample(9, 1.0).unwrap().pos.x,
                held,
                "a stall must render a clean hold"
            );
        }
        let wrapped = w.sample(9, 0.02).unwrap().pos.x;
        assert!(
            wrapped < held,
            "sanity: an un-pinned wrapping clock really would rewind the pose"
        );
        // Recovery adopts tick 10 with frac back near 0 — same render time as the
        // hold (both (9-1)+1.0 and (10-1)+0.0 are 9 ticks), so no seam.
        let mut last = held;
        for tick in 10..=12u64 {
            w.push(tick, pose_at(tick));
            for f in 0..4 {
                let x = w.sample(tick, f as f32 / 4.0).unwrap().pos.x;
                assert!(x >= last - 1e-4, "resume must not rewind: {x} < {last}");
                last = x;
            }
        }
        assert!(last > held, "the resume sweep must actually move forward");
    }

    #[test]
    fn teleport_resets_the_window_instead_of_smearing() {
        let at = |x: f32| Pose {
            pos: Vec3::new(x, 0.0, 0.0),
            orient: Quat::IDENTITY,
        };
        let mut w = PoseWindow::default();
        for tick in 1..=3u64 {
            w.push(tick, at(tick as f32 * 0.1));
        }
        // A round-RESTART respawn: the next tick's pose is across the arena.
        w.push(4, at(100.0));
        assert_eq!(
            w.sample(4, 0.5).unwrap().pos.x,
            100.0,
            "the window must restart at the arrival pose — interpolating would smear \
             the teleport over the window's span"
        );
    }
}
