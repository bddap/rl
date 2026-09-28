//! Physics-step-time pose interpolation (rl#264/rl#267) — THE one mechanism for
//! rendering any stepped pose stream. Craft and crab-part state advances a
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

/// The newest poses, each stamped with the cumulative physics step it was taken
/// after, oldest→newest. [`Self::sample`] walks a uniform physics-step clock held ONE
/// step behind ideal; the staircase never strays a full step from ideal (pinned by
/// `staircase_stays_within_one_step_of_ideal`), so per-tick feeds always cover the
/// sample point (cost: 1 step ≈ 16 ms of latency).
#[derive(Clone, Copy, Default)]
pub(super) struct PoseWindow {
    buf: [Option<(u64, Pose)>; DEPTH],
}

/// The host feeds per step, up to ~3 steps ahead of the sample clock.
const DEPTH: usize = 4;

/// A per-push pose jump no craft or crab part can cover by MOVING (plane terminal
/// ~9.1 m/s ⇒ ~0.30 m/tick) is a TELEPORT — a round-RESTART respawn, a non-finite
/// rescue (rl#137). Interpolating across one would smear the body over the window's
/// span on every observer, so the window restarts and holds the arrival pose — the
/// same motion-vs-teleport discrimination as `boarding_of`'s walk-speed guard.
const TELEPORT_RESET_METERS: f32 = 5.0;

impl PoseWindow {
    pub(super) fn push(&mut self, step: u64, p: Pose) {
        // A non-finite pose never enters the window: the rescue (rl#137) is its loud
        // surface, and one NaN entry would lerp NaN into every sample for the window's
        // span — while its NaN distance makes the teleport guard below unable to fire.
        // Hold the last finite pose; the respawn pose that follows resets via the guard.
        if !p.pos.is_finite() || !p.orient.is_finite() {
            return;
        }
        // A non-advancing step means the clock rewound under us — stale history
        // would mis-scale, so drop it.
        if self.buf[DEPTH - 1]
            .is_some_and(|(s, last)| step <= s || last.pos.distance(p.pos) > TELEPORT_RESET_METERS)
        {
            *self = Self::default();
        }
        self.buf.rotate_left(1);
        self.buf[DEPTH - 1] = Some((step, p));
    }

    /// The newest pushed pose, no interpolation — the mode-switch orientation
    /// hand-off (rl#399) reads the craft's final attitude here.
    pub(super) fn latest(&self) -> Option<Pose> {
        self.buf[DEPTH - 1].map(|(_, p)| p)
    }

    pub(super) fn sample(&self, now_tick: u64, tick_frac: f32) -> Option<Pose> {
        use crate::sim::TICK_HZ;
        use crab_world::physics::PHYSICS_HZ;

        let r = PHYSICS_HZ as f64 / TICK_HZ as f64;
        // Render time in ticks is (now_tick − 1) + frac — a frame interpolates the
        // tick interval ENDING at the last stepped tick, same clock as scene.rs's
        // accumulator alpha. The trailing −1.0 is the one-step latency hold that
        // keeps the target inside the window (see the type doc).
        let target = r * (now_tick.saturating_sub(1) as f64 + tick_frac as f64) - 1.0;
        // Clamped into the window: a filling window (engage, teleport reset) holds
        // its oldest pose until the clock reaches it.
        let mut entries = self.buf.iter().flatten();
        let &(mut s0, mut p0) = entries.next()?;
        for &(s1, p1) in entries {
            if target <= s0 as f64 {
                break;
            }
            if target < s1 as f64 {
                let w = ((target - s0 as f64) / (s1 - s0) as f64) as f32;
                return Some(Pose {
                    pos: p0.pos.lerp(p1.pos, w),
                    orient: p0.orient.slerp(p1.orient, w),
                });
            }
            (s0, p0) = (s1, p1);
        }
        Some(p0)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.buf[DEPTH - 1].is_none()
    }

    pub(super) fn newest_step(&self) -> Option<u64> {
        self.buf[DEPTH - 1].map(|(s, _)| s)
    }

    /// Ground speed of the newest interval, m/s (rl#356), divided by physics steps,
    /// not ticks: a tick spans 2 or 3 steps (rl#376). `None` until two poses arrive.
    pub(super) fn speed_mps(&self) -> Option<f32> {
        let [.., Some((s1, p1)), Some((s2, p2))] = self.buf else {
            return None;
        };
        // s2 > s1 by the push guard, so the divisor is never 0.
        Some(p1.pos.distance(p2.pos) * crab_world::physics::PHYSICS_HZ as f32 / (s2 - s1) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::cumulative_steps;

    #[test]
    fn pose_window_renders_uniform_velocity_through_the_staircase() {
        // The rl#264 pin: a body moving at CONSTANT velocity in physics time (0.1
        // units per step — craft-scale, so a multi-tick gap stays under the teleport
        // guard) must render at velocity UNIFORM IN RENDER TIME, even though the
        // 64:30 cadence bunches 2 vs 3 steps per tick — the old tick-fraction
        // interpolation surged ±50% on the 3-step ticks. When render time pauses and
        // jumps (a tick gap: both peers' clocks stall together on loss), the covered
        // distance must stay proportional to the jump — no surge, no shortfall.
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
            window.push(cumulative_steps(tick), pose_at(tick));
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
    fn speed_mps_is_uniform_through_the_staircase() {
        // rl#376: a body at CONSTANT velocity in physics time must read a CONSTANT
        // speed_mps. The tick-average form read 0.94×–1.41× of true speed as the
        // 64:30 cadence bunched 2 vs 3 steps per tick — the irregular wind wooshes.
        // Tick gaps (lost datagrams) must read exact too, not just on average.
        const STEP_METERS: f64 = 0.1;
        let true_mps = (STEP_METERS * crab_world::physics::PHYSICS_HZ as f64) as f32;
        let pose_at = |tick: u64| Pose {
            pos: Vec3::new(
                (cumulative_steps(tick) as f64 * STEP_METERS) as f32,
                0.0,
                0.0,
            ),
            orient: Quat::IDENTITY,
        };
        let mut window = PoseWindow::default();
        let mut checked = 0u32;
        for tick in 1..200u64 {
            if tick == 100 || tick == 101 {
                continue; // a 3-tick gap must not dent the reading either
            }
            window.push(cumulative_steps(tick), pose_at(tick));
            let Some(speed) = window.speed_mps() else {
                continue;
            };
            assert!(
                (speed - true_mps).abs() < 1e-3 * true_mps,
                "tick {tick}: speed_mps read {speed} for a body moving {true_mps} \
                 m/s — the staircase leaked into the speed signal"
            );
            checked += 1;
        }
        assert!(checked > 190, "the sweep must actually cover the run");
    }

    #[test]
    fn snapshot_stall_holds_still_then_resumes_forward() {
        // rl#273: during a snapshot stall the driver freezes its clock at
        // (last adopted tick, frac = 1.0) instead of letting frac wrap. Pin the
        // window's side of that contract: the frozen clock samples a CONSTANT pose,
        // and the resume sweep continues forward from exactly the held position.
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
            w.push(cumulative_steps(tick), pose_at(tick));
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
            w.push(cumulative_steps(tick), pose_at(tick));
            for f in 0..4 {
                let x = w.sample(tick, f as f32 / 4.0).unwrap().pos.x;
                assert!(x >= last - 1e-4, "resume must not rewind: {x} < {last}");
                last = x;
            }
        }
        assert!(last > held, "the resume sweep must actually move forward");
    }

    /// A captured non-finite pose (the solver blowing up in a tick's last physics
    /// step, before the next FixedUpdate's rescue) must never reach a sample: NaN
    /// would lerp into every consumer for the window's span, and a NaN distance
    /// can't trigger the teleport reset. The window holds the last finite pose and
    /// the respawn teleport-resets as usual.
    #[test]
    fn non_finite_poses_never_enter_the_window() {
        let at = |x: f32| Pose {
            pos: Vec3::new(x, 0.0, 0.0),
            orient: Quat::IDENTITY,
        };
        let mut w = PoseWindow::default();
        for tick in 1..=3u64 {
            w.push(cumulative_steps(tick), at(tick as f32 * 0.1));
        }
        w.push(cumulative_steps(4), at(f32::NAN));
        let held = w.sample(4, 0.5).expect("window still holds finite history");
        assert!(
            held.pos.is_finite(),
            "a NaN capture must not corrupt sampling: got {:?}",
            held.pos
        );
        // The rescue respawn lands across the arena: a normal teleport reset.
        w.push(cumulative_steps(5), at(50.0));
        assert_eq!(w.sample(5, 0.5).unwrap().pos.x, 50.0);
    }

    #[test]
    fn teleport_resets_the_window_instead_of_smearing() {
        let at = |x: f32| Pose {
            pos: Vec3::new(x, 0.0, 0.0),
            orient: Quat::IDENTITY,
        };
        let mut w = PoseWindow::default();
        for tick in 1..=3u64 {
            w.push(cumulative_steps(tick), at(tick as f32 * 0.1));
        }
        // A round-RESTART respawn: the next tick's pose is across the arena.
        w.push(cumulative_steps(4), at(100.0));
        assert_eq!(
            w.sample(4, 0.5).unwrap().pos.x,
            100.0,
            "the window must restart at the arrival pose — interpolating would smear \
             the teleport over the window's span"
        );
    }
}
