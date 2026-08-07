//! rl#349: the integrity-panic flight recorder. The rl#343 hard-fail prints one
//! post-hoc frame — too late to tell a one-tick solver impulse from a multi-tick
//! pump, or a joint pinned on its stop from a free swing. This ring keeps the
//! last [`TRACE_TICKS`] ticks of every env's joint (angle, rate) rows — read
//! straight off the raw observation, the sensor's one angle/rate implementation
//! — plus each tick's fastest part, and renders them into the panic message.

use std::collections::VecDeque;

use crate::bot::body::CrabJointId;
use crate::bot::sensor::OBS_SIZE;

use super::step::MaxPartSpeed;

/// ~0.75 s at 64 Hz — enough to see the whole approach of every spike class
/// observed so far (the rl#346 whip built in ≤3 ticks; a resonant pump shows
/// its period).
const TRACE_TICKS: usize = 48;

/// Joints quieter than this (rad/s) and inside their limits are elided from the
/// dump — a healthy stride is ~7 rad/s peak, so 20 keeps dumps readable while
/// missing nothing that could grow to the 300 rad/s violation class.
const QUIET_RATE: f32 = 20.0;

pub(crate) struct TraceTick {
    tick: u64,
    /// `[angle, rate]` per joint, straight from the raw obs row.
    joints: [[f32; 2]; CrabJointId::COUNT],
    speed: f32,
    lin: f32,
    ang: f32,
    part: Option<CrabJointId>,
}

pub(crate) struct IntegrityTrace {
    per_env: Vec<VecDeque<TraceTick>>,
}

impl IntegrityTrace {
    pub(crate) fn new(envs: usize) -> Self {
        Self {
            per_env: (0..envs)
                .map(|_| VecDeque::with_capacity(TRACE_TICKS))
                .collect(),
        }
    }

    /// Record env `e`'s tick from the raw obs row + the tick's speed scan.
    pub(crate) fn record(
        &mut self,
        e: usize,
        tick: u64,
        raw_obs: &[f32; OBS_SIZE],
        ms: &MaxPartSpeed,
    ) {
        let Some(ring) = self.per_env.get_mut(e) else {
            return;
        };
        let mut joints = [[0.0f32; 2]; CrabJointId::COUNT];
        for (i, j) in joints.iter_mut().enumerate() {
            j[0] = raw_obs[i * 2];
            j[1] = raw_obs[i * 2 + 1];
        }
        if ring.len() == TRACE_TICKS {
            ring.pop_front();
        }
        ring.push_back(TraceTick {
            tick,
            joints,
            speed: ms.speed,
            lin: ms.lin,
            ang: ms.ang,
            part: ms.part,
        });
    }

    /// Render env `e`'s ring for the violation panic: one line per tick, fastest
    /// part first, then every joint that is fast (|rate| > [`QUIET_RATE`]) or
    /// outside its limits, as `id angle(rad) @ rate(rad/s) [<lo|>hi if outside]`.
    pub(crate) fn dump(&self, e: usize) -> String {
        let Some(ring) = self.per_env.get(e) else {
            return "no trace for this env".to_string();
        };
        let mut out = String::new();
        for t in ring {
            let part = t
                .part
                .map_or("Carapace".to_string(), |id| format!("{id:?}"));
            out.push_str(&format!(
                "\n  tick {}: max {} lin {:.1} ang {:.1} (bound {:.1})",
                t.tick, part, t.lin, t.ang, t.speed
            ));
            for (i, [angle, rate]) in t.joints.iter().enumerate() {
                let id = CrabJointId::from_index(i).expect("trace row is joint-indexed");
                let [lo, hi] = id.limits();
                let outside = *angle < lo || *angle > hi;
                if rate.abs() > QUIET_RATE || outside {
                    let mark = if *angle < lo {
                        format!(" <lo({lo})")
                    } else if *angle > hi {
                        format!(" >hi({hi})")
                    } else {
                        String::new()
                    };
                    out.push_str(&format!(" | {id:?} {angle:.3} @ {rate:.0}{mark}"));
                }
            }
        }
        out
    }
}
