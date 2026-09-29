//! The chase eval end to end through real batched bevy+rapier worlds. Its own test
//! binary: `run_eval` pins rayon to one thread, which fails once another test in the
//! same process has built the global pool.

use crab_world::eval::{CLOSE_PROBE_DISTANCE_M, DEFAULT_TARGET_DISTANCE_M, EVAL_PAIRS, run_eval};
use crab_world::training::targets::REACH_RADIUS;

/// The end-to-end physical path (rl#341 S2-5): `run_eval` on an empty
/// (rest-pose) checkpoint through real worlds — batched far pairs, close and
/// pace probes — with the full fold and every guard live.
#[test]
fn rest_pose_has_zero_torque_and_no_progress() {
    let dir = std::env::temp_dir().join(format!("rl-eval-restpose-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Stamp the resolved plant: since the rl#293 strict adoption a recordless dir
    // REFUSES (unknown ground provenance), so the legitimate no-brain-yet
    // baseline is a plant-stamped dir with no brain — what a fresh run's
    // checkpoint dir looks like before the first save.
    crab_world::bot::body::record_plant(&dir).unwrap();

    let r = run_eval(&dir, 200, DEFAULT_TARGET_DISTANCE_M, 1.0)
        .expect("an absent checkpoint is the legitimate baseline, never a refusal");

    assert!(!r.policy_loaded, "an empty dir loads no policy (rest pose)");
    assert_eq!(r.far.pairs.len(), EVAL_PAIRS);
    assert_eq!(
        r.far.reached_count(),
        0,
        "a rest-pose crab never reaches a far ball"
    );
    // Zero floor: under the tip-based touch (rl#253) even the real Sally's
    // full-episode slump never brings a claw tip near the ball (see
    // CLOSE_PROBE_DISTANCE_M), so rest-pose reached_count is 0 on every body.
    assert_eq!(r.close.reached_count(), 0);
    assert_eq!(r.close.target_distance_m, CLOSE_PROBE_DISTANCE_M);
    for b in &r.close.per_bearing {
        assert_eq!(b.total_torque, 0.0);
        assert!(
            b.initial_distance_m > REACH_RADIUS,
            "close probe starts outside reach ({} m) at bearing {:.0}°",
            b.initial_distance_m,
            b.bearing_rad.to_degrees()
        );
        assert!(
            b.initial_distance_m < DEFAULT_TARGET_DISTANCE_M,
            "close probe is the CLOSE sweep, not another far one"
        );
    }
    assert!(
        (0.0..1.0).contains(&r.progress_m()),
        "rest-pose mean progress should be ~0, got {} m",
        r.progress_m()
    );
    for b in &r.far.pairs {
        assert_eq!(
            b.total_torque, 0.0,
            "the rest pose applies no joint torque, so total_torque must be exactly 0"
        );
        assert_eq!(b.saturation, 0.0, "zero drive saturates nothing");
        assert_eq!(b.work_j, 0.0, "zero torque does zero mechanical work");
        // Passive slump CAN clear the J/m floor (~0.55 m on some bodies) —
        // measurable or not, zero work means a zero cost of transport.
        assert_eq!(b.j_per_m().unwrap_or(0.0), 0.0);
        assert_eq!(b.active_ticks, 200, "all active ticks are measured");
        assert_eq!(b.rescues, 0, "a resting crab is never rescued");
        assert!(
            b.initial_distance_m.is_finite() && b.closest_distance_m.is_finite(),
            "distances are real finite metres"
        );
        assert!(
            b.initial_distance_m > REACH_RADIUS,
            "the ball starts far outside reach ({} m) at bearing {:.0}°",
            b.initial_distance_m,
            b.bearing_rad.to_degrees()
        );
        assert!(
            (0.0..1.5).contains(&b.progress_m),
            "rest pose shuffles nowhere at bearing {:.0}° start ({:.0},{:.0}), got {} m",
            b.bearing_rad.to_degrees(),
            b.start_xz.x,
            b.start_xz.y,
            b.progress_m
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
