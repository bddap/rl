//! THE eval wire schema — the single emitter of every `EVAL_RESULT*` line, owned
//! next to [`run_eval`](super::run_eval) so the report struct and its serialization
//! can never drift (rl#270; the lines used to be hand-formatted in `rl-train`'s
//! `main`, each new metric dodging the existing consumers' greps with a fresh
//! prefix).
//!
//! # Schema — v2 (`schema=2`, the rl#341 mean-pair flip)
//!
//! Line-oriented `key=value` records on stdout, one prefix per record kind. The
//! prefix set is CLOSED — a new metric is a new KEY on the right record, never a new
//! prefix:
//!
//! - `EVAL_RESULT_BEARING deg= start_x= start_z= progress_m= net_progress_m=
//!   closest_m= tip_m= final_m= total_torque= saturation= work_j= reached= rescues=
//!   ticks=` — one per far (heading, start) pair, plus ` j_per_m=` when that pair
//!   cleared the rl#279 progress floor. `start_x`/`start_z` are the pair's spawn
//!   locale (the arena provenance a locale index can't carry); `rescues` is the
//!   rl#341 S1-1 physics-fault tally — a nonzero pair's `progress_m` is a forfeited
//!   0. `net_progress_m` (rl#310) is the SIGNED initial−final diagnostic — negative
//!   when the episode ended farther than it started, which the zero-floored
//!   `progress_m` structurally hides.
//! - `EVAL_RESULT_CLOSE_BEARING …` — the same keys, one per close-probe bearing
//!   (rl#252); diagnostic, no consumer parses them today.
//! - `EVAL_RESULT_CLOSE progress_m= closest_m= tip_m= target_m= reached_count=
//!   bearings= worst_deg=` — the close probe's worst-bearing summary (the probe kept
//!   its compass shape and baselines; its keys are unchanged from v1).
//! - `EVAL_RESULT schema=2 progress_m= net_progress_m= min_pair_m= min_pair_deg=
//!   total_torque= target_m= reached= reached_count= pairs= rescued_pairs= ticks=
//!   settle= policy_loaded= saturation_mean= amplitude= bake= plant=` — the
//!   HEADLINE, last.
//!   `progress_m` is the MEAN over the far pairs of zero-floored best-approach
//!   progress (owner 08-03; v1's median-of-min is retired — `schema=2` marks the
//!   ruler change for any long-run series). `net_progress_m`/`total_torque`/
//!   `saturation_mean` are means over the same pairs. Tail statistics ride beside
//!   the mean so a hole can't hide in it: `min_pair_m`/`min_pair_deg` (the worst
//!   pair), `rescued_pairs` (physics-fault forfeits), `reached_count` (pairs whose
//!   claw struck), and `reached` (= reached_count > 0, kept because the monitor
//!   requires the key). Provenance rides with the data (rl#341 S2-3): `amplitude=`
//!   (the terrain relief scalar), `bake=` (the terrain artifact digest, hex), and
//!   `plant=` (the constructed plant digest, hex — a fallback-body eval is
//!   distinguishable, S3-4). `settle=` is the settle window in ticks — the one ruler
//!   knob no other key witnessed (it re-bases `initial_dist`, hence every historical
//!   `progress_m`, review round 1). When any episode exploded past the position
//!   bound it appends ` plant_unbounded=true` (bddap/rl#315 — the exploded pairs
//!   forfeited and `rl-train eval` also hard-fails; a bounded eval never prints the
//!   key). When measurable it appends ` j_per_m_mean=` and the
//!   rl#266 charge keys `charge_heights_per_s= charge_pinned= charge_drift_frac=
//!   charge_drifted= charge_target_m=`; when the charge guard SHOULD have measured
//!   and could not it appends ` charge_unmeasurable=true` instead (rl#341 S2-4 — the
//!   keys used to just vanish and no consumer watches for absence).
//!
//! # Consumer contract
//!
//! Consumers (rl-eval-monitor's `field()`, bothouse) select a record by its prefix
//! and extract values BY KEY NAME, never by position. Under that contract adding a
//! key is always safe; renaming or removing one is a breaking change to grep for in
//! bothouse first. Values are `{:.4}`/`{:.2}`/`{:.0}` decimals, `true`/`false`
//! booleans, bare integers, or `%016x` hex digests; a never-measured tip distance
//! prints as `inf`, which by-name consumers must tolerate on keys they don't
//! validate.

use std::fmt::Write;

use super::{CHARGE_SPEED_DRIFT_TOL, CRAB_CHARGE_SPEED_HEIGHTS_PER_S, EVAL_BEARINGS, EvalReport};

impl EvalReport {
    /// Every wire line of one eval, newline-terminated, in schema order (pair
    /// profiles first, headline last).
    pub fn wire_report(&self) -> String {
        let mut out = String::new();
        bearing_lines(&mut out, "EVAL_RESULT_BEARING", &self.far.pairs);
        bearing_lines(
            &mut out,
            "EVAL_RESULT_CLOSE_BEARING",
            &self.close.per_bearing,
        );

        let close_worst = self.close.worst();
        writeln!(
            out,
            "EVAL_RESULT_CLOSE progress_m={:.4} closest_m={:.4} tip_m={:.4} target_m={:.2} \
             reached_count={} bearings={} worst_deg={:.0}",
            close_worst.progress_m,
            close_worst.closest_distance_m,
            close_worst.closest_tip_distance_m,
            self.close.target_distance_m,
            self.close.reached_count(),
            EVAL_BEARINGS,
            close_worst.bearing_rad.to_degrees(),
        )
        .expect("writing to a String never fails");

        let min_pair = self.far.min_pair();
        write!(
            out,
            "EVAL_RESULT schema=2 progress_m={:.4} net_progress_m={:.4} min_pair_m={:.4} \
             min_pair_deg={:.0} total_torque={:.2} target_m={:.2} reached={} \
             reached_count={} pairs={} rescued_pairs={} ticks={} settle={} policy_loaded={} \
             saturation_mean={:.4} amplitude={:.4} bake={:016x} plant={:016x}",
            self.far.mean_progress_m(),
            self.far.mean_net_progress_m(),
            min_pair.progress_m,
            min_pair.bearing_rad.to_degrees(),
            self.far.mean_total_torque(),
            self.far.target_distance_m,
            self.far.reached_count() > 0,
            self.far.reached_count(),
            self.far.pairs.len(),
            self.far.rescued_pairs(),
            min_pair.active_ticks,
            crate::bot::RESET_GRACE_TICKS,
            self.policy_loaded,
            self.far.mean_saturation(),
            self.terrain_amplitude,
            crate::terrain::gcr_bake_digest(),
            crate::bot::body::constructed_plant_digest(),
        )
        .expect("writing to a String never fails");
        if self.plant_unbounded() {
            write!(out, " plant_unbounded=true").expect("writing to a String never fails");
        }
        if let Some(jpm_mean) = self.far.mean_j_per_m() {
            write!(out, " j_per_m_mean={jpm_mean:.2}").expect("writing to a String never fails");
        }
        if let (Some(measured), Some(drift)) = (
            self.measured_charge_heights_per_s(),
            self.charge_speed_drift(),
        ) {
            write!(
                out,
                " charge_heights_per_s={measured:.4} charge_pinned={:.4} \
                 charge_drift_frac={drift:.4} charge_drifted={} charge_target_m={:.2}",
                CRAB_CHARGE_SPEED_HEIGHTS_PER_S,
                drift.abs() > CHARGE_SPEED_DRIFT_TOL,
                self.pace.target_distance_m,
            )
            .expect("writing to a String never fails");
        } else if self.charge_unmeasurable() {
            write!(out, " charge_unmeasurable=true").expect("writing to a String never fails");
        }
        out.push('\n');
        out
    }
}

fn bearing_lines(out: &mut String, prefix: &str, reports: &[super::BearingReport]) {
    for b in reports {
        write!(
            out,
            "{prefix} deg={:.0} start_x={:.0} start_z={:.0} progress_m={:.4} \
             net_progress_m={:.4} closest_m={:.4} tip_m={:.4} final_m={:.4} \
             total_torque={:.2} saturation={:.4} work_j={:.2} reached={} rescues={} ticks={}",
            b.bearing_rad.to_degrees(),
            b.start_xz.x,
            b.start_xz.y,
            b.progress_m,
            b.net_progress_m(),
            b.closest_distance_m,
            b.closest_tip_distance_m,
            b.final_distance_m,
            b.total_torque,
            b.saturation,
            b.work_j,
            b.reached,
            b.rescues,
            b.active_ticks,
        )
        .expect("writing to a String never fails");
        if let Some(jpm) = b.j_per_m() {
            write!(out, " j_per_m={jpm:.2}").expect("writing to a String never fails");
        }
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::Vec2;

    use super::super::{
        BearingReport, CLOSE_PROBE_DISTANCE_M, CompassSweep, DEFAULT_TARGET_DISTANCE_M, EVAL_PAIRS,
        PACE_PROBE_DISTANCE_M, PairSweep,
    };
    use super::*;

    fn report(policy_loaded: bool, pace: f32) -> EvalReport {
        let bearing = |i: usize| BearingReport {
            bearing_rad: (i % EVAL_BEARINGS) as f32 * std::f32::consts::TAU / EVAL_BEARINGS as f32,
            // 50 − 50i, not −50i: pair 0's z would be −0.0 and print "-0".
            start_xz: Vec2::new(100.0 * i as f32, 50.0 - 50.0 * i as f32),
            progress_m: 1.0 + i as f32,
            total_torque: 10.0,
            saturation: 0.25,
            // 2 J per metre closed at every bearing, so pair and mean J/m agree.
            work_j: 2.0 * (1.0 + i as f32),
            initial_distance_m: 9.0,
            closest_distance_m: 8.0 - i as f32,
            final_distance_m: 8.5,
            closest_tip_distance_m: f32::INFINITY,
            reached: false,
            active_ticks: 200,
            rescues: 0,
            unbounded: false,
            sustained_pace_m_per_s: pace,
        };
        let sweep = |target: f32| CompassSweep {
            target_distance_m: target,
            per_bearing: std::array::from_fn(bearing),
        };
        EvalReport {
            policy_loaded,
            terrain_amplitude: 1.0,
            far: PairSweep {
                target_distance_m: DEFAULT_TARGET_DISTANCE_M,
                pairs: (0..EVAL_PAIRS).map(bearing).collect(),
            },
            close: sweep(CLOSE_PROBE_DISTANCE_M),
            pace: sweep(PACE_PROBE_DISTANCE_M),
        }
    }

    /// Pins the wire format the bothouse consumers grep: prefixes, key names, key
    /// order, line order — the headline last, its `progress_m` the PAIR MEAN
    /// (schema=2), tail statistics and provenance beside it.
    #[test]
    fn wire_report_matches_the_pinned_schema() {
        let wire = report(true, 0.0).wire_report();
        let lines: Vec<&str> = wire.lines().collect();
        assert_eq!(lines.len(), EVAL_PAIRS + EVAL_BEARINGS + 2);

        assert_eq!(
            lines[0],
            "EVAL_RESULT_BEARING deg=0 start_x=0 start_z=50 progress_m=1.0000 \
             net_progress_m=0.5000 closest_m=8.0000 tip_m=inf final_m=8.5000 \
             total_torque=10.00 saturation=0.2500 work_j=2.00 reached=false rescues=0 \
             ticks=200 j_per_m=2.00"
        );
        assert!(lines[1].starts_with("EVAL_RESULT_BEARING deg=45 start_x=100 start_z=0 "));
        assert!(lines[EVAL_PAIRS].starts_with("EVAL_RESULT_CLOSE_BEARING deg=0 "));
        assert_eq!(
            lines[EVAL_PAIRS + EVAL_BEARINGS],
            "EVAL_RESULT_CLOSE progress_m=1.0000 closest_m=8.0000 tip_m=inf target_m=1.00 \
             reached_count=0 bearings=8 worst_deg=0"
        );
        let headline = lines[EVAL_PAIRS + EVAL_BEARINGS + 1];
        // progress_m = mean of 1..=32 = 16.5; min pair is pair 0 at 1.0 m, bearing 0.
        let expect_mean = (0..EVAL_PAIRS).map(|i| 1.0 + i as f32).sum::<f32>() / EVAL_PAIRS as f32;
        let expect_prefix = format!(
            "EVAL_RESULT schema=2 progress_m={expect_mean:.4} net_progress_m=0.5000 \
             min_pair_m=1.0000 min_pair_deg=0 total_torque=10.00 target_m=24.00 \
             reached=false reached_count=0 pairs={EVAL_PAIRS} rescued_pairs=0 ticks=200 \
             settle={} policy_loaded=true saturation_mean=0.2500 amplitude=1.0000",
            crate::bot::RESET_GRACE_TICKS,
        );
        assert!(
            headline.starts_with(&expect_prefix),
            "headline {headline}\nvs expected prefix {expect_prefix}"
        );
        assert!(headline.contains(" bake="));
        assert!(headline.contains(" plant="));
        assert!(headline.contains(" j_per_m_mean="));
        assert!(wire.ends_with('\n'));
    }

    /// The rl#279 guard on the wire: saturation keys always print; J/m keys vanish
    /// (never a nan/inf token) on every line whose pair missed the progress floor.
    #[test]
    fn j_per_m_keys_are_guarded() {
        let mut r = report(true, 0.0);
        for b in r.far.pairs.iter_mut() {
            b.progress_m = 0.1;
        }
        for b in r.close.per_bearing.iter_mut() {
            b.progress_m = 0.1;
        }
        let wire = r.wire_report();
        let headline = wire.lines().last().unwrap();
        assert!(headline.contains(" saturation_mean=0.2500"));
        assert!(!wire.contains(" j_per_m="));
        assert!(!headline.contains(" j_per_m_mean="));
    }

    /// rl#310 on the wire: retreating pairs print a SIGNED negative
    /// `net_progress_m=` token while the zero-floored `progress_m=` stays 0 — the
    /// decomposition of a flat headline into parked vs retreating that by-name
    /// consumers (rl-eval-monitor) record.
    #[test]
    fn net_progress_key_is_signed() {
        let mut r = report(true, 0.0);
        for b in r.far.pairs.iter_mut() {
            b.progress_m = 0.0;
            b.final_distance_m = b.initial_distance_m + 2.0;
        }
        let wire = r.wire_report();
        let headline = wire.lines().last().unwrap();
        assert!(headline.contains(" progress_m=0.0000"));
        assert!(headline.contains(" net_progress_m=-2.0000"));
    }

    /// rl#341 S1-1 on the wire: rescued pairs are visible per-pair (`rescues=`) and
    /// in the headline tail (`rescued_pairs=`), and their forfeited zeros drag the
    /// mean.
    #[test]
    fn rescues_ride_the_wire() {
        let mut r = report(true, 0.0);
        r.far.pairs[2].rescues = 3;
        r.far.pairs[2].progress_m = 0.0; // the fold forfeits before the wire sees it
        let wire = r.wire_report();
        assert!(wire.lines().nth(2).unwrap().contains(" rescues=3 "));
        assert!(wire.lines().last().unwrap().contains(" rescued_pairs=1 "));
    }

    /// The charge keys ride the headline only when the guard can measure (rl#266);
    /// the rest-pose baseline carries neither the keys nor the unmeasurable marker,
    /// and a loaded-but-paceless full-budget eval carries `charge_unmeasurable=true`
    /// (rl#341 S2-4 — absence is no longer the only signal).
    #[test]
    fn plant_unbounded_is_a_conditional_headline_key() {
        // rl#315 on the wire: one exploded episode anywhere in the eval stamps the
        // headline; a fully bounded eval never prints the key, so no pre-#315
        // consumer sees a token it doesn't know.
        let bounded = report(true, 0.0).wire_report();
        assert!(!bounded.contains("plant_unbounded"));

        let mut r = report(true, 0.0);
        r.pace.per_bearing[3].unbounded = true;
        let wire = r.wire_report();
        let headline = wire.lines().last().unwrap();
        assert!(headline.contains(" plant_unbounded=true"));
    }

    #[test]
    fn charge_keys_are_conditional_headline_keys() {
        let h = crate::bot::rig::natural_body_height().expect("rig height measures");
        let mut paced_r = report(true, CRAB_CHARGE_SPEED_HEIGHTS_PER_S * h);
        for b in &mut paced_r.pace.per_bearing {
            b.active_ticks = super::super::PACE_PROBE_TICKS;
        }
        let paced = paced_r.wire_report();
        let headline = paced.lines().last().unwrap();
        assert!(headline.starts_with("EVAL_RESULT schema=2 progress_m="));
        assert!(headline.contains(" charge_heights_per_s="));
        // The pin's VALUE is a sanctioned re-measure chore (rl#266) — assert the key
        // and format, not the number, so a re-pin can't break the schema test.
        assert!(headline.contains(&format!(
            " charge_pinned={CRAB_CHARGE_SPEED_HEIGHTS_PER_S:.4}"
        )));
        assert!(headline.contains(" charge_drift_frac="));
        assert!(headline.contains(" charge_drifted=false"));
        assert!(headline.ends_with(&format!(" charge_target_m={PACE_PROBE_DISTANCE_M:.2}")));
        assert!(!headline.contains("charge_unmeasurable"));

        let baseline = report(false, 1.0).wire_report();
        assert!(!baseline.contains("charge_"), "rest pose measures nothing");

        let mut dead = report(true, 0.0);
        for b in &mut dead.pace.per_bearing {
            b.active_ticks = super::super::PACE_PROBE_TICKS;
        }
        let wire = dead.wire_report();
        assert!(
            wire.lines()
                .last()
                .unwrap()
                .ends_with(" charge_unmeasurable=true")
        );
    }
}
