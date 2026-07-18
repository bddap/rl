//! THE eval wire schema — the single emitter of every `EVAL_RESULT*` line, owned
//! next to [`run_eval`](super::run_eval) so the report struct and its serialization
//! can never drift (rl#270; the lines used to be hand-formatted in `rl-train`'s
//! `main`, each new metric dodging the existing consumers' greps with a fresh
//! prefix).
//!
//! # Schema
//!
//! Line-oriented `key=value` records on stdout, one prefix per record kind. The
//! prefix set is CLOSED — a new metric is a new KEY on the right record, never a new
//! prefix:
//!
//! - `EVAL_RESULT_BEARING deg= progress_m= closest_m= tip_m= final_m= total_torque=
//!   saturation= work_j= reached= ticks= locale=` — one per far-compass bearing
//!   (rl#239) per locale (rl#293: the far sweep repeats over `EVAL_LOCALES` terrain
//!   locales), plus ` j_per_m=` when that bearing cleared the rl#279 progress floor.
//! - `EVAL_RESULT_CLOSE_BEARING …` — the same keys, one per close-probe bearing
//!   (rl#252); diagnostic, no consumer parses them today.
//! - `EVAL_RESULT_CLOSE progress_m= closest_m= tip_m= target_m= reached_count=
//!   bearings= worst_deg=` — the close probe's worst-bearing summary.
//! - `EVAL_RESULT progress_m= total_torque= mean_torque_per_tick= initial_m=
//!   closest_m= tip_m= final_m= target_m= reached= ticks= policy_loaded= bearings=
//!   worst_deg= saturation_mean= locales= locale_worst_m=` — the HEADLINE, last: its
//!   numbers describe the MEDIAN locale's WORST bearing (rl#293 — `progress_m` is
//!   the median-over-locales of min-over-bearings); `locale_worst_m` lists every
//!   locale's min-over-bearings progress comma-joined in locale order;
//!   `saturation_mean` (rl#279) is that compass's mean torque saturation in [0, 1]. When measurable it appends ` j_per_m=` (the worst
//!   bearing's cost of transport) and ` j_per_m_mean=` (mean over the bearings past
//!   the progress floor) — effort is REPORTED here, never folded into the headline
//!   scalar or the keep-best gate. When the rl#266 charge-speed guard can measure,
//!   it appends
//!   `charge_heights_per_s= charge_pinned= charge_drift_frac= charge_drifted=
//!   charge_target_m=` (the first four replaced the old `EVAL_CHARGE_SPEED` sidecar
//!   line, which no consumer parsed — and which the release gate's `EVAL_RESULT`
//!   line filter never even logged; `charge_target_m` is the rl#280 pace probe's
//!   ball distance, which the headline's own `target_m` no longer describes).
//!
//! # Consumer contract
//!
//! Consumers (rl-eval-monitor's `field()`, bothouse) select a record by its prefix
//! and extract values BY KEY NAME, never by position. Under that contract adding a
//! key is always safe; renaming or removing one is a breaking change to grep for in
//! bothouse first. Values are `{:.4}`/`{:.2}`/`{:.0}` decimals, `true`/`false`
//! booleans, or bare integers; a never-measured tip distance prints as `inf`, which
//! by-name consumers must tolerate on keys they don't validate.

use std::fmt::Write;

use super::{
    CHARGE_SPEED_DRIFT_TOL, CRAB_CHARGE_SPEED_HEIGHTS_PER_S, CompassSweep, EVAL_BEARINGS,
    EVAL_LOCALES, EvalReport,
};

impl EvalReport {
    /// Every wire line of one eval, newline-terminated, in schema order (bearing
    /// profiles first, headline last). Far bearing lines carry a `locale=` key
    /// (rl#293) and repeat per locale; the headline's episode fields describe the
    /// MEDIAN locale's worst bearing and its `locale_worst_m=` key lists every
    /// locale's min-over-bearings progress, comma-joined in locale order.
    pub fn wire_report(&self) -> String {
        let mut out = String::new();
        for (locale, sweep) in self.far.iter().enumerate() {
            bearing_lines(&mut out, "EVAL_RESULT_BEARING", sweep, Some(locale));
        }
        bearing_lines(&mut out, "EVAL_RESULT_CLOSE_BEARING", &self.close, None);

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

        let median = self.median_far();
        let worst = median.worst();
        write!(
            out,
            "EVAL_RESULT progress_m={:.4} total_torque={:.2} mean_torque_per_tick={:.4} \
             initial_m={:.4} closest_m={:.4} tip_m={:.4} final_m={:.4} target_m={:.2} \
             reached={} ticks={} policy_loaded={} bearings={} worst_deg={:.0} \
             saturation_mean={:.4} locales={} locale_worst_m={}",
            worst.progress_m,
            worst.total_torque,
            worst.mean_torque_per_tick,
            worst.initial_distance_m,
            worst.closest_distance_m,
            worst.closest_tip_distance_m,
            worst.final_distance_m,
            median.target_distance_m,
            worst.reached,
            worst.active_ticks,
            self.policy_loaded,
            EVAL_BEARINGS,
            worst.bearing_rad.to_degrees(),
            median.mean_saturation(),
            EVAL_LOCALES,
            self.far
                .iter()
                .map(|s| format!("{:.4}", s.worst().progress_m))
                .collect::<Vec<_>>()
                .join(","),
        )
        .expect("writing to a String never fails");
        if let Some(jpm) = worst.j_per_m() {
            write!(out, " j_per_m={jpm:.2}").expect("writing to a String never fails");
        }
        if let Some(jpm_mean) = median.mean_j_per_m() {
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
        }
        out.push('\n');
        out
    }
}

fn bearing_lines(out: &mut String, prefix: &str, sweep: &CompassSweep, locale: Option<usize>) {
    for b in &sweep.per_bearing {
        write!(
            out,
            "{prefix} deg={:.0} progress_m={:.4} closest_m={:.4} tip_m={:.4} final_m={:.4} \
             total_torque={:.2} saturation={:.4} work_j={:.2} reached={} ticks={}",
            b.bearing_rad.to_degrees(),
            b.progress_m,
            b.closest_distance_m,
            b.closest_tip_distance_m,
            b.final_distance_m,
            b.total_torque,
            b.saturation,
            b.work_j,
            b.reached,
            b.active_ticks,
        )
        .expect("writing to a String never fails");
        if let Some(l) = locale {
            write!(out, " locale={l}").expect("writing to a String never fails");
        }
        if let Some(jpm) = b.j_per_m() {
            write!(out, " j_per_m={jpm:.2}").expect("writing to a String never fails");
        }
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        BearingReport, CLOSE_PROBE_DISTANCE_M, DEFAULT_TARGET_DISTANCE_M, PACE_PROBE_DISTANCE_M,
    };
    use super::*;

    fn report(policy_loaded: bool, pace: f32) -> EvalReport {
        let bearing = |i: usize| BearingReport {
            bearing_rad: i as f32 * std::f32::consts::TAU / EVAL_BEARINGS as f32,
            progress_m: 1.0 + i as f32,
            total_torque: 10.0,
            mean_torque_per_tick: 0.5,
            saturation: 0.25,
            // 2 J per metre closed at every bearing, so worst and mean J/m agree.
            work_j: 2.0 * (1.0 + i as f32),
            initial_distance_m: 9.0,
            closest_distance_m: 8.0 - i as f32,
            final_distance_m: 8.5,
            closest_tip_distance_m: f32::INFINITY,
            reached: false,
            active_ticks: 200,
            sustained_pace_m_per_s: pace,
        };
        let sweep = |target: f32| CompassSweep {
            target_distance_m: target,
            per_bearing: std::array::from_fn(bearing),
        };
        EvalReport {
            policy_loaded,
            far: [sweep(DEFAULT_TARGET_DISTANCE_M); EVAL_LOCALES],
            close: sweep(CLOSE_PROBE_DISTANCE_M),
            pace: sweep(PACE_PROBE_DISTANCE_M),
        }
    }

    /// Pins the wire format the bothouse consumers grep: prefixes, key names, key
    /// order, line order — the headline last, describing the worst (bearing-0) run.
    #[test]
    fn wire_report_matches_the_pinned_schema() {
        let wire = report(true, 0.0).wire_report();
        let lines: Vec<&str> = wire.lines().collect();
        assert_eq!(lines.len(), (EVAL_LOCALES + 1) * EVAL_BEARINGS + 2);

        assert_eq!(
            lines[0],
            "EVAL_RESULT_BEARING deg=0 progress_m=1.0000 closest_m=8.0000 tip_m=inf \
             final_m=8.5000 total_torque=10.00 saturation=0.2500 work_j=2.00 reached=false \
             ticks=200 locale=0 j_per_m=2.00"
        );
        assert!(lines[EVAL_BEARINGS].contains(" locale=1 "));
        assert!(
            lines[EVAL_LOCALES * EVAL_BEARINGS].starts_with("EVAL_RESULT_CLOSE_BEARING deg=0 ")
        );
        assert_eq!(
            lines[(EVAL_LOCALES + 1) * EVAL_BEARINGS],
            "EVAL_RESULT_CLOSE progress_m=1.0000 closest_m=8.0000 tip_m=inf target_m=1.00 \
             reached_count=0 bearings=8 worst_deg=0"
        );
        assert_eq!(
            lines[(EVAL_LOCALES + 1) * EVAL_BEARINGS + 1],
            "EVAL_RESULT progress_m=1.0000 total_torque=10.00 mean_torque_per_tick=0.5000 \
             initial_m=9.0000 closest_m=8.0000 tip_m=inf final_m=8.5000 target_m=24.00 \
             reached=false ticks=200 policy_loaded=true bearings=8 worst_deg=0 \
             saturation_mean=0.2500 locales=3 locale_worst_m=1.0000,1.0000,1.0000 \
             j_per_m=2.00 j_per_m_mean=2.00"
        );
        assert!(wire.ends_with('\n'));
    }

    /// The rl#279 guard on the wire: saturation keys always print; J/m keys vanish
    /// (never a nan/inf token) on every line whose bearing missed the progress floor.
    #[test]
    fn j_per_m_keys_are_guarded() {
        let mut r = report(true, 0.0);
        for b in r.far.iter_mut().flat_map(|s| s.per_bearing.iter_mut()) {
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

    /// The charge keys ride the headline only when the guard can measure (rl#266):
    /// present for a loaded pacing policy, absent for the rest-pose baseline.
    #[test]
    fn charge_keys_are_conditional_headline_keys() {
        let h = crate::mesh_fallback::natural_body_height().expect("rig height measures");
        let paced = report(true, CRAB_CHARGE_SPEED_HEIGHTS_PER_S * h).wire_report();
        let headline = paced.lines().last().unwrap();
        assert!(headline.starts_with("EVAL_RESULT progress_m="));
        assert!(headline.contains(" charge_heights_per_s="));
        // The pin's VALUE is a sanctioned re-measure chore (rl#266) — assert the key
        // and format, not the number, so a re-pin can't break the schema test.
        assert!(headline.contains(&format!(
            " charge_pinned={CRAB_CHARGE_SPEED_HEIGHTS_PER_S:.4}"
        )));
        assert!(headline.contains(" charge_drift_frac="));
        assert!(headline.contains(" charge_drifted=false"));
        assert!(headline.ends_with(&format!(" charge_target_m={PACE_PROBE_DISTANCE_M:.2}")));

        let baseline = report(false, 1.0).wire_report();
        assert!(!baseline.contains("charge_"), "rest pose measures nothing");
    }
}
