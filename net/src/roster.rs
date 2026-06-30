//! The match participant set as it changes over a session: an append-only, tick-aligned
//! schedule of change-points (GCR MP Stage 3, bddap/rl#151).
//!
//! Stage 1+2 froze the roster at construction. Stage 3 makes it dynamic so a player can join
//! mid-match — but a roster change is determinism-critical: it MUST take effect on the identical
//! tick on every peer, or the sims diverge. This type is the ONE source of "who is in the match at
//! tick T", consulted by BOTH the [`crate::server::Server`] (when is a tick's input set complete)
//! AND each client's [`crate::lockstep::Lockstep`] (which inputs are required + when to rebuild the
//! round for a join). Server and clients reading the same schedule is what keeps them agreeing on
//! the participant set tick-for-tick.
//!
//! With no changes scheduled the schedule is exactly the frozen initial set, so the no-join path is
//! byte-identical to the old fixed `Vec<PlayerId>` — there is ONE roster mechanism, not a parallel
//! dynamic one bolted beside the static path.

use std::collections::BTreeMap;

use crate::sim::PlayerId;

/// The participant set over time. Each change-point maps a tick to the COMPLETE participant set
/// effective FROM that tick until the next change-point. Non-empty by construction (a point at the
/// initial effective tick), so [`Self::at`] always resolves. Every set is sorted + deduped, so two
/// peers handed the same change in any order hold the identical schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterSchedule {
    /// Change-points, keyed by the tick they take effect. `points[t]` is the full roster effective
    /// from tick `t` onward (until the next key). The lowest key is the initial effective tick.
    points: BTreeMap<u64, Vec<PlayerId>>,
}

impl RosterSchedule {
    /// A schedule that is `initial` from tick 0 with no changes — the Stage-1+2 frozen-roster
    /// behaviour. `at` returns `initial` for every tick until a change is scheduled.
    pub fn frozen(initial: &[PlayerId]) -> Self {
        Self::starting_at(0, initial)
    }

    /// A schedule whose FIRST effective set is `set` from `effective_tick` — used for a joiner that
    /// enters the match at the live tick rather than from tick 0 (it never plays the pre-join
    /// ticks, so its schedule begins at the join). `at` is only ever queried at ticks
    /// `>= effective_tick` for such a peer.
    pub fn starting_at(effective_tick: u64, set: &[PlayerId]) -> Self {
        let mut points = BTreeMap::new();
        points.insert(effective_tick, sorted(set));
        Self { points }
    }

    /// The participant set effective at `tick`: the set of the latest change-point with key
    /// `<= tick`. Panics only if queried below the first effective tick, which no caller does
    /// (a peer never advances a tick before its roster exists).
    pub fn at(&self, tick: u64) -> &[PlayerId] {
        self.points
            .range(..=tick)
            .next_back()
            .map(|(_, set)| set.as_slice())
            .expect("queried the roster before its first effective tick")
    }

    /// The latest scheduled set (the current/most-future roster). Equals [`Self::at`] for any tick
    /// at or beyond the last change-point.
    pub fn current(&self) -> &[PlayerId] {
        self.points
            .last_key_value()
            .map(|(_, set)| set.as_slice())
            .expect("a schedule always holds at least the initial set")
    }

    /// The set effective EXACTLY from `tick`, if `tick` is a change-point. `None` at a tick that is
    /// not itself a change boundary (the roster simply carries over). Private: the only caller is
    /// [`Self::rebuild_at`] (the real cue), plus this module's tests.
    fn change_at(&self, tick: u64) -> Option<&[PlayerId]> {
        self.points.get(&tick).map(Vec::as_slice)
    }

    /// The latest scheduled change-point tick (the construction baseline if none added). `admit`
    /// keeps a new change strictly after this even when the emit cursor hasn't advanced between two
    /// rapid joins, so the append-only invariant holds.
    pub fn latest_change_tick(&self) -> u64 {
        *self.points.keys().next_back().expect("non-empty schedule")
    }

    /// The first tick this schedule covers — its construction baseline (0 for [`Self::frozen`], the
    /// join tick for [`Self::starting_at`]). The single source of "where this peer's participation
    /// begins": [`Self::rebuild_at`] excludes it (the round was already built for it) and a
    /// [`crate::lockstep::Lockstep`] uses it to ignore relayed hashes for ticks before it existed.
    pub fn baseline_tick(&self) -> u64 {
        *self.points.keys().next().expect("non-empty schedule")
    }

    /// The new set iff `tick` is a roster change that requires REBUILDING the round (a join) — i.e.
    /// a change-point other than the construction baseline (the schedule's earliest point, which the
    /// [`crate::sim::Sim`] was already built for, so rebuilding there would be a redundant reset).
    /// This is the cue [`crate::lockstep::Lockstep::advance_one`] keys the round-boundary join off.
    pub fn rebuild_at(&self, tick: u64) -> Option<&[PlayerId]> {
        (tick != self.baseline_tick())
            .then(|| self.change_at(tick))
            .flatten()
    }

    /// Schedule `set` to take effect from `effective_tick`. Append-only and strictly future: a NEW
    /// tick must be beyond every existing change-point, so a recorded change can never be rewritten
    /// (which would let two peers that applied it at different moments diverge). Re-scheduling an
    /// EXISTING change-point with the IDENTICAL set is an idempotent no-op — a roster change is
    /// content-addressed by `(tick, set)`, so a duplicate wire delivery (the joiner learning of its
    /// OWN boundary it already built via [`Self::starting_at`], or a host re-broadcast) is benign;
    /// only a CONFLICTING set at an existing tick is the append-only violation. The server picks
    /// `effective_tick` far enough ahead (≥ the input-delay lead) that every peer learns of the
    /// change before it is due.
    pub fn schedule_change(&mut self, effective_tick: u64, set: &[PlayerId]) {
        let set = sorted(set);
        if let Some(existing) = self.points.get(&effective_tick) {
            // Enforced in RELEASE too (`assert!`, not `debug_assert!`): a CONFLICTING set at an
            // existing tick is a real append-only violation that would silently corrupt the
            // roster in the release lockstep path — peers that applied the original change diverge
            // from any that saw the rewrite. Fail loud. An IDENTICAL re-delivery is the benign
            // idempotent no-op below, not a violation.
            assert_eq!(
                existing, &set,
                "roster changes are append-only: a change at tick {effective_tick} cannot be \
                 rewritten with a different set (identical re-delivery is a no-op)"
            );
            return; // idempotent: this exact change is already recorded
        }
        assert!(
            self.points.keys().all(|&t| effective_tick > t),
            "roster changes are append-only and strictly future: {effective_tick} must exceed every existing change-point"
        );
        self.points.insert(effective_tick, set);
    }
}

/// Sort + dedup a participant set so the schedule is order-independent (the determinism contract:
/// two peers handed the same set in any order store the identical bytes).
fn sorted(set: &[PlayerId]) -> Vec<PlayerId> {
    let mut v = set.to_vec();
    v.sort();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    #[test]
    fn frozen_returns_the_initial_set_for_every_tick() {
        let r = RosterSchedule::frozen(&ids(2));
        for t in [0, 1, 7, 1_000_000] {
            assert_eq!(r.at(t), ids(2).as_slice(), "frozen roster is constant");
        }
        assert_eq!(r.current(), ids(2).as_slice());
        assert_eq!(r.baseline_tick(), 0, "a frozen schedule begins at tick 0");
    }

    #[test]
    fn a_change_applies_from_its_tick_and_not_before() {
        let mut r = RosterSchedule::frozen(&ids(2));
        r.schedule_change(20, &ids(3));
        assert_eq!(r.at(19), ids(2).as_slice(), "old roster up to the boundary");
        assert_eq!(r.at(20), ids(3).as_slice(), "new roster from the boundary");
        assert_eq!(r.at(21), ids(3).as_slice());
        assert_eq!(
            r.change_at(20),
            Some(ids(3).as_slice()),
            "20 is a rebuild boundary"
        );
        assert_eq!(r.change_at(19), None, "19 is not a boundary");
        assert_eq!(r.change_at(21), None, "21 carries over, not a boundary");
        assert_eq!(r.current(), ids(3).as_slice());
    }

    #[test]
    fn order_independent_storage() {
        // The same change handed with the set in different orders stores identically.
        let mut a = RosterSchedule::frozen(&[PlayerId(1), PlayerId(0)]);
        let mut b = RosterSchedule::frozen(&[PlayerId(0), PlayerId(1)]);
        a.schedule_change(5, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        b.schedule_change(5, &[PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert_eq!(a, b, "set order must not affect the stored schedule");
    }

    #[test]
    fn re_scheduling_the_identical_change_is_an_idempotent_no_op() {
        // A duplicate wire delivery (or the joiner learning of its own already-built boundary)
        // re-schedules the SAME (tick, set) — must be a benign no-op, not a panic / overwrite.
        let mut r = RosterSchedule::frozen(&ids(2));
        r.schedule_change(20, &ids(3));
        r.schedule_change(20, &ids(3)); // identical re-delivery
        // Order-independent identical set is also a no-op (the schedule stores sorted).
        r.schedule_change(20, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        assert_eq!(r.at(20), ids(3).as_slice());
        assert_eq!(r.current(), ids(3).as_slice());
    }

    #[test]
    #[should_panic(expected = "append-only")]
    fn a_change_cannot_rewrite_an_earlier_or_equal_tick() {
        let mut r = RosterSchedule::frozen(&ids(2));
        r.schedule_change(20, &ids(3));
        r.schedule_change(20, &ids(4)); // same tick → not strictly future → rejected
    }

    #[test]
    fn starting_at_a_join_tick_resolves_from_there() {
        let r = RosterSchedule::starting_at(20, &ids(3));
        assert_eq!(
            r.at(20),
            ids(3).as_slice(),
            "the joiner's roster begins at the join tick"
        );
        assert_eq!(r.at(50), ids(3).as_slice());
        assert_eq!(r.baseline_tick(), 20, "the joiner's schedule begins at the join tick");
        // The join tick is the joiner's OWN baseline, so it is NOT a rebuild boundary for the joiner
        // (its sim was already built fresh over this roster) — only a LATER scheduled change rebuilds
        // it. The change-point still exists; it just isn't a redundant self-rebuild.
        assert_eq!(r.change_at(20), Some(ids(3).as_slice()), "the entry tick is a change-point");
        assert_eq!(r.rebuild_at(20), None, "but a joiner does not redundantly rebuild at its own entry");
    }
}
