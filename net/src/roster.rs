//! The match participant set as it changes over a session: an append-only, tick-aligned
//! schedule of change-points (GCR MP Stage 3, bddap/rl#151).
//!
//! Stage 1+2 froze the roster at construction. Stage 3 makes it dynamic so a player can join
//! mid-match — but a roster change is determinism-critical: it MUST take effect on the identical
//! tick everywhere, or the server and its clients disagree on who is in the match. This type is the
//! ONE source of "who is in the match at tick T", consulted by the [`crate::server::Server`] (when is
//! a tick's input set complete, and when to spawn a scheduled joiner) so completeness and the
//! authoritative spawn key off the same set.
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
        let mut points = BTreeMap::new();
        points.insert(0, sorted(initial));
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

    /// The latest scheduled change-point tick (the construction baseline if none added). `admit`
    /// keeps a new change strictly after this even when the emit cursor hasn't advanced between two
    /// rapid joins, so the append-only invariant holds.
    pub fn latest_change_tick(&self) -> u64 {
        *self.points.keys().next_back().expect("non-empty schedule")
    }

    /// The earliest tick a NEW change may take effect: at or after the caller's `floor` (the emit
    /// cursor, plus any lead), and strictly after every existing change-point — so the value is
    /// always legal to [`Self::schedule_change`]. The ONE home for this formula, shared by
    /// [`Server::admit`](crate::server::Server::admit) and
    /// [`Server::depart`](crate::server::Server::depart) so the two can't drift on the append-only
    /// rule.
    pub fn earliest_change_at(&self, floor: u64) -> u64 {
        floor.max(self.latest_change_tick() + 1)
    }

    /// Schedule `set` to take effect from `effective_tick`. Append-only and strictly future: a NEW
    /// tick must be beyond every existing change-point, so a recorded change can never be rewritten
    /// (which would let the server and a client that applied it at different moments disagree).
    /// Re-scheduling an EXISTING change-point with the IDENTICAL set is an idempotent no-op — a roster
    /// change is content-addressed by `(tick, set)`, so a duplicate wire delivery (a host re-broadcast)
    /// is benign; only a CONFLICTING set at an existing tick is the append-only violation. The server
    /// picks `effective_tick` far enough ahead ([`JOIN_LEAD`](crate::server::JOIN_LEAD)) that every
    /// client learns of the change before it is due.
    pub fn schedule_change(&mut self, effective_tick: u64, set: &[PlayerId]) {
        let set = sorted(set);
        if let Some(existing) = self.points.get(&effective_tick) {
            // Enforced in RELEASE too (`assert!`, not `debug_assert!`): a CONFLICTING set at an
            // existing tick is a real append-only violation that would silently corrupt the roster —
            // the server and a client that applied the original change diverge from any that saw the
            // rewrite. Fail loud. An IDENTICAL re-delivery is the benign idempotent no-op below.
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
    }

    #[test]
    fn a_change_applies_from_its_tick_and_not_before() {
        let mut r = RosterSchedule::frozen(&ids(2));
        r.schedule_change(20, &ids(3));
        assert_eq!(r.at(19), ids(2).as_slice(), "old roster up to the boundary");
        assert_eq!(r.at(20), ids(3).as_slice(), "new roster from the boundary");
        assert_eq!(r.at(21), ids(3).as_slice());
        assert_eq!(r.current(), ids(3).as_slice());
        assert_eq!(r.latest_change_tick(), 20);
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
        // A duplicate wire delivery re-schedules the SAME (tick, set) — must be a benign no-op, not a
        // panic / overwrite.
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

}
