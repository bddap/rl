
use std::collections::BTreeMap;

use crate::sim::PlayerId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterSchedule {
    points: BTreeMap<u64, Vec<PlayerId>>,
}

impl RosterSchedule {
    /// A schedule that is `initial` from tick 0 with no changes. `at` returns `initial` for
    /// every tick until a change is scheduled.
    pub fn frozen(initial: &[PlayerId]) -> Self {
        let mut points = BTreeMap::new();
        points.insert(0, sorted(initial));
        Self { points }
    }

    pub fn at(&self, tick: u64) -> &[PlayerId] {
        self.points
            .range(..=tick)
            .next_back()
            .map(|(_, set)| set.as_slice())
            .expect("queried the roster before its first effective tick")
    }

    pub fn current(&self) -> &[PlayerId] {
        self.points
            .last_key_value()
            .map(|(_, set)| set.as_slice())
            .expect("a schedule always holds at least the initial set")
    }

    pub fn latest_change_tick(&self) -> u64 {
        *self.points.keys().next_back().expect("non-empty schedule")
    }

    pub fn earliest_change_at(&self, floor: u64) -> u64 {
        floor.max(self.latest_change_tick() + 1)
    }

    pub fn schedule_change(&mut self, effective_tick: u64, set: &[PlayerId]) {
        let set = sorted(set);
        if let Some(existing) = self.points.get(&effective_tick) {
            assert_eq!(
                existing, &set,
                "roster changes are append-only: a change at tick {effective_tick} cannot be \
                 rewritten with a different set (identical re-delivery is a no-op)"
            );
            return;
        }
        assert!(
            self.points.keys().all(|&t| effective_tick > t),
            "roster changes are append-only and strictly future: {effective_tick} must exceed every existing change-point"
        );
        self.points.insert(effective_tick, set);
    }
}

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
        let mut a = RosterSchedule::frozen(&[PlayerId(1), PlayerId(0)]);
        let mut b = RosterSchedule::frozen(&[PlayerId(0), PlayerId(1)]);
        a.schedule_change(5, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        b.schedule_change(5, &[PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert_eq!(a, b, "set order must not affect the stored schedule");
    }

    #[test]
    fn re_scheduling_the_identical_change_is_an_idempotent_no_op() {
        let mut r = RosterSchedule::frozen(&ids(2));
        r.schedule_change(20, &ids(3));
        r.schedule_change(20, &ids(3));
        r.schedule_change(20, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        assert_eq!(r.at(20), ids(3).as_slice());
        assert_eq!(r.current(), ids(3).as_slice());
    }

    #[test]
    #[should_panic(expected = "append-only")]
    fn a_change_cannot_rewrite_an_earlier_or_equal_tick() {
        let mut r = RosterSchedule::frozen(&ids(2));
        r.schedule_change(20, &ids(3));
        r.schedule_change(20, &ids(4));
    }
}
