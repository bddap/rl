//! Deterministic simulation core for lockstep multiplayer.
//!
//! Phase 1 (world + giant crab, player/vehicle controls) replaces the trivial
//! body of [`Sim`] with the real game world. It MUST preserve the contract that
//! makes lockstep work, because peers run independent copies and only inputs cross
//! the wire (see [`crate::net::lockstep`]):
//!
//! - [`Sim::step`] is a pure function of `(prior state, inputs)` — identical
//!   inputs from an identical state always produce an identical next state, on any
//!   machine. No wall-clock, no thread-local/global RNG, no iteration over
//!   `HashMap`/`HashSet` (their order is randomized per process). Seed every random
//!   draw from [`Sim::rng`], iterate players in `PlayerId` order.
//! - [`Sim::state_hash`] folds the FULL observable state into one `u64`. It is the
//!   desync detector: two peers that diverge by even one bit hash differently next
//!   tick. When Phase 1 adds state, it must be hashed here, or a desync in that
//!   state goes undetected.

use std::collections::BTreeMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Identifies a player within the sim. Assigned from the connection set in a
/// deterministic order so every peer agrees which id is whom (see
/// [`crate::net::lockstep`]); the sim itself only relies on the ordering being
/// total and identical across peers, which `u8` gives for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlayerId(pub u8);

/// One player's input for a single tick.
///
/// Phase 0 is a 2-axis move stick; Phase 1 grows this (look, throttle, fire, …).
/// It is the unit that crosses the wire each tick, so keep it small and make it
/// encode/decode losslessly and identically on every peer — the wire bytes ARE the
/// shared truth (see [`Input::to_bytes`]). Fixed-point, not `f32`: an integer is
/// bit-identical across machines where a float round-trip need not be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Input {
    /// Move axes in fixed-point units of 1/[`Input::AXIS_SCALE`] per tick, each
    /// clamped to ±[`Input::AXIS_SCALE`] (i.e. ±1.0 in world units).
    pub move_x: i16,
    pub move_y: i16,
}

impl Input {
    /// Fixed-point denominator: `move_x / AXIS_SCALE` is the world-unit axis value.
    pub const AXIS_SCALE: i16 = 1000;
    /// Wire size of one encoded [`Input`].
    pub const WIRE_LEN: usize = 4;

    /// Build from analog axes in `[-1.0, 1.0]`, quantizing to the fixed-point grid.
    /// Quantizing at the input boundary (not in the sim) keeps the sim integer-only
    /// and means the value that crosses the wire is exactly the value applied.
    pub fn from_axes(x: f32, y: f32) -> Self {
        let q = |v: f32| (v.clamp(-1.0, 1.0) * Self::AXIS_SCALE as f32).round() as i16;
        Self {
            move_x: q(x),
            move_y: q(y),
        }
    }

    /// Encode for the wire: little-endian, fixed width, no allocation of intent
    /// beyond the bytes. Decoding the result yields exactly `self`.
    pub fn to_bytes(self) -> [u8; Self::WIRE_LEN] {
        let mut b = [0u8; Self::WIRE_LEN];
        b[0..2].copy_from_slice(&self.move_x.to_le_bytes());
        b[2..4].copy_from_slice(&self.move_y.to_le_bytes());
        b
    }

    /// Inverse of [`Input::to_bytes`].
    pub fn from_bytes(b: [u8; Self::WIRE_LEN]) -> Self {
        Self {
            move_x: i16::from_le_bytes([b[0], b[1]]),
            move_y: i16::from_le_bytes([b[2], b[3]]),
        }
    }
}

/// A player-controlled entity in the trivial Phase 0 world: a dot on a plane, moved
/// by its input. Position is fixed-point (units of 1/[`Input::AXIS_SCALE`]) for the
/// same cross-machine bit-identity reason as [`Input`] — Phase 1's real bodies will
/// likely carry `f32` transforms, at which point determinism rests on rapier's
/// enhanced-determinism rather than integer math (see rl#39).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dot {
    pub x: i64,
    pub y: i64,
}

/// The full deterministic game state. Everything that affects future ticks lives
/// here and nowhere else (no globals, no wall-clock reads), so a peer can be
/// reconstructed from it alone and [`Sim::state_hash`] can cover it completely.
#[derive(Debug, Clone)]
pub struct Sim {
    tick: u64,
    /// Player dots. `BTreeMap` (per the determinism contract above) so iteration is in
    /// `PlayerId` order on every peer.
    dots: BTreeMap<PlayerId, Dot>,
    /// The one sanctioned randomness source (see [`Sim::rng`]). Its stream position is
    /// hashed and reproduced across peers, so it is genuine sim state, not a scratch
    /// generator — never reseed it mid-sim. Phase 0's dots don't draw from it; Phase 1
    /// will.
    rng: ChaCha8Rng,
}

impl Sim {
    /// Create the initial world: one dot per player at a deterministic spawn, RNG
    /// seeded from `seed` (the shared match seed all peers agree on at session
    /// start). Spawns are laid out by id so the starting state is identical
    /// everywhere.
    pub fn new(seed: u64, players: &[PlayerId]) -> Self {
        let mut sorted: Vec<PlayerId> = players.to_vec();
        sorted.sort();
        sorted.dedup();
        let mut dots = BTreeMap::new();
        for (i, &id) in sorted.iter().enumerate() {
            // Deterministic ring of spawn points; spacing in fixed-point units.
            let i = i as i64;
            dots.insert(
                id,
                Dot {
                    x: i * 2 * Input::AXIS_SCALE as i64,
                    y: 0,
                },
            );
        }
        Self {
            tick: 0,
            dots,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Advance one tick by applying every player's input in `PlayerId` order.
    ///
    /// `inputs` must hold an entry for each player the sim knows about (the lockstep
    /// driver guarantees this by buffering a tick until all peers' inputs arrive).
    /// A missing input is treated as neutral — divergence here would be a logic bug,
    /// but defaulting keeps the step total rather than panicking on a dropped frame.
    pub fn step(&mut self, inputs: &BTreeMap<PlayerId, Input>) {
        // Iterate the dots (BTreeMap → PlayerId order), not the inputs, so the apply
        // order is the state's own order and independent of how `inputs` was built.
        for (id, dot) in self.dots.iter_mut() {
            let inp = inputs.get(id).copied().unwrap_or_default();
            dot.x += inp.move_x as i64;
            dot.y += inp.move_y as i64;
        }
        self.tick += 1;
    }

    /// Fold the entire observable state into one value. Equal hashes across peers ⇒
    /// in sync; any divergence flips it (see [`crate::net::lockstep`] desync check).
    ///
    /// Uses FNV-1a over a canonical byte serialization rather than `Hash`/`Hasher`:
    /// the algorithm is fixed in-code, so the value is stable across processes,
    /// builds, and machines — `DefaultHasher` is explicitly not (its seed/algorithm
    /// may change), which would make cross-peer comparison meaningless.
    pub fn state_hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(&self.tick.to_le_bytes());
        for (id, dot) in self.dots.iter() {
            h.write(&[id.0]);
            h.write(&dot.x.to_le_bytes());
            h.write(&dot.y.to_le_bytes());
        }
        // Hash the RNG stream position so a desync in random draws is caught even
        // before it manifests in a dot. A draw round-trips the generator state, which
        // changes the next block of bytes; folding the next u64 captures that.
        h.write(&self.rng.clone().r#gen::<u64>().to_le_bytes());
        h.finish()
    }

    /// The sanctioned randomness source for sim logic (Phase 1). Drawing from it
    /// advances shared state every peer tracks; never reach for `thread_rng`.
    pub fn rng(&mut self) -> &mut impl Rng {
        &mut self.rng
    }

    /// Current tick count (number of [`Sim::step`] calls applied).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Read a player's dot — for rendering/tests, never for sim logic decisions
    /// (those go through [`Sim::step`] so they stay deterministic).
    pub fn dot(&self, id: PlayerId) -> Option<Dot> {
        self.dots.get(&id).copied()
    }
}

/// FNV-1a 64-bit. A tiny, fixed, allocation-free hash whose result is identical on
/// every machine — the property `state_hash` needs and `std::hash::DefaultHasher`
/// does not guarantee.
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn players(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    #[test]
    fn input_bytes_roundtrip() {
        for inp in [
            Input::default(),
            Input::from_axes(1.0, -1.0),
            Input::from_axes(-0.5, 0.25),
            Input {
                move_x: i16::MIN,
                move_y: i16::MAX,
            },
        ] {
            assert_eq!(Input::from_bytes(inp.to_bytes()), inp);
        }
    }

    #[test]
    fn from_axes_clamps_and_quantizes() {
        assert_eq!(Input::from_axes(2.0, -2.0), Input { move_x: 1000, move_y: -1000 });
        assert_eq!(Input::from_axes(0.0, 0.0), Input::default());
    }

    #[test]
    fn spawn_is_deterministic_regardless_of_player_order() {
        let a = Sim::new(42, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        let b = Sim::new(42, &[PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn step_applies_input_to_owning_dot() {
        let mut sim = Sim::new(0, &players(2));
        let start = sim.dot(PlayerId(1)).unwrap();
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(1), Input::from_axes(1.0, 0.0));
        sim.step(&inputs);
        let moved = sim.dot(PlayerId(1)).unwrap();
        assert_eq!(moved.x - start.x, Input::AXIS_SCALE as i64);
        // The un-driven dot stayed put.
        assert_eq!(sim.dot(PlayerId(0)).unwrap(), Sim::new(0, &players(2)).dot(PlayerId(0)).unwrap());
    }

    #[test]
    fn hash_changes_when_state_changes() {
        let mut sim = Sim::new(0, &players(2));
        let h0 = sim.state_hash();
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), Input::from_axes(1.0, 1.0));
        sim.step(&inputs);
        assert_ne!(sim.state_hash(), h0);
    }
}
