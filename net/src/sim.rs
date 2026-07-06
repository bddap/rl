
use std::collections::BTreeMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crab_world::fnv::Fnv;

use crate::snapshot::CoreSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlayerId(pub u8);

pub mod buttons {
    pub const ACTION: u8 = 1 << 0;
    pub const RESTART: u8 = 1 << 1;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Input {
    pub move_strafe: i16,
    pub move_forward: i16,
    pub look_yaw: i16,
    pub buttons: u8,
}

impl Input {
    pub const AXIS_SCALE: i16 = 1000;
    pub const WIRE_LEN: usize = 7;

    pub fn new(strafe: f32, forward: f32, look_yaw: f32, buttons: u8) -> Self {
        let q = |v: f32| (v.clamp(-1.0, 1.0) * Self::AXIS_SCALE as f32).round() as i16;
        Self {
            move_strafe: q(strafe),
            move_forward: q(forward),
            look_yaw: q(look_yaw),
            buttons,
        }
    }

    pub fn from_axes(strafe: f32, forward: f32) -> Self {
        Self::new(strafe, forward, 0.0, 0)
    }

    pub fn pressed(self, bit: u8) -> bool {
        self.buttons & bit != 0
    }

    /// The input the server substitutes for a tick where this player's stream is STARVED
    /// (transit lag): keep the held-state move axes, zero the rest — `look_yaw` is a per-tick
    /// DELTA (re-applying it would keep the avatar turning) and a re-fired button tap would
    /// double a grab/restart. See [`crate::server`]'s hold semantics.
    pub fn hold(self) -> Self {
        Self {
            move_strafe: self.move_strafe,
            move_forward: self.move_forward,
            look_yaw: 0,
            buttons: 0,
        }
    }

    pub fn to_bytes(self) -> [u8; Self::WIRE_LEN] {
        let mut b = [0u8; Self::WIRE_LEN];
        b[0..2].copy_from_slice(&self.move_strafe.to_le_bytes());
        b[2..4].copy_from_slice(&self.move_forward.to_le_bytes());
        b[4..6].copy_from_slice(&self.look_yaw.to_le_bytes());
        b[6] = self.buttons;
        b
    }

    pub fn from_bytes(b: [u8; Self::WIRE_LEN]) -> Self {
        Self {
            move_strafe: i16::from_le_bytes([b[0], b[1]]),
            move_forward: i16::from_le_bytes([b[2], b[3]]),
            look_yaw: i16::from_le_bytes([b[4], b[5]]),
            buttons: b[6],
        }
    }
}

pub const TICK_HZ: u64 = 30;

pub const TICK_DT: f64 = 1.0 / TICK_HZ as f64;

pub const UNIT: i64 = 1000;

const PLAYER_SPEED: i64 = 166;

#[cfg(test)]
const CRAB_SPEED: i64 = 130;

#[cfg(test)]
pub(crate) fn drive_crab_toward_prey(sim: &mut Sim) {
    if sim.tick() < sim.round_start + STARTUP_GRACE_TICKS {
        return;
    }
    for idx in 0..sim.crabs().len() {
        let Some(target) = sim.nearest_living_player_pos(idx) else {
            continue;
        };
        let mut pos = sim.crabs()[idx].pos();
        let dx = target.x - pos.x;
        let dz = target.z - pos.z;
        let yaw = trig::atan2_turns(dx, dz);
        let dist = isqrt_i128(dist2_i128(dx, dz));
        if dist <= CRAB_SPEED as i128 {
            pos = target;
        } else if dist > 0 {
            pos.x += (dx as i128 * CRAB_SPEED as i128 / dist) as i64;
            pos.z += (dz as i128 * CRAB_SPEED as i128 / dist) as i64;
        }
        sim.set_external_crab_pose(idx, pos, yaw, 0);
    }
}

pub const CRAB_SCALE: i64 = 12;

pub const CRAB_GRAB_RADIUS: i64 = 3 * UNIT / 2;

const STARTUP_GRACE_TICKS: u64 = 30;

const MIN_CRAB_SPAWN_DISTANCE: i64 = 12 * UNIT;

pub const EXTRACT_RADIUS: i64 = 2 * UNIT;

pub(crate) const MAX_YAW_TURNS_PER_TICK: i32 = trig::TURN / 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerStatus {
    Alive,
    Downed,
    Extracted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pos {
    pub x: i64,
    pub z: i64,
}

impl Pos {
    /// This position in meters, as `(x, z)` — THE fixed-point→meters rule (`coord / UNIT`),
    /// stated once so the render/bridge/probe conversions can't drift. The f32 view is for
    /// presentation and diagnostics only; sim logic never reads it back.
    pub fn to_meters(self) -> (f32, f32) {
        (self.x as f32 / UNIT as f32, self.z as f32 / UNIT as f32)
    }

    /// Inverse of [`Pos::to_meters`]: meters onto the fixed-point grid (truncating cast,
    /// exactly the cast the external-crab bridge has always used).
    pub fn from_meters(x_m: f32, z_m: f32) -> Self {
        Pos {
            x: (x_m * UNIT as f32) as i64,
            z: (z_m * UNIT as f32) as i64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    pos: Pos,
    yaw: i32,
    status: PlayerStatus,
}

impl Player {
    pub(crate) fn from_parts(pos: Pos, yaw: i32, status: PlayerStatus) -> Self {
        Self { pos, yaw, status }
    }

    pub fn pos(self) -> Pos {
        self.pos
    }
    pub fn yaw(self) -> i32 {
        self.yaw
    }
    pub fn status(self) -> PlayerStatus {
        self.status
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crab {
    pos: Pos,
    yaw: i32,
}

impl Crab {
    pub(crate) fn from_parts(pos: Pos, yaw: i32) -> Self {
        Self { pos, yaw }
    }

    pub fn pos(self) -> Pos {
        self.pos
    }
    pub fn yaw(self) -> i32 {
        self.yaw
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractionPoint {
    pos: Pos,
}

impl ExtractionPoint {
    pub fn pos(self) -> Pos {
        self.pos
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    Extracted,
    Wiped,
}

#[derive(Debug, Clone)]
pub struct Sim {
    tick: u64,
    players: BTreeMap<PlayerId, Player>,
    crabs: Vec<Crab>,
    extraction: ExtractionPoint,
    outcome: Outcome,
    rng: ChaCha8Rng,
    restart_held: bool,
    round_start: u64,
    config: RoundConfig,
    external_crab_digests: Vec<u64>,
}

#[derive(Debug, Clone)]
struct RoundConfig {
    seed: u64,
    players: Vec<PlayerId>,
    crabs: usize,
}

impl Sim {
    pub fn new(seed: u64, players: &[PlayerId]) -> Self {
        let mut sorted: Vec<PlayerId> = players.to_vec();
        sorted.sort();
        sorted.dedup();
        let config = RoundConfig {
            seed,
            players: sorted,
            crabs: 1,
        };
        let (players, crabs, extraction) = Self::spawn_state(&config);
        let n_crabs = crabs.len();
        Self {
            tick: 0,
            players,
            crabs,
            extraction,
            outcome: Outcome::Ongoing,
            rng: ChaCha8Rng::seed_from_u64(seed),
            restart_held: false,
            round_start: 0,
            config,
            external_crab_digests: vec![0; n_crabs],
        }
    }

    pub fn configure_crabs(&mut self, crabs: usize) {
        assert!(crabs >= 1, "a round runs at least one giant crab (rl#114)");
        assert_eq!(
            self.tick, 0,
            "configure_crabs is round SETUP — the crab count is fixed once the round steps"
        );
        self.config.crabs = crabs;
        let (players, crabs, extraction) = Self::spawn_state(&self.config);
        self.players = players;
        self.crabs = crabs;
        self.extraction = extraction;
        self.external_crab_digests = vec![0; self.config.crabs];
    }

    pub fn set_external_crab_pose(&mut self, crab: usize, pos: Pos, yaw: i32, phys_digest: u64) {
        assert!(
            crab < self.crabs.len(),
            "external crab pose for crab {crab}, but the round has {} — the bridge and the sim \
             disagree on the binding count",
            self.crabs.len()
        );
        self.crabs[crab].pos = pos;
        self.crabs[crab].yaw = yaw;
        self.external_crab_digests[crab] = phys_digest;
    }

    fn reset(&mut self) {
        let (players, crabs, extraction) = Self::spawn_state(&self.config);
        self.round_start = self.tick;
        self.players = players;
        self.crabs = crabs;
        self.extraction = extraction;
        self.outcome = Outcome::Ongoing;
        self.rng = ChaCha8Rng::seed_from_u64(self.config.seed);
    }

    pub fn spawn_joining_player(&mut self, pid: PlayerId) {
        if self.players.contains_key(&pid) {
            return;
        }
        self.config.players.push(pid);
        self.config.players.sort();
        self.config.players.dedup();
        let idx = self
            .config
            .players
            .iter()
            .position(|p| *p == pid)
            .unwrap_or(0) as i64;
        let n = self.config.players.len() as i64;
        let x = (idx - n / 2) * 2 * UNIT;
        self.players.insert(
            pid,
            Player {
                pos: Pos { x, z: 0 },
                yaw: 0,
                status: PlayerStatus::Alive,
            },
        );
    }

    pub fn has_player(&self, pid: PlayerId) -> bool {
        self.players.contains_key(&pid)
    }

    pub fn despawn_departed_player(&mut self, pid: PlayerId) {
        self.players.remove(&pid);
        self.config.players.retain(|p| *p != pid);
    }

    fn spawn_state(cfg: &RoundConfig) -> (BTreeMap<PlayerId, Player>, Vec<Crab>, ExtractionPoint) {
        let mut map = BTreeMap::new();
        let n = cfg.players.len() as i64;
        for (i, &id) in cfg.players.iter().enumerate() {
            let x = (i as i64 - n / 2) * 2 * UNIT;
            map.insert(
                id,
                Player {
                    pos: Pos { x, z: 0 },
                    yaw: 0,
                    status: PlayerStatus::Alive,
                },
            );
        }
        let extraction = ExtractionPoint {
            pos: Pos { x: 0, z: 40 * UNIT },
        };
        let crabs = (0..cfg.crabs).map(|i| Self::spawn_crab(&map, i)).collect();
        (map, crabs, extraction)
    }

    fn spawn_crab(players: &BTreeMap<PlayerId, Player>, idx: usize) -> Crab {
        let mut pos = Pos {
            x: (6 + 8 * idx as i64) * UNIT,
            z: 20 * UNIT,
        };
        if let Some(nearest) = players
            .values()
            .min_by_key(|p| dist2_i128(pos.x - p.pos.x, pos.z - p.pos.z))
        {
            let dx = pos.x - nearest.pos.x;
            let dz = pos.z - nearest.pos.z;
            let d2 = dist2_i128(dx, dz);
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            if d2 < min * min {
                let dist = isqrt_i128(d2);
                let (ux, uz, len) = if dist > 0 {
                    (dx as i128, dz as i128, dist)
                } else {
                    (1, 0, 1)
                };
                pos.x = nearest.pos.x + (ux * min / len) as i64;
                pos.z = nearest.pos.z + (uz * min / len) as i64;
            }
        }
        Crab { pos, yaw: 0 }
    }

    fn participant_ids(&self) -> impl Iterator<Item = PlayerId> + '_ {
        self.players.keys().copied()
    }

    fn require_complete_inputs(&self, inputs: &BTreeMap<PlayerId, Input>) {
        for id in self.participant_ids() {
            assert!(
                inputs.contains_key(&id),
                "lockstep input incomplete: no input for {id:?} (have {:?}); defaulting \
                 to neutral would desync peers — refusing",
                inputs.keys().collect::<Vec<_>>(),
            );
        }
    }

    pub fn step(&mut self, inputs: &BTreeMap<PlayerId, Input>) -> bool {
        self.require_complete_inputs(inputs);
        self.tick += 1;

        let restart_now = inputs.values().any(|i| i.pressed(buttons::RESTART));
        let restart_edge = restart_now && !self.restart_held;
        self.restart_held = restart_now;
        if restart_edge {
            self.reset();
            return true;
        }

        if self.outcome != Outcome::Ongoing {
            return false;
        }

        for (id, p) in self.players.iter_mut() {
            if p.status != PlayerStatus::Alive {
                continue;
            }
            let inp = inputs[id];
            Self::advance_player(p, inp);
        }

        let armed = self.tick > self.round_start + STARTUP_GRACE_TICKS;


        if armed {
            for crab in &self.crabs {
                for p in self.players.values_mut() {
                    if p.status == PlayerStatus::Alive
                        && within(p.pos.x, p.pos.z, crab.pos.x, crab.pos.z, CRAB_GRAB_RADIUS)
                    {
                        p.status = PlayerStatus::Downed;
                    }
                }
            }
        }

        let ex = self.extraction.pos;
        for (id, p) in self.players.iter_mut() {
            if p.status == PlayerStatus::Alive
                && within(p.pos.x, p.pos.z, ex.x, ex.z, EXTRACT_RADIUS)
                && inputs[id].pressed(buttons::ACTION)
            {
                p.status = PlayerStatus::Extracted;
            }
        }

        self.outcome = self.settle_outcome();
        false
    }

    fn advance_player(p: &mut Player, inp: Input) {
        let dyaw =
            (inp.look_yaw as i64 * MAX_YAW_TURNS_PER_TICK as i64 / Input::AXIS_SCALE as i64) as i32;
        p.yaw = trig::wrap_turns(p.yaw + dyaw);

        let (sin, cos) = trig::sin_cos(p.yaw);
        let strafe = inp.move_strafe as i64;
        let forward = inp.move_forward as i64;
        let vx = sin * forward + cos * strafe;
        let vz = cos * forward - sin * strafe;
        let denom = Input::AXIS_SCALE as i64 * trig::ONE as i64;
        p.pos.x += vx * PLAYER_SPEED / denom;
        p.pos.z += vz * PLAYER_SPEED / denom;
    }

    pub(crate) fn predict_player(&mut self, id: PlayerId, inp: Input) {
        if self.outcome != Outcome::Ongoing {
            return;
        }
        if let Some(p) = self.players.get_mut(&id)
            && p.status == PlayerStatus::Alive
        {
            Self::advance_player(p, inp);
        }
    }

    pub fn nearest_living_player_pos(&self, crab: usize) -> Option<Pos> {
        self.nearest_living_player(self.crabs[crab].pos).map(|p| p.pos)
    }

    fn nearest_living_player(&self, c: Pos) -> Option<Player> {
        let mut best: Option<(i128, Player)> = None;
        for p in self.players.values() {
            if p.status != PlayerStatus::Alive {
                continue;
            }
            let d2 = dist2_i128(p.pos.x - c.x, p.pos.z - c.z);
            if best.is_none_or(|(bd, _)| d2 < bd) {
                best = Some((d2, *p));
            }
        }
        best.map(|(_, p)| p)
    }

    fn settle_outcome(&self) -> Outcome {
        if self
            .players
            .values()
            .any(|p| p.status == PlayerStatus::Extracted)
        {
            return Outcome::Extracted;
        }
        if !self.players.is_empty()
            && self
                .players
                .values()
                .all(|p| p.status == PlayerStatus::Downed)
        {
            return Outcome::Wiped;
        }
        Outcome::Ongoing
    }

    pub fn state_hash(&self) -> u64 {
        let Sim {
            tick,
            players,
            crabs,
            extraction,
            outcome,
            rng,
            restart_held,
            round_start,
            config: _,
            external_crab_digests,
        } = self;

        let mut h = Fnv::new();
        h.write(&tick.to_le_bytes());
        for (id, player) in players.iter() {
            let Player { pos, yaw, status } = player;
            h.write(&[id.0]);
            write_pos(&mut h, *pos);
            h.write(&yaw.to_le_bytes());
            h.write(&[status.tag()]);
        }
        h.write(&(crabs.len() as u32).to_le_bytes());
        for crab in crabs {
            let Crab { pos, yaw } = crab;
            write_pos(&mut h, *pos);
            h.write(&yaw.to_le_bytes());
        }
        let ExtractionPoint { pos } = extraction;
        write_pos(&mut h, *pos);
        h.write(&[outcome.tag()]);
        h.write(&[u8::from(*restart_held)]);
        h.write(&round_start.to_le_bytes());
        h.write(&rand::Rng::r#gen::<u64>(&mut rng.clone()).to_le_bytes());
        for digest in external_crab_digests {
            h.write(&digest.to_le_bytes());
        }
        h.finish()
    }

    pub fn rng(&mut self) -> &mut ChaCha8Rng {
        &mut self.rng
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn players(&self) -> impl Iterator<Item = (PlayerId, Player)> + '_ {
        self.players.iter().map(|(&id, &p)| (id, p))
    }

    pub fn player(&self, id: PlayerId) -> Option<Player> {
        self.players.get(&id).copied()
    }

    pub fn crabs(&self) -> &[Crab] {
        &self.crabs
    }

    pub fn extraction(&self) -> ExtractionPoint {
        self.extraction
    }

    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    pub fn core_snapshot(&self) -> CoreSnapshot {
        let Sim {
            tick,
            players,
            crabs,
            extraction: _,
            outcome,
            rng: _,
            restart_held: _,
            round_start: _,
            config,
            external_crab_digests: _,
        } = self;
        CoreSnapshot {
            tick: *tick,
            players: players.clone(),
            crabs: crabs.clone(),
            outcome: *outcome,
            roster: config.players.clone(),
            // Input watermarks are SERVER coordination metadata, not sim state — the sim holds
            // none. [`crate::server::Server::step_next`] stamps them; the client's `Lockstep`
            // stashes + re-stamps them for its mirror re-emit.
            input_next: std::collections::BTreeMap::new(),
        }
    }

    pub fn apply_core_snapshot(&mut self, snapshot: CoreSnapshot) {
        let CoreSnapshot {
            tick,
            players,
            crabs,
            outcome,
            roster,
            // Coordination metadata, not sim state — the client's `Lockstep` stashes it
            // (prediction-window prune + mirror re-emit) before handing the snapshot here.
            input_next: _,
        } = snapshot;
        self.tick = tick;
        self.players = players;
        self.external_crab_digests.resize(crabs.len(), 0);
        self.config.crabs = crabs.len();
        self.crabs = crabs;
        self.outcome = outcome;
        self.config.players = roster;
    }
}

impl PlayerStatus {
    pub(crate) fn tag(self) -> u8 {
        match self {
            PlayerStatus::Alive => 0,
            PlayerStatus::Downed => 1,
            PlayerStatus::Extracted => 2,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(PlayerStatus::Alive),
            1 => Some(PlayerStatus::Downed),
            2 => Some(PlayerStatus::Extracted),
            _ => None,
        }
    }
}

impl Outcome {
    pub(crate) fn tag(self) -> u8 {
        match self {
            Outcome::Ongoing => 0,
            Outcome::Extracted => 1,
            Outcome::Wiped => 2,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Outcome::Ongoing),
            1 => Some(Outcome::Extracted),
            2 => Some(Outcome::Wiped),
            _ => None,
        }
    }
}

fn dist2_i128(dx: i64, dz: i64) -> i128 {
    let dx = dx as i128;
    let dz = dz as i128;
    dx * dx + dz * dz
}

fn within(ax: i64, az: i64, bx: i64, bz: i64, r: i64) -> bool {
    dist2_i128(ax - bx, az - bz) <= (r as i128) * (r as i128)
}

fn isqrt_i128(n: i128) -> i128 {
    debug_assert!(n >= 0);
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

pub use super::cordic::{trig, trig_client};

fn write_pos(h: &mut Fnv, p: Pos) {
    let Pos { x, z } = p;
    h.write(&x.to_le_bytes());
    h.write(&z.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn players(n: u8) -> Vec<PlayerId> {
        (0..n).map(PlayerId).collect()
    }

    fn neutral_for(sim: &Sim) -> BTreeMap<PlayerId, Input> {
        sim.participant_ids()
            .map(|id| (id, Input::default()))
            .collect()
    }

    #[test]
    fn input_bytes_roundtrip() {
        for inp in [
            Input::default(),
            Input::from_axes(1.0, -1.0),
            Input::new(-0.5, 0.25, 1.0, buttons::ACTION),
            Input {
                move_strafe: i16::MIN,
                move_forward: i16::MAX,
                look_yaw: -123,
                buttons: 0xFF,
            },
        ] {
            assert_eq!(Input::from_bytes(inp.to_bytes()), inp);
        }
    }

    #[test]
    fn mixed_foot_and_neutral_pilot_steps_deterministically() {
        let run = || {
            let mut sim = Sim::new(7, &players(2));
            let p0_start = sim.player(PlayerId(0)).unwrap().pos();
            let p1_start = sim.player(PlayerId(1)).unwrap().pos();
            for _ in 0..20 {
                let mut inputs = neutral_for(&sim);
                inputs.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
                sim.step(&inputs);
            }
            let p0_end = sim.player(PlayerId(0)).unwrap().pos();
            let p1_end = sim.player(PlayerId(1)).unwrap().pos();
            (sim.state_hash(), p0_start, p0_end, p1_start, p1_end)
        };
        let (h1, ..) = run();
        let (h2, p0_start, p0_end, p1_start, p1_end) = run();
        assert_eq!(
            h1, h2,
            "the same mixed foot+pilot inputs must reproduce the state hash"
        );
        assert_eq!(
            p1_start, p1_end,
            "a piloting (neutral-input) player's foot avatar stays put"
        );
        assert_ne!(
            p0_start, p0_end,
            "the walking player actually moved (not a no-op step)"
        );
    }

    #[test]
    fn wire_len_matches_encoding() {
        assert_eq!(Input::default().to_bytes().len(), Input::WIRE_LEN);
    }

    #[test]
    fn from_axes_clamps_and_quantizes_and_is_neutral_look() {
        let i = Input::from_axes(2.0, -2.0);
        assert_eq!((i.move_strafe, i.move_forward), (1000, -1000));
        assert_eq!((i.look_yaw, i.buttons), (0, 0));
        assert_eq!(Input::from_axes(0.0, 0.0), Input::default());
    }

    #[test]
    fn spawn_is_deterministic_regardless_of_player_order() {
        let a = Sim::new(42, &[PlayerId(2), PlayerId(0), PlayerId(1)]);
        let b = Sim::new(42, &[PlayerId(0), PlayerId(1), PlayerId(2)]);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn forward_input_moves_along_facing() {
        let mut sim = Sim::new(0, &players(1));
        let p0 = sim.player(PlayerId(0)).unwrap().pos();
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        sim.step(&inputs);
        let p1 = sim.player(PlayerId(0)).unwrap().pos();
        assert_eq!(p1.x, p0.x, "no X drift facing +Z");
        let dz = p1.z - p0.z;
        assert!(
            (dz - PLAYER_SPEED).abs() <= 1,
            "forward step ≈ PLAYER_SPEED, got {dz}"
        );
    }

    #[test]
    fn strafe_input_moves_sideways_along_x() {
        let mut sim = Sim::new(0, &players(1));
        let p0 = sim.player(PlayerId(0)).unwrap().pos();
        let mut right = BTreeMap::new();
        right.insert(PlayerId(0), Input::new(1.0, 0.0, 0.0, 0));
        sim.step(&right);
        let p1 = sim.player(PlayerId(0)).unwrap().pos();
        assert_eq!(p1.z, p0.z, "no Z drift strafing at yaw 0");
        let dx = p1.x - p0.x;
        assert!(
            (dx - PLAYER_SPEED).abs() <= 1,
            "strafe-right step ≈ +PLAYER_SPEED in X, got {dx}"
        );
        let mut sim = Sim::new(0, &players(1));
        let mut left = BTreeMap::new();
        left.insert(PlayerId(0), Input::new(-1.0, 0.0, 0.0, 0));
        sim.step(&left);
        let dx_left = sim.player(PlayerId(0)).unwrap().pos().x - p0.x;
        assert_eq!(dx_left, -dx, "strafe-left mirrors strafe-right exactly");
    }

    #[test]
    fn predict_player_matches_step_for_the_local_avatar() {
        let inp = Input::new(0.6, -0.3, 0.4, 0);
        let mut stepped = Sim::new(7, &players(1));
        let mut inputs = BTreeMap::new();
        inputs.insert(PlayerId(0), inp);
        stepped.step(&inputs);

        let mut predicted = Sim::new(7, &players(1));
        predicted.predict_player(PlayerId(0), inp);

        let sp = stepped.player(PlayerId(0)).unwrap();
        let pp = predicted.player(PlayerId(0)).unwrap();
        assert_eq!(
            (pp.pos(), pp.yaw()),
            (sp.pos(), sp.yaw()),
            "predicted local avatar must equal the stepped avatar"
        );

        let before = predicted.state_hash();
        predicted.predict_player(PlayerId(9), inp);
        assert_eq!(
            predicted.state_hash(),
            before,
            "predicting an absent player must change nothing"
        );
    }

    #[test]
    fn look_then_move_turns_the_heading() {
        let mut sim = Sim::new(0, &players(1));
        let ticks = ((trig::TURN / 4) / MAX_YAW_TURNS_PER_TICK) as usize;
        for _ in 0..ticks {
            let mut inp = BTreeMap::new();
            inp.insert(PlayerId(0), Input::new(0.0, 0.0, 1.0, 0));
            sim.step(&inp);
        }
        let before = sim.player(PlayerId(0)).unwrap().pos();
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        sim.step(&fwd);
        let after = sim.player(PlayerId(0)).unwrap().pos();
        let dx = after.x - before.x;
        let dz = after.z - before.z;
        assert!(
            dx.abs() > dz.abs(),
            "after a ~quarter turn, forward should move mostly in X (dx={dx}, dz={dz})"
        );
    }

    #[test]
    fn crab_pursues_and_grabs_a_lone_player() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        let crab_armed = sim.crabs()[0].pos();
        let prey = sim.player(PlayerId(0)).unwrap().pos();
        let d_start = dist2(crab_armed, prey);
        drive_crab_toward_prey(&mut sim);
        sim.step(&neutral);
        let d_next = dist2(sim.crabs()[0].pos(), sim.player(PlayerId(0)).unwrap().pos());
        assert!(d_next < d_start, "crab must close distance once driven");
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_eq!(
            sim.outcome(),
            Outcome::Wiped,
            "standing-still player gets caught"
        );
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed
        );
    }

    #[test]
    fn external_crab_pose_seeds_and_digest_is_hashed() {
        let pos = Pos {
            x: 7 * UNIT,
            z: -3 * UNIT,
        };
        let yaw = 123;

        let mut sim = Sim::new(0, &players(1));
        let digest = 0xFEED_FACE_DEAD_BEEF;
        sim.set_external_crab_pose(0, pos, yaw, digest);
        assert_eq!(sim.crabs()[0].pos(), pos, "must seed the pose");
        assert_eq!(sim.crabs()[0].yaw(), yaw, "must seed the yaw");
        let h_seed = sim.state_hash();
        sim.set_external_crab_pose(0, pos, yaw, digest ^ 1);
        assert_ne!(
            h_seed,
            sim.state_hash(),
            "the external crab digest must be folded into the state hash"
        );
    }

    #[test]
    fn reaching_extraction_with_action_wins() {
        let mut sim = Sim::new(0, &players(1));
        let ex = sim.extraction().pos();
        let route = [
            Pos {
                x: -30 * UNIT,
                z: 0,
            },
            Pos {
                x: -30 * UNIT,
                z: ex.z,
            },
            ex,
        ];
        let mut wp = 0usize;
        let mut won = false;
        for _ in 0..4000 {
            let p = sim.player(PlayerId(0)).unwrap();
            if p.status() != PlayerStatus::Alive {
                break;
            }
            let pp = p.pos();
            if wp < route.len() - 1 && within(pp.x, pp.z, route[wp].x, route[wp].z, UNIT) {
                wp += 1;
            }
            let target = route[wp];
            let want_yaw = trig::atan2_turns(target.x - pp.x, target.z - pp.z);
            let delta = trig::wrap_turns(want_yaw - p.yaw());
            let look = if delta == 0 {
                0.0
            } else if delta < trig::TURN / 2 {
                1.0
            } else {
                -1.0
            };
            let mut inp = BTreeMap::new();
            inp.insert(PlayerId(0), Input::new(0.0, 1.0, look, buttons::ACTION));
            drive_crab_toward_prey(&mut sim);
            sim.step(&inp);
            if sim.outcome() == Outcome::Extracted {
                won = true;
                break;
            }
        }
        assert!(
            won,
            "a player who dodges the crab, reaches the point, and holds ACTION should extract"
        );
    }

    #[test]
    fn outcome_is_frozen_once_decided() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_ne!(
            sim.outcome(),
            Outcome::Ongoing,
            "round should have resolved"
        );
        let snapshot = |s: &Sim| {
            (
                s.players().collect::<Vec<_>>(),
                s.crabs().to_vec(),
                s.extraction(),
                s.outcome(),
            )
        };
        let frozen = snapshot(&sim);
        for _ in 0..10 {
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_eq!(
            snapshot(&sim),
            frozen,
            "a decided round must freeze the world"
        );
    }

    #[test]
    fn hash_changes_when_state_changes() {
        let mut sim = Sim::new(0, &players(2));
        let h0 = sim.state_hash();
        let mut inputs = neutral_for(&sim);
        inputs.insert(PlayerId(0), Input::from_axes(1.0, 1.0));
        sim.step(&inputs);
        assert_ne!(sim.state_hash(), h0);
    }

    #[test]
    #[should_panic(expected = "lockstep input incomplete")]
    fn missing_lockstep_input_panics_not_defaults_to_neutral() {
        let mut sim = Sim::new(0, &players(2));
        let mut partial = BTreeMap::new();
        partial.insert(PlayerId(0), Input::from_axes(0.0, 1.0));
        sim.step(&partial);
    }


    #[test]
    fn trig_table_hits_cardinal_points() {
        use trig::{ONE, TURN, cos, sin};
        assert_eq!(sin(0), 0);
        assert_eq!(sin(TURN / 2), 0);
        assert!((sin(TURN / 4) - ONE).abs() <= 1);
        assert!((sin(3 * TURN / 4) + ONE).abs() <= 1);
        assert!((cos(0) - ONE).abs() <= 1);
        assert!(cos(TURN / 4).abs() <= 1);
        assert!((cos(TURN / 2) + ONE).abs() <= 1);
    }

    #[test]
    fn trig_pythagorean_identity_holds() {
        use trig::{ONE, cos, sin};
        for k in 0..64 {
            let a = k * (trig::TURN / 64);
            let s = sin(a) as i64;
            let c = cos(a) as i64;
            let r2 = s * s + c * c;
            let one2 = (ONE as i64) * (ONE as i64);
            let err = (r2 - one2).abs();
            assert!(err <= one2 / 500, "sin²+cos² off at {a}: {r2} vs {one2}");
        }
    }

    #[test]
    fn atan2_recovers_cardinal_and_diagonal_directions() {
        use trig::{TURN, atan2_turns};
        let near = |a: i32, b: i32| trig::wrap_turns(a - b).min(trig::wrap_turns(b - a));
        assert!(near(atan2_turns(0, 1), 0) <= 2, "+Z is yaw 0");
        assert!(near(atan2_turns(1, 0), TURN / 4) <= 2, "+X is quarter turn");
        assert!(near(atan2_turns(0, -1), TURN / 2) <= 2, "−Z is half turn");
        assert!(
            near(atan2_turns(-1, 0), 3 * TURN / 4) <= 2,
            "−X is three-quarter turn"
        );
        assert!(
            near(atan2_turns(1, 1), TURN / 8) <= 2,
            "+X+Z diagonal is eighth turn"
        );
    }

    #[test]
    fn isqrt_matches_floor_sqrt() {
        for n in [
            0i128,
            1,
            2,
            3,
            4,
            8,
            15,
            16,
            17,
            100,
            1_000_000,
            1_000_003,
            i64::MAX as i128,
            (i64::MAX as i128) * (i64::MAX as i128),
        ] {
            let r = isqrt_i128(n);
            assert!(
                r * r <= n && (r + 1) * (r + 1) > n,
                "isqrt({n})={r} not floor sqrt"
            );
        }
    }

    #[test]
    fn cordic_table_matches_f64_reference_exactly() {
        use trig::{ONE, TURN, cos, sin};
        for a in 0..TURN {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).sin() * ONE as f64).round()
                as i32;
            assert_eq!(sin(a), want, "sin table off at {a}");
        }
        for a in (0..TURN).step_by(257) {
            let want = ((a as f64 / TURN as f64 * std::f64::consts::TAU).cos() * ONE as f64).round()
                as i32;
            assert_eq!(cos(a), want, "cos off at {a}");
        }
    }

    #[test]
    fn state_hash_is_sensitive_to_every_hashed_field() {
        let base = Sim::new(7, &players(2));
        let h0 = base.state_hash();
        let hash_after = |mutate: &dyn Fn(&mut Sim)| {
            let mut s = base.clone();
            mutate(&mut s);
            s.state_hash()
        };
        let foot = PlayerId(0);

        assert_ne!(hash_after(&|s| s.tick += 1), h0, "tick must be hashed");

        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().pos.x += 1),
            h0,
            "player pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().pos.z += 1),
            h0,
            "player pos.z must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().yaw += 1),
            h0,
            "player yaw must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.players.get_mut(&foot).unwrap().status = PlayerStatus::Downed),
            h0,
            "player status must be hashed"
        );

        assert_ne!(
            hash_after(&|s| s.crabs[0].pos.x += 1),
            h0,
            "crab pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.crabs[0].pos.z += 1),
            h0,
            "crab pos.z must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.crabs[0].yaw += 1),
            h0,
            "crab yaw must be hashed"
        );

        assert_ne!(
            hash_after(&|s| s.extraction.pos.x += 1),
            h0,
            "extraction pos.x must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.extraction.pos.z += 1),
            h0,
            "extraction pos.z must be hashed"
        );

        assert_ne!(
            hash_after(&|s| s.outcome = Outcome::Wiped),
            h0,
            "outcome must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.restart_held = !s.restart_held),
            h0,
            "restart_held must be hashed"
        );
        assert_ne!(
            hash_after(&|s| s.round_start += 1),
            h0,
            "round_start must be hashed (it gates the post-restart grace)"
        );
        assert_ne!(
            hash_after(&|s| {
                let _: u64 = rand::Rng::r#gen(s.rng());
            }),
            h0,
            "rng stream position must be hashed"
        );

        assert_eq!(
            hash_after(&|s| s.config.seed ^= 0xdead_beef),
            h0,
            "config is deliberately not hashed (see Sim::config)"
        );
        assert_ne!(
            hash_after(&|s| s.external_crab_digests[0] ^= 0xdead_beef),
            h0,
            "external_crab_digests must be hashed (always folded since rl#114)"
        );
    }

    #[test]
    fn core_snapshot_roundtrip_reproduces_authoritative_state() {
        let mut original = Sim::new(7, &players(3));
        for _ in 0..5 {
            original.set_external_crab_pose(0, Pos { x: 4200, z: -1300 }, 77, 0);
            let mut inputs = neutral_for(&original);
            *inputs.get_mut(&PlayerId(0)).unwrap() = Input::from_axes(0.3, 1.0);
            original.step(&inputs);
        }
        original.players.get_mut(&PlayerId(1)).unwrap().status = PlayerStatus::Downed;

        let restored = CoreSnapshot::from_bytes(&original.core_snapshot().to_bytes())
            .expect("a freshly-built snapshot must round-trip through bytes");

        let mut target = original.clone();
        target.tick = 999;
        target.players.get_mut(&PlayerId(0)).unwrap().pos.x += 12_345;
        target.players.get_mut(&PlayerId(2)).unwrap().yaw += 3;
        target.players.get_mut(&PlayerId(1)).unwrap().status = PlayerStatus::Extracted;
        target.crabs[0].pos = Pos { x: -1, z: -2 };
        target.crabs[0].yaw = 9;
        target.outcome = Outcome::Wiped;
        target.config.players = vec![PlayerId(0)];
        assert_ne!(target.state_hash(), original.state_hash());

        target.apply_core_snapshot(restored);
        assert_eq!(
            target.state_hash(),
            original.state_hash(),
            "applying the round-tripped snapshot reproduces every hashed carried field"
        );
        assert_eq!(
            target.config.players, original.config.players,
            "the snapshot must carry the roster too"
        );
    }


    #[test]
    fn crab_holds_and_cannot_grab_during_startup_grace() {
        let mut sim = Sim::new(0, &players(1));
        let crab0 = sim.crabs()[0].pos();
        sim.players.get_mut(&PlayerId(0)).unwrap().pos = crab0;
        let neutral = neutral_for(&sim);
        for _ in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral);
            assert_eq!(sim.crabs()[0].pos(), crab0, "crab holds its spawn during grace");
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no grab during the startup grace"
            );
            assert_eq!(sim.outcome(), Outcome::Ongoing);
        }
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "crab arms and grabs once the grace ends"
        );
    }

    #[test]
    fn crab_spawns_clear_of_every_player() {
        for n in 1..=8u8 {
            let sim = Sim::new(0, &players(n));
            let crab = sim.crabs()[0].pos();
            let nearest = sim
                .players()
                .map(|(_, p)| dist2_i128(crab.x - p.pos().x, crab.z - p.pos().z))
                .min()
                .unwrap();
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            assert!(
                nearest >= min * min,
                "n={n}: nearest player {} closer than MIN_CRAB_SPAWN_DISTANCE",
                isqrt_i128(nearest)
            );
        }
    }

    #[test]
    fn restart_resets_the_round_to_spawn() {
        let mut sim = Sim::new(0xBEEF, &players(2));
        let fresh = Sim::new(0xBEEF, &players(2));
        let mut fwd = BTreeMap::new();
        fwd.insert(PlayerId(0), Input::new(0.3, 1.0, 0.5, 0));
        fwd.insert(PlayerId(1), Input::new(-0.2, 1.0, 0.0, 0));
        for _ in 0..50 {
            sim.step(&fwd);
        }
        let world = |s: &Sim| {
            (
                s.players().collect::<Vec<_>>(),
                s.crabs().to_vec(),
                s.extraction(),
                s.outcome(),
            )
        };
        assert_ne!(
            world(&sim),
            world(&fresh),
            "the round should have diverged from spawn before restart"
        );
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        restart.insert(PlayerId(1), Input::default());
        let edge = sim.step(&restart);
        assert!(edge, "the press reports the restart edge");
        assert_eq!(sim.tick(), 51, "the tick stays monotone across a restart");
        assert_eq!(sim.outcome(), Outcome::Ongoing);
        assert_eq!(
            world(&sim),
            world(&fresh),
            "a restarted round's world matches a fresh one"
        );
    }

    #[test]
    fn restart_works_after_the_round_is_decided() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..2000 {
            if sim.outcome() != Outcome::Ongoing {
                break;
            }
            drive_crab_toward_prey(&mut sim);
            sim.step(&neutral);
        }
        assert_eq!(sim.outcome(), Outcome::Wiped, "round should have ended");
        let tick_at_loss = sim.tick();
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        sim.step(&restart);
        assert_eq!(sim.outcome(), Outcome::Ongoing, "restart revives the round");
        assert_eq!(sim.tick(), tick_at_loss + 1, "the tick keeps counting");
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Alive,
            "the player is alive again after a post-loss restart"
        );
    }

    #[test]
    fn restart_is_edge_triggered_not_level() {
        let mut sim = Sim::new(0, &players(1));
        let mut held = BTreeMap::new();
        held.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(sim.step(&held), "first press restarts");
        assert!(!sim.step(&held), "a held key doesn't re-restart");
        assert!(!sim.step(&held), "still held, still no re-restart");
        assert_eq!(sim.tick(), 3, "every tick counted, restart included");
        assert!(!sim.step(&neutral_for(&sim)), "release: no restart");
        assert!(sim.step(&held), "a new press after release restarts again");
    }

    #[test]
    fn restart_keeps_two_peers_in_lockstep() {
        let mut a = Sim::new(0x5151, &players(2));
        let mut b = Sim::new(0x5151, &players(2));
        let mut restarts = 0u32;
        for t in 0..120u64 {
            let mut inputs = BTreeMap::new();
            let restart_bit = if t == 40 { buttons::RESTART } else { 0 };
            inputs.insert(PlayerId(0), Input::new(0.4, 1.0, 0.2, 0));
            inputs.insert(PlayerId(1), Input::new(-0.3, 1.0, -0.1, restart_bit));
            restarts += u32::from(a.step(&inputs));
            b.step(&inputs);
            assert_eq!(
                a.state_hash(),
                b.state_hash(),
                "peers must stay bit-identical across a restart (tick {t})"
            );
        }
        assert_eq!(restarts, 1, "the mid-run restart fired exactly once");
        assert_eq!(a.tick(), 120, "a restart never rewinds the tick");
    }

    #[test]
    fn restart_grants_a_fresh_startup_grace() {
        let mut sim = Sim::new(0, &players(1));
        let neutral = neutral_for(&sim);
        for _ in 0..(3 * STARTUP_GRACE_TICKS) {
            sim.step(&neutral);
        }
        let mut restart = BTreeMap::new();
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(sim.step(&restart), "the restart fires");
        let crab0 = sim.crabs()[0].pos();
        sim.players.get_mut(&PlayerId(0)).unwrap().pos = crab0;
        for i in 0..STARTUP_GRACE_TICKS {
            sim.step(&neutral);
            assert_eq!(
                sim.player(PlayerId(0)).unwrap().status(),
                PlayerStatus::Alive,
                "no grab during the post-restart grace (tick {i} into the round)"
            );
        }
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(0)).unwrap().status(),
            PlayerStatus::Downed,
            "the crab arms once the post-restart grace ends"
        );
    }

    #[test]
    fn multi_crab_round_spawns_hashes_and_grabs_per_crab() {
        let mut sim = Sim::new(0, &players(2));
        sim.configure_crabs(2);
        assert_eq!(sim.crabs().len(), 2);
        assert_ne!(
            sim.crabs()[0].pos(),
            sim.crabs()[1].pos(),
            "crabs spawn staggered, not stacked"
        );
        for (i, crab) in sim.crabs().iter().enumerate() {
            let nearest = sim
                .players()
                .map(|(_, p)| dist2(crab.pos(), p.pos()))
                .min()
                .unwrap();
            let min = MIN_CRAB_SPAWN_DISTANCE as i128;
            assert!(nearest >= min * min, "crab {i} spawns clear of every player");
        }

        assert_ne!(sim.state_hash(), Sim::new(0, &players(2)).state_hash());
        let h0 = sim.state_hash();
        let c1 = sim.crabs()[1];
        sim.set_external_crab_pose(1, c1.pos(), c1.yaw(), 0xBEEF);
        assert_ne!(sim.state_hash(), h0, "crab 1's digest folds into the hash");

        let neutral = neutral_for(&sim);
        for _ in 0..=STARTUP_GRACE_TICKS {
            sim.step(&neutral);
        }
        let prey = sim.player(PlayerId(1)).unwrap().pos();
        sim.set_external_crab_pose(1, prey, 0, 0);
        sim.step(&neutral);
        assert_eq!(
            sim.player(PlayerId(1)).unwrap().status(),
            PlayerStatus::Downed,
            "the second crab's reach downs a player"
        );

        let snap = sim.core_snapshot();
        assert_eq!(snap.crabs.len(), 2);
        let restored = CoreSnapshot::from_bytes(&snap.to_bytes()).unwrap();
        let mut client = Sim::new(0, &players(2));
        client.apply_core_snapshot(restored);
        assert_eq!(
            client.crabs().len(),
            2,
            "an adopting client takes the host's crab count"
        );

        let mut restart = neutral_for(&sim);
        restart.insert(PlayerId(0), Input::new(0.0, 0.0, 0.0, buttons::RESTART));
        assert!(sim.step(&restart));
        assert_eq!(sim.crabs().len(), 2, "restart rebuilds the configured count");
    }

    #[test]
    #[should_panic(expected = "disagree on the binding count")]
    fn pose_for_an_unconfigured_crab_panics() {
        let mut sim = Sim::new(0, &players(1));
        sim.set_external_crab_pose(1, Pos::default(), 0, 0);
    }

    fn dist2(a: Pos, b: Pos) -> i128 {
        dist2_i128(a.x - b.x, a.z - b.z)
    }
}
