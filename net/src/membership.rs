use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crab_world::fnv::Fnv;

const MEMBER_TIMEOUT: Duration = Duration::from_secs(3);

pub const BEAT_EVERY: Duration = Duration::from_millis(250);

const STABLE_FOR: Duration = Duration::from_millis(1500);

pub const JOIN_WINDOW: Duration = Duration::from_secs(20);

const MAX_MEMBERS: usize = u8::MAX as usize + 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Beat {
    pub members: Vec<EndpointId>,
    /// The host's synchronized-start command: `true` once the host has clicked Start,
    /// commanding the round to begin on the roster carried in THIS same beat. A joiner
    /// closes the barrier on a direct `start` beat whose roster hash equals its own —
    /// so host and joiners freeze the byte-identical set the GO names. Always `false`
    /// outside the host-triggered menu path.
    pub start: bool,
    /// The advertiser's world identity — see [`crate::SyncStamp`]. ONE shape shared
    /// with [`crate::server::JoinRequest`] and [`PeerView`], so a new identity axis
    /// cannot be added to one carrier and silently missed in another.
    pub stamp: crate::SyncStamp,
}

impl Beat {
    pub fn roster_hash(&self) -> u64 {
        roster_hash(&self.members)
    }
}

pub fn roster_hash(ids: &[EndpointId]) -> u64 {
    let mut sorted: Vec<&EndpointId> = ids.iter().collect();
    sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    sorted.dedup();
    let mut h = Fnv::new();
    for id in sorted {
        h.write(id.as_bytes());
    }
    h.finish()
}

pub fn host_of(roster: &[EndpointId]) -> EndpointId {
    *roster
        .iter()
        .min_by(|a, b| a.as_bytes().cmp(b.as_bytes()))
        .expect("a roster always contains at least us")
}

pub struct Membership {
    me: EndpointId,
    expect: usize,
    peers: BTreeMap<EndpointId, PeerView>,
    tombstones: BTreeMap<EndpointId, Instant>,
    agreed_since: Option<(Instant, u64)>,
    started: Instant,
    /// How the barrier closes — see [`LobbyMode`]. The roster a peer freezes is the same
    /// `live_set` in every mode (only the close *moment* differs), so determinism is
    /// untouched.
    lobby: LobbyMode,
    host_go_on_my_roster: bool,
    local: crate::SyncStamp,
}

/// How a [`Membership`] barrier decides to close. A sum type so the "only a host
/// commands the start" rule is unrepresentable to violate (a [`LobbyMode::Joiner`] has
/// no `starting` to set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LobbyMode {
    Off,
    Host { started: bool },
    Joiner,
}

#[derive(Debug, Clone, Copy)]
struct PeerView {
    last_direct: Instant,
    advertised: Option<u64>,
    started: bool,
    /// `None` until a direct beat arrives — a relay-only peer stays unverified on
    /// EVERY identity axis at once (the axes are only ever learned together).
    stamp: Option<crate::SyncStamp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Forming { live: usize },
    Agreed { roster: Vec<EndpointId> },
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Host,
    Joiner,
}

impl Membership {
    pub fn new(me: EndpointId, expect: usize, now: Instant) -> Self {
        Self {
            me,
            expect: expect.max(1),
            peers: BTreeMap::new(),
            tombstones: BTreeMap::new(),
            agreed_since: None,
            started: now,
            lobby: LobbyMode::Off,
            host_go_on_my_roster: false,
            local: crate::SyncStamp::ZERO,
        }
    }

    pub fn with_stamp(mut self, stamp: crate::SyncStamp) -> Self {
        self.local = stamp;
        self
    }

    /// Begin a host-triggered (interactive lobby) formation — the boot menu's networked
    /// Host/Join. Same agreement core as [`Membership::new`], but the close trigger is a
    /// host's explicit GO ([`Membership::set_starting`]) instead of the [`STABLE_FOR`]
    /// timer; only a [`Role::Host`] can command it. `expect` is still the participant
    /// floor. The frozen roster is identical to the timer path — only the close *moment*
    /// changes.
    pub fn host_triggered(role: Role, me: EndpointId, expect: usize, now: Instant) -> Self {
        let lobby = match role {
            Role::Host => LobbyMode::Host { started: false },
            Role::Joiner => LobbyMode::Joiner,
        };
        Self {
            lobby,
            ..Self::new(me, expect, now)
        }
    }

    pub fn set_starting(&mut self) {
        if let LobbyMode::Host { started } = &mut self.lobby {
            *started = true;
        }
    }

    fn has_room_for_one_more(&self) -> bool {
        self.peers.len() + 1 < MAX_MEMBERS
    }

    pub fn on_beat(&mut self, from: EndpointId, beat: &Beat, now: Instant) {
        if from != self.me {
            self.tombstones.remove(&from);
            if self.peers.contains_key(&from) || self.has_room_for_one_more() {
                let view = self.peers.entry(from).or_insert(PeerView {
                    last_direct: now,
                    advertised: None,
                    started: false,
                    stamp: None,
                });
                view.last_direct = now;
                view.advertised = Some(beat.roster_hash());
                view.started = beat.start;
                view.stamp = Some(beat.stamp);
            }
        }
        // Transitive admission (see the method docs): seed only the id's existence
        // (`advertised = None`), capped at MAX_MEMBERS so a flooded beat can't inflate
        // our roster.
        for &id in &beat.members {
            if id != self.me
                && !self.peers.contains_key(&id)
                && !self.tombstones.contains_key(&id)
                && self.has_room_for_one_more()
            {
                self.peers.insert(
                    id,
                    PeerView {
                        last_direct: now,
                        advertised: None,
                        started: false,
                        stamp: None,
                    },
                );
            }
        }
    }

    pub fn live_set(&self) -> Vec<EndpointId> {
        let mut ids: Vec<EndpointId> = self.peers.keys().copied().collect();
        ids.push(self.me);
        ids.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        ids.dedup();
        ids
    }

    pub fn beat(&self) -> Beat {
        Beat {
            members: self.live_set(),
            start: matches!(self.lobby, LobbyMode::Host { started: true }),
            stamp: self.local,
        }
    }

    pub fn sync_verdict(&self) -> crate::SyncVerdict {
        let host = host_of(&self.live_set());
        let host_crabs = if host == self.me {
            Some(self.local.crab_count)
        } else {
            self.peers
                .get(&host)
                .and_then(|v| v.stamp)
                .map(|s| s.crab_count)
        };
        crate::SyncVerdict {
            body: self.local.body_digest != 0
                && self.peers.values().all(|v| {
                    v.stamp
                        .is_some_and(|s| s.body_digest == self.local.body_digest)
                }),
            crabs: host_crabs.is_some_and(|h| {
                h >= 1 && (self.local.crab_count == 0 || self.local.crab_count == h)
            }),
            plant: self.local.plant_digest != 0
                && self.peers.values().all(|v| {
                    v.stamp
                        .is_some_and(|s| s.plant_digest == self.local.plant_digest)
                }),
        }
    }

    pub fn poll(&mut self, now: Instant) -> Status {
        let expired: Vec<EndpointId> = self
            .peers
            .iter()
            .filter(|(_, v)| now.duration_since(v.last_direct) >= MEMBER_TIMEOUT)
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            self.peers.remove(&id);
            self.tombstones.insert(id, now);
        }
        self.tombstones
            .retain(|_, &mut t| now.duration_since(t) < MEMBER_TIMEOUT);

        let live = self.live_set();
        let my_hash = roster_hash(&live);
        let enough = live.len() >= self.expect;
        let unanimous = self.peers.values().all(|v| v.advertised == Some(my_hash));
        let agreed_now = enough && unanimous;

        // Recomputed, never latched — see `host_go_on_my_roster` for why.
        self.host_go_on_my_roster = self
            .peers
            .values()
            .any(|v| v.started && v.advertised == Some(my_hash));

        // Maintain the continuous-agreement timer — see `agreed_since` for why it is
        // keyed on the hash, not just a bool.
        self.agreed_since = match (agreed_now, self.agreed_since) {
            (true, Some((t, h))) if h == my_hash => Some((t, h)),
            (true, _) => Some((now, my_hash)),
            (false, _) => None,
        };

        let held = self
            .agreed_since
            .is_some_and(|(since, _)| now.duration_since(since) >= STABLE_FOR);
        let close = held
            && match self.lobby {
                LobbyMode::Host { started } => started,
                LobbyMode::Joiner => self.host_go_on_my_roster,
                LobbyMode::Off => true,
            };
        let timed_out =
            self.lobby == LobbyMode::Off && now.duration_since(self.started) >= JOIN_WINDOW;
        if close {
            Status::Agreed { roster: live }
        } else if timed_out {
            Status::Failed
        } else {
            Status::Forming { live: live.len() }
        }
    }
}

pub fn encode_beat(beat: &Beat) -> Vec<u8> {
    let canon = |ids: &[EndpointId]| -> Vec<EndpointId> {
        let mut v = ids.to_vec();
        v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        v.dedup();
        v
    };
    let members = canon(&beat.members);
    let mut out = Vec::with_capacity(20 + 32 * members.len());
    out.push(beat.start as u8);
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&beat.stamp.body_digest.to_le_bytes());
    out.extend_from_slice(&beat.stamp.plant_digest.to_le_bytes());
    out.push(beat.stamp.crab_count);
    for id in &members {
        out.extend_from_slice(id.as_bytes());
    }
    out
}

const MAX_BEAT_MEMBERS: usize = 256;

pub fn decode_beat(body: &[u8]) -> Result<Beat> {
    anyhow::ensure!(
        body.len() >= 20,
        "barrier frame too short for start+count+asset digest+plant digest+crab count"
    );
    let start = body[0] != 0;
    let count = u16::from_le_bytes([body[1], body[2]]) as usize;
    anyhow::ensure!(
        count <= MAX_BEAT_MEMBERS,
        "barrier frame claims {count} members (> {MAX_BEAT_MEMBERS})"
    );
    let body_digest = u64::from_le_bytes(body[3..11].try_into().expect("8-byte slice"));
    let plant_digest = u64::from_le_bytes(body[11..19].try_into().expect("8-byte slice"));
    let crab_count = body[19];
    let need = 20 + 32 * count;
    anyhow::ensure!(
        body.len() == need,
        "barrier frame length {} != expected {need} for {count} members",
        body.len()
    );
    let mut members = Vec::with_capacity(count);
    for i in 0..count {
        let off = 20 + 32 * i;
        let bytes: [u8; 32] = body[off..off + 32].try_into().expect("32-byte slice");
        let id = EndpointId::from_bytes(&bytes)
            .map_err(|e| anyhow::anyhow!("bad endpoint id in barrier frame: {e}"))?;
        members.push(id);
    }
    Ok(Beat {
        members,
        start,
        stamp: crate::SyncStamp {
            body_digest,
            plant_digest,
            crab_count,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eid(i: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[i; 32]).public()
    }

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    fn bt(members: Vec<EndpointId>) -> Beat {
        Beat {
            members,
            start: false,
            stamp: crate::SyncStamp::ZERO,
        }
    }

    fn stamp(body_digest: u64, plant_digest: u64, crab_count: u8) -> crate::SyncStamp {
        crate::SyncStamp {
            body_digest,
            plant_digest,
            crab_count,
        }
    }

    fn sorted(ids: &[EndpointId]) -> Vec<EndpointId> {
        let mut v = ids.to_vec();
        v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        v
    }

    #[test]
    fn beat_wire_roundtrips_and_is_order_independent() {
        let a = bt(vec![eid(3), eid(1), eid(2)]);
        let b = bt(vec![eid(1), eid(2), eid(3)]);
        assert_eq!(encode_beat(&a), encode_beat(&b));
        assert_eq!(a.roster_hash(), b.roster_hash());
        let decoded = decode_beat(&encode_beat(&a)).unwrap();
        assert_eq!(decoded.members, sorted(&[eid(1), eid(2), eid(3)]));
        assert!(!decoded.start, "a plain beat decodes with no start GO");
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(
            decode_beat(&[0]).is_err(),
            "too short for start+count+digests"
        );
        // A full 20-byte header claiming 5 members but carrying none — must trip the
        // length-vs-count check, not the header-size check.
        let mut truncated = vec![0u8];
        truncated.extend_from_slice(&5u16.to_le_bytes());
        truncated.extend_from_slice(&0u64.to_le_bytes());
        truncated.extend_from_slice(&0u64.to_le_bytes());
        truncated.push(0);
        assert!(decode_beat(&truncated).is_err(), "truncated body");
        let mut huge = vec![0u8];
        huge.extend_from_slice(&(300u16).to_le_bytes());
        huge.extend_from_slice(&0u64.to_le_bytes());
        huge.extend_from_slice(&0u64.to_le_bytes());
        huge.push(0);
        huge.resize(20 + 32 * 300, 0);
        assert!(decode_beat(&huge).is_err(), "over-large count rejected");
    }

    #[test]
    fn start_flag_roundtrips_without_perturbing_the_roster_hash() {
        let members = vec![eid(1), eid(2)];
        let plain = bt(members.clone());
        let go = Beat {
            members,
            start: true,
            stamp: crate::SyncStamp::ZERO,
        };
        assert_eq!(
            plain.roster_hash(),
            go.roster_hash(),
            "the GO flag must not be part of the roster hash"
        );
        assert!(!decode_beat(&encode_beat(&plain)).unwrap().start);
        assert!(decode_beat(&encode_beat(&go)).unwrap().start);
    }

    #[test]
    fn distinct_sets_hash_distinct() {
        assert_ne!(
            roster_hash(&[eid(1), eid(2)]),
            roster_hash(&[eid(1), eid(2), eid(3)]),
            "{{A,B}} and {{A,B,C}} must not collide — the close condition relies on it"
        );
    }

    fn bt_ad(members: Vec<EndpointId>, body_digest: u64) -> Beat {
        Beat {
            members,
            start: false,
            stamp: stamp(body_digest, 0, 1),
        }
    }

    #[test]
    fn body_digest_roundtrips_on_the_wire() {
        let members = vec![eid(1), eid(2)];
        let a = bt_ad(members.clone(), 0xC0FF_EE00_1234_5678);
        let b = bt_ad(members, 0x9876_5432_10AB_CDEF);
        assert_eq!(
            a.roster_hash(),
            b.roster_hash(),
            "the asset digest must not be part of the roster hash"
        );
        let decoded = decode_beat(&encode_beat(&a)).unwrap();
        assert_eq!(decoded.stamp.body_digest, 0xC0FF_EE00_1234_5678);
    }

    #[test]
    fn crab_count_gate_is_host_keyed() {
        let beat_c = |members: Vec<EndpointId>, crab_count: u8| Beat {
            members,
            start: false,
            stamp: stamp(0, 0, crab_count),
        };
        let decoded = decode_beat(&encode_beat(&beat_c(vec![eid(1)], 3))).unwrap();
        assert_eq!(decoded.stamp.crab_count, 3, "the count survives the wire");

        let t0 = Instant::now();
        let mut ids = [eid(1), eid(2)];
        ids.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let (host, client) = (ids[0], ids[1]);

        let mut a = Membership::new(client, 2, t0).with_stamp(stamp(0, 0, 2));
        a.on_beat(host, &beat_c(vec![host, client], 2), t0);
        a.poll(t0);
        assert!(a.sync_verdict().crabs, "matching host count passes");

        let mut b = Membership::new(client, 2, t0).with_stamp(stamp(0, 0, 1));
        b.on_beat(host, &beat_c(vec![host, client], 2), t0);
        b.poll(t0);
        assert!(!b.sync_verdict().crabs, "a count mismatch must not arm");

        let mut c = Membership::new(client, 2, t0).with_stamp(stamp(0, 0, 1));
        c.on_beat(host, &beat_c(vec![host, client], 0), t0);
        c.poll(t0);
        assert!(!c.sync_verdict().crabs, "a crab-less host must not arm");

        let mut d = Membership::new(host, 2, t0).with_stamp(stamp(0, 0, 2));
        d.on_beat(client, &beat_c(vec![host, client], 1), t0);
        d.poll(t0);
        assert!(
            d.sync_verdict().crabs,
            "the host self-passes on its own count"
        );

        let mut ids3 = [eid(1), eid(2), eid(3)];
        ids3.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let (h3, mid, top) = (ids3[0], ids3[1], ids3[2]);
        let mut e = Membership::new(top, 3, t0).with_stamp(stamp(0, 0, 1));
        e.on_beat(mid, &beat_c(vec![h3, mid, top], 1), t0);
        e.poll(t0);
        assert!(!e.sync_verdict().crabs, "a relay-only host is unverified");
    }

    #[test]
    fn assets_synced_only_when_all_peers_share_one_nonzero_digest() {
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        const ASSET: u64 = 0x5A11_0000_C0DE_4321;

        let mut a = Membership::new(ida, 2, t0).with_stamp(stamp(ASSET, 0, 0));
        a.on_beat(idb, &bt_ad(vec![ida, idb], ASSET), t0);
        a.poll(t0);
        assert!(
            a.sync_verdict().body,
            "equal non-zero asset digests must be synced"
        );

        let mut b = Membership::new(ida, 2, t0).with_stamp(stamp(ASSET, 0, 0));
        b.on_beat(idb, &bt_ad(vec![ida, idb], ASSET ^ 0xFF), t0);
        b.poll(t0);
        assert!(
            !b.sync_verdict().body,
            "a differing peer asset digest must not be synced"
        );

        let mut c = Membership::new(ida, 2, t0).with_stamp(stamp(ASSET, 0, 0));
        c.on_beat(idb, &bt_ad(vec![ida, idb], 0), t0);
        c.poll(t0);
        assert!(
            !c.sync_verdict().body,
            "a zero peer asset digest must not be synced"
        );

        let mut d = Membership::new(ida, 2, t0);
        d.on_beat(idb, &bt_ad(vec![ida, idb], ASSET), t0);
        d.poll(t0);
        assert!(
            !d.sync_verdict().body,
            "a zero local asset digest is never synced"
        );
    }

    #[test]
    fn plant_synced_only_when_all_peers_share_one_nonzero_digest() {
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        const PLANT: u64 = 0x7E44_A100_0BAD_5EED;
        let bt_plant = |members: Vec<EndpointId>, plant_digest: u64| Beat {
            members,
            start: false,
            stamp: stamp(0, plant_digest, 0),
        };

        let decoded = decode_beat(&encode_beat(&bt_plant(vec![ida], PLANT))).unwrap();
        assert_eq!(
            decoded.stamp.plant_digest, PLANT,
            "the digest survives the wire"
        );

        let mut a = Membership::new(ida, 2, t0).with_stamp(stamp(0, PLANT, 0));
        a.on_beat(idb, &bt_plant(vec![ida, idb], PLANT), t0);
        a.poll(t0);
        assert!(
            a.sync_verdict().plant,
            "equal non-zero plant digests must be synced"
        );

        let mut b = Membership::new(ida, 2, t0).with_stamp(stamp(0, PLANT, 0));
        b.on_beat(idb, &bt_plant(vec![ida, idb], PLANT ^ 0xFF), t0);
        b.poll(t0);
        assert!(
            !b.sync_verdict().plant,
            "a differing peer plant digest (other arena/bake/friction) must not be synced"
        );

        let mut c = Membership::new(ida, 2, t0);
        c.on_beat(idb, &bt_plant(vec![ida, idb], PLANT), t0);
        c.poll(t0);
        assert!(
            !c.sync_verdict().plant,
            "a zero local plant digest is never synced"
        );
    }

    #[test]
    fn relay_only_peer_blocks_the_asset_gate() {
        let t0 = Instant::now();
        let (me, direct, relayed) = (eid(1), eid(2), eid(3));
        const ASSET: u64 = 0x9999_8888_7777_6666;
        let mut a = Membership::new(me, 3, t0).with_stamp(stamp(ASSET, 0, 0));
        a.on_beat(direct, &bt_ad(vec![me, direct, relayed], ASSET), t0);
        a.poll(t0);
        assert!(
            !a.sync_verdict().body,
            "a relay-only peer (digest None) must block the asset gate"
        );
        a.on_beat(relayed, &bt_ad(vec![me, direct, relayed], ASSET), t0);
        a.poll(t0);
        assert!(
            a.sync_verdict().body,
            "once every peer is heard directly with the same asset, synced"
        );
    }

    #[test]
    fn lone_peer_with_expect_one_waits_then_agrees_on_itself() {
        let t0 = Instant::now();
        let mut m = Membership::new(eid(1), 1, t0);
        assert_eq!(m.poll(t0), Status::Forming { live: 1 });
        assert_eq!(m.poll(at(t0, 500)), Status::Forming { live: 1 });
        match m.poll(at(t0, STABLE_FOR.as_millis() as u64 + 1)) {
            Status::Agreed { roster, .. } => assert_eq!(roster, vec![eid(1)]),
            other => panic!("lone peer should agree on itself, got {other:?}"),
        }
    }

    #[test]
    fn lone_peer_expecting_more_times_out_instead_of_freezing_solo() {
        let t0 = Instant::now();
        let mut m = Membership::new(eid(1), 2, t0);
        assert_eq!(
            m.poll(at(t0, STABLE_FOR.as_millis() as u64 + 100)),
            Status::Forming { live: 1 }
        );
        assert_eq!(
            m.poll(at(t0, JOIN_WINDOW.as_millis() as u64 + 1)),
            Status::Failed
        );
    }

    #[test]
    fn two_peers_converge_and_freeze_the_identical_set() {
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        let mut t = 0u64;
        let mut agreed_a = None;
        let mut agreed_b = None;
        while t <= 3000 && (agreed_a.is_none() || agreed_b.is_none()) {
            let now = at(t0, t);
            let beat_a = a.beat();
            let beat_b = b.beat();
            a.on_beat(idb, &beat_b, now);
            b.on_beat(ida, &beat_a, now);
            if agreed_a.is_none()
                && let Status::Agreed { roster, .. } = a.poll(now)
            {
                agreed_a = Some(roster);
            }
            if agreed_b.is_none()
                && let Status::Agreed { roster, .. } = b.poll(now)
            {
                agreed_b = Some(roster);
            }
            t += 250;
        }
        let ra = agreed_a.expect("A must agree");
        let rb = agreed_b.expect("B must agree");
        assert_eq!(ra, rb, "both peers MUST freeze the identical set");
        assert_eq!(ra, sorted(&[ida, idb]));
    }

    #[test]
    fn staggered_join_makes_everyone_wait_for_the_late_peer() {
        let t0 = Instant::now();
        let (ida, idb, idc) = (eid(1), eid(2), eid(3));
        let mut a = Membership::new(ida, 3, t0);
        let mut b = Membership::new(idb, 3, t0);
        let mut c = Membership::new(idc, 3, t0);

        let join_at = 1000u64;
        let mut froze: BTreeMap<u8, Vec<EndpointId>> = BTreeMap::new();
        let mut t = 0u64;
        while t <= 6000 && froze.len() < 3 {
            let now = at(t0, t);
            let (ba, bb, bc) = (a.beat(), b.beat(), c.beat());
            a.on_beat(idb, &bb, now);
            b.on_beat(ida, &ba, now);
            if t >= join_at {
                a.on_beat(idc, &bc, now);
                b.on_beat(idc, &bc, now);
                c.on_beat(ida, &ba, now);
                c.on_beat(idb, &bb, now);
            }
            for (id, m) in [(1u8, &mut a), (2, &mut b), (3, &mut c)] {
                if let Status::Agreed { roster, .. } = m.poll(now)
                    && !froze.contains_key(&id)
                {
                    froze.insert(id, roster);
                }
            }
            t += 250;
        }
        assert_eq!(froze.len(), 3, "all three must eventually freeze");
        let expected = sorted(&[ida, idb, idc]);
        for (id, roster) in &froze {
            assert_eq!(roster, &expected, "peer {id} froze the wrong set");
        }
    }

    #[test]
    fn phantom_endpoint_expires_and_is_excluded() {
        let t0 = Instant::now();
        let (ida, idb, idp) = (eid(1), eid(2), eid(9));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        let phantom_beat = bt(vec![idp]);
        a.on_beat(idp, &phantom_beat, t0);
        b.on_beat(idp, &phantom_beat, t0);

        let mut froze_a = None;
        let mut froze_b = None;
        let mut t = 0u64;
        while t <= 8000 && (froze_a.is_none() || froze_b.is_none()) {
            let now = at(t0, t);
            let (ba, bb) = (a.beat(), b.beat());
            a.on_beat(idb, &bb, now);
            b.on_beat(ida, &ba, now);
            if froze_a.is_none()
                && let Status::Agreed { roster, .. } = a.poll(now)
            {
                froze_a = Some(roster);
            }
            if froze_b.is_none()
                && let Status::Agreed { roster, .. } = b.poll(now)
            {
                froze_b = Some(roster);
            }
            t += 250;
        }
        let ra = froze_a.expect("A agrees");
        let rb = froze_b.expect("B agrees");
        assert_eq!(ra, sorted(&[ida, idb]), "phantom P must be excluded");
        assert_eq!(ra, rb, "both freeze the identical phantom-free set");
        assert!(
            !ra.contains(&idp),
            "the stale endpoint must not be a member"
        );
    }

    #[test]
    fn divergent_views_never_freeze_mismatched_sets() {
        let t0 = Instant::now();
        let (ida, idb, idc) = (eid(1), eid(2), eid(3));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        let cbeat = bt(vec![idc]);
        b.on_beat(idc, &cbeat, t0);

        let mut t = 0u64;
        while t <= 2500 {
            let now = at(t0, t);
            let ba = a.beat();
            b.on_beat(idc, &cbeat, now);
            a.on_beat(idb, &bt(vec![ida, idb]), now);
            b.on_beat(ida, &ba, now);
            let sa = a.poll(now);
            let sb = b.poll(now);
            if let Status::Agreed { roster, .. } = sa {
                assert_eq!(roster, sorted(&[ida, idb]));
                assert!(
                    !matches!(sb, Status::Agreed { .. }),
                    "B must not freeze {{A,B,C}} while A freezes {{A,B}} — divergent freeze"
                );
            }
            t += 250;
        }
    }

    #[test]
    fn flickering_agreement_never_closes() {
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        let mut a = Membership::new(ida, 2, t0);
        let other = eid(7);
        let agree = bt(vec![ida, idb]);
        let disagree = bt(vec![ida, idb, other]);
        let mut t = 0u64;
        while t <= JOIN_WINDOW.as_millis() as u64 - 1000 {
            let now = at(t0, t);
            let bb = if (t / 250).is_multiple_of(2) {
                &agree
            } else {
                &disagree
            };
            a.on_beat(idb, bb, now);
            assert!(
                !matches!(a.poll(now), Status::Agreed { .. }),
                "a flickering agreement must never close (continuous-agreement guarantee)"
            );
            t += 250;
        }
    }

    #[test]
    fn late_joiner_within_settle_rewinds_everyone_off_the_partial_set() {
        let t0 = Instant::now();
        let (ida, idb, idc) = (eid(1), eid(2), eid(3));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);
        let mut c = Membership::new(idc, 2, t0);

        let join_at = 1000u64;
        let abc = sorted(&[ida, idb, idc]);
        let mut froze: BTreeMap<u8, Vec<EndpointId>> = BTreeMap::new();
        let mut t = 0u64;
        while t <= 6000 && froze.len() < 3 {
            let now = at(t0, t);
            let (ba, bb, bc) = (a.beat(), b.beat(), c.beat());
            a.on_beat(idb, &bb, now);
            b.on_beat(ida, &ba, now);
            if t >= join_at {
                a.on_beat(idc, &bc, now);
                b.on_beat(idc, &bc, now);
                c.on_beat(ida, &ba, now);
                c.on_beat(idb, &bb, now);
            }
            for (id, m) in [(1u8, &mut a), (2, &mut b), (3, &mut c)] {
                if let Status::Agreed { roster, .. } = m.poll(now)
                    && !froze.contains_key(&id)
                {
                    assert_eq!(roster, abc, "peer {id} froze a partial/wrong set");
                    froze.insert(id, roster);
                }
            }
            t += 250;
        }
        assert_eq!(froze.len(), 3, "all three must converge and freeze");
        for roster in froze.values() {
            assert_eq!(roster, &abc);
        }
        assert!(join_at < STABLE_FOR.as_millis() as u64);
    }

    #[test]
    fn relayed_phantom_does_not_ping_pong_back_into_the_set() {
        let t0 = Instant::now();
        let (ida, idb, idp) = (eid(1), eid(2), eid(9));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        let pbeat = bt(vec![idp]);
        a.on_beat(idp, &pbeat, t0);
        b.on_beat(idp, &pbeat, at(t0, 400));

        let mut froze_a = None;
        let mut froze_b = None;
        let mut t = 0u64;
        while t <= 9000 && (froze_a.is_none() || froze_b.is_none()) {
            let now = at(t0, t);
            let ba = a.beat();
            let bb = b.beat();
            a.on_beat(idb, &bb, now);
            b.on_beat(ida, &ba, now);
            if froze_a.is_none()
                && let Status::Agreed { roster, .. } = a.poll(now)
            {
                froze_a = Some(roster);
            }
            if froze_b.is_none()
                && let Status::Agreed { roster, .. } = b.poll(now)
            {
                froze_b = Some(roster);
            }
            t += 250;
        }
        let ra = froze_a.expect("A must still settle despite the phantom relay churn");
        let rb = froze_b.expect("B must still settle despite the phantom relay churn");
        assert_eq!(
            ra,
            sorted(&[ida, idb]),
            "P must not ping-pong back into the set"
        );
        assert_eq!(ra, rb, "both settle on the identical phantom-free set");
        assert!(!ra.contains(&idp));
    }

    #[test]
    fn fails_cleanly_when_agreement_never_reached() {
        let t0 = Instant::now();
        let mut m = Membership::new(eid(1), 1, t0);
        let mut t = 0u64;
        let mut saw_failed = false;
        while t <= JOIN_WINDOW.as_millis() as u64 + 2000 {
            let now = at(t0, t);
            let transient = eid((t / 250 % 200 + 10) as u8);
            m.on_beat(transient, &bt(vec![transient]), now);
            if m.poll(now) == Status::Failed {
                saw_failed = true;
                break;
            }
            t += 250;
        }
        assert!(
            saw_failed,
            "perpetual churn must FAIL at the join window, never silently freeze"
        );
    }

    #[test]
    fn host_triggered_joiner_waits_for_the_go_then_both_freeze_the_identical_set() {
        let t0 = Instant::now();
        let (idh, idj) = (eid(1), eid(2));
        let mut h = Membership::host_triggered(Role::Host, idh, 2, t0);
        let mut j = Membership::host_triggered(Role::Joiner, idj, 2, t0);

        let mut t = 0u64;
        while t <= STABLE_FOR.as_millis() as u64 + 1000 {
            let now = at(t0, t);
            let (bh, bj) = (h.beat(), j.beat());
            h.on_beat(idj, &bj, now);
            j.on_beat(idh, &bh, now);
            assert!(
                !matches!(h.poll(now), Status::Agreed { .. }),
                "host must not auto-close before its own Start click"
            );
            assert!(
                !matches!(j.poll(now), Status::Agreed { .. }),
                "joiner must lobby until the host's GO, never auto-close on the timer"
            );
            t += 250;
        }

        h.set_starting();
        let mut froze_h = None;
        let mut froze_j = None;
        while t <= STABLE_FOR.as_millis() as u64 + 4000 && (froze_h.is_none() || froze_j.is_none())
        {
            let now = at(t0, t);
            let (bh, bj) = (h.beat(), j.beat());
            h.on_beat(idj, &bj, now);
            j.on_beat(idh, &bh, now);
            if froze_h.is_none()
                && let Status::Agreed { roster, .. } = h.poll(now)
            {
                froze_h = Some(roster);
            }
            if froze_j.is_none()
                && let Status::Agreed { roster, .. } = j.poll(now)
            {
                froze_j = Some(roster);
            }
            t += 250;
        }
        let rh = froze_h.expect("host must close on its own Start once agreed");
        let rj = froze_j.expect("joiner must close on the host's GO");
        assert_eq!(rh, rj, "host and joiner MUST freeze the identical set");
        assert_eq!(rh, sorted(&[idh, idj]));
    }

    #[test]
    fn host_go_on_a_mismatched_roster_does_not_close_a_joiner() {
        let t0 = Instant::now();
        let (idh, idj) = (eid(1), eid(2));
        let mut j = Membership::host_triggered(Role::Joiner, idj, 2, t0);

        let host_go_on_partial = Beat {
            members: vec![idh],
            start: true,
            stamp: crate::SyncStamp::ZERO,
        };
        let mut t = 0u64;
        let mut closed = false;
        while t <= STABLE_FOR.as_millis() as u64 + 2000 {
            let now = at(t0, t);
            j.on_beat(idh, &host_go_on_partial, now);
            if matches!(j.poll(now), Status::Agreed { .. }) {
                closed = true;
                break;
            }
            t += 250;
        }
        assert!(
            !closed,
            "a GO whose roster hash differs from ours must never close us (no divergent freeze)"
        );
    }

    #[test]
    fn only_a_host_ever_advertises_the_start_go() {
        // The protocol guarantee the [`LobbyMode`] type ENFORCES: only a host can put
        // `start` on the wire. A timer barrier and a JOINER both stay silent on the GO
        // even after `set_starting`, so a joiner can never command a start it isn't
        // entitled to.
        let t0 = Instant::now();
        for mut m in [
            Membership::new(eid(1), 1, t0),
            Membership::host_triggered(Role::Joiner, eid(1), 2, t0),
        ] {
            m.set_starting();
            assert!(
                !m.beat().start,
                "only a Role::Host may advertise the start GO"
            );
        }
        let mut host = Membership::host_triggered(Role::Host, eid(1), 2, t0);
        assert!(!host.beat().start, "a host is silent before clicking Start");
        host.set_starting();
        assert!(host.beat().start, "a host advertises the GO after Start");
    }

    #[test]
    fn host_clicking_start_during_a_late_join_does_not_freeze_a_partial_set() {
        let t0 = Instant::now();
        let (idh, idj, idc) = (eid(1), eid(2), eid(3));
        let mut h = Membership::host_triggered(Role::Host, idh, 3, t0);
        let mut j = Membership::host_triggered(Role::Joiner, idj, 3, t0);
        let mut c = Membership::host_triggered(Role::Joiner, idc, 3, t0);
        h.set_starting();

        let join_at = 1000u64;
        let abc = sorted(&[idh, idj, idc]);
        let mut froze: BTreeMap<u8, Vec<EndpointId>> = BTreeMap::new();
        let mut t = 0u64;
        while t <= 8000 && froze.len() < 3 {
            let now = at(t0, t);
            let (bh, bj, bc) = (h.beat(), j.beat(), c.beat());
            h.on_beat(idj, &bj, now);
            j.on_beat(idh, &bh, now);
            if t >= join_at {
                h.on_beat(idc, &bc, now);
                j.on_beat(idc, &bc, now);
                c.on_beat(idh, &bh, now);
                c.on_beat(idj, &bj, now);
            }
            for (id, m) in [(1u8, &mut h), (2, &mut j), (3, &mut c)] {
                if let Status::Agreed { roster, .. } = m.poll(now)
                    && !froze.contains_key(&id)
                {
                    assert_eq!(roster, abc, "peer {id} froze a partial/wrong set");
                    froze.insert(id, roster);
                }
            }
            t += 250;
        }
        assert_eq!(
            froze.len(),
            3,
            "all three must converge and freeze {{H,J,C}}"
        );
        assert!(
            join_at < STABLE_FOR.as_millis() as u64,
            "C must join within the settle window for this to test the rewind"
        );
    }
}
