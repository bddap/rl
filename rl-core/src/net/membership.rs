//! LAN match-formation barrier: agree on ONE participant set before anyone ticks.
//!
//! The cold-start's job is to turn "some processes launched on a LAN" into a single
//! agreed match. The hard requirement is the determinism invariant of
//! [`crate::net::lockstep`]: every peer MUST freeze the byte-identical sorted
//! participant set, or the deterministic sims assign different [`PlayerId`]s and
//! desync. The old `discover_and_freeze` was best-effort — poll the connected set for
//! a few seconds, then freeze whatever showed up — which broke in three ways during a
//! live playtest: a stale/phantom `game` endpoint lingering on the LAN got pulled into
//! the roster; racy discovery let peers freeze divergent sets (A saw {A,B}, B saw
//! {A,B,C}); and a staggered launch left one peer mid-formation when another froze. Any
//! of these freezes mismatched rosters → the lockstep stalls forever on inputs from a
//! "member" who isn't really there (or whom others don't have), and the world freezes.
//!
//! This module is the real barrier. The protocol is deliberately couch-scale-simple —
//! a handful of peers on one LAN, not a matchmaking service — and splits into a pure,
//! fully-tested core ([`Membership`], the agreement state machine) and a thin async
//! driver ([`run_barrier`]) that does only I/O around it.
//!
//! ## Why it guarantees an identical frozen set
//!
//! 1. **Liveness kills phantoms.** A peer counts toward the roster only while we are
//!    receiving recent [`Beat`]s from it (a heartbeat carried on the barrier channel
//!    over its live QUIC link). A crashed/stale endpoint sends none, so it expires
//!    ([`MEMBER_TIMEOUT`]) and is never admitted — even if its mDNS record lingers.
//! 2. **Gossip to a fixpoint handles races + stagger.** Every peer periodically
//!    advertises its *full* view — its sorted live-member set — and merges any ids it
//!    hears from others (transitive: a late joiner B that only A saw is relayed by A
//!    to C). Within the join window the set only grows, and every member echoes the
//!    whole set, so all peers' views converge to the same union.
//! 3. **CONTINUOUS unanimous agreement on one hash is the close condition.** Each
//!    advertisement carries a [`roster_hash`] of the sender's set. The agreement
//!    predicate is: enough players are present (`live >= expect`) AND *every* live
//!    member is advertising the SAME hash as our own. A peer closes the barrier only
//!    when that predicate has held **continuously** for [`STABLE_FOR`] — a single
//!    beat's worth of agreement does NOT count; any peer's hash drifting away, or any
//!    membership change, rewinds the timer. That continuity is what makes the freeze
//!    identical: A cannot close on {A,B} while B closes on {A,B,C}, because closing
//!    requires B to have been echoing hash({A,B}) for the whole window — which it will
//!    not do while it knows about C. So either everyone closes on the same set, or
//!    nobody closes. (Sampling agreement only at the close instant would be a bug: a
//!    one-beat flicker of a stale cached hash could let one peer close a set another
//!    abandons. The continuous timer closes that hole.)
//! 4. **Reject, never silently freeze a partial set.** If the overall [`JOIN_WINDOW`]
//!    elapses without that sustained agreement (a flapping peer, an asymmetric link, or
//!    fewer than `expect` players ever showing), the barrier fails with an error the
//!    caller surfaces ("couldn't form a match — retry") rather than guessing a roster.
//!    A wrong guess is the exact failure this module exists to prevent. The `expect`
//!    floor also stops a peer from freezing a lone {self} match before discovery has
//!    even found the others (LAN mDNS can take a second or two).
//!
//! The frozen set is then handed to [`crate::net::net_loop`], which assigns
//! [`PlayerId`]s by sorted endpoint id exactly as before — but now over a set proven
//! identical on every peer, not merely hoped to be.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

/// How long a member may go without a [`Beat`] before we drop it from our live set.
/// Comfortably longer than [`BEAT_EVERY`] so a healthy peer is never flapped out by a
/// single late heartbeat, short enough that a crashed/stale endpoint clears within the
/// join window and can't poison the roster.
pub const MEMBER_TIMEOUT: Duration = Duration::from_secs(3);

/// How often each peer broadcasts its view ([`Beat`] + roster). Several beats fit
/// inside [`MEMBER_TIMEOUT`], so liveness survives isolated drops, and inside
/// [`STABLE_FOR`], so agreement is re-confirmed continuously while we wait to close.
pub const BEAT_EVERY: Duration = Duration::from_millis(250);

/// The agreement predicate (enough players present AND every live member advertising
/// our exact roster hash) must hold CONTINUOUSLY for this long before we close the
/// barrier. This is the settle that absorbs a staggered launch: a peer joining 1–2 s
/// late perturbs the set, breaking the predicate and rewinding the timer, so no one
/// closes until the dust settles and all views match. Long enough to outlast normal
/// mDNS discovery jitter on a LAN. Must stay comfortably below [`MEMBER_TIMEOUT`] so
/// that once everyone agrees, every peer finishes closing before the first peer to
/// close (which then goes quiet) could be expired by a straggler — the margin that
/// keeps close-time skew from flipping a late closer's set.
pub const STABLE_FOR: Duration = Duration::from_millis(1500);

/// Hard cap on total formation time. If sustained agreement is not reached within this,
/// the barrier FAILS (see module docs) rather than freezing a divergent/partial set.
/// Generous: real LAN formation settles in 2–3 s; this only trips on a genuinely sick
/// group (a flapping peer, an asymmetric link, or too few players showing up).
pub const JOIN_WINDOW: Duration = Duration::from_secs(20);

/// Hard cap on the TOTAL roster (us + peers), so a peer flooding a [`Beat`] with bogus
/// relayed ids can't inflate our set without bound (and thus can't push our own
/// advertised beat past the transport's frame cap, which would make honest peers drop
/// us). [`PlayerId`] is a `u8`, so the rest of the stack — id assignment, the beat wire
/// ([`MAX_BEAT_MEMBERS`]) — tops out at exactly 256 total; couch co-op is a handful.
/// Once `live_set` would reach this, we stop ADMITTING new ids (existing members and
/// their direct beats are unaffected) — a roster this large would never agree+freeze on
/// a real LAN, so refusing further growth only bounds memory, it can't lose a real match.
/// The cap is on the TOTAL (hence the `+ 1` for `me` at the admission sites) so our own
/// max beat stays ≤ 256 ids and every honest peer can still decode it.
pub const MAX_MEMBERS: usize = u8::MAX as usize + 1;

/// What a peer broadcasts each [`BEAT_EVERY`] on the barrier channel: a heartbeat that
/// proves it is alive plus its current view of the membership, so peers converge to a
/// common set (see module docs). Self-describing — `members` always includes the
/// sender — so a receiver needs nothing but this frame to merge a sender's view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Beat {
    /// The sender's full current live-member set, sorted by endpoint-id bytes. Includes
    /// the sender. The receiver unions these ids into its own view (transitive gossip)
    /// and reads [`Beat::roster_hash`] to check agreement.
    pub members: Vec<EndpointId>,
    /// The host's synchronized-start command (rl#58): `true` once the host has clicked
    /// Start, commanding the round to begin on the roster carried in THIS same beat. A
    /// joiner closes the barrier on a direct `start` beat whose roster hash equals its
    /// own — so host and joiners freeze the byte-identical set the GO names, never a
    /// divergent one (the same unanimity safety as the timer close, just host-triggered).
    /// Always `false` outside the host-triggered menu path, so every other caller's wire
    /// and close behaviour is unchanged.
    pub start: bool,
    /// The sender's policy-weights digest (rl#82, GCR), `0` for no checkpoint — see
    /// [`Membership::weights_synced`] for what it gates. Deliberately NOT folded into
    /// [`Beat::roster_hash`]: it is a capability advertisement, not membership, so two peers
    /// with the identical roster still freeze the same set regardless of their brains (a
    /// weights mismatch keeps the integer crab; it never breaks match formation).
    pub weights_digest: u64,
    /// The sender's crab-MODEL-asset digest (rl#100, GCR), `0` for no resolvable model — the
    /// giant crab's rapier colliders are derived from this asset
    /// ([`crate::bot::meshfit::crab_asset_digest`]), so two peers with different crab models
    /// build different colliders and silently desync. A SIBLING of `weights_digest`, handled
    /// identically: NOT folded into [`Beat::roster_hash`] (a capability advertisement, not
    /// membership — a mismatch keeps the integer crab, never breaks formation), self-declared,
    /// never relayed. See [`Membership::assets_synced`] for what it gates.
    pub asset_digest: u64,
    /// The sender's view of WHO PILOTS a plane (a subset of `members`, sorted by id bytes) —
    /// the networked half of plane mode (vehicles in multiplayer). Unlike `weights_digest`,
    /// this IS folded into [`Beat::roster_hash`]: the frozen pilot roster must be a pure
    /// function of agreed data, so a peer holding a divergent view of who flies must NOT be
    /// able to close (a split pilot set spawns different bodies per peer in
    /// [`crate::net::sim::Sim::new_with_pilots`] and desyncs instantly). A sender lists
    /// another member here only once it has heard that member's intent on the member's OWN
    /// direct beat — intent is self-declared, never relayed, so each peer learns every
    /// member's role by hearing it directly (which the barrier already requires before close).
    pub pilots: Vec<EndpointId>,
}

impl Beat {
    /// The agreement token: a hash folding BOTH `members` AND `pilots`. Two peers
    /// advertising the same token have the identical participant set AND the identical view
    /// of who pilots (the freeze precondition). Computed from the sorted, deduped id bytes so
    /// it is independent of insertion order — the same (members, pilots) always hashes the
    /// same on every peer. FNV-1a over the raw 32-byte ids: this is an agreement check among
    /// cooperating LAN peers, not an adversarial digest, so a fast non-cryptographic hash is
    /// the right tool (a collision would at worst admit a wrong freeze, which the post-freeze
    /// lockstep hash cross-check still catches). See [`agreement_token`] for why pilots are
    /// folded in (and `weights_digest` is not).
    pub fn roster_hash(&self) -> u64 {
        agreement_token(&self.members, &self.pilots)
    }
}

/// Hash a participant set for the agreement check. Canonicalizes (sort + dedup) first
/// so the result is a pure function of the SET, not the order ids were added — both
/// the advertiser and the checker must get the same number for the same membership.
pub fn roster_hash(ids: &[EndpointId]) -> u64 {
    let mut sorted: Vec<&EndpointId> = ids.iter().collect();
    sorted.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    sorted.dedup();
    // FNV-1a/64.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in sorted {
        for &byte in id.as_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The full agreement token folds the member set AND the pilot set, so two peers agree only
/// when they hold the identical roster AND the identical view of who pilots. Pilot-intent is
/// folded in (unlike [`Beat::weights_digest`], which is a side-channel capability) precisely
/// because the frozen pilot roster must be a pure function of agreed data: with it folded, a
/// divergent pilot view breaks unanimity and rewinds the [`STABLE_FOR`] timer exactly like a
/// membership change, so peers can never freeze split pilot sets (which would spawn different
/// bodies per peer and desync instantly). Role-aware: the member set seeds the hash and the
/// pilot set mixes in after, so the two dimensions can't cancel. Pilots ⊆ members always, so
/// this is just `roster_hash(members)` with the pilot subset stirred in.
pub fn agreement_token(members: &[EndpointId], pilots: &[EndpointId]) -> u64 {
    let mut h = roster_hash(members);
    for byte in roster_hash(pilots).to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The agreement state machine: who is currently live, what each peer is advertising,
/// and whether the agreement predicate has held long enough to close on an identical
/// set. Pure and time-injected (every mutator takes `now`) so the whole protocol is
/// unit-testable with a fake clock and no network — the async driver ([`run_barrier`])
/// only feeds it [`Membership::on_beat`] events and reads [`Membership::poll`].
///
/// Membership during the join window grows by admission and shrinks by expiry: a member
/// that stops beating directly EXPIRES ([`MEMBER_TIMEOUT`]) — the phantom-endpoint
/// eviction. The close decision rests on the *agreement predicate* (enough players AND
/// every live peer echoing our roster hash) having held CONTINUOUSLY for [`STABLE_FOR`];
/// any change to the live set, or any peer's advertised hash drifting off ours, rewinds
/// that timer. Sampling agreement only at the close instant would be unsound (a one-beat
/// flicker of a stale cached hash could close a set a peer is abandoning); the held timer
/// is what makes "everyone closes the same set or nobody closes" a guarantee, not a hope.
pub struct Membership {
    me: EndpointId,
    /// How many participants (incl. us) must be present before we may close. Stops a
    /// peer from freezing a lone `{self}` match before LAN discovery has even found the
    /// others. The caller's expected player count; a too-low real turnout times out
    /// ([`Status::Failed`]) instead of forming a short match.
    expect: usize,
    /// Live peers (excludes `me`): endpoint id → its view. A peer is in the match iff it
    /// is here AND not expired. `me` is always a member implicitly (see
    /// [`Membership::live_set`]).
    peers: BTreeMap<EndpointId, PeerView>,
    /// Endpoint ids we recently EXPIRED, with the expiry time. A *relay* may not
    /// re-admit an id while it sits here ([`MEMBER_TIMEOUT`] of quarantine); only the
    /// peer's OWN direct beat clears it (a peer that genuinely came back speaks for
    /// itself). This kills the phantom ping-pong: without it, two peers expiring a
    /// once-seen dead endpoint at slightly different moments would relay it back and
    /// forth, each re-admission granting a fresh lease + rewinding the agreement timer,
    /// so the barrier never settles and spuriously fails.
    tombstones: BTreeMap<EndpointId, Instant>,
    /// When the agreement predicate (see [`Membership::poll`]) most recently became true
    /// AND on which roster hash, or `None` if it is currently false. Keying on the hash
    /// (not just a bool) means *any* change to our agreed set rewinds the timer — even
    /// the (impossible-on-a-real-LAN but cheap-to-rule-out) case where every peer's set
    /// flips S→S′ in the same poll with no peer ever observing disagreement. Closing
    /// requires `Some((t, h))` with `h == my_hash` and `now - t >= STABLE_FOR`. Maintained
    /// only in `poll`.
    agreed_since: Option<(Instant, u64)>,
    /// When formation began, for the [`JOIN_WINDOW`] hard deadline.
    started: Instant,
    /// rl#58 close mode. The sum type that decides HOW the barrier closes, so the two-mode
    /// behaviour can't be a soup of bools and "only the host may GO" is enforced by the type,
    /// not by external wiring. [`LobbyMode::Off`] (the default [`Membership::new`]) is the
    /// unchanged barrier — close on the [`STABLE_FOR`] timer. The lobby variants
    /// ([`Membership::host_triggered`]) replace that with a host's explicit GO; the roster a
    /// peer freezes is the same `live_set` either way (only the close *moment* differs), so
    /// determinism is untouched.
    lobby: LobbyMode,
    /// Whether we've received a direct [`Beat`] with `start` set from a peer whose roster
    /// hash equals our CURRENT one — the host's GO landing on a roster we agree with.
    /// Recomputed each [`Membership::poll`] from peer views, so it can never latch on a
    /// stale roster: a GO seen while our sets disagreed does not close us, and a set change
    /// after a GO re-gates on the new hash. The joiner's sole close trigger in the lobby.
    host_go_on_my_roster: bool,
    /// OUR policy-weights digest (rl#82, GCR), `0` for no checkpoint. Advertised in every
    /// [`Membership::beat`] and the basis of [`Membership::weights_synced`]. `0` by default
    /// ([`Membership::with_weights_digest`] sets it), so a caller that never sets it behaves
    /// exactly as pre-rl#82 — `weights_synced` is then always false.
    local_digest: u64,
    /// OUR crab-model-asset digest (rl#100, GCR), `0` for no resolvable model. Advertised in
    /// every [`Membership::beat`] and the basis of [`Membership::assets_synced`]. `0` by default
    /// ([`Membership::with_asset_digest`] sets it), so a caller that never sets it behaves
    /// exactly as pre-rl#100 — `assets_synced` is then always false. Sibling of `local_digest`.
    local_asset_digest: u64,
    /// Whether WE intend to pilot a plane (the local `RL_VEHICLE=plane` flag). Advertised as
    /// our own entry in every [`Membership::beat`]'s `pilots` list and folded into our
    /// agreement token, so peers converge on a shared pilot roster before freezing. `false` by
    /// default ([`Membership::piloting`] sets it), so a caller that never sets it forms the
    /// unchanged foot-only match (empty pilot set ⇒ byte-identical sim).
    local_pilot: bool,
}

/// How a [`Membership`] barrier decides to close (rl#58). A sum type so the
/// determinism-critical mode is explicit, not inferred, and so the "only a host commands the
/// start" rule is unrepresentable to violate (a [`LobbyMode::Joiner`] has no `starting` to
/// set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LobbyMode {
    /// The default barrier: close on [`STABLE_FOR`] continuous agreement, fail at
    /// [`JOIN_WINDOW`]. Used by the headless `net` driver and the scripted entrypoints —
    /// byte-identical to pre-rl#58.
    Off,
    /// Host of an interactive lobby: `started` flips true on the Start click
    /// ([`Membership::set_starting`]); while true the host advertises the `start` GO and
    /// closes once its own agreement predicate holds. Open-ended (no [`JOIN_WINDOW`] fail).
    Host { started: bool },
    /// Joiner in an interactive lobby: closes only on the host's GO landing on a matching
    /// roster (`host_go_on_my_roster`). Has no way to command a start — that's the point.
    Joiner,
}

/// Our latest knowledge of one peer. Liveness comes from `last_direct` (its OWN beats —
/// relays don't refresh it, the phantom-eviction mechanism). `advertised` is the roster
/// hash it last claimed, or `None` if we've only heard of it via a relay and not yet
/// directly: a `None` can never equal our roster hash, so an unheard peer honestly
/// blocks the close until it speaks for itself (no sentinel value that a real hash could
/// collide with — the make-illegal-states-unrepresentable choice).
#[derive(Debug, Clone, Copy)]
struct PeerView {
    /// Wall-clock of the last DIRECT beat from this peer. A relay of its id by another
    /// peer does NOT bump this.
    last_direct: Instant,
    /// The roster hash this peer last advertised in its own beat, or `None` if not yet
    /// heard directly (gossip-admitted only).
    advertised: Option<u64>,
    /// Whether this peer's most recent DIRECT beat carried the host's `start` GO (rl#58).
    /// Paired with `advertised` so the close check requires the GO and a matching roster
    /// hash *from the same beat* — a host that commanded start on roster R can't close a
    /// joiner that's on a different roster. A relay never sets this (only a direct beat).
    started: bool,
    /// The policy-weights digest this peer last advertised in its OWN direct beat (rl#82),
    /// or `None` if heard only via relay. Like `advertised`, `None` can never equal our
    /// non-zero digest, so a peer not yet heard directly blocks [`Membership::weights_synced`]
    /// until it speaks for itself.
    weights_digest: Option<u64>,
    /// The crab-model-asset digest this peer last advertised in its OWN direct beat (rl#100),
    /// or `None` if heard only via relay. Sibling of `weights_digest`: like it, a `None` can
    /// never equal our non-zero digest, so a peer not yet heard directly blocks
    /// [`Membership::assets_synced`] until it speaks for itself.
    asset_digest: Option<u64>,
    /// Whether this peer declared ITSELF a pilot on its OWN direct beat — `Some(true/false)`
    /// once heard directly, `None` while only gossip-admitted. Read by [`Membership::pilot_set`]
    /// to assemble the frozen pilot roster. `None` (never heard directly) is treated as
    /// not-a-pilot, but such a peer also has `advertised == None`, so it blocks the close until
    /// it speaks for itself — by which point its true role is known. A relay never sets this:
    /// intent is self-declared, learned only from the peer's own beat (the same trust rule as
    /// liveness and `weights_digest`).
    is_pilot: Option<bool>,
}

/// The barrier's verdict on each [`Membership::poll`]: keep waiting, freeze this exact
/// set, or give up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Not yet agreed; keep beating and merging. Carries the current live set size for
    /// logging/UX (a lobby can show "2 players, waiting…").
    Forming { live: usize },
    /// The agreement predicate held continuously for [`STABLE_FOR`]. Freeze EXACTLY
    /// these ids (sorted, includes us) with EXACTLY this pilot subset. Every peer that reaches
    /// this state computed the identical roster AND the identical pilot set — that is the
    /// guarantee (both are folded into the agreement token, so neither can diverge).
    Agreed {
        roster: Vec<EndpointId>,
        /// The frozen pilots (⊆ `roster`, sorted by id bytes): the members that spawn flying.
        /// Identical on every peer by the same agreement-token guarantee as `roster`.
        pilots: Vec<EndpointId>,
    },
    /// [`JOIN_WINDOW`] elapsed without sustained agreement. The caller must surface this
    /// and retry rather than freeze a guessed set.
    Failed,
}

/// Which side of an interactive lobby a peer is (rl#58) — passed to
/// [`Membership::host_triggered`] so the barrier knows whether it may command the start. The
/// boot menu's Host button is [`Role::Host`], Join is [`Role::Joiner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The host: commands the synchronized start ([`Membership::set_starting`]).
    Host,
    /// A joiner: waits for the host's GO; cannot command a start.
    Joiner,
}

impl Membership {
    /// Begin formation. `me` is our own endpoint id (always part of the set); `expect`
    /// is the minimum participant count (incl. us) required before closing — pass the
    /// caller's expected player count, or 1 to allow a deliberate solo-over-network run.
    ///
    /// `expect` MUST be the same on every peer. It's a floor on the *agreed* set, so a
    /// mismatch is the one way two peers could diverge: a set of size N satisfies the
    /// floor for a peer expecting ≤N but not one expecting >N, so the first could close
    /// while the second times out. Every caller passes the same shared launch count, so
    /// this holds; it is a launch invariant, not something negotiated on the wire.
    pub fn new(me: EndpointId, expect: usize, now: Instant) -> Self {
        Self {
            me,
            expect: expect.max(1), // us alone is the floor; 0 would be nonsensical
            peers: BTreeMap::new(),
            tombstones: BTreeMap::new(),
            agreed_since: None,
            started: now,
            lobby: LobbyMode::Off,
            host_go_on_my_roster: false,
            local_digest: 0,
            local_asset_digest: 0,
            local_pilot: false,
        }
    }

    /// Declare whether WE pilot a plane (the local `RL_VEHICLE=plane` intent), advertised in
    /// every [`Membership::beat`] and folded into our agreement token so peers freeze ONE
    /// shared pilot roster. Builder form (chains off [`Membership::new`] /
    /// [`Membership::host_triggered`]) because the intent is a launch-time constant. `false`
    /// (the default) forms the unchanged foot-only match.
    pub fn piloting(mut self, pilot: bool) -> Self {
        self.local_pilot = pilot;
        self
    }

    /// Set OUR policy-weights digest (rl#82, GCR), advertised in every [`Membership::beat`]
    /// so peers can agree on a shared checkpoint before arming the float NN crab in lockstep.
    /// Builder form (chains off [`Membership::new`] / [`Membership::host_triggered`]) because
    /// the digest is a launch-time constant — the loaded checkpoint can't change mid-formation.
    /// `0` (the default) means "no usable checkpoint"; it never counts as synced.
    pub fn with_weights_digest(mut self, digest: u64) -> Self {
        self.local_digest = digest;
        self
    }

    /// Set OUR crab-model-asset digest (rl#100, GCR), advertised in every [`Membership::beat`]
    /// so peers can agree on a shared crab collider asset before arming the float NN crab in
    /// lockstep. Sibling of [`Membership::with_weights_digest`]; same launch-time-constant
    /// builder form (the resolved model can't change mid-formation). `0` (the default) means
    /// "no usable asset"; it never counts as synced.
    pub fn with_asset_digest(mut self, digest: u64) -> Self {
        self.local_asset_digest = digest;
        self
    }

    /// Begin a host-triggered (interactive lobby) formation (rl#58, boot-menu networked
    /// Host/Join only). Same agreement core as [`Membership::new`], but the close trigger is
    /// a host's explicit GO instead of the [`STABLE_FOR`] timer: a joiner lobbies until the
    /// host clicks Start, and the host (via [`Membership::set_starting`]) commands the start
    /// on click. `role` decides which: only a [`Role::Host`] can `set_starting` and advertise
    /// the GO. `expect` is still the participant floor. The frozen roster is identical to the
    /// timer path — only the close *moment* changes — so this introduces no new divergence.
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

    /// Host: command the round to start NOW (the boot menu's Start click). The host then
    /// advertises the GO in its [`Beat`] and closes the barrier as soon as its agreement
    /// predicate holds; joiners close when they receive that GO on a matching roster.
    /// Idempotent, and structurally a no-op for anything but a [`LobbyMode::Host`] — a joiner
    /// or a timer barrier has no `started` to set, so "only the host commands the start" is
    /// the type's guarantee, not an external one.
    pub fn set_starting(&mut self) {
        if let LobbyMode::Host { started } = &mut self.lobby {
            *started = true;
        }
    }

    /// Whether admitting one more peer would keep the TOTAL roster (peers + us) within
    /// [`MAX_MEMBERS`]. Bounds our own advertised beat to ≤ 256 ids so honest peers can
    /// always decode it (the off-by-one that a bare `peers.len() < MAX_MEMBERS` would
    /// miss: that allows 256 peers + us = 257, one past the wire/`PlayerId` ceiling).
    fn has_room_for_one_more(&self) -> bool {
        self.peers.len() + 1 < MAX_MEMBERS
    }

    /// Record a [`Beat`] received from `from` (the QUIC-authenticated sender — never a
    /// value read from the body, same trust rule as the rest of the netcode).
    ///
    /// Liveness comes ONLY from a peer's OWN direct beat: this refreshes `from`'s
    /// `last_direct` and advertised hash, and clears any tombstone on it (it's genuinely
    /// back). The members `from` *relays* are merely ADMITTED if new and not tombstoned
    /// (so an id only one peer can reach propagates to everyone, converging the views) —
    /// a relay does NOT refresh an already-known peer's liveness, nor resurrect a
    /// tombstoned one. That split is what lets a phantom expire and stay gone: a dead
    /// endpoint others still list is relayed to us, but since it never beats *directly*
    /// again its `last_direct` goes stale, [`Membership::poll`] expires it, and the
    /// tombstone keeps relays from re-admitting it. (A peer we can only ever hear via
    /// relay, never directly, will likewise expire — correct, since we couldn't exchange
    /// lockstep inputs with an unreachable peer, so it must not be frozen into the match.)
    pub fn on_beat(&mut self, from: EndpointId, beat: &Beat, now: Instant) {
        // The sender's OWN direct beat: refresh liveness + advertised hash, and lift any
        // quarantine (a genuinely-returned peer speaks for itself).
        if from != self.me {
            self.tombstones.remove(&from);
            if self.peers.contains_key(&from) || self.has_room_for_one_more() {
                let view = self.peers.entry(from).or_insert(PeerView {
                    last_direct: now,
                    advertised: None,
                    started: false,
                    weights_digest: None,
                    asset_digest: None,
                    is_pilot: None,
                });
                view.last_direct = now;
                view.advertised = Some(beat.roster_hash());
                // Latch the host's GO from THIS direct beat; paired with `advertised` so the
                // joiner close requires both the GO and a matching roster from one beat.
                view.started = beat.start;
                // Record the peer's advertised brain digest (rl#82) for the weights-synced
                // check; only a direct beat sets it, like `advertised`/`started`.
                view.weights_digest = Some(beat.weights_digest);
                // Likewise the peer's crab-asset digest (rl#100) for the assets-synced check.
                view.asset_digest = Some(beat.asset_digest);
                // Record whether the peer declared ITSELF a pilot: it lists its own id in its
                // beat's `pilots` iff it intends to fly. Only a direct beat sets this — intent
                // is self-declared, never inferred from a relay's pilot list.
                view.is_pilot = Some(beat.pilots.contains(&from));
            }
        }
        // Transitive admission: a relayed id we don't track yet and that isn't
        // quarantined is a real peer we simply haven't connected to directly — admit it
        // (seeding only its existence, `advertised = None`) so we go hear it directly and
        // the views converge. We never refresh a known member from a relay, never
        // resurrect a tombstoned id from a relay, and never invent an advertised hash —
        // so a gossip-admitted peer blocks the close until it agrees on its own beat.
        // Capped at MAX_MEMBERS (total) so a flooded beat can't inflate our roster.
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
                        weights_digest: None,
                        asset_digest: None,
                        is_pilot: None,
                    },
                );
            }
        }
    }

    /// Our current full live set (us + every non-expired peer), sorted by id bytes.
    /// This is exactly what we advertise in our own [`Beat`] and what we freeze on
    /// agreement, so advertiser and freezer can't drift. (Read-only; expiry happens in
    /// [`Membership::poll`], so call `poll` once per round before relying on this.)
    pub fn live_set(&self) -> Vec<EndpointId> {
        let mut ids: Vec<EndpointId> = self.peers.keys().copied().collect();
        ids.push(self.me);
        ids.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        ids.dedup();
        ids
    }

    /// The pilots among our current live set (us iff [`Membership::piloting`], plus every live
    /// peer whose OWN direct beat declared it a pilot), sorted by id bytes. A subset of
    /// [`Membership::live_set`]. This is exactly what we advertise in our [`Beat`] and freeze
    /// on agreement, so advertiser and freezer can't drift. A gossip-only peer (`is_pilot ==
    /// None`) is excluded — but it also blocks the close (`advertised == None`), so by the time
    /// we could freeze, every member has been heard directly and its role is known. (Read-only;
    /// expiry happens in [`Membership::poll`], so call `poll` once per round first.)
    pub fn pilot_set(&self) -> Vec<EndpointId> {
        let mut pilots: Vec<EndpointId> = self
            .peers
            .iter()
            .filter(|(_, v)| v.is_pilot == Some(true))
            .map(|(id, _)| *id)
            .collect();
        if self.local_pilot {
            pilots.push(self.me);
        }
        pilots.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        pilots.dedup();
        pilots
    }

    /// The [`Beat`] to broadcast this round: our full live set, who we believe pilots, and
    /// (host only, once [`Membership::set_starting`] has fired) the `start` GO commanding the
    /// round to begin on that set. Built from [`Membership::live_set`]/[`Membership::pilot_set`]
    /// so what we advertise is exactly what we'd freeze; `start` rides the same beat so the GO
    /// and the roster it commands are inseparable.
    pub fn beat(&self) -> Beat {
        Beat {
            members: self.live_set(),
            // Only a host that has clicked Start advertises the GO; everything else sends a
            // plain beat. So a joiner/timer barrier can never put `start` on the wire.
            start: matches!(self.lobby, LobbyMode::Host { started: true }),
            weights_digest: self.local_digest,
            asset_digest: self.local_asset_digest,
            pilots: self.pilot_set(),
        }
    }

    /// Whether every peer in the match (us + all live peers) advertised the SAME, NON-ZERO
    /// policy-weights digest (rl#82, GCR) — the upstream half of the shared-checkpoint guard.
    /// True only when we have a real checkpoint (`local_digest != 0`) AND every live peer's
    /// last DIRECT beat carried that exact digest. A `None` (relay-only, never heard directly)
    /// or any differing/zero digest fails it, so an unsynced brain can never be armed into
    /// lockstep — it stays on the deterministic integer crab. This is an OUTPUT, never a close
    /// gate: a weights mismatch still forms and plays the match (integer crab), it just refuses
    /// to hand the crab to the float NN body. Call after [`Membership::poll`] (which expires
    /// the dead) so the live set is current; pairs with [`crate::net::may_arm_external_crab`].
    pub fn weights_synced(&self) -> bool {
        self.local_digest != 0
            && self
                .peers
                .values()
                .all(|v| v.weights_digest == Some(self.local_digest))
    }

    /// Whether every peer in the match (us + all live peers) advertised the SAME, NON-ZERO
    /// crab-model-asset digest (rl#100, GCR) — the collider half of the shared-asset guard, a
    /// SIBLING of [`Membership::weights_synced`] with the identical rules. True only when we
    /// have a real asset (`local_asset_digest != 0`) AND every live peer's last DIRECT beat
    /// carried that exact digest. A `None` (relay-only) or any differing/zero digest fails it,
    /// so two peers whose crab models (and thus colliders) differ never arm the float NN crab
    /// into lockstep — they stay on the deterministic integer crab. An OUTPUT, never a close
    /// gate: an asset mismatch still forms and plays the match (integer crab), it just refuses
    /// to hand the crab to the float NN body. The NN crab arms only when BOTH this and
    /// `weights_synced` hold (peers agree on brain AND collider asset) — see
    /// [`crate::net::may_arm_external_crab`]. Call after [`Membership::poll`] so the live set
    /// is current.
    pub fn assets_synced(&self) -> bool {
        self.local_asset_digest != 0
            && self
                .peers
                .values()
                .all(|v| v.asset_digest == Some(self.local_asset_digest))
    }

    /// Advance the clock and return the verdict — the SINGLE per-round entry point.
    /// Folds expiry + agreement-timer maintenance + the verdict into one call so the
    /// ordering can't be gotten wrong (an earlier two-call `expire` then `status` split
    /// let a caller read a verdict over stale liveness). Call once per beat interval.
    ///
    /// The agreement predicate is: at least `expect` participants are live AND every live
    /// peer is advertising our exact roster hash (a peer not heard directly has
    /// `advertised == None ≠ Some(my_hash)`, so it blocks). We track when that predicate
    /// became true (`agreed_since`) and reset it the instant it goes false — so closing
    /// requires it to have held *continuously* for [`STABLE_FOR`], not merely to be true
    /// at this instant. That continuity is the safety guarantee (see the type docs).
    pub fn poll(&mut self, now: Instant) -> Status {
        // Expire peers silent past MEMBER_TIMEOUT and tombstone them so a relay can't
        // immediately re-admit. Also prune tombstones older than the quarantine window.
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

        // Evaluate the agreement predicate over the current live set AND pilot set. Folding
        // the pilot roster into the token is what makes a divergent view of who-flies block
        // the close (and rewind the timer on a mid-formation flip), so peers can only freeze
        // ONE shared pilot set — see [`agreement_token`].
        let live = self.live_set();
        let pilots = self.pilot_set();
        let my_hash = agreement_token(&live, &pilots);
        let enough = live.len() >= self.expect;
        let unanimous = self.peers.values().all(|v| v.advertised == Some(my_hash));
        let agreed_now = enough && unanimous;

        // The host's GO landing on a roster we agree with: a peer whose latest direct beat
        // both commanded start AND advertised our exact current hash. Recomputed here (never
        // latched) so a GO seen while sets disagreed can't close us, and a later set change
        // re-gates the GO on the new hash. Only the host ever sets `start`, so in practice
        // this is "the host commanded the start on the roster we're holding".
        self.host_go_on_my_roster = self
            .peers
            .values()
            .any(|v| v.started && v.advertised == Some(my_hash));

        // Maintain the continuous-agreement timer, KEYED ON the agreed hash: (re)start it
        // when the predicate holds on a hash different from what the timer tracks; clear
        // it the instant the predicate fails. So the elapsed time always measures how long
        // we've agreed on THIS exact set — a set change (which shifts `my_hash`) rewinds
        // it even in the degenerate all-peers-flip-at-once case.
        self.agreed_since = match (agreed_now, self.agreed_since) {
            (true, Some((t, h))) if h == my_hash => Some((t, h)), // same set, keep holding
            (true, _) => Some((now, my_hash)),                    // newly agreed (or new set)
            (false, _) => None,                                   // predicate broke → reset
        };

        // EVERY mode requires the agreement predicate to have held CONTINUOUSLY for
        // [`STABLE_FOR`] before closing — that settle is what absorbs a staggered/late join
        // (a peer growing its set shifts `my_hash`, breaks unanimity, and rewinds the timer),
        // so it's load-bearing for the lobby modes too, NOT just the timer one. The mode only
        // adds an EXTRA gate on top of that settle (rl#58):
        // - Host: also requires the Start click (`started`). The settle still runs, so a peer
        //   that joins right as the host clicks can't be frozen out mid-join — the host waits
        //   the dust out exactly like the timer path, then closes on the click.
        // - Joiner: also requires the host's GO on its matching roster.
        // - Off (default): no extra gate — the settle alone closes, as before rl#58.
        // Every mode freezes the IDENTICAL `live` set; only the extra gate (and, for the lobby
        // modes, the lack of a [`JOIN_WINDOW`] fail — a lobby is open-ended, the user cancels)
        // differs, so determinism is untouched.
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
            Status::Agreed {
                roster: live,
                pilots,
            }
        } else if timed_out {
            Status::Failed
        } else {
            Status::Forming { live: live.len() }
        }
    }
}

/// Encode a [`Beat`] for the wire:
/// `[start:u8][count:u16 LE][weights_digest:u64 LE][asset_digest:u64 LE][id:32]*count[pilot_count:u16 LE][pilot_id:32]*pilot_count`,
/// both id lists sorted+deduped. Fixed 32-byte ids (an [`EndpointId`] is a 32-byte public
/// key) so there is no per-id length. The leading `start` byte is the host's GO flag (rl#58);
/// `weights_digest` (rl#82) is the sender's checkpoint digest and `asset_digest` (rl#100) its
/// crab-model digest (NEITHER folded into [`roster_hash`] — capabilities, not membership); the
/// trailing `pilots` list is who the sender believes flies, which IS folded in. Both counts are
/// bounded on decode ([`MAX_BEAT_MEMBERS`], and pilots ≤ members) so a hostile/garbled frame
/// can't trigger a huge allocation.
pub fn encode_beat(beat: &Beat) -> Vec<u8> {
    let canon = |ids: &[EndpointId]| -> Vec<EndpointId> {
        let mut v = ids.to_vec();
        v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        v.dedup();
        v
    };
    let members = canon(&beat.members);
    let pilots = canon(&beat.pilots);
    let mut out = Vec::with_capacity(21 + 32 * (members.len() + pilots.len()));
    out.push(beat.start as u8);
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&beat.weights_digest.to_le_bytes());
    out.extend_from_slice(&beat.asset_digest.to_le_bytes());
    for id in &members {
        out.extend_from_slice(id.as_bytes());
    }
    out.extend_from_slice(&(pilots.len() as u16).to_le_bytes());
    for id in &pilots {
        out.extend_from_slice(id.as_bytes());
    }
    out
}

/// Upper bound on members in a decoded [`Beat`], so a bad length can't allocate without
/// bound. Couch co-op is a handful of peers; [`PlayerId`] is a `u8` so 256 is the
/// absolute ceiling the rest of the stack accepts anyway.
const MAX_BEAT_MEMBERS: usize = 256;

/// Decode a [`Beat`] produced by [`encode_beat`]. Rejects a truncated body, a member count
/// past [`MAX_BEAT_MEMBERS`], or a pilot count exceeding the member count (a malformed/hostile
/// frame) rather than panicking or over-allocating.
pub fn decode_beat(body: &[u8]) -> Result<Beat> {
    anyhow::ensure!(
        body.len() >= 19,
        "barrier frame too short for start+count+weights+asset digests"
    );
    let start = body[0] != 0;
    let count = u16::from_le_bytes([body[1], body[2]]) as usize;
    anyhow::ensure!(
        count <= MAX_BEAT_MEMBERS,
        "barrier frame claims {count} members (> {MAX_BEAT_MEMBERS})"
    );
    let weights_digest = u64::from_le_bytes(body[3..11].try_into().expect("8-byte slice"));
    let asset_digest = u64::from_le_bytes(body[11..19].try_into().expect("8-byte slice"));
    let members_end = 19 + 32 * count;
    // Room for the member ids AND the trailing 2-byte pilot count.
    anyhow::ensure!(
        body.len() >= members_end + 2,
        "barrier frame truncated before the pilot count (len {} < {})",
        body.len(),
        members_end + 2
    );
    let read_ids = |start_off: usize, n: usize| -> Result<Vec<EndpointId>> {
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let off = start_off + 32 * i;
            let bytes: [u8; 32] = body[off..off + 32].try_into().expect("32-byte slice");
            let id = EndpointId::from_bytes(&bytes)
                .map_err(|e| anyhow::anyhow!("bad endpoint id in barrier frame: {e}"))?;
            ids.push(id);
        }
        Ok(ids)
    };
    let members = read_ids(19, count)?;
    let pilot_count = u16::from_le_bytes([body[members_end], body[members_end + 1]]) as usize;
    // Pilots are a subset of members, so a count above `count` is malformed (and bounds the
    // allocation without a separate cap).
    anyhow::ensure!(
        pilot_count <= count,
        "barrier frame claims {pilot_count} pilots (> {count} members)"
    );
    let need = members_end + 2 + 32 * pilot_count;
    anyhow::ensure!(
        body.len() == need,
        "barrier frame length {} != expected {need} for {count} members + {pilot_count} pilots",
        body.len()
    );
    let pilots = read_ids(members_end + 2, pilot_count)?;
    // Pilots are a subset of members by construction. Enforce it at the trust boundary so the
    // "pilots ⊆ members" invariant the rest of the code (and three doc comments) rely on is
    // real, not asserted-by-comment. An out-of-roster pilot can't itself desync — the agreement
    // token just won't match honest peers — but rejecting it keeps a malformed frame from ever
    // entering the state machine.
    anyhow::ensure!(
        pilots.iter().all(|p| members.contains(p)),
        "barrier frame lists a pilot absent from its member set"
    );
    Ok(Beat {
        members,
        start,
        weights_digest,
        asset_digest,
        pilots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic distinct endpoint ids for tests. A raw byte pattern isn't a valid
    /// ed25519 public key, so derive each from a distinct secret key seed (`from_bytes`
    /// of a secret is infallible). Distinct seeds → distinct, valid public keys, which
    /// is all the membership logic cares about (their byte order is arbitrary but
    /// stable, which is exactly the real-world property).
    fn eid(i: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[i; 32]).public()
    }

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    /// A plain (non-start) [`Beat`] for the given members — the default heartbeat every
    /// peer sends until a host commands the start. Keeps the membership tests reading about
    /// rosters, not the rl#58 GO flag they don't exercise.
    fn bt(members: Vec<EndpointId>) -> Beat {
        Beat {
            members,
            start: false,
            weights_digest: 0,
            asset_digest: 0,
            pilots: Vec::new(),
        }
    }

    /// A heartbeat declaring a pilot roster, for the networked-planes tests.
    fn bt_p(members: Vec<EndpointId>, pilots: Vec<EndpointId>) -> Beat {
        Beat {
            members,
            start: false,
            weights_digest: 0,
            asset_digest: 0,
            pilots,
        }
    }

    /// The expected canonical (sorted) form of a set, for comparing against a roster
    /// the code returns. Real public keys sort by their key bytes in an order unrelated
    /// to the seed `i`, so tests must compare against this, not a hand-ordered literal.
    fn sorted(ids: &[EndpointId]) -> Vec<EndpointId> {
        let mut v = ids.to_vec();
        v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        v
    }

    #[test]
    fn beat_wire_roundtrips_and_is_order_independent() {
        let a = bt(vec![eid(3), eid(1), eid(2)]);
        let b = bt(vec![eid(1), eid(2), eid(3)]);
        // Same set in any order → identical bytes and identical hash (the agreement
        // token must not depend on insertion order).
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
        // Valid header (start + count=5 + weights + asset digests) but no member bodies.
        let mut truncated = vec![0u8];
        truncated.extend_from_slice(&5u16.to_le_bytes());
        truncated.extend_from_slice(&0u64.to_le_bytes()); // weights digest
        truncated.extend_from_slice(&0u64.to_le_bytes()); // asset digest
        assert!(decode_beat(&truncated).is_err(), "truncated body");
        // count past the cap (start byte + bogus count + both digests + filler).
        let mut huge = vec![0u8];
        huge.extend_from_slice(&(300u16).to_le_bytes());
        huge.extend_from_slice(&0u64.to_le_bytes()); // weights digest
        huge.extend_from_slice(&0u64.to_le_bytes()); // asset digest
        huge.resize(19 + 32 * 300, 0);
        assert!(decode_beat(&huge).is_err(), "over-large count rejected");
    }

    /// The host's start GO survives the wire as a distinct field and does NOT change the
    /// roster hash (rl#58): a `start` beat and a plain beat over the SAME members hash
    /// identically (so a host commanding the start can't accidentally fork the agreed set),
    /// but the decoded `start` flag faithfully reflects which was sent.
    #[test]
    fn start_flag_roundtrips_without_perturbing_the_roster_hash() {
        let members = vec![eid(1), eid(2)];
        let plain = bt(members.clone());
        let go = Beat {
            members,
            start: true,
            weights_digest: 0,
            asset_digest: 0,
            pilots: Vec::new(),
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

    // ── Weights-digest handshake (rl#82, GCR — the shared-checkpoint guard) ──────────

    /// A heartbeat carrying a weights digest, for the synced-brain tests.
    fn bt_d(members: Vec<EndpointId>, weights_digest: u64) -> Beat {
        Beat {
            members,
            start: false,
            weights_digest,
            asset_digest: 0,
            pilots: Vec::new(),
        }
    }

    /// A heartbeat carrying a crab-asset digest (rl#100), for the synced-asset tests. Sibling
    /// of [`bt_d`]; weights digest left `0` so these isolate the asset-handshake path.
    fn bt_ad(members: Vec<EndpointId>, asset_digest: u64) -> Beat {
        Beat {
            members,
            start: false,
            weights_digest: 0,
            asset_digest,
            pilots: Vec::new(),
        }
    }

    #[test]
    fn weights_digest_roundtrips_on_the_wire() {
        // The digest survives encode→decode and (like `start`) does NOT perturb the roster
        // hash: two beats with the same members but different brains still agree on the set.
        let members = vec![eid(1), eid(2)];
        let a = bt_d(members.clone(), 0xDEAD_BEEF_F00D_1234);
        let b = bt_d(members, 0x1111_2222_3333_4444);
        assert_eq!(
            a.roster_hash(),
            b.roster_hash(),
            "the weights digest must not be part of the roster hash"
        );
        assert_eq!(
            decode_beat(&encode_beat(&a)).unwrap().weights_digest,
            0xDEAD_BEEF_F00D_1234
        );
    }

    #[test]
    fn weights_synced_only_when_all_peers_share_one_nonzero_digest() {
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        const BRAIN: u64 = 0xABCD_0000_1234_5678;

        // Matched, non-zero brains on both peers → synced.
        let mut a = Membership::new(ida, 2, t0).with_weights_digest(BRAIN);
        a.on_beat(idb, &bt_d(vec![ida, idb], BRAIN), t0);
        a.poll(t0);
        assert!(a.weights_synced(), "equal non-zero digests must be synced");

        // A peer on a DIFFERENT brain → not synced (the mismatch case the guard exists for).
        let mut b = Membership::new(ida, 2, t0).with_weights_digest(BRAIN);
        b.on_beat(idb, &bt_d(vec![ida, idb], BRAIN ^ 0xFF), t0);
        b.poll(t0);
        assert!(!b.weights_synced(), "a differing peer digest must not be synced");

        // A peer advertising a ZERO digest (no checkpoint) → not synced.
        let mut c = Membership::new(ida, 2, t0).with_weights_digest(BRAIN);
        c.on_beat(idb, &bt_d(vec![ida, idb], 0), t0);
        c.poll(t0);
        assert!(!c.weights_synced(), "a zero peer digest must not be synced");

        // OUR OWN digest zero (no checkpoint locally) → never synced, even if a peer has one.
        let mut d = Membership::new(ida, 2, t0); // local_digest defaults to 0
        d.on_beat(idb, &bt_d(vec![ida, idb], BRAIN), t0);
        d.poll(t0);
        assert!(!d.weights_synced(), "a zero local digest is never synced");
    }

    #[test]
    fn relay_only_peer_blocks_weights_synced() {
        // A peer we know only via another's relay (never a DIRECT beat) has `weights_digest:
        // None`, which can't equal our digest — so it blocks synced until it speaks for itself,
        // exactly like the roster-agreement `advertised: None` rule. Prevents arming the NN
        // crab against a peer whose brain we've never actually heard.
        let t0 = Instant::now();
        let (ida, idb, idc) = (eid(1), eid(2), eid(3));
        const BRAIN: u64 = 0x9999_8888_7777_6666;
        let mut a = Membership::new(ida, 3, t0).with_weights_digest(BRAIN);
        // B beats directly with the matching brain AND relays C's existence (no C digest).
        a.on_beat(idb, &bt_d(vec![ida, idb, idc], BRAIN), t0);
        a.poll(t0);
        assert!(
            !a.weights_synced(),
            "a relay-only peer (digest None) must block synced"
        );
        // Now C beats directly with the matching brain → all heard, all matched → synced.
        a.on_beat(idc, &bt_d(vec![ida, idb, idc], BRAIN), t0);
        a.poll(t0);
        assert!(a.weights_synced(), "once every peer is heard with the same brain, synced");
    }

    // ── Asset-digest handshake (rl#100, GCR — the shared-collider-asset guard) ────────
    // A SIBLING of the weights handshake above: the crab-model digest rides its own Beat
    // field, is never folded into the roster hash, and gates arming the float NN crab on every
    // peer agreeing — so two peers whose crab colliders differ stay on the integer crab.

    #[test]
    fn asset_digest_roundtrips_on_the_wire() {
        // The digest survives encode→decode and (like `weights_digest`) does NOT perturb the
        // roster hash, NOR collide with the weights digest field — a beat carrying an asset
        // digest but no weights digest roundtrips both fields independently.
        let members = vec![eid(1), eid(2)];
        let a = bt_ad(members.clone(), 0xC0FF_EE00_1234_5678);
        let b = bt_ad(members, 0x9876_5432_10AB_CDEF);
        assert_eq!(
            a.roster_hash(),
            b.roster_hash(),
            "the asset digest must not be part of the roster hash"
        );
        let decoded = decode_beat(&encode_beat(&a)).unwrap();
        assert_eq!(decoded.asset_digest, 0xC0FF_EE00_1234_5678);
        assert_eq!(decoded.weights_digest, 0, "asset digest must not bleed into weights");
    }

    #[test]
    fn weights_and_asset_digests_are_independent_on_the_wire() {
        // Both digests present and DISTINCT must roundtrip to their own values — proves the two
        // sibling u64 fields don't alias (a layout bug in the wire format would swap them).
        let members = vec![eid(1), eid(2)];
        let beat = Beat {
            members,
            start: false,
            weights_digest: 0x1111_2222_3333_4444,
            asset_digest: 0x5555_6666_7777_8888,
            pilots: Vec::new(),
        };
        let decoded = decode_beat(&encode_beat(&beat)).unwrap();
        assert_eq!(decoded.weights_digest, 0x1111_2222_3333_4444);
        assert_eq!(decoded.asset_digest, 0x5555_6666_7777_8888);
    }

    #[test]
    fn assets_synced_only_when_all_peers_share_one_nonzero_digest() {
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        const ASSET: u64 = 0x5A11_0000_C0DE_4321;

        // Matched, non-zero assets on both peers → synced.
        let mut a = Membership::new(ida, 2, t0).with_asset_digest(ASSET);
        a.on_beat(idb, &bt_ad(vec![ida, idb], ASSET), t0);
        a.poll(t0);
        assert!(a.assets_synced(), "equal non-zero asset digests must be synced");

        // A peer on a DIFFERENT crab model → not synced (the desync the guard exists for).
        let mut b = Membership::new(ida, 2, t0).with_asset_digest(ASSET);
        b.on_beat(idb, &bt_ad(vec![ida, idb], ASSET ^ 0xFF), t0);
        b.poll(t0);
        assert!(!b.assets_synced(), "a differing peer asset digest must not be synced");

        // A peer advertising a ZERO digest (no resolvable model) → not synced.
        let mut c = Membership::new(ida, 2, t0).with_asset_digest(ASSET);
        c.on_beat(idb, &bt_ad(vec![ida, idb], 0), t0);
        c.poll(t0);
        assert!(!c.assets_synced(), "a zero peer asset digest must not be synced");

        // OUR OWN digest zero (no model locally) → never synced, even if a peer has one.
        let mut d = Membership::new(ida, 2, t0); // local_asset_digest defaults to 0
        d.on_beat(idb, &bt_ad(vec![ida, idb], ASSET), t0);
        d.poll(t0);
        assert!(!d.assets_synced(), "a zero local asset digest is never synced");
    }

    #[test]
    fn weights_and_assets_synced_are_independent_gates() {
        // The two guards are orthogonal: matching brains but mismatched crab assets must leave
        // `weights_synced` true yet `assets_synced` false (and the arm site ANDs them, so the
        // NN crab stays integer). Proves a refactor can't collapse the two into one digest.
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        const BRAIN: u64 = 0xB0B0_0000_1234_5678;
        const ASSET: u64 = 0xA55E_0000_8765_4321;
        let mut a = Membership::new(ida, 2, t0)
            .with_weights_digest(BRAIN)
            .with_asset_digest(ASSET);
        // Peer matches the brain but runs a DIFFERENT crab model.
        let mismatched_asset = Beat {
            members: vec![ida, idb],
            start: false,
            weights_digest: BRAIN,
            asset_digest: ASSET ^ 0x1,
            pilots: Vec::new(),
        };
        a.on_beat(idb, &mismatched_asset, t0);
        a.poll(t0);
        assert!(a.weights_synced(), "matching brains → weights synced");
        assert!(
            !a.assets_synced(),
            "a different crab asset must leave assets NOT synced (independent of weights)"
        );
    }

    // ── Pilot-intent handshake (GCR planes — wire-negotiated who flies) ──────────────

    /// A pilot list rides the wire as a distinct field, and (unlike `start`/`weights_digest`)
    /// IS folded into the agreement token: two beats with the same members but different pilots
    /// must hash DIFFERENTLY (else peers could freeze split pilot sets), while a roundtrip
    /// preserves the pilot list. The determinism crux of the whole feature in one assertion.
    #[test]
    fn pilot_list_roundtrips_and_is_folded_into_the_roster_hash() {
        let members = vec![eid(1), eid(2)];
        let no_pilots = bt_p(members.clone(), vec![]);
        let a_flies = bt_p(members.clone(), vec![eid(1)]);
        let b_flies = bt_p(members, vec![eid(2)]);
        // Different pilot rosters over the same members MUST diverge the token.
        assert_ne!(
            no_pilots.roster_hash(),
            a_flies.roster_hash(),
            "adding a pilot must change the agreement token (else split pilot sets could freeze)"
        );
        assert_ne!(
            a_flies.roster_hash(),
            b_flies.roster_hash(),
            "a DIFFERENT pilot must change the token — who flies is part of the agreement"
        );
        // The pilot list survives the wire, sorted+deduped like members.
        let decoded = decode_beat(&encode_beat(&a_flies)).unwrap();
        assert_eq!(decoded.pilots, sorted(&[eid(1)]));
        assert_eq!(decoded.members, sorted(&[eid(1), eid(2)]));
    }

    #[test]
    fn decode_rejects_a_pilot_not_in_the_member_set() {
        // The wire enforces pilots ⊆ members: a frame declaring a pilot absent from its own
        // roster is malformed and must be rejected, so the "pilots ⊆ members" invariant holds
        // at the trust boundary (not just by honest-encoder construction).
        let bogus = bt_p(vec![eid(1), eid(2)], vec![eid(9)]);
        assert!(
            decode_beat(&encode_beat(&bogus)).is_err(),
            "a pilot outside the member set must be rejected on decode"
        );
    }

    #[test]
    fn two_peers_with_one_pilot_freeze_the_identical_pilot_set() {
        // The core determinism guarantee for planes: A declares itself a pilot, B does not;
        // both must converge AND freeze the byte-identical pilot set {A} (not just the same
        // member roster). Mirrors `two_peers_converge_and_freeze_the_identical_set`, now
        // asserting the pilot subset matches too.
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        let mut a = Membership::new(ida, 2, t0).piloting(true);
        let mut b = Membership::new(idb, 2, t0).piloting(false);

        let mut t = 0u64;
        let mut agreed_a: Option<(Vec<EndpointId>, Vec<EndpointId>)> = None;
        let mut agreed_b: Option<(Vec<EndpointId>, Vec<EndpointId>)> = None;
        while t <= 3000 && (agreed_a.is_none() || agreed_b.is_none()) {
            let now = at(t0, t);
            let (ba, bb) = (a.beat(), b.beat());
            a.on_beat(idb, &bb, now);
            b.on_beat(ida, &ba, now);
            if agreed_a.is_none()
                && let Status::Agreed { roster, pilots } = a.poll(now)
            {
                agreed_a = Some((roster, pilots));
            }
            if agreed_b.is_none()
                && let Status::Agreed { roster, pilots } = b.poll(now)
            {
                agreed_b = Some((roster, pilots));
            }
            t += 250;
        }
        let (ra, pa) = agreed_a.expect("A must agree");
        let (rb, pb) = agreed_b.expect("B must agree");
        assert_eq!(ra, rb, "both peers must freeze the identical roster");
        assert_eq!(pa, pb, "both peers MUST freeze the identical pilot set");
        assert_eq!(pa, vec![ida], "only A (which declared RL_VEHICLE=plane) flies");
    }

    #[test]
    fn relay_only_pilot_blocks_the_close_until_heard_directly() {
        // A pilot's intent is learned ONLY from its own direct beat, never a relay. So a peer
        // that knows a pilot P only via another's relay must NOT close: it has P at `is_pilot:
        // None` (so P is absent from its pilot_set) AND `advertised: None` (so P blocks
        // unanimity). If a relay COULD seed P's pilot role, this peer's pilot_set would differ
        // from one that heard P directly — a split freeze. Prove the close is blocked.
        let t0 = Instant::now();
        let (ida, idb, idp) = (eid(1), eid(2), eid(9));
        let mut a = Membership::new(ida, 3, t0);
        // B beats directly (foot) and RELAYS both P's existence and P-as-pilot in its list.
        // A must NOT trust the relayed pilot bit for P — P is still only gossip-admitted.
        let mut t = 0u64;
        let mut closed = false;
        while t <= STABLE_FOR.as_millis() as u64 + 1000 {
            let now = at(t0, t);
            a.on_beat(idb, &bt_p(vec![ida, idb, idp], vec![idp]), now);
            if matches!(a.poll(now), Status::Agreed { .. }) {
                closed = true;
                break;
            }
            // A never lists P as a pilot off a relay — only P's own beat could declare it.
            assert!(
                !a.pilot_set().contains(&idp),
                "a relayed pilot bit must not make P a pilot in our view"
            );
            t += 250;
        }
        assert!(
            !closed,
            "a relay-only member (pilot or not) must block the close until heard directly"
        );
    }

    #[test]
    fn a_pilot_intent_flip_mid_formation_never_splits_the_frozen_set() {
        // The determinism trap the feature exists to defeat: a peer that FLIPS its pilot
        // intent mid-formation must not let peers freeze divergent pilot sets. B flies for a
        // stretch, then stops (or vice-versa); the flip changes B's advertised token, breaks
        // A's unanimity, and rewinds the STABLE_FOR timer — so A can only close AFTER B's
        // intent has settled, and then on B's FINAL intent. Assert: no peer ever freezes a
        // pilot set the other doesn't share, and the final freeze reflects B's settled intent.
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        let mut a = Membership::new(ida, 2, t0).piloting(false);
        // B is modeled by hand-fed beats so we can flip its declared intent mid-stream. A's
        // own beats are real. We drive A and feed it B's (flipping) beats.
        let flip_until = 1200u64; // B "flies" until here, then settles to foot
        let mut froze_a: Option<Vec<EndpointId>> = None;
        let mut last_close_t = 0u64;
        let mut t = 0u64;
        while t <= 6000 && froze_a.is_none() {
            let now = at(t0, t);
            // B's beat: lists {A,B}; declares itself a pilot only while flipping.
            let b_pilots = if t < flip_until { vec![idb] } else { vec![] };
            a.on_beat(idb, &bt_p(vec![ida, idb], b_pilots), now);
            if let Status::Agreed { pilots, .. } = a.poll(now) {
                froze_a = Some(pilots);
                last_close_t = t;
            }
            t += 250;
        }
        let pa = froze_a.expect("A must eventually close once B's intent settles");
        // A closed only AFTER the flip settled (the timer rewinds on every intent change), and
        // froze on B's FINAL intent: foot, so the pilot set is empty.
        assert!(
            last_close_t >= flip_until + STABLE_FOR.as_millis() as u64,
            "A must not close until B's flipped intent has been stable for STABLE_FOR (closed at \
             {last_close_t}ms, flip settled at {flip_until}ms)"
        );
        assert!(
            pa.is_empty(),
            "A must freeze B's SETTLED (foot) intent, not a mid-flip pilot view — got {pa:?}"
        );
    }

    #[test]
    fn lone_peer_with_expect_one_waits_then_agrees_on_itself() {
        // A deliberate solo run (expect=1) must still settle for STABLE_FOR, then agree
        // on just itself. With expect>1 a lone peer would instead time out (next test).
        let t0 = Instant::now();
        let mut m = Membership::new(eid(1), 1, t0);
        // Poll from t=0 (as the real driver does each beat) to start the agreement
        // timer; before STABLE_FOR has elapsed since, still forming.
        assert_eq!(m.poll(t0), Status::Forming { live: 1 });
        assert_eq!(m.poll(at(t0, 500)), Status::Forming { live: 1 });
        // Once the predicate has held continuously for STABLE_FOR: agreed on {self}.
        match m.poll(at(t0, STABLE_FOR.as_millis() as u64 + 1)) {
            Status::Agreed { roster, .. } => assert_eq!(roster, vec![eid(1)]),
            other => panic!("lone peer should agree on itself, got {other:?}"),
        }
    }

    #[test]
    fn lone_peer_expecting_more_times_out_instead_of_freezing_solo() {
        // The premature-lone-close guard: a peer that expects 2 players but only ever
        // sees itself must NOT freeze {self} — it has to FAIL at the join window so the
        // operator relaunches, rather than start a broken 1-player "match". (On a real
        // LAN, discovery can take a second or two; without this floor two peers that are
        // slow to find each other would each freeze solo.)
        let t0 = Instant::now();
        let mut m = Membership::new(eid(1), 2, t0);
        // Never reaches Agreed however long we wait — only itself is live.
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
        // The core guarantee: two peers that hear each other's matching beats freeze
        // the SAME set. Drive both state machines with each other's beats and assert
        // both reach Agreed with the identical roster.
        let t0 = Instant::now();
        let (ida, idb) = (eid(1), eid(2));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        // Exchange beats every 250ms; after each, neither's set changes once both know
        // each other, so the agreement timer runs and they converge on hash({A,B}).
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
        // A and B form first; C joins ~1s late. The late join must perturb A/B's set
        // and reset their settle, so NOBODY freezes {A,B} — all three converge on
        // {A,B,C}. This is the staggered-launch failure mode, defeated.
        let t0 = Instant::now();
        let (ida, idb, idc) = (eid(1), eid(2), eid(3));
        let mut a = Membership::new(ida, 3, t0);
        let mut b = Membership::new(idb, 3, t0);
        let mut c = Membership::new(idc, 3, t0); // exists but silent until join_at

        let join_at = 1000u64;
        let mut froze: BTreeMap<u8, Vec<EndpointId>> = BTreeMap::new();
        let mut t = 0u64;
        while t <= 6000 && froze.len() < 3 {
            let now = at(t0, t);
            // Everyone beats; C only participates once it has "launched".
            let (ba, bb, bc) = (a.beat(), b.beat(), c.beat());
            // A<->B always exchange.
            a.on_beat(idb, &bb, now);
            b.on_beat(ida, &ba, now);
            if t >= join_at {
                // C now reachable by all: full mesh exchange.
                a.on_beat(idc, &bc, now);
                b.on_beat(idc, &bc, now);
                c.on_beat(ida, &ba, now);
                c.on_beat(idb, &bb, now);
            }
            for (id, m) in [(1u8, &mut a), (2, &mut b), (3, &mut c)] {
                // Always poll (drives the machine); record only the first Agreed per peer.
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
        // A stale endpoint P beats once (mDNS pulled it in) then goes silent (it had
        // crashed). It must expire and NOT be in the frozen set — the phantom-endpoint
        // failure mode. A and B (the real players) freeze {A,B}.
        let t0 = Instant::now();
        let (ida, idb, idp) = (eid(1), eid(2), eid(9));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        // One stale DIRECT beat from P at t=0, then never again (its process crashed).
        let phantom_beat = bt(vec![idp]);
        a.on_beat(idp, &phantom_beat, t0);
        b.on_beat(idp, &phantom_beat, t0);

        let mut froze_a = None;
        let mut froze_b = None;
        let mut t = 0u64;
        while t <= 8000 && (froze_a.is_none() || froze_b.is_none()) {
            let now = at(t0, t);
            // A and B keep relaying P in their beats while it's still live in their view —
            // the relay must NOT keep the dead P alive (only its own direct beat would).
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
        // The determinism-critical case: while A sees {A,B} and B (transiently) sees
        // {A,B,C}, NEITHER may freeze — their hashes differ, so unanimity fails on both
        // sides. We assert that during the divergence window NO peer reports Agreed
        // with a set the other doesn't share. Here C is real-but-only-known-to-B for a
        // stretch; once B relays C to A they converge (covered above). The point of
        // THIS test is the negative: no freeze on a divergent hash.
        let t0 = Instant::now();
        let (ida, idb, idc) = (eid(1), eid(2), eid(3));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        // B hears C directly (so C is live in B with a real advertised hash) but A has
        // NOT — A's beats won't list C, and we deliberately feed A a doctored {A,B}-only
        // view of B to HOLD the views divergent and check neither closes wrongly.
        let cbeat = bt(vec![idc]);
        b.on_beat(idc, &cbeat, t0);

        let mut t = 0u64;
        while t <= 2500 {
            let now = at(t0, t);
            let ba = a.beat(); // lists {A,B} only (A never heard C)
            // Keep C alive in B's view (direct beat) so B's set stays {A,B,C}.
            b.on_beat(idc, &cbeat, now);
            // Feed A a stale {A,B}-only view of B (simulating C-bearing frames not yet
            // arriving). B hears A's real beat.
            a.on_beat(idb, &bt(vec![ida, idb]), now);
            b.on_beat(ida, &ba, now);
            let sa = a.poll(now);
            let sb = b.poll(now);
            // Neither may freeze a set the other doesn't hold: if A says Agreed it must
            // be {A,B} AND B must NOT simultaneously be Agreed on {A,B,C}. (The doctored
            // {A,B}-only beat fed to A is the artificial part — in the live protocol B's
            // real beats carry its full {A,B,C}, so A would hear C and not close early.
            // This isolates the safety check: a peer never closes on a hash a live member
            // isn't echoing. B never closes here: C advertises hash({C}) ≠ B's
            // hash({A,B,C}), so B is never unanimous.)
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
        // The CRITICAL fix (rl#44 review): the close needs CONTINUOUS agreement for
        // STABLE_FOR, not a single instant of it. A peer B whose view keeps oscillating —
        // agreeing on {A,B} one beat, then pulling in an extra member the next — must
        // NEVER let A close, because the agreement timer rewinds whenever the predicate
        // breaks. Without the held-timer fix, a one-poll sample of agreement at a
        // stale-cache moment could close a set the peer is abandoning.
        //
        // Mechanism note (honest): B's `disagree` beat both flips B's advertised hash AND
        // gossip-ADMITS `other` into A (any beat's members are relay-admitted). So on the
        // `disagree` beats A's predicate breaks two ways at once — B's hash differs and
        // `other` is an unheard member (advertised=None). EITHER alone rewinds the timer;
        // the test asserts the outcome (never Agreed across the whole window), which is the
        // continuous-agreement guarantee regardless of which condition fires.
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
        // The LOAD-BEARING staggered case the reviewers flagged as untested: with
        // expect=2, A and B could LEGITIMATELY freeze {A,B} (the count floor does NOT save
        // us here — 2 >= 2). C joins WITHIN STABLE_FOR of A+B's agreement, so the
        // timer-rewind-on-set-growth is the ONLY thing that stops a premature {A,B} freeze.
        // Assert nobody ever emits Agreed{A,B} and all three converge on {A,B,C} — the
        // determinism invariant under a within-settle staggered join.
        let t0 = Instant::now();
        let (ida, idb, idc) = (eid(1), eid(2), eid(3));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);
        let mut c = Membership::new(idc, 2, t0);

        let join_at = 1000u64; // < STABLE_FOR (1500): A+B are mid-settle on {A,B}
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
                // Always poll (drives the machine); check+record only the first Agreed per peer.
                if let Status::Agreed { roster, .. } = m.poll(now)
                    && !froze.contains_key(&id)
                {
                    // The crux: NO peer may ever freeze the partial {A,B}.
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
        // Sanity: the scenario really did put C in within the settle window (so the
        // count floor alone could not have prevented an {A,B} freeze).
        assert!(join_at < STABLE_FOR.as_millis() as u64);
    }

    #[test]
    fn relayed_phantom_does_not_ping_pong_back_into_the_set() {
        // The phantom ping-pong (rl#44 review): a once-seen dead endpoint P that two
        // peers expire at SLIGHTLY DIFFERENT times must not be relayed back and forth and
        // re-admitted forever (each re-admission would rewind the agreement timer → the
        // barrier would never settle and spuriously FAIL). The tombstone-on-expiry +
        // no-relay-resurrection rule must let A and B settle on {A,B} despite B still
        // relaying P for a beat after A has expired it.
        let t0 = Instant::now();
        let (ida, idb, idp) = (eid(1), eid(2), eid(9));
        let mut a = Membership::new(ida, 2, t0);
        let mut b = Membership::new(idb, 2, t0);

        // P beats directly to A at t=0 and to B 400ms LATER, then is silent forever. So
        // A expires P (~t=3000) a beat or two BEFORE B does (~t=3400). In that gap B is
        // still relaying P to A — the resurrection window. The tombstone A set on expiry
        // must make A REFUSE to re-admit the relayed P (only P's own direct beat could),
        // so the dead P can't ping-pong back and the set settles to {A,B}.
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
            // Each relays whatever it currently lists (P included until locally expired).
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
        // A peer that keeps flapping (a member appears and expires forever) must hit
        // JOIN_WINDOW and FAIL, not freeze a guessed set. We never let the set stabilize:
        // a new transient member id every beat keeps resetting the agreement timer so
        // STABLE_FOR is never satisfied. (expect=1 so the failure is purely the churn, not
        // a too-few-players timeout.)
        let t0 = Instant::now();
        let mut m = Membership::new(eid(1), 1, t0);
        let mut t = 0u64;
        let mut saw_failed = false;
        while t <= JOIN_WINDOW.as_millis() as u64 + 2000 {
            let now = at(t0, t);
            // A brand-new transient peer each step (id derived from t), heard once
            // directly. It expires after MEMBER_TIMEOUT, but a fresh one replaces it, so
            // the set never holds still. (Tombstones don't block these — each id is new.)
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
        // rl#58 determinism gate. A host (H) and a joiner (J) form a host-triggered barrier.
        // Neither may close on agreement ALONE — the lobby waits for H's Start click (the
        // STABLE_FOR settle still runs underneath, absorbing late joins, but it's not enough
        // to close in lobby mode) — and when H clicks, BOTH must freeze the byte-identical
        // {H,J}. The host-driven analogue of `two_peers_converge_and_freeze_the_identical_set`:
        // the roster guarantee is unchanged, only an EXTRA close gate (the GO) is layered on.
        let t0 = Instant::now();
        let (idh, idj) = (eid(1), eid(2));
        let mut h = Membership::host_triggered(Role::Host, idh, 2, t0);
        let mut j = Membership::host_triggered(Role::Joiner, idj, 2, t0);

        // Exchange beats well past STABLE_FOR with NO host click: BOTH must keep lobbying —
        // the settle elapses but neither lobby gate (Start / GO) is satisfied, so agreement
        // alone never closes a lobby.
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

        // Host clicks Start. Now both close — on the SAME set — within a couple beats.
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
        // The determinism SAFETY gate (rl#58): a host GO closes a joiner ONLY when the GO's
        // roster hash equals the joiner's own. Here the host commands start on {H} (it hasn't
        // yet heard J), while J already sees {H,J}. J must NOT close — its set {H,J} ≠ the
        // GO's {H} — so the two can never freeze divergent rosters off a premature GO. (In
        // the live protocol the host would hear J and re-GO on {H,J}; this isolates the
        // negative: a GO on the wrong set is inert.)
        let t0 = Instant::now();
        let (idh, idj) = (eid(1), eid(2));
        let mut j = Membership::host_triggered(Role::Joiner, idj, 2, t0);

        // J has heard H directly (so {H,J} is live for J) and H is "started", but H's beat
        // carries only {H} (it hasn't admitted J yet) with the GO set.
        let host_go_on_partial = Beat {
            members: vec![idh],
            start: true,
            weights_digest: 0,
            asset_digest: 0,
            pilots: Vec::new(),
        };
        let mut t = 0u64;
        let mut closed = false;
        while t <= STABLE_FOR.as_millis() as u64 + 2000 {
            let now = at(t0, t);
            // Keep H alive in J's view with the partial GO beat each round.
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
        // The protocol guarantee that the [`LobbyMode`] type now ENFORCES (rl#58): only a
        // host can put `start` on the wire. A timer barrier and a JOINER both stay silent on
        // the GO even after `set_starting` — so a non-menu caller's wire is unchanged, and a
        // joiner can never command a start it isn't entitled to (the determinism-adjacent
        // "joiner can't GO" rule, in the type rather than in external channel wiring).
        let t0 = Instant::now();
        for mut m in [
            Membership::new(eid(1), 1, t0), // timer barrier
            Membership::host_triggered(Role::Joiner, eid(1), 2, t0), // a joiner
        ] {
            m.set_starting();
            assert!(
                !m.beat().start,
                "only a Role::Host may advertise the start GO"
            );
        }
        // A host, by contrast, advertises the GO once it has clicked Start.
        let mut host = Membership::host_triggered(Role::Host, eid(1), 2, t0);
        assert!(!host.beat().start, "a host is silent before clicking Start");
        host.set_starting();
        assert!(host.beat().start, "a host advertises the GO after Start");
    }

    #[test]
    fn host_clicking_start_during_a_late_join_does_not_freeze_a_partial_set() {
        // The rl#58 race a reviewer flagged: the host has clicked Start and agrees with J on
        // {H,J}, but C joins in that same window so J's set grows to {H,J,C}. The host must
        // NOT freeze {H,J} (which would leave it waiting forever on J's inputs while J is on a
        // different roster). The per-mode [`STABLE_FOR`] settle is what saves it: C's arrival
        // breaks the host's unanimity (J now advertises hash({H,J,C})) and rewinds the timer,
        // so `held` goes false and the host keeps forming until all three converge. This is the
        // same staggered-join guarantee `late_joiner_within_settle…` proves for the timer path,
        // now confirmed for a host that has ALREADY clicked Start.
        let t0 = Instant::now();
        let (idh, idj, idc) = (eid(1), eid(2), eid(3));
        let mut h = Membership::host_triggered(Role::Host, idh, 3, t0);
        let mut j = Membership::host_triggered(Role::Joiner, idj, 3, t0);
        let mut c = Membership::host_triggered(Role::Joiner, idc, 3, t0);
        h.set_starting(); // the host has committed to start from the outset

        let join_at = 1000u64; // C joins WITHIN STABLE_FOR of H+J agreeing on {H,J}
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
                // Always poll (drives the machine); record the first Agreed per peer.
                if let Status::Agreed { roster, .. } = m.poll(now)
                    && !froze.contains_key(&id)
                {
                    // The crux: the host (or anyone) must only ever freeze the FULL {H,J,C}.
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
