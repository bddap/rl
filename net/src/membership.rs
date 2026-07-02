//! LAN match-formation barrier: agree on ONE participant set before anyone ticks.
//!
//! The cold-start's job is to turn "some processes launched on a LAN" into a single
//! agreed match. The hard requirement is the determinism invariant of
//! [`crate::lockstep`]: every peer MUST freeze the byte-identical sorted
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
//! The frozen set is then handed to [`crate::net_loop`], which assigns
//! [`PlayerId`]s by sorted endpoint id exactly as before — but now over a set proven
//! identical on every peer, not merely hoped to be.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use iroh::EndpointId;

use crab_world::fnv::Fnv;

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
    /// [`Membership::sync_verdict`] for what it gates (the HOST self-gate under host-auth).
    /// Deliberately NOT folded into [`Beat::roster_hash`]: it is a capability advertisement,
    /// not membership, so two peers with the identical roster still freeze the same set
    /// regardless of their brains (an unverified host refuses to arm the NN crab, rl#114; it
    /// never breaks match formation).
    pub weights_digest: u64,
    /// The sender's crab-MODEL-asset digest (rl#100, GCR), `0` for no resolvable model — the
    /// giant crab's rapier colliders are derived from this asset
    /// ([`crab_world::bot::meshfit::crab_asset_digest`]), so two peers with different crab models
    /// build different colliders and silently desync. A SIBLING of `weights_digest`, handled
    /// identically: NOT folded into [`Beat::roster_hash`] (a capability advertisement, not
    /// membership — a mismatch refuses to arm the NN crab, never breaks formation), self-declared,
    /// never relayed. See [`Membership::sync_verdict`] for what it gates.
    pub asset_digest: u64,
}

impl Beat {
    /// The agreement token: a hash of the member set. Two peers advertising the same token
    /// have the identical participant set (the freeze precondition). Computed from the sorted,
    /// deduped id bytes so it is independent of insertion order — the same members always hash
    /// the same on every peer. FNV-1a over the raw 32-byte ids: this is an agreement check
    /// among cooperating LAN peers, not an adversarial digest, so a fast non-cryptographic hash
    /// is the right tool (a collision would at worst admit a wrong freeze, which then surfaces
    /// as a roster/id-assignment disagreement the moment the round forms).
    pub fn roster_hash(&self) -> u64 {
        roster_hash(&self.members)
    }
}

/// Hash a participant set for the agreement check. Canonicalizes (sort + dedup) first
/// so the result is a pure function of the SET, not the order ids were added — both
/// the advertiser and the checker must get the same number for the same membership.
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
    /// [`Membership::beat`] and the basis of [`Membership::sync_verdict`]. `0` by default
    /// ([`Membership::with_weights_digest`] sets it), so a caller that never sets it behaves
    /// exactly as pre-rl#82 — the verdict's `weights` is then always false.
    local_digest: u64,
    /// OUR crab-model-asset digest (rl#100, GCR), `0` for no resolvable model. Advertised in
    /// every [`Membership::beat`] and the basis of [`Membership::sync_verdict`]. `0` by default
    /// ([`Membership::with_asset_digest`] sets it), so a caller that never sets it behaves
    /// exactly as pre-rl#100 — the verdict's `assets` is then always false. Sibling of `local_digest`.
    local_asset_digest: u64,
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
    /// or `None` if heard only via relay. Read only when this peer is the HOST (the
    /// [`Membership::sync_verdict`] self-gate); like `advertised`, `None` counts as
    /// unverified, so a host not yet heard directly blocks the gate until it speaks for
    /// itself.
    weights_digest: Option<u64>,
    /// The crab-model-asset digest this peer last advertised in its OWN direct beat (rl#100),
    /// or `None` if heard only via relay. Sibling of `weights_digest`: like it, a `None` can
    /// never equal our non-zero digest, so a peer not yet heard directly blocks
    /// [`Membership::sync_verdict`] until it speaks for itself.
    asset_digest: Option<u64>,
}

/// The barrier's verdict on each [`Membership::poll`]: keep waiting, freeze this exact
/// set, or give up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Not yet agreed; keep beating and merging. Carries the current live set size for
    /// logging/UX (a lobby can show "2 players, waiting…").
    Forming { live: usize },
    /// The agreement predicate held continuously for [`STABLE_FOR`]. Freeze EXACTLY
    /// these ids (sorted, includes us). Every peer that reaches this state computed the
    /// identical roster — that is the guarantee (it is the agreement token, so it can't diverge).
    Agreed { roster: Vec<EndpointId> },
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
        }
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

    /// The [`Beat`] to broadcast this round: our full live set and (host only, once
    /// [`Membership::set_starting`] has fired) the `start` GO commanding the round to begin on
    /// that set. Built from [`Membership::live_set`] so what we advertise is exactly what we'd
    /// freeze; `start` rides the same beat so the GO and the roster it commands are inseparable.
    pub fn beat(&self) -> Beat {
        Beat {
            members: self.live_set(),
            // Only a host that has clicked Start advertises the GO; everything else sends a
            // plain beat. So a joiner/timer barrier can never put `start` on the wire.
            start: matches!(self.lobby, LobbyMode::Host { started: true }),
            weights_digest: self.local_digest,
            asset_digest: self.local_asset_digest,
        }
    }

    /// The shared-asset guard's verdict (rl#82 weights + rl#100 crab asset, GCR), the ONE
    /// value the formation carries to the arm sites ([`crate::SyncVerdict`]).
    ///
    /// - `weights` — the HOST self-gate, host-auth's relocation of the old peer-symmetric
    ///   equality (the design doc's end state; [[real-sally-definition]]): true iff the peer
    ///   that will run the authoritative server — the lowest live endpoint id, PlayerId(0) by
    ///   [`crate::formation::assign_player_ids`]'s sorted assignment — advertised a NON-ZERO
    ///   policy-weights digest in its own direct beats. Only the host EXECUTES the brain
    ///   (clients adopt its snapshots and render its articulation, rl#151), so brain equality
    ///   across peers gates nothing real; what must hold is "the host runs a real Sally, not
    ///   a failed/absent-checkpoint rest pose". The mid-game analogue is
    ///   [`crate::server::may_admit_joiner`]'s `HostNotArmed` self-gate.
    /// - `assets` — peer-symmetric as before: true only when we have a real crab-model asset
    ///   ourselves (non-zero digest) AND every live peer's last DIRECT beat carried that exact
    ///   digest. Every peer builds its rendered crab from this asset, so a mismatch is a real
    ///   client-side divergence even under host-auth.
    ///
    /// In both halves a `None` (relay-only, never heard directly) or zero digest fails —
    /// an unverifiable host, or an asset-divergent peer, can never be armed into lockstep.
    /// This is an OUTPUT, never a close gate: a failed verdict still FORMS the match, but the
    /// round can't arm the NN crab and — with no integer fallback (rl#114) — the windowed
    /// client REFUSES it loudly rather than playing a fake crab
    /// ([`crate::may_arm_external_crab`]). Call after [`Membership::poll`] (which expires the
    /// dead) so the live set is current.
    pub fn sync_verdict(&self) -> crate::SyncVerdict {
        let host = *self
            .live_set()
            .first()
            .expect("live_set always contains at least us");
        let host_weights = if host == self.me {
            self.local_digest
        } else {
            self.peers
                .get(&host)
                .and_then(|v| v.weights_digest)
                .unwrap_or(0)
        };
        crate::SyncVerdict {
            weights: host_weights != 0,
            assets: self.local_asset_digest != 0
                && self
                    .peers
                    .values()
                    .all(|v| v.asset_digest == Some(self.local_asset_digest)),
        }
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

        // Evaluate the agreement predicate over the current live set.
        let live = self.live_set();
        let my_hash = roster_hash(&live);
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
            Status::Agreed { roster: live }
        } else if timed_out {
            Status::Failed
        } else {
            Status::Forming { live: live.len() }
        }
    }
}

/// Encode a [`Beat`] for the wire:
/// `[start:u8][count:u16 LE][weights_digest:u64 LE][asset_digest:u64 LE][id:32]*count`,
/// the id list sorted+deduped. Fixed 32-byte ids (an [`EndpointId`] is a 32-byte public
/// key) so there is no per-id length. The leading `start` byte is the host's GO flag (rl#58);
/// `weights_digest` (rl#82) is the sender's checkpoint digest and `asset_digest` (rl#100) its
/// crab-model digest (NEITHER folded into [`roster_hash`] — capabilities, not membership). The
/// member count is bounded on decode ([`MAX_BEAT_MEMBERS`]) so a hostile/garbled frame can't
/// trigger a huge allocation.
pub fn encode_beat(beat: &Beat) -> Vec<u8> {
    let canon = |ids: &[EndpointId]| -> Vec<EndpointId> {
        let mut v = ids.to_vec();
        v.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        v.dedup();
        v
    };
    let members = canon(&beat.members);
    let mut out = Vec::with_capacity(19 + 32 * members.len());
    out.push(beat.start as u8);
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&beat.weights_digest.to_le_bytes());
    out.extend_from_slice(&beat.asset_digest.to_le_bytes());
    for id in &members {
        out.extend_from_slice(id.as_bytes());
    }
    out
}

/// Upper bound on members in a decoded [`Beat`], so a bad length can't allocate without
/// bound. Couch co-op is a handful of peers; [`PlayerId`] is a `u8` so 256 is the
/// absolute ceiling the rest of the stack accepts anyway.
const MAX_BEAT_MEMBERS: usize = 256;

/// Decode a [`Beat`] produced by [`encode_beat`]. Rejects a truncated body or a member count
/// past [`MAX_BEAT_MEMBERS`] (a malformed/hostile frame) rather than panicking or
/// over-allocating.
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
    let need = 19 + 32 * count;
    anyhow::ensure!(
        body.len() == need,
        "barrier frame length {} != expected {need} for {count} members",
        body.len()
    );
    let mut members = Vec::with_capacity(count);
    for i in 0..count {
        let off = 19 + 32 * i;
        let bytes: [u8; 32] = body[off..off + 32].try_into().expect("32-byte slice");
        let id = EndpointId::from_bytes(&bytes)
            .map_err(|e| anyhow::anyhow!("bad endpoint id in barrier frame: {e}"))?;
        members.push(id);
    }
    Ok(Beat {
        members,
        start,
        weights_digest,
        asset_digest,
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
    fn weights_gate_keys_on_the_hosts_digest_alone() {
        // Host-auth: only the host executes the brain (clients adopt its snapshots), so the
        // weights half of the verdict is the HOST self-gate — "the lowest live endpoint id
        // advertised a real (non-zero) digest" — never peer-symmetric brain equality (that
        // rule is superseded; see `sync_verdict`).
        let t0 = Instant::now();
        let mut ids = [eid(1), eid(2)];
        ids.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let (host, client) = (ids[0], ids[1]);
        const BRAIN: u64 = 0xABCD_0000_1234_5678;

        // We ARE the host with a real brain: armable regardless of the client's digest — a
        // client's drifted/absent brain gates nothing (it never runs).
        let mut a = Membership::new(host, 2, t0).with_weights_digest(BRAIN);
        a.on_beat(client, &bt_d(vec![host, client], BRAIN ^ 0xFF), t0);
        a.poll(t0);
        assert!(
            a.sync_verdict().weights,
            "the host self-gate passes on its own non-zero digest"
        );

        // We ARE the host with NO checkpoint (digest 0): refused — a zero-digest host would
        // serve a rest-pose fake Sally to everyone ([[real-sally-definition]]).
        let mut b = Membership::new(host, 2, t0);
        b.on_beat(client, &bt_d(vec![host, client], BRAIN), t0);
        b.poll(t0);
        assert!(
            !b.sync_verdict().weights,
            "a zero-digest host must not arm, whatever the clients carry"
        );

        // We are a CLIENT and the host advertised a real brain: armable even with no local
        // checkpoint — we render the host's Sally, we don't run her.
        let mut c = Membership::new(client, 2, t0);
        c.on_beat(host, &bt_d(vec![host, client], BRAIN), t0);
        c.poll(t0);
        assert!(
            c.sync_verdict().weights,
            "a client verifies the HOST's digest, not its own"
        );

        // We are a CLIENT and the host advertised digest 0: refused on our side too.
        let mut d = Membership::new(client, 2, t0).with_weights_digest(BRAIN);
        d.on_beat(host, &bt_d(vec![host, client], 0), t0);
        d.poll(t0);
        assert!(
            !d.sync_verdict().weights,
            "a zero-digest HOST is refused by its clients"
        );
    }

    #[test]
    fn relay_only_host_blocks_the_weights_gate() {
        // The host's digest counts only from its OWN direct beat: a host known only via
        // another peer's relay has `weights_digest: None` — an unverifiable brain, blocked
        // exactly like the roster-agreement `advertised: None` rule. A relay-only NON-host
        // peer does not block (its brain gates nothing under host-auth).
        let t0 = Instant::now();
        let mut ids = [eid(1), eid(2), eid(3)];
        ids.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let (host, mid, top) = (ids[0], ids[1], ids[2]);
        const BRAIN: u64 = 0x9999_8888_7777_6666;

        // We are `top`; `mid` beats directly, relaying the host's existence (no host digest).
        let mut a = Membership::new(top, 3, t0).with_weights_digest(BRAIN);
        a.on_beat(mid, &bt_d(vec![host, mid, top], BRAIN), t0);
        a.poll(t0);
        assert!(
            !a.sync_verdict().weights,
            "a relay-only HOST (digest None) must block the gate"
        );
        // The host speaks for itself with a real brain → armable.
        a.on_beat(host, &bt_d(vec![host, mid, top], BRAIN), t0);
        a.poll(t0);
        assert!(
            a.sync_verdict().weights,
            "once the host is heard directly with a real brain, armable"
        );

        // Mirror: we ARE the host — a relay-only (and even zero-digest) non-host peer does
        // NOT block the weights gate.
        let mut b = Membership::new(host, 3, t0).with_weights_digest(BRAIN);
        b.on_beat(mid, &bt_d(vec![host, mid, top], 0), t0);
        b.poll(t0);
        assert!(
            b.sync_verdict().weights,
            "non-host digests (zero or relay-only) gate nothing"
        );
    }

    // ── Asset-digest handshake (rl#100, GCR — the shared-collider-asset guard) ────────
    // A SIBLING of the weights handshake above: the crab-model digest rides its own Beat
    // field, is never folded into the roster hash, and gates arming the float NN crab on every
    // peer agreeing — so two peers whose crab colliders differ can't arm Sally and the round is
    // refused (rl#114, no integer fallback).

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
        assert!(a.sync_verdict().assets, "equal non-zero asset digests must be synced");

        // A peer on a DIFFERENT crab model → not synced (the desync the guard exists for).
        let mut b = Membership::new(ida, 2, t0).with_asset_digest(ASSET);
        b.on_beat(idb, &bt_ad(vec![ida, idb], ASSET ^ 0xFF), t0);
        b.poll(t0);
        assert!(!b.sync_verdict().assets, "a differing peer asset digest must not be synced");

        // A peer advertising a ZERO digest (no resolvable model) → not synced.
        let mut c = Membership::new(ida, 2, t0).with_asset_digest(ASSET);
        c.on_beat(idb, &bt_ad(vec![ida, idb], 0), t0);
        c.poll(t0);
        assert!(!c.sync_verdict().assets, "a zero peer asset digest must not be synced");

        // OUR OWN digest zero (no model locally) → never synced, even if a peer has one.
        let mut d = Membership::new(ida, 2, t0); // local_asset_digest defaults to 0
        d.on_beat(idb, &bt_ad(vec![ida, idb], ASSET), t0);
        d.poll(t0);
        assert!(!d.sync_verdict().assets, "a zero local asset digest is never synced");
    }

    #[test]
    fn weights_and_assets_synced_are_independent_gates() {
        // The two guards are orthogonal: a real host brain but mismatched crab assets must
        // leave a verdict with `weights` true yet `assets` false (and the arm site ANDs them,
        // so the round can't arm Sally and is refused — rl#114). Proves a refactor can't
        // collapse the two into one digest.
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
        };
        a.on_beat(idb, &mismatched_asset, t0);
        a.poll(t0);
        assert!(a.sync_verdict().weights, "a real host brain → weights gate passes");
        assert!(
            !a.sync_verdict().assets,
            "a different crab asset must leave assets NOT synced (independent of weights)"
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
