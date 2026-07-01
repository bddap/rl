# GCR Multiplayer: host-authoritative state-resync (rl#151, the post-lockstep design)

Status: **design, reviewer-looped (5 lenses + one round past).** Supersedes bit-exact
lockstep for the join path. Implementation lands as the increment sequence below (each
compiles headless **and** render, has a test, keeps single-player playable). Owner
direction realized: Minecraft-model client/server ([[mp-minecraft-model]]), SP =
MP-zero-remote one path ([[sp-is-mp-special-case]]), determinism downgraded to
cool-to-have ([[gcr-nn-crab-determinism-hard-req]]), real-Sally-or-loud-refuse
([[real-sally-definition]], [[silent-fallback-antipattern]]).

## Why lockstep is dead (the finding that forced this)

Job 509 (probe `nn-crab-join-xpeer`) proved a warm-incumbent ↔ cold-joiner mid-game join
**cannot** be bit-exact: `cold_respawn_armed_crab` (75710b61) respawns the crab *entities*
but not the incumbent's warm `RapierContext` — the solver/contact warm-start caches **and**
the rigid-body/collider handle-arena free-list survive, so the incumbent's first post-join
step differs from a fresh joiner's. Resetting the warm caches narrows it; the handle-arena
free-list still diverges. Closing it would need a deterministic rebuild of the *whole*
rapier world in pinned order on every incumbent every join — impractical in live bevy.

**The dissolve.** Under host-authoritative, **clients never run the rapier solver** — they
render the host's *output* transforms. The warm-cache / handle-arena state that breaks
bit-identity is never on the wire and never compared. The 509 blocker doesn't get worked
around; it ceases to exist. (Cross-peer pose differences would now be *cosmetic*, not a
correctness fault — but with host-sends-pose, below, there aren't any: every client shows
the host's exact pose.)

## The reframe that shapes everything

The authoritative game state is the **integer `Sim`** (`sim.rs:344`), not the float
physics. Per tick it owns: `tick`, a few `Player{pos,yaw,status}` (BTreeMap), `crab{pos,
yaw}`, the fixed `extraction`, `outcome`, the restart latch, the RNG stream position — the
exact fields `Sim::state_hash` (`sim.rs:749`) destructures exhaustively. ~100–200 bytes for
the whole match. The float rapier crab is a side-car: the NN policy drives it, and only its
quantized `Pos{i64 x,i64 z} + yaw_turns` + a physics digest are injected into the integer
sim via `set_external_crab_pose` (`sim.rs:435`). Game logic (grab, extract, outcome) is all
integer.

Consequence: **the snapshot is cheap and already enumerated.** Full-state-every-tick is
viable at ~KB/s; no delta compression, no interest management, no AAA netcode — fit the
model to GCR, a few players + one crab in a small arena ([[vibe-coded-dont-cargo-cult]]).

## Target architecture

**One authoritative server (in-process for SP, dialed over iroh for MP).** The server owns
the integer `Sim` **and** the rapier world (the one Sally) and steps them. Each tick:

1. Server collects each client's `Input` (clients send inputs up; the server's own local
   client included). No INPUT_DELAY barrier — the server applies what it has and steps.
2. Server steps the NN policy + rapier, derives the crab pose, injects it via
   `set_external_crab_pose`, then `Sim::step(inputs)` — the authoritative advance.
3. Server emits a **snapshot** (built at exactly one site, post-injection — see types).
4. Server broadcasts it; clients apply and render.

**Snapshot — two layers, split for the headless build** (a single struct carrying the
render-only articulation would break the no-`render` trainer build — [[verify-all-bins-on-module-moves]]):
- **`CoreSnapshot`** (lives beside `sim`, headless-safe, serializable): the `Sim` game
  state — `tick`, `players` (reuse `Sim`'s own `BTreeMap<PlayerId,Player>`, not a re-encoded
  vec, so a new `Player` field is caught at the serialize site), `crab` (Pos+yaw), `outcome`,
  `roster`. This is authoritative; it is everything game logic needs. ~100–200 B.
- **render articulation** (`#[cfg(feature="render")]`, carried in a render-gated extension
  frame, not in `CoreSnapshot`): the ~13 crab body-part `(pos,rot)` transforms the skinned
  Sally mesh needs. Render-only garnish the host already computes; the trainer never sees it.
- Total ≈ 0.5 KB/tick × 30 Hz ≈ **~15 KB/s**. Quantization/decimation is deferred, not
  needed to ship.

**One construction site, no dual crab-state drift.** The integer `crab{pos,yaw}` is *derived
from* the float pose by `set_external_crab_pose` in the same step; the snapshot is built
immediately after, at one function, so the integer Pos and the articulation can't disagree.
Don't scatter snapshot construction.

**Channel: latest-wins, never retransmit stale state.** A snapshot is a full state, so a
newer one *supersedes* an older — send on an unreliable/superseding channel (QUIC datagram
or a "drop if older than last-applied `tick`" discipline), never a reliable stream that
retransmits a stale snapshot behind head-of-line blocking. The snapshot `tick` IS the
version: a client applies the highest `tick` it has seen and discards older arrivals — this
also versions the roster, so no separate roster epoch is needed.

**Cadence:** one snapshot per sim tick (30 Hz, `cadence.rs`). Clients render **one snapshot
behind** and **interpolate** (lerp pos / slerp rot) between the two latest — the existing
tick-interpolation in `render/scene.rs` (alpha-tween `prev`→live) is exactly this, salvaged
wholesale.

**Client reconciliation — minimal, fit to GCR:**
- **Remote players + the crab**: pure interpolation. Remote authoritative entities; render
  slightly behind and tween — no prediction, no snap. The crab is **never predicted** (the
  cargo-cult trap; a 30 Hz interpolated articulated body reads fine on LAN). If the crab's
  lunge ever feels laggy on higher-latency links, *extrapolation* from the last two snapshots
  is the deferred knob — gated on the owner's eye (the visual oracle, [[owner-visual-spatial-oracle]]),
  not baked in speculatively.
- **The local foot player only**: client-side prediction + reconciliation. Apply own input
  immediately (responsive WASD); keep a small ring of recent `(tick, input)`; on each
  snapshot re-seat to the authoritative position and replay inputs newer than the snapshot's
  tick. The foot player is a trivial integer facing-relative mover (`sim.rs:616`), so this is
  a few lines and the correction is sub-tick on LAN; it's the one place WAN latency would
  show, and keeping it makes the local/remote player path uniform.

**Vehicle / plane mode is an input mode, not an SP fork** ([[rl-vehicles-plane-mode-required]]
— a hard req). Today the pilot toggle is gated on `is_solo()` (`render/driver.rs:444`) — a
parallel codepath. Under host-auth a vehicle is a client *input source* whose inputs the
server simulates like any other; fold the `is_solo()` gate away so piloting works in MP too,
on the one path.

**Why host-sends-pose, not clients-run-the-policy** (the one real fork in the design):
running the NN policy + rapier on *every* client costs compute that grows with the network
([[rl-gpu-training-bigger-nns]]) and reintroduces cosmetic crab drift needing anchoring.
Host-sends-pose makes the client a **pure renderer of authoritative state** — no drift, no
per-client brain, the cleanest single path, a literal Minecraft model. Bonus: any *recurrent*
policy state (LSTM/GRU hidden state) stays **host-local** and never has to be synced or
transferred on join — only the host runs the brain.

## SP = MP-zero-remote, ONE path

SP **already** funnels through the server path: it boots as `Coordinator::Server { net: None
}` (`net_loop.rs:271`) — a server with no remote peers (verified: `Coordinator::exchange` has
one path per arm, no hidden SP/MP branch). The pivot keeps that shape and changes only *what
the server produces*: the in-process server steps the authoritative `Sim` + rapier, and the
local client reads its state **through the same snapshot the wire client uses**. **Always
serialize, even in SP** — by-reference-in-SP-but-bytes-in-MP is two code paths and a
[[silent-fallback-antipattern]]; the copy is ~500 B/tick (trivial), and one serialized path
means "apply snapshot" is type-identical for local and remote. There is no `SoloHost`, no
SP/MP branch: the client always reads "the server's latest snapshot"; for SP that server is
in-process with zero remote clients.

## Real-Sally guarantee under host-auth (the guard that must MOVE, not vanish)

Lockstep enforced [[real-sally-definition]] *symmetrically* — every peer independently ran
the same real brain + colliders or the round refused (`may_arm_external_crab`, `lib.rs:91`,
requires `weights_synced && assets_synced` when networked). Host-auth removes the symmetric
run, so it must **add the upstream guard host-auth needs** rather than drop it:

1. **The host must prove it runs the real Sally.** A host whose checkpoint failed to load has
   `weights_digest == 0` and the policy emits a zero-action rest pose (`policy.rs`); today it
   would still arm in solo (`net_is_none → true`) and serve a non-Sally crab. **Add a host
   self-gate:** refuse to host a networked match (and surface it loudly in SP — a non-Sally
   crab is a failure, not a fallback) when `weights_digest == 0`. This is the upstream half
   that replaces the peer-symmetric `weights_synced`.
2. **The joiner must verify the host serves real Sally.** On dial, the joiner checks the
   host's advertised weights digest equals the canonical real-Sally digest (refuse a host
   running a random/zero brain — loud, never connect-and-render-a-fake). `may_admit_joiner`
   keeps refusing a digest-mismatched joiner; the reciprocal host-side non-zero self-gate
   (`AdmissionRefusal::HostNotArmed`, checked FIRST) makes two both-missing peers (`0 == 0`)
   unable to admit each other into a fake-crab match. Realized (incr 4) as: admission requires
   `host_weights != 0` AND `joiner == host`, so an admitted joiner is guaranteed the host runs
   its own non-zero brain — transitive verification, since a joiner only dials after loading its
   own real checkpoint (`net-join` bails loud on a local digest of `0`). NOTE — the fleet has no
   single hardcoded "canonical Sally" constant (that would drift every retrain); "real Sally" is
   *whatever trained checkpoint the fleet carries*, so the check authenticates digest **sameness**
   (both peers run the identical non-zero brain), not pedigree — two peers on the same wrong-but-
   non-zero checkpoint would still admit each other. That residual is inherent to the digest model,
   not an incr-4 gap.
3. **Asset digest already protects the joiner's render.** Under host-sends-pose the joiner
   *renders the Sally mesh skinned to the host's pose*. The existing `asset_digest` is
   `crab_asset_digest()`, which **hashes the raw `sally.glb` file bytes** (`crab-world/src/
   bot/meshfit.rs:107`) — so it binds the render mesh + rig *and* the colliders (the colliders
   are a deterministic pure function of those bytes). The current admission check therefore
   already gates exactly what the joiner renders; no `mesh_digest`/collider split is needed.
   Keep the asset check as-is — a joiner with the wrong `sally.glb` is refused loudly.

`may_arm_external_crab` thus reduces *on the authoritative host* to "the host has a real
(non-zero) checkpoint" — but the weights guarantee **relocates** to (1)+(2), it does not
vanish. The loud-refuse machinery (`Refuse` frame, typed refusal) and the principle survive.

## SALVAGE vs DELETE (against current `net/src/`)

The DELETE list is the **end state**, realized at increment 5 — nothing is deleted before
its host-auth replacement exists and is proven (finish the swap before landing; no
stopgap-first dual-broken window, [[no-stopgap-first-stepping-stone]]).

### DELETE — lockstep machinery host-authority obsoletes ([[deletions-welcome]])

1. **The state-hash desync oracle as live game control.** The `Fault::Desync`/`Unverifiable`
   cross-check in `Lockstep` (`lockstep.rs:62-81`, `333-339`), the per-tick `applied_hashes`
   + out-of-order `pending_peer_hashes` buffers (`lockstep.rs:100-110`), and the
   `Confirmed (tick,hash)` field of `TickMsg` (`lockstep.rs:55`). One authoritative sim →
   nothing to cross-check. **Keep `Sim::state_hash` itself** — it stays as the determinism
   *test* helper (`lib.rs:155-294` `desync_test`, `determinism_probe`) and its
   exhaustive-no-`..` destructure is the discipline `CoreSnapshot` reuses. It is just no
   longer a runtime cross-check, and no hash is shipped in the snapshot.
2. **`RosterSchedule` (entire `roster.rs`, 218 lines)** — exists only so peers agree
   tick-for-tick on the roster without a server. The server now owns the roster and ships it
   in `CoreSnapshot.roster` ([[collapse-global-mutable-state-not-coordinate]]). Delete only
   after `Server` is rewritten to own the roster directly (`Server` embeds it today,
   `server.rs:146`).
3. **`Lockstep`'s peer-symmetric advance** (`lockstep.rs:287-357`): the
   stall-until-every-peer's-input loop, **`INPUT_DELAY`** (`lockstep.rs:19`, also `Server::new
   next_emit`, `server.rs:172`), the symmetric `TickMsg` broadcast. Replaced by: server
   collects inputs → steps once → broadcasts snapshot; client sends input + applies snapshot +
   predicts its own avatar. `TickMsg` shrinks to a plain input frame (tick+input, no hash).
4. **`Sim::rebuild_with_roster` as the lockstep round-boundary join** (`sim.rs:483` +
   `lockstep.rs:318-320`): the *bit-identical-rebuild-on-every-peer* purpose is gone. Join =
   server adds the player, the next snapshot carries them, joiner boots from it. The method
   body (`config.players` update + `reset()`) **stays for round restart**.

### SALVAGE

1. **iroh/QUIC transport + frames** (`transport.rs`) — a generic framed pipe. Keep
   `Tick`(input up, minus the hash), `JoinRequest`/`Refuse`/`Welcome`/`RosterChange`;
   **replace** `TickSet`(complete input set down) with the snapshot frame(s). Extend the
   `Frame` enum, don't fork it.
2. **The digest-based admission guard + loud refusal** — `Server::may_admit_joiner`
   (`server.rs:97`), the `Refuse` frame ([[real-sally-definition]], [[silent-fallback-antipattern]]).
   It **moves and gains the host self-check** above; keep the machinery + the principle.
3. **`game net` / `game net-join` entry points** (efd63c9d, `net_loop.rs`
   `connect_and_form`/`connect_and_join`) — reused; the joiner boots from a snapshot instead
   of `Lockstep::join_at`.
4. **`cold_respawn_armed_crab`** (`external_crab.rs:256`) as the **SP / round-RESTART** path —
   correct for restart (despawn+respawn on the restart edge); only ever wrong as a lockstep
   *join* mechanism. Keep for restart; stop using it for join.
5. **Render interpolation** (`render/scene.rs` alpha-tween, `render/driver.rs` accumulator) —
   already the client-side interpolation host-auth needs. Salvage wholesale.
6. **`set_external_crab_pose`** (`sim.rs:435`); the **discovery/dial** part of `membership.rs`
   (simplified — clients find + dial the host; no symmetric roster-hash agreement, so much of
   `membership.rs` is more delete than salvage once the server owns the roster).

## Increment plan (each compiles headless **and** render, has a test, reviewer-loopable; SP never breaks)

**0 — the snapshot seam (SP-only, no behavior change).** Add `CoreSnapshot` (in a new
headless-safe module beside `sim`) reusing `Sim`'s own field types, with the
compile-enforced no-`..` destructure (a new `Sim`/`Player` field is a build error until it's
carried — make-illegal-states-unrepresentable), plus `Sim::core_snapshot()` /
`Sim::apply_core_snapshot()`. `Player`/`Crab` have private fields, so `CoreSnapshot` lives
*beside* `sim` (same module, or `sim` exposes the constructors) and carries a **deterministic
encoding** — a serde derive or a hand-rolled `to_bytes`/`from_bytes`. The fields are POD
integers (`pos`/`yaw`/`status`), so a derive is **inert w.r.t. the determinism firewall**: that
firewall governs the *step* (no HashMap walk / `thread_rng` / wall-clock), not the wire, so
serializing at the snapshot boundary adds no nondeterminism. Route the local client to read
game state through the snapshot accessor. No wire change, no lockstep removal, **no render-only field in `CoreSnapshot`** (so
the trainer build is untouched). **Test:** round-trip `apply(serialize→deserialize(snapshot))
.state_hash() == original.state_hash()` — completeness proven before anything depends on it.
SP plays identically.

**1 — server steps, client applies (still SP/in-process, always serialized).** Replace the
local client's `Lockstep`-advance with `server.step(inputs) → snapshot → client.apply`. SP =
one server, one local client, zero remote; remove the local client's re-simulation. The old
peer-symmetric path still exists for the untouched remote path. **Test:** snapshot-driven
state hash-equals the old sim-driven path over a scripted input log. SP identical.

**2 — snapshot frame(s) + remote client renders (no client-side sim).** Add the snapshot
frame to `transport` (`CoreSnapshot` always; render articulation in a render-gated extension
frame); server broadcasts; remote client renders via the salvaged interpolation and sends
inputs up (`Tick`). A 2-process host+client runs host-authoritatively. **Test:** headless
2-process host/join asserts the joiner's `CoreSnapshot` matches the host's per tick; a render
screenshot shows the posed crab + players.

**3 — local-player prediction + reconciliation + fold the vehicle gate.** Predict the local
foot player only (input ring + replay); remote players + crab interpolated. Fold the
`is_solo()` vehicle toggle into the one input path so piloting works in MP. **Test:**
reconcile leaves no visible snap; a networked round can enter vehicle mode.

**4 — mid-game join via snapshot transfer (the 509 fix).** Verify the asset-digest semantics
(open item above) first. Joiner dials → admission (host self-check + joiner host-verify +
asset/mesh check, loud `Refuse`) → server adds to roster → next snapshot carries the joiner →
joiner boots from it. The armed-Sally join that *failed* under lockstep now works by
construction. **Test:** the multi-process armed-Sally join (job 509's failing probe) shows
the joiner seeing the real Sally at the host's exact pose; an asset/brain-mismatched joiner —
and a zero-digest host — are refused loudly.

**5 — delete the dead lockstep machinery.** Host-auth now primary and proven: remove the
`Fault` cross-check / `applied_hashes` / `pending_peer_hashes` / `Confirmed` / `INPUT_DELAY` /
`RosterSchedule` / `Lockstep` peer-symmetric advance / `rebuild_with_roster`-as-join. Collapse
SP and MP to the one client-dials-server path. Net-negative diff. **Test:** full build
(headless + render) + all tests + SP + 2-proc MP + join green; and an explicit no-second-path
grep, e.g. `grep -rn 'rebuild_with_roster\|pending_peer_hashes\|INPUT_DELAY\|is_solo' net/src`
returns only dead comments ([[silent-fallback-antipattern]]).

Increments 0–1 are SP-only and behavior-preserving; 2–4 add the MP path beside the old one;
5 deletes the old one last — so SP (the live demo) is playable at every step.

## Premise challenges that changed the shape

1. **No rapier state on the wire.** The 509 killers (warm solver caches, handle-arena
   free-list) are irrelevant because clients render the host's *output* transforms, never the
   solver. The snapshot carries pose, never rapier internals — host-auth *dissolves* 509.
2. **The crab is never predicted.** Remote authoritative entity → interpolate (already built);
   prediction is for the local foot player only. Extrapolation is a deferred feel-knob, not a
   default.
3. **Full-snapshot-every-tick, no delta/interest-management.** The integer sim is ~hundreds of
   bytes; ~KB/s at 30 Hz. Compression is deferred, not a design requirement.
4. **Host-sends-pose beats clients-run-the-policy** — no per-client brain, no drift, dumb
   renderer, recurrent policy state stays host-local. It narrows real-Sally admission to "can
   the joiner render her" + "does the host run her," with the weights guard *relocated* (host
   self-check + joiner host-verify), not dropped.
5. **The real-Sally guard had to MOVE, not vanish** (the reviewer-loop's sharpest catch):
   removing the peer-symmetric `weights_synced` opens a silent non-Sally-host hole unless
   host-auth adds the upstream non-zero-digest self-gate. Folded into the design above.

### `CoreSnapshot` sketch (increment 0 — the seam everything builds on)

```rust
// Headless-safe, serializable. The render articulation (CrabPose, ~13 body-part
// transforms) rides a SEPARATE #[cfg(feature="render")] extension frame — NOT this
// struct — so the no-render trainer build never pulls it in.
//
// Completeness is compile-enforced like `Sim::state_hash`: `Sim::core_snapshot` builds
// this by destructuring `Sim { tick, players, crab, outcome, roster, .. }` with NO `..`
// over the carried fields, so a new authoritative field is a build error until carried.
pub struct CoreSnapshot {
    pub tick: u64,                          // also the version: apply-highest, drop-stale
    pub players: BTreeMap<PlayerId, Player>, // reuse Sim's own type (no re-encode drift)
    pub crab: Crab,                          // Pos + yaw — integer, authoritative
    pub outcome: Outcome,
    pub roster: Vec<PlayerId>,               // server-owned; replaces RosterSchedule
    // extraction is fixed (not sent); no state hash is shipped.
}
```
