# Mesh-fit: auto-deriving physics colliders from the skinned glTF

**Status:** spike / feasibility kickoff. Prototype lives in `src/bot/meshfit.rs`
(`#[cfg(test)]`, zero release-build impact). Run the validation table with:

```
cd /tmp/wt-meshfit && ( set -a; source /tmp/rl/.sandbox-env; set +a; \
  export HOME=/tmp; cargo test --release meshfit_validation -- --nocapture )
```

Nothing here is wired into `spawn_crab`. This document records what the spike
proved, where the auto-fit diverges from the hand-coded body and why, the hard
problems, and a phased plan to land it.

---

## 1. The idea, and what the spike actually does

`sally.glb` (the pretty model the cosmetic skin already rides — see
`src/bot/skin.rs`) carries everything a physics body needs *implicitly*: a
skeleton (114 deform/control bones with bind-pose transforms) and skin weights
(4 bone influences per vertex over 26 271 vertices). The hand-coded body
(`src/bot/body.rs`) re-specifies all of that by hand: per-part primitive
colliders, dimensions, densities, joint anchors, axes, and limits.

The spike pipeline:

1. **Load** the GLB with the `gltf` crate (already in the tree via bevy; added as
   a `dev-dependency` with only `utils`+`names`, no new crate pulled in).
2. **Cluster** every vertex to a physics part by its *dominant* bone, reusing the
   exact bone→part mapping the skin drives with (`bone_to_part`, a copy of
   `skin::bone_target`). 6 model bones per leg collapse to 3 physics segments.
3. **Fit** each cluster: a capsule (axis = largest-variance PCA eigenvector;
   radius = 95th-percentile perpendicular spread; endpoints pulled in by the
   radius so the caps land on the tips) **and** an oriented box (OBB, principal
   axes + 98th-percentile extents) as the non-elongated alternative.
4. **Mass properties** analytically (cylinder + 2 hemispheres for the capsule;
   solid box / ball for the others) under the *hand-coded density*, so fitted
   and reference mass are compared at equal density. The capsule formula was
   cross-checked independently and reproduces rapier's `capsule_y` + `Density`
   numbers (ref tibia ≈ 12 g, matching body.rs's own "~10–14 g" note).
5. **Validate** against `body.rs` — the deliverable. Per part: fitted vs
   hand-coded half-height/radius/mass/inertia, plus a capsule-fit residual and an
   OBB "blobbiness" (mid-axis / major-axis) that objectively says *which
   primitive the cloud wants*.

The reference numbers are read from `body.rs`'s own constants via a `#[cfg(test)]`
re-export (`body::reference`), so the comparison can never drift from the live
physics.

---

## 2. Validation results (real numbers, sally.glb)

### Headline

| metric | result |
|---|---|
| Vertices clustered | 26 271 across 33 parts, **every bone routes to a part** (no orphans) |
| Leg **lengths** (fit vs hand-coded, total) | **70–130 %** — skeleton recovers limb proportions well |
| Leg **radii** | **2–3× too fat** (tibia 0.078 vs 0.025, femur 0.068 vs 0.035) |
| Total **mass** | fitted **3.25 kg vs 1.17 kg hand-coded (279 %)** — driven by radius² |
| Carapace **footprint** | width 1.04 m vs 1.0 m — **clean match** |
| Carapace **height** | fitted half-extent **0.39 m vs 0.12 m** — model is a tall dome, physics a flat slab |
| Two middle coxae | **degenerate** capsule fit (blobbiness > 1.0, radius blows up to 0.20) |
| Claw hand / pincer | **not capsule-shaped** (blobbiness 0.5–0.6) — wants a box/hull |

### What fits cleanly (capsule is the right primitive)

- **Femur and tibia, all 8 legs.** Blobbiness 0.29–0.40 (clearly stick-like),
  lengths within ±30 %. These are unambiguous capsules. The radius is
  systematically larger than hand-coded because the **art legs are fleshy** and
  the **physics legs are deliberately thin sticks**, and they *taper* toward the
  foot (a constant-radius capsule shows a moderate residual ~0.5 purely from
  taper, not from being the wrong shape).
- **Carapace footprint.** The OBB recovers width/depth (half-extents ≈
  0.52 × 0.17 horizontal) within a few percent of the hand-coded 0.50 × 0.35
  box.

### What diverges, and why (this is the point of the kickoff)

1. **Radii / mass run heavy (≈2.8× total).** Mass ∝ radius²; the visual mesh is
   chunkier than the stick colliders. The hand-coded densities were *tuned*
   (distal links pushed UP to 14, claws DOWN to 1) to land a balance-able crab
   on thin geometry. **Re-fitting the geometry invalidates the density tuning**
   — you cannot keep both. Either the fitted colliders get their own density
   pass, or the fit is shrunk toward the stick model. This is the central
   coupling, not a bug.

2. **Two middle coxae fit degenerately** (`LegCoxa(*,1)`: blobbiness 1.13, radius
   → 0.20). The coxa bone (`Def_leg_0N.000`) owns a *tall, near-isotropic* blob
   of attachment flesh where the leg meets the shell (bbox ext ≈ 0.34×0.27×0.15),
   so there is **no dominant axis** for PCA to find and the perpendicular spread
   explodes. The clean front/back coxae (leg 0/3) fit fine; the middle ones,
   nearest the widest part of the shell, don't. A capsule is marginal for the
   coxa generally.

3. **Carapace is a dome, not a slab.** Bind-pose shell bones reach to y=0.61; the
   cluster is a rounded carapace, while the physics uses HALF_H=0.12 — a 3×
   flatter box chosen for *stance stability*, not visual fidelity. The fitted
   box would raise the CoM and change how the crab balances. **This is a physics
   decision the art does not encode.**

4. **Claw hand and pincer are genuinely non-capsule** (blobbiness 0.5–0.6, all
   three axes comparable). `ClawFore` clusters the pincer palm + fixed jaw
   (`pincer.002–005`) into one wide flat blob; a capsule mis-fits it (radius
   0.147, mass 8×). These want a **box or convex hull**.

5. **Eyes:** the model's "antennae" stand in for eye stalks; their cloud is a
   short stalk, fitted as a small capsule (≈0.05 r) vs the hand-coded 0.03 ball.
   Cosmetically irrelevant, dynamically negligible (sub-gram).

6. **Art asymmetry:** the right pincer has a stray extra bone
   (`Def_pincer.004.R.001`) the left lacks — the model is not perfectly
   bilaterally symmetric, so a naive per-side fit yields slightly different
   left/right colliders. Mitigation: mirror one side, or average.

### Skeleton placement (does the joint structure come from the skeleton?)

Bind-pose bone origins give clean per-leg segment spans, e.g.:

```
hand-coded full lengths:   coxa 0.300  femur 0.360  tibia 0.400 m
model leg_01.L bind spans: coxa 0.168  femur 0.373  tibia 0.381 m
model leg_03.L bind spans: coxa 0.221  femur 0.431  tibia 0.415 m
front-left foot reaches 1.008 m from the carapace bone
```

Femur/tibia spans land within ~15 % of hand-coded; the coxa span is shorter
(the model splits the hip differently). **The skeleton clearly carries the
joint-chain geometry** — anchor positions and segment lengths are right there in
the bind pose. What the skeleton does **not** carry: joint *axes*, *limits*,
*motor/friction* params, and the deliberate rest *stance* (the model's bind pose
is a flat splay; the physics rest is a planted Λ). Those stay hand-authored.

---

## 3. Hard problems

1. **Mesh ≠ physics intent.** The biggest finding: the art is a faithful crab;
   the physics body is a *gameplay abstraction* (flat shell for stability, thin
   light legs for clean dynamics, claws under-massed so the CoM sits over the
   feet). A literal fit reproduces the art, not the intent, and would need its
   densities (and probably a deliberate slim-down) re-derived. **Auto-fit gives
   geometry; the tuning is still design work.**

2. **Skinning bleed / no-clean-axis clusters.** Dominant-bone assignment is
   crude. Attachment regions (coxa↔shell) produce isotropic blobs that PCA can't
   axis-align. Needs either weighted assignment (weight-blend, not winner-take-
   all), per-bone (not per-segment) clustering, or trimming the cluster to its
   elongated core before fitting.

3. **Non-capsule parts need convex decomposition.** Carapace, claw hand, pincer
   are boxes/hulls, not capsules. A real pipeline must pick the primitive per
   part (the spike's blobbiness flag is a start) and run convex-hull (parry's
   `ConvexPolyhedron::from_convex_hull`) or VHACD-style decomposition for the
   chunky parts. Convex hulls of raw art can be high-vertex and slow; they need
   simplification, and a 30-link multibody with hull colliders may cost solver
   time the current capsules don't.

4. **Watertightness / art irregularities.** The mesh is one skinned primitive,
   not per-part watertight solids; the right-pincer extra bone shows the art
   isn't perfectly symmetric. Mass from a *surface* fit (what the spike does) is
   fine; mass from *volume integration* of a raw art mesh would need watertight,
   manifold per-part geometry the model doesn't guarantee.

5. **Training stability across a physics-model change.** Any collider change is a
   **new MDP**: masses, inertias, contact geometry, and balance all shift, so the
   current checkpoint will not transfer and reward curves reset. This is the
   riskiest part operationally — see the phased plan's gating.

6. **Joint structure is only half in the skeleton.** Anchors/lengths: yes. Axes,
   limits, rest stance, motor/friction: no. So "derive the body from the
   skeleton" can replace the *geometry* consts but not the *articulation* design;
   the two must stay co-authored, and the mapping from 114 bones → 32 DOFs is the
   hand-specified glue (today `bone_target`).

---

## 4. Integration options

| Option | What | Pros | Cons |
|---|---|---|---|
| **A. Offline bake** (recommended) | A dev subcommand fits once, writes a `colliders.ron`/`.json` the body loads | Deterministic; reviewable diff; no per-launch cost; art changes are an explicit re-bake + retrain | One more artifact to keep in sync; needs a loader |
| **B. Build-time fit** | `build.rs` fits at compile time | No runtime cost; always fresh | `build.rs` is untrusted-ish, slow, pulls glTF into the build graph; hard to inspect |
| **C. Runtime fit at spawn** | Fit on app start | Always matches the model | Per-launch parse of a 36 MB GLB; fit cost on the hot path; non-determinism risk |
| **D. Hand-coded + fit-as-check** | Keep `body.rs`, add the spike as a CI guard that flags drift between art and physics | Zero risk to training; catches "art moved, physics didn't" | Doesn't deliver the auto-body; just monitors |

**Recommendation: D now, A as the real landing.** Ship the validation as a guard
(it already is one) so art/physics drift is visible, then move to an offline bake
that emits a *typed* collider table the body consumes — keeping a hand-coded
fallback until a fit-derived body trains to parity.

---

## 5. Phased plan

**Phase 0 — spike (this PR-to-be).** Validation harness + numbers + this doc.
Capsule + OBB + mass/inertia + placement, all `#[cfg(test)]`. *Done.* Outcome:
**feasible for legs, needs work for coxa/carapace/claws; densities must be
re-derived.**

**Phase 1 — fit quality.** Fix the degenerate clusters: weight-blended vertex
assignment, per-bone clustering with core-trimming, and a per-part primitive
choice (capsule vs box vs hull) driven by blobbiness/residual. Add convex-hull
fitting (parry) for carapace/claw/pincer. Gate: every part fits a sane primitive
with residual under a threshold; left/right symmetric within tolerance.

**Phase 2 — typed collider table + offline bake.** Define a
`FittedBody { parts: Vec<FittedPart> }` type (collider enum + transform + density)
and a baker subcommand that writes it. `body.rs` gains a path that builds
colliders from the table when present, hand-coded otherwise. Joints/axes/limits/
rest stance stay hand-authored (the skeleton doesn't encode them). Make illegal
states unrepresentable: one collider enum, mass derived from it, no parallel
dims. Gate: a unit test asserts the baked body's per-part mass/inertia within X %
of a chosen target (whether that target is "the art" or "the current stick body"
is the design call from §3.1).

**Phase 3 — density / balance re-derivation.** With fitted geometry, re-pick
densities (or a global scale) so total mass and CoM keep the crab balance-able —
the fitted 2.8× mass and dome carapace will not stand as-is. Validate statically
(CoM over the support polygon) before any training.

**Phase 4 — retrain & A/B.** Train the fit-derived body from scratch (it's a new
MDP — no checkpoint transfer). Keep the hand-coded body as the shipped default
until the fitted one reaches parity on stand/walk. Only then flip the default.
Gate: fitted-body policy matches or beats hand-coded on the standing/locomotion
metrics over a full run.

---

## 6. Verdict

**Feasible, with scope.** The skeleton + skin genuinely carry the body's
geometry: limb segmentation, lengths, and the carapace footprint all fall out of
the fit cleanly, and the bone→part mapping the skin already uses is exactly the
clustering the fit needs. But "auto-generate the colliders" is **not** a drop-in
replacement for `body.rs`: the hand-coded body is a *tuned gameplay abstraction*
(flat shell, thin light legs, under-massed claws) that the art does not encode,
so the auto-fit must be followed by a primitive-selection pass (capsule/box/hull),
a density re-derivation, and a from-scratch retrain. The realistic win is an
**offline-baked, typed collider table with a hand-coded fallback**, landed behind
a train-to-parity gate — not a runtime fit and not a blind swap.
