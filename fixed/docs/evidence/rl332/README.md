# rl#332 — Sally "flight": repro, root cause, fix evidence

Report (build 2eb10a9): "Oh I saw Sally flying
again by the way." This dir is the evidence pack for the diagnosis chain and the
fix. All runs: `game sally-soak` (landed with this issue), the live released
checkpoint (`rl-releases/bf5cf70174dc-w1785011429/checkpoints/best`), canonical
GCR terrain, 64 Hz 1:1 cadence, deterministic per seed. "Baseline" =
bf5cf70 (pre-fix), "post-fix" = carapace air drag (this change).

## Repro (stage 1)

4 seeds × 300 000 ticks (~78 sim-min each, ~5.2 sim-h total), baseline:

| seed | events | peak above ground | peak \|vy\| | peak up-vy |
|---|---|---|---|---|
| s1 | 0 | 0.98 m | 2.6 m/s | 1.3 m/s |
| s2 | 4 | **46.5 m** | **55.7 m/s** | 9.3 m/s |
| s3 | 1 | 7.4 m | 22.0 m/s | 4.7 m/s |
| s4 | 0 | 1.2 m | 1.9 m/s | 1.1 m/s |

Flight is real, terrain-dependent (seeds that wander into steep relief fly;
seeds on gentle ground never do), and reproducible in minutes of sim time.
`baseline-s2-event2-46m-luge.jsonl.gz` is the s2 peak event: a 296-tick window
of full-body state (per-part positions/velocities, contacts) showing her
crossing the map at 30+ m/s and falling 40+ m at −55 m/s.

## Diagnosis chain (what was ruled out, in order)

1. **Solver momentum leak (rl#321 class)** — RULED OUT. The rl#321 gates
   measure ≤ 1.3e-4 m/s² residual contact-free and 0.18 m/s² under
   self-contact (both at their post-cc09092 floors).
2. **Solver energy injection, flat ground** — RULED OUT.
   `passive_storm_energy_instrument`: thrash-energized crab with drives zeroed
   dissipates monotonically (E 4.42 J → 1.54 J over 10 s; max gain +0.000 J).
   Ablations (limits off / friction motors off / both) all dissipate.
3. **Solver energy injection, GCR terrain** — RULED OUT.
   `passive_storm_energy_on_terrain_instrument`: E −201 J → −1511 J over 20 s
   passive, max gain +0.000 J.
4. **Gravity-powered luge** — CONFIRMED as the mechanism of the sustained
   runs. `sally-soak --zero-drive-after 600` (seed s2, mid-storm): a fully
   PASSIVE ragdoll kept accelerating 21 → 46 m/s over 15 s while total
   mechanical energy fell monotonically (E −8210 → −12771 J): pure PE→KE down
   ~100 m+ of mountainside, near-lossless because a tumbling compact body
   neither slides (no friction work) nor feels any air. Recorded trace
   (progress lines, zero-drive at 600):

   ```
   tick  750  speed=14.2 m/s  E=-8209.7 J
   tick 1400  speed=36.0 m/s  E=-9302.9 J
   tick 1600  speed=46.1 m/s  E=-9302.9→-12296 J (falling)
   ```

**Root cause of the flight-scale speeds: the plant models no aerodynamic drag,
so nothing bounds speed on the km-scale GCR relief.** At 30–55 m/s every
terrain lip is a ski jump; "flight" is ballistic hang-time from illegitimate
speed, not an illegitimate force.

## Fix

Quadratic drag on the carapace (`crab-world/src/bot/aero.rs`), sized for
terminal velocity √(m·g/DRAG) ≈ 15 m/s at her measured 0.781 kg — bluff-body
plausible (C_d ≈ 0.4 at 0.14 m² frontal), a few % of body weight at gait
speed, ~5× her full-charge pace. Carapace-only: per-limb drag would damp the
fast limb whips the frozen policy's gait is built from.

- Regression: `airborne_crab_reaches_terminal_velocity` — the 4-s free-fall
  that integrated to ~39 m/s pre-fix must land in 11–18 m/s (measures
  14.87 m/s). Both-sided: catches missing/doubled drag AND re-bake mass drift.
- The rl#321 momentum gates now carry the drag impulse in their books
  (residuals stay at the 1e-4 floor).
- Behavior: `rl-train eval` worst-bearing progress 0.00 → 0.00 (unchanged
  headline; net_progress −13.7 → −5.3 m improved), `nn-crab-probe`
  deterministic A/B still MATCHES.

## Post-fix soaks (same seeds, same budget) — honest residual

| seed | events | peak above ground | peak \|vy\| | peak up-vy |
|---|---|---|---|---|
| s1 | 0 | 0.98 m | 1.8 m/s | 1.6 m/s |
| s2 | 23 | 34.1 m | 54.5 m/s | 29.6 m/s |
| s3 | 2 | 6.1 m | 14.2 m/s | 4.2 m/s |
| s4 | 0 † | 1.2 m | 1.7 m/s | ≤1.7 m/s |

† s4 is a CENSORED negative at 250 000/300 000 ticks: two independent post-fix
runs were both killed by host memory pressure at ~250 k, both with zero events
and a 1.2 m altitude ceiling to that point (baseline s4: zero events at the
full 300 k).

The passive luge is dead (nothing passive exceeds ~15 m/s any more), but s2
still storms — and the discriminator says why:

- `--zero-drive-after 103000` (just before s2's biggest post-fix storm window,
  103.5k–118.6k, 21 events): with drives zeroed the storm NEVER HAPPENS —
  zero further events, she settles. The residual flight is **policy-driven**.
- Power measurement (`P=` in soak progress lines, the eval rl#279 observable:
  commanded torque × sensed hinge rate): storm windows run 55–237 W gross
  actuator power vs 50–148 W during normal gait — full overlap, so no
  torque-speed power cap can starve storms without touching the gait.
- Scale: 100–200 W through a 0.781 kg body is ~200 W/kg (a real crab is
  ~2 W/kg). The torque ceilings (2.5–7 N·m per joint) that make the ragdoll
  trainable also make the OOD policy a legitimate catapult: it can climb a
  canyon wall at 9 m/s² against drag (`postfix-s2-event3-wallstorm.jsonl.gz`,
  ticks 104441–104513: +22 m/s in 2.3 s, 65–79 contacts, rising).

**Conclusion:** the sim physics is now honest — momentum conserved, passive
energy strictly dissipates, terminal velocity real. The remaining flight is
the frozen policy exercising a plant whose power-to-weight is ~100× animal
scale, concentrated in steep terrain far from the arena's play locales (max
uphill grade 0.36; seeds that stay on gentle ground: zero events across 156
sim-min). Fixing THAT means rescaling actuator strength — a new MDP, i.e. a
retrain — out of scope while training is off (this job's brief). Staged as
the follow-up on the issue.
