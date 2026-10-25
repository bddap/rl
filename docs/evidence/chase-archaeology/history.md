# rl#351 history: the chase archaeology record

Everything [rl#351](https://github.com/bddap/rl/issues/351) recorded before its body was cut to a status page: the issue body as it stood, then every comment in posting order, verbatim. Commit SHAs are as written at the time; [rl#414](https://github.com/bddap/rl/issues/414) later rewrote history, and comment 54 below maps old SHAs to new. Images and clips link to files beside this one.

## Index

| # | posted | headline |
|---:|---|---|
| [1](#c1) | — | Report and directive (verbatim) |
| [2](#c2) | 2026-08-07 | [xhigh bump is landing (job 2231, needs a hub restart). Stage 1 queued (2233): recover the Jun-28 training source — the era binary pin rl-demo-3ddcf6f is the …](https://github.com/bddap/rl/issues/351#issuecomment-5210606733) |
| [3](#c3) | 2026-08-11 | [Probe A assess — verdict: **confounded (red-leaning)**](https://github.com/bddap/rl/issues/351#issuecomment-5247534132) |
| [4](#c4) | 2026-08-11 | [Correction (owner review): the end-of-bound behavior is **falling**, not "drift" — and the render leg is off-distribution: this pin's `rl-demo` renders only …](https://github.com/bddap/rl/issues/351#issuecomment-5247638450) |
| [5](#c5) | 2026-08-11 | [**Audit result (follow-up to the correction above): the eval instrument never saw the probe's terrain. The eval leg is struck; the verdict survives on the in…](https://github.com/bddap/rl/issues/351#issuecomment-5247672165) |
| [6](#c6) | 2026-08-11 | [Probe B assess — verdict: **GREEN**](https://github.com/bddap/rl/issues/351#issuecomment-5247760466) |
| [7](#c7) | 2026-08-12 | [Probe C assess — verdict: **RED**](https://github.com/bddap/rl/issues/351#issuecomment-5263261832) |
| [8](#c8) | 2026-08-13 | [Probe D — 24 h in: upward-trend, no cross → ONE 24 h extension (per the rule)](https://github.com/bddap/rl/issues/351#issuecomment-5286428448) |
| [9](#c9) | 2026-08-15 | [Probe D verdict: RED (qualified — partial learn, not A/C's flatline)](https://github.com/bddap/rl/issues/351#issuecomment-5302193567) |
| [10](#c10) | 2026-08-17 | [Probe E verdict: GREEN](https://github.com/bddap/rl/issues/351#issuecomment-5310533531) |
| [11](#c11) | 2026-08-18 | [Probe F assess — DEGRADED. `56754c32` (mlp256 → mlp512x3) convicted for cold-start learning.](https://github.com/bddap/rl/issues/351#issuecomment-5322994955) |
| [12](#c12) | 2026-08-18 | [**Arch differential vs the #342 roast — the conviction's mechanism, posted in full at https://github.com/bddap/rl/issues/342#issuecomment-5326903217.**](https://github.com/bddap/rl/issues/351#issuecomment-5326904740) |
| [13](#c13) | 2026-08-19 | [Probe G assess — verdict: **RED (flatline). The second breaker is real — main flatlines with the arch controlled.**](https://github.com/bddap/rl/issues/351#issuecomment-5341064760) |
| [14](#c14) | 2026-08-19 | [Correction to the sps note above: A's train.log gives median 515 steps/s (peak 804, n=6263 steady-state samples), not ~800. G held 261; H is holding 273. The…](https://github.com/bddap/rl/issues/351#issuecomment-5344403309) |
| [15](#c15) | 2026-08-20 | [Probe H assess — verdict: **VOID (instrument wedge — no REACH verdict; relaunch as H2)**](https://github.com/bddap/rl/issues/351#issuecomment-5355932495) |
| [16](#c16) | 2026-08-20 | [sps anomaly follow-up — the in-train chase-eval is disconfirmed as the cause of the G/H ~270 sps crawl:](https://github.com/bddap/rl/issues/351#issuecomment-5357804179) |
| [17](#c17) | 2026-08-21 | [Probe H2 assess — verdict: **RED (floor with era REACH restored) — `8bb5009` REACH 0.2 is ACQUITTED as the main-line breaker.**](https://github.com/bddap/rl/issues/351#issuecomment-5370893881) |
| [18](#c18) | 2026-08-25 | [Probe I assess — verdict: **NOT-FLOOR** (D/F-like slow learning) — everything ≤07-17 ACQUITTED, breaker narrows to `2760e243..4cdf934c` (07-18..07-27)](https://github.com/bddap/rl/issues/351#issuecomment-5403783019) |
| [19](#c19) | 2026-08-31 | [Probe K assess — verdict: **FLOOR (trendless at the full I/J budget)** — floor-maker narrows to (`69d39c8`..`074c1263`]: contact stiffness / shoves, singles …](https://github.com/bddap/rl/issues/351#issuecomment-5472191034) |
| [20](#c20) | — | Report (verbatim) |
| [21](#c21) | 2026-09-02 | [Probe L assess — verdict: **NOT-FLOOR (strongest climb of the series)** — contact stiffness `19a1526` ACQUITTED → random shoves `074c1263` CONVICTED as the f…](https://github.com/bddap/rl/issues/351#issuecomment-5502872204) |
| [22](#c22) | 2026-09-02 | [rl#415 fix landed in 6e0ac5d: shoves are now 0.1–0.4× body weight (Δv ≤ 0.5 m/s) and OFF by default — `rl-train learn --shove-prob <p>` turns them on (1/640 …](https://github.com/bddap/rl/issues/351#issuecomment-5503585285) |
| [23](#c23) | 2026-09-05 | [For the next sanctioned training run — terrain-amplitude curriculum (verbatim):](https://github.com/bddap/rl/issues/351#issuecomment-5553185430) |
| [24](#c24) | 2026-09-06 | [Probe N assess — verdict: **CLIMB (L-like, qualified)** — `2e5605f` + shoves OFF learns where main + shoves OFF floors; the second main-side breaker sits in …](https://github.com/bddap/rl/issues/351#issuecomment-5562379444) |
| [25](#c25) | 2026-09-08 | [Probe O assess — verdict: **FLOOR (deep flatline)** — `40fde25` + shoves OFF never learns; the second main-side breaker sits in `2e5605f`..`40fde25` (85 comm…](https://github.com/bddap/rl/issues/351#issuecomment-5592833129) |
| [26](#c26) | 2026-09-11 | [**Probe P — 30M-tick early read (not yet legible; next read at 40M)**](https://github.com/bddap/rl/issues/351#issuecomment-5639856499) |
| [27](#c27) | 2026-09-12 | [Probe P assess — M-LIKE at the band's low edge ⇒ two breakers, one each side of `7184c17`](https://github.com/bddap/rl/issues/351#issuecomment-5648208263) |
| [28](#c28) | 2026-09-12 | [**Probe Q — early read at 0.79M ticks (2.2 h in; not legible, next read at 30M)**](https://github.com/bddap/rl/issues/351#issuecomment-5649125691) |
| [29](#c29) | 2026-09-16 | [**Probe Q — 30M-tick read (O-LIKE-shaped; verdict + render at 40M)**](https://github.com/bddap/rl/issues/351#issuecomment-5690363461) |
| [30](#c30) | 2026-09-17 | [Probe Q assess — O-LIKE through 40M; bump convicted as breaker 2](https://github.com/bddap/rl/issues/351#issuecomment-5706819167) |
| [31](#c31) | 2026-09-23 | [Probe R assess — P-LIKE: reverting the bump restores P-level learning ⇒ breaker 2 is the bevy 0.19 / rapier 0.35 bump](https://github.com/bddap/rl/issues/351#issuecomment-5804590760) |
| [32](#c32) | 2026-09-25 | [**Probe S — 10M-tick read (P-LIKE-shaped; verdict + render at 20M)**](https://github.com/bddap/rl/issues/351#issuecomment-5824719912) |
| [33](#c33) | 2026-09-26 | [Probe S assess — P-LIKE: `c9f2428` learns like P ⇒ breaker 1 lies in commits 1–31](https://github.com/bddap/rl/issues/351#issuecomment-5842304022) |
| [34](#c34) | 2026-09-26 | [**Probe T — 10M-tick read (N-LIKE-shaped; verdict + render at 20M)**](https://github.com/bddap/rl/issues/351#issuecomment-5846248931) |
| [35](#c35) | 2026-09-26 | [Probe T assess — N-LIKE: `b11ef09` climbs where `c9f2428` floors ⇒ breaker 1 is rl#340 stage 2](https://github.com/bddap/rl/issues/351#issuecomment-5849787113) |
| [36](#c36) | 2026-09-27 | [The retrain after this endgame should wear #20's hull body: feet and pincers as convex hulls (branch [`rl20-hull-candidate`](https://github.com/bddap/rl/tree…](https://github.com/bddap/rl/issues/351#issuecomment-5853064287) |
| [37](#c37) | 2026-09-27 | [**Endgame A/B, 10M read.** Not decisive: M separated from N only after ~15M.](https://github.com/bddap/rl/issues/351#issuecomment-5857281232) |
| [38](#c38) | 2026-09-28 | [**Endgame A/B: fourth arm E12 added at 2026-09-28 05:22Z.** E12 is E1 + E2 together, T's whole parameter surface on main: contact ζ5, solver (4,1,1), linear …](https://github.com/bddap/rl/issues/351#issuecomment-5864000085) |
| [39](#c39) | 2026-09-28 | [**Endgame A/B, 20M read.** Still not separable: N and M were tied at 20M (0.047 and 0.042) and only split at 30–40M.](https://github.com/bddap/rl/issues/351#issuecomment-5867748560) |
| [40](#c40) | 2026-09-29 | [**Endgame A/B, 30M read: every arm is climbing off M's floor.**](https://github.com/bddap/rl/issues/351#issuecomment-5889017511) |
| [41](#c41) | 2026-09-29 | [Handled in flight (botq 2996): the pre-purge main is kept in a private recovery archive off GitHub (pushing old history anywhere there would re-upload the bl…](https://github.com/bddap/rl/issues/351#issuecomment-5890782239) |
| [42](#c42) | 2026-09-29 | [Probe A launched (job 2260)](https://github.com/bddap/rl/issues/351#issuecomment-5891859705) |
| [43](#c43) | 2026-09-29 | [Probe B launched (job 2262)](https://github.com/bddap/rl/issues/351#issuecomment-5891861444) |
| [44](#c44) | 2026-09-29 | [Decision-tree step → Probe C launched (job 2294)](https://github.com/bddap/rl/issues/351#issuecomment-5891867667) |
| [45](#c45) | 2026-09-29 | [Decision-tree step → Probe D launched (job 2406)](https://github.com/bddap/rl/issues/351#issuecomment-5891881266) |
| [46](#c46) | 2026-09-29 | [Decision-tree step → Probe E launched (job 2470)](https://github.com/bddap/rl/issues/351#issuecomment-5891889223) |
| [47](#c47) | 2026-09-29 | [Probe I launched (job 2673) — bisect at the 07-17 collider-regen boundary](https://github.com/bddap/rl/issues/351#issuecomment-5891942084) |
| [48](#c48) | 2026-09-29 | [**Probe I `charge_drifted` (rl#266) — diagnosis, for the bound assess**](https://github.com/bddap/rl/issues/351#issuecomment-5891959467) |
| [49](#c49) | 2026-09-29 | [Probe I assess — interim: extension spent (upward trend, no cross)](https://github.com/bddap/rl/issues/351#issuecomment-5891970095) |
| [50](#c50) | 2026-09-29 | [Probe J launched (job 2716) — bisect midpoint of `2760e243..4cdf934c`](https://github.com/bddap/rl/issues/351#issuecomment-5891979679) |
| [51](#c51) | 2026-09-29 | [Decision-tree step → Probe K launched (job 2839)](https://github.com/bddap/rl/issues/351#issuecomment-5891990245) |
| [52](#c52) | 2026-09-29 | [Probe L launched (job 2946)](https://github.com/bddap/rl/issues/351#issuecomment-5891993898) |
| [53](#c53) | 2026-09-29 | [Lateral read of the probe logs (2026-09-01T19:45Z, Probe L at 65.8M ticks, 5 h before its bound): **L vs K is a clean A/B of the shove commit** — K = L + `07…](https://github.com/bddap/rl/issues/351#issuecomment-5891997770) |
| [54](#c54) | 2026-09-29 | [rl#414 rewrote every rl SHA after the first 14 commits (filter-repo also rewrote SHA references inside commit messages, so the cascade reached the whole hist…](https://github.com/bddap/rl/issues/351#issuecomment-5892008659) |
| [55](#c55) | 2026-09-29 | [Probe M assess — verdict: **FLOOR** — shoves OFF on current main does NOT restore learning; a second main-side breaker sits in `d577e9c9`..`138d1eb` (old `07…](https://github.com/bddap/rl/issues/351#issuecomment-5892024189) |
| [56](#c56) | 2026-09-29 | [Probe P launched — `7184c17` (commit 43/85, pre bevy 0.19 / rapier 0.35 bump), bound in TICKS](https://github.com/bddap/rl/issues/351#issuecomment-5892057561) |
| [57](#c57) | 2026-09-29 | [Decision-tree step → Probe R launched (job 4155)](https://github.com/bddap/rl/issues/351#issuecomment-5892067385) |
| [58](#c58) | 2026-09-29 | [Stage 1 — source recovered, changes enumerated, bisect plan](https://github.com/bddap/rl/issues/351#issuecomment-5900501079) |
| [59](#c59) | 2026-09-29 | [Probe J assess — interim: extension spent (overlap-zone read, no separation yet)](https://github.com/bddap/rl/issues/351#issuecomment-5900518787) |
| [60](#c60) | 2026-09-29 | [Probe J assess — verdict: **NOT-FLOOR** (I-like climb) — everything ≤07-19 `69d39c8` ACQUITTED, floor-maker narrows to (`69d39c8`..`4cdf934c`] (07-19..07-27)](https://github.com/bddap/rl/issues/351#issuecomment-5900521457) |
| [61](#c61) | 2026-09-29 | [Probe K assess — interim: extension spent (overlap-zone read, no separation)](https://github.com/bddap/rl/issues/351#issuecomment-5900524701) |
| [62](#c62) | 2026-09-29 | [Decision-tree step → Probe F launched (job 2508)](https://github.com/bddap/rl/issues/351#issuecomment-5900585909) |
| [63](#c63) | 2026-09-29 | [Decision-tree step → Probe G launched (job 2551)](https://github.com/bddap/rl/issues/351#issuecomment-5900587713) |
| [64](#c64) | 2026-09-29 | [Decision-tree step → Probe H launched (job 2594)](https://github.com/bddap/rl/issues/351#issuecomment-5900590426) |
| [65](#c65) | 2026-09-29 | [Decision-tree step → Probe H2 launched (job 2602)](https://github.com/bddap/rl/issues/351#issuecomment-5900594719) |
| [66](#c66) | 2026-09-29 | [Probe M launched (job 3001) — the main-side confirmation half of the shove conviction](https://github.com/bddap/rl/issues/351#issuecomment-5900599167) |
| [67](#c67) | 2026-09-29 | [Probe N launched (job 3161) — bisect midpoint of the post-shove range](https://github.com/bddap/rl/issues/351#issuecomment-5900603006) |
| [68](#c68) | 2026-09-29 | [Probe O launched (job 3402) — bisect midpoint of `2e5605f`..`138d1eb`](https://github.com/bddap/rl/issues/351#issuecomment-5900605493) |
| [69](#c69) | 2026-09-29 | [Decision-tree step → Probe Q launched (job 3704)](https://github.com/bddap/rl/issues/351#issuecomment-5900609966) |
| [70](#c70) | 2026-09-29 | [Probe S launched (job 4561) — breaker 1 content split at `c9f2428`](https://github.com/bddap/rl/issues/351#issuecomment-5900615218) |
| [71](#c71) | 2026-09-29 | [Probe T launched (job 4881) — single `b11ef09`, stage 2's parent](https://github.com/bddap/rl/issues/351#issuecomment-5900619177) |
| [72](#c72) | 2026-09-29 | [Endgame A/B launched (job 5026): three arms on current main](https://github.com/bddap/rl/issues/351#issuecomment-5900622864) |
| [73](#c73) | 2026-09-30 | [Endgame A/B: E0 and E2 have finished their 40M budgets; E1 stopped on an integrity failure; E12 is still running.](https://github.com/bddap/rl/issues/351#issuecomment-5914960248) |
| [74](#c74) | 2026-10-01 | [Endgame render update (job 5026): E0/E1/E2 completed; E12 still training.](https://github.com/bddap/rl/issues/351#issuecomment-5925081278) |
| [75](#c75) | 2026-10-01 | [Endgame verdict: retain the tested main settings. The historical floor-remedy revert is obsolete at this pin; deterministic chase is still unsolved.](https://github.com/bddap/rl/issues/351#issuecomment-5926592440) |
| [76](#c76) | 2026-10-01 | [Matched flat diagnostic instrumentation landed in **a9372f7**. `render-video` can now use the trainer's flat ground and ≤9 m target band, and compare action …](https://github.com/bddap/rl/issues/351#issuecomment-5942681198) |
| [77](#c77) | 2026-10-02 | [First 10M-tick read of the cold hull-body flat ≤9 m baseline at `a9372f7`:](https://github.com/bddap/rl/issues/351#issuecomment-5947533650) |
| [78](#c78) | 2026-10-02 | [Matched hull flat-baseline verdict at **a9372f7**: training improved but was **censored by an integrity failure at 34,240,512 of 40M ticks**. It was not rest…](https://github.com/bddap/rl/issues/351#issuecomment-5961152278) |
| [79](#c79) | 2026-10-02 | [**Pre-registration — flat short-band integrity reproduction** (registered before the hunt starts; numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-5961509043) |
| [80](#c80) | 2026-10-03 | [**Flat short-band integrity reproduction — verdict: REPRODUCED; no single demonstrated cause qualifies for a fix → next bounded diagnostic appended.**](https://github.com/bddap/rl/issues/351#issuecomment-5966200102) |
| [81](#c81) | 2026-10-03 | [**Pre-registration — multibody-solve integrity rate A/B** (registered before any arm runs; the numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-5967242336) |
| [82](#c82) | 2026-10-03 | [**Multibody-solve integrity rate A/B: verdict NOT CONVICTABLE.** The shipped arm tripped once in 9.44M ticks, below the pre-registered ≥5, so no solver lever…](https://github.com/bddap/rl/issues/351#issuecomment-5972121711) |
| [83](#c83) | 2026-10-03 | [**Pre-registration — trip-rate process/host split** (registered before either arm runs; the numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-5972221733) |
| [84](#c84) | 2026-10-04 | [**Trip-rate process/host split: verdict CHANCE.** Neither the host nor the process layout explains the 0.64/M vs 0.11/M gap. Under the pre-registered rule (b…](https://github.com/bddap/rl/issues/351#issuecomment-5976833153) |
| [85](#c85) | 2026-10-04 | [**Pre-registration — re-sized multibody-solve lever A/B** (registered before any arm runs; the numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-5976973289) |
| [86](#c86) | 2026-10-04 | [**Re-sized multibody-solve lever A/B: verdict CONVICTED. Both levers pass, and (4,12,3)×2 ships on main in [`73e7b88`](https://github.com/bddap/rl/commit/73e…](https://github.com/bddap/rl/issues/351#issuecomment-5980377900) |
| [87](#c87) | 2026-10-05 | [**Outer-4 matched cold rerun — 10M early read** (`rl-target@canary-hull-flat-outer4-73e7b88`, `73e7b88`, solver (4,12,3)×2)](https://github.com/bddap/rl/issues/351#issuecomment-5986582860) |
| [88](#c88) | 2026-10-09 | [**Outer-4 matched cold rerun: integrity KEPT, chase NOT demonstrated.** The render half fails, so the next bounded diagnostic is appended below.](https://github.com/bddap/rl/issues/351#issuecomment-6074530946) |
| [89](#c89) | 2026-10-09 | [**Outer-4 budget continuation — launched** (warm, same run, `--ticks 80000000`)](https://github.com/bddap/rl/issues/351#issuecomment-6074669047) |
| [90](#c90) | 2026-10-09 | [**Outer-4 budget continuation: stopped by an rl#343 hard-fail at 48,172,032 ticks. No chase.** Reach peaked at 35–40M and fell after it. Two boxes are append…](https://github.com/bddap/rl/issues/351#issuecomment-6081002141) |
| [91](#c91) | 2026-10-09 | [**Pre-registration — right-carpus residual: onset capture and trip rate before/after** (registered before any arm runs; the numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-6082780452) |
| [92](#c92) | 2026-10-09 | [**Right-carpus residual: cause found and fixed in the multibody friction solve. Kept by the pre-registered rule: 0 trips and 122 → 0 near-miss events.**](https://github.com/bddap/rl/issues/351#issuecomment-6090077345) |
| [93](#c93) | 2026-10-09 | [**Pre-registration — mean-policy inversion sweep, frozen 40M and 48.17M outer-4 checkpoints** (registered before any sweep render; the definitions and rule b…](https://github.com/bddap/rl/issues/351#issuecomment-6090384266) |
| [94](#c94) | 2026-10-09 | [**Mean-policy inversion sweep: inversion grew from 3 to 7 of 20 seeds between 40M and 48.17M, and no deeply inverted body ever recovered. The reward never re…](https://github.com/bddap/rl/issues/351#issuecomment-6090609779) |
| [95](#c95) | 2026-10-09 | [**Pre-registration — outer-2 integrity on the fixed plant** (registered before any frozen-checkpoint arm runs; the numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-6090849435) |
| [96](#c96) | 2026-10-10 | [**Outer-2 integrity with the friction fix: HOLDS. Main returns to (2,12,3)×2.** By the pre-registered rule, both outer-2 arms tripped 0 times with 0 near mis…](https://github.com/bddap/rl/issues/351#issuecomment-6094384747) |
| [97](#c97) | 2026-10-10 | [**Pre-registration — upright-shaping A/B (inversion recovery)** (registered before either arm launches; the numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-6100998864) |
| [98](#c98) | 2026-10-10 | [**Upright-shaping A/B: both arms launched 2026-10-10 18:51Z** ([pre-registration](https://github.com/bddap/rl/issues/351#issuecomment-6100998864))](https://github.com/bddap/rl/issues/351#issuecomment-6101035108) |
| [99](#c99) | 2026-10-12 | [**Upright-shaping A/B: REVERTED by the pre-registered rule.** Shaping removed inversion from the mean policy (0/20 inverted at 20 s, against 8/20 for control…](https://github.com/bddap/rl/issues/351#issuecomment-6125760321) |
| [100](#c100) | 2026-10-12 | [**Pre-registration — upright-shaping seed replicate** (registered before any arm launches; the numbers below are fixed)](https://github.com/bddap/rl/issues/351#issuecomment-6125902831) |
| [101](#c101) | 2026-10-12 | [**Upright-shaping seed replicate: four arms launched 2026-10-12 14:05Z** ([pre-registration](https://github.com/bddap/rl/issues/351#issuecomment-6125902831))](https://github.com/bddap/rl/issues/351#issuecomment-6126660633) |
| [102](#c102) | 2026-10-16 | [**Upright-shaping seed replicate: REVERTED by the pre-registered rule.** k = 12 cost no training reach (pooled 30–40M over seeds 351–353: 0.113 vs control 0.…](https://github.com/bddap/rl/issues/351#issuecomment-6223519208) |
| [103](#c103) | 2026-10-16 | [**Pre-registration: entropy-floor A/B.** [Protocol](https://github.com/bddap/rl/blob/c14e253/docs/evidence/chase-archaeology/entropy-floor-ab/README.md).](https://github.com/bddap/rl/issues/351#issuecomment-6223666009) |
| [104](#c104) | 2026-10-16 | [**Entropy-floor A/B: seed-351 pair launched cold.** [Launch evidence and binary hashes](https://github.com/bddap/rl/tree/c655927/docs/evidence/chase-archaeol…](https://github.com/bddap/rl/issues/351#issuecomment-6224023125) |
| [105](#c105) | 2026-10-18 | [**Entropy-floor A/B: training seed 351 completed; seed 352 launched.** [Evidence](https://github.com/bddap/rl/tree/5cf9c59/docs/evidence/chase-archaeology/en…](https://github.com/bddap/rl/issues/351#issuecomment-6255154924) |
| [106](#c106) | 2026-10-20 | [**Entropy-floor A/B: training seed 352 completed; seed 353 launched.** [Evidence](https://github.com/bddap/rl/tree/83badf8/docs/evidence/chase-archaeology/en…](https://github.com/bddap/rl/issues/351#issuecomment-6294383060) |
| [107](#c107) | 2026-10-22 | [Completed the architecture/optimizer A/B and rejected the constant −0.7 log-std floor; retain the existing −1 endpoint. Landed evidence: [96d6067](https://gi…](https://github.com/bddap/rl/issues/351#issuecomment-6337580474) |
| [108](#c108) | 2026-10-24 | [**Outer-4 frame budget on device: TV half measured, 60 fps holds. Deck half needs a person: the box is marked `(needs: …)`.**](https://github.com/bddap/rl/issues/351#issuecomment-6387852516) |
| [109](#c109) | 2026-10-24 | [**Full target-distance band: waits on a flat chasing baseline, which does not exist yet.** Nothing was trained or landed for this box.](https://github.com/bddap/rl/issues/351#issuecomment-6387909445) |
| [110](#c110) | 2026-10-24 | [**Pre-registration: sampled-vs-mean sweep of the entropy-floor checkpoints** (first bounded step of *Flat mean-policy chasing baseline*; no training).](https://github.com/bddap/rl/issues/351#issuecomment-6388229137) |
| [111](#c111) | 2026-10-24 | [**Sampled-vs-mean sweep: the policy has not learned to close.** [Evidence](https://github.com/bddap/rl/tree/6a7abcf/docs/evidence/chase-archaeology/sampled-v…](https://github.com/bddap/rl/issues/351#issuecomment-6388532097) |
| [112](#c112) | 2026-10-24 | [**Pre-registration: long-distance closing A/B** ([protocol](https://github.com/bddap/rl/tree/029f8e1/docs/evidence/chase-archaeology/long-distance-closing-ab…](https://github.com/bddap/rl/issues/351#issuecomment-6389834106) |

## The issue body before the rewrite

Directive (verbatim): "Those weights didn't work iirc. Gotta go back further to see a Sally that actually chased. Pre terrain, pre eval script even."

Era lines: chase-eval landed 79d11ae (2026-06-30), terrain rl#281 (2026-07-16). So the target era is ≤ Jun 29. Pre-2026-06-29 source history did not survive the history rewrite — the earliest surviving commit is 9e3d3b4 (06-29) — so the era pin is the **archived binary** `rl-demo-3ddcf6f` (commit 3ddcf6f, 07-01, in-history), which loads the Jun-28 checkpoint format. All three candidates were re-verified **today** (2026-08-06) by rendering 20 s flat-floor chases (`RL_TARGET_BALL=1`, lavapipe headless) under that binary. Evidence committed at [`docs/evidence/chase-archaeology/`](https://github.com/bddap/rl/tree/main/docs/evidence/chase-archaeology) (5b678cd); checkpoints archived durably at `archive/sally377-jun28-chase/` on the training host.

#### Ranked shortlist

**1. `brain.bin` — Jun 28 17:38, it≈500 after the 2× energy tax (job-377 AFTER brain)** — paired normalizer. Fresh render: closes from spawn and ends with the ball at her claws. Era evidence (job 377, Jun 28): natural ball-chase rollout 670 ticks, "reach intact, mesh is real Sally", mean|drive|≈0.82.
![sally377 fresh render](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/sally377-recheck-tiles.png)

**2. `brain-pre.bin` — ≤ Jun 27, pre-energy-penalty (the "d20bcdd" brain)** — paired normalizer. Fresh render: reaches the ball mid-clip and stays with it. Era clip (`era-jun28-before_chase.mp4`) also shows a full chase.
![pre-penalty fresh render](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/prepenalty-paired-tiles.png)

**3. `brain-reach1.0-iter1046.bin` — Jun 28 03:19, the run's reach-rate-1.0 peak at iter 1046** — caveat: its paired normalizer didn't survive; rendered with the closest (Jul-01 lineage) one. Fresh render: approaches and holds ~1 body-length off the ball without closing the last bit — plausibly the normalizer mismatch, not the brain.
![reach1.0 fresh render](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/reach1.0-iter1046-tiles.png)

Clips (20 s each): [sally377-recheck.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/sally377-recheck.mp4) · [prepenalty-paired.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/prepenalty-paired.mp4) · [reach1.0-iter1046.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/reach1.0-iter1046.mp4) · era originals `era-jun28-{before,after}_chase.mp4`.

Everything later that survives (mlp256 Jul 3+, mlp512x3 Jul 6+, fitbody Jul 18+) is post-eval-script and mostly post-terrain — checked and excluded by construction, per the directive's era dating. Not promoting anything; promotion is a separate call. To eyeball a candidate: `archive/sally377-jun28-chase/render.sh <ckpt-dir> out.mp4 20 640 360` (uses the archived era binary env).


#### Increment checklist (bisect d20bcdd1→main; background in the Stage-1 comment below)

Note superseding the history claim above: full history IS reachable on origin/main; recovered baseline = `d20bcdd1` (Stage-1 comment).

- [x] Probe A launch — cold canary on main, `--terrain flat --band-max-m 9`, floor −1.0, 24 h via `canary-stop-bound`; decide GPU co-residency with Probe B (mlp256 is small — co-run if it fits); post launch evidence (pin, command, bound) here
- [x] Probe B launch — cold canary on baseline `d20bcdd1`, era-native config (flat, band ≤9, mlp256); build the pin, verify its eval/proxy path (backport `5a897b90` or use the render + train.log proxy), launch 24 h; post launch evidence here
- [x] Probe A assess — wait until the bound ends, then record sustained-1m crossing tick, integrity hard-fail count, end-of-bound 20 s render; post green/red/confounded verdict here
- [x] Probe B assess — wait until the bound ends; era proxy (band advance past rung 1 at reach 0.6) + `rl-demo-3ddcf6f` render; upward-trend no-cross gets ONE 24 h extension before a red call; post verdict here
- [x] Decision-tree step — read the A/B verdicts, pick the next probe per the Stage-1 tree (flag-flip singles on main / boundary `4cdf934c` / bound escalation on B), launch it, and APPEND its assess item as the next unchecked box (repeat this launch+assess pattern until one change is convicted)
- [x] Probe C assess — wait until the bound ends (2026-08-12 02:52 UTC); judge TRAIN-SIDE only (chase-eval invalid for flat probes, and this pin is pre-`5a897b90`): `[bearing-reach]`/per-iter reach on the flat ≤9 m band (era health reach ~0.6 within ~1000 iters; Probe A red was 0.00–0.05) + end-of-bound rollout-world render; upward-trend no-cross gets ONE 24 h extension; hard-fail N/A (quiet-reset era); post the verdict here — green convicts the 07-28..08-04 physics/integrity cluster (next: singles within it), red moves the boundary to `981c00ee`-era (07-06, pre-REACH-0.2) — then APPEND the next decision-tree/probe item as the next unchecked box (repeat until one change is convicted)
- [x] Decision-tree step — Probe C RED: the 07-28..08-04 physics/integrity cluster is acquitted as sole cause (C's pin predates all of it and is red anyway); boundary moves to `981c00ee`-era (07-06, pre-REACH-0.2). Launch Probe D: cold 24 h canary on a `981c00ee`-era pin, flat ≤9 m band (terrain doesn't exist there — verify the band lever natively or backport `f49957ca`), same rollout shape, `canary-stop-bound`; judge train-side reach + rollout-world render (pin's evaluator is pre-`5a897b90` — chase-eval invalid); post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe D assess — wait until the bound ends (2026-08-13 12:20 UTC); judge TRAIN-SIDE only (chase-eval doubly invalid: flat probe + pre-`5a897b90` evaluator): per-iter reach on the flat ≤9 m band under era REACH 0.8 (era health reach ~0.6 within ~1000 iters; B crossed rung 1 at iter 317; A/C red were 0.00–0.08) + end-of-bound render with the pin's own flat-native `rl-demo`; upward-trend no-cross gets ONE 24 h extension; hard-fail N/A (quiet-reset era); post the verdict here — GREEN convicts the `8bb5009`..`4cdf934c` span (REACH 0.2 prime suspect; next: flip REACH alone), RED pushes into the 06-29..07-10 window (OU noise/σ-anneal, curriculum deletion, mlp512x3, unclamped obs) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe D RED, qualified (48 h/73M ticks: reach plateau ~0.4, one transient 0.63 window, never sustained 0.6; extension spent — vs A/C flatline 0.00–0.08): the `d20bcdd1`..`0bf9076e` window DEGRADES cold learning; a harder breaker remains in `8bb5009`..main (unacquitted). Launch Probe E — bisect midpoint: cold 24 h canary on the `3ddcf6f` pin (07-01: post 06-30 cluster [OU noise+σ-anneal, curriculum deletion, unclamped obs, GAE tail/Huber, target-KL], pre-mlp512x3), era-native mlp256, flat ≤9 band, same rollout shape, `canary-stop-bound`; judge TRAIN-SIDE reach (pre-`5a897b90` — chase-eval invalid) + end-of-bound render (`rl-demo-3ddcf6f` is era-verified, loads this format); GREEN convicts 07-04..07-06 (`56754c32` mlp512x3 arch swap prime suspect), RED convicts the 06-30 cluster (OU noise/σ-anneal first single); post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe E assess — wait until the bound ends (2026-08-16 15:13:31 UTC); judge TRAIN-SIDE only (chase-eval doesn't exist at this rev — pre-`79d11ae`): per-iter reach on the flat ≤9 m band under era REACH 0.8 (era health reach ~0.6 within ~1000 iters; B rung-1 at iter 317; A/C red 0.00–0.08; D plateau ~0.4 with one transient 0.63) + end-of-bound render with the era-verified `rl-demo-3ddcf6f` (same commit — loads the format by construction); upward-trend no-cross gets ONE 24 h extension; hard-fail N/A (quiet-reset era); post the verdict here — GREEN (sustained ≥0.6) convicts 07-04..07-06 (`56754c32` mlp512x3 arch swap prime suspect; next: arch single), RED (plateau ≤~0.4) convicts the 06-30 cluster (OU noise/σ-anneal first single) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe E GREEN (reach 0.6 sustained by ~3k iters, 0.85–0.87 final; end-of-bound render chases, mean|drive| 0.80): the 06-30 cluster (OU noise/σ-anneal, curriculum deletion, unclamped obs, GAE tail/Huber, target-KL) is ACQUITTED — all in E's pin. With D's qualified RED on the 07-06 pin, the degradation window narrows to `3ddcf6f`..`0bf9076e` (07-01..07-06): `56754c32`+1 (07-04, mlp256 culled → mlp512x3) prime suspect; `981c00ee`+1 (07-06, keep-best on chase-eval) selection-only/low. Launch Probe F — arch single-flip: cold 24 h canary on the `3ddcf6f` pin with ONLY the mlp512x3 swap backported (`56754c32`), flat ≤9 band, same rollout shape, `canary-stop-bound`; judge train-side reach against E's own curve (0.6 by ~3k, 0.85 end — same pin, same infra, arch the only diff); DEGRADED convicts mlp512x3 cold learning, HEALTHY acquits the arch → suspicion moves to `981c00ee` keep-best / remaining 07-06 diffs; post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe F assess — wait until the bound ends (2026-08-18 02:13:07 UTC); judge TRAIN-SIDE only, **against Probe E's own curve** (same `3ddcf6f` pin + infra, arch the only diff: E sustained 0.6 by ~2.7k iters, 0.83–0.87 from 10k; D plateau ~0.4; A/C red 0.00–0.08): per-iter reach on the flat ≤9 m band under era REACH 0.8 + end-of-bound render with the PATCHED `rl-demo-3ddcf6f-mlp512x3` (stock era demo cannot load a 512×3 ckpt); upward-trend no-cross gets ONE 24 h extension; hard-fail N/A (quiet-reset era); post the verdict here — DEGRADED (≤~0.4–0.5 plateau) CONVICTS `56754c32` mlp512x3 cold learning (endgame next: minimal A/B confirmation, then fix-or-revert on main), HEALTHY (E-like) acquits the arch → suspicion moves to `981c00ee`+1 keep-best / the remaining 07-01..07-06 diffs — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe F DEGRADED (reach plateau 0.26–0.35 from ~4k iters through 12.7k iters / 39.1M ticks, vs E on the identical pin+infra: 0.6 by ~2.5k, 0.83–0.87 from 10k; end-of-bound render armed, mean|drive| 0.869, no chase): **`56754c32` mlp512x3 CONVICTED for cold-start learning** — and E-vs-F IS the minimal A/B (arch the only diff), so the endgame confirmation half is already in hand. Conviction is PARTIAL: D/F plateau ~0.3–0.4 while A/C flatline 0.00–0.08 — at least one more breaker lives in `8bb5009`..main, unacquitted. Next probe (serves both endgame halves): main, `--terrain flat --band-max-m 9`, floor −1.0, cold 24 h `canary-stop-bound`, with the trunk restored to mlp256 via the arch registry (`3415332` seam) — E-like learning clears the remaining window AND green-lights the mlp256 fix-or-revert on main; a plateau/flatline isolates the second breaker with the arch controlled; launch it, post evidence, then APPEND its assess item as the next unchecked box
- [x] Probe G assess — wait until the bound ends (2026-08-19 04:35:41 UTC); judge TRAIN-SIDE only (chase-eval INVALID for flat probes — evals stay canonical-GCR, struck at Probe A): per-iter/per-episode reach on the flat ≤9 m band under main's REACH 0.2 — the LIKE-FOR-LIKE yardstick is Probe A (same rev family, same 0.2 m criterion, arch the only diff: flatline 0.01–0.05 over 19.2M ticks); E (0.6 by ~2.5k, 0.83–0.87 from 10k) and D/F (degraded 0.3–0.4) are directional only (REACH 0.8 era) + end-of-bound render with `rl-demo-966b0ab` (canonical-GCR ground — off-distribution for a flat-trained policy, weight as on A); log integrity hard-fail count (rl#343 live at this rev; >0 with no liftoff = confounded, not falsified); upward-trend no-cross gets ONE 24 h extension; post the verdict here — CLEAR LEARNING vs A green-lights the mlp256 fix-or-revert on main AND clears the 07-10..main window on this task shape; plateau/flatline ISOLATES the second breaker with the arch controlled (`8bb5009` REACH 0.2 first single) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe G RED (flatline 0.009–0.021 ≈ A's 0.01–0.05, arch controlled, 0 hard-fails): the second breaker in `8bb5009`..main dominates cold learning regardless of arch; mlp512x3's conviction stays era-scoped and the mlp256 default-flip stays on hold. Launch Probe H — REACH single: Probe G's exact recipe (main, `--arch mlp256 --terrain flat --band-max-m 9`, floor −1.0, cold 24 h `canary-stop-bound`) with REACH_RADIUS 0.2→0.8 as the ONLY change (flag or build-local patch as with F; smoke-verify 0.8 is in the artifact — note the reach metric then matches E's definition directly, so E's 0.6-by-~2.5k becomes a like-for-like yardstick); GREEN convicts `8bb5009` REACH 0.2 for cold-start learning on main, RED acquits REACH and pushes to the remaining 07-10..main window (baked-collider regen `2760e243`, env swap `27c1376f`, band/obs/terrain cluster, physics cluster); also check H's sps at launch — G held 264 sps where A held ~800 on the same recipe, unexplained; post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe H assess — wait until the bound ends (2026-08-20 12:06:59 UTC); judge TRAIN-SIDE only (chase-eval INVALID for flat probes — evals stay canonical-GCR, struck at Probe A): per-iter/per-episode reach on the flat ≤9 m band under REACH 0.8 — like-for-like yardsticks are now E (same 0.8 criterion: 0.6 by ~2.5k iters, 0.83–0.87 from 10k) and B (rung-1 at iter 317); A/C/G (0.01–0.08, REACH 0.2) and D/F (0.3–0.4) are directional; + end-of-bound render with `rl-demo-966b0ab` (canonical-GCR ground — off-distribution for a flat-trained policy, weight as on A/G); log integrity hard-fail count (rl#343 live; >0 with no liftoff = confounded, not falsified); note G/H hold ~270 sps vs A's ~800 so the bound buys ~7M ticks — judge on the per-iter axis (identical 6144 transitions/update; G's 2233 iters were past every liftoff precedent); upward-trend no-cross gets ONE 24 h extension; post the verdict here — GREEN (E-like, sustained ≥0.6) CONVICTS `8bb5009` REACH 0.2 for cold-start learning on main (endgame: fix-or-revert reach), RED acquits REACH and pushes to the remaining 07-10..main window (baked-collider regen `2760e243`, env swap `27c1376f`, band/obs/terrain cluster, physics cluster) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe H VOID (instrument wedge, no REACH verdict): relaunch as Probe H2 — same recipe as H (`966b0ab` tree + build-local REACH_RADIUS 0.8, `--arch mlp256 --terrain flat --band-max-m 9`, floor −1.0, cold, 24 h `canary-stop-bound`) with the periodic chase-eval NEUTRALIZED: it is canonical-GCR (struck as a ruler for flat probes at Probe A) and its 2nd invocation deadlocked H's learner at 13:28 UTC after ~10/32 locales, starving the run 22.6 h; G completed 24/24 evals on the same tree under REACH 0.2, so the hang is reach-0.8-correlated or nondeterministic — root-cause optional, neutralization mandatory. Prefer LANDING the skip on main (`--terrain flat` ⇒ no canonical-GCR eval — the eval is documented as invalid for flat rollouts, so this is strategic deletion, not a hack) over a build-local patch; `[bearing-reach]`/per-iter train telemetry stays the ruler. Post launch evidence, then APPEND its assess item (same H split: GREEN convicts `8bb5009` REACH 0.2, RED acquits and pushes to the remaining 07-10..main window) as the next unchecked box
- [x] Probe H2 assess — wait until the bound ends (2026-08-21 13:18:02 UTC); judge TRAIN-SIDE only (moot by construction now: the periodic chase-eval is SKIPPED for flat runs since `2809138` — the wedge is unreachable): per-iter/per-episode reach on the flat ≤9 m band under REACH 0.8 — like-for-like yardsticks are E (same 0.8 criterion: 0.6 by ~2.5k iters, 0.83–0.87 from 10k) and B (rung-1 at iter 317); A/C/G (0.01–0.08, REACH 0.2) and D/F (0.3–0.4) are directional; + end-of-bound render with a `2809138`-family `rl-demo` (canonical-GCR ground — off-distribution for a flat-trained policy, weight as on A/G); log integrity hard-fail count (rl#343 live; >0 with no liftoff = confounded, not falsified); note H2 holds ~920 sps steady — the G/H ~270-sps anomaly did NOT reproduce, so the bound buys ~25M ticks and the anomaly needs re-dating against `966b0ab`..`2809138`; upward-trend no-cross gets ONE 24 h extension; post the verdict here — GREEN (E-like, sustained ≥0.6) CONVICTS `8bb5009` REACH 0.2 for cold-start learning on main (endgame: fix-or-revert reach), RED acquits REACH and pushes to the remaining 07-10..main window (baked-collider regen `2760e243`, env swap `27c1376f`, band/obs/terrain cluster, physics cluster) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe H2 RED (floor 0.08–0.12 with era REACH 0.8 restored, 10.9k iters = 4.3× E's liftoff horizon; verdict clean on hard-fail-free segment 1): `8bb5009` REACH 0.2 ACQUITTED; the floor-maker landed in 07-06..main and acts regardless of arch (D on mlp512x3 still reached ~0.4 vs A/G/H2 floor). Launch Probe I — bisect at the 07-17 collider-regen boundary: cold 24 h canary on a `2760e243` pin (post-collider-regen; pre-`932add78` terrain-ground, pre-`8eda58af` band128, pre-`98392819` obs117, pre-env-swap — flat arena + band ≤9 are the rev's DEFAULTS, mlp512x3 the only arch, REACH 0.2 live), same rollout shape (`--workers 6 --envs 2 --horizon 512`, floor −1.0), `canary-stop-bound`; NEUTRALIZE the periodic chase-eval build-locally (struck as a flat-probe ruler at Probe A; H-wedge precedent — the `2809138` skip doesn't exist at this rev; keep-best keyed on it since `981c00ee` and the locale fix `5a897b90` is 08-01, so the pin's eval is invalid anyway); verify pre-launch that flat/band-≤9 defaults hold natively (no `--terrain`/`--band-max-m` until `f49957ca`) and smoke cold iter 0 + checkpoint save; yardsticks: D/F mlp512x3 plateau 0.3–0.4 = arch-normal cold learning, A/G/H2 0.01–0.12 = the floor; PLATEAU (≥~0.3) acquits everything ≤07-17 incl. the collider regen and pushes to 07-18..main (obs relayout/+25 channels `98392819`, contact stiffness `19a15264`, shoves `074c1263`, env swap `27c1376f`, physics cluster), FLOOR convicts the 07-10..07-17 span with collider regen `2760e243` prime (REACH acquitted; close-mix/digest/terrain-opt-in low); post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe I assess — wait until the bound ends (2026-08-23 12:19:37 UTC); judge TRAIN-SIDE only (the periodic chase-eval is NEUTRALIZED build-locally — canonical-GCR, struck as a flat-probe ruler at Probe A; the skip preserves the `[bearing-reach]` flush cadence, verified in-log): per-iter/per-episode reach on the flat ≤9 m band under REACH 0.2 — like-for-like yardsticks are A/G/H2 (same 0.2 criterion, floor 0.01–0.12; A also mlp512x3); D/F (0.3–0.4 plateau, mlp512x3 under era REACH 0.8) and E/B (0.8-criterion greens) are directional; + end-of-bound render, preferring the pin's own `rl-demo` (flat is native at this rev — an IN-distribution render, unlike A/G/H2's canonical-GCR clips; fall back to `rl-demo-966b0ab` weighted as on A/G); hard-fail N/A by construction (quiet-reset era — rl#343 is 08-03); sps at launch ~900 steady (the G/H ~270 anomaly did not reproduce on this pin); upward-trend no-cross gets ONE 24 h extension; post the verdict here — PLATEAU (≥~0.3, D/F-like) acquits everything ≤07-17 incl. the collider regen and pushes the breaker to 07-18..main (obs relayout/+25 channels `98392819`, band128 `8eda58af`, terrain-ground `932add78`, contact stiffness `19a15264`, shoves `074c1263`, env swap `27c1376f`, physics cluster), FLOOR (0.01–0.12, A/G/H2-like) convicts the 07-10..07-17 span with collider regen `2760e243` prime (REACH acquitted; close-mix/digest/terrain-opt-in low) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe I NOT-FLOOR (24 h + the ONE extension = 73.3M ticks / 23.5k iters: reach 0.02 → ~0.045 → 0.07 → 0.14–0.15 late surge, still rising; end-of-bound render with the pin's own flat-native `rl-demo` CHASES — closes from spawn, stays at the ball; vs A's trendless 0.01–0.05 same criterion+arch): everything ≤07-17 incl. the collider regen `2760e243` is ACQUITTED of the floor. With Probe C's floor on `4cdf934c` (07-27), the floor-maker narrows to **`2760e243..4cdf934c` (07-18..07-27)** — candidates: obs relayout/+25 channels `98392819`, band128 `8eda58af`, terrain-ground `932add78`, contact stiffness `19a15264`, shoves `074c1263`, env swap `27c1376f` (launch job: date/ancestry-verify each against the span before picking). Launch Probe J — bisect midpoint of that span: cold 24 h canary, flat + band ≤9 m (native or via `f49957ca` flags per the pin's ancestry — verify pre-launch), REACH 0.2, mlp512x3, same rollout shape (`--workers 6 --envs 2 --horizon 512`, floor −1.0), `canary-stop-bound`; NEUTRALIZE the periodic chase-eval build-locally as on Probe I (the `2809138` flat-skip doesn't exist mid-span; keep the `[bearing-reach]` flush cadence, verify in-log); smoke cold iter 0 + checkpoint save; yardsticks: Probe I's climb (0.045 by 10M ticks, 0.07+ by 40M, 0.14 late) = healthy-for-this-era, A/C floor 0.01–0.05 trendless = broken; I-LIKE CLIMB acquits ≤midpoint and narrows to (midpoint..07-27], FLOOR convicts (07-17..midpoint]; post launch evidence, then APPEND its assess item as the next unchecked box. NOTE the acquittal is of the FLOOR only — I's climb is far below era-healthy liftoff (B rung-1 at iter 317; E 0.6 by ~2.5k iters), consistent with the mlp512x3 slow-learning conviction (E-vs-F) still applying at this rev.
- [x] Probe J assess — wait until the bound ends (2026-08-26 01:48:11 UTC); judge TRAIN-SIDE only (periodic chase-eval NEUTRALIZED build-locally as on Probe I — canonical-GCR, struck as a flat-probe ruler at Probe A; the skip preserves the `[bearing-reach]` flush cadence, verified in-log at launch): per-iter/per-episode reach on the flat ≤9 m band under REACH 0.2 — yardsticks: Probe I's climb (0.045 by 10M ticks, 0.07+ by 40M, 0.14–0.15 late, still rising) = healthy-for-this-span, A/C/G/H2 floor 0.01–0.12 trendless = broken; + end-of-bound render, preferring an `rl-demo` built from the probe's own patched tree (pin `69d39c8` + `build-local.patch` in the instance dir — flat is reachable via the backported `--terrain` seam; fall back to `rl-demo-966b0ab` weighted as on A/G); hard-fail N/A by construction (quiet-reset era — rl#343 is 08-03); sps at launch ~884 steady; upward-trend no-cross gets ONE 24 h extension; post the verdict here — I-LIKE CLIMB acquits ≤`69d39c8` and narrows the floor-maker to (`69d39c8`..`4cdf934c`] (contact stiffness `19a1526`, shoves `074c126`, env swap `27c1376` prime + the 07-20..27 tail), FLOOR convicts (`2760e243`..`69d39c8`] (terrain-ground `932add7`, band128 `8eda58a`, obs relayout/+25ch `9839281`, rl#236 claw-only-down) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe J NOT-FLOOR (24 h + the ONE extension = 72.6M ticks / 23.6k iters, zero restarts in bound 1, one verified-warm resume into the extension: reach 0.013 floor → 0.038–0.059 → extension climb to run-high windows 0.097/0.120/0.121, extension episode-weighted 0.075 vs 0.038 in bound 1 — tracks Probe I's curve nearly point-for-point at every horizon; end-of-bound render on the pin's own `rl-demo` with flat wired build-local (in-distribution, seed 351): armed — mean|drive| 0.853, effort 30.0 ≈ I's 0.890/32.0 — but no close in the seeded 20 s episode; weighted secondary, J's late reach 0.10–0.12 trails I's 0.14–0.15): everything ≤`69d39c8` (07-19) ACQUITTED of the floor — terrain-ground `932add7`, band128 `8eda58a`, obs relayout/+25ch `9839281`, rl#236 claw-only-down all cleared. With Probe C's floor on `4cdf934c` (07-27), the floor-maker narrows to **(`69d39c8`..`4cdf934c`] (07-19..07-27)** — candidates: contact stiffness `19a1526` (07-19 23:41), shoves `074c1263` (07-20 03:01), env swap `27c1376f` (07-20 05:53), + the 07-20..27 tail (mostly vehicles/MP, low prior). Launch Probe K — bisect: suggested pin `074c1263` (post contact-stiffness + shoves, pre env-swap — splits the candidate set 2/1; launch job date/ancestry-verifies before picking), cold 24 h canary, flat + band ≤9 m via the J-precedent `f49957ca` backport (re-resolve conflicts against this pin's ancestry), REACH 0.2, mlp512x3, same rollout shape (`--workers 6 --envs 2 --horizon 512`, floor −1.0), `canary-stop-bound`; NEUTRALIZE the periodic chase-eval build-locally as on I/J (keep the `[bearing-reach]` flush cadence, verify in-log); smoke cold iter 0 + checkpoint save; yardsticks: I/J climb (~0.045 by 10M ticks, 0.07+ by 40M, 0.10–0.15 late) = healthy-for-this-span, A/C/G/H2 floor 0.01–0.12 trendless = broken; I/J-LIKE CLIMB acquits ≤pin → env swap `27c1376f` + tail prime, FLOOR convicts (`69d39c8`..pin] (contact stiffness / shoves); post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe K assess — wait until the bound ends (2026-08-28 21:09:36 UTC); judge TRAIN-SIDE only (periodic chase-eval NEUTRALIZED build-locally as on Probes I/J — canonical-GCR, struck as a flat-probe ruler at Probe A; the skip preserves the `[bearing-reach]` flush cadence, verified in-log at launch): per-iter/per-episode reach on the flat ≤9 m band under REACH 0.2 — yardsticks: I/J climb (~0.045 by 10M ticks, 0.07+ by 40M, 0.10–0.15 late) = healthy-for-this-span, A/C/G/H2 floor 0.01–0.12 trendless = broken; + end-of-bound render, preferring an `rl-demo` built from the probe's own patched tree (pin `074c1263` + `build-local.patch` in the instance dir — flat is reachable via the backported `--terrain` seam; fall back to `rl-demo-966b0ab` weighted as on A/G); hard-fail N/A by construction (quiet-reset era — rl#343 is 08-03); sps at launch ~890 steady (~30M ticks in-bound; the G/H ~270 anomaly absent); upward-trend no-cross gets ONE 24 h extension; post the verdict here — I/J-LIKE CLIMB acquits ≤`074c126` (contact stiffness `19a1526` + shoves `074c126` cleared) and narrows the floor-maker to (`074c126`..`4cdf934c`] (env swap `27c1376f` prime + the 07-20..27 tail), FLOOR convicts (`69d39c8`..`074c126`] (contact stiffness / shoves — singles next) — then APPEND the next decision-tree/probe item as the next unchecked box
- [x] Decision-tree step — Probe K FLOOR (24 h + the ONE extension = 72.1M ticks / 23.5k iters, one verified-warm mid-extension resume: reach trendless 0.028–0.054 the whole run, bound 1 weighted 0.0345 / extension 0.0394, ending 0.031–0.038 — at the horizons where I/J separated (55–72M ticks: 0.07–0.15) K reads 0.034–0.043; end-of-bound render (pin's own `rl-demo`, flat wired, seed 351): armed — mean|drive| 0.962, effort 35.9 — but static, no close): **the floor-maker is in (`69d39c8`..`074c1263`] — contact stiffness `19a1526` (07-19 23:41) or shoves `074c1263` (07-20 03:01)** (the span's other 7 commits are rl#298 stage 2–4 seam prep + rl#299 evidence, low prior); env swap `27c1376f` + the 07-20..27 tail are no longer needed to explain the floor. Launch Probe L — single: pin `19a1526` (post contact-stiffness, pre-shoves; date/ancestry-verify before picking), cold canary, flat + band ≤9 m via the K-precedent `f49957ca` build-local backport (re-resolve against this pin's ancestry), REACH 0.2, mlp512x3, same rollout shape (`--workers 6 --envs 2 --horizon 512`, floor −1.0), `canary-stop-bound`; NEUTRALIZE the periodic chase-eval build-locally as on I/J/K (keep the `[bearing-reach]` flush cadence, verify in-log); smoke cold iter 0 + checkpoint save; BUDGET: separation shows only at ~55M+ ticks (I/J/K precedent — all three read 0.03–0.05 at ≤36M), so plan the 24 h bound + the ONE extension from the start; FLOOR convicts contact stiffness `19a1526` (endgame next: minimal A/B + fix-or-revert on main), I/J-LIKE CLIMB acquits it → convicts shoves `074c1263` (same endgame on the shove path); post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe L assess — wait until the bound ends (2026-09-02 00:35:36 UTC — a 48 h bound: the 24 h + the ONE extension were PRE-COMMITTED at launch per the budget rule, so NO further extension is available); judge TRAIN-SIDE only (periodic chase-eval NEUTRALIZED build-locally as on I/J/K — canonical-GCR, struck as a flat-probe ruler at Probe A; the skip preserves the `[bearing-reach]` flush cadence, verified at smoke): per-iter/per-episode reach on the flat ≤9 m band under REACH 0.2 — yardsticks: I/J climb (~0.045 by 10M ticks, 0.07+ by 40M, 0.10–0.15 late) = healthy-for-this-span, A/C/G/H2/K floor 0.01–0.12 trendless = broken; + end-of-bound render, preferring an `rl-demo` built from the probe's own patched tree (pin `19a1526` + `build-local.patch` in the instance dir — flat is reachable via the backported `--terrain` seam; fall back to `rl-demo-966b0ab` weighted as on A/G); hard-fail N/A by construction (quiet-reset era — rl#343 is 08-03); sps at launch ~890 steady (~60M+ ticks in-bound); post the verdict here — FLOOR CONVICTS contact stiffness `19a1526` for the cold-learning floor, with J-vs-L already the minimal A/B (single commit apart) → endgame next: fix-or-revert on main; I/J-LIKE CLIMB acquits it → CONVICTS shoves `074c1263` (K = stiffness+shoves floored; the span's 7 remaining rl#298 seam-prep commits stay low-prior) → endgame on the shove path — then APPEND the next decision-tree/endgame item as the next unchecked box
- [x] Probe M launch — the main-side confirmation half of the shove conviction (L NOT-FLOOR 0.079→0.170/73M, K = L+shoves FLOOR → `074c1263` convicted, fix tracked rl#415): cold canary on current `main`, `--terrain flat --band-max-m 9`, floor −1.0, same rollout shape — Probe A's exact config with shoves NEUTRALIZED (the rl#415 fix if landed by launch time, else `SHOVE_START_PROB=0`/equivalent build-local, delta saved as `build-local.patch`); chase-eval is valid on main flat post-`2809138` — keep it, plus `[bearing-reach]`; smoke cold iter 0 + checkpoint save; BUDGET: 48 h bound PRE-COMMITTED via `canary-stop-bound` (L precedent — separation needs ~55M+ ticks; no further extension); I/J/L-LIKE CLIMB confirms the conviction transfers to main with every latest feature live (then the Endgame re-enable canaries proceed from a working baseline); FLOOR means a second main-side breaker post-`074c1263` hides in A's stack (mlp512x3 slow-learning and the target-height/0.2 m-ruler ceiling are the ranked suspects — mechanism items 2–3) → bisect `074c1263`..main next; post launch evidence, then APPEND its assess item as the next unchecked box
- [x] Probe M assess — wait until the bound ends (2026-09-04 01:46:32 UTC — a 48 h bound: the 24 h + the ONE extension were PRE-COMMITTED at launch per the budget rule, so NO further extension is available); judge TRAIN-SIDE only (main skips the periodic chase-eval on flat by construction — `[best] chase-eval DISABLED` in-log, `[bearing-reach]` flush cadence kept, both verified at smoke): per-iter/per-episode reach on the flat ≤9 m band under REACH 0.2 — yardsticks: I/J/L climb (~0.045 by 10M ticks, 0.07+ by 40M, 0.10–0.17 late; L read 0.079 in its FIRST 5M bucket) = the shove conviction TRANSFERS to main with every latest feature live; A/C/G/H2/K floor 0.01–0.12 trendless = a second main-side breaker post-`074c1263` (Probe A is the direct control: same lineage, shoves ON, 0.01–0.05); + end-of-bound render (an `rl-demo` from main at/after `138d1eb`, flat in-distribution, seed 351, weighted as on L); integrity hard-fail count logged (rl#343 is live on this pin — >0 with no liftoff is CONFOUNDED, not falsified); sps at launch ~940 steady (~75M ticks expected in-bound); post the verdict here — CLIMB → Endgame proceeds on the shove path (land the rl#415 fix on main as the accused-change fix, then the re-enable-feature canaries from a working baseline); FLOOR → bisect `074c1263`..main next (mlp512x3 slow-learning and the target-height/0.2 m-ruler ceiling are the ranked suspects — mechanism items 2–3) — then APPEND the next decision-tree/endgame item as the next unchecked box
- [x] Decision-tree step — Probe M FLOOR (main `138d1eb` + shoves OFF: 0.023→0.056 through 65M, K-band, no chase in the render; the shove fix alone does not restore learning on main). Launch Probe N: cold 48 h canary on the post-shove bisect midpoint `2e5605f` (commit 170/339 of `d577e9c9`..`138d1eb`) + the same 2-line `SHOVE_START_PROB=0` build-local patch (or the rl#415 fix if landed), Probe A recipe (`--terrain flat --band-max-m 9`, mlp512x3, floor −1.0), `canary-stop-bound` 48 h pre-committed (24 h + the one extension); judge TRAIN-SIDE reach (chase-eval disabled on flat by construction) — L-like (first 5M bucket ≥0.06, ≥0.07 by 40M) moves the breaker to `2e5605f`..`138d1eb`, M-like (≤0.056 through 65M) to `d577e9c9`..`2e5605f`; wire the pin's `rl-demo` ground flat build-local for the end-of-bound render; post launch evidence (pin, command, bound), then APPEND its assess item as the next unchecked box
- [x] Probe N assess — wait until the bound ends (2026-09-06 03:11:01 UTC); judge TRAIN-SIDE only (chase-eval skipped build-local — `reach.awk` 5M-tick episode-weighted buckets on `canary-probe-n-2e5605f/train.log`, 0.2 m ruler): L-LIKE (first 5M bucket ≥0.06, ≥0.07 by 40M, 0.10–0.17 late) moves the breaker to `2e5605f`..`138d1eb`, M-LIKE (≤0.056 through 65M, K-band) to `d577e9c9`..`2e5605f`; log the integrity hard-fail count; end-of-bound 20 s render with the pre-built `rl-demo-2e5605f-flat-shoveoff` (seed 351, trainer asset root) → mp4 + tiles + reach curve landed under `docs/evidence/chase-archaeology/`; post the verdict here, then APPEND the next bisect-midpoint launch item (same recipe, ~85 commits either side) as the next unchecked box
- [x] Probe O launch — bisect midpoint of `2e5605f`..`138d1eb` (169 commits; Probe N CLIMBED, M floored): cold 48 h canary on `40fde25` (08-17, commit 85/169; if it does not build or its rollout world is broken, take the nearest buildable neighbor and say which) + the same build-local patch as N (`SHOVE_START_PROB` → 0.0; periodic canonical-GCR chase-eval skipped in `BestKeeper::maybe_snapshot` if this rev still runs it on flat — keep the `[bearing-reach]` flush cadence), Probe A's recipe (`--terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512`, `RL_LOG_STD_FLOOR_END=-1.0`, fresh init), `canary-stop-bound` 48 h pre-committed (24 h + the ONE extension; the assess gets none); pre-build the pin's own `rl-demo` flat-wired + shove-off for the end-of-bound render; smoke iter 0→10 on GPU, verify the running trainer's exe = the pinned binary, post launch evidence (pin, sha256, command, bound readback) here, then APPEND its assess item as the next unchecked box — split: N-LIKE (out of the ≤0.056 band by 30M, ≥0.07 at 40M; also report the 50M+ trend — N faded 0.116→0.068) → breaker in `40fde25`..`138d1eb`; M-LIKE (≤0.056 through 65M) → in `2e5605f`..`40fde25` — DONE (job 3402): `40fde25` builds and its world is intact, launched 2026-09-06 22:18:32 UTC, bound to 09-08 22:18:37 UTC; ~250 sps ⇒ ~43M ticks in-bound (caveat carried into the assess)
- [x] Probe O assess — wait until the bound ends (2026-09-08 22:18:37 UTC; 48 h pre-committed = 24 h + the ONE extension, NO further extension); judge TRAIN-SIDE only (chase-eval skipped build-local — `reach.awk` 5M-tick episode-weighted buckets on `canary-probe-o-40fde25/train.log`, 0.2 m ruler): N-LIKE (out of the ≤0.056 band by 30M, ≥0.07 at 40M; also report the end-of-run trend — N faded 0.116→0.068 past 50M, so a climb-then-collapse is not read as clean) moves the breaker to `40fde25`..`138d1eb` (84 commits), M-LIKE (≤0.056 through the full bound, K-band) to `2e5605f`..`40fde25` (85 commits); NOTE this pin rolls out at ~250 sps (G/H-era slowdown), so the bound yields ~43M ticks, not 65M — judge on the per-tick axis against N/M at the same tick marks; log the integrity hard-fail count; end-of-bound 20 s render with the pre-built `rl-demo-40fde25-flat-shoveoff` (seed 351, trainer asset root) → mp4 + tiles + reach curve landed under `docs/evidence/chase-archaeology/`; post the verdict here, then APPEND the next bisect-midpoint launch item (same recipe, ~42 commits either side) as the next unchecked box
- [x] Decision-tree step — Probe O FLOOR (flatline 0.006–0.008 through its 21M-tick bound, a statue on render; deeper than every prior floor probe; throughput 240 sps vs N's 720 — the slowdown landed in the same range): the second main-side breaker sits in `2e5605f`..`40fde25` (85 commits, 08-12 → 08-17). Launch Probe P — cold canary on `7184c17` (commit 43/85, 08-15, the LAST commit before the bevy 0.19 / rapier 0.35 bump `4a0a3a6`..`88be0c1`) + the same shove-off + chase-eval-skip build-local patch, Probe A's recipe (flat ≤9 m band, mlp512x3, floor −1.0). **Bound in TICKS, not hours:** smoke the pin's sps first and pre-commit a wall bound that lands ≥40M ticks (ticks/s = sps/2; 48 h at 720 sps = 62M, at 240 sps only 21M — O's launch conflated samples with ticks); state the bound and the projected tick count in the launch evidence. Record the smoke sps in the launch comment: ≥600 sps at `7184c17` puts the slowdown in `4a0a3a6`..`40fde25` too. Split for the assess: N-LIKE (≥0.02 by 5–10M, out of the ≤0.056 band by 30M, ≥0.07 at 40M) → breaker in `4a0a3a6`..`40fde25` (commits 44–85: the rapier 0.35 bump + rl#340 stages 7–10) — next single: `88be0c1` (the bump landing); O-LIKE (≤0.01 through 20M, drift <0.6 m) → in `12a6a77`..`7184c17` (commits 1–43: rl#340 stages 2–5 contact/foot-stiffness work) — next midpoint ~commit 22; M-LIKE (0.02–0.056 through the bound, never separating) → two breakers, one each side — single `88be0c1` first. Post launch evidence, then APPEND its assess item as the next unchecked box — DONE (job 3494): pin `7184c17` builds, world intact; smoke 248–250 sps ⇒ ~127 ticks/s, so the slowdown is in commits 1–43 (`12a6a77`..`7184c17`), not the bump; launched 2026-09-08 23:07:56 UTC, bound 96 h (345600 s) pre-committed → stops 2026-09-12 23:08:16 UTC, projected ~44M ticks
- [x] Probe P assess — wait until the bound ends (2026-09-12 23:08:16 UTC; 96 h pre-committed in TICKS — ~127 ticks/s ⇒ ~44M ticks — NO extension); judge TRAIN-SIDE only (chase-eval skipped build-local — `reach.awk` 5M-tick episode-weighted buckets on `canary-probe-p-7184c17/train.log`, 0.2 m ruler), on the per-tick axis against N/O/M at the same tick marks: N-LIKE (≥0.02 by 5–10M, out of the ≤0.056 band by 30M, ≥0.07 at 40M; report the end-of-run trend — N faded 0.116→0.068 past 50M) → breaker in `4a0a3a6`..`40fde25` (commits 44–85: the rapier 0.35 bump + rl#340 stages 7–10) — next: single `88be0c1` (the bump landing); O-LIKE (≤0.01 through 20M, drift <0.6 m, statue on render) → in `12a6a77`..`7184c17` (commits 1–43: rl#340 stages 2–5 contact/foot-stiffness work; the 720→250 sps slowdown is ALSO in this span — flag any commit that moved both) — next: midpoint ~commit 22; M-LIKE (0.02–0.056 through the bound, never separating) → two breakers, one each side — single `88be0c1` first; log the integrity hard-fail count; end-of-bound 20 s render with the pre-built `rl-demo-7184c17-flat-shoveoff` (seed 351, trainer asset root) → mp4 + tiles + reach curve landed under `docs/evidence/chase-archaeology/`; post the verdict here, then APPEND the next launch item as the next unchecked box — DONE (job 3497): read at the 40M mark (41.2M ticks; 96 h bound left to run out): 5M buckets 0.008/0.007/0.008/0.011/0.015/0.021/0.022/0.022, drift 0.46→0.91 m, reward −5.8→+8.8, hard-fails 0 ⇒ M-LIKE at the low edge (N excluded at 5–10M and 40M; O excluded past 20M); render no chase (DRIVE_STATS 0.478/11.68); evidence b120283c88f7; verdict https://github.com/bddap/rl/issues/351#issuecomment-5648208263
- [x] Decision-tree step — Probe P M-LIKE, low edge (0.008 through 15M, plateau 0.021–0.022 from 25M, never separating; no chase on render): two breakers — breaker 1 in `12a6a77`..`7184c17` (commits 1–43, rl#340 stages 2–5; the 5× N→P drop AND the 720→250 sps slowdown), breaker 2 in `4a0a3a6`..`40fde25` (commits 44–85; the P→O slow-climb→flatline drop, O censored at 21M). Launch Probe Q — the pre-registered single: cold canary on `88be0c1` (commit 48/85, 08-15: the bevy 0.19 / rapier 0.35 bump `4a0a3a6` + the rl#212 pins that landed with it; if it does not build or its rollout world is broken, take the nearest buildable neighbor and say which) + the same shove-off + chase-eval-skip build-local patches (P/O archive: `canary-probe-p-7184c17/build-local*.patch`), Probe A recipe (`--terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512`, `RL_LOG_STD_FLOOR_END=-1.0`, fresh init). Bound in TICKS: smoke sps first, pre-commit a wall bound that lands ≥40M ticks (~96 h at ~250 sps), `canary-stop-bound`, NO extension; pre-build the pin's own flat-wired shove-off `rl-demo` for the render; verify the running trainer's exe = the pinned binary; post launch evidence (pin, sha256, command, bound readback, smoke sps), then APPEND its assess item as the next unchecked box — assess reads at the 30M and 40M tick marks (wait to each, not to the bound), same `reach.awk` ruler on the per-tick axis: O-LIKE (≤0.01 through 30M, drift <0.6 m, reward never positive) convicts the bump `4a0a3a6`..`88be0c1` as breaker 2 → next: revert-test it on top of `40fde25`; P-LIKE (≥0.015 by 25–30M, drift ≥0.7 m, reward positive by ~3M ticks) puts breaker 2 in commits 49–85 (rl#340 stages 7–10) → next midpoint ~commit 67; either way breaker 1's bisect (midpoint ~commit 22 of `12a6a77`..`7184c17`, watching sps as the co-signal) is the box after
- [x] Probe Q assess — wait to the **30M tick mark** (~2026-09-16 00:00Z at ~110 ticks/s; re-derive from `canary-probe-q-88be0c1/ckpt/ticks.txt`), post an early read, then defer to **40M** (~2026-09-17 02:00Z) for the verdict — NOT to the bound (2026-09-17 20:18:37Z, 120 h, NO extension). Same `reach.awk` ruler (5M buckets, episode-weighted, 0.2 m) on the per-tick axis + first-2000 vs last-2000 iter reach/drift/reward + integrity hard-fail count + any warm-resume seam; end-of-bound render with `rl-demo-88be0c1-flat-shoveoff` (seed 351, 20 s, `RL_TARGET_BALL=1`, lavapipe, `RUST_LOG=info` to see the `play: loaded checkpoint` line) + `DRIVE_STATS`; evidence under `docs/evidence/chase-archaeology/probe-q-88be0c1-*`. **O-LIKE** (≤0.01 through 30M, drift <0.6 m, reward never positive) convicts the bump `4a0a3a6`..`88be0c1` as breaker 2 → next box: revert-test the bump on top of `40fde25`; **P-LIKE** (≥0.015 by 25–30M, drift ≥0.7 m, reward positive by ~3M ticks) puts breaker 2 in commits 49–85 (rl#340 stages 7–10) → next box: midpoint ~commit 67 (`git rev-list --reverse 2e5605f..40fde25 | sed -n 67p`). Either way, the box after that is breaker 1's bisect (midpoint ~commit 22 of `12a6a77`..`7184c17`, sps as the co-signal). Post the verdict, then APPEND the next decision-tree item as the next unchecked box
- [x] Decision-tree step — Probe Q O-LIKE through 40M (≤0.010 every 5M bucket, reward never positive by bucket; no chase; drive 0.205/effort 2.42): **breaker 2 is the bevy 0.19 / rapier 0.35 bump landing `4a0a3a6`..`88be0c1`**. Launch Probe R — revert-test the bump on top of `40fde25`: build a diagnostic tree at `40fde25` with the bump/rl#212 pin set reverted to the `7184c17` physics stack (archive the exact build-local patch and verify the resulting Cargo/physics versions; if a literal revert conflicts with stages 7–10, port the pre-bump dependency and solver/contact settings explicitly and document the reconciliation), plus the same shove-off + chase-eval-skip patches. Use Probe A’s cold flat ≤9 m/mlp512x3/floor −1.0 recipe; smoke sps and checkpoint save; pre-build the matching flat-wired demo; set a tick-sized bound reaching ≥40M with no extension. P-LIKE recovery (≥0.015 by 25–30M, reward positive, ≥0.02 at 40M) confirms the bump mechanism; O-LIKE (≤0.01 through 40M, negative reward) moves breaker 2 into commits 49–85 despite Q. Post launch evidence, then append the assess box; after this revert-test, return to breaker 1’s midpoint (~commit 22 of `12a6a77`..`7184c17`, with sps as co-signal).
- [x] Probe R assess — read `canary-probe-r-40fde25-revertbump` at the **30M and 40M tick marks** (not the 120 h wall bound; NO extension) using the same 5M-bucket `reach.awk` ruler, first-2000 vs last-2000 reach/drift/reward, reward-positive bucket fractions, integrity hard-fail count, and any warm-resume seam. Render the 40M checkpoint with `rl-demo-40fde25-revertbump-flat-shoveoff` (seed 351, 20 s, `RL_TARGET_BALL=1`, lavapipe, `RUST_LOG=info` checkpoint-load proof) and report `DRIVE_STATS`. **P-LIKE recovery** (≥0.015 by 25–30M, reward positive, ≥0.02 at 40M) confirms the bevy 0.19 / rapier 0.35 bump mechanism; **O-LIKE** (≤0.01 through 40M, negative reward) moves breaker 2 into commits 49–85 despite Q. Then return to breaker 1’s midpoint (~commit 22 of `12a6a77`..`7184c17`, with sps as co-signal) and append its launch item.
- [x] Probe S launch — breaker 1 content split at `c9f2428` (commit 31 of 43 in `2e5605f..7184c17`: after rl#340 stage 2's 30 Hz contact stiffness + solver-iteration guard, before stage 3's carapace drag `bd5e7c7`/`5aefee4`). Only commits 30–31 touch `crab-world/src/physics/`, only 35–36 touch body aero/spawn; 1–29 are render/audio/UI/voice/plane tuning plus one collider check. Same recipe as N–R: build-local shoves off + periodic chase-eval skip, `--terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512`, `RL_LOG_STD_FLOOR_END=-1.0`, cold. Record the smoke's steady sps first (N 720, P ~250): it places the slowdown on its own. Bound in ticks for a ≥20M read (~45 h at P's ~124 ticks/s, ~16 h at N's); pre-register reads at 10M and 20M: **N-LIKE** (≥0.02 by 5–10M, ≥0.04 at 20M) ⇒ breaker 1 in commits 32–43 (stage 3 drag prime); **P-LIKE** (≤0.012 through 20M) ⇒ in 1–31 (stage 2 prime; next single `b11ef09`, commit 29). Post launch evidence, then APPEND its assess item as the next unchecked box — DONE (job 4561): pin `c9f2428` builds unmodified, world intact, stage 2's 8/4/4 iterations + ζ10 live in-log; smoke 245 sps steady ⇒ ~122 ticks/s, so the 720→250 slowdown is in commits 1–31; launched 2026-09-23 23:55:28 UTC, bound 60 h (216000 s) pre-committed → stops 2026-09-26 11:55:31 UTC, projected ~26M ticks, NO extension; launch evidence https://github.com/bddap/rl/issues/351#issuecomment-5805041095
- [x] Probe S assess — read `canary-probe-s-c9f2428` at the **10M and 20M tick marks** (wait to each; ETA from `canary-probe-s-c9f2428/ckpt/ticks.txt` — ~2026-09-24 22:40Z and ~09-25 21:30Z at ~122 ticks/s, ~09-25 01:00Z and ~09-26 02:00Z at 111), not the 60 h wall bound (2026-09-26 11:55:31Z, NO extension). Same `reach.awk` ruler (5M buckets, episode-weighted, 0.2 m) on the per-tick axis against N/P/R at the same marks; first-2000 vs last-2000 iter reach/drift/reward; reward-positive iteration fractions per bucket; integrity hard-fail count; any warm-resume seam. **N-LIKE** (≥0.02 by 5–10M, ≥0.04 at 20M; N 0.028/0.046/0.053/0.047) ⇒ breaker 1 in commits 32–43, where stage 3's carapace drag `bd5e7c7`/`5aefee4` (35–36) is the only physics/body change → next box: single `5aefee4` (commit 36); **P-LIKE** (≤0.012 through 20M; P 0.008/0.007/0.008/0.011) ⇒ in 1–31, stage 2 `96be084` (commit 30) prime → next box: single `b11ef09` (commit 29, stage 2's parent), whose smoke sps also tests whether `96be084` owns the 720→250 slowdown; a 20M read between the two boxes is reported as such, not forced into one. Render the checkpoint at the 20M read (the trainer keeps only its latest set) with `rl-demo-c9f2428-flat-shoveoff` (seed 351, 20 s, `RL_TARGET_BALL=1`, lavapipe, `RUST_LOG=info` for the `play: loaded checkpoint` line) + `DRIVE_STATS`; evidence under `docs/evidence/chase-archaeology/probe-s-c9f2428-*`. Post the verdict, then APPEND the next launch item as the next unchecked box — DONE (job 4565): P-LIKE — reach 0.006/0.006/0.008/0.011 at 5/10/15/20M, on P's curve (P 0.008/0.007/0.008/0.011; N 0.028/0.046/0.053/0.047); 0 hard-fails, one cold start, no seam; breaker 1 lies in commits 1–31 and stage 3's drag (32–43) is acquitted; 20M render no chase (`DRIVE_STATS` 0.363/7.44); evidence `217379e`; verdict https://github.com/bddap/rl/issues/351#issuecomment-5842304022
- [x] Probe T launch — single `b11ef09` (commit 29 of 43 in `2e5605f..7184c17`, stage 2's parent): Probe S's pin minus exactly commits 30–31 (`96be084` rl#340 stage 2: `PHYSICS_SUBSTEPS` 2→4, contact damping ratio 5→10 at 30 Hz, solver iterations 8/4/4; `c9f2428` its iteration guard). Same recipe as N–S: build-local shoves off + periodic chase-eval skip, `--terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512`, `RL_LOG_STD_FLOOR_END=-1.0`, cold. Record the smoke's steady sps first (N 720, S 245): N-level puts the 720→250 slowdown on stage 2, S-level puts it in 1–29. Bound in ticks for a ≥20M read, sized from the smoke's ticks/s, NO extension; pre-register reads at 10M and 20M on the same `reach.awk` ruler: **N-LIKE** (≥0.02 by 5–10M, ≥0.04 at 20M; N 0.028/0.046/0.053/0.047) ⇒ stage 2 (`96be084`/`c9f2428`) is breaker 1, and T-vs-S is itself the minimal A/B (commits 30–31 the only diff); **P-LIKE** (≤0.012 through 20M; S 0.006/0.006/0.008/0.011) ⇒ breaker 1 lies in 1–29, which touch no `crab-world/src/physics/` file → next: bisect 1–29; a 20M read between the two is reported as such. Pre-build the matching flat-wired demo (`rl-demo-b11ef09-flat-shoveoff`). Post launch evidence, then APPEND its assess item as the next unchecked box — DONE (job 4881): pin `b11ef09` builds unmodified, world intact, pre-stage-2 plant live in-log (ζ5, 4/1/1 iterations; `PHYSICS_SUBSTEPS` 2 in source); slowdown placed on stage 2: T costs N's CPU per tick (23.4 vs 23.6 ms; S 73.2) and runs 3.0× S live (620–661 vs 207–214 sps in one slice); launched 2026-09-26 03:14:50 UTC, bound 48 h (172800 s) pre-committed → stops 2026-09-28 03:14:59 UTC, 20M lands at ≥116 ticks/s (188 live while S co-runs), NO extension; launch evidence https://github.com/bddap/rl/issues/351#issuecomment-5842822175
- [x] Probe T assess — read `canary-probe-t-b11ef09` at the **10M and 20M tick marks** (wait to each; ETA from `canary-probe-t-b11ef09/ckpt/ticks.txt` — ~190–300 ticks/s while S co-runs to 2026-09-26 11:55Z, ~250–330 solo after: ~09-26 12:30–16:30Z and ~09-26 21:00Z–09-27 03:30Z), not the 48 h wall bound (2026-09-28 03:14:59Z, NO extension). Same `reach.awk` ruler (`awk -v B=5000000 [-v CUT=<ticks>] -f canary-probe-t-b11ef09/reach.awk train.log`: 5M buckets, episode-weighted, 0.2 m) on the per-tick axis against N/S at the same marks; first-2000 vs last-2000 iter reach/drift/reward; reward-positive iteration fractions per bucket; integrity hard-fail count; any warm-resume seam; T's solo steady sps once S's bound has passed. **N-LIKE** (≥0.02 by 5–10M, ≥0.04 at 20M; N 0.028/0.046/0.053/0.047) ⇒ stage 2 (`96be084`/`c9f2428`) is breaker 1, with T-vs-S as its minimal A/B (commits 30–31 the only diff); both breakers are then convicted (breaker 2 = the bevy 0.19 / rapier 0.35 bump, Q/R) and the existing Endgame box is next; **P-LIKE** (≤0.012 through 20M; S 0.006/0.006/0.008/0.011) ⇒ breaker 1 lies in commits 1–29, which touch nothing under `crab-world/src/physics/` or `training/` → next: bisect 1–29, content-guided (crab-world: controls/instrument, sky/moon, ground-look shaders, play mode, plane aero `vehicle.rs` 18–19, collider-check fixture 26, wav 27); a 20M read between the two is reported as such, not forced into one. Render the checkpoint at the 20M read (snapshot the set first — the trainer keeps only its latest) with `rl-demo-b11ef09-flat-shoveoff` (seed 351, 20 s, `RL_TARGET_BALL=1`, lavapipe, `RUST_LOG=info` for the `play: loaded checkpoint` line) + `DRIVE_STATS`; evidence under `docs/evidence/chase-archaeology/probe-t-b11ef09-*`. Post the verdict; unless N-LIKE, APPEND the next launch item as the next unchecked box, before Endgame — DONE (job 4909): N-LIKE — reach 0.027/0.051/0.054/0.085 at 5/10/15/20M (N 0.028/0.046/0.053/0.047; S 0.006/0.006/0.008/0.011); 0 hard-fails, one cold start, no seam; solo 684 sps (S 225–229); breaker 1 = rl#340 stage 2 (`96be084`/`c9f2428`), T-vs-S its minimal A/B, so with breaker 2 (the bump, Q/R) both are convicted and Endgame is next — no box appended; 20M render: she closes on the ball by ~9 s, no claw-tip touch (`DRIVE_STATS` 0.573/15.75); evidence `a2dc721`; verdict https://github.com/bddap/rl/issues/351#issuecomment-5849787113
- [x] Endgame — minimal A/B confirmation of each convicted change, `--shove-prob` stated explicitly on both arms; fix-or-revert on main; then one re-enable-feature canary box per remaining feature, added above the Ask box, until training works with scalable terrain + new sensors. From the thread: "the endgame A/B should state the flag explicitly on both arms." and the convictions: "breaker 2 is the bevy 0.19 / rapier 0.35 bump" (breaker 1 per box 42). — DONE (job 5026): historical confirming A/Bs retained; at main faa21d7, E0 clears the old floor (final reach 0.097), E2 0.083 and E12 0.048 do not improve on control, and E1 is censored by an integrity failure at 36.07M. New floor-remedy revert is obsolete at this pin: retain main, no production change warranted. All four matching 20 s renders completed but none demonstrates chase; deterministic-chase and full-band canary boxes added below, existing terrain/pace/Ask remain open. Explicit shove probability 0 in every arm (flag deleted). Read-only review 8/10. Full verdict: https://github.com/bddap/rl/issues/351#issuecomment-5926592440
- [x] Flat-baseline deterministic chase gap — the Endgame control clears the old training floor (35–40M reach 0.097) but its seed-351, 20 s mean-policy render tumbles then stalls; E1 and E2 also fail to demonstrate pursuit. After rl#20's hull body lands, establish a matching flat, ≤9 m baseline with the new sensors retained; compare deterministic mean-policy and stochastic rollouts against the rendered behavior, record integrity and plant identity, and fix the demonstrated cause or append the next bounded diagnostic. Do not treat stochastic training reach as proof of deterministic chase. Post a successful closing-and-holding render before widening the task. Completed the bounded diagnostic branch: matched instrumentation landed in a9372f7; hull baseline hard-failed at 34,240,512 ticks, and six paired renders did not demonstrate chase. [Verdict and next diagnostic](https://github.com/bddap/rl/issues/351#issuecomment-5961152278).
- [x] Flat short-band integrity reproduction — retain the hull plant, sensors, architecture and ≤9 m flat task; use the last coherent 34,240,512-tick checkpoint from a9372f7 to pre-register a finite seeded rollout reproduction of the right-claw 236 m/s failure. Frozen-checkpoint rollouts are reproduction attempts, not an exact replay of online training. Capture replayable pre-failure physics/actions if the flight recorder is insufficient, isolate the first divergent contact/joint event, and fix only a demonstrated cause with a regression and matched finite rerun; otherwise append the next bounded diagnostic. Do not silently restart or relax integrity thresholds. A successful mean-policy closing-and-holding render remains required before widening. — DONE (job 5983): reproduced, 4 rl#343 trips in ≥6.26M frozen ticks (one right-claw-wrist-only, 110.7 m/s; registered budget stopped at 49.7% once the primary endpoint was met), 7 exact-replay onset captures + lever table landed in `ede952b`; onset = one-tick limit-saturated multibody solve blow-up, not drive-caused, PGS iterations diverge; no lever qualifies as a demonstrated cause under the box constraints → rate A/B appended below. Verdict: https://github.com/bddap/rl/issues/351#issuecomment-5966200102
- [x] Multibody-solve integrity rate A/B on the frozen checkpoint — keep the hull plant, sensors, architecture and flat ≤9 m task. The reproduction box found 4 rl#343 trips in ≥6.26M frozen ticks from the a9372f7 34,240,512-tick checkpoint, and 7 exact-replay onset captures are in `docs/evidence/chase-archaeology/integrity-repro/`. Every onset is one tick in which a limit-saturated multibody solve blows up (21–36 of 38 joints at a limit). It survives zeroed drives, and more PGS iterations make it diverge. (a) Attribution, cheap: add `sally-replay` rows for "all joint limits off" and "terrain contact off" (and, if feasible, the shape swap with masses held fixed) to split the limit solve from contact and from the mass perturbation. (b) Rate A/B: add a solver/substep override to `rl-train repro`, then run the same checkpoint, config and seed per arm at a pre-registered 12 × 786,432 = 9,437,184 ticks (the baseline expects ~6 trips). Arms: shipped; (4,12,3)×2; (2,12,3)×4; plus any lever (a) implicates. Record each arm's ticks/s, since stage 2's (8,4,4)×4 removed every captured kick but was convicted as breaker 1 at ~3× CPU per tick. Keep a lever only if it trips 0 times where the shipped arm trips ≥5. Land it with a regression that replays the captured states, then run a matched cold finite flat ≤9 m rerun and a mean-policy closing-and-holding render. Otherwise append the next bounded diagnostic. Do not relax integrity thresholds or silently restart. — DONE (job 6023): NOT CONVICTABLE. (a) all-limits-off and terrain-contact-off each remove the blow-up (self-contact and drives not needed; shape swaps were already mass-pinned, massΔ 0). (b) at 9,437,184 ticks per arm the shipped arm tripped once (0.11/M), below the pre-registered ≥5; (4,12,3)×2 0 trips at 1.37× CPU/tick, (2,12,3)×4 2 trips at 1.60×. Tooling landed in 114ab54..edeb7db, evidence in d9bcadb. Shipped rate is 6× below the reproduction hunt's 0.64/M, so the next box splits that first. [Verdict](https://github.com/bddap/rl/issues/351#issuecomment-5972121711).
- [x] Trip-rate process/host split — the rate A/B's shipped arm tripped once in 9.44M frozen ticks (0.11/M, 12 single-worker `rl-train repro` processes on 32-vCPU burst VMs), while the reproduction hunt saw 4 in 6.26M (0.64/M, one 12-worker process on the training host): P(≤1) ≈ 0.017. Before any further lever A/B, on the training host with the same checkpoint, config, shipped solver and seed 3510, run (i) one `--workers 12` process and (ii) 12 `--workers 1 --first-worker i` processes, each at 12 × 786,432 ticks; record trips, near misses and ticks/CPU-s. If (i) trips well above (ii), in-process worker coupling raises the trip rate: find the shared state with a regression. If both are near 0.6/M, the host matters: isolate what differs. If both are near 0.1/M, the hunt's rate was chance: re-size the lever A/B from the pooled rate to expect ≥10 shipped trips. Keep the hull plant, sensors, architecture and flat ≤9 m task; do not relax integrity thresholds. — DONE (job 6033): CHANCE. Shipped solver, 12 × 786,432 ticks per arm: (ii) split tripped 1 in 9,437,574, bit-identical on the training host and a burst VM (2,303 digests, the same worker-3 trip at tick 332,678 as the rate A/B's shipped arm, pre-trip snapshot byte-identical), so the host is ruled out; (i) one 12-worker process tripped 2 in 9,438,092 (burst VM; the host's CPU share put it at days, a deviation stated in the verdict). In-process workers diverge only at warm-up (a cold brain from burn's global RNG, https://github.com/bddap/rl/issues/431), so they are independent samples, not a rate mechanism. Pooled independent rate 7 in 25.13M = 0.28/M → the lever A/B is re-sized below. Evidence `a293b63`. [Verdict](https://github.com/bddap/rl/issues/351#issuecomment-5976833153)
- [x] Re-sized multibody-solve lever A/B on the frozen checkpoint — the process/host split found the shipped trip rate on the a9372f7 34,240,512-tick checkpoint to be chance-level, 0.28/M pooled (7 in 25.13M; 0.16/M from the current binary alone), and showed single-worker `rl-train repro` processes are bit-reproducible on any host. Keep the hull plant, sensors, architecture and flat ≤9 m task, seed 3510, `--workers 1 --first-worker i`. Run arms shipped, (4,12,3)×2 and (2,12,3)×4 at 46 × 786,432 = 36,175,872 ticks per arm (≥10 expected shipped trips at 0.28/M, ~6 at 0.16/M). Workers 0–11 of each arm are the rate A/B's `d9bcadb` evidence and are reproduced exactly, so run workers 12–45 only; spot-check that one reused worker reproduces its recorded digest. Record trips, near-miss events and ticks/CPU-s per arm. Keep a lever only if shipped trips ≥8 and the lever's share of the pair's trips is significant at one-sided p < 0.01 (e.g. 8–0, 11–1). Land it with a regression that replays the captured states, then run a matched cold finite flat ≤9 m rerun and a mean-policy closing-and-holding render. Otherwise append the next bounded diagnostic. Do not relax integrity thresholds or silently restart. — DONE (job 6080): CONVICTED. Over 36.2M ticks per arm, shipped tripped 13 times against 2 for each lever (13–2, p = 0.0037). (4,12,3)×2 is the cheaper at 1.33× CPU/tick and ships on main in `73e7b88`. Regression `integrity_captures` replays 21 of 22 committed onsets quiet and is mutation-proven; one carpus-spin onset is a named residual. Headless step p50 rose 6.8 → 10.4 ms. Matched cold rerun `rl-target@canary-hull-flat-outer4-73e7b88` launched 2026-10-04 13:14Z; its assess is the next box. [Verdict](https://github.com/bddap/rl/issues/351#issuecomment-5980377900)
- [x] Outer-4 matched cold rerun assess — read `rl-target@canary-hull-flat-outer4-73e7b88` (`73e7b88`, solver (4,12,3)×2, the 5820 baseline recipe: seed 351, flat ≤9 m, mlp512x3, 6×2×512, floor −1, cold, `--ticks 40000000`, ~306 ticks/s ⇒ ends ~2026-10-06 01:30Z). wait to the 10M mark for an early read, then to the end. Reads: episode-weighted reach in 5M buckets against the 5820 baseline at the same marks; the rl#343 hard-fail count (the baseline hard-failed at 34.24M; any failure stops the run, no silent restart, and its flight recorder plus the onset capture are posted); mean-policy 20 s renders on seeds 351–353 with the pinned `rl-demo` (sha256 `331ce5b2…`, in the instance dir). Kept if the run completes 40M ticks with 0 hard-fails AND a render shows Sally closing on the ball and holding. Otherwise record which half failed and append the next bounded diagnostic. Stochastic training reach is not proof of deterministic chase. — DONE: integrity KEPT, chase NOT demonstrated. 40,000,512 ticks, exit 0, 0 rl#343 hard-fails (baseline failed at 34.24M); 235 warm resumes, no integrity restart. Reach 5M buckets 0.023→0.101, behind the baseline to 30M, past its 0.078–0.085 plateau at 35–40M and still rising. Mean renders on 351–353: 351/352 flip inverted (upright −0.95), 353 upright but creeps 5.63→4.91 m. Render half failed; next diagnostic appended. Evidence `f9b26e0`. [Verdict](https://github.com/bddap/rl/issues/351#issuecomment-6074530946)
- [x] Outer-4 budget continuation — the 40M cold rerun cleared integrity but its mean policy flips inverted (351, 352) or creeps without closing (353), while train reach was still rising at the budget end (35–40M bucket 0.101, the steepest). Test budget alone: warm-continue that same run from its 40,000,512-tick checkpoint (same `73e7b88` binary, solver (4,12,3)×2, seed 351, flat ≤9 m, mlp512x3, 6×2×512, floor −1) to `--ticks 80000000`, changing nothing else. Read the 5M reach buckets with the same ruler, the rl#343 hard-fail count (any failure stops the run, no silent restart), and mean-policy 20 s renders on seeds 351–353 at 60M and 80M with the pinned `rl-demo`. Kept if a mean render closes on the ball and holds it upright; otherwise record whether reach plateaued and whether inversion persists, and append the next bounded diagnostic (inversion recovery or architecture). Stochastic reach is not chase.
- [x] Outer-4 right-carpus residual — the (4,12,3)×2 continuation hard-failed at 48,172,032 ticks on a one-tick LegCarpus(Right, 3) spin (3.4 → 76 m/s, 346 rad/s; `outer4-continuation-48m/hardfail-48172032.txt`), the same shape as the residual `inproc-vm-w0 t720366` excluded when `73e7b88` landed. Keep the hull plant, sensors, architecture, flat ≤9 m task and rl#343 thresholds. Capture onsets with single-worker `repro` on the frozen 48,172,032 checkpoint, which is bit-reproducible; pre-register a tick budget sized to the measured (4,12,3)×2 rate. Find what drives the right-carpus joint to a one-tick spin, and fix it at its cause rather than by raising solver counts further. Land a regression that replays every captured onset, including `t720366`, and passes them on the fixed plant. Mutation check: reverting the fix turns it red. Then post the trip rate before and after.
- [x] Mean-policy inversion and post-40M regression — on the outer-4 run, reach peaked at 35–40M (0.101; 0.127 at 38–40M) and fell to 0.066 by 48M, while mean reward(20) dropped from +9.4 to −4.9. Seed 352 ends on her back at both 40M and 48M; seed 351 stopped flipping at 48M but still did not hold. With the frozen 40,000,512 and 48,172,032 checkpoints and the pinned `rl-demo`, run a pre-registered mean-policy sweep over seeds 351–370: the fraction inverted at 20 s, the fraction that ever right themselves, and the fraction that close within 0.5 m and stay. Read what the reward and observations give an inverted body: is there any gradient toward righting? Post the reading, and append the next bounded change: an inversion-recovery change or an architecture/optimizer change, chosen by the evidence. — DONE (job 7004): inversion recovery chosen. Mean policy, seeds 351–370: inverted at 20 s 3/20 at 40M → 7/20 at 48.17M; righted 0/3 → 3/10, all from shallow tilts (0 of 7 below upright −0.5 recovered); close-and-hold 2/20 → 0/20 (both on the body, not the claws); failures inverted/upright 3/15 → 7/13. The reward (progress 24/m, effort, grab 50) never reads orientation, there is no inversion terminal, about 12% of respawns already start inverted: no direct righting gradient. Optimizer recorded: KL 0.08–0.09/update, ≤2 epochs on 86–92% of iterations after 40M. Upright-shaping A/B appended below the outer-2 box. Evidence `510d48f`. [Verdict](https://github.com/bddap/rl/issues/351#issuecomment-6090609779)
- [x] Outer-2 integrity with the friction fix — rapier `5ec3305` removed the cause of the rl#343 blow-ups (multibody friction: exact 2×2 tangent solve, clamped anchor bias; `integrity-carpus-48m/`: near-miss events 122 → 0 at (4,12,3)×2). Outer 4 was the symptom-level lever and costs the frame budget (driven step p50 6.8 → 10.4 ms). Re-run the pre-registered frozen-checkpoint arms at (2,12,3)×2 and (4,12,3)×2 on the fixed plant, plus the 145-onset regression at (2,12,3)×2. If outer 2 holds integrity, ship it back and re-measure the step profile; otherwise record which onsets still blow up.
- [x] Upright-shaping A/B (inversion recovery) — the inversion sweep (`inversion-sweep/`, mean policy, seeds 351–370) found inverted-at-20 s rising 3/20 → 7/20 from 40M to 48.17M, none of the 7 deep inversions (upright < −0.5) recovering, and a reward that never reads orientation (progress 24/m, effort, grab 50), although about 12% of respawns already start inverted. Add potential-based upright shaping, Φ(s) = k·upright, r += γΦ(s′) − Φ(s). Handle Φ at the grab terminal and at truncation so the term stays a pure potential difference and the grab is not penalised. Pre-register k against the progress scale, with its justification. Run a matched cold A/B on one main pin, which carries the solver the outer-2 box ships: control vs shaping, seed 351, flat ≤9 m, mlp512x3, 6×2×512, floor −1, 40M ticks each. Read, on both arms: the 5M reach buckets, the rl#343 hard-fail count (no silent restart), and this sweep at 40M (`analyze.py`, the same definitions) with an `rl-demo` built at the pin. Keep the shaping only if its arm ends inverted at 20 s on ≤ 1/20 seeds and below control, with close-and-hold and reach not below control. Otherwise revert it and record why. If upright bodies that do not close remain the majority on the kept arm, append the architecture/optimizer box from the sweep's optimizer reading (KL 0.08–0.09 per update, ≤2 epochs on 62–92% of iterations, entropy pinned at the floor). — DONE (job 7058): REVERTED. Matched cold 40M pair on `c19a393`, 0 hard-fails on both arms. k = 12 was inverted at 20 s on 0/20 seeds (control 8/20), closed and held 1/20 (control 0/20) and closed a median 1.35 m (control 0.06 m), but its 35–40M reach was 0.059 against 0.111 (z = 10.9), which failed the reach bar. Reverted in `b6b6bf0`; evidence `2643498`; seed replicate appended below. [Verdict](https://github.com/bddap/rl/issues/351#issuecomment-6125760321)
- [x] Upright-shaping seed replicate — the seed-351 A/B (`upright-shaping-ab/`) took the 40M mean policy from 8/20 to 0/20 inverted at 20 s and median distance closed from 0.06 to 1.35 m, but k = 12's train reach swung (0.210 at 22–24M, 0.040 at 38–40M) and its 35–40M bucket (0.059) fell below control's (0.111, z = 10.9), so it was reverted in `b6b6bf0`. One seed per arm cannot separate that late collapse from seed noise. Re-land the shaping (revert `b6b6bf0`) and run the same matched cold pair, k = 0 vs k = 12, on seeds 352 and 353: four 40M runs on one pin with the `c19a393` recipe. Pre-register before launch. Read per run: 5M reach buckets, rl#343 hard-fails, and the 40M mean sweep (seeds 351–370, `analyze.py`). Keep the shaping only if pooled 30–40M reach over seeds 351–353 is not below control's (one-sided two-proportion test, p < 0.05), and on each new seed the k = 12 sweep is inverted at 20 s on ≤ 1/20 seeds and below control, with close-and-hold not below control. Otherwise revert it again and record why. Either way, append the architecture/optimizer box from the sweep's optimizer reading (KL 0.08–0.09 per update, ≤ 2 epochs on 62–92% of iterations, entropy pinned at the floor), which k = 12's oscillating reach also points to. — DONE: REVERTED. Four cold 40M runs on the `c19a393` binary (seeds 352/353 × k = 0/12), 0 hard-fails. Pooled 30–40M reach over 351–353: k = 12 0.113 vs control 0.103 (not below). The 40M mean sweep failed the per-seed bar: k = 12 inverted at 20 s on 6/20 (352) and 2/20 (353), and close-and-hold 0 vs 5 on 352. Over three seeds, inversion at 20 s fell from 27/60 to 8/60, but close-and-hold went from 6/60 to 2/60, and 50 of 58 shaped failures end upright. Reverted in `24b3c49`; evidence `92c595a`; architecture/optimizer box appended below. [Verdict](https://github.com/bddap/rl/issues/351#issuecomment-6223519208)
- [x] Architecture/optimizer A/B for the upright-but-not-closing mean policy — the upright-shaping seed replicate (`upright-shaping-replicate/`) reverted the shaping, and across seeds 351–353 the 40M mean policy fails mostly upright and not closing: control closes and holds on 6/60 eval seeds, and 50 of the shaped arms' 58 failures end upright. The optimizer reading (`inversion-sweep/optim-4m.txt`) shows per-update KL 0.077–0.091, the update stopped at ≤ 2 epochs on 62–92% of iterations, and entropy pinned at 0.419 (log-std at the −1 floor) from 4M ticks. Pick one bounded change from that reading, such as the learning rate, the target-KL/epoch budget, or the log-std floor schedule, and pre-register it with its justification. Then run a matched cold A/B on one main pin against control on seeds 351–353: flat ≤ 9 m, mlp512x3, 6×2×512, 40M ticks each, the arms differing only by that change. Read per run: the 5M reach buckets, rl#343 hard-fails, and the 40M mean sweep (seeds 351–370, `upright-shaping-ab/analyze.py`). Keep the change only if pooled 30–40M reach over the three seeds is not below control's and pooled close-and-hold is above control's (each one-sided, p < 0.05). Otherwise revert it and record why.
- [ ] (needs: someone to leave one Steam Deck awake and untouched for ~15 min) Outer-4 frame budget on device — `73e7b88` moved the headless driven step p50 from 6.8 to 10.4 ms (`integrity-lever-ab-36m/step-profile.txt`). Measure windowed frame time, step-carrying frames included, on the TV (when idle) and on one awake, idle deck, at a main ≥`73e7b88` against `3ed15e4`. If 60 fps holds, record the reading here. If not, find frame budget elsewhere: the rl#343 integrity fix is not reverted for frames. Then post the measurement and the change.
- [ ] Long-distance closing A/B — the sampled-vs-mean sweep (`sampled-vs-mean-sweep/`, `6a7abcf`) found that the 40M entropy-floor policies have not learned to close: sampled actions hold 4/120 and the mean 4/120 (paired McNemar p = 0.64). Every approach within 0.5 m but one starts within 2.5 m of the ball; from farther out she gains at most about 1 m in 20 s, sampled or not, so the 0.07–0.13 training reach is consistent with short-spawn credit. First, at the A/B's pin, run the scripted pace instrument (`scripted_flail_gait_pace`) to bound how far the plant can travel in 20 s; if it cannot cover the ≤9 m band, record that and append a plant box above this one instead. Otherwise pick ONE learning change from this reading (the reward's travel incentive, the reach criterion, or the spawn-distance distribution within ≤9 m), pre-register it with its justification, and run a matched cold A/B against control on training seeds 351–353: flat ≤9 m, mlp512x3, 6×2×512, 40M ticks, the arms differing only by that change. Read per run: the 5M reach buckets, rl#343 hard-fails, the 40M mean sweep (eval seeds 351–370, `upright-shaping-ab/analyze.py` unchanged) and closing by spawn distance (`sampled-vs-mean-sweep/by-spawn.py`). Keep the change only if pooled mean-policy close-and-hold is above control's (one-sided Fisher p < 0.05) and pooled 30–40M reach is not below control's (one-sided z, p < 0.05); otherwise revert it and record why. Stochastic reach is not chase.
- [ ] (waits: Long-distance closing A/B) Flat mean-policy chasing baseline — the full-band box below starts from a verified flat chasing baseline, and none exists yet: the entropy-floor A/B (`entropy-floor-ab/verdict.md`, `96d6067`) closed and held on 4/60 control and 0/60 treatment mean-policy sweeps although treatment's sampled 30–40M reach was 0.126, and it appended no next diagnostic. Establish the baseline on current main with the shipped plant, sensors, mlp512x3 and flat ≤9 m task: verified means one cold 40M run whose mean policy closes and holds on ≥ 10/20 of eval seeds 351–370 under `upright-shaping-ab/analyze.py` unchanged, with 0 rl#343 hard-fails and its render posted. First bounded step, no training: sweep the six registered 40M entropy-floor checkpoints on seeds 351–370 with sampled actions as well as the mean (restore a sampled-action render mode if main lacks one) and score both with the same analyzer. Sampled holds well above mean holds put the failure in the deterministic policy, so the next change targets the mean action; sampled holds as rare as mean holds mean the policy has not learned to close, so the next change targets learning (budget, reach criterion or reward), chosen from the reading. Pre-register each step, append the next bounded change above this box when a step does not establish the baseline, and check this box only when the bar is met. — First step DONE: sampled actions hold 4/120 against the mean's 4/120 (McNemar p = 0.64), so the policy has not learned to close and the next change targets learning, appended above. Evidence `6a7abcf`.
- [ ] (waits: Flat mean-policy chasing baseline) Re-enable full target-distance band — from the verified flat chasing baseline, change only the ≤9 m band to the current full band, retain the same plant, sensors and architecture, pre-register a finite tick budget and matched evaluation, then post the train-side and end-of-bound render verdict. Keep the wider band only if chase survives; otherwise record the failing feature and next diagnostic. Terrain is the separate amplitude-curriculum canary below.
- [ ] Terrain-amplitude curriculum: a cold run starts at low terrain amplitude and ramps to today's unchanged band as competence appears; kept if Sally then navigates the slopes at today's amplitude (rl#385's slope test turned into a training eval), else the reason recorded here. From the thread (verbatim): "Hoping we can keep terrain near where we are now and eventually Sally will learn to navigate the slopes. She might learn with terrain amplitude turned down then ramp up. Maybe that will work."
- [ ] At the pin the endgame retrain launches on, re-run the scripted pace instrument (`scripted_flail_gait_pace`) and post the reading here. From rl#20: "so re-run this instrument at whatever pin the retrain launches on."
- [ ] Render-video startup visibility — matched seed-351 flat smoke renders begin with Sally absent from the first captured frame, while later frames show her and the policy trace is active. Diagnose camera/asset readiness and make the recorded interval show the intended scene from its first frame; verify with a bounded clip and frame inspection. This is a capture-validity follow-up, not evidence that a render defect causes stalled pursuit.
- [ ] Ask: a cold training run on current main, with as many of the latest features live as possible (at least scalable terrain near today's amplitude and the new sensors), produces a Sally that actually chases: she clears chase-eval against this epic's archaeology baselines, her end-of-run render shows her closing on the target and staying with it, on the slopes too, and the test Steam Decks' demo shows it on-device. From the directive: "Goal is to get training working with as many of our latest features (e.g. scalable terrain, new sensors) as possible." and "Gotta go back further to see a Sally that actually chased."

## Comments

<a id="c1"></a>
### 1. Report (verbatim)

> brain-reach1.0-iter1046.bin is Sally just chilling. Looks like no joint torques. I can confirm that for the other two Sally does chase.
>
> Unfortunate that git history went missing, not sure how that happened. Are you able to scrounge up the source code used to train that version? If so, I'd suggest making a list of all the changes to training since that one, and systematically (preferably efficienly) determining just which change borked training. Though that's just a suggestion you may in your silicon wisdom know a better way. Goal is to get training working with as many of our latest features (e.g. scalable terrain, new sensors) as possible.
>
> Iirc the successful training runs did take a while before Sally started performing that well.
>
> (@bddap-bot , be xhigh before triaging this)

<a id="c2"></a>
### 2. 2026-08-07 00:53Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5210606733)

xhigh bump is landing (job 2231, needs a hub restart). Stage 1 queued (2233): recover the Jun-28 training source — the era binary pin rl-demo-3ddcf6f is the lead — then an ordered list of every training change since, and an efficient bisect plan posted here as a stage checklist. Labeled epic so the sweep drives the follow-on stages. iter1046 noted as a non-chaser.

<a id="c3"></a>
### 3. 2026-08-11 00:17Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5247534132)

#### Probe A assess — verdict: **confounded (red-leaning)**

Bound ran 2026-08-09 18:44 → 08-10 18:44 UTC, stopped on time by `canary-stop-bound`. 5583 iters, **19.24M ticks**.

**Sustained-1m crossing: none.** 24 chase-eval samples (~hourly); max single sample 0.509 m (the final one), max 5-window median **0.382 m** vs the 1.0 m bar. Far pairs reached 0/32 on every sample; train reach 0.00. Full series (m):
`0.402 0.323 0.362 0.195 0.181 0.334 0.250 0.249 0.322 0.337 0.368 0.424 0.312 0.377 0.371 0.389 0.265 0.235 0.279 0.365 0.243 0.382 0.431 0.509`

**Integrity hard-fails: 1** (rl#343) — tick 348,113 (2 h 15 m in): ClawPincer(Right) blow-up at 470.7 rad/s, exit 101; systemd warm-resumed from the last checkpoint 10 s later. The remaining **21.7 h ran clean**.

**End-of-bound render** (pin's own `rl-demo` built at `8012a4f8`, end-of-bound ckpt, seed 351, canonical GCR tile — the demo's only ground since rl#293, same locale family as the eval): policy armed (mean|drive| 0.885, mean effort 31.4) but **no chase** — vigorous aimless drift downhill, camera barely keeps her in frame. [probe-a-8012a4f-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-a-8012a4f-endofbound.mp4) (landed b5995a42)
![probe A end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-a-8012a4f-endofbound-tiles.png)

**Verdict reasoning.** By the plan rule (>0 hard-fails + no liftoff ⇒ confounded, not falsified) this is formally **confounded** — but the confound is minor (one event, ~2% of the bound; vs 29 on `7949759`), and the clean 21.7 h post-resume still oscillated 0.18–0.51 m with no crossing at 19.2M ticks, past the c9db5a2-ext24 slow-cross precedent (18.9M). Read it as red-leaning: flat+band9 flags do NOT rescue main. The last two samples (0.431, 0.509) are the run's two highest — a weak end-trend; the ONE extension (warm-resume + re-arm) is available but not spent here. Decision-tree step: weigh this with Probe B's control verdict; if B is green, the accused set is the un-flaggable remainder (REACH 0.2, physics cluster, env-swap, obs-114, mlp512x3, shoves) and extension of A is probably not worth 24 h.

<a id="c4"></a>
### 4. 2026-08-11 00:31Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5247638450)

Correction (review): the end-of-bound behavior is **falling**, not "drift" — and the render leg is off-distribution: this pin's `rl-demo` renders only the canonical GCR tile, so the clip shows a flat-trained policy on hills it never trained on. It supports "no chase" only weakly. Auditing whether the 24 chase-eval samples shared the terrain mismatch — the red-lean verdict is held at reduced weight until that reads out.

<a id="c5"></a>
### 5. 2026-08-11 00:35Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5247672165)

**Audit result (follow-up to the correction above): the eval instrument never saw the probe's terrain. The eval leg is struck; the verdict survives on the in-distribution ruler.**

- **Code:** `--terrain` changes the ROLLOUT worlds only — documented at the flag (`crab-world/src/lib.rs:218-224`: "The eval and the plant sidecar/digest stay canonical-GCR either way"). The periodic evaluator hardcodes canonical amplitude (`best.rs:161-167` → `run_eval(..., 1.0)`; grid built at `eval.rs:778` via `gcr_with_amplitude`).
- **Log:** 25/25 eval invocations in `canary-probe-a-8012a4f/train.log` print `amplitude 1`, with start coordinates out to ±14 km — impossible on the ±512 m flat training grid. All 24 assessment samples scored a flat-trained policy on canonical GCR at 24 m far-pairs (vs its 9 m band cap). The sustained-1m proxy is therefore **invalid for this probe** — its crossing precedents are GCR-trained canaries judged in-distribution.
- **In-distribution read (train-side, valid):** per-bearing train-episode reach on the probe's own flat ≤9 m band sat at 0.01–0.05 (~76–173 ep/bearing/hr) through 19.2M ticks / 5583 iters; per-iter reach almost always 0.00. Era health is reach ~0.6 with band advance within ~1000 iters. **Probe A stays red-leaning (still confounded by the one hard-fail) — same verdict, valid ruler.**
- **Plan amendment for stage 4+:** `run_eval` is a valid proxy only for GCR probes. Any flat probe (e.g. the flat+band128 flag-flip single) must be judged by train-side reach/band telemetry plus a rollout-world render, never chase-eval. Probe B is unaffected — its plan already judges by the era proxy + era-binary renders.

<a id="c6"></a>
### 6. 2026-08-11 00:48Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5247760466)

#### Probe B assess — verdict: **GREEN**

Bound ran 2026-08-09 19:17 → 08-10 19:17 UTC, stopped on time by `canary-stop-bound`. 7661 iters, **23.54M ticks**.

**Era proxy: PASSED, decisively.** Band advanced past rung 1 (1.5–3.0 m) at **iter 317 / 0.98M ticks (~1.6 h in)**, reach climbing 0.43→0.57→0.71 over the last three rung-1 iters. It then climbed every rung — 3.5–5.0 m by iter 399, and the **9 m cap (7.5–9.0 m) by iter 579 / 1.78M ticks (~2 h in)** — where it spent the remaining 22 h. Final-stretch reach at the cap: 0.83–1.00, mostly **1.00**; final reward(20) ≈ 206. Rung dwell counts (iters): 317 / 82 / 35 / 49 / 37 / 59 / 7083-at-cap.

**End-of-bound render** (`rl-demo-3ddcf6f`, end-of-bound ckpt, flat floor, 20 s): **she chases** — closes on the ball and ends with it at her claws, repeatedly across the clip; mean|drive| 0.742, mean effort 24.1. [probe-b-d20bcdd1-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-b-d20bcdd1-endofbound.mp4) (landed d9080e7d)
![probe B end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-b-d20bcdd1-endofbound-tiles.png)

No extension needed (that rule was for an upward-trend no-cross; B crossed in hour 2). Integrity hard-fails: N/A by construction (quiet-reset era). Confounders: none observed.

**Reading A+B together (for the decision-tree step, next box):** the Jun-28 baseline **learns to chase cold in ~1–2M ticks** — so cold-probe negatives are interpretable, and the report's "took a while" applies to polish, not to liftoff. Probe A (main, flat+band9 flags) sat at reach ≈0.00–0.05 for 19.2M ticks on the same task shape. The accused set is therefore the **un-flaggable remainder**: REACH_RADIUS 0.2, advancing-curriculum deletion, OU noise + σ-floor anneal, mlp512x3, obs-114/unclamped target, env-swap world, shoves, physics cluster. Notably, Probe A's flags cap the band but do NOT restore the advancing curriculum — B's rung ladder (easy-first) is exactly what 3ddcf6f7+3 deleted, making **curriculum deletion + REACH 0.2** the cheapest high-suspicion singles for the next probe.

<a id="c7"></a>
### 7. 2026-08-12 06:40Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5263261832)

#### Probe C assess — verdict: **RED**

Bound ran 2026-08-11 02:51 → 08-12 02:52 UTC, stopped on time by `canary-stop-bound`. 10,520 iters, **32.3M ticks** — 1.7× Probe A's sample.

**Train-side reach: flat zero the whole bound.** Episode-weighted reach 0.032 over 43,070 episodes; first-2000-iter window 0.037 vs last-2000 0.030 — no upward trend (if anything a drift down), so the ONE extension is not spent. Per-iter reach almost always 0.00; final reward(20) median (last 1000 iters) −108. All 23 hourly `[bearing-reach]` samples sat at 0.01–0.08 across every bearing (min-bearing series topped out at 0.03) — indistinguishable from Probe A's red profile, vs era health reach ~0.6 within ~1000 iters (Probe B crossed rung 1 at iter 317). Hard-fails N/A by construction (quiet-reset era); the run consumed its full 24 h wall clock with no incident.

**End-of-bound render — in-distribution this time.** Per the Probe A correction, the clip is on the probe's own rollout world: rl-demo rebuilt from the pin's tree with a one-line diagnostic patch (ArenaWorldPlugin grid gcr→flat 512 m; build-local, not landed). Policy armed (mean|drive| 0.836, mean effort 28.5) but **no chase** — Sally paces in place on the flat and never closes on the ball (never in frame across 20 s). [probe-c-4cdf934c-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-c-4cdf934c-endofbound.mp4) (landed fcf83351)
![probe C end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-c-4cdf934c-endofbound-tiles.png)

**Implication (decision tree, next box appended):** `4cdf934c` (07-27) predates the entire 07-28..08-04 physics/integrity cluster (momentum solver, dampers, drag, hard-fail, implicit-dof fork) and is red anyway — that cluster is **acquitted as the sole cause**. The breakage lives at or before 07-27. Boundary moves to **`981c00ee`-era (07-06, pre-REACH-0.2)**: still in the accused set there — REACH 0.2 (07-10), env-swap (07-20), shoves (07-20), terrain-era collider/contact changes (07-16..19), obs relayouts. Probe D on that pin splits the remainder: green convicts the 07-06..07-27 span (REACH 0.2 the prime suspect), red pushes into the 06-29..07-06 window (OU noise, curriculum deletion, mlp512x3, unclamped obs).

<a id="c8"></a>
### 8. 2026-08-13 21:10Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5286428448)

#### Probe D — 24 h in: upward-trend, no cross → ONE 24 h extension (per the rule)

First bound (2026-08-12 12:20 → 08-13 12:20 UTC, 36.3M ticks, 11 814 iters) ended with **no era-health crossing but a clear learning trend** — unlike A/C (flat 0.00–0.08 throughout):

| iters | reach (ep-weighted) |
|---|---|
| 0–1k | 0.072 |
| 1–4k | 0.10–0.21 |
| 4–8k | 0.11–0.18 |
| 8–9k | 0.307 |
| 9–10k | **0.455** (best 200-iter window 0.530 @ iter 9800) |
| 10–12k | 0.414 → 0.370 |

Threshold is era health ~0.6; best sustained window 0.53 — not crossed, but this is the first cold post-baseline probe that *learns at all* on the flat ≤9 m band. Per the pre-registered rule, red is not callable yet: **extension armed** — resumed from ckpt (brain+normalizer+optimizer restored, ticks continue at 36.3M; verified in-log), `canary-stop-bound` stops **2026-08-14 21:02:34 UTC** (timer readback verified). Caveat noted for the final read: the restart re-bases the σ-anneal epoch, so the extension's first ~5M ticks re-anneal exploration σ from −0.7.

Final verdict (cross/no-cross + end-of-bound render with the pin's own flat-native `rl-demo`) after the extension bound. Assess job re-parked to then.

<a id="c9"></a>
### 9. 2026-08-15 12:20Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5302193567)

#### Probe D verdict: RED (qualified — partial learn, not A/C's flatline)

48 h / 73M ticks on the `0bf9076e` pin (flat ≤9 m, REACH 0.8, mlp512x3, cold). Train-side reach, episode-weighted:

- **First 24 h:** 0.07 → 0.46 (peak bucket iter 9–10k), best 200-iter window 0.53.
- **Extension 24 h:** plateau 0.34–0.52; ONE transient 200-iter window at 0.633 (iter 6400), immediately back to ~0.4. Never sustained the era-health ~0.6 bar (era: sustained within ~1000 iters; B rung-1 at iter 317). Extension allowance spent → **no-cross = RED**.
- End-of-bound render (pin's own flat-native `rl-demo`, final ckpt): upright, driving hard (`DRIVE_STATS ticks=1310 mean_abs_drive=0.915`), reaches the ball mid-clip but doesn't hold it — visual match for reach ~0.4.

![probe D end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-d-endbound-tiles.png)
[probe-d-endbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-d-endbound.mp4) · landed `7444384`. Caveat on the extension read: the ckpt restart re-based the σ-anneal (first ~5M ticks re-noisy) — direction unaffected (plateau, not dip-then-recover).

**Reading:** per the pre-registered split, RED pushes into the 06-29..07-10 window (`d20bcdd1`..`0bf9076e`). The qualifier matters: D *learns partially* (0.4) where A/C on later pins flatlined (0.00–0.08) — so the window's changes **degrade** cold learning (0.6 → 0.4 plateau) while a **harder breaker** still lives in `8bb5009`..main (not acquitted, just not this probe's target). Bisect midpoint next: `3ddcf6f` (07-01) sits after the 06-30 cluster (OU noise + σ-anneal, curriculum deletion, unclamped obs, GAE tail/Huber, target-KL) and before mlp512x3 (07-04) — and its mlp256 + demo binary are already era-verified. Probe E box appended.

<a id="c10"></a>
### 10. 2026-08-17 00:35Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5310533531)

#### Probe E verdict: GREEN

24 h bound complete (08-15 15:13 → 08-16 15:13 UTC), pin `3ddcf6f` (07-01), cold, era-native mlp256, flat ≤9 m band: **18316 iters, 56.3 M ticks, reach climbs 0.22 → 0.85 and stays there.**

Weighted reach per 1k iters: 0.22 → 0.42 → 0.54 → 0.66 (3–4k) → … → 0.83–0.87 (10k+, ~10 000 ep/bucket). Sustained-0.6 crossing ≈ iter 2 700 — slightly slower than era health (~0.6@1000) but unambiguous, vs A/C flatline 0.00–0.08 and D's 0.4 plateau. Final `reward(20)` 88.7.

End-of-bound render (`rl-demo-3ddcf6f`, 20 s flat): active gait, ball at her claws mid-clip, mean|drive| 0.80 (era chaser: 0.82).

![reach curve](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-e-reach-curve.png)
![end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-e-endbound-tiles.png)

Clip: [probe-e-endbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-e-endbound.mp4) · evidence landed at `aefcc8d`.

**Consequence:** the 06-30 cluster (OU noise/σ-anneal, curriculum deletion, unclamped obs, GAE tail/Huber, target-KL) is **acquitted** — all present in this pin. With D's qualified RED on the 07-06 pin, the window narrows to 07-01..07-06: **`56754c32` (mlp256 → mlp512x3) prime suspect**; `981c00ee` (keep-best on chase-eval) selection-only. Probe F (arch single-flip atop `3ddcf6f`) appended as the next box.

<a id="c11"></a>
### 11. 2026-08-18 03:03Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5322994955)

#### Probe F assess — DEGRADED. `56754c32` (mlp256 → mlp512x3) convicted for cold-start learning.

Full bound: 24 h / 12,714 iters / 39.1M ticks (stop timer fired on schedule 2026-08-18 02:13 UTC). Train-side reach (episode-weighted 500-iter buckets, ~2.2–2.6k ep each) vs Probe E — same `3ddcf6f` tree, same infra, arch the only diff:

![E vs F reach](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-f-vs-e-reach.png)

- **F: plateau 0.26–0.35 from ~4k iters, best bucket 0.351, final 0.34.** Never approaches E's 0.6 crossing (~2.5k iters) or 0.83–0.87 tail. No upward trend after 4k (0.311 @ 4k → 0.341 @ 12.5k) — the extension clause doesn't apply; this is the convicting DEGRADED shape, not a slow riser.
- F's plateau matches Probe D (~0.4, on the 07-06 pin that already carries mlp512x3) — coherent.
- End-of-bound render, patched `rl-demo-3ddcf6f-mlp512x3`, era flat floor: policy armed (`DRIVE_STATS ticks=1310 mean_abs_drive=0.869`) but no chase — ball never closed.

![probe F endbound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-f-endbound-tiles.png)

[probe-f-endbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-f-endbound.mp4) · evidence landed `acd9625`.

**E-vs-F is itself the minimal A/B** (single-flip, controlled) — the endgame's confirmation half is done. Caveat kept on the record: the conviction is partial — A/C on later pins flatlined 0.00–0.08, well below the mlp512x3 plateau, so ≥1 more breaker in `8bb5009`..main remains unacquitted. Next box appended: main + mlp256-restored probe, which tests the fix-or-revert AND isolates the second breaker in one run.

<a id="c12"></a>
### 12. 2026-08-18 10:27Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5326904740)

**Arch differential vs the #342 roast — the conviction's mechanism, posted in full at https://github.com/bddap/rl/issues/342#issuecomment-5326903217.**

Short form: no roast-cited constant differs by leaf — the mlp256/mlp512x3 diff is width+depth only (`LOG_STD_INIT`, head init, tanh bound, value-head shape all byte-identical, verified on main `964c378` and the era `3ddcf6f` tree). Under the probe floor (−1.0), `log_std` −1.6 sat below the floor's whole range with its gradient clamp-gated, so **E and F ran bit-identical σ schedules with a dead entropy term for the entire run — F's DEGRADED is not σ collapse; the gap is mean-pathway commitment speed** at mlp256-tuned constants (un-fan-in-scaled head init + per-param Adam on the wider head + third layer, lr unchanged since `d20bcdd1`). Coherent with 5a's warm superiority: cold dynamics, not raw capacity. E-vs-F can't separate head-geometry vs depth vs lr; the discriminator (512x3 cold, fan-in-scaled head) is recorded on #395 if big-net cold start ever matters — Probe G's mlp256 path doesn't need it. Spin-offs: #394 (mechanical fixes post-G), #395 (next-run exploration recipe).

<a id="c13"></a>
### 13. 2026-08-19 10:49Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5341064760)

#### Probe G assess — verdict: **RED (flatline). The second breaker is real — main flatlines with the arch controlled.**

Bound ran 2026-08-18 04:35 → 08-19 04:35:41 UTC, stopped on time by `canary-stop-bound`. 2233 iters, **6.86M ticks**, one uninterrupted process — **0 integrity hard-fails** (vs A's 1): unconfounded.

**Train-side reach (the like-for-like ruler: Probe A, same REACH 0.2): flatline 0.009–0.021.** Episode-weighted 250-iter buckets (~1000 ep each):
`0.018 0.017 0.011 0.012 0.009 0.021 0.021 0.014 0.014`
No trend (first 0.018 → last 0.014); reward(20) drifted −38 → −47. Indistinguishable from A's 0.01–0.05 floor. The extension clause doesn't apply — there is no upward trend to extend.

![G vs A reach](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-g-vs-a-reach.png)

**Throughput caveat:** the run held 264 sps steady from iter 1 (vs A's ~800 — uniform from launch, so not a co-tenant onset; cause not chased here), so the bound bought 6.86M ticks vs A's 19.2M. The per-iter axis is the comparison anyway (identical 6144 transitions/update), and 2233 iters is past every liftoff precedent: B capped the band by iter 579, E sustained 0.6 by ~2.5k, F's plateau was set by ~4k. Zero slope at 2.2k iters is a red, not a censored maybe.

**End-of-bound render** (`rl-demo-966b0ab`, seed 351, canonical-GCR ground — off-distribution for a flat-trained policy, weighted as on A): policy armed (`DRIVE_STATS ticks=1310 mean_abs_drive=0.635 mean_effort=18.1`) but no chase — she tumbles down the slope, ball never engaged. [probe-g-endbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-g-endbound.mp4) · evidence landed `afc59a3f`.

![probe G endbound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-g-endbound-tiles.png)

**Reading — the arch factorial closes:** on the era pin the arch flip was decisive (E 0.83–0.87 vs F 0.26–0.35); on main it moves nothing (G ≈ A, both on the floor). So the `8bb5009`..main breaker(s) dominate cold learning regardless of arch, and mlp512x3's conviction is era-scoped — it degrades, but the main-line flatline is owned by something else. mlp256 stays an opt-in leaf (no green-light to flip `ArchId::DEFAULT`). First single per the tree: **`8bb5009` REACH_RADIUS 0.8→0.2** — decision-tree box appended.

<a id="c14"></a>
### 14. 2026-08-19 15:36Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5344403309)

Correction to the sps note above: A's train.log gives median 515 steps/s (peak 804, n=6263 steady-state samples), not ~800. G held 261; H is holding 273. The anomaly stands (G/H at roughly half A's throughput on the same recipe) and remains unexplained; verdicts are unaffected — they read per-iteration rulers.

<a id="c15"></a>
### 15. 2026-08-20 12:33Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5355932495)

#### Probe H assess — verdict: **VOID (instrument wedge — no REACH verdict; relaunch as H2)**

The stop timer fired on schedule (2026-08-19 12:06:54 → 08-20 12:06:59 UTC), but the run only **trained for the first hour: 98 iters / 0.30M ticks** (`ticks.txt` 301,056 == the iter-97 line). At 13:07:27 UTC the 2nd periodic chase-eval began (32 canonical-GCR far pairs); it rolled ~10 locales and **deadlocked at 13:28:27 UTC mid-locale — the learner then starved for the remaining 22.6 h**: zero log lines after 13:28, no checkpoint writes after 13:07, ~0 CPU (systemd: 4 h 20 m CPU over 24 h wall, all in hour 0–1.4). The process never crashed — it hung; integrity hard-fails: **0**.

**Train-side read (the only ruler — and it is censored):** episode-weighted reach **0.071 over 409 ep** across iters 1–97; iter 0 read reach 1.00/2 ep — the random policy already terminating grabs under 0.8, proving the flip semantically live. 98 iters is far short of every liftoff precedent (B rung-1 at iter 317; E 0.6 at ~2.5k), so this is a **censored non-result, not a red**: `8bb5009` REACH 0.2 is neither convicted nor acquitted. The extension clause doesn't apply (nothing to extend — the run was dead, not slow). End-of-bound render skipped as moot: the last checkpoint is from hour 1.

**The wedge:** the eval leg was already STRUCK as a ruler for flat probes (Probe A amendment); here it graduated from invalid to actively destructive — an in-process canonical-GCR eval hang cost the whole bound. Probe G (same `966b0ab` tree, REACH 0.2) completed 24/24 evals, and A did 25/25, so the hang is reach-0.8-correlated or nondeterministic. H2 must neutralize the eval for flat probes (prefer landing `--terrain flat` ⇒ skip chase-eval on main — the eval is documented as canonical-GCR-only and invalid there, so this is deletion of a wrong instrument, not a workaround); `[bearing-reach]`/per-iter telemetry stays the ruler. Box appended.

<a id="c16"></a>
### 16. 2026-08-20 15:09Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5357804179)

sps anomaly follow-up — the in-train chase-eval is disconfirmed as the cause of the G/H ~270 sps crawl:

- Probe G completed 23 periodic chase-evals and was still uniformly slow: steady-rollout median 261 sps, p25 259 / p75 263 over 2234 iters. A periodic-eval cost would produce a spiky distribution, not flat 259–263.
- Every other flat run with the eval enabled holds ~830–906 sps (flat-s1 881, flat-sigma-s1 891, probe-c 828, probe-f 906).
- H2 (eval-free, main 2809138) holds 820 sps median (steady, n=848).

The consistent story is a build-range effect, not the eval: A (pin 8012a4f, 08-09) ran 515; G/H (966b0ab) ran ~270; H2 (2809138) runs 820. Parsimonious candidate: the expensive solver arrived after 8012a4f and 964c378 ("restore the cheap driven solver", rl#392) removed it again — which would also account for the TV 2 fps episode. d199714 (rayon build_global pin) is the alternate. Not chased further here; recorded so the eventual chase starts from evidence.

<a id="c17"></a>
### 17. 2026-08-21 14:06Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5370893881)

#### Probe H2 assess — verdict: **RED (floor with era REACH restored) — `8bb5009` REACH 0.2 is ACQUITTED as the main-line breaker.**

Full bound: 2026-08-20 13:17:47 → 08-21 13:18:02 UTC, stopped on schedule by `canary-stop-bound`. **10,921 iters / 33,549,312 ticks** (`ticks.txt` == 10,921 × 3072 exactly). Steady rollout mean **785 sps** — A-class throughput held for the whole bound; the G/H ~270-sps anomaly stayed absent (its window remains `966b0ab`..`2809138`, not chased here).

**Train-side reach (the ruler; E is the like-for-like yardstick — same REACH 0.8): floor 0.08–0.12 across all 10.9k iters.** Episode-weighted 500-iter buckets (~2.1k ep each, 46,394 ep total):
`0.093 0.093 0.085 0.079 0.093 0.090 0.091 0.090 0.095 0.102 0.087 0.097 0.088 0.096 0.090 0.095 0.097 0.102 0.096 0.117 0.108 0.116`
E crossed 0.6 by ~2.5k iters and held 0.83–0.87 from 10k; B took rung 1 at iter 317. H2 ran **4.3× E's liftoff horizon** and never left the floor. The tail drift (+0.02 over the last 1.5k buckets, slope ≈ +0.003/1k iters → 0.6 at ~170k iters) is the F-precedent non-trend — the extension clause does not apply. reward(20) flat at −39..−45 throughout.

![H2 vs E reach](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-h2-vs-e-reach.png)

**Integrity hard-fails: 2** (rl#343; both `lifecycle.rs` physics blow-ups — LegCarpus overspeed, at cumulative iters 3337 and 10,063). Each panicked the process; systemd warm-resumed from ckpt within ~10 s (brain+optimizer restored — the log carries exactly one cold-start line), and the panic fires pre-update, so the poisoned in-flight rollout never reached the learner. Per the recorded rule, >0 hard-fails with no liftoff = confounded, not falsified — **but the verdict stands on unconfounded data alone: segment 1 (3337 uninterrupted, hard-fail-free iters — 1.3× E's liftoff horizon) is flat 0.079–0.097.** The full trajectory is continuous across the resumes (dotted lines in the chart) and adds only confirmation. Rate context: 2 hard-fails / 33.5M ticks ≈ A's 1 / 19.2M.

**End-of-bound render** (`rl-demo-966b0ab`, seed 351, canonical-GCR ground — off-distribution for a flat-trained policy, weighted as on A/G; checkpoint format identical by construction — mlp256 `brain.bin` 556,880 B, the G-verified loader): policy armed (`DRIVE_STATS ticks=1310 mean_abs_drive=0.628 mean_effort=18.4` — ≈ G's 0.635/18.1) but no chase: she tumbles down the slope, ball never engaged.
[probe-h2-endbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-h2-endbound.mp4)

![probe H2 endbound tiles](https://raw.githubusercontent.com/bddap/rl/main/docs/evidence/chase-archaeology/probe-h2-endbound-tiles.png)

**Reading:** restoring era REACH 0.8 on main moved the floor only from 0.01–0.08 (A/C/G, REACH 0.2) to 0.09–0.12 — exactly the criterion getting 4× easier, zero learning gained. If REACH 0.2 were THE breaker, this flip had to lift; it didn't. Caveat on the record: H2's pin (`2809138`) sits 11 commits past G's `966b0ab` (incl. `964c378` driven-solver restore, `775f843` spawn clearance), so H2-vs-G isn't a pure single-flip — but a masking new breaker in that span would have to exactly cancel a REACH rescue, and the solver restore moves physics *toward* era behavior; second-order.

**Implication — the bisect continues in 07-10..main with REACH struck from the suspect list:** baked-collider regen `2760e243` (07-17, "a new MDP no live checkpoint can drive"), env swap `27c1376f` (07-20), the 07-18 band/obs/terrain cluster, and the physics cluster. Known brackets: D (07-06 pin, mlp512x3) still learned to ~0.4; A (main, mlp512x3, flags neutralizing ground+band) floored — so the floor-maker landed 07-06..main and acts regardless of arch. Decision-tree box appended: **Probe I bisects that window at the 07-17 collider-regen boundary.**

<a id="c18"></a>
### 18. 2026-08-25 01:24Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5403783019)

#### Probe I assess — verdict: **NOT-FLOOR** (D/F-like slow learning) — everything ≤07-17 ACQUITTED, breaker narrows to `2760e243..4cdf934c` (07-18..07-27)

Extension ran 2026-08-24 00:05 → 08-25 00:05 UTC, stopped on time. Full run: **73.3M ticks / 23,455 iters / 98.5k episodes**, cold from scratch, verified warm across every restart.

**Train-side reach (0.2 m criterion, 500-iter buckets):** 0.02 → flat ~0.045 (10–35M ticks) → 0.07 through the mid-extension → late surge to **0.14–0.15** (~68M), ending 0.10–0.15 and still rising. Extension segment episode-weighted 0.080 vs 0.042 in the first bound. The like-for-like floor exemplar A (same criterion, same arch, main) sat trendless at 0.01–0.05 for 19.2M ticks — Probe I's last full windows exceed the entire A/G/H2 floor band (0.01–0.12) and the shape is a monotone climb, not a flatline.

![Probe I reach curve](https://raw.githubusercontent.com/bddap/rl/f556b1943282/docs/evidence/chase-archaeology/probe-i-2760e243-reach-curve.png)

**End-of-bound render — in-distribution (the pin's own `rl-demo`, flat native, seed 351): she chases.** Closes from a distant spawn and stays at the ball for the rest of the 20 s clip; mean|drive| 0.890, mean effort 32.0. First render in the A-family bisect line that actually chases. [probe-i-2760e243-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-i-2760e243-endofbound.mp4) (landed f556b194)

![Probe I end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/f556b1943282/docs/evidence/chase-archaeology/probe-i-2760e243-endofbound-tiles.png)

**Verdict reasoning.** The split asked FLOOR (convicts 07-10..07-17, collider regen prime) vs PLATEAU ≥0.3 (acquits ≤07-17). The literal ≥0.3 bar was written on D/F's 0.8 m-criterion scale; on the 0.2 m criterion the operative discriminator is floor-shape vs climb, and Probe I unambiguously climbs (and its render chases, where D/F's "armed, no chase" didn't). So: **collider regen `2760e243` and everything ≤07-17 acquitted of the floor.** Combined with Probe C's floor on `4cdf934c` (07-27), the floor-maker lives in **07-18..07-27** — obs relayout/+25ch `98392819`, band128 `8eda58af`, terrain-ground `932add78`, contact stiffness `19a15264`, shoves `074c1263`, env swap `27c1376f` (dates to be ancestry-verified at launch). Probe J (bisect midpoint) appended as the next box.

**Scope note:** the acquittal is of the FLOOR only. I's climb is far below era-healthy liftoff (B rung-1 at iter 317; E 0.6 by ~2.5k iters) — consistent with the E-vs-F mlp512x3 slow-learning conviction still applying at this rev. Charge-guard event: benign (job 2698 — fresh canary at 18% of the prod-era pin; reach decoupled, rl#343 N/A).

<a id="c19"></a>
### 19. 2026-08-31 00:19Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5472191034)

#### Probe K assess — verdict: **FLOOR (trendless at the full I/J budget)** — floor-maker narrows to (`69d39c8`..`074c1263`]: contact stiffness / shoves, singles next

Extension ran 2026-08-28 21:52 → 08-29 21:53:15 UTC, stopped on time by `canary-stop-bound`. Full run: **72.1M ticks / 23.5k iters / 94,342 ep**, cold from scratch; bound 1 zero restarts; one process death mid-extension (~08-29 10:02 UTC) with a verified-warm resume (brain + both normalizers + Adam restored in-log; tick counter continuous at 53.74M; exactly one cold-start line in the whole log). ~845–890 sps steady. Integrity events: **0**; hard-fail N/A by construction (quiet-reset era).

**Train-side reach (0.2 m criterion, 500-iter buckets, episode-weighted):** bound 1 0.029–0.041 (weighted 0.0345) → extension 0.028–0.054 (weighted **0.0394**), ending 0.031–0.038. The run-high is a single 0.054 bucket at ~46M; the last ~20M ticks decline back to 0.028–0.043. At every horizon where I/J separated, K does not: ~55M I/J 0.07–0.08 vs K 0.043; 65–72M I/J 0.10–0.15 vs K 0.034. The interim's watch feature resolved — no rise materialized; the shape is the A/C/G/H2 trendless floor (0.01–0.12 band), not a slow climb.

![Probe K vs J vs I reach curves](https://raw.githubusercontent.com/bddap/rl/68b4d4b619bb/docs/evidence/chase-archaeology/probe-k-074c1263-reach-curve.png)

**End-of-bound render** (the pin's own `rl-demo`, flat wired build-local + the instance `build-local.patch` — in-distribution, seed 351; ckpt body digest verified against the trainer's sally.glb): **armed but no chase** — `DRIVE_STATS mean_abs_drive=0.962 mean_effort=35.9` (above I's chasing 0.890/32.0 in raw magnitude), yet the 20 s tableau is static — she never closes on the ball. Motor activity without directed progress: the floor signature, weighted secondary per the box.
[probe-k-074c1263-endofbound.mp4](https://github.com/bddap/rl/raw/20a5fa73a993/docs/evidence/chase-archaeology/probe-k-074c1263-endofbound.mp4)

![Probe K end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/20a5fa73a993/docs/evidence/chase-archaeology/probe-k-074c1263-endofbound-tiles.png)

**Reading:** with J NOT-FLOOR on `69d39c8` and K FLOOR on `074c1263`, the floor-maker lives in **(`69d39c8`..`074c1263`]** — the named candidates are **contact stiffness `19a1526` (07-19 23:41)** and **shoves `074c1263` (07-20 03:01)**; the span's other 7 commits are rl#298 stage 2–4 seam prep + rl#299 evidence, low prior. Env swap `27c1376f` + the 07-20..27 tail are no longer needed to explain the floor (K's pin predates them and floors anyway). Budget note for the singles: separation shows only at ~55M+ ticks (I/J/K all read alike at ≤36M), so plan bound + the ONE extension from the start.

**Decision-tree box appended: Probe L** — single at `19a1526` (post contact-stiffness, pre-shoves): FLOOR convicts contact stiffness, I/J-LIKE CLIMB acquits it → convicts shoves. Evidence landed: `68b4d4b6` (curve), `20a5fa73` (mp4 + tiles).

<a id="c20"></a>
### 20. Report (verbatim)

> fyi i messed with git history in https://github.com/bddap/rl/issues/414 . you might want to keep a copy of the old main branch so the commits referenced in this issue keep making sense while this issue is still being worked, or some other solution.
>
> sorry

<a id="c21"></a>
### 21. 2026-09-02 01:16Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5502872204)

#### Probe L assess — verdict: **NOT-FLOOR (strongest climb of the series)** — contact stiffness `19a1526` ACQUITTED → random shoves `074c1263` CONVICTED as the floor-maker

Bound ran 2026-08-31 00:35 → 09-02 00:35:36 UTC, stopped on time by `canary-stop-bound` (the single pre-committed 48 h bound — no extension seam). Full run: **73.25M ticks / 23,844 iters / ~102.0k ep**, cold from scratch, **zero restarts** (exactly one cold-start line in the whole log), ~859 sps steady. Integrity events: **0**; hard-fail N/A by construction (quiet-reset era). Progress-glitch drops 358 full-run (~1 per 205k ticks — the stiffness-side contact blow-ups noted in the lateral read; negligible).

**Train-side reach (0.2 m criterion, 5M-tick buckets, episode-weighted):**
`0.079 0.090 0.095 0.087 0.084 0.092 0.114 0.123 0.120 0.124 0.129 0.123 0.146 0.170 0.127(partial)` — no floor phase at all (first bucket 0.079 vs I/J's 0.014–0.023), monotone climb to a 0.170 run-high at 70M, last `[bearing-reach]` windows 0.09–0.22. **Above I and J at every horizon** (55M: L 0.129 vs I/J 0.071–0.073; 65M: L 0.146 vs J 0.097) and 3–4× K everywhere (K trendless 0.028–0.054, weighted 0.039).

![Probe L vs K/J/I reach curves](https://raw.githubusercontent.com/bddap/rl/138d1eb75d4c/docs/evidence/chase-archaeology/probe-l-19a1526-reach-curve.png)

**End-of-bound render** (the pin's own `rl-demo`, flat wired build-local as on J/K — in-distribution, seed 351, trainer's own sally.glb asset root): **she chases** — closes from a distant spawn to the ball at her claws by the clip's end, upright active gait throughout; `DRIVE_STATS mean_abs_drive=0.915 mean_effort=33.4` (≈ I's chasing 0.890/32.0). Second chasing render of the bisect line, at the highest reach yet.
[probe-l-19a1526-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-l-19a1526-endofbound.mp4)

![Probe L end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/138d1eb75d4c/docs/evidence/chase-archaeology/probe-l-19a1526-endofbound-tiles.png)

**Reading — the bisect is resolved.** L is a perfect single (J's pin + ONLY the contact-stiffness commit) and it out-learns J from the first bucket: `19a1526` acquitted — plausibly even beneficial (feet no longer 65 mm inside the floor). K = L + shoves `074c1263` + 7 seam commits whose entire crab-world/src delta is 6 doc-comment lines, and K floors — **random shoves `074c1263` convicted** as the (`69d39c8`..`4cdf934c`] floor-maker, completing the chain: J NOT-FLOOR → K FLOOR → L NOT-FLOOR. Mechanism (2–8× body-weight bursts, return variance ×16, chase gradient drowned) + falsifiers in the 09-01 lateral-read comment; **fix tracked in rl#415** (scale to body weight, default off for cold runs).

**Scope notes:** the shove conviction explains the FLOOR only. L's 0.17 is still far below era-healthy (B rung-1 at iter 317, E 0.6 by ~2.5k iters) — the mlp512x3 slow-learning conviction (E-vs-F) stands, and the target-height/0.2 m-ruler ceiling (mechanism item 3) remains unmeasured speculation.

**Decision-tree box appended: Probe M** — the main-side confirmation half: cold canary on current main with shoves neutralized (the rl#415 fix if landed, else `SHOVE_START_PROB=0` build-local), `--terrain flat --band-max-m 9`, floor −1.0 — Probe A's exact config minus shoves. I/J/L-like climb confirms the conviction transfers to main with every latest feature live (the stated goal); a floor means a second main-side breaker post-`074c1263` still hides in Probe A's stack. 48 h bound pre-committed up front (L precedent: separation needs ~55M+ ticks). Evidence landed: `138d1eb` (curve + mp4 + tiles).

<a id="c22"></a>
### 22. 2026-09-02 02:49Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5503585285)

rl#415 fix landed in 6e0ac5d: shoves are now 0.1–0.4× body weight (Δv ≤ 0.5 m/s) and OFF by default — `rl-train learn --shove-prob <p>` turns them on (1/640 was the old per-tick rate). A cold canary on main now runs unshoved unless the flag is passed; the endgame A/B should state the flag explicitly on both arms. Before/after numbers and test names on https://github.com/bddap/rl/issues/415.

<a id="c23"></a>
### 23. 2026-09-05 16:28Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5553185430)

For the next sanctioned training run — terrain-amplitude curriculum (verbatim):

> Hoping we can keep terrain near where we are now and eventually Sally will learn to navigate the slopes. She might learn with terrain amplitude turned down then ramp up. Maybe that will work.

Shape: start training at low terrain amplitude, ramp toward the current (unchanged) band as competence appears. Context: rl#340 established that zero-input slope hold ≥40° is physically out of reach on the mesh body, so steep slopes are a learned capability by decision — the plant stays as-is. Success signal available when it works: un-ignoring/converting rl#385's slope test into a training eval. Prod training remains off; this is recorded, not implemented.

<a id="c24"></a>
### 24. 2026-09-06 21:43Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5562379444)

#### Probe N assess — verdict: **CLIMB (L-like, qualified)** — `2e5605f` + shoves OFF learns where main + shoves OFF floors; the second main-side breaker sits in `2e5605f`..`138d1eb` (169 commits)

Bound ran 2026-09-04 03:10 → 09-06 03:11:01 UTC, stopped on time by `canary-stop-bound` (48 h pre-committed, no extension). Full run: **62.2M ticks / 20,252 iters / ~84.1k ep**, cold from scratch (exactly one cold-start line), NRestarts 0, ~720 sps steady. Integrity hard-fail (rl#343): **0** — not confounded.

**Train-side reach (0.2 m ruler, 5M-tick buckets, episode-weighted, `reach.awk`):**
`0.028 0.046 0.053 0.047 0.054 0.078 0.089 0.106 0.112 0.116 0.099 0.086 0.068(partial, 2988 ep)` — last 200 iters 0.070; `[bearing-reach]` windows 0.05–0.14 across all eight bearings.

Against the pre-registered split:
- **M-LIKE (≤0.056 through 65M)? No.** N leaves the floor band at 30M (0.078) and stays out through 65M; at 40M it is **0.106 vs M 0.054 / K 0.034** (2×), at 50M **0.116 vs M 0.044** (2.6×). A/K/M never exceed 0.056 in any bucket ≤65M.
- **L-LIKE? Yes on the two later marks, no on the first:** first bucket **0.028** (bar ≥0.06; L 0.079) — N starts in M's band and takes ~25M ticks to separate; 40M **0.106** (bar ≥0.07 ✓); peak **0.116** at 50M (0.10–0.17 band ✓). L was 0.123–0.129 over the same 40–55M stretch, so N tracks L ~0.01–0.02 under once it is up.
- **Qualification — a late fade:** 50M→65M N drops 0.116 → 0.099 → 0.086 → 0.068 while L kept rising (0.129 → 0.146 → 0.170). The last bucket is partial (722 iters) and still above every M/K/A bucket, so the split verdict stands, but the shape is peak-then-decline, not L's monotone climb; a `2e5605f`-side breaker that acts late, or plain cold-run instability, are both open.

![Probe N vs L/M/K/A reach curves](https://raw.githubusercontent.com/bddap/rl/51364319fbe4/docs/evidence/chase-archaeology/probe-n-2e5605f-reach-curve.png)

**End-of-bound render** (the pre-built `rl-demo-2e5605f-flat-shoveoff` — the pin's own demo + shove-off hunk + flat ground build-local; seed 351, trainer asset root, final ckpt at 62.2M): **no chase** — she rights up once, then rolls onto her back by ~8 s and stays inverted for the rest of the clip, legs waving, ball untouched at ~3 m; `DRIVE_STATS mean_abs_drive=0.572 mean_effort=15.8` (M's no-chase: 0.588/16.9; L chasing: 0.915/33.4). The render is at the fade end of the curve (0.068 bucket), so it under-reads N's peak; weighted secondary per the box, and consistent with the late decline (a policy losing its footing, literally).
[probe-n-2e5605f-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-n-2e5605f-endofbound.mp4)

![Probe N end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/51364319fbe4/docs/evidence/chase-archaeology/probe-n-2e5605f-endofbound-tiles.png)

**Reading.** `2e5605f` (08-12) + shoves OFF is on the learning side of the split: it separates from the M/K/A floor by 2× from 30M on, which main + shoves OFF (M) never did in 70M. So the breaker that survives the shove fix landed **after** `2e5605f`: `2e5605f`..`138d1eb`, 169 commits (08-12 → 09-01). Inside it: the rl#353 later stages, rl#372/386 work, the moisture/target-height items, and whatever touched the rollout world after 08-12 — the ranked mechanism suspect (target-height band / 0.2 m-ruler ceiling) stays in range if it landed there. The fade is the caveat carried forward: the next assess should score the 30–50M separation AND the 50M+ trend, so a probe that climbs then collapses is not read as clean.

**Decision-tree box appended: Probe O** — bisect midpoint of `2e5605f`..`138d1eb`: cold canary on `40fde25` (08-17, commit 85/169) + the same shove-off build-local patch (+ chase-eval skip if that rev still runs the periodic GCR eval on flat), Probe A's recipe (flat ≤9 m band, mlp512x3, floor −1.0), 48 h pre-committed. N-LIKE (out of the ≤0.056 band by 30M, ≥0.07 at 40M) → the breaker is in `40fde25`..`138d1eb`; M-LIKE (≤0.056 through 65M) → it is in `2e5605f`..`40fde25`. Evidence landed: `5136431` (curve + mp4 + tiles).

<a id="c25"></a>
### 25. 2026-09-08 22:37Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5592833129)

#### Probe O assess — verdict: **FLOOR (deep flatline)** — `40fde25` + shoves OFF never learns; the second main-side breaker sits in `2e5605f`..`40fde25` (85 commits, 08-12 → 08-17)

Bound ran 2026-09-06 22:18:32 → 09-08 22:18:37 UTC, stopped on time by `canary-stop-bound` (48 h pre-committed, no extension). Full run: **20.95M ticks / 6,821 iters / 27.4k ep**, cold from scratch (exactly one cold-start line), NRestarts 0, **~240 sps steady**. Integrity hard-fail (rl#343): **0** — not confounded.

**Correction to the launch note:** "~43M ticks" conflated samples with ticks — `sps` counts samples and each tick yields two (6144 samples per 3072-tick iter), so 48 h × 240 sps = 41M samples = **21M ticks**, a third of N's 62M. The pre-registered 30M/40M marks were never reached; the verdict rests on the buckets that were.

**Train-side reach (0.2 m ruler, 5M-tick buckets, episode-weighted, `reach.awk`):**
`0.006 0.007 0.008 0.006 0.007(partial, 1250 ep)` — last 200 iters 0.0062; `[bearing-reach]` 0.00–0.05 across all eight bearings, min 0.00 in 47 of 48 windows. Drift 0.40–0.53 m (N 1.0–1.3, M 0.8–1.0). Reward climbs −7.8 → 4.8 while reach stays at zero: the economy is paying for standing still.

Against the pre-registered split:
- **N-LIKE? No, at every horizon reached.** O is 4–8× under N in the same buckets (N 0.028 / 0.046 / 0.053 / 0.047 at 5–20M) and never shows N's early 0.03–0.05 band.
- **M-LIKE (≤0.056 through the bound)? Yes — and below it.** O sits 3–6× under M (0.023 / 0.032 / 0.026 / 0.042), under K (0.031–0.036) and under A (0.021–0.024). No probe A–N has a 5M bucket below 0.02; O has five. This is not the M/K/A floor, it is a new class: a policy that does not move.
- **Censoring caveat:** at 21M ticks a late N-style separation (N left the band at 30M) cannot be formally excluded. It would have to start from a level no learning probe has ever shown at 5–20M, so the FLOOR call stands; the next probe carries a tick-sized bound so this does not recur.

![Probe O vs L/N/M/K/A reach curves](https://raw.githubusercontent.com/bddap/rl/464e1e90efbc/docs/evidence/chase-archaeology/probe-o-40fde25-reach-curve.png)

**End-of-bound render** (`rl-demo-40fde25-flat-shoveoff` — the pin's own demo + shove-off hunk + flat ground build-local; seed 351, trainer asset root, final ckpt at 20.95M): **a statue** — she stands upright where she spawned for the full 20 s, ball untouched ~3 m to her left, no step toward it; `DRIVE_STATS mean_abs_drive=0.379 mean_effort=7.55` (M's no-chase 0.588/16.9, N 0.572/15.8, L chasing 0.915/33.4). Half the drive of any earlier floor probe — consistent with the 0.4 m drift.
[probe-o-40fde25-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-o-40fde25-endofbound.mp4)

![Probe O end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/464e1e90efbc/docs/evidence/chase-archaeology/probe-o-40fde25-endofbound-tiles.png)

**Reading.** N (`2e5605f`, 08-12) climbs; O (`40fde25`, 08-17) is flat at zero; both carry the same shove-off patch. The breaker landed in `2e5605f`..`40fde25`, 85 commits over five days. Two things sit in that range and co-locate with the 3× throughput drop (720 → 240 sps; G/H at `966b0ab`, 08-18, ran 264–279): the **bevy 0.19 / rapier 0.35 bump** (`4a0a3a6`..`88be0c1`, commits 44–48, landed 08-15 as rl#212 — new contact semantics, a 400 m/s solver cap and contact recycling each pinned off after the fact as "silent default swaps": `88be0c1`, `c16933d`) and the **rl#340 contact/foot-stiffness stages** (`96be084` commit 30, `a57c0e0` 39, `c16933d` 55). A plant whose feet no longer transmit drive into motion explains both a policy that learns to stand still and a solver doing more work per tick.

**Method fix carried into the next box:** bounds are sized in ticks — smoke the pin's sps, pre-commit a wall bound that lands ≥40M ticks, state the projection at launch.

**Decision-tree box appended: Probe P** — cold canary on `7184c17` (commit 43/85, the last commit before the rapier 0.35 bump) + the same shove-off + eval-skip build-local, Probe A's recipe, tick-sized bound. N-LIKE → breaker in `4a0a3a6`..`40fde25` (the bump + rl#340 stages 7–10; next single `88be0c1`); O-LIKE flatline → in `12a6a77`..`7184c17` (rl#340 stages 2–5); M-LIKE → two breakers, one each side. Evidence landed: `464e1e9` (curve + mp4 + tiles).

<a id="c26"></a>
### 26. 2026-09-11 19:51Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5639856499)

**Probe P — 30M-tick early read (not yet legible; next read at 40M)**

`reach.awk` 5M buckets (episode-weighted, 0.2 m ruler) at 30.78M ticks: **0.008 / 0.007 / 0.008 / 0.011 / 0.015 / 0.021**, partial 30–35M **0.028**. First-2000 vs last-2000 iters: reach 0.0078 → 0.0216, drift 0.46 → 0.85 m, reward −5.8 → +8.4 (monotone climb since ~20M). Integrity hard-fails: 0.

- N-LIKE: **excluded** — 0.008 at 5–10M (N ≥0.02), still inside the ≤0.056 band at 30M.
- O-LIKE: **excluded past 20M** — O flatlined 0.006–0.008 with drift <0.6 m; P rises every bucket since 20M with drift 0.85 m and climbing reward.
- M-LIKE (0.02–0.056 through the bound, never separating) vs a late separation (≥0.07 at 40M): the 40M mark decides it.

Caveat carried: probe J (CLIMB) sat at 0.014/0.016 at 5M/10M before jumping to 0.050 at 15M, so a sub-0.02 start does not by itself mean floor; P is ~3.4× under J at matched ticks and passed 15M without a jump.

Seam: the unit stopped 2026-09-11 00:29:22Z and restarted 00:33:25Z (host reboot, NRestarts=0), warm-resumed from ckpt/ at 22,480,896 ticks; tick counter continuous, one cold-start line; the uptick predates the seam.

Throughput 223 sps steady ≈ 111 ticks/s ⇒ 40M at ~2026-09-12 19:00Z, bound 23:08Z unchanged.

<a id="c27"></a>
### 27. 2026-09-12 19:35Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5648208263)

#### Probe P assess — M-LIKE at the band's low edge ⇒ two breakers, one each side of `7184c17`

Read at the pre-registered 40M mark (41.2M ticks, 13.4k iters, ~111 ticks/s; the 96 h bound runs to 23:08Z and stays uncut — the plateau is legible now and the remaining ~2M ticks cannot move it).

`reach.awk` 5M buckets, episode-weighted, 0.2 m ruler: **0.008 / 0.007 / 0.008 / 0.011 / 0.015 / 0.021 / 0.022 / 0.022** (5M…40M), 40–45M partial 0.013. First-2000 vs last-2000 iters: reach 0.0078 → 0.0198, drift 0.46 → 0.91 m, reward −5.8 → +8.8 (reward turns positive at ~2.7M ticks and climbs monotonically to ~9, then flat from ~30M). Integrity hard-fails: 0. One warm-resume seam at 22,480,896 ticks (host reboot, NRestarts=0, counter continuous; the uptick predates it).

Against the pre-registered boxes:
- **N-LIKE — no.** 0.008 at 5–10M (N ≥0.02); 0.022 at 40M (N ≥0.07, and out of the ≤0.056 band by 30M — P never leaves it). 3.5–6× under N at every bucket.
- **O-LIKE — no.** O flatlined 0.006–0.008 with drift 0.4–0.5 m and no reward climb through its 21M; P rises off that floor from ~20M (0.011 → 0.022), drift 0.9 m, reward +9.
- **M-LIKE — yes, low edge.** In the 0.02–0.056 band from 30M on, never separating; M itself sat 0.023–0.056 and started at 0.023 where P needed 25M to get there.

End-of-bound render (`rl-demo-7184c17-flat-shoveoff`, seed 351, 20 s, checkpoint at 41.25M ticks): **no chase** — Sally holds near spawn for the full clip with the ball 2–3 m off her left; `DRIVE_STATS mean_abs_drive=0.478 mean_effort=11.68` (O's statue: 0.379 / 7.55 — P moves the limbs more, to no effect).

Evidence landed in b120283c88f7: [reach curve](https://github.com/bddap/rl/blob/b120283c88f72a504c8866470c086bc5cb3741e5/docs/evidence/chase-archaeology/probe-p-7184c17-reach-curve.png) · [render](https://github.com/bddap/rl/blob/b120283c88f72a504c8866470c086bc5cb3741e5/docs/evidence/chase-archaeology/probe-p-7184c17-endofbound.mp4) · [tiles](https://github.com/bddap/rl/blob/b120283c88f72a504c8866470c086bc5cb3741e5/docs/evidence/chase-archaeology/probe-p-7184c17-endofbound-tiles.png)

![reach curve](https://raw.githubusercontent.com/bddap/rl/b120283c88f72a504c8866470c086bc5cb3741e5/docs/evidence/chase-archaeology/probe-p-7184c17-reach-curve.png)

**Narrowing.** N (`2e5605f`) climbs to 0.106 at 40M; P (`7184c17`, 43 commits later) plateaus at 0.022; O (`40fde25`, 42 more) never leaves 0.006–0.008.
- **Breaker 1 — `12a6a77`..`7184c17` (commits 1–43, rl#340 stages 2–5 contact/foot-stiffness work):** the 5× N→P drop, uncensored at every horizon. The 720→250 sps slowdown lands in the same span (P smoked 248–250 sps like O; N ran 720) — any commit that moved both is the prime suspect.
- **Breaker 2 — `4a0a3a6`..`40fde25` (commits 44–85):** the P→O drop from a slow climb to a true flatline. Weaker evidence: O was censored at 21M, where P had only just left the floor (0.011 vs 0.006; drift 0.58 vs 0.45; reward +1.8 vs O's never-positive).

Per the pre-registered M-LIKE branch the single `88be0c1` (commit 48: the bevy 0.19 / rapier 0.35 bump `4a0a3a6` plus the rl#212 pins that landed with it) goes first — one probe settles breaker 2's bump-or-not question; breaker 1's bisect (midpoint ~commit 22) follows. Next box appended.

Method note carried forward: read at the 30M/40M tick marks instead of the wall bound — this verdict landed 3.8 h before the bound with the same information, and the 30M read had already excluded N and O.

<a id="c28"></a>
### 28. 2026-09-12 22:31Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5649125691)

**Probe Q — early read at 0.79M ticks (2.2 h in; not legible, next read at 30M)**

Cold start confirmed (`optimizer not warm-started: the brain cold-started`), NRestarts 0, integrity hard-fails 0, no warm-resume seam. Throughput 187→193 sps steady with Probe P still co-resident (P's bound ends 2026-09-12 23:08Z); 786k ticks over 255 iters ≈ 99 ticks/s contended.

Same ruler at 250k-tick buckets, matched against P and O at the same ticks:

| bucket | Q reach / drift / rew | P | O |
|---|---|---|---|
| 0–0.25M | 0.000 / 0.53 / −20.3 | 0.003 / 0.38 / −15.6 | 0.009 / 0.35 / −12.0 |
| 0.25–0.5M | 0.009 / 0.47 / −16.4 | 0.009 / 0.28 / −15.7 | 0.003 / 0.38 / −14.6 |
| 0.5–0.75M | 0.006 / 0.39 / −16.1 | 0.012 / 0.38 / −13.4 | 0.003 / 0.35 / −14.0 |

First-100 vs last-100 iters: reach 0.000 → 0.005, drift 0.55 → 0.40 m, reward −20.4 → −16.2. Q is inside the noise band P and O shared at this horizon (all three sat at 0.003–0.012 through 1M; P and O did not separate until ~20M), so neither branch of the split is reachable yet. Reward starts ~5 lower than P/O under the same recipe — noted, not weighted; the 30M read carries the 5M-bucket ruler.

Evaluation running; next read at 30M ticks.

<a id="c29"></a>
### 29. 2026-09-16 00:52Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5690363461)

**Probe Q — 30M-tick read (O-LIKE-shaped; verdict + render at 40M)**

`reach.awk` 5M buckets (episode-weighted, 0.2 m ruler) at 30.90M ticks: **0.004 / 0.003 / 0.006 / 0.006 / 0.006 / 0.009**, partial 30–35M 0.013. First-2000 vs last-2000 iters: reach 0.0040 → 0.0096, drift 0.44 → 0.60 m, reward −12.4 → −8.3. Integrity hard-fails: 0. NRestarts 0, cold start only, no warm-resume seam. 221 sps steady solo (~110 ticks/s).

Matched against P (commit 43) and O (commit 85) on the same ruler:

| bucket | Q reach / drift / rew | P | O |
|---|---|---|---|
| 5M | 0.004 / 0.44 / −13.5 | 0.008 / 0.45 / −7.0 | 0.006 / 0.40 / −7.8 |
| 10M | 0.003 / 0.43 / −7.7 | 0.007 / 0.51 / −0.6 | 0.007 / 0.46 / −1.9 |
| 15M | 0.006 / 0.46 / −6.1 | 0.008 / 0.52 / −0.3 | 0.008 / 0.48 / +0.7 |
| 20M | 0.006 / 0.48 / −6.4 | 0.011 / 0.58 / +1.8 | 0.006 / 0.53 / +2.5 |
| 25M | 0.006 / 0.46 / −5.5 | 0.015 / 0.75 / +5.6 | 0.007 / 0.64 / +4.8 |
| 30M | 0.009 / 0.60 / −8.6 | 0.021 / 0.85 / +8.5 | — |

Fraction of iters with `reward(20) > 0` per 5M bucket: Q **0.00 / 0.00 / 0.04 / 0.05 / 0.06 / 0.06**; P 0.04 / 0.40 / 0.47 / 0.70 / 0.92 / 0.99; O 0.02 / 0.27 / 0.61 / 0.86 / 0.91.

- **P-LIKE: excluded** — needs ≥0.015 by 25–30M and reward positive by ~3M; Q is 0.009 and its reward has never gone positive on any bucket.
- **O-LIKE: matched on all three criteria** (≤0.01 through 30M, drift <0.6 m until the last bucket's 0.60, reward never positive) — and Q sits BELOW O on reward at every horizon: O's reward crossed zero by 15M and its positive-iter fraction reached 0.86 by 20M; Q's is 0.06 at 30M. The bump's pin learns less than the tip it precedes.

Late-bucket vloss is climbing (0.49 → 0.80 → 0.93) with drift, the shape P showed before its 20M separation, so the 40M read decides whether that is a late liftoff or noise. If it holds O-LIKE, breaker 2 = the bevy 0.19 / rapier 0.35 bump `4a0a3a6`..`88be0c1`; next box: revert-test the bump on top of `40fde25`.

Evaluation running; next read at 40M ticks. Bound unchanged, no extension.

<a id="c30"></a>
### 30. 2026-09-17 01:01Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5706819167)

#### Probe Q assess — O-LIKE through 40M; bump convicted as breaker 2

Read at 40.94M ticks (pre-registered 40M mark; 221–224 sps steady, ~110 ticks/s). `reach.awk` 5M buckets, episode-weighted, 0.2 m ruler: **0.004 / 0.003 / 0.006 / 0.006 / 0.006 / 0.009 / 0.010 / 0.009** (5M…40M), 40–45M partial 0.007. First-2000 vs last-2000 iters: reach **0.0040 → 0.0093**, drift **0.437 → 0.489 m**, reward **−12.39 → −6.38**. Reward-positive iteration fractions by bucket remain only 0.000 / 0.004 / 0.039 / 0.048 / 0.063 / 0.058 / 0.043 / 0.068. Integrity hard-fails: **0**. Exactly one cold-start line, no warm-resume seam.

Against the pre-registered split:
- **P-LIKE: excluded.** Q never reaches the required ≥0.015 by 25–30M (0.009), never attains drift ≥0.7 m (0.60 peak bucket), and every 5M reward bucket remains negative. At 40M Q is 0.009 versus P 0.022.
- **O-LIKE: yes.** Q stayed ≤0.01 through 40M, with low drift and persistently negative reward. It is at least as broken as O; its 35–40M bucket fell back to 0.009 rather than separating late.

**Verdict:** the P→Q collapse occurs at the bevy 0.19 / rapier 0.35 bump landing, so breaker 2 is convicted in **`4a0a3a6`..`88be0c1`**. The next experiment is the pre-registered revert-test of that bump on top of `40fde25`; breaker 1 (`12a6a77`..`7184c17`) remains independently open.

End-of-bound render (`rl-demo-88be0c1-flat-shoveoff`, seed 351, 20 s, lavapipe) explicitly logged `play: loaded checkpoint`: **no chase** — Sally stays at spawn and the ball remains untouched. `DRIVE_STATS mean_abs_drive=0.205 mean_effort=2.42`, weaker than O (0.379 / 7.55) and P (0.478 / 11.68).

Evidence landed at `e92842a`: [render](https://github.com/bddap/rl/blob/e92842a009c64053a34f207b82c82d964f53f793/docs/evidence/chase-archaeology/probe-q-88be0c1-endofbound.mp4) · [tiles](https://github.com/bddap/rl/blob/e92842a009c64053a34f207b82c82d964f53f793/docs/evidence/chase-archaeology/probe-q-88be0c1-endofbound-tiles.png)

![Probe Q end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/e92842a009c64053a34f207b82c82d964f53f793/docs/evidence/chase-archaeology/probe-q-88be0c1-endofbound-tiles.png)

<a id="c31"></a>
### 31. 2026-09-23 23:22Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5804590760)

#### Probe R assess — P-LIKE: reverting the bump restores P-level learning ⇒ breaker 2 is the bevy 0.19 / rapier 0.35 bump

Read at the pre-registered tick marks: 30M at iter 9765 (2026-09-19 21:50Z) and 40M at iter 13020 (2026-09-20 18:50Z). The 120 h bound stopped the run on schedule at 53.58M ticks (2026-09-22 02:19:53Z); no extension. Steady rollout 246 sps, ~124 ticks/s.

`reach.awk` 5M buckets, episode-weighted, 0.2 m ruler:

| probe | 5M | 10M | 15M | 20M | 25M | 30M | 35M | 40M |
|---|---|---|---|---|---|---|---|---|
| **R** `40fde25`, bump reverted | 0.008 | 0.006 | 0.010 | 0.010 | **0.017** | **0.024** | 0.031 | **0.024** |
| P `7184c17`, before the bump | 0.008 | 0.007 | 0.008 | 0.011 | 0.015 | 0.021 | 0.022 | 0.022 |
| Q `88be0c1`, bump landed | 0.004 | 0.003 | 0.006 | 0.006 | 0.006 | 0.009 | 0.010 | 0.009 |

After the 40M read R eased to 0.018 and 0.017 (45M, 50M; partial 55M 0.016) with reward still +7 to +9.

| R, first-2000 → last-2000 iters | reach | drift | reward(20) |
|---|---|---|---|
| at 30M | 0.0072 → 0.0223 | 0.448 → 0.844 m | −5.90 → +9.02 |
| at 40M | 0.0072 → 0.0248 | 0.448 → 0.859 m | −5.90 → +6.56 |

Reward-positive iteration fraction by bucket, 5M…40M: 0.061 / 0.489 / 0.806 / 0.950 / 0.998 / 0.992 / 0.965 / 0.945. Integrity hard-fails: 0. One cold start, no warm-resume seam.

Against the split:
- **P-LIKE: yes.** ≥0.015 by 25–30M (0.017, 0.024); reward positive from the 15M bucket on; 0.024 at 40M against the ≥0.02 bar. R matches or beats P in every bucket from 25M through 40M.
- **O-LIKE: no.** Q, the same bump on P's base, sat at 0.006–0.010 over that span.

Correction to the Probe P assess: O's reward did turn positive: bucket means +0.7 and +2.5 in its 15M and 20M buckets, and 86% of its iterations between 15M and 20M ticks were positive. Reward sign separates Q from P and R, not O. The reach clauses carry this verdict.

**Verdict: breaker 2 is the bevy 0.19 / rapier 0.35 bump, `7184c17..88be0c1` (5 commits), convicted both ways.** Adding it to P floors learning (Q); removing it from `40fde25` restores P's curve (R). The 37 later commits add no second breaker at this ruler's resolution. Reverting to rapier 0.32 also dropped rl#340 stage 7's contact-recycling pin (`c16933d`, `b30ba55`), whose field exists only in 0.35.

**Lead for the endgame:** the floored pins already carry the bump's known silent default swaps pinned back. Q has the first three: static-contact softness and friction in the bias pass (`a0d853d`), and the 400 m/s solver speed cap (`88be0c1`). O has all four, adding contact recycling (`c16933d`). The breaking difference is elsewhere in the bevy 0.19 / rapier 0.35 surface.

**End-of-bound render** (`rl-demo-40fde25-revertbump-flat-shoveoff`, seed 351, 20 s, lavapipe; `RUST_LOG=info` logged `play: loaded checkpoint from …/canary-probe-r-40fde25-revertbump/ckpt`). The 40M checkpoint no longer exists, because the trainer keeps only its latest set, so this is the 53.58M-tick set. She faces the ball and creeps toward it; by 19.9 s it sits just ahead of her claws, untouched (a claw touch re-rolls it). `DRIVE_STATS mean_abs_drive=0.499 mean_effort=12.69` (P 0.478 / 11.68, O 0.379 / 7.55, Q 0.205 / 2.42).

Evidence landed at `3a1ae98`: [reach + reward overlay](https://github.com/bddap/rl/blob/3a1ae9821c109525d9bd318374465aa5014d1840/docs/evidence/chase-archaeology/probe-r-40fde25-revertbump-reach-curve.png) · [render](https://github.com/bddap/rl/blob/3a1ae9821c109525d9bd318374465aa5014d1840/docs/evidence/chase-archaeology/probe-r-40fde25-revertbump-endofbound.mp4) · [tiles](https://github.com/bddap/rl/blob/3a1ae9821c109525d9bd318374465aa5014d1840/docs/evidence/chase-archaeology/probe-r-40fde25-revertbump-endofbound-tiles.png)

![Probe R reach and reward overlay](https://raw.githubusercontent.com/bddap/rl/3a1ae9821c109525d9bd318374465aa5014d1840/docs/evidence/chase-archaeology/probe-r-40fde25-revertbump-reach-curve.png)

![Probe R end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/3a1ae9821c109525d9bd318374465aa5014d1840/docs/evidence/chase-archaeology/probe-r-40fde25-revertbump-endofbound-tiles.png)

Throughput stays at P's level (N ran 720 sps), so the 3× slowdown is not the bump; it lives in breaker 1's range.

**Next: breaker 1, `2e5605f..7184c17` (43 commits).** The appended Probe S launch splits at `c9f2428` (commit 31) instead of the blind midpoint (22). Only commits 30–31 (rl#340 stage 2: 30 Hz contact stiffness, solver-iteration guard) touch `crab-world/src/physics/`. Only 35–36 (stage 3: carapace drag sized off body mass) touch body aero and spawn. Commits 1–29 carry render, audio, UI, voice, plane tuning and one collider check. One probe at `c9f2428` separates the two prime suspects.

<a id="c32"></a>
### 32. 2026-09-25 00:43Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5824719912)

**Probe S — 10M-tick read (P-LIKE-shaped; verdict + render at 20M)**

`reach.awk` 5M buckets (episode-weighted, 0.2 m ruler) at 10.2M ticks: **0.006 / 0.006**. First-2000 vs last-2000 iters at the 10M mark (3,255 iters, so the windows overlap): reach 0.0062 → 0.0061, drift 0.44 → 0.48 m, reward −5.9 → −1.3. Integrity hard-fails: 0. NRestarts 0, one cold start, no warm-resume seam. 225 sps steady (~114 ticks/s over the last 24 h).

| bucket | S `c9f2428` reach / drift / rew | N `2e5605f` | P `7184c17` | R `40fde25`, bump reverted |
|---|---|---|---|---|
| 5M | 0.006 / 0.44 / −6.6 | 0.028 / 1.04 / −5.5 | 0.008 / 0.45 / −7.0 | 0.008 / 0.44 / −7.1 |
| 10M | 0.006 / 0.48 / −0.7 | 0.046 / 1.25 / +2.3 | 0.007 / 0.51 / −0.6 | 0.006 / 0.49 / +0.0 |

Fraction of iters with `reward(20) > 0` per 5M bucket: S **0.03 / 0.38**; N 0.34 / 0.60; P 0.04 / 0.40; R 0.06 / 0.49.

- **N-LIKE: excluded.** Its first clause needs ≥0.02 by 5–10M; S reads 0.006 at both marks against N's 0.028 and 0.046, at under half N's drift.
- **P-LIKE: on track.** S sits with P and R, not N, on every column; ≤0.012 through 20M is the remaining clause.

If the 20M read holds, breaker 1 lies in commits 1–31 with stage 2 `96be084` (commit 30) prime, and the next box is the single `b11ef09` (commit 29).

Evaluation running; next read at 20M ticks. Bound unchanged, no extension.

<a id="c33"></a>
### 33. 2026-09-26 02:17Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5842304022)

#### Probe S assess — P-LIKE: `c9f2428` learns like P ⇒ breaker 1 lies in commits 1–31

Tick marks crossed at iter 3255 (10M, 2026-09-25 00:14Z; early read https://github.com/bddap/rl/issues/351#issuecomment-5824719912) and iter 6510 (20M, 2026-09-26 00:08Z). Steady rollout 225–229 sps, ~115 ticks/s. The run continues to its 60 h bound (2026-09-26 11:55:31Z); no extension.

`reach.awk` 5M buckets, episode-weighted, 0.2 m ruler:

| probe | 5M | 10M | 15M | 20M |
|---|---|---|---|---|
| **S** `c9f2428`, commit 31 | 0.006 | 0.006 | 0.008 | **0.011** |
| P `7184c17`, commit 43 | 0.008 | 0.007 | 0.008 | 0.011 |
| N `2e5605f`, before commit 1 | 0.028 | 0.046 | 0.053 | 0.047 |
| R `40fde25`, bump reverted | 0.008 | 0.006 | 0.010 | 0.010 |

S, first-2000 → last-2000 iters at 20M: reach 0.0062 → 0.0108, drift 0.44 → 0.62 m, reward −5.89 → +4.27. Reward-positive iteration fraction by bucket, 5M…20M: S 0.029 / 0.381 / 0.871 / 0.910; P 0.041 / 0.404 / 0.474 / 0.701; N 0.339 / 0.595 / 0.641 / 0.886. Integrity hard-fails: 0. One cold start, NRestarts 0, no warm-resume seam.

Against the split:
- **P-LIKE: yes.** ≤0.012 in every bucket through 20M; S's 20M bucket equals P's.
- **N-LIKE: no.** 0.006 at 5–10M against the ≥0.02 bar, 0.011 at 20M against ≥0.04; N read 0.028–0.053 over the same marks.

S's reward turns positive sooner than P's (87% of 15M-bucket iterations vs 47%), as R's did; reach, not reward, carries the split.

**Verdict: breaker 1 lies in commits 1–31 of `2e5605f..7184c17`.** Commits 32–43, which carry stage 3's carapace drag `bd5e7c7`/`5aefee4`, are acquitted: S has none of them and already sits on P's curve. In 1–31 the only change to `crab-world/src/physics/` is rl#340 stage 2: `96be084` (commit 30: `PHYSICS_SUBSTEPS` 2→4, contact damping ratio 5→10 at 30 Hz, solver iterations set to 8/4/4) and its review round `c9f2428` (31). Commits 1–29 are render, audio, UI, voice and plane-handling changes plus a test-only fixture move.

**20M render** (`rl-demo-c9f2428-flat-shoveoff`, seed 351, 20 s, lavapipe; `RUST_LOG=info` logged `play: loaded checkpoint from …/ckpt-20m`, a consistent copy of the trainer's set taken at 20.14M ticks): **no chase**. She faces the ball, which stays several body lengths off for the whole clip, untouched. `DRIVE_STATS mean_abs_drive=0.363 mean_effort=7.44` (end-of-bound renders: O 0.379 / 7.55 at 21.0M, P 0.478 / 11.68 at 42.7M).

Evidence landed at `217379e`: [reach + reward overlay](https://github.com/bddap/rl/blob/217379e259dda434ac8313e9047649fc3754f9ad/docs/evidence/chase-archaeology/probe-s-c9f2428-reach-curve.png) · [render](https://github.com/bddap/rl/blob/217379e259dda434ac8313e9047649fc3754f9ad/docs/evidence/chase-archaeology/probe-s-c9f2428-20m.mp4) · [tiles](https://github.com/bddap/rl/blob/217379e259dda434ac8313e9047649fc3754f9ad/docs/evidence/chase-archaeology/probe-s-c9f2428-20m-tiles.png)

![Probe S reach and reward overlay](https://raw.githubusercontent.com/bddap/rl/217379e259dda434ac8313e9047649fc3754f9ad/docs/evidence/chase-archaeology/probe-s-c9f2428-reach-curve.png)

![Probe S 20M render tiles](https://raw.githubusercontent.com/bddap/rl/217379e259dda434ac8313e9047649fc3754f9ad/docs/evidence/chase-archaeology/probe-s-c9f2428-20m-tiles.png)

**Next: single `b11ef09` (commit 29).** It is S minus exactly commits 30–31, so N-LIKE convicts stage 2 and P-LIKE moves breaker 1 into 1–29. Its smoke sps also says whether stage 2 owns the 720→250 slowdown (S runs at P's rate). Launch box appended.

<a id="c34"></a>
### 34. 2026-09-26 12:24Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5846248931)

**Probe T — 10M-tick read (N-LIKE-shaped; verdict + render at 20M)**

`reach.awk` 5M buckets (episode-weighted, 0.2 m ruler), cut at 10M: **0.027 / 0.051**. First-2000 vs last-2000 iters at the mark (3,255 iters, so the windows overlap): reach 0.0294 → 0.0472, drift 1.024 → 1.140 m, reward −10.55 → +2.23. Integrity hard-fails: 0. NRestarts 0, one cold start, no warm-resume seam.

| bucket | T `b11ef09` reach / drift / rew | N `2e5605f` | S `c9f2428` |
|---|---|---|---|
| 5M | 0.027 / 1.01 / −11.9 | 0.028 / 1.04 / −5.5 | 0.006 / 0.44 / −6.6 |
| 10M | 0.051 / 1.17 / +3.8 | 0.046 / 1.25 / +2.3 | 0.006 / 0.48 / −0.7 |

Fraction of iters with `reward(20) > 0` per 5M bucket: T **0.07 / 0.65**; N 0.34 / 0.60; S 0.03 / 0.38. T's reward trails N's in the first bucket and passes it by 10M; reach carries the split.

- **N-LIKE: first clause met.** ≥0.02 by 5–10M: T reads 0.027 and 0.051 against N's 0.028 and 0.046, at N's drift. ≥0.04 at 20M remains.
- **P-LIKE: excluded.** ≤0.012 through 20M already fails at 5M.

T is S minus commits 30–31, so a 20M read that holds convicts rl#340 stage 2 (`96be084`/`c9f2428`) as breaker 1.

Solo since S's bound ended (11:55:31Z): 627 rollout sps, 312 ticks/s wall over 176 iters. 10M crossed at iter 3255, 2026-09-26 12:21:25Z. Evaluation running; next read at 20M ticks. Bound unchanged, no extension.

<a id="c35"></a>
### 35. 2026-09-26 20:52Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5849787113)

#### Probe T assess — N-LIKE: `b11ef09` climbs where `c9f2428` floors ⇒ breaker 1 is rl#340 stage 2

Tick marks crossed at iter 3255 (10M, 2026-09-26 12:21Z; early read https://github.com/bddap/rl/issues/351#issuecomment-5846248931) and iter 6510 (20M, 20:30Z). Solo from the end of S's bound (11:55:31Z) to 20M: 684 rollout sps, 339 ticks/s (N 720 sps, S 225–229). The run continues to its 48 h bound (2026-09-28 03:14:59Z); no extension.

`reach.awk` 5M buckets, episode-weighted, 0.2 m ruler, cut at 20M:

| probe | 5M | 10M | 15M | 20M |
|---|---|---|---|---|
| **T** `b11ef09`, commit 29 | 0.027 | 0.051 | 0.054 | **0.085** |
| N `2e5605f`, before commit 1 | 0.028 | 0.046 | 0.053 | 0.047 |
| S `c9f2428`, commit 31 | 0.006 | 0.006 | 0.008 | 0.011 |

T, first-2000 → last-2000 iters at 20M: reach 0.0294 → 0.0820, drift 1.024 → 1.439 m, reward −10.55 → +23.42. Reward-positive iteration fraction by bucket, 5M…20M: T 0.074 / 0.650 / 0.889 / 0.999; N 0.339 / 0.595 / 0.641 / 0.886; S 0.029 / 0.381 / 0.871 / 0.910. Integrity hard-fails: 0. One cold start, NRestarts 0, no warm-resume seam.

Against the split:
- **N-LIKE: yes.** ≥0.02 by 5–10M (0.027, 0.051); ≥0.04 at 20M (0.085).
- **P-LIKE: no.** Above 0.012 from the first bucket.

T pulls ahead of N in the 20M bucket (0.085 vs 0.047). With one seed per arm that gap carries no weight; the split is floor versus climb.

**Verdict: breaker 1 is rl#340 stage 2, `96be084` + `c9f2428`.** T and S differ by exactly those two commits (`git diff b11ef09 c9f2428` touches only `crab-world/src/physics/mod.rs` and `crab-world/src/bot/sim_truth_test.rs`; Cargo.lock identical), and T climbs where S floors, so T-vs-S is the minimal A/B. Stage 2 sets `PHYSICS_SUBSTEPS` 2→4, contact `damping_ratio` 5→10 at 30 Hz, and solver iterations (8, 4, 4) over Rapier's (4, 1, 1); it also carries the 720→250 sps slowdown. With breaker 2 convicted as the bevy 0.19 / rapier 0.35 bump (Q/R), both breakers in `2e5605f..40fde25` are named. Whether `40fde25..138d1eb` holds a third is untested (O and M both floor); a fix-or-revert canary on main answers that. The existing Endgame box is next.

**20M render** (`rl-demo-b11ef09-flat-shoveoff`, seed 351, 20 s, lavapipe; `RUST_LOG=info` logged `play: loaded checkpoint from …/ckpt-20m`, a consistent copy of the set saved at 20.16M ticks): she turns to the ball and closes on it by ~9 s; it then stays against her body to the end, with no claw-tip touch (a touch re-rolls it). `DRIVE_STATS mean_abs_drive=0.573 mean_effort=15.75` (S at 20M: 0.363 / 7.44).

Evidence landed at `a2dc721`: [reach + reward overlay](https://github.com/bddap/rl/blob/a2dc721579fe38f913a20aebb6f3e1e09731edd1/docs/evidence/chase-archaeology/probe-t-b11ef09-reach-curve.png) · [render](https://github.com/bddap/rl/blob/a2dc721579fe38f913a20aebb6f3e1e09731edd1/docs/evidence/chase-archaeology/probe-t-b11ef09-20m.mp4) · [tiles](https://github.com/bddap/rl/blob/a2dc721579fe38f913a20aebb6f3e1e09731edd1/docs/evidence/chase-archaeology/probe-t-b11ef09-20m-tiles.png)

![Probe T reach and reward overlay](https://raw.githubusercontent.com/bddap/rl/a2dc721579fe38f913a20aebb6f3e1e09731edd1/docs/evidence/chase-archaeology/probe-t-b11ef09-reach-curve.png)

![Probe T 20M render tiles](https://raw.githubusercontent.com/bddap/rl/a2dc721579fe38f913a20aebb6f3e1e09731edd1/docs/evidence/chase-archaeology/probe-t-b11ef09-20m-tiles.png)

<a id="c36"></a>
### 36. 2026-09-27 05:39Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5853064287)

The retrain after this endgame should wear #20's hull body: feet and pincers as convex hulls (branch [`rl20-hull-candidate`](https://github.com/bddap/rl/tree/rl20-hull-candidate), fit audit 39/39), a new plant with its own body digest. #20's Ask job ([5087](https://bddap-bot.github.io/botq/#/job/5087)) waits on [5026](https://bddap-bot.github.io/botq/#/job/5026) and merges it once this endgame's fix-or-revert lands, so a canary or retrain launched after that trains on it. One launched before it produces checkpoints for a body main is about to drop.

<a id="c37"></a>
### 37. 2026-09-27 15:35Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5857281232)

**Endgame A/B, 10M read.** Not decisive: M separated from N only after ~15M.

| arm | 5M | 10M | drift, first → last 2000 iters | reward-positive iters, 5M / 10M |
|---|---|---|---|---|
| E0, main control | 0.033 | **0.048** | 0.89 → 1.01 m | 0.11 / 0.27 |
| E1, ζ5 + solver (4,1,1) | 0.023 | **0.034** | 0.75 → 0.87 m | 0.04 / 0.20 |
| E2, 0.32 contact params | 0.030 | **0.043** | 0.86 → 1.01 m | 0.05 / 0.23 |
| N (climbed) | 0.028 | 0.046 | | |
| T (climbed) | 0.027 | 0.051 | | |
| M (main 09-02, floored) | 0.023 | 0.032 | | |
| S (floored) | 0.006 | 0.006 | | |

Episode-weighted reach (0.2 m ruler), ~6.6k episodes per bucket. No integrity hard-fails. Correction: every arm had one warm-resume seam before this read, when all three units were stopped and restarted together at 2026-09-27 13:31Z. Tick counters are continuous across it. The seams sit at 11.59M (E0), 9.41M (E1) and 11.61M (E2), so only E1's falls inside these buckets. E0 and E2 sit on the climbers' line, and E1 sits on M's. Undoing stage 2's contact damping and solver counts on main has not helped so far. Next read at 20M, around 2026-09-28 04:30Z, when E1 gets there.

<a id="c38"></a>
### 38. 2026-09-28 05:23Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5864000085)

**Endgame A/B: fourth arm E12 added at 2026-09-28 05:22Z.** E12 is E1 + E2 together, T's whole parameter surface on main: contact ζ5, solver (4,1,1), linear error 0.001, corrective cap 10, prediction 0.002, clustering off. Trainer sha256 `2ef4139b…`; all six values show in its startup `IntegrationParameters` line.

At 15M, E0/E1/E2 sit together at 0.043/0.040/0.038 (N 0.053, T 0.054, M 0.026). If each arm floors alone, E12 says whether the two halves only work together. If E12 floors as well, the breaker is in rapier 0.35's solver internals or in main after 08-17, not in any parameter. Same recipe, `--ticks 40000000`, same reads; E12 reaches each mark about 20M behind the others.

Seams since the 10M read: the host rebooted at 2026-09-27 20:35Z, and all three arms warm-resumed with continuous tick counters. E2 then hit one rl#343 integrity hard-fail at 21:30Z (a part past 100 m/s) and warm-resumed; E0 and E1 have none.

<a id="c39"></a>
### 39. 2026-09-28 10:02Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5867748560)

**Endgame A/B, 20M read.** Still not separable: N and M were tied at 20M (0.047 and 0.042) and only split at 30–40M.

| arm | 5M | 10M | 15M | 20M |
|---|---|---|---|---|
| E0, main control | 0.033 | 0.048 | 0.043 | **0.044** |
| E1, ζ5 + solver (4,1,1) | 0.023 | 0.034 | 0.040 | 0.055 (partial, 17.7M) |
| E2, 0.32 contact params | 0.030 | 0.043 | 0.038 | **0.036** |
| E12, E1 + E2 | 0.018 (partial, 2.0M) | | | |
| N (climbed) | 0.028 | 0.046 | 0.053 | 0.047 |
| T (climbed) | 0.027 | 0.051 | 0.054 | 0.085 |
| M (main 09-02, floored) | 0.023 | 0.032 | 0.026 | 0.042 |

Integrity hard-fails: E2 has one (reported earlier); the others have none. Four co-running arms leave ~110–135 ticks/s each, so E0 and E2 reach 30M around 2026-09-29 07:00Z. E1 gets there about 8 h later, and E12 trails by ~20M.

<a id="c40"></a>
### 40. 2026-09-29 11:08Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5889017511)

**Endgame A/B, 30M read: every arm is climbing off M's floor.**

| arm | 20M | 25M | 30M |
|---|---|---|---|
| E0, main control | 0.044 | 0.049 | **0.060** |
| E1, ζ5 + solver (4,1,1) | 0.058 | 0.068 (partial, 24.9M) | |
| E2, 0.32 contact params | 0.036 | 0.043 | **0.063** |
| E12, E1 + E2 | 0.021 (partial, 9.2M) | | |
| N (climbed) | 0.047 | 0.054 | 0.078 |
| M (main 09-02, floored) | 0.042 | 0.042 | 0.046 |

Reward turned positive in the latest buckets for all three. Host restarts added more warm-resume seams; tick counters are continuous. Integrity hard-fails: E2 1, the others 0. The verdict stays at 40M, about 27 h out for E0 and E2.

<a id="c41"></a>
### 41. 2026-09-29 12:58Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5890782239)

Handled in flight (botq 2996): the pre-purge main is kept in a private recovery archive off GitHub (pushing old history anywhere there would re-upload the blob), and an old→new table for every SHA this thread cites lands here as a comment.

<a id="c42"></a>
### 42. 2026-09-29 14:01Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891859705)

#### Probe A launched (job 2260)

**Pin:** main `8012a4f8` (08-09 HEAD, includes the `5a897b90` eval-instrument fix). Binary `rl-train-8012a4f8`, sha256 `3a22289fa02262c308cb935aa8fcb840d5c48f740b1dd5add286a1f65c812c3b`, from the canonical autodeploy build of that exact rev.

**Run:** trainer instance `canary-probe-a-8012a4f`, cold (fresh scratch ckpt), started 2026-08-09 18:44 UTC.
```
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
Config otherwise identical to the canary lineage (terrain-refit-s1, reward-econ-s3). Assets share verified byte-identical to `origin/main` for `sally.glb` + `assets/`.

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-10 18:44:07 UTC**, stopguard +120 s.

**Verified live:** iter 1 at steady 804 sps rollout (~8 s/iter → ~2.7M ticks in-bound); iter-0 chase-eval constructed 32 sound far-pair locales (post-fix instrument, eval stays canonical-GCR by design). 0 integrity hard-fails so far — count will be logged in the assess.

**GPU co-residency with Probe B: CO-RUN.** Probe A takes 354 MiB; total 2.0/8.2 GiB used (the only other tenants: a 131 MiB display process, job-2198's recal 362 MiB). mlp256 is strictly smaller and the training host has co-run three mlp512x3 sims before — ~6 GiB headroom is ample. Probe B launches alongside; no serialization needed.

<a id="c43"></a>
### 43. 2026-09-29 14:01Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891861444)

#### Probe B launched (job 2262)

**Pin:** baseline `d20bcdd1` (Jun-28, the EFFORT_WEIGHT 2× commit both confirmed chasers bracket). Built from the era tree + era Cargo.lock (shared clone was shallow to Jun-30 — deepened with `--shallow-since=2026-06-26`; `d20bcdd1` content-verified as exactly the EFFORT_WEIGHT diff). Binary `rl-train-d20bcdd1`, sha256 `fcf7f9a76888f0c5148675f538f50308ddac43e898cfca6ad21aecdbce6f0790`.

**Era-native config = the rev's defaults, by construction:** flat arena (terrain doesn't exist), advancing curriculum from rung 1 [1.5, 3.0) m hard-capped at 9 m (`TARGET_ARENA_HALF` = 10 − 1), mlp256 (`HIDDEN_SIZE=256`, the only arch), EFFORT_WEIGHT 0.0012, reach radius 0.8, advance at reach-fraction 0.6 over 200 episodes. No `--terrain`/`--band-max-m`/`--arch`/sigma-floor knobs exist at this rev (all post-baseline). Rollout shape matches Probe A for comparability: `--workers 6 --envs 2 --horizon 512`.

**Proxy path VERIFIED pre-launch (no `5a897b90` backport — chase-eval doesn't exist in this era; the plan's fallback is the proxy):**
- **train.log half:** per-iter line carries the era proxy natively — `band 1.5-3.0m (reach 0.00)` — assess = band advance past rung 1 at reach 0.6.
- **render half:** a 2-iter smoke checkpoint from this exact binary was rendered by `rl-demo-3ddcf6f` via the archive `render.sh` (lavapipe): loaded clean, rolled 350 ticks, wrote the mp4. Fresh-d20bcdd1 checkpoint format ⇄ era demo binary is proven end-to-end.
- Integrity hard-fail doesn't exist at this rev (quiet-reset era) — confounder count N/A by construction.

**Run:** trainer instance `canary-probe-b-d20bcdd1`, cold (fresh scratch init), started 2026-08-09 19:16 UTC. Live at iter 2: steady 945 sps rollout (~1.7M ticks/h → ~40M ticks in-bound), GPU update on the RTX 2080, checkpoints saving.

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-10 19:17:24 UTC**, stopguard +120 s.

**Co-residency:** B takes 90 MiB beside A's 362 MiB (2.0/8.2 GiB total) — co-run as decided at A's launch; both instances active.

Assess rule (next box): band advance past rung 1 at reach 0.6 + `rl-demo-3ddcf6f` render at bound end; upward-trend no-cross gets ONE 24 h extension before a red call.

<a id="c44"></a>
### 44. 2026-09-29 14:02Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891867667)

#### Decision-tree step → Probe C launched (job 2294)

**Branch taken: A red + B green** → commit-walk the always-on path, flat/band-9 throughout. **First boundary: `4cdf934c` (07-27, pre-physics-cluster).**

**Pin sanity (ancestry-verified):** `4cdf934c` ("rl#318 review loop", 07-27) contains `f49957ca` (the `--terrain/--band-max-m` flags) and `8557ba6` (rl#318 terrain friction), and does **not** contain the momentum solver `cc09092a`, the rl#315 dampers (`a6990d8` — the 07-27 transient later reverted by `f8293e1` is not an ancestor either), air drag, integrity hard-fail, or the implicit-dof-damping fork. So the only delta vs Probe A's pin is the 07-28..08-04 physics/integrity cluster + the rl#337 refactor: **green convicts that cluster; red moves the boundary to `981c00ee`-era (07-06, pre-REACH-0.2).**

**Pin:** built from the pin's own tree + Cargo.lock. Binary `rl-train-4cdf934c` in the trainer's state dir, sha256 `8cfb22541e5e3b5674083a3f66284f5bcacc111cb7e1104fc3d9ca0644d515a2` (build: `Finished release profile [optimized] in 2m 31s`).

**Run:** trainer instance `canary-probe-c-4cdf934c`, cold (fresh scratch init — ckpt files stamped by this run), started 2026-08-11 02:51 UTC. Config identical to Probe A: 
```
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
Rollout-world banner verified in-log: `terrain Flat, band 1.5..9 m`. Live at iter 3, 807 sps steady rollout (~2.9M ticks projected in-bound). GPU 354 MiB / 8.2 GiB (sole trainer; A and B stopped at their bounds).

**Bound:** 24 h via the stop-bound timer — armed, timer readback verified: stops **2026-08-12 02:52:03 UTC**, stopguard +120 s.

**Ruler (per the Probe A audit amendment):** chase-eval is invalid for flat probes — and this pin's evaluator is pre-`5a897b90` anyway (broken locales; its iter-0 eval pass ran ~7 min and completed, non-crashing, same as the pre-08-01 canaries). Assess = **train-side** `[bearing-reach]`/per-iter reach on the probe's own flat ≤9 m band (era health: reach ~0.6 within ~1000 iters; Probe A red sat at 0.00–0.05 for 5583 iters) + end-of-bound rollout-world render. Integrity hard-fail doesn't exist at this rev (quiet-reset era) — confounder N/A by construction.

<a id="c45"></a>
### 45. 2026-09-29 14:03Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891881266)

#### Decision-tree step → Probe D launched (job 2406)

**Branch taken: C RED** → the 07-28..08-04 physics/integrity cluster is acquitted as sole cause; boundary moves to the `981c00ee`-era (pre-REACH-0.2).

**Pin: `0bf9076e`** ("Chase eval: close-range probe" rl#252, 2026-07-10) — the last commit before `8bb5009` (REACH_RADIUS 0.8→0.2), i.e. the **sharpest pre-REACH split** the era allows. Ancestry verified: contains `981c00ee` (07-06 keep-best-on-chase-eval); does NOT contain `8bb5009`. The pin keeps the whole 06-29..07-10 accused set live (OU noise + σ-anneal, advancing-curriculum deletion, mlp512x3 — the only arch since 07-04, unclamped target obs) and predates everything later: REACH 0.2, CLOSE_FRAC default-off, terrain, band-128, obs relayouts, collider regen, contact stiffness, shoves, env-swap, the physics cluster.

**Band lever NATIVE — no `f49957ca` backport:** at this rev the flat ±10 m walled arena is the only ground and the chase band is fixed sampling on [1.5, `TARGET_ARENA_HALF` = 9] m (`targets.rs`; curriculum already deleted). REACH_RADIUS 0.8 (era value).

**Pin binary:** built from the pin's own tree + Cargo.lock. `rl-train-0bf9076e` in the trainer's state dir, sha256 `e122278d7aa860f3387b6e294e2dfc000c74e59870d4d096d73486266fc9ae4e` (`Finished release profile [optimized] in 2m 26s`).

**Run:** trainer instance `canary-probe-d-0bf9076e`, cold (fresh scratch init), started 2026-08-12 12:20 UTC. Rollout shape matches A/B/C:
```
rl-train learn --arch mlp512x3 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
(no `--terrain`/`--band-max-m` at this rev — flat band ≤9 is the default by construction, banner-verified in-log). Live at iter 2: 1034 sps steady rollout, ~8.4 s/iter (~31M ticks projected in-bound), GPU update on the RTX 2080 (354 MiB, sole trainer). Assets: the same read-only `sally.glb` share as probes A/B/C.

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-13 12:20:39 UTC**, stopguard +120 s.

**Ruler (assess, next box):** TRAIN-SIDE only — chase-eval is doubly invalid here (flat probe + pre-`5a897b90` evaluator; its iter-0 pass completed non-crashing, same as C — reads are void, the `981c00ee` keep-best gate consuming them is harmless for diagnosis). Judge per-iter reach on the flat ≤9 m band under era REACH 0.8 semantics (era health: reach ~0.6 within ~1000 iters; Probe B crossed rung 1 at iter 317; A/C red = 0.00–0.08 throughout) + an end-of-bound render — the pin's own `rl-demo` is flat-native (pre-terrain), no diagnostic patch needed. Hard-fail N/A (quiet-reset era).

**Split:** GREEN convicts the `8bb5009`..`4cdf934c` span — REACH 0.2 is its first commit and prime suspect (next: flip REACH alone). RED pushes into the 06-29..07-10 window (`d20bcdd1`..`0bf9076e`: OU noise + σ-floor anneal, curriculum deletion, mlp512x3, unclamped obs, GAE tail/Huber, target-KL early-stop).

<a id="c46"></a>
### 46. 2026-09-29 14:03Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891889223)

#### Decision-tree step → Probe E launched (job 2470)

**Branch taken: D RED, qualified** → bisect midpoint of the `d20bcdd1`..`0bf9076e` window (the 06-30 cluster vs the 07-04..07-10 span).

**Pin: `3ddcf6f7`** ("training/curriculum: near-heavy full-range target sampling", 2026-07-01 — the curriculum-deletion commit itself, and the same commit as the era-verified demo binary). Ancestry verified: contains `76e65dc7` (target-KL early-stop), `7ba0771e`+`889607e4` (OU noise + σ-floor anneal), `a00eaaa6` (unclamped target obs), `5195a15c` (GAE tail/Huber), `1683a26a` (EFFORT_WEIGHT back to 0.0006 — acquitted, both chasers bracket it); does NOT contain `56754c32` (mlp512x3), `981c00ee`, `8bb5009`. So the full 06-30 cluster is live and everything from 07-04 on is absent.

**Era-native config, content-verified at the rev:** flat ±10 m walled arena the only ground; fixed near-heavy band [`BAND_START_MIN` 1.5, `TARGET_ARENA_HALF` = 10−1 = 9] m (curriculum deleted at this very commit); **mlp256** (`HIDDEN_SIZE=256`, the only arch — no `--arch` flag exists); `CURRICULUM_REACH_RADIUS` 0.8; `RL_LOG_STD_FLOOR_END` knob native (algorithm.rs).

**Pin binary:** built from the pin's own tree + Cargo.lock. `rl-train-3ddcf6f` in the trainer's state dir, sha256 `9962911fb7e36abe61714b190fa7587637a8c5284b647b208d0f1d37819f2f28` (`Finished release profile [optimized] in 2m 25s`).

**Run:** trainer instance `canary-probe-e-3ddcf6f`, cold (fresh scratch init — fresh RNG seed logged, keep-best seeded from nothing), started **2026-08-15 15:13 UTC**. Rollout shape matches A/B/C/D:
```
rl-train learn --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
Banner-verified in-log: `band 1.5-9.0m`; mlp256 confirmed live (GPU matmul tuning keys n=256). Live at iter 10: 1299 sps steady rollout, 3072 ticks/iter @ ~5 s (~2.2M ticks/h → ~53M projected in-bound), GPU update on the RTX 2080 (220 MiB, sole trainer). Assets: the same read-only `sally.glb` share as probes A–D.

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-16 15:13:31 UTC**, stopguard +120 s.

**Ruler (assess, next box):** TRAIN-SIDE only — chase-eval doesn't exist at this rev (pre-`79d11ae`). Per-iter reach on the flat ≤9 m band under era REACH 0.8 (era health ~0.6 within ~1000 iters; B rung-1 at iter 317; A/C red 0.00–0.08; D plateau ~0.4) + end-of-bound render via `rl-demo-3ddcf6f` — same commit, loads the format by construction (re-verified 08-06). Hard-fail N/A (quiet-reset era).

**Split:** GREEN (sustained ≥0.6) convicts **07-04..07-06** — `56754c32` mlp512x3 arch swap the prime suspect (next: arch single). RED (plateau ≤~0.4, D's profile) convicts the **06-30 cluster** — OU noise/σ-anneal the first single.

<a id="c47"></a>
### 47. 2026-09-29 14:06Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891942084)

#### Probe I launched (job 2673) — bisect at the 07-17 collider-regen boundary

Decision-tree step after **H2 RED**: `8bb5009` REACH 0.2 is **acquitted** (floor 0.08–0.12 persisted with era REACH 0.8 restored, 10.9k iters); the floor-maker landed in 07-06..main and acts regardless of arch.

**Pin:** `2760e243` (07-17, the regenerated baked colliders — rl#20 stage 2). POST collider regen; PRE terrain-ground `932add78`, PRE band128 `8eda58af`, PRE obs117 `98392819`, PRE env-swap `27c1376f`, PRE the physics cluster. **Era-native defaults verified in-tree pre-launch:** arena `walled_box` flat (RL_ARENA unset — `world.rs:96`), band fixed [1.5, 9] m (`TARGET_ARENA_HALF = 10−1`, curriculum-less), `REACH_RADIUS 0.2` (`targets.rs:61`), mlp512x3 the default arch; no `--terrain`/`--band-max-m` flags exist yet (pre-`f49957ca`) — none passed.

**Chase-eval neutralized BUILD-LOCALLY** (the `2809138` flat-skip doesn't exist at this rev; F/H2 build-local precedent): `BestKeeper::maybe_snapshot` returns `true` right after the period reset — the canonical-GCR eval world (struck as a flat-probe ruler at Probe A; wedged Probe H 22.6 h) is never constructed, and the `true` return keeps the learner's `[bearing-reach]` flush cadence, which IS the ruler. 8-line diff to `best.rs` only; nothing landed on main.

**Binary:** `rl-train-2760e243-evalskip`, sha256 `aef8bb8d02b2c58d203029c56b474350ab1a80c76fea58f19d78a3a55d8ce674`, built from the pin's own tree + Cargo.lock. **Smoke-verified pre-launch:** cold iter 0 + checkpoint save (`brain.bin` 2,909,114 B = the mlp512x3 size), `[best] periodic chase-eval SKIPPED — probe-I build-local neutralization` at the first period, σ-floor −0.7→−1.0 anneal live, ~875 sps.

**Run:** trainer instance `canary-probe-i-2760e243`, cold (fresh scratch init), started 2026-08-22 12:19 UTC.
```
rl-train learn --workers 6 --envs 2 --horizon 512   # flat/band9/mlp512x3/REACH 0.2 all rev defaults
RL_LOG_STD_FLOOR_END=-1.0
```
Live at iter 3: **~900 sps steady** (the G/H ~270-sps anomaly did not reproduce on this pin — bound buys ~25M ticks), GPU update on the RTX 2080 (362 MiB; ample headroom). Assets share byte-identical `sally.glb` (probes A–H2).

**Bound:** 24 h via the stop-bound timer — armed, timer readback verified: stops **2026-08-23 12:19:37 UTC**, stopguard +120 s.

**Split (assess, next box):** PLATEAU (≥~0.3, D/F-like) acquits everything ≤07-17 incl. the collider regen → breaker in 07-18..main; FLOOR (0.01–0.12, A/G/H2-like) convicts the 07-10..07-17 span, collider regen `2760e243` prime.

<a id="c48"></a>
### 48. 2026-09-29 14:07Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891959467)

**Probe I `charge_drifted` (rl#266) — diagnosis, for the bound assess**

The eval monitor paged `charge_drifted=true` on every Probe I eval tick since the probe's first eval (08-22 12:26Z). Verdicts on the three questions:

1. **Not an instrument mismatch.** The monitor evals with the run's own BIN pin (`rl-train-2760e243-evalskip`); the charge pin compiled into that binary is era-native — `CRAB_CHARGE_SPEED_HEIGHTS_PER_S = 2.61` at 2760e243 (main's 4.42 is never consulted). Reproduced off-monitor with the same binary + live checkpoint: `charge_heights_per_s=0.4822 charge_pinned=2.6100 charge_drift_frac=-0.8152 charge_drifted=true`.

2. **Real measurement, benign — reach numbers NOT exploit-tainted.** The 2.61 pin was measured 2026-07-16 on the then-live mlp512x3-s2 brain (a competent policy). Probe I is a fresh-scratch bounded canary at reach ≈0.0–0.17; it charges at 18% of the pinned pace — *slow*, the opposite direction of a charge-speed exploit. rl#343 hard-fail covers integrity violations; rl#266 is a pin-staleness alarm for the spawn-safety derivation and does not apply to a bounded diagnostic run. `progress_m` is measured directly against the real ball, decoupled from the charge pin (rl#280 pace probe), so the FLOOR/PLATEAU split reads as measured. Probe H2 fired the same flag for the same reason.

3. **Probe untouched.** No integrity ground to stop it; the 08-23 12:19Z bound stop stands. The deployment config now suppresses the rl#266 page for `canary-*` runs (the sample still records the wire truth; prod runs still page), so canary probes stop minting guaranteed-false fleet-error classes.

Net for the assess: treat Probe I's charge flag as expected cold-policy behavior; weigh its reach/progress curve at face value.

<a id="c49"></a>
### 49. 2026-09-29 14:08Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891970095)

#### Probe I assess — interim: extension spent (upward trend, no cross)

Bound ran 2026-08-22 12:19 → 08-23 12:19 UTC, stopped on time by `canary-stop-bound`. 11,430 iters, 47,557 episodes, **35.7M ticks** (three mid-bound service restarts, each verified warm — brain + normalizers + Adam restored; tick counter continuous).

**Train-side reach (REACH 0.2, like-for-like vs A/G/H2):** episode-weighted 0.042 overall. Per-1000-iter curve: 0.021 → 0.041 by iter 3k, then **flat 0.041–0.047 for ~8k iters** — squarely A's floor band (0.01–0.05, same criterion + arch) — but the tail rises: iters 7000–7999 hit 0.061 and the final partial window (iters 11000+, 1817 ep) hit **0.073**, ~5.7σ above the mid-bound plateau; last `[bearing-reach]` sample 0.04–0.13 (min 0.04) vs 0.01–0.07 earlier. Nonzero-reach iters 1807/11430, max single-iter reach 1.00.

**Charge-guard event (rl#266, tick 18653184):** diagnosed benign in job 2698 — the era pin (2.61, measured 07-16 on the competent prod brain) vs a fresh canary charging at 18% of it; slow, the opposite of an exploit. progress_m is decoupled from the charge pin (rl#280), so reach numbers are untainted; rl#343 hard-fail N/A by construction (quiet-reset era). Durable fix landed in the deploy tooling (canary-* runs record but never page that guard).

**Why the extension:** an end-of-bound upward trend with no cross is exactly the ONE-extension rule (D precedent). It also matters more than usual here: under the 0.2 m criterion a D/F-like partial learner and the A-floor may both read ~0.04–0.07 (G→H2 shows the 0.2→0.8 criterion alone moves floor readings ~6×), so 24 h more either climbs clear of A's band or stalls — that separation IS the verdict.

**Extension live:** warm-resumed 2026-08-24 00:05 UTC (brain/normalizer/optimizer load verified in-log, tick counter continues at 35.71M, 917 sps steady), new bound armed and timer-verified: stops **2026-08-25 00:05:37 UTC**. Final verdict (incl. end-of-bound render, preferring a pin-native `rl-demo`) after the extension; the split stands: sustained climb clear of the floor band = D/F-like (acquits ≤07-17), stall ~0.05 = FLOOR (convicts 07-10..07-17, collider regen `2760e243` prime).

<a id="c50"></a>
### 50. 2026-09-29 14:08Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891979679)

#### Probe J launched (job 2716) — bisect midpoint of `2760e243..4cdf934c`

Decision-tree step after **Probe I NOT-FLOOR**: everything ≤07-17 incl. the collider regen is **acquitted** of the floor; with Probe C's floor on `4cdf934c` (07-27), the floor-maker narrows to `2760e243..4cdf934c` (07-18..07-27).

**Candidates date/ancestry-verified in the span** (git log `2760e243..4cdf934c`, all six present): terrain-ground `932add7` (07-18 11:45), band128 `8eda58a` (07-18 14:47), obs relayout/+25ch `9839281` (07-18 20:57), contact stiffness `19a1526` (07-19 23:41), shoves `074c126` (07-20 03:01), env swap `27c1376` (07-20 05:53).

**Pin: `69d39c8`** (07-19 20:58, "rl#300: demo reset re-rolls the locale") — the **parent of contact-stiffness `19a1526`**, splitting the candidate set exactly 3/3: pin-side (tested) terrain-ground + band128 + obs relayout (+ rl#236 claw-only-down, rl#298 stage 1); excluded contact stiffness + shoves + env swap + the 07-20..27 tail. Span split 28/46 commits. Chosen over the raw commit-count midpoint (~`27c1376`), which would have put all six candidates on the pin side and wasted the probe on the mostly-vehicles/MP tail.

**Flat/band backport (build-local, NOT landed):** `--terrain`/`--band-max-m` don't exist at this rev (pre-`f49957c`) and `932add7` deleted the flat arena — cherry-picked `f49957c` onto the pin; one conflict (its `inproc.rs` hunk carries post-env-swap code) resolved by wiring the pin's own `build_rollout_app` to `config.terrain.grid()`. **Chase-eval neutralized build-locally** as on Probe I (the `2809138` flat-skip doesn't exist mid-span): `BestKeeper::maybe_snapshot` returns `true` right after the period reset; the `[bearing-reach]` flush cadence is preserved. Full working-tree delta vs the pin: `build-local.patch` in the instance dir (441 lines, 13 files).

**Era defaults verified in-tree:** `REACH_RADIUS 0.2` (`targets.rs:81`), mlp512x3 the default arch — checkpoint `brain.bin` 2,960,314 B = Probe I's mlp512x3 2,909,114 B + the +25 obs channels (25×512×4 B), exact.

**Binary:** `rl-train-69d39c8-evalskip-flatport` (trainer state dir), sha256 `cac7527851dd1d740c8cd92c4bbf929d4f8edfd04a2538bc2323ea412cdecba0`, built from the pin's own tree + Cargo.lock. **Smoke-verified pre-launch:** `[learner] rollout world: terrain Flat, band 1.5..9 m`, cold iter 0 + checkpoint save, `[best] periodic chase-eval SKIPPED — probe-J build-local neutralization` at the first period, σ-floor −0.7→−1.0 anneal live, ~912 sps.

**Run:** trainer instance `canary-probe-j-69d39c8`, cold (fresh scratch init), started 2026-08-25 01:47 UTC.
```
rl-train learn --terrain flat --band-max-m 9 --workers 6 --envs 2 --horizon 512   # REACH 0.2 / mlp512x3 rev defaults
RL_LOG_STD_FLOOR_END=-1.0
```
Live at iter 1: **884 sps steady**, GPU update on the RTX 2080. Assets share byte-identical `sally.glb` (probes A–I).

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-26 01:48:11 UTC**, stopguard +120 s.

**Split (assess, next box):** I-LIKE CLIMB (0.045 by 10M ticks, 0.07+ by 40M, 0.14 late) acquits ≤`69d39c8` → floor-maker in (`69d39c8`..`4cdf934c`] (contact stiffness / shoves / env swap prime); FLOOR (0.01–0.12 trendless, A/C/G/H2-like) convicts (`2760e243`..`69d39c8`] (terrain-ground / band128 / obs relayout / claw-only-down).

<a id="c51"></a>
### 51. 2026-09-29 14:09Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891990245)

#### Decision-tree step → Probe K launched (job 2839)

**Branch taken: J NOT-FLOOR** → everything ≤`69d39c8` (07-19) acquitted of the floor; with Probe C's floor on `4cdf934c` (07-27), the floor-maker lives in (`69d39c8`..`4cdf934c`].

**Pin: `074c1263`** ("rl#298 stage 4: random external shoves during training", 2026-07-20 03:01) — post contact-stiffness `19a1526` (07-19 23:41) AND shoves itself, pre env-swap `27c1376f` (07-20 05:53). Ancestry verified: contains `19a1526` + `69d39c8`; does NOT contain `27c1376f`; 9 commits `69d39c8`..pin (the others are rl#298 stage 2–4 seam prep + rl#299 evidence — infra), 37 pin..`4cdf934c`. Splits the span's named candidates 2/1: pin-side (tested here) contact stiffness + shoves; excluded env swap + the 07-20..27 tail.

**Backport + eval neutralization:** Probe J's `build-local.patch` applies **clean** at this pin — the working-tree delta is content-identical to J's (only hunk offsets/blob ids differ; copy beside the run script). `--terrain flat --band-max-m 9` restored via the `f49957c` seam; periodic chase-eval neutralized as on I/J (`[bearing-reach]` flush cadence kept). Era defaults content-verified at the rev: REACH_RADIUS 0.2 (`targets.rs:81`), mlp512x3 the default arch (smoke `brain.bin` 2,960,314 B — the obs-117 mlp512x3 signature, same as J).

**Pin binary:** built from the pin's own tree + Cargo.lock (+ build-local.patch). `rl-train-074c1263-evalskip-flatport` (trainer state dir), sha256 `916d35761ac0bef319d9d6d5cb7ae4d64503aa4a66850354104b92f558b03e0f` (`Finished release profile [optimized] in 1m 52s`).

**Smoke (pre-launch, cold):** iter 0–5 + checkpoint set saved; banner `terrain Flat, band 1.5..9 m`; first period logged `chase-eval SKIPPED — probe-J build-local neutralization` + one `[bearing-reach]` flush; GPU update on the RTX 2080.

**Run:** trainer instance `canary-probe-k-074c1263`, cold (fresh scratch init), started 2026-08-27 21:09 UTC. Rollout shape matches A–J: `--workers 6 --envs 2 --horizon 512`, `RL_LOG_STD_FLOOR_END=-1.0`. Live at iter 10: **890 sps steady** (~30M ticks projected in-bound) — the G/H ~270-sps anomaly absent; checkpoints saving; eval-skip firing on cadence.

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-28 21:09:36 UTC**, stopguard +120 s.

**Split (assess, next box):** I/J-LIKE CLIMB (~0.045 by 10M ticks, 0.07+ by 40M, 0.10–0.15 late) acquits ≤`074c126` → **env swap `27c1376f` + the 07-20..27 tail prime**; FLOOR (A/C/G/H2-like 0.01–0.12 trendless) convicts (`69d39c8`..`074c126`] → contact stiffness / shoves, singles next. Hard-fail N/A by construction (quiet-reset era).

<a id="c52"></a>
### 52. 2026-09-29 14:09Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891993898)

#### Probe L launched (job 2946)

**Pin:** `19a1526` (2026-07-19 23:41, "rl#299: contact stiffness to rapier default"). Ancestry verified: contains `69d39c8`, excludes `074c1263` — and `69d39c8..19a1526` contains ONLY this commit, so **Probe L is a perfect single: Probe J's pin + exactly the contact-stiffness change.** J-vs-L is the minimal A/B by construction (E-vs-F precedent) — a FLOOR verdict arrives with its confirmation half already in hand.

**Build:** Probe K's `build-local.patch` applies CLEAN at this pin (f49957c `--terrain flat`/`--band-max-m 9` backport + chase-eval neutralization). Era defaults verified in-tree: REACH_RADIUS 0.2 (`training/targets.rs:81`), mlp512x3 the default arch. Binary `rl-train-19a1526-evalskip-flatport` (trainer state dir), sha256 `8e6aed7398820a8faaa3c548a8f703e23e4095adf6f89edfb615a7780335fd31`; full working-tree delta saved as `build-local.patch` beside the run script.

**Smoke (pre-launch):** cold iter 0 through iter 31, GPU update path live, checkpoint set saving, `chase-eval SKIPPED` + `[bearing-reach]` flush cadence in-log, smoke `brain.bin` 2,960,314 B — the obs-117 mlp512x3 signature, same as J/K.

**Run:** trainer instance `canary-probe-l-19a1526`, cold (fresh scratch init), started 2026-08-31 00:35 UTC. Live at iter 5: 890 sps steady rollout, GPU update on the 2080, checkpoints saving, 0 restarts. GPU co-residency: sole canary (1.1/8.2 GiB before launch).

**Bound: 48 h** via `canary-stop-bound` — armed, timer readback verified: stops **2026-09-02 00:35:36 UTC**, stopguard +120 s. This is the launch item's BUDGET rule executed: the 24 h bound + the ONE extension pre-committed up front (I/J/K all read an uninformative 0.03–0.05 at ≤36M ticks; separation only shows at ~55M+, and 24 h buys only ~30M). The assess gets NO further extension — and no mid-extension resume seam this time. ~60M+ ticks expected in-bound at this sps.

**Split (assess, next box):** FLOOR (A/C/G/H2/K-like 0.01–0.12 trendless) CONVICTS contact stiffness `19a1526`; I/J-LIKE CLIMB (~0.045 by 10M, 0.07+ by 40M, 0.10–0.15 late) acquits it → CONVICTS shoves `074c1263` (K = stiffness+shoves floored, so a clean climb on stiffness-alone leaves shoves as the difference, modulo the span's 7 low-prior rl#298 seam-prep commits).

<a id="c53"></a>
### 53. 2026-09-29 14:09Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5891997770)

Lateral read of the probe logs (2026-09-01T19:45Z, Probe L at 65.8M ticks, 5 h before its bound): **L vs K is a clean A/B of the shove commit** — K = L + `074c1263` + seam commits whose crab-world/src delta is 6 doc-comment lines. L reads NOT-FLOOR and above J and I at every 5M-tick bucket, so contact stiffness `19a1526` is acquitted and random shoves `074c1263` convicted. Mechanism + falsifiers below; the fix is tracked separately.

### rl#351 mechanism — lateral read, 2026-09-01T19:45Z (read-only; Probe L live to 09-02 00:35Z)

#### (a) Problem, restated from evidence
Cold flat/band-9 canaries: the Jun-28 baseline chases within 1-2M ticks (B rung-1 @ iter 317; E reach 0.6 @ ~2.7k iters, 0.85 tail).
Every pin ≥ 07-20 sits at train reach 0.02-0.05 for 20-72M ticks (A, C, G, H2, K). It is three stacked degradations, not one:
1. `56754c32` mlp256→mlp512x3: E 0.85 vs F 0.30 on the same tree (REACH 0.8) — convicted by Probe F.
2. Ruler change, not plant: `8bb5009` REACH 0.8→0.2 (targets.rs:81) + `413c2c5` (07-17, ancestor of every probe ≥ I) targets 0.15-0.7 m ABOVE ground
   (targets.rs:12-13, sample_target y draw). A 3-D "claw tip within 0.2 m" of a ball up to 0.7 m up has an unmeasured ceiling; I/J/L top out 0.12-0.18.
3. The FLOOR-maker in (69d39c8..074c1263] — resolved below by L vs K.

#### (b) Findings (evidence: the probes' `canary-probe-*/train.log`, code at the named pins)
F1. Probe L is NOT-FLOOR and ABOVE J and I at every horizon (episode-weighted train reach, 5M-tick buckets, 0.2 m ruler, `reach.awk`):
      ticks   5M    15M   35M   55M   65M   last [bearing-reach] window
      I      .023  .045  .049  .073  .067  0.14-0.15 late
      J      .014  .050  .047  .071  .097  0.09-0.14
      K      .031  .036  .039  .040  .037  0.02-0.05
      L      .079  .095  .114  .129  .146  0.15-0.21   (65.8M ticks at 19:35Z, 0 restarts, 861 sps)
    K = L + `074c1263` + 7 seam commits whose whole crab-world/src delta is 6 doc-comment lines in policy.rs tests
    (`git diff --stat 19a1526 074c1263~1 -- crab-world/src`); build-local.patch byte-identical across J/K/L. So L-vs-K is a clean A/B of the
    shove commit: contact stiffness `19a1526` ACQUITTED, shoves `074c1263` CONVICTED — readable now, 5 h before the bound.
F2. Mechanism signature, same logs, reward(20)/drift (last 3000 iters mean±sd): J +9±11 / 1.5 m; L −40±48 / 2.6 m; K −164±45 / 5.2 m.
    Same reward code (reward.rs: 24·Δ(carapace–target planar dist) − 0.0006·effort, +50 grab; effort tax ≈ 27-32/episode from the renders'
    mean_effort 30-36). ⇒ K's crab ends each 1500-tick episode ≈ 6 m FARTHER from the target; L ≈ 0; J closes ≈ 1.5 m. K's per-iter reward sd is
    4× J's ⇒ return variance ×16 feeding a ReturnNormalizer (algorithm.rs:232-259: GAE on denormalized values, returns scaled by running std) —
    the chase advantage is drowned by an exogenous, unobservable, uncontrollable term paid at 24/m.
F3. Shove scale vs plant. shove.rs:23-27 (identical on origin/main; registered inproc.rs:170-173; NO off-switch): 40-160 N for 8 ticks (0.125 s),
    p=1/640/tick/env ⇒ ~2.3 bursts per max-length episode, rollout-only. Crab total mass 1.97 kg (baked.rs shapes×densities: carapace 0.72 +
    42 links 1.25; weight 19 N). Impulse 5-20 N·s ⇒ Δv 2.5-10 m/s; feet friction (μ 1.5, spawn.rs:193) resists ≤ ~29 N so every burst > 29 N
    accelerates her; applied at the carapace ~0.4 m above the feet ⇒ 16-64 N·m tipping torque vs ~10 N·m gravity-restoring ⇒ first-order,
    every burst tips AND launches her (K drift 5.2 m vs L 2.6 m). The anchor was a vehicle ram at 8 m/s; the only bound ever tested is
    `x1 − x0 > 0.05 m` (reset_test.rs:167). Code path is sound (apply_actions zeroes ExternalForce each tick, actuator.rs; no accumulation) —
    it is a SCALE bug: 2-8× body weight, twice an episode, on a 2 kg body.
F4. On main the launches are damped (air drag `aero::CarapaceDrag`, 08-03: A drift 2.5, reward −80) but the floor persists (A .022, G .015,
    H2 .09 on the 0.8 ruler) — consistent with shoves still the binding constraint; later physics not needed to explain the floor.
F5. Stiffness side-findings: 5→30 Hz at substep dt=1/128 ⇒ rapier erp/substep 0.024→0.128 (erp_inv_dt = ω/(dt·ω+2ζ)). L outlearns J from
    the first bucket (0.079 vs 0.014) — plausibly real (feet no longer 65 mm inside the floor) but ONE seed. Cost: L logs 286 progress-glitch
    drops (|Δd| > 0.5 m/tick) vs J 0, K 185 — rare contact blow-ups, ~1 per 230k ticks, negligible.
F6. Instrument notes: `total` (samples) resets on every warm resume (J 70.1M vs 72.6M ticks; K 36.0M vs 72.1M) — ticks is the only axis.
    DRIFT_REBASE_M 9 m recenter (targets.rs:147) fires routinely only in K (mean drift 5.2 m) — a K-only regime the other probes never enter.

#### (c) Ranked mechanisms, each with its falsifier
1. CONVICTED (high): random shoves `074c1263` — 2-8×-body-weight launches, unobservable in advance, ~2.3/episode, charged at 24/m by the
   progress term ⇒ mean return −6 m/ep, return sd ×4, chase gradient drowned. Falsify: K-pin or main with SHOVE_START_PROB=0 must climb
   J/L-like; a settled crab after ONE 100 N burst must move > 1 m / roll > 90° (if < 0.3 m and upright, the variance needs another source).
2. Contributing, already convicted: mlp512x3 cold start (E vs F). Falsify: L-pin + mlp256 beats L.
3. SPECULATION: the 0.2 m ruler is ceiling-limited by target height (0.15-0.7 m). Falsify: bin K/L episodes by target y; if reach(y ≤ 0.3)
   ≫ reach(y ≥ 0.5), "reach 0.15" is a ceiling artifact and era (0.8 ruler, no y-band) vs now is apples/oranges.
4. ACQUITTED: contact stiffness `19a1526` — L ≥ J at every horizon.

#### (d) Commands
- Table: `for p in i-2760e243 j-69d39c8 k-074c1263 l-19a1526; do echo $p; awk -v B=5000000 -f reach.awk canary-probe-$p/train.log; done`
- Reward sd: `grep '^\[learner\] iter' <log> | tail -3000 | awk '{match($0,/reward\(20\) (-?[0-9.]+)/,r);x=r[1];s+=x;ss+=x*x;n++}END{m=s/n;print m,sqrt(ss/n-m*m)}'`
- Mass: `git show 074c1263:crab-world/src/bot/rig/baked.rs | awk '...cuboid 8hxhyhz, capsule πr²(4r/3+2h), ×density...'` (session script; total 1.972 kg)
- Unit-level convict (AFTER L's bound; cargo, ~2 min): in shove.rs `armed_shove_displaces_the_crab_and_expires`, arm `force: Vec3::new(100,0,0)`,
  tick 1500, print `(carapace_xz − p0).length()` and min over ticks of `(rotation*Vec3::Y).dot(Vec3::Y)`; `cargo test -p crab-world armed_shove -- --nocapture`.
- Main-line canary (24 h, only if a main-side proof is wanted): main + shoves off (SHOVE_START_PROB=0.0 build-local or a new flag),
  `rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512`, RL_LOG_STD_FLOOR_END=-1.0 — expect ≥ J's curve.
- Fix shape: scale shoves to body weight (impulse-cap Δv ≤ ~0.5 m/s, or 2-10 N ≈ 0.1-0.5·mg), default OFF for cold runs, curriculum-on later.

<a id="c54"></a>
### 54. 2026-09-29 14:10Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5892008659)

rl#414 rewrote every rl SHA after the first 14 commits (filter-repo also rewrote SHA references inside commit messages, so the cascade reached the whole history). Old→new for every SHA this thread cites (63), matched by author-date + subject over the full 1030-commit history, 0 unmatched:

<details><summary>old → new table</summary>

| old | new | subject |
|---|---|---|
| `074c1263` | `d577e9c9` | rl#298 stage 4: random external shoves during training |
| `0bf9076e` | `9ce25302` | Chase eval: close-range probe (rl#252) |
| `11607209` | `5a02e87d` | rl#250: close-range/under-carapace target curriculum behind RL_TARGET_ |
| `1683a26a` | `1938ff81` | training/reward: revert EFFORT_WEIGHT 0.0012 -> 0.0006 |
| `19a15264` | `70711622` | rl#299: contact stiffness to rapier default — limbs no longer rest 6-1 |
| `204f76df` | `d6af6323` | rl#338: fix the 21 stale comments — code-grade's honest-no-stale-claim |
| `20a5fa73` | `40e933a2` | rl#351 Probe K assess: end-of-bound render (flat in-distribution, seed |
| `2760e243` | `44e7808b` | rl#20 stage 2 CANDIDATE: scored shape search + mass-preserving density |
| `27c1376f` | `28d20e25` | rl#298 stage 4: the env swap — the trainer trains in the headless serv |
| `2809138f` | `320372bd` | training: flat runs skip the periodic chase-eval (rl#351) |
| `2885a99a` | `93db494e` | rl#330 review fixes: release-frame taps join the code; sub-frame modif |
| `34153322` | `433dc5d6` | crab-world: AnyBrain seam + arch registry; CrabBrain -> bot/arch/mlp25 |
| `3a50ccda` | `e0e880c4` | rl#311: remove body.pos (spawn-relative position) from the observation |
| `3ddcf6f7` | `d6745cf3` | training/curriculum: near-heavy full-range target sampling (replaces u |
| `413c2c5a` | `adb582dc` | rl#281 stage 4: RL_ARENA plant knob — train, eval, and keep-best on th |
| `414519ab` | `aa999f56` | rl#332: carapace air drag — the speed bound the plant was missing |
| `436ae9c0` | `db05582c` | rl#351 Probe J assess: reach-curve evidence (J vs I, floor band) |
| `449050c8` | `3e592e5c` | rl#271: hide obs/action layout behind typed accessors + layout digest  |
| `47625f78` | `47625f78` | init |
| `4cdf934c` | `0afeca88` | rl#318 review loop: vehicle friction pin, tumble assert, one codec fix |
| `5195a15c` | `0294bc7a` | training: GAE tail bootstrap from successor value, on the rollout back |
| `56754c32` | `86cc624e` | arch: cull mlp256 — the 5b evidence-based leaf delete (part of #200) |
| `5a897b90` | `c35356a8` | rl#328: derive progressable eval locales — un-scoreable locales are un |
| `5b678cd7` | `390f16b1` | chase-archaeology evidence: Jun-28 Sallys re-verified chasing under th |
| `68b4d4b6` | `48bd3cbf` | rl#351 Probe K assess: reach-curve evidence (K vs J vs I, floor band) |
| `69d39c8a` | `cf9bd58f` | rl#300: demo reset re-rolls the locale — Sally + ball respawn at a ran |
| `7444384e` | `31a213e5` | chase-archaeology: Probe D end-of-bound render (0bf9076e pin, 48h, rea |
| `76957e67` | `bc92b65a` | rl#343: a physics-integrity violation hard-fails training, no silent r |
| `76e65dc7` | `2f44f7e1` | trainer: keep a best-by-competence checkpoint (part of #157) |
| `775f8433` | `a83fa300` | rl#397: spawn clearance decoupled from the chase-speed pin; cascade pr |
| `79497599` | `4fad9206` | rl#347: the claw whip was the rl#315 impulse-joint damper injecting en |
| `79d11aeb` | `39331db7` | Add headless training-success eval: progress toward a far ball + appli |
| `7ba0771e` | `d81b5307` | training: temporally-correlated (OU) exploration noise to bootstrap a  |
| `8012a4f8` | `26a824ef` | code-grade re-pass: fix 5 stale comments the rl#352 fix round left beh |
| `8557ba6d` | `2a51f86b` | ground: explicit terrain friction so Sally holds steep slopes (rl#318) |
| `889607e4` | `872bdea2` | training: annealed exploration-σ floor to bootstrap a gait (rl#161) |
| `8bb50096` | `2356a520` | rl#253: REACH_RADIUS 0.8→0.2 + one tip_touch predicate across training |
| `8eda58af` | `fcddc209` | rl#292+rl#293 stage 2b: 0-128 m log-uniform chase band; pace-pinned mu |
| `932add78` | `8e66c9c0` | rl#293 stage 2a: terrain is THE ground — TrainArena/RL_ARENA deleted,  |
| `964c378e` | `23dc343b` | rl#392: 2 fps on a test display — restore the cheap driven solver; rest is SLE |
| `966b0ab2` | `67cb2a6f` | arch: restore mlp256 as an opt-in leaf (rl#351 Probe F conviction) |
| `981c00ee` | `1ba293f6` | rl#233: key the keep-best gate on the unified chase-eval, not train-re |
| `98392819` | `9d763bfa` | rl#295: terrain sensing — 25-channel yaw-frame height scan in obs |
| `9e3d3b47` | `6d334215` | GCR: de-invert plane pitch + one VEHICLE_STICK_SENS knob (directive: contr |
| `a00eaaa6` | `2bfd933b` | training+GCR: feed the policy the real target at any distance — delete |
| `a6990d82` | `d0b94cee` | rl#315: limbs no longer interpenetrate under sustained max actuator dr |
| `acd96253` | `51be2434` | rl#351 Probe F evidence: mlp512x3 single-flip DEGRADED vs Probe E (rea |
| `aefcc8d5` | `0a285ff7` | evidence: Probe E (3ddcf6f cold canary) GREEN — reach 0.85 sustained,  |
| `afc59a3f` | `ddc9f81b` | rl#351 Probe G evidence: end-of-bound render + tiles + G-vs-A reach ch |
| `b25c3c4e` | `f0dcb117` | rl#281 stage 2: one terrain path — grid sampler, rapier heightfield, s |
| `b5995a42` | `d236a4d9` | rl#351 Probe A assess: end-of-bound render + tiles (no chase; verdict  |
| `b7909fce` | `e17d50d8` | rl#346: delete roll_with_recovery — an integrity panic now kills the r |
| `c9db5a20` | `c811a466` | Fix stale VehicleControls doc: remote pilots' entries already arrive ( |
| `cc09092a` | `ec7abb5c` | rl#321: momentum-conserving multibody solver — un-ignore the strict co |
| `ce22e7ed` | `eefad283` | rl#315: viscous joint dampers + soft-CCD; eval hard-fails an unbounded |
| `d1997149` | `c73593b8` | rl#345 review round: pin rayon via build_global; banner reads the pool |
| `d20bcdd1` | `3c494441` | GCR reward: strengthen energy-use penalty 2× (EFFORT_WEIGHT 0.0006→0.0 |
| `d9080e7d` | `a335731a` | rl#351 Probe B end-of-bound evidence: d20bcdd1 baseline cold run chase |
| `f49957ca` | `958107d9` | TrainConfig: --terrain and --band-max-m rollout diagnostics (1807 cana |
| `f556b194` | `9d2cd065` | rl#351 Probe I evidence: end-of-bound render + reach curve — NOT-FLOOR |
| `f8293e1e` | `1fce4207` | Revert rl#315 plant changes: rigid contacts explode the live plant (rl |
| `f945ff80` | `ccc0ffb3` | rl#351 Probe J assess: end-of-bound render (flat in-distribution, seed |
| `fc683d84` | `81df4325` | rl#337: split TrainingState by mode, collapse per-env parallel slices |
| `fcf83351` | `236fb810` | rl#351 Probe C assess evidence: end-of-bound flat-world render (red) |

</details>

Full 1030-line map on the build host: `rl414-sha-map-full.txt` (`old new subject`). The pre-rewrite main (old tip `20a5fa73`) is preserved LOCALLY as `rl-old-main-pre-414.bundle` (321 MB, `git bundle verify` clean, never to be pushed anywhere on GitHub — that would re-upload the purged blob). Restore an old commit on the build host: `git fetch rl-old-main-pre-414.bundle <old-sha>`. Probe L (job 2948) eval pins map through the same table (`19a15264` → `70711622`, `074c1263` → `d577e9c9`).

<a id="c55"></a>
### 55. 2026-09-29 14:11Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5892024189)

#### Probe M assess — verdict: **FLOOR** — shoves OFF on current main does NOT restore learning; a second main-side breaker sits in `d577e9c9`..`138d1eb` (old `074c1263`..main, 339 commits)

Bound ran 2026-09-02 01:46 → 09-04 01:46:32 UTC, stopped on time by the stop-bound timer (48 h pre-committed, no extension seam). Full run: **70.15M ticks / 22,835 iters / ~93.4k ep**, cold from scratch, ~790 sps mean steady. **One restart, not a run fault:** the training host rebooted at 01:34 UTC 09-04 (12 min before the bound); the unit warm-resumed from `ckpt/` at 69.90M ticks with the tick counter continuous (exactly one cold-start line in the log) and ran the last 0.25M. Integrity hard-fail (rl#343): **0** — no rollout panic, no unit crash; not confounded.

**Train-side reach (0.2 m criterion, 5M-tick buckets, episode-weighted):**
`0.023 0.032 0.026 0.042 0.042 0.046 0.047 0.054 0.046 0.044 0.040 0.049 0.056 0.054 0.079(partial, 201 ep)` — last 200 iters 0.057; `[bearing-reach]` windows 0.02–0.10. Against the pre-registered yardsticks: 10M **0.032** (I/J ~0.045; L 0.079 in its FIRST bucket), 40M **0.054** (I/J/L 0.07+), 55M **0.040** (L 0.129, I/J 0.071–0.073), 65M **0.056** (L 0.146, J 0.097). M lives in K's band (0.028–0.054) — a mild 0.02→0.055 drift, 2–3× under L at every horizon past 10M, and above A (the shoves-ON control, 0.022–0.024 at 20M) only by ~0.02.

![Probe M vs A/L/K reach curves](https://raw.githubusercontent.com/bddap/rl/56a955580b36/docs/evidence/chase-archaeology/probe-m-138d1eb-reach-curve.png)

**End-of-bound render** (the pin's own `rl-demo` at `138d1eb` + the same 2-line shove-off patch, ground wired flat build-local as on J/K/L, seed 351, trainer's sally.glb asset root): **no chase** — she holds ~3–4 m from the ball for the whole clip, a slow turn toward it, never closes; `DRIVE_STATS mean_abs_drive=0.588 mean_effort=16.9` (L chasing: 0.915/33.4; A armed-no-chase: 0.885). Less armed than A, not more.
[probe-m-138d1eb-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-m-138d1eb-endofbound.mp4)

![Probe M end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/56a955580b36/docs/evidence/chase-archaeology/probe-m-138d1eb-endofbound-tiles.png)

**Reading.** The shove conviction (J→K→L) stands for the `19a1526`-era pin, but it does not transfer alone: main with shoves neutralized floors like A. So the FLOOR branch of the split fires — at least one more breaker landed in `d577e9c9`..`138d1eb` (339 commits: rl#298 chase-eval-on-server-world, terrain/obs channels, moisture, the target-height band — mechanism item 3, the 0.2 m-ruler ceiling, is the ranked in-range suspect; mlp512x3 is shared with L, so it is a handicap, not this breaker). rl#415 (shove scaling) remains worth landing, but it is no longer the Endgame candidate on its own.

**Decision-tree box appended: Probe N** — bisect midpoint of the post-shove range: cold canary on `2e5605f` (rl#353 stage 8, commit 170/339) + the same shove-off build-local patch, Probe A's recipe (flat ≤9 m band, mlp512x3, floor −1.0), 48 h pre-committed. L-like (first bucket ≥0.06, ≥0.07 by 40M) → the breaker is in `2e5605f`..`138d1eb`; M-like (≤0.056 through 65M) → it is in `d577e9c9`..`2e5605f`. Evidence landed: `56a9555` (curve + mp4 + tiles).

<a id="c56"></a>
### 56. 2026-09-29 14:13Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5892057561)

#### Probe P launched — `7184c17` (commit 43/85, pre bevy 0.19 / rapier 0.35 bump), bound in TICKS

**Pin:** `7184c17` (2026-08-15 16:28, "rl#340 stage 5: record the measured settle-bounce floor in the assert") — `git rev-list --count 2e5605f..7184c17` = 43; its topo-successor in the range is `4a0a3a6` (the bump). Builds unmodified, rollout world intact (`rollout world: terrain Flat, band 1.5..9 m`).

**Build-local (never landed):** the same two hunks as N/O — `SHOVE_START_PROB` 1/640 → 0.0, periodic canonical-GCR chase-eval skipped in `BestKeeper::maybe_snapshot` (`[best] periodic chase-eval SKIPPED` present in-log; `[bearing-reach]` flush cadence kept). Demo adds the flat `TerrainGrid::gcr()` hunk. Patches archived in the instance dir as `build-local.patch` and `build-local-demo.patch`.

- `rl-train-7184c17-shoveoff-evalskip` sha256 `b60604c84d0ae245ea19c319d8a1d7491bff004950488b79c4d99e6c3686e682` — `Finished release profile [optimized] target(s) in 2m 54s`
- `rl-demo-7184c17-flat-shoveoff` sha256 `e047c78762a65d82bc4ad4cdc9902c85272ffae68b078cfe958208b23b644e07` — `Finished release profile [optimized] target(s) in 6m 43s` (pre-built for the end-of-bound render)

**Smoke (240 s cold, scratch ckpt, wiped):** iter 0→8 on the RTX 2080, checkpoint set saved (brain.bin 2,954,170 B), `sps(steady rollout)` **248–250**, rollout 24.1–24.3 s per 3072 ticks ⇒ **~127 ticks/s**.

**Slowdown localized:** 250 sps at `7184c17` ≈ O's 240, vs N's 720 at `2e5605f` ⇒ the throughput drop landed in `12a6a77`..`7184c17` (commits 1–43, rl#340 stages 2–5) — NOT in the bump range.

**Command** (Probe A's recipe, fresh init):
```
RL_LOG_STD_FLOOR_END=-1.0 rl-train-7184c17-shoveoff-evalskip learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --checkpoint-dir <instance-dir>/canary-probe-p-7184c17/ckpt
```

**Run:** trainer instance `canary-probe-p-7184c17` active since 2026-09-08 23:07:56 UTC, NRestarts 0, MainPID exe = the pinned binary, 358 MiB GPU, iter 2 at 247 sps steady.

**Bound (ticks, not hours):** 40M ticks at ~127 ticks/s needs ~88 h; pre-committed **96 h = 345600 s** ⇒ projected **~44M ticks** in-bound. `canary-stop-bound canary-probe-p-7184c17 345600` → `armed: stops at 2026-09-12 23:08:16 UTC (timer verified)`, stopguard +120 s; `stop-at` record written. NO extension for the assess.

**Split for the assess:** N-LIKE (≥0.02 by 5–10M, out of the ≤0.056 band by 30M, ≥0.07 at 40M) → breaker in `4a0a3a6`..`40fde25` (commits 44–85) — next single `88be0c1`; O-LIKE (≤0.01 through 20M, drift <0.6 m) → in `12a6a77`..`7184c17` (commits 1–43) — next midpoint ~commit 22; M-LIKE (0.02–0.056 through the bound) → two breakers — single `88be0c1` first.

<a id="c57"></a>
### 57. 2026-09-29 14:14Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5892067385)

#### Decision-tree step → Probe R launched (job 4155)

**Branch taken:** Probe Q stayed O-LIKE through 40M, so breaker 2 is provisionally the bevy 0.19 / rapier 0.35 bump landing `4a0a3a6`..`88be0c1`. Probe R is the pre-registered revert-test on top of `40fde25`.

**Diagnostic tree:** base `40fde25`; reverse-applied the complete `7184c17..88be0c1` bump/rl#212 patch. The literal reverse conflicted where rl#340 stages 7–10 had rewritten `crab-world/src/physics/mod.rs`, so the reconciliation takes the pre-bump physics implementation there while retaining `40fde25` as the source base. Verified resolved stack:

- Bevy `0.18.1`
- bevy_rapier3d `0.34.0`
- Rapier `0.32.0`, fork rev `6624551227535fecbc367f2ea03b5509f2aaa05f`
- pre-0.35 `IntegrationParameters` (no 0.35-only static-contact, friction-bias, speed-cap, or contact-recycling fields)

The same build-local shove-off + periodic chase-eval skip used by N/O/P/Q is applied. Exact patches are archived in the run's state dir (`build-local.patch`, `build-local-demo.patch`).

**Builds** (pin tree + pin lockfile, kache-warm):

- `rl-train-40fde25-revertbump-shoveoff-evalskip`, sha256 `2fe8fe10f9783f728ebcc50aed84f210b801d570734480f906f2719a75f304d5` — `Finished release profile [optimized] target(s) in 5m 13s`
- `rl-demo-40fde25-revertbump-flat-shoveoff`, sha256 `88a42f9eafdff4b341838fbe5b00cba3895d04db3f407bb3b6c55ecde2fc43c7` — `Finished release profile [optimized] target(s) in 9m 00s`

**Smoke:** 240 s cold run, iter 0→7; checkpoint set saved (`brain.bin` 2,954,170 B), `[best] periodic chase-eval SKIPPED` and flat `rollout world: terrain Flat, band 1.5..9 m` verified. Steady rollout was **211–221 sps** while Q was co-resident, ≈105–110 ticks/s. The live run's first steady iteration is **236 sps**, ≈118 ticks/s.

**Run:** trainer instance `canary-probe-r-40fde25-revertbump`, cold fresh init, started **2026-09-17 02:19:50 UTC**. The main process resolves to the pinned trainer; `NRestarts=0`; first checkpoint save verified. Probe A recipe: flat ≤9 m, mlp512x3, workers 6, envs 2, horizon 512, `RL_LOG_STD_FLOOR_END=-1.0`.

**Tick-sized bound:** 120 h (`432000 s`) via the stop-bound timer, timer readback verified for **2026-09-22 02:19:53 UTC**, stopguard +120 s, NO extension. At the measured 105–118 ticks/s this projects **45.4–51.0M ticks**, clearing 40M. Assess at 30M and 40M, not at the wall bound.

**Split:** P-LIKE recovery (≥0.015 by 25–30M, reward positive, ≥0.02 at 40M) confirms the bump mechanism. O-LIKE (≤0.01 through 40M, negative reward) moves breaker 2 into commits 49–85 despite Q. The appended assess box carries the render and reporting requirements; breaker 1's midpoint follows this revert-test.

<a id="c58"></a>
### 58. 2026-09-29 22:44Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900501079)

#### Stage 1 — source recovered, changes enumerated, bisect plan

##### 1. Recovery: the history is not missing

Correction to this issue's body: `origin/main` today reaches the full history — root `47625f78` (2026-02-11 init) through now, including all 80 commits dated Jun 27–29. Whatever the 08-06 archaeology saw (stale clone or a since-repaired rewrite), the era is recoverable directly.

**Baseline = `d20bcdd1`** (2026-06-28 15:06, "reward: strengthen energy-use penalty 2× (EFFORT_WEIGHT 0.0006→0.0012)"). Provenance, cross-checked three ways:
- job-377's own result names the pre-tax checkpoint "brain-pre-energy-penalty-**d20bcdd**" — `brain-pre.bin` (confirmed chaser) is the checkpoint immediately before this commit; `brain.bin` (confirmed chaser) is it≈500 after it, trained Jun 28 ~15:06–17:38 under `d20bcdd1` (next commits that day are render/otel-only).
- `d20bcdd1` content-verified: exactly the `EFFORT_WEIGHT 0.0006→0.0012` diff.
- The era binary pin `3ddcf6f7` (07-01) is reachable from main and loads these checkpoints (re-verified 08-06).

So both confirmed chasers bracket `d20bcdd1` with effort weight 0.0006 and 0.0012 — that coefficient is acquitted within that range. Recovery is **complete**: `git checkout d20bcdd1` builds the training source that produced the chasing brains. Nothing partial, nothing missing.

##### 2. Training-relevant changes d20bcdd1 → main (204f76df)

524 commits total; the training-relevant set, dated (verified by diff content, not just messages):

| date | sha | change | break mechanism |
|---|---|---|---|
| 06-29 | 76e65dc7+2 | target-KL 0.03 early-stop; π_old recomputed on update backend; keep-best ckpt | update dynamics; low |
| 06-30 | 7ba0771e, 889607e4 | OU exploration noise (ρ=0.95) + σ-floor anneal −0.7→−2.0 over 5M ticks | first 5M ticks forced-noisy; plausible |
| 06-30 | a00eaaa6 | obs target_local unclamped (real target at any distance) | obs range grows with band; plausible |
| 06-30 | 5195a15c+1 | GAE tail bootstrap; Huber value loss | mild |
| 06-30 | 3ddcf6f7+3 | advancing curriculum deleted → fixed near-heavy band sampling | no easy-first progression for cold runs; plausible |
| 07-01 | 1683a26a | EFFORT_WEIGHT 0.0012→0.0006 (revert of the baseline commit) | acquitted (both chasers) |
| 07-04 | 56754c32+1 | mlp256 culled → mlp512x3 (3×512) is THE arch | bigger-net cold dynamics; plausible |
| 07-06 | 981c00ee+1 | keep-best keyed on chase-eval | selection only; low |
| 07-10 | 8bb50096 | **REACH_RADIUS 0.8→0.2 m** | terminal +50 4× harder; success episodes stop terminating; HIGH |
| 07-10 | 11607209+1 | close-range target mix (CLOSE_FRAC) | low |
| 07-13 | 449050c8+3 | obs/action digest gate (rl#271); env knobs → clap | infra |
| 07-16 | b25c3c4e+4 | terrain lands (rl#281), opt-in `RL_ARENA` | opt-in at this date |
| 07-17 | 2760e243+1 | **regenerated baked colliders** — new shapes + re-derived densities ("a new MDP no live checkpoint can drive") | body dynamics changed; HIGH |
| 07-18 | 932add78+1 | **terrain is THE ground** — flat arena deleted | difficulty step; HIGH |
| 07-18 | 8eda58af+1 | **chase band → log-uniform [1.5, 128] m** (was ≤9) | return scale ×14; most far episodes unfinishable in 1500 ticks; HIGH |
| 07-18 | 98392819+1 | **+25 terrain obs channels**, obs 92→117 | relayout + new channels to learn; HIGH-ish |
| 07-19 | 19a15264 | contact stiffness → rapier default (was ~36× softer) | ground contact changed; plausible |
| 07-20 | 074c1263 | random external shoves (40–160 N, p=1/640/tick, no off-switch) | uncompensated perturbations; plausible |
| 07-20 | 27c1376f+3 | **env swap (rl#298)**: trainer rolls out in the headless net-server world (vehicles as world content) | wholesale env reconstruction; HIGH |
| 07-23 | 3a50ccda | rl#311: body.pos dropped from obs, 117→114 | loses drift cue; plausible-mild |
| 07-25 | f49957ca | `--terrain {gcr,flat}` + `--band-max-m` diagnostic flags | the bisect levers |
| 07-27 | 8557b8ad+1 | explicit terrain friction (rl#318) | plausible |
| 07-29 | cc09092a | **momentum-conserving multibody solver** (rapier fork bump; flail pace +70%) | core physics under every gait; HIGH |
| 08-03 | ce22e7ed | **viscous joint dampers on every articulation + soft-CCD** (rl#315) | gait dynamics changed; HIGH |
| 08-03 | 414519ab | carapace air drag (0.034) | caps body speed; plausible |
| 08-03 | 76957e67 | **rl#343: integrity violation hard-fails (panics) the run** (was quiet reset) | runs die instead of resetting; HIGH as confounder |
| 08-04 | b7909fce+1 | roll_with_recovery deleted (rl#346) | same class |
| 08-04 | 79497599 | **rl#347: damper swap → implicit multibody dof damping** (fork bump 66245512) | plant dynamics again; HIGH-ish |
| 08-06 | fc683d84+1 | rl#337 TrainingState refactor ("pure") | verify-by-content candidate for a semantic slip |

Unchanged end-to-end: PPO core (γ 0.99, λ 0.95, clip 0.2, entropy 0.001, lr 3e-4, epochs 4, batch 64), PROGRESS_WEIGHT 24, GRAB 50, episode 1500 ticks, Welford normalizer + clip 5.0. Excluded: render/UI/netcode/eval-instrumentation-only commits.

##### 3. Existing evidence (what's already ruled in/out)

- **All rl#319 canary reads before 08-01 are VOID** — the eval instrument was broken (unconstructible eval locales, fixed `5a897b90` / job 2018) and the "flat-regime breaker" verdict was overturned. The #3–#11 acquittals don't stand; do not cite them.
- Post-fix, clean: **`2885a99` (08-01) cold on gcr: no liftoff in 28.8M ticks/24 h.** That's a measured no-learn on a pin that already has terrain+band128+obs114+env-swap+solver-swap but **predates** the 08-03/04 dampers, drag, and integrity hard-fail — so those last four cannot be the sole cause.
- **`7949759` (08-04) cold on gcr: flat ~0.3 m over 24.7M ticks, 29 integrity hard-fails in 24 h (rl#349, open)** — no-learn AND confounded.
- Never measured: whether the **baseline itself learns to chase from scratch** — the Jun-28 brains were a long warm-start lineage. This is a mandatory control (successful runs did take a while).

##### 4. Bisect plan

**Cost model:** each probe = one fresh-scratch canary (trainer instance <name> + `canary-stop-bound`, 24 h bound, ~1–1.5M ticks/h). Proxy: **chase-eval sustained-1m** (first 5-sample window with median ≥1.0 m) — healthy learners cross at 0.7–3.3M ticks (~0.5–3 h); justified against archaeology: era chase competence emerged within ~500–1050 iters of the Jun-28 run. Greens are cheap, reds cost the full bound → order probes to maximize expected greens. Ground truth at each bound end: a 20 s render with the pin's own demo binary (the method that settled this issue's shortlist). Rules: a 24 h no-cross with an upward trend gets ONE extension (c9db5a2-ext24 crossed at 18.9M); a probe with >0 integrity hard-fails and no liftoff is **confounded, not falsified** (rl#349 is one enumerated candidate, not the presumed cause — log the hard-fail count on every probe); use each pin's own eval binary and verify its locale construction is sound (backport `5a897b90` where not — or fall back to the render + era reach/band log proxy). Prod training stays off; canaries are diagnostic, checkpoints never deployed.

**Two anchors first (informative in all outcomes):**
- **Probe A — main, June-approximated by flags:** `--terrain flat --band-max-m 9`, floor-end −1.0, cold 24 h. Neutralizes the two flaggable MDP deltas; everything un-flaggable (physics cluster, REACH 0.2, shoves, env-swap world, obs-114+terrain channels, mlp512x3, hard-fail) stays live.
- **Probe B — baseline `d20bcdd1`, cold:** era-native config (flat, band ≤9, mlp256). Judge by era proxy (band advance past rung 1 at reach 0.6 in train.log) + renders via `rl-demo-3ddcf6f`. If B itself doesn't chase in-bound, cold-probe negatives are uninterpretable and the plan shifts to longer bounds / lineage replay before any bisecting.

**Decision tree:**
- A green + B green → the breakage is in terrain/band: flag-flip singles on main (gcr+band9, flat+band128) to convict one; then it's a difficulty/curriculum tuning problem (e.g. band or amplitude ramp), not a bug hunt.
- A red + B green → commit-walk the always-on path, flat/band-9 throughout. First boundary: **`4cdf934c` (07-27, pre-physics-cluster)** — green convicts the 07-29–08-04 physics/integrity cluster (~5 changes, 1–2 more probes); red pushes earlier: next **`981c00ee`-era (07-06, pre-REACH-0.2)**, then midpoints. Pins inside 07-18–07-25 lack the flat flag — avoid or cherry-pick `f49957ca` onto them. Expected total: 2–4 probes past the anchors.
- A red + B red → no cold anchor exists; escalate bounds (48–72 h) on B before trusting any negative.

Wall-clock: anchors 48 h serial (stage-2 job assesses GPU co-residency — the mlp256 baseline is small), then 2–4 probes ≤24 h each ≈ **3–7 days** to conviction.

**End game:** confirm the accused change with a minimal A/B (change ± on otherwise-identical config), fix-or-revert it on main, then re-enable the remaining features one canary at a time — goal: training working with max features (scalable terrain, new sensors) per the directive.

##### Stage checklist

- [x] Stage 1 — recover the training source (`d20bcdd1`), enumerate changes, bisect plan (this comment)
- [ ] Stage 2 — Probe A: main cold canary, `--terrain flat --band-max-m 9`, floor −1.0, 24 h; record sustained-1m tick, integrity hard-fail count, end-of-bound render
- [ ] Stage 3 — Probe B: baseline `d20bcdd1` cold canary, era proxy + `rl-demo-3ddcf6f` renders; validates cold-start chase at the recovered baseline
- [ ] Stage 4 — branch per the decision tree (flag-flip singles, or boundary probes `4cdf934c` → earlier); iterate to the guilty change
- [ ] Stage 5 — minimal A/B confirmation; fix-or-revert on main; re-enable features one-per-canary until training works with max features

<a id="c59"></a>
### 59. 2026-09-29 22:46Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900518787)

#### Probe J assess — interim: extension spent (overlap-zone read, no separation yet)

Bound ran 2026-08-25 01:47 → 08-26 01:48:11 UTC, stopped on time by `canary-stop-bound`. **11,953 iters / 36.72M ticks** (`ticks.txt` 36,722,688 == 11,953 × 3072 exact), cold from scratch, **zero restarts**, ~857 sps steady.

**Train-side reach (REACH 0.2, per-1000-iter episode-weighted, 49,060 ep total):**
`0.014 0.012 0.015 0.038 0.059 0.049 0.046 0.039 0.038 0.046 0.050 0.049` — floor through 2.5k, rise to ~0.059 by 4k, then oscillation 0.038–0.050; final partial window (iters 11,500+, 1889 ep) **0.056**, 5.5σ above the 5.5k+ plateau (0.0435, 24.7k ep). Tail OLS slope +0.0015/1k iters. Last `[bearing-reach]` windows 0.02–0.07 across all bearings.

**Why the extension (the ONE, D/I precedent):** J's first bound is statistically the twin of I's — overall 0.038 vs I's 0.042, final-window significance 5.5σ vs I's 5.7σ — squarely in the documented overlap zone (~0.04–0.07) where a D/F-like slow learner and the A-floor read alike under the 0.2 m criterion. I separated only in its extension (0.080 ext-weighted, 0.14–0.15 late). Judging J FLOOR at a horizon where I read the same would break the like-for-like. The feature to watch: J's tail (0.056) is NOT a run-high (the 4k bucket already hit 0.059), where I's 0.073 was — if the extension stalls ~0.05 instead of climbing, that IS the separation.

**Extension live:** warm-resumed 2026-08-26 03:53 UTC (brain + both normalizers + Adam restored, verified in-log; tick counter continuous at 36.73M; the running process's executable = the pinned `cac75278…` binary; 934 sps), bound armed and timer-verified: stops **2026-08-27 03:54:11 UTC**, stopguard +120 s.

**Split unchanged:** I-like separation (climbs clear of the floor band) acquits ≤`69d39c8` → breaker in (`69d39c8`..`4cdf934c`] (contact stiffness / shoves / env swap); stall ~0.04–0.06 trendless = FLOOR → convicts (`2760e243`..`69d39c8`] (terrain-ground / band128 / obs relayout / claw-only-down).

<a id="c60"></a>
### 60. 2026-09-29 22:46Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900521457)

#### Probe J assess — verdict: **NOT-FLOOR** (I-like climb) — everything ≤07-19 `69d39c8` ACQUITTED, floor-maker narrows to (`69d39c8`..`4cdf934c`] (07-19..07-27)

Extension ran 2026-08-26 03:53 → 08-27 03:54:11 UTC, stopped on time by `canary-stop-bound`. Full run: **72.6M ticks / 23,623 iters / 97,957 ep**, cold from scratch; bound 1 had zero restarts, the extension resumed verified-warm (brain + both normalizers + Adam in-log, tick counter continuous, the running process's executable = the pinned `cac75278…` binary). Steady 857 sps median. Integrity events: **0**; zero in-log charge flags.

**Train-side reach (0.2 m criterion, per-1000 cumulative iters):** floor 0.012–0.015 through 2.5k → 0.038–0.059 by 4–5k → oscillation ~0.04–0.05 through 12k (bound 1 ended 0.038 overall — the interim's overlap-zone read) → extension climbs through 0.06–0.08 to run-high late windows **0.097 / 0.120 / 0.087 / 0.121**, extension episode-weighted **0.075** vs 0.038 in bound 1, last-1.7k-iters 0.102 vs ext-body 0.070. The shape is the Probe I trajectory nearly point-for-point at every horizon (I: 0.042 bound 1 → 0.080 ext, 0.14–0.15 late) — not the trendless A/C/G floor (0.01–0.05, same criterion + arch, 19.2M ticks).

![Probe J vs I reach curves](https://raw.githubusercontent.com/bddap/rl/f945ff807da2/docs/evidence/chase-archaeology/probe-j-69d39c8-reach-curve.png)

**End-of-bound render** — the pin's own `rl-demo`, flat wired build-local (one line: `ArenaWorldPlugin` grid → `TrainTerrain::Flat.grid()`; in-distribution, seed 351): **armed** — `DRIVE_STATS mean_abs_drive=0.853 mean_effort=30.0`, ≈ I's chasing 0.890/32.0 and far above H2's tumble 0.628/18.4 — active gait, upright, but **no close on the ball in the seeded 20 s episode**. Weighted secondary per the box: J's late reach (0.10–0.12) trails I's at render time (0.14–0.15), so a non-chasing single episode is unsurprising; the verdict is the curve.
[probe-j-69d39c8-endofbound.mp4](https://github.com/bddap/rl/raw/main/docs/evidence/chase-archaeology/probe-j-69d39c8-endofbound.mp4)

![Probe J end-of-bound tiles](https://raw.githubusercontent.com/bddap/rl/f945ff807da2/docs/evidence/chase-archaeology/probe-j-69d39c8-endofbound-tiles.png)

**Reading:** the pin-side candidates — terrain-ground `932add7`, band128 `8eda58a`, obs relayout/+25ch `9839281`, rl#236 claw-only-down — are all in J's tree and J learns like I; they are **acquitted of the floor**. With Probe C's floor on `4cdf934c` (07-27), the floor-maker lives in **(`69d39c8`..`4cdf934c`]**: contact stiffness `19a1526` (07-19 23:41), shoves `074c1263` (07-20 03:01), env swap `27c1376f` (07-20 05:53), + the low-prior 07-20..27 tail. Scope note as on I: acquittal is of the FLOOR only — J's climb stays far below era-healthy liftoff (B rung-1 at iter 317; E 0.6 by ~2.5k), consistent with the mlp512x3 slow-learning conviction (E-vs-F) still applying.

**Decision-tree box appended: Probe K** — bisect at suggested pin `074c1263` (splits the candidate set 2/1: FLOOR convicts contact-stiffness/shoves, CLIMB moves prime suspicion to the env swap + tail). Evidence landed: `436ae9c0` (curve PNG), `f945ff80` (mp4 + tiles).

<a id="c61"></a>
### 61. 2026-09-29 22:46Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900524701)

#### Probe K assess — interim: extension spent (overlap-zone read, no separation)

Bound ran 2026-08-27 21:09 → 08-28 21:09:36 UTC, stopped on time by `canary-stop-bound`. **11,668 iters / 35.85M ticks** (`ticks.txt` 35,847,168 == 11,668 × 3072 exact), cold from scratch, **zero restarts**, ~845 sps steady. Integrity events: 0; hard-fail N/A by construction (quiet-reset era).

**Train-side reach (REACH 0.2, per-1000-iter episode-weighted, 47,809 ep total):**
`0.033 0.029 0.034 0.036 0.033 0.036 0.033 0.034 0.035 0.038 0.038 0.035` — overall **0.0345**. Two shape departures from I/J at the same horizon: no early floor phase (bucket 1 already 0.033 vs I/J's 0.012–0.021 — consistent with the σ-anneal window reading higher by chance at this pin) and no distinct rise (run-high buckets 0.038 at 9–11k, ≈ +1.7σ over the 0–9k body vs I's 5.7σ / J's 5.5σ tails). Last `[bearing-reach]` flushes 0.044 / 0.037 / 0.038, the run-high flush third-from-last.

**Why the extension (the ONE, D/I/J precedent):** K's bound-1 level sits inside the documented overlap zone (J 0.038, I 0.042 at the same ~36M-tick horizon) where a D/F-like slow learner and the A-floor read alike under the 0.2 m criterion; both I and J separated only in extension (I 0.080 ext-weighted, 0.14–0.15 late; J 0.075 ext, 0.10–0.12 late). Convicting contact-stiffness/shoves on a read that matched two acquitted profiles would break the like-for-like. The watch feature is sharper than J's: K has shown **no rise at all** — an extension that climbs I/J-like acquits ≤`074c1263`; a stall at 0.03–0.04 trendless IS the separation.

**Extension live:** warm-resumed 2026-08-28 21:52 UTC (brain + both normalizers + Adam restored in-log; tick counter continuous at 35.85M; the running process's executable = the pinned `916d3576…` binary; 850 sps steady). Bound armed and timer-verified: stops **2026-08-29 21:53:15 UTC**, stopguard +120 s.

**Split unchanged** from the launch box: I/J-like climb clear of the floor band acquits ≤`074c126` → **env swap `27c1376f` + the 07-20..27 tail prime**; 0.03–0.04 trendless = FLOOR → convicts (`69d39c8`..`074c126`] (contact stiffness `19a1526` / shoves `074c1263`), singles next.

<a id="c62"></a>
### 62. 2026-09-29 22:52Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900585909)

#### Decision-tree step → Probe F launched (job 2508)

**Branch taken: E GREEN** → the 06-30 cluster is acquitted; window `3ddcf6f`..`0bf9076e` (07-01..07-06), `56754c32` (mlp256 → mlp512x3) prime suspect. Probe F is the **arch single-flip**: the exact tree that just went green, with the architecture as the only change.

**Pin: the SAME `3ddcf6f7` tree as Probe E, one build-local diff** — `brain.rs` mlp256 → 3×512 (HIDDEN_SIZE 512 + a third trunk fc/ln layer), exactly the depth×width delta `56754c32`'s 5a A/B isolated ("everything but trunk depth×width held equal"). A literal cherry-pick is impossible — the arch-registry seam (`3415332`) postdates this pin — so the flip is hand-applied; patch archived as `probe-f-mlp512x3-backport.patch` in the trainer's state dir and inlined below. Not landed (diagnostic, same as Probe C's demo patch). Everything else — infra, OU noise/σ-anneal, band, REACH 0.8, floor −1.0 — bit-identical to E's build.

**Pin binaries** (pin tree + patch + pin Cargo.lock):
- `rl-train-3ddcf6f-mlp512x3` sha256 `5a71c18b7b41ebb99323d6a7fcf36ca1256b478b7dff328b78284310ff7414da`
- `rl-demo-3ddcf6f-mlp512x3` sha256 `ab9ae959e844c1f72be58524ed52c7218c9c0680747ba49a4e5875ef82fd78a7` — needed because the stock era demo cannot load a 512×3 checkpoint.

**Smoke-verified pre-launch, end-to-end:** 3-min cold run → `brain.bin` 2.9 MB (≈730k params, the mlp512x3 size; mlp256's is ~0.5 MB — the flip is in the artifact, not just the source) → patched demo loaded the checkpoint and rendered it (lavapipe).

**Run:** trainer instance `canary-probe-f-mlp512x3`, cold (fresh scratch init), started 2026-08-17 02:13 UTC. Rollout shape matches A–E: `--workers 6 --envs 2 --horizon 512`, `RL_LOG_STD_FLOOR_END=-1.0`. Live at iter 3: 962 sps steady rollout (~35–40M ticks projected in-bound), banner-verified flat band 1.5..9 m native; GPU 362 MiB sole trainer, the running process's executable = the pinned binary.

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-18 02:13:07 UTC**, stopguard +120 s.

**Ruler (assess, next box):** TRAIN-SIDE reach judged **against Probe E's own curve** — same pin, same infra, arch the only diff (E: sustained 0.6 by ~2.7k iters, 0.83–0.87 from 10k; D plateau ~0.4; A/C red 0.00–0.08). **DEGRADED (≤~0.4–0.5 plateau) convicts mlp512x3 cold learning; HEALTHY (E-like) acquits the arch** → suspicion moves to `981c00ee`+1 keep-best / the remaining 07-01..07-06 diffs. End-of-bound render via the patched demo. Chase-eval doesn't exist at this rev; hard-fail N/A (quiet-reset era).

<details><summary>the backport diff (brain.rs, the only file)</summary>

```diff
diff --git a/crab-world/src/bot/brain.rs b/crab-world/src/bot/brain.rs
index 709fb8b..874a03a 100644
--- a/crab-world/src/bot/brain.rs
+++ b/crab-world/src/bot/brain.rs
@@ -8,7 +8,11 @@ use burn::prelude::*;
 use super::actuator::ACTION_SIZE;
 use super::sensor::OBS_SIZE;
 
-const HIDDEN_SIZE: usize = 256;
+// PROBE F (rl#351, diagnostic, build-local — NOT landed): the mlp512x3 single-flip
+// backport. 256 -> 512 and a third trunk fc/ln layer, matching 56754c32's arch
+// exactly (its 5a A/B held everything but trunk depth x width equal). Arch is the
+// ONLY diff vs the green Probe E build of this same tree.
+const HIDDEN_SIZE: usize = 512;
 
 /// Initial policy log-std (std ≈ 0.2): start with low exploration, see `new`.
 const LOG_STD_INIT: f32 = -1.6;
@@ -25,8 +29,10 @@ const LOG_STD_MAX: f32 = 0.5;
 pub struct CrabBrain<B: Backend> {
     trunk_fc1: nn::Linear<B>,
     trunk_fc2: nn::Linear<B>,
+    trunk_fc3: nn::Linear<B>,
     trunk_ln1: nn::LayerNorm<B>,
     trunk_ln2: nn::LayerNorm<B>,
+    trunk_ln3: nn::LayerNorm<B>,
 
     policy_fc: nn::Linear<B>,
 
@@ -46,8 +52,12 @@ impl<B: Backend> CrabBrain<B> {
         let trunk_fc2 = nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
             .with_bias(true)
             .init(device);
+        let trunk_fc3 = nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
+            .with_bias(true)
+            .init(device);
         let trunk_ln1 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);
         let trunk_ln2 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);
+        let trunk_ln3 = nn::LayerNormConfig::new(HIDDEN_SIZE).init(device);
 
         // Small-gain init on the policy head. The trunk ends in a LayerNorm
         // (unit-scale output), so a default-initialised head emits means ~±1 —
@@ -82,8 +92,10 @@ impl<B: Backend> CrabBrain<B> {
         Self {
             trunk_fc1,
             trunk_fc2,
+            trunk_fc3,
             trunk_ln1,
             trunk_ln2,
+            trunk_ln3,
             policy_fc,
             value_fc1,
             value_fc2,
@@ -97,6 +109,9 @@ impl<B: Backend> CrabBrain<B> {
         let x = burn::tensor::activation::relu(x);
         let x = self.trunk_fc2.forward(x);
         let x = self.trunk_ln2.forward(x);
+        let x = burn::tensor::activation::relu(x);
+        let x = self.trunk_fc3.forward(x);
+        let x = self.trunk_ln3.forward(x);
         burn::tensor::activation::relu(x)
     }
 
```

</details>

<a id="c63"></a>
### 63. 2026-09-29 22:52Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900587713)

#### Decision-tree step → Probe G launched (job 2551)

**Branch taken: F DEGRADED** → `56754c32` (mlp256 → mlp512x3) convicted for cold-start learning; E-vs-F was itself the minimal A/B. Probe G runs main with the **arch controlled** — it serves both endgame halves: E-like learning green-lights the mlp256 fix-or-revert on main AND clears the remaining `8bb5009`..main window on this task shape; a plateau/flatline isolates the second breaker with the arch out of the equation.

**Landed first: mlp256 restored to the arch registry** (`966b0ab`, "arch: restore mlp256 as an opt-in leaf"). The leaf is byte-compatible with the culled original (`56754c32^` — era checkpoints load unchanged); `ArchId::DEFAULT` stays mlp512x3 pending G's verdict. So Probe G runs a **genuine main binary** — no build-local patch this time; `--arch mlp256` is the recipe difference vs Probe A, everything else is A's exact June-approximation (`--terrain flat --band-max-m 9`, floor −1.0).

**Pin:** main `966b0ab` (the restore commit itself). Binary `rl-train-966b0ab`, sha256 `7b1a6f83fc728e2e3163b655cff4e00fb852cff2e2be43d8ccf3703f494799b7`, built by job 2551 from that rev (fmt+clippy clean; full workspace suite green: 341+270 passed, 0 failed). Render pin `rl-demo-966b0ab` sha256 `843adaaf7deaa449cca33baafecf765d1eda89e2ba3d5e5f9b5bc4ef0e7d692e` — same rev, loads the mlp256 checkpoint via the restored registry.

**Smoke-verified pre-launch:** cold iter 0 + checkpoint save under the exact probe flags; `brain.bin` 557 KB — the mlp256 size (mlp512x3 writes 2.9 MB), so the flag is in the artifact, not just the argv.

**Run:** trainer instance `canary-probe-g-mlp256`, cold (fresh scratch init), started **2026-08-18 04:35 UTC**. Rollout shape matches A–F:
```
rl-train learn --arch mlp256 --terrain flat --band-max-m 9 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
Verified live: the running process's executable = the pinned binary, banner `terrain Flat, band 1.5..9 m`, iter 0 complete, GPU-resident (218 MiB, RTX 2080; 1.2 GiB total in use — ample headroom). Assets: the same read-only `sally.glb` share as probes A–F.

**Bound:** 24 h via the stop-bound timer — armed, timer readback verified: stops **2026-08-19 04:35:41 UTC**, stopguard +120 s.

**Ruler (assess, next box):** TRAIN-SIDE reach on the probe's own flat ≤9 m band — chase-eval stays struck for flat probes (canonical-GCR evals, Probe A amendment). Like-for-like yardstick is **Probe A** (same rev family, same REACH 0.2 criterion, arch the only diff: 0.01–0.05 flatline); E/D/F comparisons are directional (REACH 0.8 era). Integrity hard-fails (rl#343) counted; end-of-bound render via `rl-demo-966b0ab` (canonical-GCR ground — off-distribution, weighted as on A).

<a id="c64"></a>
### 64. 2026-09-29 22:52Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900590426)

#### Decision-tree step → Probe H launched (job 2594)

**Branch taken: G RED** (flatline 0.009–0.021 ≈ A's 0.01–0.05, arch controlled, 0 hard-fails) → the second breaker in `8bb5009`..main dominates cold learning regardless of arch; mlp512x3's conviction stays era-scoped, the mlp256 default-flip stays on hold. First single per the tree: **`8bb5009` REACH_RADIUS 0.8→0.2**.

**Pin: the `966b0ab` tree (Probe G's exact rev) + build-local REACH_RADIUS 0.2→0.8 — the ONLY change vs G.** REACH_RADIUS is a const woven into tests/const-asserts, so this is an F-style build-local patch, not a landed flag. Full functional diff: `targets.rs` `REACH_RADIUS 0.2 → 0.8`, plus the rollout banner patched to print the compiled reach so the ARTIFACT self-proves in-log.

**Binary:** `rl-train-966b0ab-reach0.8` in the trainer's state dir, sha256 `da2458b355fd66a637acd1e0493e7652ef7ba529e6ba7e26c2e4606610fe4392`. Smoke-verified pre-launch: banner `reach 0.8 m`; cold iter 0 + checkpoint save; `brain.bin` 556,880 B (the mlp256 size) — arch AND reach are in the artifact, not just the argv.

**Run:** trainer instance `canary-probe-h-reach0.8`, cold (fresh scratch init), started **2026-08-19 12:06:54 UTC**. Rollout shape matches A–G:
```
rl-train learn --arch mlp256 --terrain flat --band-max-m 9 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
Verified live: the running process's executable = the pinned binary; banner `terrain Flat, band 1.5..9 m, reach 0.8 m`; wgpu adapter RTX 2080. **Iter 0: reward(20) 50.6, reach 1.00 over 2 episodes** — under 0.8 the random policy already terminates grab episodes, so the flip is semantically live from tick one (G's iter 0: reach `-` over 0 ep).

**Bound:** 24 h via `canary-stop-bound` — armed, timer readback verified: stops **2026-08-20 12:06:59 UTC**, stopguard +120 s.

**sps check (the G caveat):** H holds ~279 sps at iter-0 rollout — G's 264, not A's ~800, on the same recipe shape. So the slowdown reproduces across G/H (arch-restored main builds) and isn't G-specific co-tenancy; cause still not chased here — the per-iter axis is the comparison (identical 6144 transitions/update).

**Split (assess, next box):** GREEN convicts `8bb5009` REACH 0.2 for cold-start learning on main (endgame: fix-or-revert reach); RED acquits REACH and pushes to the remaining 07-10..main window (baked-collider regen `2760e243`, env swap `27c1376f`, band/obs/terrain cluster, physics cluster). With reach 0.8 the train-side reach metric matches E's definition directly — E (0.6 by ~2.5k iters, 0.83–0.87 from 10k) is now a like-for-like yardstick.

<a id="c65"></a>
### 65. 2026-09-29 22:53Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900594719)

#### Decision-tree step → Probe H2 launched (job 2602)

**The instrument fix is LANDED, not patched: main `2809138`** — "training: flat runs skip the periodic chase-eval". A flat run's `BestKeeper` now carries no evaluator (`Option<Evaluator>` = `None` keyed on `TrainTerrain::Flat` at the one construction site): no eval spend, no `best/` promotion (flat checkpoints are diagnostic-only anyway), and the H-class wedge is unreachable by construction. The period still fires so the per-bearing train-reach window flushes on the same cadence a GCR run's would — that telemetry stays the flat ruler. The `--terrain` flag doc now states the skip; new unit test `no_evaluator_spends_nothing_but_paces_the_flush`. Full touched-path suite green before land: fmt + clippy `-D warnings` clean, release builds of `rl-train`/`rl-demo`/`game`/`rl-update-ui` green, workspace tests 344+275 passed / 0 failed, and the ignored `rest_pose_has_zero_torque_and_no_progress` eval test passed (183 s).

**Probe: identical semantics to H — REACH single-flip on G's recipe, on the fixed tree.** Pin main `2809138` + the F/H-precedent build-local patch (`REACH_RADIUS 0.2 → 0.8` + banner prints the compiled reach); nothing but the eval skip landed.

**Binary:** `rl-train-2809138-reach0.8`, sha256 `39250531ebb1c2ae76697f42f8564ed2326847c9c5d6d9e9550e8cb21fac7591`. Smoke-verified pre-launch: banner `terrain Flat, band 1.5..9 m, reach 0.8 m`; boot line `[best] chase-eval DISABLED for this run: flat rollout terrain … (rl#351)`; cold iter 0 + checkpoint save; `brain.bin` 556,880 B (the mlp256 size) — arch, reach, AND the skip are all in the artifact.

**Run:** trainer instance `canary-probe-h2-reach0.8`, cold (fresh scratch init), started **2026-08-20 13:17:47 UTC**. Rollout shape matches A–H:
```
rl-train learn --arch mlp256 --terrain flat --band-max-m 9 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
Verified live: the running process's executable = the pinned binary; GPU-resident (218 MiB, RTX 2080); iters advancing (iter 3 at ticks 12,288, checkpoints saving).

**Bound:** 24 h via the stop-bound timer — armed, timer readback verified: stops **2026-08-21 13:18:02 UTC**, stopguard +120 s.

**sps note:** H2 holds **~920 sps steady** — the G/H ~270-sps anomaly did NOT reproduce on the `2809138` build (A-class throughput is back; bound buys ~25M ticks vs H's 0.3M). The anomaly now dates somewhere in `966b0ab`..`2809138` build/config space, or was environmental; not chased here.

**Split (assess, next box):** GREEN (E-like, sustained ≥0.6) convicts `8bb5009` REACH 0.2 for cold-start learning on main (endgame: fix-or-revert reach); RED acquits REACH → remaining suspects are the 07-10..main window (baked-collider regen `2760e243`, env swap `27c1376f`, band/obs/terrain cluster, physics cluster). With reach 0.8, E (0.6 by ~2.5k iters) and B (rung-1 at iter 317) are the like-for-like yardsticks.

<a id="c66"></a>
### 66. 2026-09-29 22:53Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900599167)

#### Probe M launched (job 3001) — the main-side confirmation half of the shove conviction

**Pin:** current `main` `138d1eb` (2026-09-02, the Probe L assess landing — post-rl#414 rewrite; the convicted shove commit is `074c1263` → `d577e9c9` in new SHAs) **+ ONE build-local change: `SHOVE_START_PROB` 1/640 → 0.0** (`shove.rs`; the draw still consumes the training RNG each tick so seeding is unchanged — no burst ever arms). rl#415's fix was not landed at launch time, so this is the item's fallback path; the full working-tree delta is `build-local.patch` beside the run script (2 lines, that constant only).

**Config = Probe A's exact recipe minus shoves:**
```
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```
Everything else on main stays LIVE (momentum solver, joint dampers, air drag, REACH 0.2, target-height band, obs + terrain channels, integrity hard-fail) — the point of the probe.

**Binary:** `rl-train-138d1eb-shoveoff` in the trainer's state dir, sha256 `6bb321c0a673555a83930fde4296f0973d44f355348a1ac5b101ed0c45da9396`, release build of that worktree (kache-warm, 1m42s).

**Smoke (pre-launch, 110 s cold):** iter 0 → 23, GPU update path live (RTX 2080), `terrain Flat, band 1.5..9 m` banner, `[best] chase-eval DISABLED for this run: flat rollout terrain` (main's own flat rule — so the box's "chase-eval valid on main flat" premise is inverted: main SKIPS it on flat by construction since `2809138`, exactly the I/J/K/L neutralization, now landed), `[bearing-reach]` flush present, checkpoint set saved, smoke `brain.bin` 2,954,170 B (mlp512x3 at main's obs width). Assets: the same shared `sally.glb` as A–L (body digest `0x33628dfeaeb29dd3`, preflight OK).

**Run:** trainer instance `canary-probe-m-138d1eb`, cold (fresh scratch init), started **2026-09-02 01:46 UTC**. Verified live: the instance's main process executable = the pinned binary; 362 MiB on the 2080 (sole canary — ample headroom); ~940 sps steady at iter 4.

**Bound: 48 h** via the stop-bound timer — armed, timer readback verified: stops **2026-09-04 01:46:32 UTC**, stopguard +120 s, stop-at record written. The 24 h + the ONE extension pre-committed per the budget rule (L precedent: separation needs ~55M+ ticks; ~75M expected in-bound at this sps). The assess gets NO further extension.

**Split (assess, next box):** I/J/L-LIKE CLIMB (~0.045 by 10M ticks, 0.07+ by 40M, 0.10–0.17 late) confirms the conviction transfers to main with every latest feature live → Endgame on the shove path (rl#415 fix, then the re-enable canaries from a working baseline); FLOOR (A/C/G/H2/K-like 0.01–0.12 trendless; Probe A is the direct shoves-ON control at 0.01–0.05) means a second main-side breaker post-`074c1263` → bisect `074c1263`..main. Nothing landed in the repo by this job (build-local only).

<a id="c67"></a>
### 67. 2026-09-29 22:53Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900603006)

#### Probe N launched (job 3161) — bisect midpoint of the post-shove range

**Pin:** `2e5605f` (2026-08-12, "rl#353 stage 8 review round" — commit 170/339 of `d577e9c9`..`138d1eb`, ancestor of main) **+ build-local patch** (`build-local.patch` beside the run script, 3 hunks): `SHOVE_START_PROB` 1/640 → 0.0 (`shove.rs`; the draw stays in the RNG stream, no burst ever arms) and the periodic canonical-GCR chase-eval skipped in `BestKeeper::maybe_snapshot` (`best.rs`; this rev predates main's flat-skips-eval rule, so the I–L neutralization is reapplied — the `true` return keeps the `[bearing-reach]` flush cadence, the ruler). rl#415's fix is not in this pin's range, so the constant patch is the item's fallback path.

**Config = Probe A/M's exact recipe:**
```
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```

**Binary:** `rl-train-2e5605f-shoveoff-evalskip` in the trainer's state dir, sha256 `3e774e02ba7cf53a3ed336bff615eb86986190d70eca8c57c3b6d11dd8ffde43` (release build of the pin worktree, kache-warm, `Finished release profile [optimized] in 2m 59s`).

**Smoke (110 s cold):** iter 0 → 10, GPU update path live (RTX 2080), `rollout world: terrain Flat, band 1.5..9 m`, `[best] periodic chase-eval SKIPPED — probe-N build-local neutralization` at iter 0 (never again in-period, as on I–L), `[bearing-reach]` flush present, checkpoint set saved, `brain.bin` 2,954,170 B (mlp512x3 at this rev's obs width — same size as M's). Assets: the same `assets/sally.glb` share as A–M (body digest `0x33628dfeaeb29dd3`, preflight OK).

**Run:** trainer instance `canary-probe-n-2e5605f`, cold (fresh scratch init), started **2026-09-04 03:10 UTC**. Verified live: the main process's executable = the pinned binary; NRestarts 0, exactly one cold-start line; 358 MiB on the 2080 (4.1/8.2 GiB used with the test display's 3.6 GiB — ample); ~720 sps steady at smoke, 621 during a concurrent build, → ~60–70M ticks expected in-bound.

**Bound: 48 h** via `canary-stop-bound` — armed, timer readback verified: stops **2026-09-06 03:11:01 UTC**, stopguard +120 s, stop-at record written. 24 h + the ONE extension pre-committed (L precedent: separation needs ~55M+ ticks). The assess gets NO further extension.

**End-of-bound render wired now, not at assess:** `rl-demo-2e5605f-flat-shoveoff` in the trainer's state dir, sha256 `8ac1d5c7c986c2d6b658a40954e14a75c1811758e48f44d767407981763e4536` — the pin's own `rl-demo` + the same shove-off hunk + `TerrainGrid::gcr()` returning the flat 512 m grid build-local (one seam, every consumer — spawn frame, ground sampling, render — sees the probe's ground; `build-local-demo.patch` beside the run script).

**Split (assess, next box):** L-LIKE (first 5M bucket ≥0.06, ≥0.07 by 40M) → the breaker is in `2e5605f`..`138d1eb` (169 commits; next midpoint ~`d577e9c9`+255); M-LIKE (≤0.056 through 65M) → it is in `d577e9c9`..`2e5605f` (170 commits; next midpoint ~commit 85). Nothing landed in the repo by this job (build-local only).

<a id="c68"></a>
### 68. 2026-09-29 22:53Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900605493)

#### Probe O launched (job 3402) — bisect midpoint of `2e5605f`..`138d1eb`

**Branch taken: N CLIMB / M FLOOR** → the second main-side breaker sits in `2e5605f`..`138d1eb` (169 commits, 08-12 → 09-01). Midpoint pin **`40fde25`** (2026-08-17, "rl#386: ram test measures net body momentum, 32 m/s ram" — commit 85/169, ancestor of main; 84 commits remain above it). It builds unmodified, and its rollout world is intact — no neighbor needed.

**Build-local patch = N's two hunks, byte-identical** (`build-local.patch` beside the run script): `SHOVE_START_PROB` 1/640 → 0.0 (`shove.rs`) and the periodic canonical-GCR chase-eval skipped in `BestKeeper::maybe_snapshot` (`best.rs` — this rev still runs it on flat: no flat-skip rule exists at `40fde25`; the `true` return keeps the `[bearing-reach]` flush cadence, the ruler). Both applied clean with no offset.

**Config = Probe A/M/N's exact recipe:**
```
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512
RL_LOG_STD_FLOOR_END=-1.0
```

**Binary:** `rl-train-40fde25-shoveoff-evalskip`, sha256 `5a183268bd3bfd1f2c820f6907e9660dfda9fb4e134d6830c0a29c244f45a1f9` (release build of the pin worktree, kache-warm, `Finished release profile [optimized] in 3m 58s`).

**Smoke (240 s cold, scratch ckpt dir, wiped after):** iter 0 → 8, GPU update path live (RTX 2080, `adapter confirmed as hardware GPU`), `rollout world: terrain Flat, band 1.5..9 m`, `[best] periodic chase-eval SKIPPED` at iter 0, `[bearing-reach]` flush present, full checkpoint set saved, `brain.bin` 2,954,170 B (mlp512x3 — same size as M/N). Assets: the same `rl-train-main/assets/sally.glb` share as A–N.

**Run:** trainer instance `canary-probe-o-40fde25`, cold (fresh scratch init, exactly one cold-start line), started **2026-09-06 22:18:32 UTC**. Verified live: the main process's executable = the pinned binary; NRestarts 0; 358 MiB on the 2080 (2.2/8.2 GiB used including a concurrent game process — ample); integrity hard-fail 0 so far.

**Bound: 48 h** via `canary-stop-bound` — armed, timer readback verified: stops **2026-09-08 22:18:37 UTC**, stopguard +120 s. 24 h + the ONE extension pre-committed; the assess gets NO further extension.

**Throughput caveat (carry into the assess):** this pin rolls out at **~250 sps** (iter 0–2: 247/255/235) vs N's ~720 and M's ~800–1000 on the same recipe shape — the G/H-era slowdown (264–279 sps on `966b0ab`, 08-18, "cause still not chased") is evidently present at `40fde25` and gone by `2e5605f`'s successors on main. A concurrent `game play` process (~1.1 cores) is the only CPU co-tenant and does not explain 3×. At 250 sps the 48 h bound yields **~43M ticks**, not N's 62M: the N-LIKE marks (out of the ≤0.056 band by 30M, ≥0.07 at 40M) are reachable; "≤0.056 through 65M" is not, so M-LIKE is judged as "≤0.056 through the full bound (~43M)" — M/K/A never left that band in any bucket, and N had separated by 30M, so the split stays decidable on the per-tick axis.

**End-of-bound render wired now:** `rl-demo-40fde25-flat-shoveoff`, sha256 `18e30f140b9d3500169c6fcfb86d5126ea384aa9042005a68f7a957e23c5ee9c` — the pin's own `rl-demo` + the same shove-off hunk + `TerrainGrid::gcr()` returning the flat 512 m grid build-local (`build-local-demo.patch` beside the run script, N's hunk applied clean).

**Split (assess, next box):** N-LIKE (out of the ≤0.056 band by 30M, ≥0.07 at 40M; report the end-of-run trend — N faded 0.116 → 0.068 past 50M) → breaker in `40fde25`..`138d1eb` (84 commits; next midpoint ~commit 42); M-LIKE (≤0.056 through the bound) → in `2e5605f`..`40fde25` (85 commits; next midpoint ~commit 42). Nothing landed in the repo by this job (build-local only).

<a id="c69"></a>
### 69. 2026-09-29 22:54Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900609966)

#### Decision-tree step → Probe Q launched (job 3704)

**Branch taken: P M-LIKE at the low edge** → the pre-registered single for breaker 2: the bevy 0.19 / rapier 0.35 bump.

**Pin: `88be0c1`** (2026-08-15, commit 48/85 of `2e5605f`..`40fde25`): `4a0a3a6` (bevy 0.18.1→0.19, bevy_rapier3d→0.36, rapier3d 0.35) plus the rl#212 pins that landed with it — `a0d853d` (momentum-fix port onto rapier 0.35 + the 0.35 contact-semantics pin, rl#318/rl#321), `a489a5e`, `5be7da6` (rebase reconciliation), `88be0c1` (solver 400 m/s speed-cap pin-off). Builds unmodified; rollout world intact (banner `rollout world: terrain Flat, band 1.5..9 m`).

**Patches:** P's `build-local.patch` (shove-off + chase-eval skip) and the demo terrain hunk apply clean, byte-identical to N/O/P; archived in the probe's `canary-probe-q-88be0c1/` state dir. Nothing landed in the repo (diagnostic, same as N/O/P).

**Pin binaries** (nix-shell, kache-warm, pin tree + pin Cargo.lock):
- `rl-train-88be0c1-shoveoff-evalskip` sha256 `18de2f44e2f54663e9a1c3ee57e4ce10695a50921efa648c27e6267405c196d1` — `Finished release profile [optimized] in 9m 22s`
- `rl-demo-88be0c1-flat-shoveoff` sha256 `658e453934a273ad6522420c86739449209ed63c9780c9e869a2e04c9b24b156` — `13m 58s`; pre-verified: loaded the smoke checkpoint (`play: loaded checkpoint from …` — visible only under `RUST_LOG=info`, the demo's default filter hides `crab_world::policy`) and rendered under lavapipe, rc 0.

**Smoke** (240 s cold, scratch ckpt, wiped): iter 0→6 on the RTX 2080, `[best] periodic chase-eval SKIPPED`, `[bearing-reach]` flush present, ckpt set saved, brain.bin 2,954,170 B (mlp512x3 — P's size exactly). sps(steady rollout) **191** (175–204/iter) **with Probe P co-resident** (P's 96 h bound runs to 23:08Z, left uncut per its verdict). P itself fell ~245 → ~210 sps under the overlap, so Q's sole-trainer rate is ~220 sps ≈ 110 ticks/s — **P-class throughput, not N's 720**: the bump moves sps neither way, consistent with commits 1–43 owning the slowdown. (A first smoke overlapped the demo build and read 50–74 sps; discarded, kept as `smoke-contaminated-by-build.log`.)

**Run:** trainer instance `canary-probe-q-88be0c1`, cold (fresh scratch init), started **2026-09-12 20:18:37 UTC**. Probe A recipe: `--terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512`, `RL_LOG_STD_FLOOR_END=-1.0`. the running process's executable = the pinned binary, NRestarts 0, 358 MiB GPU. Live at iter 1: 190 sps.

**Bound in ticks:** 40M needs ~101 h solo (~117 h at the contended rate) → pre-committed **120 h (432 000 s)** via the stop-bound timer, timer readback verified: stops **2026-09-17 20:18:37 UTC**, stopguard +120 s; projected 41–48M ticks. **NO extension.** The assess reads at the **30M and 40M tick marks** (re-derived from `ckpt/ticks.txt`), not at the bound.

**Split (assess, next box):** same `reach.awk` ruler on the per-tick axis. **O-LIKE** (≤0.01 through 30M, drift <0.6 m, reward never positive) convicts the bump `4a0a3a6`..`88be0c1` as breaker 2 → next: revert-test it on top of `40fde25`. **P-LIKE** (≥0.015 by 25–30M, drift ≥0.7 m, reward positive by ~3M ticks) puts breaker 2 in commits 49–85 (rl#340 stages 7–10) → next midpoint ~commit 67. Either way the box after is breaker 1's bisect (midpoint ~commit 22 of `12a6a77`..`7184c17`, sps as the co-signal).

<a id="c70"></a>
### 70. 2026-09-29 22:54Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900615218)

#### Probe S launched (job 4561) — breaker 1 content split at `c9f2428`

**Pin: `c9f2428`** (2026-08-13, "rl#340 stage 2 review round: guard the solver iteration counts") = `git rev-list --reverse 2e5605f..7184c17 | sed -n 31p`. Its parent `96be084` (commit 30) is rl#340 stage 2: `PHYSICS_SUBSTEPS` 2→4, contact damping ratio 5→10 at 30 Hz, solver iterations 4/1/1→8/4/4. `c9f2428` adds only the `SOLVER_ITERATIONS` const, its runtime assert and `#[cfg(test)]` suite edits. Above the pin, stage 3's carapace drag `bd5e7c7`/`5aefee4` (35–36) is the only physics/body change: 32 is an audio bitcrush constant, 33–34 are evidence, 37–43 are `#[cfg(test)]` suites. Cargo.lock is identical to P's. The pin builds unmodified, and its rollout world is intact (`rollout world: terrain Flat, band 1.5..9 m`). Stage 2 is live in the log: `num_solver_iterations: 8, num_internal_pgs_iterations: 4, num_internal_stabilization_iterations: 4`, `damping_ratio: 10.0`.

**Build-local (never landed):** P's `build-local.patch` and `build-local-demo.patch`, byte-identical to N/P/Q (sha256 `636b3cc6…` / `ff6ebe2c…`), applied clean. `SHOVE_START_PROB` is 0.0 and the periodic canonical-GCR chase-eval is skipped; `[best] periodic chase-eval SKIPPED` and the `[bearing-reach]` flush both appear in the log. The demo adds the flat `TerrainGrid::gcr()` hunk. Patches, run script and smoke log are archived in the run's state dir `canary-probe-s-c9f2428/`.

- `rl-train-c9f2428-shoveoff-evalskip` sha256 `28f6dc414b5bcb36a4c6acec1cf0e1181882706b9eaa2dcd71586a274853a714`, `Finished release profile [optimized] target(s) in 2m 57s`
- `rl-demo-c9f2428-flat-shoveoff` sha256 `5e239e535a58aab03e811b764d4f3a30a657e9cb23ecc70c24557f5be9d55bb5`, `7m 56s`. Pre-verified on the smoke checkpoint under lavapipe: `play: loaded checkpoint from …/smoke-ckpt`, 192 frames written on flat ground, `DRIVE_STATS` printed.

**Smoke (240 s cold, sole trainer, nice 19):** iter 0→8 on the RTX 2080, checkpoint set saved (`brain.bin` 2,954,170 B, the mlp512x3 size). `sps(steady rollout)` read **245** at iter 8 (per-iter 235–255), with 23.9–26.2 s per 3072-tick rollout ⇒ **~122 ticks/s**.

**Slowdown placed:** 245 sps at commit 31, against N's 720 at `2e5605f`, puts the 720→250 drop in commits 1–31. That fits the ~3× solver cost `96be084`'s message records (720/245 = 2.9×).

**Command** (Probe A's recipe, fresh init):
```
RL_LOG_STD_FLOOR_END=-1.0 rl-train-c9f2428-shoveoff-evalskip learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --checkpoint-dir canary-probe-s-c9f2428/ckpt
```

**Run:** trainer instance `canary-probe-s-c9f2428`, cold (one `cold-started` line), started **2026-09-23 23:55:28 UTC**. The main process runs the pinned binary (the sha256 of the running process's executable matches), NRestarts 0, 350 MiB GPU, and every iter saves a checkpoint. At iter 23 it had 73,728 ticks. Per-iter sps was 181–231 while the demo built and 237–259 after.

**Bound in ticks:** a 20M read at ~122 ticks/s needs ~45.5 h, so the bound is **60 h (216000 s)**, pre-committed via `canary-stop-bound`. Readback: `armed: canary-probe-s-c9f2428 stops at 2026-09-26 11:55:31 UTC (timer verified)`, stopguard +120 s, `stop-at` recorded. That projects ~26M ticks, and 20M still lands at rates down to ~93 ticks/s. **NO extension.** The assess reads at the **10M and 20M tick marks**, not at the bound: ~2026-09-24 22:40Z and ~09-25 21:30Z at 122 ticks/s, or ~09-25 01:00Z and ~09-26 02:00Z at P's contended 111. Re-derive both from `ckpt/ticks.txt`.

**Split (assess, next box):** same `reach.awk` ruler. **N-LIKE** (≥0.02 by 5–10M, ≥0.04 at 20M; N read 0.028/0.046/0.053/0.047 at 5/10/15/20M) puts breaker 1 in commits 32–43, with stage 3 drag the prime suspect. Next: single `5aefee4` (commit 36). **P-LIKE** (≤0.012 through 20M; P read 0.008/0.007/0.008/0.011) puts it in 1–31, with stage 2 the prime suspect. Next: single `b11ef09` (commit 29, stage 2's parent). That probe's smoke sps also tests whether `96be084` owns the slowdown.

<a id="c71"></a>
### 71. 2026-09-29 22:54Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900619177)

#### Probe T launched (job 4881) — single `b11ef09`, stage 2's parent

**Pin: `b11ef09`** = `git rev-list --reverse 2e5605f..7184c17 | sed -n 29p`. It is Probe S's pin minus exactly `96be084` and `c9f2428` (rl#340 stage 2 and its review round): `git diff b11ef09 c9f2428` touches only `crab-world/src/physics/mod.rs` and `crab-world/src/bot/sim_truth_test.rs`, and Cargo.lock is identical. The pin builds unmodified. The pre-stage-2 plant is live in-log: `damping_ratio: 5.0` and `num_solver_iterations: 4, num_internal_pgs_iterations: 1, num_internal_stabilization_iterations: 1` (S logs 10.0 and 8/4/4), with `PHYSICS_SUBSTEPS = 2` in source. The world is intact: `rollout world: terrain Flat, band 1.5..9 m`, and the Sally preflight digest `0x33628dfeaeb29dd3` is S's.

**Build-local (never landed):** the N–S patches, applied byte-identically (resulting diff sha256 `636b3cc6…` trainer, `ff6ebe2c…` demo). They set `SHOVE_START_PROB` to 0.0 and skip the periodic chase-eval; `[best] periodic chase-eval SKIPPED` and `[bearing-reach]` both appear in-log. Patches, run script, smoke logs and the ruler are archived in the run's state dir.

- `rl-train-b11ef09-shoveoff-evalskip` sha256 `56cc3c159eb467bc7b1be0ea92a0ffd00ee5cc685ebcd0f35ecabf286fa5a101`, `Finished release profile [optimized] target(s) in 11m 58s`
- `rl-demo-b11ef09-flat-shoveoff` sha256 `0de15b8e56dd9b5edd2202c7ae30ada811474354c1198abf653986566f6e6988`, pre-verified on the smoke checkpoint under lavapipe: `play: loaded checkpoint from …/smoke-ckpt`, 192 frames on flat ground, `DRIVE_STATS ticks=222 mean_abs_drive=0.30750 mean_effort=5.11802`

**Smoke (300 s cold):** iter 0→11, checkpoint set saved (`brain.bin` 2,954,170 B). Per-iter sps was 324–351 over iters 1–4, then 178–326 as host load rose from ~25 to ~40; steady sps was 272 at iter 11 (~134 ticks/s). The smoke shares a cgroup with the job-queue workers, so these wall numbers carry their load. Two controlled reads place the slowdown instead:

| read | N `2e5605f` | T `b11ef09` | S `c9f2428` |
|---|---|---|---|
| cold smokes side by side, one cgroup, same 7 min: CPU-ms per tick after iter 0 | 23.6 | 23.4 | 73.2 (one 3,072-tick interval; its rollouts took 100–144 s vs N/T's 34–39 s) |
| live runs, one slice, 03:27:08–03:27:56Z: per-iter sps | — | 620–661 | 207–214 |

**Slowdown placed: stage 2.** T costs what N costs per tick; S costs 3.1× that. Live, T runs 3.0× S side by side, against N/S = 720/245 = 2.9× historically. The 720→250 drop is `96be084`/`c9f2428`, which fits the "~3× solver cost" in `96be084`'s commit message.

**Command** (Probe A's recipe, fresh init):
```
RL_LOG_STD_FLOOR_END=-1.0 rl-train-b11ef09-shoveoff-evalskip learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --checkpoint-dir canary-probe-t-b11ef09/ckpt
```

**Run:** trainer instance `canary-probe-t-b11ef09`, cold (one `cold-started` line), started **2026-09-26 03:14:50 UTC**. The main process runs the pinned binary (the sha256 of its running executable matches), NRestarts 0, 350 MiB GPU, a checkpoint save every iter, 0 integrity lines. The first 47 iters covered 144,384 ticks at 188 ticks/s, co-resident with S (S's bound ends 2026-09-26 11:55:31Z).

**Bound in ticks:** 48 h (`172800 s`) via `canary-stop-bound`. Readback: `armed: canary-probe-t-b11ef09 stops at 2026-09-28 03:14:59 UTC (timer verified: Mon 2026-09-28 03:14:59 UTC); stopguard fires +120s after.`, and `stop-at` is recorded. 20M lands in-bound at ≥116 ticks/s: the smoke's contended 134 ticks/s projects 23M, the live 188 ticks/s 32M, N's 360 ticks/s 62M. **NO extension.** The assess reads at the **10M and 20M tick marks**, not at the bound.

**Split (assess, next box):** same `reach.awk` ruler. A copy in the run's state dir reproduces S's 0.006/0.006/0.008/0.011.
- **N-LIKE** (≥0.02 by 5–10M, ≥0.04 at 20M; N read 0.028/0.046/0.053/0.047) ⇒ stage 2 is breaker 1, and T-vs-S is its minimal A/B. Both breakers would then be convicted, so the Endgame box follows.
- **P-LIKE** (≤0.012 through 20M; S read 0.006/0.006/0.008/0.011) ⇒ breaker 1 lies in commits 1–29, which touch nothing under `crab-world/src/physics/` or `training/`. Their crab-world changes are controls/instrument, sky/moon, ground-look shaders, play mode, plane aero (`vehicle.rs`, 18–19), a collider-check fixture (26) and wav (27). Next: bisect 1–29.

A 20M read between the two is reported as such.

<a id="c72"></a>
### 72. 2026-09-29 22:55Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5900622864)

#### Endgame A/B launched (job 5026): three arms on current main

**`--shove-prob` is moot.** rl#416 (`fed23bb`) deleted random shoves a day after rl#415 added the flag; `rl-train learn --help` at `faa21d7` lists no shove option, so every arm is shove-free by construction.

**Main's plant against the two convicted physics changes:**

| knob | T `b11ef09` (climbs) | S `c9f2428` (floors) | main `faa21d7` |
|---|---|---|---|
| substeps | 2 | 4 | 2 (rl#392) |
| contact ζ at 30 Hz | 5 | 10 | 20 |
| solver (outer, PGS, stab) | 4, 1, 1 | 8, 4, 4 | 2, 12, 3 |
| rapier | 0.32 | 0.32 | 0.35, four default swaps pinned back |

Main leaves four other 0.35 default swaps live: allowed linear error 1 → 5 mm, corrective velocity cap 10 → 3 m/s, prediction distance 2 → 20 mm, contact clustering (new, on).

Probe M (main `138d1eb`, same substeps-2 / ζ20 family) started like the climbers. Its first-1M-tick drift was 0.59–0.82 m (T 0.82–1.11, N 0.69–0.97; S 0.33–0.43, P 0.28–0.43), and its 5M reach was 0.023 (T 0.027, N 0.028; S 0.006, P 0.008). Its floor set in late, at ≤0.056 through 70M, which is the bump's shape (P → Q). So rl#392's substeps restore has likely already neutralized breaker 1, and the bump is the live suspect. Each arm tests one half.

**Arms.** Pin `faa21d7` (current main), cold, Probe A's recipe: `rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --ticks 40000000` with `RL_LOG_STD_FLOOR_END=-1.0`. Main skips the periodic chase-eval on flat itself; the arms carry no other build-local patch.

| arm | change vs E0 | trainer sha256 |
|---|---|---|
| E0 | none (control) | `0c2b9846…` |
| E1 | breaker-1 residue reverted: contact ζ 20 → 5, solver (2,12,3) → (4,1,1) | `a31d9115…` |
| E2 | the four unpinned swaps at 0.32 values: linear error 0.001, corrective cap 10, prediction 0.002, clustering off | `93176bbc…` |

Each patch shows in its run's startup `IntegrationParameters` line (E1: `damping_ratio: 5.0`, `num_solver_iterations: 4, num_internal_pgs_iterations: 1, num_internal_stabilization_iterations: 1`; E2: `normalized_allowed_linear_error: 0.001`, `normalized_max_corrective_velocity: 10.0`, `normalized_prediction_distance: 0.002`, `contact_clustering: false`). Patches and run scripts are in each arm's `canary-endgame-*/` state dir.

**Run.** Trainer instances `canary-endgame-{e0-faa21d7,e1-faa21d7-prestage2,e2-faa21d7-rapier032params}` started 2026-09-27 01:35:43Z. Each shows one cold start, NRestarts 0, 358 MiB GPU, and a running-executable hash matching its pinned trainer. Co-run steady rollout: E0 470, E1 384, E2 475 sps (~235 / 192 / 238 ticks/s), so 40M lands in ~47–58 h.

**Bound in ticks.** `--ticks 40000000` makes the learner exit cleanly at the budget, and its odometer survives restarts. A 144 h stop-bound timer is the backstop; all three timers read back 2026-10-03 01:36:10Z. Probe T was stopped at 25.37M ticks to free its cores. Its verdict stands; its 20M and 25M buckets read 0.085 and 0.081.

**Pre-registered reads** at 10, 20, 30 and 40M on the `reach.awk` 5M buckets (0.2 m ruler). Per arm at 40M:
- CLIMB: 35–40M bucket ≥ 0.07 and ≥ 1.5× E0's (N read 0.106 there; M never passed 0.056 through 70M).
- M-LIKE: every bucket ≤ 0.056.
- DEEP: ≤ 0.012 through 20M.

What each outcome means:
- E1 CLIMB vs E0: stage 2's residue still breaks main. E1 ≈ E0: breaker 1 is already neutralized on main.
- E2 CLIMB vs E0: breaker 2 lives in the four unpinned swaps, and the fix is to pin them. E2 ≈ E0: it lives in the 0.35 solver itself.
- E0 CLIMB: main already learns.

Each arm's 40M checkpoint is rendered with its own flat-wired `rl-demo`. Fix-or-revert and the re-enable boxes follow the verdict.

<a id="c73"></a>
### 73. 2026-09-30 16:01Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5914960248)

Endgame A/B: E0 and E2 have finished their 40M budgets; E1 stopped on an integrity failure; E12 is still running.

| arm | 30M | 35M | 40M | integrity hard-fails |
|---|---|---|---|---|
| E0, main control | 0.060 | 0.062 | **0.097** | 0 |
| E1, ζ5 + solver (4,1,1) | 0.062 | 0.058 | 0.044 partial at 36.07M | **1** |
| E2, 0.32 contact params | 0.063 | 0.063 | **0.083** | 1 |
| E12, both reverts | pending | pending | pending (24.8M now) | 0 |

Episode-weighted reach in 5M-tick buckets, same 0.2 m ruler. E0 and E2 each stopped cleanly at 40,000,512 ticks. Main control crosses the pre-registered absolute 0.07 threshold; E2 is 0.86× control, below the 1.5× improvement requirement. No evidence here favors reverting these contact defaults. This clears the earlier M-like floor on this flat, short-band task; it does not establish terrain chase or attribute the recovery to a single intervening commit.

E1 hard-failed at 2026-09-30 12:54:54Z: ClawWrist(Right) reached 120.6 m/s at episode-step 981. Its intact checkpoint is at 36,065,280 ticks. Per the remediation notice, I am counting the failure as experimental instability evidence and leaving E1 stopped. Its partial final bucket is not a 40M verdict. E2's earlier hard-fail and all previously reported warm-resume seams remain qualifications.

First→last 2000 iterations: E0 reach 0.0359→0.0878, drift 0.885→1.320 m, reward −10.24→+9.54; E1 0.0259→0.0553, 0.749→1.061 m, −9.48→−2.40; E2 0.0315→0.0798, 0.862→1.241 m, −11.46→+7.84. Final complete-bucket reward-positive fractions: E0 0.908, E2 0.881.

All arms are shove-free: main deleted shoves in rl#416, so `--shove-prob` no longer exists. E12 retains its finite 40M budget. The Endgame box stays unchecked until its final read and the arm renders are assessed. No production code changed at this read.

<a id="c74"></a>
### 74. 2026-10-01 05:01Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5925081278)

Endgame render update (job 5026): E0/E1/E2 completed; E12 still training.

All three matching flat-ground demos loaded their arm's checkpoint and exited 0. Each MP4 is verified as 20.000 s, 1,280 frames, 640×360, seed 351, target ball enabled, lavapipe. All are shove-free by construction (`--shove-prob` unavailable after rl#416; effective probability 0).

| arm | checkpoint ticks | mean absolute drive | mean effort | hard-CCD clamp candidates | visual assessment |
|---|---:|---:|---:|---:|---|
| E0 | 40,000,512 | 0.74514 | 24.36799 | 0 | tumbles early, then essentially static from ~7.5 s; no demonstrated chase |
| E1 | 36,065,280 | 0.77081 | 25.54440 | 0 | overturned/rotating, remains separated from the ball; no demonstrated chase |
| E2 | 40,000,512 | 0.69287 | 21.53453 | 2 | moves and turns, but no convincing closing-and-holding sequence |

The 0.097 E0 training result clears the old floor; it is NOT restored deterministic chase. Demo inference takes action means, while train-side reach includes stochastic exploration. These clips are one seed, and no render/training graph defect has been established. E1's terminal training blowup and E2's earlier hard-fail remain qualifications.

Read-only review: 7/10, conditional support for retaining the tested main settings; no tested revert is justified by the learning results so far. The original CLIMB wording was ambiguous for the control: E0 passes the absolute 0.07 criterion; the 1.5×-control improvement condition applies to treatment arms, not E0 itself. E2 tests four exposed parameters, not a full dependency downgrade.

Added unchecked boxes for the flat deterministic-chase gap (after rl#20's hull body lands) and then a one-feature full-distance-band canary. Terrain keeps its existing separate amplitude-curriculum box. New sensors remain enabled. Endgame is still unchecked: E12 has about 38.9M/40M ticks. Its matching demo has now built successfully (release build exit 0, 22m27s); all temporary source edits are restored.

Clips, contact sheets, raw render logs, validation hashes, build log and reviewer findings are preserved in job 5026's durable artifacts as `e0-40m.*`, `e1-36m.*`, `e2-40m.*`, `render-validation.json`, `build-e12-final.log`, and `review.md`.

<a id="c75"></a>
### 75. 2026-10-01 07:15Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5926592440)

Endgame verdict: retain the tested main settings. The historical floor-remedy revert is obsolete at this pin; deterministic chase is still unsolved.

The unmodified `faa21d7` control clears the pre-registered absolute 0.07 threshold. Neither parameter revert, alone or combined, improves on it. This follows the planned E0-clears-floor branch, so no new production revert is warranted. It does not identify which intervening change repaired the floor, establish statistical equivalence, or claim restored chase. E2 tests four exposed parameters, not a full dependency downgrade.

| Arm | Change from control | Final checkpoint ticks | 30–35M reach | 35–40M reach | Training integrity hard-fails |
|---|---|---:|---:|---:|---:|
| E0 | none | 40,000,512 | 0.062 | **0.097** | 0 |
| E1 | contact damping 20→5; solver (2,12,3)→(4,1,1) | 36,065,280 | 0.058 | 0.044, **partial/censored** | 1 |
| E2 | linear error .005→.001; corrective cap 3→10; prediction .02→.002; clustering off | 40,000,512 | 0.063 | **0.083** | 1, resumed |
| E12 | E1 and E2 together | 40,000,512 | 0.072 | **0.048** | 0 |

E2 is 0.86× control; E12 is 0.49×. Both miss the 1.5× treatment-improvement criterion. E12 regressed late after reaching 0.072; it was not floor-like throughout. E1 stopped on the recorded ClawWrist(Right) 120.6 m/s violation and was not restarted. Its partial bucket is not a 40M verdict. E0, E2 and E12 exited cleanly at their tick budgets. All training units are now inactive.

The original CLIMB wording was ambiguous for the control: E0 cannot exceed 1.5× itself. The explicit E0 branch is interpreted against the absolute 0.07 threshold; the relative condition applies to treatments. This qualification is recorded, not hidden by relabeling all arms CLIMB.

Historical confirmations and disposition:

- Architecture: E/F isolated mlp256 versus mlp512x3 on the era pin; G showed that restoring mlp256 alone did not rescue main. The conviction remains era-scoped; no default architecture change follows from these results.
- Random shoves: J/K/L confirmed the harm. Main deleted shoves in rl#416. **`--shove-prob`: unavailable; effective shove probability 0 on E0, E1, E2 and E12.** No arm silently relies on a shove default.
- Stage 2: T/S differ only by `96be084` and `c9f2428`. Main already restored two substeps and subsequently changed damping/solver settings; E1 tests the remaining old-parameter restoration and is unstable.
- Dependency bump: P/Q adds it; O/R removes it and restores P-level learning. Those historical confirmations stand. The later control clears the old floor, while E2/E12 provide no reason to restore the tested old contact parameters on main.

All new arms use the same `faa21d7` source pin, flat terrain, band ≤9 m, mlp512x3, 6 workers, 2 environments, horizon 512, log-std floor −1.0, one cold initialization and a 40M-tick budget. Main skips canonical-GCR evaluation on flat. The ruler is episode-weighted 0.2 m reach in 5M-tick buckets; `CUT=40000000` excludes the final overshoot iteration. Numerous warm seams remain a limitation: GPU initializations were 11/12/13/9 for E0/E1/E2/E12, with strictly increasing logged tick counters. There is one stochastic training run per arm.

All four matching flat-ground renders loaded the intended checkpoint and exited 0. Each is 20.000 seconds, 1,280 frames, 640×360, seed 351, target ball enabled, using lavapipe. The contact sheets were inspected.

| Arm | Mean absolute drive | Mean effort | Hard-CCD clamp candidates | Visual verdict |
|---|---:|---:|---:|---|
| E0 | 0.74514 | 24.36799 | 0 | tumbles, then essentially static from ~7.5 s; no demonstrated chase |
| E1 | 0.77081 | 25.54440 | 0 | overturned/turning and separated from target; no demonstrated chase |
| E2 | 0.69287 | 21.53453 | 2 | moves/turns without convincing closing and holding |
| E12 | 0.81118 | 27.66004 | 2 | early motion, then settles with the ball still distant; no demonstrated chase |

Training reach includes exploration noise; the demos use deterministic action means. One render seed is supplementary evidence, not a success-rate estimate. No render/headless graph defect has been established. The absence of convincing pursuit is an explicit remaining problem, not erased by E0's training metric.

Read-only Explore review: initial 7/10, final **8/10**, approving this narrow closure subject to the completed E12 render and retained qualifications. Both conditions are now met. No reviewer changed files or built code.

Endgame disposition: check this increment via the explicitly allowed obsolete-increment route, retaining the tested main settings. No production-code change or new dependency was justified. Temporary diagnostic patches were reversed; tracked diff is empty and `botq pushed HEAD` succeeds. There is no new commit to land and no touched-path test suite owed for this artifact-only verdict. The matching E12 release demo build passed, exit 0, in 22m 27s; its CLI check and all four render/encoding and ffprobe checks passed. Earlier interrupted render attempts are superseded by these complete clips, not counted as successes.

Remaining unchecked boxes, in order: resolve the flat deterministic-chase gap after rl#20's hull-body landing; re-enable the full distance band as a single-feature canary; run the existing terrain-amplitude curriculum; record the scripted pace instrument at the retrain pin; satisfy the final Ask with baseline comparison, slope pursuit and on-device proof. The new sensors remain enabled. No canary is launched here on the body about to be replaced. The epic remains open.

Evidence is retained in job 5026's durable artifact directory: `final-buckets.txt`, `run-integrity.json`, `final-run-states.txt`, four MP4s and their `*-tiles.png` contact sheets, `render-validation.json`, `render-*-final.log`, `build-e12-final.log`, `review.md`, the experiment commands/patches/ruler and `experiment-file-hashes.json`. This verdict supersedes the interim read and render updates.

<a id="c76"></a>
### 76. 2026-10-01 23:31Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5942681198)

Matched flat diagnostic instrumentation landed in **a9372f7**. `render-video` can now use the trainer's flat ground and ≤9 m target band, and compare action means with the trainer's OU/Gaussian exploration. The target RNG is separate from exploration, and the trace covers only recorded ticks. No physics, architecture, or sensor change was made. The former flat-only local render patch left the target band unmatched; this fixes the comparison instrument, not yet the learned pursuit.

A **cold hull-body baseline is running**, pinned to that commit:

```sh
rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3   --workers 6 --envs 2 --horizon 512 --seed 351   --log-std-floor-end -1 --ticks 40000000 --checkpoint-dir "$CKPT"
```

The finite budget is **40M ticks**, with reads at 10M and 40M using episode-weighted 5M-tick reach buckets. The end-of-budget comparison is paired mean/stochastic 20 s, 640×360 renders at seeds 351–353, using the same complete checkpoint and plant. At 40M the exploration-floor schedule is at -1. A mean-policy closing-and-holding render remains the gate for widening; stochastic reach alone cannot clear it. An integrity failure censors the run and is reported rather than silently restarted.

Plant identity: hull body `9e779a4ff65774dd`, sensor layout `f58bd65704a5e084`, simulation `4489626817134930`; new sensors retained. Trainer SHA256 `ab1ae1530ea6ff389dd5735728bb3b4849a62fd766e3040807346c813cca9c0f`; renderer `7d763513b99b7457f5245d6c221ab401ab3258f2d5ddfa0f6dc98be80e192c9c`.

Instrumentation smoke: cold training exited 0 at 12,288 ticks, with no hard integrity failures and 41 CCD clamp candidates. Both 2 s renders loaded the same stamped checkpoint, matched simulation identity and target, exited 0, and verified as 128 frames. Mean drive 0.22364 versus stochastic 0.44799; both renders reported zero integrity faults and zero clamp candidates. These are wiring checks, **not a learning verdict**.

All covering checks passed: fmt, clippy, both isolated release builds, full workspace tests, web check, simulation tests (4/4), and meshfit with the verified model (16/16). Read-only review finished at 9.5/10 after fixes to trace timing, diagnostic-only fault reporting, and policy-step integration coverage. Protocol: [flat-chase-diagnostic.md](https://github.com/bddap/rl/blob/a9372f7/docs/flat-chase-diagnostic.md).

The flat-baseline box stays unchecked while this run matures. The separate startup-visibility box above Ask records the smoke clip's initially missing Sally; no render defect has been established as the cause of stalled pursuit. Full band, terrain, and on-device acceptance remain later increments.

<a id="c77"></a>
### 77. 2026-10-02 07:38Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5947533650)

First 10M-tick read of the cold hull-body flat ≤9 m baseline at `a9372f7`:

| Training interval | Episode-weighted reach | Episodes | Mean reward(20) | Mean drift |
|---|---:|---:|---:|---:|
| 0–5M ticks | 0.034 | 6,662 | −14.9 | 0.91 m |
| 5–10M ticks | 0.039 | 6,666 | −4.1 | 0.99 m |

The unchanged Endgame ruler includes 3,255 logged iterations through 9,999,360 ticks. Its first/last 2,000-iteration windows overlap at this read, so the disjoint buckets above are the useful comparison. Reach is computed from the logged, rounded per-iteration values.

At the integrity read, training had reached 13,682,688 ticks with one cold start, one GPU initialization, zero restarts, strictly increasing logged ticks, and no logged integrity failure or rescue. CCD clamp candidates are separate from integrity failures: 10,242 through 10M, 14,573 through the current read, maximum 9 in an episode. The live checkpoint's four members share a save stamp and retain body `9e779a4ff65774dd`, sensor layout `f58bd65704a5e084`, and simulation `4489626817134930`.

The finite 40M run continues unchanged. These stochastic training metrics do not demonstrate deterministic pursuit. The flat-baseline box remains unchecked until the final checkpoint and paired mean/stochastic 20 s renders at seeds 351–353 are assessed; full-distance and terrain expansion remain gated.

<a id="c78"></a>
### 78. 2026-10-02 20:43Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5961152278)

Matched hull flat-baseline verdict at **a9372f7**: training improved but was **censored by an integrity failure at 34,240,512 of 40M ticks**. It was not restarted. The paired checkpoint render comparison is recorded below; this stage does not clear the widening gate.

| Tick interval | Episode-weighted reach | Episodes |
|---|---:|---:|
| 0–5M | 0.034 | 6,662 |
| 5–10M | 0.039 | 6,666 |
| 10–15M | 0.043 | 6,680 |
| 15–20M | 0.057 | 6,720 |
| 20–25M | 0.085 | 6,823 |
| 25–30M | 0.082 | 6,796 |
| 30–34.240512M, partial | 0.078 | 5,745 |

Same episode-weighted ruler over rounded logged reach; there is no 35–40M result. One cold start, one GPU initialization, zero restarts, and strictly increasing ticks across 11,146 iterations. There were 83,496 CCD clamp candidates (maximum 32 per episode), counted separately from the single hard integrity failure. This is one stochastic training run and cannot establish deterministic chase or a hull treatment effect against the earlier Endgame control.

The terminal event was a right-claw blow-up at worker tick 5,706,897 / episode step 633: wrist linear speed 236.1 m/s and angular speed 507.9 rad/s. The flight recorder localizes the event but does not establish whether contact, joint solving or drive caused it. The failure notification was honored: no silent restart, no continuing the interrupted budget.

The last complete checkpoint has matching envelope stamps for brain, observation normalizer, optimizer and return normalizer. Body `9e779a4ff65774dd`, sensor layout `f58bd65704a5e084`, simulation `4489626817134930`; all new sensors retained. Trainer and renderer remain pinned to the landed source and published binary hashes. Floor annealing had reached −1 after 5M ticks.

All six final paired renders exited 0 and verified as **20.000 s, 1,280 frames, 640×360 at 64 fps**. Each clip has 20 captured-interval trace samples, no sampled target change, and no logged rescue, panic or integrity fault. Paired modes share the same initial target. Contact sheets and traces were inspected.

| Mode / seed | Body distance at 1 s → 20 s | Minimum sampled tip distance | CCD candidates | Observed behavior |
|---|---:|---:|---:|---|
| mean-351 | 1.738 → 0.953 m | 1.085 m | 1 | tumbles, ends inverted away from target |
| stochastic-351 | 2.041 → 1.904 m | 1.603 m | 8 | tumbles, ends inverted away from target |
| mean-352 | 6.798 → 6.359 m | 5.760 m | 1 | overturns early, nearly stationary far away |
| stochastic-352 | 7.049 → 6.511 m | 5.889 m | 2 | overturns early, nearly stationary far away |
| mean-353 | 6.324 → 5.965 m | 5.246 m | 0 | stays upright, little progress; target distance from trace |
| stochastic-353 | 6.317 → 6.073 m | 5.348 m | 0 | stays upright, little progress; target distance from trace |

**None demonstrates closing and holding.** Adding exploration does not restore pursuit in these three paired clips; this is not a success-rate estimate or proof that action selection never matters. Seed 353 remains upright but scarcely advances. No target reseed appears in the sampled trace.

Execution qualification: two software-render batches were externally terminated with exit 143; incomplete seed-352 captures are excluded. All six final clips were rerun consistently on hardware NVIDIA Vulkan using the same pinned renderer, checkpoint and rollout settings. Both completed software seed-351 clips are retained; their sampled traces and drive statistics exactly match the hardware repeats. The checkpoint hashes remained unchanged. The initially invisible Sally also appears in these clips, with visibility later; the existing startup-visibility follow-up remains separate.


The renderer uses the same flat ground, ≤9 m target mixture and policy head as the trainer. Its default mode takes action means; the paired mode uses stationary OU/Gaussian exploration at floor −1 with an independent RNG. Continuous renders are not replays of training episode resets, evolving normalizers or the failed worker. Compare distance only within unchanged-target segments. The 60 pre-capture startup ticks and separate startup-visibility limitation still apply.

This increment follows its explicit **append-the-next-bounded-diagnostic** branch. The landed change fixes the previously unmatched render target band and provides policy-mode comparison; it does not claim to repair learned chase or the physical failure. Added a finite flat short-band integrity reproduction box **before full-band expansion**, preserving the current plant and sensors and requiring a demonstrated cause before a fix. Full band and terrain remain gated on a successful mean-policy closing-and-holding render.

All eight covering suites passed on the landed commit: fmt, clippy, trainer release build, renderer/game release build, full workspace tests, web check, simulation 4/4 and meshfit 16/16. Read-only implementation review finished at 9.5/10; final evidence review also finished at 9.5/10 with no blockers to this bounded closure. No further production change was made during training or this read.

Evidence is preserved in job 5820’s durable artifacts: `final-buckets.txt`, `final-integrity.json`, `failure-flight-recorder.txt`, `final-checkpoint-identity.json`, six `mean-{351,352,353}.mp4` / `stochastic-{351,352,353}.mp4` clips with contact sheets and logs, `render-validation.json`, `render-results.json`, `final-render-checks.txt`, `review.md` and `validation.md`. The source and protocol are landed in [a9372f7](https://github.com/bddap/rl/commit/a9372f7b8ac067d6c618f79d45e04ca047dc5268).

<a id="c79"></a>
### 79. 2026-10-02 21:08Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5961509043)

**Pre-registration — flat short-band integrity reproduction** (registered before the hunt starts; numbers below are fixed)

Checkpoint: the last coherent set from the a9372f7 hull baseline, 34,240,512 ticks (`brain.bin` sha256 `bc44ff79…7a18`, `normalizer.bin` `058ec69d…aaf7`). It is the exact brain + normalizer snapshot that was rolling when the violation fired: the learner persists the set at iteration start, then rolls it.

Harness: new `rl-train repro` (source commit `aae90fc`, tree `bd1cea2b`, on a9372f7; binary sha256 `649976762e97…6000`). It reuses the trainer's own rollout world (`build_rollout_app`), horizon loop (`roll_one_horizon`), stochastic sampling, and the unchanged rl#343 check and thresholds. There are no updates, and the normalizer stays frozen at the checkpoint. Simulation identity is `4489626817134930` and the plant record matches.

Fixed config: `--terrain flat --band-max-m 9 --envs 2 --horizon 512 --log-std-floor-end -1` (floor −1.000 at 34.24M), arch mlp512x3 from the checkpoint tag, `--seed 3510`, **12 workers × 1,048,576 ticks = 12,582,912 ticks** (~37% of the original run's tick count). Each worker stops at its first violation; nothing restarts.

Endpoints:
1. **Primary:** ≥1 rl#343 violation within the budget. A hit is replayed by re-running that worker's seeded stream alone (`--only-worker`). Determinism must hold: the replay has to trip at the same tick with the same part. That replay saves a per-tick plant snapshot of the failing env from 64 ticks before the trip, and `sally-replay` finds the first tick where a kick appears, the contacts on the kicked link, and which single solver/limit/shape lever removes it. A fix lands only if one lever demonstrably removes the kick, and it comes with a regression on the captured state plus a matched finite rerun.
2. **Secondary:** near misses, meaning a part ≥30 m/s in bound units (`lin.max(ang/3)`). These are logged per worker/tick and are replayable the same way, but they are labelled as near misses and are not the failure. Calibration saw 0 in 98,304 ticks.

If 0 violations: "not reproduced at 12.58M frozen ticks". Any near misses get analysed, and the next bounded diagnostic is appended as a box. Integrity thresholds are not relaxed.

<a id="c80"></a>
### 80. 2026-10-03 06:04Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5966200102)

**Flat short-band integrity reproduction — verdict: REPRODUCED; no single demonstrated cause qualifies for a fix → next bounded diagnostic appended.**

Pre-registration: the comment above. Harness `rl-train repro` landed in `aae90fc`..`e49bd48`; evidence landed in [`ede952b`](https://github.com/bddap/rl/tree/ede952b/docs/evidence/chase-archaeology/integrity-repro).

**Primary endpoint met.** The frozen 34,240,512-tick checkpoint (`brain.bin` `bc44ff79…7a18`, `normalizer.bin` `058ec69d…aaf7`, hashes re-verified; simulation `4489626817134930`) was run with the registered config: flat, band ≤9 m, 2 envs × horizon 512, floor −1, seed 3510, 12 workers. That produced **4 rl#343 trips in ≥6,259,163 worker-ticks** (≈0.64 per M ticks; the training run had 1 in 34.24M):

| worker | tick | episode step | fastest part | lin m/s | ang rad/s |
|---|---:|---:|---|---:|---:|
| 0 | 323,648 | 238 | LegCarpus(Right, 0) — whole body | 195.5 | 340.9 |
| 9 | 334,611 | 548 | **ClawWrist(Right)** — right claw only | 110.7 | 324.9 |
| 6 | 414,530 | 452 | LegMerus(Right, 3) | 152.1 | 1933.0 |
| 8 | 525,126 | 269 | LegMerus(Right, 2) | 1137.9 | 37027.0 |

Worker 9 is the original failure's signature: the right claw wrist alone, from a calm state (0.42 m/s the tick before), as in the a9372f7 run (236.1 m/s, 507.9 rad/s). The other three are the same kick on other links.

**Deviations from the registration, stated:**
1. I stopped the hunt at 49.7% of the 12,582,912-tick budget. By then the binary primary endpoint was already decided. Host contention (load ~65, run reniced to 19) put the remainder at ~10 h, and the next A/B measures its own matched baseline rate anyway. Per-worker ticks are in `per-worker-ticks.txt`.
2. I did not use the seeded `--only-worker` replay. A worker's stream does not reproduce across processes: three same-seed runs gave different action digests. The harness therefore captures the failing env's plant snapshot in-run, at the tick before any part passes 30 m/s. Every capture replays its kick exactly (`sally-replay` self-check dev 0.000), which is the stronger form of the determinism the registration asked for.

**First divergent event, isolated.** Seven kick-onset captures (the four trips here, the 427.9 m/s right-wrist trip from the first hunt, and two ≥65 m/s near misses) each went through `game sally-replay`, with one lever changed per row ([`sally-replay.txt`](https://github.com/bddap/rl/blob/ede952b/docs/evidence/chase-archaeology/integrity-repro/sally-replay.txt)):

| lever (one vs shipped) | kicks surviving / 7 |
|---|---:|
| shipped (self-check) | 7 |
| drives zeroed / ×0.5 | 3 / 3 — **not drive-caused**: worker 0's trip still reaches 91.7 m/s with zero drive |
| PGS internal iterations 12→24 / 48 | 4 / 4 — and **diverges**: 7e4 → 7e13 m/s |
| capsule radius ×0.5 / ×1.5 | 7 / 7, bit-identical (the 4 capsule links are not involved) |
| any single claw joint's limits off | 6–7 (shoulder-left 2) |
| limit spring 40 Hz / = contact class | 3 / 5 |
| solver (4,12,3)×2 | 1 (residual 6.2 m/s, below the near-miss line) |
| 4 substeps (2,12,3)×4 | 2 (residual 6.2 / 10.5 m/s) |
| solver (8,4,4)×4 · (32,8,8)×8 | 0 · 0 |
| 28 cuboid links → capsules · all links → balls | 0 · 0 |

In the 192 flight-recorder ticks before the four trips, **21–36 of 38 joints sit at or past a limit** on every tick. The kicked link has a resting terrain contact but no impulse spike. The onset is one tick in which the solve of the whole limit-saturated multibody blows up. Adding PGS iterations amplifies it instead of converging, which points to an inconsistent or ill-conditioned limit + contact system, not under-iteration.

**Why no fix lands here.** None of these levers is a cause demonstrated in isolation under this box's constraints:
- Every removing lever perturbs the whole solve. The shape swaps also change density-derived mass and inertia on 28–38 links and replace the hull plant this box must keep.
- (8,4,4)×4 is rl#340 stage 2, convicted as breaker 1 for cold learning, at ~3× the CPU per tick.
- A one-tick replay shows that a lever removes *this* kick. It does not show a lower trip rate.

So, per the box's otherwise-branch, the next bounded diagnostic is appended as a matched frozen-harness rate A/B (new box below). Integrity thresholds are untouched, nothing restarted, and no production code changed. The mean-policy closing-and-holding render gate still holds before any widening.

<a id="c81"></a>
### 81. 2026-10-03 08:36Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5967242336)

**Pre-registration — multibody-solve integrity rate A/B** (registered before any arm runs; the numbers below are fixed)

**(a) Attribution, done first** (one-tick replays of the 7 committed onset captures; full table lands with the evidence). The new rows:

| lever (one vs shipped) | kicks surviving / 7 | max part speed after |
|---|---:|---|
| all joint limits off | 5/7 by the 4× detector, but every one is ≤4.5 m/s | 4.0–4.5 m/s (shipped: 65–4004 m/s) |
| terrain contact off (solver ignores ground contacts) | 0/7 | 0.3–1.5 m/s |
| all contacts off | 0/7 | identical to terrain-off, so self-contact plays no part |

Removing either the joint limits or the ground contact removes the blow-up. The kick is a property of the limit and ground-contact system solved together, not of drives (3/7 survive zeroed drives) or self-contact. The new `massΔ` column reads 0 on every row, shape swaps included: `sally-replay` already pins each collider's mass properties, so the earlier remark that the shape swaps also change mass and inertia was wrong. They move geometry only. Neither removal is a lever that keeps the plant, so (a) adds no arm.

**(b) Rate A/B.** Harness: `rl-train repro --solver OUTER,PGS,STAB[xSUBSTEPS]`, which applies the counts after the shipped-solver warm-up. One change from the reproduction box: a trip no longer ends a worker. The worker builds a fresh world on a new seed and keeps going, so every arm spends its whole budget and the trip rate is trips over ticks actually rolled. Each worker reports its thread CPU time.

Fixed per arm: the same frozen 34,240,512-tick checkpoint (`brain.bin` `bc44ff79…7a18`, `normalizer.bin` `058ec69d…aaf7`), `--terrain flat --band-max-m 9 --envs 2 --horizon 512 --log-std-floor-end -1 --seed 3510 --workers 12 --ticks 786432` = **9,437,184 ticks**. rl#343 thresholds are unchanged.

Arms:
1. shipped (2,12,3)×2
2. (4,12,3)×2
3. (2,12,3)×4

Host: one short-lived burst VM runs all three arms at once, so the per-arm ticks per CPU-second are directly comparable.

Decision rule: a lever is kept only if it trips **0** times while the shipped arm trips **≥5** (the baseline expects ~6). If the shipped arm trips fewer than 5 times, the A/B cannot convict and the next bounded diagnostic is appended. A kept lever lands with a regression that replays the captured states, then a matched cold finite flat ≤9 m rerun and a mean-policy closing-and-holding render. Its CPU price is recorded against stage 2's (8,4,4)×4, at ~3× per tick.

<a id="c82"></a>
### 82. 2026-10-03 18:21Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5972121711)

**Multibody-solve integrity rate A/B: verdict NOT CONVICTABLE.** The shipped arm tripped once in 9.44M ticks, below the pre-registered ≥5, so no solver lever is kept. The next bounded diagnostic is appended below.

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-5967242336
Harness: [`114ab54`](https://github.com/bddap/rl/commit/114ab54) (+ `2fc0465`, `edeb7db`).
Evidence: [`d9bcadb`](https://github.com/bddap/rl/tree/d9bcadb/docs/evidence/chase-archaeology/integrity-rate-ab).

**(a) Attribution** (one-tick replays of the 7 committed onset captures, [`sally-replay.txt`](https://github.com/bddap/rl/blob/d9bcadb/docs/evidence/chase-archaeology/integrity-repro/sally-replay.txt)):
- **All joint limits off:** no blow-up, max part speed ≤4.5 m/s on all 7.
- **Terrain contact off:** 0/7 kicks, and the result is identical to all contacts off.
- **Mass and geometry:** the `massΔ` column is 0 on every row. Shape swaps already pin mass properties, so they move geometry only. A mutation that unpins the masses makes `massΔ` read 0.9–670, so the column is live.
- **Capsule-radius rows cut:** they touch only the four antenna links and left every kick bit-identical.

So the onset needs the joint-limit system and the ground contact solved together. Drives and self-contact are not needed. Neither removal is a lever that keeps the plant.

**(b) Rate A/B.** Config: same checkpoint and config, seed 3510, 12 × 786,432 ticks per arm. Each worker ran as its own `--workers 1 --first-worker i` process, on the same seeds as a 12-worker run. Each arm had 4 workers on each of three identical 32-vCPU VMs, all three arms running concurrently. A trip rebuilds the world on a new seed, so every worker spent its full budget.

| arm | ticks | trips | per M | near misses ≥30 m/s | ticks / CPU-s | CPU per tick vs shipped |
|---|---:|---:|---:|---:|---:|---:|
| shipped (2,12,3)×2 | 9,437,574 | **1** | 0.106 | 18 | 62.4 | 1.00× |
| (4,12,3)×2 | 9,437,184 | 0 | 0 | 22 | 45.4 | 1.37× |
| (2,12,3)×4 | 9,437,933 | 2 | 0.212 | 50 | 38.9 | 1.60× |

The trips:
- shipped w3, right claw wrist, 104 m/s
- (2,12,3)×4 w3, right claw wrist, 130 m/s
- (2,12,3)×4 w9, right leg carpus (Right, 2), 41 m/s lin / 341 rad/s

All three captures replay exactly (dev 0.000). On those one-tick replays, every alternative solver and terrain-off removes the kick. Yet the ×4-substep arm still tripped at twice the shipped rate. **A one-tick replay removal does not predict a lower trip rate.** The (4,12,3)×2 arm's 0 against shipped's 1 is not evidence either way. Its near misses are not lower (22 vs 18).

**Discrepancy that now gates any lever A/B.** The reproduction hunt ran one 12-worker process on the training host: 4 trips in 6.26M ticks (0.64/M). Here the shipped arm ran as 12 single-worker processes on different hardware: 1 in 9.44M (0.11/M). P(≤1 | 0.64/M × 9.44M) ≈ 0.017. Three things could explain it:
- in-process worker coupling (shared global state or pools between worlds in one process)
- the host or CPU
- chance

At the observed rate, a convicting A/B would need roughly 50M ticks per arm. The appended box separates these explanations first.

Integrity thresholds are untouched, nothing was silently restarted, and no production physics changed. Read-only review took two rounds: 6/10, then 8/10. The fixed findings were a restart seed collision with the worker seed scheme, capture names that collided across worlds, a sally-replay base solver option, the CPU clock, and docs. The burst VMs are destroyed.

<a id="c83"></a>
### 83. 2026-10-03 18:34Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5972221733)

**Pre-registration — trip-rate process/host split** (registered before either arm runs; the numbers below are fixed)

Host: the training host (16 cores), both arms running at the same time so they share its contention. Binary: `rl-train` built from `d9bcadb` (sha256 `57737db1…0c77`), shipped solver (2,12,3)×2. Checkpoint: the frozen 34,240,512-tick set (`brain.bin` `bc44ff79…7a18`, `normalizer.bin` `058ec69d…aaf7`). Config: `--terrain flat --band-max-m 9 --envs 2 --horizon 512 --log-std-floor-end -1 --seed 3510 --ticks 786432`. rl#343 thresholds unchanged; a trip rebuilds that worker's world on a fresh seed, as in the rate A/B.

Arms, 12 × 786,432 = 9,437,184 ticks each:
- **(i) in-process:** one `--workers 12` process.
- **(ii) split:** 12 `--workers 1 --first-worker i` processes, i = 0..11.

Recorded per arm: trips, near misses (≥30 m/s, `lin.max(ang/3)`), ticks per CPU-second, captures of every trip onset.

Decision rule (Poisson; 0.64/M expects ~6 trips per arm, 0.11/M expects ~1):
1. **Coupling:** (i) ≥4 trips and (i)'s share of all trips is significant under a ticks-weighted conditional binomial at one-sided p < 0.05 (e.g. 5–0, 6–0, 7–1). Next: find the shared in-process state, with a regression.
2. **Host:** both arms ≥3 trips and no significant difference. Next: isolate what differs between this host and the burst VMs.
3. **Chance:** both arms ≤2 trips. Next: re-size the lever A/B from the pooled rate of all three shipped-solver runs on this checkpoint, so that the shipped arm expects ≥10 trips.
4. Anything else is reported as mixed, and the next bounded diagnostic is appended.

<a id="c84"></a>
### 84. 2026-10-04 05:11Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5976833153)

**Trip-rate process/host split: verdict CHANCE.** Neither the host nor the process layout explains the 0.64/M vs 0.11/M gap. Under the pre-registered rule (both arms ≤2 trips), the lever A/B is re-sized from the pooled rate. That box is appended below.

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-5972221733
Evidence: [`a293b63`](https://github.com/bddap/rl/tree/a293b63/docs/evidence/chase-archaeology/integrity-process-split)

Fixed for both arms: `rl-train` from `d9bcadb`, the frozen 34,240,512-tick checkpoint (hashes re-verified), shipped solver (2,12,3)×2, flat, band ≤9 m, 2 envs × horizon 512, floor −1, seed 3510, 12 × 786,432 ticks per arm.

| arm | host | ticks | trips | per M | near-miss events | ticks / CPU-s |
|---|---|---:|---:|---:|---:|---:|
| (ii) 12 × `--workers 1 --first-worker i` | training host (Ryzen 9 3900X) | 9,437,574 | **1** | 0.106 | 14 | 62.1 |
| (ii) same | burst VM (Xeon Platinum 8168) | 9,437,574 | 1 | 0.106 | 14 | 43.7 |
| (i) one `--workers 12` | burst VM (Xeon Platinum 8168) | 9,438,092 | **2** | 0.212 | 25 | 45.7 |

**The host doesn't matter: single-worker runs are bit-reproducible.** All 12 split workers produced identical action digests on both hosts at every 4,096-tick mark (2,303 digests), and identical near-miss lines. The one trip is the rate A/B's shipped-arm trip, bit for bit: worker 3, tick 332,678, ClawWrist(Right) at 104.06165 m/s. Its pre-trip snapshot is byte-identical (sha256 `eec19e24…`) across both hosts here and job 6023's separately built binary on another VM. The (ii) arm therefore *is* the rate A/B's shipped arm, not a new sample.

**The process layout doesn't raise the rate.** In-process tripped 2 against split's 1, far from the coupling rule (≥4 and significant). In-process workers *are* coupled, but only at start-up. A same-seed 2-worker process gives a different worker-0 stream on each run, while single-worker runs repeat exactly. A per-tick plant hash puts the first divergence at the second warm-up update: warm-up drives the plant with a cold brain drawn from burn's process-global RNG, which concurrent world builds consume in racy order. After that, physics and inference are deterministic. In-process worker 0 and solo worker 0 had identical plants and actions from their first episode reset (tick 1,503) through 4,096. Filed as https://github.com/bddap/rl/issues/431. The race makes each in-process world an independent random sample. It explains why the hunt's streams never replayed, but it is no mechanism for more trips. The in-process arm shares no (worker, tick) near miss with the split arm.

**What remains is chance.** The independent shipped-solver samples on this checkpoint:
- hunt (in-process): 4 in ≥6,259,163
- rate A/B shipped (= this split arm): 1 in 9,437,574
- this in-process arm: 2 in 9,438,092

Pooled, that is 7 in 25.13M = **0.28/M**. Against the two current-binary samples (3 in 18.88M, 0.16/M), the hunt's 4 has P(≥4) ≈ 0.02. Against the pooled rate it is ≈ 0.10. Pooling the in-process samples (6 in 15.70M) against split (1 in 9.44M) gives a conditional binomial p ≈ 0.19. One residual difference, untested: the hunt ran an earlier `rl-train repro` build (`aae90fc`..`e49bd48`, a trip ended its worker). The shipped plant path is unchanged since then: `--solver` writes the same counts after warm-up.

**Re-sizing.** At 0.28/M, ≥10 expected shipped trips needs 36.2M ticks per arm = 46 × 786,432. Run as single-worker processes, each arm is bit-reproducible on any host. Workers 0–11 of all three rate-A/B arms are therefore already measured, and only workers 12–45 remain to run. If the true rate is the current-binary 0.16/M, that budget expects ~6 shipped trips. The appended box says so.

**Deviations from the registration:**
1. The in-process arm ran on a burst VM, not the training host. On the training host this job's CPU share fell to ~1 core under other jobs' 8× weight, which put the arms at days. A session restart also killed the first attempt at 3.8 h, with 0 trips in 2.7M split + 1.2M in-process ticks. The VM survives that. The bit-identity above shows host does not change the trajectories. A host in-process run was started when the host went idle, then stopped at 0.49M ticks (0 trips) when contention returned: ~10 h remained. It is reported, not pooled.
2. Both arms ran concurrently, on one c-32 VM (in-process + split) and on the training host (split), as registered.

Compute: one c-32 for ≈5.7 h (≈$6), plus two droplets destroyed within minutes because the cloud-init key route left root's password expired. All deleted by id, and the temporary SSH key removed. Integrity thresholds are untouched, nothing restarted silently, and no production code changed. The landing is evidence only, so it owed no suites (`botq land`: "ran 0 owed command(s)").

<a id="c85"></a>
### 85. 2026-10-04 05:34Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5976973289)

**Pre-registration — re-sized multibody-solve lever A/B** (registered before any arm runs; the numbers below are fixed)

Fixed per arm: the frozen 34,240,512-tick checkpoint (`brain.bin` `bc44ff79…7a18`, `normalizer.bin` `058ec69d…aaf7`), `rl-train repro --terrain flat --band-max-m 9 --envs 2 --horizon 512 --log-std-floor-end -1 --seed 3510 --workers 1 --first-worker i --ticks 786432`, hull plant, sensors, mlp512x3 unchanged. rl#343 thresholds unchanged; a trip rebuilds that worker's world on a fresh seed, as in the rate A/B. Binary: `rl-train` built from `a293b63` (= `d9bcadb` + evidence only), sha256 `1b68a13a…b558`; a local smoke reproduced recorded worker 0's action digests at 4,096 and 8,192 ticks bit for bit.

Arms, 46 × 786,432 = **36,175,872 ticks** each:
1. shipped (2,12,3)×2
2. (4,12,3)×2
3. (2,12,3)×4

Workers 0–11 of each arm are the rate A/B's `d9bcadb` evidence (single-worker processes are bit-reproducible across hosts and builds); this run adds workers 12–45. Spot-check: shipped worker 3 is re-run on the training host past its recorded trip (tick 332,678) and must reproduce the recorded digests and trip.

Host: six c-32 burst VMs, 17 single-worker processes each (≈ one per physical core), every VM carrying 5–6 workers of every arm, so ticks per CPU-second stay comparable across arms.

Recorded per arm: trips, near-miss events (≥30 m/s by `lin.max(ang/3)`; a new event is a (worker, world, env) near miss more than 64 ticks after its previous one), ticks per CPU-second, a capture of every trip onset.

Decision rule:
- A lever is **kept** only if the shipped arm trips **≥8** and the lever's share of that pair's trips is significant under a ticks-weighted one-sided binomial at **p < 0.01** (e.g. 8–0, 11–1).
- A kept lever lands with a regression that replays the captured states, then a matched cold finite flat ≤9 m rerun and a mean-policy closing-and-holding render.
- Otherwise (shipped <8, or no lever significant) the next bounded diagnostic is appended. Expected shipped trips: ~10 at the pooled 0.28/M, ~6 at the current-binary 0.16/M.

<a id="c86"></a>
### 86. 2026-10-04 13:18Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5980377900)

**Re-sized multibody-solve lever A/B: verdict CONVICTED. Both levers pass, and (4,12,3)×2 ships on main in [`73e7b88`](https://github.com/bddap/rl/commit/73e7b88).**

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-5976973289
Evidence: [`integrity-lever-ab-36m/`](https://github.com/bddap/rl/tree/73e7b88/docs/evidence/chase-archaeology/integrity-lever-ab-36m) (`4f0c7ac`)

Every arm used the frozen 34,240,512-tick checkpoint (hashes re-verified), flat ≤9 m, 2 envs × horizon 512, floor −1, seed 3510, and 46 × 786,432 ticks. rl#343 thresholds were unchanged. Workers 0–11 are the rate A/B's. Workers 12–45 ran as single-worker processes, 17 per c-32 on six identical VMs, each VM carrying all three arms.

| arm | trips (46 workers) | per M | trips w12–45 | near-miss events w12–45 | ticks / CPU-s | CPU per tick |
|---|---:|---:|---:|---:|---:|---:|
| shipped (2,12,3)×2 | **13** | 0.359 | 12 | 55 | 55.6 | 1.00× |
| (4,12,3)×2 | **2** | 0.055 | 2 | 44 | 41.8 | 1.33× |
| (2,12,3)×4 | **2** | 0.055 | 0 | 21 | 36.2 | 1.54× |

- **Decision rule.** Shipped tripped 13 times, past the ≥8 threshold. Each lever's split of the trips against shipped is 13–2. With ticks almost exactly equal across arms, the one-sided ticks-weighted binomial gives p = 0.0037 for each lever, under the 0.01 bar. Both levers are kept.
- **Choice.** Main ships (4,12,3)×2, the cheaper of the two. (2,12,3)×4 has fewer near misses but costs 1.54× the CPU per tick. Neither lever brings trips to zero.
- **Spot-check.** A rebuilt `rl-train` re-ran shipped worker 3 on the training host and matched every one of the 81 recorded action digests. It also reproduced the trip at tick 332,678, and its pre-trip snapshot is byte-identical (`eec19e24…`). So reusing workers 0–11 is sound.

**Landed** (`73e7b88`, on `4f0c7ac`): `SOLVER_ITERATIONS` is now (4,12,3).
- Regression `crab-world/tests/integrity_captures.rs` replays 21 of the 22 committed (2,12,3)×2 onsets one tick on the shipped solver. All 21 stay below the 30 m/s near-miss line.
- A mutation pass reverted the constant to (2,12,3), and the test went red on all 21 captures.
- The 22nd onset, `inproc-vm-w0 t720366` (a right-carpus spin), still reaches 43 m/s / 286 rad/s at outer 4. It is excluded and named as a residual.
- `game sally-replay` gains a (2,12,3)×2 row. Captures from before this change self-check with `--solver 2,12,3x2`.
- All 7 covering suites were green at `botq land`, including the full `cargo test`.

**Price on the game side.** `game step-profile`, run headless on the training host with the hull checkpoint, puts the driven step p50 at **6.8 → 10.4 ms**. That is the frame budget rl#396 stage 4 had bought with outer 2. The TV and the decks have not been measured. A box below covers that.

**Matched cold rerun launched.** This is the 5820 baseline recipe at `73e7b88`, cold: seed 351, flat ≤9 m, mlp512x3, 6×2×512, floor −1, `--ticks 40000000`. It runs as `rl-target@canary-hull-flat-outer4-73e7b88`, started 2026-10-04 13:14:06Z.
- The running exe's sha256 is `6db62ecb…16b8`, the same as the pinned `rl-train`. The rl-demo pin is `331ce5b2…8f01`.
- The log shows `num_solver_iterations: 4`, a cold brain, and a Flat 1.5–9 m band. The new simulation identity is `37e61920ac4ad672`.
- It runs at ~306 ticks/s, so 40M ticks lands around 2026-10-06 01:30Z.
- The baseline hard-failed at 34.24M. An integrity failure stops this run too, with no silent restart.

The assess box and an on-device frame-budget box are appended below.

Compute: six c-32 droplets for ≈7 h (≈$42). All six were deleted by id, and the temporary SSH key was removed.

<a id="c87"></a>
### 87. 2026-10-05 01:30Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-5986582860)

**Outer-4 matched cold rerun — 10M early read** (`rl-target@canary-hull-flat-outer4-73e7b88`, `73e7b88`, solver (4,12,3)×2)

Episode-weighted reach per 5M-tick bucket from the `[learner] iter` lines, same ruler on both logs:

| ticks | outer-4 rerun | 5820 baseline (2,12,3)×2 |
|---|---:|---:|
| 0–5M | 0.0230 (6,608 ep) | 0.0340 (6,662 ep) |
| 5–10M | 0.0183 (6,584 ep) | 0.0391 (6,666 ep) |

- **Integrity:** 0 rl#343 hard-fails through 10,251,264 ticks.
- **Restarts:** two warm resumes, both from training-host reboots (18:58Z, unclean; 20:43Z, NVIDIA driver 580→595). Each restored brain, normalizers and Adam state from the per-iteration checkpoint, with simulation identity `37e61920ac4ad672` unchanged. No integrity restart.
- **Reading:** the rerun is behind the baseline at both marks, and its 5–10M bucket is lower than its first. The baseline itself only separated from 20M (0.085), so this is not yet a red. Throughput is ~160–275 ticks/s under host load, against the planned 306. The 40M end is now ≈ 2026-10-06 evening UTC rather than 01:30Z.

The final read at 40M covers reach to 40M against the baseline's 0.078–0.085 plateau, the hard-fail count, and mean-policy 20 s renders on seeds 351–353.

<a id="c88"></a>
### 88. 2026-10-09 04:54Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6074530946)

**Outer-4 matched cold rerun: integrity KEPT, chase NOT demonstrated.** The render half fails, so the next bounded diagnostic is appended below.

Evidence: [`outer4-cold-rerun-40m/`](https://github.com/bddap/rl/tree/f9b26e0/docs/evidence/chase-archaeology/outer4-cold-rerun-40m) (`f9b26e0`)

The run is the 5820 baseline recipe at `73e7b88` with solver (4,12,3)×2: seed 351, flat ≤9 m, mlp512x3, 6×2×512, floor −1, cold, `--ticks 40000000`. Simulation identity `37e61920ac4ad672` held at every launch.

**Integrity: passed.**
- The run logged `Tick budget reached (40000512 ticks)` and exited 0.
- It had **0** rl#343 hard-fails. The (2,12,3)×2 baseline hard-failed at 34,240,512.
- There were 235 process launches: two host reboots, and the rest scheduled pauses for interactive GPU use. Every launch restored brain, normalizers and Adam state from the per-iteration checkpoint. No integrity restart happened, and ticks rose strictly.

**Train-side reach** (episode-weighted, 5M-tick buckets, same ruler on both logs):

![reach](https://github.com/bddap/rl/raw/f9b26e0/docs/evidence/chase-archaeology/outer4-cold-rerun-40m/outer4-vs-baseline-reach.png)

| ticks | outer-4 | baseline (2,12,3)×2 |
|---|---:|---:|
| 0–5M | 0.023 | 0.034 |
| 10–15M | 0.026 | 0.043 |
| 20–25M | 0.043 | 0.085 |
| 25–30M | 0.063 | 0.082 |
| 30–35M | 0.076 | 0.078 (to 34.24M) |
| 35–40M | **0.101** | — |

Outer-4 trails the baseline until 30M, then keeps climbing past the baseline's plateau. At 40M it is still rising, with its steepest slope in the last bucket.

**Mean-policy 20 s renders: no chase.** These use the pinned `rl-demo` (sha256 `331ce5b2…8f01`) on the final checkpoint, flat, band 9, 640×360, NVIDIA Vulkan. All six clips exited 0, with one target per clip.

| seed | distance to ball, start → end | upright | behaviour |
|---|---|---|---|
| 351 | 1.34 → 0.81 m | −0.95 from 2 s | flips onto her back next to the ball and stays |
| 352 | 8.38 → 7.25 m | −0.96 from 4 s | flips and stalls |
| 353 | 5.63 → 4.91 m | 0.82 | upright, creeps 0.04 m/s, no closing |

![mean-351](https://github.com/bddap/rl/raw/f9b26e0/docs/evidence/chase-archaeology/outer4-cold-rerun-40m/mean-351-tiles.png)
![mean-353](https://github.com/bddap/rl/raw/f9b26e0/docs/evidence/chase-archaeology/outer4-cold-rerun-40m/mean-353-tiles.png)

The stochastic clip on seed 351 gets within 0.12–0.25 m from 12 s on, but inverted. Stochastic training reach is not proof of deterministic chase.

**Verdict.** Per the box's rule this is "otherwise": the integrity half held and the render half failed. (4,12,3)×2 stays on main; it removes the hard-fail and costs no learning by 40M. The failing half is mean-policy chase. Train reach is still rising at the budget's end, so the next bounded diagnostic is whether budget alone closes the gap: a warm continuation of this same run to 80M, with the same renders at 60M and 80M. It is appended above the frame-budget box.

<a id="c89"></a>
### 89. 2026-10-09 05:06Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6074669047)

**Outer-4 budget continuation — launched** (warm, same run, `--ticks 80000000`)

The only change from the 40M run is the tick budget: `--ticks 40000000` → `--ticks 80000000` in the same instance's launch script. Everything else stays: the same `rl-train` (sha256 `6db62ecb…16b8`, `73e7b88`), solver (4,12,3)×2, seed 351, flat ≤9 m, mlp512x3, 6×2×512, floor −1, and the same checkpoint dir and log.

- **Resume, 2026-10-09 05:04Z:** simulation identity `37e61920ac4ad672` matches checkpoint and build. Brain, normalizer, return normalizer and Adam state are restored. The σ schedule is anneal epoch 0 over 5M ticks, so it is already at the −1 floor, and `--ticks` does not move it. The first iteration counts on from 40,000,512.
- **40M checkpoint kept** as `ckpt-40000512/` beside the live one (brain sha256 `31052611…c539`).
- **60M:** a watcher copies the checkpoint set once ticks first reach 60,000,000 (`ckpt-<ticks>/`). It keeps the copy only if `ticks.txt` is unchanged across it, so the renders get a consistent set.
- **Throughput:** ~535 ticks/s steady. Without TV pauses, 60M comes ≈ 15:30Z today and 80M ≈ 2026-10-10 02:00Z. Pauses push both later.
- **Integrity:** an rl#343 hard-fail stops the run with no restart, as before.

At 60M and 80M: the 5M reach buckets with the same ruler, the hard-fail count, and mean-policy 20 s renders on seeds 351–353 with the pinned `rl-demo` (`331ce5b2…8f01`).

<a id="c90"></a>
### 90. 2026-10-09 12:35Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6081002141)

**Outer-4 budget continuation: stopped by an rl#343 hard-fail at 48,172,032 ticks. No chase.** Reach peaked at 35–40M and fell after it. Two boxes are appended below.

Evidence: [`outer4-continuation-48m/`](https://github.com/bddap/rl/tree/99d7735/docs/evidence/chase-archaeology/outer4-continuation-48m) (`99d7735`)

This is the 40M outer-4 run warm-continued in place. The only change is `--ticks 40000000` → `--ticks 80000000`: same `73e7b88` binary, solver (4,12,3)×2, seed 351, flat ≤9 m, mlp512x3, 6×2×512, floor −1. It had two launches, both warm, and simulation identity `37e61920ac4ad672` held at each.

**Integrity: failed.**
- One hard-fail, at 48,172,032 ticks: rollout thread 5, env 0, episode step 22.
- The fault was LegCarpus(Right, 3) at **76 m/s, 346 rad/s**, up from 3.4 m/s / 15 rad/s one tick earlier. That is a one-tick right-carpus spin, the same shape as `inproc-vm-w0 t720366`, the residual named when (4,12,3)×2 landed.
- The run exited with no restart. 60M and 80M were never reached.
- Totals for (4,12,3)×2 on this run: 0 trips in the first 40.0M ticks, 1 in the next 8.17M.

**Train-side reach** (same 5M ruler):

![reach](https://github.com/bddap/rl/raw/99d7735/docs/evidence/chase-archaeology/outer4-continuation-48m/outer4-continuation-reach.png)

| ticks | reach | episodes |
|---|---:|---:|
| 35–40M | **0.101** | 6,550 |
| 40–45M | 0.095 | 6,771 |
| 45–48.17M | 0.066 | 4,266 |

At 2M resolution, reach is 0.127 at 38–40M, then 0.106 → 0.103 → 0.049 → 0.079. Mean reward(20) falls from +9.4 to −4.9 over the same span. Reach plateaued and then regressed, so budget alone does not close the gap on this recipe.

**Mean-policy 20 s renders: no chase.** These use the pinned `rl-demo` (`331ce5b2…8f01`) on the final checkpoint (brain `6ab9fb93…f1cffc`). It stands in for the unreached 60M/80M marks. The same recipe on the 40M checkpoint reproduces the committed 40M traces byte for byte.

| seed | distance to ball, start → end | upright | behaviour |
|---|---|---|---|
| 351 | 2.20 → 1.11 m at 12 s → 1.35 m | 0.66–0.87 | stays upright (at 40M she flipped), closes halfway, then drifts off |
| 352 | 7.78 → 6.52 m | −0.78 → −0.93 | on her back throughout, rolls 1.3 m, stalls |
| 353 | 5.91 → 4.62 m at 13 s → 4.66 m | 0.75–0.99 | upright, creeps, then stops |

![mean-351](https://github.com/bddap/rl/raw/99d7735/docs/evidence/chase-archaeology/outer4-continuation-48m/mean-351-tiles.png)
![mean-352](https://github.com/bddap/rl/raw/99d7735/docs/evidence/chase-archaeology/outer4-continuation-48m/mean-352-tiles.png)

**Verdict.** Per the box's rule this is "otherwise": no mean render closes and holds. Reach plateaued at 35–40M and then fell. Inversion persists on 352 but not on 351. A hard-fail ended the run, so integrity on (4,12,3)×2 is not closed.

Two bounded boxes are appended, integrity first, because any longer run on this plant can hit the same spin:
1. **Right-carpus residual.** Capture onsets bit-reproducibly from the frozen 48.17M checkpoint and fix the one-tick spin at its cause. A regression replays every onset, `t720366` included.
2. **Mean-policy inversion and post-40M regression.** A 20-seed mean-policy sweep on the frozen 40M and 48.17M checkpoints measures how often she ends inverted, how often she rights herself, and how often she closes and holds. It also reads whether reward and observations give an inverted body any gradient toward righting. The evidence picks the next change: inversion recovery, or architecture/optimizer.

Both checkpoints are kept frozen beside the run.

<a id="c91"></a>
### 91. 2026-10-09 14:19Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6082780452)

**Pre-registration — right-carpus residual: onset capture and trip rate before/after** (registered before any arm runs; the numbers below are fixed)

**Diagnosis so far** (one-tick replays of the 24 committed blow-up onsets, outer 1–8 each, 208 replays): every blow-up is a Gauss-Seidel sweep that diverges *inside* one substep's contact solve, and every one needs friction (collider friction ×0: 0/208 blow up; as shipped: 32/208). Two defects in the multibody (generic) contact friction drive it:
1. The two friction rows of a multibody contact were solved together with only their diagonal effective masses — a Jacobi step inside the sweep. Through the joints a link's tangent-plane mobility is strongly anisotropic, so the ignored coupling `K₀₁ = J_t0·M⁻¹·J_t1ᵀ` is large and the step can diverge. Rigid-body contacts already solve the exact 2×2; multibody contacts did not.
2. The friction rows' anchor bias `(p1 − p2)·t / dt` is unclamped, unlike the normal bias. A sliding point's frozen anchors drift by the whole step's slip, and undoing that within one substep asks friction for (substeps × slip speed). On `t720366` the right-leg contact's friction targets were 68–132 m/s. With foot–ground μ = 2.0 and `μ·|K_nt| / K_nn ≈ 2.5` on that light link, the saturated friction drives the link into the ground, the normal impulse answers, the friction cap rises, and both impulses roughly double per sweep. That is the one-tick carpus spin.

Lever ablation (208 replays): exact 2×2 only → 3 loud (all `t720366`); bias clamp only → 26; both → **0**. Both land in the rapier fork as one commit (`5ec3305`), clamping the anchor bias to `max_corrective_velocity` in all three contact kinds. Solver counts are unchanged: (4,12,3)×2.

**Fixed per arm:** the frozen 48,172,032-tick checkpoint (`brain.bin` `6ab9fb93…f1cffc`, `normalizer.bin` `60033299…21e7`), `rl-train repro --terrain flat --band-max-m 9 --envs 2 --horizon 512 --log-std-floor-end -1 --seed 4810 --workers 1 --first-worker i --ticks 786432 --capture-dir cap`, solver (4,12,3)×2, hull plant, sensors, mlp512x3 unchanged. rl#343 thresholds unchanged; a trip rebuilds that worker's world on a fresh seed.

Arms, 34 × 786,432 = **26,738,688 ticks** each:
1. **before** — the run's own `rl-train` (sha256 `6db62ecb…16b8`, `73e7b88`), the binary that hard-failed at 48.17M
2. **after** — `rl-train` built with the fix (rapier `5ec3305`; sha256 `afc9ad7f…4492`)

Budget sizing: (4,12,3)×2 measured 2 trips in 36.2M (0.055/M) on the 34.24M checkpoint and 1 in 8.17M on this run. Near-miss events ran 44 in 26.7M (1.65/M). At 26.7M per arm the before arm expects ~1.5–3 trips and ~44 near-miss events.

Host: four c-32 burst VMs, 17 single-worker processes each, each VM carrying both arms' workers, so ticks per CPU-second stay comparable.

Recorded per arm: trips, near-miss events (≥30 m/s by `lin.max(ang/3)`; a new event is a (worker, world, env) near miss more than 64 ticks after its previous one), ticks per CPU-second, a capture of every trip onset.

**Decision rule:**
- The fix is **kept** if the after arm has **0 trips** and its near-miss events are fewer than the before arm's under a ticks-weighted one-sided binomial at **p < 0.01**.
- Every trip onset captured in the before arm joins the regression (`integrity_captures`, which also takes `t720366` and the two (4,12,3)×2 onsets `outer4-w26`/`outer4-w34`), and must replay below the 30 m/s line on the fixed plant. A mutation check pins the old rapier rev and must turn it red.
- Otherwise the onsets the after arm captures are the next diagnostic, appended as a box.

<a id="c92"></a>
### 92. 2026-10-09 22:09Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6090077345)

**Right-carpus residual: cause found and fixed in the multibody friction solve. Kept by the pre-registered rule: 0 trips and 122 → 0 near-miss events.**

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-6082780452
Evidence: [`integrity-carpus-48m/`](https://github.com/bddap/rl/tree/8f368ae/docs/evidence/chase-archaeology/integrity-carpus-48m) (`2fe6516`). Fix: rapier fork `5ec3305`, pinned in [`0975bcd`](https://github.com/bddap/rl/commit/0975bcd).

**What drives the one-tick spin.** Stage-by-stage traces of the replayed onsets show the blow-up happens *inside* one substep's contact solve: the generalized velocity on two or three dofs roughly doubles every few Gauss-Seidel sweeps. Every committed onset needs friction (collider friction ×0: 0 of 208 replays blow up; as shipped: 32). Two defects in the multibody friction rows:
1. A multibody contact solved its two friction rows together using only their diagonal effective masses: a Jacobi step inside the Gauss-Seidel sweep. Through the joints, a light link's tangent-plane mobility is strongly anisotropic, so the ignored `K₀₁` is large and the step can diverge. Rigid-body contacts already solved the exact 2×2.
2. The friction anchor bias `(p1 − p2)·t / dt_substep` was unclamped, unlike the normal bias. A sliding point's anchors drift with its slip, and the bias asks friction to undo all of it within one substep. On `t720366` the right-leg contact's friction target was 68–132 m/s. With foot–ground μ = 2.0 and `μ·|K_nt| / K_nn ≈ 2.5` on that link, saturated friction drives the link into the ground, the normal impulse answers, the friction cap rises, and both impulses about double each sweep. That is the right-carpus spin.

The fix solves the exact coupled 2×2 and clamps the anchor bias to `max_corrective_velocity` in all three contact kinds. Solver counts are unchanged at (4,12,3)×2, and the hull plant, sensors, architecture, task and rl#343 thresholds are untouched. Lever ablation over 208 replays: 2×2 alone → 3 loud (all `t720366`), clamp alone → 26, both → 0.

**Trip rate, frozen 48,172,032-tick checkpoint**, 34 × 786,432 = 26,738,688 ticks per arm, both arms on every VM:

| arm | trips | near-miss events | peak lin.max(ang/3) | ticks / CPU-s |
|---|---:|---:|---:|---:|
| before (`6db62ecb…`, the binary that hard-failed) | 0 | **122** | 93.4 m/s | 38.99 |
| after (rapier `5ec3305`) | 0 | **0** | — | 38.23 |

- Neither arm tripped. At the measured (4,12,3)×2 rate, 26.7M ticks expected ~1.5 trips before, so trips alone cannot separate the arms. The near-miss events can: 122 vs 0, one-sided ticks-weighted binomial p = 1.9e-37. The before arm's commonest near-miss part is LegCarpus(Right, 3) (47 of 135 lines), the same part as the 48.17M hard-fail.
- Price: about 2% more CPU per tick.

![onset replays](https://github.com/bddap/rl/raw/8f368ae/docs/evidence/chase-archaeology/integrity-carpus-48m/onset-replays.png)

**Regression.** `integrity_captures` now replays every onset committed under `integrity-*/`, 145 in all. That covers the 119 before-arm captures that reproduce their near miss on the old plant, plus `t720366` and the two (4,12,3)×2 trips. All 145 stay under 30 m/s on the fixed plant: `t720366` goes 46.8 → 7.1 m/s, and the 123 new captures replay ≤ 7.9 m/s. Four before-arm captures do not reproduce on the old plant and are listed in `replays.txt` but not committed.
- **Mutation check:** pinning the old rapier rev turns the test red on 122 onsets; restoring it turns it green.
- `ReplayOutcome`'s maxima now keep NaN, so a non-finite replay can no longer read as quiet.

**Other effects.**
- One existing test moved: the render-build flail link check (`slot_pumped_crab_motion_reaches_the_snapshot_clients_decode`). On the fixed plant she shuffles 0.5–0.8 m around spawn throughout the window and ends 0.33 m out. On the old plant, the 2.7 m it measured was a single 2 m lurch in 25 ticks. The test pins the link, so it now reads her largest excursion.
- Every covering suite is green at the landing (`8f368ae`), the full `cargo test` included. The fork's own suite is green too.

**Review.** Round 1 found no correctness defects. Design review asked whether rebasing the anchor offset beats clamping it. I kept the clamp: the anchor exists to stop resting creep, where drift is small and the clamp never binds, and the rest/sleep gates depend on that. It also flagged that a rank-deficient tangent `K` is unguarded; the crab's free root keeps `K` full rank, and the rigid path has the same behaviour, so it is recorded but not changed. Round 2 found the NaN masking, now fixed.

Compute: four 32-vCPU droplets for ≈6.4 h (≈$25), each deleted by id, and the temporary SSH key removed.

One box is appended above the frame-budget box: with the cause fixed, retest whether outer 2 holds integrity, since outer 4 is what cost the frame budget.

<a id="c93"></a>
### 93. 2026-10-09 22:34Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6090384266)

**Pre-registration — mean-policy inversion sweep, frozen 40M and 48.17M outer-4 checkpoints** (registered before any sweep render; the definitions and rule below are fixed)

**Fixed:** the pinned `rl-demo` (sha256 `331ce5b2…8f01`, `73e7b88`), mean policy, `render-video --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed S`, NVIDIA Vulkan, the run's own plant asset root. Checkpoints: `ckpt-40000512` (brain `31052611…c539`) and `ckpt-48172032` (brain `6ab9fb93…f1cffc`). Seeds 351–370 on each: 40 renders. Recipe check before the sweep: seed 351 on 48.17M reproduces the committed `outer4-continuation-48m/mean-351-trace.txt` byte for byte.

**Read:** `ROLLOUT_TRACE`, one sample per simulated second (t = 1…20 s): planar body-to-ball distance `distance_m`, claw-tip distance, `upright` = carapace up-axis · world Y.

**Definitions:**
- *inverted* — `upright < 0`.
- **inverted at 20 s** — the t = 20 s sample is inverted.
- **rights herself** — an inverted sample followed by a later sample with `upright ≥ 0.5`. Reported over all 20 seeds and over the seeds that ever invert.
- **closes and holds** — `distance_m ≤ 0.5` at every sample from some t ≤ 18 s through t = 20 s (≥ 3 consecutive samples, to the end). Secondary: ever within 0.5 m; the same rule on claw-tip distance.

**The reward/observation reading** (from source at `73e7b88`, not from the renders): what an inverted body earns per term, whether any term rises as `upright` rises from −1 toward +1, whether the observation carries her orientation, whether episodes terminate or reset on inversion, and whether spawns or shoves ever put her on her back in training.

**Decision rule for the appended box.** Pool the 40 renders. Each render that does not close and hold is a failure, classed *inverted* (inverted at 20 s) or *upright* (not).
- **Inversion-recovery change** if inverted failures ≥ upright failures on either checkpoint, or if they rose from 40M to 48M by ≥ 3 seeds. Its form follows the reading: a missing righting gradient gets a reward/termination change; a gradient that exists but is not learned gets a training-distribution change (inverted spawns).
- **Architecture/optimizer change** otherwise: most failures are upright bodies that do not close. Its form follows the 40M→48M comparison and the run's reward(20) collapse after 40M (an optimizer/stability change if the 48M policy is worse than 40M on close-and-hold or distance closed, an architecture change if they are equal).

<a id="c94"></a>
### 94. 2026-10-09 22:51Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6090609779)

**Mean-policy inversion sweep: inversion grew from 3 to 7 of 20 seeds between 40M and 48.17M, and no deeply inverted body ever recovered. The reward never reads orientation. By the pre-registered rule the next change is inversion recovery: upright shaping, as an A/B.**

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-6090384266
Evidence: [`inversion-sweep/`](https://github.com/bddap/rl/tree/510d48f/docs/evidence/chase-archaeology/inversion-sweep) (`510d48f`)

The run used the pinned `rl-demo` (`331ce5b2…8f01`), mean policy, flat ≤9 m, 20 s, seeds 351–370 on each frozen checkpoint. That is 40 renders, and all exited 0. Seeds 351–353 reproduce the committed 40M and 48M traces byte for byte.

![sweep](https://github.com/bddap/rl/raw/510d48f/docs/evidence/chase-archaeology/inversion-sweep/inversion-sweep.png)

| | 40,000,512 | 48,172,032 |
|---|---:|---:|
| inverted at 20 s | 3/20 | **7/20** |
| ever inverted | 3/20 | 10/20 |
| rights herself (inverted, then upright ≥ 0.5) | 0/3 | 3/10 |
| closes ≤ 0.5 m and holds to 20 s (body) | **2/20** | 0/20 |
| ever within 0.5 m | 7/20 | 1/20 |
| holds on claw-tip distance | 0/20 | 0/20 |
| failures: inverted / upright | 3 / 15 | 7 / 13 |
| median distance closed | 0.76 m | 0.25 m |

- **Righting.** All three recoveries at 48M started from shallow tilts, with minimum upright −0.02, −0.17 and −0.40. None of the 7 bodies that went below −0.5 came back.
- **When she flips.** Inversion is early: 5 of the 10 at 48M, and all 3 at 40M, are already inverted at the 2 s sample. That is about 2 s of control after an upright spawn.
- **Closing and holding.** The two 40M holds sit on the ball with the body, not the claws. Seed 366 stays 0.04–0.26 m from 10 s with its tips ≥ 0.45 m away. 48M is worse than 40M on every closing measure.

![40M seed 366](https://github.com/bddap/rl/raw/510d48f/docs/evidence/chase-archaeology/inversion-sweep/mean-40m-366-tiles.png)
![48M seed 363](https://github.com/bddap/rl/raw/510d48f/docs/evidence/chase-archaeology/inversion-sweep/mean-48m-363-tiles.png)

**What the reward and observations give an inverted body** (source at `73e7b88`; nothing relevant changed through `8f368ae`):
- **Reward.** It is three terms: `24 × planar distance closed`, minus `0.0006 × Σ drive²`, plus 50 on a claw-tip grab (terminal). No term reads orientation, so an inverted carapace that slides toward the ball earns exactly what an upright one does.
- **Episodes.** There is no termination or penalty on inversion; the old fall terminal was deleted as a dying exploit. Episodes truncate at 1500 ticks with a bootstrap. γ 0.99 and λ 0.95 give a horizon of about 1.6 s.
- **Observations.** Orientation enters only as the world-frame quaternion, where upright = 1 − 2(x² + z²). There is no body-frame gravity channel.
- **Exposure.** Training already visits inverted states. About 12% of respawns start with upright < 0 (the heavy-tilt branch of `random_spawn_rotation`), plus her own flips. There are no shoves.
- **Gradient toward righting.** There is no direct one. The only pull is indirect: righting pays only through progress made afterwards, discounted over about 1.6 s. Inverted bodies here stall, so righting does have value, but nothing steers the gradient toward it, and 48M ticks with about 12% inverted starts did not find it.

**Decision** (pre-registered): inverted failures rose from 3 to 7, an increase of 4, which meets the ≥ 3 threshold, so the next change is **inversion recovery**. Because the reading shows a missing gradient, the change goes into the reward. Upright bodies that fail to close are still the majority on both checkpoints (15 and 13), so the box below measures both classes.

**Optimizer** (for the box after this one; per-4M table in `optim-4m.txt`):
- Per-update KL is 0.077–0.091 throughout.
- The update stops at ≤ 2 epochs on 62–92% of iterations. After 40M this rises to 86–92%, while mean reward(20) goes +8.5 → +3.4 → −3.0.
- Entropy has been pinned at the log-std floor since 4M.

If upright shaping fixes inversion but the upright bodies still don't close, this is where to look next.

One box is appended below the outer-2 integrity box: a matched cold A/B with potential-based upright shaping. It sits there so both arms run on the solver that box ships.

<a id="c95"></a>
### 95. 2026-10-09 23:13Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6090849435)

**Pre-registration — outer-2 integrity on the fixed plant** (registered before any frozen-checkpoint arm runs; the numbers below are fixed)

The friction fix (rapier `5ec3305`) removed the cause of the one-tick spins. Outer 4 was the symptom-level lever and costs the frame budget (driven step p50 6.8 → 10.4 ms). This tests whether outer 2 now holds integrity.

**Stage 1, already run (cheap gate):** the 145-onset regression (`integrity_captures`, every onset under `integrity-*/`) replayed one tick at **(2,12,3)×2** on the fixed plant (main `510d48f`): **145 of 145 stay under 30 m/s, max 7.93 m/s.** Had any onset been loud, outer 2 would have failed here.

**Stage 2, frozen-checkpoint arms.** Binary: `rl-train` built from main `510d48f` (rapier `5ec3305`), sha256 `2e1924c3…577f`; the arm differs only by `--solver`. Every run is `rl-train repro --terrain flat --band-max-m 9 --envs 2 --horizon 512 --log-std-floor-end -1 --workers 1 --first-worker i --ticks 786432 --capture-dir cap`, the run's own plant asset root, hull plant, sensors, mlp512x3 unchanged, rl#343 thresholds unchanged (a trip rebuilds that worker's world on a fresh seed).

| checkpoint | seed | workers | arms | ticks per arm |
|---|---|---|---|---:|
| 34,240,512 (`bc44ff79…`, the lever A/B's) | 3510 | 0–45 | (2,12,3)×2 and (4,12,3)×2 | 36,175,872 |
| 48,172,032 (`6ab9fb93…`, the carpus run's) | 4810 | 0–33 | (2,12,3)×2 | 26,738,688 |

The 48M checkpoint's (4,12,3)×2 control on the fixed plant is the carpus box's *after* arm (0 trips, 0 near-miss events in 26,738,688 ticks): same checkpoint, seed, workers and rapier rev; the commits since change only replay NaN handling and a test, not the step. It is reused, not re-run.

Pre-fix reference, same 34M checkpoint and seed: (2,12,3)×2 tripped **13** times in 36.2M (0.36/M); at that rate the outer-2 arms expect ~22 trips over 62.9M ticks.

Host: eight c-32 burst VMs, 15–16 single-worker processes each, every VM carrying workers of every arm, so ticks per CPU-second stay comparable.

Recorded per arm: trips, near-miss events (≥30 m/s by `lin.max(ang/3)`; a new event is a (worker, world, env) near miss more than 64 ticks after its previous one), ticks per CPU-second, a capture of every near-miss and trip onset.

**Decision rule:**
- Outer 2 **holds integrity** if both (2,12,3)×2 arms trip **0** times, and on each checkpoint its near-miss events are not more than the (4,12,3)×2 control's under a ticks-weighted one-sided binomial at **p < 0.01**.
- If it holds: `SOLVER_ITERATIONS` returns to (2,12,3), `integrity_captures` keeps replaying all 145 onsets (plus any new ones) on the shipped solver, a mutation check confirms the test still turns red on the old rapier rev, and `game step-profile` re-measures the driven step, interleaved against (4,12,3)×2.
- Otherwise outer 4 stays. Every outer-2 trip onset is committed with its replay at each solver, naming which onsets still blow up, and the next bounded diagnostic is appended.

<a id="c96"></a>
### 96. 2026-10-10 05:52Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6094384747)

**Outer-2 integrity with the friction fix: HOLDS. Main returns to (2,12,3)×2.** By the pre-registered rule, both outer-2 arms tripped 0 times with 0 near misses. The driven step p50 is back to ~6.5 ms from ~10 ms.

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-6090849435
Evidence: `docs/evidence/chase-archaeology/integrity-outer2-fixed/` (landed with the change in [`ea60e14`](https://github.com/bddap/rl/commit/ea60e14))

**Stage 1, onset regression at (2,12,3)×2.** All 145 committed onsets replay one tick on the fixed plant under 30 m/s, max 7.93 m/s (`replays-outer2.txt`).

**Stage 2, frozen-checkpoint arms.** `rl-train` from `510d48f` (rapier `5ec3305`, sha256 `2e1924c3…577f`). Single-worker processes, 786,432 ticks each, flat ≤9 m, 2 envs × horizon 512, floor −1, the run's own plant. Eight c-32 VMs, every VM carrying every arm.

| checkpoint, seed, workers | solver | ticks | trips | near-miss events | ticks / CPU-s |
|---|---|---:|---:|---:|---:|
| 34,240,512, 3510, 0–45 | (2,12,3)×2 | 36,175,872 | **0** | **0** | 56.7 |
| 34,240,512, 3510, 0–45 | (4,12,3)×2 | 36,175,872 | 0 | 0 | 43.2 |
| 48,172,032, 4810, 0–33 | (2,12,3)×2 | 26,738,688 | **0** | **0** | 56.5 |

- **Decision rule.** Both outer-2 arms tripped 0 times. Their near-miss events (0) cannot be significantly above the outer-4 arm's (0 here; 0 in the carpus after arm on 48M). Outer 2 holds.
- **What 0 bounds.** 0 trips in 62.9M pooled outer-2 ticks bounds the rate at ≤ 0.048/M (95%). That is below the 0.055/M outer 4 was shipped at. Before the fix, outer 2 on the same 34M checkpoint and seed tripped 13 times in 36.2M (an older binary, not interleaved). At that rate these ticks would expect ~23 trips. Those pre-fix workers 12–45 also had 55 near-miss events; these have none.
- **Cause tied by replay.** Pinning rapier `8f2af05` at outer 2 turns `integrity_captures` red on 73 of the 145 onsets (exit 101). With `5ec3305` restored it is green.
- **Price.** Outer 2 costs 0.76× the CPU per tick of outer 4 (56.7 vs 43.2 ticks/CPU-s).
- **Limit.** These were frozen checkpoints on flat ground. In live training, the rl#343 hard-fail stays the detector.

**Step profile.** `game step-profile --ticks 1024` on the 34M hull checkpoint, headless, on the training host. Runs were interleaved, three each, on a loaded host (load ~8):

| run | (4,12,3)×2 p50 | (2,12,3)×2 p50 |
|---|---:|---:|
| 1 | 10.28 ms | 6.88 ms |
| 2 | 10.54 ms | 6.20 ms |
| 3 | 8.94 ms | 6.25 ms |

Physics share at p50: ~8.6 ms at outer 4, ~5.2 ms at outer 2. Outer 2 is back at the pre-`73e7b88` 6.8 ms baseline.

**Landed** ([`ea60e14`](https://github.com/bddap/rl/commit/ea60e14)): `SOLVER_ITERATIONS` is (2,12,3) again, and its doc comment carries the new reasons. `integrity_captures` now replays all 145 onsets at outer 2, because it follows the shipped solver. `game sally-replay`'s comparison row goes back to (4,12,3)×2. Captures taken while outer 4 shipped self-check with `--solver 4,12,3x2`.

**Effect on the boxes below.**
- The upright-shaping A/B runs on outer 2, the solver this box ships.
- The on-device frame-budget box was written against outer 4's 10.4 ms. Main now runs the 6.8 ms baseline it compared against, so whoever takes that box measures the current pin.

Compute: eight c-32 droplets for ≈5.4 h (≈$43). All eight were deleted by id, and the temporary SSH key was removed.

<a id="c97"></a>
### 97. 2026-10-10 18:51Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6100998864)

**Pre-registration — upright-shaping A/B (inversion recovery)** (registered before either arm launches; the numbers below are fixed)

**The change** ([`c19a393`](https://github.com/bddap/rl/commit/c19a393)): potential-based shaping Φ(s) = k·upright (upright = carapace up-axis · world Y), paid on every transition as γΦ(s′) − Φ(s), γ = 0.99, the same constant GAE uses. Selected by `rl-train learn --upright-shaping-k K` (default 0 = the unshaped reward, bit-for-bit).
- **Grab terminal.** The grab is absorbing with Φ = 0, so that step pays γ·0 − Φ(s). Over any episode the discounted shaping then sums to −Φ(s₀) whether or not she grabs, so the grab's value relative to every alternative is unchanged and it is not penalised. Paying the real Φ(s′) there instead would add a γᵀ·k bonus for grabbing upright, which is no longer a potential difference. Both review lenses flagged that variant.
- **Cap truncation.** Pays the real γΦ(s′) − Φ(s); the GAE bootstrap V(s′) carries the shaping beyond the cap.
- Unit tests: the discounted sum telescopes to −Φ(s₀) on paths ending in the grab, and each step end pays the right term. Mutation-checked: dropping γ, or paying the real Φ(s′) at the grab, turns them red.

**k = 12, pre-registered against the progress scale.** Φ spans 2k = 24 = `PROGRESS_WEIGHT` × 1 m, so righting from fully inverted to upright is worth closing 1 m.
- That is below the 50 grab and below the near-band approach (24 × (1.5 − 0.2) = 31). Shaping cannot outbid the task, and since it is a potential difference it cannot change the optimum anyway.
- It is large enough to be heard. A 1 s righting (64 ticks) pays about 0.37 per tick, about 3× what a 0.3 m/s approach pays (0.11 per tick). With λ = 0.95, about 0.83 of k reaches the acting step through the advantage even before the critic learns Φ.
- No sweep over k. The A/B tests this one value.

**Arms** (one binary, built from main at `c19a393`; they differ only by `--upright-shaping-k`, stated explicitly on both):

| instance | k |
|---|---:|
| `canary-upright-ab-ctrl-c19a393` | 0 |
| `canary-upright-ab-k12-c19a393` | 12 |

Both arms: `rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --seed 351 --log-std-floor-end -1 --ticks 40000000 --read-mark 40000000`, cold, with the shipped solver (2,12,3)×2 and the hull plant (`sally.glb` from the pinned asset release). They run side by side on the training host under the same runner, so GPU yields to the TV pause both together. Each run's `train.log` records k in its `[learner] reward economy` line.

**Reads, on both arms:**
1. **Reach.** Episode-weighted reach per 5M-tick bucket, with the same `reach5m.awk` ruler as `outer4-cold-rerun-40m/`. Shaping changes reward(20) by construction, so reward is not compared across arms.
2. **Integrity.** The rl#343 hard-fail count. Any failure stops that arm, with no silent restart, and its onset is posted.
3. **Inversion sweep at 40M.** Run on `marks/40000000/` with an `rl-demo` built at `c19a393`: mean policy, `render-video --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed S`, seeds 351–370, NVIDIA Vulkan, the run's plant. Scored with `inversion-sweep/analyze.py`'s definitions unchanged: inverted = upright < 0 at the t = 20 s sample; close-and-hold = body-to-ball ≤ 0.5 m at every sample from some t ≤ 18 s through 20 s; righted = inverted, then a later upright ≥ 0.5.

**Decision rule.** Keep the shaping, and make k = 12 the default, only if all of these hold:
- the k = 12 arm is inverted at 20 s on **≤ 1/20** seeds **and fewer than control**;
- its close-and-hold count is **≥ control's**;
- its reach is **not below control's**. "Below" means its 35–40M bucket is lower by a one-sided two-proportion z-test at p < 0.05, about 0.009 at ~6,500 episodes per arm.

Otherwise the shaping is reverted on main (the flag and the term are deleted), and the reason is recorded here.

- **If kept,** and upright bodies that do not close are still the majority of the k = 12 arm's sweep failures, the architecture/optimizer box is appended. It works from the sweep's optimizer reading: per-update KL 0.08–0.09, ≤ 2 epochs on 62–92% of iterations, entropy pinned at the floor.
- **If an arm hard-fails before 40M,** the A/B is censored and no keep decision is made. The onset is posted, and the next integrity diagnostic is appended.

<a id="c98"></a>
### 98. 2026-10-10 18:55Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6101035108)

**Upright-shaping A/B: both arms launched 2026-10-10 18:51Z** ([pre-registration](https://github.com/bddap/rl/issues/351#issuecomment-6100998864))

- **Pin:** main [`c19a393`](https://github.com/bddap/rl/commit/c19a393). Both arms run one `rl-train`, sha256 `054b4a8f…2a16`, read back from each run's own launch line. The `rl-demo` for the 40M sweep is built from the same tree, sha256 `35e3e69c…aab0`.
- **Plant:** `sally.glb` sha256 `119d5a4f…bf0b`, verified against the baked pin. This is the same hull plant the outer-4 runs used. `simulation=ac3ad4a12cc60fdc` on both arms.
- **Common command:** `learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --seed 351 --log-std-floor-end -1 --ticks 40000000 --read-mark 40000000`. Both arms started cold, with cold optimizer moments.

| instance | flag | `[learner] reward economy` line |
|---|---|---|
| `canary-upright-ab-ctrl-c19a393` | `--upright-shaping-k 0` | `upright shaping k 0` |
| `canary-upright-ab-k12-c19a393` | `--upright-shaping-k 12` | `upright shaping k 12` |

- **Smoke:** both arms run side by side at about 720–740 steady sps each, which is about 335 ticks/s on a loaded host. At that rate 40M lands about 2026-10-12 04:00Z, later if TV pauses intervene. Both arms pause together.
- **Reward(20) is not comparable across arms.** On the k = 12 arm the undiscounted episode sum carries the −(1−γ)·k·upright term every tick, which telescopes only under discounting. Reach is the pre-registered ruler.

<a id="c99"></a>
### 99. 2026-10-12 13:57Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6125760321)

**Upright-shaping A/B: REVERTED by the pre-registered rule.** Shaping removed inversion from the mean policy (0/20 inverted at 20 s, against 8/20 for control), but its training reach in the 35–40M bucket fell below control's (0.059 vs 0.111, z = 10.9). The rule required reach not below control, so the shaping is reverted on main ([`b6b6bf0`](https://github.com/bddap/rl/commit/b6b6bf0)).

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-6100998864
Evidence: [`upright-shaping-ab/`](https://github.com/bddap/rl/tree/2643498/docs/evidence/chase-archaeology/upright-shaping-ab) (`2643498`)

![A/B](https://github.com/bddap/rl/raw/2643498/docs/evidence/chase-archaeology/upright-shaping-ab/upright-shaping-ab.png)

**Runs.** Both arms used one `rl-train` from `c19a393`, cold, with seed 351, flat ≤ 9 m, mlp512x3, 6×2×512 and floor −1, run side by side. Each completed "Tick budget reached (40000512 ticks)" with exit 0. **Neither arm had an rl#343 hard-fail.**

**Reach** (`reach5m.awk`, episode-weighted):

| bucket | 0–5M | 5–10M | 10–15M | 15–20M | 20–25M | 25–30M | 30–35M | **35–40M** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0.026 | 0.054 | 0.077 | 0.069 | 0.075 | 0.082 | 0.089 | **0.111** |
| k = 12 | 0.023 | 0.078 | 0.041 | 0.041 | 0.144 | 0.104 | 0.121 | **0.059** |

The k = 12 arm swings much harder than control. In 2M buckets it peaks at 0.210 at 22–24M, reaches 0.141 at 32–34M, then drops to 0.067 and 0.040 over 36–40M. Pooled over 20–40M, k = 12 is ahead, 0.104 against 0.089.

**Mean-policy sweep at 40M** (seeds 351–370, `rl-demo` from the pin, `inversion-sweep/analyze.py` definitions unchanged; all 40 renders exit 0 with 20 samples each):

| | control | k = 12 |
|---|---:|---:|
| inverted at 20 s | 8/20 | **0/20** |
| ever inverted | 10/20 | 0/20 |
| rights herself | 2/10 | — |
| closes ≤ 0.5 m and holds (body) | 0/20 | 1/20 (seed 358, from 16 s) |
| ever within 0.5 m | 2/20 | 2/20 |
| holds on claw-tip distance | 1/20 | 0/20 |
| failures: inverted / upright | 8 / 12 | 0 / 19 |
| median distance closed | 0.06 m | 1.35 m |

**Decision (pre-registered).**
- Inverted at 20 s on ≤ 1/20 seeds and below control: 0/20 vs 8/20. **Met.**
- Close-and-hold ≥ control: 1 vs 0. **Met.**
- Reach not below control: 35–40M at 0.059 vs 0.111. **Failed.**

The shaping is therefore reverted: the flag and the term are deleted. The architecture/optimizer box was conditional on keeping the shaping, so it is not appended.

**What this leaves.** Shaping does what it was built for: no inversions at all, and every seed walks toward the ball. One seed per arm cannot tell whether its late reach collapse comes from the shaping or is seed noise. Control's own 2M buckets wobble by ±0.03, and k = 12's earlier buckets were above control's.

A seed-replicate box is appended below. It re-lands the shaping and runs the same pair on seeds 352 and 353. The keep rule moves to pooled 30–40M reach across the three seeds.

<a id="c100"></a>
### 100. 2026-10-12 14:05Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6125902831)

**Pre-registration — upright-shaping seed replicate** (registered before any arm launches; the numbers below are fixed)

The seed-351 A/B ([verdict](https://github.com/bddap/rl/issues/351#issuecomment-6125760321)) removed inversion from the 40M mean policy (8/20 → 0/20) but k = 12's 35–40M reach fell below control's (0.059 vs 0.111). This replicate runs the same pair on seeds 352 and 353 to separate that collapse from seed noise.

**Code.** The shaping is re-landed on main by reverting `b6b6bf0` (`--upright-shaping-k`, default 0).

**Binary.** All four runs use the `rl-train` built at `c19a393` that ran the seed-351 pair, sha256 `054b4a8f…2a16`. The sweep uses its `rl-demo`, sha256 `35e3e69c…aab0`. Main has since gained the rl#431 warm-up change (`1345119`..`882ee82`), which changes the rollout worlds' warm-up. With one binary, all six runs pooled below differ only by seed and k. Plant: `sally.glb` `119d5a4f…bf0b`, solver (2,12,3)×2.

**Runs** (cold, side by side on the training host under one runner; they differ only by `--seed` and `--upright-shaping-k`, both stated explicitly):

| instance | seed | k |
|---|---:|---:|
| `canary-upright-rep-ctrl-s352-c19a393` | 352 | 0 |
| `canary-upright-rep-k12-s352-c19a393` | 352 | 12 |
| `canary-upright-rep-ctrl-s353-c19a393` | 353 | 0 |
| `canary-upright-rep-k12-s353-c19a393` | 353 | 12 |

Common: `rl-train learn --terrain flat --band-max-m 9 --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --log-std-floor-end -1 --ticks 40000000 --read-mark 40000000`. The batch shape 6×2×512 is the recipe, so four runs share the CPU rather than each taking fewer workers.

**Reads per run:**
1. **Reach.** Episode-weighted reach per 5M-tick bucket, using `upright-shaping-ab/reach5m.awk` unchanged.
2. **Integrity.** The rl#343 hard-fail count. A failure stops that run with no silent restart, and its onset is posted.
3. **Mean sweep at 40M.** `marks/40000000/`, the mean policy, `render-video --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed S`, seeds 351–370. It is scored with `upright-shaping-ab/analyze.py` unchanged: inverted = upright < 0 at t = 20 s; close-and-hold = body-to-ball ≤ 0.5 m from some t ≤ 18 s through 20 s.

**Decision rule.** Keep the shaping only if both hold:
- **Pooled reach.** Pool the 30–40M episodes over seeds 351–353, with seed 351 taken from the existing `upright-shaping-ab/` arms. Each arm's rate is its reached episodes over its episodes. k = 12 must not be below control: a one-sided two-proportion z-test, control > k = 12, must not reach p < 0.05 (z ≥ 1.645).
- **Each new seed, 352 and 353.** The k = 12 sweep is inverted at 20 s on ≤ 1/20 seeds and fewer than that seed's control, and its close-and-hold count is ≥ control's.

If kept, k = 12 becomes the default. Otherwise the shaping is reverted on main again, and the reason is recorded.

- **Hard fail.** If any run hard-fails before 40M, the replicate is censored. The shaping is reverted with no keep decision, the onset is posted, and the next integrity diagnostic is appended.
- **Either way,** the architecture/optimizer box is appended. It works from the sweep's optimizer reading: KL 0.08–0.09 per update, ≤ 2 epochs on 62–92% of iterations, entropy pinned at the floor. k = 12's oscillating reach points the same way.

<a id="c101"></a>
### 101. 2026-10-12 14:50Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6126660633)

**Upright-shaping seed replicate: four arms launched 2026-10-12 14:05Z** ([pre-registration](https://github.com/bddap/rl/issues/351#issuecomment-6125902831))

- **Re-land:** the shaping is back on main in [`d7e9e61`](https://github.com/bddap/rl/commit/d7e9e61), which reverts `b6b6bf0`. All six landing suites were green.
- **Binary:** every arm runs the `c19a393` `rl-train`, sha256 `054b4a8f…2a16`, the same one as the seed-351 pair. It is read back from each run's own launch line. The sweep `rl-demo` is `35e3e69c…aab0`. Plant: `sally.glb` `119d5a4f…bf0b`, `simulation=ac3ad4a12cc60fdc` on all four.

| instance | `[learner] reward economy` line |
|---|---|
| `canary-upright-rep-ctrl-s352-c19a393` (`--seed 352`) | `upright shaping k 0` |
| `canary-upright-rep-k12-s352-c19a393` (`--seed 352`) | `upright shaping k 12` |
| `canary-upright-rep-ctrl-s353-c19a393` (`--seed 353`) | `upright shaping k 0` |
| `canary-upright-rep-k12-s353-c19a393` (`--seed 353`) | `upright shaping k 12` |

- **Smoke:** with all four side by side, each runs at about 155–165 ticks/s (steady rollout ~295 sps). 40M then lands around 2026-10-15 12:00Z, later if the TV pauses them. All four pause together.

<a id="c102"></a>
### 102. 2026-10-16 19:53Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6223519208)

**Upright-shaping seed replicate: REVERTED by the pre-registered rule.** k = 12 cost no training reach (pooled 30–40M over seeds 351–353: 0.113 vs control 0.103). Its 40M mean policy still ended inverted on 6/20 eval seeds (seed 352) and 2/20 (seed 353), against a bar of ≤ 1/20. On seed 352 it also closed and held 0/20 against control's 5/20. The shaping is reverted on main ([`24b3c49`](https://github.com/bddap/rl/commit/24b3c49)).

Pre-registration: https://github.com/bddap/rl/issues/351#issuecomment-6125902831
Evidence: [`upright-shaping-replicate/`](https://github.com/bddap/rl/tree/92c595a/docs/evidence/chase-archaeology/upright-shaping-replicate) (`92c595a`)

![replicate](https://github.com/bddap/rl/raw/92c595a/docs/evidence/chase-archaeology/upright-shaping-replicate/upright-shaping-replicate.png)

**Runs.** Four cold runs on the `c19a393` binary (`054b4a8f…2a16`) that ran the seed-351 pair: seeds 352 and 353 × k = 0 / 12, flat ≤ 9 m, mlp512x3, 6×2×512, floor −1. All four ended "Tick budget reached (40000512 ticks)" with exit 0. **Hard-fails: 0 on every run.** Each run warm-resumed 125–136 times when the trainer yielded the host; the 351 pair was measured under the same runner.

**Reach** (`reach5m.awk`, episode-weighted):

| bucket | 30–35M | 35–40M | 30–40M |
|---|---:|---:|---:|
| control s352 | 0.086 | 0.082 | 0.084 |
| k = 12 s352 | 0.086 | 0.092 | 0.089 |
| control s353 | 0.126 | 0.120 | 0.123 |
| k = 12 s353 | 0.173 | 0.144 | 0.159 |
| **pooled 351–353** | | | **control 0.103 (41,159 ep), k = 12 0.113 (41,227 ep)** |

One-sided control > k = 12: z = −4.84, so k = 12 is not below control. Seed 351's late collapse did not recur.

**Mean sweep at 40M** (seeds 351–370, `analyze.py` definitions unchanged; all 80 renders exit 0, 20 samples each):

| | ctrl s352 | k12 s352 | ctrl s353 | k12 s353 |
|---|---:|---:|---:|---:|
| inverted at 20 s | 9/20 | **6/20** | 10/20 | **2/20** |
| ever inverted | 10/20 | 7/20 | 10/20 | 2/20 |
| closes ≤ 0.5 m and holds | 5/20 | **0/20** | 1/20 | 1/20 |
| ever within 0.5 m | 5/20 | 0/20 | 5/20 | 8/20 |
| failures inverted / upright | 9 / 6 | 6 / 14 | 10 / 9 | 2 / 17 |
| median distance closed | 0.57 m | 0.53 m | 1.30 m | 1.20 m |

**Decision.**
- Pooled reach not below control: **met.**
- Seed 352: inverted 6/20 > 1/20, **failed**. Close-and-hold 0 < 5, **failed**.
- Seed 353: inverted 2/20 > 1/20, **failed**. Close-and-hold 1 = 1, met.

The shaping is therefore reverted: the flag and the term are deleted, and the code is back to `b6b6bf0`.

**What this shows.** Over three seeds, shaping cut inversion at 20 s from 27/60 to 8/60, but it did not remove it. It did not help the mean policy close: 2/60 close-and-hold against control's 6/60. 50 of the shaped arms' 58 failures end upright, so inversion was not what kept the policy from chasing.

**Next box.** The architecture/optimizer A/B is appended below, built from the optimizer reading: KL 0.077–0.091 per update, ≤ 2 epochs on 62–92% of iterations, entropy pinned at the floor.

<a id="c103"></a>
### 103. 2026-10-16 20:02Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6223666009)

**Pre-registration: entropy-floor A/B.** [Protocol](https://github.com/bddap/rl/blob/c14e253/docs/evidence/chase-archaeology/entropy-floor-ab/README.md).

The single change is `--log-std-floor-end -1` → `-0.7`: hold the existing initial exploration floor instead of annealing it down. The optimizer reading shows entropy pinned at the floor from early training. This tests whether maintaining exploration helps the mean policy learn to close; it also raises the variance used in the KL calculation, without changing the learning rate, KL guard or epoch budget. No source change or default change is needed to run it.

Six cold runs use one main-source binary, seeds 351–353 × the two floors, flat ≤9 m, mlp512x3, 6×2×512, 40M ticks with a registered 40M checkpoint. Matched pairs run concurrently, seeds in ascending order. Earlier shaping controls are not reused. The existing 5M reach ruler and 20-seed mean sweep are unchanged.

Keep only with no significant pooled 30–40M reach harm (one-sided control > treatment z-test, p < 0.05) **and** a significant pooled close-and-hold gain (one-sided treatment > control Fisher exact test, p < 0.05). Report each training seed as well as pooled results; shared policies limit the independence of the 60 evaluation cases per arm. Any integrity hard-fail censors the experiment. Otherwise retain the existing floor and record the result. No interim selection or sweep over endpoints.

<a id="c104"></a>
### 104. 2026-10-16 20:24Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6224023125)

**Entropy-floor A/B: seed-351 pair launched cold.** [Launch evidence and binary hashes](https://github.com/bddap/rl/tree/c655927/docs/evidence/chase-archaeology/entropy-floor-ab).

Both arms use one trainer built from main source `24b3c49`, with the matching evaluator preserved. The only extra commit present while building was the documentation-only pre-registration. Release trainer and evaluator builds both exited 0.

The launch logs confirm cold optimizer moments, flat 1.5–9 m, mlp512x3, 6×2×512, simulation `ac3ad4a12cc60fdc`, and the two intended schedules: control −0.7 → −1, treatment −0.7 → −0.7. Their first update is identical. Both saved checkpoints and advanced beyond 190,000 ticks without a panic at the launch check.

Each arm is bounded at 40M with a registered 40M checkpoint. Seeds 352 and 353 follow as matched pairs under the same protocol. The checklist stays open: no keep/reject verdict is available until all six runs and the 120 mean-policy reads finish.

<a id="c105"></a>
### 105. 2026-10-18 21:07Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6255154924)

**Entropy-floor A/B: training seed 351 completed; seed 352 launched.** [Evidence](https://github.com/bddap/rl/tree/5cf9c59/docs/evidence/chase-archaeology/entropy-floor-ab/s351).

Both seed-351 arms reached 40,000,512 ticks and exited 0, with zero panic lines and one warm resume per arm. All 40 mean-policy renders passed: 20 ordered samples each, using the immutable 40M checkpoint and pinned evaluator. One interrupted control render lacked an exit receipt; the identical seed was rerun successfully and the interrupted attempt retained separately.

| Read | Control −1 | Treatment −0.7 |
|---|---:|---:|
| 30–40M episode-weighted reach | 0.0966 | 0.1727 |
| Close-and-hold | 1/20 | 0/20 |
| Inverted at 20 s | 8/20 | 19/20 |
| Median distance closed | 0.56 m | 0.05 m |

The unchanged scorer uses its historical `k12` filename label for treatment; neither arm has shaping. Published traces are labelled `floor07`.

This is an interim seed result, **not a keep/reject decision**. The seed-352 pair has now started cold with the same trainer and recipe, differing only in the floor endpoint. [Launch telemetry](https://github.com/bddap/rl/blob/5cf9c59/docs/evidence/chase-archaeology/entropy-floor-ab/launch-s352.txt). Seed 353 and the pooled tests remain required; the checklist stays open.

<a id="c106"></a>
### 106. 2026-10-20 15:21Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6294383060)

**Entropy-floor A/B: training seed 352 completed; seed 353 launched.** [Evidence](https://github.com/bddap/rl/tree/83badf8/docs/evidence/chase-archaeology/entropy-floor-ab/s352).

Both seed-352 arms reached 40,000,512 ticks and exited 0, with zero panic lines. Control had three warm resumes and treatment four. All 40 accepted mean-policy renders passed: 20 ordered samples each, using the immutable 40M checkpoint and pinned evaluator. One interrupted treatment render lacked an exit receipt; the identical seed was rerun successfully and the interrupted attempt retained separately.

| Read | Control −1 | Treatment −0.7 |
|---|---:|---:|
| 30–40M episode-weighted reach | 0.0781 | 0.1008 |
| Close-and-hold | 2/20 | 0/20 |
| Inverted at 20 s | 2/20 | 11/20 |
| Median distance closed | 0.31 m | 0.35 m |

The unchanged scorer uses its historical `k12` filename label for treatment; neither arm has shaping. Published traces are labelled `floor07`.

This is an interim seed result, **not a keep/reject decision**. The final seed-353 pair has started cold with the same trainer and recipe, differing only in the floor endpoint. [Launch telemetry](https://github.com/bddap/rl/blob/83badf8/docs/evidence/chase-archaeology/entropy-floor-ab/launch-s353.txt). Its 40M sweeps and the pooled tests remain required; the checklist stays open.

<a id="c107"></a>
### 107. 2026-10-22 09:35Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6337580474)

Completed the architecture/optimizer A/B and rejected the constant −0.7 log-std floor; retain the existing −1 endpoint. Landed evidence: [96d6067](https://github.com/bddap/rl/commit/96d6067b810cdc214e679fc1fbbdb21d4f44a2d4), [pooled verdict](https://github.com/bddap/rl/blob/main/docs/evidence/chase-archaeology/entropy-floor-ab/verdict.md).

All six matched cold runs reached 40,000,512 ticks on the registered pin, with zero integrity panic lines. All 120 accepted mean-policy renders used the registered 40M checkpoint sets and exited successfully; 2,400 trace samples were validated.

| Registered read | Control −1 | Treatment −0.7 | Decision |
|---|---:|---:|---|
| Pooled 30–40M reach | 0.0724 / 40,581 episodes | 0.1260 / 41,795 episodes | No-harm gate passes; H1 control > treatment, z=−25.647, p=1 |
| Close-and-hold | 4/60 | 0/60 | Gain gate fails; H1 treatment > control, Fisher p=1 |
| Inverted at 20 s | 20/60 | 40/60 | Diagnostic |

Per-training-seed hold counts were control 1,2,1 and treatment 0,0,0. Improved sampled training reach did not transfer to the required mean-policy holds. No source/default change was applied, so rejecting the experimental option leaves the existing recipe intact. Warm resumes, excluded render attempts and exact dependency restoration are recorded in the evidence; the binary hashes remained unchanged. The pooled test is the registered descriptive ruler, with only three independently trained policies per arm.

The architecture/optimizer box is checked. The other increments, including device frame budget, full-distance band, terrain curriculum, pace reading, capture startup visibility and the final chasing Ask, remain open. This completes this bounded experiment; it does not establish a new chasing baseline.

<a id="c108"></a>
### 108. 2026-10-24 18:20Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6387852516)

**Outer-4 frame budget on device: TV half measured, 60 fps holds. Deck half needs a person: the box is marked `(needs: …)`.**

Main has shipped (2,12,3)×2 again since `ea60e14`, so this measures the current pin against `3ed15e4`.

**Builds:** main `96d6067` (the deployed game) and `3ed15e4`, built in release and launched through the same launcher, with the same assets and the same shipped brain (`checkpoint-check OK` on both).

**Method:** a solo round, launched through the game's Steam shortcut. The run was 30 s in the menu, then 10 s of warm-up, then 60 s of scripted stop-and-go walking with camera turns. The per-frame ruler is `RL_POS_TRACE`: each `F` row's `Time` delta paired with its `S` row (sim cost, ticks pumped). A step-carrying frame is one that pumped at least one tick. Arms ran in the order main, `3ed15e4`, `3ed15e4`, main, on the games' CPUs (8-11,20-23), under host load average 66–71.

| run | build | fps | frame p10/50/90/99 ms | frames in 13.3–18.4 ms | frames > 20 ms | step-carrying frames | their p50/90/99 ms | in band | sim_ms p50/90/99 |
|---|---|---:|---|---:|---:|---:|---|---:|---|
| 1 | main | 59.94 | 16.27/16.67/17.07/18.38 | 99.0% | 0.11% | 1801 | 16.67/17.08/18.97 | 98.3% | 6.16/10.58/16.04 |
| 2 | `3ed15e4` | 59.79 | 16.27/16.67/17.07/18.80 | 98.7% | 0.28% | 1801 | 16.66/17.05/18.68 | 98.8% | 6.19/10.41/14.91 |
| 3 | `3ed15e4` | 59.77 | 16.30/16.67/17.04/18.53 | 98.9% | 0.22% | 1801 | 16.65/17.02/18.53 | 98.9% | 6.13/10.00/14.64 |
| 4 | main | 59.80 | 16.27/16.67/17.09/18.96 | 98.7% | 0.36% | 1801 | 16.67/17.09/18.82 | 98.8% | 6.17/10.30/15.81 |

**Reading:** on the TV, 60 fps holds at main, and main is indistinguishable from `3ed15e4`. The difference between arms is no larger than the run-to-run spread within an arm. No frame pumped more than one tick. The longest frame in every run was 67 ms.

**Side observation, not diagnosed:** in both main runs the crab-slot log showed the prey distance growing steadily, by about 6 m/s, from 84 to 292 m and from 111 to 400 m over 50 s. With the same brain, `3ed15e4` held it at 6–15 m. Per-frame sim cost was the same in all four runs, so the frame reading stands either way.

**Deck half: needs a person.** Over two days the decks were checked about ten times. They were either asleep, or awake with someone playing on them. The idle gate (a running game, or recent input) refused every awake deck, and a deck that is awake and unused soon goes back to sleep. The one idle window found was on a deck whose full disk kept Steam from launching anything; that disk has since been freed. This measurement needs one deck left awake, untouched and undocked for about 15 minutes. The runner and the `3ed15e4` binary are prepared and can be staged again in a minute.

<a id="c109"></a>
### 109. 2026-10-24 18:26Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6387909445)

**Full target-distance band: waits on a flat chasing baseline, which does not exist yet.** Nothing was trained or landed for this box.

The box widens the band "from the verified flat chasing baseline" and keeps the wider band "only if chase survives". No flat run on the current plant has shown chase. The last measurement, the entropy-floor A/B ([verdict](https://github.com/bddap/rl/blob/main/docs/evidence/chase-archaeology/entropy-floor-ab/verdict.md)), had the mean policy close and hold on 4/60 control and 0/60 treatment sweeps, and it says it "does not establish a new chasing baseline". Earlier boxes also required a closing-and-holding mean-policy render before widening the task. A full-band run now could not answer this box's question.

That A/B appended no next diagnostic, so the missing precondition is now its own box above this one: **Flat mean-policy chasing baseline**. Its bar is one cold 40M flat ≤9 m run on current main whose mean policy closes and holds on ≥ 10/20 eval seeds (351–370, unchanged `analyze.py`), with 0 rl#343 hard-fails. Its first step needs no training. It sweeps the six existing 40M entropy-floor checkpoints with sampled actions and with the mean. That splits a deterministic-policy gap (treatment: sampled reach 0.126, mean holds 0/60) from a policy that has not learned to close. The full-band box carries `(waits: Flat mean-policy chasing baseline)`.

<a id="c110"></a>
### 110. 2026-10-24 18:57Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6388229137)

**Pre-registration: sampled-vs-mean sweep of the entropy-floor checkpoints** (first bounded step of *Flat mean-policy chasing baseline*; no training).

**Inputs, frozen:** the six registered 40M checkpoints of the entropy-floor A/B (control −1 and treatment −0.7 × training seeds 351–353, `ckpt/marks/40000000`, 40,000,512 ticks), the pinned evaluator from that A/B (`rl-demo` sha256 `70746b5e…`, built from `24b3c49`), the pinned plant (`sally.glb` `119d5a4f…`). The pinned evaluator already has a sampled-action mode (`--exploration-log-std-floor`, the trainer's seeded OU exploration), so nothing is restored or rebuilt.

**Sampled arm:** for each checkpoint, eval seeds 351–370, `render-video --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed S --exploration-log-std-floor F`, with F the checkpoint's own training floor at 40M (control −1, treatment −0.7). One draw per eval seed; 120 renders.

**Mean arm:** the 120 published mean-policy traces from the A/B. Two of them (one per training floor) are re-rendered first; they must match the published traces sample for sample, otherwise all 120 are re-rendered.

**Scoring:** `upright-shaping-ab/analyze.py` unchanged, per checkpoint and mode. Primary read: close-and-hold (body ≤ 0.5 m from some t ≤ 18 s through 20 s), pooled over the 120 cases per mode, with per-checkpoint counts. Secondary: ever ≤ 0.5 m, inverted at 20 s, median distance closed.

**Decision, fixed now:** cases are paired by checkpoint × eval seed. Let b = sampled-hold/mean-miss pairs and c = mean-hold/sampled-miss pairs.
- **Deterministic-policy gap** if sampled holds ≥ 2× mean holds and the one-sided exact McNemar test (H1 b > c) gives p < 0.05. Then the next change targets the mean action.
- **Not learned to close** otherwise. Then the next change targets learning (budget, reach criterion or reward), chosen from the reading.

Whichever branch holds, the next bounded change is appended above the baseline box, and the baseline box waits on it. The baseline box is checked only when a cold 40M run meets its bar.

<a id="c111"></a>
### 111. 2026-10-24 19:26Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6388532097)

**Sampled-vs-mean sweep: the policy has not learned to close.** [Evidence](https://github.com/bddap/rl/tree/6a7abcf/docs/evidence/chase-archaeology/sampled-vs-mean-sweep), run under the pre-registration above. No training was run.

![distance at 1 s against distance at 20 s](https://raw.githubusercontent.com/bddap/rl/6a7abcf/docs/evidence/chase-archaeology/sampled-vs-mean-sweep/closing.png)

There were 120 sampled-action renders: six 40M checkpoints × eval seeds 351–370, each at its own training floor. All exited 0, and every trace has 20 samples at 40,000,512 ticks. Two re-rendered mean traces matched the published ones sample for sample, so the A/B's 120 mean traces stand as the mean arm. Scoring used the unchanged `analyze.py`.

| | Mean | Sampled |
|---|---:|---:|
| Close-and-hold, pooled | 4/120 | 4/120 |
| Discordant pairs | 4 (mean only) | 4 (sampled only) |
| Inverted at 20 s | 60/120 | 60/120 |

The one-sided exact McNemar test gives p = 0.64, so the registered gap rule (≥ 2× and p < 0.05) fails. The failure is not in the deterministic mean action.

Spawn distance accounts for the few approaches. Every case that got within 0.5 m, in either mode, started within 2.5 m of the ball except one. From 4.5 m or more, the median gain in 20 s was 0.66 m with the mean action and 1.06 m sampled. What she has not learned is to travel toward a ball a few metres away.

The next bounded change is appended above the baseline box: **Long-distance closing A/B**. It first runs a scripted-pace bound on the plant, then one pre-registered learning change: travel incentive, reach criterion or spawn distance. The baseline box now waits on it, and it stays unchecked until a cold 40M run holds on ≥ 10/20 seeds.

<a id="c112"></a>
### 112. 2026-10-24 21:36Z — [comment](https://github.com/bddap/rl/issues/351#issuecomment-6389834106)

**Pre-registration: long-distance closing A/B** ([protocol](https://github.com/bddap/rl/tree/029f8e1/docs/evidence/chase-archaeology/long-distance-closing-ab), option landed in [`1ab752d`](https://github.com/bddap/rl/commit/1ab752d)). Registered before any training run.

**Plant bound first.** At main `1ab752d`, on flat ground, 20 s from a 300-tick settle:

| Gait | Planar travel in 20 s |
|---|---:|
| `scripted_flail_gait_pace` (every drive ±1, 10-tick period; restored build-locally, it was deleted from main) | 1.51 m |
| best of 96 random open-loop gaits (per-joint sine or square, periods 6–96 ticks, random phase, sign and bias) | 16.55 m |
| median of those 96 | 1.48 m |

Four of the 96 open-loop gaits cover more than 9 m. The plant can cross the band, so no plant box is appended. The registered flail is one fixed gait, and it falls short. The search only gives a lower bound on what the plant can do. It does not check orientation, so a gait that slides while inverted would count. Logs: [pace.txt](https://github.com/bddap/rl/blob/029f8e1/docs/evidence/chase-archaeology/long-distance-closing-ab/pace.txt) and [gait-search.txt](https://github.com/bddap/rl/blob/029f8e1/docs/evidence/chase-archaeology/long-distance-closing-ab/gait-search.txt); the build-local instrument is [pace-instrument.patch](https://github.com/bddap/rl/blob/029f8e1/docs/evidence/chase-archaeology/long-distance-closing-ab/pace-instrument.patch), with the deleted `scripted_flail_gait_pace` extended to 20 s on both grounds and the gait search beside it. On the canonical GCR ground the same flail slides 66 m downhill, so only the flat reading bounds the gait.

**The one change.** Treatment draws the band distance uniformly in metres over [1.5, 9] m (`--band-draw uniform-distance`). Control keeps the log-uniform draw (`--band-draw log-uniform`, the default). The 10% close disc, the reward, the reach criterion and everything else are the same in both arms.

| Training spawn share (close disc included) | < 2.5 m | 2.5–4.5 m | ≥ 4.5 m |
|---|---:|---:|---:|
| control, log-uniform | 35.7% | 29.5% | 34.8% |
| treatment, uniform | 22.0% | 24.0% | 54.0% |

**Why this change.** In the sampled-vs-mean sweep, every approach to within 0.5 m but one started within 2.5 m. From 4.5 m or farther, the median closed in 20 s was 0.66 m with the mean action and 1.06 m sampled, which is no better than a random open-loop gait. Training reach therefore looks like credit from short spawns: a lunge and grab earns +50 there without any locomotion. A uniform draw cuts the share of short spawns and puts most episodes where only travel pays.

The other two options were rejected:
- The travel incentive (24 per metre closed) is already dense and the same at every distance, and it beats the grab bonus on a far approach. Raising it would mostly shrink the effort tax and the grab bonus relative to it, and the effort tax is already a minor term.
- Changing the reach criterion would change the reach ruler that the keep gate reads.

Evaluation stays log-uniform (`rl-demo` render rollouts), so both arms are scored on the same distribution as earlier sweeps.

**Runs.** Six cold runs, one trainer binary built at `1ab752d`, with `rl-demo` from the same pin and plant `sally.glb` `119d5a4f…`:

`rl-train learn --terrain flat --band-max-m 9 --band-draw {log-uniform|uniform-distance} --arch mlp512x3 --workers 6 --envs 2 --horizon 512 --seed {351,352,353} --log-std-floor-end -1 --ticks 40000000 --read-mark 40000000`

All six run concurrently under the same runner and share the CPU. The 6×2×512 rollout is the recipe, so throughput is not traded against it. Warm resumes are recorded, and settings never change on a resume. An rl#343 hard-fail stops that arm with no silent restart and censors the experiment, which blocks a keep.

**Reads, per run.**
- 5M reach buckets using `../upright-shaping-ab/reach5m.awk`, unchanged.
- The rl#343 hard-fail count.
- The 40M mean sweep on `ckpt/marks/40000000`, eval seeds 351–370: `render-video --rollout-terrain flat --band-max-m 9 --seconds 20 --width 640 --height 360 --seed S` on NVIDIA Vulkan, scored by `../upright-shaping-ab/analyze.py` unchanged.
- Closing by spawn distance with `../sampled-vs-mean-sweep/by-spawn.py`.

**Decision, fixed now.** Keep uniform-distance only if both gates pass and no arm hard-failed:
1. Pooled mean-policy close-and-hold over 60 cases per arm is above control's, by a one-sided Fisher exact test at p < 0.05.
2. Pooled episode-weighted 30–40M training reach is not below control's. The test is the replicate's one-sided two-proportion z test, H1 control > treatment, where p < 0.05 counts as harm.

Training reach is stochastic and is not evidence of chase. If the change is kept, it becomes the flat baseline recipe. If it is rejected, `--band-draw` is removed from main and the reason is recorded.

Full SHA-256 digests (release builds of `-p rl-train -p rl-demo` at `1ab752d`, exit 0):

```
ba75f0450a046ee7ed76dd5815baeb22489900e02d11764d9f9f4366e99b9ec8  rl-train
bb342275aff9148f73ccfe13df9c16896085296a29f04465b1cc32b0ce5d3bfc  rl-demo
119d5a4f9986f3739e01677910f9880ddc48a47b0a12a24236c1781426a7bf0b  sally.glb
```

No plant, physics or body source changed between the entropy-floor pin `24b3c49` and `1ab752d`.

