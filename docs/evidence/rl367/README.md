# rl#367 — jump feel: liftoff 1.23→1.45 m/s + hold-to-float

`before-after.gif`: seed-7 locale, on-foot, camera pitched 25° down, captured
headless at tick rate (30 Hz) and played at 12 fps. Same scripted input on both
sides: a 2-frame JUMP tap at frame 95, then JUMP held over frames 130–175.
BEFORE is the pre-change sim (tap apex ~1.5 player heights, ~0.25 s airtime,
and the held window reads as a stutter of micro-hops); AFTER is the landed
tuning — tap apex ~2.1 heights, and the held window rides the half-gravity
rise to ~4.2 heights with a full-gravity snap fall (~0.5 s airtime per hop).

Repro: `game fp-screenshot --seed 7 --players 1 --settle 90 --anim-frames 120
--anim-every 1 --cam-pitch=-25 --jump-holds 95:97,130:175 --out f.png`
