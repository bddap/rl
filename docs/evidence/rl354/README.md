# rl#354 — on-foot nearby-ground jitter (f32 render precision)

`before-after.gif`: seed-7 locale (10.3 km from the tile origin), on-foot walk,
camera pitched 35° down at the nearby ground, captured headless at tick rate
(30 Hz) and played at 12 fps. BEFORE is pre-fix main (8da8cb1); AFTER is the
render-frame rebase. Template-tracked nearest-ground stride: BEFORE 20–29 px per
frame (std 3.9 px — the judder); AFTER exactly 29 px every frame (std 0.0).

Repro: `game fp-screenshot --seed 7 --players 1 --walk-at 1 --settle 90
--anim-frames 96 --anim-every 1 --cam-pitch=-35 --out f.png`
