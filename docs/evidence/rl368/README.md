# rl#368 — slide: entry burst + crouched eye

`before-after.gif`: seed-7 locale, on-foot, camera pitched 25° down, captured
headless at tick rate (30 Hz), played at 12 fps, frames 80–150. Same scripted
input on both sides: walking from frame 1, SPRINT held from frame 40, SLIDE
held over frames 100–160. BEFORE is the shipped slide (rl#355): entry needs
sprint pace and the skid is a pure 1.8×→1.5×-walk decay over ~0.4 s at a fixed
eye height — nothing visibly happens. AFTER is the fix: entry bursts ×5/4
(1.8× → 2.25× walk, then friction, ~0.85 s skid) and the eye eases down to
half stature for the duration — the slide reads as a low, fast skid.

Repro: `game fp-screenshot --seed 7 --players 1 --settle 90 --anim-frames 150
--anim-every 1 --cam-pitch=-25 --walk-at 1 --sprint-holds '40:'
--slide-holds '100:160' --out f.png`
