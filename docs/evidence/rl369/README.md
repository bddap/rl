# rl#369 — free musical play past valid-command length

The old `MAX_CHORD_LEN = 8` capture poison ("code too long — release to
cancel") is gone — deleted with the context menu in 5c056d7 (rl#358), and
rl#380 made code length structurally unbounded (capture, walk clamp, map).
This records the player-level verification that nothing re-caps it.

`free-play-mash.wav`: a 20-tap mash rendered through the live scheme
(`DPAD_EVIDENCE_DIR=… cargo test -p crab-world --features render dpad_evidence
-- --ignored`, the `free-play-mash` clip). Every tap sounds — measured peaks
−8 to −25 dB at taps 1/10/17/20 — and the release is the unknown cadence at
4.0 s, never silence.

`free-play-then-code.mp4`: the real app (`game fp-screenshot`, headless
lavapipe, 854×480@60), scripted input: X held with a 20-tap mash (frames
120–291, release 330 — the combo map dives 20 levels into undiscovered
space, capture never cancels), then a second hold entering `v^^^` (frames
450–585, release 645) — the ground flips to night-bloom on release, so a
valid code still executes after arbitrary free play. Audio is the pure-path
render of both captures at matching frame timings.

Repro: `game fp-screenshot --settle 90 --width 854 --height 480
--anim-frames 660 --anim-every 1 --chord-hold-at 110 --chord-release-at 330
--chord-holds '420:645' --chord-taps
'120:U,129:R,138:U,147:D,156:L,165:D,174:U,183:U,192:R,201:D,210:L,219:L,228:U,237:R,246:D,255:U,264:D,273:L,282:U,291:R,450:D,495:U,540:U,585:U'
--out f.png`
