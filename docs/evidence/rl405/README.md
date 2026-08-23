# rl#405 — in-game screenshot binding

Verification renders from the windowed client (`game play`) under Xvfb on lavapipe,
F12 injected via xdotool.

- `in-round.png` — a captured gameplay frame, written to the screenshots dir by the
  new binding (native window res, 1280x720 in this run).
- `confirmation-flash.png` — a second capture 1s after the first: the on-screen
  confirmation line (`saved shot-….png`) from the first shot is visible bottom-center.

Each save also emitted the export log line, e.g.:

    INFO screenshot: screenshot saved path=…/shots/shot-20260822-190211.053.png

Repeat-fire check: three F12 presses → three PNGs + three log lines, same session.
