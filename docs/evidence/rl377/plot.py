#!/usr/bin/env python3
"""rl#377 parked-craft wiggle: per-tick orientation chatter, before/after the sleep fix.

Reads the RL_POS_TRACE CSVs (Q = craft orientation entering the pose window)
captured by `game fp-screenshot --pilot-toggle-at 60 --pilot-park …` before and
after the parked-craft sleep fix, and renders the evidence figure.
Usage: plot.py <capture-dir> <out-dir>
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"  # before
ORANGE = "#eb6834"  # after


def read_rot(path: Path):
    """Tick -> per-tick rotation angle (deg) from consecutive Q quaternions."""
    quats = []
    for line in path.read_text().splitlines():
        f = line.split(",")
        try:
            if f[0] == "Q" and len(f) == 6:
                quats.append((int(f[1]), tuple(float(v) for v in f[2:6])))
        except ValueError:
            continue  # torn tail line of a killed capture
    rot = {}
    for (t0, a), (t1, b) in zip(quats, quats[1:]):
        d = min(abs(sum(x * y for x, y in zip(a, b))), 1.0)
        rot[t1] = math.degrees(2 * math.acos(d))
        _ = t0
    return rot


def main() -> None:
    cap = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
    before = read_rot(cap / "before.csv")
    after = read_rot(cap / "after.csv")

    # The craft boards at tick ~30 and finishes its touch-down settle well before
    # tick 150; everything after is a parked craft with a neutral command.
    lo = 150
    hi = min(max(before), max(after))

    fig, axes = plt.subplots(
        2, 1, figsize=(9.6, 5.4), sharex=True, facecolor=SURFACE, dpi=160
    )
    for ax, rot, color, label in (
        (axes[0], before, BLUE, "before: never sleeps, chatters indefinitely"),
        (axes[1], after, ORANGE, "after: asleep, bit-exact still"),
    ):
        ticks = [t for t in sorted(rot) if lo <= t <= hi]
        vals = [rot[t] for t in ticks]
        ax.set_facecolor(SURFACE)
        ax.plot(ticks, vals, color=color, lw=0.9)
        moving = sum(1 for v in vals if v > 1e-4)
        ax.set_title(
            f"{label} — max {max(vals):.4f}°/tick, {moving}/{len(vals)} ticks moving",
            color=INK,
            fontsize=10,
            loc="left",
        )
        ax.set_ylabel("rotation, °/tick", color=INK2, fontsize=9)
        ax.set_ylim(-0.002, 0.05)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(INK2)
    axes[1].set_xlabel("sim tick (parked: neutral command, on ground)", color=INK2, fontsize=9)
    fig.suptitle(
        "rl#377 — parked plane orientation chatter, seed 7 boarding spot",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "parked-rotation-before-after.png", facecolor=SURFACE)
    print(out / "parked-rotation-before-after.png")


if __name__ == "__main__":
    main()
