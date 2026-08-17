#!/usr/bin/env python3
"""rl#376 wind-airspeed trace plots: the 64:30 staircase in the wind's speed read.

Reads the RL_POS_TRACE CSVs (W = airspeed the wind synth is driven with,
S = per-frame measured sim cost + pumped ticks) captured by
`game fp-screenshot --pilot-toggle-at 60 …` before and after the speed_mps
step-clock fix, and renders the evidence figure.
Usage: plot.py <capture-dir> <out-dir>
"""

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
BAND = "#e7e6e3"  # 3-step tick band, recessive neutral

PHYSICS_HZ, TICK_HZ = 64, 30


def steps_for_tick(t: int) -> int:
    return PHYSICS_HZ * t // TICK_HZ - PHYSICS_HZ * (t - 1) // TICK_HZ


def read_trace(path: Path):
    wind, cost = {}, {}
    for line in path.read_text().splitlines():
        f = line.split(",")
        try:
            if f[0] == "W" and len(f) == 3:
                wind[int(f[1])] = float(f[2])
            elif f[0] == "S" and len(f) == 4:
                cost[int(f[1])] = float(f[2])
        except ValueError:
            continue  # torn tail line of a killed capture
    return wind, cost


def shade_3step(ax, lo, hi):
    for t in range(lo, hi + 1):
        if steps_for_tick(t) == 3:
            ax.axvspan(t - 0.5, t + 0.5, color=BAND, zorder=0)


def main():
    cap, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    wind_b, cost_b = read_trace(cap / "before.csv")
    wind_a, _ = read_trace(cap / "after.csv")

    # Cruise window: past boarding + spool-up, present in both captures.
    lo = 120
    hi = min(max(wind_b), max(wind_a))
    tb = [t for t in sorted(wind_b) if lo <= t <= hi and wind_b[t] > 0]
    ta = [t for t in sorted(wind_a) if lo <= t <= hi and wind_a[t] > 0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 2]
    )
    fig.patch.set_facecolor(SURFACE)

    shade_3step(ax1, lo, hi)
    ax1.plot(tb, [wind_b[t] for t in tb], color=BLUE, lw=2, label="before (tick-average)")
    ax1.plot(ta, [wind_a[t] for t in ta], color=ORANGE, lw=2, label="after (step-clock)")
    ax1.set_ylabel("wind-seen airspeed, m/s", color=INK)
    ax1.legend(loc="upper right", frameon=False, labelcolor=INK)
    ax1.set_title(
        "rl#376 — the airspeed the wind synth hears, same cruise, before/after\n"
        "shaded bands = the 64:30 staircase's 3-step ticks",
        color=INK,
        fontsize=11,
    )

    shade_3step(ax2, lo, hi)
    tc = [t for t in sorted(cost_b) if lo <= t <= hi]
    ax2.plot(tc, [cost_b[t] for t in tc], color=INK2, lw=2)
    m2 = [cost_b[t] for t in tc if steps_for_tick(t) == 2]
    m3 = [cost_b[t] for t in tc if steps_for_tick(t) == 3]
    ax2.set_ylabel("sim cost / frame, ms", color=INK)
    ax2.set_xlabel("sim tick", color=INK)
    ax2.set_title(
        f"measured sim cost per frame (before capture) — "
        f"2-step mean {sum(m2)/len(m2):.1f} ms, 3-step mean {sum(m3)/len(m3):.1f} ms",
        color=INK,
        fontsize=10,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.grid(axis="y", color=BAND, lw=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out / "wind-speed-before-after.png", dpi=150)
    print(out / "wind-speed-before-after.png")


if __name__ == "__main__":
    main()
