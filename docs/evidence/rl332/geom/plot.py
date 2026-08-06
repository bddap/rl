#!/usr/bin/env python3
"""rl#332 launch-geometry plots (job 2186). Usage:

    plot.py out_dir window.jsonl[.gz] ...

Per window: vy / horizontal speed / |v| time series with launch ticks marked,
plus above-ground altitude and carapace specific energy, one shared time axis
(separate subplots — never a dual axis). Across all windows: histogram of
launch angle to the local ground normal (0 deg = the owner's "perpendicular").
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze

# dataviz reference palette (light), validated via validate_palette.js.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "lines.linewidth": 2.0,
    }
)


def plot_window(path, out_dir):
    header, ticks, launches = analyze.analyze(path)
    t0 = ticks[0]["tick"]
    xs = [t["tick"] - t0 for t in ticks]
    vy = [t["linvel"][1] for t in ticks]
    vh = [analyze.hspeed(t["linvel"]) for t in ticks]
    sp = [math.hypot(a, b) for a, b in zip(vh, vy)]
    above = [t["above"] for t in ticks]
    whole = analyze.whole_energy(ticks[0]) is not None
    energy = [
        analyze.whole_energy(t) if whole else analyze.spec_energy(t) for t in ticks
    ]
    e0 = energy[0]
    energy = [e - e0 for e in energy]
    e_label = (
        "whole-body mech. energy (rel.)" if whole else "carapace specific energy (rel.)"
    )
    e_unit = "J vs window start" if whole else "J/kg vs window start"

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9, 7), sharex=True, height_ratios=[3, 2, 2]
    )
    ax1.plot(xs, vy, color=BLUE, label="vertical vy")
    ax1.plot(xs, vh, color=ORANGE, label="horizontal speed")
    ax1.plot(xs, sp, color=AQUA, label="|v|", linewidth=1.2)
    ax1.axhline(0, color=MUTED, linewidth=0.8)
    ax1.set_ylabel("m/s")
    ax1.legend(loc="upper left", frameon=False, fontsize=8)
    ax2.plot(xs, above, color=BLUE, label="carapace above ground")
    ax2.set_ylabel("m above ground")
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    ax3.plot(xs, energy, color=ORANGE, label=e_label)
    ax3.set_ylabel(e_unit)
    ax3.set_xlabel(f"tick − {t0} (64 Hz)")
    ax3.legend(loc="upper left", frameon=False, fontsize=8)
    for ax in (ax1, ax2, ax3):
        for l in launches:
            ax.axvline(l["launch_tick"] - t0, color=MUTED, linewidth=0.8, linestyle=":")
    for l in launches:
        ax1.annotate(
            f'{l["class"]}\nrise {l["vy_rise_ticks"]}t',
            (l["launch_tick"] - t0, max(vy) * 0.9),
            fontsize=7,
            color=MUTED,
            ha="left",
        )
    fig.suptitle(
        f'{path.name} — onset tick {header["onset_tick"]}, {len(launches)} launch(es)',
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(out_dir) / (path.name.split(".jsonl")[0] + "-velocity.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return launches


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    all_launches = []
    for p in map(Path, sys.argv[2:]):
        all_launches.extend(plot_window(p, out_dir))

    angles = [l["angle_to_normal_deg"] for l in all_launches if "angle_to_normal_deg" in l]
    elevs = [l["vel_elev_deg"] for l in all_launches]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4))
    axa.hist(angles, bins=list(range(0, 100, 10)), color=BLUE, edgecolor=SURFACE, linewidth=2)
    axa.set_xlabel("angle from local ground NORMAL, deg\n(0 = perpendicular to the slope)")
    axa.set_ylabel("launches")
    axa.set_title(f"vs terrain normal, n={len(angles)}", fontsize=10)
    axb.hist(elevs, bins=list(range(-40, 100, 10)), color=BLUE, edgecolor=SURFACE, linewidth=2)
    axb.set_xlabel("velocity elevation above horizon, deg\n(90 = straight up in the world frame)")
    axb.set_ylabel("launches")
    axb.set_title(f"vs world horizon, n={len(elevs)}", fontsize=10)
    fig.suptitle("rl#332 launch-velocity direction at liftoff", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "launch-angle-histogram.png", dpi=130)
    plt.close(fig)

    json.dump(all_launches, (out_dir / "launches.json").open("w"), indent=1)
    print(f"{len(all_launches)} launches, {len(angles)} with angle; plots in {out_dir}")


if __name__ == "__main__":
    main()
