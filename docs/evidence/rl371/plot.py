#!/usr/bin/env python3
"""rl#371 position-trace plots: sim-true vs render-side walking jitter.

Reads the RL_POS_TRACE CSVs captured by `game fp-screenshot --walk-straight
--frame-hz 60 [--frame-jitter-ms J]` and renders the before/after evidence.
Usage: plot.py <capture-dir> <out-dir>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UNIT = 100_000  # grid units per meter
TICK_HZ = 30.0

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def load(path):
    """-> (tick_speed_mm_s, frame_speed_mm_s, frame_dt_s). Speeds are horizontal
    (xz) step magnitude over the step's own duration — one ruler for both."""
    ticks, frames = [], []
    for line in Path(path).read_text().splitlines():
        f = line.split(",")
        if f[0] == "T":
            ticks.append((int(f[1]), int(f[2]), int(f[3])))
        elif f[0] == "F":
            frames.append((float(f[2]), int(f[3]), float(f[4]), float(f[6])))
    t = np.array(list(dict((tk, (x, z)) for tk, x, z in ticks).values()), float)
    tick_speed = np.hypot(*np.diff(t, axis=0).T) / UNIT * TICK_HZ * 1000.0
    fr = np.array(frames, float)
    dt = fr[1:, 1] / 1e6
    frame_speed = np.hypot(*np.diff(fr[:, 2:4], axis=0).T) * 1000.0 / dt
    return tick_speed, frame_speed, dt


def panel(ax, y, color, label, ideal):
    ax.plot(np.arange(len(y)), y, color=color, lw=1.4, marker="o", ms=2.6, mew=0)
    ax.axhline(ideal, color=INK2, lw=0.8, ls=(0, (4, 4)), alpha=0.6)
    med = np.median(y)
    worst = np.max(np.abs(y - ideal))
    ax.text(
        0.01, 0.96,
        f"{label}   median {med:.1f} mm/s · worst dev {worst:.1f} mm/s",
        transform=ax.transAxes, va="top", fontsize=9, color=INK,
    )
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e8e7e3", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def fig(panels, title, out, window, ylim):
    n = len(panels)
    f, axes = plt.subplots(n, 1, figsize=(8, 1.9 * n + 0.8), sharex=True, sharey=True)
    f.patch.set_facecolor(SURFACE)
    for ax, (y, color, label, ideal) in zip(np.atleast_1d(axes), panels):
        panel(ax, y[window], color, label, ideal)
        ax.set_ylim(*ylim)
    np.atleast_1d(axes)[-1].set_xlabel("frame (60 Hz)", fontsize=9, color=INK2)
    f.supylabel("horizontal speed per step  (mm/s)", fontsize=9, color=INK2)
    f.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    f.tight_layout(rect=(0.02, 0, 1, 0.97))
    f.savefig(out, dpi=160)
    plt.close(f)
    print(out)


def main(cap, out):
    cap, out = Path(cap), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tick_b, frame_b, _ = load(cap / "far-fixed60.csv")
    _, frame_a, _ = load(cap / "far-fixed60-after.csv")
    _, jframe_b, _ = load(cap / "far-jitter60.csv")
    _, jframe_a, _ = load(cap / "far-jitter60-after.csv")
    ideal = float(np.median(tick_b))
    w = slice(1200, 1320)  # a 2 s steady-walk window, same for every panel

    fig(
        [
            (np.repeat(tick_b, 2), BLUE, "sim, per-tick (÷2 to 60 Hz)", ideal),
            (frame_b, ORANGE, "camera, per-frame — BEFORE", ideal),
            (frame_a, AQUA, "camera, per-frame — AFTER (delta-form lerp_pos)", ideal),
        ],
        "Walking at a 14.4 km locale, exact 60 Hz frame clock — rl#371",
        out / "fixed60.png", w, (100, 180),
    )
    fig(
        [
            (jframe_b, ORANGE, "camera, per-frame — BEFORE", ideal),
            (jframe_a, AQUA, "camera, per-frame — AFTER (delta-form lerp_pos)", ideal),
        ],
        "Same locale, 60 Hz ± 1.5 ms frame-time jitter — rl#371",
        out / "jitter60.png", w, (90, 195),
    )
    for name, y in [
        ("sim per-tick", tick_b * 0 + tick_b),
        ("camera before", frame_b),
        ("camera after", frame_a),
        ("camera before (jittered clock)", jframe_b),
        ("camera after (jittered clock)", jframe_a),
    ]:
        s = y[600:2300] if len(y) > 2300 else y
        print(f"{name:34s} median {np.median(s):7.2f}  p2p {np.ptp(s):7.2f} mm/s")


if __name__ == "__main__":
    main(*sys.argv[1:3])
