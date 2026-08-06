#!/usr/bin/env python3
"""rl#332 launch-geometry analysis (job 2186, owner follow-up 2026-08-04).

Question under test: the owner saw launches "near perpendicular to the ground"
and suspects a physics bug (e.g. solver depenetration popping) rather than the
landed luge/ramp-conversion story. The two mechanisms differ in the SHAPE of
vertical velocity at launch:

  INJECTION (solver kick) - vy steps up by several m/s within 1-2 ticks
      (64 Hz: gravity moves vy only 0.153 m/s/tick) coincident with ground
      contact, WHILE total speed rises and carapace specific mechanical
      energy steps up beyond actuator reach (soak-measured gross <= 237 W
      through a 0.781 kg body -> O(300-600 J/kg/s); a from-nothing 5 m/s
      pop at 20 m/s reads as >3000 J/kg/s). Energy appears.
  BOUNCE (impact rebound) - vy also steps in 1-2 ticks, but it REDIRECTS a
      large incoming downward vy: total speed drops and energy dissipates
      hard (the 46 m baseline event: in at 63 m/s falling, out at 32 m/s,
      dE ~ -70k J/kg/s). Legitimate collision response.
  CONVERSION (ramp/luge or policy climb) - vy grows over tens of ticks;
      specific-energy rate stays within the actuator bound.

Input: sally-soak JSONL windows (header line + per-tick state, y-up, 64 Hz).
`contacts` counts ALL narrow-phase points including self-contacts (68+ while
37 m airborne), so ground contact is instead inferred from part clearance:
min over parts of part.y minus the terrain height under the CARAPACE
(cara.y - above). Flat-local approximation: on a 20 deg slope across her
~0.5 m half-span this errs by ~0.18 m, hence the wide hysteresis band.

Outputs: per-launch records (angle to local ground normal, vy rise time,
kick scan) as JSON on stdout; plots via plot.py.
"""
import gzip
import json
import math
import sys
from pathlib import Path

G = 9.81
DT = 1.0 / 64.0
GROUNDED_M = 0.12  # min part clearance below this = on the ground
LAUNCHED_M = 0.35  # sustained clearance above this = airborne
AIRBORNE_MIN = 16  # ticks (0.25 s) of clearance to call it a launch
APPROACH = 16  # ticks used to fit the ground slope before liftoff
KICK_DVY = 2.0  # 1-tick carapace vy gain (m/s) that no gravity tick explains
IMPULSIVE_RISE_TICKS = 2
IMPULSIVE_GAIN = 3.0  # m/s vy gain across the rise for the impulsive call
IMPULSIVE_DE = 2000.0  # J/kg/s carapace specific-energy rate beyond actuators


def load_window(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[0], lines[1:]


def hspeed(v):
    return math.hypot(v[0], v[2])


def spec_energy(t):
    v = t["linvel"]
    return 0.5 * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + G * t["cara"][1]


def clearance(t):
    ground = t["cara"][1] - t["above"]
    return min(p[1] for p in t["parts"]) - ground


def ground_slope(ticks, i):
    """Signed rise/run of terrain under the carapace along the track over the
    APPROACH ticks before index i (least squares); None if she barely moved."""
    lo = max(0, i - APPROACH)
    pts, s, prev = [], 0.0, None
    for t in ticks[lo : i + 1]:
        x, y, z = t["cara"]
        if prev is not None:
            s += math.hypot(x - prev[0], z - prev[1])
        pts.append((s, y - t["above"]))
        prev = (x, z)
    if s < 0.05:
        return None
    n = len(pts)
    ms = sum(p[0] for p in pts) / n
    mh = sum(p[1] for p in pts) / n
    var = sum((p[0] - ms) ** 2 for p in pts)
    if var < 1e-9:
        return None
    return sum((p[0] - ms) * (p[1] - mh) for p in pts) / var


def find_launches(ticks):
    """Indices of grounded -> sustained-airborne transitions (clearance
    hysteresis). The launch index is the last grounded tick + 1."""
    out = []
    i = 0
    n = len(ticks)
    while i < n:
        if clearance(ticks[i]) < GROUNDED_M:
            j = i + 1
            while j < n and clearance(ticks[j]) < LAUNCHED_M:
                j += 1
            k = j
            while k < n and clearance(ticks[k]) >= LAUNCHED_M:
                k += 1
            if j < n and k - j >= AIRBORNE_MIN:
                # back up to the last grounded tick before j
                last = j - 1
                while last > i and clearance(ticks[last]) >= GROUNDED_M:
                    last -= 1
                out.append(last + 1)
                i = k
                continue
        i += 1
    return out


def rise_time(ticks, i):
    """vy gain around launch and how many ticks the 25%->75% rise takes.
    Baseline = min vy in the 32 ticks before launch; peak = max vy in the 48
    after."""
    lo, hi = max(0, i - 32), min(len(ticks), i + 48)
    seg = [t["linvel"][1] for t in ticks[lo:hi]]
    base = min(seg[: i - lo + 1]) if i > lo else seg[0]
    peak = max(seg[i - lo :])
    gain = peak - base
    if gain <= 0:
        return gain, None
    t25 = t75 = None
    for k, vy in enumerate(seg):
        if t25 is None and vy >= base + 0.25 * gain:
            t25 = k
        if vy >= base + 0.75 * gain:
            t75 = k
            break
    return gain, (t75 - t25) if t25 is not None and t75 is not None else None


def kick_scan(ticks):
    """Every 1-tick carapace vy gain >= KICK_DVY anywhere in the window, with
    context. Gravity's tick step is -0.153; actuators push the carapace only
    through leg linkages."""
    kicks = []
    for k in range(1, len(ticks)):
        a, b = ticks[k - 1], ticks[k]
        dvy = b["linvel"][1] - a["linvel"][1]
        if dvy >= KICK_DVY:
            sa = math.hypot(hspeed(a["linvel"]), a["linvel"][1])
            sb = math.hypot(hspeed(b["linvel"]), b["linvel"][1])
            kicks.append(
                {
                    "tick": b["tick"],
                    "dvy": dvy,
                    "speed": f"{sa:.1f}->{sb:.1f}",
                    "gained_speed": sb > sa * 1.05,
                    "dE_per_s": (spec_energy(b) - spec_energy(a)) / DT,
                    "clearance": clearance(b),
                    "contacts": b["contacts"],
                    "above": b["above"],
                }
            )
    return kicks


def analyze(path):
    header, ticks = load_window(path)
    launches = []
    for i in find_launches(ticks):
        t = ticks[i]
        v = t["linvel"]
        vh = hspeed(v)
        theta_v = math.degrees(math.atan2(v[1], vh))
        gain, rise = rise_time(ticks, i)
        lo = max(1, i - IMPULSIVE_RISE_TICKS)
        de2 = max(
            (spec_energy(ticks[k]) - spec_energy(ticks[k - 1])) / DT
            for k in range(lo, min(i + IMPULSIVE_RISE_TICKS + 1, len(ticks)))
        )
        before = ticks[max(0, i - 4)]["linvel"]
        after = ticks[min(len(ticks) - 1, i + 2)]["linvel"]
        speed_before = math.hypot(hspeed(before), before[1])
        speed_after = math.hypot(hspeed(after), after[1])
        fast = rise is not None and rise <= IMPULSIVE_RISE_TICKS and gain >= IMPULSIVE_GAIN
        if fast and speed_after > speed_before * 1.05 and de2 >= IMPULSIVE_DE:
            kind = "injection"
        elif fast:
            kind = "bounce"
        else:
            kind = "conversion"
        rec = {
            "file": path.name,
            "event_onset_tick": header["onset_tick"],
            "launch_tick": t["tick"],
            "idx": i,
            "vy": v[1],
            "vh": vh,
            "speed": math.hypot(vh, v[1]),
            "vel_elev_deg": theta_v,
            "vy_gain": gain,
            "vy_rise_ticks": rise,
            "speed_before": speed_before,
            "speed_after": speed_after,
            "max_dE_per_s_2tick": de2,
            "class": kind,
        }
        slope = ground_slope(ticks, i)
        if slope is not None:
            theta_g = math.degrees(math.atan(slope))
            rec["ground_slope_deg"] = theta_g
            # angle between velocity and the local ground NORMAL in the
            # along-track vertical plane; 0 = the owner's "perpendicular".
            rec["angle_to_normal_deg"] = abs(90.0 - (theta_v - theta_g))
        launches.append(rec)
    return header, ticks, launches


def main(paths):
    out = {"launches": [], "kicks": []}
    for p in map(Path, paths):
        header, ticks, launches = analyze(p)
        out["launches"].extend(launches)
        for k in kick_scan(ticks):
            k["file"] = p.name
            out["kicks"].append(k)
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
