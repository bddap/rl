# rl#332 launch geometry — is there a depenetration kick? (job 2186)

Owner (2026-08-04): "Launches I saw tended to be near perpendicular to the
ground. Was hoping it was a lead to a physics bug." This dir tests that against
recorded launch kinematics, treating job 2133's luge-conversion conclusion as a
hypothesis under challenge, not a given.

## Method (`analyze.py`, plots via `plot.py`)

Inputs: sally-soak full-body JSONL windows (64 Hz, y-up) from three sources —
the two archived rl#332 windows, fresh soaks of the released af58cf1 binary,
and fresh soaks of an energy-instrumented build of BOTH main and the PRE-drag
commit (78d1a2f, the physics regime the owner watched on 2026-08-01; solver
identical, no air drag). Instrumented windows carry whole-body mechanical
energy per tick (mass-weighted, incl. rotation — `crab_mech_energy`).

Per launch (grounded → ≥0.25 s airborne, by part clearance — the `contacts`
field counts self-contacts, 60+ even mid-flight, so it cannot detect liftoff):

- **vy rise time** — an impulsive solver kick steps vy several m/s in 1–2
  ticks (gravity moves it 0.153 m/s/tick); a ramp conversion takes tens.
- **speed ratio** across the launch — redirection preserves or loses |v|;
  injection gains it.
- **net whole-body energy across the strike** (±8 ticks) — the decisive
  ruler. 16 ticks of full measured actuator gross (≤440 W) is ~110 J; the
  soft 30 Hz contact springs can return within a strike only what the same
  strike stored. A net gain ≫ that is energy from nowhere. (2-tick rates
  false-alarm on spring return; the net cannot.)
- **launch angle** vs the local ground normal (along-track slope fit over the
  16 approach ticks) and vs the world horizon.

Plus, independent of launches: a whole-window kick scan (every 1-tick carapace
vy gain ≥ 2 m/s, with concurrent speed and energy change) and a ballistic
check (airborne stretches must fall at −g; parts-mean COM proxy).

Rapier's depenetration pop mechanism exists and is bounded
(`normalized_max_corrective_velocity` = 10 m/s default) — but firing it needs
deep penetration, and since rl#299 (in every build here) contact stiffness
rests limbs ≲1 cm into terrain.

## Results (13 launches: 2 archived windows + 7 fresh instrumented — main s2,
pre-fix s2 + a second pre-fix seed; coverage note below)

**Zero impulsive-injection launches. 8 conversions, 5 bounces.**

- Every instrumented launch LOSES whole-body energy across the strike: net
  −41 to −1045 J over ±8 ticks (`launches.json`,
  `net_whole_dE_strike`). The four injection-candidate kicks the carapace-only
  ruler flags all live in the two archived (pre-instrumentation) windows —
  crash-rebound transfer (in at 63 m/s falling, out at 32: −70 kJ/kg/s) and
  wall-climb leg thrust at 10–21 m clearance — and the whole-body ruler
  clears every kick in every instrumented window.
- **Launch angles: 31–84° OFF the local ground normal (median 55°), liftoff
  velocity elevations −36° to +31°** — `launch-angle-histogram.png`. Nothing
  near-perpendicular, in either build. The steepest ascent anywhere in any
  window (peak-ascent, what a viewer tracks mid-flight) is 43° above the
  horizon.
- Ballistic phases are clean: airborne parts-mean dvy/dt residuals vs −g are
  −2.1..+3.1 m/s² with no positive bias (unweighted-mean COM proxy noise;
  the instrumented main window shows −0.3 m/s², i.e. drag).
- Teleports (the rescue path): 0 in every run.

The pre-fix (owner-regime) soaks luge continuously — seed s2 fires an event
per 512-tick cooldown through the first storm (peak vy 31 m/s, peak 15 m
above ground), which is what the TV session showed on 2026-08-01. Measured
geometry says those launches leave the slope at ~35° above the slope plane,
shallow in the world frame; the "near perpendicular" look is the arc seen
against terrain that is itself falling away at 20–47°: she rises 10–47 m
RELATIVE to ground on a world-shallow trajectory (e.g.
`prefix-s2-event-3…velocity.png`: vy snaps −30→−5 at a dissipative strike,
E −1000 J, then "climbs" 10 m of receding slope).

**Verdict: luge-conversion story confirmed in both regimes; no depenetration
kick.** Rapier's pop mechanism (10 m/s corrective-velocity cap) never fires
detectably — consistent with rl#299 stiffness keeping penetration ≲1 cm.

Coverage: instrumented soaks — pre-fix s2/x3 and main s2 to ≥100 k ticks at
sweep time (events cluster in the first storm; none later up to 100 k), main
x1 150 k censored-zero, main x2 ~50 k censored-zero. The release-binary s2
soak (200 k, 1 event) duplicates instrumented main s2 tick-for-tick and is
excluded from counts.
