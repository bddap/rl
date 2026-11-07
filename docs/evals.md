# Crab evals

What a good crab does, and how a policy driving Sally, the purchased *Grapsus
grapsus* model, is scored on it. The design starts from how real *Grapsus* crabs move.

## What a good crab does

The reference animal is *Grapsus tenuicrustatus*, the congener whose size and mass
`scripts/sally/skeleton.md` gives Sally (Martinez et al. 1998). Walking in air, those
crabs:

- **carry the body on the legs.** Carapace height 0.044 ± 0.004 m (Table 4); "on land,
  crabs almost never had fewer than three legs in contact with the ground."
- **walk sideways.** Travel is lateral, at a mean 9.7 ± 0.7 cm s⁻¹ (Table 1). The
  fastest terrestrial run recorded for the species is 1.4 m s⁻¹ (Martinez 2001). The
  carapace rolls 30 ± 5° while walking (Table 7), so a crab need not be level to be up.
- **leave the claws out of it.** The crabs "were not observed to use them [the
  chelipeds] during locomotion" (Martinez et al. 1998).

An overturned crab rights itself. We found no righting time for a grapsid; the
subtidal *Ebalia tuberosa* takes 2 to 900 s (Schembri 1981), which says the behaviour
exists, not how fast a *Grapsus* does it.

## The body the policy drives

The crab is the rig `scripts/sally/sally.sh rig` builds from the purchased mesh: 61
rigid bodies on 60 limited hinges. The mesh may not be redistributed, so the script
reads it from `$SALLY_BLEND` or the project's private asset store. The harness gives every hinge a servo; a policy
sets servo targets and nothing else.

- **Action.** One number per hinge, clamped to [−1, 1]: 0 targets the rest angle, +1
  and −1 the joint's upper and lower limits, linearly between. The zero action targets
  the purchased mesh's own posture, which the servos hold against gravity to within
  their error at the torque cap.
- **Servo.** Torque `k·(target − angle) − c·rate`, capped at τ_max, solved implicitly
  by rapier's force-based joint motor. τ_max is the crab's weight, from its bodies'
  hulls and density, at a lever of its carapace's width, the same for every joint;
  `sally-eval` prints it. k reaches the cap at 0.1 rad of error; c = 0.01 s · k. These
  three are this harness's choices: no source gives crab joint torques. A servo whose
  stiffness scales with each link's own inertia (rapier's default acceleration-based
  motor) cannot carry the body on gram-scale distal segments.
- **Observation.** Each tick the policy sees every bone's world pose and velocity,
  every hinge angle, the tick, the task, and the horizontal direction the crab's left
  side faced at spawn.
- **World.** Flat ground, gravity 9.81 m s⁻², 64 ticks per second with 16 physics
  substeps each: at one substep the gram-scale links pass through their joint limits.
  The crab spawns with its lowest point 10 mm above the ground. Bodies never sleep, so
  a still crab still answers its servos. Nothing is random but the policy, so an
  episode replays bit for bit. An episode whose physics goes non-finite ends and
  scores 0.

**Supported** is the state every task measures through: the carapace is tilted less
than 45° from its rest attitude, which admits the 30° walking roll, and the carapace's
own hull is at least 2 mm above the ground, so its weight is on the legs. 2 mm clears
the fraction of a millimetre rapier's contacts let a resting hull sink.

## Tasks

| task | start | length | metric | why it measures the behaviour |
|---|---|---|---|---|
| `stand` | upright | 8 s | fraction of ticks supported over the last 6 s | Carrying the body is the precondition of every gait. The first 2 s let the drop settle. |
| `walk_left`, `walk_right` | upright | 10 s each | net carapace displacement toward the side that faced left (right) at spawn, summed over supported ticks only, over 10 s, in m s⁻¹ | Sideways walking is how the animal travels. Counting only supported ticks scores walking, not sliding on the belly or tumbling. The mirrored pair is read together: a policy that drifts one way scores the same magnitude with opposite signs. |
| `righting` | overturned about the fore–aft axis | 10 s | 1 − t/10 s, where t is the start of the first 0.5 s spent supported; 0 if it never is | Recovering from an overturn decides whether the crab can keep going. Holding 0.5 s excludes passing through upright mid-tumble. |

The reference for the walks is the animal's 0.097 m s⁻¹ routine pace, with 1.4 m s⁻¹
its ceiling. `stand` and `righting` top out at 1.

Not scored, each by choice:

- **Gait resemblance** (stride frequency 1.17 Hz, duty factor 0.57, Table 1). It
  measures likeness, not function.
- **Energy.** No source gives a crab's metabolic cost for these servos to be charged
  against, and a made-up cost would rank policies by the made-up part.
- **Rough ground, turning, the claws.** Flat ground isolates the gait. Turning and
  claw use are behaviours to add once a policy stands and walks.

## Harness

`crab_world::eval` runs the tasks. `Env` is one episode a caller steps itself
(observe, act, step); `Policy` is the per-tick interface `run` and `score` drive.
`sally-eval` scores a baseline policy on the rig and prints one `SALLY_EVAL <json>`
line, with every episode's score:

```sh
scripts/sally/sally.sh rig sally.glb
nix-shell --run 'cargo run --release -p crab-world --bin sally-eval -- sally.glb --policy zero'
nix-shell --run 'cargo run --release -p crab-world --bin sally-eval -- sally.glb --policy random'
```

The zero policy holds the rest pose. The random policy draws every servo target
uniform in [−1, 1], independently each tick, from a seeded ChaCha8 stream; it runs
one episode per seed, 0 to 7 by default.

## Baselines

Scored on the rig from the purchased mesh. Random is the mean over seeds 0 to 7, with
the range across seeds.

| task | zero | random |
|---|---|---|
| `stand` | 1.0 | 0.045 (0 to 0.18) |
| `walk_left` | −0.0001 m s⁻¹ | 0.0016 m s⁻¹ (−0.0013 to 0.0079) |
| `walk_right` | 0.0001 m s⁻¹ | −0.0016 m s⁻¹ (−0.0079 to 0.0013) |
| `righting` | 0 | 0 (0 to 0) |

Holding the bought posture stands, so `stand` alone cannot tell a policy from the
zero baseline; walking and righting can. Neither baseline walks: zero drifts at 0.1% of the animal's
pace and random at 1.7% on average, 8% on its best seed, and each pair is equal and
opposite because neither policy reads the task. Neither rights itself in 10 s. The zero crab stands
with its carapace origin at 25 mm, against the animal's 44 mm walking height.

## References

- Martinez, M. M., Full, R. J. & Koehl, M. A. R. (1998). Underwater punting by an
  intertidal crab: a novel gait revealed by the kinematics of pedestrian locomotion in
  air versus water. *J. Exp. Biol.* 201, 2609–2623. https://doi.org/10.1242/jeb.201.18.2609
- Martinez, M. M. (2001). Running in the surf: hydrodynamics of the shore crab
  *Grapsus tenuicrustatus*. *J. Exp. Biol.* 204, 3097–3112.
  https://doi.org/10.1242/jeb.204.17.3097
- Schembri, P. J. (1981). Substratum preferences, burrowing and righting in *Ebalia
  tuberosa* (Pennant) (Crustacea: Decapoda: Leucosiidae). *Mar. Behav. Physiol.* 8,
  149–161.
