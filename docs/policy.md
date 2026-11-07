# Crab policy

The policy that drives Sally on the tasks `docs/evals.md` defines, and how it is
trained. The design starts from the body, the tasks and the cost of simulating them.

**No old checkpoint.** The policy trains from a seeded random initialization. No
earlier checkpoint is loaded, distilled, warm-started from or consulted, and no
earlier architecture informs this one. Archived checkpoints stay outside every code
path.

## What the design has to respect

- **The body repeats itself.** Ten appendages, each a chain of six hinges: two
  chelipeds and four pairs of walking legs. Left and right mirror each other, and the
  rig's joint convention (positive is dorsal, anterior or opening, on both sides)
  makes a mirrored motion the same action on both sides.
- **The tasks are mirror images.** `walk_left` is `walk_right` seen in a mirror.
  `stand` and `righting` are their own mirror images.
- **Walking is rhythmic.** *Grapsus* strides at 1.17 Hz at its routine pace
  (Martinez et al. 1998, Table 1), and the legs alternate. In decapods the walking
  rhythm comes from pattern generators in the thoracic ganglia (Chrachri & Clarac
  1990), shaped by sensory feedback from the legs (Büschges 2005).
- **Simulation is the scarce resource.** The 61-body multibody at 16 substeps costs
  about 7 CPU-seconds per simulated second, so a day on 16 threads is about 10⁷
  ticks. The design has to learn from that many samples.

## Inputs and outputs

The harness's per-tick `Observation` and its action, one servo target per hinge in
[−1, 1], are the policy's interface (`docs/evals.md`).

One network, shared by all ten appendages, maps one appendage's view to that
appendage's six servo targets. A right appendage's view is mirrored through the
crab's sagittal plane first, so the network only ever sees a left appendage, and its
output needs no mirroring back.

Each appendage's view, 30 numbers:

| input | n | scale |
|---|---|---|
| its hinge angles, in action units (0 rest, ±1 the limits) | 6 | 1 |
| its hinge rates, from the bones' angular velocities | 6 | per 10 rad s⁻¹ |
| gravity's direction in the carapace frame | 3 | 1 |
| the carapace's velocity in its own frame | 3 | per 0.1 m s⁻¹ |
| the carapace's angular velocity in its own frame | 3 | per 10 rad s⁻¹ |
| the carapace's height above the ground | 1 | per 0.05 m |
| the commanded walk: +1 toward the appendage's own side, −1 away, 0 for `stand` and `righting` | 1 | 1 |
| a clock: sin and cos of 2π · 1.17 Hz · t | 2 | 1 |
| which appendage: cheliped or leg 1–4, one-hot | 5 | 1 |

The carapace frame is the carapace's rest axes (left, up, forward) carried by its
current pose. Mirroring negates the left component of a vector and the up and
forward components of an angular velocity, and swaps the command's sign.

- **Sharing.** The walking legs differ in position, not in kind, so one network
  conditioned on which appendage it drives learns once what ten would learn
  separately, and every tick trains it ten times.
- **Mirroring.** It makes the policy exactly symmetric: whatever it learns walking
  left it does walking right, and a gait that drifts one way is not representable.
- **The clock** is the rhythm generator at its simplest, fixed at the animal's stride
  frequency. The network learns each appendage's phase against it; the feedback inputs
  let it correct the rhythm, not replace it. The tick is in the observation, so the
  clock needs no state.
- **No memory.** The view is the body's full mechanical state plus the clock, so a
  feedforward network suffices. A recurrent one costs training time for state the
  physics already shows.
- **The claws stay in.** The chelipeds are not used to walk, but `righting` may need
  them; the network learns what to do with them.

## Network

The actor is an MLP, 30 → 64 → 64 → 6, tanh on every layer. Its output is the mean of
a diagonal Gaussian over the six servo targets, with one learned log standard deviation
per hinge index, shared by every appendage. The action the harness sees is a sample
while training and the mean when scored. That makes the scored policy deterministic,
and each episode replays bit for bit as the harness requires.

The critic estimates the return from the whole crab: every hinge angle and rate, the
carapace's gravity, velocity, angular velocity and height, the task one-hot, the clock
and the fraction of the episode elapsed, 137 numbers, through an MLP 137 → 64 → 64 → 1.
The elapsed fraction lets it value a cut-off episode's last ticks. It runs only in training.

Initialization is seeded and random: weights uniform in ±√(6 / (fan_in + fan_out)),
biases zero, each network's last layer scaled by 0.01 so the first policy is close
to holding the rest pose and the first value estimates are near zero, and the log standard deviations at ln 0.2. Holding the rest pose
already stands (`docs/evals.md`), and a crab that falls on every exploratory step
never sees what walking pays.

## Training algorithm

Proximal policy optimization (Schulman et al. 2017) with generalized advantage
estimation (Schulman et al. 2016). It reuses each batch of ticks for several gradient
steps and assigns credit per tick, so it learns from fewer simulated seconds than
methods that score whole episodes, and simulated seconds are what binds.

- **Rollouts.** 16 parallel episodes, one per thread, each its own world. Each
  episode's task is drawn uniformly from the four eval tasks, and its length is the
  task's. Every iteration steps each episode 256 ticks, continuing across iterations
  and resetting to a new task when one ends; that is 4096 ticks, or 64 simulated
  seconds, per iteration.
- **Reward,** per tick, is built from the quantities the harness scores each tick,
  whether the crab is supported and how far it moved:
  - `stand`: 1 if supported. The metric counts only the last 6 s; paying the first
    2 s too pays for not falling during the drop.
  - each walk: the carapace's displacement toward the commanded side on a supported
    tick, over the animal's routine 0.097 m s⁻¹ pace times the tick length.
  - `righting`: 1 if supported. Its metric rewards the first held support; rewarding
    every supported tick pays for getting up early and staying up.
  No shaping term is added; one would rank policies by the made-up part.
- **Ends.** An episode reaching its task's length is cut off, not terminated: its
  value bootstraps from the critic. Non-finite physics terminates with no further
  reward.
- **Update.** γ = 0.99 and λ = 0.95, advantages normalized over the batch. Four
  epochs of minibatches of 1024 ticks, clip 0.2. The actor's loss sums the ten
  appendages' log-probabilities into the tick's one action; the critic's is half the
  squared error of its estimate. Actor and critic each have their own Adam at
  3 × 10⁻⁴ and their gradient norm clipped to 0.5.
- **Determinism.** Each episode slot runs on its own thread and draws its noise and
  tasks from its own seeded stream, so a run replays exactly given its seed and slot
  count; a different slot count is a different run.

The network, its backward pass and Adam are plain Rust, about the size of this
document: the networks are tens of thousands of parameters, and a tensor library would
be a larger dependency than the code it replaces.

## Running it

`crab_world::policy` holds the network and the views, `crab_world::train` the
learner. `sally-train` writes the actor to `policy.json` every 10 iterations and after
the last, and to `train.jsonl` its config, then one line per iteration with every
ended episode scored by its task's metric; `sally-eval` scores
the actor's mean action on the evals:

```sh
scripts/sally/sally.sh rig sally.glb
nix-shell --run 'cargo run --release -p crab-world --bin sally-train -- sally.glb --out run --iterations 200'
nix-shell --run 'cargo run --release -p crab-world --bin sally-eval -- sally.glb --policy trained --weights run/policy.json'
```

## First run

Seed 0, 320 iterations: 1.31 × 10⁶ ticks, about 5.7 simulated hours, from random
initialization. Scored by `sally-eval` on its mean action, beside the baselines of
`docs/evals.md`, all re-run on the same build:

| task | zero | random | trained |
|---|---|---|---|
| `stand` | 1.0 | 0.045 (0 to 0.18) | 1.0 |
| `walk_left` | −0.0001 m s⁻¹ | 0.0016 m s⁻¹ (−0.0013 to 0.0079) | −0.0002 m s⁻¹ |
| `walk_right` | 0.0001 m s⁻¹ | −0.0016 m s⁻¹ (−0.0079 to 0.0013) | −0.0006 m s⁻¹ |
| `righting` | 0 | 0 | 0 |

The trained policy stands and does not walk or right itself; on the evals it is no
better than holding the rest pose. Its training episodes, which sample actions around
the mean with σ ≈ 0.19, did walk: over the last 40 iterations they averaged
0.022 m s⁻¹ left and 0.021 m s⁻¹ right, a fifth of the animal's routine pace and
13 times the random baseline's mean. The walking is in the sampled noise,
not in the mean the evals score. No episode, trained or sampled, righted itself.

## Not in it, by choice

- **Learned clock frequency.** The animal's 1.17 Hz is enough to walk at its routine
  pace; a policy that needs another speed earns a frequency input later.
- **Observation normalization.** Fixed scales from the body's own magnitudes, so the
  weights alone are the policy.
- **Curricula, domain randomization, demonstrations.** Each is a design decision a
  result has to justify first. The tasks are flat ground with one start each.

## References

- Büschges, A. (2005). Sensory control and organization of neural networks mediating
  coordination of multisegmental organs for locomotion. *J. Neurophysiol.* 93,
  1127–1135. https://doi.org/10.1152/jn.00896.2004
- Chrachri, A. & Clarac, F. (1990). Fictive locomotion in the fourth thoracic ganglion
  of the crayfish, *Procambarus clarkii*. *J. Neurosci.* 10, 707–719.
  https://doi.org/10.1523/JNEUROSCI.10-03-00707.1990
- Martinez, M. M., Full, R. J. & Koehl, M. A. R. (1998). Underwater punting by an
  intertidal crab: a novel gait revealed by the kinematics of pedestrian locomotion in
  air versus water. *J. Exp. Biol.* 201, 2609–2623. https://doi.org/10.1242/jeb.201.18.2609
- Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. (2016). High-dimensional
  continuous control using generalized advantage estimation. *ICLR*.
  https://arxiv.org/abs/1506.02438
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). Proximal
  policy optimization algorithms. https://arxiv.org/abs/1707.06347
