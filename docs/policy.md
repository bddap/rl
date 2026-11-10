# Crab policy

The policy that drives Sally on the tasks `docs/evals.md` defines, and how it is
found. The design starts from the body, the tasks and the cost of simulating them.

**No old checkpoint.** The search starts from a seeded random initialization. No
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
  about 6 CPU-seconds per simulated second on a core of its own, so a day on 16
  threads is about 10⁷ ticks of 1/64 s. The design has to find a gait in that many.
- **Shaking moves this body faster than striding.** A servo covers most of the way to
  a new target within one tick, and a crab whose targets jump every tick slides along
  its fore–aft axis: targets drawn afresh each tick around the rest pose, with a
  standard deviation of a fifth of the way to a hinge's limit, carry it backward at
  0.062 m s⁻¹ and turn it 158° in 9 s. A stride scripted for all eight legs at the
  animal's frequency carries it sideways at 0.028 m s⁻¹ over its first three seconds,
  turning 10° a second. Whatever finds the policy has to be kept from the first and
  led to the second. These measurements, and the others below, are recorded in
  rl#440.

## Inputs and outputs

The harness's per-tick `Observation` and its action, one servo target per hinge in
[−1, 1], are the policy's interface (`docs/evals.md`).

One network, shared by all ten appendages, maps one appendage's view to that
appendage's six servo targets. A right appendage's view is mirrored through the
crab's sagittal plane first, so the network only ever sees a left appendage, and its
output needs no mirroring back.

The action the harness gets is not the network's targets but trails them: each tick
it moves a quarter of the way to them, a lag of 54 ms. The harness hands the policy
its last action back every tick and zeros it when an episode starts, so the lag
needs no state in the policy.

Each appendage's view, 38 numbers:

| input | n | scale |
|---|---|---|
| its hinge angles, in action units (0 rest, ±1 the limits) | 6 | 1 |
| its hinge rates, from the bones' angular velocities | 6 | per 10 rad s⁻¹ |
| its action as last sent, the state the lag carries | 6 | 1 |
| gravity's direction in the carapace frame | 3 | 1 |
| the carapace's velocity in its own frame | 3 | per 0.1 m s⁻¹ |
| the carapace's angular velocity in its own frame | 3 | per 10 rad s⁻¹ |
| the carapace's height above the ground | 1 | per 0.05 m |
| the direction the task walks, in the carapace frame; zero for `stand` and `righting` | 3 | 1 |
| a clock: sin and cos of 2π · 1.17 Hz · t | 2 | 1 |
| which appendage: cheliped or leg 1–4, one-hot | 5 | 1 |

The carapace frame is the carapace's rest axes (left, up, forward) carried by its
current pose. Mirroring negates the left component of a vector and the up and
forward components of an angular velocity.

- **Sharing.** The walking legs differ in position, not in kind, so one network
  conditioned on which appendage it drives holds once what ten would hold
  separately.
- **Mirroring.** It makes the policy exactly symmetric: what it does walking left it
  does, mirrored, walking right.
- **The clock** is the rhythm generator at its simplest, fixed at the animal's stride
  frequency. The network sets each appendage's phase against it; the feedback inputs
  let it correct the rhythm, not replace it. The tick is in the observation, so the
  clock needs no state.
- **The lag** keeps the action in the band a stride lives in. A 1.17 Hz stride passes
  it at 93% of its amplitude; targets that flip every tick pass at 14%. Without it
  the search below, run on the two walks and `stand`, finds the shaking: a policy
  scoring 0.04 m s⁻¹ on both walks whose leg targets reverse direction on 85% of the
  ticks they move, the joints at 733° s⁻¹, while the crab spins half way round. With
  it the same search finds a stride: one per 55 ticks, the clock's period, with the
  leg joints at 95° s⁻¹, near the scripted stride's 74° s⁻¹.
- **The walk as a direction.** The walks count sideways travel toward the side that
  faced left at spawn (`docs/evals.md`), and a walking crab turns. The view carries
  that direction as the carapace sees it, so the policy can tell where its travel
  counts once it has turned.
- **No memory.** An appendage's view holds its own joints, the carapace's motion, the
  action the lag holds and the clock, and the network is feedforward.
- **The claws stay in.** The chelipeds are not used to walk, but `righting` may need
  them.

## Network

The actor is an MLP, 38 → 64 → 64 → 6, tanh on every layer; its output is the six
servo targets the appendage's action trails. Nothing in the policy is drawn at random,
so each episode replays bit for bit as the harness requires.

Initialization is seeded and random: weights uniform in ±√(6 / (fan_in + fan_out)),
biases zero, the last layer scaled by 0.01 so the first policy is close to holding
the rest pose. Holding the rest pose already stands (`docs/evals.md`).

## Training

The actor is trained by direct search on the evals: a (1, λ) evolution strategy
(Beyer & Schwefel 2002) that keeps the fittest actor it has scored.

- **A generation** draws 8 candidates, each its parent with Gaussian noise of standard
  deviation 0.075 added to its last layer, 390 weights and biases. The two layers
  under it stay the random features they were seeded as. The first parent is the
  seeded actor.
- **Fitness** is built from the scores `sally-eval` reports: every candidate plays one
  episode of each of the four tasks, and its fitness is the sum of its scores, each
  over the task's reference (the animal's 0.097 m s⁻¹ pace on a walk, 1 on `stand`
  and `righting`). A candidate whose physics diverges is unfit, and a generation of
  nothing else leaves the parent where it was.
- **Selection.** A generation's fittest candidate parents the next, whether or not
  it beats its own parent, and the fittest actor scored in any generation is kept as
  the result. Searches that moved only to a fitter candidate stayed on the first gait
  they met: from the same first generation, 0.0067 m s⁻¹ on `walk_left`, one with
  this step stayed there for the two generations it ran, and one whose step narrowed
  after each failure stayed for eight and was at 0.010 m s⁻¹ after fourteen, where
  this one was at 0.014 m s⁻¹ after five.
- **No noise on the actions.** A learner that explores by perturbing actions trains
  the perturbed policy, not the scored one, and on this body the perturbation is the
  shaking above: what improves is where the shaking carries the crab, and nothing in
  that asks the policy to move once the shaking stops. Perturbing the weights
  instead, every candidate is a policy `sally-eval` would score as it stands.
- **No gradient.** Around a standing policy the walk scores do not turn over when a
  perturbation does: three perturbations of the last layer that moved the crab
  −0.0075, −0.0057 and +0.0015 m s⁻¹ toward the task's side had opposites that moved
  it −0.0009, −0.0019 and −0.0001 m s⁻¹. A slope there says little about where a
  stride lies; taking the best of several perturbations climbs regardless.
- **Determinism.** The candidates come from one seeded stream and each episode
  replays bit for bit, so a run replays exactly from its seed, on any number of
  threads.
- **Cost.** An actor's four episodes are 2432 ticks, a generation 19 456; the default
  budget of 1.25 × 10⁶ ticks covers the first actor and 64 generations, about 5.4
  simulated hours.

## Running it

`crab_world::policy` holds the network and the views, `crab_world::train` the
search. `sally-train` writes the kept actor to `policy.json` at the start and
whenever a generation improves it, and to `train.jsonl` its config and the first
actor's episodes, then one line per generation with every candidate's fitness
(`null` for one whose physics diverged) and the kept actor's episodes;
`sally-eval --policy trained` scores a `policy.json` on the evals. `sally.sh run` does
all of it from the purchased `.blend`: it rigs the crab, trains for `sally-train`'s
default budget from seed 0 unless given a budget in ticks and a seed, scores the
result and both baselines, and writes them side by side to `scores.md`:

```sh
scripts/sally/sally.sh run out [<ticks> [<seed>]]
```

`sally.sh clips` shows what a policy does: one episode of a task (`walk_left` unless
named) from the same start under the zero, random and trained policies, each
recorded by `sally-eval --record` and rendered by `clip.py` as the rig's own mesh
posed by its physics bodies every third tick, to `<out>/<policy>.mp4`:

```sh
scripts/sally/sally.sh clips clips out/train/policy.json
```

The steps one at a time:

```sh
scripts/sally/sally.sh rig sally.glb
nix-shell --run 'cargo run --release -p crab-world --bin sally-train -- sally.glb --out run'
nix-shell --run 'cargo run --release -p crab-world --bin sally-eval -- sally.glb --policy trained --weights run/policy.json'
```

## A run

Seed 0, cut after 14 of the default 64 generations (`scripts/sally/sally.sh run out
274816 0`): 2.7 × 10⁵ ticks, 72 simulated minutes. Scored by `sally-eval` beside the
baselines of `docs/evals.md`:

| task | zero | random | trained |
|---|---|---|---|
| `stand` | 1 | 0.038 (0 to 0.073) | 1 |
| `walk_left` | −9 × 10⁻⁷ m s⁻¹ | −0.0002 m s⁻¹ (−0.0082 to 0.0058) | 0.0139 m s⁻¹ |
| `walk_right` | 9 × 10⁻⁷ m s⁻¹ | 0.0002 m s⁻¹ (−0.0058 to 0.0082) | 0.0050 m s⁻¹ |
| `righting` | 0 | 0 (0 to 0) | 0.80 |

The kept actor, found in the eleventh generation, stands, walks toward the commanded
side on both walks at 14% and 5% of the animal's pace, and turns itself off its back
in 2.0 s. The two walks differ though the policy is symmetric and the rig mirrors to
about 1 mm.

## Not in it, by choice

- **Learned clock frequency.** The animal's 1.17 Hz is enough to walk at its routine
  pace; a policy that needs another speed earns a frequency input later.
- **Observation normalization.** Fixed scales from the body's own magnitudes, so the
  weights alone are the policy.
- **Curricula, domain randomization, demonstrations.** Each is a design decision a
  result has to justify first. The tasks are flat ground with one start each.
- **Searching under the last layer.** The layers under it join the search when a
  result shows the random features hold a gait back.

## References

- Beyer, H.-G. & Schwefel, H.-P. (2002). Evolution strategies: a comprehensive
  introduction. *Nat. Comput.* 1, 3–52. https://doi.org/10.1023/A:1015059928466
- Büschges, A. (2005). Sensory control and organization of neural networks mediating
  coordination of multisegmental organs for locomotion. *J. Neurophysiol.* 93,
  1127–1135. https://doi.org/10.1152/jn.00896.2004
- Chrachri, A. & Clarac, F. (1990). Fictive locomotion in the fourth thoracic ganglion
  of the crayfish, *Procambarus clarkii*. *J. Neurosci.* 10, 707–719.
  https://doi.org/10.1523/JNEUROSCI.10-03-00707.1990
- Martinez, M. M., Full, R. J. & Koehl, M. A. R. (1998). Underwater punting by an
  intertidal crab: a novel gait revealed by the kinematics of pedestrian locomotion in
  air versus water. *J. Exp. Biol.* 201, 2609–2623. https://doi.org/10.1242/jeb.201.18.2609
