# Crab policy

The policy that drives Sally on the tasks `docs/evals.md` defines, and how it is
found. The design starts from the body, the tasks and the cost of simulating them.

**No old checkpoint.** The search starts from a policy that holds the rest pose. No
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
  led to the second.
- **The legs take turns.** A leg that stays on the ground while the body travels is
  dragged. A policy whose legs all swung in phase kept its trailing legs down for 96
  to 98% of a walk, slid them 0.27 to 0.29 m in 8 s, the distance the body covered,
  and swept its feet 15 to 33 mm a stride where they reach 50 to 71 mm. None of
  three searches of such policies passed 0.036 m s⁻¹.

These measurements, and the others below, are recorded in rl#440 and rl#442.

## Inputs and outputs

The harness's per-tick `Observation` and its action, one servo target per hinge in
[−1, 1], are the policy's interface (`docs/evals.md`).

One readout, shared by all ten appendages, maps one appendage's view to that
appendage's six servo targets. A right appendage's view is mirrored through the
crab's sagittal plane first, so the readout only ever sees a left appendage, and its
output needs no mirroring back.

The action the harness gets is not the readout's targets but trails them: each tick
it moves a quarter of the way to them, a lag of 54 ms. The harness hands the policy
its last action back every tick and zeros it when an episode starts, so the lag
needs no state in the policy.

Each appendage's view, 66 numbers:

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
| the stride of its own kind of appendage (cheliped or leg 1–4), seven numbers; zero in the other four kinds' places | 35 | 1 |

The carapace frame is the carapace's rest axes (left, up, forward) carried by its
current pose. Mirroring negates the left component of a vector and the up and
forward components of an angular velocity.

The stride is `1, a, a·sin, a·cos, b, b·sin, b·cos`. `a` is how far the walk points
to the appendage's own side: the left component of the walk's direction in its
mirrored view, or 0 where that is negative. `b` is how far it points to the other
side: the same component negated, or 0. The sine and cosine are of a clock at
2π · 1.17 Hz · t. So a leg that leads is given a centre and a swing through `a`, and
one that trails another centre and swing through `b`.

- **Sharing.** The walking legs differ in position, not in kind, so what an appendage
  does about its joints, the carapace and the walk's direction is held once for all
  ten. Only the stride is held per kind.
- **Mirroring.** It makes the policy exactly symmetric: what it does walking left it
  does, mirrored, walking right.
- **The clock** is the rhythm generator at its simplest, fixed at the animal's stride
  frequency. The feedback inputs let the policy correct the rhythm, not replace it.
  The tick is in the observation, so the clock needs no state.
- **A stride per kind and per side.** For the legs to take turns, each has to be able
  to swing at its own phase, and a leg on the side the crab walks toward pulls where
  one on the other side pushes. A weight on the clock that all appendages share
  swings them together: over the eight legs' views on a walk, a weighted sum of a
  view holding the clock once and the kind as a one-hot fits a stride of eight legs
  in phase with R² 1.00 and one of two alternating sets of four with 0.08. With the
  stride above it fits both with 1.00. With nowhere to walk, on `stand` and
  `righting`, the stride is its constant alone and the crab is driven by its posture
  and its feedback.
- **The lag** keeps the action in the band a stride lives in. A 1.17 Hz stride passes
  it at 93% of its amplitude; targets that flip every tick pass at 14%. Without it a
  search run on the two walks and `stand` finds the shaking: a policy scoring
  0.04 m s⁻¹ on both walks whose leg targets reverse direction on 85% of the ticks
  they move, the joints at 733° s⁻¹, while the crab spins half way round. With it the
  same search finds a stride: one per 55 ticks, the clock's period, with the leg
  joints at 95° s⁻¹, near the scripted stride's 74° s⁻¹.
- **The walk as a direction.** The walks count sideways travel toward the side that
  faced left at spawn (`docs/evals.md`), and a walking crab turns. The view carries
  that direction as the carapace sees it, so the policy can tell where its travel
  counts once it has turned, and the stride fades as the crab turns away from it.
- **No memory.** An appendage's view holds its own joints, the carapace's motion, the
  action the lag holds and the clock, and the readout keeps nothing between ticks.
- **The claws stay in.** The chelipeds are not used to walk, but `righting` may need
  them.

## Readout

Each servo target is the tanh of a weighted sum of the appendage's view: six rows of
66 weights, 396 numbers. Nothing in the policy is drawn at random, so each episode
replays bit for bit as the harness requires.

The first policy is all zeros: it holds the rest pose, which already stands
(`docs/evals.md`).

## Training

The actor is trained by direct search on the evals: a (1, λ) evolution strategy
(Beyer & Schwefel 2002) that keeps the fittest actor it has scored, and beside it a
search of that kept actor's strides.

- **A generation** draws 8 candidates, each its parent with Gaussian noise of standard
  deviation 0.075 added to every weight, and then 8 refinements, each the kept actor
  with such noise added to its stride weights alone: the 180 weights on the six `a`
  and `b` inputs of each kind. The first parent is the zero actor. The run's seed
  seeds two streams of noise, one for the candidates and one for the refinements, so
  the parents a seed leads through are the same whatever is refined on the way.
- **Fitness** is built from the scores `sally-eval` reports: an actor's fitness is the
  sum of its scores on the four tasks, each over the task's reference (the animal's
  0.097 m s⁻¹ pace on a walk, 1 on `stand` and `righting`). A candidate plays one
  episode of each task. A refinement plays the two walks and is given the kept
  actor's `stand` and `righting` episodes, which it would repeat tick for tick: the
  inputs its changed weights sit on are zero there. An actor whose physics diverges
  is unfit, and a generation whose candidates are all unfit leaves the parent where
  it was.
- **Selection.** A generation's fittest candidate parents the next, whether or not
  it beats its own parent, and the fittest actor scored in any generation is kept as
  the result. A search that moves only to a fitter candidate stays on the first gait
  it meets: from one first generation (rl#440), a search with this step stayed there
  for the two generations it ran and one whose step narrowed after each failure for
  eight, where this rule had doubled the walks' speed after five. A refinement
  parents nothing: the fittest of a generation's refinements replaces the kept actor
  if it is the fitter. The kept actor is the result, so it only ever moves to a
  fitter one, and it is refined in its strides alone, where such a move can lose
  nothing on `stand` or `righting`.
- **Why refinements.** A candidate changes every weight at once. Measured on one kept
  actor, the first of a run to right itself (`righting` 0.955, walking 0.024 and
  0.009 m s⁻¹): none of 16 candidates drawn around it righted itself and 10 no
  longer stood, and over the ten generations that followed the search's parents
  scored about one task's worth of fitness less than it, so nothing improved it. A
  refinement of that actor stands and rights itself exactly as it does, whatever its
  noise; of 16, one walked 0.033 and 0.028 m s⁻¹ and the other fifteen slower than
  the actor itself (rl#442).
- **No noise on the actions.** A learner that explores by perturbing actions trains
  the perturbed policy, not the scored one, and on this body the perturbation is the
  shaking above: what improves is where the shaking carries the crab, and nothing in
  that asks the policy to move once the shaking stops. Perturbing the weights
  instead, every candidate is a policy `sally-eval` would score as it stands.
- **No gradient.** Around a standing policy the walk scores do not turn over when a
  perturbation does: three perturbations of a standing policy's weights that moved
  the crab −0.0075, −0.0057 and +0.0015 m s⁻¹ toward the task's side had opposites
  that moved it −0.0009, −0.0019 and −0.0001 m s⁻¹. A slope there says little about where a
  stride lies; taking the best of several perturbations climbs regardless.
- **Determinism.** The candidates and the refinements each come from a seeded
  stream and each episode replays bit for bit, so a run replays exactly from its seed, on any
  number of threads. An interrupted run goes on from its log: the actors of the
  generations it finished are drawn again and take the fitnesses the log holds, the
  kept actor and the parent the log ends on are scored on every task to check that
  the log is this search's, and the search continues as if it had never stopped.
- **Cost.** A candidate's four episodes are 2432 ticks and a refinement's two walks
  1280, a generation 29 696; the default budget of 1.25 × 10⁶ ticks covers the first
  actor and 42 generations, about 5.4 simulated hours.

## Running it

`crab_world::policy` holds the readout and the views, `crab_world::train` the
search. `sally-train` writes the kept actor to `policy.json` at the start and
whenever a generation improves it, and to `train.jsonl` its config and the first
actor's episodes, then one line per generation with every candidate's and every
refinement's fitness (`null` for one whose physics diverged) and the kept actor's
episodes. Run again on
an output directory whose `train.jsonl` starts as this run would, it goes on from
there, a last line cut short by a kill dropped. It refuses, until the log is
removed, one that starts otherwise, one with a line it cannot read, a generation
out of place or another count of fitnesses than the config's, one holding more
generations than are asked for, and one whose actors score otherwise.
`sally-eval --policy trained` scores a `policy.json` on the evals.
`sally.sh run` does all of it from the purchased `.blend`: it rigs the crab, trains
for `sally-train`'s default budget with seed 0 unless given a budget in ticks and a
seed, scores the result and both baselines, and writes them side by side to
`scores.md`:

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

Seed 0 at the default budget (`scripts/sally/sally.sh run out`): 42 generations,
1.25 × 10⁶ ticks. Scored by `sally-eval` beside the baselines of `docs/evals.md`:

| task | zero | random | trained |
|---|---|---|---|
| `stand` | 1 | 0.038 (0 to 0.073) | 1 |
| `walk_left` | −9 × 10⁻⁷ m s⁻¹ | −0.0002 m s⁻¹ (−0.0082 to 0.0058) | 0.0563 m s⁻¹ |
| `walk_right` | 9 × 10⁻⁷ m s⁻¹ | 0.0002 m s⁻¹ (−0.0058 to 0.0082) | 0.0679 m s⁻¹ |
| `righting` | 0 | 0 (0 to 0) | 0.91 |

The kept actor stands, walks toward the commanded side at 58% and 70% of the
animal's pace, and turns itself off its back in 0.9 s. After generation 28 the kept
actor walked 0.072 and 0.070 m s⁻¹ and had never righted itself. A candidate of
generation 29 righted itself and replaced it; with that generation's refinements it
walked 0.031 and 0.029 m s⁻¹, and the refinements of the thirteen generations left
brought the walks to the table's speeds, the last of them in the final generation.

## Not in it, by choice

- **Learned clock frequency.** The animal's 1.17 Hz is enough to walk at its routine
  pace; a policy that needs another speed earns a frequency input later.
- **Observation normalization.** Fixed scales from the body's own magnitudes, so the
  weights alone are the policy.
- **Curricula, domain randomization, demonstrations.** Each is a design decision a
  result has to justify first. The tasks are flat ground with one start each.
- **A hidden layer.** Over a walk, two tanh layers of 64 seeded random features are a
  linear function of the view to 99.0% of their variance. A layer joins when a result
  shows that a weighted sum of the view holds a gait back.

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
