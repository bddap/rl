//! Direct search for the [`Actor`], per `docs/policy.md`: every generation scores
//! perturbed copies of its parent on the evals themselves and the fittest copy
//! parents the next, the fittest actor ever scored is kept, and every generation
//! also scores copies of that kept actor perturbed in its strides alone.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Deserializer, Serialize};

use crate::eval::{Episode, Policy, Task, score};
use crate::policy::{Actor, Anatomy, Driver};
use crate::sally::Skeleton;

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub seed: u64,
    /// Copies of the parent scored each generation, perturbed in every weight.
    pub candidates: usize,
    /// Copies of the kept actor scored each generation, perturbed in its stride
    /// weights alone.
    pub refinements: usize,
    /// Standard deviation of a perturbation.
    pub step: f32,
}

/// The tasks whose episodes the stride weights can change.
fn walks() -> Vec<Task> {
    let walks = |t: &Task| t.walk_sign() != 0.0;
    Task::ALL.into_iter().filter(walks).collect()
}

impl Config {
    /// The generations a budget of simulated ticks covers once the first actor's own
    /// episodes are paid for, counting every episode at its full length.
    pub fn generations(&self, ticks: usize) -> usize {
        let ticks_of = |tasks: &[Task]| tasks.iter().map(|t| t.ticks()).sum::<usize>();
        let (candidate, refinement) = (ticks_of(&Task::ALL), ticks_of(&walks()));
        if self.candidates == 0 {
            return 0;
        }
        ticks.saturating_sub(candidate)
            / (self.candidates * candidate + self.refinements * refinement)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            candidates: 8,
            refinements: 8,
            step: 0.075,
        }
    }
}

/// One actor's episodes, one per task, and their fitness.
#[derive(Debug, Clone, Serialize)]
pub struct Scored {
    fitness: f32,
    episodes: Vec<Episode>,
}

impl Scored {
    /// Every task's score over its reference, summed; an actor whose physics
    /// diverged anywhere is unfit, though the harness scores that episode 0.
    fn new(episodes: Vec<Episode>) -> Self {
        let fitness = if episodes.iter().all(|e| e.finite) {
            episodes.iter().map(|e| e.score / e.task.reference()).sum()
        } else {
            f32::NEG_INFINITY
        };
        Self { fitness, episodes }
    }

    /// These episodes with `others` in the place of those of the same tasks.
    fn with(&self, others: &[Episode]) -> Self {
        let other = |e: &Episode| *others.iter().find(|o| o.task == e.task).unwrap_or(e);
        Self::new(self.episodes.iter().map(other).collect())
    }

    pub fn fitness(&self) -> f32 {
        self.fitness
    }
}

/// A generation's fitnesses, in the order its actors were drawn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fitnesses {
    #[serde(deserialize_with = "null_is_unfit")]
    pub candidates: Vec<f32>,
    #[serde(deserialize_with = "null_is_unfit")]
    pub refinements: Vec<f32>,
}

/// JSON has no infinity: the fitness of an actor whose physics diverged is logged
/// as `null`.
fn null_is_unfit<'de, D: Deserializer<'de>>(logged: D) -> Result<Vec<f32>, D::Error> {
    let logged = Vec::<Option<f32>>::deserialize(logged)?;
    let unfit = |f: Option<f32>| f.unwrap_or(f32::NEG_INFINITY);
    Ok(logged.into_iter().map(unfit).collect())
}

/// What one generation did, for the training log.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub generation: usize,
    #[serde(flatten)]
    pub fitnesses: Fitnesses,
    /// Whether a candidate or a refinement beat the kept actor.
    pub improved: bool,
    pub kept: Scored,
}

impl Report {
    pub fn logged(line: &str, generation: usize) -> Result<Fitnesses, String> {
        #[derive(Deserialize)]
        struct Line {
            generation: usize,
            #[serde(flatten)]
            fitnesses: Fitnesses,
        }
        let line: Line = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if line.generation != generation {
            return Err(format!(
                "generation {} is logged where generation {generation} belongs",
                line.generation
            ));
        }
        Ok(line.fitnesses)
    }
}

/// Uniform in [0, 1).
fn uniform(rng: &mut ChaCha8Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    let u = 1.0 - uniform(rng);
    let v = uniform(rng);
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}

fn perturbed(
    rng: &mut ChaCha8Rng,
    step: f32,
    actor: &Actor,
    moves: impl Fn(usize) -> bool,
    count: usize,
) -> Vec<Actor> {
    let copy = |_| {
        let mut copy = actor.clone();
        for (i, w) in copy.weights.iter_mut().enumerate() {
            if moves(i) {
                *w += step * gaussian(rng);
            }
        }
        copy
    };
    (0..count).map(copy).collect()
}

/// The fittest of the actors whose physics held, if any did.
fn fittest(fitnesses: &[f32]) -> Option<usize> {
    let fit = fitnesses.iter().enumerate().filter(|(_, f)| f.is_finite());
    fit.max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i)
}

pub struct Trainer<'a> {
    skeleton: &'a Skeleton,
    candidates: usize,
    refinements: usize,
    step: f32,
    /// Which of a view's inputs are strides, read by the walks alone.
    strides: Vec<bool>,
    /// The actor the next generation's candidates are drawn around: the first one,
    /// then each generation's fittest candidate, fitter than its own parent or not.
    parent: Actor,
    actor: Actor,
    kept: Scored,
    generation: usize,
    candidate_noise: ChaCha8Rng,
    /// A stream of its own, so that the parents a seed leads through do not depend
    /// on what is refined on the way.
    refinement_noise: ChaCha8Rng,
}

impl<'a> Trainer<'a> {
    pub fn new(skeleton: &'a Skeleton, config: &Config) -> Result<Self, String> {
        if config.candidates == 0 {
            return Err("a generation needs a candidate".into());
        }
        let anatomy = Anatomy::new(skeleton)?;
        let actor = Actor::new(&anatomy);
        let kept = Self::scored(skeleton, std::slice::from_ref(&actor))?.remove(0);
        let mut refinement_noise = ChaCha8Rng::seed_from_u64(config.seed);
        refinement_noise.set_stream(1);
        Ok(Self {
            skeleton,
            candidates: config.candidates,
            refinements: config.refinements,
            step: config.step,
            strides: anatomy.strides(),
            parent: actor.clone(),
            actor,
            kept,
            generation: 0,
            candidate_noise: ChaCha8Rng::seed_from_u64(config.seed),
            refinement_noise,
        })
    }

    /// The fittest actor scored so far.
    pub fn actor(&self) -> &Actor {
        &self.actor
    }

    pub fn kept(&self) -> &Scored {
        &self.kept
    }

    /// Every actor's episodes on `tasks`, in their order: the ones `sally-eval` would
    /// score.
    fn episodes(
        skeleton: &Skeleton,
        actors: &[Actor],
        tasks: &[Task],
    ) -> Result<Vec<Vec<Episode>>, String> {
        if actors.is_empty() {
            return Ok(Vec::new());
        }
        let drivers = actors
            .iter()
            .map(|actor| Driver::new(skeleton, actor.clone()))
            .collect::<Result<Vec<_>, String>>()?;
        let each: Vec<u64> = (0..actors.len() as u64).collect();
        let make = |i: u64| -> Box<dyn Policy> { Box::new(drivers[i as usize].clone()) };
        let scored = score(skeleton, tasks, &each, &make, None)?;
        Ok((0..actors.len())
            .map(|i| scored.iter().map(|t| t.episodes[i]).collect())
            .collect())
    }

    fn scored(skeleton: &Skeleton, actors: &[Actor]) -> Result<Vec<Scored>, String> {
        let episodes = Self::episodes(skeleton, actors, &Task::ALL)?;
        Ok(episodes.into_iter().map(Scored::new).collect())
    }

    fn draw_candidates(&mut self) -> Vec<Actor> {
        let (noise, count) = (&mut self.candidate_noise, self.candidates);
        perturbed(noise, self.step, &self.parent, |_| true, count)
    }

    /// Copies of the kept actor that differ from it in stride weights alone, so on
    /// every task but the walks they do exactly what it does.
    fn draw_refinements(&mut self) -> Vec<Actor> {
        let (noise, count) = (&mut self.refinement_noise, self.refinements);
        let stride = |i| self.strides[i % self.actor.inputs];
        perturbed(noise, self.step, &self.actor, stride, count)
    }

    fn select(
        &mut self,
        mut candidates: Vec<Actor>,
        fitnesses: &[f32],
        kept: f32,
    ) -> Option<usize> {
        let best = fittest(fitnesses)?;
        self.parent = candidates.swap_remove(best);
        (fitnesses[best] > kept).then(|| {
            self.actor = self.parent.clone();
            best
        })
    }

    fn refine(
        &mut self,
        mut refinements: Vec<Actor>,
        fitnesses: &[f32],
        kept: f32,
    ) -> Option<usize> {
        let best = fittest(fitnesses).filter(|&i| fitnesses[i] > kept)?;
        self.actor = refinements.swap_remove(best);
        Some(best)
    }

    pub fn generation(&mut self) -> Result<Report, String> {
        let drawn = self.draw_candidates();
        let mut scored = Self::scored(self.skeleton, &drawn)?;
        let candidates: Vec<f32> = scored.iter().map(Scored::fitness).collect();
        let selected = self.select(drawn, &candidates, self.kept.fitness);
        if let Some(i) = selected {
            self.kept = scored.swap_remove(i);
        }

        let drawn = self.draw_refinements();
        let walked = Self::episodes(self.skeleton, &drawn, &walks())?;
        let mut scored: Vec<Scored> = walked.iter().map(|w| self.kept.with(w)).collect();
        let refinements: Vec<f32> = scored.iter().map(Scored::fitness).collect();
        let refined = self.refine(drawn, &refinements, self.kept.fitness);
        if let Some(i) = refined {
            self.kept = scored.swap_remove(i);
        }

        let report = Report {
            generation: self.generation,
            fitnesses: Fitnesses {
                candidates,
                refinements,
            },
            improved: selected.or(refined).is_some(),
            kept: self.kept.clone(),
        };
        self.generation += 1;
        Ok(report)
    }

    /// Moves on from the generations an interrupted run of this search logged
    /// without simulating them again: its actors are drawn as that run drew them and
    /// take the fitnesses it logged. The kept actor and the parent the log ends on
    /// are then scored on every task, and a log in which either scored otherwise is
    /// refused as another search's.
    pub fn replay(mut self, logged: &[Fitnesses]) -> Result<Self, String> {
        let (mut kept, mut parent) = (self.kept.fitness, None);
        for fitnesses in logged {
            let (candidates, refinements) = (&fitnesses.candidates, &fitnesses.refinements);
            if (candidates.len(), refinements.len()) != (self.candidates, self.refinements) {
                return Err(format!(
                    "generation {} logged {} candidates and {} refinements where {} and {} belong",
                    self.generation,
                    candidates.len(),
                    refinements.len(),
                    self.candidates,
                    self.refinements
                ));
            }
            let drawn = self.draw_candidates();
            if let Some(i) = self.select(drawn, candidates, kept) {
                kept = candidates[i];
            }
            parent = fittest(candidates).map(|i| candidates[i]).or(parent);
            let drawn = self.draw_refinements();
            if let Some(i) = self.refine(drawn, refinements, kept) {
                kept = refinements[i];
            }
            self.generation += 1;
        }
        if logged.is_empty() {
            return Ok(self);
        }
        let both = [self.actor.clone(), self.parent.clone()];
        let mut scored = Self::scored(self.skeleton, &both)?;
        let other = |whose: &str, there: f32, here: f32| {
            format!("the log is another search's: its {whose} scored {there} there and {here} here")
        };
        if scored[0].fitness != kept {
            return Err(other("kept actor", kept, scored[0].fitness));
        }
        match parent {
            Some(parent) if scored[1].fitness != parent => {
                return Err(other("last parent", parent, scored[1].fitness));
            }
            _ => {}
        }
        self.kept = scored.swap_remove(0);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::sally::test_rig::biped;

    const CONFIG: Config = Config {
        seed: 1,
        candidates: 3,
        refinements: 2,
        step: 1.0,
    };

    struct Run {
        start: Actor,
        first: Scored,
        /// Each generation's report, then the kept actor and the parent it left.
        generations: Vec<(Report, Actor, Actor)>,
    }

    impl Run {
        fn logged(&self) -> Vec<Fitnesses> {
            let fitnesses = |(report, ..): &(Report, Actor, Actor)| report.fitnesses.clone();
            self.generations.iter().map(fitnesses).collect()
        }

        /// The kept actor's fitness as generation `g` began.
        fn kept_before(&self, g: usize) -> f32 {
            match g.checked_sub(1) {
                Some(before) => self.generations[before].0.kept.fitness(),
                None => self.first.fitness(),
            }
        }

        /// The kept actor's fitness as generation `g`'s refinements were drawn.
        fn kept_between(&self, g: usize) -> f32 {
            let candidates = &self.generations[g].0.fitnesses.candidates;
            let best = fittest(candidates).map_or(f32::NEG_INFINITY, |i| candidates[i]);
            best.max(self.kept_before(g))
        }
    }

    /// The first generations from [`CONFIG`], shared: every candidate scored costs
    /// four episodes and every refinement two.
    fn run() -> &'static Run {
        static RUN: OnceLock<Run> = OnceLock::new();
        RUN.get_or_init(|| {
            let skeleton = biped();
            let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
            let (start, first) = (trainer.actor().clone(), trainer.kept().clone());
            let generations = (0..3)
                .map(|_| {
                    let report = trainer.generation().unwrap();
                    (report, trainer.actor.clone(), trainer.parent.clone())
                })
                .collect();
            Run {
                start,
                first,
                generations,
            }
        })
    }

    #[test]
    fn a_run_replays_exactly_from_its_seed() {
        let skeleton = biped();
        let (report, kept, _) = &run().generations[0];
        let mut again = Trainer::new(&skeleton, &CONFIG).unwrap();
        assert_eq!(again.generation().unwrap().fitnesses, report.fitnesses);
        assert_eq!(again.actor(), kept);
        let other = Config { seed: 6, ..CONFIG };
        let mut other = Trainer::new(&skeleton, &other).unwrap();
        let other = other.generation().unwrap().fitnesses;
        assert_ne!(other.candidates, report.fitnesses.candidates);
        assert_ne!(other.refinements, report.fitnesses.refinements);
    }

    #[test]
    fn the_candidates_a_seed_draws_do_not_depend_on_what_is_refined() {
        let skeleton = biped();
        let mut refining = Trainer::new(&skeleton, &CONFIG).unwrap();
        assert_ne!(
            refining.candidate_noise.get_stream(),
            refining.refinement_noise.get_stream()
        );
        let mut plain = Trainer::new(&skeleton, &CONFIG).unwrap();
        let first = refining.draw_candidates();
        assert!(first == plain.draw_candidates());
        let refinements = refining.draw_refinements();
        assert!(refining.draw_candidates() == plain.draw_candidates());
        assert!(refinements != refining.draw_refinements());
        assert!(first[..CONFIG.refinements] != refinements[..]);
    }

    #[test]
    fn the_fittest_candidate_parents_the_next_generation_and_is_kept_only_if_it_beats_the_kept() {
        assert_eq!(fittest(&[0.5, 1.5, 1.2]), Some(1));
        assert_eq!(fittest(&[f32::NEG_INFINITY, -0.5]), Some(1));
        assert_eq!(fittest(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), None);

        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
        let start = trainer.actor().clone();
        let drawn = trainer.draw_candidates();
        assert_eq!(trainer.select(drawn.clone(), &[0.5, 1.5, 1.2], 1.5), None);
        assert!(trainer.parent == drawn[1] && trainer.actor == start);
        assert_eq!(
            trainer.select(drawn.clone(), &[0.5, 1.2, 1.6], 1.5),
            Some(2)
        );
        assert!(trainer.parent == drawn[2] && trainer.actor == drawn[2]);
        let unfit = [f32::NEG_INFINITY; 3];
        assert_eq!(
            trainer.select(drawn.clone(), &unfit, f32::NEG_INFINITY),
            None
        );
        assert!(trainer.parent == drawn[2] && trainer.actor == drawn[2]);

        let next = trainer.draw_candidates().remove(0);
        let rms_from = |from: &Actor| {
            let squares = next
                .weights
                .iter()
                .zip(&from.weights)
                .map(|(n, f)| (n - f).powi(2));
            (squares.sum::<f32>() / next.weights.len() as f32).sqrt()
        };
        assert!(rms_from(&drawn[2]) < rms_from(&start));
        let step = rms_from(&drawn[2]);
        assert!((step / CONFIG.step - 1.0).abs() < 0.2, "{step}");
    }

    #[test]
    fn the_fittest_refinement_is_kept_only_if_it_beats_the_kept_and_parents_nothing() {
        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
        let start = trainer.actor().clone();
        let drawn = trainer.draw_refinements();
        assert_eq!(trainer.refine(drawn.clone(), &[1.5, 1.2], 1.5), None);
        assert_eq!(
            trainer.refine(drawn.clone(), &[f32::NEG_INFINITY; 2], 0.0),
            None
        );
        assert!(trainer.actor == start);
        assert_eq!(trainer.refine(drawn.clone(), &[1.2, 1.6], 1.5), Some(1));
        assert!(trainer.actor == drawn[1] && trainer.parent == start);
    }

    #[test]
    fn a_generation_is_scored_as_its_report_says() {
        let skeleton = biped();
        let run = run();
        let mut was = &run.start;
        for (g, (report, kept, parent)) in run.generations.iter().enumerate() {
            assert_eq!(report.generation, g);
            let Fitnesses {
                candidates,
                refinements,
            } = &report.fitnesses;
            assert_eq!(candidates.len(), CONFIG.candidates);
            assert_eq!(refinements.len(), CONFIG.refinements);
            let best = candidates.iter().chain(refinements);
            let best = best.copied().fold(run.kept_before(g), f32::max);
            assert_eq!(report.kept.fitness(), best);
            assert_eq!(report.improved, best > run.kept_before(g));
            assert_eq!(report.improved, kept != was);
            let scored = Trainer::scored(&skeleton, &[kept.clone(), parent.clone()]).unwrap();
            assert_eq!(scored[0].fitness(), report.kept.fitness());
            let fit = fittest(candidates).expect("a candidate whose physics held");
            assert_eq!(scored[1].fitness(), candidates[fit]);
            was = kept;
        }
        assert!(run.generations[0].0.improved);
    }

    #[test]
    fn a_replayed_log_leaves_the_search_where_its_run_was() {
        let skeleton = biped();
        let run = run();
        let logged = run.logged();
        let resume =
            |logged: &[Fitnesses]| Trainer::new(&skeleton, &CONFIG).unwrap().replay(logged);
        let (last, kept, parent) = run.generations.last().unwrap();

        let whole = resume(&logged).unwrap();
        assert!(whole.actor() == kept);
        assert!(&whole.parent == parent);
        assert_eq!(whole.kept().fitness(), last.kept.fitness());
        assert_eq!(whole.generation, logged.len());

        let mut part = resume(&logged[..logged.len() - 1]).unwrap();
        let next = part.generation().unwrap();
        assert_eq!(next.generation, last.generation);
        assert_eq!(next.fitnesses, last.fitnesses);
        assert_eq!(next.improved, last.improved);
        assert_eq!(next.kept.fitness(), last.kept.fitness());
        assert!(part.actor() == kept);

        let none = resume(&[]).unwrap();
        assert!(none.actor() == &run.start && none.parent == run.start);
        assert_eq!(none.generation, 0);

        let mut low = Trainer::new(&skeleton, &CONFIG).unwrap();
        low.kept.fitness = f32::MIN;
        let low = low.replay(&logged).unwrap();
        assert!(low.actor() == kept);

        let (first, kept, parent) = &run.generations[0];
        let unfit = Fitnesses {
            candidates: vec![f32::NEG_INFINITY; CONFIG.candidates],
            refinements: vec![f32::NEG_INFINITY; CONFIG.refinements],
        };
        let after = resume(&[logged[0].clone(), unfit]).unwrap();
        assert!(after.actor() == kept && &after.parent == parent);
        assert_eq!(after.kept().fitness(), first.kept.fitness());
        assert_eq!(after.generation, 2);
    }

    #[test]
    fn a_log_of_another_search_is_refused() {
        let skeleton = biped();
        let run = run();
        let logged = run.logged();
        let resume =
            |logged: &[Fitnesses]| Trainer::new(&skeleton, &CONFIG).unwrap().replay(logged);
        let another = |whose: &str, result: Result<Trainer, String>| {
            let refused = result.err().expect("refused");
            let said = format!("another search's: its {whose} scored");
            assert!(refused.contains(&said), "{refused}");
        };
        let best = |fitnesses: &[f32]| fittest(fitnesses).map(|i| fitnesses[i]);

        let kept_by_a_candidate = (0..logged.len())
            .find(|&g| best(&logged[g].candidates) > Some(run.kept_before(g)))
            .expect("a generation whose fittest candidate was kept");
        let mut raised = logged.clone();
        let at = fittest(&raised[kept_by_a_candidate].candidates).unwrap();
        raised[kept_by_a_candidate].candidates[at] += 8.0;
        another("kept actor", resume(&raised));

        let parent_only = (0..logged.len())
            .find(|&g| best(&logged[g].candidates).is_some_and(|b| b <= run.kept_before(g)))
            .expect("a generation whose fittest candidate was not kept");
        let mut other = logged[..=parent_only].to_vec();
        other.last_mut().unwrap().candidates.rotate_left(1);
        another("last parent", resume(&other));

        let kept_by_a_refinement = (0..logged.len())
            .find(|&g| best(&logged[g].refinements) > Some(run.kept_between(g)))
            .expect("a generation whose fittest refinement was kept");
        let mut other = logged[..=kept_by_a_refinement].to_vec();
        other.last_mut().unwrap().refinements.rotate_left(1);
        another("kept actor", resume(&other));

        let mut short = logged[..1].to_vec();
        short[0].refinements.pop();
        let short = resume(&short).err().expect("refused");
        assert!(short.contains("3 candidates and 1 refinements"), "{short}");
        let mut long = logged[..1].to_vec();
        long[0].candidates.push(9.0);
        let long = resume(&long).err().expect("refused");
        assert!(long.contains("4 candidates and 2 refinements"), "{long}");
    }

    #[test]
    fn a_logged_generation_reads_back_its_fitnesses() {
        let report = Report {
            generation: 4,
            fitnesses: Fitnesses {
                candidates: vec![1.25, f32::NEG_INFINITY, -0.1],
                refinements: vec![f32::NEG_INFINITY, 2.5],
            },
            improved: false,
            kept: Scored::new(vec![episode(Task::Stand, 1.0)]),
        };
        let line = serde_json::to_string(&report).unwrap();
        assert_eq!(line.matches("null").count(), 2, "{line}");
        assert_eq!(Report::logged(&line, 4).unwrap(), report.fitnesses);
        let misplaced = Report::logged(&line, 3).unwrap_err();
        assert!(misplaced.contains("generation 4"), "{misplaced}");
        assert!(Report::logged(&line[..line.len() - 2], 4).is_err());
    }

    #[test]
    fn every_actor_keeps_its_own_score() {
        let skeleton = biped();
        let run = run();
        let (report, kept, _) = &run.generations[0];
        let both = Trainer::scored(&skeleton, &[run.start.clone(), kept.clone()]).unwrap();
        assert_eq!(both[0].fitness(), run.first.fitness());
        assert_eq!(both[1].fitness(), report.kept.fitness());
        assert_ne!(both[0].fitness(), both[1].fitness());
        assert!(Trainer::scored(&skeleton, &[]).unwrap().is_empty());
    }

    #[test]
    fn a_candidate_differs_from_its_parent_in_every_weight() {
        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
        let candidates = trainer.draw_candidates();
        assert_eq!(candidates.len(), CONFIG.candidates);
        let parent = &trainer.parent.weights;
        for candidate in &candidates {
            assert_eq!(candidate.weights.len(), parent.len());
            assert!(candidate.weights.iter().zip(parent).all(|(c, p)| c != p));
        }
        assert!(candidates[0] != candidates[1]);

        let quarter = Config {
            step: 0.25,
            ..CONFIG
        };
        let mut quarter = Trainer::new(&skeleton, &quarter).unwrap();
        let scaled = |(w, q): (&f32, &f32)| *w != 0.0 && *q == 0.25 * w;
        let quarter = quarter.draw_candidates().remove(0);
        assert!(
            candidates[0]
                .weights
                .iter()
                .zip(&quarter.weights)
                .all(scaled)
        );
    }

    #[test]
    fn a_refinement_differs_from_the_kept_actor_in_every_stride_weight_and_no_other() {
        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
        trainer.actor = run().generations[0].1.clone();
        let kept = trainer.actor.clone();
        assert!(kept != trainer.parent);
        let strides = Anatomy::new(&skeleton).unwrap().strides();
        assert!(strides.contains(&true) && strides.contains(&false));
        let refinements = trainer.draw_refinements();
        assert_eq!(refinements.len(), CONFIG.refinements);
        for refinement in &refinements {
            for (i, (r, k)) in refinement.weights.iter().zip(&kept.weights).enumerate() {
                assert_eq!(r != k, strides[i % kept.inputs], "weight {i}");
            }
        }
        assert!(refinements[0] != refinements[1]);
    }

    #[test]
    fn a_refinement_does_what_the_kept_actor_does_on_every_task_but_the_walks() {
        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
        trainer.actor = run().generations[0].1.clone();
        let refinement = trainer.draw_refinements().remove(0);
        let both = [trainer.actor.clone(), refinement.clone()];
        let both = Trainer::scored(&skeleton, &both).unwrap();
        let bits = |e: &Episode| {
            let of = [e.score, e.supported_fraction, e.mean_body_height_m];
            (e.finite, of.map(f32::to_bits))
        };
        let mut walked = Vec::new();
        for (kept, refined) in both[0].episodes.iter().zip(&both[1].episodes) {
            assert_eq!(kept.task, refined.task);
            if walks().contains(&kept.task) {
                assert_ne!(bits(kept), bits(refined), "{:?}", kept.task);
                walked.push(*refined);
            } else {
                assert_eq!(bits(kept), bits(refined), "{:?}", kept.task);
            }
        }
        assert_eq!(walked.len(), 2);
        let on_the_walks = Trainer::episodes(&skeleton, &[refinement], &walks()).unwrap();
        let merged = both[0].with(&on_the_walks[0]);
        assert_eq!(merged.fitness().to_bits(), both[1].fitness().to_bits());
        assert_ne!(merged.fitness(), both[0].fitness());
    }

    fn episode(task: Task, score: f32) -> Episode {
        Episode {
            task,
            score,
            finite: true,
            supported_fraction: 1.0,
            mean_body_height_m: 0.03,
        }
    }

    #[test]
    fn fitness_counts_every_task_against_its_reference() {
        let pace = Task::WalkLeft.reference();
        let scored = Scored::new(vec![
            episode(Task::Stand, 1.0),
            episode(Task::WalkLeft, pace),
            episode(Task::WalkRight, 0.5 * pace),
            episode(Task::Righting, 0.25),
        ]);
        assert!(
            (scored.fitness() - 2.75).abs() < 1e-6,
            "{}",
            scored.fitness()
        );
        let faster = scored.with(&[episode(Task::WalkRight, pace)]);
        assert!(
            (faster.fitness() - 3.25).abs() < 1e-6,
            "{}",
            faster.fitness()
        );
        let tasks = |s: &Scored| s.episodes.iter().map(|e| e.task).collect::<Vec<_>>();
        assert_eq!(tasks(&faster), tasks(&scored));
    }

    #[test]
    fn an_actor_whose_physics_diverged_is_unfit() {
        let mut diverged = episode(Task::WalkLeft, 0.0);
        diverged.finite = false;
        let scored = Scored::new(vec![episode(Task::Stand, 1.0), diverged]);
        assert_eq!(scored.fitness(), f32::NEG_INFINITY);
        let held = Scored::new(vec![
            episode(Task::Stand, 1.0),
            episode(Task::WalkLeft, 0.0),
        ]);
        assert_eq!(held.with(&[diverged]).fitness(), f32::NEG_INFINITY);
    }

    #[test]
    fn the_first_actor_is_scored_as_the_evals_score_it() {
        let skeleton = biped();
        let run = run();
        assert_eq!(run.first.episodes.len(), Task::ALL.len());
        for episode in &run.first.episodes {
            let mut driver = Driver::new(&skeleton, run.start.clone()).unwrap();
            let again = crate::eval::run(&skeleton, episode.task, &mut driver).unwrap();
            assert_eq!(episode.score.to_bits(), again.score.to_bits());
        }
    }

    #[test]
    fn a_budget_buys_whole_generations_after_the_first_actor() {
        let per_actor: usize = Task::ALL.iter().map(|t| t.ticks()).sum();
        let on_the_walks = Task::WalkLeft.ticks() + Task::WalkRight.ticks();
        assert_eq!(walks(), [Task::WalkLeft, Task::WalkRight]);
        let generation = 3 * per_actor + 2 * on_the_walks;
        assert_eq!(CONFIG.generations(per_actor + generation - 1), 0);
        assert_eq!(CONFIG.generations(per_actor + generation), 1);
        assert_eq!(CONFIG.generations(per_actor + 3 * generation - 1), 2);
        assert_eq!(CONFIG.generations(0), 0);
        let none = Config {
            candidates: 0,
            ..CONFIG
        };
        assert_eq!(none.generations(usize::MAX), 0);
        let unrefined = Config {
            refinements: 0,
            ..CONFIG
        };
        assert_eq!(unrefined.generations(4 * per_actor), 1);
        assert_eq!(Config::default().generations(1_250_000), 42);
    }
}
