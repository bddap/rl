//! Direct search for the [`Actor`], per `docs/policy.md`: every generation scores
//! perturbed copies of its parent on the evals themselves, the fittest copy parents
//! the next, and the fittest actor ever scored is kept.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use serde::Serialize;

use crate::eval::{Episode, Policy, Task, score};
use crate::policy::{Actor, Anatomy, Driver, uniform};
use crate::sally::Skeleton;

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub seed: u64,
    /// Perturbed copies scored each generation.
    pub candidates: usize,
    /// Standard deviation of a perturbation.
    pub step: f32,
}

impl Config {
    /// The generations a budget of simulated ticks covers once the first actor's own
    /// episodes are paid for, counting every episode at its full length.
    pub fn generations(&self, ticks: usize) -> usize {
        let per_actor: usize = Task::ALL.iter().map(|t| t.ticks()).sum();
        ticks
            .saturating_sub(per_actor)
            .checked_div(self.candidates * per_actor)
            .unwrap_or(0)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            candidates: 8,
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

    pub fn fitness(&self) -> f32 {
        self.fitness
    }
}

/// What one generation did, for the training log.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub generation: usize,
    /// Each candidate's fitness, in the order drawn.
    pub fitnesses: Vec<f32>,
    /// Whether its fittest candidate beat the kept actor.
    pub improved: bool,
    pub kept: Scored,
}

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    let u = 1.0 - uniform(rng);
    let v = uniform(rng);
    (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
}

/// The fittest of the candidates whose physics held, if any did.
fn fittest(fitnesses: &[f32]) -> Option<usize> {
    let fit = fitnesses.iter().enumerate().filter(|(_, f)| f.is_finite());
    fit.max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i)
}

pub struct Trainer<'a> {
    skeleton: &'a Skeleton,
    candidates: usize,
    step: f32,
    /// The actor the next generation's candidates are drawn around: the seeded one,
    /// then each generation's fittest, fitter than its own parent or not.
    parent: Actor,
    actor: Actor,
    kept: Scored,
    generation: usize,
    rng: ChaCha8Rng,
}

impl<'a> Trainer<'a> {
    /// A seeded random actor for `skeleton`, scored.
    pub fn new(skeleton: &'a Skeleton, config: &Config) -> Result<Self, String> {
        if config.candidates == 0 {
            return Err("a generation needs a candidate".into());
        }
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let actor = Actor::new(&Anatomy::new(skeleton)?, &mut rng);
        let kept = Self::evaluate(skeleton, std::slice::from_ref(&actor))?.remove(0);
        Ok(Self {
            skeleton,
            candidates: config.candidates,
            step: config.step,
            parent: actor.clone(),
            actor,
            kept,
            generation: 0,
            rng,
        })
    }

    /// The fittest actor scored so far.
    pub fn actor(&self) -> &Actor {
        &self.actor
    }

    pub fn kept(&self) -> &Scored {
        &self.kept
    }

    /// Every actor's episode on every task: the ones `sally-eval` would score.
    fn evaluate(skeleton: &Skeleton, actors: &[Actor]) -> Result<Vec<Scored>, String> {
        let drivers = actors
            .iter()
            .map(|actor| Driver::new(skeleton, actor.clone()))
            .collect::<Result<Vec<_>, String>>()?;
        let each: Vec<u64> = (0..actors.len() as u64).collect();
        let make = |i: u64| -> Box<dyn Policy> { Box::new(drivers[i as usize].clone()) };
        let tasks = score(skeleton, &Task::ALL, &each, &make, None)?;
        Ok((0..actors.len())
            .map(|i| Scored::new(tasks.iter().map(|t| t.episodes[i]).collect()))
            .collect())
    }

    fn perturbed(&mut self) -> Actor {
        let mut actor = self.parent.clone();
        for p in actor.last_layer_mut() {
            *p += self.step * gaussian(&mut self.rng);
        }
        actor
    }

    /// Scores one generation of candidates. The fittest becomes the next parent and,
    /// if it beats the kept actor, the kept actor.
    pub fn generation(&mut self) -> Result<Report, String> {
        let mut candidates: Vec<Actor> = (0..self.candidates).map(|_| self.perturbed()).collect();
        let mut scored = Self::evaluate(self.skeleton, &candidates)?;
        let fitnesses: Vec<f32> = scored.iter().map(Scored::fitness).collect();
        let best = fittest(&fitnesses);
        let improved = best.is_some_and(|i| fitnesses[i] > self.kept.fitness);
        if let Some(i) = best {
            self.parent = candidates.swap_remove(i);
            if improved {
                self.actor = self.parent.clone();
                self.kept = scored.swap_remove(i);
            }
        }
        let report = Report {
            generation: self.generation,
            fitnesses,
            improved,
            kept: self.kept.clone(),
        };
        self.generation += 1;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::sally::test_rig::biped;

    const CONFIG: Config = Config {
        seed: 0,
        candidates: 3,
        step: 1.0,
    };

    struct First {
        scored: Scored,
        actor: Actor,
        report: Report,
        kept: Actor,
        parent: Actor,
    }

    /// One generation from [`CONFIG`], shared: every actor scored costs four episodes.
    fn first_generation() -> &'static First {
        static RUN: OnceLock<First> = OnceLock::new();
        RUN.get_or_init(|| {
            let skeleton = biped();
            let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
            let (scored, actor) = (trainer.kept().clone(), trainer.actor().clone());
            let report = trainer.generation().unwrap();
            First {
                scored,
                actor,
                report,
                kept: trainer.actor,
                parent: trainer.parent,
            }
        })
    }

    #[test]
    fn a_run_replays_exactly_from_its_seed() {
        let skeleton = biped();
        let first = first_generation();
        let mut again = Trainer::new(&skeleton, &CONFIG).unwrap();
        assert_eq!(again.actor().params, first.actor.params);
        assert_eq!(
            again.generation().unwrap().fitnesses,
            first.report.fitnesses
        );
        let other = Config { seed: 6, ..CONFIG };
        let mut other = Trainer::new(&skeleton, &other).unwrap();
        assert_ne!(other.actor().params, first.actor.params);
        assert_ne!(
            other.generation().unwrap().fitnesses,
            first.report.fitnesses
        );
    }

    #[test]
    fn the_fittest_candidate_parents_the_next_generation_and_is_kept_only_if_it_beats_the_kept() {
        assert_eq!(fittest(&[0.5, 1.5, 1.2]), Some(1));
        assert_eq!(fittest(&[f32::NEG_INFINITY, -0.5]), Some(1));
        assert_eq!(fittest(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), None);

        let first = first_generation();
        let best = fittest(&first.report.fitnesses).unwrap();
        assert!(first.report.fitnesses[best] > first.scored.fitness());
        assert!(first.report.improved);
        assert_eq!(first.report.kept.fitness(), first.report.fitnesses[best]);
        assert!(first.kept.params != first.actor.params);
        assert!(first.parent.params == first.kept.params);

        let skeleton = biped();
        let mut tied = Trainer::new(&skeleton, &CONFIG).unwrap();
        tied.kept.fitness = first.report.fitnesses[best];
        let report = tied.generation().unwrap();
        assert_eq!(report.fitnesses, first.report.fitnesses);
        assert!(!report.improved);
        let scores = |s: &Scored| s.episodes.iter().map(|e| e.score).collect::<Vec<_>>();
        assert_eq!(scores(&report.kept), scores(&first.scored));
        assert!(tied.actor().params == first.actor.params);
        assert!(tied.parent.params == first.parent.params);

        let next = tied.perturbed();
        let rms_from = |from: &Actor| {
            let squares = next
                .params
                .iter()
                .zip(&from.params)
                .map(|(n, f)| (n - f).powi(2));
            (squares.sum::<f32>() / next.clone().last_layer_mut().len() as f32).sqrt()
        };
        assert!(rms_from(&first.parent) < rms_from(&first.actor));
        let step = rms_from(&first.parent);
        assert!((step / CONFIG.step - 1.0).abs() < 0.2, "{step}");
    }

    #[test]
    fn every_actor_keeps_its_own_score() {
        let skeleton = biped();
        let first = first_generation();
        let both = [first.actor.clone(), first.kept.clone()];
        let both = Trainer::evaluate(&skeleton, &both).unwrap();
        assert_eq!(both[0].fitness(), first.scored.fitness());
        assert_eq!(both[1].fitness(), first.report.kept.fitness());
        assert_ne!(both[0].fitness(), both[1].fitness());
    }

    #[test]
    fn a_candidate_differs_from_its_parent_in_its_last_layer_only() {
        let skeleton = biped();
        let mut trainer = Trainer::new(&skeleton, &CONFIG).unwrap();
        let mut parent = trainer.parent.clone();
        let mut candidate = trainer.perturbed();
        let under = parent.params.len() - parent.last_layer_mut().len();
        assert!(parent.params[..under] == candidate.params[..under]);
        let moved = parent
            .last_layer_mut()
            .iter()
            .zip(candidate.last_layer_mut())
            .filter(|(p, c)| p != c)
            .count();
        assert_eq!(moved, parent.last_layer_mut().len());
        assert!(trainer.perturbed().params != candidate.params);
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
    }

    #[test]
    fn an_actor_whose_physics_diverged_is_unfit() {
        let mut diverged = episode(Task::WalkLeft, 0.0);
        diverged.finite = false;
        let scored = Scored::new(vec![episode(Task::Stand, 1.0), diverged]);
        assert_eq!(scored.fitness(), f32::NEG_INFINITY);
    }

    #[test]
    fn the_first_actor_is_scored_as_the_evals_score_it() {
        let skeleton = biped();
        let first = first_generation();
        assert_eq!(first.scored.episodes.len(), Task::ALL.len());
        for episode in &first.scored.episodes {
            let mut driver = Driver::new(&skeleton, first.actor.clone()).unwrap();
            let again = crate::eval::run(&skeleton, episode.task, &mut driver).unwrap();
            assert_eq!(episode.score.to_bits(), again.score.to_bits());
        }
    }

    #[test]
    fn a_budget_buys_whole_generations_after_the_first_actor() {
        let per_actor: usize = Task::ALL.iter().map(|t| t.ticks()).sum();
        let generation = 3 * per_actor;
        assert_eq!(CONFIG.generations(per_actor + generation - 1), 0);
        assert_eq!(CONFIG.generations(per_actor + generation), 1);
        assert_eq!(CONFIG.generations(per_actor + 3 * generation - 1), 2);
        assert_eq!(CONFIG.generations(0), 0);
        let none = Config {
            candidates: 0,
            ..CONFIG
        };
        assert_eq!(none.generations(usize::MAX), 0);
        assert_eq!(Config::default().generations(1_250_000), 64);
    }
}
