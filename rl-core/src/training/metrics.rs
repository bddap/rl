//! Per-episode CSV metrics for the training run. The learner's host writes to "tmp"
//! (where the plotting scripts read); each rollout thread writes to its own scratch dir
//! so K threads never clobber one shared CSV.

use std::io::Write;
use std::path::Path;

pub(crate) struct MetricsLogger {
    episode_file: std::fs::File,
}

impl MetricsLogger {
    /// `dir` is where `episodes.csv` lands. A rollout thread passes its own scratch
    /// dir so K threads don't clobber one shared CSV; the learner's host uses "tmp"
    /// (the established location the plotting scripts read).
    pub(crate) fn new(dir: &Path) -> Self {
        std::fs::create_dir_all(dir).expect("failed to create metrics dir");

        let ep_path = dir.join("episodes.csv");
        let mut episode_file =
            std::fs::File::create(&ep_path).expect("failed to create episodes.csv");
        writeln!(
            episode_file,
            "episode,reward,steps,avg_reward_10,mean_height,mean_upright,mean_sq_angvel"
        )
        .expect("failed to write header");

        Self { episode_file }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn log_episode(
        &mut self,
        episode: u32,
        reward: f32,
        steps: u32,
        avg_reward: f32,
        mean_height: f32,
        mean_upright: f32,
        mean_sq_angvel: f32,
    ) {
        writeln!(
            self.episode_file,
            "{},{:.4},{},{},{:.4},{:.4},{:.4}",
            episode, reward, steps, avg_reward, mean_height, mean_upright, mean_sq_angvel
        )
        .ok();
        if episode.is_multiple_of(10) {
            self.episode_file.flush().ok();
        }
    }
}
