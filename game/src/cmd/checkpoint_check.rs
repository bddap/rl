//! `checkpoint-check`: rig-compatibility gate for the release/deploy pipeline.

use anyhow::{Result, bail};
use clap::Parser;

#[derive(Parser)]
pub(crate) struct Args {
    /// Checkpoint dir holding the `brain.bin` to rig-check against this binary. Required —
    /// the gate names exactly the checkpoint it's about to ship, no implicit default.
    #[arg(long, value_name = "DIR")]
    checkpoint: std::path::PathBuf,
}

/// Rig-compatibility gate (see [`crate::cmd::Command::CheckpointCheck`]): does the checkpoint at
/// `--checkpoint` fit THIS binary's crab rig? crab-world owns the verdict
/// ([`crab_world::play::checkpoint_fits_rig`]) and the rig spec
/// ([`crab_world::play::rig_dims`]), so the binary answers for itself — no hand-kept number
/// to drift; here we only turn the verdict into a message + exit code. Any non-`Ok` verdict
/// is an error (→ nonzero exit), which the release builder treats as "do not publish this
/// checkpoint".
pub(crate) fn run(args: Args) -> Result<()> {
    use crab_world::play::{RigDims, RigFit};
    let RigDims {
        obs: rig_obs,
        action: rig_act,
    } = crab_world::play::rig_dims();
    let dir = args.checkpoint.display();
    match crab_world::play::checkpoint_fits_rig(&args.checkpoint) {
        RigFit::Ok => {
            println!("checkpoint-check OK: {dir} matches the rig ({rig_obs} obs, {rig_act} act)");
            Ok(())
        }
        RigFit::Missing => bail!(
            "checkpoint-check: no brain.bin in {dir} (this binary's rig is {rig_obs} \
             obs, {rig_act} act) — nothing to ship",
        ),
        RigFit::Unreadable => bail!(
            "checkpoint-check: brain.bin in {dir} exists but won't deserialize (truncated or \
             corrupt) — do not ship it",
        ),
        RigFit::Mismatch(RigDims { obs, action }) => bail!(
            "checkpoint-check MISMATCH: {dir} is {obs} obs, {action} act but this binary's rig \
             is {rig_obs} obs, {rig_act} act — the NN crab would silently hold its rest pose. \
             Retrain/redeploy on the current rig (or rebuild the binary to match the checkpoint).",
        ),
    }
}
