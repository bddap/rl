use anyhow::{Result, bail};
use clap::Parser;

#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "DIR")]
    checkpoint: std::path::PathBuf,
}

pub(crate) fn run(args: Args) -> Result<()> {
    use crab_world::play::{CheckpointUnusable, RigDims};
    let RigDims {
        obs: rig_obs,
        action: rig_act,
    } = crab_world::play::rig_dims();
    let dir = args.checkpoint.display();
    match crab_world::play::checkpoint_fits_rig(&args.checkpoint) {
        Ok(()) => {
            println!("checkpoint-check OK: {dir} matches the rig ({rig_obs} obs, {rig_act} act)");
            Ok(())
        }
        Err(CheckpointUnusable::Missing) => bail!(
            "checkpoint-check: no brain.bin in {dir} (this binary's rig is {rig_obs} \
             obs, {rig_act} act) — nothing to ship",
        ),
        Err(CheckpointUnusable::Mismatch(RigDims { obs, action })) => bail!(
            "checkpoint-check MISMATCH: {dir} is {obs} obs, {action} act but this binary's rig \
             is {rig_obs} obs, {rig_act} act — the NN crab would silently hold its rest pose. \
             Retrain/redeploy on the current rig (or rebuild the binary to match the checkpoint).",
        ),
        Err(CheckpointUnusable::Refused(why)) => bail!(
            "checkpoint-check REFUSED: {dir} — {why}. The runtime loader would refuse this \
             checkpoint the same way, so it must not ship.",
        ),
    }
}
