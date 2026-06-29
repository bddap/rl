//! `nn-crab-xpeer`: the decisive GCR #82 cross-peer NN-crab determinism gate.

use anyhow::Result;
use clap::Parser;

use super::shared::{nn_crab_checkpoint_dir, write_tick_hash_log};

#[derive(Parser)]
pub(crate) struct Args {
    /// Trained crab checkpoint dir (`brain.bin` + `normalizer.bin`). Same resolution as
    /// `nn-crab-probe` / `play --nn-crab-checkpoint`.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<std::path::PathBuf>,
    /// Sim ticks to run both peers for before comparing.
    #[arg(long, default_value_t = 600)]
    ticks: u64,
    /// Shared match seed (identical on both peers, as it would be on the wire).
    #[arg(long, default_value_t = 0x6372_6162)]
    seed: u64,
    /// Write peer A's per-tick `<tick> <state_hash>` log here. With B's log, `diff` them: a
    /// byte-identical pair is the cross-peer determinism proof.
    #[arg(long, value_name = "FILE", default_value = "xpeer_a.log")]
    hash_log_a: std::path::PathBuf,
    /// Write peer B's per-tick `<tick> <state_hash>` log here.
    #[arg(long, value_name = "FILE", default_value = "xpeer_b.log")]
    hash_log_b: std::path::PathBuf,
}

/// The cross-peer NN-crab determinism gate (GCR #82): run the real rapier-NN crab on two
/// independent in-process peers exchanging lockstep inputs, write each peer's per-tick hash log,
/// and confirm they stayed byte-identical. Exits nonzero on any divergence so it doubles as a CI
/// gate on "the NN crab is the deterministic multiplayer crab".
pub(crate) fn run(args: Args) -> Result<()> {
    use net::external_crab::run_cross_peer_probe;

    let dir = nn_crab_checkpoint_dir(args.checkpoint)?;
    println!("nn-crab-xpeer: checkpoint={}", dir.display());
    println!("nn-crab-xpeer: seed={:#x} ticks={}", args.seed, args.ticks);

    let result = run_cross_peer_probe(&dir, args.seed, args.ticks);
    if result.ticks.is_empty() {
        anyhow::bail!("nn-crab-xpeer: no ticks applied — the peers never advanced");
    }

    // Per-tick `<tick> <hash>` logs for each peer, so an operator can `diff` them directly.
    write_tick_hash_log(
        &args.hash_log_a,
        result.ticks.iter().map(|t| (t.tick, t.hash_a)),
    )?;
    write_tick_hash_log(
        &args.hash_log_b,
        result.ticks.iter().map(|t| (t.tick, t.hash_b)),
    )?;
    println!(
        "nn-crab-xpeer: wrote {} per-tick hashes per peer to {} / {}",
        result.ticks.len(),
        args.hash_log_a.display(),
        args.hash_log_b.display(),
    );

    println!(
        "nn-crab-xpeer: lockstep desync faults = {} (the peers' own cross-check)",
        result.faults
    );
    match result.first_divergence() {
        None => {
            let last = result.ticks.last().unwrap();
            println!(
                "nn-crab-xpeer: per-tick hashes IDENTICAL across both peers \
                 (final tick {} hash {:#018x})",
                last.tick, last.hash_a
            );
        }
        Some(d) => {
            println!(
                "nn-crab-xpeer: FIRST DIVERGENCE at tick {} — A={:#018x} B={:#018x}",
                d.tick, d.hash_a, d.hash_b
            );
        }
    }

    if result.is_deterministic() {
        println!(
            "nn-crab-xpeer: PASS — the trained NN crab is the deterministic multiplayer crab \
             (outcome 3: bit-identical across peers, 0 desyncs)"
        );
        Ok(())
    } else {
        anyhow::bail!(
            "nn-crab-xpeer: FAIL — the float NN crab DIVERGED across peers (outcome 4: \
             netcode-rethink trigger; the diverging hash logs are the evidence)"
        )
    }
}
