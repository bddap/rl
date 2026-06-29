//! `nn-crab-join-xpeer`: the GCR MP Stage 3 mid-game-JOIN armed-Sally determinism PROBE (rl#151).
//!
//! FINDING (this probe's reason to exist): it measures whether the round-boundary join keeps the
//! armed rapier crab bit-identical across a mid-game join, and DEMONSTRATES THAT IT DOES NOT under
//! the current `cold_respawn_armed_crab` (75710b61). Despawning + respawning the crab ENTITIES does
//! not reset the incumbent's warm `RapierContext` (44+ ticks of solver/contact warm-start caches AND
//! the rigid-body/collider handle-arena free-list), so its first post-join physics step diverges from
//! a fresh-process joiner's zero-history one — job 412's restored-vs-live divergence, surfacing at the
//! join tick. Full bit-exactness would require a deterministic rebuild of the WHOLE physics world
//! (reset every rapier set + respawn the arena AND the crab in a pinned order on every incumbent),
//! which is impractical in the live bevy app. This empirical result CONFIRMS the owner's 2026-06-28
//! direction (determinism downgraded to cool-to-have; host-authoritative STATE-RESYNC is the chosen
//! GCR MP approach, NOT bit-exact lockstep). The probe is kept as the executable evidence + a guard:
//! if a future change claims lockstep join determinism, run this and watch it stay green.

use anyhow::Result;
use clap::Parser;

use super::shared::nn_crab_checkpoint_dir;

#[derive(Parser)]
pub(crate) struct Args {
    /// Trained crab checkpoint dir (`brain.bin` + `normalizer.bin`). Same resolution as
    /// `nn-crab-xpeer` / `play --nn-crab-checkpoint`.
    #[arg(long, value_name = "DIR")]
    checkpoint: Option<std::path::PathBuf>,
    /// Sim ticks the incumbent runs SOLO (armed) before the joiner is admitted.
    #[arg(long, default_value_t = 90)]
    pre_join_ticks: u64,
    /// Sim ticks both peers run together AFTER the join took effect — the window the
    /// warm-incumbent vs cold-joiner rapier crabs must stay byte-identical over.
    #[arg(long, default_value_t = 240)]
    post_join_ticks: u64,
    /// Shared match seed (identical on every peer, as it would be on the wire).
    #[arg(long, default_value_t = 0x6372_6162)]
    seed: u64,
}

/// The mid-game-join armed-Sally determinism gate (GCR MP Stage 3): run the real rapier-NN crab
/// on an INCUMBENT that hosts a solo round, admit a fresh COLD joiner over the round-boundary
/// mechanism, and confirm every peer computes the byte-identical per-tick `state_hash` for every
/// tick the joiner participates in. Exits nonzero on any divergence (the honest signal: determinism
/// does not hold), so it doubles as a guard if a future change ever claims lockstep join determinism.
/// It exercises the existential job-412 risk the pure-core join test (which folds
/// `external_crab_digest = 0`) cannot — and currently REPORTS the divergence (see the module doc).
pub(crate) fn run(args: Args) -> Result<()> {
    use net::external_crab::run_cross_peer_join_probe;

    let dir = nn_crab_checkpoint_dir(args.checkpoint)?;
    println!("nn-crab-join-xpeer: checkpoint={}", dir.display());
    println!(
        "nn-crab-join-xpeer: seed={:#x} pre_join_ticks={} post_join_ticks={}",
        args.seed, args.pre_join_ticks, args.post_join_ticks
    );

    let result =
        run_cross_peer_join_probe(&dir, args.seed, args.pre_join_ticks, args.post_join_ticks);

    if result.ticks.is_empty() {
        anyhow::bail!("nn-crab-join-xpeer: no ticks applied — the peers never advanced");
    }
    let compared = result.compared_post_join_ticks();
    println!(
        "nn-crab-join-xpeer: join effective at tick {}; {} post-join tick(s) had BOTH peers present \
         (the warm-vs-cold rapier comparison)",
        result.effective_tick, compared,
    );
    println!(
        "nn-crab-join-xpeer: lockstep desync faults = {} (the peers' own cross-check)",
        result.faults
    );

    match result.first_divergence() {
        None => {
            let last = result.ticks.last().unwrap();
            println!(
                "nn-crab-join-xpeer: every tick's per-peer hashes IDENTICAL (final tick {})",
                last.tick,
            );
        }
        Some(d) => {
            println!(
                "nn-crab-join-xpeer: FIRST DIVERGENCE at tick {} — hashes {:?}",
                d.tick, d.hashes,
            );
        }
    }

    if result.is_deterministic() {
        println!(
            "nn-crab-join-xpeer: PASS — the cold-respawned incumbent crab and the joiner's fresh \
             crab evolved bit-identically across the mid-game join (0 desyncs, all hashes agree)"
        );
        Ok(())
    } else if compared == 0 {
        anyhow::bail!(
            "nn-crab-join-xpeer: FAIL — the join was never compared (no post-join tick had both \
             peers present); raise post_join_ticks or check the JOIN_LEAD margin"
        )
    } else {
        anyhow::bail!(
            "nn-crab-join-xpeer: FAIL — the armed rapier crab DIVERGED across the mid-game join \
             (job 412's restored-vs-live risk; the diverging hashes are the evidence)"
        )
    }
}
