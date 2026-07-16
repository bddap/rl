//! Offline mesh-fitting tool (bddap/rl#20): fits Sally's physics colliders from the
//! skinned glTF and bakes them into the committed table the runtime consumes
//! (`crab-world/src/bot/rig/baked.rs`). Runtime crates carry NO fitting code — a
//! re-fit is a rare, deliberate event (a sally.glb change), done here, and any
//! geometry change is a loud, reviewed diff of the baked table (a new MDP: plan a
//! retrain). The collider<->mesh audits that need the fitter's containment machinery
//! live here too (they were `rl-train --verify-colliders` / `--verify-pivots`).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod audit;
mod bake;
mod containment;
mod fit;
mod gltf_load;

#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Fit colliders from sally.glb and regenerate crab-world/src/bot/rig/baked.rs.
    Bake {
        /// Output path (default: the in-repo baked.rs next to this tool's crate).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Collider<->mesh agreement over every rest collider of the CURRENT fit of the
    /// current asset (equals the shipped baked table iff the bake is fresh — the
    /// `baked_matches_refit` test is what pins that).
    VerifyColliders,
    /// Joint-pivot / capsule-endpoint containment in the bind mesh.
    VerifyPivots,
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Bake { out } => bake_cmd(out),
        Command::VerifyColliders => audit::verify_colliders().map(ExitCode::from),
        Command::VerifyPivots => audit::verify_pivots().map(ExitCode::from),
    };
    result.unwrap_or_else(|e| {
        eprintln!("{e}");
        ExitCode::FAILURE
    })
}

fn default_baked_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../crab-world/src/bot/rig/baked.rs")
}

fn bake_cmd(out: Option<PathBuf>) -> Result<ExitCode, String> {
    let Some(path) = crab_world::mesh_fallback::model_path() else {
        return Err(format!(
            "bake: {}",
            crab_world::mesh_fallback::MESH_ABSENT_REASON
        ));
    };
    // One read: the digest stamped into the table describes the exact bytes fitted.
    let bytes = std::fs::read(&path).map_err(|e| format!("bake: read {path:?}: {e}"))?;
    let digest = crab_world::fnv::fnv1a(&bytes);
    let model = gltf_load::LoadedModel::from_slice(&bytes)
        .map_err(|e| format!("bake: load {path:?}: {e}"))?;
    let unmapped = model.unmapped_bones();
    if !unmapped.is_empty() {
        return Err(format!(
            "bake: refusing — bones map to no physics part (their flesh would fit into \
             nothing): {unmapped:?}"
        ));
    }
    let recipe =
        bake::fitted_recipe(&model).ok_or_else(|| format!("bake: {path:?} built no rig recipe"))?;
    let rendered = bake::render_baked_rs(&recipe, digest);
    let out = out.unwrap_or_else(default_baked_path);
    std::fs::write(&out, &rendered).map_err(|e| format!("bake: write {out:?}: {e}"))?;
    println!(
        "baked {} links from {} (digest {digest:#018x}) -> {}",
        recipe.links.len(),
        path.display(),
        out.display()
    );
    Ok(ExitCode::SUCCESS)
}
