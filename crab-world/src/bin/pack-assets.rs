//! Bake an asset tree into one `assets.pack` blob (rl#411 stage 6) — the web
//! bundle's whole asset payload. See [`crab_world::asset_pack`] for the format.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Pack every file under --root into an RLPACK1 blob at --out")]
struct Args {
    /// The asset tree to bake (the dir the game would mount as `assets/`).
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let pack = crab_world::asset_pack::pack_dir(&args.root)?;
    std::fs::write(&args.out, &pack)?;
    println!(
        "packed {} -> {} ({} bytes)",
        args.root.display(),
        args.out.display(),
        pack.len()
    );
    Ok(())
}
