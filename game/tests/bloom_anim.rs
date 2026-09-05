//! rl#420: the night-bloom glow breathes — one shared mechanism over every
//! variant row, period `BLOOM_CYCLE_S`. Proven on the rendered pixels through
//! the game's own frame-sequence path: half a period apart the frame changes,
//! a whole period apart it repeats. A static scene (sky frozen, no crab, no
//! walk) so the ground's light is the only thing that can move.

use std::path::PathBuf;
use std::process::Command;

use crab_world::ground::BLOOM_CYCLE_S;
use net::sim::TICK_HZ;

test_watchdog::arm!();

fn frame(path: PathBuf) -> Vec<u8> {
    image::open(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_rgb8()
        .into_raw()
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

#[test]
fn night_bloom_glow_breathes_on_its_period() {
    let half_period_frames = BLOOM_CYCLE_S as f64 / 2.0 * TICK_HZ as f64;
    assert_eq!(
        half_period_frames.fract(),
        0.0,
        "half a bloom cycle must be a whole number of render frames"
    );
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bloom-anim/f.png");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_game"))
        .env("RL_ALLOW_MISSING_ASSETS", "1")
        .args([
            "fp-screenshot",
            "--ground-look=night-bloom",
            "--players=1",
            "--width=320",
            "--height=180",
            "--settle=90",
            "--anim-frames=3",
            "--cam-pitch=-55",
            "--cam-height=25",
            "--moon-azimuth-deg=200",
            "--moon-elevation-deg=45",
        ])
        .arg(format!("--anim-every={half_period_frames}"))
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "fp-screenshot failed: {status}");

    let shot = |k: u32| frame(out.with_extension(format!("{k:04}.png")));
    let (f0, f1, f2) = (shot(0), shot(1), shot(2));
    let half = mean_abs_diff(&f0, &f1);
    let full = mean_abs_diff(&f0, &f2);
    assert!(
        full < 0.5,
        "one full cycle later the frame should repeat: mean |Δ| = {full:.3}/255"
    );
    assert!(
        half > 1.0,
        "half a cycle later the glow should have moved: mean |Δ| = {half:.3}/255 (full-cycle {full:.3})"
    );
}
