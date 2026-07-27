//! rl#318: zero-input slope holding. Sally must not slide downhill under no drive
//! on slopes GCR actually generates — the manual-control "uncontrollable downhill
//! slide" is a contact-physics failure, not a policy one.

use bevy::prelude::*;

use super::body::{CrabBodyPart, CrabCarapace};
use super::headless::{HeadlessStack, WorldRole, headless_stack, tick};
use crate::terrain::TerrainGrid;

/// A uniform ramp of `angle_deg` along +x, built through the real artifact codec
/// (the parse path IS the seam production grids ride). ±512 m span so band
/// sampling's clamp assert is satisfied (same sizing as `flat_headless_app`).
fn ramp(angle_deg: f32) -> TerrainGrid {
    const N: usize = 257; // 256 cells × 4 m = ±512 m
    const CELL: f32 = 4.0;
    const SCALE: f32 = 0.1;
    let meta = format!(r#"{{"rows":{N},"cols":{N},"cell_size_m":{CELL},"height_scale":{SCALE}}}"#);
    let mut bytes = b"RLTERR01".to_vec();
    bytes.extend((meta.len() as u32).to_le_bytes());
    bytes.extend(meta.as_bytes());
    let slope = angle_deg.to_radians().tan();
    for _row in 0..N {
        for col in 0..N {
            let h = slope * CELL * (col as f32 - (N as f32 - 1.0) / 2.0) / SCALE;
            bytes.extend((h.round() as i16).to_le_bytes());
        }
    }
    TerrainGrid::parse(&bytes).expect("test artifact parses")
}

/// Carapace xz drift over `secs` of zero-input settling, after 1 s of touchdown.
/// Returns (drift_m, rescues) — a rescue teleports the body and voids the measure.
fn zero_input_drift(grid: TerrainGrid, secs: f32) -> (f32, u64) {
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        grid: std::sync::Arc::new(grid),
        visuals: crate::Visuals(false),
    });
    tick(&mut app, 64); // 1 s: spawn + touch down

    let start = carapace_xz(&mut app);
    tick(&mut app, (secs * crate::physics::PHYSICS_HZ as f32) as u32);
    let drift = (carapace_xz(&mut app) - start).length();
    let rescues = app.world().resource::<super::RescueStats>().total;
    (drift, rescues)
}

fn carapace_xz(app: &mut App) -> Vec2 {
    let mut q = app
        .world_mut()
        .query_filtered::<&Transform, (With<CrabBodyPart>, With<CrabCarapace>)>();
    let t = q.single(app.world()).expect("one carapace");
    Vec2::new(t.translation.x, t.translation.z)
}

/// What the fix must hold: GCR's steep ground — cell slopes run p50 26.6° /
/// p90 43.5° / p99 54.6° (`gcr_slope_census`) — within about a body length
/// (~0.9 m carapace; 1.5 m gives touchdown-settle headroom) over 10 s: the rl#318
/// acceptance. Pre-fix the 40° ramp slid 70 m/10 s (the terrain collider had no
/// Friction component — see `physics::world::GROUND_FRICTION` for the full story).
#[test]
fn crab_holds_steep_slopes_with_zero_input() {
    for angle in [30.0, 40.0, 45.0, 50.0, 55.0] {
        let (drift, rescues) = zero_input_drift(ramp(angle), 10.0);
        println!("slope {angle:>4.1}°: drift {drift:.2} m over 10 s ({rescues} rescues)");
        assert_eq!(rescues, 0, "{angle}° ramp: rescue voided the measurement");
        assert!(
            drift < 1.5,
            "crab slid {drift:.2} m in 10 s of zero input on a {angle}° ramp (rl#318)"
        );
    }
}

/// Diagnostic census of the committed GCR bake: per-cell max gradient angle, so the
/// holding test above targets slopes the terrain actually generates. Ignored in
/// normal runs; `--ignored --nocapture` prints the distribution.
#[test]
#[ignore]
fn gcr_slope_census() {
    let g = TerrainGrid::gcr();
    let ext = 15_000.0f32; // inside the 1024 × 30 m tile
    let step = 30.0f32;
    let mut angles = Vec::new();
    let mut x = -ext;
    while x < ext {
        let mut z = -ext;
        while z < ext {
            let dx = (g.height(x + step, z) - g.height(x, z)) / step;
            let dz = (g.height(x, z + step) - g.height(x, z)) / step;
            angles.push((dx * dx + dz * dz).sqrt().atan().to_degrees());
            z += step;
        }
        x += step;
    }
    angles.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| angles[((angles.len() - 1) as f64 * p) as usize];
    println!(
        "GCR cell slopes: p50 {:.1}° p90 {:.1}° p99 {:.1}° max {:.1}°",
        pct(0.5),
        pct(0.9),
        pct(0.99),
        pct(1.0)
    );
}
