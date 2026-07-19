//! rl#295: measured cost of rapier `QueryPipeline::cast_ray` vs the closed-form
//! [`TerrainGrid::height`] lookup at training shape (25 scan samples × live envs ×
//! 64 Hz). Informational, not a gate — it prices the future swap to direction-cast
//! rays when non-heightfield geometry (trees, obstacles) lands. Run manually:
//! `cargo test -p crab-world --release --test terrain_query_bench -- --ignored --nocapture`

use std::hint::black_box;
use std::time::{Duration, Instant};

use crab_world::terrain::TerrainGrid;

/// The rl#295 scan shape: center + 8 bearings × radii {3, 9, 27} m. Probe placement
/// only (the ratio is shape-independent); the assert against the sensor's own
/// TERRAIN_SAMPLES below keeps the headline per-wall-second budget from going stale
/// if the scan pattern changes.
const SCAN_RADII_M: [f32; 3] = [3.0, 9.0, 27.0];
const SCAN_BEARINGS: usize = 8;
const SAMPLES_PER_ENV: usize = 1 + SCAN_RADII_M.len() * SCAN_BEARINGS;
/// Live training shape: 2 workers × `--envs 4`.
const ENVS: usize = 8;
const TICK_HZ: f64 = 64.0;
/// Above the bake's highest peak (datum-shifted span is 4213 m), so every vertical
/// ray enters from open air.
const RAY_Y: f32 = 5000.0;

/// Deterministic xorshift — the bench must not depend on run-to-run point luck.
fn rand01(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / (1u64 << 24) as f32
}

/// ENVS env locales on the tile interior, each with the 25-point yaw-frame scan.
fn sample_points(grid: &TerrainGrid) -> Vec<(f32, f32)> {
    let half = grid.extent_x() / 2.0 - 100.0;
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut points = Vec::with_capacity(ENVS * SAMPLES_PER_ENV);
    for _ in 0..ENVS {
        let cx = (rand01(&mut state) * 2.0 - 1.0) * half;
        let cz = (rand01(&mut state) * 2.0 - 1.0) * half;
        let yaw = rand01(&mut state) * std::f32::consts::TAU;
        points.push((cx, cz));
        for radius in SCAN_RADII_M {
            for b in 0..SCAN_BEARINGS {
                let a = yaw + b as f32 * std::f32::consts::TAU / SCAN_BEARINGS as f32;
                points.push((cx + radius * a.sin(), cz + radius * a.cos()));
            }
        }
    }
    points
}

/// ns/query for one full pass over the point set: auto-scales the batch to ≥50 ms
/// (so the clock resolution is noise), then takes the best of 3 batches.
fn bench_ns_per_query(mut pass: impl FnMut() -> f32, queries_per_pass: usize) -> f64 {
    let mut batch = 1usize;
    loop {
        let timed = |batch: usize, pass: &mut dyn FnMut() -> f32| {
            let t = Instant::now();
            let mut acc = 0.0f32;
            for _ in 0..batch {
                acc += pass();
            }
            black_box(acc);
            t.elapsed()
        };
        let elapsed = timed(batch, &mut pass);
        if elapsed >= Duration::from_millis(50) {
            let best = (0..2)
                .map(|_| timed(batch, &mut pass))
                .min()
                .unwrap_or(elapsed)
                .min(elapsed);
            return best.as_nanos() as f64 / (batch * queries_per_pass) as f64;
        }
        batch *= 2;
    }
}

#[test]
#[ignore = "informational benchmark (rl#295), run manually with --release --nocapture"]
fn cast_ray_vs_height_lookup_at_training_shape() {
    use bevy_rapier3d::rapier::math::Vector;
    use bevy_rapier3d::rapier::parry::query::DefaultQueryDispatcher;
    use bevy_rapier3d::rapier::prelude::{
        BroadPhaseBvh, ColliderBuilder, ColliderSet, IntegrationParameters, QueryFilter, Ray, Real,
        RigidBodySet,
    };

    assert_eq!(SAMPLES_PER_ENV, crab_world::bot::sensor::TERRAIN_SAMPLES);
    let grid = TerrainGrid::gcr();
    let points = sample_points(&grid);
    assert_eq!(points.len(), ENVS * SAMPLES_PER_ENV);

    let bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let handle = colliders.insert(ColliderBuilder::new(grid.collider().raw).build());
    let mut broad_phase = BroadPhaseBvh::new();
    let mut events = Vec::new();
    broad_phase.update(
        &IntegrationParameters::default(),
        &colliders,
        &bodies,
        &[handle],
        &[],
        &mut events,
    );
    let dispatcher = DefaultQueryDispatcher;
    let pipeline =
        broad_phase.as_query_pipeline(&dispatcher, &bodies, &colliders, QueryFilter::default());
    let cast = |x: f32, z: f32| -> f32 {
        let ray = Ray::new(Vector::new(x, RAY_Y, z), Vector::new(0.0, -1.0, 0.0));
        let (_, toi) = pipeline
            .cast_ray(&ray, Real::MAX, true)
            .expect("vertical ray hits the tile");
        RAY_Y - toi
    };

    // Both instruments must measure the same surface before we time them.
    for &(x, z) in &points {
        let (ray_h, sample_h) = (cast(x, z), grid.height(x, z));
        assert!(
            (ray_h - sample_h).abs() < 0.1,
            "ray and sampler disagree at ({x},{z}): {ray_h} vs {sample_h}"
        );
    }

    let height_ns = bench_ns_per_query(
        || points.iter().map(|&(x, z)| grid.height(x, z)).sum(),
        points.len(),
    );
    let ray_ns = bench_ns_per_query(
        || points.iter().map(|&(x, z)| cast(x, z)).sum(),
        points.len(),
    );

    let per_sec = |ns: f64| ns * (ENVS * SAMPLES_PER_ENV) as f64 * TICK_HZ / 1e6;
    println!(
        "terrain query cost, {ENVS} envs x {SAMPLES_PER_ENV} samples @ {TICK_HZ} Hz (rl#295):"
    );
    println!(
        "  TerrainGrid::height     {height_ns:9.1} ns/query -> {:8.3} ms of one core per wall-second",
        per_sec(height_ns)
    );
    println!(
        "  QueryPipeline::cast_ray {ray_ns:9.1} ns/query -> {:8.3} ms of one core per wall-second",
        per_sec(ray_ns)
    );
    println!("  ray/height ratio        {:9.1}x", ray_ns / height_ns);
}
