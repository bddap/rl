//! rl#318: zero-input slope holding. Sally must not slide downhill under no drive
//! on slopes GCR actually generates — the manual-control "uncontrollable downhill
//! slide" is a contact-physics failure, not a policy one.

use bevy::prelude::*;

use super::body::{CrabBodyPart, CrabCarapace};
use super::headless::{HeadlessStack, WorldRole, headless_stack, tick};
use crate::terrain::TerrainGrid;

/// A uniform ramp of `angle_deg` along +x. ±512 m span so band sampling's clamp
/// assert is satisfied (same sizing as `flat_headless_app`).
fn ramp(angle_deg: f32) -> TerrainGrid {
    const N: usize = 257; // 256 cells × 4 m = ±512 m
    const CELL: f32 = 4.0;
    const SCALE: f32 = 0.1;
    let slope = angle_deg.to_radians().tan();
    let heights: Vec<i16> = (0..N * N)
        .map(|i| {
            let col = i % N;
            (slope * CELL * (col as f32 - (N as f32 - 1.0) / 2.0) / SCALE).round() as i16
        })
        .collect();
    TerrainGrid::test_grid(N, N, CELL, SCALE, &heights)
}

/// Carapace xz drift over `secs` of zero-input settling, after 1 s of touchdown.
/// Returns (drift_m, up_y, rescues): a rescue teleports the body and voids the
/// measure, and a small/negative final carapace-up y means the crab tumbled — a
/// "held position" that ends wedged on its back must not pass as holding.
fn zero_input_drift(grid: TerrainGrid, secs: f32) -> (f32, f32, u64) {
    let hz = crate::physics::PHYSICS_HZ as f32;
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        grid: std::sync::Arc::new(grid),
        visuals: crate::Visuals(false),
    });
    tick(&mut app, hz as u32); // 1 s: spawn + touch down

    let start = carapace(&mut app).translation;
    tick(&mut app, (secs * hz) as u32);
    let end = carapace(&mut app);
    let drift = (end.translation - start).xz().length();
    let up_y = (end.rotation * Vec3::Y).y;
    let rescues = app.world().resource::<super::RescueStats>().total;
    (drift, up_y, rescues)
}

fn carapace(app: &mut App) -> Transform {
    let mut q = app
        .world_mut()
        .query_filtered::<&Transform, (With<CrabBodyPart>, With<CrabCarapace>)>();
    *q.single(app.world()).expect("one carapace")
}

/// What the fix must hold: through the p99 of GCR's cell-slope distribution
/// (`gcr_slope_census`; the distribution and the tuning story live on
/// `physics::world::GROUND_FRICTION`'s doc) — within about a body length
/// (~0.9 m carapace; 1.5 m gives touchdown-settle headroom) over 10 s, upright:
/// the rl#318 acceptance.
#[test]
fn crab_holds_steep_slopes_with_zero_input() {
    for angle in [30.0f32, 40.0, 45.0, 50.0, 55.0] {
        let (drift, up_y, rescues) = zero_input_drift(ramp(angle), 10.0);
        println!(
            "slope {angle:>4.1}°: drift {drift:.2} m over 10 s (up_y {up_y:.2}, {rescues} rescues)"
        );
        assert_eq!(rescues, 0, "{angle}° ramp: rescue voided the measurement");
        assert!(
            drift < 1.5,
            "crab slid {drift:.2} m in 10 s of zero input on a {angle}° ramp (rl#318)"
        );
        // Slope-parallel stance bounds up_y near cos(angle); a tumbled crab goes
        // far below it (inverted is negative).
        assert!(
            up_y > angle.to_radians().cos() - 0.3,
            "crab ended tumbled on the {angle}° ramp: carapace up_y {up_y:.2}"
        );
    }
}

/// Diagnostic: settle upright on FLAT ground for 2 s, then tilt gravity to
/// `TILT_DEG` (default 40) and watch the drift — pure holding with zero touchdown
/// energy, which the ramp test above cannot separate from landing dynamics.
/// `--ignored --nocapture`; combine with `RL_JOINT_FRICTION_CAP` to bound the
/// joint-yield contribution.
///
/// rl#340 stage 4 evidence this instrument produced (all at 40° unless noted, one
/// draw each — realization spread is real, see rl#340 stage 3, but the failures
/// are decisive, not marginal):
/// - stock plant: slides from the FIRST tick at ~2.2 m/s² upright, then trips and
///   lands on its back — effective friction ≈ 0.55 against a nominal foot↔ground
///   pair of 2.0, so the coefficients are NOT what binds;
/// - `GROUND_FRICTION` 2.5→4.0 on the ramp test: 18.8→19.2 m, no effect — the
///   stage's "retune GROUND_FRICTION" hypothesis is falsified;
/// - rigid joints (`RL_JOINT_FRICTION_CAP=5.0`, ~60× the passive leg torque the
///   slope demands): creep drops 4× but still 1.9 m/10 s at 40° and a runaway
///   tumble at 55° — passive joint yield under the 0.04 N·m stiction cap is the
///   DOMINANT leak, yet closing it completely still fails the 1.5 m bar;
/// - `PHYSICS_SUBSTEPS` 4→8, rigid: no better (3.9 m, flipped) — not substep
///   convergence.
///
/// Conclusion: no constant retune passes ≥40°. The remainder lives in contact
/// geometry/delivery on the mesh body (capsule foot tips on 1–4 mm-deep flickering
/// 30 Hz-soft contacts; at rest only 4 of 8 feet + one claw wrist even touch —
/// see `stance_stats`).
#[test]
#[ignore]
fn tilted_gravity_hold() {
    let hz = crate::physics::PHYSICS_HZ as f32;
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        grid: std::sync::Arc::new(ramp(0.0)),
        visuals: crate::Visuals(false),
    });
    tick(&mut app, (2.0 * hz) as u32);
    let angle = std::env::var("TILT_DEG")
        .map(|v| v.parse::<f32>().expect("TILT_DEG must be a number"))
        .unwrap_or(40.0)
        .to_radians();
    println!("  pre-tilt rest contacts:");
    dump_ground_contacts(&mut app);
    {
        let mut q = app
            .world_mut()
            .query::<&mut bevy_rapier3d::plugin::RapierConfiguration>();
        let mut cfg = q.single_mut(app.world_mut()).expect("rapier config");
        cfg.gravity = 9.81 * Vec3::new(angle.sin(), -angle.cos(), 0.0);
    }
    let start = carapace(&mut app).translation;
    for s in 0..10 {
        tick(&mut app, hz as u32);
        let c = carapace(&mut app);
        let d = (c.translation - start).xz().length();
        let u = (c.rotation * Vec3::Y).y;
        println!("  tilted t={}s drift {d:.2} m up_y {u:.2}", s + 1);
        if s == 0 {
            dump_ground_contacts(&mut app);
        }
    }
}

/// Which links touch the heightfield right now, at what depth, with what friction
/// coefficients on each side — the ground-truth for "who carries the crab".
fn dump_ground_contacts(app: &mut App) {
    use bevy_rapier3d::plugin::context::{RapierContextColliders, RapierContextSimulation};
    let mut parts_q = app
        .world_mut()
        .query::<(Entity, Option<&super::body::CrabJoint>, Has<CrabCarapace>)>();
    let names: std::collections::HashMap<Entity, String> = parts_q
        .iter(app.world())
        .map(|(e, j, cara)| {
            let n = j.map(|j| format!("{:?}", j.id)).unwrap_or_else(|| {
                if cara {
                    "Carapace".into()
                } else {
                    "part".into()
                }
            });
            (e, n)
        })
        .collect();
    let mut q = app
        .world_mut()
        .query::<(&RapierContextColliders, &RapierContextSimulation)>();
    let (cols, sim) = q.single(app.world()).expect("rapier ctx");
    let mut rows = Vec::new();
    for pair in sim.narrow_phase.contact_pairs() {
        let ground_h = [pair.collider1, pair.collider2].into_iter().find(|&h| {
            cols.colliders
                .get(h)
                .is_some_and(|c| c.shape().as_heightfield().is_some())
        });
        let Some(gh) = ground_h else { continue };
        let other = if gh == pair.collider1 {
            pair.collider2
        } else {
            pair.collider1
        };
        let Some(oe) = cols.collider_entity(other) else {
            continue;
        };
        // `dist` is negative inside; a pair can exist with zero touching points
        // (speculative margin), which is itself diagnostic — grazing, not resting.
        let (depth, npts) = pair
            .manifolds
            .iter()
            .flat_map(|m| m.points.iter())
            .fold((f32::MIN, 0usize), |(d, n), pt| {
                (d.max(-pt.dist), n + usize::from(-pt.dist > 0.0))
            });
        let fo = cols.colliders.get(other).map(|c| c.friction());
        let fg = cols.colliders.get(gh).map(|c| c.friction());
        let name = names.get(&oe).cloned().unwrap_or_else(|| "non-part".into());
        rows.push(format!(
            "    contact {name}: depth {depth:.4} pts {npts} μ_part {fo:?} μ_ground {fg:?}"
        ));
    }
    rows.sort();
    for r in rows {
        println!("{r}");
    }
}

/// Diagnostic: bind-pose mass distribution vs foot support polygon → static topple
/// thresholds per direction, from the baked recipe alone (no sim). Baseline for the
/// mesh body: com height 0.279 m, feet x ±1.04 m (75° threshold) but z only
/// [-0.58, +0.15] — 36.5° forward. `--ignored --nocapture`.
#[test]
#[ignore]
fn stance_stats() {
    use crate::bot::rig;
    let recipe = rig::baked_recipe();
    let world_pos = rig::link_world_origins(&recipe.links, recipe.hub_bind_world);
    let mut com = Vec3::ZERO;
    let mut total = 0.0f32;
    let mut feet = Vec::new();
    {
        let c = bevy_rapier3d::prelude::Collider::cuboid(
            recipe.carapace_half.x,
            recipe.carapace_half.y,
            recipe.carapace_half.z,
        );
        let m = c.raw.mass_properties(recipe.carapace_density).mass();
        let p = recipe.hub_bind_world + recipe.carapace_offset;
        com += p * m;
        total += m;
        println!("carapace m {m:.3} at {p:?}");
    }
    for (link, &wp) in recipe.links.iter().zip(&world_pos) {
        let c = match rig::link_rest_shape(link, Vec3::ZERO) {
            rig::RestShape::Capsule { a, b, radius } => {
                bevy_rapier3d::prelude::Collider::capsule(a, b, radius)
            }
            rig::RestShape::Cuboid { half, .. } => {
                bevy_rapier3d::prelude::Collider::cuboid(half.x, half.y, half.z)
            }
        };
        let m = c.raw.mass_properties(link.density).mass();
        let p = wp + link.center;
        com += p * m;
        total += m;
        if matches!(link.actuated, Some(super::body::CrabJointId::LegCarpus(..))) {
            feet.push(p);
        }
    }
    com /= total;
    println!("total m {total:.3}  com {com:?}");
    let ground = feet.iter().map(|f| f.y).fold(f32::MAX, f32::min) - 0.05;
    let h = com.y - ground;
    let max_x = feet.iter().map(|f| f.x).fold(f32::MIN, f32::max);
    let min_x = feet.iter().map(|f| f.x).fold(f32::MAX, f32::min);
    let max_z = feet.iter().map(|f| f.z).fold(f32::MIN, f32::max);
    let min_z = feet.iter().map(|f| f.z).fold(f32::MAX, f32::min);
    for f in &feet {
        println!("foot at {f:?}");
    }
    println!("com height {h:.3}; feet x [{min_x:.3},{max_x:.3}] z [{min_z:.3},{max_z:.3}]");
    println!(
        "topple thresholds: +x {:.1}° -x {:.1}° +z {:.1}° -z {:.1}°",
        ((max_x - com.x) / h).atan().to_degrees(),
        ((com.x - min_x) / h).atan().to_degrees(),
        ((max_z - com.z) / h).atan().to_degrees(),
        ((com.z - min_z) / h).atan().to_degrees(),
    );
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
