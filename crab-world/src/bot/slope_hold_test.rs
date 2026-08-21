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
    ramp_scaled(angle_deg, 0.1)
}

/// `ramp` with an explicit vertical scale: ppm-level jitter on the scale re-rolls
/// the solver's rest-noise realization at physically negligible world change (the
/// rl#340 stage-3 chaos finding — one draw of this system is a lottery ticket).
/// Small ANGLE jitter can not do this: heights are i16-quantized, so a
/// milli-degree perturbation leaves the cells near the spawn point bit-identical.
fn ramp_scaled(angle_deg: f32, scale: f32) -> TerrainGrid {
    const N: usize = 257; // 256 cells × 4 m = ±512 m
    const CELL: f32 = 4.0;
    const SCALE: f32 = 0.1;
    let slope = angle_deg.to_radians().tan();
    // The i16 counts always use the canonical 0.1 divisor, so a jittered `scale`
    // shifts every world height by the same ppm factor instead of re-quantizing.
    let heights: Vec<i16> = (0..N * N)
        .map(|i| {
            let col = i % N;
            (slope * CELL * (col as f32 - (N as f32 - 1.0) / 2.0) / SCALE).round() as i16
        })
        .collect();
    TerrainGrid::test_grid(N, N, CELL, scale, &heights)
}

/// Carapace xz drift over `secs` of zero-input settling, after 1 s of touchdown.
/// Returns (drift_m, up_y, rescues): a rescue teleports the body and voids the
/// measure, and a small/negative final carapace-up y means the crab tumbled — a
/// "held position" that ends wedged on its back must not pass as holding.
///
/// Prints a per-second trace plus the t=2 joint-rail dump: on a failure the
/// captured output separates a touchdown skid (drift front-loaded into the first
/// seconds while joints collapse to their stops) from steady creep — the two need
/// different fixes, and one final number cannot tell them apart.
fn zero_input_drift(grid: TerrainGrid, secs: u32) -> (f32, f32, u64) {
    let hz = crate::physics::PHYSICS_HZ as u32;
    let mut app = headless_stack(HeadlessStack {
        num_envs: 1,
        role: WorldRole::Standalone,
        grid: std::sync::Arc::new(grid),
        visuals: crate::Visuals(false),
    });
    tick(&mut app, hz); // 1 s: spawn + touch down

    let start = carapace(&mut app).translation;
    for s in 1..=secs {
        tick(&mut app, hz);
        let c = carapace(&mut app);
        let d = (c.translation - start).xz().length();
        println!(
            "    ramp t={s}s drift {d:.2} m up_y {:.2}",
            (c.rotation * Vec3::Y).y
        );
        if s == 2 {
            dump_joint_angles(&mut app);
        }
    }
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
#[ignore = "rl#385: physically red at ≥40° on the kept plant (rl#340 stages 4–4c); stays ignored until slope navigation is a demonstrated LEARNED behavior, then becomes a training eval — never un-ignore as-is"]
fn crab_holds_steep_slopes_with_zero_input() {
    for angle in [30.0f32, 40.0, 45.0, 50.0, 55.0] {
        let (drift, up_y, rescues) = zero_input_drift(ramp(angle), 10);
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

/// Diagnostic: the zero-input hold table over arbitrary ramp angles — the
/// instrument that prices the rl#318 band call (drift and posture per angle,
/// where the acceptance test above only reports its fixed band). `RAMP_DEGS`
/// is a comma-separated angle list; `RAMP_DRAWS` (default 3) runs each angle
/// that many times under ppm terrain-scale jitter (see [`ramp_scaled`]) so the
/// realization spread is visible, not hidden inside one draw.
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn slope_hold_table() {
    let degs = std::env::var("RAMP_DEGS").unwrap_or_else(|_| "25,30,35,40,45".into());
    let draws: u32 = std::env::var("RAMP_DRAWS")
        .map(|v| v.parse().expect("RAMP_DRAWS must be a count"))
        .unwrap_or(3);
    for angle in degs
        .split(',')
        .map(|s| s.trim().parse::<f32>().expect("RAMP_DEGS must be numbers"))
    {
        for k in 0..draws {
            let scale = 0.1 * (1.0 + k as f32 * 1e-6);
            let (drift, up_y, rescues) = zero_input_drift(ramp_scaled(angle, scale), 10);
            println!(
                "TABLE {angle:>4.1}° draw {k}: drift {drift:.2} m up_y {up_y:.2} rescues {rescues}"
            );
        }
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
///
/// rl#340 stage 4b (job 2476) falsified direction (a) — foot-contact redesign —
/// too, one draw each (this run added the imp_n/imp_t columns to the contact
/// dump: the solver's summed normal/tangent impulses per pair):
/// - flat foot pads (round-cuboid under each carpus tip, ground-parallel in bind
///   pose, sitting proud of the capsule sphere): first-second tilt slide
///   1.09→0.83 m but still flips — and the RAMP test DEGRADES (30°: 0.99→1.28 m
///   ending half-tumbled, 40°: 18.75→21.4 m): pads mush the touchdown;
/// - `ContactSkin` 5 mm on the feet (keeps manifold points alive across the mm
///   flicker): pure-hold creep decelerates (near-hold to t=4 at cap 0.3) yet
///   every combination still trips by t≤9, and the ramp degrades as above;
/// - solver iterations 16/8/8: no change; contact spring 30→120 Hz: ballistic
///   chatter explosion (24 m in the first tilted second);
/// - the friction cone is NEVER railed while creeping (Σ|imp_t|/Σimp_n measured
///   0.3–1.3 vs the 2.0 pair coefficient): the slide is point-flicker windows
///   plus joint yield, not cone-limited slip — coefficients stay innocent;
/// - UPPER BOUND: rigid joints (cap 5.0) + pads + skin — the most any foot
///   geometry can deliver — still creeps to 2.25 m by t=8 and trips at t=9 at
///   40°. Every foot-contact redesign is bounded by the rigid-joint hold, so
///   direction (a) alone cannot pass the 1.5 m/10 s bar; what remains is
///   stance/bake work (direction b: all 8 feet grounded, wider support polygon
///   — which moves the rigid bound itself) or the rl#318 band re-scope
///   (direction c). The pad/skin plant changes were reverted (they regress the
///   ramp acceptance); only this instrumentation landed.
///
/// rl#340 stage 4c (job 2485) ran direction (b) to its bound. The key mechanism
/// (`dump_joint_angles`, the landed increment): with 0.04 N·m stiction no
/// load-bearing joint holds mid-range, so the zero-input standing skeleton is
/// the gravity-railed LIMIT-STOP pose — at rest 6/8 basis joints sit railed on
/// their −0.6 lo stop, the body hangs nose-down on the 4 front feet + one claw
/// wrist (μ 0.5, half the normal load), and the back feet never reach ground.
/// Stance is therefore a LIMITS property; spawn-time joint coordinates cannot
/// hold against load. Experiments at 40°, one draw each:
/// - claw shoulders preset to −0.3 (wrist prop removed): still 4-footed,
///   still flips — the prop was not the binding failure;
/// - basis lo stop −0.6 → −0.2 (+ spawn preset at the stop, claws lifted):
///   7–8 feet grounded, and the FLIP IS GONE — first upright 40° hold in the
///   epic (tilt instrument: up_y 0.97–1.00 through t=10). Ramp: 30° drift
///   0.99 → 0.61–0.75 m, 40° 18.75 → 1.97 m upright — but the bar is 1.5 m.
///   The trace: ~1.45 m is t≤2 touchdown skid while the downhill merus/carpus
///   collapse to their +1.0/+1.1 stops, then ~60 mm/s creep;
/// - the neighborhood is sharp: basis lo −0.15 → 2.91 m, −0.25 → 7.55 m
///   tumbling, and −0.3 → 6.93 m ends tumbled; −0.2 is the optimum;
/// - tightening merus/carpus stops to ±0.8 (to cut the collapse): 16.07 m —
///   that compliance is what conforms the legs to the slope; falsified;
/// - stiction cap 0.04 → 0.1 on the best stance: 6.49 m with a mid-run
///   tumble; stiffer joints conform worse; falsified;
/// - 45° with the best stance: 29.09 m, flipped — the band above 40° stays
///   unreachable by any stance found.
///
/// Verdict: stance work moves the rigid bound exactly as 4b predicted (a 9×
/// slide cut and flip-free 40°) yet cannot pass 1.5 m at 40°, and ≥45° is
/// structurally out of reach. The basis-stop plant change was reverted per the
/// revert rule (the gated test stays red either way, and halving the basis
/// range is an MDP call bundled with direction (c) — the owner's rl#318 band
/// re-scope, which now has this frontier to price against the 30–55° demand).
///
/// rl#340 stage 9 (job 2506) re-priced both plants on the post-recycling plant
/// (a4e4b89 pinned rapier-0.35 contact recycling off, which moved the slope
/// economics stage 4c was priced on). `slope_hold_table`, 3 scale-jitter draws
/// per angle, drift m over 10 s (bar 1.5):
/// - stock plant: 25° 0.98–1.04 up_y 0.85; 30° 2.57–2.69 TUMBLED (up_y −0.87,
///   every draw — pre-recycling this held at 0.99 m); 35° 3.59–3.75 half-tumbled
///   (up_y 0.35–0.42); 40° 4.68–17.60 heavy-tailed (stage 7's one-draw 3.71 was
///   the tail's lucky end); 45° 21.8–33.7 tumbled;
/// - 4c candidate (basis lo −0.2 + spawn preset + claws lifted): 25° 0.34–0.65;
///   30° 0.78–0.81; 35° 0.82–1.65 straddling the bar; 40° 2.07–2.59 UPRIGHT
///   every draw (up_y 0.75–0.82); 45° 3.71–19.63 (up_y down to 0.15).
///
/// The candidate's ordering vs stock survives recycling unchanged; the stock
/// plant's zero-input floor dropped from "holds 30°" to "holds 25°".
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
    dump_joint_angles(&mut app);
    {
        let mut q = app
            .world_mut()
            .query::<&mut bevy_rapier3d::plugin::RapierConfiguration>();
        let mut cfg = q.single_mut(app.world_mut()).expect("rapier config");
        cfg.gravity =
            -crate::physics::PHYSICS_GRAVITY.y * Vec3::new(angle.sin(), -angle.cos(), 0.0);
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

/// Per-joint coordinate vs its limits at rest: the zero-input skeleton is gravity
/// railed against limit stops (stiction is 0.04 N·m — no load-bearing joint holds
/// mid-range), so THIS pose, not the bind pose, is the standing geometry.
fn dump_joint_angles(app: &mut App) {
    use bevy_rapier3d::plugin::context::RapierContextJoints;
    let mut jq = app.world_mut().query::<(
        &bevy_rapier3d::prelude::RapierMultibodyJointHandle,
        &super::body::CrabJoint,
    )>();
    let pairs: Vec<_> = jq.iter(app.world()).map(|(h, j)| (h.0, j.id)).collect();
    let mut ctx = app.world_mut().query::<&RapierContextJoints>();
    let joints = ctx.single(app.world()).expect("rapier joints");
    let mut rows = Vec::new();
    for (handle, id) in pairs {
        // A missing row would silently shrink the "N/8 railed" denominator this
        // dump exists to count, so a stale handle announces itself instead.
        let Some(link) = joints
            .multibody_joints
            .get(handle)
            .and_then(|(mb, link_id)| mb.link(link_id))
        else {
            rows.push(format!(
                "    joint {id:?}: NO MULTIBODY LINK (stale handle)"
            ));
            continue;
        };
        // Spatial-vector slots 0-2 are linear; a revolute's one angular dof
        // lands in slot 3.
        let angle = link.joint().coords()[3];
        let [lo, hi] = id.limits();
        let railed = if angle <= lo + 0.02 {
            " RAILED-LO"
        } else if angle >= hi - 0.02 {
            " RAILED-HI"
        } else {
            ""
        };
        rows.push(format!(
            "    joint {id:?}: {angle:+.3} in [{lo:+.2},{hi:+.2}]{railed}"
        ));
    }
    rows.sort();
    for r in rows {
        println!("{r}");
    }
}

/// Which links touch the heightfield right now, at what depth, with what friction
/// coefficients on each side — the ground-truth for "who carries the crab".
fn dump_ground_contacts(app: &mut App) {
    use bevy_rapier3d::plugin::context::{RapierContextColliders, RapierContextSimulation};
    let mut parts_q = app.world_mut().query_filtered::<(
        Entity,
        Option<&super::body::CrabJoint>,
        Has<CrabCarapace>,
    ), With<CrabBodyPart>>();
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
        // Solver ground truth: what the friction cone actually delivered last
        // step. Slip with |tangent| ≈ μ·normal means the cone is railed (bound
        // too low); |tangent| ≪ μ·normal during slip means the solver never
        // applies the friction it is allowed.
        let (n_imp, t_imp) = pair
            .manifolds
            .iter()
            .flat_map(|m| m.points.iter())
            .fold((0.0f32, 0.0f32), |(n, t), pt| {
                (n + pt.data.impulse, t + pt.data.tangent_impulse.norm())
            });
        let fo = cols.colliders.get(other).map(|c| c.friction());
        let fg = cols.colliders.get(gh).map(|c| c.friction());
        let name = names.get(&oe).cloned().unwrap_or_else(|| "non-part".into());
        let depth = if depth == f32::MIN {
            "none".to_string()
        } else {
            format!("{depth:.4}")
        };
        rows.push(format!(
            "    contact {name}: depth {depth} pts {npts} μ_part {fo:?} μ_ground {fg:?} \
             imp_n {n_imp:.4} imp_t {t_imp:.4}"
        ));
    }
    rows.sort();
    for r in rows {
        println!("{r}");
    }
}

/// Diagnostic: bind-pose mass distribution vs foot support polygon → static topple
/// thresholds per direction, from the baked recipe alone (no sim). Baseline for the
/// mesh body: com height 0.399 m, foot tips x ±1.09 m (70° threshold across) but z
/// only [-0.63, +0.13] — 24.3° forward, well under the rl#318 band's floor.
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn stance_stats() {
    use crate::bot::rig;
    let recipe = rig::baked_recipe();
    let world_pos = rig::link_world_origins(&recipe.links, recipe.hub_bind_world);
    let mut com = Vec3::ZERO;
    let mut total = 0.0f32;
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
    // A foot is the LOW capsule endpoint — the point that actually meets the
    // ground — not the link center, which sits half a capsule higher and inboard
    // on splayed legs.
    let mut tips = Vec::new();
    for (link, &wp) in recipe.links.iter().zip(&world_pos) {
        let shape = rig::link_rest_shape(link, Vec3::ZERO);
        let c = match shape {
            rig::RestShape::Capsule { a, b, radius } => {
                bevy_rapier3d::prelude::Collider::capsule(a, b, radius)
            }
            rig::RestShape::Cuboid { half, .. } => {
                bevy_rapier3d::prelude::Collider::cuboid(half.x, half.y, half.z)
            }
        };
        let m = c.raw.mass_properties(link.density).mass();
        com += (wp + link.center) * m;
        total += m;
        if matches!(link.actuated, Some(super::body::CrabJointId::LegCarpus(..))) {
            let rig::RestShape::Capsule { a, b, radius } = shape else {
                panic!("carpus links are capsules in every bake so far");
            };
            let tip = wp + if a.y < b.y { a } else { b };
            tips.push((tip, radius));
        }
    }
    com /= total;
    println!("total m {total:.3}  com {com:?}");
    let feet: Vec<Vec3> = tips.iter().map(|&(t, _)| t).collect();
    let ground = tips.iter().map(|&(t, r)| t.y - r).fold(f32::MAX, f32::min);
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
