use anyhow::Result;
use clap::Parser;

use crab_world::bot::body::LIMIT_SOFTNESS;
use crab_world::physics::CONTACT_SOFTNESS;
use crab_world::physics::snapshot::{
    ContactFilter, PlantSnapshot, ReplayConfig, ReplayOutcome, ShapeVariant, SpringCoefficients,
};

/// rl#332 T1: replay ONE tick from each `sally-soak --dump-state-at` snapshot,
/// varying ONE lever at a time against the shipped configuration — drives, solver
/// counts, joint limit spring, link collider shape, same-crab contact filter — and
/// print whether the recorded kick survives. The first row is the self-check: the
/// recorded contact filter with recorded drives — it must reproduce the original
/// run; the second is this build's filter.
#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "FILE", required = true, num_args = 1..)]
    state: Vec<std::path::PathBuf>,
}

struct Row {
    label: String,
    cfg: ReplayConfig,
}

fn rows() -> Vec<Row> {
    let shipped = ReplayConfig {
        drive_scale: 1.0,
        iterations: crab_world::physics::SOLVER_ITERATIONS,
        substeps: crab_world::physics::PHYSICS_SUBSTEPS,
        limit_softness: None,
        shape: ShapeVariant::AsIs,
        filter: ContactFilter::Shipped,
    };
    let row = |label: &str, cfg: ReplayConfig| Row {
        label: label.to_string(),
        cfg,
    };
    let soft = |hz: f32, zeta: f32| {
        Some(SpringCoefficients {
            natural_frequency: hz,
            damping_ratio: zeta,
        })
    };
    let mut rows = vec![
        row(
            "self-check (filter as recorded)",
            ReplayConfig {
                filter: ContactFilter::AsRecorded,
                ..shipped
            },
        ),
        row("shipped filter (no same-crab)", shipped),
    ];
    for (label, scale) in [("drives zeroed", 0.0), ("drives ×0.5", 0.5)] {
        rows.push(row(
            label,
            ReplayConfig {
                drive_scale: scale,
                ..shipped
            },
        ));
    }
    for (iterations, substeps) in [
        ((2, 2, 3), 2),
        ((2, 24, 3), 2),
        ((2, 48, 3), 2),
        ((4, 12, 3), 2),
        ((2, 12, 3), 4),
        ((8, 4, 4), 4),
        ((32, 8, 8), 8),
    ] {
        rows.push(row(
            &format!("solver {iterations:?}×{substeps}"),
            ReplayConfig {
                iterations,
                substeps,
                ..shipped
            },
        ));
    }
    rows.push(row(
        "limit spring 40 Hz",
        ReplayConfig {
            limit_softness: soft(40.0, LIMIT_SOFTNESS.damping_ratio),
            ..shipped
        },
    ));
    rows.push(row(
        "limit spring = contact class",
        ReplayConfig {
            limit_softness: soft(
                CONTACT_SOFTNESS.natural_frequency,
                CONTACT_SOFTNESS.damping_ratio,
            ),
            ..shipped
        },
    ));
    for (label, shape) in [
        ("capsule radius ×1.5", ShapeVariant::CapsuleRadius(1.5)),
        ("capsule radius ×0.5", ShapeVariant::CapsuleRadius(0.5)),
        ("cuboids → capsules", ShapeVariant::CuboidsToCapsules),
        ("all links thin balls", ShapeVariant::Balls { fat: false }),
        ("all links fat balls", ShapeVariant::Balls { fat: true }),
    ] {
        rows.push(row(label, ReplayConfig { shape, ..shipped }));
    }
    rows
}

fn cell(out: &ReplayOutcome) -> String {
    if out.kicks == 0 {
        format!("–  {:>5.2}", out.max_speed_after)
    } else {
        format!("K{} {:>5.2}", out.kicks, out.worst_kick.2)
    }
}

pub(crate) fn run(args: Args) -> Result<()> {
    let snaps: Vec<PlantSnapshot> = args
        .state
        .iter()
        .map(|p| PlantSnapshot::load(p).map_err(anyhow::Error::from))
        .collect::<Result<_>>()?;
    let rows = rows();
    let results: Vec<Vec<ReplayOutcome>> = rows
        .iter()
        .map(|r| snaps.iter().map(|s| s.replay(&r.cfg)).collect())
        .collect();

    for (snap, out) in snaps.iter().zip(&results[0]) {
        let part = out.worst_kick.0;
        println!(
            "state after tick {}: E={:.1} J; original tick {} max part speed {:.2} m/s; self-check dev {:.3}; worst kick part {} ({:?}) {:.2}→{:.2} m/s",
            snap.tick,
            snap.energy(),
            snap.tick + 1,
            snap.original_max_speed(),
            out.max_dev_from_original,
            part,
            snap.part_joint(part),
            out.worst_kick.1,
            out.worst_kick.2
        );
        for c in &out.worst_kick_contacts {
            println!(
                "    contact with {}: {} pts, penetration {:.4} m, normal on kicked ({:+.2},{:+.2},{:+.2}), impulse {:.4}",
                match c.other {
                    None => "terrain".to_string(),
                    Some(0) => "carapace".to_string(),
                    Some(i) => format!("part {i} ({:?})", snap.part_joint(i)),
                },
                c.points,
                c.penetration,
                c.normal_on_kicked.x,
                c.normal_on_kicked.y,
                c.normal_on_kicked.z,
                c.impulse
            );
        }
    }

    println!();
    println!(
        "cell = kick count on the replayed tick + kicked/max part speed after (m/s); '–' = no kick"
    );
    print!("{:<30} {:>9}", "variant (one lever vs shipped)", "survives");
    for s in &snaps {
        print!(" {:>10}", format!("t{}", s.tick + 1));
    }
    println!();
    for (r, outs) in rows.iter().zip(&results) {
        let survived = outs.iter().filter(|o| o.kicks > 0).count();
        print!("{:<30} {:>5}/{:<3}", r.label, survived, outs.len());
        for o in outs {
            print!(" {:>10}", cell(o));
        }
        println!();
    }
    Ok(())
}
