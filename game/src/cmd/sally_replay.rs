use anyhow::Result;
use clap::Parser;

use crab_world::bot::body::{CrabJointId, LIMIT_SOFTNESS, Side};
use crab_world::physics::snapshot::{
    ContactsOff, FreeLimits, PlantSnapshot, ReplayConfig, ReplayOutcome, ShapeVariant,
    SpringCoefficients,
};
use crab_world::physics::{CONTACT_SOFTNESS, SHIPPED_SOLVER, SolverCounts};

/// rl#332 T1: replay ONE tick from each `sally-soak --dump-state-at` or
/// `rl-train repro --capture-dir` snapshot, varying ONE lever at a time against the
/// baseline configuration — drives, solver counts, joint limit spring, one claw
/// joint's or every joint's limits, terrain or all contacts, link collider shape
/// (masses pinned; the `massΔ` column proves it) — and print whether the recorded kick
/// survives. The first row is the self-check: baseline configuration, recorded
/// drives — it must reproduce the original run.
#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "FILE", required = true, num_args = 1..)]
    state: Vec<std::path::PathBuf>,

    /// The solver the captures ran under (`rl-train repro --solver`): the self-check
    /// row and every other lever's base. A snapshot stores the counts but not the
    /// substeps, so this names both.
    #[arg(long, default_value_t = SHIPPED_SOLVER)]
    solver: SolverCounts,
}

struct Row {
    label: String,
    cfg: ReplayConfig,
}

fn rows(base: SolverCounts) -> Vec<Row> {
    let baseline = ReplayConfig {
        drive_scale: 1.0,
        solver: base,
        limit_softness: None,
        free_limits: FreeLimits::None,
        contacts_off: ContactsOff::None,
        shape: ShapeVariant::AsIs,
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
    let mut rows = vec![row(&format!("self-check ({base})"), baseline)];
    for (label, scale) in [("drives zeroed", 0.0), ("drives ×0.5", 0.5)] {
        rows.push(row(
            label,
            ReplayConfig {
                drive_scale: scale,
                ..baseline
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
        ((2, 11, 3), 2),
        ((2, 13, 3), 2),
        ((2, 12, 0), 2),
        ((2, 12, 2), 2),
        ((2, 12, 4), 2),
    ] {
        let solver = SolverCounts {
            iterations,
            substeps,
        };
        rows.push(row(
            &format!("solver {solver}"),
            ReplayConfig { solver, ..baseline },
        ));
    }
    rows.push(row(
        "limit spring 40 Hz",
        ReplayConfig {
            limit_softness: soft(40.0, LIMIT_SOFTNESS.damping_ratio),
            ..baseline
        },
    ));
    rows.push(row(
        "limit spring = contact class",
        ReplayConfig {
            limit_softness: soft(
                CONTACT_SOFTNESS.natural_frequency,
                CONTACT_SOFTNESS.damping_ratio,
            ),
            ..baseline
        },
    ));
    for id in [Side::Left, Side::Right].into_iter().flat_map(|side| {
        [
            CrabJointId::ClawShoulder(side),
            CrabJointId::ClawWrist(side),
            CrabJointId::ClawPincer(side),
        ]
    }) {
        rows.push(row(
            &format!("limits off: {id:?}"),
            ReplayConfig {
                free_limits: FreeLimits::Joint(id),
                ..baseline
            },
        ));
    }
    rows.push(row(
        "all joint limits off",
        ReplayConfig {
            free_limits: FreeLimits::All,
            ..baseline
        },
    ));
    for (label, contacts_off) in [
        ("terrain contact off", ContactsOff::Terrain),
        ("all contacts off", ContactsOff::All),
    ] {
        rows.push(row(
            label,
            ReplayConfig {
                contacts_off,
                ..baseline
            },
        ));
    }
    for (label, shape) in [
        ("cuboids → capsules", ShapeVariant::CuboidsToCapsules),
        ("all links thin balls", ShapeVariant::Balls { fat: false }),
        ("all links fat balls", ShapeVariant::Balls { fat: true }),
    ] {
        rows.push(row(label, ReplayConfig { shape, ..baseline }));
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
    for (path, s) in args.state.iter().zip(&snaps) {
        let p = &s.params;
        let counts = (
            p.num_solver_iterations,
            p.num_internal_pgs_iterations,
            p.num_internal_stabilization_iterations,
        );
        if counts != args.solver.iterations {
            println!(
                "WARNING {} ran with counts {counts:?}, not --solver {}: its self-check will not reproduce",
                path.display(),
                args.solver
            );
        }
    }
    let rows = rows(args.solver);
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
    print!(
        "{:<30} {:>9} {:>7}",
        "variant (one lever vs base)", "survives", "massΔ"
    );
    for s in &snaps {
        print!(" {:>10}", format!("t{}", s.tick + 1));
    }
    println!();
    for (r, outs) in rows.iter().zip(&results) {
        let survived = outs.iter().filter(|o| o.kicks > 0).count();
        let mass_dev = outs
            .iter()
            .map(|o| o.max_mass_props_dev)
            .fold(0.0, f32::max);
        print!(
            "{:<30} {:>5}/{:<3} {:>7.1e}",
            r.label,
            survived,
            outs.len(),
            mass_dev
        );
        for o in outs {
            print!(" {:>10}", cell(o));
        }
        println!();
    }
    Ok(())
}
