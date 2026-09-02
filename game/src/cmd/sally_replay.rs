use anyhow::Result;
use clap::Parser;

use crab_world::bot::body::LIMIT_SOFTNESS;
use crab_world::physics::CONTACT_SOFTNESS;
use crab_world::physics::snapshot::{PlantSnapshot, ReplayConfig, SpringCoefficients};

/// rl#332 T1: replay ONE tick from a `sally-soak --dump-state-at` snapshot under
/// every combination of {drives as recorded | zeroed} × solver counts × joint
/// limit spring, and print where the tick's energy and speed jumps went. The
/// first row is the self-check: shipped configuration, recorded drives — it must
/// reproduce the original run.
#[derive(Parser)]
pub(crate) struct Args {
    #[arg(long, value_name = "FILE")]
    state: std::path::PathBuf,
}

pub(crate) fn run(args: Args) -> Result<()> {
    let snap = PlantSnapshot::load(&args.state)?;
    let shipped = (
        crab_world::physics::SOLVER_ITERATIONS,
        crab_world::physics::PHYSICS_SUBSTEPS,
    );
    let orig_max = snap.original_max_speed();
    println!(
        "sally-replay: state after tick {} ({} parts, {} joints), E={:.1} J; original run's tick {} ended with max part speed {:.2} m/s",
        snap.tick,
        snap.parts.len(),
        snap.joints.len(),
        snap.energy(),
        snap.tick + 1,
        orig_max
    );
    println!(
        "{:<8} {:<12} {:<9} {:<8} {:>8} {:>8} {:>8} {:>8} {:>5} {:>26} {:>9}",
        "drives",
        "iters×sub",
        "limit",
        "",
        "ΔE J",
        "v0 max",
        "v1 max",
        "ω1 max",
        "kicks",
        "worst kick (part v0→v1)",
        "dev orig"
    );
    let softs = [
        ("400Hz", None),
        ("40Hz", Some((40.0, LIMIT_SOFTNESS.damping_ratio))),
        (
            "contact",
            Some((
                CONTACT_SOFTNESS.natural_frequency,
                CONTACT_SOFTNESS.damping_ratio,
            )),
        ),
    ];
    let sweep = [200.0, 120.0, 80.0, 60.0];
    for zero_drives in [false, true] {
        for (iterations, substeps) in [
            shipped,
            ((2, 2, 3), 2),
            ((2, 4, 3), 2),
            ((2, 12, 3), 2),
            ((2, 16, 3), 2),
            ((2, 32, 3), 2),
            ((3, 8, 3), 2),
            ((4, 8, 3), 2),
            ((2, 8, 3), 4),
            ((8, 4, 4), 4),
            ((32, 8, 8), 8),
        ] {
            let fine: Vec<(String, Option<(f32, f32)>)> = if (iterations, substeps) == shipped {
                sweep
                    .iter()
                    .map(|hz| (format!("{hz}Hz"), Some((*hz, LIMIT_SOFTNESS.damping_ratio))))
                    .collect()
            } else {
                Vec::new()
            };
            let rows = softs.iter().map(|(l, hz)| (l.to_string(), *hz)).chain(fine);
            for (label, limit_hz) in rows {
                let cfg = ReplayConfig {
                    zero_drives,
                    iterations,
                    substeps,
                    limit_softness: limit_hz.map(|(f, z)| SpringCoefficients {
                        natural_frequency: f,
                        damping_ratio: z,
                    }),
                };
                let out = snap.replay(&cfg);
                println!(
                    "{:<8} {:<12} {:<9} {:<8} {:>+8.1} {:>8.2} {:>8.2} {:>8.1} {:>5} {:>26} {:>9.3}",
                    if zero_drives { "zeroed" } else { "as-is" },
                    format!("{:?}×{}", iterations, substeps),
                    label,
                    "",
                    out.energy_after - out.energy_before,
                    out.max_speed_before,
                    out.max_speed_after,
                    out.max_angvel_after,
                    out.kicks,
                    format!(
                        "{} {:.2}→{:.2}",
                        out.worst_kick.0, out.worst_kick.1, out.worst_kick.2
                    ),
                    out.max_dev_from_original
                );
            }
        }
    }
    Ok(())
}
