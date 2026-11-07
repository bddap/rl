//! Drop the rigged Sally `.glb` onto flat ground with every joint passive and report
//! where it comes to rest: a check that the bodies and hinges derived from its
//! skeleton hold together under gravity.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_rapier3d::prelude::TimestepMode;
use clap::Parser;
use crab_world::physics::{PHYSICS_DT, PHYSICS_HZ};
use crab_world::sally::{SUBSTEPS, Skeleton};

#[derive(Parser)]
struct Args {
    rig: PathBuf,
    #[arg(long, default_value_t = 4.0)]
    seconds: f32,
    #[arg(long, default_value_t = 0.01)]
    drop_m: f32,
    #[arg(long, default_value_t = SUBSTEPS)]
    substeps: usize,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let glb = std::fs::read(&args.rig).map_err(|e| format!("{}: {e}", args.rig.display()))?;
    let skeleton = Skeleton::from_glb(&glb)?;
    let lowest = skeleton.lowest_point(&skeleton.rest_poses(Transform::IDENTITY));
    let at = Transform::from_xyz(0.0, args.drop_m - lowest, 0.0);

    let (mut app, entities) = skeleton.world(at);
    app.insert_resource(TimestepMode::Fixed {
        dt: PHYSICS_DT,
        substeps: args.substeps,
    });
    let poses = |app: &App| -> Vec<Transform> {
        entities
            .iter()
            .map(|&e| *app.world().get::<Transform>(e).unwrap())
            .collect()
    };

    app.update();
    let first = poses(&app);
    let start = skeleton.limit_excess(&first);
    let ticks = (args.seconds * PHYSICS_HZ as f32) as usize;
    let mut worst = 0.0f32;
    for _ in 1..ticks {
        app.update();
        let e = skeleton.limit_excess(&poses(&app));
        worst = worst.max(e);
    }
    let end = poses(&app);
    let finite = end
        .iter()
        .all(|t| t.translation.is_finite() && t.rotation.is_finite());
    let body = end[0].translation;
    let lowest_now = skeleton.lowest_point(&end);
    let report = serde_json::json!({
        "bones": skeleton.bones.len(),
        "seconds": args.seconds,
        "substeps": args.substeps,
        "limit_excess_deg_at_spawn": start.to_degrees(),
        "limit_excess_deg_worst": worst.to_degrees(),
        "limit_excess_deg_final": skeleton.limit_excess(&end).to_degrees(),
        "body_height_m": [first[0].translation.y, body.y],
        "body_drift_m": (body - first[0].translation).xz().length(),
        "lowest_point_m": lowest_now,
        "finite": finite,
    });
    println!("SALLY_DROP {report}");
    if !finite || worst > 0.0 {
        return Err("the derived bodies did not hold together".into());
    }
    Ok(())
}
