mod bot;
mod debug_sliders;
mod physics;
mod play;
mod player;
mod training;

use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use clap::{Parser, Subcommand};

use training::session::STEPS_PER_ROLLOUT;

/// Crab Combat — RL-trained crab bots learn to stand, walk, and fight.
///
/// Training is the `learn` subcommand (the sole trainer). With no subcommand the
/// binary only renders a trained policy — `--demo` or `--screenshot` — from the
/// flags below. The training knobs live on the `learn` subcommand, so a stray
/// `--workers` on a render invocation is a parse error rather than a silent no-op.
#[derive(Parser, Debug, Clone)]
#[command(version)]
pub struct Cli {
    #[command(flatten)]
    args: Args,

    #[command(subcommand)]
    command: Option<Command>,
}

/// The trainer. One learner (the main thread) owns the policy + optimizer +
/// normalizer; K rollout THREADS each step their own rapier world on their own core
/// and feed buffers back over a channel — wall-clock-parallel rollouts, crash-
/// isolated per worker, with no multiprocess IPC. See `training::inproc`.
#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Run the trainer: spawn K rollout threads, snapshot the policy to each, collect
    /// their rollouts, and run the PPO update. Resumes from `--checkpoint-dir` and
    /// stops at the `--ticks` budget.
    Learn(LearnArgs),
}

/// Training config (consumed by the learner and its rollout threads, which build a
/// `TrainingState`) plus the render modes' shared knobs. The `learn` subcommand
/// flattens it so e.g. `--checkpoint-dir` / `--ticks` mean the same thing
/// everywhere.
#[derive(Parser, Debug, Clone)]
pub struct TrainConfig {
    /// Directory for checkpoint files. On startup, if the directory contains a
    /// previous checkpoint it will be loaded automatically. During training,
    /// checkpoints are saved here periodically and on exit.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Stop training after this many physics ticks (0 = run until killed). The budget
    /// is counted in ticks, never wall-clock, so a run simulates an identical amount
    /// regardless of machine speed or load — the "fixed ticks, not real time"
    /// guarantee an assumed time↔tick relation can't give. The learner checks the
    /// budget once per PPO iteration, so it stops at the first iteration boundary at
    /// or after N (overshooting by up to one K·(--envs)·H iteration's worth of ticks).
    #[arg(long, default_value_t = 0)]
    pub ticks: u64,

    /// Benchmark only: skip NN inference in the train loop (hold zero actions),
    /// isolating physics + engine overhead from network cost. Training is
    /// meaningless under this flag — it exists to measure the per-step bottleneck.
    #[arg(long)]
    pub bench_skip_nn: bool,

    /// Environments M each rollout thread steps in its one world per tick (one
    /// batched NN pass over the M crabs, which sit on a 4 m grid). Total parallel
    /// envs = `--workers` × M. Capped at 16 (the ±10 m arena's grid limit per world).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..=16))]
    pub envs: u64,
}

/// Render-mode config: the shared `TrainConfig` (for `--checkpoint-dir`, the policy
/// the demo/screenshot loads) plus the demo/screenshot/window knobs. Training knobs
/// live on the `learn` subcommand, not here.
#[derive(Parser, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub train: TrainConfig,

    /// Play with a trained crab: load the checkpoint, drive it with the policy
    /// (no learning), orbit camera + poke/reset controls.
    #[arg(long)]
    demo: bool,

    /// Run the demo in a window instead of the default borderless fullscreen.
    #[arg(long)]
    windowed: bool,

    /// Render one frame to this PNG and exit (windowless, GPU on). For inspecting
    /// the trained crab without a display.
    #[arg(long, value_name = "PATH")]
    screenshot: Option<PathBuf>,

    /// Physics steps to simulate before taking the screenshot.
    #[arg(long, default_value_t = 200)]
    screenshot_settle: u32,

    /// Screenshot width in pixels.
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Screenshot height in pixels.
    #[arg(long, default_value_t = 720)]
    height: u32,

    /// Directory the DEMO hot-reloads the policy from while running — the LIVE
    /// training output. The demo loads its initial policy from `--checkpoint-dir`
    /// (a stable copy) and then, every couple of seconds, swaps in a newer
    /// checkpoint that appears here, so a left-open demo tracks training without a
    /// relaunch. Unset, or a missing/half-written dir, = no swap. Demo mode only.
    #[arg(long, value_name = "PATH")]
    live_checkpoint_dir: Option<PathBuf>,

    /// Demo only: replace the trained policy with hands-on gamepad control — D-pad
    /// up/down picks a joint, the right stick drives its torque, all else held at
    /// zero. A physics feel-test, not a learned driver.
    #[arg(long)]
    manual_control: bool,

    /// Demo only: show a TEMPORARY egui panel of physics sliders (contact spring,
    /// length_unit, joint limit spring + leg friction, restitution, substeps) that
    /// live-tune the running crab. A throwaway feel-tuning aid; off by default and
    /// never wired into training/headless. See `src/debug_sliders.rs`.
    #[arg(long)]
    debug_sliders: bool,

    /// DEV: score every live collider against the mesh it stands in for and print a
    /// per-part agreement table (signed surface distance, in model units), then exit
    /// (no window). Exits nonzero if any part fails, so it doubles as a regression
    /// gate on rig changes. Model is `CRAB_MODEL_PATH`, else the dev `sally.glb`.
    #[arg(long)]
    verify_colliders: bool,

    /// DEV: test whether every joint pivot and collider endpoint lies INSIDE the
    /// bind-pose mesh, via the generalized winding number against the model's
    /// triangle soup, then exit (no window). Reports per-point winding number +
    /// signed nearest-surface distance and ranks the worst out-of-mesh offenders.
    /// Model is `CRAB_MODEL_PATH`, else the dev `sally.glb`.
    #[arg(long)]
    verify_pivots: bool,

    /// DEV: spawn the crab, settle it to rest, then test every pair of body
    /// colliders for interpenetration at the settled pose and flag any overlap the
    /// solver is actively fighting — a collision the rig didn't suppress as a jointed
    /// anchor or a group-filtered nested link. Those expected overlaps are reported
    /// but never failed. Exits nonzero on any illegal one, so it gates rig changes.
    #[arg(long)]
    check_rest_colliders: bool,
}

/// Learner orchestration: the shared training config plus how many rollout threads
/// to fan out.
#[derive(Parser, Debug, Clone)]
struct LearnArgs {
    #[command(flatten)]
    train: TrainConfig,

    /// Number of rollout threads K, each stepping its own world on its own core.
    /// Default is PHYSICAL cores minus 2 (floored at one) — physical, not logical,
    /// so it never oversubscribes a hyperthreaded pair onto one core — leaving the
    /// rest of the machine a couple of cores. Pass an explicit value to use more.
    /// Clamped to 1..=64.
    #[arg(long)]
    workers: Option<usize>,

    /// Rollout horizon H: physics ticks each thread rolls per iteration before
    /// handing its buffers back. Per-iteration sample count is K·(--envs)·H.
    #[arg(long, default_value_t = STEPS_PER_ROLLOUT as u64)]
    horizon: u64,

    /// Stop after this many PPO iterations (0 = unbounded). A benchmark / A-B knob;
    /// the production budget is `--ticks` (total physics ticks). Whichever limit is
    /// hit first stops the run.
    #[arg(long, default_value_t = 0)]
    iters: u64,

    /// Niceness applied to the whole process — the learner and its rollout threads
    /// share it (POSIX priority is per-process; higher = yields more CPU). Positive
    /// so a foreground game always preempts training even when the threads saturate
    /// their cores. Clamped to 0..=19 (0 disables; a negative nice would raise
    /// priority and needs privilege, so it is floored to 0 rather than attempted).
    #[arg(long, default_value_t = 10)]
    nice: i32,
}

/// What the no-subcommand binary is rendering this run (training is `rl learn`,
/// which never builds an `AppMode`). Both modes always render.
#[derive(Clone)]
enum AppMode {
    Demo,
    Screenshot { path: PathBuf, settle: u32 },
}

/// Whether to spawn visual assets (meshes, lights). The `rl learn` rollout worlds
/// set this false (rendering off entirely); the rendering modes here set it true.
#[derive(Resource, Clone, Copy)]
pub struct Visuals(pub bool);

fn main() {
    let cli = Cli::parse();

    // The `learn` entry point short-circuits the normal Bevy app: the learner steps
    // no world itself (it owns the policy and runs PPO), and spawns K rollout
    // threads that each drive their own headless app by hand. See `training::inproc`.
    if let Some(Command::Learn(l)) = cli.command {
        // run_learner owns nicing (it lowers process priority before building any
        // world) so a foreground game preempts training.
        training::inproc::run_learner(
            &l.train,
            training::inproc::default_workers(l.workers),
            l.horizon,
            l.iters,
            l.nice,
        );
        return;
    }

    let args = cli.args;

    // DEV verify: score the live colliders against the mesh, print, exit.
    if args.verify_colliders {
        std::process::exit(verify_colliders());
    }

    // DEV verify: test joint pivots + collider endpoints for mesh containment, exit.
    if args.verify_pivots {
        std::process::exit(verify_pivots());
    }

    // Every remaining mode spawns the rig-derived body, which needs the glTF
    // skeleton. Resolve + load it now so a missing or corrupt model fails fast with
    // the real reason, instead of panicking deep in Startup (or blaming
    // CRAB_MODEL_PATH for a parse error in a model that was actually present).
    match bot::meshfit::model_path() {
        None => {
            eprintln!(
                "crab model not found: set CRAB_MODEL_PATH (an asset path under BEVY_ASSET_ROOT/assets), or place sally.glb there"
            );
            std::process::exit(1);
        }
        Some(p) => match bot::meshfit::LoadedModel::load(&p) {
            Err(e) => {
                eprintln!("crab model {p:?}: {e}");
                std::process::exit(1);
            }
            // A model that loads but lacks the expected crab bones builds no recipe.
            // Reject it here, not as `spawn_crab`'s expect deep in Startup with a
            // message that wrongly blames a missing/corrupt file.
            Ok(model) => {
                if bot::rig::build_recipe(&model).is_none() {
                    eprintln!(
                        "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
                    );
                    std::process::exit(1);
                }
            }
        },
    }

    // DEV check: settle the crab and audit its rest-pose colliders for illegal
    // interpenetration, then exit. Placed after the model preflight so a missing
    // model fails with the message above, not a spawn panic deep in the check.
    if args.check_rest_colliders {
        std::process::exit(bot::collider_check::run());
    }

    let mode = if let Some(path) = args.screenshot.clone() {
        AppMode::Screenshot {
            path,
            settle: args.screenshot_settle,
        }
    } else if args.demo {
        AppMode::Demo
    } else {
        // Training is `rl learn` (the sole trainer) — the no-subcommand binary only
        // renders (demo/screenshot). A bare `rl` with no mode flag has nothing to do.
        eprintln!(
            "no mode selected. Train with `rl learn` (the sole trainer); the bare \
             binary needs --demo or --screenshot."
        );
        std::process::exit(2);
    };

    let mut app = App::new();

    match &mode {
        AppMode::Screenshot { .. } => {
            // No window, but GPU ON so we can render to an image. Real-time 60 Hz
            // loop at default 1x: one physics step + one render per frame, so the
            // capture counter (render frames) also tracks simulated time and the
            // GPU pipeline warms over the same frames. (A fast/100x clock decouples
            // them — physics races while render frames crawl, and early frames
            // render black before the pipeline warms up.)
            app.add_plugins(
                DefaultPlugins
                    .set(bevy::window::WindowPlugin {
                        primary_window: None,
                        exit_condition: bevy::window::ExitCondition::DontExit,
                        ..default()
                    })
                    .disable::<bevy::winit::WinitPlugin>(),
            );
            app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
                Duration::from_secs_f64(1.0 / 60.0),
            ));
            // Rapier collider wireframes draw via gizmos, which DO render into the
            // offscreen screenshot camera (Bevy 0.18) — but only if the plugin is
            // present. The other arms add it; Screenshot has its own arm, so gate it
            // here on the same flag or the captured PNG never shows the colliders.
            if std::env::var_os("RL_DEBUG_COLLIDERS").is_some() {
                app.add_plugins(RapierDebugRenderPlugin {
                    // Collider shapes only — the default also draws per-body axes +
                    // joint markers, which on a 31-part body is an unreadable tangle.
                    mode: DebugRenderMode::COLLIDER_SHAPES,
                    ..default()
                });
                // Bright always-in-front markers at each joint pivot — the companion
                // to the collider cage, gated on the same flag (see body.rs).
                bot::body::register_pivot_markers(&mut app);
            }
        }
        AppMode::Demo => {
            // The demo defaults to borderless fullscreen (the Steam launch target is
            // a couch screen) unless --windowed.
            let fullscreen = !args.windowed;
            app.add_plugins(DefaultPlugins.set(bevy::window::WindowPlugin {
                primary_window: Some(bevy::window::Window {
                    title: "Crab RL".into(),
                    mode: if fullscreen {
                        bevy::window::WindowMode::BorderlessFullscreen(
                            bevy::window::MonitorSelection::Primary,
                        )
                    } else {
                        bevy::window::WindowMode::Windowed
                    },
                    ..default()
                }),
                ..default()
            }));
            // With the stand-in primitive meshes removed, Rapier's debug-render is
            // the only in-engine view of the true colliders, so the skin can be
            // checked against the actual physics shapes. `enabled` only sets the
            // INITIAL state — the demo's right-arrow toggles `DebugRenderContext`
            // live — and the demo starts the cage on iff RL_DEBUG_COLLIDERS is set.
            app.add_plugins(RapierDebugRenderPlugin {
                enabled: std::env::var_os("RL_DEBUG_COLLIDERS").is_some(),
                // Collider shapes only — the default also draws per-body axes +
                // joint markers, which on a 31-part body is an unreadable tangle.
                mode: DebugRenderMode::COLLIDER_SHAPES,
                ..default()
            });
            // Pivot markers gate on RL_DEBUG_COLLIDERS: a deliberate diagnostic, not
            // default chrome.
            if std::env::var_os("RL_DEBUG_COLLIDERS").is_some() {
                bot::body::register_pivot_markers(&mut app);
            }
        }
    }

    // Demo and screenshot always render and drive exactly one crab (visuals on,
    // 1 env — parallel envs are a training concept, and training is `rl learn` only).
    app.insert_resource(Visuals(true))
        .insert_resource(bot::NumEnvs(1))
        // The dt + sub-steps live in one place, physics::fixed_timestep, shared with
        // every headless test and the `rl learn` rollout worlds, so the physics the
        // demo/screenshot renders can't drift from the physics training optimizes.
        .insert_resource(physics::fixed_timestep())
        // Seed Rapier's context with the softened contact spring (physics::
        // CONTACT_SOFTNESS) — one source, shared with training + tests — before the
        // plugin spawns the context from it.
        .insert_resource(physics::rapier_context_init())
        // Run physics IN FixedUpdate, lockstep with the Sense→Think→Act brain loop
        // (which also lives in FixedUpdate): one physics step per brain step, the
        // same coupling the `rl learn` rollout worlds use, so the demo replays the
        // exact dynamics the policy trained under.
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default().in_fixed_schedule())
        .add_plugins(physics::PhysicsWorldPlugin)
        .add_plugins(bot::BotPlugin)
        .add_systems(FixedUpdate, contact_audit);

    match mode {
        AppMode::Demo => {
            app.add_plugins(play::DemoPlugin {
                checkpoint_dir: args.train.checkpoint_dir.clone(),
                live_checkpoint_dir: args.live_checkpoint_dir.clone(),
                manual_control: args.manual_control,
            });
            // TEMPORARY physics-tuning overlay, demo-only and off by default.
            if args.debug_sliders {
                app.add_plugins(debug_sliders::DebugSlidersPlugin);
            }
        }
        AppMode::Screenshot { path, settle } => {
            app.add_plugins(play::ScreenshotPlugin {
                checkpoint_dir: args.train.checkpoint_dir.clone(),
                path,
                settle,
                width: args.width,
                height: args.height,
            });
        }
    }

    app.run();
}

/// DEV `--verify-colliders`: load the model, reconstruct every live collider in
/// bind-pose world, and score it against the mesh vertices it stands in for. Prints
/// a per-part agreement table (signed surface distance, model units) + a worst-
/// offender ranking, and returns a process exit code (0 = all pass, 1 = a part
/// fails or the model is unavailable) so it serves as both a diagnostic and a
/// regression gate.
fn verify_colliders() -> i32 {
    use bot::meshfit::{score_box, score_capsule};
    use bot::rig::RestShape;

    let Some(model_path) = bot::meshfit::model_path() else {
        eprintln!(
            "verify-colliders: no model — set CRAB_MODEL_PATH or place sally.glb at the dev path"
        );
        return 1;
    };
    let model = match bot::meshfit::LoadedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("verify-colliders: load {model_path:?}: {e}");
            return 1;
        }
    };
    let Some(recipe) = bot::rig::build_recipe(&model) else {
        eprintln!("verify-colliders: model built no rig recipe");
        return 1;
    };
    let clouds = model.vertices_by_part();
    let trunk = model.vertices_for_bones(&bot::rig::TRUNK_BONES);

    println!("collider<->mesh agreement (model units; +out = mesh pokes OUT of collider):");
    println!(
        "  {:<22} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>6}  {:>7}",
        "part", "n", "r", "fOut%", "pk95", "pkMax", "bulge", "skew", "rRat", "verdict"
    );

    let fmt = |x: f32| {
        if x.is_finite() {
            format!("{x:.2}")
        } else {
            "-".to_string()
        }
    };
    // (label, severity = pk95/r, failed) for the worst-offender ranking.
    let mut ranking: Vec<(String, f32, bool)> = Vec::new();
    let mut any_fail = false;

    for rc in bot::rig::rest_colliders(&model, &recipe) {
        let label = format!("{:?}", rc.part);
        let (score, rnorm, fail) = match rc.shape {
            RestShape::Capsule { a, b, radius } => {
                let pts = clouds
                    .get(&rc.part)
                    .map(|(p, _)| p.as_slice())
                    .unwrap_or(&[]);
                let s = score_capsule(pts, a, b, radius);
                // Pass: little flesh escapes, the worst poke is shallow vs the part's
                // own radius, the collider isn't grossly oversized, the axis tracks
                // the limb, and the radius isn't starved/ballooned.
                let fail = s.frac_outside > 0.05
                    || s.poke_out_p95 > (0.15 * radius).max(0.005)
                    || s.bulge_p95 > 0.5 * radius
                    || (s.axis_skew_deg.is_finite() && s.axis_skew_deg > 15.0)
                    || (s.radius_ratio.is_finite() && !(0.85..=1.4).contains(&s.radius_ratio));
                (s, radius.max(1e-3), fail)
            }
            RestShape::Cuboid { center, half } => {
                let s = score_box(&trunk, center, half);
                // A box over-covering the shell is cosmetically fine; only flag flesh
                // escaping it (absolute, since a box has no single radius).
                let fail = s.frac_outside > 0.03 || s.poke_out_p95 > 0.02;
                (s, half.min_element().max(1e-3), fail)
            }
        };
        any_fail |= fail;
        ranking.push((label.clone(), score.poke_out_p95 / rnorm, fail));
        println!(
            "  {:<22} {:>5} {:>6.3} {:>6.1} {:>6.3} {:>6.3} {:>6.3} {:>5} {:>6}  {}",
            label,
            score.n,
            rnorm,
            score.frac_outside * 100.0,
            score.poke_out_p95,
            score.poke_out_max,
            score.bulge_p95,
            fmt(score.axis_skew_deg),
            fmt(score.radius_ratio),
            if fail { "FAIL" } else { "pass" },
        );
    }

    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst: Vec<String> = ranking
        .iter()
        .take(6)
        .map(|(l, s, f)| format!("{l} {:.2}{}", s, if *f { "!" } else { "" }))
        .collect();
    println!("worst (pk95/r): {}", worst.join(", "));
    println!(
        "{}",
        if any_fail {
            "VERDICT: FAIL — some colliders sit off the mesh"
        } else {
            "VERDICT: pass"
        }
    );
    i32::from(any_fail)
}

/// DEV `--verify-pivots`: empirically test whether each joint pivot and each fitted
/// collider endpoint lies INSIDE the crab's bind-pose visual mesh. Loads the model's
/// triangle soup (bind-world-skinned, same frame as the bone origins + clouds), then
/// for every query point computes the generalized winding number (inside/outside,
/// robust to a non-watertight mesh) and the signed nearest-surface distance (how far
/// in/out). Prints a per-link table + a worst-offender ranking, and exits 0/1.
fn verify_pivots() -> i32 {
    use bot::rig::RestShape;

    let Some(model_path) = bot::meshfit::model_path() else {
        eprintln!(
            "verify-pivots: no model — set CRAB_MODEL_PATH or place sally.glb at the dev path"
        );
        return 1;
    };
    let model = match bot::meshfit::LoadedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("verify-pivots: load {model_path:?}: {e}");
            return 1;
        }
    };
    let mesh = match bot::meshfit::load_bind_mesh(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("verify-pivots: load mesh {model_path:?}: {e}");
            return 1;
        }
    };
    let Some(recipe) = bot::rig::build_recipe(&model) else {
        eprintln!("verify-pivots: model built no rig recipe");
        return 1;
    };
    let pos = &mesh.positions;
    let tris = &mesh.triangles;

    let (lo, hi) = bot::meshfit::aabb(pos);

    // Mesh-containment probe over the bind soup — the same single path the skin-diag
    // audit uses. Its `orient` is the global winding sign from the soup's signed
    // volume: makes interior points read +1 whatever the triangle order, without
    // trusting any single "is this inside?" probe. The crab's vertex centroid sits in
    // a cavity (legs splayed, hollow shell), so it reads ~0 and is useless as the
    // orientation reference — the earlier bug. The carapace pivot (leg-hub centroid,
    // deep in the thorax) is the honest interior probe, used below only to *report*
    // the self-check, not to set the sign.
    let soup = bot::meshfit::MeshContainment::new(pos, tris);
    let signed_vol = soup.signed_vol();
    let orient = soup.orient();
    // Adapt the verdict to this reporter's `(wn, signed_dist, inside)` table layout.
    let probe = |p: Vec3| {
        let c = soup.probe(p);
        (c.wn, c.signed_dist, c.inside)
    };

    // Self-checks. Interior reference = the leg-hub centroid (the carapace pivot the
    // rig anchors every limb to), which is solidly inside the body shell; it must
    // read ~+1. A point 10 units past the bbox must read ~0.
    let hub = bot::rig::rest_colliders(&model, &recipe)
        .iter()
        .find(|rc| rc.part == bot::meshfit::PartId::Carapace)
        .map(|rc| rc.pivot)
        .unwrap_or((lo + hi) * 0.5);
    let centroid = pos.iter().copied().sum::<Vec3>() / pos.len().max(1) as f32;
    let far = hi + (hi - lo).max(Vec3::splat(1.0)) + Vec3::splat(10.0);
    let (hub_wn, _, _) = probe(hub);
    let (cen_wn, _, _) = probe(centroid);
    let (far_wn, _, _) = probe(far);

    println!(
        "mesh: {} verts, {} triangles, bbox {:.3}..{:.3}, signed_vol={:.4}",
        pos.len(),
        tris.len(),
        lo,
        hi,
        signed_vol
    );
    println!(
        "self-check: hub(interior) wn={:+.3} (expect ~+1), vertex-centroid wn={:+.3} (in a cavity → ~0 ok), far-point wn={:+.3} (expect ~0){}",
        hub_wn,
        cen_wn,
        far_wn,
        if orient < 0.0 {
            "  [triangle winding is CW/flipped — normalised via signed volume]"
        } else {
            ""
        }
    );

    println!();
    println!("per-link containment (signed dist: + = OUTSIDE mesh, - = inside):");
    println!(
        "  {:<24} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4}",
        "link", "piv.wn", "piv.dist", "in?", "a.wn", "a.dist", "in?", "b.wn", "b.dist", "in?"
    );

    // (label, signed outside distance) for the worst-offender ranking; only OUTSIDE
    // points (positive signed distance) are offenders. `windings` collects every
    // query point's winding so the watertight verdict can measure how tightly they
    // cluster at integers (clean) vs scatter fractionally (open/non-manifold).
    let mut pivots_out = 0usize;
    let mut endpoints_out = 0usize;
    let mut offenders: Vec<(String, f32)> = Vec::new();
    let mut windings: Vec<f32> = Vec::new();
    let yn = |b: bool| if b { "IN" } else { "OUT" };

    for rc in bot::rig::rest_colliders(&model, &recipe) {
        let label = format!("{:?}", rc.part);
        let (pwn, pdist, pin) = probe(rc.pivot);
        windings.push(pwn);
        if !pin {
            pivots_out += 1;
            offenders.push((format!("{label} pivot"), pdist));
        }
        match rc.shape {
            RestShape::Capsule { a, b, .. } => {
                let (awn, adist, ain) = probe(a);
                let (bwn, bdist, bin) = probe(b);
                windings.push(awn);
                windings.push(bwn);
                for (tag, inside, dist) in [("a", ain, adist), ("b", bin, bdist)] {
                    if !inside {
                        endpoints_out += 1;
                        offenders.push((format!("{label} {tag}"), dist));
                    }
                }
                println!(
                    "  {:<24} | {:>+7.3} {:>+8.4} {:>4} | {:>+7.3} {:>+8.4} {:>4} | {:>+7.3} {:>+8.4} {:>4}",
                    label,
                    pwn,
                    pdist,
                    yn(pin),
                    awn,
                    adist,
                    yn(ain),
                    bwn,
                    bdist,
                    yn(bin)
                );
            }
            RestShape::Cuboid { center, half } => {
                // The carapace box has no segment endpoints; test its 8 corners + the
                // center so we still learn whether the box surface escapes the shell.
                println!(
                    "  {:<24} | {:>+7.3} {:>+8.4} {:>4} | {:>7} {:>8} {:>4} | {:>7} {:>8} {:>4}",
                    label,
                    pwn,
                    pdist,
                    yn(pin),
                    "(box)",
                    "corners",
                    "↓",
                    "",
                    "",
                    ""
                );
                for sx in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        for sz in [-1.0f32, 1.0] {
                            let corner = center + half * Vec3::new(sx, sy, sz);
                            let (cwn, cdist, cin) = probe(corner);
                            windings.push(cwn);
                            if !cin {
                                endpoints_out += 1;
                                offenders.push((
                                    format!("{label} corner({sx:+.0},{sy:+.0},{sz:+.0})"),
                                    cdist,
                                ));
                            }
                            println!(
                                "      corner ({:+.0},{:+.0},{:+.0})         | {:>+7.3} {:>+8.4} {:>4}",
                                sx,
                                sy,
                                sz,
                                cwn,
                                cdist,
                                yn(cin)
                            );
                        }
                    }
                }
                let (ccwn, ccdist, ccin) = probe(center);
                println!(
                    "      center                       | {:>+7.3} {:>+8.4} {:>4}",
                    ccwn,
                    ccdist,
                    yn(ccin)
                );
            }
        }
    }

    // Watertight verdict: a clean closed mesh makes every winding land near an
    // integer (0 outside, ±1 inside). Count query points whose winding is clearly
    // fractional (off the nearest integer by >0.1) — many ⇒ the surface is open or
    // non-manifold and the IN/OUT calls near the boundary are soft.
    let fractional = windings
        .iter()
        .filter(|&&w| (w - w.round()).abs() > 0.1)
        .count();
    let clean = (hub_wn > 0.9) && (far_wn.abs() < 0.1) && fractional == 0;

    offenders.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    println!();
    println!(
        "watertight: {} — {}/{} query windings are fractional (off integer by >0.1); interior wn={:+.3}, exterior wn={:+.3}",
        if clean { "CLEAN/closed" } else { "MESSY/open" },
        fractional,
        windings.len(),
        hub_wn,
        far_wn
    );
    println!(
        "SUMMARY: {pivots_out} pivot(s) OUTSIDE mesh, {endpoints_out} endpoint/corner(s) OUTSIDE mesh"
    );
    println!("worst offenders (model units outside the surface):");
    for (label, d) in offenders.iter().take(12) {
        println!("  {:<34} {:+.4}", label, d);
    }
    if offenders.is_empty() {
        println!("  (none — every query point is inside the mesh)");
    }

    let pass = pivots_out == 0;
    println!(
        "VERDICT: {}",
        if pass {
            "all joint pivots lie INSIDE the mesh"
        } else {
            "some joint pivots lie OUTSIDE the mesh — see ranking"
        }
    );
    i32::from(!pass)
}

/// Diagnostic (enable with RL_CONTACT_AUDIT=1): every 64 ticks, prints every
/// crab-part-vs-crab-part contact pair currently penetrating more than 5 mm,
/// deepest first. Ground contacts are excluded. Answers "are the legs
/// actually intersecting" with numbers instead of squinting at renders.
fn contact_audit(
    sim: Query<&bevy_rapier3d::plugin::context::RapierContextSimulation>,
    cols: Query<&bevy_rapier3d::plugin::context::RapierContextColliders>,
    parts: Query<
        (Option<&bot::body::CrabJoint>, Has<bot::body::CrabCarapace>),
        With<bot::body::CrabBodyPart>,
    >,
    mut tick: Local<u32>,
) {
    if std::env::var_os("RL_CONTACT_AUDIT").is_none() {
        return;
    }
    *tick += 1;
    if *tick % 64 != 2 {
        return;
    }
    let (Ok(sim), Ok(cols)) = (sim.single(), cols.single()) else {
        return;
    };
    let name = |p: (Option<&bot::body::CrabJoint>, bool)| {
        p.0.map(|j| format!("{:?}", j.id))
            .unwrap_or_else(|| "Carapace".to_string())
    };
    let mut worst: Vec<(f32, String, String)> = Vec::new();
    for pair in sim.narrow_phase.contact_pairs() {
        let (Some(e1), Some(e2)) = (
            cols.collider_entity(pair.collider1),
            cols.collider_entity(pair.collider2),
        ) else {
            continue;
        };
        let (Ok(p1), Ok(p2)) = (parts.get(e1), parts.get(e2)) else {
            continue;
        };
        let mut depth = 0.0f32;
        for m in &pair.manifolds {
            for pt in &m.points {
                depth = depth.max(-pt.dist);
            }
        }
        if depth > 0.005 {
            worst.push((depth, name(p1), name(p2)));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "AUDIT tick {}: {} crab-crab pairs >5mm penetration",
        *tick,
        worst.len()
    );
    for (d, a, b) in worst.iter().take(6) {
        println!("  {:>4.0}mm {a} vs {b}", d * 1000.0);
    }
}

