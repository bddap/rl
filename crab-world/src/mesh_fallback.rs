use crate::bot;
use std::path::PathBuf;
use std::sync::OnceLock;

pub const MESH_ABSENT_REASON: &str = "no crab model resolved (CRAB_MODEL_PATH / default `sally.glb` not found under \
     BEVY_ASSET_ROOT/assets)";

#[derive(Clone, Copy)]
pub enum Surface {
    RlDemo,
    Game,
    /// The headless trainer/eval, when `--allow-fallback-body` deliberately runs the
    /// procedural body (bddap/rl#214). Without the flag it refuses instead
    /// ([`require_canonical_body`]).
    Trainer,
}

impl Surface {
    fn as_str(self) -> &'static str {
        match self {
            Surface::RlDemo => "rl-demo",
            Surface::Game => "game",
            Surface::Trainer => "rl-train",
        }
    }
}

/// Resolve the crab model file: `CRAB_MODEL_PATH` (absolute, or relative to the
/// asset root) with `sally.glb` under `BEVY_ASSET_ROOT/assets` as the default.
pub fn model_path() -> Option<PathBuf> {
    resolve(
        std::env::var_os("CRAB_MODEL_PATH").as_deref(),
        &crate::assets::asset_root(),
        |p| p.exists(),
    )
}

fn resolve(
    crab_model_path: Option<&std::ffi::OsStr>,
    asset_root: &std::path::Path,
    exists: impl Fn(&std::path::Path) -> bool,
) -> Option<PathBuf> {
    let rel = crab_model_path.map_or_else(|| PathBuf::from("sally.glb"), PathBuf::from);
    if rel.is_absolute() {
        return exists(&rel).then_some(rel);
    }
    let asset = asset_root.join("assets").join(rel);
    exists(&asset).then_some(asset)
}

/// A verified-usable crab model: the path whose bytes matched
/// [`bot::rig::BAKED_ASSET_DIGEST`] (any other byte state is refused, so there is no
/// digest field to carry — the const IS the digest) and the baked collider recipe.
pub struct UsableModel {
    pub path: PathBuf,
    pub recipe: bot::rig::RigRecipe,
}

fn checked_model() -> Result<UsableModel, String> {
    let Some(path) = model_path() else {
        return Err(MESH_ABSENT_REASON.to_string());
    };
    let recipe = checked_recipe(&path)?;
    Ok(UsableModel { path, recipe })
}

/// The bddap/rl#20 Phase 2 gate: the runtime does not fit colliders from the mesh —
/// it verifies the asset's bytes are EXACTLY the ones the committed
/// [`bot::rig::baked_recipe`] table was baked from, then serves that table. Any other
/// byte state (a changed model, a corrupt download) is refused loudly: serving a
/// stale table under a new mesh would silently fork physics from render, and a
/// re-fit is a deliberate offline event (`cargo run -p meshfit -- bake`), never a
/// side effect of swapping a file.
fn checked_recipe(p: &std::path::Path) -> Result<bot::rig::RigRecipe, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("crab model {p:?}: read: {e}"))?;
    let digest = crate::fnv::fnv1a(&bytes);
    if digest != bot::rig::BAKED_ASSET_DIGEST {
        return Err(format!(
            "crab model {p:?}: digest {digest:#018x} does not match the baked collider \
             table ({:#018x}) — the asset changed (or is corrupt) without a re-bake. \
             Fetch the canonical sally.glb (scripts/fetch-sally.sh), or re-bake \
             deliberately (`cargo run -p meshfit -- bake`): a geometry change is a new \
             MDP — review the baked.rs diff and plan a retrain (rl#277)",
            bot::rig::BAKED_ASSET_DIGEST
        ));
    }
    Ok(bot::rig::baked_recipe())
}

pub fn usable_model() -> &'static Result<UsableModel, String> {
    static VERDICT: OnceLock<Result<UsableModel, String>> = OnceLock::new();
    VERDICT.get_or_init(checked_model)
}

pub fn usable_model_path() -> Option<PathBuf> {
    usable_model().as_ref().ok().map(|u| u.path.clone())
}

/// Natural rest-pose height of the crab body THIS process constructs (arena m) — THE
/// scale bridge between her rig and any sized frame: net's arena→sim render seam and
/// the eval's rl#266 charge-speed guard both divide by this one measurement, so the
/// two can never disagree on what "one crab height" is. `None` when the silhouette is
/// degenerate — callers must treat that as unmeasurable, never as scale 1.0 (an
/// identity conversion silently re-opens the rl#254 creep).
pub fn natural_body_height() -> Option<f32> {
    static H: OnceLock<Option<f32>> = OnceLock::new();
    *H.get_or_init(|| {
        let h =
            bot::rig::recipe_silhouette(&bot::body::render_recipe(usable_model_path().is_some()))
                .natural_height();
        (h > 1e-4).then_some(h)
    })
}

/// Digest of the crab body THIS process constructs — THE one "which body is this?"
/// value: the checkpoint body-identity stamp (bddap/rl#214) AND the per-peer collider
/// digest the MP membership handshake advertises (rl#100/rl#114). When the
/// [`usable_model`] verdict is `Ok`, [`bot::rig::baked_body_digest`] — asset bytes
/// chained with the baked collider table's [`bot::rig::RigRecipe::digest`] — else `0`: the
/// fallback/no-asset value, which never counts as synced in the handshake and never
/// passes for Sally at a checkpoint load. Keyed off the VERDICT, not bare
/// `model_path()`: a present-but-unloadable glb constructs the fallback body, so it must
/// advertise/stamp `0`, not the broken file's hash (two identically-corrupt peers must
/// not "agree on the collider asset" while both run not-Sally bodies).
///
/// WHY both halves (bddap/rl#20 stage 1): the asset digest alone certified the render
/// mesh but NOT the body — a `baked.rs` regen (a re-fit of the same sally.glb) changes
/// every collider under an unchanged asset digest, which is a new MDP no live
/// checkpoint can drive (rl#277). Folding the table's own bit-exact digest in makes a
/// table move refuse loudly at trainer resume, eval, demo arm, and the MP handshake —
/// no float-reproducibility risk, since both halves hash compiled constants.
/// Remaining caveat: the binary axis beyond the table stays unguarded — spawn/joint
/// CODE changes still alter the body under an unchanged digest.
pub fn constructed_body_digest() -> u64 {
    if usable_model().is_ok() {
        bot::rig::baked_body_digest()
    } else {
        0
    }
}

/// The trainer/eval body-preflight verdict: proceed (on which body), or refuse. A value
/// of this type is the PROOF the preflight ran — [`crate::training::inproc::run_learner`]
/// and [`crate::eval::run_eval`] take one as a required argument, so a new entry point
/// can't recreate rl#214 by forgetting the gate; production obtains it only from
/// [`require_canonical_body`], and a test writing a `BodyGate::…` literal is a visible,
/// greppable opt-in rather than a silent hole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyGate {
    /// The canonical Sally mesh is usable — the body every checkpoint should train on.
    RealSally,
    /// No usable mesh, but the caller explicitly opted into the procedural fallback.
    FallbackAllowed,
}

/// Pure gate decision, split from [`require_canonical_body`] so the refusal matrix is
/// unit-testable without touching the memoized global verdict or process env.
fn body_gate(
    verdict: Result<(), &str>,
    allow_fallback: bool,
    context: &str,
) -> Result<BodyGate, String> {
    match verdict {
        Ok(()) => Ok(BodyGate::RealSally),
        Err(_) if allow_fallback => Ok(BodyGate::FallbackAllowed),
        Err(reason) => Err(format!(
            "REFUSING to {context}: {reason}. A policy trained or judged on the procedural \
             fallback body is NOT Sally (bddap/rl#214). Fetch the mesh \
             (scripts/fetch-sally.sh, or point CRAB_MODEL_PATH / BEVY_ASSET_ROOT at it), or \
             pass --allow-fallback-body for a deliberate procedural-body dev run."
        )),
    }
}

/// MANDATORY body preflight for training and eval (bddap/rl#214): the trainer used to hit
/// [`crate::bot::body::render_recipe`]'s silent fallback and train the procedural body as
/// if it were Sally. Now `rl-train learn`/`eval` call this before building any world and
/// pass the returned [`BodyGate`] proof down to `run_learner`/`run_eval` — `Err` is the
/// refusal, ALREADY logged loudly here (`tracing::error!`, so it also
/// exports over OTEL; the binary lacks a tracing dep of its own), for the caller to exit
/// nonzero on; `--allow-fallback-body` downgrades it to the SAME latched [`log_fallback`]
/// error the player-facing surfaces raise, plus a warning that checkpoints will carry
/// body digest 0. On the happy path it logs one positive line naming the mesh and
/// digest, so "which body is this run on?" is always answerable from the log.
pub fn require_canonical_body(context: &str, allow_fallback: bool) -> Result<BodyGate, String> {
    let verdict = usable_model();
    let gate = body_gate(
        verdict.as_ref().map(|_| ()).map_err(String::as_str),
        allow_fallback,
        context,
    );
    match (&gate, verdict) {
        (Err(refusal), _) => {
            tracing::error!(target: "crab_world::canonical_mesh", "{refusal}");
        }
        (Ok(BodyGate::RealSally), Ok(u)) => tracing::info!(
            "canonical Sally mesh preflight OK for {context}: {} (body digest {:#018x})",
            u.path.display(),
            constructed_body_digest(),
        ),
        (Ok(BodyGate::FallbackAllowed), Err(reason)) => {
            log_fallback(Surface::Trainer, reason);
            tracing::warn!(
                "--allow-fallback-body: {context} proceeds on the procedural fallback \
                 body (NOT Sally); checkpoints will carry body digest 0"
            );
        }
        (Ok(BodyGate::RealSally), Err(_)) | (Ok(BodyGate::FallbackAllowed), Ok(_)) => {
            unreachable!("gate outcome co-derives with the verdict")
        }
    }
    gate
}

pub fn log_fallback(surface: Surface, reason: &str) {
    tracing::error!(
        target: "crab_world::canonical_mesh",
        surface = %surface.as_str(),
        host = %hostname(),
        reason = %reason,
        "canonical Sally mesh could not be resolved — falling back to the honest collider \
         wireframe (the real physics colliders, NOT the real Sally rig). Fetch it with \
         scripts/fetch-sally.sh or point CRAB_MODEL_PATH at the model."
    );
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

#[cfg(feature = "render")]
pub use banner::spawn_banner;

#[cfg(feature = "render")]
mod banner {
    use bevy::prelude::*;

    const BANNER_HEADLINE: &str =
        "SALLY MESH NOT LOADED — showing physics colliders (NOT the real Sally rig)";

    pub fn spawn_banner(commands: &mut Commands, reason: &str) -> Entity {
        commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(0.0),
                    left: Val::Px(0.0),
                    right: Val::Px(0.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.15, 0.0, 0.0, 0.85)),
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new(BANNER_HEADLINE),
                    TextFont {
                        font_size: 20.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.5, 0.5)),
                ));
                b.spawn((
                    Text::new(format!("{reason}  —  fetch with scripts/fetch-sally.sh")),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgba(0.95, 0.85, 0.85, 0.9)),
                ));
            })
            .id()
    }
}

#[cfg(test)]
mod model_path_tests {
    use super::resolve;
    use std::path::{Path, PathBuf};

    #[test]
    fn relative_resolves_under_asset_root() {
        let got = resolve(Some("sally.glb".as_ref()), Path::new("/srv/app"), |p| {
            p == Path::new("/srv/app/assets/sally.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/srv/app/assets/sally.glb")));
    }

    #[test]
    fn defaults_to_sally_under_asset_root() {
        let got = resolve(None, Path::new("/crate"), |p| {
            p == Path::new("/crate/assets/sally.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/crate/assets/sally.glb")));
    }

    #[test]
    fn absolute_path_used_as_is() {
        let got = resolve(Some("/models/x.glb".as_ref()), Path::new("/srv"), |p| {
            p == Path::new("/models/x.glb")
        });
        assert_eq!(got, Some(PathBuf::from("/models/x.glb")));
    }

    #[test]
    fn none_when_missing() {
        assert_eq!(
            resolve(Some("sally.glb".as_ref()), Path::new("/srv"), |_| false),
            None
        );
    }
}

#[cfg(test)]
mod tests {
    use super::checked_recipe;

    #[test]
    fn present_but_unloadable_glb_is_err_not_panic() {
        let dir = std::env::temp_dir().join(format!("rl154-badglb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("sally.glb");
        std::fs::write(&bad, b"this is not a glb, it is garbage bytes").unwrap();

        let status = checked_recipe(&bad);
        assert!(
            status.is_err(),
            "a garbage present-but-unloadable glb must report Err (rl#154), got a recipe: {:?}",
            status.is_ok()
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod body_gate_tests {
    use super::{BodyGate, body_gate};

    /// The bddap/rl#214 refusal matrix: a usable mesh always proceeds on Sally; without
    /// one, training/eval refuse unless the fallback was explicitly opted into.
    #[test]
    fn refuses_fallback_body_unless_opted_in() {
        assert_eq!(body_gate(Ok(()), false, "learn"), Ok(BodyGate::RealSally));
        assert_eq!(body_gate(Ok(()), true, "learn"), Ok(BodyGate::RealSally));
        assert_eq!(
            body_gate(Err("no mesh"), true, "learn"),
            Ok(BodyGate::FallbackAllowed)
        );
        let refusal = body_gate(Err("no mesh"), false, "learn").unwrap_err();
        assert!(refusal.contains("REFUSING to learn"), "{refusal}");
        assert!(refusal.contains("--allow-fallback-body"), "{refusal}");
    }
}
