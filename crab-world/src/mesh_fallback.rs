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

pub struct UsableModel {
    pub path: PathBuf,
    pub recipe: bot::rig::RigRecipe,
    /// FNV-1a/64 of the exact file bytes the `recipe` was built from — hashed and
    /// parsed from ONE read ([`checked_recipe`]), so the digest can never describe a
    /// different file state than the body actually spawned (bddap/rl#214). Never `0`
    /// (that value means "fallback/no asset", see [`constructed_body_digest`]).
    pub digest: u64,
}

fn checked_model() -> Result<UsableModel, String> {
    let Some(path) = bot::meshfit::model_path() else {
        return Err(MESH_ABSENT_REASON.to_string());
    };
    let (recipe, digest) = checked_recipe(&path)?;
    Ok(UsableModel {
        path,
        recipe,
        digest,
    })
}

fn checked_recipe(p: &std::path::Path) -> Result<(bot::rig::RigRecipe, u64), String> {
    let bytes = std::fs::read(p).map_err(|e| format!("crab model {p:?}: read: {e}"))?;
    let model = bot::meshfit::LoadedModel::from_slice(&bytes)
        .map_err(|e| format!("crab model {p:?}: {e}"))?;
    let recipe = bot::rig::build_recipe(&model).ok_or_else(|| {
        format!(
            "crab model {p:?}: loaded but has none of the expected crab bones (e.g. Def_leg_01.000.L)"
        )
    })?;
    Ok((recipe, crate::fnv::fnv1a(&bytes)))
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
/// digest the MP membership handshake advertises (rl#100/rl#114). The [`usable_model`]
/// verdict's own byte digest when it is `Ok` (the body spawns the mesh-fitted recipe),
/// else `0` — the fallback/no-asset value, which never counts as synced in the handshake
/// and never passes for Sally at a checkpoint load. Keyed off the VERDICT, not bare
/// `model_path()`: a present-but-unloadable glb constructs the fallback body, so it must
/// advertise/stamp `0`, not the broken file's hash (two identically-corrupt peers must
/// not "agree on the collider asset" while both run not-Sally bodies).
///
/// WHY hash the raw file bytes (not the post-load mesh or the fitted capsule spec): the
/// giant crab's rapier colliders are a DETERMINISTIC pure function of this file's bytes
/// GIVEN a fixed binary — glb → skin-to-bind-world → capsule/box fit. The binary is
/// already an unstated GCR baseline (two peers on different binaries desync on
/// everything, not just colliders), so the only collider-affecting input left to guard
/// is the asset — hashed conservatively (any byte change ⇒ mismatch ⇒ refuse) and
/// WITHOUT re-introducing float-reproducibility questions that hashing the skinned
/// vertex cloud or fitted f32 spec would. Same caveat for the checkpoint stamp: it
/// guards the ASSET axis only; a meshfit/rig code change alters the body under an
/// unchanged digest (the binary axis stays unguarded, as in the handshake).
pub fn constructed_body_digest() -> u64 {
    usable_model().as_ref().map(|u| u.digest).unwrap_or(0)
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
            u.digest,
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

#[cfg(feature = "render")]
pub fn initial_render_mode(
    flag: Option<crate::crab_view::RenderMode>,
    surface: Surface,
) -> crate::crab_view::RenderMode {
    use crate::crab_view::RenderMode;
    let mesh_err = usable_model().as_ref().err();
    if let Some(reason) = mesh_err {
        log_fallback(surface, reason);
    }
    flag.or_else(RenderMode::env_mode)
        .unwrap_or(if mesh_err.is_some() {
            RenderMode::Colliders
        } else {
            RenderMode::Mesh
        })
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
                        font_size: FontSize::Px(20.0),
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.5, 0.5)),
                ));
                b.spawn((
                    Text::new(format!("{reason}  —  fetch with scripts/fetch-sally.sh")),
                    TextFont {
                        font_size: FontSize::Px(13.0),
                        ..default()
                    },
                    TextColor(Color::srgba(0.95, 0.85, 0.85, 0.9)),
                ));
            })
            .id()
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
