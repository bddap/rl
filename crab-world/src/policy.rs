use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::Tensor;

use crate::bot::actuator::ACTION_SIZE;
use crate::bot::arch::{AnyBrain, ArchId};
use crate::bot::sensor::OBS_SIZE;
use crate::training::InferBackend;
use crate::training::checkpoint::{BrainLoadError, CheckpointDir, load_brain_file};
use crate::training::envelope::EnvelopeError;
use crate::training::normalizer::{NORMALIZER_CLIP, ObsNormalizer};

pub struct Policy {
    device: NdArrayDevice,
    /// The checkpoint dir whose brain currently drives [`Self::state`] — the identity
    /// [`Self::cycle_brain`] keys its cursor off.
    dir: PathBuf,
    /// The roster root the brain-swap button cycles under (rl#232): the boot checkpoint
    /// dir, retargeted to the live dir when one is set (that is the dir being DRIVEN).
    /// `dir == swap_root` IS the "latest" slot, so an unprefixed label always means
    /// latest — no cached slot name to drift.
    swap_root: PathBuf,
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    live_dir: Option<PathBuf>,
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    last_loaded: Option<std::time::SystemTime>,
    /// Brain-file mtime whose REFUSAL was already logged, so a persistently-bad
    /// checkpoint logs once per distinct file while still being retried each poll —
    /// unlike `last_loaded`, a refusal never suppresses the retry itself, because it may
    /// be the transient mid-save window (see [`Self::try_hot_reload`], bddap/rl#215).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    last_refused: Option<std::time::SystemTime>,
    /// `last_refused`'s plant sibling (bddap/rl#285), kept separate: deduping both
    /// refusal classes on one field would suppress the log when the CLASS flips at an
    /// unchanged brain mtime (a sidecar can heal or flip without touching the brain).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    last_plant_refused: Option<std::time::SystemTime>,
    state: PolicyState,
}

#[allow(clippy::large_enum_variant)]
enum PolicyState {
    Rest {
        refused: Option<String>,
    },
    Diagnostic {
        brain: AnyBrain<InferBackend>,
        normalizer: ObsNormalizer,
    },
    Loaded {
        brain: AnyBrain<InferBackend>,
        normalizer: ObsNormalizer,
        digest: NonZeroU64,
    },
}

fn checkpoint_digest(dir: &Path) -> u64 {
    let paths = CheckpointDir::new(dir);
    let Ok(mut bytes) = std::fs::read(paths.brain_file()) else {
        return 0;
    };
    if let Ok(norm) = std::fs::read(paths.normalizer_path()) {
        bytes.extend_from_slice(&norm);
    }
    crate::fnv::fnv1a(&bytes)
}

/// What a rest-bound [`Policy::load`] arms when no usable checkpoint loads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RestFallback {
    /// Hold the zero-action rest pose (the default).
    Rest,
    /// DIAGNOSTIC: drive an untrained random brain (rl-demo `--random-policy`).
    RandomBrain,
}

pub(crate) fn dims_fit_rig(obs: usize, action: usize) -> bool {
    (obs, action) == (OBS_SIZE, ACTION_SIZE)
}

/// Classify the on-disk checkpoint without arming it — for surfaces that only ever
/// report a verdict (`checkpoint-check`). Any caller whose Ok leads to DRIVING a crab
/// must use [`load_armed`] instead: classify-then-reload is two reads, and a checkpoint
/// swap landing between them arms a policy the gate never vetted (bddap/rl#241).
pub fn checkpoint_fits_rig(dir: &Path) -> Result<(), CheckpointUnusable> {
    match load_brain_normalizer(dir, &NdArrayDevice::Cpu) {
        Loaded::Fit(..) => Ok(()),
        Loaded::Absent => Err(CheckpointUnusable::Missing),
        Loaded::Mismatch(dims) => Err(CheckpointUnusable::Mismatch(dims)),
        Loaded::Refused(why) => Err(CheckpointUnusable::Refused(why)),
    }
}

/// Load AND arm in one read — the Ok IS the armed policy, so there is no second read
/// for a checkpoint swap to straddle (bddap/rl#241).
pub fn load_armed(checkpoint_dir: &Path) -> Result<Policy, CheckpointUnusable> {
    let device = NdArrayDevice::Cpu;
    match load_brain_normalizer(checkpoint_dir, &device) {
        Loaded::Fit(brain, normalizer) => {
            info!("loaded armed checkpoint from {}", checkpoint_dir.display());
            Ok(Policy {
                device,
                dir: checkpoint_dir.to_owned(),
                swap_root: checkpoint_dir.to_owned(),
                live_dir: None,
                last_loaded: None,
                last_refused: None,
                last_plant_refused: None,
                state: loaded_state(brain, normalizer, checkpoint_digest(checkpoint_dir)),
            })
        }
        Loaded::Absent => Err(CheckpointUnusable::Missing),
        Loaded::Mismatch(dims) => Err(CheckpointUnusable::Mismatch(dims)),
        Loaded::Refused(why) => Err(CheckpointUnusable::Refused(why)),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RigDims {
    pub obs: usize,
    pub action: usize,
}

/// Why a checkpoint cannot arm — the one refusal classification both
/// [`checkpoint_fits_rig`] and [`load_armed`] speak.
#[derive(Debug)]
pub enum CheckpointUnusable {
    /// No `brain.bin` — the legitimate "no brain yet" case.
    Missing,
    /// The checkpoint was refused before any dims existed to compare, with the reason
    /// preformatted: a truncated/corrupt file (distinct from `Missing` so the operator
    /// redeploys the file they can SEE instead of chasing path resolution), a legacy
    /// unmigrated file, an unregistered architecture named in the reason (the bddap/rl#200
    /// §2 arch arm), a mis-copied artifact, or a missing/mis-paired obs normalizer.
    Refused(String),
    Mismatch(RigDims),
}

#[allow(clippy::large_enum_variant)]
enum Loaded {
    /// Brain + its paired normalizer parsed and the dims fit the rig — arm the NN crab.
    Fit(AnyBrain<InferBackend>, ObsNormalizer),
    /// No `brain.bin`. The LEGITIMATE "no brain yet" case — keep the current policy /
    /// hold the neutral rest pose. Not an error.
    Absent,
    Mismatch(RigDims),
    /// The envelope refused the checkpoint — corrupt/truncated, legacy (unmigrated), an
    /// unregistered arch, a mis-copied file, or a missing/mis-paired normalizer. Like
    /// `Mismatch`, an operator error refused LOUDLY with the reason; what used to degrade
    /// to the quiet rest pose misattributed as "no checkpoint" (bddap/rl#200 §2's
    /// silent-fallback class) is now this distinct verdict.
    Refused(String),
}

/// A cross-member coherence refusal (the normalizer's save stamp / arch vs the brain's
/// set key) can be a STRADDLED READ rather than a bad set on disk: the set lands in one
/// atomic dir swap (bddap/rl#238), but the two member opens below are separate path
/// lookups, so a swap landing between them pairs generation N's brain with generation
/// N+1's normalizer — both files valid, stamps mismatched (2026-07-09: a false
/// rl-release red from exactly this). Re-reading disambiguates: a straddle heals on the
/// next read, a genuine on-disk mis-pair (an out-of-band torn copy) persists through
/// every read and still refuses.
const SET_READ_ATTEMPTS: u32 = 3;
const SET_READ_RETRY: std::time::Duration = std::time::Duration::from_millis(25);

/// Read a brain + normalizer from `dir`, classifying the result for the caller — THE one
/// checkpoint classifier: the runtime loaders (initial + hot-reload) and the gates
/// ([`checkpoint_fits_rig`]) all speak this verdict, so they can't drift — and all get
/// the straddled-read retry ([`SET_READ_ATTEMPTS`]) for free.
fn load_brain_normalizer(dir: &Path, device: &NdArrayDevice) -> Loaded {
    let mut mispair = None;
    for attempt in 1..=SET_READ_ATTEMPTS {
        if attempt > 1 {
            std::thread::sleep(SET_READ_RETRY);
        }
        match load_set_once(dir, device) {
            SetRead::Done(loaded) => return loaded,
            SetRead::Mispaired(why) => mispair = Some(why),
        }
    }
    let why = mispair.expect("the loop always runs");
    Loaded::Refused(format!(
        "{why} — persisted across {SET_READ_ATTEMPTS} re-reads, so this is a real \
         mis-pair on disk, not a read racing a concurrent set swap"
    ))
}

/// One verdict from a single read of the set, split so [`load_brain_normalizer`] can
/// retry the possibly-transient case without string-matching the refusal.
#[allow(clippy::large_enum_variant)]
enum SetRead {
    Done(Loaded),
    /// The normalizer refused against the brain's set key — possibly a straddled read.
    Mispaired(String),
}

/// One read of the checkpoint set. The brain load dispatches on the envelope's arch tag
/// ([`load_brain_file`]); the normalizer is brain-PAIRED — it must exist and carry the
/// brain's own set key, because a real brain normalizing against cold or mis-paired
/// stats acts silently wrong, a worse failure than not arming.
fn load_set_once(dir: &Path, device: &NdArrayDevice) -> SetRead {
    let paths = CheckpointDir::new(dir);
    let loaded = match load_brain_file::<InferBackend>(&paths.brain_file(), device) {
        Ok(loaded) => loaded,
        Err(BrainLoadError::Envelope(EnvelopeError::Absent)) => {
            return SetRead::Done(Loaded::Absent);
        }
        Err(e) => {
            return SetRead::Done(Loaded::Refused(format!(
                "{}: {e}",
                crate::training::checkpoint::BRAIN_FILENAME
            )));
        }
    };
    // Body↔policy identity (bddap/rl#214): a policy trained on a different body than the
    // one this process constructs is not this crab — refuse to arm it (same class as the
    // rig-dims mismatch below), never drive the wrong body "mostly fine". A pre-stamp
    // checkpoint passes on trust (read-only surfaces don't stamp; the trainer's next
    // save migrates it).
    if let Err(why) = crate::training::checkpoint::check_body_identity(
        loaded.body_digest,
        crate::mesh_fallback::constructed_body_digest(),
    ) {
        return SetRead::Done(Loaded::Refused(format!(
            "{}: {why}",
            crate::training::checkpoint::BRAIN_FILENAME
        )));
    }
    // Channel-layout identity (bddap/rl#271): the dims gate below checks only COUNTS, so
    // a same-count channel reorder would arm clean and drive the wrong joints — refuse
    // it like a wrong body.
    if let Err(why) = crate::training::checkpoint::check_channel_layout(
        loaded.layout_digest,
        crate::bot::channel_layout_digest(),
    ) {
        return SetRead::Done(Loaded::Refused(format!(
            "{}: {why}",
            crate::training::checkpoint::BRAIN_FILENAME
        )));
    }
    let key = loaded.set_key();
    let brain = loaded.brain;
    let (obs, action) = brain.io_dims();
    if !dims_fit_rig(obs, action) {
        return SetRead::Done(Loaded::Mismatch(RigDims { obs, action }));
    }
    // The set key (bddap/rl#215) keys the pairing to the brain's own SAVE, not
    // just its arch: a partial save (or a copy torn across one) leaves one member from an
    // older set, which would normalize this brain with another save's statistics —
    // refused here like any other mis-pair.
    match ObsNormalizer::load(&paths.normalizer_path(), key) {
        Ok(normalizer) => SetRead::Done(Loaded::Fit(brain, normalizer)),
        // The two coherence variants are the ones a straddled read can produce — every
        // other refusal is a property of the file itself and re-reading can't change it.
        Err(e @ (EnvelopeError::SaveStampMismatch { .. } | EnvelopeError::ArchMismatch { .. })) => {
            SetRead::Mispaired(format!(
                "{}: {e} (a brain never arms without its paired obs normalizer)",
                crate::training::checkpoint::NORMALIZER_FILENAME
            ))
        }
        Err(e) => SetRead::Done(Loaded::Refused(format!(
            "{}: {e} (a brain never arms without its paired obs normalizer)",
            crate::training::checkpoint::NORMALIZER_FILENAME
        ))),
    }
}

/// The one spelling of the wrong-rig refusal reason that rides into
/// [`PolicyState::Rest`] (and from there the on-screen brain label) — shared by the
/// initial load and the hot-reload so the attribution can't be re-phrased per site.
fn rig_mismatch_reason(dims: RigDims) -> String {
    let RigDims { obs, action } = dims;
    format!("wrong rig: {obs} obs/{action} act (this build: {OBS_SIZE}/{ACTION_SIZE})")
}

fn log_rig_mismatch(surface: &str, dir: &Path, dims: RigDims) {
    let RigDims { obs, action } = dims;
    error!(
        "{surface}: checkpoint at {} was built for a DIFFERENT rig — its brain wants \
         {obs} obs / {action} act but this binary's crab rig is {OBS_SIZE} obs / \
         {ACTION_SIZE} act. REFUSING to arm the NN crab: it would hold an inert rest \
         pose that looks frozen-but-fine. Rebuild the checkpoint for this rig, or run a \
         binary whose rig matches the checkpoint.",
        dir.display(),
    );
}

/// The loud refusal for an envelope-level rejection (corrupt/legacy/unknown-arch/
/// mis-paired) — `log_rig_mismatch`'s sibling, one message for both arming sites.
fn log_checkpoint_refusal(surface: &str, dir: &Path, why: &str) {
    error!(
        "{surface}: REFUSING checkpoint at {} — {why}. The NN crab will NOT arm (it \
         would hold an inert rest pose that looks frozen-but-fine).",
        dir.display(),
    );
}

/// bddap/rl#285: refuse to arm a checkpoint whose recorded plant disagrees with the
/// plant this process resolved. A feed flip (e.g. terrain→flat) swaps a NEW RUN into
/// the live dir under a long-lived kiosk, and hot-reloading that run's weights into
/// the old run's arena drives them in a world they never trained in — indefinitely,
/// since nothing restarts the demo on deploy. Force-resolving both plant locks first
/// makes `adopt_recorded_plant` agreement-or-error BY CONSTRUCTION: an embedder that
/// never adopted at boot resolves from env here rather than silently adopting the
/// checkpoint's plant into a world that was already built without it. Reuses the one
/// adoption primitive instead of growing a second comparison path.
fn plant_agreement(dir: &Path) -> Result<(), String> {
    let _ = crate::bot::body::friction_cap_override();
    crate::bot::body::adopt_recorded_plant(dir)
}

/// The loud refusal for a plant disagreement (bddap/rl#285) — `log_rig_mismatch`'s
/// third sibling: the checkpoint is healthy but trained in a different world than
/// this process simulates.
fn log_plant_refusal(surface: &str, dir: &Path, why: &str) {
    error!(
        "{surface}: REFUSING checkpoint at {} — {why}. The current brain keeps \
         driving; the checkpoint arms once its plant agrees (a fresh launch adopts it).",
        dir.display(),
    );
}

fn loaded_state(
    brain: AnyBrain<InferBackend>,
    normalizer: ObsNormalizer,
    digest: u64,
) -> PolicyState {
    match NonZeroU64::new(digest) {
        Some(digest) => PolicyState::Loaded {
            brain,
            normalizer,
            digest,
        },
        None => PolicyState::Diagnostic { brain, normalizer },
    }
}

impl Policy {
    /// The explicit zero-action rest-pose policy — for tests and neutral-pose
    /// inspection. Arming paths never construct this; they go through [`load_armed`],
    /// whose failure is a refusal, not a quiet statue (bddap/rl#241).
    pub fn rest() -> Self {
        Self {
            device: NdArrayDevice::Cpu,
            // No checkpoint identity: empty and equal, so the label is unprefixed
            // ("latest") and a swap cycle finds an empty roster rather than a phantom.
            dir: PathBuf::new(),
            swap_root: PathBuf::new(),
            live_dir: None,
            last_loaded: None,
            last_refused: None,
            last_plant_refused: None,
            state: PolicyState::Rest { refused: None },
        }
    }

    /// Load brain + normalizer from a checkpoint dir. A missing checkpoint falls back
    /// QUIETLY to the zero-action rest pose so the app still launches (useful before the
    /// first checkpoint exists, and to inspect the body's neutral pose); a present-but-
    /// unusable one (wrong rig, or envelope-refused — corrupt/legacy/wrong-arch) is
    /// refused LOUDLY and also rests. `fallback` picks what a rest-bound load arms
    /// instead — an explicit caller decision, never ambient process state (rl#272).
    pub fn load(checkpoint_dir: &Path, fallback: RestFallback) -> Self {
        let device = NdArrayDevice::Cpu;
        let loaded = load_brain_normalizer(checkpoint_dir, &device);
        // The loud refusals fire FIRST, unconditionally — the diagnostic override below
        // must not swallow the reason (a corrupt/legacy/wrong-rig checkpoint is exactly
        // what an operator debugging with the random brain needs to see).
        match &loaded {
            Loaded::Mismatch(dims) => log_rig_mismatch("play", checkpoint_dir, *dims),
            Loaded::Refused(why) => log_checkpoint_refusal("play", checkpoint_dir, why),
            Loaded::Fit(..) | Loaded::Absent => {}
        }
        let state = match loaded {
            Loaded::Fit(brain, normalizer) => {
                info!("play: loaded checkpoint from {}", checkpoint_dir.display());
                loaded_state(brain, normalizer, checkpoint_digest(checkpoint_dir))
            }
            Loaded::Absent | Loaded::Mismatch(_) | Loaded::Refused(_)
                if fallback == RestFallback::RandomBrain =>
            {
                warn!(
                    "play: --random-policy — driving with an untrained random brain \
                     (no usable checkpoint at {})",
                    checkpoint_dir.display()
                );
                PolicyState::Diagnostic {
                    brain: AnyBrain::<InferBackend>::init(ArchId::DEFAULT, &device),
                    normalizer: ObsNormalizer::new(NORMALIZER_CLIP),
                }
            }
            Loaded::Absent => {
                warn!(
                    "play: no usable checkpoint at {} — using zero-action pose",
                    checkpoint_dir.display()
                );
                PolicyState::Rest { refused: None }
            }
            // Refusal already logged above; the state is the ATTRIBUTED rest pose — the
            // reason rides in the state so the on-screen brain label renders it (rl#200).
            Loaded::Mismatch(dims) => PolicyState::Rest {
                refused: Some(rig_mismatch_reason(dims)),
            },
            Loaded::Refused(why) => PolicyState::Rest { refused: Some(why) },
        };

        Self {
            device,
            dir: checkpoint_dir.to_owned(),
            swap_root: checkpoint_dir.to_owned(),
            live_dir: None,
            last_loaded: None,
            last_refused: None,
            last_plant_refused: None,
            state,
        }
    }

    /// Point this policy at a different checkpoint dir NOW — the one brain-swap
    /// primitive ([`Self::cycle_brain`]; the demo's button and the GCR host's button
    /// both land here). Only a `Fit` load replaces the driving state; any other verdict
    /// is logged loudly and keeps the current brain, exactly like a hot-reload of a bad
    /// file — a swap must never blank a working Sally. A policy that hot-follows a live
    /// dir follows the SWITCHED dir from now on (so "latest" keeps streaming the trainer
    /// and "best" tracks keep-best promotions), one that never hot-reloads stays static
    /// on the new brain.
    fn switch_dir(&mut self, dir: &Path) -> bool {
        // Stat BEFORE loading (mirroring `try_hot_reload`): stamping a post-load mtime
        // would skip a save that landed mid-load — forever, if it was the run's final
        // save (the rl#215 class). A stale-early stamp costs one redundant reload.
        let mtime = std::fs::metadata(CheckpointDir::new(dir).brain_file())
            .and_then(|m| m.modified())
            .ok();
        match load_brain_normalizer(dir, &self.device) {
            Loaded::Fit(brain, normalizer) => {
                // rl#285: a roster slot synced from a different run must not arm into
                // this process's world. Checked on Fit only — an empty slot stays the
                // gentle "no brain" refusal below, not a phantom plant mismatch.
                if let Err(why) = plant_agreement(dir) {
                    log_plant_refusal("brain swap", dir, &why);
                    return false;
                }
                self.state = loaded_state(brain, normalizer, checkpoint_digest(dir));
                self.dir = dir.to_owned();
                if self.live_dir.is_some() {
                    self.live_dir = Some(dir.to_owned());
                }
                self.last_loaded = mtime;
                self.last_refused = None;
                self.last_plant_refused = None;
                info!(
                    "brain swap: {} now driving from {}",
                    self.brain_label(),
                    dir.display(),
                );
                true
            }
            Loaded::Absent => {
                warn!(
                    "brain swap: no brain at {} — keeping the current brain",
                    dir.display()
                );
                false
            }
            Loaded::Mismatch(dims) => {
                log_rig_mismatch("brain swap", dir, dims);
                false
            }
            Loaded::Refused(why) => {
                log_checkpoint_refusal("brain swap", dir, &why);
                false
            }
        }
    }

    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn set_live_dir(&mut self, dir: Option<PathBuf>) {
        if let Some(d) = &dir {
            self.swap_root = d.clone();
        }
        self.live_dir = dir;
    }

    /// Hot-reload's plant-refusal bookkeeping (bddap/rl#285): log once per distinct
    /// save, and give an UNARMED policy the refusal attribution (same promise as the
    /// rig-mismatch and envelope-refusal arms) — a driving brain is left untouched.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    fn refuse_plant(&mut self, dir: &Path, mtime: std::time::SystemTime, why: String) {
        if self.last_plant_refused != Some(mtime) {
            log_plant_refusal("play (hot-reload)", dir, &why);
            self.last_plant_refused = Some(mtime);
        }
        if matches!(self.state, PolicyState::Rest { .. }) {
            self.state = PolicyState::Rest { refused: Some(why) };
        }
    }

    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn try_hot_reload(&mut self) -> bool {
        let Some(dir) = self.live_dir.clone() else {
            return false;
        };
        let brain_bin = CheckpointDir::new(&dir).brain_file();
        let Ok(mtime) = std::fs::metadata(&brain_bin).and_then(|m| m.modified()) else {
            return false;
        };
        if self.last_loaded == Some(mtime) {
            return false;
        }
        // rl#285: a feed flip swaps a NEW RUN into the live dir; its weights must not
        // arm into the old run's world. Checked BEFORE the load so a persistently-
        // disagreeing run costs one sidecar read per poll, not a brain load — and NOT
        // stamped into `last_loaded`: the healing sidecar can land without touching the
        // brain's mtime (the torn-set reasoning below), so every poll re-checks until
        // agreement. `last_plant_refused` keeps the log at once per distinct save.
        if let Err(why) = plant_agreement(&dir) {
            self.refuse_plant(&dir, mtime, why);
            return false;
        }
        match load_brain_normalizer(&dir, &self.device) {
            Loaded::Fit(brain, normalizer) => {
                // Re-checked between the coherent set read and arming: a flip landing
                // mid-load would arm behind the stale pre-check (the rl#241 classify-
                // then-act straddle). A brain landing before its sidecar within one
                // sync tick remains uncovered — this guard is a mismatch net, not a
                // transaction; the durable close is a plant leg in the save-stamped set.
                if let Err(why) = plant_agreement(&dir) {
                    self.refuse_plant(&dir, mtime, why);
                    return false;
                }
                self.state = loaded_state(brain, normalizer, checkpoint_digest(&dir));
                self.dir = dir;
                self.last_loaded = Some(mtime);
                true
            }
            Loaded::Absent => false, // no brain file yet — keep the current policy
            // A wrong-rig brain landed in the live dir: refuse it loudly but keep
            // whatever we're DRIVING (a bad file must not blank a working demo). An
            // UNARMED policy has nothing to protect — re-attribute its rest state so the
            // on-screen label says REFUSED and why, not "no brain" (the attribution
            // promise holds on hot-reload, not just the initial load). A rig mismatch is
            // a property of the brain FILE itself, so stamp `last_loaded` — log once per
            // distinct file (mtime), no pointless re-reads.
            Loaded::Mismatch(dims) => {
                log_rig_mismatch("play (hot-reload)", &dir, dims);
                self.last_loaded = Some(mtime);
                if matches!(self.state, PolicyState::Rest { .. }) {
                    self.state = PolicyState::Rest {
                        refused: Some(rig_mismatch_reason(dims)),
                    };
                }
                false
            }
            // Same keep-driving policy (with the same rest re-attribution), but do NOT
            // stamp `last_loaded`: the refusal may be the transient mid-save window above,
            // and the completed set won't change the brain's mtime — stamping here would
            // refuse a run's FINAL save forever. A genuinely bad checkpoint just gets
            // re-read (and re-refused) each poll; `last_refused` keeps the log at once per
            // distinct file. (A transient REFUSED label self-heals the same way: the next
            // poll's Fit overwrites it.)
            Loaded::Refused(why) => {
                if self.last_refused != Some(mtime) {
                    log_checkpoint_refusal("play (hot-reload)", &dir, &why);
                    self.last_refused = Some(mtime);
                }
                if matches!(self.state, PolicyState::Rest { .. }) {
                    self.state = PolicyState::Rest { refused: Some(why) };
                }
                false
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        !matches!(self.state, PolicyState::Rest { .. })
    }

    /// The human-facing identity of the brain driving this policy — the on-screen crab label
    /// (rl#200 increment 7): `arch @shortdigest` for a real checkpoint, and every OTHER state
    /// attributed honestly ("who's who" includes who FAILED and why). THE one label formatter:
    /// the demo, the GCR host, and (via the articulation wire) every GCR client render this
    /// exact string, so the label can't drift per surface.
    pub fn brain_label(&self) -> String {
        let core = self.state_label();
        // Driving a roster subdir (a brain swap, rl#232) carries the slot's dir name as
        // a prefix — derived from `dir` vs `swap_root` at read time, never cached, so an
        // unprefixed label always MEANS the latest/primary brain. ASCII separator: the
        // in-world label font has no '·' glyph (renders tofu on the TV).
        match self.dir.file_name() {
            Some(name) if self.dir != self.swap_root => {
                format!("{}: {core}", name.to_string_lossy())
            }
            _ => core,
        }
    }

    fn state_label(&self) -> String {
        match &self.state {
            PolicyState::Loaded { brain, digest, .. } => {
                // First 8 hex chars of the full checkpoint digest — enough to tell two live
                // checkpoints apart at a glance, short enough to read in a playtest.
                let hex = format!("{:016x}", u64::from(*digest));
                format!("{} @{}", brain.arch(), &hex[..8])
            }
            PolicyState::Diagnostic { brain, .. } => {
                format!("{} (random, untrained)", brain.arch())
            }
            PolicyState::Rest { refused: None } => "no brain (rest pose)".to_string(),
            PolicyState::Rest { refused: Some(why) } => {
                // A floating label, not a log line: keep it readable in-world. The full
                // reason is already in the load-time `error!`.
                const MAX: usize = 60;
                let short = crate::truncate_at_char_boundary(why, MAX);
                let ellipsis = if short.len() < why.len() { "…" } else { "" };
                format!("REFUSED: {short}{ellipsis}")
            }
        }
    }

    pub fn act(&self, raw_obs: &[f32; OBS_SIZE]) -> [f32; ACTION_SIZE] {
        let (brain, normalizer) = match &self.state {
            PolicyState::Loaded {
                brain, normalizer, ..
            }
            | PolicyState::Diagnostic { brain, normalizer } => (brain, normalizer),
            PolicyState::Rest { .. } => return [0.0; ACTION_SIZE],
        };
        let obs = normalizer.normalize_frozen(raw_obs);
        let input =
            Tensor::<InferBackend, 1>::from_floats(obs.as_slice(), &self.device).unsqueeze();
        let (means, _log_std) = brain.policy(input);
        let flat: Vec<f32> = means.flatten::<1>(0, 1).to_data().to_vec().unwrap();

        flat.try_into()
            .expect("policy mean count == ACTION_SIZE (the rig gates refuse mismatched brains)")
    }

    /// Swap to the next loadable brain in the roster under [`Self::swap_root`] — THE one
    /// brain-swap code path (rl#232): the demo's button and the GCR host's button both
    /// call this, so the two surfaces can't drift. Re-enumerates the roster per press (a
    /// `best/` promoted since boot, or a run dir dropped in, joins without a restart)
    /// and skips slots that refuse to load (each refusal already logged loudly); `false`
    /// when there is nothing to swap to or every other slot refused.
    pub fn cycle_brain(&mut self) -> bool {
        let slots = brain_slots(&self.swap_root);
        if slots.len() < 2 {
            info!(
                "brain swap: only one brain under {} — nothing to swap to",
                self.swap_root.display()
            );
            return false;
        }
        // An off-roster cursor (the boot dir before the first hot-reload retargets onto
        // the live root) resolves so the next step lands on slot 0 — the primary brain
        // is always the first stop.
        let current = slots
            .iter()
            .position(|d| *d == self.dir)
            .unwrap_or(slots.len() - 1);
        for step in 1..slots.len() {
            if self.switch_dir(&slots[(current + step) % slots.len()]) {
                return true;
            }
        }
        false
    }
}

/// The swap list rooted at one primary checkpoint dir: the dir itself (the "latest"
/// slot) plus every direct subdirectory holding a `brain.bin`, sorted. Data-driven by
/// construction (rl#232): keep-best's `best/` joins because it is such a subdir, and
/// dropping in more dirs (or symlinks — another run's ckpt, another architecture)
/// extends the cycle with no code change; each brain's arch comes from its own envelope
/// at load time.
fn brain_slots(primary: &Path) -> Vec<PathBuf> {
    let mut subs: Vec<PathBuf> = std::fs::read_dir(primary)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|dir| dir.is_dir() && CheckpointDir::new(dir).brain_file().is_file())
        .collect();
    subs.sort();
    let mut slots = vec![primary.to_owned()];
    slots.append(&mut subs);
    slots
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

    use crate::bot::arch::Mlp512x3;
    use crate::training::TrainBackend;
    use crate::training::envelope::{ArtifactKind, BrainStamps, write_envelope};

    fn save_brain(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let paths = CheckpointDir::new(dir);
        crate::training::checkpoint::save_brain(&brain, &paths.brain_file(), 21).unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 21)
            .unwrap();
        // Every real checkpoint records its terrain arena (rl#293's load condition).
        std::fs::write(
            dir.join(crate::bot::body::PLANT_FILENAME),
            "arena terrain\n",
        )
        .unwrap();
    }

    /// The fixture obs the golden `actions.hex` was generated over: deterministic,
    /// integer-derived (no libm), shared by the golden tests.
    fn golden_obs() -> [f32; OBS_SIZE] {
        let mut obs = [0.0f32; OBS_SIZE];
        for (i, o) in obs.iter_mut().enumerate() {
            *o = ((i * 37 % 101) as f32) / 50.5 - 1.0;
        }
        obs
    }

    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden-mlp512x3-env")
    }

    fn golden_action_bits() -> Vec<u32> {
        std::fs::read_to_string(golden_dir().join("actions.hex"))
            .expect("read golden actions.hex")
            .lines()
            .map(|l| u32::from_str_radix(l.trim(), 16).expect("parse golden bits"))
            .collect()
    }

    /// GOLDEN FILE (bddap/rl#200 increment 2): `tests/data/golden-mlp512x3-env/` holds
    /// an ENVELOPED golden brain (v1/TOFU shape, plus an identity obs normalizer) with
    /// its action bits over [`golden_obs`] pinned in `actions.hex` at generation time.
    /// Loading it through today's loader and reproducing those bits EXACTLY proves the
    /// tagged on-disk format did not drift (a reader/writer change that re-mapped
    /// envelope fields would strand every fleet checkpoint while all save/load-symmetric
    /// tests stayed green). Regenerate with `regenerate_enveloped_golden_fixture` below
    /// ONLY on a deliberate format version bump or an arch cull.
    ///
    /// Post-rl#298 this bit-exactness guard is justified as TRAINER/EVAL REPRODUCIBILITY —
    /// the same forward pass the host runs in its crab slot is what the trainer trains and
    /// the eval measures, so it must stay bit-stable across toolchain/backend bumps — not as
    /// MP correctness: clients never run the policy (the host streams poses), so no
    /// cross-peer forward determinism is required.
    #[test]
    fn golden_enveloped_checkpoint_loads_and_acts_bit_identically() {
        let policy = Policy::load(&golden_dir(), RestFallback::Rest);
        assert!(
            policy.is_loaded(),
            "the enveloped golden checkpoint no longer loads — the tagged on-disk format \
             drifted (every migrated fleet checkpoint would refuse to arm)"
        );

        let expected = golden_action_bits();
        assert_eq!(
            expected.len(),
            ACTION_SIZE,
            "golden fixture is for this rig"
        );
        let act = policy.act(&golden_obs());
        let got: Vec<u32> = act.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            got, expected,
            "actions on the fixture obs are not bit-identical to the pinned golden bits"
        );
    }

    #[test]
    fn legacy_checkpoint_is_refused() {
        let dir = std::env::temp_dir().join(format!("rl-legacy-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain: AnyBrain<TrainBackend> = AnyBrain::init(ArchId::DEFAULT, &device);
        let raw = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        std::fs::write(CheckpointDir::new(&dir).brain_file(), raw).unwrap();

        let policy = Policy::load(&dir, RestFallback::Rest);
        assert!(
            !policy.is_loaded(),
            "a legacy untagged brain.bin must not load — the loader has no untagged path"
        );
        // The non-arming is graceful: the rest pose, never a panic.
        assert_eq!(policy.act(&golden_obs()), [0.0; ACTION_SIZE]);

        match checkpoint_fits_rig(&dir) {
            Err(CheckpointUnusable::Refused(why)) => assert!(
                why.contains("pre-envelope"),
                "the refusal must name the legacy (pre-envelope) diagnosis, got: {why}"
            ),
            _ => panic!("a legacy checkpoint must classify as Refused, not Missing/Ok"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "fixture generator, run manually on a deliberate format change or arch cull"]
    fn regenerate_enveloped_golden_fixture() {
        let dir = golden_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let paths = CheckpointDir::new(&dir);
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let bytes = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        crate::training::envelope::write_v1_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            brain.arch(),
            bytes,
        )
        .unwrap();
        crate::training::envelope::write_v1_envelope(
            &paths.normalizer_path(),
            ArtifactKind::ObsNormalizer,
            brain.arch(),
            bincode::serialize(&ObsNormalizer::new(NORMALIZER_CLIP).snapshot()).unwrap(),
        )
        .unwrap();

        let policy = Policy::load(&dir, RestFallback::Rest);
        assert!(policy.is_loaded(), "the fixture just written must load");
        let bits: Vec<String> = policy
            .act(&golden_obs())
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect();
        std::fs::write(dir.join("actions.hex"), bits.join("\n") + "\n").unwrap();
    }

    /// bddap/rl#214 WIRING guard (the pure `check_body_identity` matrix is unit-tested
    /// beside it): a checkpoint stamped with a different body digest than this process
    /// constructs must refuse to arm through the one classifier — deleting or
    /// reordering the identity check in `load_brain_normalizer` turns this red.
    #[test]
    fn wrong_body_digest_checkpoint_refuses_to_arm() {
        let dir = std::env::temp_dir().join(format!("rl-bodydigest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let bytes = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        let paths = CheckpointDir::new(&dir);
        // Differs from the constructed digest whatever the test env's body is; the
        // layout stamp is CORRECT so this test isolates the body axis.
        let wrong = crate::mesh_fallback::constructed_body_digest() ^ 0xdead_beef;
        write_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            bytes,
            Some(BrainStamps {
                body_digest: wrong,
                layout_digest: crate::bot::channel_layout_digest(),
            }),
            21,
        )
        .unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(ArchId::DEFAULT, &paths.normalizer_path(), 21)
            .unwrap();

        let policy = Policy::load(&dir, RestFallback::Rest);
        assert!(!policy.is_loaded(), "a wrong-body checkpoint must not arm");
        match checkpoint_fits_rig(&dir) {
            Err(CheckpointUnusable::Refused(why)) => assert!(
                why.contains("DIFFERENT crab body"),
                "the refusal must name the body mismatch, got: {why}"
            ),
            _ => panic!("a wrong-body checkpoint must classify as Refused"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// bddap/rl#271 WIRING guard, `wrong_body_digest_checkpoint_refuses_to_arm`'s
    /// layout sibling: a brain whose channel-layout stamp differs from this build's
    /// layout has the right dims but remapped channels, and must refuse to arm.
    #[test]
    fn wrong_layout_digest_checkpoint_refuses_to_arm() {
        let dir = std::env::temp_dir().join(format!("rl-layoutdigest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let device = NdArrayDevice::Cpu;

        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let bytes = brain
            .record_leaf(&BinBytesRecorder::<FullPrecisionSettings>::default(), ())
            .unwrap();
        let paths = CheckpointDir::new(&dir);
        // The body stamp is CORRECT so this test isolates the layout axis.
        write_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            bytes,
            Some(BrainStamps {
                body_digest: crate::mesh_fallback::constructed_body_digest(),
                layout_digest: crate::bot::channel_layout_digest() ^ 0xdead_beef,
            }),
            21,
        )
        .unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(ArchId::DEFAULT, &paths.normalizer_path(), 21)
            .unwrap();

        let policy = Policy::load(&dir, RestFallback::Rest);
        assert!(
            !policy.is_loaded(),
            "a wrong-layout checkpoint must not arm"
        );
        match checkpoint_fits_rig(&dir) {
            Err(CheckpointUnusable::Refused(why)) => assert!(
                why.contains("SILENTLY REMAP"),
                "the refusal must name the layout remap hazard, got: {why}"
            ),
            _ => panic!("a wrong-layout checkpoint must classify as Refused"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hot_reload_swaps_in_a_new_checkpoint() {
        let tmp = std::env::temp_dir();
        let live = tmp.join(format!("rl-hotreload-live-{}", std::process::id()));
        let empty = tmp.join(format!("rl-hotreload-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&empty).unwrap();

        let mut policy = Policy::load(&empty, RestFallback::Rest);
        assert!(
            !policy.is_loaded(),
            "empty checkpoint dir should give an unloaded policy"
        );
        assert!(
            !policy.try_hot_reload(),
            "no live_dir set → nothing to reload"
        );

        policy.live_dir = Some(live.clone());
        assert!(
            !policy.try_hot_reload(),
            "live dir without a brain → no reload"
        );

        save_brain(&live);
        assert!(
            policy.try_hot_reload(),
            "a new brain in the live dir must reload"
        );
        assert!(
            policy.is_loaded(),
            "a successful hot-reload marks the policy loaded"
        );
        assert!(
            !policy.try_hot_reload(),
            "the same checkpoint must not reload again"
        );

        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// bddap/rl#285: a feed flip swaps a NEW RUN into the live dir; hot-reload must not
    /// arm its weights into the OLD run's world. With the process plant resolved (here:
    /// the default), a live checkpoint recording a different plant must refuse — keeping
    /// the policy unarmed with the refusal attributed — and must arm the moment the
    /// sidecar agrees again, even with no new save (the lagging per-file-sync case).
    #[test]
    fn hot_reload_refuses_a_plant_disagreeing_checkpoint() {
        let tmp = std::env::temp_dir();
        let live = tmp.join(format!("rl-hotreload-plant-{}", std::process::id()));
        let empty = tmp.join(format!("rl-hotreload-plant-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&live).unwrap();
        std::fs::create_dir_all(&empty).unwrap();

        save_brain(&live);
        // A friction cap the process (env-unset → default) never resolved: the
        // agreement check must refuse it. The arena record is present and canonical —
        // the friction leg alone carries the disagreement. (After save_brain: the
        // fixture writes the canonical sidecar this test then perturbs.)
        std::fs::write(
            live.join(crate::bot::body::PLANT_FILENAME),
            "joint_friction_cap 2.5\narena terrain\n",
        )
        .unwrap();

        let mut policy = Policy::load(&empty, RestFallback::Rest);
        policy.live_dir = Some(live.clone());
        assert!(
            !policy.try_hot_reload(),
            "a plant-disagreeing checkpoint must refuse, keeping the current policy"
        );
        assert!(!policy.is_loaded());
        assert!(
            policy.brain_label().contains("REFUSED"),
            "the unarmed rest state must carry the plant refusal, got: {}",
            policy.brain_label()
        );
        assert!(
            !policy.try_hot_reload(),
            "still disagreeing — every poll re-checks, still refusing"
        );

        // The sidecar heals WITHOUT the brain's mtime changing (a lagging per-file
        // sync): the unstamped refusal must re-check and arm on the very next poll.
        std::fs::write(
            live.join(crate::bot::body::PLANT_FILENAME),
            "arena terrain\n",
        )
        .unwrap();
        assert!(
            policy.try_hot_reload(),
            "an agreeing plant must arm on the next poll, no new save needed"
        );
        assert!(policy.is_loaded());

        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// bddap/rl#215: an out-of-band copy can land a set-torn dir (new brain, previous
    /// save's normalizer). The refusal must keep the current policy AND not cache the
    /// verdict against the brain's mtime: the healing normalizer can land WITHOUT
    /// touching `brain.bin`, so the next poll must arm the completed set. (Stamping
    /// `last_loaded` on the refusal would refuse a run's FINAL save forever — there is
    /// no later save to bump the mtime.)
    #[test]
    fn hot_reload_retries_a_set_torn_refusal_once_the_set_completes() {
        let tmp = std::env::temp_dir();
        let live = tmp.join(format!("rl-hotreload-torn-{}", std::process::id()));
        let empty = tmp.join(format!("rl-hotreload-torn-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&live).unwrap();
        std::fs::create_dir_all(&empty).unwrap();

        // The mid-save window: brain from save 22, normalizer still from save 21.
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let paths = CheckpointDir::new(&live);
        crate::training::checkpoint::save_brain(&brain, &paths.brain_file(), 22).unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 21)
            .unwrap();
        std::fs::write(
            live.join(crate::bot::body::PLANT_FILENAME),
            "arena terrain\n",
        )
        .unwrap();

        let mut policy = Policy::load(&empty, RestFallback::Rest);
        policy.live_dir = Some(live.clone());
        assert!(
            !policy.try_hot_reload(),
            "a set-torn dir must refuse, keeping the current policy"
        );

        // The trailing normalizer lands; brain.bin is untouched (same mtime).
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 22)
            .unwrap();
        assert!(
            policy.try_hot_reload(),
            "the completed set must arm on the next poll despite the unchanged brain mtime"
        );
        assert!(policy.is_loaded());

        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&empty);
    }

    /// The straddled-read retry (`SET_READ_ATTEMPTS`): a stamp mis-pair that PERSISTS
    /// across the re-reads is a real on-disk mis-pair and must still refuse — and the
    /// refusal must SAY the re-reads happened, so an operator reading it doesn't chase
    /// the concurrent-swap race the retry already ruled out. (The heal-on-re-read half
    /// needs a mid-load concurrent swap, which has no deterministic test; production
    /// coverage is the 2026-07-09 rl-release false red this exists to end.)
    #[test]
    fn persistent_mispair_still_refuses_and_names_the_re_reads() {
        let dir = std::env::temp_dir().join(format!("rl-mispair-retry-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let paths = CheckpointDir::new(&dir);
        crate::training::checkpoint::save_brain(&brain, &paths.brain_file(), 22).unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 21)
            .unwrap();

        match checkpoint_fits_rig(&dir) {
            Err(CheckpointUnusable::Refused(why)) => {
                assert!(
                    why.contains("DIFFERENT saves"),
                    "the refusal must name the mis-pair, got: {why}"
                );
                assert!(
                    why.contains("re-reads"),
                    "the refusal must attribute the ruled-out swap race, got: {why}"
                );
            }
            _ => panic!("a persistently mis-paired set must classify as Refused"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn save_brain_record(
        dir: &Path,
        record: crate::bot::arch::mlp512x3::Mlp512x3Record<TrainBackend>,
    ) {
        std::fs::create_dir_all(dir).unwrap();
        let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(record, ())
            .unwrap();
        let paths = CheckpointDir::new(dir);
        write_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            bytes,
            Some(BrainStamps {
                body_digest: crate::mesh_fallback::constructed_body_digest(),
                layout_digest: crate::bot::channel_layout_digest(),
            }),
            21,
        )
        .unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(ArchId::DEFAULT, &paths.normalizer_path(), 21)
            .unwrap();
    }

    fn save_brain_with_obs_dim(dir: &Path, obs_dim: usize) {
        use burn::module::{Param, ParamId};
        let device = NdArrayDevice::Cpu;
        let mut record = Mlp512x3::<TrainBackend>::new(&device).into_record();
        let [_obs, hidden] = record.trunk_fc1.weight.shape().dims();
        let weight = Tensor::<TrainBackend, 2>::zeros([obs_dim, hidden], &device);
        record.trunk_fc1.weight = Param::initialized(ParamId::new(), weight);
        save_brain_record(dir, record);
    }

    #[test]
    fn dim_mismatched_checkpoint_falls_back_instead_of_panicking() {
        let tmp = std::env::temp_dir();
        let dir = tmp.join(format!("rl-dimmismatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);

        let policy = Policy::load(&dir, RestFallback::Rest);
        assert!(
            !policy.is_loaded(),
            "a dim-mismatched checkpoint must fall back to unloaded, not load"
        );
        assert_eq!(
            policy.act(&[0.0; OBS_SIZE]),
            [0.0; ACTION_SIZE],
            "an unloaded policy holds the zero-action pose"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn save_nan_brain(dir: &Path) {
        use burn::module::{Param, ParamId};
        let device = NdArrayDevice::Cpu;
        let mut record = Mlp512x3::<TrainBackend>::new(&device).into_record();
        let dims: [usize; 2] = record.policy_fc.weight.shape().dims();
        let weight = Tensor::<TrainBackend, 2>::full(dims, f32::NAN, &device);
        record.policy_fc.weight = Param::initialized(ParamId::new(), weight);
        save_brain_record(dir, record);
    }

    #[test]
    fn act_propagates_a_non_finite_brain_output() {
        let dir = std::env::temp_dir().join(format!("rl-nanbrain-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        save_nan_brain(&dir);

        let policy = Policy::load(&dir, RestFallback::Rest);
        assert!(
            policy.is_loaded(),
            "the NaN brain is shape-valid, so the dim gates must load it"
        );
        let act = policy.act(&golden_obs());
        assert!(
            act.iter().any(|v| !v.is_finite()),
            "a numerically-broken brain's non-finite mean must reach the caller \
             (the actuator owns zeroing + the loud latched error), got {act:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkpoint_fits_rig_classifies_the_on_disk_brain() {
        let tmp = std::env::temp_dir();
        let dir = tmp.join(format!("rl-rigfit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        assert!(matches!(
            checkpoint_fits_rig(&dir),
            Err(CheckpointUnusable::Missing)
        ));

        // A current-rig brain (with its paired normalizer) fits.
        save_brain(&dir);
        assert!(checkpoint_fits_rig(&dir).is_ok());

        // A brain whose paired normalizer is MISSING is Refused — a real brain must never
        // arm against silently-cold normalizer stats.
        std::fs::remove_file(CheckpointDir::new(&dir).normalizer_path()).unwrap();
        assert!(matches!(
            checkpoint_fits_rig(&dir),
            Err(CheckpointUnusable::Refused(_))
        ));
        save_brain(&dir); // restore the pair

        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);
        assert!(matches!(
            checkpoint_fits_rig(&dir),
            Err(CheckpointUnusable::Mismatch(RigDims { obs, .. })) if obs == OBS_SIZE + 4
        ));

        // A present-but-corrupt brain.bin is Refused, not Missing.
        std::fs::write(CheckpointDir::new(&dir).brain_file(), b"truncated garbage").unwrap();
        assert!(matches!(
            checkpoint_fits_rig(&dir),
            Err(CheckpointUnusable::Refused(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// rl#232: the swap roster is the primary dir + every brain-holding subdir, and the
    /// cycle walks it latest → best → latest, stamping the slot into the label; a slot
    /// that refuses to load is skipped, and a roster of one is a no-op.
    #[test]
    fn cycle_brain_walks_the_roster_and_labels_the_slot() {
        let root = std::env::temp_dir().join(format!("rl-brain-cycle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        save_brain(&root);
        save_brain(&root.join("best"));
        // A non-checkpoint subdir (keep-best's own sidecar litter, a stray dir) never
        // becomes a slot.
        std::fs::create_dir_all(root.join("not-a-brain")).unwrap();

        assert_eq!(brain_slots(&root), vec![root.clone(), root.join("best")]);

        let mut policy = Policy::load(&root, RestFallback::Rest);
        assert!(policy.is_loaded());
        let boot_label = policy.brain_label();
        assert!(
            !boot_label.contains('·'),
            "no slot prefix on the primary brain: {boot_label}"
        );

        assert!(policy.cycle_brain());
        assert!(
            policy.brain_label().starts_with("best: "),
            "{}",
            policy.brain_label()
        );
        assert_eq!(policy.dir, root.join("best"));

        assert!(policy.cycle_brain());
        assert!(
            !policy.brain_label().contains('·'),
            "back on the primary brain the prefix drops: {}",
            policy.brain_label()
        );
        assert_eq!(policy.dir, root);

        // A corrupt slot never lands: the swap reports nothing to switch to and the
        // driving brain never blanks.
        std::fs::write(CheckpointDir::new(&root.join("best")).brain_file(), b"junk").unwrap();
        assert!(!policy.cycle_brain());
        assert!(policy.is_loaded());
        assert_eq!(policy.dir, root, "a failed swap keeps the current slot");

        // Roster of one: nothing to swap to.
        std::fs::remove_dir_all(root.join("best")).unwrap();
        let mut solo = Policy::load(&root, RestFallback::Rest);
        assert!(!solo.cycle_brain());

        let _ = std::fs::remove_dir_all(&root);
    }
}
