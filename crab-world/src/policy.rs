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

pub fn checkpoint_digest(dir: &Path) -> u64 {
    let paths = CheckpointDir::new(dir);
    let Ok(mut bytes) = std::fs::read(paths.brain_file()) else {
        return 0;
    };
    if let Ok(norm) = std::fs::read(paths.normalizer_path()) {
        bytes.extend_from_slice(&norm);
    }
    crate::fnv::fnv1a(&bytes)
}

pub(crate) fn dims_fit_rig(obs: usize, action: usize) -> bool {
    (obs, action) == (OBS_SIZE, ACTION_SIZE)
}

pub fn checkpoint_fits_rig(dir: &Path) -> RigFit {
    match load_brain_normalizer(dir, &NdArrayDevice::Cpu) {
        Loaded::Fit(..) => RigFit::Ok,
        Loaded::Absent => RigFit::Missing,
        Loaded::Mismatch(dims) => RigFit::Mismatch(dims),
        Loaded::Refused(why) => RigFit::Refused(why),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RigDims {
    pub obs: usize,
    pub action: usize,
}

pub enum RigFit {
    Ok,
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
        Err(
            e @ (EnvelopeError::SaveStampMismatch { .. } | EnvelopeError::ArchMismatch { .. }),
        ) => SetRead::Mispaired(format!(
            "{}: {e} (a brain never arms without its paired obs normalizer)",
            crate::training::checkpoint::NORMALIZER_FILENAME
        )),
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
    /// Load brain + normalizer from a checkpoint dir. A missing checkpoint falls back
    /// QUIETLY to the zero-action rest pose so the app still launches (useful before the
    /// first checkpoint exists, and to inspect the body's neutral pose); a present-but-
    /// unusable one (wrong rig, or envelope-refused — corrupt/legacy/wrong-arch) is
    /// refused LOUDLY and also rests.
    pub fn load(checkpoint_dir: &Path) -> Self {
        let device = NdArrayDevice::Cpu;
        let random_override = std::env::var("RL_RANDOM_POLICY").is_ok_and(|v| v == "1");
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
            Loaded::Absent | Loaded::Mismatch(_) | Loaded::Refused(_) if random_override => {
                warn!(
                    "play: RL_RANDOM_POLICY — driving with an untrained random brain \
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
            live_dir: None,
            last_loaded: None,
            last_refused: None,
            state,
        }
    }

    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn set_live_dir(&mut self, dir: Option<PathBuf>) {
        self.live_dir = dir;
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
        match load_brain_normalizer(&dir, &self.device) {
            Loaded::Fit(brain, normalizer) => {
                self.state = loaded_state(brain, normalizer, checkpoint_digest(&dir));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Module;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

    use crate::bot::arch::Mlp512x3;
    use crate::training::TrainBackend;
    use crate::training::envelope::{ArtifactKind, write_envelope};

    fn save_brain(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let device = NdArrayDevice::Cpu;
        let brain = AnyBrain::<TrainBackend>::init(ArchId::DEFAULT, &device);
        let paths = CheckpointDir::new(dir);
        crate::training::checkpoint::save_brain(&brain, &paths.brain_file(), 21).unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(brain.arch(), &paths.normalizer_path(), 21)
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
    #[test]
    fn golden_enveloped_checkpoint_loads_and_acts_bit_identically() {
        let policy = Policy::load(&golden_dir());
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

        let policy = Policy::load(&dir);
        assert!(
            !policy.is_loaded(),
            "a legacy untagged brain.bin must not load — the loader has no untagged path"
        );
        // The non-arming is graceful: the rest pose, never a panic.
        assert_eq!(policy.act(&golden_obs()), [0.0; ACTION_SIZE]);

        match checkpoint_fits_rig(&dir) {
            RigFit::Refused(why) => assert!(
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

        let policy = Policy::load(&dir);
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
        // Differs from the constructed digest whatever the test env's body is.
        let wrong = crate::mesh_fallback::constructed_body_digest() ^ 0xdead_beef;
        write_envelope(
            &paths.brain_file(),
            ArtifactKind::Brain,
            ArchId::DEFAULT,
            bytes,
            Some(wrong),
            21,
        )
        .unwrap();
        ObsNormalizer::new(NORMALIZER_CLIP)
            .save(ArchId::DEFAULT, &paths.normalizer_path(), 21)
            .unwrap();

        let policy = Policy::load(&dir);
        assert!(!policy.is_loaded(), "a wrong-body checkpoint must not arm");
        match checkpoint_fits_rig(&dir) {
            RigFit::Refused(why) => assert!(
                why.contains("DIFFERENT crab body"),
                "the refusal must name the body mismatch, got: {why}"
            ),
            _ => panic!("a wrong-body checkpoint must classify as Refused"),
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

        let mut policy = Policy::load(&empty);
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

        let mut policy = Policy::load(&empty);
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
            RigFit::Refused(why) => {
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
            Some(crate::mesh_fallback::constructed_body_digest()),
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

        let policy = Policy::load(&dir);
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

        let policy = Policy::load(&dir);
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

        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Missing));

        // A current-rig brain (with its paired normalizer) fits.
        save_brain(&dir);
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Ok));

        // A brain whose paired normalizer is MISSING is Refused — a real brain must never
        // arm against silently-cold normalizer stats.
        std::fs::remove_file(CheckpointDir::new(&dir).normalizer_path()).unwrap();
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Refused(_)));
        save_brain(&dir); // restore the pair

        save_brain_with_obs_dim(&dir, OBS_SIZE + 4);
        assert!(matches!(
            checkpoint_fits_rig(&dir),
            RigFit::Mismatch(RigDims { obs, .. }) if obs == OBS_SIZE + 4
        ));

        // A present-but-corrupt brain.bin is Refused, not Missing.
        std::fs::write(CheckpointDir::new(&dir).brain_file(), b"truncated garbage").unwrap();
        assert!(matches!(checkpoint_fits_rig(&dir), RigFit::Refused(_)));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
