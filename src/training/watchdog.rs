//! Startup watchdog: re-exec a `learn` process that deadlocks BEFORE iteration 0.
//!
//! # Why this exists
//! `matrixmultiply`'s shared inner-gemm thread tree can deadlock when K rollout
//! threads first race it during their world build/warmup (see
//! [`super::inproc::init_process_pools`]). Pinning `MATMUL_NUM_THREADS=1` is the
//! real fix and clears it in the overwhelming majority of runs, but the race is
//! probabilistic at process startup and has still been observed to wedge a fresh
//! run under load — all threads futex-blocked, iteration 0 never reached.
//!
//! A deadlocked process cannot un-deadlock itself, but a *fresh* process re-rolls
//! that startup race and almost always clears it. So a dedicated watchdog thread —
//! spawned before any world is built, so it can never be one of the threads that
//! blocks on the gemm tree — waits for proof that training is progressing. If that
//! proof does not arrive within a timeout, it concludes a pre-iter-0 deadlock and
//! replaces the whole process image with a fresh copy via `execv`, wiping the
//! wedged worker threads. An attempt counter carried in the environment bounds the
//! re-execs so a genuine (non-race) hang can't loop forever.
//!
//! The watchdog only ever fires before iteration 0: once training proves it is
//! progressing the signal is set and the thread exits, so it cannot interfere with
//! a long run that later stalls for an unrelated reason.

use std::os::unix::process::CommandExt;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Env var (seconds): how long to wait for iteration 0 before concluding a
/// pre-iter-0 deadlock and re-execing. `0` disables the watchdog entirely (escape
/// hatch). World build normally takes seconds, so the default leaves huge margin
/// and ~zero false-positive risk.
const TIMEOUT_ENV: &str = "RL_WATCHDOG_TIMEOUT_SECS";
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Env var: maximum number of re-execs before giving up. Bounds the retry so a
/// hang that is NOT the probabilistic matmul race (which a fresh process can't
/// clear) can't re-exec forever — after this many attempts the watchdog exits
/// non-zero for external intervention instead.
const MAX_ATTEMPTS_ENV: &str = "RL_WATCHDOG_MAX_ATTEMPTS";
const DEFAULT_MAX_ATTEMPTS: u32 = 5;

/// Env var: internal re-exec counter, incremented across the `execv` and read back
/// by the fresh process so the bound survives the image replacement (env is the
/// natural channel — it carries across `execv` for free). Not user-facing; an
/// operator overrides the bound via [`MAX_ATTEMPTS_ENV`], not this.
const ATTEMPT_ENV: &str = "RL_WATCHDOG_ATTEMPT";

/// Process exit code when the watchdog gives up or can't re-exec. `EX_SOFTWARE`
/// (sysexits.h): a non-zero "internal error" a supervisor can distinguish from a
/// clean stop, signalling that automated re-rolling did not clear the hang.
const GIVE_UP_EXIT_CODE: i32 = 70;

/// Resolved watchdog configuration, parsed once from the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WatchdogConfig {
    /// `None` = disabled (timeout env was `0`); `Some` = wait this long for iter 0.
    pub timeout: Option<Duration>,
    /// This process's re-exec attempt number: 0 on the first launch, N after N
    /// re-execs.
    pub attempt: u32,
    /// Re-exec at most this many times before giving up.
    pub max_attempts: u32,
}

impl WatchdogConfig {
    /// Read the config from the environment, applying the documented defaults. A
    /// non-numeric or absent value falls back to the default; a timeout of `0`
    /// disables the watchdog.
    pub(crate) fn from_env() -> Self {
        let timeout_secs = parse_env_or(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECS);
        Self {
            timeout: (timeout_secs != 0).then(|| Duration::from_secs(timeout_secs)),
            attempt: parse_env_or(ATTEMPT_ENV, 0),
            max_attempts: parse_env_or(MAX_ATTEMPTS_ENV, DEFAULT_MAX_ATTEMPTS),
        }
    }
}

/// Parse an env var as `T`, falling back to `default` when unset or unparsable.
fn parse_env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

/// What the watchdog should do when the iter-0 timeout elapses without the
/// "first iteration reached" signal. Pure decision so it is unit-testable without
/// spawning threads or actually re-execing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeoutAction {
    /// Re-exec as attempt `next_attempt` of `max_attempts`, re-rolling the race.
    ReExec {
        next_attempt: u32,
        max_attempts: u32,
    },
    /// The retry budget is spent — log and exit non-zero for external intervention.
    GiveUp { attempts: u32, max_attempts: u32 },
}

/// Decide the timeout action from the attempt counter alone (pure). Called only
/// once the timeout has already elapsed without the progress signal.
///
/// `attempt` is how many re-execs have happened so far (0 on the first launch).
/// Re-exec while strictly fewer than `max_attempts` re-execs have been done; the
/// re-exec that would make the count reach `max_attempts` is still allowed, and the
/// NEXT process — finding `attempt == max_attempts` — is the one that gives up. So
/// `max_attempts` is exactly the number of re-execs performed across the chain.
fn decide_timeout_action(attempt: u32, max_attempts: u32) -> TimeoutAction {
    if attempt < max_attempts {
        TimeoutAction::ReExec {
            next_attempt: attempt + 1,
            max_attempts,
        }
    } else {
        TimeoutAction::GiveUp {
            attempts: attempt,
            max_attempts,
        }
    }
}

/// A one-shot flag the learner sets the instant it KNOWS the world build completed
/// and training is progressing (iteration 0 reached). Setting it disarms the
/// watchdog; the watchdog polls it and, once set, exits quietly without ever
/// touching the process.
///
/// Cloning shares the same underlying flag (it is an `Arc`), so the learner holds
/// one handle and the watchdog thread another.
#[derive(Clone)]
pub(crate) struct ProgressSignal(Arc<AtomicBool>);

impl ProgressSignal {
    fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Mark iteration 0 reached — the world build is past, no deadlock occurred.
    /// Idempotent: calling it more than once is harmless (only the first matters).
    pub(crate) fn mark_reached(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    fn is_reached(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Arm the startup watchdog for a `learn` run and return the [`ProgressSignal`] the
/// learner must call [`ProgressSignal::mark_reached`] on once iteration 0 is reached.
///
/// MUST be called BEFORE any rollout world is built, so the watchdog thread is not
/// one of the threads that can block on the shared gemm tree. Disabled (no thread
/// spawned, returns immediately) when the timeout env is `0`.
///
/// On the timeout firing without the signal, the watchdog either re-execs this
/// process (a fresh image re-rolls the probabilistic startup race) or, once the
/// retry budget is spent, exits the process non-zero. Both paths terminate the
/// current image, so this never returns control on a fired timeout.
pub(crate) fn arm(config: WatchdogConfig) -> ProgressSignal {
    let signal = ProgressSignal::new();
    let Some(timeout) = config.timeout else {
        eprintln!("[watchdog] disabled ({TIMEOUT_ENV}=0)");
        return signal;
    };

    eprintln!(
        "[watchdog] armed: re-exec if iteration 0 is not reached within {}s \
         (attempt {} of max {} re-execs)",
        timeout.as_secs(),
        config.attempt,
        config.max_attempts,
    );

    let watch = signal.clone();
    // A dedicated thread that does nothing but poll the flag and watch the clock.
    // It must NOT touch the gemm tree, build a world, or run any matmul, so it can
    // never be a party to the deadlock it exists to break.
    std::thread::Builder::new()
        .name("rl-watchdog".into())
        .spawn(move || run(watch, timeout, config.attempt, config.max_attempts))
        .expect("spawn watchdog thread");

    signal
}

/// Poll interval. Short relative to the (≥ seconds, default 120 s) timeout, so the
/// extra wait past the deadline is negligible while the thread stays near-idle.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The watchdog loop: wait for the progress signal until the deadline, then act.
fn run(signal: ProgressSignal, timeout: Duration, attempt: u32, max_attempts: u32) {
    // Clamp the deadline rather than computing `now + timeout` directly, so an absurd
    // operator-set timeout can't overflow `Instant` and panic the watchdog thread
    // (which would silently disarm it — the opposite of what an over-cautious timeout
    // intends). The fallback offset (~136 years) is "effectively never" in practice.
    let deadline = Instant::now()
        .checked_add(timeout)
        .unwrap_or_else(|| Instant::now() + Duration::from_secs(u32::MAX as u64));
    // Loop until the deadline passes. The final check after the loop also catches the
    // signal firing inside the last poll gap, so a process that reached iter 0 just
    // under the wire is never re-execed.
    while Instant::now() < deadline && !signal.is_reached() {
        std::thread::sleep(POLL_INTERVAL);
    }
    if signal.is_reached() {
        // Iteration 0 reached — the build is past and no deadlock occurred. The
        // watchdog has done its job; leave the process untouched.
        eprintln!("[watchdog] iteration 0 reached — disarmed");
        return;
    }

    match decide_timeout_action(attempt, max_attempts) {
        TimeoutAction::ReExec {
            next_attempt,
            max_attempts,
        } => {
            eprintln!(
                "[watchdog] iteration 0 not reached within {}s — assuming pre-iter-0 \
                 deadlock (matmul shared-gemm-tree race); re-execing (attempt {next_attempt} of {max_attempts})",
                timeout.as_secs(),
            );
            re_exec(next_attempt);
        }
        TimeoutAction::GiveUp {
            attempts,
            max_attempts,
        } => {
            eprintln!(
                "[watchdog] exhausted {attempts} of {max_attempts} watchdog re-execs; \
                 pre-iter-0 hang is not clearing, may not be the matmul race — exiting for \
                 external intervention"
            );
            std::process::exit(GIVE_UP_EXIT_CODE);
        }
    }
}

/// Replace this process image with a fresh copy of the same binary, re-running the
/// original arguments and bumping the attempt counter so the bound survives. `execv`
/// does not return on success (the image is gone); a failure to even launch is fatal
/// and reported.
///
/// `current_exe()` (canonicalized, symlinks resolved) becomes the new `argv[0]` — so
/// the program's *arguments* are preserved verbatim but `argv[0]` is the resolved
/// binary path rather than however it was originally invoked. The trainer never
/// dispatches on `argv[0]`, so this is immaterial here.
fn re_exec(next_attempt: u32) -> ! {
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("[watchdog] cannot resolve current exe to re-exec: {e}");
        std::process::exit(GIVE_UP_EXIT_CODE);
    });
    // The inherited environment carries through `execv` unchanged except for the
    // bumped attempt counter set here, which is how the retry bound survives.
    let mut cmd = Command::new(exe);
    cmd.args(std::env::args_os().skip(1));
    cmd.env(ATTEMPT_ENV, next_attempt.to_string());

    let err = cmd.exec();
    // Only reachable if exec itself failed — the new image never started.
    eprintln!("[watchdog] re-exec failed: {err}");
    std::process::exit(GIVE_UP_EXIT_CODE);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The retry bound: re-exec while strictly fewer than `max` re-execs have been
    /// done, and the re-exec that reaches the cap is itself allowed; the process
    /// that finds `attempt == max` is the one that gives up. So a `max` of M yields
    /// exactly M re-execs across the chain.
    #[test]
    fn timeout_action_respects_attempt_bound() {
        // First launch (no re-execs yet) with a budget of 5: re-exec as attempt 1.
        assert_eq!(
            decide_timeout_action(0, 5),
            TimeoutAction::ReExec {
                next_attempt: 1,
                max_attempts: 5
            }
        );
        // Mid-chain: still under the cap.
        assert_eq!(
            decide_timeout_action(4, 5),
            TimeoutAction::ReExec {
                next_attempt: 5,
                max_attempts: 5
            }
        );
        // The Mth re-exec has happened; this process is the one that gives up.
        assert_eq!(
            decide_timeout_action(5, 5),
            TimeoutAction::GiveUp {
                attempts: 5,
                max_attempts: 5
            }
        );
        // Over the cap (e.g. a hand-set counter) also gives up — never re-execs past.
        assert_eq!(
            decide_timeout_action(6, 5),
            TimeoutAction::GiveUp {
                attempts: 6,
                max_attempts: 5
            }
        );
    }

    /// `max_attempts == 0` means "never re-exec": the very first timeout gives up.
    /// (Distinct from `timeout == 0`, which disables the watchdog so it never even
    /// reaches a timeout.)
    #[test]
    fn zero_max_attempts_never_re_execs() {
        assert_eq!(
            decide_timeout_action(0, 0),
            TimeoutAction::GiveUp {
                attempts: 0,
                max_attempts: 0
            }
        );
    }

    /// `mark_reached` latches the flag, and clones observe it (the learner sets one
    /// handle, the watchdog thread polls another).
    #[test]
    fn progress_signal_latches_and_shares() {
        let signal = ProgressSignal::new();
        assert!(!signal.is_reached(), "fresh signal must be unset");
        let clone = signal.clone();
        signal.mark_reached();
        assert!(signal.is_reached(), "mark_reached() must latch the signal");
        assert!(
            clone.is_reached(),
            "a clone shares the underlying flag, so it sees the latch"
        );
    }

    /// The real cancel path: a signal already set when `run` starts makes it return
    /// at once — before the deadline and WITHOUT re-execing. Driving the actual `run`
    /// (not just the flag) is what pins spec item (2); with the signal set the only
    /// reachable outcome is the early disarm return, so this can never reach `exec`
    /// and kill the test process. A long timeout proves it returns immediately rather
    /// than waiting out the clock.
    #[test]
    fn run_returns_immediately_when_already_signalled() {
        let signal = ProgressSignal::new();
        signal.mark_reached();
        let watch = signal.clone();

        let handle = std::thread::spawn(move || {
            // attempt/max are irrelevant: the set signal short-circuits before any
            // timeout decision. A 1-hour timeout would block this thread forever if
            // the signal were ignored — the join-deadline below would then catch it.
            run(watch, Duration::from_secs(3600), 0, 5);
        });

        let start = Instant::now();
        while !handle.is_finished() && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            handle.is_finished(),
            "run must return at once when the signal is already set, not wait the timeout"
        );
        handle.join().expect("watchdog run thread panicked");
    }

    /// `RL_WATCHDOG_TIMEOUT_SECS=0` disables the watchdog (no thread, no re-exec).
    #[test]
    fn zero_timeout_disables() {
        let config = WatchdogConfig {
            timeout: None,
            attempt: 0,
            max_attempts: 5,
        };
        // arm() with a None timeout returns a fresh (unset) signal and spawns no
        // thread; the signal it hands back is inert. We assert the config shape that
        // drives that branch here (arm's no-thread path has no observable effect to
        // assert beyond not blocking, which this config selects).
        assert!(config.timeout.is_none(), "timeout=0 must disable");
        let signal = arm(config);
        assert!(
            !signal.is_reached(),
            "disabled watchdog still returns a usable (unset) signal"
        );
    }

    /// Env parsing: defaults when unset/garbage, `0` timeout disables, values honored.
    /// All env mutation is confined to this one test so its set/remove pairs don't
    /// interleave with a sibling's — cargo runs the module's tests on parallel threads
    /// in one process, and `set_var`/`remove_var` are UB if another thread is in any
    /// `getenv`/`setenv` at the same time. No other test here mutates the environment
    /// (the rest take explicit configs), so within this crate these are the only env
    /// writers and the writes are safe.
    #[test]
    fn config_from_env_parses_and_defaults() {
        // Helper to run a closure with a clean set of the three vars, restoring after.
        fn with_env(vars: &[(&str, Option<&str>)], f: impl FnOnce()) {
            let saved: Vec<(String, Option<String>)> = vars
                .iter()
                .map(|(k, _)| (k.to_string(), std::env::var(k).ok()))
                .collect();
            for (k, v) in vars {
                match v {
                    // SAFETY: see the test's doc comment — this is the sole env-mutating
                    // test in the module, so no other thread reads/writes env concurrently.
                    Some(v) => unsafe { std::env::set_var(k, v) },
                    None => unsafe { std::env::remove_var(k) },
                }
            }
            f();
            for (k, v) in saved {
                match v {
                    Some(v) => unsafe { std::env::set_var(&k, v) },
                    None => unsafe { std::env::remove_var(&k) },
                }
            }
        }

        with_env(
            &[
                (TIMEOUT_ENV, None),
                (MAX_ATTEMPTS_ENV, None),
                (ATTEMPT_ENV, None),
            ],
            || {
                let c = WatchdogConfig::from_env();
                assert_eq!(c.timeout, Some(Duration::from_secs(DEFAULT_TIMEOUT_SECS)));
                assert_eq!(c.max_attempts, DEFAULT_MAX_ATTEMPTS);
                assert_eq!(c.attempt, 0);
            },
        );

        with_env(
            &[
                (TIMEOUT_ENV, Some("0")),
                (MAX_ATTEMPTS_ENV, Some("3")),
                (ATTEMPT_ENV, Some("2")),
            ],
            || {
                let c = WatchdogConfig::from_env();
                assert_eq!(c.timeout, None, "0 disables");
                assert_eq!(c.max_attempts, 3);
                assert_eq!(c.attempt, 2);
            },
        );

        with_env(
            &[
                (TIMEOUT_ENV, Some("45")),
                (MAX_ATTEMPTS_ENV, Some("garbage")),
                (ATTEMPT_ENV, Some("")),
            ],
            || {
                let c = WatchdogConfig::from_env();
                assert_eq!(c.timeout, Some(Duration::from_secs(45)));
                assert_eq!(
                    c.max_attempts, DEFAULT_MAX_ATTEMPTS,
                    "garbage falls back to default"
                );
                assert_eq!(c.attempt, 0, "empty falls back to default");
            },
        );
    }
}
