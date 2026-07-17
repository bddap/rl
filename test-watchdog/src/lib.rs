//! rl#282 suite-level stall watchdog.
//!
//! Under heavy trainer load, test suites in this workspace occasionally wedge
//! forever: every thread parked in futex_wait at 0% CPU (mechanism undiagnosed;
//! observed in the crab_world lib suite and the armed-render probe, 45 min–1 h 20 m
//! until an external kill). The watchdog keys on that exact signature — process
//! CPU time flatlining while the suite is still running — so a merely *slow* run
//! on a saturated box cannot false-fire: a starved-but-runnable thread still
//! accrues CPU every scheduler pass; a wedged one accrues none.
//!
//! Arm once per test binary with `test_watchdog::arm!();` (in a lib suite, gate
//! the invocation with `#[cfg(test)]`). On a stall it dumps every thread's name
//! and state — libtest names worker threads after the test they run, so the dump
//! names the wedged test — then aborts, turning an hour-plus silent hang into a
//! loud, attributable failure.

use std::fmt::Write as _;
use std::sync::Once;
use std::time::Duration;

#[doc(hidden)]
pub use ctor;

/// Arm the watchdog at test-binary load, before any test runs — expands to a
/// `ctor` constructor so no individual test has to remember to call [`arm`].
#[macro_export]
macro_rules! arm {
    () => {
        #[$crate::ctor::ctor(unsafe, crate_path = $crate::ctor)]
        fn __rl282_arm_stall_watchdog() {
            $crate::arm();
        }
    };
}

/// CPU flatline long enough to declare the wedge.
const STALL_WINDOW: Duration = Duration::from_secs(120);

/// Progress = at least this much fresh CPU since the last anchor. Far above the
/// watchdog's own polling cost (~1 ms per poll), far below what any runnable
/// thread earns per window even on a box saturated by four trainers.
const PROGRESS_MIN: Duration = Duration::from_millis(250);

pub fn arm() {
    arm_with(STALL_WINDOW);
}

fn arm_with(window: Duration) {
    static ARMED: Once = Once::new();
    ARMED.call_once(|| {
        std::thread::Builder::new()
            .name("stall-watchdog".into())
            .spawn(move || {
                // A dead watchdog is a silently un-guarded suite — the failure
                // shape this crate exists to kill. Nothing in the loop should
                // panic on Linux; if it does, go down loudly.
                let _ = std::panic::catch_unwind(move || watch(window));
                eprintln!(
                    "test-watchdog: stall-watchdog thread died — this suite is no \
                     longer guarded against the rl#282 wedge; aborting rather than \
                     running unprotected"
                );
                std::process::abort();
            })
            .expect("spawn stall watchdog");
    });
}

fn watch(window: Duration) -> ! {
    let poll = (window / 24).clamp(Duration::from_millis(50), Duration::from_secs(5));
    let mut detector = StallDetector::new(process_cpu(), window);
    loop {
        std::thread::sleep(poll);
        if detector.observe(process_cpu(), poll) {
            fire(window);
        }
    }
}

struct StallDetector {
    window: Duration,
    anchor: Duration,
    stalled: Duration,
}

impl StallDetector {
    fn new(cpu: Duration, window: Duration) -> Self {
        Self {
            window,
            anchor: cpu,
            stalled: Duration::ZERO,
        }
    }

    /// True once the process has gone a full window without accruing
    /// `PROGRESS_MIN` of fresh CPU. Accrual is measured against the last anchor,
    /// not per poll, so a starved suite dripping CPU slowly still re-anchors.
    fn observe(&mut self, cpu: Duration, elapsed: Duration) -> bool {
        if cpu.saturating_sub(self.anchor) >= PROGRESS_MIN {
            self.anchor = cpu;
            self.stalled = Duration::ZERO;
            return false;
        }
        self.stalled += elapsed;
        self.stalled >= self.window
    }
}

/// utime+stime of the whole process (every thread), from /proc/self/stat.
/// Excludes live child-process CPU (cutime/cstime land only after wait), so an
/// armed test that parked a full window while a subprocess worked would
/// false-fire — no armed suite spawns subprocesses today; keep it that way or
/// widen this.
fn process_cpu() -> Duration {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("read /proc/self/stat");
    // comm may contain spaces or parens; real fields resume after the LAST ')'.
    let (_, rest) = stat.rsplit_once(')').expect("malformed /proc/self/stat");
    let mut fields = rest.split_whitespace();
    let utime: u64 = fields.nth(11).and_then(|f| f.parse().ok()).expect("utime");
    let stime: u64 = fields.next().and_then(|f| f.parse().ok()).expect("stime");
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as u64;
    Duration::from_secs_f64((utime + stime) as f64 / hz as f64)
}

fn fire(window: Duration) -> ! {
    let mut dump = String::new();
    if let Ok(tasks) = std::fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            let status = std::fs::read_to_string(task.path().join("status")).unwrap_or_default();
            let field = |key: &str| {
                status
                    .lines()
                    .find_map(|l| l.strip_prefix(key))
                    .map_or("?", str::trim)
            };
            let _ = writeln!(
                dump,
                "  tid {:>8}  {:<20} {}",
                field("Pid:"),
                field("Name:"),
                field("State:")
            );
        }
    }
    eprintln!(
        "\ntest-watchdog: process CPU flatlined for {window:?} with the suite still \
         running — the rl#282 wedge (every thread parked at 0% CPU; so far seen only \
         alongside 4+ live trainers). Aborting loudly instead of hanging until an \
         external kill. Check machine load before suspecting the code under test; a \
         fire on a QUIET box is a new diagnostic lead — reopen rl#282 with this dump.\n\
         threads (name = the test a libtest worker is running):\n{dump}"
    );
    std::process::abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Duration = Duration::from_secs(120);
    const POLL: Duration = Duration::from_secs(5);

    #[test]
    fn steady_progress_never_fires() {
        let mut d = StallDetector::new(Duration::ZERO, WINDOW);
        let mut cpu = Duration::ZERO;
        for _ in 0..100 {
            cpu += Duration::from_millis(300);
            assert!(!d.observe(cpu, POLL));
        }
    }

    #[test]
    fn slow_drip_reanchors_instead_of_firing() {
        // 100 ms per poll is under PROGRESS_MIN per poll but crosses it
        // cumulatively every third poll — a starved suite never fires.
        let mut d = StallDetector::new(Duration::ZERO, WINDOW);
        let mut cpu = Duration::ZERO;
        for _ in 0..100 {
            cpu += Duration::from_millis(100);
            assert!(!d.observe(cpu, POLL));
        }
    }

    #[test]
    fn flatline_fires_exactly_at_the_window() {
        let base = Duration::from_secs(100);
        let mut d = StallDetector::new(base, WINDOW);
        // Only the watchdog's own polling cost accrues.
        let cpu = base + Duration::from_millis(10);
        for _ in 0..23 {
            assert!(!d.observe(cpu, POLL));
        }
        assert!(d.observe(cpu, POLL));
    }

    /// End-to-end fire path: re-exec this test binary as a wedged child (armed,
    /// then parked at 0 CPU) and require the loud abort with the rl#282 signature
    /// and the wedged test's name in the thread dump.
    #[test]
    fn fires_and_aborts_the_wedged_process() {
        const CHILD: &str = "TEST_WATCHDOG_WEDGED_CHILD";
        if std::env::var_os(CHILD).is_some() {
            arm_with(Duration::from_secs(1));
            // Long enough that only the watchdog's abort can end the child in
            // time, short enough that a broken watchdog fails this test in ~1 min
            // instead of wedging the suite.
            std::thread::sleep(Duration::from_secs(60));
            unreachable!("watchdog should have aborted the wedged child");
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["fires_and_aborts_the_wedged_process", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .expect("spawn wedged child");
        assert!(
            !out.status.success(),
            "child should have aborted, got {:?}",
            out.status
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("rl#282"), "no wedge signature:\n{stderr}");
        // comm truncates at 15 bytes, so match the truncated test-thread name.
        assert!(
            stderr.contains("tests::fires"),
            "dump should name the wedged test:\n{stderr}"
        );
    }
}
