//! The RECORDER half of frame-time telemetry (bddap/rl#309), platform-free by
//! construction: the render thread owns a fixed-size log-bucketed histogram —
//! recording is float math + an array increment, no allocation, no lock, no
//! syscall — and ~1 Hz of accumulated frames it snapshots the buckets into a
//! lock-free bounded queue (backpressure drops the OLDEST second — a lost second
//! is fine, growth is not).
//!
//! What happens to the snapshots is the SINK's business, and the sink is the
//! platform seam (rl#411): the process' telemetry init registers one via
//! [`install_sink`] before the app boots — natively `otel` spawns a flusher
//! thread that replays snapshots into the OTLP metrics SDK; a platform with no
//! exporter installs nothing and the bounded queue sheds the snapshots. This
//! crate itself has no thread, no clock, and no exporter dependency.

use std::sync::{Arc, OnceLock};

use crossbeam_queue::ArrayQueue;

/// Bucket count for the fixed histogram. 64 log-spaced buckets over 1..1000 ms is a
/// ~11% width per bucket — comfortably finer than the p50/p95/1%-low queries need,
/// while a snapshot stays one 256-byte memcpy.
pub const BUCKETS: usize = 64;

/// Top of the log range in ms (the bottom is 1 ms = 10^0). A >897 ms frame lands in
/// the open-ended top bucket; a <1.1 ms frame in the bottom one.
const MAX_MS: f64 = 1000.0;

/// Seconds of frames per snapshot — the owner's "flush ~1 Hz" bound on frame-path RAM.
const FLUSH_EVERY_SECS: f32 = 1.0;

/// Snapshots the queue holds before dropping oldest: a few seconds of sink stall
/// is absorbed, anything longer sheds data instead of RAM.
const QUEUE_DEPTH: usize = 8;

/// Upper boundaries of buckets `0..BUCKETS-1` (the last bucket is open-ended), in ms:
/// `1000^((j+1)/64)`. A sink that re-buckets the replayed [`midpoint_ms`] values with
/// these boundaries reproduces the frame-path counts exactly.
pub fn boundaries_ms() -> Vec<f64> {
    (0..BUCKETS - 1)
        .map(|j| MAX_MS.powf((j + 1) as f64 / BUCKETS as f64))
        .collect()
}

/// Geometric midpoint of bucket `i` — strictly inside `(boundary[i-1], boundary[i])`,
/// so a sink re-buckets a replayed midpoint into exactly bucket `i`.
pub fn midpoint_ms(i: usize) -> f64 {
    MAX_MS.powf((i as f64 + 0.5) / BUCKETS as f64)
}

/// Bucket index for a frame time. Pure float math — safe on the frame path.
pub fn bucket_index(dt_ms: f64) -> usize {
    // NaN and non-positive both file into bucket 0.
    if dt_ms.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return 0;
    }
    let idx = (dt_ms.ln() / MAX_MS.ln() * BUCKETS as f64).floor();
    idx.clamp(0.0, (BUCKETS - 1) as f64) as usize
}

/// One second of bucketed frame counts — what crosses the recorder→sink queue.
pub type Snapshot = [u32; BUCKETS];

/// The sink's end of the snapshot queue.
pub struct SnapshotRx(Arc<ArrayQueue<Snapshot>>);

impl SnapshotRx {
    /// Oldest pending snapshot, if any. Lock-free; safe to poll from anywhere.
    pub fn pop(&self) -> Option<Snapshot> {
        self.0.pop()
    }
}

type Sink = Box<dyn Fn(SnapshotRx) + Send + Sync>;

static SINK: OnceLock<Sink> = OnceLock::new();

/// Register the process' snapshot sink — called once by telemetry init, BEFORE the
/// app boots a recorder. The sink receives each recorder's [`SnapshotRx`] and owns
/// draining it (natively: spawn a flusher thread; a poll-driven platform can stash
/// the rx and drain it from its own loop). A second install is a wiring bug and is
/// refused loudly.
pub fn install_sink(sink: impl Fn(SnapshotRx) + Send + Sync + 'static) {
    if SINK.set(Box::new(sink)).is_err() {
        tracing::error!("frametime sink installed twice — keeping the first");
    }
}

/// The render-thread half: owns the fixed buckets, pushes a snapshot every
/// [`FLUSH_EVERY_SECS`] of accumulated frame time. Create once via [`start`].
pub struct FrameTelemetry {
    counts: Snapshot,
    seconds: f32,
    queue: Arc<ArrayQueue<Snapshot>>,
}

impl FrameTelemetry {
    /// Record one frame delta. Elapsed time is accumulated from the deltas the caller
    /// already has — no clock read on the frame path.
    pub fn frame(&mut self, dt_secs: f32) {
        if dt_secs <= 0.0 {
            return; // first-frame zero delta / paused tick — nothing rendered
        }
        self.counts[bucket_index(dt_secs as f64 * 1000.0)] += 1;
        self.seconds += dt_secs;
        if self.seconds >= FLUSH_EVERY_SECS {
            self.queue.force_push(self.counts); // full queue: drops the OLDEST snapshot
            self.counts = [0; BUCKETS];
            self.seconds = 0.0;
        }
    }
}

/// Build the recorder and hand its queue to the installed sink. With no sink
/// installed (a surface that never armed telemetry export) the bounded queue sheds
/// snapshots and the recorder still costs the frame path nothing.
pub fn start() -> FrameTelemetry {
    let queue = Arc::new(ArrayQueue::new(QUEUE_DEPTH));
    match SINK.get() {
        Some(sink) => sink(SnapshotRx(Arc::clone(&queue))),
        None => tracing::debug!("no frametime sink installed — frame histograms drop"),
    }
    FrameTelemetry {
        counts: [0; BUCKETS],
        seconds: 0.0,
        queue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An OTLP-style sink buckets a value into the first bucket whose upper bound
    /// is at least it (last bucket open-ended). The replay midpoints must
    /// round-trip: sink bucket of midpoint(i) == i, and our own frame-path index
    /// agrees.
    #[test]
    fn midpoints_rebucket_to_their_own_bucket() {
        let bounds = boundaries_ms();
        assert_eq!(bounds.len(), BUCKETS - 1);
        for i in 0..BUCKETS {
            let mid = midpoint_ms(i);
            let sink_bucket = bounds.iter().position(|b| mid <= *b).unwrap_or(BUCKETS - 1);
            assert_eq!(
                sink_bucket, i,
                "a sink would misfile midpoint of bucket {i}"
            );
            assert_eq!(bucket_index(mid), i, "frame-path index disagrees at {i}");
        }
    }

    #[test]
    fn bucket_index_covers_extremes() {
        assert_eq!(bucket_index(0.0), 0);
        assert_eq!(bucket_index(-1.0), 0);
        assert_eq!(bucket_index(f64::NAN), 0);
        assert_eq!(bucket_index(0.01), 0); // way under 1 ms
        assert_eq!(bucket_index(1e9), BUCKETS - 1); // absurd hitch
        // A 60 fps frame sits in the middle of the range, not at an edge.
        let idx = bucket_index(16.7);
        assert!((10..BUCKETS - 10).contains(&idx), "16.7ms at bucket {idx}");
    }

    /// Buckets around common frame rates are narrow enough that p50 read from a
    /// bucket midpoint is within ~6% of truth.
    #[test]
    fn resolution_is_fine_near_frame_rates() {
        for target in [7.0, 11.1, 16.7, 33.3] {
            let i = bucket_index(target);
            let width_ratio = midpoint_ms(i + 1) / midpoint_ms(i);
            assert!(
                width_ratio < 1.12,
                "bucket ratio {width_ratio} too coarse at {target}ms"
            );
        }
    }

    fn test_recorder(depth: usize) -> FrameTelemetry {
        FrameTelemetry {
            counts: [0; BUCKETS],
            seconds: 0.0,
            queue: Arc::new(ArrayQueue::new(depth)),
        }
    }

    #[test]
    fn snapshots_flush_at_one_hz_and_reset() {
        let mut t = test_recorder(QUEUE_DEPTH);
        // 1/64 s is exact in binary, so 64 deltas sum to exactly 1.0.
        for _ in 0..63 {
            t.frame(1.0 / 64.0);
        }
        assert!(t.queue.is_empty(), "flushed before a second accumulated");
        t.frame(1.0 / 64.0);
        let snap = t.queue.pop().expect("one second of frames must flush");
        assert_eq!(snap.iter().sum::<u32>(), 64);
        assert_eq!(t.counts.iter().sum::<u32>(), 0, "buckets must reset");
        assert_eq!(t.seconds, 0.0);
    }

    #[test]
    fn backpressure_drops_oldest_not_newest() {
        let mut t = test_recorder(2);
        // Three one-second batches with distinguishable frame counts: 1, 2, 3.
        for batch in 1..=3u32 {
            for _ in 0..batch {
                t.frame(1.1 / batch as f32);
            }
        }
        let first = t.queue.pop().expect("queue holds two snapshots");
        let second = t.queue.pop().expect("queue holds two snapshots");
        assert_eq!(first.iter().sum::<u32>(), 2, "oldest snapshot was dropped");
        assert_eq!(second.iter().sum::<u32>(), 3);
        assert!(t.queue.pop().is_none());
    }

    #[test]
    fn zero_and_negative_deltas_are_ignored() {
        let mut t = test_recorder(QUEUE_DEPTH);
        t.frame(0.0);
        t.frame(-0.5);
        assert_eq!(t.counts.iter().sum::<u32>(), 0);
        assert_eq!(t.seconds, 0.0);
    }
}
