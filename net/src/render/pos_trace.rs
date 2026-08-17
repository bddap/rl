//! Position trace (rl#371): per-tick sim player position and per-frame
//! render-resolved camera position, interleaved in one CSV — the discriminator
//! between true (sim) jitter and jitter introduced between sim and pixels.
//!
//! Armed by `RL_POS_TRACE=<path>` at round install; unset, every hook is one
//! `Option` check. Sim positions are written on the exact i64 grid (10 µm) so the
//! trace adds no quantization of its own; the camera line is the render-frame f32
//! translation actually handed to the GPU, printed at full f32 precision.
//!
//! Line format (comma-separated, one record per line):
//! - `O,<tick>,<x>,<z>` — the [`super::RenderOrigin`] in grid units, written
//!   whenever it changes (in practice once per round).
//! - `T,<tick>,<x>,<z>,<alt>` — local player sim position after this tick, grid
//!   units.
//! - `F,<tick>,<frac>,<dt_us>,<cx>,<cy>,<cz>` — the frame's [`RenderClock`] and
//!   camera translation (render-frame meters) after `apply_transforms`, plus the
//!   frame's `Time` delta in microseconds.

use std::fs::File;
use std::io::{BufWriter, Write};

use bevy::prelude::*;

use crate::sim::Pos;

/// The armed trace, or `None` (the shipping default). Lives outside the round
/// scope: one file per process, rounds delimited by their `O` lines.
#[derive(Resource, Default)]
pub(crate) struct PosTrace(pub(crate) Option<Trace>);

pub(crate) struct Trace {
    w: BufWriter<File>,
    origin: Option<Pos>,
}

impl PosTrace {
    /// Arm from `RL_POS_TRACE`. A path that cannot be created is a hard error:
    /// a capture run that silently records nothing is the silent fallback this
    /// repo bans.
    pub(crate) fn from_env() -> Self {
        let Some(path) = std::env::var_os("RL_POS_TRACE") else {
            return Self(None);
        };
        let file = File::create(&path)
            .unwrap_or_else(|e| panic!("RL_POS_TRACE={}: {e}", path.to_string_lossy()));
        let mut w = BufWriter::new(file);
        writeln!(w, "# rl371 pos trace v1").expect("pos trace header");
        info!("pos trace armed: {}", path.to_string_lossy());
        Self(Some(Trace { w, origin: None }))
    }

    /// The local player's sim position after tick `tick` just advanced.
    pub(crate) fn tick(&mut self, tick: u64, pos: Pos, alt: i64) {
        if let Some(t) = &mut self.0 {
            writeln!(t.w, "T,{tick},{},{},{alt}", pos.x, pos.z).expect("pos trace write");
        }
    }

    /// The frame's resolved camera, after every camera write of the frame.
    pub(crate) fn frame(
        &mut self,
        tick: u64,
        frac: f32,
        dt: std::time::Duration,
        cam: Vec3,
        origin: Pos,
    ) {
        if let Some(t) = &mut self.0 {
            if t.origin != Some(origin) {
                t.origin = Some(origin);
                writeln!(t.w, "O,{tick},{},{}", origin.x, origin.z).expect("pos trace write");
            }
            writeln!(
                t.w,
                "F,{tick},{frac},{},{:.9e},{:.9e},{:.9e}",
                dt.as_micros(),
                cam.x,
                cam.y,
                cam.z
            )
            .expect("pos trace write");
        }
    }
}
