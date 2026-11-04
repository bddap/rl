//! Per-frame timing probe: with `RL_FRAME_PROBE=<path>` set at launch, each stage
//! boundary of every frame appends a `stage,frame,t_ns` row stamped on
//! CLOCK_MONOTONIC, the clock an input-driving harness can log on too.
//!
//! `frame` is the main update's number; render stages carry the number of the main
//! update they draw, since pipelined rendering overlaps the two. `acquire_*`
//! brackets `prepare_windows`, which blocks in `get_current_texture`; `submit_*`
//! brackets the render graph's command-buffer submit, and `present_end` follows
//! `present()`. Between `main_end` and `extract` the main thread waits for the
//! render world to come back.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc;

use bevy::prelude::*;
use bevy::render::renderer::{RenderGraph, RenderGraphSystems, render_system};
use bevy::render::view::{create_surfaces, prepare_windows};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};

pub struct FrameProbePlugin;

impl Plugin for FrameProbePlugin {
    fn build(&self, app: &mut App) {
        let Some(path) = std::env::var_os("RL_FRAME_PROBE") else {
            return;
        };
        let file = File::create(&path)
            .unwrap_or_else(|e| panic!("RL_FRAME_PROBE={}: {e}", path.to_string_lossy()));
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("frame-probe".into())
            .spawn(move || write_rows(BufWriter::new(file), rx))
            .expect("spawn frame-probe writer");

        app.insert_resource(Probe(tx.clone()))
            .init_resource::<ProbeFrame>()
            .add_systems(First, main_begin)
            .add_systems(Last, stamp(Stage::MainEnd));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(Probe(tx))
            .init_resource::<ProbeFrame>()
            .add_systems(ExtractSchedule, extract)
            .add_systems(
                Render,
                (
                    stamp(Stage::RenderBegin).in_set(RenderSystems::ExtractCommands),
                    stamp(Stage::AcquireBegin)
                        .in_set(RenderSystems::PrepareViews)
                        .after(create_surfaces)
                        .before(prepare_windows),
                    stamp(Stage::AcquireEnd)
                        .in_set(RenderSystems::PrepareViews)
                        .after(prepare_windows),
                    stamp(Stage::PresentEnd)
                        .in_set(RenderSystems::Render)
                        .after(render_system),
                ),
            )
            .add_systems(
                RenderGraph,
                (
                    stamp(Stage::SubmitBegin)
                        .after(RenderGraphSystems::Render)
                        .before(RenderGraphSystems::Submit),
                    stamp(Stage::SubmitEnd).in_set(RenderGraphSystems::Finish),
                ),
            );
    }
}

#[derive(Clone, Copy)]
enum Stage {
    MainBegin,
    MainEnd,
    Extract,
    RenderBegin,
    AcquireBegin,
    AcquireEnd,
    SubmitBegin,
    SubmitEnd,
    PresentEnd,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::MainBegin => "main_begin",
            Stage::MainEnd => "main_end",
            Stage::Extract => "extract",
            Stage::RenderBegin => "render_begin",
            Stage::AcquireBegin => "acquire_begin",
            Stage::AcquireEnd => "acquire_end",
            Stage::SubmitBegin => "submit_begin",
            Stage::SubmitEnd => "submit_end",
            Stage::PresentEnd => "present_end",
        }
    }
}

struct Row {
    stage: Stage,
    frame: u64,
    t_ns: u64,
}

fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid out-pointer and CLOCK_MONOTONIC exists on Linux.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn write_rows(mut out: BufWriter<File>, rx: mpsc::Receiver<Row>) {
    let mut write = || -> std::io::Result<()> {
        out.write_all(b"stage,frame,t_ns\n")?;
        // Flushed each time the queue drains, so a killed game loses only rows in flight.
        while let Ok(first) = rx.recv() {
            for row in std::iter::once(first).chain(rx.try_iter()) {
                writeln!(out, "{},{},{}", row.stage.name(), row.frame, row.t_ns)?;
            }
            out.flush()?;
        }
        Ok(())
    };
    if let Err(e) = write() {
        warn!("frame probe stopped writing: {e}");
    }
}

#[derive(Resource)]
struct Probe(mpsc::Sender<Row>);

impl Probe {
    fn send(&self, stage: Stage, frame: u64) {
        // A dead writer has already warned; the game keeps running.
        let _ = self.0.send(Row {
            stage,
            frame,
            t_ns: now_ns(),
        });
    }
}

#[derive(Resource, Default)]
struct ProbeFrame(u64);

fn main_begin(mut frame: ResMut<ProbeFrame>, probe: Res<Probe>) {
    frame.0 += 1;
    probe.send(Stage::MainBegin, frame.0);
}

fn extract(main: Extract<Res<ProbeFrame>>, mut frame: ResMut<ProbeFrame>, probe: Res<Probe>) {
    frame.0 = main.0;
    probe.send(Stage::Extract, frame.0);
}

fn stamp(stage: Stage) -> impl FnMut(Res<ProbeFrame>, Res<Probe>) {
    move |frame, probe| probe.send(stage, frame.0)
}
