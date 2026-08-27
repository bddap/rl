//! The browser entry adapter (rl#411) — the web peer of `game`'s CLI adapter.
//! Owns exactly the platform inputs, then calls the one shared entry
//! [`net::render::run_game`]: a tracing subscriber to the JS console, the baked
//! asset pack (fetched by the PAGE — index.html owns every network byte so it can
//! show download progress, rl#413), a console frame-rate sink, and the asset-root
//! pin. Solo play makes ZERO network contact beyond the page's same-origin fetches
//! — a session binds only when the player enters Host/Join (rl#412 cross-play: the
//! same lobby as native, relay-backed).
#![cfg(target_family = "wasm")]

use std::path::PathBuf;

use anyhow::{Context, Result};
use wasm_bindgen::prelude::*;

/// The page's entry point, called with the fetched `assets.pack` bytes once both
/// downloads finish (index.html shows progress until then; a ~150 MB cold load is
/// minutes of otherwise-black screen on residential bandwidth — rl#413).
#[wasm_bindgen]
pub fn boot(pack: Vec<u8>) {
    console_error_panic_hook::set_once();
    init_console_tracing();
    // spawn_local, though nothing awaits: winit's wasm event loop exits `app.run()`
    // by throwing a control-flow JS exception, and from a plain exported fn that
    // throw would surface at the page's `boot()` callsite as a bogus error.
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(e) = run(pack) {
            // The panic hook routes this to the console loudly; a broken bundle
            // must refuse to run, not degrade (rl#375).
            panic!("web boot failed: {e:#}");
        }
    });
}

fn run(pack: Vec<u8>) -> Result<()> {
    let entries = crab_world::asset_pack::read_pack(&pack)
        .context("parsing assets.pack — a torn pack refuses to boot (rl#375)")?;
    tracing::info!(
        "WEB_ASSETS_PRELOADED count={} pack_bytes={}",
        entries.len(),
        pack.len()
    );
    crab_world::assets::preload_web_assets(
        entries
            .into_iter()
            .map(|(path, bytes)| (PathBuf::from(path), bytes)),
    );
    install_console_frametime_sink();

    // The web launch surface: menu boot — solo, host, and join all drive the same
    // menu as native (rl#412). No telemetry collector (TelemetrySender is
    // uninhabited on wasm) and no scripted lobby.
    net::render::run_game(net::render::GameConfig {
        launch: net::render::Launch::Menu,
        telemetry: None,
        nn_crab_checkpoints: Vec::new(),
        view: crab_world::RenderArgs::default(),
        // Page-relative: bevy's wasm reader and the page's prefetch both resolve
        // `assets/…` against the page URL, one tree for both byte paths.
        asset_root: PathBuf::new(),
    })
}

/// A tracing subscriber that writes each event to the JS console. `without_time`:
/// the fmt layer's default timer reads `SystemTime::now`, which panics on wasm.
fn init_console_tracing() {
    use tracing_subscriber::fmt::MakeWriter;

    struct ConsoleWriter(Vec<u8>);
    impl std::io::Write for ConsoleWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Drop for ConsoleWriter {
        fn drop(&mut self) {
            let line = String::from_utf8_lossy(&self.0);
            let line = line.trim_end();
            if !line.is_empty() {
                web_sys::console::log_1(&line.into());
            }
        }
    }
    struct MakeConsole;
    impl<'a> MakeWriter<'a> for MakeConsole {
        type Writer = ConsoleWriter;
        fn make_writer(&'a self) -> ConsoleWriter {
            ConsoleWriter(Vec::new())
        }
    }

    tracing_subscriber::fmt()
        .with_writer(MakeConsole)
        .with_ansi(false)
        .without_time()
        .with_max_level(tracing::Level::INFO)
        .init();
}

/// The web frametime sink: stash the recorder's queue and drain it on a JS interval,
/// logging one `WEB_FRAMETIME` line per snapshot — the machine-readable steady-rate
/// evidence the headless verify greps (and a human reads in devtools).
///
/// The first nonzero snapshot also retires the page's loading overlay (rl#413):
/// real frames are the one signal the game is actually presenting — removing the
/// overlay any earlier (e.g. at `boot()`) would re-open a silent black gap while
/// shaders compile.
fn install_console_frametime_sink() {
    frametime::install_sink(|rx| {
        let tick = Closure::<dyn FnMut()>::new(move || {
            while let Some(snapshot) = rx.pop() {
                let frames: u32 = snapshot.iter().sum();
                if frames == 0 {
                    continue;
                }
                remove_loading_overlay();
                let q = |p| frametime::percentile_ms(&snapshot, p).unwrap_or(f64::NAN);
                tracing::info!(
                    "WEB_FRAMETIME frames={frames} median_ms={:.1} p90_ms={:.1}",
                    q(0.5),
                    q(0.9),
                );
            }
        });
        if let Some(win) = web_sys::window() {
            let _ = win.set_interval_with_callback_and_timeout_and_arguments_0(
                tick.as_ref().unchecked_ref(),
                1000,
            );
        }
        // The interval owns the closure for the page's lifetime.
        tick.forget();
    });
}

fn remove_loading_overlay() {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("gcr-loading"))
    {
        el.remove();
    }
}
