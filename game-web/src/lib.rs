//! The browser entry adapter (rl#411) — the web peer of `game`'s CLI adapter.
//! Owns exactly the platform inputs, then calls the one shared entry
//! [`net::render::run_game`]: a tracing subscriber to the JS console, the one
//! `assets.pack` fetch that fills [`crab_world::assets`]' baked web store, a console
//! frame-rate sink, and the asset-root pin. Solo play makes ZERO network contact
//! beyond that same-origin pack fetch — a session binds only when the player enters
//! Host/Join (rl#412 cross-play: the same lobby as native, relay-backed).
#![cfg(target_family = "wasm")]

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// The whole baked asset tree, one blob relative to the page (rl#411 stage 6): the
/// dist step bakes every asset — weights, model, ambience, glyphs — so a hosted
/// build serves compiled artifacts only, never loose asset files.
const PACK_URL: &str = "assets.pack";

#[wasm_bindgen(start)]
pub fn boot() {
    console_error_panic_hook::set_once();
    init_console_tracing();
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = run().await {
            // The panic hook routes this to the console loudly; a broken bundle
            // must refuse to run, not degrade (rl#375).
            panic!("web boot failed: {e:#}");
        }
    });
}

async fn run() -> Result<()> {
    let pack = fetch_bytes(PACK_URL)
        .await
        .context("fetching the baked asset pack — build the bundle via game-web/run.sh dist")?;
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
        // Page-relative: bevy's wasm reader and the prefetch above both resolve
        // `assets/…` against the page URL, one tree for both byte paths.
        asset_root: PathBuf::new(),
    })
}

fn window() -> Result<web_sys::Window> {
    web_sys::window().ok_or_else(|| anyhow!("no JS window"))
}

async fn fetch_response(url: &str) -> Result<web_sys::Response> {
    let resp = JsFuture::from(window()?.fetch_with_str(url))
        .await
        .map_err(|e| anyhow!("fetch {url}: {e:?}"))?;
    let resp: web_sys::Response = resp
        .dyn_into()
        .map_err(|_| anyhow!("fetch {url}: not a Response"))?;
    anyhow::ensure!(resp.ok(), "fetch {url}: HTTP {}", resp.status());
    Ok(resp)
}

async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let resp = fetch_response(url).await?;
    let buf = JsFuture::from(resp.array_buffer().map_err(|e| anyhow!("{e:?}"))?)
        .await
        .map_err(|e| anyhow!("reading {url}: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
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
fn install_console_frametime_sink() {
    frametime::install_sink(|rx| {
        let tick = Closure::<dyn FnMut()>::new(move || {
            while let Some(snapshot) = rx.pop() {
                let frames: u32 = snapshot.iter().sum();
                if frames == 0 {
                    continue;
                }
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
