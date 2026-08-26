//! Browser half of the browser↔native CROSS-PLAY verify (rl#412) — see
//! `examples/web-form/run.sh` for the whole flow. Drives the REAL lobby path a
//! browser player takes: `menu::begin(Join(code))` over the pollable web bind,
//! formation to agreement with a native host through the n0 relays, then the REAL
//! round drivers (`Coordinator` as a remote client) shipping inputs UP and adopting
//! authoritative snapshots DOWN. Logs one `HASH <tick> <hash>` line per adopted tick
//! in `game net --hash-log`'s shape, so the harness can diff the two sides for
//! byte-identical state — the both-ways proof: a matching hash requires the host to
//! have integrated THIS browser's inputs into the state the browser adopted.

#[cfg(target_family = "wasm")]
mod wasm {
    use std::time::Duration;

    use net::menu::{self, StartChoice};
    use net::net_loop::Coordinator;
    use net::sim::{Input, TICK_DT, TICK_HZ};
    use wasm_bindgen::prelude::*;

    fn log(s: &str) {
        web_sys::console::log_1(&JsValue::from_str(s));
    }

    const FORM_TIMEOUT_TICKS: u32 = 120_000 / 50;

    #[wasm_bindgen]
    pub async fn run_form(host_code: String, run_secs: u32) -> Result<(), JsValue> {
        console_error_panic_hook::set_once();
        match form_and_run(host_code, run_secs).await {
            Ok(()) => Ok(()),
            Err(e) => {
                log(&format!("FORM_FAIL={e:#}"));
                Err(JsValue::from_str(&format!("{e:#}")))
            }
        }
    }

    async fn form_and_run(host_code: String, run_secs: u32) -> anyhow::Result<()> {
        let host: iroh::EndpointId = host_code.parse()?;
        // Crabless stamp, matching `game net`'s headless host (rest-pose statue).
        let stamp = net::SyncStamp::local(0);
        let mut forming = menu::begin(&StartChoice::Join(Some(host)), 7, None, stamp);

        let mut announced = false;
        let mut last_roster = 0usize;
        let mut waited = 0u32;
        let result = loop {
            if let Some(id) = forming.my_id()
                && !announced
            {
                announced = true;
                log(&format!("FORM_BOUND id={id}"));
            }
            let roster = forming.lobby_len();
            if roster != last_roster {
                last_roster = roster;
                log(&format!("FORM_ROSTER n={roster}"));
            }
            if let Some(r) = forming.poll() {
                break r?;
            }
            waited += 1;
            anyhow::ensure!(waited < FORM_TIMEOUT_TICKS, "formation timed out");
            n0_future::time::sleep(Duration::from_millis(50)).await;
        };

        let ready = menu::ready_from(result, 7).expect("verify probe never cancels");
        let net = ready.net.ok_or_else(|| {
            anyhow::anyhow!("formed ALONE — the host was never reached (dial/relay path failed)")
        })?;
        let mut client = ready.client;
        anyhow::ensure!(!net.is_host(), "the native peer must host; we joined");
        let peers = client.peers().to_vec();
        let me = client.me();
        log(&format!("FORM_OK players={} me={me:?}", peers.len()));

        let mut coord = Coordinator::for_round(Some(net), &peers, me, client.sim().clone());
        let mut adopted = 0usize;
        let tick_dt = Duration::from_secs_f64(TICK_DT);
        for _ in 0..u64::from(run_secs) * TICK_HZ {
            let t = client.next_tick() as f32 * 0.1;
            let input = Input::from_axes(t.cos(), t.sin());
            let msg = client.submit_local_input(input, None);
            let ex = match coord.exchange(msg) {
                Ok(ex) => ex,
                Err(down) => anyhow::bail!("round ended early: {down}"),
            };
            let mut hashes: Vec<(u64, u64)> = Vec::new();
            adopted += client.adopt_snapshots(ex.snapshots, |c| {
                hashes.push((c.sim().tick().saturating_sub(1), c.sim().state_hash()));
            });
            for (tick, hash) in hashes {
                log(&format!("HASH {tick} {hash:#018x}"));
            }
            n0_future::time::sleep(tick_dt).await;
        }
        anyhow::ensure!(adopted > 0, "no snapshots adopted — state never flowed down");
        log(&format!(
            "XPLAY_OK final_tick={} adopted={adopted}",
            client.sim().tick()
        ));
        Ok(())
    }
}

// A cdylib has no entry point; this only satisfies the native compile of the example
// target (its wasm module above is cfg'd out there).
#[cfg(not(target_family = "wasm"))]
#[allow(dead_code)]
fn main() {}
