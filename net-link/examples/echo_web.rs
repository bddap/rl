//! Browser half of the browser↔native relay echo probe (rl#411 stage 4) — see
//! `examples/web-echo/run.sh` for the whole flow. Binds a wasm iroh endpoint
//! (relay-only in browsers), dials the native peer by explicit EndpointAddr (id +
//! relay URL — the join-code shape), then round-trips datagrams on the game's ALPN
//! and reports per-round RTT plus a final verdict on the console.

#[cfg(target_family = "wasm")]
mod wasm {
    use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, endpoint::presets};
    use wasm_bindgen::prelude::*;

    fn log(s: &str) {
        web_sys::console::log_1(&JsValue::from_str(s));
    }

    #[wasm_bindgen]
    pub async fn run_probe(peer_id: String, relay_url: String, rounds: u32) -> Result<(), JsValue> {
        console_error_panic_hook::set_once();
        match probe(peer_id, relay_url, rounds).await {
            Ok(()) => Ok(()),
            Err(e) => {
                log(&format!("PROBE_FAIL={e:#}"));
                Err(JsValue::from_str(&format!("{e:#}")))
            }
        }
    }

    async fn probe(peer_id: String, relay_url: String, rounds: u32) -> anyhow::Result<()> {
        let id: EndpointId = peer_id.parse()?;
        let relay = relay_url.parse().map_err(anyhow::Error::msg)?;
        let addr = EndpointAddr::new(id).with_relay_url(relay);

        let t0 = js_sys::Date::now();
        let ep = Endpoint::builder(presets::Minimal)
            .relay_mode(RelayMode::Default)
            .bind()
            .await?;
        log(&format!(
            "PROBE_BOUND_MS={:.0} id={}",
            js_sys::Date::now() - t0,
            ep.id()
        ));

        let t0 = js_sys::Date::now();
        let conn = ep.connect(addr, net_proto::codec::ALPN).await?;
        log(&format!("PROBE_CONNECT_MS={:.0}", js_sys::Date::now() - t0));

        let mut rtts = Vec::new();
        for i in 0..rounds {
            let payload = format!("echo-{i}-{}", "x".repeat(200));
            let t0 = js_sys::Date::now();
            conn.send_datagram(payload.clone().into_bytes().into())?;
            let d = conn.read_datagram().await?;
            let rtt = js_sys::Date::now() - t0;
            anyhow::ensure!(d.as_ref() == payload.as_bytes(), "echo mismatch on round {i}");
            log(&format!("PROBE_RTT_MS={rtt:.1}"));
            rtts.push(rtt);
        }
        rtts.sort_by(f64::total_cmp);
        log(&format!(
            "PROBE_OK rounds={} median_rtt_ms={:.1} min={:.1} max={:.1}",
            rtts.len(),
            rtts[rtts.len() / 2],
            rtts[0],
            rtts[rtts.len() - 1],
        ));
        Ok(())
    }
}

// A cdylib has no entry point; this only satisfies the native compile of the example
// target (its wasm module above is cfg'd out there).
#[cfg(not(target_family = "wasm"))]
#[allow(dead_code)]
fn main() {}
