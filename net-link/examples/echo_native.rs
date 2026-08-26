//! Native half of the browser↔native relay echo probe (rl#411 stage 4) — see
//! `examples/web-echo/run.sh` for the whole flow. Binds with the n0 dev relays ON
//! (unlike the game's native link, which is relay-off — the probe exists to exercise
//! the relay path a browser peer needs), prints our EndpointId + home relay, then
//! echoes every datagram back to its sender on the game's ALPN.

#[cfg(not(target_family = "wasm"))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use iroh::{Endpoint, RelayMode, endpoint::presets};

    let ep = Endpoint::builder(presets::Minimal)
        .relay_mode(RelayMode::Default)
        .alpns(vec![net_proto::codec::ALPN.to_vec()])
        .bind()
        .await?;
    ep.online().await;
    let relay = ep
        .addr()
        .addrs
        .iter()
        .find_map(|a| match a {
            iroh::TransportAddr::Relay(u) => Some(u.to_string()),
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("no home relay after online()"))?;
    println!("PROBE_ID={}", ep.id());
    println!("PROBE_RELAY={relay}");

    while let Some(incoming) = ep.accept().await {
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    println!("PROBE_ACCEPT_ERR={e}");
                    return;
                }
            };
            println!("PROBE_CONN_FROM={}", conn.remote_id());
            loop {
                match conn.read_datagram().await {
                    Ok(d) => {
                        println!("PROBE_ECHO_BYTES={}", d.len());
                        if let Err(e) = conn.send_datagram(d) {
                            println!("PROBE_SEND_ERR={e}");
                            break;
                        }
                    }
                    Err(e) => {
                        println!("PROBE_CONN_END={e}");
                        break;
                    }
                }
            }
        });
    }
    Ok(())
}

#[cfg(target_family = "wasm")]
fn main() {}
