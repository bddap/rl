#!/usr/bin/env bash
set -euo pipefail

OTELCOL="${OTELCOL:?set OTELCOL to the otelcol-contrib store path}"
TUNNEL="${TUNNEL:-$HOME/.local/bin/iroh-tunnel}"
WT="$(cd "$(dirname "$0")/.." && pwd)"
# Off the standard 4318 so a collector already serving it is never fed proof data.
PORT="${OTLP_PORT:-24318}"
# Keys and tunnel logs stay out of the collector's sandbox, which could read or forge them.
WORK="$(mktemp -d /tmp/rl-proof.XXXXXX)"
SBX="$(mktemp -d /tmp/untrusted.rl-proof.XXXXXX)"
mkdir -p "$SBX/sink"; cd "$SBX"

cat > otelcol.yaml <<YAML
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:$PORT
exporters:
  file:
    path: ./sink/otlp-*.jsonl
    group_by: { enabled: true, resource_attribute: host.name }
processors: { batch: {} }
service:
  pipelines:
    metrics: { receivers: [otlp], processors: [batch], exporters: [file] }
    logs:    { receivers: [otlp], processors: [batch], exporters: [file] }
  telemetry: { metrics: { level: none } }
YAML

pids=(); cleanup(){ for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null||true; done; rm -rf "$WORK" "$SBX"; }; trap cleanup EXIT

run-untrusted -p "$PORT" "$OTELCOL/bin/otelcol-contrib" --config ./otelcol.yaml >"$WORK/otelcol.log" 2>&1 & pids+=($!)
cd "$WORK"
# The forwarded port accepts before otelcol listens, so wait for an HTTP answer.
for i in $(seq 1 50); do [ "$(curl -s --max-time 1 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/logs")" != 000 ] && break; sleep 0.2; done

ROOT="$("$TUNNEL" forward --root-key "$WORK/root.key" --server x --print-root)"
"$TUNNEL" serve --key-file "$WORK/server.key" --allow "$ROOT" --target "127.0.0.1:$PORT" >serve.log 2>&1 & pids+=($!)
for i in $(seq 1 50); do grep -q "endpoint id:" serve.log && break; sleep 0.2; done
SRV_ID="$(grep 'endpoint id:' serve.log|awk '{print $NF}')"; SRV_ADDR="$(grep 'direct addr:' serve.log|awk '{print $NF}'|head -1)"
ADDR=(); [ -n "${SRV_ADDR:-}" ] && ADDR=(--server-addr "$SRV_ADDR")
"$TUNNEL" forward --root-key "$WORK/root.key" --server "$SRV_ID" "${ADDR[@]}" --listen 127.0.0.1:14318 >forward.log 2>&1 & pids+=($!)
for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/14318)2>/dev/null && { exec 3>&-; break; }; sleep 0.2; done 2>/dev/null || true

echo "== run the rl otel SDK smoke example through the tunnel =="
cd "$WT"
DECK_ID=testdeck OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:14318 \
  taskset -c 0-13 nix-shell --run "cargo run -q -p otel --example smoke" 2>&1 | tail -5
cd "$WORK"

SINK="$SBX/sink/otlp-testdeck.jsonl"
for i in $(seq 1 40); do [ -s "$SINK" ] && grep -q hello-otel-from-rust-LOG "$SINK" && grep -q rl_otel_smoke_counter "$SINK" && break; sleep 0.3; done
echo "sink: $SINK"; ls -l "$SBX/sink/" 2>/dev/null
ok=0
for n in hello-otel-from-rust-LOG rl_otel_smoke_counter; do
  if grep -q "$n" "$SINK" 2>/dev/null; then echo "  FOUND  $n"; ok=$((ok+1)); else echo "  MISSING $n"; fi
done
[ -f "$SINK" ] && echo "  partition tag confirmed: file named for DECK_ID host.name=testdeck"
[ "$ok" -eq 2 ] && echo "RESULT: PASS — rl Rust SDK emitted its logs and metrics through iroh, tagged by deck." || { echo "RESULT: FAIL ($ok/2)"; tail -20 otelcol.log; exit 1; }
