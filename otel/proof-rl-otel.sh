#!/usr/bin/env bash
# Full app→iroh→sink proof for the rl OTEL instrumentation: the rl `otel` crate's Rust
# SDK emits real OTLP (trace + log + metric) which traverses the iroh tunnel and lands in
# the stock collector's sink. Proves (a) the 0.28 thread-based exporters work with NO
# tokio runtime, and (b) rl-emitted OTLP is valid end to end.
#
#   otel SDK (this example) → :14318 iroh-tunnel forward → iroh → serve → :4318 otelcol → JSONL
set -euo pipefail

OTELCOL="${OTELCOL:?set OTELCOL to the otelcol-contrib store path}"
TUNNEL="${TUNNEL:-$HOME/.local/bin/iroh-tunnel}"
WT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HOME/.local/state/telemetry/rl-proof"
rm -rf "$WORK"; mkdir -p "$WORK/sink"; cd "$WORK"

cat > otelcol.yaml <<'YAML'
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318
exporters:
  file:
    path: ./sink/otlp-*.jsonl
    group_by: { enabled: true, resource_attribute: host.name }
processors: { batch: {} }
service:
  pipelines:
    traces:  { receivers: [otlp], processors: [batch], exporters: [file] }
    metrics: { receivers: [otlp], processors: [batch], exporters: [file] }
    logs:    { receivers: [otlp], processors: [batch], exporters: [file] }
  telemetry: { metrics: { level: none } }
YAML

pids=(); cleanup(){ for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null||true; done; }; trap cleanup EXIT

run-untrusted "$OTELCOL/bin/otelcol-contrib" --config ./otelcol.yaml >otelcol.log 2>&1 & pids+=($!)
for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/4318)2>/dev/null && { exec 3>&-; break; }; sleep 0.2; done 2>/dev/null || true
sleep 1

ROOT="$("$TUNNEL" forward --root-key "$WORK/root.key" --server x --print-root)"
"$TUNNEL" serve --key-file "$WORK/server.key" --allow "$ROOT" --target 127.0.0.1:4318 >serve.log 2>&1 & pids+=($!)
for i in $(seq 1 50); do grep -q "endpoint id:" serve.log && break; sleep 0.2; done
SRV_ID="$(grep 'endpoint id:' serve.log|awk '{print $NF}')"; SRV_ADDR="$(grep 'direct addr:' serve.log|awk '{print $NF}'|head -1)"
ADDR=(); [ -n "${SRV_ADDR:-}" ] && ADDR=(--server-addr "$SRV_ADDR")
"$TUNNEL" forward --root-key "$WORK/root.key" --server "$SRV_ID" "${ADDR[@]}" --listen 127.0.0.1:14318 >forward.log 2>&1 & pids+=($!)
for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/14318)2>/dev/null && { exec 3>&-; break; }; sleep 0.2; done 2>/dev/null || true

echo "== run the rl otel SDK smoke example through the tunnel =="
cd "$WT"
DECK_ID=ablaised OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:14318 \
  taskset -c 0-13 nix-shell --run "cargo run -q -p otel --example smoke" 2>&1 | tail -5
cd "$WORK"

SINK="$WORK/sink/otlp-ablaised.jsonl"
for i in $(seq 1 40); do [ -s "$SINK" ] && grep -q hello-otel-from-rust-LOG "$SINK" && grep -q rl_otel_smoke_counter "$SINK" && grep -q smoke_span "$SINK" && break; sleep 0.3; done
echo "sink: $SINK"; ls -l "$WORK/sink/" 2>/dev/null
ok=0
for n in smoke_span hello-otel-from-rust-LOG rl_otel_smoke_counter; do
  if grep -q "$n" "$SINK" 2>/dev/null; then echo "  FOUND  $n"; ok=$((ok+1)); else echo "  MISSING $n"; fi
done
[ -f "$SINK" ] && echo "  partition tag confirmed: file named for DECK_ID host.name=ablaised"
[ "$ok" -eq 3 ] && echo "RESULT: PASS — rl Rust SDK emitted all 3 OTLP signals through iroh, tagged by deck." || { echo "RESULT: FAIL ($ok/3)"; tail -20 otelcol.log; exit 1; }
