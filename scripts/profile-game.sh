#!/usr/bin/env bash
# profile-game.sh — capture a CPU/GPU/FPS/frame-cost profile of an rl binary.
#
# Answers the recurring "why is the game slow?" question with one command instead
# of rediscovering the toolchain each time (the 2026-06-28 GCR slideshow diag).
# Read signals: GPU idle + slow ⇒ CPU-bound or software render; one thread pegged
# ⇒ serial bottleneck; whole proc capped at ~1 core while loadavg ≫ cores ⇒
# preemption starvation; backend != Vulkan/NVIDIA ⇒ software fallback.
#
# Two ways to point it at a process:
#   profile-game.sh --pid <PID>            attach to a running process (read-only;
#                                          safe against a live owner session)
#   profile-game.sh [-- <BIN> <ARGS...>]   launch a target (default: the deployed
#                                          /home/a/rl-game game), profile, then kill
#
# Sampling (nvidia-smi, top -H, /proc) is read-only — attach mode never signals the
# target. Launched targets are pinned to cores 14-23 (botq builds are confined to
# 0-13) so profiling isn't self-contended; for the real Vulkan client run the whole
# script as user `a` (it owns the Wayland session and the GPU process).
set -uo pipefail

SECS=15
PID=""
PERF=0
CORES="14-23"
OUT=""
LOG=""
GAME_DIR=/home/a/rl-game
declare -a LAUNCH_CMD=()

usage() { awk 'NR>1{ if(!/^#/)exit; sub(/^# ?/,""); print }' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --pid)   PID="${2:?--pid needs a value}"; shift 2 ;;
    --secs)  SECS="${2:?--secs needs a value}"; shift 2 ;;
    --perf)  PERF=1; shift ;;
    --cores) CORES="${2:?--cores needs a value}"; shift 2 ;;
    --out)   OUT="${2:?--out needs a value}"; shift 2 ;;
    --log)   LOG="${2:?--log needs a value}"; shift 2 ;;
    -h|--help) usage 0 ;;
    --) shift; LAUNCH_CMD=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }
TS=/run/current-system/sw/bin/taskset
have "$TS" || TS="$(command -v taskset 2>/dev/null || true)"

[ -n "$OUT" ] || OUT="$(mktemp -d /tmp/profile-game.XXXXXX)"
mkdir -p "$OUT"
REPORT="$OUT/report.md"

# --- resolve the target PID ----------------------------------------------------
LAUNCHED=""
if [ -z "$PID" ]; then
  # launch mode: default to the deployed game's launcher unless a binary was given
  if [ ${#LAUNCH_CMD[@]} -eq 0 ]; then
    if [ -x "$GAME_DIR/run-game.sh" ]; then
      LAUNCH_CMD=("$GAME_DIR/run-game.sh")
    else
      echo "no --pid and no launch target, and $GAME_DIR/run-game.sh not runnable here." >&2
      echo "give a PID (--pid N) or a binary (-- /path/to/bin args...)." >&2
      exit 1
    fi
  fi
  echo "launching: ${LAUNCH_CMD[*]}  (pinned to cores $CORES)" >&2
  # setsid → own process group, so cleanup kills the game even if the launcher
  # (run-game.sh / safe-exec trampoline) forks rather than exec's into it.
  if [ -n "$TS" ] && [ -x "$TS" ]; then
    setsid "$TS" -c "$CORES" "${LAUNCH_CMD[@]}" >"$OUT/target.log" 2>&1 &
  else
    setsid "${LAUNCH_CMD[@]}" >"$OUT/target.log" 2>&1 &
  fi
  PID=$!
  LAUNCHED=$PID
  # the launcher writes the renderer's stdout/stderr (incl. the AdapterInfo line)
  # here — read our own capture, not a stale deployed launch.log.
  [ -n "$LOG" ] || LOG="$OUT/target.log"
  sleep 3  # let the window/renderer come up before sampling
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "target PID $PID is not alive" >&2
  [ -n "$LAUNCHED" ] && tail -20 "$OUT/target.log" >&2
  exit 1
fi

CMD="$(tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null | cut -c1-200)"
PUSER="$(ps -o user= -p "$PID" 2>/dev/null | tr -d ' ')"

# Only ever kill a process WE launched (its whole group) — attach mode never signals.
cleanup() {
  [ -n "$LAUNCHED" ] || return
  kill -- "-$LAUNCHED" 2>/dev/null; sleep 1; kill -9 -- "-$LAUNCHED" 2>/dev/null
}
trap cleanup EXIT

# --- GPU time series (nvidia-smi) ---------------------------------------------
gpu_series() {
  have nvidia-smi || { echo "(nvidia-smi absent)"; return; }
  echo "t  gpu%  mem%  sm_mhz  watt"
  local t
  for t in $(seq 1 "$SECS"); do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | awk -F, -v t="$t" '{print t" "$1" "$2" "$3" "$4}'
    sleep 1
  done
}
gpu_proc() {
  have nvidia-smi || return
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
}

# --- per-thread CPU (top -H, two snapshots SECS apart; 2nd reflects the window) -
cpu_threads() {
  if have top; then
    top -H -b -n 2 -d "$SECS" -p "$PID" 2>/dev/null \
      | awk 'BEGIN{s=0} /^ *PID +USER/{s++} s==2' | head -25
  else
    echo "(top absent)"
  fi
}
# involuntary context switches corroborate preemption starvation
ctxt_switches() {
  awk '/nonvoluntary_ctxt_switches|voluntary_ctxt_switches/{print}' "/proc/$PID/status" 2>/dev/null
}
loadavg() { cat /proc/loadavg 2>/dev/null; nproc --all 2>/dev/null | sed 's/^/cores: /'; }

# --- render backend + FPS from the target log ---------------------------------
backend() {
  [ -n "$LOG" ] && [ -r "$LOG" ] || { echo "(no readable log: ${LOG:-none})"; return; }
  grep -aoE 'AdapterInfo \{[^}]*\}' "$LOG" | tail -1 \
    || echo "(no AdapterInfo line in $LOG — backend unknown)"
}
fps() {
  if [ -n "$LOG" ] && [ -r "$LOG" ] && grep -aiqE 'fps|frame_time|frametime' "$LOG"; then
    grep -aiE 'fps|frame_time|frametime' "$LOG" | tail -5
  else
    echo "(target does not log FPS — add bevy FrameTimeDiagnosticsPlugin +"
    echo " LogDiagnosticsPlugin to self-report; meanwhile read GPU% + per-thread CPU"
    echo " below: GPU idle while slow ⇒ not GPU-bound.)"
  fi
}

# --- optional perf frame-breakdown (flag-gated, degrades gracefully) ----------
perf_breakdown() {
  [ "$PERF" = 1 ] || { echo "(skipped — pass --perf for a frame breakdown)"; return; }
  have perf || { echo "(perf not installed — frame breakdown skipped)"; return; }
  local data="$OUT/perf.data"
  if ! perf record -g -o "$data" -p "$PID" -- sleep "$SECS" 2>"$OUT/perf.err"; then
    echo "(perf record failed — likely needs privilege; see perf.err)"; sed 's/^/  /' "$OUT/perf.err"; return
  fi
  echo "top self-cost functions (perf report):"
  perf report -i "$data" --stdio 2>/dev/null | grep -aE '^\s+[0-9]' | head -20
}

echo "profiling PID $PID ($PUSER) for ${SECS}s — report → $REPORT" >&2

# GPU time-series and per-thread CPU sampled CONCURRENTLY → one overlapping window.
gpu_series >"$OUT/gpu.txt" 2>/dev/null &
GJOB=$!
cpu_threads >"$OUT/cpu.txt" 2>/dev/null
wait "$GJOB" 2>/dev/null
DIED=""
if [ -n "$LAUNCHED" ] && ! kill -0 "$PID" 2>/dev/null; then DIED=1; fi

{
  echo "# profile — PID $PID"
  echo
  echo "- process: \`$CMD\`"
  echo "- user: $PUSER   window: ${SECS}s   ${LAUNCHED:+launched, pinned to cores $CORES}"
  echo "- log: ${LOG:-none}"
  [ -n "$DIED" ] && { echo; echo "> ⚠️ launched target EXITED during sampling — samples truncated; tail:"; echo '```'; tail -15 "$OUT/target.log" 2>/dev/null; echo '```'; }
  echo
  echo "## render backend (Vulkan/NVIDIA = GPU; llvmpipe/lavapipe = software)"
  echo '```'; backend; echo '```'
  echo "## FPS"
  echo '```'; fps; echo '```'
  echo "## GPU utilization over time (idle while slow ⇒ NOT GPU-bound)"
  echo '```'; cat "$OUT/gpu.txt"; echo '```'
  echo "### GPU compute clients"
  echo '```'; gpu_proc; echo '```'
  echo "## host load (loadavg ≫ cores ⇒ oversubscription starves frame-locked threads)"
  echo '```'; loadavg; echo '```'
  echo "## per-thread CPU (top -H; one thread ~100% ⇒ serial; proc ~1 core total ⇒ starved)"
  echo '```'; cat "$OUT/cpu.txt"; echo '```'
  echo "### context switches (high nonvoluntary ⇒ preempted off-core)"
  echo '```'; ctxt_switches; echo '```'
  echo "## frame breakdown (perf; adds a separate ${SECS}s window)"
  echo '```'; perf_breakdown; echo '```'
} >"$REPORT"

cat "$REPORT"
echo >&2
echo "report saved: $REPORT" >&2
