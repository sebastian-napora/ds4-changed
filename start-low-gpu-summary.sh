#!/usr/bin/env bash
#
# Start start-low-gpu.sh and keep a live, timestamped runtime summary.
#
# Each run writes to:
#   summary-logs/low-gpu-YYYYmmdd-HHMMSS/
#     run.log        Full stdout/stderr from start-low-gpu.sh
#     summary.md     Latest snapshot, overwritten every interval
#     timeline.log   Append-only snapshots for later review
#     env.log        Startup environment/configuration
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${DS4_SUMMARY_LOG_ROOT:-"$SCRIPT_DIR/summary-logs"}"
SESSION_NAME="${DS4_SUMMARY_SESSION_NAME:-low-gpu-$(date +%Y%m%d-%H%M%S)}"
SESSION_DIR="$LOG_ROOT/$SESSION_NAME"
INTERVAL="${DS4_SUMMARY_INTERVAL:-30}"

RUN_LOG="$SESSION_DIR/run.log"
SUMMARY_FILE="$SESSION_DIR/summary.md"
TIMELINE_LOG="$SESSION_DIR/timeline.log"
ENV_LOG="$SESSION_DIR/env.log"

START_PID=""
SUMMARY_PID=""

usage() {
    cat <<EOF
Usage: $0 [--help]

Starts ./start-low-gpu.sh and writes live runtime summaries while the model is
running. The latest summary is kept in summary.md; historical snapshots are
appended to timeline.log.

Environment:
  DS4_SUMMARY_LOG_ROOT      Log root. Default: ./summary-logs
  DS4_SUMMARY_SESSION_NAME  Session directory name. Default: low-gpu-timestamp
  DS4_SUMMARY_INTERVAL      Snapshot interval in seconds. Default: 30

All start-low-gpu.sh environment variables are passed through, for example:
  DS4_MODEL, DS4_PORT, LITE_LLM_PORT, DS4_CTX, MAX_RAM_GB, GPU_POWER_LIMIT

Examples:
  $0
  DS4_SUMMARY_INTERVAL=10 $0
  DS4_MODEL=/models/ds4.gguf DS4_SUMMARY_SESSION_NAME=low-gpu-test $0
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

if [ "$#" -gt 0 ]; then
    echo "Unknown option: $1" >&2
    usage >&2
    exit 1
fi

mkdir -p "$SESSION_DIR"

write_env_log() {
    {
        echo "session=$SESSION_NAME"
        echo "started_at=$(date -Iseconds)"
        echo "script_dir=$SCRIPT_DIR"
        echo "summary_interval=$INTERVAL"
        echo "ds4_model=${DS4_MODEL:-ds4flash.gguf}"
        echo "ds4_host=${DS4_HOST:-0.0.0.0}"
        echo "ds4_port=${DS4_PORT:-11112}"
        echo "litellm_port=${LITE_LLM_PORT:-11111}"
        echo "ds4_ctx=${DS4_CTX:-100000}"
        echo "kv_disk_dir=${DS4_KV_DISK_DIR:-/tmp/ds4-kv}"
        echo "kv_disk_space_mb=${DS4_KV_DISK_SPACE:-8192}"
        echo "max_ram_gb=${MAX_RAM_GB:-124}"
        echo "max_ram_percent=${MAX_RAM_PERCENT:-95}"
        echo "gpu_power_limit_percent=${GPU_POWER_LIMIT:-15}"
    } > "$ENV_LOG"
}

read_pid_file() {
    local path="$1"
    if [ -s "$path" ]; then
        tr -d '[:space:]' < "$path"
    fi
}

pid_status() {
    local label="$1"
    local pid="$2"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        printf "%s: running pid=%s\n" "$label" "$pid"
    elif [ -n "$pid" ]; then
        printf "%s: stopped pid=%s\n" "$label" "$pid"
    else
        printf "%s: no pid file\n" "$label"
    fi
}

memory_snapshot() {
    if command -v free >/dev/null 2>&1; then
        free -h
    elif command -v vm_stat >/dev/null 2>&1 && command -v sysctl >/dev/null 2>&1; then
        local page_size
        page_size="$(vm_stat | awk '/page size of/ {print $8}' | tr -d '.')"
        local total
        total="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
        echo "total_bytes=$total"
        echo "page_size=$page_size"
        vm_stat
    else
        echo "memory tools unavailable"
    fi
}

gpu_snapshot() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,utilization.gpu,power.draw,power.limit,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>&1 || true
    else
        echo "nvidia-smi unavailable"
    fi
}

endpoint_snapshot() {
    local host="${DS4_HOST:-127.0.0.1}"
    local ds4_port="${DS4_PORT:-11112}"
    local litellm_port="${LITE_LLM_PORT:-11111}"

    if [ "$host" = "0.0.0.0" ]; then
        host="127.0.0.1"
    fi

    if command -v curl >/dev/null 2>&1; then
        printf "ds4 /v1/models: "
        curl -fsS -m 2 "http://$host:$ds4_port/v1/models" >/dev/null 2>&1 && echo "ok" || echo "unavailable"
        printf "litellm /v1/models: "
        curl -fsS -m 2 "http://$host:$litellm_port/v1/models" >/dev/null 2>&1 && echo "ok" || echo "unavailable"
    else
        echo "curl unavailable"
    fi
}

write_summary() {
    local state="${1:-running}"
    local now
    local ds4_pid
    local litellm_pid
    local memory_pid
    local gpu_pid

    now="$(date -Iseconds)"
    ds4_pid="$(read_pid_file /tmp/ds4-server.pid || true)"
    litellm_pid="$(read_pid_file /tmp/ds4-litellm.pid || true)"
    memory_pid="$(read_pid_file /tmp/ds4-memory-monitor.pid || true)"
    gpu_pid="$(read_pid_file /tmp/ds4-gpu-monitor.pid || true)"

    {
        echo "# DS4 Low GPU Summary"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- State: $state"
        echo "- Updated: $now"
        echo "- Log directory: $SESSION_DIR"
        echo "- Run log: $RUN_LOG"
        echo
        echo "## Processes"
        echo
        pid_status "wrapper" "${START_PID:-}"
        pid_status "ds4-server" "$ds4_pid"
        pid_status "litellm" "$litellm_pid"
        pid_status "memory-monitor" "$memory_pid"
        pid_status "gpu-monitor" "$gpu_pid"
        echo
        echo "## Endpoints"
        echo
        endpoint_snapshot
        echo
        echo "## GPU"
        echo
        gpu_snapshot
        echo
        echo "## Memory"
        echo
        memory_snapshot
        echo
        echo "## Recent Run Log"
        echo
        if [ -s "$RUN_LOG" ]; then
            tail -n 40 "$RUN_LOG"
        else
            echo "run.log is empty so far"
        fi
    } > "$SUMMARY_FILE.tmp"

    mv "$SUMMARY_FILE.tmp" "$SUMMARY_FILE"

    {
        echo "===== $now state=$state ====="
        sed -n '1,120p' "$SUMMARY_FILE"
        echo
    } >> "$TIMELINE_LOG"
}

summary_loop() {
    while true; do
        write_summary "running"
        sleep "$INTERVAL"
    done
}

cleanup() {
    local status="${1:-stopping}"

    if [ -n "${SUMMARY_PID:-}" ]; then
        kill "$SUMMARY_PID" 2>/dev/null || true
        wait "$SUMMARY_PID" 2>/dev/null || true
    fi

    write_summary "$status" || true

    for pid_file in /tmp/ds4-server.pid /tmp/ds4-litellm.pid /tmp/ds4-memory-monitor.pid /tmp/ds4-gpu-monitor.pid; do
        if [ -s "$pid_file" ]; then
            local pid
            pid="$(read_pid_file "$pid_file" || true)"
            if [ -n "$pid" ]; then
                kill "$pid" 2>/dev/null || true
            fi
        fi
    done

    if [ -n "${START_PID:-}" ]; then
        kill "$START_PID" 2>/dev/null || true
        wait "$START_PID" 2>/dev/null || true
    fi
}

trap 'cleanup interrupted; exit 130' INT
trap 'cleanup terminated; exit 143' TERM

write_env_log

echo "Summary session: $SESSION_DIR"
echo "Starting ./start-low-gpu.sh ..."

./start-low-gpu.sh > >(tee -a "$RUN_LOG") 2>&1 &
START_PID=$!

summary_loop &
SUMMARY_PID=$!

set +e
wait "$START_PID"
START_STATUS=$?
set -e

cleanup "exited:$START_STATUS"
echo "Summary written to: $SUMMARY_FILE"
exit "$START_STATUS"
