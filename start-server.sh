#!/bin/bash
#
# ds4-server startup script
# DeepSeek V4 Flash inference server with optional LiteLLM proxy
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
DS4_HOST="${DS4_HOST:-0.0.0.0}"
DS4_PORT="${DS4_PORT:-11112}"          # ds4 backend port
LITE_LLM_PORT="${LITE_LLM_PORT:-11111}" # LiteLLM proxy port
DS4_CTX="${DS4_CTX:-100000}"
KV_DISK_DIR="${DS4_KV_DISK_DIR:-/tmp/ds4-kv}"
KV_DISK_SPACE="${DS4_KV_DISK_SPACE:-8192}"
MODEL="${DS4_MODEL:-ds4flash.gguf}"

# Memory limit settings (shutdown if exceeded)
MAX_RAM_GB="${MAX_RAM_GB:-124}"        # Max RAM before shutdown (GB)
MAX_RAM_PERCENT="${MAX_RAM_PERCENT:-95}" # Or max % of total RAM

# GPU throttling settings (for 5-10% utilization)
GPU_POWER_LIMIT="${GPU_POWER_LIMIT:-50}"     # Power limit in watts (percentage of max)
GPU_COMPUTE_MODE="${GPU_COMPUTUTE_MODE:-default}"
GPU_PERSISTENT_MODE="${GPU_PERSISTENT_MODE:-1}"

# Optional: enable trace logging
TRACE_FILE="${DS4_TRACE:-}"

# Build ds4-server command
DS4_CMD="./ds4-server"
DS4_CMD="$DS4_CMD --host $DS4_HOST"
DS4_CMD="$DS4_CMD --port $DS4_PORT"
DS4_CMD="$DS4_CMD --ctx $DS4_CTX"
DS4_CMD="$DS4_CMD --kv-disk-dir $KV_DISK_DIR"
DS4_CMD="$DS4_CMD --kv-disk-space-mb $KV_DISK_SPACE"
DS4_CMD="$DS4_CMD -m $MODEL"

# Optional MTP (speculative decoding)
if [ -n "$DS4_MTP" ]; then
    DS4_CMD="$DS4_CMD --mtp $DS4_MTP"
    DS4_CMD="$DS4_CMD --mtp-draft ${DS4_MTP_DRAFT:-2}"
fi

# Optional trace
if [ -n "$TRACE_FILE" ]; then
    DS4_CMD="$DS4_CMD --trace $TRACE_FILE"
fi

# Kill existing processes
kill_existing() {
    echo "🧹 Cleaning up existing processes..."
    pkill -f "ds4-server" 2>/dev/null || true
    pkill -f "litellm" 2>/dev/null || true
    sleep 1
}

# Setup GPU throttling
setup_gpu_throttle() {
    if command -v nvidia-smi &>/dev/null; then
        echo "🎮 Setting GPU throttling..."
        
        # Get max power
        MAX_POWER=$(nvidia-smi --query-gpu="power.limit" --format=csv,noheader,nounits | awk '{print $1}')
        NEW_POWER=$(echo "$MAX_POWER * $GPU_POWER_LIMIT / 100" | bc)
        
        # Set power limit
        sudo nvidia-smi -pl "$NEW_POWER" 2>/dev/null || \
            nvidia-smi -pl "$NEW_POWER" 2>/dev/null || \
            echo "⚠️  Could not set power limit (may need sudo)"
        
        echo "   Power limit: ${NEW_POWER}W (was ${MAX_POWER}W)"
    else
        echo "⚠️  nvidia-smi not found, skipping GPU throttling"
    fi
}

# Memory monitoring function
get_used_ram_gb() {
    free -b | awk 'NR==2 {print $3/1024/1024/1024}'
}

get_used_ram_percent() {
    free | awk 'NR==2 {printf "%.0f", $3/$2*100}'
}

get_total_ram_gb() {
    free -b | awk 'NR==2 {print $2/1024/1024/1024}'
}

# Start ds4 backend
start_ds4() {
    echo "🚀 Starting ds4-server (port $DS4_PORT)..."
    echo "   Command: $DS4_CMD"
    $DS4_CMD &
    DS4_PID=$!
    echo "   PID: $DS4_PID"
    echo $DS4_PID > /tmp/ds4-server.pid
}

# Start LiteLLM proxy
start_litellm() {
    VENV_LITELLM="$SCRIPT_DIR/venv/bin/litellm"
    
    if [ ! -f "$VENV_LITELLM" ]; then
        echo "⚠️  LiteLLM venv not found. Run: ./setup-venv.sh"
        return 1
    fi
    
    echo "🚀 Starting LiteLLM proxy (port $LITE_LLM_PORT)..."
    $VENV_LITELLM --config $SCRIPT_DIR/lite_llm_config.yaml --port $LITE_LLM_PORT &
    LITE_LLM_PID=$!
    echo "   PID: $LITE_LLM_PID"
    echo $LITE_LLM_PID > /tmp/ds4-litellm.pid
}

# Monitor memory and shutdown if exceeded
monitor_memory() {
    echo ""
    echo "📊 Memory monitoring (limit: ${MAX_RAM_GB}GB / ${MAX_RAM_PERCENT}%)..."
    echo ""
    
    while true; do
        sleep 10
        
        USED_GB=$(get_used_ram_gb)
        USED_PCT=$(get_used_ram_percent)
        TOTAL_GB=$(get_total_ram_gb)
        
        # Check if limit exceeded
        EXCEEDED=0
        
        if (( $(echo "$USED_GB > $MAX_RAM_GB" | bc -l) )); then
            echo ""
            echo "🚨 MEMORY LIMIT EXCEEDED: ${USED_GB}GB > ${MAX_RAM_GB}GB"
            EXCEEDED=1
        fi
        
        if [ "$USED_PCT" -ge "$MAX_RAM_PERCENT" ]; then
            echo ""
            echo "🚨 MEMORY PERCENT EXCEEDED: ${USED_PCT}% >= ${MAX_RAM_PERCENT}%"
            EXCEEDED=1
        fi
        
        if [ "$EXCEEDED" -eq 1 ]; then
            echo ""
            echo "⚠️  Shutting down to prevent OOM..."
            echo "   Used: ${USED_GB}GB / ${TOTAL_GB}GB (${USED_PCT}%)"
            echo ""
            
            # Kill processes
            [ -f /tmp/ds4-server.pid ] && kill $(cat /tmp/ds4-server.pid) 2>/dev/null
            [ -f /tmp/ds4-litellm.pid ] && kill $(cat /tmp/ds4-litellm.pid) 2>/dev/null
            pkill -f "ds4-server" 2>/dev/null
            pkill -f "litellm" 2>/dev/null
            
            echo "✅ Graceful shutdown complete"
            exit 1
        fi
        
        # Print status every 30 seconds
        if [ $(( $(date +%s) % 30 )) -eq 0 ]; then
            echo -ne "   RAM: ${USED_GB}GB / ${TOTAL_GB}GB (${USED_PCT}%)    \r"
        fi
    done
}

# Print startup info
print_info() {
    TOTAL_GB=$(get_total_ram_gb)
    echo ""
    echo "=========================================="
    echo "  ds4-server - DeepSeek V4 Flash"
    echo "=========================================="
    echo "  Model:       $MODEL"
    echo "  Context:     $DS4_CTX tokens"
    echo "  KV Cache:    $KV_DISK_DIR ($KV_DISK_SPACE MB)"
    echo "  Max RAM:     ${MAX_RAM_GB}GB / ${MAX_RAM_PERCENT}% (of ${TOTAL_GB}GB)"
    echo "=========================================="
    echo ""
    echo "Services:"
    echo "  ds4 backend:  http://$DS4_HOST:$DS4_PORT/v1/chat/completions"
    echo "  LiteLLM:      http://$DS4_HOST:$LITE_LLM_PORT/v1/chat/completions"
    echo ""
    echo "API Keys:"
    echo "  ds4 backend:  dsv4-local"
    echo "  LiteLLM:      dsv4-local"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
}

# Usage
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --help              Show this help
  --ds4-only          Start only ds4 backend (no LiteLLM)
  --gpu-throttle      Apply GPU throttling (50% power)

Environment variables:
  DS4_HOST           Bind address (default: 0.0.0.0)
  DS4_PORT           ds4 backend port (default: 11112)
  LITE_LLM_PORT      LiteLLM proxy port (default: 11111)
  DS4_CTX            Context size (default: 100000)
  DS4_KV_DISK_DIR    KV cache directory (default: /tmp/ds4-kv)
  DS4_KV_DISK_SPACE  KV cache size in MB (default: 8192)
  MAX_RAM_GB         Max RAM GB before shutdown (default: 124)
  MAX_RAM_PERCENT    Max RAM % before shutdown (default: 95)
  GPU_POWER_LIMIT    GPU power limit in % (default: 50)

Examples:
  $0                         # Start both ds4 + LiteLLM
  $0 --ds4-only              # Start only ds4 backend
  $0 --gpu-throttle          # Start with GPU throttling
  MAX_RAM_GB=120 $0          # Custom RAM limit

EOF
}

# Parse arguments
MODE="both"
GPU_THROTTLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        --ds4-only)
            MODE="ds4"
            ;;
        --gpu-throttle)
            GPU_THROTTLE=true
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# Main
kill_existing

if [ "$GPU_THROTTLE" = true ]; then
    setup_gpu_throttle
fi

print_info

case $MODE in
    both)
        start_ds4
        echo "Waiting 10s for ds4 to initialize..."
        sleep 10
        start_litellm
        ;;
    ds4)
        start_ds4
        ;;
esac

# Start memory monitoring in background
monitor_memory &
MONITOR_PID=$!
echo $MONITOR_PID > /tmp/ds4-memory-monitor.pid

# Wait for user interrupt or memory limit
wait
