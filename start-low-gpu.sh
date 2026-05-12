#!/bin/bash
#
# ds4-server startup script - LOW GPU mode (~15% utilization)
# DeepSeek V4 Flash inference server with memory protection
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
DS4_HOST="${DS4_HOST:-0.0.0.0}"
DS4_PORT="${DS4_PORT:-11112}"
LITE_LLM_PORT="${LITE_LLM_PORT:-11111}"
DS4_CTX="${DS4_CTX:-100000}"
KV_DISK_DIR="${DS4_KV_DISK_DIR:-/tmp/ds4-kv}"
KV_DISK_SPACE="${DS4_KV_DISK_SPACE:-8192}"
MODEL="${DS4_MODEL:-ds4flash.gguf}"

# GPU throttling settings (~15% utilization)
GPU_POWER_LIMIT="${GPU_POWER_LIMIT:-15}"  # 15% of max power

# Memory limit settings (shutdown if exceeded)
MAX_RAM_GB="${MAX_RAM_GB:-124}"          # Max RAM before shutdown (GB)
MAX_RAM_PERCENT="${MAX_RAM_PERCENT:-95}"   # Or max % of total RAM

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
VENV_LITELLM="$SCRIPT_DIR/venv/bin/litellm"

# Kill existing processes
kill_existing() {
    echo "🧹 Cleaning up existing processes..."
    pkill -f "ds4-server" 2>/dev/null || true
    pkill -f "litellm" 2>/dev/null || true
    sleep 1
}

# Setup GPU throttling for ~15% utilization
setup_gpu_throttle() {
    if command -v nvidia-smi &>/dev/null; then
        echo "🎮 Setting GPU throttling to ~15%..."
        
        # Get current GPU info
        GPU_NAME=$(nvidia-smi --query-gpu="name" --format=csv,noheader)
        MAX_POWER=$(nvidia-smi --query-gpu="power.limit" --format=csv,noheader,nounits | awk '{print $1}')
        CURRENT_POWER=$(nvidia-smi --query-gpu="power.draw" --format=csv,noheader,nounits | awk '{print $1}')
        
        # Calculate new power limit
        NEW_POWER=$(echo "scale=1; $MAX_POWER * $GPU_POWER_LIMIT / 100" | bc)
        
        echo "   GPU: $GPU_NAME"
        echo "   Max Power: ${MAX_POWER}W"
        echo "   Current: ${CURRENT_POWER}W"
        
        # Try to set power limit
        if sudo nvidia-smi -pl "$NEW_POWER" 2>/dev/null; then
            echo "   ✅ Power limit set to ${NEW_POWER}W"
        elif nvidia-smi -pl "$NEW_POWER" 2>/dev/null; then
            echo "   ✅ Power limit set to ${NEW_POWER}W"
        else
            echo "   ⚠️  Could not set power limit (needs sudo)"
        fi
        
        # Try to enable persistence mode
        if sudo nvidia-smi -pm 1 2>/dev/null || nvidia-smi -pm 1 2>/dev/null; then
            echo "   ✅ Persistence mode enabled"
        fi
        
        # Show current utilization
        sleep 2
        echo ""
        echo "📊 Current GPU status:"
        nvidia-smi --query-gpu="utilization.gpu,power.draw,power.limit,temperature.gpu" \
            --format=csv,noheader,nounits | \
            awk -F', ' '{printf "   GPU: %s | Power: %sW | Temp: %s°C\n", $1, $2, $4}'
    else
        echo "⚠️  nvidia-smi not found"
    fi
}

# Memory monitoring functions
get_used_ram_gb() {
    free -b | awk 'NR==2 {print $3/1024/1024/1024}'
}

get_used_ram_percent() {
    free | awk 'NR==2 {printf "%.0f", $3/$2*100}'
}

get_total_ram_gb() {
    free -b | awk 'NR==2 {print $2/1024/1024/1024}'
}

# Build ds4-server command
build_ds4_cmd() {
    DS4_CMD="./ds4-server"
    DS4_CMD="$DS4_CMD --host $DS4_HOST"
    DS4_CMD="$DS4_CMD --port $DS4_PORT"
    DS4_CMD="$DS4_CMD --ctx $DS4_CTX"
    DS4_CMD="$DS4_CMD --kv-disk-dir $KV_DISK_DIR"
    DS4_CMD="$DS4_CMD --kv-disk-space-mb $KV_DISK_SPACE"
    DS4_CMD="$DS4_CMD -m $MODEL"
    echo "$DS4_CMD"
}

# Start ds4 backend
start_ds4() {
    CMD=$(build_ds4_cmd)
    echo "🚀 Starting ds4-server (port $DS4_PORT)..."
    echo "   Command: $CMD"
    $CMD &
    DS4_PID=$!
    echo "   PID: $DS4_PID"
    echo $DS4_PID > /tmp/ds4-server.pid
}

# Start LiteLLM proxy
start_litellm() {
    # Check if venv exists
    if [ ! -f "$VENV_LITELLM" ]; then
        echo "⚠️  LiteLLM venv not found. Run: ./setup-venv.sh"
        echo "   Starting without LiteLLM..."
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
        
        # Print status periodically
        echo -ne "   RAM: ${USED_GB}GB / ${TOTAL_GB}GB (${USED_PCT}%)    \r"
        sleep 30
    done
}

# Monitor GPU usage
monitor_gpu() {
    echo ""
    echo "📊 GPU monitoring (Ctrl+C to stop):"
    while true; do
        nvidia-smi --query-gpu="utilization.gpu,power.draw,temperature.gpu" \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F', ' '{printf "\r   GPU: %-4s | Power: %sW | Temp: %s°C  ", $1, $2, $3}'
        sleep 5
    done
}

# Print startup info
print_info() {
    TOTAL_GB=$(get_total_ram_gb)
    echo ""
    echo "=========================================="
    echo "  ds4-server - DeepSeek V4 Flash"
    echo "  ⚡ LOW GPU MODE (~15%)"
    echo "=========================================="
    echo "  Model:       $MODEL"
    echo "  Context:     $DS4_CTX tokens"
    echo "  KV Cache:    $KV_DISK_DIR"
    echo "  Power Limit: ${GPU_POWER_LIMIT}%"
    echo "  Max RAM:     ${MAX_RAM_GB}GB / ${MAX_RAM_PERCENT}%"
    echo "=========================================="
    echo ""
    echo "Services:"
    echo "  ds4 backend:  http://$DS4_HOST:$DS4_PORT/v1/chat/completions"
    echo "  LiteLLM:     http://$DS4_HOST:$LITE_LLM_PORT/v1/chat/completions"
    echo ""
    echo "API Key: dsv4-local"
    echo ""
}

# Main
kill_existing
setup_gpu_throttle
print_info

# Start ds4 backend
start_ds4
echo "Waiting 10s for ds4 to initialize..."
sleep 10

# Start LiteLLM (optional)
start_litellm 2>/dev/null || true

# Start monitoring in background
echo ""
echo "📊 Starting GPU and memory monitoring..."

(monitor_gpu) &
echo $! > /tmp/ds4-gpu-monitor.pid

(monitor_memory) &
echo $! > /tmp/ds4-memory-monitor.pid

# Wait for user interrupt
wait
