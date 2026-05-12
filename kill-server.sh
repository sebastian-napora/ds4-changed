#!/bin/bash
#
# Kill ds4-server and all related processes
#

echo "🛑 Stopping ds4 services..."

# Kill by PID files
[ -f /tmp/ds4-server.pid ] && kill $(cat /tmp/ds4-server.pid) 2>/dev/null && echo "   ✅ ds4-server killed" || true
[ -f /tmp/ds4-litellm.pid ] && kill $(cat /tmp/ds4-litellm.pid) 2>/dev/null && echo "   ✅ LiteLLM killed" || true
[ -f /tmp/ds4-gpu-monitor.pid ] && kill $(cat /tmp/ds4-gpu-monitor.pid) 2>/dev/null && echo "   ✅ GPU monitor killed" || true
[ -f /tmp/ds4-memory-monitor.pid ] && kill $(cat /tmp/ds4-memory-monitor.pid) 2>/dev/null && echo "   ✅ Memory monitor killed" || true

# Also kill any remaining processes
pkill -f "ds4-server" 2>/dev/null && echo "   ✅ ds4-server (pkill) killed" || true
pkill -f "litellm" 2>/dev/null && echo "   ✅ LiteLLM (pkill) killed" || true
pkill -f "ds4-memory-monitor" 2>/dev/null && echo "   ✅ Memory monitor (pkill) killed" || true
pkill -f "ds4-gpu-monitor" 2>/dev/null && echo "   ✅ GPU monitor (pkill) killed" || true

# Clean up PID files
rm -f /tmp/ds4-server.pid /tmp/ds4-litellm.pid /tmp/ds4-gpu-monitor.pid /tmp/ds4-memory-monitor.pid

echo "✅ All services stopped"