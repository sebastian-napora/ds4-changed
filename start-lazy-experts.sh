#!/bin/bash
#
# Experimental CUDA startup mode for DeepSeek V4 Flash.
# Keeps dense/router/shared tensors in the normal startup cache, but skips
# routed MoE expert tensors so they are reached through the existing lazy
# model-range path instead of being uploaded before serving starts.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export DS4_CUDA_LAZY_ROUTED_EXPERTS="${DS4_CUDA_LAZY_ROUTED_EXPERTS:-1}"
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-32}"
export DS4_CUDA_WEIGHT_CACHE_VERBOSE="${DS4_CUDA_WEIGHT_CACHE_VERBOSE:-1}"
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-4096}"

echo "=========================================="
echo "  ds4-server lazy routed experts experiment"
echo "=========================================="
echo "  DS4_CUDA_LAZY_ROUTED_EXPERTS=$DS4_CUDA_LAZY_ROUTED_EXPERTS"
echo "  DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=$DS4_CUDA_WEIGHT_CACHE_LIMIT_GB"
echo "  DS4_CUDA_Q8_F16_CACHE_MB=$DS4_CUDA_Q8_F16_CACHE_MB"
echo "  DS4_CUDA_WEIGHT_CACHE_VERBOSE=$DS4_CUDA_WEIGHT_CACHE_VERBOSE"
echo "=========================================="
echo ""

exec ./start-server.sh "$@"
