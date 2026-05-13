#!/bin/bash
#
# Experimental CUDA startup mode for DeepSeek V4 Flash.
# Keeps dense/router/shared tensors in the normal startup cache, but skips
# routed MoE expert tensors at startup. During inference, only a small number
# of selected routed experts are copied into the lazy expert cache; the rest
# stay on the mapped model path instead of uploading the whole 256-expert tensor.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export DS4_CUDA_LAZY_ROUTED_EXPERTS="${DS4_CUDA_LAZY_ROUTED_EXPERTS:-1}"
export DS4_CUDA_LAZY_MAX_RESIDENT_EXPERTS="${DS4_CUDA_LAZY_MAX_RESIDENT_EXPERTS:-6}"
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-32}"
export DS4_CUDA_WEIGHT_CACHE_VERBOSE="${DS4_CUDA_WEIGHT_CACHE_VERBOSE:-1}"
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-4096}"
export DS4_SERVER_REQUEST_PROGRESS="${DS4_SERVER_REQUEST_PROGRESS:-1}"

echo "=========================================="
echo "  ds4-server lazy routed experts experiment"
echo "=========================================="
echo "  DS4_CUDA_LAZY_ROUTED_EXPERTS=$DS4_CUDA_LAZY_ROUTED_EXPERTS"
echo "  DS4_CUDA_LAZY_MAX_RESIDENT_EXPERTS=$DS4_CUDA_LAZY_MAX_RESIDENT_EXPERTS"
echo "  DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=$DS4_CUDA_WEIGHT_CACHE_LIMIT_GB"
echo "  DS4_CUDA_Q8_F16_CACHE_MB=$DS4_CUDA_Q8_F16_CACHE_MB"
echo "  DS4_CUDA_WEIGHT_CACHE_VERBOSE=$DS4_CUDA_WEIGHT_CACHE_VERBOSE"
echo "  DS4_SERVER_REQUEST_PROGRESS=$DS4_SERVER_REQUEST_PROGRESS"
echo "=========================================="
echo ""

if [ "${DS4_LAZY_SKIP_BUILD:-0}" != "1" ]; then
    if [ ! -x ./ds4-server ] || [ ds4.c -nt ./ds4-server ] || [ ds4_cuda.cu -nt ./ds4-server ] || [ ds4_server.c -nt ./ds4-server ]; then
        echo "🔧 Building ds4-server for lazy routed experts..."
        make ds4-server
        echo ""
    fi
fi

if [ "$#" -eq 0 ]; then
    set -- --ds4-only
fi

exec ./start-server.sh "$@"
