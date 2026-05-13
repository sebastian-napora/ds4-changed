#!/bin/bash
#
# Compatibility launcher for the old "40 experts" preset.
# It now uses the 128 GB-friendly high-memory lazy expert cache.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export DS4_CUDA_LAZY_ROUTED_EXPERTS=1
export DS4_CUDA_LAZY_MAX_RESIDENT_EXPERTS=8192
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-32}"
export DS4_CUDA_WEIGHT_CACHE_VERBOSE="${DS4_CUDA_WEIGHT_CACHE_VERBOSE:-0}"
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-4096}"
export DS4_SERVER_REQUEST_PROGRESS="${DS4_SERVER_REQUEST_PROGRESS:-1}"

exec ./start-lazy-experts.sh "$@"
