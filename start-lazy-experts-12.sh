#!/bin/bash
#
# Convenience launcher for lazy routed experts with 12 resident experts.
# Run this instead of remembering environment variables by hand.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export DS4_CUDA_LAZY_ROUTED_EXPERTS=1
export DS4_CUDA_LAZY_MAX_RESIDENT_EXPERTS=12
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-32}"
export DS4_CUDA_WEIGHT_CACHE_VERBOSE="${DS4_CUDA_WEIGHT_CACHE_VERBOSE:-1}"
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-4096}"
export DS4_SERVER_REQUEST_PROGRESS="${DS4_SERVER_REQUEST_PROGRESS:-1}"

exec ./start-lazy-experts.sh "$@"
