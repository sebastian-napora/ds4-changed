#!/usr/bin/env bash
#
# One-command setup and launcher for DS4 MTP speculative decoding.
#
# Defaults:
#   - Ensures ds4flash.gguf exists, downloading q2-imatrix if needed.
#   - Ensures the MTP support GGUF exists.
#   - Builds ds4 / ds4-server if needed.
#   - Starts ds4-server with --mtp and --mtp-draft 2.
#
# Examples:
#   ./setup-mtp.sh
#   ./setup-mtp.sh --cli --prompt "Explain Redis streams."
#   ./setup-mtp.sh --setup-only
#   DS4_MAIN_MODEL=q4-imatrix ./setup-mtp.sh --server -- --port 11112

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
GGUF_DIR="${DS4_GGUF_DIR:-"$SCRIPT_DIR/gguf"}"
case "$GGUF_DIR" in
    /*) ;;
    *) GGUF_DIR="$SCRIPT_DIR/$GGUF_DIR" ;;
esac

MODE="server"
PROMPT=""
SETUP_ONLY=0
DO_DOWNLOAD=1
DO_BUILD=1
MAIN_MODEL_TARGET="${DS4_MAIN_MODEL:-q2-imatrix}"
MTP_PATH="${DS4_MTP:-"$GGUF_DIR/$MTP_FILE"}"
MTP_DRAFT="${DS4_MTP_DRAFT:-2}"
MTP_MARGIN="${DS4_MTP_MARGIN:-3}"
DS4_CTX="${DS4_CTX:-100000}"
DS4_HOST="${DS4_HOST:-0.0.0.0}"
DS4_PORT="${DS4_PORT:-11112}"
KV_DISK_DIR="${DS4_KV_DISK_DIR:-/tmp/ds4-kv}"
KV_DISK_SPACE="${DS4_KV_DISK_SPACE:-8192}"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [-- EXTRA_ARGS...]

Options:
  --server             Start ds4-server after setup (default).
  --cli                Start the ds4 CLI after setup.
  --prompt TEXT        CLI prompt to run with -p. Implies --cli.
  --setup-only         Download/build only; do not launch anything.
  --no-download        Do not download missing GGUF files.
  --no-build           Do not run make.
  --mtp PATH           MTP GGUF path. Default: ./gguf/$MTP_FILE
  --mtp-draft N        Speculative draft tokens. Default: 2
  --mtp-margin F       MTP confidence margin. Default: 3
  -h, --help           Show this help.

Environment:
  DS4_MAIN_MODEL       Main model download target if ds4flash.gguf is missing.
                       Default: q2-imatrix. Other values: q4-imatrix, q2, q4.
  DS4_GGUF_DIR         GGUF download directory. Default: ./gguf
  DS4_MTP             MTP GGUF path override.
  DS4_MTP_DRAFT       Same as --mtp-draft.
  DS4_MTP_MARGIN      Same as --mtp-margin.
  DS4_CTX             Server context. Default: 100000
  DS4_HOST            Server host. Default: 0.0.0.0
  DS4_PORT            Server port. Default: 11112
  DS4_KV_DISK_DIR     Server KV disk directory. Default: /tmp/ds4-kv
  DS4_KV_DISK_SPACE   Server KV disk budget in MB. Default: 8192

Notes:
  MTP speculative decoding is used only for greedy generation. For API clients,
  send temperature: 0. Thinking mode also disables the greedy path internally.

Examples:
  $0
  $0 --cli --prompt "Explain Redis streams."
  $0 --setup-only
  $0 --server -- --trace /tmp/ds4-trace.log
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --server)
            MODE="server"
            ;;
        --cli)
            MODE="cli"
            ;;
        --prompt)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --prompt" >&2
                exit 1
            fi
            PROMPT="$1"
            MODE="cli"
            ;;
        --setup-only)
            SETUP_ONLY=1
            ;;
        --no-download)
            DO_DOWNLOAD=0
            ;;
        --no-build)
            DO_BUILD=0
            ;;
        --mtp)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --mtp" >&2
                exit 1
            fi
            MTP_PATH="$1"
            ;;
        --mtp-draft)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --mtp-draft" >&2
                exit 1
            fi
            MTP_DRAFT="$1"
            ;;
        --mtp-margin)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --mtp-margin" >&2
                exit 1
            fi
            MTP_MARGIN="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

require_file_or_download() {
    local path="$1"
    local target="$2"
    local description="$3"

    if [ -s "$path" ]; then
        echo "Found $description: $path"
        return
    fi

    if [ "$DO_DOWNLOAD" -ne 1 ]; then
        echo "Missing $description: $path" >&2
        echo "Re-run without --no-download or download it manually." >&2
        exit 1
    fi

    echo "Downloading $description with ./download_model.sh $target"
    ./download_model.sh "$target"
}

if [ ! -x ./download_model.sh ]; then
    echo "Missing executable ./download_model.sh" >&2
    exit 1
fi

if [ ! -e ./ds4flash.gguf ]; then
    require_file_or_download "./ds4flash.gguf" "$MAIN_MODEL_TARGET" "main DS4 model link"
else
    echo "Found main DS4 model link: ./ds4flash.gguf"
fi

require_file_or_download "$MTP_PATH" "mtp" "MTP support model"

if [ "$DO_BUILD" -eq 1 ]; then
    if [ ! -x ./ds4 ] || [ ! -x ./ds4-server ]; then
        echo "Building ds4 binaries"
        make
    else
        echo "Found ds4 binaries"
    fi
else
    echo "Skipping build"
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
    echo
    echo "Setup complete."
    echo "MTP path: $MTP_PATH"
    exit 0
fi

case "$MODE" in
    cli)
        CMD=(./ds4 --nothink --mtp "$MTP_PATH" --mtp-draft "$MTP_DRAFT" --mtp-margin "$MTP_MARGIN")
        if [ -n "$PROMPT" ]; then
            CMD+=(-p "$PROMPT")
        fi
        CMD+=("${EXTRA_ARGS[@]}")
        echo "Starting CLI with MTP speculative decoding"
        echo "Command: ${CMD[*]}"
        exec "${CMD[@]}"
        ;;
    server)
        CMD=(./ds4-server
            --host "$DS4_HOST"
            --port "$DS4_PORT"
            --ctx "$DS4_CTX"
            --kv-disk-dir "$KV_DISK_DIR"
            --kv-disk-space-mb "$KV_DISK_SPACE"
            --mtp "$MTP_PATH"
            --mtp-draft "$MTP_DRAFT"
            --mtp-margin "$MTP_MARGIN")
        CMD+=("${EXTRA_ARGS[@]}")
        echo "Starting server with MTP speculative decoding"
        echo "Server: http://$DS4_HOST:$DS4_PORT/v1/chat/completions"
        echo "Use temperature: 0 in requests to hit the speculative path."
        echo "Command: ${CMD[*]}"
        exec "${CMD[@]}"
        ;;
    *)
        echo "Invalid mode: $MODE" >&2
        exit 1
        ;;
esac
