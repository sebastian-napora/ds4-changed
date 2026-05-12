#!/bin/bash
#
# Setup Python venv for ds4 with LiteLLM
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

echo "🐍 Setting up Python venv..."
python3 -m venv "$VENV_DIR"

echo "📦 Installing packages..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install litellm

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate venv:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run LiteLLM:"
echo "  $VENV_DIR/bin/litellm --config $SCRIPT_DIR/lite_llm_config.yaml --port 11111"
