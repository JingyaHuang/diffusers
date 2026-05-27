#!/usr/bin/env bash
# Launch vllm-omni to serve FLUX.2-klein-4B on a single Neuron device using
# the diffusers backend.  Run this script on a Trainium or Inferentia instance.
#
# Usage:
#   bash run_server.sh [--model <model_id>] [--port <port>] [--cache-dir <path>]
#
# Defaults:
#   model     : black-forest-labs/FLUX.2-klein-4B
#   port      : 8091
#   cache-dir : /tmp/neff_cache   (lost on reboot; pass a persistent path to
#                                  avoid recompilation across runs)
#
# Environment variables (set before running):
#   DIFFUSERS_REPO   : path to local diffusers source (default: ~/diffusers)
#   HF_TOKEN         : Hugging Face access token for gated models

set -euo pipefail

MODEL="${MODEL:-black-forest-labs/FLUX.2-klein-4B}"
PORT="${PORT:-8091}"
CACHE_DIR="${CACHE_DIR:-/tmp/neff_cache}"
DIFFUSERS_REPO="${DIFFUSERS_REPO:-$HOME/diffusers}"
PYTHON="${PYTHON:-$HOME/.venv/bin/python}"

# Neuron compiler env — required for NKI version compatibility
export NEURONX_DISABLE_NKI_FLEX_ATTENTION="${NEURONX_DISABLE_NKI_FLEX_ATTENTION:-1}"
export TORCH_NEURONX_ENABLE_NKI_SDPA="${TORCH_NEURONX_ENABLE_NKI_SDPA:-0}"
export TORCH_NEURONX_NEFF_CACHE_DIR="${CACHE_DIR}"

# Use local diffusers source so Neuron-specific fixes are in effect
export PYTHONPATH="${DIFFUSERS_REPO}/src:${PYTHONPATH:-}"

# Neuron runtime library path (needed when NRT is installed under /tmp)
if [ -d /tmp/neuron_lib/opt/aws/neuron/lib ]; then
    export LD_LIBRARY_PATH="/tmp/neuron_lib/opt/aws/neuron/lib:${LD_LIBRARY_PATH:-}"
fi

echo "=== vllm-omni Neuron serving ==="
echo "  model     : ${MODEL}"
echo "  port      : ${PORT}"
echo "  cache-dir : ${CACHE_DIR}"
echo "  diffusers : ${DIFFUSERS_REPO}"
echo ""
echo "  Note: first run compiles XLA graphs (~10-15 min). Subsequent runs"
echo "        load NEFFs from cache-dir and start in <2 min."
echo ""

exec "${PYTHON}" -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
    --omni \
    --diffusion-load-format diffusers \
    --dtype bfloat16 \
    --port "${PORT}" \
    --init-timeout 3600 \
    "$@"
