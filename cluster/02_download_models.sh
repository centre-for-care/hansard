#!/usr/bin/env bash
# Download model weights into $HF_HOME on /well. Login node only (compute nodes
# have no internet); use tmux for the LLM batch. Idempotent/resumable.
# Gated models (Gemma, Nemotron) need `hf auth login` once.
#
# Prerequisites (cluster/README.md §§3–5):
#   source ~/.config/hansard_llm.env
#   source "$VENV_DIR/bin/activate"
#   bash cluster/02_download_models.sh [embedders|llms|all|<hf-model-id>...]

set -euo pipefail
WHAT="${1:-all}"

: "${HF_HOME:?source ~/.config/hansard_llm.env first (sets HF_HOME)}"
command -v hf >/dev/null || {
  echo "hf CLI not found — source \"\$VENV_DIR/bin/activate\" first (huggingface_hub)" >&2
  exit 1
}

# Env file forces offline mode for compute jobs; downloads must reach the hub.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

EMBEDDERS=(
  Qwen/Qwen3-Embedding-0.6B
  Qwen/Qwen3-Embedding-4B
  Qwen/Qwen3-Embedding-8B
  BAAI/bge-base-en-v1.5
  BAAI/bge-large-en-v1.5
  Alibaba-NLP/gte-large-en-v1.5
  intfloat/e5-large-v2
  nomic-ai/nomic-embed-text-v1.5
)

# Panel + extended axis. Nemotron-Super-49B: FP8 on one 80GB GPU.
LLMS=(
  Qwen/Qwen3-30B-A3B-Instruct-2507
  google/gemma-3-27b-it
  nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
  Qwen/Qwen3-32B
  Qwen/Qwen3-14B
  Qwen/Qwen3-4B-Instruct-2507
  mistralai/Mistral-Small-3.2-24B-Instruct-2506
)

dl() { for m in "$@"; do echo "== $m"; hf download "$m" && echo "   ok"; done; }

case "$WHAT" in
  embedders) dl "${EMBEDDERS[@]}" ;;
  llms)      dl "${LLMS[@]}" ;;
  all)       dl "${EMBEDDERS[@]}"; dl "${LLMS[@]}" ;;
  -h|--help) echo "usage: $0 [embedders|llms|all|<hf-model-id>...]"; exit 0 ;;
  *)         dl "$@" ;;
esac

echo; echo "cache usage:"; du -sh "$HF_HOME"
