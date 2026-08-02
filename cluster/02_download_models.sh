#!/usr/bin/env bash
# Download model weights into the HF cache on scratch. Run on whichever node
# has internet (discovery step tells you); if only login nodes do, run it there
# (downloads are IO-bound, no GPU needed). Idempotent — hf download resumes.
#
#   source ~/.config/hansard_llm.env && source "$VENV_DIR/bin/activate"
#   bash cluster/02_download_models.sh [embedders|llms|all]
#
# Sizes (fp16/bf16 unless noted): embedders ~20GB total; LLMs ~340GB total.
# Check scratch quota first. Gated models (Llama, Gemma) need
# `hf auth login` once with a HF token that has accepted their licenses.

set -euo pipefail
WHAT="${1:-all}"

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

# Panel + extended model axis (see plan C2). Nemotron-Super-49B replaces
# Llama-3.3-70B as the large-model panelist; FP8 fits one 80GB GPU.
LLMS=(
  Qwen/Qwen3-30B-A3B-Instruct-2507
  google/gemma-3-27b-it
  nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
  Qwen/Qwen3-32B
  Qwen/Qwen3-14B
  Qwen/Qwen3-4B-Instruct-2507
  mistralai/Mistral-Small-3.2-24B-Instruct-2506
)

dl() { for m in "$@"; do echo "== $m"; hf download "$m" >/dev/null && echo "   ok"; done; }

case "$WHAT" in
  embedders) dl "${EMBEDDERS[@]}" ;;
  llms)      dl "${LLMS[@]}" ;;
  all)       dl "${EMBEDDERS[@]}"; dl "${LLMS[@]}" ;;
  *) echo "usage: $0 [embedders|llms|all]"; exit 1 ;;
esac

echo; echo "cache usage:"; du -sh "${HF_HOME:-$HOME/.cache/huggingface}"
