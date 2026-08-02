#!/usr/bin/env bash
#SBATCH --job-name=embed-grid
#SBATCH --partition=gpu            # EDIT: from 00_discovery output
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7                # one task per embedder (see MODELS below)
#SBATCH --output=logs/%x-%A_%a.out

# Embedder-sensitivity grid: job array, one embedding model per task, each
# embedding the eval subset in all representations and scoring all definition
# queries. Driven by hansard_llm/embedder_grid.py (local sentence-transformers
# backend — no server needed).
#
#   sbatch cluster/embed_grid.sbatch

set -euo pipefail
source ~/.config/hansard_llm.env
source "$VENV_DIR/bin/activate"
mkdir -p logs

MODELS=(
  Qwen/Qwen3-Embedding-0.6B
  Qwen/Qwen3-Embedding-4B
  Qwen/Qwen3-Embedding-8B
  BAAI/bge-base-en-v1.5
  BAAI/bge-large-en-v1.5
  Alibaba-NLP/gte-large-en-v1.5
  intfloat/e5-large-v2
  nomic-ai/nomic-embed-text-v1.5
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "embedder grid: $MODEL"
python -m hansard_llm.embedder_grid --model "$MODEL" --backend st
