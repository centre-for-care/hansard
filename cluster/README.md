# Cluster setup (SLURM + vLLM)

The pipeline's LLM client is OpenAI-compatible, so a cluster-hosted
`vllm serve` is a drop-in backend: point `LLM_BASE_URL` at it and every row
records `backend=vllm-<cluster>` in provenance. Nebius stays as fallback
(leave its credentials in the `.env` on scratch).

Order of operations:

| Step | Script | Where | Notes |
|---|---|---|---|
| 1 | `00_discovery.sh` | login node | read-only; paste output back before proceeding |
| 2 | `01_setup_env.sh` | login node | needs `HANSARD_SCRATCH=<big filesystem>`; makes uv env + `~/.config/hansard_llm.env` |
| 3 | rsync data up | laptop | `full_data_enriched.parquet`, `artifacts/llm/pilot_sample.parquet`, `artifacts/llm/legacy/` (see comments in `01_setup_env.sh`) |
| 4 | `02_download_models.sh` | wherever internet is | `embedders` first (~20GB), `llms` later (~340GB); gated repos need `hf auth login` |
| 5 | `embed_grid.sbatch` | sbatch | job array, one embedder per task |
| 6 | `run_grid.sbatch` | sbatch | serve model + run experiment in one job |
| 7 | `serve_llm.sbatch` | sbatch | standalone server for interactive use |

Smoke test once the env exists (steps 1–4 done, before committing to the
full grid):

```bash
# 50-speech pilot slice through a local server, cheapest model:
sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--n-speeches 5" \
       cluster/run_grid.sbatch
```

Every `#SBATCH --partition/--gres` line is a placeholder until discovery
reports the real partition names and GPU types — edit them once, they are the
only cluster-specific bits.
