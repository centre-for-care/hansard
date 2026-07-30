"""hansard_llm — LLM-based targeted topic extraction over the Hansard corpus.

Pilot goal: a reproducible, robustness-aware pipeline that, for a *fixed*
topic (default: health & social care), asks an LLM per speech:

    1. does this speech substantively discuss the topic?   (presence)
    2. which controlled sub-themes does it address?         (details)
    3. a supporting verbatim quote                          (for human spot-check)

The pilot measures *consistency* (not yet validity) of these answers across
nuisance factors we want invariance to:

    role   x  task-wording  x  output-format  x  model  x  repetition

so we can attribute instability to each factor (variance decomposition) and
judge how robust the downstream estimand (topic prevalence over time) is.

Modules
-------
    config   - env, model registry, topic definition, paths
    prompts  - component-based prompt assembly + deterministic variant IDs
    schema   - output schema + JSON / free-text parsers
    client   - Nebius (OpenAI-compatible) client with provenance logging
    sample   - stratified pilot sampling over the Parquet via DuckDB
    run      - idempotent grid runner + result store
    metrics  - consistency, variance decomposition, estimand robustness
"""

from __future__ import annotations

__all__ = ["config", "prompts", "schema"]
