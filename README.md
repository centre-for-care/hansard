# hansard

Scraping and analysing UK parliamentary debates (Hansard), 1803–present.

The repo has two eras of work:

1. **Data acquisition** (`scrapers/`) — scrape debate transcripts and assemble
   speaker metadata (Wikipedia/Wikidata with GPT-assisted verification).
2. **LLM analysis pilot** (`hansard_llm/`) — a robustness-aware LLM
   topic-extraction pipeline over the scraped corpus, plus the cluster
   infrastructure to run it at scale.

Data and run artifacts live **outside this repo** in a sibling checkout
(`../hansard_eda` by default; override with `HANSARD_LLM_DATA_DIR` /
`HANSARD_LLM_ARTIFACTS_DIR`). Path resolution is centralised in
[hansard_llm/config.py](hansard_llm/config.py).

## Repo map

| Directory | What it is | Details |
|---|---|---|
| [hansard_llm/](hansard_llm/) | The main Python package: stratified sampling, prompt-grid LLM extraction, embedding retrieval, robustness metrics, report builders (`docs/`). | [hansard_llm/README.md](hansard_llm/README.md) |
| [cluster/](cluster/) | SLURM scripts for Oxford BMRC (`cluster/`) and ARC HTC (`cluster/arc/`). | [cluster/README.md](cluster/README.md), [cluster/arc/README.md](cluster/arc/README.md) |
| [scrapers/](scrapers/) | Standalone CLI scripts that built the corpus and speaker metadata. Numbered by execution order; run rarely (re-scrapes only). | below |
| [scripts/](scripts/) | One-off verification/maintenance scripts (migration checks, cache verification). | docstrings in each |
| [tests/](tests/) | Pytest suite guarding past bugs: output parsing, results-store cache keys, era binning. Run `python -m pytest tests/ -q`. | |

Install: `pip install -e .` (deps pinned in [pyproject.toml](pyproject.toml);
duckdb is hard-pinned because sample reproducibility depends on its version).

## Scrapers

Filenames are numbered (`NN_`) by execution order within each pipeline. All
are standalone `argparse` CLIs.

### Transcripts — [scrapers/hansard/](scrapers/hansard/)

- `scraper_historical_hansard_01_calendar_index.py` — index sittings (1803–2005) → chambers/sections.
- `scraper_historical_hansard_02_enrich_sections.py` — enrich each section with full text, year-by-year with checkpoints.
- `scraper_historical_hansard_03_collect_speaker_metadata.py` — crawl Hansard speaker index pages.
- `scraper_modern_hansard_01_scrape_debates.py` — modern Hansard: calendar → section URLs → hierarchical debate tree.

### Speaker metadata — [scrapers/historical/](scrapers/historical/) (1803–2005) and [scrapers/modern/](scrapers/modern/) (2005–present)

Parallel pipelines with the same shape: collect candidate Wikipedia pages,
verify identity with GPT, extract a structured biography JSON, then fall back
to Wikidata and finally the open web (fallback-sourced entries are flagged).

| Step | Historical | Modern |
|---|---|---|
| Seed / gender | `historical_00_infer_speaker_gender.py` | `modern_00_fetch_members_api.py` (Parliament Members API) |
| Collect Wikipedia candidates | `historical_01_wiki_collect_and_download.py` | `modern_01_collect_wikipedia_links.py` + `modern_02_download_wiki_texts.py` |
| GPT identity check | `historical_02_verify_wikipedia_pages.py` | `modern_03_verify_pages.py` |
| GPT bio extraction | `historical_03_extract_bio_json.py` | `modern_04_extract_bio_json.py` |
| Wikipedia/Wikidata fallback | `historical_04_fallback_search_fetch.py` | `modern_05_fallback_search_fetch.py` |
| Open-web fallback | `historical_05_openweb_fallback.py` | `modern_06_openweb_fallback.py` |
| No-URL edge cases | `historical_99_no_url_pipeline.py` | — |
