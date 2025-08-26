# hansard
A repo for scraping and analysing Hansard data!

This project provides end-to-end, reproducible pipelines to:

- **Scrape Hansard debate transcripts** — Historical (1803–2005) and Modern (2005–present)
- **Assemble speaker metadata** — Wikipedia/Wikidata with GPT-assisted verification and open-web fallbacks
- Export tidy **CSV/JSON** tables for analysis

---

## Table of contents
- [hansard](#hansard)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Project structure](#project-structure)
    - [1) Hansard transcripts (web scrapers)](#1-hansard-transcripts-web-scrapers)
    - [2) Speaker metadata (Wikipedia/Wikidata + GPT)](#2-speaker-metadata-wikipediawikidata--gpt)

---

## Overview

The repo has two complementary domains:

1) **Hansard transcripts (scrapers)**  
   Crawl the Hansard calendar, collect section hierarchies, and extract debate text.

2) **Speaker metadata (Wikipedia/Wikidata + GPT)**  
   Build queries, fetch candidate pages, verify identity with GPT, extract a structured biography JSON, then fall back to Wikidata or open web when needed.

Both domains are implemented for **Historical (1803–2005)** and **Modern (2005–present)** periods.

---

## Project structure

> Filenames are numbered (`NN_`) to reflect execution order within each pipeline.

### 1) Hansard transcripts (web scrapers)

**Historical (HTML scraping)**
- `historical_hansard_01_calendar_index.py`  
  Index sittings (1803–2005) → chambers/sections → `historical_step1.json`.
- `historical_hansard_02_enrich_sections.py`  
  Enrich each section with full text (speeches & procedural), year-by-year with checkpoints.

**Modern (HTML scraping + API seed)**
- `modern_hansard_00_fetch_members_api.py`  
  Retrieve modern members metadata from the UK Parliament Members API (seed list).
- `modern_hansard_01_scrape_debates.py`  
  Scrape modern Hansard: calendar → section URLs → hierarchical debate tree.

**Historical speaker index (Hansard “People”)**
- `historical_speakers_00_collect_index.py`  
  Crawl Hansard speaker index pages → `speaker_details.json`.

### 2) Speaker metadata (Wikipedia/Wikidata + GPT)

**Historical (1803–2005)**
- `historical_01_hansard_wikipedia_collector_full.py` — Build Google queries per speaker; collect candidate Wikipedia URLs; download article text.
- `historical_02_verify_wikipedia_pages.py` — GPT identity check (`yes`/`no`/`uncertain`).
- `historical_03_extract_bio_json.py` — GPT extraction of structured biography JSON from verified pages.
- `historical_04_fallback_search_fetch.py` — If missing, GPT search **restricted to Wikipedia + Wikidata**, record URLs and fetch text.
- `historical_05_openweb_fallback.py` — Final open-web fallback (Britannica, gov.uk, reputable press/archives); extracted entries are **flagged** as open-web sourced.
- `historical_99_no_url_pipeline.py` — Utilities and edge-case handling for historical speakers **without** Wikipedia URLs.
- `infer_speaker_gender.py` — Gender inference for historical speakers (titles → heuristics → `gender_guesser` → manual overrides).

**Modern (2005–present)**
- `modern_01_hansard_wikipedia_collector.py` — Build queries; collect **top-2** Wikipedia URLs per member.
- `modern_02_download_wiki_texts.py` — Download article content for candidates.
- `modern_03_verify_pages.py` — GPT identity check against Parliament API metadata.
- `modern_04_extract_bio_json.py` — GPT extraction of structured biography JSON.
- `modern_05_fallback_search_fetch.py` — Wikipedia/Wikidata-only fallback search + fetch.
- `modern_06_openweb_fallback.py` — Final open-web fallback and flagging.
