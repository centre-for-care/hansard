#!/usr/bin/env python3
"""historical_missing_url_pipeline.py

End-to-end pipeline for *historical Hansard speakers without URLs*.

This script reproduces (and streamlines) your notebook workflow:

1) Parse the Hansard JSONL transcripts and **collect speakers without a URL**
   (deduplicated per name; keep first/last appearance dates).
2) **Clean & normalise names**, detect generic titles, and compute spans.
3) **Block + fuzzy-cluster** near-duplicate spellings into canonical
   people clusters, then assign a stable `person_id` per cluster.
4) Build minimal **profiles** (`canonical_name`, `aliases`, `first_speech`,
   `last_speech`) and use an LLM (default `gpt-4o-search-preview`) to
   **research structured biographies** for each candidate person *without URLs*.
5) Expand those biographies back to the **raw Hansard spellings** to produce:
     • `speaker_bios.csv` (core facts per Hansard spelling)
     • `speaker_parties_without_url.csv` (standardised parties)
     • `speaker_constituencies_without_url.csv` (seat periods)
     • `speaker_education_without_url.csv` (school + university rows)
6) **Gender inference** for raw Hansard spellings (rule-based + gender_guesser).
7) A short **coverage report** to measure how many rows/names are mapped.

Designed to be *idempotent* and resumable (checkpoints on long steps).
All OpenAI calls read API key from env `OPENAI_API_KEY` or `--api-key`.

USAGE (examples)
----------------
# full pipeline (default filenames)
python historical_missing_url_pipeline.py \
  extract --jsonl combined_patched2.jsonl --out transcript_no_url.csv
python historical_missing_url_pipeline.py cluster --in transcript_no_url.csv --out canonical_clusters.csv
python historical_missing_url_pipeline.py research --clusters canonical_clusters.csv --out historical_missing_url_all.csv \
  --model gpt-4o-search-preview --batch 100
python historical_missing_url_pipeline.py bios --meta historical_missing_url_all.csv --transcript transcript_no_url.csv --out speaker_bios.csv
python historical_missing_url_pipeline.py parties --meta historical_missing_url_all.csv --transcript transcript_no_url.csv --out speaker_parties_without_url.csv
python historical_missing_url_pipeline.py constituencies --meta historical_missing_url_all.csv --transcript transcript_no_url.csv --out speaker_constituencies_without_url.csv
python historical_missing_url_pipeline.py education --meta historical_missing_url_all.csv --transcript transcript_no_url.csv --out speaker_education_without_url.csv
python historical_missing_url_pipeline.py gender --transcript transcript_no_url.csv --out speaker_genders_without_url.csv
python historical_missing_url_pipeline.py report --meta historical_missing_url_all.csv --transcript transcript_no_url.csv

# one-shot: run *everything*
python historical_missing_url_pipeline.py all --jsonl combined_patched2.jsonl --api-key sk-...

Dependencies
------------
  pip install pandas rapidfuzz jellyfish networkx python-dateutil openai gender-guesser tqdm bs4

Notes
-----
• The LLM steps are conservative: temperature=0, strict JSON-only outputs.
• No Selenium or Wikipedia scraping here (this script targets *missing-URL* cases).
• If you already have precomputed files, you may run subcommands individually.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dateutil import parser as dparser
from rapidfuzz import fuzz, process as rf_process
import jellyfish

# Optional, loaded lazily in gender step
try:
    from gender_guesser.detector import Detector  # type: ignore
except Exception:  # pragma: no cover
    Detector = None  # resolved at runtime in gender step

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# Optional OpenAI import (v1 SDK)
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # resolved at runtime in research step

# ──────────────────────────────────────────────────────────────────────────────
# Constants & small utilities
# ──────────────────────────────────────────────────────────────────────────────
GENERIC_TITLES = {
    # single-word generic roles / parliamentary forms of address
    "speaker", "chairman", "chair", "president", "secretary", "treasurer",
    "minister", "advocate", "attorney", "solicitor", "chancellor",
    "archbishop", "bishop",
    # multi-word generic phrases
    "lord chancellor", "lord president", "lord bishop",
    "deputy chairman of committees", "hon member", "an hon member",
    "noble lord", "noble lords",
}

TRAILING_VERBS_RE = re.compile(r"\s+(asked|said|replied|continued|added)$", re.I)
HONORIFICS_RE = re.compile(
    r"^(mr|mrs|ms|miss|lord|lady|sir|dr|colonel|general|captain|major|earl|viscount|baron|baroness|duke|archbishop)\.?,?\s+",
    re.I,
)
ARTICLES_RE = re.compile(r"^(the|a|an)\s+", re.I)
NON_NAME_EDGES_RE = re.compile(r"^[^a-z\s-]+|[^a-z\s-]+$", re.I)
WS_RE = re.compile(r"\s+")

SOUND_BLOCK_PREFIX = True  # include first-letter with soundex to reduce collisions

# For date normalisation
_DATE_FULL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_YM = re.compile(r"^\d{4}-\d{2}$")
_DATE_Y = re.compile(r"^\d{4}$")
_THIS_YEAR = dt.datetime.now().year

CHECKPOINT_SLEEP = 0.8  # seconds between API calls (defensive)

# Party canonicalisation map (trimmed from your notebook; extend as needed)
PARTY_MAP: Dict[str, Optional[str]] = {
    # Labour family
    'Labour': 'Labour', 'Labour Party': 'Labour', 'Labour (UK)': 'Labour',
    'British Labour': 'Labour', 'British Labour Party': 'Labour',
    'Welsh Labour': 'Labour', 'Scottish Labour': 'Labour',
    'Scottish Labour Party': 'Labour', 'Blue Labour': 'Labour',
    'Labour Co-op': 'Labour Co-operative', 'Labour Co-operative': 'Labour Co-operative',
    'Labour and Co-operative': 'Labour Co-operative',
    'Labour and Co-operative Party': 'Labour Co-operative',
    'Welsh Labour (Labour and Co-operative)': 'Labour Co-operative',
    'Scottish Labour & Co-operative': 'Labour Co-operative',
    'Scottish Labour and Co-operative Party': 'Labour Co-operative',

    # Conservative family
    'Conservative': 'Conservative', 'Conservatives': 'Conservative',
    'Conservative and Unionist': 'Conservative', 'Conservative & Unionist': 'Conservative',
    'Welsh Conservative': 'Conservative', 'Scottish Conservative': 'Conservative',
    'Scottish Conservative Party': 'Conservative',
    'Scottish Conservative and Unionist Party': 'Conservative',
    'Northern Ireland Conservative': 'Conservative',
    'Conservative and National Liberal': 'Conservative',

    # Liberal / Lib-Dem family
    'Liberal Democrats': 'Liberal Democrats', 'Liberal Democrat': 'Liberal Democrats',
    'Welsh Liberal Democrats': 'Liberal Democrats', 'Scottish Liberal Democrats': 'Liberal Democrats',
    'SDP-Liberal Alliance': 'Liberal Democrats',
    'Liberal Party': 'Liberal', 'Liberal Party (UK)': 'Liberal', 'Liberal': 'Liberal',
    'Liberal Unionist': 'Liberal', 'Liberal National': 'Liberal', 'National Liberal': 'Liberal',

    # Others (examples)
    'SNP': 'SNP', 'Scottish National Party': 'SNP', 'Plaid Cymru': 'Plaid Cymru',
    'Green Party': 'Green Party', 'Green Party of England and Wales': 'Green Party',
    'UKIP': 'UKIP', 'Brexit Party': 'Brexit Party', 'Reform UK': 'Reform UK',
    'Alliance': 'Alliance', 'SDLP': 'SDLP', 'DUP': 'DUP', 'UUP': 'UUP',

    # Independents & crossbench
    'Independent': 'Independent', 'Speaker': 'Speaker',
    'Crossbench': 'Crossbench', 'Cross-bench': 'Crossbench', 'Crossbencher': 'Crossbench',

    # None / non-affiliated
    'None': None, 'Non-affiliated': None, 'Unaffiliated': None,
}

# ──────────────────────────────────────────────────────────────────────────────
# Name cleaning & blocking
# ──────────────────────────────────────────────────────────────────────────────
def clean_name(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    n = s.strip()
    n = TRAILING_VERBS_RE.sub("", n)
    n = HONORIFICS_RE.sub("", n)
    n = ARTICLES_RE.sub("", n)
    n = n.lower()
    n = NON_NAME_EDGES_RE.sub("", n)
    n = WS_RE.sub(" ", n).strip()
    return n

def block_key(name: str) -> str:
    """Surname soundex with optional first-letter prefix to reduce collisions."""
    toks = name.split()
    if not toks:
        return ""
    last = toks[-1]
    sx = jellyfish.soundex(last) if last else ""
    return (toks[0][0] + "-" + sx) if (SOUND_BLOCK_PREFIX and toks and sx) else sx

# ──────────────────────────────────────────────────────────────────────────────
# Transcript parsing
# ──────────────────────────────────────────────────────────────────────────────
def stream_transcript_rows(jsonl_path: Path) -> Iterator[dict]:
    """Yield {date, title, speaker, speaker_url, speech} from nested JSONL."""
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            date = rec.get("day")
            chambers = rec.get("data", {}).get("chambers", {})
            for entries in chambers.values():
                for entry in entries or []:
                    yield from _walk_entry(entry, date)


def _walk_entry(entry: dict, date: str) -> Iterator[dict]:
    title = entry.get("title", "")
    for content in entry.get("content") or []:
        yield {
            "date": date,
            "title": title,
            "speaker": content.get("speaker"),
            "speaker_url": content.get("speaker_url"),
            "speech": content.get("text") or content.get("speech") or "",
        }
    for sub in entry.get("subsections") or []:
        yield from _walk_entry(sub, date)


# ──────────────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────────────
def cluster_speakers(df_no_url: pd.DataFrame) -> pd.DataFrame:
    """Return canonical clusters with person_id and alias sets.

    Input: dataframe with columns [date, title, speaker, speaker_clean].
    Output columns: [person_id, canonical_name, aliases, first, last,
                     gap_days, n_variants, n_rows]
    """
    work = df_no_url.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    # Span per cleaned spelling
    span = (work.groupby("speaker_clean")["date"].agg(["min", "max"]).reset_index())
    span["gap_days"] = (span["max"] - span["min"]).dt.days
    span["block"] = span["speaker_clean"].apply(block_key)

    # Build edges only within blocks using RapidFuzz
    rows = []
    for _, g in span.groupby("block"):
        names = g["speaker_clean"].tolist()
        idxs = g.index.tolist()
        # pairwise within block
        for i, ia in enumerate(idxs):
            a = span.loc[ia, "speaker_clean"]
            for ib in idxs[i+1:]:
                b = span.loc[ib, "speaker_clean"]
                score = fuzz.token_set_ratio(a, b)
                if score > 90:
                    rows.append((a, b))

    G = nx.Graph()
    G.add_edges_from(rows)
    # Add singleton nodes
    for n in span["speaker_clean"].tolist():
        if n not in G:
            G.add_node(n)

    clusters = list(nx.connected_components(G))

    canon_rows = []
    for cid, names in enumerate(clusters, start=1):
        names = sorted(names)
        subset = span[span["speaker_clean"].isin(names)]
        first, last = subset["min"].min(), subset["max"].max()
        gap = (last - first).days if pd.notna(first) and pd.notna(last) else np.nan
        # simple representative: longest token length
        rep = max(names, key=len)
        canon_rows.append({
            "person_id": f"P{cid:05d}",
            "canonical_name": rep,
            "aliases": "; ".join(names),
            "first": first,
            "last": last,
            "gap_days": gap,
            "n_variants": len(names),
            "n_rows": int(work[work["speaker_clean"].isin(names)].shape[0]),
        })

    canon_df = pd.DataFrame(canon_rows).sort_values(["first", "canonical_name"]).reset_index(drop=True)
    return canon_df


# ──────────────────────────────────────────────────────────────────────────────
# Profiles & LLM research
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Profile:
    canonical_name: str
    aliases: List[str]
    first_speech: str  # YYYY-MM-DD
    last_speech: str   # YYYY-MM-DD


def _iso_date(val) -> str:
    if pd.isna(val) or val == "":
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d")
    except Exception:
        return str(val)[:10]


def build_profiles(canon_df: pd.DataFrame) -> List[Profile]:
    profiles: List[Profile] = []
    for _, r in canon_df.iterrows():
        alias_list = [a.strip() for a in str(r.get("aliases", "")).split(";") if a.strip()]
        profiles.append(Profile(
            canonical_name=str(r["canonical_name"]),
            aliases=alias_list,
            first_speech=_iso_date(r["first"]),
            last_speech=_iso_date(r["last"]),
        ))
    return profiles


RESEARCH_INSTRUCTION = (
    """
You are an automated web-research assistant.

TASK
• You will receive a short **JSON profile** for one UK parliamentarian or peer.
  The profile always contains:
    { "canonical_name": "...",      // primary string to search
      "aliases": [ ... ],             // 0..N alternative spellings / titles
      "first_speech": "YYYY-MM-DD", // earliest Hansard appearance
      "last_speech":  "YYYY-MM-DD" } // latest Hansard appearance
• Search reputable sources (Wikipedia, Wikidata, Britannica, official
  parliamentary or government sites, reputable newspapers/archives).
• If a fact is missing or cannot be verified with high confidence, output null.

RETURN FORMAT
Return **only** a single JSON object — no commentary or code fences. The very
first character must be '{' and the very last must be '}'.
Fill fields as completely as possible; if unknown, use null.

━━━━━━━━  OUTPUT SCHEMA  ━━━━━━━━
{
  "name": "",
  "date_of_birth": "",            // Prefer YYYY-MM-DD; allow YYYY-MM or YYYY; else null
  "date_of_death": "",
  "place_of_birth": "",
  "party_affiliation": [
    { "party": "", "start_year": null, "end_year": null }
  ],
  "education": {
    "school_type": "",            // "Clarendon", "HMC schools", "Other private", "All other", or null
    "school_name": "",
    "school_country": "",
    "universities": [
      {
        "university_name": "",
        "university_city": "",
        "university_country": "",
        "degree_level": "",
        "field_of_study": "",
        "start_year": null,
        "end_year": null
      }
    ]
  },
  "occupation_before_politics": "",
  "political_career": {
    "first_elected": null,
    "last_elected": null,
    "years_in_parliament": null,
    "ministerial_positions": [],
    "leadership_positions": []
  },
  "constituencies": [              // chronological order earliest → latest
    { "seat": "", "start": "", "end": "" }
  ]
}
━━━━━━━━  END OF SCHEMA  ━━━━━━━━
"""
).strip()


def ensure_openai_client(api_key: Optional[str]) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai package not available; `pip install openai`.")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key missing (use --api-key or set OPENAI_API_KEY).")
    return OpenAI(api_key=key)


def llm_research_profiles(
    canon_df: pd.DataFrame,
    profiles: List[Profile],
    out_csv: Path,
    model: str = "gpt-4o-search-preview",
    batch_size: int = 100,
    api_key: Optional[str] = None,
    sleep_sec: float = CHECKPOINT_SLEEP,
) -> pd.DataFrame:
    """Query the LLM once per profile; write incremental checkpoints.

    Output columns: [person_id, canonical_name, aliases, gpt_json]
    """
    client = ensure_openai_client(api_key)

    out_cols = ["person_id", "canonical_name", "aliases", "gpt_json"]
    if out_csv.exists():
        df_meta = pd.read_csv(out_csv, dtype=str)
        missing = len(canon_df) - len(df_meta)
        if missing > 0:
            # pad if an earlier run aborted
            for _ in range(missing):
                df_meta.loc[len(df_meta)] = {c: "" for c in out_cols}
    else:
        df_meta = pd.DataFrame([{c: "" for c in out_cols} for _ in range(len(canon_df))])

    done = df_meta["gpt_json"].astype(str).str.strip().ne("").sum()

    for i in tqdm(range(len(canon_df)), desc="LLM research"):
        if i < done:
            continue

        pid = str(canon_df.at[i, "person_id"])  # align by row
        prof = profiles[i]
        user_payload = json.dumps(dataclasses.asdict(prof), ensure_ascii=False)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RESEARCH_INSTRUCTION},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0,
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # pragma: no cover
            answer = ""
            print(f"[warn] API error at row {i} (pid={pid}): {e}")

        df_meta.loc[i, "person_id"] = pid
        df_meta.loc[i, "canonical_name"] = prof.canonical_name
        df_meta.loc[i, "aliases"] = "; ".join(prof.aliases)
        df_meta.loc[i, "gpt_json"] = answer

        # checkpoint
        if (i + 1) % batch_size == 0 or (i + 1) == len(canon_df):
            df_meta.to_csv(out_csv, index=False, encoding="utf-8-sig")
            time.sleep(sleep_sec)

    return df_meta


# ──────────────────────────────────────────────────────────────────────────────
# Expansion to raw Hansard names & table builders
# ──────────────────────────────────────────────────────────────────────────────
def build_raw_map(df_transcript: pd.DataFrame) -> Dict[str, List[str]]:
    return (
        df_transcript.groupby("speaker_clean")["speaker"]
        .apply(lambda s: sorted(set(s.dropna())))
        .to_dict()
    )


def deep_get(d: dict, *keys, default=""):
    cur = d
    for k in keys:
        if cur is None or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def normalise_date(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    if _DATE_FULL.match(s):
        return s
    if _DATE_YM.match(s):
        return f"{s}-01"
    if _DATE_Y.match(s):
        y = int(s)
        if 1000 <= y <= _THIS_YEAR:
            return f"{y}-01-01"
    # free-form fallback
    try:
        return pd.to_datetime(s, errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return ""


def expand_bios_to_hansard(df_meta: pd.DataFrame, df_transcript: pd.DataFrame) -> pd.DataFrame:
    raw_map = build_raw_map(df_transcript)

    rows = []
    for _, m in df_meta.iterrows():
        pid = m.get("person_id")
        blob = m.get("gpt_json")
        if not isinstance(blob, str) or not blob.strip():
            continue
        try:
            bio = json.loads(blob)
        except Exception:
            continue

        clean_set = set()
        if isinstance(m.get("canonical_name"), str):
            clean_set.add(m["canonical_name"].strip().lower())
        if isinstance(m.get("aliases"), str):
            clean_set.update(a.strip().lower() for a in m["aliases"].split(";") if a.strip())
        clean_set.update(
            df_transcript.loc[df_transcript["person_id"] == pid, "speaker_clean"].dropna().str.lower()
        )

        for clean_nm in clean_set:
            raw_variants = raw_map.get(clean_nm, [clean_nm])
            for raw in raw_variants:
                rows.append({
                    "name": deep_get(bio, "name"),
                    "name_from_hansard": raw,
                    "date_of_birth": normalise_date(deep_get(bio, "date_of_birth")),
                    "date_of_death": normalise_date(deep_get(bio, "date_of_death")),
                    "place_of_birth": deep_get(bio, "place_of_birth"),
                    "occupation_before_politics": deep_get(bio, "occupation_before_politics"),
                    "first_elected": normalise_date(deep_get(bio, "political_career", "first_elected")),
                    "last_elected": normalise_date(deep_get(bio, "political_career", "last_elected")),
                    "ministerial_positions": "; ".join(deep_get(bio, "political_career", "ministerial_positions", default=[])),
                    "leadership_positions": "; ".join(deep_get(bio, "political_career", "leadership_positions", default=[])),
                })

    bio_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["name_from_hansard"], keep="first")
        .reset_index(drop=True)
    )

    # fallback: ensure name has something
    bio_df["name"] = bio_df["name"].fillna("").str.strip()
    mask_blank = bio_df["name"].eq("")
    bio_df.loc[mask_blank, "name"] = bio_df.loc[mask_blank, "name_from_hansard"]
    return bio_df


def build_parties(df_meta: pd.DataFrame, df_transcript: pd.DataFrame) -> pd.DataFrame:
    raw_map = build_raw_map(df_transcript)

    rows = []
    for _, m in df_meta.iterrows():
        pid = m.get("person_id")
        blob = m.get("gpt_json")
        if not isinstance(blob, str) or not blob.strip():
            continue
        try:
            bio = json.loads(blob)
        except Exception:
            continue
        affs = bio.get("party_affiliation") or []

        # gather spellings for this pid
        spellings = set()
        if isinstance(m.get("canonical_name"), str):
            spellings.add(m["canonical_name"].strip().lower())
        if isinstance(m.get("aliases"), str):
            spellings.update(a.strip().lower() for a in m["aliases"].split(";") if a.strip())
        spellings.update(df_transcript.loc[df_transcript["person_id"] == pid, "speaker"].dropna().str.lower())

        for aff in affs:
            if not isinstance(aff, dict):
                continue
            party_raw = str(aff.get("party", "")).strip()
            if not party_raw:
                continue
            party_std = PARTY_MAP.get(party_raw, party_raw)

            for clean_nm in spellings:
                for raw in raw_map.get(clean_nm, [clean_nm]):
                    rows.append({
                        "name_from_hansard": raw,
                        "party": party_std,
                        "start_year": _to_year(aff.get("start_year")),
                        "end_year": _to_year(aff.get("end_year")),
                    })

    out = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["name_from_hansard", "start_year", "party"])
        .reset_index(drop=True)
    )
    for col in ("start_year", "end_year"):
        out[col] = out[col].astype("Int64")
    return out


def _to_year(x) -> Optional[int]:
    if x in (None, "") or (isinstance(x, float) and np.isnan(x)):
        return pd.NA
    try:
        y = int(float(x))
        if 1000 <= y <= _THIS_YEAR + 5:
            return y
    except Exception:
        return pd.NA
    return pd.NA


def _fix_zero_date(val: str) -> str:
    if not isinstance(val, str):
        return ""
    s = val.strip()
    if not s:
        return ""
    m = re.match(r"^(\d{4})-00-00$", s)
    if m:
        return f"{m.group(1)}-01-01"
    m = re.match(r"^(\d{4})-(\d{2})-00$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    return s


def build_constituencies(df_meta: pd.DataFrame, df_transcript: pd.DataFrame) -> pd.DataFrame:
    raw_map = build_raw_map(df_transcript)
    rows = []

    for _, m in df_meta.iterrows():
        pid = m.get("person_id")
        blob = m.get("gpt_json")
        if not isinstance(blob, str) or not blob.strip():
            continue
        try:
            bio = json.loads(blob)
        except Exception:
            continue
        consts = bio.get("constituencies") or []

        spellings = set()
        if isinstance(m.get("canonical_name"), str):
            spellings.add(m["canonical_name"].strip().lower())
        if isinstance(m.get("aliases"), str):
            spellings.update(a.strip().lower() for a in m["aliases"].split(";") if a.strip())
        spellings.update(df_transcript.loc[df_transcript["person_id"] == pid, "speaker"].dropna().str.lower())

        for c in consts:
            if not isinstance(c, dict):
                continue
            seat = str(c.get("seat", "")).strip()
            start = normalise_date(c.get("start"))
            end = normalise_date(c.get("end"))
            start, end = _fix_zero_date(start), _fix_zero_date(end)

            for clean_nm in spellings:
                for raw in raw_map.get(clean_nm, [clean_nm]):
                    rows.append({
                        "name_from_hansard": raw,
                        "seat": seat or np.nan,
                        "start": start,
                        "end": end,
                    })

    out = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["name_from_hansard", "start", "seat"])
        .reset_index(drop=True)
    )
    return out


def build_education(df_meta: pd.DataFrame, df_transcript: pd.DataFrame, drop_years: bool = True) -> pd.DataFrame:
    raw_map = build_raw_map(df_transcript)

    def _norm_blank(x):
        return np.nan if (x in ("", None)) else x

    def _to_year_nullable(x):
        y = _to_year(x)
        return y if y is not pd.NA else pd.NA

    rows = []
    for _, m in df_meta.iterrows():
        pid = m.get("person_id")
        blob = m.get("gpt_json")
        if not isinstance(blob, str) or not blob.strip():
            continue
        try:
            bio = json.loads(blob)
        except Exception:
            continue
        edu = bio.get("education") or {}

        spellings = set()
        if isinstance(m.get("canonical_name"), str):
            spellings.add(m["canonical_name"].strip().lower())
        if isinstance(m.get("aliases"), str):
            spellings.update(a.strip().lower() for a in m["aliases"].split(";") if a.strip())
        spellings.update(df_transcript.loc[df_transcript["person_id"] == pid, "speaker"].dropna().str.lower())

        # School row (if any)
        if any(edu.get(k) for k in ("school_name", "school_type", "school_country")):
            for clean_nm in spellings:
                for raw in raw_map.get(clean_nm, [clean_nm]):
                    rows.append({
                        "name_from_hansard": raw,
                        "level": "school",
                        "institution": _norm_blank(edu.get("school_name")),
                        "type": _norm_blank(edu.get("school_type")),
                        "country": _norm_blank(edu.get("school_country")),
                        "degree": pd.NA,
                        "field": pd.NA,
                        "start_year": pd.NA,
                        "end_year": pd.NA,
                    })

        # Universities
        for uni in edu.get("universities") or []:
            if isinstance(uni, str):
                uni_dict = {
                    "institution": uni.strip(),
                    "type": pd.NA,
                    "country": pd.NA,
                    "degree": pd.NA,
                    "field": pd.NA,
                    "start_year": pd.NA,
                    "end_year": pd.NA,
                }
            elif isinstance(uni, dict):
                uni_dict = {
                    "institution": _norm_blank(uni.get("university_name") or uni.get("name")),
                    "type": _norm_blank(uni.get("university_type") or uni.get("type")),
                    "country": _norm_blank(uni.get("university_country") or uni.get("country")),
                    "degree": _norm_blank(uni.get("degree_level") or uni.get("degree")),
                    "field": _norm_blank(uni.get("field_of_study") or uni.get("subject")),
                    "start_year": _to_year_nullable(uni.get("start_year")),
                    "end_year": _to_year_nullable(uni.get("end_year")),
                }
            else:
                continue

            for clean_nm in spellings:
                for raw in raw_map.get(clean_nm, [clean_nm]):
                    rows.append({
                        "name_from_hansard": raw,
                        "level": "university",
                        **uni_dict,
                    })

    out = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["name_from_hansard", "level", "institution"])
        .reset_index(drop=True)
    )

    if drop_years and not out.empty:
        out = out.drop(columns=[c for c in ("start_year", "end_year") if c in out.columns])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Gender inference for raw Hansard spellings
# ──────────────────────────────────────────────────────────────────────────────
MR_TOKENS = {"mr", "sir", "lord", "baron", "viscount", "duke", "earl", "marquess"}
MRS_TOKENS = {"mrs", "ms", "miss", "lady", "dame", "duchess", "countess", "baroness", "viscountess"}
TOKEN_RE = re.compile(r"\b(" + "|".join(MR_TOKENS | MRS_TOKENS) + r")\b", re.I)
TITLE_MAP = {t: "male" for t in MR_TOKENS} | {t: "female" for t in MRS_TOKENS}
PREFIX_RE = re.compile(
    r"^(?:lieut[\-\s]?colonel|lt[\-\s]?col|major|colonel|brigadier|"
    r"lieut[\-\s]?commander|lieutenant|captain|commander|commodore|"
    r"wing[\-\s]?commander|group[\-\s]?captain|air[\-\s]?commodore|"
    r"rear[\-\s]?admiral|vice[\-\s]?admiral|admiral|field[\-\s]?marshal|"
    r"major[\-\s]?general|lieut[\-\s]?general|general|"
    r"viscountess|viscount|baroness|baron|lord|lady|earl|countess|"
    r"marquess|marchioness|duke|duchess|hon|sir|dame|dr|professor|prof|"
    r"reverend|rev)\s+",
    re.I,
)


def _gender_explicit(txt: str) -> Optional[str]:
    m = TOKEN_RE.search(txt)
    if not m:
        return None
    tok = m.group(1).lower()
    return "male" if tok in MR_TOKENS else "female"


def _gender_first_word(txt: str) -> Optional[str]:
    first = txt.split()[0] if txt else ""
    return TITLE_MAP.get(first)


def _gender_from_firstname(txt: str, det: Optional["Detector"]) -> Optional[str]:
    stripped = PREFIX_RE.sub("", txt).strip()
    if not stripped:
        return None
    first = stripped.split()[0].capitalize()
    if det is None:
        return None
    g = det.get_gender(first)
    if g in ("male", "mostly_male"):
        return "male"
    if g in ("female", "mostly_female"):
        return "female"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CLI subcommands
# ──────────────────────────────────────────────────────────────────────────────
def cmd_extract(args: argparse.Namespace) -> None:
    jsonl = Path(args.jsonl)
    out_csv = Path(args.out)

    rows = []
    for rec in tqdm(stream_transcript_rows(jsonl), desc="scan"):
        spk = rec.get("speaker")
        if not spk:
            continue
        if rec.get("speaker_url"):
            continue  # only missing-URL cases here
        clean = clean_name(spk)
        if not clean:
            continue
        rows.append({
            "date": rec.get("date"),
            "title": rec.get("title", ""),
            "speaker": spk,
            "speaker_clean": clean,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✓ wrote {len(df):,} rows → {out_csv}")


def cmd_cluster(args: argparse.Namespace) -> None:
    inp = Path(args.in_path)
    out = Path(args.out)
    df = pd.read_csv(inp, dtype=str)
    canon = cluster_speakers(df)

    # Map canonical person_id back onto transcript rows
    name2id = {}
    for _, r in canon.iterrows():
        pid = r["person_id"]
        for a in str(r["aliases"]).split(";"):
            a = a.strip()
            if a:
                name2id[a] = pid
        name2id[r["canonical_name"]] = pid

    df["person_id"] = df["speaker_clean"].map(name2id)
    # persist both
    canon.to_csv(out, index=False, encoding="utf-8-sig")
    df.to_csv(inp, index=False, encoding="utf-8-sig")  # update in-place
    print(f"✓ clusters → {out}  | transcript updated with person_id")


def cmd_research(args: argparse.Namespace) -> None:
    clusters = Path(args.clusters)
    out = Path(args.out)
    canon_df = pd.read_csv(clusters, dtype=str)
    profiles = build_profiles(canon_df)
    df_meta = llm_research_profiles(
        canon_df=canon_df,
        profiles=profiles,
        out_csv=out,
        model=args.model,
        batch_size=args.batch,
        api_key=args.api_key,
        sleep_sec=args.sleep,
    )
    print(f"✓ wrote {len(df_meta):,} rows → {out}")


def cmd_bios(args: argparse.Namespace) -> None:
    meta = Path(args.meta)
    transcript = Path(args.transcript)
    out = Path(args.out)

    df_meta = pd.read_csv(meta, dtype=str)
    df_tr = pd.read_csv(transcript, dtype=str)

    # Ensure transcript has person_id mapping
    if "person_id" not in df_tr.columns:
        raise SystemExit("transcript CSV must include person_id; run 'cluster' first")

    bios = expand_bios_to_hansard(df_meta, df_tr)
    bios = bios[[
        "name", "name_from_hansard", "date_of_birth", "date_of_death",
        "place_of_birth", "occupation_before_politics",
        "first_elected", "last_elected",
        "ministerial_positions", "leadership_positions",
    ]]
    bios.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✓ wrote {len(bios):,} rows → {out}")


def cmd_parties(args: argparse.Namespace) -> None:
    meta = Path(args.meta)
    transcript = Path(args.transcript)
    out = Path(args.out)

    df_meta = pd.read_csv(meta, dtype=str)
    df_tr = pd.read_csv(transcript, dtype=str)

    part = build_parties(df_meta, df_tr)
    part.to_csv(out, index=False, encoding="utf-8-sig", na_rep="")
    print(f"✓ wrote {len(part):,} rows → {out}")


def cmd_const(args: argparse.Namespace) -> None:
    meta = Path(args.meta)
    transcript = Path(args.transcript)
    out = Path(args.out)

    df_meta = pd.read_csv(meta, dtype=str)
    df_tr = pd.read_csv(transcript, dtype=str)

    const_df = build_constituencies(df_meta, df_tr)
    const_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✓ wrote {len(const_df):,} rows → {out}")


def cmd_edu(args: argparse.Namespace) -> None:
    meta = Path(args.meta)
    transcript = Path(args.transcript)
    out = Path(args.out)

    df_meta = pd.read_csv(meta, dtype=str)
    df_tr = pd.read_csv(transcript, dtype=str)

    edu_df = build_education(df_meta, df_tr, drop_years=True)
    edu_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✓ wrote {len(edu_df):,} rows → {out}")


def cmd_gender(args: argparse.Namespace) -> None:
    transcript = Path(args.transcript)
    out = Path(args.out)
    df_tr = pd.read_csv(transcript, dtype=str)

    # lazy import gender_guesser
    global Detector
    if Detector is None:
        try:
            from gender_guesser.detector import Detector  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit("pip install gender-guesser") from e

    det = Detector(case_sensitive=False)

    rows = []
    seen = set()
    for raw in sorted(df_tr["speaker"].dropna().unique()):
        if raw in seen:
            continue
        seen.add(raw)
        cleaned = clean_name(raw)
        if cleaned.startswith("the "):
            cleaned = cleaned[4:].lstrip()

        gender = _gender_explicit(cleaned)
        if gender is None:
            first_tok = cleaned.split()[0] if cleaned else ""
            if first_tok in GENERIC_TITLES or not cleaned:
                gender = None
            else:
                gender = _gender_first_word(cleaned) or _gender_from_firstname(cleaned, det)

        rows.append({"name_from_hansard": raw, "gender": gender})

    gender_df = pd.DataFrame(rows).sort_values("name_from_hansard").reset_index(drop=True)
    # drop blank names if any
    gender_df = gender_df[gender_df["name_from_hansard"].astype(str).str.strip().ne("")]
    gender_df.to_csv(out, index=False, encoding="utf-8-sig", na_rep="")
    print(f"✓ wrote {len(gender_df):,} rows → {out}")


def cmd_report(args: argparse.Namespace) -> None:
    meta = Path(args.meta)
    transcript = Path(args.transcript)

    df_meta = pd.read_csv(meta, dtype=str)
    df_tr = pd.read_csv(transcript, dtype=str)

    meta_ids = set(df_meta["person_id"].dropna())
    mask = df_tr["person_id"].isin(meta_ids)
    rows_with_meta = mask.sum()
    total_rows = len(df_tr)
    uniq_ids_df = df_tr["person_id"].nunique()
    uniq_ids_with_meta = len(set(df_tr["person_id"]) & meta_ids)

    print(
        f"{rows_with_meta:,} of {total_rows:,} transcript rows "
        f"({rows_with_meta/total_rows:.2%}) have metadata."
    )
    print(
        f"{uniq_ids_with_meta:,} of {uniq_ids_df:,} unique person_ids "
        f"({(uniq_ids_with_meta / max(1, uniq_ids_df)):.2%}) have metadata."
    )


def cmd_all(args: argparse.Namespace) -> None:
    # Paths
    jsonl = Path(args.jsonl)
    tr_csv = Path(args.transcript or "transcript_no_url.csv")
    clusters_csv = Path(args.clusters or "canonical_clusters.csv")
    meta_csv = Path(args.meta or "historical_missing_url_all.csv")

    # 1) extract
    if not tr_csv.exists():
        cmd_extract(argparse.Namespace(jsonl=str(jsonl), out=str(tr_csv)))

    # 2) cluster
    if not clusters_csv.exists():
        cmd_cluster(argparse.Namespace(in_path=str(tr_csv), out=str(clusters_csv)))

    # 3) research
    if not meta_csv.exists():
        cmd_research(argparse.Namespace(
            clusters=str(clusters_csv), out=str(meta_csv), model=args.model,
            batch=args.batch, api_key=args.api_key, sleep=args.sleep
        ))

    # 4) tables
    cmd_bios(argparse.Namespace(meta=str(meta_csv), transcript=str(tr_csv), out=str(args.bios)))
    cmd_parties(argparse.Namespace(meta=str(meta_csv), transcript=str(tr_csv), out=str(args.parties)))
    cmd_const(argparse.Namespace(meta=str(meta_csv), transcript=str(tr_csv), out=str(args.constituencies)))
    cmd_edu(argparse.Namespace(meta=str(meta_csv), transcript=str(tr_csv), out=str(args.education)))
    cmd_gender(argparse.Namespace(transcript=str(tr_csv), out=str(args.gender)))
    cmd_report(argparse.Namespace(meta=str(meta_csv), transcript=str(tr_csv)))


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pipeline for historical speakers without URLs")
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    sp = sub.add_parser("extract", help="Parse JSONL and collect speakers without URL")
    sp.add_argument("--jsonl", required=True, help="combined_patched2.jsonl")
    sp.add_argument("--out", required=True, help="Output CSV for transcript (no URL)")
    sp.set_defaults(func=cmd_extract)

    # cluster
    sp = sub.add_parser("cluster", help="Cluster spellings into canonical people + assign person_id")
    sp.add_argument("--in", dest="in_path", required=True, help="transcript_no_url.csv")
    sp.add_argument("--out", required=True, help="canonical_clusters.csv")
    sp.set_defaults(func=cmd_cluster)

    # research
    sp = sub.add_parser("research", help="LLM research to produce structured JSON per person")
    sp.add_argument("--clusters", required=True, help="canonical_clusters.csv")
    sp.add_argument("--out", required=True, help="historical_missing_url_all.csv")
    sp.add_argument("--model", default="gpt-4o-search-preview")
    sp.add_argument("--batch", type=int, default=100)
    sp.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    sp.add_argument("--sleep", type=float, default=CHECKPOINT_SLEEP)
    sp.set_defaults(func=cmd_research)

    # bios
    sp = sub.add_parser("bios", help="Expand JSON bios back to raw Hansard spellings")
    sp.add_argument("--meta", required=True)
    sp.add_argument("--transcript", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_bios)

    # parties
    sp = sub.add_parser("parties", help="Build party affiliations table")
    sp.add_argument("--meta", required=True)
    sp.add_argument("--transcript", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_parties)

    # constituencies
    sp = sub.add_parser("constituencies", help="Build constituencies table")
    sp.add_argument("--meta", required=True)
    sp.add_argument("--transcript", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_const)

    # education
    sp = sub.add_parser("education", help="Build education table (school/universities)")
    sp.add_argument("--meta", required=True)
    sp.add_argument("--transcript", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_edu)

    # gender
    sp = sub.add_parser("gender", help="Infer gender for raw Hansard spellings")
    sp.add_argument("--transcript", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_gender)

    # report
    sp = sub.add_parser("report", help="Coverage report for mapping")
    sp.add_argument("--meta", required=True)
    sp.add_argument("--transcript", required=True)
    sp.set_defaults(func=cmd_report)

    # all-in-one
    sp = sub.add_parser("all", help="Run the entire pipeline end-to-end")
    sp.add_argument("--jsonl", required=True)
    sp.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    sp.add_argument("--model", default="gpt-4o-search-preview")
    sp.add_argument("--batch", type=int, default=100)
    sp.add_argument("--sleep", type=float, default=CHECKPOINT_SLEEP)
    sp.add_argument("--transcript", default="transcript_no_url.csv")
    sp.add_argument("--clusters", default="canonical_clusters.csv")
    sp.add_argument("--meta", default="historical_missing_url_all.csv")
    sp.add_argument("--bios", default="speaker_bios.csv")
    sp.add_argument("--parties", default="speaker_parties_without_url.csv")
    sp.add_argument("--constituencies", default="speaker_constituencies_without_url.csv")
    sp.add_argument("--education", default="speaker_education_without_url.csv")
    sp.add_argument("--gender", default="speaker_genders_without_url.csv")
    sp.set_defaults(func=cmd_all)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
