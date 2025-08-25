#!/usr/bin/env python3
"""historical_openweb_fallback.py

Open‑web fallback for **historical Hansard** speakers that still lack a
structured biography after the Wikipedia/Wikidata passes.

This script mirrors your notebook Step 10 but makes it idempotent,
checkpointed, and CLI‑friendly. It:

1) Loads the working CSV (typically the merged/enriched file).
2) Finds rows where `extracted_json` is still null/blank (configurable).
3) For each such row, builds a compact profile from
   `speaker_details_with_gender.json` (aligned by row index or an explicit
   index column) and calls an LLM with web search enabled (default
   `gpt-4o-search-preview`) to **research the open web** and return a strict
   JSON object following your schema.
4) Writes the JSON back to `extracted_json` (or a custom column), along with
   a few audit columns.

Designed to be *resumable*: it skips rows that already have output in the
selected column and saves every N rows.

USAGE
-----
# Basic: fill blanks using open‑web search, saving in place
python historical_openweb_fallback.py \
  --in merged_wikipedia_full_history_enriched.csv \
  --profiles speaker_details_with_gender.json \
  --api-key $OPENAI_API_KEY

# Write to a side column and a new file
python historical_openweb_fallback.py \
  --in merged_wikipedia_full_history_enriched.csv \
  --out merged_wikipedia_full_history_enriched_openweb.csv \
  --out-col extracted_json_openweb

# Use an explicit original index column in the CSV to look up the profile
python historical_openweb_fallback.py \
  --in merged_wikipedia_full_history_enriched.csv \
  --profiles speaker_details_with_gender.json \
  --index-col index

DEPENDENCIES
------------
  pip install pandas tqdm openai

Notes
-----
• API key is taken from --api-key or env `OPENAI_API_KEY`.
• Model defaults to `gpt-4o-search-preview` (web‑enabled).
• Output is strict JSON (best‑effort sanitised if the model drifts).
• Adds audit columns: `openweb_status`, `openweb_model`, `openweb_ts`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

# Optional OpenAI v1 SDK import (preferred)
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # resolved at runtime


# ────────────────────────────────────────────────────────────────
# Prompts
# ────────────────────────────────────────────────────────────────
OPENWEB_INSTRUCTION = (
    """
You are an automated web-research assistant.

TASK
• You will receive a short JSON profile for an historical UK parliamentarian or peer.
• Search the open web for reliable biographical information about this exact individual.
• Preferred sources: major encyclopedias (Wikipedia, Wikidata, Britannica), government or parliamentary sites, reputable newspapers, archival databases.
• If a fact is missing or cannot be verified with high confidence, output null (do NOT guess).

RETURN FORMAT
Return a single JSON object that follows exactly the schema described.
Fill fields as completely as possible; if a value is missing or cannot be inferred, return `null`.
Return only the JSON object—no extra keys, comments, or trailing text.

━━━━━━━━  OUTPUT SCHEMA  ━━━━━━━━
{
  "name": "",
  "date_of_birth": "",
  "date_of_death": "",
  "place_of_birth": "",
  "party_affiliation": [
    { "party": "", "start_year": null, "end_year": null }
  ],
  "education": {
    "school_type": "",
    "school_name": "",
    "school_country": "",
    "universities": [
      {
        "university_name": "",
        "university_city": "",
        "university_country": "",
        "degree_level": "",
        "field_of_study": ""
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
  }
}
━━━━━━━━  END OF SCHEMA  ━━━━━━━━

INPUT PROFILE:
"""
).strip()


# ────────────────────────────────────────────────────────────────
# OpenAI client
# ────────────────────────────────────────────────────────────────

def ensure_client(api_key: Optional[str]):
    if OpenAI is None:
        raise RuntimeError(
            "openai package not available; run `pip install openai`.")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key missing (use --api-key or set OPENAI_API_KEY).")
    return OpenAI(api_key=key)


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────

def is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def best_effort_extract_json(text: str) -> str:
    """Return the most plausible JSON object substring from *text*.

    If *text* already looks like JSON (starts with '{' and ends with '}'),
    return as-is. Otherwise, find the first '{' and the last '}' and extract.
    If nothing plausible, return "null" to signal failure but keep the row
    resumable.
    """
    if not isinstance(text, str):
        return "null"
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    m1 = t.find("{")
    m2 = t.rfind("}")
    if m1 != -1 and m2 != -1 and m2 > m1:
        candidate = t[m1 : m2 + 1]
        # quick sanity: balanced braces count (rough check)
        if candidate.count("{") == candidate.count("}"):
            return candidate
    return "null"


def build_profile_payload(speaker: Dict[str, Any]) -> str:
    """Clean a *speaker* dict into a compact JSON payload for the prompt.

    - drop URL if present
    - accept either `dates` or `lifespan` and normalise to `lifespan`
    - keep gender / constituencies / titles / offices text fields if present
    """
    sp = dict(speaker or {})
    sp.pop("url", None)
    if "dates" in sp and "lifespan" not in sp:
        sp["lifespan"] = sp.pop("dates")
    return json.dumps(sp, ensure_ascii=False)


def load_profiles(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_target_indices(
    df: pd.DataFrame,
    out_col: str,
    only_missing: bool = True,
    require_no_wiki_text: bool = False,
    index_slice: Optional[Tuple[int, int]] = None,
) -> List[int]:
    """Compute which row indices to process.

    By default, selects rows where *out_col* is null/blank. If
    *require_no_wiki_text* is True, also require wiki_text to be missing.
    Optionally slice by [start, stop) using *index_slice*.
    """
    mask = df[out_col].isna() | df[out_col].astype(str).str.strip().eq("")
    if not only_missing:
        mask = pd.Series([True] * len(df), index=df.index)
    if require_no_wiki_text and "wiki_text" in df.columns:
        m2 = df["wiki_text"].isna() | df["wiki_text"].astype(str).str.strip().eq("")
        mask = mask & m2
    idx = df.index[mask].tolist()
    if index_slice is not None:
        a, b = index_slice
        idx = [i for i in idx if a <= i < b]
    return idx


# ────────────────────────────────────────────────────────────────
# Core runner
# ────────────────────────────────────────────────────────────────

def run_openweb(
    in_csv: Path,
    profiles_json: Path,
    out_csv: Optional[Path] = None,
    out_col: str = "extracted_json",
    model: str = "gpt-4o-search-preview",
    batch: int = 50,
    sleep_sec: float = 1.0,
    api_key: Optional[str] = None,
    index_col: Optional[str] = None,
    only_missing: bool = True,
    require_no_wiki_text: bool = False,
    start: Optional[int] = None,
    stop: Optional[int] = None,
) -> None:
    client = ensure_client(api_key)

    # Load frame
    df = pd.read_csv(in_csv, dtype=str)
    if out_col not in df.columns:
        df[out_col] = ""

    # Load speaker profiles
    speakers = load_profiles(profiles_json)

    # Prepare target indices
    idx_slice = (start, stop) if (start is not None and stop is not None) else None
    targets = select_target_indices(
        df,
        out_col=out_col,
        only_missing=only_missing,
        require_no_wiki_text=require_no_wiki_text,
        index_slice=idx_slice,
    )

    if not targets:
        print("Nothing to process — all rows already filled or mask empty.")
        return

    # Resolve output path
    out_path = out_csv or in_csv

    # Process
    print(f"Open‑web fallback on {len(targets):,} rows → {out_path}")
    for n, i in enumerate(tqdm(targets, desc="openweb"), start=1):
        # Build the speaker payload (aligned by index or an explicit mapping column)
        try:
            if index_col and index_col in df.columns and not is_blank(df.at[i, index_col]):
                orig_idx = int(float(df.at[i, index_col]))
            else:
                orig_idx = int(i)
            speaker_obj = speakers[orig_idx]
        except Exception:
            # Fallback to row index if anything goes wrong
            orig_idx = int(i)
            speaker_obj = speakers[orig_idx] if 0 <= orig_idx < len(speakers) else {}

        payload = build_profile_payload(speaker_obj)

        # Skip if this row already has content (resumability)
        current = str(df.at[i, out_col]) if out_col in df.columns else ""
        if current and current.strip():
            continue

        # Call the model
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": OPENWEB_INSTRUCTION},
                    {"role": "user", "content": payload},
                ],
                temperature=0,
            )
            reply = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # pragma: no cover
            reply = ""
            df.at[i, "openweb_status"] = f"error: {e}"[:200]
        else:
            df.at[i, "openweb_status"] = "ok"
            df.at[i, "openweb_model"] = model
            df.at[i, "openweb_ts"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # Best-effort JSON extraction
        cleaned = best_effort_extract_json(reply)
        df.at[i, out_col] = cleaned

        # Checkpointing
        if (n % batch) == 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            time.sleep(sleep_sec)

    # Final write
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✓ saved → {out_path}")


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Open‑web fallback to fill missing biographies for historical Hansard speakers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_csv", required=True,
                   help="Input CSV (e.g., merged_wikipedia_full_history_enriched.csv)")
    p.add_argument("--profiles", required=True,
                   help="speaker_details_with_gender.json (list aligned to original order)")
    p.add_argument("--out", dest="out_csv", default=None,
                   help="Optional output CSV (defaults to in place)")
    p.add_argument("--out-col", default="extracted_json",
                   help="Column to write JSON into")
    p.add_argument("--model", default="gpt-4o-search-preview",
                   help="Web-enabled model for open‑web research")
    p.add_argument("--batch", type=int, default=50,
                   help="Rows between checkpoints")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="Seconds to sleep between checkpoints (rate-limit buffer)")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"),
                   help="OpenAI API key (else read from env OPENAI_API_KEY)")
    p.add_argument("--index-col", default=None,
                   help="Optional column name that stores the original speaker index")
    p.add_argument("--all", dest="only_missing", action="store_false",
                   help="Process all rows (instead of only where out-col is blank)")
    p.add_argument("--require-no-wiki-text", action="store_true",
                   help="Require wiki_text to be empty when selecting rows")
    p.add_argument("--start", type=int, default=None, help="Start row index (inclusive)")
    p.add_argument("--stop", type=int, default=None, help="Stop row index (exclusive)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_openweb(
        in_csv=Path(args.in_csv),
        profiles_json=Path(args.profiles),
        out_csv=Path(args.out_csv) if args.out_csv else None,
        out_col=args.out_col,
        model=args.model,
        batch=args.batch,
        sleep_sec=args.sleep,
        api_key=args.api_key,
        index_col=args.index_col,
        only_missing=args.only_missing,
        require_no_wiki_text=args.require_no_wiki_text,
        start=args.start,
        stop=args.stop,
    )


if __name__ == "__main__":
    main()
