#!/usr/bin/env python3
"""
modern_06_openweb_fallback.py

Final open-web fallback for modern speakers whose biography JSON is still
missing after all Wikipedia/Wikidata passes.

What it does
------------
* Select rows where `extracted_json` is blank/NaN.
* Build a compact member profile (from the Members API JSON) per row.
* Call GPT-4o-search-preview (open web) and expect a single strict JSON object.
* Save JSON to `extracted_json` and raw API response to `api_response_full`.
* Resumable: already-filled rows are skipped; checkpoints every --batch-size.

Inputs
------
--input        : CSV from the previous step (e.g., after step 05/03/04 merges)
--members-json : JSON from step 00 (e.g., modern_members_full_responses.json)

Output
------
--output : same CSV shape plus filled `extracted_json` / `api_response_full`.

Usage
-----
python modern_06_openweb_fallback.py \
  --input modern_verify_or_merge.csv \
  --members-json modern_members_full_responses.json \
  --output modern_verify_or_merge_openweb.csv \
  --api-key $OPENAI_API_KEY \
  --batch-size 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI  # OpenAI v1 SDK

# ──────────────────────────────────────────────────────────────────────────────
# Prompt (open-web)
# ──────────────────────────────────────────────────────────────────────────────
OPENWEB_SYSTEM = """
You are an automated web-research assistant.

TASK
• You will receive a short JSON profile for a UK parliamentarian or peer.
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
  },
  "constituencies": [
    { "seat": "", "start": "", "end": "" }
  ]
}
━━━━━━━━  END OF SCHEMA  ━━━━━━━━

INPUT PROFILE:
""".strip()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def iter_members(data: Any) -> Iterable[dict]:
    """
    Yield member dicts from common Members API shapes:
      • [ {...}, {...}, ... ]
      • [ {"value": {...}}, ... ]
      • { "items": [ {"value": {...}}, ... ] }
      • a single {"value": {...}} or a single {...}
    """
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "value" in item and isinstance(item["value"], dict):
                yield item["value"]
            elif isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "value" in item and isinstance(item["value"], dict):
                    yield item["value"]
                elif isinstance(item, dict):
                    yield item
        else:
            yield data

def index_members_by_id(members: List[dict]) -> Dict[int, dict]:
    """Build a map from member `id` → member dict (best effort)."""
    out: Dict[int, dict] = {}
    for m in members:
        mid = m.get("id")
        if isinstance(mid, str):
            try:
                mid = int(mid)
            except Exception:
                mid = None
        if isinstance(mid, int):
            out[mid] = m
    return out

def get_member_for_row(row: pd.Series, members_by_id: Dict[int, dict], members_list: List[dict], row_index: int) -> dict:
    """
    Prefer lookup by `member_id` column if present; else fallback to positional index.
    """
    if "member_id" in row and pd.notna(row["member_id"]):
        try:
            mid = int(float(row["member_id"]))
            if mid in members_by_id:
                return members_by_id[mid]
        except Exception:
            pass
    if 0 <= row_index < len(members_list):
        return members_list[row_index]
    return {}

def best_effort_extract_json(text: str) -> str:
    """
    Extract the most plausible JSON object from text (defensive).
    Returns "null" if nothing plausible is found.
    """
    if not isinstance(text, str):
        return "null"
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b != -1 and b > a:
        cand = t[a:b+1]
        if cand.count("{") == cand.count("}"):
            return cand
    return "null"

def call_openweb(client: OpenAI, profile_json: str, retries: int = 2) -> tuple[str, str]:
    """
    Call GPT with the open-web instruction. Returns (clean_json, raw_api_json).
    clean_json is the best-effort extracted JSON object as a string (or "null").
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-search-preview",
                messages=[
                    {"role": "system", "content": OPENWEB_SYSTEM},
                    {"role": "user", "content": profile_json},
                ],
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()
            cleaned = best_effort_extract_json(content)
            return cleaned, json.dumps(resp.to_dict(), ensure_ascii=False)
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            if ("rate limit" in msg or "429" in msg) and attempt < retries:
                logging.warning("Rate-limited; sleeping 20s (attempt %d/%d)…", attempt, retries)
                time.sleep(20)
                continue
            logging.error("OpenAI error: %s", exc)
            break
    # On failure, return a null JSON with empty raw (but keep resumability)
    return "null", ("" if last_err is None else json.dumps({"error": str(last_err)}))

# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────
def run_openweb(
    in_csv: Path,
    members_json: Path,
    out_csv: Path,
    api_key: str,
    batch_size: int,
) -> None:
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(in_csv, dtype=str)
    for col in ("extracted_json", "api_response_full"):
        if col not in df.columns:
            df[col] = ""

    raw_members = json.loads(members_json.read_text(encoding="utf-8"))
    members_list = list(iter_members(raw_members))
    members_by_id = index_members_by_id(members_list)

    # Targets: rows where extracted_json is blank/NaN
    to_process = df.index[df["extracted_json"].isna() | df["extracted_json"].astype(str).str.strip().eq("")].tolist()
    logging.info("Open-web fallback needed for %d rows", len(to_process))

    for n, idx in enumerate(to_process, start=1):
        row = df.loc[idx]
        prof = get_member_for_row(row, members_by_id, members_list, idx)
        profile_json = json.dumps(prof or {}, ensure_ascii=False)

        cleaned_json, raw_api = call_openweb(client, profile_json)
        df.at[idx, "extracted_json"] = cleaned_json
        df.at[idx, "api_response_full"] = raw_api

        if n % batch_size == 0 or n == len(to_process):
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d → %s", n, len(to_process), out_csv)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final open-web fallback to fill missing biography JSON.")
    p.add_argument("--input",        type=Path, required=True, help="CSV from previous step (with extracted_json column)")
    p.add_argument("--members-json", type=Path, required=True, help="Members metadata JSON from step 00")
    p.add_argument("--output",       type=Path, required=True, help="Destination CSV path")
    p.add_argument("--api-key",      default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    p.add_argument("--batch-size",   type=int, default=100, help="Rows per checkpoint")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key required via --api-key or OPENAI_API_KEY")

    run_openweb(
        in_csv=args.input,
        members_json=args.members_json,
        out_csv=args.output,
        api_key=args.api_key,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()
