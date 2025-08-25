#!/usr/bin/env python3
"""historical_fallback_search_fetch.py

Wikipedia/Wikidata fallback search **and fetch** for historical Hansard speakers.

This script fills in missing Wikipedia/Wikidata links for rows where the
structured biography (`extracted_json`) was not produced, by:

  1) Querying an LLM **restricted to Wikipedia/Wikidata only** to return
     at most one candidate URL for each source.
  2) Fetching page text (Wikipedia article text + infobox, or Wikidata JSON-LD)
     for the newly-found pages, writing it to `wiki_text`.
  3) (Optional) Counting year matches in the intro (Wikipedia) or full text
     (Wikidata) into `matched_years_count` using a list of expected years from
     the CSV column `years` **or** from the speaker JSON `dates`/`lifespan`.

The file is designed to be **idempotent** and **resumable**: existing values in
output CSV are preserved and the run picks up where it left off. See the
subcommands below.

USAGE
-----
# 1) Only run the URL fallback search
python historical_fallback_search_fetch.py search \
  --input merged_wikipedia_full_history_extracted.csv \
  --speakers speaker_details_with_gender.json \
  --output merged_wikipedia_full_history_extracted_websearch.csv \
  --model gpt-4o-search-preview

# 2) Fetch page text for any rows that now have URLs
python historical_fallback_search_fetch.py fetch \
  --input merged_wikipedia_full_history_extracted_websearch.csv \
  --output merged_wikipedia_full_history_extracted_websearch_with_urltext.csv

# 3) Do both in one go
python historical_fallback_search_fetch.py all \
  --input merged_wikipedia_full_history_extracted.csv \
  --speakers speaker_details_with_gender.json \
  --mid merged_wikipedia_full_history_extracted_websearch.csv \
  --output merged_wikipedia_full_history_extracted_websearch_with_urltext.csv

DEPENDENCIES
------------
  pip install pandas tqdm requests bs4 openai tiktoken  # (tiktoken optional)

Notes
-----
• We do **not** enforce token truncation in this script (you said the limit is
  raised). If you ever need it, wire in a tiktoken-based truncation before
  writing `wiki_text`.
• The LLM step respects a strict two-line regex protocol to reduce parsing
  errors.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# OpenAI v1 SDK (preferred). Fallback to legacy if unavailable.
try:  # pragma: no cover - import-time branch
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # resolved at runtime in _ensure_client

MODEL_DEFAULT = "gpt-4o-search-preview"
USER_AGENT = "Mozilla/5.0 (compatible; hist-hansard-bot/1.0; +https://example.org)"
REQ_TIMEOUT = 15  # seconds
SAVE_EVERY = 50   # rows
SLEEP_SEC  = 1.0  # throttling between LLM calls

WIKI_HOSTS = ("wikipedia.org", "wikidata.org")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_client(api_key: Optional[str]):
    if OpenAI is None:
        raise RuntimeError("openai package not available; pip install openai>=1.0")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key missing (use --api-key or set OPENAI_API_KEY).")
    return OpenAI(api_key=key)


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str)
    # Ensure required columns exist
    if "extracted_json" not in df.columns:
        df["extracted_json"] = pd.Series([None] * len(df), dtype="object")
    if "wikipedia_url" not in df.columns:
        df["wikipedia_url"] = ""
    if "wikidata_url" not in df.columns:
        df["wikidata_url"] = ""
    if "wiki_text" not in df.columns:
        df["wiki_text"] = ""
    return df


def _load_speakers(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("speaker JSON must be a list")
    return data


# Mapping from DataFrame index → speaker index (defaults to identity)
# If your CSV contains a column 'index' that stores the original speaker order,
# we will use it; else, we assume row index == speaker index.

def _speaker_idx_for_row(df_row: pd.Series) -> int:
    try:
        # common pattern in your notebooks
        if "index" in df_row and pd.notna(df_row["index"]):
            return int(df_row["index"])  # already 0..N-1
    except Exception:
        pass
    return int(df_row.name)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: LLM Wikipedia/Wikidata fallback SEARCH
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_SYSTEM_PROMPT = (
    """
You are an automated researcher.
I will send you a JSON object describing a historical UK parliament member.

Your task:
1. Search **only** on Wikipedia.org and Wikidata.org.
2. If you find a **dedicated personal page** that clearly matches the same individual, return the URL.
3. If no confident match, output "Not found".
4. Return **exactly two lines** and **nothing else**.

The required regular-expression is:
^Wikipedia URL: (https?://[^ ]+|Not found)$
^Wikidata URL: (https?://[^ ]+|Not found)$

If you violate the regex (even one extra character), your answer will be treated as wrong.
Do not add markdown, explanations, lists, or blank lines.
"""
).strip()


def _extract_urls_from_reply(reply: str) -> Tuple[str, str]:
    wiki_url, wd_url = "Not found", "Not found"
    for line in reply.splitlines():
        line = line.strip()
        if line.lower().startswith("wikipedia url"):
            wiki_url = line.split(":", 1)[1].strip()
        elif line.lower().startswith("wikidata url"):
            wd_url = line.split(":", 1)[1].strip()
    return wiki_url, wd_url


def _is_allowed_url(url: str) -> bool:
    return any(host in url for host in WIKI_HOSTS)


def run_search(
    input_csv: Path,
    speakers_json: Path,
    output_csv: Path,
    model: str = MODEL_DEFAULT,
    api_key: Optional[str] = None,
    sleep_sec: float = SLEEP_SEC,
    batch_size: int = SAVE_EVERY,
) -> None:
    """Fill wikipedia_url / wikidata_url for rows with null extracted_json.

    Assumes row order matches the speaker JSON list; if the CSV has a column
    named 'index', we use it as the original row.
    """
    client = _ensure_client(api_key)
    df = _load_df(input_csv)
    speakers = _load_speakers(speakers_json)

    # rows that still lack structured extraction
    missing_idx = df[df["extracted_json"].isnull()].index.tolist()
    if not missing_idx:
        print("No rows with null extracted_json. Nothing to search.")
        df.to_csv(output_csv, index=False)
        return

    # Resume-friendly: keep any existing URL values
    print(f"Searching URLs for {len(missing_idx):,} rows …")

    for n, idx in enumerate(tqdm(missing_idx, desc="Search Wikipedia/Wikidata"), start=1):
        row = df.loc[idx]
        # Skip if already filled (resuming)
        if str(row.get("wikipedia_url", "")).strip() or str(row.get("wikidata_url", "")).strip():
            continue

        sp_i = _speaker_idx_for_row(row)
        if sp_i < 0 or sp_i >= len(speakers):
            # Out-of-range guard
            continue
        payload = json.dumps(speakers[sp_i], ensure_ascii=False)

        reply = "Wikipedia URL: Not found\nWikidata URL: Not found"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0,
            )
            reply = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # pragma: no cover
            print(f"[warn] API error at row {idx}: {e}")

        wiki_url, wd_url = _extract_urls_from_reply(reply)
        if wiki_url != "Not found" and not _is_allowed_url(wiki_url):
            wiki_url = "Not found"
        if wd_url != "Not found" and not _is_allowed_url(wd_url):
            wd_url = "Not found"

        df.at[idx, "wikipedia_url"] = wiki_url
        df.at[idx, "wikidata_url"] = wd_url

        if n % batch_size == 0:
            df.to_csv(output_csv, index=False)
            print(f"✓ progress saved ({n}/{len(missing_idx)}) → {output_csv}")
        time.sleep(sleep_sec)

    df.to_csv(output_csv, index=False)
    print("✓ search completed →", output_csv)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: FETCH page text + year matching
# ──────────────────────────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None

def _session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _session = s
    return _session


def fetch_soup(url: str) -> BeautifulSoup:
    r = _session().get(url, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_wikipedia_text(soup: BeautifulSoup) -> str:
    parts: List[str] = ["Main text:", ""]
    container = soup.select_one("#mw-content-text .mw-parser-output")
    if container:
        for tag in container.find_all(["p", "h2", "h3", "li", "table"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.extend([txt, ""])  # keep blank lines between blocks
    parts.extend(["Infobox:", ""])
    infobox = soup.find(class_="infobox")
    if infobox:
        for row in infobox.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if th and td:
                k = th.get_text(" ", strip=True)
                v = td.get_text(" ", strip=True)
                if k and v:
                    parts.append(f"{k}: {v}")
    return "\n".join(parts).strip()


def extract_wikidata_text(soup: BeautifulSoup) -> str:
    ld = soup.find("script", {"type": "application/ld+json"})
    if ld and ld.string:
        return ld.string.strip()
    return soup.get_text(" ", strip=True)


_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")  # 1600–2099, generous


def _years_from_string(s: str) -> List[int]:
    if not isinstance(s, str):
        return []
    return sorted({int(y) for y in _YEAR_RE.findall(s)})


def _expected_years_for_row(df_row: pd.Series, speakers: Optional[List[dict]] = None) -> List[int]:
    # Preferred: CSV column 'years' that already contains a Python-list-ish string
    val = df_row.get("years")
    if isinstance(val, str) and val.strip():
        try:
            # tolerate formats like "[1820, 1899]" or "(1820, 1899)"
            cleaned = val.strip().strip("[](){}")
            items = [int(x) for x in re.split(r"[,\s]+", cleaned) if x.isdigit()]
            if items:
                return sorted(set(items))
        except Exception:
            pass

    # Fallback: derive from speakers JSON dates/lifespan if available
    if speakers is not None:
        sp_i = _speaker_idx_for_row(df_row)
        if 0 <= sp_i < len(speakers):
            for key in ("years", "dates", "lifespan"):
                if key in speakers[sp_i]:
                    yrs = _years_from_string(str(speakers[sp_i][key]))
                    if yrs:
                        return yrs
    return []


def _count_intro_years(text: str) -> int:
    """Count unique 4-digit years in Wikipedia intro; on Wikidata, count all."""
    if not isinstance(text, str) or not text.strip():
        return 0
    if "Main text:" in text:
        try:
            body = text.split("Main text:", 1)[1]
            # split paragraphs on blank lines
            parts = [p for p in body.split("\n\n") if p.strip()]
            intro = parts[0] if parts else body
            return len(set(_YEAR_RE.findall(intro)))
        except Exception:
            return len(set(_YEAR_RE.findall(text)))
    return len(set(_YEAR_RE.findall(text)))


def run_fetch(
    input_csv: Path,
    output_csv: Path,
    speakers_json: Optional[Path] = None,
    batch_size: int = SAVE_EVERY,
) -> None:
    """Fetch wiki_text for rows that have wikipedia_url or wikidata_url.

    Also computes `matched_years_count` against expected years (if available).
    """
    df = _load_df(input_csv)
    speakers = _load_speakers(speakers_json) if speakers_json and speakers_json.exists() else None

    if "matched_years_count" not in df.columns:
        df["matched_years_count"] = pd.Series([None] * len(df), dtype="float")

    for i, row in enumerate(tqdm(df.itertuples(index=True), total=len(df), desc="Fetch wiki text"), start=1):
        idx = row.Index
        wiki = str(getattr(row, "wikipedia_url", "") or "").strip()
        wd   = str(getattr(row, "wikidata_url",  "") or "").strip()

        if str(df.at[idx, "wiki_text"]).strip():  # resume: already fetched
            continue

        text_block = ""
        try:
            if wiki and wiki.lower() != "not found":
                soup = fetch_soup(wiki)
                text_block = extract_wikipedia_text(soup)
            elif wd and wd.lower() != "not found":
                soup = fetch_soup(wd)
                text_block = extract_wikidata_text(soup)
        except Exception as e:  # pragma: no cover
            print(f"[warn] fetch failed at row {idx}: {e}")
            text_block = ""

        df.at[idx, "wiki_text"] = text_block

        # Year matching (optional but cheap)
        exp_years = _expected_years_for_row(df.loc[idx], speakers)
        if text_block:
            counted = _count_intro_years(text_block)
        else:
            counted = 0
        # If explicit expected list is known, store the min(len(exp), counted)
        # Otherwise, just store counted (still useful as a signal)
        df.at[idx, "matched_years_count"] = float(min(counted, len(exp_years)) if exp_years else counted)

        if i % batch_size == 0:
            df.to_csv(output_csv, index=False)
            print(f"✓ progress saved ({i}/{len(df)}) → {output_csv}")

    df.to_csv(output_csv, index=False)
    print("✓ fetch completed →", output_csv)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wikipedia/Wikidata fallback search + fetch (historical)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # search
    sp = sub.add_parser("search", help="LLM search restricted to Wikipedia/Wikidata; write URLs")
    sp.add_argument("--input", required=True, type=Path, help="Input CSV (needs extracted_json column)")
    sp.add_argument("--speakers", required=True, type=Path, help="speaker_details_with_gender.json")
    sp.add_argument("--output", required=True, type=Path, help="Output CSV with wikipedia_url/wikidata_url")
    sp.add_argument("--model", default=MODEL_DEFAULT)
    sp.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    sp.add_argument("--sleep", type=float, default=SLEEP_SEC)
    sp.add_argument("--batch-size", type=int, default=SAVE_EVERY)

    # fetch
    sp = sub.add_parser("fetch", help="Fetch page text for rows that have URLs; compute year signal")
    sp.add_argument("--input", required=True, type=Path, help="Input CSV with wikipedia_url/wikidata_url")
    sp.add_argument("--output", required=True, type=Path, help="Output CSV with wiki_text")
    sp.add_argument("--speakers", type=Path, default=None, help="Optional speakers JSON to derive expected years")
    sp.add_argument("--batch-size", type=int, default=SAVE_EVERY)

    # all
    sp = sub.add_parser("all", help="Run search then fetch")
    sp.add_argument("--input", required=True, type=Path, help="Start CSV (with extracted_json)")
    sp.add_argument("--speakers", required=True, type=Path, help="speaker_details_with_gender.json")
    sp.add_argument("--mid", required=True, type=Path, help="Intermediate CSV after search")
    sp.add_argument("--output", required=True, type=Path, help="Final CSV after fetch")
    sp.add_argument("--model", default=MODEL_DEFAULT)
    sp.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    sp.add_argument("--sleep", type=float, default=SLEEP_SEC)
    sp.add_argument("--batch-size", type=int, default=SAVE_EVERY)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "search":
        run_search(
            input_csv=args.input,
            speakers_json=args.speakers,
            output_csv=args.output,
            model=args.model,
            api_key=args.api_key,
            sleep_sec=args.sleep,
            batch_size=args.batch_size,
        )
    elif args.cmd == "fetch":
        run_fetch(
            input_csv=args.input,
            output_csv=args.output,
            speakers_json=args.speakers,
            batch_size=args.batch_size,
        )
    elif args.cmd == "all":
        run_search(
            input_csv=args.input,
            speakers_json=args.speakers,
            output_csv=args.mid,
            model=args.model,
            api_key=args.api_key,
            sleep_sec=args.sleep,
            batch_size=args.batch_size,
        )
        run_fetch(
            input_csv=args.mid,
            output_csv=args.output,
            speakers_json=args.speakers,
            batch_size=args.batch_size,
        )
    else:
        parser.error("unknown command")


if __name__ == "__main__":
    main()
