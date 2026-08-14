#!/usr/bin/env python3
"""
modern_02_download_wiki_texts.py

Download and flatten up to **two** Wikipedia articles for each modern member row.

Input (from modern_01_collect_wikipedia_candidates.py)
------------------------------------------------------
CSV with columns including:
- wikipedia_links       : JSON-encoded list (<= 2) of Wikipedia URLs
- wikipedia_full_texts  : (optional) JSON-encoded list of article texts if
                          modern_01 was run without --no-download

Output
------
Same CSV schema plus two new columns:
- wikipedia_text_1 : full text for the first URL (or empty)
- wikipedia_text_2 : full text for the second URL (or empty)

Notes
-----
- If wikipedia_full_texts is already present (from modern_01 without --no-download),
  this script copies those values into wikipedia_text_1/2 and skips refetching.
- Resumable: rows that already have both wikipedia_text_1 and wikipedia_text_2 are skipped.
- Checkpointing: the output CSV is written every --batch-size rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- Configuration (CLI-overridable) ----------------
DEFAULT_INPUT  = "modern_01_wikipedia_candidates.csv"          # default output of modern_01
DEFAULT_OUTPUT = "modern_02_wikipedia_candidates_with_text.csv"
DEFAULT_BATCH  = 100
TIMEOUT        = 60     # seconds
SLEEP_SEC      = 0.3    # polite delay between requests
USER_AGENT     = "ModernHansardBot/1.0 (+https://example.com)"

TXT_COL_1 = "wikipedia_text_1"
TXT_COL_2 = "wikipedia_text_2"
LINKS_COL = "wikipedia_links"
PRELOAD_COL = "wikipedia_full_texts"


# ---------------- HTTP session with retries ----------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = make_session()


# ---------------- Helpers ----------------
def parse_json_list(cell) -> List[str]:
    """Parse a JSON-encoded list; return [] if invalid/blank."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        data = json.loads(cell)
        return data if isinstance(data, list) else []
    except Exception:
        # Fallback for legacy Python-literal list strings
        try:
            import ast
            data = ast.literal_eval(cell)
            return data if isinstance(data, list) else []
        except Exception:
            return []


def fetch_soup(url: str) -> BeautifulSoup:
    r = _SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_full_text(soup: BeautifulSoup) -> str:
    """Flatten intro, headings, paragraphs, lists, tables, infobox into one text block."""
    parts: List[str] = ["Main text:", ""]
    main = soup.select_one("#mw-content-text .mw-parser-output")
    if main:
        for tag in main.find_all(["p", "h2", "h3", "li", "table"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.extend([txt, ""])

    parts.extend(["Infobox:", ""])
    box = soup.find(class_="infobox")
    if box:
        for row in box.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if th and td:
                k = th.get_text(" ", strip=True)
                v = td.get_text(" ", strip=True)
                if k and v:
                    parts.append(f"{k}: {v}")

    return "\n".join(parts).strip()


def maybe_copy_preloaded_texts(row: pd.Series) -> bool:
    """
    If the CSV already has wikipedia_full_texts (from modern_01 without --no-download),
    copy those texts into wikipedia_text_1/2. Returns True if values were copied.
    """
    if PRELOAD_COL not in row or not isinstance(row[PRELOAD_COL], str):
        return False
    texts = parse_json_list(row[PRELOAD_COL])
    if not texts:
        return False

    t1 = texts[0] if len(texts) >= 1 and isinstance(texts[0], str) else ""
    t2 = texts[1] if len(texts) >= 2 and isinstance(texts[1], str) else ""
    row[TXT_COL_1] = t1
    row[TXT_COL_2] = t2
    return bool(t1 or t2)


# ---------------- Core ----------------
def process_csv(input_path: Path, output_path: Path, batch_size: int) -> None:
    logging.info("Loading %s", input_path)
    df = pd.read_csv(input_path, dtype=str)

    if LINKS_COL not in df.columns:
        raise SystemExit(f"Input CSV must contain column '{LINKS_COL}'.")

    # Ensure output columns exist (resumable)
    for col in (TXT_COL_1, TXT_COL_2):
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    for idx in df.index:
        # Skip row if already filled (resume support)
        if str(df.at[idx, TXT_COL_1]).strip() and str(df.at[idx, TXT_COL_2]).strip():
            continue

        row = df.loc[idx].copy()

        # If modern_01 already downloaded full texts, copy them over
        if maybe_copy_preloaded_texts(row):
            df.loc[idx, [TXT_COL_1, TXT_COL_2]] = row[[TXT_COL_1, TXT_COL_2]]
        else:
            # Otherwise, fetch from wikipedia_links
            links = parse_json_list(df.at[idx, LINKS_COL])
            texts = ["", ""]
            for i in range(min(2, len(links))):
                url = links[i]
                try:
                    texts[i] = extract_full_text(fetch_soup(url))
                except Exception as e:
                    logging.warning("[%d] failed to fetch %s: %s", idx, url, e)
                time.sleep(SLEEP_SEC)
            df.at[idx, TXT_COL_1] = texts[0]
            df.at[idx, TXT_COL_2] = texts[1]

        if (idx + 1) % batch_size == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d rows", idx + 1, total)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("Done → %s", output_path)


# ---------------- CLI ----------------
def parse_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download & flatten Wikipedia articles for modern Hansard candidates."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Input CSV (output of modern_01_collect_wikipedia_candidates.py)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="Rows between checkpoints")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging verbosity")
    return p


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    process_csv(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
