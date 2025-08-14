#!/usr/bin/env python3
"""Download Wikipedia article text for modern Hansard links (simple version).

Reads `merged_wikipedia_modern.csv`, fetches up to two Wikipedia URLs per row,
extracts full article text (paragraphs, headings, lists, tables, infobox) and
writes two new columns `wikipedia_text_1`, `wikipedia_text_2` to
`merged_wikipedia_modern_with_text.csv`.
"""

import ast
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ==== 0. CONFIG =============================================================
INPUT_CSV  = "merged_wikipedia_modern.csv"
OUTPUT_CSV = "merged_wikipedia_modern_with_text.csv"
BATCH_SIZE = 100  # save checkpoint every N rows
TIMEOUT    = 60   # seconds

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; my-wiki-bot/1.0)"
})

# ==== 1. Helpers ============================================================

def fetch_wikipedia_soup(url: str) -> BeautifulSoup:
    """Download the page and return BeautifulSoup (raises on HTTP errors)."""
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_full_text(soup: BeautifulSoup) -> str:
    """Flatten intro, main text, tables and infobox into one newline‑delimited string."""
    parts = ["Main text:", ""]
    container = soup.select_one("#mw-content-text .mw-parser-output")
    if container:
        for tag in container.find_all(["p", "h2", "h3", "li", "table"]):
            text = tag.get_text(" ", strip=True)
            if text:
                parts.extend([text, ""])
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

# ==== 2. Load & initialise ==================================================

df = pd.read_csv(INPUT_CSV, dtype=str)
# convert stringified lists → list objects
df["wikipedia_links"] = df["wikipedia_links"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

df["wikipedia_text_1"] = ""
df["wikipedia_text_2"] = ""

# ==== 3. Iterate & checkpoint ==============================================

for idx in df.index:
    links = df.at[idx, "wikipedia_links"]
    text1, text2 = "", ""

    if len(links) >= 1:
        try:
            text1 = extract_full_text(fetch_wikipedia_soup(links[0]))
        except Exception as e:  # pylint: disable=broad-except
            print(f"[{idx}] first link failed: {e}")

    if len(links) >= 2:
        try:
            text2 = extract_full_text(fetch_wikipedia_soup(links[1]))
        except Exception as e:  # pylint: disable=broad-except
            print(f"[{idx}] second link failed: {e}")

    df.at[idx, "wikipedia_text_1"] = text1
    df.at[idx, "wikipedia_text_2"] = text2
#!/usr/bin/env python3
"""modern_download_wiki_texts.py
================================
Download and flatten up to **two** Wikipedia articles for each speaker record.

The script expects an input CSV created by *modern_hansard_wikipedia_collector.py*
with a column **`wikipedia_links`** (a JSON-encoded list of ≤ 2 URLs).
It appends two new columns:

* ``wikipedia_text_1`` – full text for the first URL (or empty).
* ``wikipedia_text_2`` – full text for the second URL (or empty).

Features
--------
* **Resumable** – already-filled rows are skipped; checkpoints written every
  ``--batch-size`` rows.
* **Polite fetching** – 0.3 s delay between requests and custom UA string.
* **Robust** – catches per-row HTTP errors without stopping the run.
* **Typed & logged** – uses ``logging`` for progress instead of ``print``.
"""
from __future__ import annotations

import argparse
import ast
import logging
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Default config (overridden by CLI)
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = "merged_wikipedia_modern.csv"
DEFAULT_OUTPUT = "merged_wikipedia_modern_with_text.csv"
DEFAULT_BATCH  = 100
TIMEOUT        = 60  # seconds
SLEEP_SEC      = 0.3  # politeness delay
USER_AGENT     = "ModernHansardBot/1.0 (+https://example.com)"

# Column names
TXT_COL_1 = "wikipedia_text_1"
TXT_COL_2 = "wikipedia_text_2"

# ---------------------------------------------------------------------------
# HTTP session (module-level reuse)
# ---------------------------------------------------------------------------
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_soup(url: str) -> BeautifulSoup:
    """Return parsed HTML for *url* (raises ``requests`` errors if any)."""
    resp = _SESSION.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_full_text(soup: BeautifulSoup) -> str:
    """Flatten intro, headings, paragraphs, lists, tables, infobox → plain text."""
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
                key = th.get_text(" ", strip=True)
                val = td.get_text(" ", strip=True)
                if key and val:
                    parts.append(f"{key}: {val}")

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def process_csv(input_path: Path, output_path: Path, batch_size: int) -> None:
    """Fill ``wikipedia_text_1/2`` columns for every row in *input_path*."""
    logging.info("Loading %s", input_path)
    df = pd.read_csv(input_path, dtype=str)

    # ensure list type in-memory
    df["wikipedia_links"] = df["wikipedia_links"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    # add empty columns if missing (makes script resumable)
    for col in (TXT_COL_1, TXT_COL_2):
        if col not in df.columns:
            df[col] = ""

    total_rows = len(df)
    for idx in df.index:
        # skip row if both texts already exist (resume support)
        if df.at[idx, TXT_COL_1] and df.at[idx, TXT_COL_2]:
            continue

        links: List[str] = df.at[idx, "wikipedia_links"]
        texts = ["", ""]

        for i in range(min(2, len(links))):
            url = links[i]
            try:
                texts[i] = extract_full_text(fetch_soup(url))
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("[%d] failed to fetch %s: %s", idx, url, exc)
            time.sleep(SLEEP_SEC)

        df.at[idx, TXT_COL_1], df.at[idx, TXT_COL_2] = texts

        if (idx + 1) % batch_size == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d rows", idx + 1, total_rows)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("Done – written %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download & flatten Wikipedia articles for modern Hansard links.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Rows processed before checkpoint save")
    parser.add_argument("--log-level", default="INFO", choices=[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    process_csv(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()

    if (idx + 1) % BATCH_SIZE == 0:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Saved first {idx + 1} rows to {OUTPUT_CSV}")

# ==== 4. Final save =========================================================

df.to_csv(OUTPUT_CSV, index=False)
