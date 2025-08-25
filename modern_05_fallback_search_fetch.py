#!/usr/bin/env python3
"""modern_fallback_search_fetch.py

For speakers whose initial Wikipedia pages were *not* verified,
probe GPT‑4o‑search‑preview to locate a dedicated Wikipedia **or** Wikidata page
and immediately download its full text.

After this script you can simply re‑run `modern_verify_pages.py` on the produced
CSV, followed by `modern_extract_bio_json.py` for any new “yes” pages.

Workflow
--------
1. Load the verification CSV produced earlier (needs `gpt_reply_1`, `gpt_reply_2`).
2. Identify rows where neither reply == "yes".
3. For each, send the speaker JSON profile to GPT with a constrained two‑line
   regex output. Parse out `wikipedia_url`, `wikidata_url`.
4. Fetch the page; if it is Wikipedia, flatten text via the same logic as the
   downloader; if Wikidata, keep the `<script type='application/ld+json'>` or
   raw text.
5. Store `wiki_text` so subsequent verification can run.

Columns added / touched
-----------------------
* `wikipedia_url`   – URL or "Not found"
* `wikidata_url`    – URL or "Not found"
* `wiki_text`       – downloaded article text (empty if fetch failed)
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# GPT prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an automated researcher.  \
I will send you a JSON object describing a modern UK parliamentarian or peer.

Your task:
1. Search **only** on Wikipedia.org and Wikidata.org.  \
2. If you find a **dedicated personal page** that clearly matches the *same individual*, return the URL.  \
3. If no confident match, output “Not found”.  \
4. Return **exactly two lines** and **nothing else**.

The required regular-expression is:
^Wikipedia URL: (https?://[^ ]+|Not found)$
^Wikidata URL: (https?://[^ ]+|Not found)$

If you violate the regex (even one extra character),
your answer will be treated as wrong. Do not add markdown,
explanations, lists, or blank lines."""

WIKI_RE = re.compile(r"^Wikipedia URL: (.+)$", re.I)
WD_RE   = re.compile(r"^Wikidata URL: (.+)$", re.I)

# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ModernHansardBot/1.0 (+https://example.com)"})
TIMEOUT = 25  # seconds


def fetch_soup(url: str) -> BeautifulSoup:
    resp = SESSION.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_wikipedia_text(soup: BeautifulSoup) -> str:
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


def extract_wikidata_text(soup: BeautifulSoup) -> str:
    ld = soup.find("script", {"type": "application/ld+json"})
    if ld and ld.string:
        return ld.string.strip()
    return soup.get_text(" ", strip=True)


# ---------------------------------------------------------------------------
# GPT wrapper
# ---------------------------------------------------------------------------

def search_wiki_wikidata(client: openai.OpenAI, profile_json: str, retries: int = 2) -> Tuple[str, str]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-search-preview",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": profile_json},
                ],
            )
            reply = resp.choices[0].message.content.strip()
            wiki_url, wd_url = "Not found", "Not found"
            for line in reply.splitlines():
                m1 = WIKI_RE.match(line)
                m2 = WD_RE.match(line)
                if m1:
                    wiki_url = m1.group(1).strip()
                if m2:
                    wd_url = m2.group(1).strip()
            return wiki_url, wd_url
        except Exception as exc:  # pylint: disable=broad-except
            if "rate limit" in str(exc).lower() and attempt < retries:
                logging.warning("Rate‑limit; sleeping 20 s (attempt %d)…", attempt)
                time.sleep(20)
                continue
            logging.error("OpenAI error: %s", exc)
            break
    return "Not found", "Not found"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def process_fallback(
    df: pd.DataFrame,
    members: List[dict],
    client: openai.OpenAI,
    batch_size: int,
    out_path: Path,
) -> None:
    # Ensure required columns exist
    for col in ("wikipedia_url", "wikidata_url", "wiki_text"):
        if col not in df.columns:
            df[col] = ""

    # Mask: neither reply == "yes"
    mask = (~df["gpt_reply_1"].str.lower().eq("yes")) & (~df["gpt_reply_2"].str.lower().eq("yes"))
    idxs = df.index[mask].tolist()
    logging.info("Need fallback search for %d speakers", len(idxs))

    for count, idx in enumerate(tqdm(idxs, desc="Fallback search"), start=1):
        profile_json = json.dumps(members[idx]["value"], ensure_ascii=False)
        wiki_url, wd_url = search_wiki_wikidata(client, profile_json)
        df.at[idx, "wikipedia_url"] = wiki_url
        df.at[idx, "wikidata_url"] = wd_url

        # pick whichever is a real URL first
        url = wiki_url if wiki_url.startswith("http") else wd_url if wd_url.startswith("http") else ""
        text = ""
        if url:
            try:
                soup = fetch_soup(url)
                text = extract_wikipedia_text(soup) if "wikipedia.org" in url else extract_wikidata_text(soup)
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("[%d] Failed to fetch %s: %s", idx, url, exc)
        df.at[idx, "wiki_text"] = text

        if count % batch_size == 0 or count == len(idxs):
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d", count, len(idxs))

    logging.info("Fallback search complete – results in %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fallback Wikipedia/Wikidata search + text fetch for unmatched speakers.")
    parser.add_argument("--input",        type=Path, required=True, help="CSV from previous verify step")
    parser.add_argument("--members-json", type=Path, required=True, help="Members metadata JSON")
    parser.add_argument("--output",       type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--api-key",      default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--batch-size",   type=int, default=50, help="Rows before checkpoint save (default 50)")
    parser.add_argument("--log-level",    default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key required via --api-key or OPENAI_API_KEY env var")

    client = openai.OpenAI(api_key=args.api_key)

    df = pd.read_csv(args.input, dtype=str)
    members = json.loads(Path(args.members_json).read_text(encoding="utf-8"))

    # ensure wikipedia_links col is list for compatibility downstream
    if "wikipedia_links" in df.columns:
        df["wikipedia_links"] = df["wikipedia_links"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    process_fallback(df, members, client, args.batch_size, args.output)


if __name__ == "__main__":
    main()
