#!/usr/bin/env python3
"""
modern_05_fallback_search_fetch.py

Fallback search for modern (post-2005) speakers whose initial Wikipedia pages
were NOT verified in step 03. Uses GPT-4o-search-preview to find a dedicated
Wikipedia or Wikidata page, downloads the page text, and writes it back to the
verification CSV so you can re-run step 03 (and then step 04).

Inputs
------
--input        : CSV from step 03 (modern_03_verify_pages.py)
--members-json : JSON from step 00 (modern_00_fetch_members_api.py), e.g.
                 modern_members_full_responses.json

What this writes
----------------
- wikipedia_url       : URL or "Not found" (from GPT)
- wikidata_url        : URL or "Not found" (from GPT)
- wikipedia_links     : JSON list (if Wikipedia URL was found, it is at index 0)
- wikipedia_text_1    : Flattened Wikipedia text for index 0 (if fetched)
- wiki_text           : Flattened text for the found page (Wikipedia **or** Wikidata)
                        (kept to support step 04 which can consume `wiki_text`)
- Checkpoint saves every N rows (resumable)

Typical next steps
------------------
1) Re-run step 03 (modern_03_verify_pages.py) on the CSV this script produced.
2) Then run step 04 (modern_04_extract_bio_json.py) for rows with final_reply == "yes".
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
from typing import Any, Iterable, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# OpenAI v1 SDK
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# GPT prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an automated researcher.
I will send you a JSON object describing a modern UK parliamentarian or peer.

Your task:
1) Search only on Wikipedia.org and Wikidata.org.
2) If you find a dedicated personal page that clearly matches the same individual, return the URL.
3) If no confident match, output "Not found".
4) Return exactly two lines and nothing else.

The required regular-expression is:
^Wikipedia URL: (https?://[^ ]+|Not found)$
^Wikidata URL: (https?://[^ ]+|Not found)$

If you violate the regex (even one extra character), your answer will be treated as wrong.
Do not add markdown, explanations, lists, or blank lines.
"""

WIKI_RE = re.compile(r"^Wikipedia URL: (.+)$", re.I)
WD_RE   = re.compile(r"^Wikidata URL: (.+)$", re.I)

# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ModernHansardBot/1.0 (+https://example.com)"})
TIMEOUT = 25  # seconds

def fetch_soup(url: str) -> BeautifulSoup:
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

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

# ──────────────────────────────────────────────────────────────────────────────
# Members JSON helpers (robust to multiple shapes)
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

# ──────────────────────────────────────────────────────────────────────────────
# GPT wrapper
# ──────────────────────────────────────────────────────────────────────────────
def search_wiki_wikidata(client: OpenAI, profile_json: str, retries: int = 2) -> Tuple[str, str]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-search-preview",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": profile_json},
                ],
                temperature=0,
            )
            reply = (resp.choices[0].message.content or "").strip()
            wiki_url, wd_url = "Not found", "Not found"
            for line in reply.splitlines():
                m1 = WIKI_RE.match(line)
                m2 = WD_RE.match(line)
                if m1:
                    wiki_url = m1.group(1).strip()
                if m2:
                    wd_url = m2.group(1).strip()
            return wiki_url, wd_url
        except Exception as exc:  # keep going on RL/transients
            msg = str(exc).lower()
            if ("rate limit" in msg or "429" in msg) and attempt < retries:
                logging.warning("Rate-limited; sleeping 20s (attempt %d/%d)…", attempt, retries)
                time.sleep(20)
                continue
            logging.error("OpenAI error: %s", exc)
            break
    return "Not found", "Not found"

# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────
def process_fallback(
    df: pd.DataFrame,
    members_list: List[dict],
    client: OpenAI,
    batch_size: int,
    out_path: Path,
) -> None:
    # Ensure required columns exist
    for col in ("wikipedia_url", "wikidata_url", "wiki_text",
                "wikipedia_links", "wikipedia_text_1"):
        if col not in df.columns:
            df[col] = ""

    # Make replies lowercase strings (avoid .str errors on NaN)
    for col in ("gpt_reply_1", "gpt_reply_2"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    # Select rows needing fallback: neither reply is "yes"
    mask = (~df["gpt_reply_1"].str.lower().eq("yes")) & (~df["gpt_reply_2"].str.lower().eq("yes"))
    idxs = df.index[mask].tolist()
    logging.info("Fallback search needed for %d rows", len(idxs))

    for count, idx in enumerate(tqdm(idxs, desc="Fallback search"), start=1):
        # Build a compact member profile JSON for GPT
        if idx < len(members_list):
            profile_obj = members_list[idx]
        else:
            profile_obj = {}
        profile_json = json.dumps(profile_obj, ensure_ascii=False)

        wiki_url, wd_url = search_wiki_wikidata(client, profile_json)
        df.at[idx, "wikipedia_url"] = wiki_url
        df.at[idx, "wikidata_url"] = wd_url

        chosen_url = ""
        if isinstance(wiki_url, str) and wiki_url.lower().startswith("http"):
            chosen_url = wiki_url
        elif isinstance(wd_url, str) and wd_url.lower().startswith("http"):
            chosen_url = wd_url

        text = ""
        wiki_text_1 = ""
        wiki_links_json = "[]"

        if chosen_url:
            try:
                soup = fetch_soup(chosen_url)
                if "wikipedia.org" in chosen_url:
                    text = extract_wikipedia_text(soup)
                    wiki_text_1 = text
                    wiki_links_json = json.dumps([chosen_url], ensure_ascii=False)
                else:
                    # Wikidata (store text into wiki_text for step 04)
                    text = extract_wikidata_text(soup)
            except Exception as exc:
                logging.warning("[%d] Failed to fetch %s: %s", idx, chosen_url, exc)

        # Write columns (keep both the generic wiki_text and the step-03 friendly fields)
        df.at[idx, "wiki_text"] = text
        df.at[idx, "wikipedia_text_1"] = wiki_text_1
        df.at[idx, "wikipedia_links"] = wiki_links_json

        # Checkpoint
        if (count % batch_size == 0) or (count == len(idxs)):
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d → %s", count, len(idxs), out_path)

    logging.info("Fallback search complete → %s", out_path)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fallback Wikipedia/Wikidata search + text fetch for unmatched speakers.")
    p.add_argument("--input",        type=Path, required=True, help="CSV from step 03 (modern_03_verify_pages.py)")
    p.add_argument("--members-json", type=Path, required=True, help="Members JSON from step 00")
    p.add_argument("--output",       type=Path, required=True, help="Destination CSV path")
    p.add_argument("--api-key",      default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    p.add_argument("--batch-size",   type=int, default=50, help="Rows per checkpoint (default 50)")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key required via --api-key or OPENAI_API_KEY")

    client = OpenAI(api_key=args.api_key)

    # Load CSV and members JSON
    df = pd.read_csv(args.input, dtype=str)
    raw_members = json.loads(Path(args.members_json).read_text(encoding="utf-8"))
    members_list = list(iter_members(raw_members))

    # Normalise existing wikipedia_links to JSON lists in memory if present
    if "wikipedia_links" in df.columns:
        def _ensure_list(x):
            if isinstance(x, str) and x.strip():
                try:
                    return json.loads(x)
                except Exception:
                    try:
                        return ast.literal_eval(x)
                    except Exception:
                        return []
            return x if isinstance(x, list) else []
        df["wikipedia_links"] = df["wikipedia_links"].apply(_ensure_list)

    process_fallback(df, members_list, client, args.batch_size, args.output)

if __name__ == "__main__":
    main()
