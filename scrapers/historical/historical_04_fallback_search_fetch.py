#!/usr/bin/env python3
"""
historical_04_fallback_search_fetch.py

Wikipedia/Wikidata fallback *search* and *fetch* for historical Hansard speakers,
PLUS a final merge step that writes an enriched master table.

Pipeline
--------
1) search  : fill wikipedia_url / wikidata_url (LLM restricted to Wikipedia/Wikidata)
2) fetch   : fetch page text → wiki_text (+ matched_years_count)
3) all     : run search → fetch → MERGE back into base CSV
             and write `merged_wikipedia_full_history_enriched.csv`
             (or a path you provide via --enriched)

Typical usage
-------------
# 1) Only run URL fallback search
python historical_04_fallback_search_fetch.py search \
  --input merged_wikipedia_with_bio.csv \
  --speakers speaker_details_with_gender.json \
  --output merged_wikipedia_full_history_extracted_websearch.csv

# 2) Fetch page text for any rows that now have URLs
python historical_04_fallback_search_fetch.py fetch \
  --input  merged_wikipedia_full_history_extracted_websearch.csv \
  --output merged_wikipedia_full_history_extracted_websearch_with_urltext.csv

# 3) Do both, then merge and write the enriched master file
python historical_04_fallback_search_fetch.py all \
  --input    merged_wikipedia_with_bio.csv \
  --speakers speaker_details_with_gender.json \
  --mid      merged_wikipedia_full_history_extracted_websearch.csv \
  --output   merged_wikipedia_full_history_extracted_websearch_with_urltext.csv \
  --enriched merged_wikipedia_full_history_enriched.csv

Dependencies
------------
pip install pandas tqdm requests bs4 openai
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# OpenAI v1 SDK (preferred)
try:  # pragma: no cover
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # checked at runtime

MODEL_DEFAULT = "gpt-4o-search-preview"
USER_AGENT = "Mozilla/5.0 (compatible; hist-hansard-bot/1.0; +https://example.org)"
REQ_TIMEOUT = 15  # seconds
SAVE_EVERY = 50   # rows
SLEEP_SEC  = 1.0  # throttle between LLM calls
ENRICHED_DEFAULT = "merged_wikipedia_full_history_enriched.csv"

WIKI_HOSTS = ("wikipedia.org", "wikidata.org")

# =============================================================================
# Utilities
# =============================================================================

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
    # Ensure required columns exist (resumable/idempotent)
    for col, default in [
        ("extracted_json", None),
        ("wikipedia_url", ""),
        ("wikidata_url", ""),
        ("wiki_text", ""),
        ("matched_years_count", ""),
    ]:
        if col not in df.columns:
            df[col] = default
    return df


def _load_speakers(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("speaker JSON must be a list")
    return data


def _speaker_idx_for_row(df_row: pd.Series) -> int:
    """
    Map DataFrame row → original speaker index.

    If the CSV has an 'index' column carrying the original 0..N-1 ordering,
    use it; otherwise fall back to the row's current index.
    """
    try:
        if "index" in df_row and pd.notna(df_row["index"]):
            return int(df_row["index"])
    except Exception:
        pass
    return int(df_row.name)

# =============================================================================
# Stage 1: LLM Wikipedia/Wikidata fallback SEARCH
# =============================================================================

SEARCH_SYSTEM_PROMPT = """
You are an automated researcher.
I will send you a JSON object describing a historical UK parliament member.

Your task:
1. Search only on Wikipedia.org and Wikidata.org.
2. If you find a dedicated personal page that clearly matches the same individual, return the URL.
3. If no confident match, output "Not found".
4. Return exactly two lines and nothing else.

The required regular-expression is:
^Wikipedia URL: (https?://[^ ]+|Not found)$
^Wikidata URL: (https?://[^ ]+|Not found)$

If you violate the regex (even one extra character), your answer will be treated as wrong.
Do not add markdown, explanations, lists, or blank lines.
""".strip()


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
    """
    Fill `wikipedia_url` / `wikidata_url` for rows with null `extracted_json`.

    Assumes row order matches the speakers JSON list; if the CSV contains an
    'index' column with original order, that is used to select the speaker.
    """
    client = _ensure_client(api_key)
    df = _load_df(input_csv)
    speakers = _load_speakers(speakers_json)

    missing_idx = df[df["extracted_json"].isnull()].index.tolist()
    if not missing_idx:
        print("No rows with null extracted_json. Nothing to search.")
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return

    print(f"Searching URLs for {len(missing_idx):,} rows …")

    for n, idx in enumerate(tqdm(missing_idx, desc="Search Wikipedia/Wikidata"), start=1):
        row = df.loc[idx]

        # Skip if already filled (resume-friendly)
        if str(row.get("wikipedia_url", "")).strip() or str(row.get("wikidata_url", "")).strip():
            continue

        sp_i = _speaker_idx_for_row(row)
        if not (0 <= sp_i < len(speakers)):
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
        df.at[idx, "wikidata_url"]  = wd_url

        if n % batch_size == 0:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"✓ progress saved ({n}/{len(missing_idx)}) → {output_csv}")
        time.sleep(sleep_sec)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("✓ search completed →", output_csv)

# =============================================================================
# Stage 2: FETCH page text + year counting
# =============================================================================

_REQ_SESSION: Optional[requests.Session] = None

def get_session() -> requests.Session:
    global _REQ_SESSION
    if _REQ_SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _REQ_SESSION = s
    return _REQ_SESSION


def fetch_soup(url: str) -> BeautifulSoup:
    r = get_session().get(url, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_wikipedia_text(soup: BeautifulSoup) -> str:
    """Return a compact 'full text' block: main content + infobox."""
    parts: List[str] = ["Main text:", ""]
    container = soup.select_one("#mw-content-text .mw-parser-output")
    if container:
        for tag in container.find_all(["p", "h2", "h3", "li", "table"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.extend([txt, ""])
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
    """Prefer LD+JSON; fallback to full page text."""
    ld = soup.find("script", {"type": "application/ld+json"})
    if ld and ld.string:
        return ld.string.strip()
    return soup.get_text(" ", strip=True)


_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")  # 1600–2099 (generous)


def _count_intro_years(text: str) -> int:
    """Wikipedia: years in intro only; Wikidata: years across the whole text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    if "Main text:" in text:
        try:
            body = text.split("Main text:", 1)[1]
            parts = [p for p in body.split("\n\n") if p.strip()]
            intro = parts[0] if parts else body
            return len(set(_YEAR_RE.findall(intro)))
        except Exception:
            return len(set(_YEAR_RE.findall(text)))
    return len(set(_YEAR_RE.findall(text)))


def run_fetch(
    input_csv: Path,
    output_csv: Path,
    batch_size: int = SAVE_EVERY,
) -> None:
    """Fetch `wiki_text` for rows that have `wikipedia_url` or `wikidata_url`."""
    df = _load_df(input_csv)

    for i, row in enumerate(tqdm(df.itertuples(index=True), total=len(df), desc="Fetch wiki text"), start=1):
        idx = row.Index
        wiki = str(getattr(row, "wikipedia_url", "") or "").strip()
        wd   = str(getattr(row, "wikidata_url",  "") or "").strip()

        # Resume-friendly: skip if wiki_text already present
        if str(df.at[idx, "wiki_text"]).strip():
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
        df.at[idx, "matched_years_count"] = _count_intro_years(text_block) if text_block else 0

        if i % batch_size == 0:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"✓ progress saved ({i}/{len(df)}) → {output_csv}")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("✓ fetch completed →", output_csv)

# =============================================================================
# Stage 3: MERGE back into the base CSV
# =============================================================================

_BLANK = re.compile(r"^\s*$", re.IGNORECASE)

def _is_blank(val: pd.Series) -> pd.Series:
    s = val.astype(str)
    return s.isna() | s.str.match(_BLANK)

def overlay_cols(base: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """
    Overlay *src* onto *base*:
    - Add any missing columns.
    - For columns present in *base*, fill blanks in *base* with non-blank values from *src*.
    """
    out = base.copy()
    for col in src.columns:
        if col not in out.columns:
            out[col] = src[col]
        else:
            mask = _is_blank(out[col]) & (~_is_blank(src[col]))
            out.loc[mask, col] = src.loc[mask, col]
    return out

def run_merge(base_csv: Path, wf_csv: Path, enriched_csv: Path) -> None:
    base = pd.read_csv(base_csv, encoding="utf-8-sig")
    wf   = pd.read_csv(wf_csv,   encoding="utf-8-sig")
    merged = overlay_cols(base, wf)
    merged.to_csv(enriched_csv, index=False, encoding="utf-8-sig")
    print("✓ enriched written →", enriched_csv)

# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wikipedia/Wikidata fallback search + fetch (historical) + merge")
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
    sp.add_argument("--batch-size", type=int, default=SAVE_EVERY)

    # merge (optional standalone)
    sp = sub.add_parser("merge", help="Overlay the fetch output back onto the base CSV")
    sp.add_argument("--base", required=True, type=Path, help="Base CSV (e.g., merged_wikipedia_with_bio.csv)")
    sp.add_argument("--wf",   required=True, type=Path, help="Fetch output CSV (with wiki_text)")
    sp.add_argument("--enriched", default=ENRICHED_DEFAULT, type=Path,
                   help=f"Path for enriched output (default: {ENRICHED_DEFAULT})")

    # all
    sp = sub.add_parser("all", help="Run search → fetch → merge")
    sp.add_argument("--input", required=True, type=Path, help="Base CSV (with extracted_json)")
    sp.add_argument("--speakers", required=True, type=Path, help="speaker_details_with_gender.json")
    sp.add_argument("--mid", required=True, type=Path, help="Intermediate CSV after search")
    sp.add_argument("--output", required=True, type=Path, help="Final CSV after fetch")
    sp.add_argument("--enriched", default=ENRICHED_DEFAULT, type=Path,
                   help=f"Path for enriched output (default: {ENRICHED_DEFAULT})")
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
            batch_size=args.batch_size,
        )
    elif args.cmd == "merge":
        run_merge(
            base_csv=args.base,
            wf_csv=args.wf,
            enriched_csv=args.enriched,
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
            batch_size=args.batch_size,
        )
        run_merge(
            base_csv=args.input,
            wf_csv=args.output,
            enriched_csv=args.enriched,
        )
    else:
        parser.error("unknown command")


if __name__ == "__main__":
    main()
