#!/usr/bin/env python3
"""
modern_04_extract_bio_json.py

Extract structured biography JSON *after* a Wikipedia page has been verified
as the correct person.

Input (recommended): the output CSV from modern_03_verify_pages.py, which
typically includes:
  - wikipedia_text_1, wikipedia_text_2
  - wikipedia_links (JSON list)
  - gpt_reply_1, gpt_reply_2
  - matched_url
  - final_reply (rows with value "yes" are treated as verified)

This script will:
  • Derive a single `wiki_text` for verified rows (or use an existing column if present)
  • Call GPT to extract a strict JSON biography
  • Save results into:
      - extracted_json       (strict JSON from the model)
      - api_response_full    (raw API response for audit)

Idempotent/resumable: skips rows where `extracted_json` is already non-blank.

Example:
  python modern_04_extract_bio_json.py \
    --input modern_03_wikipedia_verify.csv \
    --output modern_04_bio_extracted.csv \
    --api-key $OPENAI_API_KEY \
    --batch-size 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# OpenAI v1 SDK
from openai import OpenAI

# tiktoken is optional; we fall back if unavailable
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # fallback later

# ---------------------------------------------------------------------------
# Prompt (single constant)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a research assistant extracting structured metadata from Wikipedia articles about British politicians and peers.

Read the full article text below and return a single JSON object that follows exactly the schema described.
Fill fields as completely as possible; if a value is missing or cannot be inferred, return `null`.

Return only the JSON object—no extra keys, comments, or trailing text.

────────────────  SCHOOL-TYPE RULES  ────────────────
• Clarendon : one of the fixed nine “Great Public Schools”
  ["Charterhouse", "Eton College", "Harrow School", "Merchant Taylors' School",
   "Rugby School", "Shrewsbury School", "St Paul's School", "Westminster School",
   "Winchester College"]

• HMC schools: any UK independent (fee-paying) school that is/was a member
  of the Headmasters' & Headmistresses' Conference and is not in the Clarendon list.

• Other private: independent private schools *not* in Clarendon or HMC
  (including overseas private/boarding schools).

• All other: state-funded, grammar, comprehensive, foreign public schools,
  or unknown / unclassified cases.

Apply the first rule that matches; Clarendon overrides everything else.

━━━━━━━━  OUTPUT SCHEMA  ━━━━━━━━
{
  "name": "",

  "date_of_birth": "",              // Prefer YYYY-MM-DD; allow YYYY-MM or YYYY; else null
  "date_of_death": "",              // same rules as date_of_birth
  "place_of_birth": "",

  "party_affiliation": [
    { "party": "", "start_year": null, "end_year": null }
  ],

  "education": {
    "school_type": "",              // "Clarendon", "HMC schools", "Other private", "All other", or null
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

Wikipedia article text:
"""

# Heading pattern for truncation before References
REF_PATTERN = re.compile(r"(?i)\n+references\b")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_after_references(text: str) -> str:
    """Remove content after the first 'References' heading to save tokens."""
    parts = REF_PATTERN.split(text, maxsplit=1)
    return parts[0]


def _get_encoder(model_name: str):
    """
    Best-effort token encoder:
      - try model-specific encoding
      - fallback to cl100k_base
      - else return a naive byte-counter approximation (≈4 chars/token)
    """
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def token_count(text: str, enc) -> int:
    if enc is None:
        # very rough fallback: ~4 chars per token
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def ask_gpt(client: OpenAI, model: str, system_prompt: str, page_text: str, retries: int = 2) -> Optional[Tuple[str, str]]:
    """
    Send (system,user) messages. Returns (json_string, raw_api_response_json) or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": page_text},
                ],
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()
            return content, json.dumps(resp.to_dict(), ensure_ascii=False)
        except Exception as exc:  # pragma: no cover
            msg = str(exc).lower()
            if ("rate limit" in msg or "429" in msg) and attempt < retries:
                logging.warning("Rate-limited; sleeping 20s (attempt %d/%d)…", attempt, retries)
                time.sleep(20)
                continue
            logging.error("OpenAI error: %s", exc)
            return None
    return None


def _parse_json_list(cell) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        data = json.loads(cell)
        return data if isinstance(data, list) else []
    except Exception:
        try:
            import ast
            data = ast.literal_eval(cell)
            return data if isinstance(data, list) else []
        except Exception:
            return []


def pick_verified_text(row: pd.Series) -> str:
    """
    Choose the verified article text for a row from modern_03:
      - If 'wiki_text' column already exists and is non-blank → use it.
      - If final_reply == 'yes':
          • Prefer match by matched_url index, else by gpt_reply_1/2 == 'yes'.
      - Otherwise return empty string (not verified).
    """
    # use provided wiki_text if present
    if "wiki_text" in row and isinstance(row["wiki_text"], str) and row["wiki_text"].strip():
        return row["wiki_text"].strip()

    final_reply = str(row.get("final_reply", "")).strip().lower()
    if final_reply != "yes":
        return ""

    text1 = str(row.get("wikipedia_text_1", "") or "")
    text2 = str(row.get("wikipedia_text_2", "") or "")
    links = _parse_json_list(row.get("wikipedia_links", "")) or []

    matched_url = str(row.get("matched_url", "") or "").strip()
    if matched_url and matched_url in links:
        i = links.index(matched_url)
        return text1 if i == 0 else (text2 if i == 1 else "")

    # fallback to the first “yes”
    if str(row.get("gpt_reply_1", "")).strip().lower() == "yes" and text1:
        return text1
    if str(row.get("gpt_reply_2", "")).strip().lower() == "yes" and text2:
        return text2

    # last resort: whichever non-empty is available
    return text1 or text2


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def extract_bio(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    token_limit: int,
    batch_size: int,
    out_path: Path,
) -> None:
    enc = _get_encoder(model)

    total = len(df)
    for n, (row_idx, row) in enumerate(df.iterrows(), start=1):
        # Skip if already extracted
        existing = str(row.get("extracted_json", "") or "").strip()
        if existing:
            continue

        # Determine the verified wiki_text (or skip)
        wiki_text = pick_verified_text(row)
        if not wiki_text:
            continue

        # Rough token budgeting (system + user)
        combined = SYSTEM_PROMPT + "\n" + wiki_text
        if token_count(combined, enc) > token_limit:
            wiki_text = truncate_after_references(wiki_text)
            combined = SYSTEM_PROMPT + "\n" + wiki_text
            if token_count(combined, enc) > token_limit:
                logging.warning("[%d] Still too long after truncation; skipped.", row_idx)
                continue
            # persist truncated text for audit
            df.at[row_idx, "wiki_text"] = wiki_text

        result = ask_gpt(client, model, SYSTEM_PROMPT, wiki_text)
        if result:
            json_out, raw_resp = result
            df.at[row_idx, "extracted_json"] = json_out
            df.at[row_idx, "api_response_full"] = raw_resp

        if (n % batch_size) == 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d rows → %s", n, total, out_path)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info("Extraction complete – results in %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured biography JSON from verified Wikipedia texts (modern era).")
    parser.add_argument("--input",       type=Path, required=True, help="CSV from modern_03 (or compatible).")
    parser.add_argument("--output",      type=Path, required=True, help="Destination CSV path.")
    parser.add_argument("--api-key",     default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key (env fallback).")
    parser.add_argument("--model",       default="gpt-4o", help="Model name (default: gpt-4o).")
    parser.add_argument("--token-limit", type=int, default=30000, help="Max tokens (system+user).")
    parser.add_argument("--batch-size",  type=int, default=100, help="Rows processed between checkpoints.")
    parser.add_argument("--log-level",   default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key is required (via --api-key or OPENAI_API_KEY).")

    client = OpenAI(api_key=args.api_key)

    df = pd.read_csv(args.input, dtype=str)

    # Ensure output columns exist
    for col in ("extracted_json", "api_response_full"):
        if col not in df.columns:
            df[col] = ""

    # Ensure wiki_text column exists (even if we derive per-row)
    if "wiki_text" not in df.columns:
        df["wiki_text"] = ""

    extract_bio(df, client, args.model, args.token_limit, args.batch_size, args.output)


if __name__ == "__main__":
    main()
