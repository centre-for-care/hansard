#!/usr/bin/env python3
"""historical_extract_bio_json.py

For every *verified* Wikipedia article (``match_decision == "yes"``) in the
collector CSV, call GPT-4o to extract a rich biography JSON.

Input  (required)
-----------------
--input-csv       CSV from *historical_verify_wikipedia_pages.py*
                  – must contain columns:
                       • wikipedia_gpt_input   (raw article text)
                       • match_decision        (yes / no / uncertain)

Output  (required)
------------------
--output-csv      Same CSV plus two new columns
                       • extracted_json       – GPT-parsed biography
                       • api_response_full    – full OpenAI API response (JSON)

API key
-------
Give your OpenAI key either **via environment variable**
``OPENAI_API_KEY=sk-…`` *or* the flag ``--api-key sk-…``.

Typical usage
-------------
```bash
python historical_extract_bio_json.py \
  --input-csv  merged_wikipedia.csv \
  --output-csv merged_wikipedia_with_bio.csv \
  --api-key    sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX \
  --batch-size 100
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import openai
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "gpt-4o"
ARTICLE_COL     = "wikipedia_gpt_input"
DECISION_COL    = "match_decision"
OUT_JSON_COL    = "extracted_json"
RAW_API_COL     = "api_response_full"
DEFAULT_BATCH   = 100

# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_HEADER = (
    """
You are a research assistant extracting structured metadata from Wikipedia articles about British politicians and peers
(active at any point since 1800 and recorded in Hansard).

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
  "name": "",                       // Full legal name

  "date_of_birth": "",              // Prefer full YYYY-MM-DD.
                                     // If only “July 1820” is known, return "1820-07".
                                     // If only year (e.g. "1820") is known, return "1820".
                                     // If nothing reliable, return null
  "date_of_death": "",              // Same rules as date_of_birth
  "place_of_birth": "",             // Record **exactly** what Wikipedia gives, in order.
                                     // e.g. "Stoke Newington, London, England"
                                     // If only “London” appears, return "London".
                                     // Do **not** invent missing parts; if no birthplace,
                                     // return null.

  "party_affiliation": [            // One element per party period; keep chronological order
    {
      "party": "",                  // e.g. "Conservative", "Liberal Unionist"
      "start_year": null,           // int or null
      "end_year": null              // int or null (null = still affiliated or unknown end)
    }
  ],

  "education": {
    "school_type": "",              // "Clarendon", "HMC schools", "Other private", "All other", or null
    "school_name": "",              // Full secondary-school name, or null
    "school_country": "",           // "UK", "France", "USA", etc., or null
    "universities": [               // one object per degree / course
      {
        "university_name": "",      // e.g. "University of Oxford", "University of Cambridge", or null
        "university_city": "",      // e.g. "Oxford", "Cambridge", or null
        "university_country": "",   // "UK","USA", etc.
        "degree_level": "",         // "Undergraduate","Masters","Doctorate",
                                    // "Diploma","Professional Qualification", or null
        "field_of_study": ""        // e.g. "Law","PPE","History", or null
      }
      /* repeat for each additional qualification */
    ]
  },

  "occupation_before_politics": "", // Main profession(s) prior to entering parliament, or null

  "political_career": {
    "first_elected":,          // Year first elected to Commons/Lords; null if unknown
    "last_elected":,           // Year last elected / final term, or null
    "years_in_parliament":,    // Total years served as MP/Lord; null if unknown
    "ministerial_positions": [],    // List of "Title (start–end)" strings
    "leadership_positions": []      // List of "Role (start–end)" strings, e.g. "Chief Whip (1886–1892)"
  }
}
━━━━━━━━  END OF SCHEMA  ━━━━━━━━

Wikipedia article text:
"""

# ──────────────────────────────────────────────────────────────────────────────
# GPT call helper
# ──────────────────────────────────────────────────────────────────────────────

def ask_gpt(client: openai.OpenAI, article_text: str) -> Tuple[str, str]:
    """Return (json_text, raw_response) from GPT-4o or ("", "ERROR …") on failure."""
    prompt = f"{PROMPT_HEADER}\n\nWikipedia article text:\n{article_text}"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip(), json.dumps(resp.model_dump(), ensure_ascii=False)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("OpenAI error: %s", exc)
        return "", f"ERROR: {exc}"

# ──────────────────────────────────────────────────────────────────────────────
# Extraction loop
# ──────────────────────────────────────────────────────────────────────────────

def run_extraction(df: pd.DataFrame, client: openai.OpenAI, batch_size: int, out_path: Path) -> None:
    """Populate OUT_JSON_COL & RAW_API_COL for every verified row needing extraction."""
    for col in (OUT_JSON_COL, RAW_API_COL):
        if col not in df.columns:
            df[col] = ""

    targets = df.index[
        (df[DECISION_COL].str.lower() == "yes")
        & df[OUT_JSON_COL].astype(str).str.strip().eq("")
        & df[ARTICLE_COL].astype(str).str.strip().ne("")
    ].tolist()

    logging.info("Need extraction for %d rows", len(targets))

    for i, idx in enumerate(targets, start=1):
        article_text = df.at[idx, ARTICLE_COL]
        json_out, raw_resp = ask_gpt(client, article_text)
        df.at[idx, OUT_JSON_COL] = json_out
        df.at[idx, RAW_API_COL]  = raw_resp

        if i % batch_size == 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d", i, len(targets))

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info("Extraction finished – written to %s", out_path)

# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract structured biography JSON from Wikipedia articles")
    p.add_argument("--input-csv",  type=Path, required=True, help="CSV with wikipedia_gpt_input & match_decision")
    p.add_argument("--output-csv", type=Path, required=True, help="Destination CSV path (overwritten)")
    p.add_argument("--api-key",    default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Checkpoint every N rows")
    p.add_argument("--log-level",  choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key required – provide via OPENAI_API_KEY or --api-key")

    client = openai.OpenAI(api_key=args.api_key)
    df      = pd.read_csv(args.input_csv, dtype=str)
    logging.info("Loaded %d rows from %s", len(df), args.input_csv)

    run_extraction(df, client, args.batch_size, args.output_csv)


if __name__ == "__main__":
    main()
