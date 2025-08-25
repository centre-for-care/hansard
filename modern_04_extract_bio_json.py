"""modern_extract_bio_json.py

Extract structured biography JSON *after* a Wikipedia page has
been verified as the correct person.

Input CSV requirements
----------------------
* ``wiki_text``         – article text already flattened (non‑empty for rows to process)
* ``extracted_json``    – will be filled with GPT output (kept if already present)

Columns added / updated
-----------------------
* ``extracted_json``    – the biography JSON or empty on failure
* ``api_response_full`` – raw OpenAI response (useful for audit / re‑prompt)

The script is resumable and skips rows where ``extracted_json`` is non‑blank.

Example
-------
```bash
python modern_extract_bio_json.py \
  --input modern_step6_input.csv \
  --output modern_step6_output.csv \
  --api-key $OPENAI_API_KEY \
  --batch-size 100
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List

import openai
import pandas as pd
import tiktoken

# ---------------------------------------------------------------------------
# Prompt (long string constant to avoid extra file dependency)
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
  },

{
  "constituencies": [         // chronological order, earliest → latest
    {
      "seat": "",             // Constituency name exactly as it appears, e.g. "Tamworth"
      "start": "",            // Prefer full ISO date  "YYYY-MM-DD".
                              // If only month/year known:  "YYYY-MM".
                              // If only year known:        "YYYY".
                              // If unknown:                null
      "end": ""               // Same rules as "start".  null = still in office or unknown end.
    }
    /* repeat for every distinct seat-holding period */
  ]
}
━━━━━━━━  END OF SCHEMA  ━━━━━━━━

Wikipedia article text:
"""

# heading pattern for truncation
REF_PATTERN = re.compile(r"(?i)\n+references\b")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_after_references(text: str) -> str:
    """Remove content after the last 'References' heading to save tokens."""
    parts = REF_PATTERN.split(text, maxsplit=1)
    return parts[0]


def token_count(text: str, enc) -> int:
    return len(enc.encode(text))


def ask_gpt(client: openai.OpenAI, model: str, system_prompt: str, page_text: str, retries: int = 2) -> str | None:
    """Send (system,user) messages, return raw JSON string or None on failure."""
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
            return resp.choices[0].message.content.strip(), json.dumps(resp.to_dict(), ensure_ascii=False)
        except Exception as exc:  # pylint: disable=broad-except
            if "rate limit" in str(exc).lower() and attempt < retries:
                logging.warning("Rate‑limit; sleeping 20 s (attempt %d)…", attempt)
                time.sleep(20)
                continue
            logging.error("OpenAI error: %s", exc)
            return None
    return None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def extract_bio(
    df: pd.DataFrame,
    client: openai.OpenAI,
    model: str,
    token_limit: int,
    batch_size: int,
    out_path: Path,
) -> None:
    enc = tiktoken.encoding_for_model(model)

    total = len(df)
    for row_idx in df.index:
        # skip if wiki_text empty or bio already extracted
        if not str(df.at[row_idx, "wiki_text").strip():
            continue
        if str(df.at[row_idx, "extracted_json").strip():
            continue

        text = str(df.at[row_idx, "wiki_text"])
        if token_count(text, enc) > token_limit:
            text = truncate_after_references(text)
            if token_count(text, enc) > token_limit:
                logging.warning("[%d] Still too long after truncation; skipped.", row_idx)
                continue
            df.at[row_idx, "wiki_text"] = text  # update truncated version for audit

        result = ask_gpt(client, model, SYSTEM_PROMPT, text)
        if result:
            json_out, raw_resp = result
            df.at[row_idx, "extracted_json"]    = json_out
            df.at[row_idx, "api_response_full"] = raw_resp

        # checkpoint
        if (row_idx + 1) % batch_size == 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d rows", row_idx + 1, total)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info("Extraction complete – results in %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured biography JSON from verified Wikipedia texts.")
    parser.add_argument("--input",       type=Path, required=True, help="CSV with wiki_text column ready")
    parser.add_argument("--output",      type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--api-key",     default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key (env fallback)")
    parser.add_argument("--model",      default="gpt-4o", help="Model name (default: gpt-4o)")
    parser.add_argument("--token-limit", type=int, default=30000, help="Max tokens allowed incl. prompt")
    parser.add_argument("--batch-size",  type=int, default=100, help="Rows processed before checkpoint save")
    parser.add_argument("--log-level",   default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key is required (via --api-key or OPENAI_API_KEY env var)")

    client = openai.OpenAI(api_key=args.api_key)

    df = pd.read_csv(args.input, dtype=str)
    # Ensure required cols exist
    for col in ("extracted_json", "api_response_full"):
        if col not in df.columns:
            df[col] = ""

    extract_bio(df, client, args.model, args.token_limit, args.batch_size, args.output)


if __name__ == "__main__":
    main()
