#!/usr/bin/env python3
"""modern_openweb_fallback.py

Final open‑web fallback for speakers whose biography JSON is
still missing after all Wikipedia/Wikidata passes.

Logic
-----
* Identify rows where ``extracted_json`` is blank / NaN.
* Send the speaker profile to GPT‑4o‑search‑preview with an "open web" prompt
  (no domain restriction). GPT must return **only** the JSON object following
  the unified schema.
* Store GPT output in ``extracted_json`` and raw response in
  ``api_response_full``.
* Resumable: already‑filled rows are skipped; checkpoints written every
  ``--batch-size`` rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import openai
import pandas as pd

# ---------------------------------------------------------------------------
# Prompt for open‑web search
# ---------------------------------------------------------------------------
OPENWEB_PROMPT = """
You are an automated web-research assistant.

TASK
• You will receive a short JSON profile for an UK parliamentarian or peer.
• Search the open web for reliable biographical information about this exact individual. 
• Preferred sources: major encyclopedias (Wikipedia, Wikidata, Britannica), government or parliamentary sites, reputable newspapers, archival databases.  
• If a fact is missing or cannot be verified with high confidence, output null (do NOT guess).

RETURN FORMAT  
Return a single JSON object that follows exactly the schema described.
Fill fields as completely as possible; if a value is missing or cannot be inferred, return `null`.
Return only the JSON object—no extra keys, comments, or trailing text.

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

INPUT PROFILE:
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_openweb_gpt(client: openai.OpenAI, profile_json: str, retries: int = 2) -> tuple[str | None, str | None]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-search-preview",
                messages=[
                    {"role": "system", "content": OPENWEB_PROMPT},
                    {"role": "user", "content": profile_json},
                ],
            )
            return resp.choices[0].message.content.strip(), json.dumps(resp.to_dict(), ensure_ascii=False)
        except Exception as exc:  # pylint: disable=broad-except
            if "rate limit" in str(exc).lower() and attempt < retries:
                logging.warning("Rate‑limit hit; sleeping 20 s (attempt %d)…", attempt)
                time.sleep(20)
                continue
            logging.error("OpenAI error: %s", exc)
            break
    return None, None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def openweb_fallback(
    df: pd.DataFrame,
    members: List[dict],
    client: openai.OpenAI,
    batch_size: int,
    out_path: Path,
) -> None:
    # ensure columns exist
    for col in ("extracted_json", "api_response_full"):
        if col not in df.columns:
            df[col] = ""

    to_process = df.index[
        df["extracted_json"].isna() | df["extracted_json"].astype(str).str.strip().eq("")
    ].tolist()
    logging.info("Open‑web fallback needed for %d speakers", len(to_process))

    for i, idx in enumerate(to_process, start=1):
        profile_json = json.dumps(members[idx]["value"], ensure_ascii=False)
        bio_json, raw = ask_openweb_gpt(client, profile_json)
        if bio_json:
            df.at[idx, "extracted_json"]    = bio_json
            df.at[idx, "api_response_full"] = raw

        if i % batch_size == 0 or i == len(to_process):
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d", i, len(to_process))

    logging.info("Open‑web fallback complete – results in %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final open‑web fallback to fill missing biography JSON.")
    parser.add_argument("--input",        type=Path, required=True, help="CSV after Step 8")
    parser.add_argument("--members-json", type=Path, required=True, help="Members metadata JSON")
    parser.add_argument("--output",       type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--api-key",      default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--batch-size",   type=int, default=50, help="Rows before checkpoint save")
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

    openweb_fallback(df, members, client, args.batch_size, args.output)


if __name__ == "__main__":
    main()
