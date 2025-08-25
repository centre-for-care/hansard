#!/usr/bin/env python3
"""historical_verify_wikipedia_pages.py

Use GPT-4o to decide whether each candidate Wikipedia article really belongs
to a historical Hansard speaker.

Input
-----

1. --speakers-json  : *speaker_details_with_gender.json*  
2. --input-csv      : CSV from *historical_wikipedia_collector_full.py*
                      (must include a column **wikipedia_gpt_input**).

Output
------

The same CSV plus one extra column:

    match_decision   →  "yes" | "no" | "uncertain" | ""  (if skipped)

The file is overwritten in-place every *--batch-size* rows so a long run can be
resumed safely.

Example
-------

```bash
python historical_verify_wikipedia_pages.py \
    --speakers-json speaker_details_with_gender.json \
    --input-csv     merged_wikipedia.csv \
    --output-csv    merged_wikipedia_with_match.csv \
    --api-key       sk-... \
    --batch-size    100
```
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

# ────────────────────────────────
# Constants
# ────────────────────────────────
GPT_MODEL       = "gpt-4o"
WIKI_COL        = "wikipedia_gpt_input"
DECISION_COL    = "match_decision"
DEFAULT_BATCH   = 100

# ────────────────────────────────
# Speaker-profile helpers
# ────────────────────────────────
_PROFILE_KEYS = (
    "name",
    "dates",
    "gender",
    "alternative_names",
    "other_titles",
    "constituencies",
    "titles_in_lords",
    "offices",
)


def _serialise_profile(raw: dict) -> str:
    """Return speaker profile as a human-readable block."""
    lines: List[str] = []
    if raw.get("name"):
        lines.append(f"Name: {raw['name']}")
    if raw.get("dates"):
        lines.append(f"Lifespan: {raw['dates']}")
    if raw.get("gender"):
        lines.append(f"Gender: {raw['gender']}")
    if raw.get("alternative_names"):
        lines.append(f"Alternative Names: {raw['alternative_names']}")
    if raw.get("other_titles"):
        lines.append(f"Other Titles: {raw['other_titles']}")
    if raw.get("constituencies"):
        lines.append(f"Constituencies: {raw['constituencies']}")
    if raw.get("titles_in_lords"):
        lines.append(f"Titles in Lords: {raw['titles_in_lords']}")
    if raw.get("offices"):
        lines.append(f"Offices: {raw['offices']}")
    return "\n".join(lines)


PROMPT = """
You are provided with structured information about a historical speaker from Historical Hansard, and the full text from a Wikipedia article.

Important note: Information from Historical Hansard may contain inaccuracies, such as incorrect birth/death dates or slightly inaccurate details. Keep this possibility in mind when comparing the details.

Speaker information:
{speaker_block}

Wikipedia article full text:
<<<WIKIPEDIA ARTICLE>>>
{article_text}
<<<END ARTICLE>>>

Is this Wikipedia article the personal Wikipedia page of the speaker described above?

Make your judgment based primarily on matching:
- Full names or known alternative names
- Major positions, constituencies, or offices held
- Significant life events or titles

Minor discrepancies (e.g., slightly different birth/death years) should not alone lead to a "no" answer, but should prompt caution ("uncertain") unless other details strongly match or contradict.

Answer strictly with one of these three words:
- yes
- no
- uncertain

Answer:
""".strip()

# ────────────────────────────────
# GPT helper
# ────────────────────────────────

def ask_gpt(client: openai.OpenAI, prompt: str) -> str:
    """Ask GPT once; fallback to 'uncertain' on error."""
    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().lower()
        if answer in ("yes", "no", "uncertain"):
            return answer
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("OpenAI error: %s", exc)
    return "uncertain"

# ────────────────────────────────
# Core verification loop
# ────────────────────────────────

def verify(
    df: pd.DataFrame,
    speakers: List[dict],
    client: openai.OpenAI,
    batch_size: int,
    out_path: Path,
) -> None:
    """Populate `match_decision` for every non-empty article row."""
    if DECISION_COL not in df.columns:
        df[DECISION_COL] = ""

    for idx, row in df.iterrows():
        if str(row.get(DECISION_COL)).strip():          # already processed
            continue
        article = row.get(WIKI_COL, "")
        if not isinstance(article, str) or not article.strip():
            continue                                    # nothing to verify

        # speaker index equals CSV row index (collector preserved order)
        speaker_block = _serialise_profile(speakers[idx])

        prompt = PROMPT.format(
            speaker_block=speaker_block,
            article_text=article,
        )
        decision = ask_gpt(client, prompt)
        df.at[idx, DECISION_COL] = decision

        if (idx + 1) % batch_size == 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed up to row %d", idx + 1)

    # final save
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info("Verification complete – results at %s", out_path)


# ────────────────────────────────
# CLI
# ────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-verify historical Hansard ↔ Wikipedia matches")
    p.add_argument("--speakers-json", type=Path, required=True, help="speaker_details_with_gender.json")
    p.add_argument("--input-csv",     type=Path, required=True, help="CSV with wikipedia_gpt_input column")
    p.add_argument("--output-csv",    type=Path, required=True, help="Destination CSV path")
    p.add_argument("--api-key",       default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    p.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH, help="Rows between checkpoints")
    p.add_argument("--log-level",     choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key required (env OPENAI_API_KEY or --api-key)")

    client   = openai.OpenAI(api_key=args.api_key)
    speakers = json.loads(args.speakers_json.read_text(encoding="utf-8"))
    df       = pd.read_csv(args.input_csv, dtype=str)

    logging.info("Loaded %d speaker rows, %d CSV rows", len(speakers), len(df))
    verify(df, speakers, client, args.batch_size, args.output_csv)


if __name__ == "__main__":
    main()
