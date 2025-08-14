#!/usr/bin/env python3
"""modern_verify_pages.py

Verify that a Wikipedia (or Wikidata) article truly refers
to the parliamentarian / peer in question, using a *yes / no / uncertain* GPT
check.  The script is **idempotent** and **resumable**: rows that already
contain a value in ``final_reply`` are skipped.

Usage
-----
```bash
python modern_verify_pages.py \
  --input merged_wikipedia_modern_with_text.csv \
  --members-json post2005_parliament_members_full_responses.json \
  --output wikipedia_verify.csv \
  --api-key $OPENAI_API_KEY \
  --batch-size 200
```

Input CSV must have:
* ``wikipedia_text_1`` / ``wikipedia_text_2`` – full article texts (may be empty)
* ``wikipedia_links`` – list (JSON string) with ≤2 URLs (used only for
  recording ``matched_url``)

Columns added / updated:
* ``gpt_reply_1`` – yes/no/uncertain/NA for first article
* ``gpt_reply_2`` – same for second
* ``matched_url`` – URL when GPT says *yes*
* ``final_reply`` – summary decision for the row (yes/no/uncertain)
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import openai
import pandas as pd

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are provided with structured information about a speaker from Hansard, and the full text from a Wikipedia article.

Speaker information:
- Name: {name}
- Party: {party}
- Constituency: {constituency}
- Service: {service}
- Gender: {gender}

Wikipedia article full text:
<<<WIKIPEDIA ARTICLE>>>
{wiki_text}
<<<END ARTICLE>>>

Is this Wikipedia article the personal Wikipedia page of the speaker described above?

Answer **strictly** with one of these three words:
- yes
- no
- uncertain"""


def build_prompt(member: dict, wiki_text: str) -> str:
    """Format the system + user prompt for GPT verification."""
    name = member.get("nameDisplayAs") or member.get("nameListAs", "NA")
    party = (member.get("latestParty") or {}).get("name", "NA")
    constituency = (member.get("latestHouseMembership") or {}).get("membershipFrom", "NA")
    gender = member.get("gender", "NA")

    hs = member.get("latestHouseMembership") or {}
    start = (hs.get("membershipStartDate") or "")[:4]
    end = (hs.get("membershipEndDate") or "")[:4]
    service = f"{start or 'NA'} to {end or 'NA'}"

    return PROMPT_TEMPLATE.format(
        name=name,
        party=party,
        constituency=constituency,
        service=service,
        gender=gender,
        wiki_text=wiki_text,
    )


# ---------------------------------------------------------------------------
# GPT utility
# ---------------------------------------------------------------------------

def ask_gpt(client: openai.OpenAI, model: str, prompt: str, retries: int = 2) -> str:
    """Call GPT, return first non‑blank trimmed line; retries on rate‑limit."""
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            for line in resp.choices[0].message.content.splitlines():
                if line.strip():
                    return line.strip().lower()
        except Exception as exc:  # pylint: disable=broad-except
            if "rate limit" in str(exc).lower() and attempt < retries:
                logging.warning("Rate‑limit; sleeping 20s (attempt %d/%d)…", attempt, retries)
                time.sleep(20)
                continue
            logging.error("OpenAI error: %s", exc)
    return "uncertain"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def verify_pages(
    df: pd.DataFrame,
    members: List[dict],
    client: openai.OpenAI,
    model: str,
    batch_size: int,
    out_path: Path,
) -> None:
    total = len(df)
    for row_idx in df.index:
        # skip rows that already have a decision
        if str(df.at[row_idx, "final_reply").lower() in {"yes", "no", "uncertain"}:
            continue

        mp = members[row_idx]["value"]
        texts = [df.at[row_idx, "wikipedia_text_1"] or "", df.at[row_idx, "wikipedia_text_2"] or ""]
        links = df.at[row_idx, "wikipedia_links"]
        if isinstance(links, str):  # JSON‑encoded list if not already parsed
            links = ast.literal_eval(links)

        replies = []
        matched_url = ""
        final = "no"

        for idx_in, txt in enumerate(texts):
            if not txt.strip():
                replies.append("NA")
                continue
            label = ask_gpt(client, model, build_prompt(mp, txt))
            replies.append(label)
            if label == "yes":
                matched_url = links[idx_in] if idx_in < len(links) else ""
                final = "yes"
                break  # no need to check second text
            if label == "uncertain" and final != "yes":
                final = "uncertain"  # may be overridden by later "yes"

        # Fill any missing reply slots with "NA"
        while len(replies) < 2:
            replies.append("NA")

        df.at[row_idx, "gpt_reply_1"] = replies[0]
        df.at[row_idx, "gpt_reply_2"] = replies[1]
        df.at[row_idx, "matched_url"] = matched_url
        df.at[row_idx, "final_reply"] = final

        # checkpoint
        if (row_idx + 1) % batch_size == 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d rows", row_idx + 1, total)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info("Verification complete – results in %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Wikipedia matches for modern Hansard speakers (GPT yes/no/uncertain).")
    parser.add_argument("--input",        type=Path, required=True, help="CSV with article texts & links")
    parser.add_argument("--members-json", type=Path, required=True, help="post2005_parliament_members_full_responses.json")
    parser.add_argument("--output",       type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--api-key",      type=str,   default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key (env if omitted)")
    parser.add_argument("--model",        type=str,   default="gpt-4o", help="Model name (default: gpt-4o)")
    parser.add_argument("--batch-size",   type=int,   default=200, help="Rows processed before checkpoint save")
    parser.add_argument("--log-level",    default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY in environment.")

    client = openai.OpenAI(api_key=args.api_key)

    members = json.loads(Path(args.members_json).read_text(encoding="utf-8"))

    df = pd.read_csv(args.input, dtype=str)
    # ensure list dtype in memory
    if "wikipedia_links" in df.columns:
        df["wikipedia_links"] = df["wikipedia_links"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # create missing cols for idempotency
    for col in ("gpt_reply_1", "gpt_reply_2", "matched_url", "final_reply"):
        if col not in df.columns:
            df[col] = ""

    verify_pages(df, members, client, args.model, args.batch_size, args.output)


if __name__ == "__main__":
    main()
