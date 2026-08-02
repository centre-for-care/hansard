#!/usr/bin/env python3
"""
modern_03_verify_pages.py

Verify that a Wikipedia (or Wikidata) article truly refers to the modern
parliamentarian / peer in question using a GPT yes/no/uncertain check.

Idempotent & resumable:
- Rows with a non-empty final_reply in {"yes","no","uncertain"} are skipped.
- Checkpoints written every --batch-size rows.

Typical usage
-------------
python modern_03_verify_pages.py \
  --input modern_02_wikipedia_candidates_with_text.csv \
  --members-json modern_members_full_responses.json \
  --output modern_03_wikipedia_verify.csv \
  --api-key $OPENAI_API_KEY \
  --batch-size 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from openai import OpenAI  # OpenAI v1 SDK

# ------------------------------- Prompt --------------------------------------

PROMPT_TEMPLATE = """You are provided with structured information about a UK parliamentarian and the full text from a Wikipedia article.

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

Answer strictly with one of these three words:
- yes
- no
- uncertain
"""

def build_prompt(name: str, party: str, constituency: str, gender: str, service: str, wiki_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        name=name or "NA",
        party=party or "NA",
        constituency=constituency or "NA",
        gender=gender or "NA",
        service=service or "NA",
        wiki_text=wiki_text or "",
    )

# -------------------------- Members JSON helpers -----------------------------

def iter_members_payloads(data: Any) -> Iterable[dict]:
    """
    Yield member dicts from common shapes:
      • [ {...}, {...} ]
      • [ {"value": {...}}, ... ]
      • { "items": [ {"value": {...}}, ... ] }
      • single {"value": {...}} or single {...}
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
        elif isinstance(data, dict) and "value" in data and isinstance(data["value"], dict):
            yield data["value"]
        else:
            yield data


def load_members_map(path: Path) -> Dict[int, dict]:
    """
    Load the members JSON and return id -> member payload mapping.
    Only keeps entries with an integer 'id'.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, dict] = {}
    for m in iter_members_payloads(raw):
        mid = m.get("id")
        try:
            mid_int = int(mid) if mid is not None else None
        except Exception:
            mid_int = None
        if mid_int is not None:
            out[mid_int] = m
    return out


def enrich_from_members(member: Optional[dict]) -> Tuple[str, str]:
    """
    From a Members API payload, derive (gender, service) where:
      service = "YYYY to YYYY" using latestHouseMembership start/end years.
    Returns ("", "") if member is None or missing.
    """
    if not isinstance(member, dict):
        return "", ""
    gender = str(member.get("gender") or "").strip()

    hs = member.get("latestHouseMembership") or {}
    start = str(hs.get("membershipStartDate") or "")
    end   = str(hs.get("membershipEndDate") or "")
    svc   = ""
    if start or end:
        svc = f"{(start[:4] or 'NA')} to {(end[:4] or 'NA')}"
    return gender, svc

# ------------------------------ GPT call -------------------------------------

def ask_gpt(client: OpenAI, model: str, prompt: str, retries: int = 2, cooldown: float = 20.0) -> str:
    """
    Call GPT; return 'yes'/'no'/'uncertain' (lowercased).
    Retries on rate-limit style errors.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            # take first non-empty line, be conservative
            for line in text.splitlines():
                if line.strip():
                    val = line.strip()
                    if val in {"yes", "no", "uncertain"}:
                        return val
                    break
        except Exception as exc:
            msg = str(exc).lower()
            if ("rate limit" in msg or "429" in msg) and attempt < retries:
                logging.warning("Rate-limited; sleeping %.0fs (attempt %d/%d)…", cooldown, attempt, retries)
                time.sleep(cooldown)
                continue
            logging.warning("OpenAI error (attempt %d/%d): %s", attempt, retries, exc)
    return "uncertain"

# ------------------------------ Utilities ------------------------------------

def parse_json_list(cell) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        data = json.loads(cell)
        return data if isinstance(data, list) else []
    except Exception:
        # legacy Python literal list fallback
        try:
            import ast
            data = ast.literal_eval(cell)
            return data if isinstance(data, list) else []
        except Exception:
            return []


def coalesce(*vals: Optional[str]) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# ------------------------------- Core ----------------------------------------

def verify_pages(
    df: pd.DataFrame,
    members_map: Dict[int, dict],
    client: OpenAI,
    model: str,
    batch_size: int,
    out_path: Path,
) -> None:
    """
    For each row, verify up to two article texts.
    Writes gpt_reply_1, gpt_reply_2, matched_url, final_reply.
    """
    # Ensure output columns exist
    for col in ("gpt_reply_1", "gpt_reply_2", "matched_url", "final_reply"):
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    for n, (row_idx, row) in enumerate(df.iterrows(), start=1):
        existing = str(row.get("final_reply", "")).strip().lower()
        if existing in {"yes", "no", "uncertain"}:
            continue

        # Pull basic fields from CSV (always available from modern_01)
        member_id = None
        try:
            mid_raw = row.get("member_id")
            member_id = int(mid_raw) if str(mid_raw).strip() != "" else None
        except Exception:
            member_id = None

        member_name = coalesce(row.get("member_name"))
        party = coalesce(row.get("party"))
        constituency = coalesce(row.get("constituency_or_lords"))

        # Enrich from Members JSON if available via id
        gender = ""
        service = ""
        if member_id is not None and member_id in members_map:
            g, s = enrich_from_members(members_map[member_id])
            gender = g or gender
            service = s or service

        texts = [
            coalesce(row.get("wikipedia_text_1")),
            coalesce(row.get("wikipedia_text_2")),
        ]
        links = parse_json_list(row.get("wikipedia_links", ""))

        replies: List[str] = []
        matched_url = ""
        final = "no"

        for i, txt in enumerate(texts):
            if not txt:
                replies.append("NA")
                continue
            prompt = build_prompt(
                name=member_name,
                party=party,
                constituency=constituency,
                gender=gender,
                service=service,
                wiki_text=txt,
            )
            decision = ask_gpt(client, model, prompt)
            replies.append(decision)

            if decision == "yes":
                matched_url = links[i] if i < len(links) else ""
                final = "yes"
                break
            elif decision == "uncertain" and final != "yes":
                final = "uncertain"

        while len(replies) < 2:
            replies.append("NA")

        df.at[row_idx, "gpt_reply_1"] = replies[0]
        df.at[row_idx, "gpt_reply_2"] = replies[1]
        df.at[row_idx, "matched_url"] = matched_url
        df.at[row_idx, "final_reply"] = final

        if (n % batch_size) == 0:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info("Checkpointed %d/%d rows → %s", n, total, out_path)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info("Verification complete – results in %s", out_path)

# -------------------------------- CLI ----------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify Wikipedia matches for modern Hansard speakers (GPT yes/no/uncertain).")
    p.add_argument("--input",        type=Path, required=True,
                   help="CSV with article texts & links (output of modern_02 or modern_01).")
    p.add_argument("--members-json", type=Path, required=True,
                   help="Members API dump (output of modern_00_fetch_members_api.py).")
    p.add_argument("--output",       type=Path, required=True,
                   help="Destination CSV path.")
    p.add_argument("--api-key",      type=str, default=os.getenv("OPENAI_API_KEY"),
                   help="OpenAI API key (env if omitted).")
    p.add_argument("--model",        type=str, default="gpt-4o",
                   help="Model name (default: gpt-4o).")
    p.add_argument("--batch-size",   type=int, default=200,
                   help="Rows processed between checkpoints.")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging verbosity.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.api_key:
        raise SystemExit("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=args.api_key)

    members_map = load_members_map(args.members_json)
    logging.info("Loaded %d members from %s", len(members_map), args.members_json)

    df = pd.read_csv(args.input, dtype=str)

    # Ensure required columns exist
    for col in ("wikipedia_links", "member_name", "party", "constituency_or_lords"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV must contain column '{col}' (from modern_01).")

    # Add text columns if missing (e.g., if you skipped modern_02 and want to verify only URLs)
    for col in ("wikipedia_text_1", "wikipedia_text_2"):
        if col not in df.columns:
            df[col] = ""

    verify_pages(df, members_map, client, args.model, args.batch_size, args.output)


if __name__ == "__main__":
    main()
