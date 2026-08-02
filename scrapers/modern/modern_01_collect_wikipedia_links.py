#!/usr/bin/env python3
"""
modern_01_collect_wikipedia_candidates.py

Collect Wikipedia candidate pages for *modern* (post-2005) Hansard speakers.

This script mirrors the historical-era collector but adapts to the JSON shape
returned by the **UK Parliament Members API**. For each member it:

1) Builds a concise Google query from `nameDisplayAs` (or `nameListAs`),
   latest party, and latest constituency (or the string "House of Lords"),
   plus the keyword "wikipedia".
2) Performs a Google search (Selenium/Firefox) and captures the first N
   `wikipedia.org` links on the results page (default: 2).
3) Optionally downloads each Wikipedia page and stores (intro + main text +
   infobox) for downstream GPT verification/extraction.
4) Writes incremental results to a CSV so the job can be resumed safely.

Because modern member records rarely include birth/death years, no automatic
year-match filter is applied; identity verification is delegated to GPT in a
later pipeline stage.

USAGE
-----
# URLs only (faster)
python modern_01_collect_wikipedia_candidates.py \
  --input modern_members_full_responses.json \
  --output modern_01_wikipedia_candidates.csv \
  --geckodriver /path/to/geckodriver \
  --checkpoint-every 200 \
  --sleep 0.25 \
  --no-download

# URLs + full page texts (slower)
python modern_01_collect_wikipedia_candidates.py \
  --input modern_members_full_responses.json \
  --output modern_01_wikipedia_candidates_with_text.csv \
  --geckodriver /path/to/geckodriver \
  --checkpoint-every 200 \
  --sleep 0.25

Output CSV columns
------------------
* member_id            – Member numeric ID (if present in source).
* member_name          – `nameDisplayAs` (or `nameListAs`) used for search.
* party                – Latest party name/abbr (best effort).
* constituency_or_lords– Latest constituency, or "House of Lords".
* search_query         – Full Google query string.
* wikipedia_links      – JSON list (≤ max-links) of candidate URLs.
* wikipedia_full_texts – JSON list of article texts (empty if `--no-download`).

Dependencies
------------
pip install selenium bs4 requests
# And install Firefox + geckodriver on your system.

Notes
-----
• The script is robust to common Members API shapes:
  - a list of plain member dicts,
  - a list of {"value": {...}} items,
  - a single object with {"items": [...]}, each {"value": {...}}.
• Use `--max-links` if you want more/less than the default 2 Wikipedia links.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service as GeckoService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
GOOGLE_URL = "https://www.google.com"
WIKI_DOMAIN = "wikipedia.org"
MAX_GOOGLE_WAIT_SEC = 60
USER_AGENT = "ModernHansardBot/1.0 (+https://example.com)"  # set your project URL

CSV_FIELDNAMES = [
    "member_id",
    "member_name",
    "party",
    "constituency_or_lords",
    "search_query",
    "wikipedia_links",
    "wikipedia_full_texts",
]

# Single shared requests session for optional Wikipedia downloads
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: input shape + member accessor
# ──────────────────────────────────────────────────────────────────────────────
def iter_members(data: Any) -> Iterable[dict]:
    """
    Yield member dicts from several common API result shapes.
    Supports:
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
            # single object (maybe already the value)
            yield data
    else:
        logging.warning("Unsupported input JSON shape; got %s", type(data).__name__)


def extract_member_core(member: dict) -> Optional[Tuple[Optional[int], str, str, str]]:
    """
    Extract core fields we need:
      → (member_id, display_name, party, constituency_or_lords)
    Returns None if no usable name is found.
    """
    if not isinstance(member, dict):
        return None

    # Name (display > list)
    name = (member.get("nameDisplayAs") or member.get("nameListAs") or "").strip()
    if not name:
        return None

    # Member ID (if present)
    member_id = member.get("id")
    if isinstance(member_id, str):
        try:
            member_id = int(member_id)
        except Exception:
            pass
    elif not isinstance(member_id, int):
        member_id = None

    # Latest party (best effort)
    party_info = (member.get("latestParty") or {}) if isinstance(member.get("latestParty"), dict) else {}
    party = (party_info.get("name") or party_info.get("abbreviation") or "").strip()

    # Constituency or Lords
    memb = (member.get("latestHouseMembership") or {}) if isinstance(member.get("latestHouseMembership"), dict) else {}
    constituency = (memb.get("membershipFrom") or "").strip() or "House of Lords"

    return member_id, name, party, constituency


def build_search_query(name: str, party: str, constituency_or_lords: str) -> str:
    """
    Construct a Google query for the member.
    """
    parts: List[str] = [name]
    if party:
        parts.append(party)
    parts.extend([constituency_or_lords, "wikipedia"])
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Google search utilities
# ──────────────────────────────────────────────────────────────────────────────
def init_driver(geckodriver: Path) -> webdriver.Firefox:
    """Spawn a headless Firefox driver."""
    opts = webdriver.FirefoxOptions()
    opts.add_argument("--headless")
    # Mild hardening to reduce noise:
    opts.set_preference("dom.webnotifications.enabled", False)
    service = GeckoService(executable_path=str(geckodriver))
    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_page_load_timeout(MAX_GOOGLE_WAIT_SEC)
    return driver


def google_wikipedia_links(driver: webdriver.Firefox, query: str, max_links: int) -> List[str]:
    """Return up to *max_links* Wikipedia URLs from the first Google results page."""
    driver.get(GOOGLE_URL)

    # Accept simple consent forms if present (best effort; ignore failures)
    try:
        # Some regional variants show a one-time consent dialog
        buttons = driver.find_elements(By.CSS_SELECTOR, "button, div[role='button']")
        for b in buttons:
            label = (b.text or "").lower()
            if "accept" in label or "i agree" in label:
                try:
                    b.click()
                    break
                except Exception:
                    pass
    except Exception:
        pass

    box = driver.find_element(By.NAME, "q")
    box.clear()
    box.send_keys(query)
    box.send_keys(Keys.RETURN)

    try:
        WebDriverWait(driver, MAX_GOOGLE_WAIT_SEC).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"a[href*='{WIKI_DOMAIN}']"))
        )
    except Exception:
        logging.warning("Google timeout for query: %s", query)
        return []

    anchors = driver.find_elements(By.CSS_SELECTOR, f"a[href*='{WIKI_DOMAIN}']")
    links: List[str] = []
    for a in anchors:
        href = a.get_attribute("href")
        if href and href not in links:
            links.append(href)
        if len(links) >= max_links:
            break
    return links


# ──────────────────────────────────────────────────────────────────────────────
# Wikipedia helpers (optional download)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_wiki_text(url: str) -> str:
    """Download Wikipedia article and return intro + main + infobox as one string."""
    try:
        resp = _SESSION.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:  # keep going on failures
        logging.debug("Failed to GET %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Intro – first non-empty <p> under content text
    intro = next(
        (p.get_text(strip=True) for p in soup.select("#mw-content-text p") if p.get_text(strip=True)),
        "",
    )

    parts: List[str] = []
    if intro:
        parts.extend([intro, ""])

    parts.append("Main text:")
    container = soup.select_one("#mw-content-text .mw-parser-output")
    if container:
        for tag in container.find_all(["p", "h2", "h3", "li"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append(txt)

    parts.extend(["", "Infobox:"])
    infobox = soup.find(class_="infobox")
    if infobox:
        for row in infobox.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if th and td:
                parts.append(f"{th.get_text(' ', strip=True)}: {td.get_text(' ', strip=True)}")

    return "\n".join(parts).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_members(
    members: Sequence[dict],
    geckodriver: Path,
    out_path: Path,
    checkpoint_every: int,
    sleep_sec: float,
    download: bool,
    max_links: int,
) -> None:
    """
    Iterate over member records, run Google queries, capture up to max_links
    Wikipedia URLs (and optionally page texts), and write CSV incrementally.
    """
    out_fp = out_path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_fp, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()

    driver = init_driver(geckodriver)
    try:
        for idx, member in enumerate(members, start=1):
            core = extract_member_core(member)
            if core is None:
                logging.debug("Skipping member without usable name (row %d)", idx)
                continue
            member_id, name, party, const_or_lords = core
            query = build_search_query(name, party, const_or_lords)

            links = google_wikipedia_links(driver, query, max_links=max_links)
            texts: List[str] = []
            if download and links:
                for link in links:
                    texts.append(fetch_wiki_text(link))

            writer.writerow(
                {
                    "member_id": "" if member_id is None else member_id,
                    "member_name": name,
                    "party": party,
                    "constituency_or_lords": const_or_lords,
                    "search_query": query,
                    "wikipedia_links": json.dumps(links, ensure_ascii=False),
                    "wikipedia_full_texts": json.dumps(texts, ensure_ascii=False),
                }
            )

            if idx % checkpoint_every == 0:
                out_fp.flush()
                logging.info("Checkpointed %d rows to %s", idx, out_path)

            if sleep_sec:
                time.sleep(sleep_sec)
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        out_fp.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Wikipedia candidates for modern Hansard members.")
    p.add_argument("--input", type=Path, required=True, help="Path to Parliament Members JSON file")
    p.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    p.add_argument("--geckodriver", type=Path, required=True, help="Path to geckodriver executable")
    p.add_argument("--checkpoint-every", type=int, default=200, help="Flush CSV after N members")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to pause between Google queries")
    p.add_argument("--no-download", action="store_true", help="Skip downloading full Wikipedia text (URLs only)")
    p.add_argument("--max-links", type=int, default=2, help="Maximum Wikipedia links to collect per member")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging verbosity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    members = list(iter_members(raw))
    logging.info("Loaded %d member records from %s", len(members), args.input)

    process_members(
        members=members,
        geckodriver=args.geckodriver,
        out_path=args.output,
        checkpoint_every=args.checkpoint_every,
        sleep_sec=args.sleep,
        download=not args.no_download,
        max_links=max(1, args.max_links),
    )
    logging.info("Finished. Results at %s", args.output)


if __name__ == "__main__":
    main()
