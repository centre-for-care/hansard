#!/usr/bin/env python3
"""Collect Wikipedia candidate pages for *modern* (post‑2005) Hansard speakers.

This script mirrors the historical‐era collector but adapts to the JSON shape
returned by the **UK Parliament Members API**.  For each member it:

1.  **Builds a concise Google query** from `nameDisplayAs` (or `nameListAs`),
    latest party, and latest constituency (or the string *House of Lords*),
    plus the keyword *wikipedia*.
2.  **Performs a Google search** (Selenium/Firefox) and captures the first
    **two** `wikipedia.org` links on the results page.
3.  **Optionally** downloads each Wikipedia page and stores (intro + main text +
    infobox) for downstream GPT verification/extraction.
4.  Writes incremental results to a CSV so the job can be resumed safely.

Because modern member records rarely include birth/death years, no automatic
year‑match filter is applied; identity verification is delegated to GPT in a
later pipeline stage.

USAGE
-----
```bash
python modern_hansard_wikipedia_collector.py \
    --input post2005_parliament_members.json \
    --output members_wikipedia_candidates.csv \
    --geckodriver /path/to/geckodriver \
    --checkpoint-every 200 \
    --sleep 0.5
```

Output CSV columns
------------------
* **search_query**        – Full Google query string.
* **wikipedia_links**     – JSON‑serialised list (≤2 items) of candidate URLs.
* **wikipedia_full_texts**– JSON‑serialised list of article texts (empty if
                             `--no-download`).

If you only need the URLs (and will fetch text later), pass `--no-download` to
skip Wikipedia fetches for a substantial speed‑up.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service as GeckoService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOOGLE_URL = "https://www.google.com"
WIKI_DOMAIN = "wikipedia.org"
MAX_GOOGLE_WAIT_SEC = 60
MAX_WIKI_LINKS = 2
USER_AGENT = "ModernHansardBot/1.0 (+https://example.com)"
CSV_FIELDNAMES = [
    "search_query",
    "wikipedia_links",
    "wikipedia_full_texts",
]

# ---------------------------------------------------------------------------
# Speaker helpers
# ---------------------------------------------------------------------------

def build_search_query(member: dict) -> Optional[str]:
    """Return a Google query string for *member* or *None* if insufficient data."""
    if not isinstance(member, dict):
        return None
    val = member.get("value", {})
    if not isinstance(val, dict):
        return None

    # 1) Name (display > list)
    name = (val.get("nameDisplayAs") or val.get("nameListAs") or "").strip()
    if not name:
        return None

    # 2) Party (may be missing)
    party_info = val.get("latestParty", {})
    party = (party_info.get("name") or party_info.get("abbreviation") or "").strip()

    # 3) Constituency or Lords indicator
    memb = val.get("latestHouseMembership", {})
    constituency = (memb.get("membershipFrom") or "").strip() or "House of Lords"

    parts: List[str] = [name]
    if party:
        parts.append(party)
    parts.append(constituency)
    parts.append("wikipedia")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Google search utilities
# ---------------------------------------------------------------------------

def init_driver(geckodriver: Path) -> webdriver.Firefox:
    """Spawn a headless Firefox driver."""
    opts = webdriver.FirefoxOptions()
    opts.add_argument("--headless")
    service = GeckoService(executable_path=str(geckodriver))
    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_page_load_timeout(MAX_GOOGLE_WAIT_SEC)
    return driver


def google_wikipedia_links(driver: webdriver.Firefox, query: str) -> List[str]:
    """Return up to *MAX_WIKI_LINKS* Wikipedia URLs from the first results page."""
    driver.get(GOOGLE_URL)
    box = driver.find_element(By.NAME, "q")
    box.clear()
    box.send_keys(query)
    box.send_keys(Keys.RETURN)

    try:
        WebDriverWait(driver, MAX_GOOGLE_WAIT_SEC).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"a[href*='{WIKI_DOMAIN}']"))
        )
    except Exception:  # pylint: disable=broad-except
        logging.warning("Google timeout for query: %s", query)
        return []

    anchors = driver.find_elements(By.CSS_SELECTOR, f"a[href*='{WIKI_DOMAIN}']")
    links: List[str] = []
    for a in anchors:
        href = a.get_attribute("href")
        if href and href not in links:
            links.append(href)
        if len(links) == MAX_WIKI_LINKS:
            break
    return links


# ---------------------------------------------------------------------------
# Wikipedia helpers (optional download)
# ---------------------------------------------------------------------------
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


def fetch_wiki_text(url: str) -> str:
    """Download Wikipedia article and return intro+main+infobox (one string)."""
    try:
        resp = _SESSION.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:  # pylint: disable=broad-except
        logging.debug("Failed to GET %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    intro = next(
        (p.get_text(strip=True) for p in soup.select("#mw-content-text p") if p.get_text(strip=True)),
        "",
    )
    parts: List[str] = [intro, "", "Main text:"]
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content:
        for tag in content.find_all(["p", "h2", "h3", "li"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
    parts.append("Infobox:")
    infobox = soup.find(class_="infobox")
    if infobox:
        for row in infobox.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if th and td:
                parts.append(f"{th.get_text(' ', strip=True)}: {td.get_text(' ', strip=True)}")
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_members(
    members: Sequence[dict],
    geckodriver: Path,
    out_path: Path,
    checkpoint_every: int,
    sleep_sec: float,
    download: bool,
) -> None:
    out_fp = out_path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_fp, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()

    driver = init_driver(geckodriver)
    try:
        for idx, member in enumerate(members, start=1):
            query = build_search_query(member)
            if not query:
                logging.debug("Skipping member without query (idx=%d).", idx)
                continue

            links = google_wikipedia_links(driver, query)
            texts: List[str] = []
            if download:
                for link in links:
                    texts.append(fetch_wiki_text(link))
            writer.writerow(
                {
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
        driver.quit()
        out_fp.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Wikipedia candidates for modern Hansard members.")
    parser.add_argument("--input", type=Path, required=True, help="Path to Parliament Members JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--geckodriver", type=Path, required=True, help="Path to geckodriver executable")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Flush CSV after N members (default 200)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to pause between Google queries")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading full Wikipedia text (URLs only)")
    parser.add_argument("--log-level", default="INFO", choices=[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    members = json.loads(Path(args.input).read_text(encoding="utf-8"))
    logging.info("Loaded %d member records from %s", len(members), args.input)

    process_members(
        members=members,
        geckodriver=args.geckodriver,
        out_path=args.output,
        checkpoint_every=args.checkpoint_every,
        sleep_sec=args.sleep,
        download=not args.no_download,
    )
    logging.info("Finished. Results at %s", args.output)


if __name__ == "__main__":
    main()
