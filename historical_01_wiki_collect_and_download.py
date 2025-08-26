#!/usr/bin/env python3
"""Collect Wikipedia pages for historical Hansard speakers.

This script reads a JSON file containing speaker metadata (as produced by the
Historical Hansard Data Collection workflow) and attempts to locate the correct
Wikipedia article for each speaker.  It follows these steps:

1.  **Clean speaker metadata** into a minimal, serialisable form.
2.  **Generate a Google search query** that combines name, lifespan, aliases,
    constituencies, titles, etc., plus the keyword "wikipedia".
3.  **Retrieve search‑result links** using Selenium‑driven Firefox and extract
    all links that point to *wikipedia.org* on the first result page.
4.  **Filter candidate pages by year match** – we download the lead paragraph
    of each article and test whether it contains any of the speaker's lifespan
    years.  The first hit is considered a match.
5.  **Extract article content** (intro + main text + infobox) for verified
    pages and store it for downstream GPT processing.
6.  Persist results incrementally to CSV so that the job can be resumed.

The script is written in **Google Python style** (PEP‑484 type hints, doctrings,
logging, constants in ALL_CAPS, small cohesive functions).  It remains
CPU‑light; network I/O (Google + Wikipedia) dominates runtime, so the logic can
optionally be parallelised with a thread pool when fetching Wikipedia pages.

USAGE
-----

```bash
python historical_hansard_wikipedia_collector.py \
    --input speaker_details_with_gender.json \
    --output speaker_details_with_wikipedia.csv \
    --geckodriver /path/to/geckodriver \
    --checkpoint-every 100
```

The script will create (or append to) the output CSV with these columns:

* **search_query**          – The Google query string.
* **wikipedia_links**       – Raw list of candidate links from the first
                               results page.
* **years**                 – Speaker lifespan years as integers.
* **wikipedia_match**       – The URL that passed the year‑match filter (or
                               empty).
* **matched_years_count**   – How many years overlapped between lead paragraph
                               and lifespan.
* **wikipedia_gpt_input**   – Full article text (intro + main + infobox) ready
                               for GPT extraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service as GeckoService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MAX_GOOGLE_WAIT_SEC = 60
GOOGLE_URL = "https://www.google.com"
WIKI_DOMAIN = "wikipedia.org"
USER_AGENT = "HistoricalHansardBot/1.0 (+https://example.com)"
CSV_FIELDNAMES = [
    "search_query",
    "wikipedia_links",
    "years",
    "wikipedia_match",
    "matched_years_count",
    "wikipedia_gpt_input",
]

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def load_speakers(path: Path) -> List[dict]:
    """Load raw JSON containing speaker records."""
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def extract_years(text: str) -> Set[int]:
    """Return all four‑digit years (integers) found in *text*."""
    return {int(y) for y in re.findall(r"\b(\d{4})\b", text)}


def clean_speaker(raw: dict) -> dict:
    """Serialise key speaker fields into simple strings.

    The cleaning logic mirrors the exploratory notebook but is encapsulated
    here.  All lists are collapsed into semicolon‑separated strings with
    parenthetical date ranges preserved.
    """
    def _serialise(items: Optional[Sequence[dict]], *, field: str) -> str:
        if not items:
            return ""
        out: List[str] = []
        for itm in items:
            content = (itm.get("content") or "").strip()
            dates = (itm.get("dates") or "").strip()
            if content and dates:
                out.append(f"{content} ({dates})")
            elif content:
                out.append(content)
        return "; ".join(out)

    return {
        "name": (raw.get("name") or "").strip(),
        "dates": (raw.get("dates") or "").strip(),
        "gender": raw.get("gender", ""),
        "constituencies": _serialise(raw.get("constituencies"), field="constituency"),
        "titles_in_lords": _serialise(raw.get("titles_in_lords"), field="content"),
        "offices": _serialise(raw.get("offices"), field="content"),
        "alternative_names": _serialise(raw.get("alternative_names"), field="content"),
        "other_titles": _serialise(raw.get("other_titles"), field="content"),
    }


def build_search_query(spk: dict) -> str:
    """Construct a Google query for the speaker."""
    parts: List[str] = [spk["name"]]
    if spk["dates"]:
        parts.append(spk["dates"])
    for key in ("alternative_names", "constituencies", "titles_in_lords"):
        if spk[key]:
            # Remove parentheticals; split on ; or ,
            cleaned = re.sub(r"\s*\([^)]*\)", "", spk[key])
            parts.extend([p.strip() for p in re.split(r"[;,]", cleaned) if p.strip()])
    parts.append("wikipedia")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Google search utilities (Selenium)
# ----------------------------------------------------------------------------

def init_webdriver(geckodriver: Path) -> webdriver.Firefox:
    """Instantiate a headless Firefox WebDriver."""
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    service = GeckoService(executable_path=str(geckodriver))
    driver = webdriver.Firefox(service=service, options=options)
    driver.set_page_load_timeout(MAX_GOOGLE_WAIT_SEC)
    return driver


def get_wikipedia_links(driver: webdriver.Firefox, query: str) -> List[str]:
    """Return all Wikipedia links that appear on the first Google results page."""
    driver.get(GOOGLE_URL)

    search_box = driver.find_element(By.NAME, "q")
    search_box.clear()
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    # Wait until at least one Wikipedia result appears (or timeout)
    try:
        WebDriverWait(driver, MAX_GOOGLE_WAIT_SEC).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"a[href*='{WIKI_DOMAIN}']"))
        )
    except Exception:  # pylint: disable=broad-except
        logging.warning("Google results timed out for query: %s", query)

    anchors = driver.find_elements(By.CSS_SELECTOR, "a[href]")
    links = [a.get_attribute("href") for a in anchors if WIKI_DOMAIN in a.get_attribute("href")]
    return links


# ----------------------------------------------------------------------------
# Wikipedia page processing
# ----------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


def fetch_wikipedia_page(url: str) -> Optional[BeautifulSoup]:
    """Download the Wikipedia article and return a BeautifulSoup object."""
    try:
        resp = _SESSION.get(url, timeout=10)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Failed to fetch %s: %s", url, exc)
        return None


def extract_intro_and_full_text(soup: BeautifulSoup) -> Tuple[str, str]:
    """Return (intro_paragraph, full_text) from a Wikipedia article soup."""
    # Intro – first non‑empty <p>
    intro = next(
        (p.get_text(strip=True) for p in soup.select("#mw-content-text p") if p.get_text(strip=True)),
        "",
    )

    parts: List[str] = ["Main text:", ""]
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content:
        for tag in content.find_all(["p", "h2", "h3", "li"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
                parts.append("")
        for tbl in content.find_all("table"):
            tbl_txt = tbl.get_text(" ", strip=True)
            if tbl_txt:
                parts.append(tbl_txt)
                parts.append("")

    parts.append("Infobox:")
    parts.append("")
    infobox = soup.find(class_="infobox")
    if infobox:
        for row in infobox.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if th and td:
                field = th.get_text(" ", strip=True)
                value = td.get_text(" ", strip=True)
                if field and value:
                    parts.append(f"{field}: {value}")
        parts.append("")

    full_text = "\n".join(parts).strip()
    return intro, full_text


def pick_matching_page(
    links: Sequence[str], target_years: Set[int]
) -> Tuple[Optional[str], int, str]:
    """Return (matched_url, overlap_count, full_text) for the first good link."""
    for url in links:
        soup = fetch_wikipedia_page(url)
        if not soup:
            continue
        intro, full_text = extract_intro_and_full_text(soup)
        intro_years = extract_years(intro)
        overlap = intro_years & target_years
        if overlap:
            return url, len(overlap), full_text
    return None, 0, ""


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def process_speakers(
    speakers: List[dict],
    geckodriver: Path,
    out_path: Path,
    checkpoint_every: int,
    sleep_sec: float = 0.0,
) -> None:
    """Iterate through *speakers* and write results to *out_path* CSV."""
    out_fp = out_path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_fp, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()

    driver = init_webdriver(geckodriver)
    try:
        for idx, raw in enumerate(speakers, start=1):
            spk = clean_speaker(raw)
            query = build_search_query(spk)
            years = list(extract_years(spk["dates"]))

            links = get_wikipedia_links(driver, query)
            match_url, n_overlap, full_text = pick_matching_page(links, set(years))

            writer.writerow(
                {
                    "search_query": query,
                    "wikipedia_links": json.dumps(links, ensure_ascii=False),
                    "years": json.dumps(years),
                    "wikipedia_match": match_url or "",
                    "matched_years_count": n_overlap,
                    "wikipedia_gpt_input": full_text,
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


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Collect Wikipedia pages for historical Hansard speakers.")
    parser.add_argument("--input", type=Path, required=True, help="Path to speaker_details_with_gender.json")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--geckodriver", type=Path, required=True, help="Path to geckodriver executable")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Flush CSV every N speakers (default: 100)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between Google queries")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    """Script entrypoint."""
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    speakers = load_speakers(args.input)
    logging.info("Loaded %d speakers from %s", len(speakers), args.input)

    process_speakers(
        speakers=speakers,
        geckodriver=args.geckodriver,
        out_path=args.output,
        checkpoint_every=args.checkpoint_every,
        sleep_sec=args.sleep,
    )
    logging.info("Finished. Output written to %s", args.output)


if __name__ == "__main__":
    main()
