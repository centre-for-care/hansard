#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modern_hansard_01_scrape_debates.py

Scrape Modern Hansard debate structure from https://hansard.parliament.uk/

What it does
------------
1) Collect available sitting dates for a year range (Commons by default).
2) For each date, extract "HTML download" debate links.
3) For each debate URL, parse the hierarchical structure:
   - Section headings (h2/h3)
   - Info lines (date/time/metadata)
   - Speaker contributions (name + paragraphs)
   - Nested subsections

Outputs
-------
- Dates checkpoint JSON     (default: modern_available_dates.json)
- Full scraped data as JSON (default: modern_hansard_data.json)

Usage
-----
# Basic (Commons, 2020–2021)
python modern_hansard_01_scrape_debates.py --start-year 2020 --end-year 2021

# Lords only, custom files, be nice with 4s delay
python modern_hansard_01_scrape_debates.py --chamber Lords \
  --start-year 2019 --end-year 2019 \
  --dates-file lords_dates_2019.json \
  --out lords_2019.json --sleep 4

# Reuse existing dates file (faster)
python modern_hansard_01_scrape_debates.py --start-year 2020 --end-year 2020 --reuse-dates
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    # cloudscraper gracefully handles Cloudflare
    import cloudscraper
    from cloudscraper import CloudScraper
except Exception as e:  # pragma: no cover
    raise SystemExit("Please `pip install cloudscraper bs4`") from e


BASE_URL = "https://hansard.parliament.uk"
DEFAULT_SLEEP = 3.0
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class Contribution:
    speaker: str
    speaker_url: Optional[str]
    paragraphs: List[str]


@dataclass
class DebateNode:
    heading: Optional[str]
    items: List[str]
    contributions: List[Contribution]
    subdebates: List["DebateNode"]
    depth: int


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------
def make_scraper() -> CloudScraper:
    s = cloudscraper.create_scraper(browser={"browser": "firefox", "platform": "linux", "mobile": False})
    s.headers.update({"User-Agent": "ModernHansardScraper/1.0 (academic use)"})
    return s


def get_with_retry(s: CloudScraper, url: str, *, timeout: int = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES, sleep: float = DEFAULT_SLEEP) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            resp = s.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logging.warning("GET failed (%s/%s) %s → %s", attempt, retries, url, e)
            if attempt == retries:
                raise
            time.sleep(sleep)
    raise RuntimeError("unreachable")  # pragma: no cover


# -----------------------------------------------------------------------------
# Step 1 – find available sitting dates
# -----------------------------------------------------------------------------
def month_url(chamber: str, year: int, month: int) -> str:
    # URL pattern is /{Chamber}/{YYYY}-{MM}-01
    return f"{BASE_URL}/{chamber}/{year}-{month:02d}-01"


def extract_dates_from_calendar(html: str) -> List[str]:
    """Return dates (YYYY-MM-DD) that have business from a month calendar."""
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("div.d-none.d-lg-block table.calendar-grid a.day-link")
    out: List[str] = []
    for a in links:
        aria = a.get("aria-label", "")
        # Typically contains "has business"
        if "has business" in aria.lower():
            try:
                # Formats like: "Wednesday 11 March 2020"
                date_str = aria.split(".")[-1].strip()
                dt = datetime.strptime(date_str, "%A %d %B %Y")
                out.append(dt.strftime("%Y-%m-%d"))
            except Exception:
                # Fallback: try parsing any date-like tail
                pass
    # de-dup while preserving order
    return list(dict.fromkeys(out))


def get_available_dates(
    scraper: CloudScraper,
    chamber: str,
    start_year: int,
    end_year: int,
    sleep: float,
) -> List[str]:
    all_dates: List[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = month_url(chamber, year, month)
            try:
                resp = get_with_retry(scraper, url)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (404, 410):
                    logging.info("No page for %s-%02d (%s)", year, month, e.response.status_code)
                else:
                    logging.warning("Skipping %s-%02d due to error: %s", year, month, e)
                continue
            dates = extract_dates_from_calendar(resp.text)
            logging.info("Found %d dates in %s-%02d", len(dates), year, month)
            all_dates.extend(dates)
            time.sleep(sleep)
    # unique preserve order
    return list(dict.fromkeys(all_dates))


# -----------------------------------------------------------------------------
# Step 2 – extract HTML download links for each date
# -----------------------------------------------------------------------------
def day_url(chamber: str, date_str: str) -> str:
    # /Commons/YYYY-MM-DD or /Lords/YYYY-MM-DD
    return f"{BASE_URL}/{chamber}/{date_str}"


def extract_html_download_links(soup: BeautifulSoup) -> List[str]:
    """
    Try multiple strategies to find "HTML Downloads" links.
    Returns absolute URLs.
    """
    links: List[str] = []

    # Strategy A: explicit dropdown labeled "HTML Downloads"
    label_div = soup.find(lambda tag: tag.name == "div" and tag.get_text(strip=True) == "HTML Downloads")
    if label_div:
        menu = label_div.find_parent("div", class_="dropdown-menu")
        if menu:
            for a in menu.select("a.dropdown-item[href]"):
                href = a.get("href", "")
                if href.lower().endswith(".html") or "html" in a.get_text(" ", strip=True).lower():
                    links.append(BASE_URL + href if href.startswith("/") else href)

    # Strategy B: any dropdown-item with 'html' in text or href
    if not links:
        for a in soup.select("a.dropdown-item[href]"):
            text = a.get_text(" ", strip=True).lower()
            href = a.get("href", "").lower()
            if "html" in text or href.endswith(".html") or "/html" in href:
                full = a.get("href")
                if full:
                    links.append(BASE_URL + full if full.startswith("/") else full)

    # de-dup preserve order
    dedup: List[str] = []
    for u in links:
        if u not in dedup:
            dedup.append(u)
    return dedup


# -----------------------------------------------------------------------------
# Step 3 – parse a debate page (hierarchical)
# -----------------------------------------------------------------------------
def parse_debate_page(url: str, scraper: CloudScraper, sleep: float) -> Optional[List[Dict[str, Any]]]:
    resp = get_with_retry(scraper, url)
    soup = BeautifulSoup(resp.text, "html.parser")
    root = soup.find("div", class_="child-debate-list")
    if not root:
        return None

    def parse_one(div: BeautifulSoup, depth: int) -> DebateNode:
        # Heading
        heading_el = div.find(["h2", "h3"], recursive=False)
        heading = heading_el.get_text(strip=True) if heading_el else None

        # Direct debate items (info lines)
        info_lines: List[str] = []
        direct_divs = [d for d in div.find_all("div", recursive=False) if "debate-item" in (d.get("class") or [])]
        for item in direct_divs:
            # skip ones that are contributions
            if item.find("div", class_="contribution"):
                continue
            for p in item.find_all("p"):
                txt = p.get_text(strip=True)
                if txt:
                    info_lines.append(txt)

        # Contributions
        contributions: List[Contribution] = []
        for item in direct_divs:
            c = item.find("div", class_="contribution")
            if not c:
                continue

            a = c.select_one("a.attributed-to-details")
            speaker_url = None
            if a:
                # Try to assemble a more robust label (primary + secondary)
                primary = a.find("div", class_="primary-text")
                secondary = a.find("div", class_="secondary-text")
                if primary and secondary:
                    speaker = (primary.get_text(strip=True) + " — " + secondary.get_text(strip=True)).strip()
                else:
                    speaker = a.get_text(" ", strip=True)
                if a.get("href"):
                    href = a.get("href")
                    speaker_url = BASE_URL + href if href.startswith("/") else href
            else:
                speaker = "UNKNOWN"

            paras: List[str] = []
            content = c.find("div", class_="content")
            if content:
                for p in content.find_all("p"):
                    txt = p.get_text(strip=True)
                    if txt:
                        paras.append(txt)

            contributions.append(Contribution(speaker=speaker, speaker_url=speaker_url, paragraphs=paras))

        # Subdebates
        subdebates: List[DebateNode] = []
        for sub_list in div.find_all("div", class_="child-debate-list", recursive=False):
            for sub in sub_list.find_all("div", class_="child-debate", recursive=False):
                subdebates.append(parse_one(sub, depth + 1))

        return DebateNode(
            heading=heading,
            items=info_lines,
            contributions=contributions,
            subdebates=subdebates,
            depth=depth,
        )

    top_nodes: List[DebateNode] = []
    for d in root.find_all("div", class_="child-debate", recursive=False):
        top_nodes.append(parse_one(d, depth=0))

    time.sleep(sleep)  # politeness
    # Convert to plain dicts for JSON
    def node_to_dict(n: DebateNode) -> Dict[str, Any]:
        return {
            "heading": n.heading,
            "items": n.items,
            "contributions": [asdict(c) for c in n.contributions],
            "subdebates": [node_to_dict(x) for x in n.subdebates],
            "depth": n.depth,
        }

    return [node_to_dict(n) for n in top_nodes]


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(
    chamber: str,
    start_year: int,
    end_year: int,
    dates_file: str,
    out_file: str,
    sleep: float,
    timeout: int,
    retries: int,
    reuse_dates: bool,
) -> None:
    scraper = make_scraper()

    # Dates
    if reuse_dates:
        with open(dates_file, "r", encoding="utf-8") as f:
            available_dates = json.load(f)
        logging.info("Loaded %d dates from %s", len(available_dates), dates_file)
    else:
        available_dates = get_available_dates(scraper, chamber, start_year, end_year, sleep)
        with open(dates_file, "w", encoding="utf-8") as f:
            json.dump(available_dates, f, indent=2)
        logging.info("Saved %d dates → %s", len(available_dates), dates_file)

    # For each date: collect debate HTML links
    results: Dict[str, List[str]] = {}
    for date in available_dates:
        url = day_url(chamber, date)
        try:
            resp = get_with_retry(scraper, url, timeout=timeout, retries=retries, sleep=sleep)
        except requests.RequestException as e:
            logging.warning("Skipping date %s due to error: %s", date, e)
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        links = extract_html_download_links(soup)
        results[date] = links
        logging.info("[%s] %d HTML links", date, len(links))
        time.sleep(sleep)

    # Parse each debate URL into structure
    all_data: Dict[str, List[Dict[str, Any]]] = {}
    for date, links in results.items():
        all_data[date] = []
        for url in links:
            logging.info("Parsing %s", url)
            try:
                struct = parse_debate_page(url, scraper, sleep)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    logging.info("  404 Not Found → %s (skipping)", url)
                    continue
                logging.warning("  HTTP error on %s: %s", url, e)
                continue
            except Exception as e:
                logging.warning("  Error on %s: %s", url, e)
                continue
            all_data[date].append({"url": url, "data": struct})

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %d days → %s", len(all_data), out_file)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape Modern Hansard debates (structure, not PDFs).")
    p.add_argument("--chamber", default="Commons", choices=["Commons", "Lords"], help="Parliamentary chamber")
    p.add_argument("--start-year", type=int, required=True, help="Start year (inclusive)")
    p.add_argument("--end-year", type=int, required=True, help="End year (inclusive)")
    p.add_argument("--dates-file", default="modern_available_dates.json", help="Checkpoint JSON for dates")
    p.add_argument("--out", dest="out_file", default="modern_hansard_data.json", help="Output JSON file")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Seconds to sleep between requests")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout (seconds)")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="HTTP retries")
    p.add_argument("--reuse-dates", action="store_true", help="Reuse existing dates file if present")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    run(
        chamber=args.chamber,
        start_year=args.start_year,
        end_year=args.end_year,
        dates_file=args.dates_file,
        out_file=args.out_file,
        sleep=args.sleep,
        timeout=args.timeout,
        retries=args.retries,
        reuse_dates=args.reuse_dates,
    )


if __name__ == "__main__":
    main()
