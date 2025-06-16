#!/usr/bin/env python3
"""
scraper_HistoricalHansard_step1.py

Fetches all Hansard sittings from 1802-01-01 to 2006-01-01 and writes out
a JSON checkpoint file ("historical_step1.json") containing, for each day:
  {
    "YYYY-MM-DD": {
      "url": "...",
      "chambers": {
        "Commons": [...],
        "Lords": [...],
        ...
      }
    },
    ...
  }

This script only performs Step 1 (structure extraction) and does not
enrich leaf sections or write the full content.
"""

import datetime
import calendar
import json
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin

BASE_FMT = "https://api.parliament.uk/historic-hansard/sittings/{year}/{mon}/{day}"
START_DATE = datetime.date(1802, 1, 1)
END_DATE = datetime.date(2006, 1, 1)
CHECKPOINT_FILE = "historical_step1.json"


def next_relevant(tag):
    """Return the next Tag sibling, skipping whitespace & column refs."""
    sib = tag.next_sibling
    while sib and (not isinstance(sib, Tag) or
                   (sib.name == "span" and "section-column-reference" in sib.get("class", []))):
        sib = sib.next_sibling
    return sib


def parse_list(ol_tag, base_url):
    """
    Recursively parse an <ol> of sections. Returns a list of dicts:
      {
        "title": str,
        "url": str,
        "column": str|None,
        "word_length": str|None,
        "subsections": [ ...same structure... ]
      }
    """
    sections = []
    if not ol_tag:
        return sections

    for li in ol_tag.find_all("li", recursive=False):
        link = li.select_one("span.section-link a")
        if not link:
            continue

        title = link.get_text(strip=True)
        url = urljoin(base_url, link["href"])

        # find preceding column reference
        col = None
        prev = li.find_previous_sibling()
        while prev and not (prev.name == "span" and
                            "section-column-reference" in prev.get("class", [])):
            prev = prev.find_previous_sibling()
        if prev:
            col = prev.get_text(strip=True)

        # optional word length
        wl = li.select_one("span.section-word-length")
        word_length = wl.get_text(strip=True) if wl else None

        # nested <ol>?
        child_ol = None
        nr = next_relevant(li)
        if nr and nr.name == "ol":
            child_ol = nr

        sections.append({
            "title":       title,
            "url":         url,
            "column":      col,
            "word_length": word_length,
            "subsections": parse_list(child_ol, base_url)
        })
    return sections


def main():
    all_data = {}
    last_decade = None
    curr = START_DATE
    delta = datetime.timedelta(days=1)

    while curr <= END_DATE:
        decade = (curr.year // 10) * 10
        if decade != last_decade:
            print(f"… checkpoint {decade}s …")
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2)
            last_decade = decade

        url = BASE_FMT.format(
            year=curr.year,
            mon=calendar.month_abbr[curr.month].lower(),
            day=curr.day
        )
        try:
            resp = requests.get(url, timeout=20)
        except requests.RequestException:
            curr += delta
            continue

        text = resp.text if resp.status_code == 200 else ""
        if resp.status_code == 200 and "Page not found" not in text:
            soup = BeautifulSoup(text, "html.parser")
            day_str = curr.isoformat()
            day_info = {"url": url, "chambers": {}}

            for h3 in soup.find_all("h3"):
                chamber = h3.get_text(" ", strip=True)
                top_ol = None
                for sib in h3.next_siblings:
                    if isinstance(sib, Tag) and sib.name == "ol":
                        top_ol = sib
                        break
                day_info["chambers"][chamber] = parse_list(top_ol, url)

            all_data[day_str] = day_info

        curr += delta

    # final write
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    print(f"✅ Wrote {CHECKPOINT_FILE} with {len(all_data)} days")


if __name__ == "__main__":
    main()
