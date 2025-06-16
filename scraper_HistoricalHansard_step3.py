"""
scraper_HistoricalHansard_step3.py

This script scrapes UK Parliament Hansard speaker index and detail pages,
building a JSON list of speaker profiles including:
  - name and life dates
  - constituencies served and date ranges
  - other listed roles or contributions with associated dates

It performs two main steps:
  1. fetch_all_records(): crawl alphabetical index pages to collect speaker names,
     dates text, and profile URLs for all Hansard-indexed persons.
  2. parse_speaker_page(): load each speaker's detail page to extract structured
     information under each <h2> heading, splitting content by optional date-range
     suffix.

Output:
  - speaker_details.json: array of speaker profile dicts

Dependencies:
  - requests
  - beautifulsoup4

Configuration:
  - HEADERS: custom User-Agent header for HTTP requests
  - DATE_RANGE_RE: regex to separate content text from trailing date or date-range
"""

import re
import json
import string
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

# --- Custom headers ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; hansard-scraper/0.1; +https://github.com/centre-for-care)"
}

# --- Regex for splitting content + date ranges ---
DATE_RANGE_RE = re.compile(
    r"^(?P<content>.+?)\s+"
    r"(?P<dates>"
      r"(?:[A-Za-z]+\s+\d{1,2},\s*\d{4})"
      r"(?:\s*[-–]\s*(?:[A-Za-z]+\s+\d{1,2},\s*\d{4})?)?"
    r")$"
)

def split_content_and_dates(text):
    m = DATE_RANGE_RE.match(text)
    if m:
        return m.group("content"), m.group("dates")
    return text, None

def parse_speaker_page(session, person_url):
    resp = session.get(person_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    data = {"url": person_url}  # ← include the URL here
    h1 = soup.find("h1", class_="vcard")
    if not h1:
        raise ValueError("Missing <h1 class='vcard'> on " + person_url)

    data["name"]  = h1.get_text(" ", strip=True)
    data["dates"] = h1.next_sibling.strip()

    for h2 in soup.find_all("h2"):
        title = h2.get_text(" ", strip=True)
        key   = title.lower().replace(" ", "_")
        if key == "contributions":
            continue

        entries = []
        for sib in h2.next_siblings:
            if isinstance(sib, Tag) and sib.name == "h2":
                break

            if isinstance(sib, Tag) and sib.name in ("ol", "ul"):
                for li in sib.find_all("li"):
                    text = li.get_text(" ", strip=True)
                    if key == "constituencies":
                        name      = li.find("a").get_text(strip=True)
                        dates_txt = text.replace(name, "").strip()
                        entries.append({
                            "constituency": name,
                            "dates": dates_txt
                        })
                    else:
                        content, dates = split_content_and_dates(text)
                        entries.append({
                            "content": content,
                            "dates": dates
                        })

            elif isinstance(sib, Tag) and sib.name == "p":
                text = sib.get_text(" ", strip=True)
                if text:
                    content, dates = split_content_and_dates(text)
                    entries.append({
                        "content": content,
                        "dates": dates
                    })

        if entries:
            data[key] = entries

    return data

def fetch_all_records(session):
    INDEX_FMT = "https://api.parliament.uk/historic-hansard/people/{}"
    records   = []
    for letter in (c for c in string.ascii_lowercase if c != "x"):
        url = INDEX_FMT.format(letter)
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for li in soup.select("li"):
                a = li.find("a", href=True)
                if not a:
                    continue
                name       = a.get_text(strip=True)
                person_url = requests.compat.urljoin(url, a["href"])
                dates_txt  = li.get_text(" ", strip=True).replace(name, "").strip()
                records.append((name, dates_txt, person_url))
        except Exception as e:
            print(f"Failed to fetch index for '{letter}': {e}")
    return records

def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    print("➡️  Fetching index pages…")
    records = fetch_all_records(session)
    print(f"🔎  Found {len(records)} speakers. Fetching details…")

    speaker_details = []
    for idx, (_, _, url) in enumerate(records, 1):
        try:
            detail = parse_speaker_page(session, url)
            detail["url"] = url
            speaker_details.append(detail)
        except Exception as e:
            print(f"[{idx}/{len(records)}] Error parsing {url}: {e}")

    print("💾  Writing to speaker_details.json")
    with open("speaker_details.json", "w", encoding="utf-8") as f:
        json.dump(speaker_details, f, ensure_ascii=False, indent=2)

    print("✅  Done.")

if __name__ == "__main__":
    main()
