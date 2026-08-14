"""
scraper_historical_hansard_02_enrich_sections.py

This script loads a precomputed JSON index of Hansard sittings (historical_step1.json),
then fetches and enriches each section's full text (member contributions and procedural
paragraphs) for sittings within a specified year range.

It processes years in ascending order, retries on failures, and checkpoints output
after each year. The enriched data is written to:
  - historical_hansard_full.json
  - failed_urls.json

Configuration:
  * START_YEAR / END_YEAR: inclusive range of years to process
  * INPUT_FILE: path to the JSON index from step 1
  * OUTPUT_FILE: path for enriched data
  * FAILED_FILE: path for recording permanently failed URLs

Dependencies:
  - requests
  - beautifulsoup4

"""

import re
import time
import json
import argparse
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────

# Years to scrape (inclusive)
START_YEAR = 1950
END_YEAR   = 2007

# Input / output paths
INPUT_FILE   = "historical_step1.json"
OUTPUT_FILE  = f"historical_hansard_full.json"
FAILED_FILE  = f"failed_urls.json"

# HTTP
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; hansard-scraper/0.1; +https://github.com/centre-for-care)"
}
session = requests.Session()
session.headers.update(HEADERS)

# Regex for splitting content + date‐ranges (unused here but kept for reference)
DATE_RANGE_RE = re.compile(
    r"^(?P<content>.+?)\s+"
    r"(?P<dates>"
      r"(?:[A-Za-z]+\s+\d{1,2},\s*\d{4})"
      r"(?:\s*[-–]\s*(?:[A-Za-z]+\s+\d{1,2},\s*\d{4})?)?"
    r")$"
)

# Accumulate permanently failed URLs
failed_urls = []

# ─── UTILITIES ─────────────────────────────────────────────────────────────────

def split_content_and_dates(text):
    """
    If text ends with “<content> <Month Day, Year>” (optionally date-range),
    return (content, dates). Otherwise return (text, None).
    """
    m = DATE_RANGE_RE.match(text)
    if m:
        return m.group("content"), m.group("dates")
    return text, None

def next_relevant(tag):
    """
    Given a BeautifulSoup Tag, skip over text nodes and
    any <span class="section-column-reference">…</span> siblings,
    returning the next sibling that is a “real” Tag.
    """
    sib = tag.next_sibling
    while sib and (not isinstance(sib, Tag) or
                   (sib.name == "span" and
                    "section-column-reference" in sib.get("class", []))):
        sib = sib.next_sibling
    return sib

# ─── PARSERS ───────────────────────────────────────────────────────────────────

def parse_list(ol_tag, base_url):
    """
    Parse an <ol> of <li> entries, each representing a “section” with:
      - title
      - URL
      - column reference (if any)
      - word length (if any)
      - nested subsections
    Returns a list of dicts, where each dict has keys:
      'title', 'url', 'column', 'word_length', 'subsections'.
    """
    sections = []
    if not ol_tag:
        return sections

    for li in ol_tag.find_all("li", recursive=False):
        link = li.select_one("span.section-link a")
        if not link:
            continue

        title       = link.get_text(strip=True)
        section_url = urljoin(base_url, link["href"])

        # column ref (if any)
        col = None
        prev = li.previous_sibling
        while prev and not (prev.name == "span" and 
                            "section-column-reference" in prev.get("class", [])):
            prev = prev.previous_sibling
        if prev:
            col = prev.get_text(strip=True)

        # word length (if any)
        wl = li.select_one("span.section-word-length")
        word_length = wl.get_text(strip=True) if wl else None

        # nested subsections?
        child_ol = next_relevant(li)
        subsections = []
        if child_ol and child_ol.name == "ol":
            subsections = parse_list(child_ol, base_url)

        sections.append({
            "title": title,
            "url": section_url,
            "column": col,
            "word_length": word_length,
            "subsections": subsections
        })
    return sections

def fetch_section_text(section_url, retries=5, delay=1, pause=5):
    """
    Fetch the raw HTML at `section_url`, parse member contributions and/or
    fallback paragraphs, returning a list of dicts:
      {
        "speaker": ...,
        "speaker_url": ...,
        "speech": ...
      }
    If the page contains "Page not found" (case‐insensitive), or if it fails
    after two phases of retries, records `section_url` into the global
    `failed_urls` and returns an empty list.
    """
    # ─── Phase 1: up to `retries` attempts, small delays ────────────────────────
    for i in range(retries):
        try:
            r = session.get(section_url, timeout=5)
            r.raise_for_status()
            break
        except Exception as e:
            print(f"[Phase1 {i+1}/{retries}] {e} — {section_url}")
            time.sleep(delay)
    else:
        # ─── Phase 2: pause, then try `retries` more times ───────────────────────
        print(f"Phase1 failed for {section_url}, pausing {pause}s…")
        time.sleep(pause)
        for i in range(retries):
            try:
                r = session.get(section_url, timeout=5)
                r.raise_for_status()
                break
            except Exception as e:
                print(f"[Phase2 {i+1}/{retries}] {e} — {section_url}")
                time.sleep(delay)
        else:
            print(f"❌ Giving up on {section_url}")
            failed_urls.append(section_url)
            return []

    # ─── At this point, `r` holds a successful response ────────────────────────
    soup = BeautifulSoup(r.text, "html.parser")

    # ─── Check for “Page not found” anywhere in the rendered text ──────────────
    page_text = soup.get_text(separator="\n").strip()
    if re.search(r"Page\s+not\s+found", page_text, re.IGNORECASE):
        print(f"⚠️ Page not found detected at {section_url}")
        failed_urls.append(section_url)
        return []

    entries = []

    # ─── 1) Try extracting “member_contribution” blocks ───────────────────────
    for contrib in soup.select('div.hentry.member_contribution'):
        cite = contrib.select_one('cite.member.author.entry-title')
        if not cite:
            continue

        a = cite.find('a')
        speaker     = a.get_text(strip=True) if a else cite.get_text(strip=True)
        speaker_url = urljoin(section_url, a['href']) if a else ""
        block       = contrib.find('blockquote', class_='contribution_text')
        paras       = block.find_all('p') if block else []
        speech      = "\n\n".join(p.get_text(strip=True) for p in paras).strip()

        if speech:
            entries.append({
                "speaker": speaker,
                "speaker_url": speaker_url,
                "speech": speech
            })

    # ─── 2) Fallback: any <p> inside #content (excluding member_contribution) ──
    content_div = soup.find(id='content') or soup.find(class_='house-of-commons-sitting')
    if content_div:
        for p in content_div.find_all('p'):
            if p.find_parent('div', class_='member_contribution'):
                # skip paragraphs already captured above
                continue
            txt = p.get_text(strip=True)
            if txt:
                entries.append({
                    "speaker": "",
                    "speaker_url": "",
                    "speech": txt
                })

    return entries

def enrich_leaves(sections):
    """
    Recursively descend into `sections` (list of dicts). If a section has
    "subsections", recurse. Otherwise, fetch its content and store it under
    sec["content"] as a list of {speaker, speaker_url, speech}.
    """
    for sec in sections:
        if sec.get("subsections"):
            enrich_leaves(sec["subsections"])
        else:
            sec["content"] = fetch_section_text(sec["url"])

# ─── MAIN ────────────────────────────────────────────────────────────────────────

def main(start_year, end_year):
    # 1) Load everything from step1
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 2) Filter to only those days in the desired range [start_year, end_year]
    filtered = {
        day: all_data[day]
        for day in all_data
        if start_year <= int(day.split('-')[0]) <= end_year
    }

    # 3) Build a year→list_of_days mapping
    years = {}
    for day in filtered:
        yr = int(day.split('-')[0])
        years.setdefault(yr, []).append(day)

    # 4) Iterate through each year in ascending order
    for year in sorted(years):
        print(f"🕰️  Processing {year}…")
        for day in sorted(years[year]):
            day_info = filtered[day]
            for chamber, sections in day_info.get("chambers", {}).items():
                print(f"  • {day} / {chamber}")
                enrich_leaves(sections)

        # 5) After finishing this entire year, save both the “full” JSON and the failed-URLs JSON
        print(f"  ✔️  Finished {year}; saving output…")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # We dump the entire `filtered` dict (i.e. everything processed so far, up through this year).
            json.dump(filtered, f, indent=2)

        with open(FAILED_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_urls, f, indent=2)

        print(f"  ✅  Saved up through {year}.")

    print("✅ All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year",   type=int, default=END_YEAR)
    args, _ = parser.parse_known_args()
    main(args.start_year, args.end_year)
