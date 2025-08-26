#!/usr/bin/env python3
"""
Hansard Scraper

This script scrapes Modern Hansard parliamentary debate data from https://hansard.parliament.uk/. Users only need to set 
the START_YEAR and END_YEAR values to run the entire pipeline.

The script performs the following tasks:
  1. Scrapes the Hansard calendar to get all available business dates between 
     the given years.
  2. Saves available dates to a JSON file.
  3. For each available date, extracts the HTML download links for that day.
  4. For each section URL, parses the hierarchical debate structure including:
       - Section headings (e.g., "House of Commons", "Prayers", "Cabinet Office", etc.)
       - Info lines (e.g., "Thursday 6 March 2025", "The House met at half-past Nine o’clock")
       - Speaker contributions (with speaker names and text paragraphs)
  5. Saves the full scraped data to a JSON file.

Usage:
    Simply edit the START_YEAR and END_YEAR constants as needed, then run:
        python hansard_scraper.py
"""

import time
import json
from typing import List, Dict, Optional, Any, Tuple
import cloudscraper
from cloudscraper import CloudScraper, create_scraper
from bs4 import BeautifulSoup
from datetime import datetime
import requests


# -----------------------------
# Configuration: Only modify these
START_YEAR: int = 2020
END_YEAR: int = 2021

# Output file names
AVAILABLE_DATES_FILE: str = "available_dates.json"
OUTPUT_FILE: str = "hansard_data.json"
# -----------------------------

# Internal constants
BASE_URL: str = "https://hansard.parliament.uk"
SLEEP_TIME: int = 3  # seconds between requests

def get_available_dates(scraper: CloudScraper, start_year: int, end_year: int) -> List[str]:
    """
    Scrape each month's first day page to collect all available business dates.
    
    Args:
        scraper: A CloudScraper instance.
        start_year: The starting year (inclusive).
        end_year: The ending year (inclusive).
    
    Returns:
        A list of date strings formatted as 'YYYY-MM-DD'.
    """
    all_dates: List[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url: str = f"{BASE_URL}/Commons/{year}-{month:02d}-01"
            response = scraper.get(url)
            if not response.ok:
                print(f"  -> No page for {year}-{month:02d} (status {response.status_code}). Skipping.")
                continue
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.select('div.d-none.d-lg-block table.calendar-grid a.day-link')
            for link in links:
                aria_label = link.get("aria-label", "")
                if "has business" in aria_label.lower():
                    try:
                        # Example: "Wednesday 11 March 2020"
                        date_str = aria_label.split('.')[-1].strip()
                        date_obj = datetime.strptime(date_str, "%A %d %B %Y")
                        formatted = date_obj.strftime("%Y-%m-%d")
                        all_dates.append(formatted)
                    except Exception as e:
                        print("Error parsing date from:", aria_label, e)
            time.sleep(SLEEP_TIME)
    # Remove duplicates while preserving order.
    return list(dict.fromkeys(all_dates))


def save_available_dates(dates: List[str], filename: str = AVAILABLE_DATES_FILE) -> None:
    """
    Save the list of available dates to a JSON file.
    
    Args:
        dates: List of date strings.
        filename: Output file name.
    """
    with open(filename, 'w') as f:
        json.dump(dates, f, indent=2)


def load_available_dates(filename: str = AVAILABLE_DATES_FILE) -> List[str]:
    """
    Load available dates from a JSON file.
    
    Args:
        filename: The JSON file containing the dates.
    
    Returns:
        A list of date strings.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def extract_html_download_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """
    Extract all HTML download links from the "HTML Downloads" section.
    
    Args:
        soup: BeautifulSoup-parsed HTML of the page.
        base_url: The base URL for constructing full links.
    
    Returns:
        A list of full URLs for HTML downloads.
    """
    links: List[str] = []
    html_header = soup.find("div", string="HTML Downloads")
    if not html_header:
        return links
    dropdown_div = html_header.find_parent("div", class_="dropdown-menu")
    if not dropdown_div:
        return links
    found_section = False
    for child in dropdown_div.find_all(["div", "a"], recursive=False):
        if child.name == "div" and child.get_text(strip=True) == "HTML Downloads":
            found_section = True
            continue
        if found_section:
            if child.name == "div" and "dropdown-header" in child.get("class", []):
                break
            if child.name == "a" and "dropdown-item" in child.get("class", []):
                href = child.get("href")
                if href:
                    links.append(base_url + href)
    return links


def parse_debate_page(url: str, scraper: CloudScraper) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a debate page and return a hierarchical data structure.
    
    Each node is a dictionary with the following keys:
      - "heading": Heading text (from a direct h2/h3 element).
      - "items": List of info lines (e.g. dates, times, and other metadata).
      - "contributions": List of speaker contributions (each a dict with keys "speaker" and "text").
      - "subdebates": List of child nodes (subsections).
      - "depth": The hierarchical level (integer).
       
    Args:
        url: The URL of the debate page.
        scraper: A CloudScraper instance.
    
    Returns:
        A list of nodes representing the debate structure, or None if not found.
    """
    response = scraper.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    root_list = soup.find("div", class_="child-debate-list")
    if not root_list:
        return None

    # DFS stack: each element is (current_div, depth, parent_node, node_dict)
    stack: List[Tuple[BeautifulSoup, int, Optional[Dict[str, Any]], Dict[str, Any]]] = []
    structure: List[Dict[str, Any]] = []
    top_debates = root_list.find_all("div", class_="child-debate", recursive=False)
    for debate in reversed(top_debates):
        stack.append((debate, 0, None, {}))
    
    while stack:
        current_div, depth, parent, node = stack.pop()
        node["depth"] = depth
        
        # (A) Extract heading from direct h2/h3 elements.
        heading_el = current_div.find(["h2", "h3"], recursive=False)
        node["heading"] = heading_el.get_text(strip=True) if heading_el else None
        
        # (B) Extract info lines from direct children with "debate-item" in their class list.
        info_items: List[str] = []
        debate_items = [
            d for d in current_div.find_all("div", recursive=False)
            if d.get("class") and "debate-item" in d.get("class")
        ]
        for item in debate_items:
            if item.find("div", class_="contribution"):
                continue  # Skip those with speaker contributions.
            for p in item.find_all("p"):
                text = p.get_text(strip=True)
                if text:
                    info_items.append(text)
        node["items"] = info_items
        
        # (C) Extract speaker contributions.
        contributions: List[Dict[str, Any]] = []
        for item in debate_items:
            contrib = item.find("div", class_="contribution")
            if contrib:
                speaker_a = contrib.select_one("a.attributed-to-details")
                if speaker_a:
                    primary = speaker_a.find("div", class_="primary-text")
                    secondary = speaker_a.find("div", class_="secondary-text")
                    if primary and secondary:
                        speaker_name = primary.get_text(strip=True) + "\n" + secondary.get_text(strip=True)
                    else:
                        speaker_name = speaker_a.get_text(strip=True)
                    speaker_name = speaker_name.strip()
                else:
                    speaker_name = "UNKNOWN"
                paras: List[str] = []
                content_div = contrib.find("div", class_="content")
                if content_div:
                    for p in content_div.find_all("p"):
                        ptext = p.get_text(strip=True)
                        if ptext:
                            paras.append(ptext)
                contributions.append({
                    "speaker": speaker_name,
                    "text": paras
                })
        node["contributions"] = contributions
        node["subdebates"] = []
        
        if parent is not None:
            parent.setdefault("subdebates", []).append(node)
        else:
            structure.append(node)
        
        # (D) Process nested child-debate-list nodes.
        sub_lists = current_div.find_all("div", class_="child-debate-list", recursive=False)
        for sub_list in sub_lists:
            sub_debates = sub_list.find_all("div", class_="child-debate", recursive=False)
            for sub_debate in reversed(sub_debates):
                stack.append((sub_debate, depth + 1, node, {}))
    
    return structure


def main() -> None:
    scraper: CloudScraper = create_scraper(browser="firefox")
    
    # Step 1: Get available dates and save them.
    available_dates: List[str] = get_available_dates(scraper, START_YEAR, END_YEAR)
    save_available_dates(available_dates, AVAILABLE_DATES_FILE)
    print("Available dates saved to", AVAILABLE_DATES_FILE)
    
    # Step 2: For each available date, extract HTML download links.
    available_dates = load_available_dates(AVAILABLE_DATES_FILE)
    
    results: Dict[str, List[str]] = {}
    for date in available_dates:
        day_url: str = f"{BASE_URL}/Commons/{date}"
        try:
            response = scraper.get(day_url)
        except Exception as e:
            print(f"  -> Error fetching {day_url}: {e}")
            continue
        if not response.ok:
            print(f"  -> Failed to get data for {date} (status {response.status_code}). Skipping.")
            continue
        soup = BeautifulSoup(response.content, "html.parser")
        html_links: List[str] = extract_html_download_links(soup, BASE_URL)
        results[date] = html_links
        time.sleep(SLEEP_TIME)
    
    # Step 3: For each date and URL, parse the debate page structure.
    all_data: Dict[str, List[Dict[str, Any]]] = {}
    for date, url_list in results.items():
        all_data[date] = []
        for url in url_list:
            print(f"Processing {url}...")
            try:
                debate_structure = parse_debate_page(url, scraper)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"Skipping {url} (404 Not Found).")
                    continue
                else:
                    raise
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
            all_data[date].append({
                "url": url,
                "data": debate_structure
            })
            time.sleep(SLEEP_TIME)
    
    # Save the full scraped data to a JSON file.
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f, indent=2)
    print("Scraped data saved to", OUTPUT_FILE)


if __name__ == '__main__':
    main()