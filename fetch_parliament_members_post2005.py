#!/usr/bin/env python3
"""
fetch_parliament_members_post2005.py

Fetches member records from the UK Parliament API (https://members-api.parliament.uk/)
and writes successful JSON payloads and errors to separate files.
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Default configuration
DEFAULT_START_ID = 1
DEFAULT_END_ID = 5402
DEFAULT_BATCH_SIZE = 1000
DEFAULT_RATE_LIMIT_SLEEP = 0.0
BASE_URL = "https://members-api.parliament.uk/api/Members/"


def setup_logging() -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_session(
    retries: int = 3, backoff_factor: float = 0.3, status_forcelist: List[int] = None
) -> requests.Session:
    """
    Create a requests.Session with retry logic.

    :param retries: number of retry attempts
    :param backoff_factor: factor to apply between retries
    :param status_forcelist: HTTP status codes to retry on
    :return: configured Session
    """
    session = requests.Session()
    status_forcelist = status_forcelist or [500, 502, 503, 504]
    retry_strategy = Retry(
        total=retries,
        status_forcelist=status_forcelist,
        backoff_factor=backoff_factor,
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_members(
    session: requests.Session,
    start_id: int,
    end_id: int,
    batch_size: int,
    rate_limit_sleep: float,
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    Iterate member IDs and fetch from the API.

    :param session: HTTP session
    :param start_id: first member ID to fetch
    :param end_id: last member ID to fetch (inclusive)
    :param batch_size: log progress every batch_size IDs
    :param rate_limit_sleep: delay between requests
    :return: tuple of (successes, errors)
    """
    successes: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for member_id in range(start_id, end_id + 1):
        url = f"{BASE_URL}{member_id}"
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                successes.append(resp.json())
            elif resp.status_code == 404:
                errors.append({"MemberId": member_id, "status": 404, "note": "Not found"})
            else:
                errors.append(
                    {
                        "MemberId": member_id,
                        "status": resp.status_code,
                        "text": resp.text[:200],
                    }
                )
        except requests.RequestException as exc:
            errors.append({"MemberId": member_id, "error": str(exc)})

        if rate_limit_sleep:
            time.sleep(rate_limit_sleep)

        if member_id % batch_size == 0:
            logging.info(
                "Processed up to Member ID %d — %d successes, %d errors",
                member_id,
                len(successes),
                len(errors),
            )

    return successes, errors


def save_json(data: Any, filename: str) -> None:
    """
    Write Python object as pretty JSON.

    :param data: object to serialize
    :param filename: output file path
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info("Wrote %s (%d items)", filename, len(data) if isinstance(data, list) else 0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch UK Parliament members data.")
    parser.add_argument(
        "--start-id",
        type=int,
        default=DEFAULT_START_ID,
        help="First Member ID to retrieve (inclusive).",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=DEFAULT_END_ID,
        help="Last Member ID to retrieve (inclusive).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="How often to log progress.",
    )
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=DEFAULT_RATE_LIMIT_SLEEP,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="parliament_members",
        help="Prefix for output JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    setup_logging()
    args = parse_args()

    logging.info(
        "Starting fetch: IDs %d to %d", args.start_id, args.end_id
    )
    session = create_session()

    successes, errors = fetch_members(
        session,
        args.start_id,
        args.end_id,
        args.batch_size,
        args.rate_limit_sleep,
    )

    success_file = f"{args.output_prefix}_full_responses.json"
    error_file = f"{args.output_prefix}_errors.json"

    save_json(successes, success_file)
    save_json(errors, error_file)

    logging.info(
        "Finished. Retrieved %d members, %d errors.",
        len(successes),
        len(errors),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Unhandled exception")
        sys.exit(1)
