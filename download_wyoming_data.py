#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: download_wyoming_data.py
Description: Download sounding data from the University of Wyoming website
             (https://weather.uwyo.edu/upperair/sounding.shtml).

Author: Guilherme Almeida dos Santos
ORCID: https://orcid.org/0009-0006-3696-3468
Lattes: http://lattes.cnpq.br/7666680077808755
Project/Paper: "The Record-Breaking Low-Level Jet Event in Subtropical South America during the Winter of 2026"
Repository: https://github.com/GuiMeteorology/record-breaking-llj-2026
"""

import argparse
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import requests

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
STATION_NUMBER = "83554"                 # Station ID (e.g., 83554 for Corumbá/MS, Brazil)
START_DATE = "2007-01-01"                # Default start date (YYYY-MM-DD)
END_DATE = "2026-07-21"                  # Default end date (YYYY-MM-DD)
OUTPUT_DIR = f"soundings_{STATION_NUMBER}"  # Default directory to save output files

BASE_URL = "https://weather.uwyo.edu/wsgi/sounding"
FORM_URL = "https://weather.uwyo.edu/upperair/sounding.shtml"

STANDARD_HOURS = [0, 12]                 # Standard radiosonde launch hours (UTC)
ALL_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]  # All synoptic hours offered by the website

# Regular expressions for parsing the HTML response
RE_STATION_HEADER = re.compile(r"Observations for Station\s+\S+", re.IGNORECASE)
RE_H1 = re.compile(r"<H1>(.*?)</H1>", re.IGNORECASE | re.DOTALL)
RE_H3 = re.compile(r"<H3>(.*?)</H3>", re.IGNORECASE | re.DOTALL)
RE_LATLON = re.compile(r"Latitude:\s*[-\d.]+\s*Longitude:\s*[-\d.]+", re.IGNORECASE)
RE_PRE_BLOCKS = re.compile(r"<PRE>(.*?)</PRE>", re.IGNORECASE | re.DOTALL)


def _clean_html_tags(text: str) -> str:
    """Remove residual HTML tags (e.g., <B>, <I>) while preserving text content."""
    return re.sub(r"<[^>]+>", "", text)


class InternetFailureError(RuntimeError):
    """
    Download error specifically attributable to local connectivity issues
    (e.g., connection refused/reset, timeout), meaning the request never
    received a server response.

    This does NOT include HTTP error status codes (400, 404, 500, etc.), as
    those are valid responses from the server and do not indicate a local network failure.
    """


# Request exception types indicating connectivity failures
CONNECTION_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

MAX_ATTEMPTS = 4
DELAY_BETWEEN_REQUESTS = 1.5  # Seconds, to avoid server rate-limiting
DELAY_BETWEEN_ATTEMPTS = 6.0  # Seconds, delay before retrying after a failure/timeout

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": FORM_URL,
    "Connection": "close",
}

_session = None
_session_lock = threading.Lock()


def get_session() -> requests.Session:
    """
    Creates and initializes a thread-safe HTTP session.

    Visits the sounding form page first to obtain session cookies expected by
    subsequent requests to the /wsgi/sounding endpoint. Without this, certain
    valid date/time queries return empty responses.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:  # Double-checked locking pattern
                new_session = requests.Session()
                new_session.headers.update(HEADERS)
                try:
                    new_session.get(FORM_URL, timeout=30)
                except requests.RequestException as e:
                    logging.warning("Could not warm up session by visiting %s: %s", FORM_URL, e)
                _session = new_session
    return _session


# Data source formats to attempt sequentially
CANDIDATE_SOURCES = ["BUFR", "FM35", "UNKNOWN"]


def build_url(day: date, hour: int, src: str) -> str:
    """
    Constructs the target URL for a specific sounding query.

    Query details required by the server:
    - The space in the 'datetime' parameter must be '%20' (not '+').
    - The hour component must NOT have a leading zero (e.g., '0:00:00' instead of '00:00:00').
    - The 'src' parameter is mandatory (BUFR, FM35, or UNKNOWN).
    """
    datetime_str = f"{day.isoformat()}%20{hour}:00:00"
    return f"{BASE_URL}?src={src}&datetime={datetime_str}&id={STATION_NUMBER}&type=TEXT:LIST"


def download_sounding(day: date, hour: int):
    """
    Attempts to download sounding data for a given date/hour, trying each candidate
    source format (BUFR, FM35, UNKNOWN) until valid data is retrieved.

    Returns:
        str: Raw sounding text if data exists.
        None: If no data is available for the given timestamp across all sources.

    Raises:
        InternetFailureError: If all candidate attempts failed due to local network connectivity.
        RuntimeError: If failures were caused by server-side errors or invalid responses.
    """
    network_errors = []  # Stores tuple entries: (src, error_type, message)

    for src in CANDIDATE_SOURCES:
        url = build_url(day, hour, src)
        text = _try_download_url(day, hour, src, url, network_errors)
        if text is not None:
            return text
        time.sleep(1)

    if network_errors and len(network_errors) == len(CANDIDATE_SOURCES):
        details = "; ".join(f"src={s}: {msg}" for s, _type, msg in network_errors)
        error_types = {err_type for _s, err_type, _msg in network_errors}
        
        if error_types == {"internet"}:
            raise InternetFailureError(
                f"Connectivity failure on {day.isoformat()} {hour:02d}Z (all sources): {details}"
            )
        raise RuntimeError(
            f"Persistent failure on {day.isoformat()} {hour:02d}Z (all sources): {details}"
        )

    return None


def _try_download_url(day: date, hour: int, src: str, url: str, network_errors: list):
    """Executes request retry loop for a specific URL and data source."""
    last_error = None
    last_error_type = "other"
    session = get_session()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=60)

            if resp.status_code in (400, 404):
                # Valid server response indicating no data exists for this specific source format.
                logging.info(
                    "src=%s has no data for %s %02dZ (HTTP %d) - trying next source",
                    src, day.isoformat(), hour, resp.status_code,
                )
                return None

            resp.raise_for_status()
            text = resp.text

            if RE_STATION_HEADER.search(text):
                parts = []

                m_h1 = RE_H1.search(text)
                if m_h1:
                    parts.append(_clean_html_tags(m_h1.group(1)).strip())

                m_h3 = RE_H3.search(text)
                if m_h3:
                    parts.append(_clean_html_tags(m_h3.group(1)).strip())

                m_latlon = RE_LATLON.search(text)
                if m_latlon:
                    parts.append(_clean_html_tags(m_latlon.group(0)).strip())

                pre_blocks = RE_PRE_BLOCKS.findall(text)
                for block in pre_blocks:
                    parts.append(block.strip())

                content = "\n\n".join(p for p in parts if p)
                return content if content.strip() else None
            else:
                logging.warning(
                    "src=%s returned HTTP 200 without station header for %s %02dZ. "
                    "Size: %d bytes. Response preview: %r",
                    src, day.isoformat(), hour, len(text), text[:300],
                )
                return None

        except CONNECTION_ERRORS as e:
            last_error = e
            last_error_type = "internet"
            logging.warning(
                "Possible network connection failure (attempt %d/%d) for %s %02dZ [src=%s]: %s",
                attempt, MAX_ATTEMPTS, day.isoformat(), hour, src, e,
            )
            time.sleep(DELAY_BETWEEN_ATTEMPTS)

        except requests.RequestException as e:
            last_error = e
            last_error_type = "other"
            logging.warning(
                "Error processing response (attempt %d/%d) for %s %02dZ [src=%s]: %s",
                attempt, MAX_ATTEMPTS, day.isoformat(), hour, src, e,
            )
            time.sleep(DELAY_BETWEEN_ATTEMPTS)

    network_errors.append((src, last_error_type, str(last_error)))
    return None


def save_sounding(output_dir: str, day: date, hour: int, text: str) -> str:
    """Saves sounding text content to a structured directory hierarchy (YYYY/YYYYMMDD_HHZ.txt)."""
    year = f"{day.year:04d}"
    year_dir = os.path.join(output_dir, year)
    os.makedirs(year_dir, exist_ok=True)
    filename = f"{day.strftime('%Y%m%d')}_{hour:02d}Z.txt"
    file_path = os.path.join(year_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return file_path


# ----------------------------------------------------------------------
# Retry Queue Management (Network Failures)
#
# Pending items are stored as empty marker files in a designated folder,
# named "YYYYMMDD_HHZ.pending". This allows tracking interrupted items
# and resuming execution across multiple script runs.
# ----------------------------------------------------------------------
PENDING_FOLDER_NAME = "pending_queue_internet"
RE_PENDING_NAME = re.compile(r"^(\d{8})_(\d{2})Z\.pending$")


def _get_pending_dir(output_dir: str) -> str:
    return os.path.join(output_dir, PENDING_FOLDER_NAME)


def mark_pending(output_dir: str, day: date, hour: int) -> None:
    """Creates a marker file indicating a task pending retry."""
    folder = _get_pending_dir(output_dir)
    os.makedirs(folder, exist_ok=True)
    filename = f"{day.strftime('%Y%m%d')}_{hour:02d}Z.pending"
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")


def remove_pending(output_dir: str, day: date, hour: int) -> None:
    """Removes the pending marker file once a task completes or fails permanently."""
    folder = _get_pending_dir(output_dir)
    filename = f"{day.strftime('%Y%m%d')}_{hour:02d}Z.pending"
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass


def list_pending(output_dir: str):
    """Returns a list of tuples (date, hour) representing current pending tasks."""
    folder = _get_pending_dir(output_dir)
    if not os.path.isdir(folder):
        return []
    items = []
    for filename in os.listdir(folder):
        m = RE_PENDING_NAME.match(filename)
        if not m:
            continue
        date_str, hour_str = m.groups()
        day = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        items.append((day, int(hour_str)))
    return items


def generate_days(start: date, end: date):
    """Generator yielding daily date objects between start and end inclusive."""
    current_day = start
    while current_day <= end:
        yield current_day
        current_day += timedelta(days=1)


def _format_duration(seconds: float) -> str:
    """Formats time duration in seconds into 'Xh Ym Zs' string representation."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(
        description=f"Download University of Wyoming sounding data for station {STATION_NUMBER}."
    )
    parser.add_argument("--start", default=START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output", default=OUTPUT_DIR, help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--log", default="download_log.txt", help="Execution log file name"
    )
    parser.add_argument(
        "--all-hours",
        action="store_true",
        help="Attempt all 8 synoptic hours (00, 03, 06, 09, 12, 15, 18, 21Z) instead of standard 00Z/12Z.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_BETWEEN_REQUESTS,
        help="Delay between HTTP requests in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel execution threads (default: 1 = sequential). "
             "Values between 4 and 10 speed up downloads without overloading the server.",
    )
    parser.add_argument(
        "--retry-pause",
        type=float,
        default=30.0,
        help="Pause duration in seconds between network failure retry iterations (default: %(default)s).",
    )
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    if start_date > end_date:
        print("Error: Start date cannot be after end date.")
        sys.exit(1)

    target_hours = ALL_HOURS if args.all_hours else STANDARD_HOURS

    os.makedirs(args.output, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output, args.log), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    total_days = (end_date - start_date).days + 1
    start_execution_time = datetime.now()
    
    logging.info("Starting sounding data retrieval for station %s", STATION_NUMBER)
    logging.info("Execution start time: %s", start_execution_time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(
        "Period: %s to %s (%d days) | Target hours: %s",
        start_date, end_date, total_days, target_hours,
    )
    logging.info("Estimated total requests: %d", total_days * len(target_hours))

    total_downloaded = 0
    total_no_data = 0
    total_errors = 0
    total_pending_internet = 0
    detailed_errors = []
    counter = 0
    total_requests = total_days * len(target_hours)
    counter_lock = threading.Lock()

    def process_item(day: date, hour: int):
        """Processes a single date/hour task: downloads data and updates metrics."""
        nonlocal total_downloaded, total_no_data, total_errors, total_pending_internet, counter

        try:
            text = download_sounding(day, hour)
            error = None
            is_internet_failure = False
        except InternetFailureError as e:
            text = None
            error = str(e)
            is_internet_failure = True
        except RuntimeError as e:
            text = None
            error = str(e)
            is_internet_failure = False

        with counter_lock:
            counter += 1

            if is_internet_failure:
                mark_pending(args.output, day, hour)
                logging.warning(
                    "Network failure - item queued/retained in pending list "
                    "(%s): %s %02dZ -> %s",
                    PENDING_FOLDER_NAME, day.isoformat(), hour, error,
                )
                total_pending_internet += 1
            else:
                remove_pending(args.output, day, hour)
                if error is not None:
                    logging.error(error)
                    detailed_errors.append((day.isoformat(), hour, error))
                    total_errors += 1
                elif text is None:
                    logging.info("No data available: %s %02dZ", day.isoformat(), hour)
                    total_no_data += 1
                else:
                    saved_path = save_sounding(args.output, day, hour, text)
                    logging.info("Successfully saved: %s", saved_path)
                    total_downloaded += 1

            if counter % 50 == 0:
                elapsed_seconds = (datetime.now() - start_execution_time).total_seconds()
                avg_time_per_request = elapsed_seconds / counter
                estimated_remaining_seconds = avg_time_per_request * (total_requests - counter)
                logging.info(
                    "Progress: %d/%d requests | %d saved | %d missing | %d errors | "
                    "%d pending (network) | Elapsed: %s | Estimated remaining: %s",
                    counter, total_requests,
                    total_downloaded, total_no_data, total_errors, total_pending_internet,
                    _format_duration(elapsed_seconds),
                    _format_duration(estimated_remaining_seconds),
                )

    if args.threads <= 1:
        # Sequential processing mode
        for day in generate_days(start_date, end_date):
            for hour in target_hours:
                process_item(day, hour)
                time.sleep(args.delay)
    else:
        # Multithreaded parallel processing mode
        logging.info("Parallel mode activated using %d worker threads.", args.threads)
        tasks = [
            (day, hour)
            for day in generate_days(start_date, end_date)
            for hour in target_hours
        ]
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(process_item, day, hour) for day, hour in tasks]
            for future in as_completed(futures):
                future.result()

    # ------------------------------------------------------------------
    # Phase 2: Process pending queue (retry network failures)
    # ------------------------------------------------------------------
    pending_items = list_pending(args.output)
    if pending_items:
        logging.info("=" * 60)
        logging.info(
            "Starting retry process for network-failed pending items (%d remaining).",
            len(pending_items),
        )
        round_count = 0
        while pending_items:
            round_count += 1
            logging.info(
                "--- Pending Retry Queue: Round %d (%d items remaining) ---",
                round_count, len(pending_items),
            )
            for day, hour in pending_items:
                process_item(day, hour)
                time.sleep(args.delay)

            pending_items = list_pending(args.output)
            if pending_items:
                logging.info(
                    "%d items remain in queue after round %d. "
                    "Waiting %.0fs before next retry attempt...",
                    len(pending_items), round_count, args.retry_pause,
                )
                time.sleep(args.retry_pause)

        logging.info("Pending retry queue successfully cleared.")

    end_execution_time = datetime.now()
    total_duration = (end_execution_time - start_execution_time).total_seconds()

    logging.info("=" * 60)
    logging.info("Execution complete.")
    logging.info("Start time: %s", start_execution_time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("End time: %s", end_execution_time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Total duration: %s", _format_duration(total_duration))
    logging.info("Soundings downloaded successfully: %d", total_downloaded)
    logging.info("Date/hour combinations with no data: %d", total_no_data)
    logging.info("Permanent non-network errors: %d", total_errors)
    
    if detailed_errors:
        logging.info("Detailed error logs:")
        for day_str, hour, err_msg in detailed_errors:
            logging.info("  %s %02dZ -> %s", day_str, hour, err_msg)


if __name__ == "__main__":
    main()
