"""
FDA Warning Letters Scraper
============================
Step 1: Hits the FDA's internal DataTables AJAX endpoint to paginate through
        all warning letters and collect metadata + URLs.

Step 2: Fetches the full text of each letter from its individual fda.gov page.

Install dependencies:
    pip install requests beautifulsoup4 pandas tqdm
"""

import os
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path

# ------------ Configuration ------------

BASE_DIR      = Path(__file__).parent
OUTPUT_DIR    = BASE_DIR / "fda_warning_letters"
METADATA_CSV  = OUTPUT_DIR / "metadata.csv"
FULL_TEXT_DIR = OUTPUT_DIR / "full_text"
ERRORS_FILE   = OUTPUT_DIR / "errors.json"

# Seconds to wait between requests
REQUEST_DELAY = 0.1

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchBot/1.0; "
        "FDA Warning Letter ML Dataset)"
    )
}

FULL_TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Collect metadata + URLs via the DataTables AJAX endpoint
# ══════════════════════════════════════════════════════════════════════════════

# The FDA warning letters page uses a server-side DataTables table.
# All data is loaded via this AJAX endpoint using start/length pagination.
AJAX_URL = "https://www.fda.gov/datatables/views/ajax"

# These parameters come directly from the page's embedded JS config
AJAX_BASE_PARAMS = {
    "_drupal_ajax":     "1",
    "view_name":        "warning_letter_solr_index",
    "view_display_id":  "warning_letter_solr_block",
    "view_base_path":   (
        "inspections-compliance-enforcement-and-criminal-investigations"
        "/compliance-actions-and-activities/warning-letters/datatables-data"
    ),
    "view_dom_id": (
        "b3376f9d9b2ab318b05e02d9440378cdb3b394491759ac589e7bc5778193edaa"
    ),
    "view_path": (
        "/inspections-compliance-enforcement-and-criminal-investigations"
        "/compliance-actions-and-activities/warning-letters"
    ),
    "pager_element": "0",
    "view_args":     "",
}

# Total letters known from the page's JS config — used as a ceiling for pagination
TOTAL_LETTERS = 3366


def fetch_urls_from_search(max_letters: int | None = None) -> list[dict[str, str]]:
    """
    Paginate the FDA DataTables AJAX endpoint and return raw metadata records.

    Requests 100 rows per page and stops early if max_letters is reached,
    avoiding unnecessary HTTP requests. Each record contains the fields
    posted_date, date, company, issuing_office, subject, response_letter,
    closeout_letter, and url.
    """
    print("Step 1: Collecting letter URLs from FDA DataTables AJAX endpoint...")

    records = []
    start  = 0
    length = 100

    def plain(html: str) -> str:
        return BeautifulSoup(html, "html.parser").get_text(strip=True)

    while True:
        params = {
            **AJAX_BASE_PARAMS,
            "start":  start,
            "length": length,
        }

        try:
            resp = requests.get(AJAX_URL, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Request failed at start={start}: {e}")
            break

        rows = data.get("data", [])
        if not rows:
            print(f"  No rows at start={start}, stopping.")
            break

        total = data.get("recordsTotal", TOTAL_LETTERS)

        for row in rows:
            link_html = row[2]
            soup = BeautifulSoup(link_html, "html.parser")
            link = soup.find("a")
            if not link:
                continue

            href = str(link["href"])
            if not href.startswith("http"):
                href = "https://www.fda.gov" + href

            records.append({
                "posted_date":     plain(row[0]),
                "date":            plain(row[1]),
                "company":         link.get_text(strip=True),
                "issuing_office":  plain(row[3]),
                "subject":         plain(row[4]),
                "response_letter": plain(row[5]) if len(row) > 5 else "",
                "closeout_letter": plain(row[6]) if len(row) > 6 else "",
                "url":             href,
            })

            if max_letters and len(records) >= max_letters:
                break  # stop parsing this page's rows immediately

        print(f"  start={start:>5}: +{len(rows)} rows  ({len(records)} total)")

        if max_letters and len(records) >= max_letters:
            break  # stop fetching the next page
        if start + length >= total:
            break

        start += length
        time.sleep(REQUEST_DELAY)

    return records


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Fetch the full text of each individual letter page
# ══════════════════════════════════════════════════════════════════════════════

def fetch_letter_text(url: str) -> tuple[str | None, str | None]:
    """
    Fetch and return the cleaned full text of a single FDA warning letter page.
    Returns (text, error_string). On success, error_string is None.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, str(e)

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try selectors from most to least specific for FDA's page layout
    for selector in [
        "div.drug-warning-letter-content",
        "div.field--name-body",
        "main#main-content article",
        "main#main-content",
        "div#content-wrapper",
        "article",
    ]:
        el = soup.select_one(selector)
        if el:
            # Strip out nav/header/footer/sidebar noise
            for tag in el.select("nav, .nav, footer, header, script, style, aside"):
                tag.decompose()
            lines = [
                line for line in
                el.get_text(separator="\n", strip=True).splitlines()
                if line.strip()
            ]
            return "\n".join(lines), None

    return None, "No content element found"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_scraper(max_letters:int | None = None) -> None:
    """
    Run the full two-step data acquisition pipeline.

    Step 1 — Calls fetch_urls_from_search to collect metadata and letter URLs,
    then writes the results to metadata.csv.

    Step 2 — Fetches the full text of each letter from its individual fda.gov
    page and writes one .txt file per letter to the full_text directory.
    Already-downloaded letters are skipped so the scraper is safe to resume
    after an interruption. Failures are logged to errors.json.
    """
    print("FDA Warning Letters Collector")
    print("=" * 50)

    # ------------ Step 1: Collect all metadata + URLs ------------
    # There are over 3k letters
    records = fetch_urls_from_search(max_letters=max_letters)

    if not records:
        print("\nNo records collected. Check your internet connection and try again.")
        return

    df = pd.DataFrame(records)
    df.insert(0, "id", df.index.map(lambda i: f"{i:05d}"))  # add id column first
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nMetadata ({len(df)} rows) saved to: {METADATA_CSV}")

    # ------------ Step 2: Fetch full text for each letter ------------
    print(f"\nStep 2: Fetching full text for {len(df)} letters...")
    errors  = []
    fetched = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fetching letters"):
        url = row.get("url", "")
        if not url or not str(url).startswith("http"):
            errors.append({"index": i, "company": row.get("company"), "error": "No URL"})
            continue

        # Build a safe filename from company name + date
        safe_name = "".join(
            c if c.isalnum() or c in "-_ " else "_"
            for c in f"{row.get('company', 'unknown')}_{row.get('date', 'nodate')}"
        ).replace(" ", "_")[:100]
        out_path = os.path.join(FULL_TEXT_DIR, f"{i:05d}_{safe_name}.txt")

        # Resume-safe: skip letters already downloaded
        if os.path.exists(out_path):
            fetched += 1
            continue

        text, err = fetch_letter_text(str(url))

        if err:
            errors.append({"index": i, "url": url, "error": err})
        elif text:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"COMPANY: {row.get('company', '')}\n")
                f.write(f"DATE: {row.get('date', '')}\n")
                f.write(f"SUBJECT: {row.get('subject', '')}\n")
                f.write(f"ISSUING_OFFICE: {row.get('issuing_office', '')}\n")
                f.write(f"URL: {url}\n")
                f.write("=" * 60 + "\n\n")
                f.write(text)
            fetched += 1
        else:
            errors.append({"index": i, "url": url, "error": "Empty text returned"})

        time.sleep(REQUEST_DELAY)

    # ------------ Save errors ------------
    if errors:
        with open(ERRORS_FILE, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n{len(errors)} errors logged to: {ERRORS_FILE}")

    # ------------ Summary ------------
    print(f"\nDone! {fetched} letters saved in: {FULL_TEXT_DIR}/")
    print("\nSample rows:")
    print(df[["company", "date", "subject"]].head(10).to_string(index=False))


def main() -> None:
    print("Running Warning Letter Scraper")
    num_letters = 10
    run_scraper(num_letters)
    print(f"Finished scraping {num_letters} letters")


if __name__ == "__main__":
    main()
