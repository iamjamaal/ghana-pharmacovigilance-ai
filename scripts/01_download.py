#!/usr/bin/env python3
"""
Step 1: Download all Ghanaian ADR data sources.
- DrugLens PDFs from Ghana FDA website
- FDA annual reports and guidelines
- PMC open-access articles (XML format)
"""

import json
import os
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_PATH = BASE_DIR / "config" / "sources.json"

HEADERS = {
    "User-Agent": "GhanaADR-Research-Pipeline/1.0 (academic research; Ghana AI Innovation Challenge 2026)"
}

PMC_OAI_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
PMC_EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def download_pdf(url: str, dest: Path, label: str) -> bool:
    """Download a PDF file with retry logic."""
    if dest.exists():
        print(f"  [SKIP] {label} already downloaded")
        return True

    for attempt in range(3):
        try:
            print(f"  [GET]  {label} (attempt {attempt + 1})")
            resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            resp.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = dest.stat().st_size / 1024
            print(f"  [OK]   {label} — {size_kb:.0f} KB")
            return True

        except requests.RequestException as e:
            print(f"  [FAIL] {label}: {e}")
            if dest.exists():
                dest.unlink()
            time.sleep(2 * (attempt + 1))

    return False


def download_pmc_xml(pmcid: str, dest: Path, label: str) -> bool:
    """Download a PMC article as full-text XML via E-utilities."""
    if dest.exists():
        print(f"  [SKIP] {label} already downloaded")
        return True

    # Use efetch to get full XML
    params = {
        "db": "pmc",
        "id": pmcid.replace("PMC", ""),
        "rettype": "xml",
        "retmode": "xml"
    }

    for attempt in range(3):
        try:
            print(f"  [GET]  {label} ({pmcid}, attempt {attempt + 1})")
            resp = requests.get(PMC_EFETCH_BASE, params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()

            with open(dest, "w", encoding="utf-8") as f:
                f.write(resp.text)

            size_kb = dest.stat().st_size / 1024
            print(f"  [OK]   {label} — {size_kb:.0f} KB XML")
            return True

        except requests.RequestException as e:
            print(f"  [FAIL] {label}: {e}")
            if dest.exists():
                dest.unlink()
            time.sleep(2 * (attempt + 1))

    return False


def main():
    config = load_config()

    # Create subdirectories
    pdf_dir = RAW_DIR / "pdfs"
    xml_dir = RAW_DIR / "pmc_xml"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    results = {"success": [], "failed": []}

    # ── Download DrugLens PDFs ──
    print("\n═══ Downloading DrugLens Newsletters ═══")
    for item in config["druglens_pdfs"]:
        dest = pdf_dir / f"{item['id']}.pdf"
        ok = download_pdf(item["url"], dest, f"DrugLens Issue {item['issue']}")
        (results["success"] if ok else results["failed"]).append(item["id"])
        time.sleep(1)  # polite delay

    # ── Download FDA reports ──
    print("\n═══ Downloading FDA Reports ═══")
    for item in config["fda_reports"]:
        dest = pdf_dir / f"{item['id']}.pdf"
        ok = download_pdf(item["url"], dest, item["id"])
        (results["success"] if ok else results["failed"]).append(item["id"])
        time.sleep(1)

    # ── Download PMC articles ──
    print("\n═══ Downloading PMC Articles ═══")
    for item in config["pmc_articles"]:
        dest = xml_dir / f"{item['id']}_{item['pmcid']}.xml"
        ok = download_pmc_xml(item["pmcid"], dest, item["title"][:60])
        (results["success"] if ok else results["failed"]).append(item["id"])
        time.sleep(0.5)  # NCBI rate limit: 3 requests/second without API key

    # ── Summary ──
    print("\n═══ Download Summary ═══")
    print(f"  Successful: {len(results['success'])}")
    print(f"  Failed:     {len(results['failed'])}")
    if results["failed"]:
        print(f"  Failed items: {', '.join(results['failed'])}")

    # Save manifest
    manifest_path = RAW_DIR / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
