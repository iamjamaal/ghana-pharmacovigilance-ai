"""
Step 1 — Harvest drug package inserts from DailyMed.

For each of the 15 target drugs, queries the DailyMed public REST API,
pulls the Adverse Reactions section, sentence-splits with spaCy, and
saves to corpora/inserts/{drug_name}.jsonl.

Usage:
    python src/weak_supervision/harvest_inserts.py
    python src/weak_supervision/harvest_inserts.py --out-dir corpora/inserts
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import io
import zipfile

import requests
import spacy
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Target drugs (15 from the task specification)
# ---------------------------------------------------------------------------

TARGET_DRUGS = [
    "hydroxychloroquine",
    "chloroquine",
    "AstraZeneca",
    "zidovudine",
    "artesunate",
    "Sputnik V",
    "amodiaquine",
    "lamivudine",
    "lopinavir",
    "hyoscine butylbromide",
    "ritonavir",
    "paracetamol",
    "doxycycline",
    "ceftriaxone",
    "amoxicillin",
]

# DailyMed search terms (some drugs use their INN or US name; None = skip)
DAILYMED_QUERY_MAP = {
    "AstraZeneca": None,             # Not FDA-approved; not in DailyMed
    "Sputnik V": None,               # Not FDA-approved; not in DailyMed
    "hyoscine butylbromide": None,   # Not FDA-approved in US; not in DailyMed
    "artesunate": None,              # IV artesunate available only via CDC/IND; no labelled NDA
    "amodiaquine": None,             # Not FDA-approved; not in DailyMed
    "paracetamol": "acetaminophen",  # US name
}

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
ADVERSE_SECTION_RE = re.compile(
    r"adverse\s+react", re.IGNORECASE
)

# Sections to skip even if they mention "adverse"
SKIP_SECTION_RE = re.compile(
    r"(reporting|post.?market|mechanism|drug\s+interaction|clinical\s+trial)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DailyMed API helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict | None = None, timeout: int = 20) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"    [warn] GET {url} failed: {exc}")
        return None


def _get_xml(url: str, timeout: int = 30) -> str | None:
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"Accept": "application/xml"})
        r.raise_for_status()
        return r.text
    except Exception as exc:
        print(f"    [warn] GET {url} failed: {exc}")
        return None


def search_drug(drug_name: str) -> str | None:
    """Return the setid of the first search result, or None."""
    data = _get_json(f"{DAILYMED_BASE}/spls.json", params={"drug_name": drug_name})
    if not data:
        return None
    results = data.get("data", [])
    if not results:
        return None
    return results[0].get("setid")


def fetch_spl_xml(setid: str) -> str | None:
    """Download the SPL zip for a setid and return the XML text.

    DailyMed provides the full SPL as a ZIP file via getFile.cfm?type=zip.
    The REST API v2 does not serve individual SPL XML directly.
    """
    url = f"https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid={setid}&type=zip"
    try:
        r = requests.get(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            xml_files = [name for name in z.namelist() if name.endswith(".xml")]
            if not xml_files:
                print(f"    [warn] ZIP for {setid} contains no XML files")
                return None
            return z.read(xml_files[0]).decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"    [warn] ZIP download failed for {setid}: {exc}")
        return None


# LOINC code for Adverse Reactions section in FDA drug labels
_ADVERSE_LOINC = "34084-4"


def extract_adverse_text(xml_text: str) -> str:
    """Extract text from the Adverse Reactions section of an SPL XML document."""
    if not xml_text:
        return ""
    try:
        soup = BeautifulSoup(xml_text, "lxml-xml")
    except Exception:
        try:
            soup = BeautifulSoup(xml_text, "lxml")
        except Exception:
            return ""

    texts: list[str] = []

    # Primary: match by LOINC code (most reliable)
    for section in soup.find_all("section"):
        code_tag = section.find("code")
        if code_tag and code_tag.get("code") == _ADVERSE_LOINC:
            texts.append(_extract_section_text(section))

    # Fallback: match by title text
    if not texts:
        for section in soup.find_all("section"):
            title_tag = section.find("title")
            if title_tag:
                title_text = title_tag.get_text(" ", strip=True)
                if ADVERSE_SECTION_RE.search(title_text) and not SKIP_SECTION_RE.search(title_text):
                    texts.append(_extract_section_text(section))

    return "\n\n".join(t for t in texts if t.strip())


def _extract_section_text(section) -> str:
    """Extract all readable text from an XML section element."""
    # Remove table elements which contain structured data not useful as sentences
    for tag in section.find_all(["table", "caption", "col", "colgroup"]):
        tag.decompose()
    return section.get_text(" ", strip=True)


# ---------------------------------------------------------------------------
# Main harvest loop
# ---------------------------------------------------------------------------

def harvest(out_dir: Path, nlp) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for drug in TARGET_DRUGS:
        safe_name = re.sub(r"[^\w\-]", "_", drug.lower())
        out_path = out_dir / f"{safe_name}.jsonl"

        query = DAILYMED_QUERY_MAP.get(drug, drug)
        if query is None:
            print(f"  {drug}: skipped — not in DailyMed (expected for vaccines/Sputnik V)")
            out_path.write_text("")  # empty file, not an error
            counts[drug] = 0
            continue

        print(f"  {drug}: searching DailyMed for '{query}'…")
        setid = search_drug(query)
        if not setid:
            print(f"  {drug}: no DailyMed result for '{query}'")
            out_path.write_text("")
            counts[drug] = 0
            continue

        print(f"    setid = {setid}, fetching SPL XML…")
        xml_text = fetch_spl_xml(setid)
        if not xml_text:
            print(f"    fetch failed")
            out_path.write_text("")
            counts[drug] = 0
            continue

        raw_text = extract_adverse_text(xml_text)

        if not raw_text:
            print(f"    no adverse reactions text found in SPL")
            out_path.write_text("")
            counts[drug] = 0
            continue

        sentences = _split_sentences(raw_text, nlp)
        sentences = [s for s in sentences if len(s.split()) >= 5]  # filter noise

        with open(out_path, "w", encoding="utf-8") as fh:
            for sent in sentences:
                fh.write(json.dumps({
                    "drug": drug,
                    "sentence": sent,
                    "source": "dailymed",
                    "setid": setid,
                }) + "\n")

        counts[drug] = len(sentences)
        print(f"    {len(sentences)} sentences saved -> {out_path.name}")
        time.sleep(0.5)  # polite rate limiting

    return counts


def _split_sentences(text: str, nlp) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Harvest DailyMed drug inserts")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: corpora/inserts)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "corpora" / "inserts")

    print("Loading spaCy model…")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  en_core_web_sm not found; downloading…")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000

    print(f"\nHarvesting {len(TARGET_DRUGS)} drugs -> {out_dir}\n")
    counts = harvest(out_dir, nlp)

    print("\n--- Harvest summary ---")
    total = 0
    for drug in TARGET_DRUGS:
        n = counts.get(drug, 0)
        total += n
        status = "ZERO" if n == 0 else str(n)
        print(f"  {drug:<30} {status:>6} sentences")
    print(f"  {'TOTAL':<30} {total:>6} sentences")


if __name__ == "__main__":
    main()
