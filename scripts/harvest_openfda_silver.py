#!/usr/bin/env python3
"""
scripts/harvest_openfda_silver.py
===================================
Download real ICSR records from the OpenFDA adverse events API and convert
them to CLF-only training records (empty adr_spans/drug_spans).

Focus: drugs relevant to Ghana ADR Pipeline — antimalarials, ARVs, anti-TB,
vaccines, and common chronic disease medications.

Each API record is converted to a natural-language sentence summarising the
ICSR key fields, so the model learns to recognise regulatory register language.

Usage:
    python scripts/harvest_openfda_silver.py --dry-run   # preview first 5
    python scripts/harvest_openfda_silver.py             # write output

Output: data/silver/openfda_harvest_v1.jsonl (CLF-only, empty spans)

Note: Requires internet access. Rate limit: 240 req/min without API key.
      For higher volume, set env var OPENFDA_KEY=<your key>.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

ROOT   = Path(__file__).parent.parent
OUTDIR = ROOT / "data" / "silver"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT    = OUTDIR / "openfda_harvest_v1.jsonl"

BASE_URL = "https://api.fda.gov/drug/event.json"
API_KEY  = os.environ.get("OPENFDA_KEY", "")

# Ghana-relevant drug terms — mapped to canonical name
DRUG_QUERIES = [
    # Antimalarials
    ("artesunate",              "artesunate"),
    ("artemether+lumefantrine", "artemether-lumefantrine"),
    ("amodiaquine",             "amodiaquine"),
    ("quinine",                 "quinine"),
    ("chloroquine",             "chloroquine"),
    # ARVs
    ("efavirenz",               "efavirenz"),
    ("tenofovir",               "tenofovir"),
    ("lamivudine",              "lamivudine"),
    ("nevirapine",              "nevirapine"),
    ("lopinavir",               "lopinavir"),
    # Anti-TB
    ("isoniazid",               "isoniazid"),
    ("rifampicin",              "rifampicin"),
    ("rifampin",                "rifampicin"),
    ("pyrazinamide",            "pyrazinamide"),
    # Antibiotics
    ("cotrimoxazole",           "cotrimoxazole"),
    ("trimethoprim",            "cotrimoxazole"),
    # Chronic disease
    ("metformin",               "metformin"),
    ("glibenclamide",           "glibenclamide"),
    ("atorvastatin",            "atorvastatin"),
    ("amlodipine",              "amlodipine"),
    ("lisinopril",              "lisinopril"),
    ("warfarin",                "warfarin"),
]

# Seriousness criterion labels
SERIOUSNESS_MAP = {
    "seriousnesshospitalization":    "hospitalisation",
    "seriousnesslifethreatening":    "life-threatening",
    "seriousnessdisabling":          "disability/incapacity",
    "seriousnessdeath":              "death",
    "seriousnesscongenitalanomali":  "congenital anomaly",
    "seriousnessother":              "medically significant",
}

OUTCOME_MAP = {
    "1": "recovered",
    "2": "recovering",
    "3": "not recovered",
    "4": "recovered with sequelae",
    "5": "fatal",
    "6": "unknown",
}


def _fetch(drug_query, canonical, limit, skip):
    """Fetch up to `limit` ICSR records for drug_query; returns list of raw dicts."""
    params = {
        "search": f"patient.drug.medicinalproduct:{drug_query}",
        "limit":  str(limit),
        "skip":   str(skip),
    }
    if API_KEY:
        params["api_key"] = API_KEY
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ghana-adr-pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        return data.get("results", [])
    except Exception as exc:
        print(f"  [WARN] {canonical}: {exc}", file=sys.stderr)
        return []


def _icsr_to_sentence(result, canonical):
    """Convert one OpenFDA result dict to a natural-language ICSR summary."""
    patient = result.get("patient", {})

    # Reactions
    reactions = [
        r.get("reactionmeddrapt", "").strip()
        for r in patient.get("reaction", [])
        if r.get("reactionmeddrapt", "").strip()
    ]
    if not reactions:
        return None, None

    # Drug indication
    indications = []
    for d in patient.get("drug", []):
        ind = d.get("drugindication", "").strip()
        if ind and ind.lower() not in ("", "unknown", "not reported"):
            indications.append(ind)
    indication_str = indications[0].title() if indications else "unspecified indication"

    # Seriousness criteria
    serious_flags = [
        label
        for field, label in SERIOUSNESS_MAP.items()
        if str(result.get(field, "")).strip() == "1"
    ]
    if not serious_flags and str(result.get("serious", "")).strip() != "1":
        serious_str = "non-serious"
    elif serious_flags:
        serious_str = "; ".join(serious_flags)
    else:
        serious_str = "serious"

    # Outcome
    reactions_list = patient.get("reaction", [])
    outcome_code = str(reactions_list[0].get("reactionoutcome", "6")).strip() if reactions_list else "6"
    outcome_str = OUTCOME_MAP.get(outcome_code, "unknown")

    # Age/sex
    age = patient.get("patientonsetage", "")
    sex_code = str(patient.get("patientsex", "")).strip()
    sex_str = {"1": "male", "2": "female"}.get(sex_code, "")

    demo_parts = []
    if age:
        try:
            age_int = int(float(age))
            demo_parts.append(f"{age_int}-year-old")
        except ValueError:
            pass
    if sex_str:
        demo_parts.append(sex_str)
    demo_str = " ".join(demo_parts) if demo_parts else "patient"

    # Build sentence
    reaction_str = "; ".join(reactions[:3])  # cap at 3 reactions
    parts = [
        f"Drug: {canonical.title()}.",
        f"Indication: {indication_str}.",
        f"Adverse reaction: {reaction_str}.",
        f"Seriousness: {serious_str}.",
        f"Outcome: {outcome_str}.",
    ]
    if demo_str != "patient":
        parts.insert(0, f"Patient: {demo_str}.")

    sentence = " ".join(parts)

    # Drug field for schema
    return canonical, sentence


def harvest(per_drug_limit=15, delay_secs=0.5):
    records = []
    seen_sents = set()
    stats = {"queries": 0, "raw": 0, "kept": 0, "dedup_skipped": 0}

    for drug_query, canonical in DRUG_QUERIES:
        results = _fetch(drug_query, canonical, per_drug_limit, 0)
        stats["queries"] += 1
        stats["raw"] += len(results)

        for r in results:
            drug, sent = _icsr_to_sentence(r, canonical)
            if drug is None:
                continue
            if sent in seen_sents:
                stats["dedup_skipped"] += 1
                continue
            seen_sents.add(sent)
            records.append({
                "drug":         drug,
                "sentence":     sent,
                "source":       "openfda",
                "setid":        f"openfda_{canonical.replace('/', '_')}_{stats['kept']:04d}",
                "contains_adr": 1,
                "adr_spans":    [],
                "drug_spans":   [],
            })
            stats["kept"] += 1
        time.sleep(delay_secs)

    return records, stats


def main():
    parser = argparse.ArgumentParser(description="Harvest OpenFDA ICSR silver data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and preview; don't write output")
    args = parser.parse_args()

    print("Querying OpenFDA adverse events API...")
    records, stats = harvest(per_drug_limit=15)

    print(f"\nQueries: {stats['queries']}  Raw results: {stats['raw']}")
    print(f"Kept: {stats['kept']}  Dedup skipped: {stats['dedup_skipped']}")

    if args.dry_run:
        print("\n[dry-run] Sample records:")
        for r in records[:5]:
            print(f"\n  drug: {r['drug']}")
            print(f"  text: {r['sentence']}")
        print("\n[dry-run] No output written.")
        return

    if not records:
        print("No records collected — check network access and API availability.")
        sys.exit(1)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWritten {len(records)} records -> {OUT}")
    print("NOTE: CLF-only data. Do not pass to NER — adr_spans are empty.")


if __name__ == "__main__":
    main()
