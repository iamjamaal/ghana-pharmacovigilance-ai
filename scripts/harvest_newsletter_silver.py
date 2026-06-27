#!/usr/bin/env python3
"""
scripts/harvest_newsletter_silver.py
=====================================
Harvest newsletter-domain silver NER from DrugLens PDFs.

Reads sentences from data/extracted/druglens_{03,05,07,09,10}.json,
applies DrugGazetteerAnnotator + ADRGazetteerAnnotator from lfs.py,
keeps sentences with >=1 DRUG or ADR match, and writes to
data/silver/newsletter_silver_v1.jsonl in the same format as inserts_v1.jsonl.

Usage:
    python scripts/harvest_newsletter_silver.py --dry-run   # count only
    python scripts/harvest_newsletter_silver.py             # write output
"""
import sys
import re
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import spacy

from weak_supervision.lfs import load_gazetteers, DrugGazetteerAnnotator, ADRGazetteerAnnotator

EXTRACTED = ROOT / "data" / "extracted"
OUTPUT    = ROOT / "data" / "silver" / "newsletter_silver_v1.jsonl"

DRUGLENS_FILES = [
    EXTRACTED / "druglens_03.json",
    EXTRACTED / "druglens_05.json",
    EXTRACTED / "druglens_07.json",
    EXTRACTED / "druglens_09.json",
    EXTRACTED / "druglens_10.json",
]

MIN_SENT_LEN   = 25
MAX_CHUNK_LEN  = 600
MIN_SPAN_CHARS = 4   # reject gazetteer matches shorter than this (e.g. "as", "be")


def _split_text(text: str):
    """Split page text into candidate sentences."""
    # Split on double newlines first (paragraph/section boundaries)
    chunks = re.split(r'\n{2,}', text)
    sentences = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= MAX_CHUNK_LEN:
            # Join single-newline lines within the chunk into one sentence candidate
            joined = ' '.join(ln.strip() for ln in chunk.split('\n') if ln.strip())
            if len(joined) >= MIN_SENT_LEN:
                sentences.append(joined)
        else:
            # Long chunk: split on sentence boundaries (". " + uppercase)
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk)
            for part in parts:
                part = ' '.join(part.split())
                if len(part) >= MIN_SENT_LEN:
                    sentences.append(part)
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Harvest DrugLens newsletter silver")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count sentences and entity hits; don't write output")
    args = parser.parse_args()

    nlp = spacy.blank("en")
    drug_terms, adr_terms = load_gazetteers()
    drug_ann = DrugGazetteerAnnotator(drug_terms)
    adr_ann  = ADRGazetteerAnnotator(adr_terms)

    records = []
    stats = {
        "files": 0, "pages": 0, "sentences_total": 0,
        "drug_hit": 0, "adr_hit": 0, "kept": 0,
    }

    for fpath in DRUGLENS_FILES:
        if not fpath.exists():
            print(f"[SKIP] {fpath.name} not found", file=sys.stderr)
            continue
        with open(fpath, encoding="utf-8") as f:
            doc_data = json.load(f)
        doc_id = doc_data["doc_id"]
        stats["files"] += 1

        for page in doc_data["pages"]:
            stats["pages"] += 1
            for sent_idx, sentence in enumerate(_split_text(page["text"])):
                stats["sentences_total"] += 1
                spacy_doc = nlp(sentence)
                drug_ann(spacy_doc)
                adr_ann(spacy_doc)

                drug_spans = [s for s in spacy_doc.user_data.get("lf_spans", {}).get("lf_drug_gazetteer", [])
                              if s["end_char"] - s["start_char"] >= MIN_SPAN_CHARS]
                adr_spans  = [s for s in spacy_doc.user_data.get("lf_spans", {}).get("lf_adr_gazetteer", [])
                              if s["end_char"] - s["start_char"] >= MIN_SPAN_CHARS]

                if not drug_spans and not adr_spans:
                    continue

                if drug_spans:
                    stats["drug_hit"] += 1
                if adr_spans:
                    stats["adr_hit"] += 1
                stats["kept"] += 1

                # Canonical drug name from first match
                if drug_spans:
                    raw = sentence[drug_spans[0]["start_char"]:drug_spans[0]["end_char"]]
                    drug_name = drug_terms.get(raw.lower().strip(".,"), raw)
                else:
                    drug_name = "unknown"

                records.append({
                    "drug":         drug_name,
                    "sentence":     sentence,
                    "source":       "druglens",
                    "setid":        f"{doc_id}_p{page['page_number']}_s{sent_idx}",
                    "contains_adr": 1 if adr_spans else 0,
                    "adr_spans":    adr_spans,
                    "drug_spans":   drug_spans,
                })

    hit_rate = round(100.0 * stats["kept"] / max(stats["sentences_total"], 1), 1)
    print(f"Files={stats['files']}  Pages={stats['pages']}"
          f"  Sentences={stats['sentences_total']}  Hit rate={hit_rate}%")
    print(f"  drug_hit={stats['drug_hit']}  adr_hit={stats['adr_hit']}  kept={stats['kept']}")

    if args.dry_run:
        print("[dry-run] No output written.")
        for r in records[:3]:
            print(f"\n  text: {r['sentence'][:110]!r}")
            print(f"  drugs: {[r['sentence'][s['start_char']:s['end_char']] for s in r['drug_spans']]}")
            print(f"  adrs:  {[r['sentence'][s['start_char']:s['end_char']] for s in r['adr_spans']]}")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} records -> {OUTPUT}")


if __name__ == "__main__":
    main()
