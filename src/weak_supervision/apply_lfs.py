"""
Step 2 — Apply labeling functions to harvested insert sentences.

Reads corpora/inserts/*.jsonl, runs the 4 LFs + HMM aggregation,
and writes silver labels to data/silver/inserts_v1.jsonl.

Usage:
    python src/weak_supervision/apply_lfs.py
    python src/weak_supervision/apply_lfs.py --inserts-dir corpora/inserts --out data/silver/inserts_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import spacy
from tqdm import tqdm

from lfs import load_gazetteers, build_annotators, aggregate_batch


BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inserts-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    inserts_dir = Path(args.inserts_dir) if args.inserts_dir else (repo_root / "corpora" / "inserts")
    out_path = Path(args.out) if args.out else (repo_root / "data" / "silver" / "inserts_v1.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading spaCy model…")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    print("Loading gazetteers…")
    drug_terms, adr_terms = load_gazetteers()
    annotators = build_annotators(drug_terms, adr_terms)

    # Load all insert records
    records = []
    for jsonl_file in sorted(inserts_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    print(f"\nApplying LFs to {len(records)} insert sentences…")

    written = 0
    with open(out_path, "w", encoding="utf-8") as out_fh:
        for i in range(0, len(records), BATCH_SIZE):
            batch_records = records[i: i + BATCH_SIZE]
            texts = [r["sentence"] for r in batch_records]
            docs = list(nlp.pipe(texts))
            agg_results = aggregate_batch(docs, annotators)

            for rec, doc, (contains_adr, adr_spans) in zip(batch_records, docs, agg_results):
                lf_spans = doc.user_data.get("lf_spans", {})
                out_fh.write(json.dumps({
                    "drug": rec["drug"],
                    "sentence": rec["sentence"],
                    "source": rec["source"],
                    "setid": rec.get("setid", ""),
                    "contains_adr": contains_adr,
                    "adr_spans": adr_spans,
                    "drug_spans": lf_spans.get("lf_drug_gazetteer", []),
                }) + "\n")
                written += 1

            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  processed {min(i + BATCH_SIZE, len(records))}/{len(records)}", end="\r")

    print(f"\nWrote {written} silver-labelled sentences -> {out_path}")
    pos = sum(1 for r in _read_jsonl(out_path) if r["contains_adr"] == 1)
    print(f"  contains_adr=1: {pos} ({100*pos/max(written,1):.1f}%)")
    print(f"  contains_adr=0: {written - pos}")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    main()
