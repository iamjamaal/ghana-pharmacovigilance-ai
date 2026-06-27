#!/usr/bin/env python3
"""
Step 4: Export unified Ghana ADR dataset.
Reads the annotated JSONL and produces:
  1. ghana_adr_dataset.jsonl    — full annotated dataset (model training)
  2. ghana_adr_dataset.csv      — flat CSV for quick exploration
  3. ghana_adr_ner.jsonl        — NER-format for token classification
  4. ghana_adr_re.jsonl         — relation extraction format
  5. dataset_card.md            — HuggingFace-style dataset card
  6. label_studio_import.json   — for manual annotation review
"""

import json
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = BASE_DIR / "data" / "annotated"
OUTPUT_DIR = BASE_DIR / "output"
CONFIG_PATH = BASE_DIR / "config" / "sources.json"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_annotated_data() -> list[dict]:
    input_path = ANNOTATED_DIR / "ghana_adr_sentences.jsonl"
    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_full_jsonl(rows: list[dict], output_dir: Path):
    """Export the complete annotated dataset."""
    path = output_dir / "ghana_adr_dataset.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [1/6] Full dataset:      {path.name} ({len(rows)} rows)")


def export_csv(rows: list[dict], output_dir: Path):
    """Export flat CSV for spreadsheet exploration."""
    path = output_dir / "ghana_adr_dataset.csv"

    fieldnames = [
        "sentence_id", "doc_id", "source_type", "year", "hospital",
        "section", "text", "contains_adr",
        "n_drugs", "drugs_found", "n_adrs", "adrs_found",
        "n_relations", "auto_annotated", "manually_reviewed"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            drugs = row.get("entities", {}).get("drugs", [])
            adrs = row.get("entities", {}).get("adrs", [])

            writer.writerow({
                "sentence_id": row["sentence_id"],
                "doc_id": row["doc_id"],
                "source_type": row["source_type"],
                "year": row.get("year", ""),
                "hospital": row.get("hospital", ""),
                "section": row.get("section", ""),
                "text": row["text"],
                "contains_adr": row["contains_adr"],
                "n_drugs": len(drugs),
                "drugs_found": "; ".join(d["canonical"] for d in drugs),
                "n_adrs": len(adrs),
                "adrs_found": "; ".join(a["canonical"] for a in adrs),
                "n_relations": len(row.get("relations", [])),
                "auto_annotated": row.get("auto_annotated", True),
                "manually_reviewed": row.get("manually_reviewed", False)
            })

    print(f"  [2/6] CSV export:        {path.name}")


def export_ner_format(rows: list[dict], output_dir: Path):
    """Export in NER token-classification format (text + entity spans)."""
    path = output_dir / "ghana_adr_ner.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            entities = []
            for drug in row.get("entities", {}).get("drugs", []):
                entities.append({
                    "start": drug["start"],
                    "end": drug["end"],
                    "label": "DRUG",
                    "text": drug["text"]
                })
            for adr in row.get("entities", {}).get("adrs", []):
                entities.append({
                    "start": adr["start"],
                    "end": adr["end"],
                    "label": "ADR",
                    "text": adr["text"]
                })

            # Sort entities by position
            entities.sort(key=lambda e: e["start"])

            ner_row = {
                "id": row["sentence_id"],
                "text": row["text"],
                "entities": entities,
                "meta": {
                    "doc_id": row["doc_id"],
                    "source_type": row["source_type"]
                }
            }
            f.write(json.dumps(ner_row, ensure_ascii=False) + "\n")

    print(f"  [3/6] NER format:        {path.name}")


def export_re_format(rows: list[dict], output_dir: Path):
    """Export relation-extraction format (drug-ADR pairs with context)."""
    path = output_dir / "ghana_adr_re.jsonl"
    count = 0

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            for rel in row.get("relations", []):
                re_row = {
                    "id": f"{row['sentence_id']}_r{count}",
                    "text": row["text"],
                    "drug": rel["drug"],
                    "drug_span": rel["drug_span"],
                    "adr": rel["adr"],
                    "adr_span": rel["adr_span"],
                    "relation": rel["type"],
                    "confidence": rel["confidence"],
                    "needs_review": rel["needs_review"],
                    "meta": {
                        "doc_id": row["doc_id"],
                        "source_type": row["source_type"]
                    }
                }
                f.write(json.dumps(re_row, ensure_ascii=False) + "\n")
                count += 1

    print(f"  [4/6] RE format:         {path.name} ({count} relations)")


def export_label_studio(rows: list[dict], output_dir: Path):
    """
    Export Label Studio import format for manual annotation review.
    Team members load this into Label Studio to correct auto-annotations.
    """
    path = output_dir / "label_studio_import.json"

    tasks = []
    for row in rows:
        # Only include rows that need review (have auto-annotations)
        drugs = row.get("entities", {}).get("drugs", [])
        adrs = row.get("entities", {}).get("adrs", [])

        if not drugs and not adrs and not row.get("contains_adr"):
            continue

        predictions = []
        for drug in drugs:
            predictions.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": drug["start"],
                    "end": drug["end"],
                    "text": drug["text"],
                    "labels": ["DRUG"]
                }
            })
        for adr in adrs:
            predictions.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": adr["start"],
                    "end": adr["end"],
                    "text": adr["text"],
                    "labels": ["ADR"]
                }
            })

        task = {
            "data": {
                "text": row["text"],
                "meta": {
                    "sentence_id": row["sentence_id"],
                    "doc_id": row["doc_id"],
                    "source_type": row["source_type"],
                    "contains_adr_auto": row["contains_adr"]
                }
            },
            "predictions": [{
                "model_version": "lexicon_v1",
                "result": predictions
            }] if predictions else []
        }
        tasks.append(task)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"  [5/6] Label Studio:      {path.name} ({len(tasks)} tasks)")


def export_dataset_card(rows: list[dict], config: dict, output_dir: Path):
    """Generate a HuggingFace-style dataset card."""
    path = output_dir / "dataset_card.md"

    # Compute statistics
    total = len(rows)
    adr_pos = sum(1 for r in rows if r["contains_adr"])
    source_counts = Counter(r["source_type"] for r in rows)
    year_counts = Counter(r.get("year") for r in rows)

    all_drugs = []
    all_adrs = []
    for r in rows:
        for d in r.get("entities", {}).get("drugs", []):
            all_drugs.append(d["canonical"])
        for a in r.get("entities", {}).get("adrs", []):
            all_adrs.append(a["canonical"])

    top_drugs = Counter(all_drugs).most_common(15)
    top_adrs = Counter(all_adrs).most_common(15)

    card = f"""---
language: en
license: cc-by-4.0
task_categories:
  - token-classification
  - text-classification
task_ids:
  - named-entity-recognition
  - binary-classification
tags:
  - pharmacovigilance
  - adverse-drug-reactions
  - ghana
  - healthcare
  - nlp
  - africa
pretty_name: Ghana ADR Dataset
size_categories:
  - 1K<n<10K
---

# Ghana Adverse Drug Reaction (ADR) NLP Dataset

## Dataset Description

A curated corpus of adverse drug reaction mentions extracted from publicly
available Ghanaian health sources, built for the **Ghana AI Innovation
Challenge 2026** (Ghana AI Summit & Awards).

**Primary Ghanaian data sources:**
- Ghana FDA DrugLens Newsletters (Issues 3, 5, 7, 9, 10)
- Ghana FDA Annual Report 2023
- Open-access case reports from Korle-Bu, KATH, and regional hospitals
- Patient ADR interview transcripts from 28 Ghanaian health facilities
- COVID-19 vaccine AEFI surveillance data from Ghana

### Dataset Statistics

| Metric | Value |
|---|---|
| Total sentences | {total:,} |
| ADR-positive sentences | {adr_pos:,} ({100*adr_pos/max(total,1):.1f}%) |
| Unique drugs mentioned | {len(set(all_drugs)):,} |
| Unique ADR terms | {len(set(all_adrs)):,} |
| Total drug mentions | {len(all_drugs):,} |
| Total ADR mentions | {len(all_adrs):,} |
| Source documents | {len(set(r['doc_id'] for r in rows))} |
| Year range | {min(y for y in year_counts if y)}-{max(y for y in year_counts if y)} |

### Source Distribution

| Source Type | Sentences |
|---|---|
"""
    for st, count in source_counts.most_common():
        card += f"| {st} | {count:,} |\n"

    card += f"""
### Top 15 Drugs Mentioned

| Drug | Mentions |
|---|---|
"""
    for drug, count in top_drugs:
        card += f"| {drug} | {count} |\n"

    card += f"""
### Top 15 ADR Terms

| ADR | Mentions |
|---|---|
"""
    for adr, count in top_adrs:
        card += f"| {adr} | {count} |\n"

    card += f"""
## Annotation Schema

- **DRUG**: Medication names (brand or generic)
- **ADR**: Adverse drug reaction symptoms or conditions
- **SEVERITY**: Severity indicators (mild, moderate, severe, fatal)
- **PATIENT_DEMO**: Patient demographics (age, sex, region)
- **CAUSES**: Relation linking DRUG → ADR

Sentence-level label: `contains_adr` (binary)

## Usage

```python
import json

# Load full dataset
with open("ghana_adr_dataset.jsonl") as f:
    data = [json.loads(line) for line in f]

# Filter ADR-positive sentences
adr_sentences = [row for row in data if row["contains_adr"] == 1]

# Get all drug-ADR relations
relations = []
for row in data:
    for rel in row.get("relations", []):
        relations.append(rel)
```

## Licensing

- DrugLens newsletters: Ghana Government Publication (public domain)
- PMC articles: CC-BY (Creative Commons Attribution)
- Dataset compilation: CC-BY-4.0

## Citation

```bibtex
@dataset{{ghana_adr_2026,
  title={{Ghana ADR NLP Dataset}},
  author={{[Your Team Name]}},
  year={{2026}},
  note={{Built for the Ghana AI Innovation Challenge 2026}},
  url={{https://github.com/[your-repo]}}
}}
```

## Ethical Considerations

- No individual patient identifiers are included
- All source data is from publicly available, open-access publications
- The dataset is intended for research and development of pharmacovigilance tools
- Auto-annotations are preliminary and require manual review before clinical use

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} by the Ghana ADR Pipeline.
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(card)

    print(f"  [6/6] Dataset card:      {path.name}")


def main():
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═══ Loading annotated data ═══")
    rows = load_annotated_data()
    print(f"  Loaded {len(rows):,} sentences\n")

    print("═══ Exporting dataset ═══")
    export_full_jsonl(rows, OUTPUT_DIR)
    export_csv(rows, OUTPUT_DIR)
    export_ner_format(rows, OUTPUT_DIR)
    export_re_format(rows, OUTPUT_DIR)
    export_label_studio(rows, OUTPUT_DIR)
    export_dataset_card(rows, config, OUTPUT_DIR)

    print(f"\n═══ All exports complete ═══")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\n  Next steps:")
    print(f"  1. Load label_studio_import.json into Label Studio for manual review")
    print(f"  2. Have 2-3 annotators correct drug/ADR entities and relations")
    print(f"  3. Re-export with manually_reviewed=True after correction")
    print(f"  4. Fine-tune BioBERT/PubMedBERT on ghana_adr_ner.jsonl")


if __name__ == "__main__":
    main()
