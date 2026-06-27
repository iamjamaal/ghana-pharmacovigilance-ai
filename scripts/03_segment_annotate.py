#!/usr/bin/env python3
"""
Step 3: Segment & Auto-Annotate.
- Split extracted text into individual sentences
- Auto-tag DRUG and ADR entities using lexicon matching
- Classify each sentence as contains_adr = 1 or 0
- Output: one JSONL row per sentence, ready for manual review

This is the FIRST PASS — lexicon-based. Manual correction follows.
"""

import json
import re
import hashlib
from pathlib import Path

# ── Simple sentence splitter (no NLTK dependency for portability) ──
SENT_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'   # Split after sentence-ending punctuation before capital
    r'|(?<=\.\))\s+(?=[A-Z])'    # After .)
    r'|(?<=\.\")\s+(?=[A-Z])'    # After ."
)

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
ANNOTATED_DIR = BASE_DIR / "data" / "annotated"
CONFIG_PATH = BASE_DIR / "config" / "sources.json"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Build lexicons from config ──
def build_drug_lexicon(config: dict) -> list[dict]:
    """Build drug-matching patterns from the seed lexicon."""
    drugs = config.get("ghana_drug_lexicon_seeds", [])
    patterns = []
    for drug in drugs:
        # Create case-insensitive regex pattern
        escaped = re.escape(drug)
        pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
        patterns.append({"name": drug, "pattern": pattern})
    return patterns


def build_adr_lexicon(config: dict) -> list[dict]:
    """Build ADR/symptom-matching patterns from the seed lexicon."""
    symptoms = config.get("ghana_symptom_lexicon_seeds", {})
    patterns = []

    for category in ["english_clinical", "english_colloquial"]:
        for symptom in symptoms.get(category, []):
            escaped = re.escape(symptom)
            pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
            patterns.append({
                "name": symptom,
                "category": category,
                "pattern": pattern
            })

    # Add common ADR terms not in the seed list
    extra_adrs = [
        "nausea", "vomiting", "diarrhoea", "diarrhea", "fever", "pyrexia",
        "pruritus", "urticaria", "erythema", "oedema", "edema", "cough",
        "dyspnoea", "dyspnea", "tachycardia", "bradycardia", "hypotension",
        "hypertension", "jaundice", "hepatitis", "renal failure",
        "anaphylaxis", "angioedema", "seizure", "convulsion", "tremor",
        "insomnia", "fatigue", "malaise", "myalgia", "arthralgia",
        "injection site pain", "injection-site pain", "chills", "rigors",
        "alopecia", "photosensitivity", "tinnitus", "blurred vision",
        "dysgeusia", "anosmia", "hyperhidrosis", "skin peeling",
        "elevated transaminases", "neutropenia", "leukopenia",
        "pancytopenia", "haemolytic anaemia", "hemolytic anemia",
        "acute kidney injury", "renal impairment", "drug eruption",
        "fixed drug eruption", "toxic epidermal necrolysis"
    ]
    for symptom in extra_adrs:
        escaped = re.escape(symptom)
        pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
        patterns.append({
            "name": symptom,
            "category": "extra_clinical",
            "pattern": pattern
        })

    return patterns


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Returns non-empty sentences."""
    if not text or len(text.strip()) < 10:
        return []

    sentences = SENT_SPLIT.split(text)
    # Further split on newlines that look like sentence breaks
    expanded = []
    for s in sentences:
        parts = re.split(r'\n+', s)
        expanded.extend(parts)

    # Clean and filter
    cleaned = []
    for s in expanded:
        s = s.strip()
        # Skip very short fragments, headers, page numbers
        if len(s) < 20:
            continue
        if re.match(r'^(page|figure|table|fig\.|tab\.)\s*\d', s, re.IGNORECASE):
            continue
        if re.match(r'^\d+$', s):
            continue
        cleaned.append(s)

    return cleaned


def find_entities(sentence: str, drug_patterns: list, adr_patterns: list) -> dict:
    """Find drug and ADR entity mentions in a sentence."""
    drugs_found = []
    adrs_found = []

    for dp in drug_patterns:
        for match in dp["pattern"].finditer(sentence):
            drugs_found.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "label": "DRUG",
                "canonical": dp["name"]
            })

    for ap in adr_patterns:
        for match in ap["pattern"].finditer(sentence):
            adrs_found.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "label": "ADR",
                "canonical": ap["name"],
                "category": ap.get("category", "unknown")
            })

    return {
        "drugs": drugs_found,
        "adrs": adrs_found
    }


def generate_sentence_id(doc_id: str, sentence: str, idx: int) -> str:
    """Generate a unique, stable ID for each sentence."""
    content = f"{doc_id}:{idx}:{sentence[:50]}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def infer_relations(drugs: list, adrs: list) -> list:
    """
    Simple proximity-based relation inference.
    If a DRUG and an ADR appear in the same sentence, infer a CAUSES relation.
    """
    relations = []
    for drug in drugs:
        for adr in adrs:
            relations.append({
                "type": "CAUSES",
                "drug": drug["canonical"],
                "drug_span": [drug["start"], drug["end"]],
                "adr": adr["canonical"],
                "adr_span": [adr["start"], adr["end"]],
                "confidence": "auto_cooccurrence",
                "needs_review": True
            })
    return relations


def main():
    config = load_config()
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    drug_patterns = build_drug_lexicon(config)
    adr_patterns = build_adr_lexicon(config)

    print(f"  Drug lexicon:  {len(drug_patterns)} patterns")
    print(f"  ADR lexicon:   {len(adr_patterns)} patterns")

    all_rows = []
    stats = {
        "total_sentences": 0,
        "adr_positive": 0,
        "drug_mentions": 0,
        "adr_mentions": 0,
        "relations_inferred": 0,
        "by_source_type": {}
    }

    # Process each extracted document
    extracted_files = sorted(EXTRACTED_DIR.glob("*.json"))
    extracted_files = [f for f in extracted_files if f.name != "_index.json"]

    for doc_path in extracted_files:
        with open(doc_path, encoding="utf-8") as f:
            doc = json.load(f)

        doc_id = doc["doc_id"]
        source_type = doc.get("source_type", "unknown")
        print(f"\n  Processing {doc_id} ({source_type})...")

        # Get text chunks — prefer sections for PMC, pages for PDFs
        text_chunks = []
        if "sections" in doc and doc["sections"]:
            for sec in doc["sections"]:
                text_chunks.append({
                    "text": sec["text"],
                    "section": sec.get("section_title", "unknown")
                })
        elif "pages" in doc and doc["pages"]:
            for page in doc["pages"]:
                text_chunks.append({
                    "text": page["text"],
                    "section": f"page_{page['page_number']}"
                })
        elif doc.get("full_text"):
            text_chunks.append({
                "text": doc["full_text"],
                "section": "full_document"
            })

        sent_idx = 0
        for chunk in text_chunks:
            sentences = split_sentences(chunk["text"])

            for sentence in sentences:
                entities = find_entities(sentence, drug_patterns, adr_patterns)
                has_drug = len(entities["drugs"]) > 0
                has_adr = len(entities["adrs"]) > 0
                contains_adr = 1 if (has_adr or _sentence_suggests_adr(sentence)) else 0

                relations = infer_relations(entities["drugs"], entities["adrs"])

                row = {
                    "sentence_id": generate_sentence_id(doc_id, sentence, sent_idx),
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "year": doc.get("year"),
                    "hospital": doc.get("hospital", ""),
                    "section": chunk["section"],
                    "sentence_index": sent_idx,
                    "text": sentence,
                    "contains_adr": contains_adr,
                    "entities": {
                        "drugs": entities["drugs"],
                        "adrs": entities["adrs"]
                    },
                    "relations": relations,
                    "auto_annotated": True,
                    "manually_reviewed": False
                }

                all_rows.append(row)
                sent_idx += 1

                # Update stats
                stats["total_sentences"] += 1
                if contains_adr:
                    stats["adr_positive"] += 1
                stats["drug_mentions"] += len(entities["drugs"])
                stats["adr_mentions"] += len(entities["adrs"])
                stats["relations_inferred"] += len(relations)

        source_count = stats["by_source_type"].get(source_type, 0)
        stats["by_source_type"][source_type] = source_count + sent_idx
        print(f"    → {sent_idx} sentences, {sum(1 for r in all_rows[-sent_idx:] if r['contains_adr'])} ADR-positive")

    # ── Save annotated dataset ──
    output_path = ANNOTATED_DIR / "ghana_adr_sentences.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Save stats ──
    stats_path = ANNOTATED_DIR / "annotation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ──
    print(f"\n═══ Segmentation & Auto-Annotation Summary ═══")
    print(f"  Total sentences:      {stats['total_sentences']:,}")
    print(f"  ADR-positive:         {stats['adr_positive']:,} ({100*stats['adr_positive']/max(stats['total_sentences'],1):.1f}%)")
    print(f"  Drug mentions:        {stats['drug_mentions']:,}")
    print(f"  ADR mentions:         {stats['adr_mentions']:,}")
    print(f"  Relations inferred:   {stats['relations_inferred']:,}")
    print(f"\n  By source type:")
    for st, count in sorted(stats["by_source_type"].items()):
        print(f"    {st:30s} {count:5d} sentences")
    print(f"\n  Output: {output_path}")
    print(f"  Stats:  {stats_path}")


def _sentence_suggests_adr(text: str) -> bool:
    """Heuristic: does sentence language suggest ADR content even without exact lexicon match?"""
    indicators = [
        r'\badverse\b.*\breaction\b',
        r'\bside\s+effect',
        r'\bdrug.?induced\b',
        r'\bdrug.?related\b',
        r'\bcaused\s+by\s+\w+\s+(therapy|treatment|medication|drug)',
        r'\bfollowing\s+(administration|treatment|intake|ingestion)\b',
        r'\bafter\s+taking\b',
        r'\bsuspected\s+(adr|adverse|reaction|event)',
        r'\bcausality\s+assessment\b',
        r'\bNaranjo\b',
        r'\bWHO-UMC\b',
        r'\bprobable\b.*\b(adr|reaction)\b',
        r'\bpossible\b.*\b(adr|reaction)\b'
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in indicators)


if __name__ == "__main__":
    main()
