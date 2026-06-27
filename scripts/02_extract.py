#!/usr/bin/env python3
"""
Step 2: Extract text from downloaded sources.
- PDFs → plain text via PyMuPDF
- PMC XML → structured sections via BeautifulSoup/lxml
Outputs one JSON file per source document in data/extracted/.
"""

import json
import re
import os
from pathlib import Path
from bs4 import BeautifulSoup

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    exit(1)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
CONFIG_PATH = BASE_DIR / "config" / "sources.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """Normalize whitespace, fix common PDF artifacts."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[a-z])-\s+(?=[a-z])', '', text)  # fix hyphenation
    text = text.replace('\x00', '')
    return text.strip()


def extract_pdf(pdf_path: Path) -> dict:
    """Extract all text from a PDF, page by page."""
    doc = fitz.open(str(pdf_path))
    pages = []
    full_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        cleaned = clean_text(text)
        if cleaned:
            pages.append({
                "page_number": i + 1,
                "text": cleaned
            })
            full_text.append(cleaned)

    doc.close()

    return {
        "total_pages": len(pages),
        "pages": pages,
        "full_text": " ".join(full_text)
    }


def extract_pmc_xml(xml_path: Path, sections_of_interest: list) -> dict:
    """
    Extract structured text from PMC JATS XML.
    Pulls title, abstract, and specified sections (case-presentation, results, discussion).
    """
    with open(xml_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")

    result = {
        "title": "",
        "abstract": "",
        "sections": [],
        "full_text": ""
    }

    # ── Title ──
    title_tag = soup.find("article-title")
    if title_tag:
        result["title"] = clean_text(title_tag.get_text())

    # ── Abstract ──
    abstract_tag = soup.find("abstract")
    if abstract_tag:
        result["abstract"] = clean_text(abstract_tag.get_text())

    # ── Body sections ──
    body = soup.find("body")
    if body:
        for sec in body.find_all("sec", recursive=True):
            # Get section title
            sec_title_tag = sec.find("title", recursive=False)
            sec_title = clean_text(sec_title_tag.get_text()) if sec_title_tag else "untitled"
            sec_id = sec.get("id", "").lower()
            sec_type = sec.get("sec-type", "").lower()

            # Check if this section is of interest
            sec_key = sec_title.lower().replace(" ", "-")
            is_relevant = any(
                s in sec_key or s in sec_id or s in sec_type
                for s in sections_of_interest
            )

            # Also include any section that mentions drugs or reactions
            sec_text = clean_text(sec.get_text())

            if is_relevant or _text_likely_contains_adr(sec_text):
                result["sections"].append({
                    "section_title": sec_title,
                    "section_id": sec_id or sec_key,
                    "text": sec_text,
                    "is_priority": is_relevant
                })

    # Build full text
    all_parts = [result["abstract"]]
    all_parts.extend(s["text"] for s in result["sections"])
    result["full_text"] = " ".join(filter(None, all_parts))

    return result


def _text_likely_contains_adr(text: str) -> bool:
    """Quick heuristic check if text might contain ADR mentions."""
    adr_keywords = [
        "adverse", "reaction", "side effect", "toxicity", "rash",
        "nausea", "vomiting", "headache", "dizziness", "fever",
        "hepato", "nephro", "anaemia", "anemia", "thrombocytopenia",
        "stevens-johnson", "anaphyla", "seizure", "convulsion",
        "drug-induced", "drug related", "suspected", "causality"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in adr_keywords)


def main():
    config = load_config()
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    all_documents = []

    # ── Extract DrugLens PDFs ──
    print("\n═══ Extracting DrugLens PDFs ═══")
    for item in config["druglens_pdfs"]:
        pdf_path = RAW_DIR / "pdfs" / f"{item['id']}.pdf"
        if not pdf_path.exists():
            print(f"  [SKIP] {item['id']} — not downloaded")
            continue

        print(f"  [EXTRACT] {item['id']}...")
        extracted = extract_pdf(pdf_path)

        doc = {
            "doc_id": item["id"],
            "source_type": item["source_type"],
            "year": item["year"],
            "origin": f"Ghana FDA DrugLens Issue {item['issue']}",
            "license": "Ghana Government Publication",
            "extraction_method": "pymupdf",
            **extracted
        }
        all_documents.append(doc)

        # Save individual file
        out_path = EXTRACTED_DIR / f"{item['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"  [OK]   {item['id']} — {len(extracted['full_text'])} chars, {extracted['total_pages']} pages")

    # ── Extract FDA reports ──
    print("\n═══ Extracting FDA Reports ═══")
    for item in config["fda_reports"]:
        pdf_path = RAW_DIR / "pdfs" / f"{item['id']}.pdf"
        if not pdf_path.exists():
            print(f"  [SKIP] {item['id']} — not downloaded")
            continue

        print(f"  [EXTRACT] {item['id']}...")
        extracted = extract_pdf(pdf_path)

        doc = {
            "doc_id": item["id"],
            "source_type": item["source_type"],
            "year": item["year"],
            "origin": f"Ghana FDA {item['source_type'].replace('_', ' ').title()}",
            "license": "Ghana Government Publication",
            "extraction_method": "pymupdf",
            **extracted
        }
        all_documents.append(doc)

        out_path = EXTRACTED_DIR / f"{item['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"  [OK]   {item['id']} — {len(extracted['full_text'])} chars")

    # ── Extract PMC articles ──
    print("\n═══ Extracting PMC Articles ═══")
    for item in config["pmc_articles"]:
        xml_path = RAW_DIR / "pmc_xml" / f"{item['id']}_{item['pmcid']}.xml"
        if not xml_path.exists():
            print(f"  [SKIP] {item['id']} — not downloaded")
            continue

        print(f"  [EXTRACT] {item['id']} ({item['pmcid']})...")
        extracted = extract_pmc_xml(xml_path, item.get("sections_of_interest", []))

        doc = {
            "doc_id": item["id"],
            "pmcid": item["pmcid"],
            "source_type": item["source_type"],
            "year": item["year"],
            "hospital": item.get("hospital", ""),
            "origin": f"PMC Open Access — {item['title']}",
            "license": "CC-BY (PMC Open Access)",
            "extraction_method": "pmc_xml_jats",
            "title": extracted["title"],
            "abstract": extracted["abstract"],
            "sections": extracted["sections"],
            "full_text": extracted["full_text"],
            "n_sections": len(extracted["sections"]),
            "n_priority_sections": sum(1 for s in extracted["sections"] if s.get("is_priority"))
        }
        all_documents.append(doc)

        out_path = EXTRACTED_DIR / f"{item['id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"  [OK]   {item['id']} — {len(extracted['full_text'])} chars, {len(extracted['sections'])} sections")

    # ── Summary ──
    print(f"\n═══ Extraction Summary ═══")
    print(f"  Documents extracted: {len(all_documents)}")
    total_chars = sum(len(d.get("full_text", "")) for d in all_documents)
    print(f"  Total text:          {total_chars:,} characters (~{total_chars // 5:,} tokens)")

    # Save master index
    index_path = EXTRACTED_DIR / "_index.json"
    index = [
        {
            "doc_id": d["doc_id"],
            "source_type": d["source_type"],
            "year": d["year"],
            "origin": d["origin"],
            "char_count": len(d.get("full_text", ""))
        }
        for d in all_documents
    ]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"  Index saved to {index_path}")


if __name__ == "__main__":
    main()
