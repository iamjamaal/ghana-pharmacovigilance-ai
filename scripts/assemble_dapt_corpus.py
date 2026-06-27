#!/usr/bin/env python3
"""
assemble_dapt_corpus.py — Build DAPT pretraining corpus for Ghana ADR pipeline Phase 3.

Sources (in priority order):
  1. corpora/inserts/       — DailyMed sentences (already downloaded)
  2. DrugLens PDFs          — https://www.fdaghana.gov.gh/druglens.html
  3. PMC Open Access        — NCBI Entrez, African pharmavigilance (up to 500 papers)
  4. AJOL open-access       — Ghana Medical Journal + West African Journal of Medicine

Output: corpora/dapt/{inserts,druglens,pmc_african,ajol}.txt
        Each file: one clean sentence per line.
"""

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz  # PyMuPDF
import nltk
import requests
from bs4 import BeautifulSoup

# Force UTF-8 output on Windows so Unicode in print() never raises.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── paths ─────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
INSERTS_DIR = ROOT / "corpora" / "inserts"
DAPT_DIR   = ROOT / "corpora" / "dapt"
DAPT_DIR.mkdir(parents=True, exist_ok=True)

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ── nltk setup ────────────────────────────────────────────────────────────────────
nltk.download("punkt_tab", quiet=True)


# ── text utilities ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove page headers, URLs, normalize whitespace."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[-=_]{5,}", " ", text)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)   # bare page numbers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list:
    """Sentence-split text; filter trivial lines."""
    sentences = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        try:
            sents = nltk.sent_tokenize(para)
        except Exception:
            sents = [para]
        for s in sents:
            s = " ".join(s.split())
            if len(s) < 15:
                continue
            alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
            if alpha_ratio < 0.40:
                continue
            sentences.append(s)
    return sentences


def write_sentences(sentences: list, path: Path) -> None:
    path.write_text("\n".join(sentences), encoding="utf-8")


def print_stats(path: Path, label: str) -> tuple:
    """Print per-file stats; return (sentences, tokens, mb)."""
    if not path.exists() or path.stat().st_size == 0:
        print(f"  {label}: EMPTY / NOT CREATED")
        return 0, 0, 0.0
    text = path.read_text(encoding="utf-8")
    sents = [l for l in text.splitlines() if l.strip()]
    words = len(text.split())
    tokens = int(words * 1.3)
    mb = path.stat().st_size / 1_000_000
    print(f"  {label}: {mb:.3f} MB | {len(sents):,} sentences | ~{tokens:,} tokens")
    return len(sents), tokens, mb


# ── Source 1: DailyMed inserts ────────────────────────────────────────────────────

def build_inserts(out_path: Path) -> None:
    print("\n[1/4] DailyMed inserts (corpora/inserts/)...")
    sentences = []
    files = sorted(INSERTS_DIR.glob("*.jsonl"))
    print(f"  {len(files)} JSONL files found")
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text = row.get("sentence", "")
                    if text:
                        sentences.extend(split_sentences(clean_text(text)))
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")

    sentences = list(dict.fromkeys(sentences))  # dedup, preserve order
    write_sentences(sentences, out_path)
    print_stats(out_path, "inserts.txt")


# ── Source 2: DrugLens PDFs ───────────────────────────────────────────────────────

DRUGLENS_URL = "https://fdaghana.gov.gh/druglens-2/"


def collect_pdf_urls(page_url: str) -> list:
    """Return all PDF URLs found on page_url and one link-level deep."""
    resp = requests.get(page_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    base = "https://www.fdaghana.gov.gh"

    def normalise(href: str) -> str:
        if href.startswith("http"):
            return href
        return base + "/" + href.lstrip("/")

    pdf_urls = set()
    sub_links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            pdf_urls.add(normalise(href))
        elif "druglens" in href.lower() or "drug" in href.lower():
            if not href.startswith("#") and len(href) > 2:
                sub_links.add(normalise(href))

    print(f"  Top-level PDFs: {len(pdf_urls)}, sub-pages to check: {len(sub_links)}")

    for url in list(sub_links)[:30]:
        if url == page_url:
            continue
        try:
            r2 = requests.get(url, headers=HEADERS, timeout=30)
            soup2 = BeautifulSoup(r2.text, "lxml")
            for a in soup2.find_all("a", href=True):
                href = a["href"].strip()
                if href.lower().endswith(".pdf"):
                    pdf_urls.add(normalise(href))
            time.sleep(0.5)
        except Exception:
            pass

    return sorted(pdf_urls)


def extract_pdf_sentences(pdf_bytes: bytes) -> list:
    """Extract sentence-split text from PDF bytes via PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        raw = "\n\n".join(pages)
        return split_sentences(clean_text(raw))
    except Exception as e:
        print(f"    PDF extract error: {e}")
        return []


def build_druglens(out_path: Path) -> None:
    print("\n[2/4] DrugLens PDFs (fdaghana.gov.gh)...")
    try:
        pdf_urls = collect_pdf_urls(DRUGLENS_URL)
        print(f"  Total PDF URLs found: {len(pdf_urls)}")
        sentences = []
        for url in pdf_urls:
            try:
                r = requests.get(url, headers=HEADERS, timeout=90)
                r.raise_for_status()
                new_sents = extract_pdf_sentences(r.content)
                sentences.extend(new_sents)
                fname = url.split("/")[-1]
                print(f"    {fname} ({len(r.content)//1024}KB) -> {len(new_sents)} sentences")
                time.sleep(1.0)
            except Exception as e:
                print(f"    FAILED {url}: {e}")

        sentences = list(dict.fromkeys(sentences))
        write_sentences(sentences, out_path)
    except Exception as e:
        print(f"  DrugLens FAILED entirely: {e}")
        if not out_path.exists():
            write_sentences([], out_path)
    print_stats(out_path, "druglens.txt")


# ── Source 3: PMC Open Access — African pharmacovigilance ─────────────────────────

PMC_QUERY = (
    "(adverse drug reaction[tiab] OR adverse event[tiab] OR pharmacovigilance[tiab])"
    ' AND (Ghana[tiab] OR Nigeria[tiab] OR Kenya[tiab] OR Uganda[tiab]'
    ' OR Ethiopia[tiab] OR "sub-Saharan Africa"[tiab])'
    ' AND "open access"[filter]'
)


def pmc_esearch(query: str, retmax: int = 500) -> list:
    params = {
        "db": "pmc",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "tool": "ghana-adr-pipeline",
        "email": "jnnoah3@st.knust.edu.gh",
    }
    r = requests.get(f"{ENTREZ_BASE}/esearch.fcgi", params=params,
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    result = data.get("esearchresult", {})
    count = result.get("count", "?")
    ids = result.get("idlist", [])
    print(f"  PMC search: {count} total hits, fetching {len(ids)}")
    return ids


def pmc_efetch_xml(pmc_ids: list) -> str:
    params = {
        "db": "pmc",
        "id": ",".join(pmc_ids),
        "retmode": "xml",
        "tool": "ghana-adr-pipeline",
        "email": "jnnoah3@st.knust.edu.gh",
    }
    r = requests.get(f"{ENTREZ_BASE}/efetch.fcgi", params=params,
                     headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.text


def extract_jats_text(xml_str: str) -> str:
    """Extract abstract + body paragraphs from JATS XML (PMC format)."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        print(f"    XML parse error: {e}")
        return ""

    texts = []
    for article in root.iter("article"):
        for title in article.iter("article-title"):
            t = " ".join(title.itertext()).strip()
            if t:
                texts.append(t)
        for abstract in article.iter("abstract"):
            for p in abstract.iter("p"):
                t = " ".join(p.itertext()).strip()
                if t:
                    texts.append(t)
        for body in article.iter("body"):
            for p in body.iter("p"):
                t = " ".join(p.itertext()).strip()
                if t:
                    texts.append(t)

    return "\n\n".join(texts)


def build_pmc(out_path: Path) -> None:
    print("\n[3/4] PMC Open Access — African pharmacovigilance...")
    try:
        pmc_ids = pmc_esearch(PMC_QUERY, retmax=500)
        if not pmc_ids:
            print("  No results — writing empty file")
            write_sentences([], out_path)
            print_stats(out_path, "pmc_african.txt")
            return

        sentences = []
        batch_size = 20
        total_batches = (len(pmc_ids) + batch_size - 1) // batch_size

        for i in range(0, len(pmc_ids), batch_size):
            batch = pmc_ids[i: i + batch_size]
            batch_num = i // batch_size + 1
            try:
                xml_str = pmc_efetch_xml(batch)
                text = extract_jats_text(xml_str)
                new_sents = split_sentences(clean_text(text))
                sentences.extend(new_sents)
                print(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"+{len(new_sents)} sentences (total {len(sentences)})"
                )
            except Exception as e:
                print(f"  Batch {batch_num}/{total_batches} FAILED: {e}")
            time.sleep(0.35)

        sentences = list(dict.fromkeys(sentences))
        write_sentences(sentences, out_path)
    except Exception as e:
        print(f"  PMC FAILED entirely: {e}")
        if not out_path.exists():
            write_sentences([], out_path)
    print_stats(out_path, "pmc_african.txt")


# ── Source 4: AJOL open-access ────────────────────────────────────────────────────

AJOL_JOURNALS = [
    ("Ghana Medical Journal",           "https://www.ajol.info/index.php/gmj"),
    ("West African Journal of Medicine","https://www.ajol.info/index.php/wajm"),
]


def ajol_issue_urls(journal_base: str, max_issues: int = 20) -> list:
    archive = f"{journal_base}/issue/archive"
    r = requests.get(archive, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/issue/view/" in href:
            url = href if href.startswith("http") else "https://www.ajol.info" + href
            urls.append(url)
    return list(dict.fromkeys(urls))[:max_issues]


def ajol_article_urls(issue_url: str) -> list:
    r = requests.get(issue_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/article/view/" in href and "download" not in href.lower():
            url = href if href.startswith("http") else "https://www.ajol.info" + href
            urls.append(url)
    return list(dict.fromkeys(urls))


def ajol_article_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    parts = []

    h1 = soup.find("h1") or soup.find("h2")
    if h1:
        parts.append(h1.get_text(" ", strip=True))

    for tag in soup.find_all(["div", "section", "article"],
                              class_=re.compile(r"abstract|full.?text|body", re.I)):
        parts.append(tag.get_text(" ", strip=True))

    if not parts:
        main = (soup.find("div", id=re.compile(r"main|content", re.I))
                or soup.find("main")
                or soup.find("article"))
        if main:
            parts.append(main.get_text(" ", strip=True))

    return "\n\n".join(parts)


def build_ajol(out_path: Path) -> None:
    print("\n[4/4] AJOL open-access (GMJ + WAJM)...")
    all_sentences = []

    for journal_name, journal_base in AJOL_JOURNALS:
        print(f"  {journal_name}...")
        try:
            issues = ajol_issue_urls(journal_base, max_issues=20)
            print(f"    {len(issues)} issues found")

            article_urls = []
            for issue_url in issues:
                try:
                    article_urls.extend(ajol_article_urls(issue_url))
                    time.sleep(0.4)
                except Exception:
                    pass

            article_urls = list(dict.fromkeys(article_urls))
            print(f"    {len(article_urls)} article URLs")

            journal_sentences = []
            for url in article_urls[:120]:
                try:
                    text = ajol_article_text(url)
                    journal_sentences.extend(split_sentences(clean_text(text)))
                    time.sleep(0.5)
                except Exception:
                    pass

            all_sentences.extend(journal_sentences)
            print(f"    -> {len(journal_sentences)} sentences")
        except Exception as e:
            print(f"  {journal_name} FAILED: {e}")

    all_sentences = list(dict.fromkeys(all_sentences))
    write_sentences(all_sentences, out_path)
    print_stats(out_path, "ajol.txt")


# ── main ──────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("DAPT Corpus Assembly — Ghana ADR Pipeline Phase 3")
    print("=" * 64)

    build_inserts(DAPT_DIR / "inserts.txt")
    build_druglens(DAPT_DIR / "druglens.txt")
    build_pmc(DAPT_DIR / "pmc_african.txt")
    build_ajol(DAPT_DIR / "ajol.txt")

    print("\n" + "=" * 64)
    print("CORPUS SUMMARY")
    print("=" * 64)
    total_sents = total_tokens = 0
    total_mb = 0.0
    for name in ["inserts.txt", "druglens.txt", "pmc_african.txt", "ajol.txt"]:
        s, t, m = print_stats(DAPT_DIR / name, name)
        total_sents += s
        total_tokens += t
        total_mb += m

    print(f"\n  TOTAL: {total_mb:.3f} MB | {total_sents:,} sentences | ~{total_tokens:,} tokens")
    ok_sents = total_sents >= 20_000
    ok_mb    = total_mb >= 5.0
    if ok_sents and ok_mb:
        print("  TARGETS MET (>=20k sentences, >=5 MB)")
    else:
        gaps = []
        if not ok_sents:
            gaps.append(f"sentences short by {20_000 - total_sents:,}")
        if not ok_mb:
            gaps.append(f"size short by {5.0 - total_mb:.3f} MB")
        print(f"  TARGETS MISSED: {'; '.join(gaps)}")


if __name__ == "__main__":
    main()
