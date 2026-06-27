"""
Labeling functions for the Ghana ADR weak supervision pipeline.

Four LFs operate on spaCy Doc objects and emit span annotations:
  lf_drug_gazetteer     -- drug name match → DRUG span
  lf_adr_gazetteer      -- ADR term match  → ADR span
  lf_trigger_proximity  -- drug + trigger word + ADR within 15 tokens → ADR span
  lf_negation           -- NegEx cue before ADR candidate → NEG_ADR span

Aggregation (aggregate_doc / aggregate_batch) uses skweak HMM when available;
falls back to a lightweight precision-weighted majority vote if skweak cannot be
imported (e.g. Python version incompatibility).

Public helpers
--------------
load_gazetteers()          -- returns (drug_terms, adr_terms)
build_annotators(nlp, ...)  -- returns (drug_ann, adr_ann, trigger_ann, neg_ann)
aggregate_batch(docs, annotators)  -- returns list of (doc, contains_adr, spans)
predict_single(text, nlp, annotators) -- returns dict with all LF outputs
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import spacy
from spacy.tokens import Doc, Span

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIGGER_WORDS: List[str] = [
    "after",
    "following",
    "caused",
    "associated with",
    "reported with",
    "cases of",
    "due to",
    "secondary to",
    "induced by",
    "related to",
]

NEGATION_CUES: Set[str] = {
    "no",
    "not",
    "without",
    "denies",
    "denied",
    "ruled out",
    "negative for",
    "absence of",
    "none",
    "neither",
    "nor",
    "contraindicated",
    "contraindicated in",
    "patients with known",
    "history of",
}

_GAZETTEER_DIR = Path(__file__).parent.parent.parent / "data" / "gazetteers"


# ---------------------------------------------------------------------------
# Gazetteer loading
# ---------------------------------------------------------------------------

def load_gazetteers(
    drugs_path: Path | None = None,
    adrs_path: Path | None = None,
) -> Tuple[Dict[str, str], Set[str]]:
    """Return (drug_terms, adr_terms).

    drug_terms: {lowercase_term → canonical_name}
    adr_terms:  {lowercase_term}
    """
    drugs_path = drugs_path or (_GAZETTEER_DIR / "drugs.json")
    adrs_path = adrs_path or (_GAZETTEER_DIR / "adrs.json")

    drug_terms: Dict[str, str] = {}
    with open(drugs_path, encoding="utf-8") as fh:
        for entry in json.load(fh):
            canonical = entry["canonical"]
            all_forms = (
                [canonical]
                + entry.get("synonyms", [])
                + entry.get("lay_terms", [])
            )
            for form in all_forms:
                drug_terms[form.lower().strip(".,")] = canonical

    adr_terms: Set[str] = set()
    with open(adrs_path, encoding="utf-8") as fh:
        for entry in json.load(fh):
            adr_terms.add(entry["term"].lower())

    return drug_terms, adr_terms


# ---------------------------------------------------------------------------
# Span utility: longest-match, no overlaps
# ---------------------------------------------------------------------------

def _longest_match(
    text: str, term_set: Set[str] | Dict[str, str], label: str
) -> List[Tuple[int, int, str]]:
    """Case-insensitive longest-match scan. Returns list of (start_char, end_char, label)."""
    lower = text.lower()
    terms = sorted(
        (term_set if isinstance(term_set, set) else term_set.keys()),
        key=len,
        reverse=True,  # longest first so they shadow shorter overlapping matches
    )
    occupied: List[Tuple[int, int]] = []
    hits: List[Tuple[int, int, str]] = []

    for term in terms:
        pos = 0
        while True:
            idx = lower.find(term, pos)
            if idx == -1:
                break
            end = idx + len(term)
            # Enforce word boundaries: char before/after must not be alphanumeric or '-'
            before_ok = idx == 0 or not (lower[idx - 1].isalnum() or lower[idx - 1] == "-")
            after_ok = end == len(lower) or not (lower[end].isalnum() or lower[end] == "-")
            if before_ok and after_ok:
                # Check for overlap with already recorded spans
                overlap = any(s < end and e > idx for s, e in occupied)
                if not overlap:
                    hits.append((idx, end, label))
                    occupied.append((idx, end))
            pos = idx + 1

    return hits


def _char_to_token_span(doc: Doc, start_char: int, end_char: int) -> Span | None:
    """Convert character offsets to a spaCy Span; returns None if alignment fails."""
    try:
        span = doc.char_span(start_char, end_char, alignment_mode="expand")
        return span
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LF base class (lightweight, no skweak dependency)
# ---------------------------------------------------------------------------

class LFAnnotator:
    """Base class for labeling function annotators.

    Each annotator adds its results to doc.user_data[name] as a list of
    {"start": int, "end": int, "label": str} dicts (token indices).
    """

    def __init__(self, name: str):
        self.name = name

    def annotate(self, doc: Doc) -> List[Dict]:
        """Return a list of span dicts with token-level start/end and label."""
        raise NotImplementedError

    def __call__(self, doc: Doc) -> Doc:
        if "lf_spans" not in doc.user_data:
            doc.user_data["lf_spans"] = {}
        doc.user_data["lf_spans"][self.name] = self.annotate(doc)
        return doc


# ---------------------------------------------------------------------------
# LF 1: Drug gazetteer
# ---------------------------------------------------------------------------

class DrugGazetteerAnnotator(LFAnnotator):
    def __init__(self, drug_terms: Dict[str, str]):
        super().__init__("lf_drug_gazetteer")
        self.drug_terms = drug_terms

    def annotate(self, doc: Doc) -> List[Dict]:
        spans = []
        for start_char, end_char, _ in _longest_match(doc.text, self.drug_terms, "DRUG"):
            tok_span = _char_to_token_span(doc, start_char, end_char)
            if tok_span is not None:
                spans.append({"start": tok_span.start, "end": tok_span.end,
                               "start_char": start_char, "end_char": end_char, "label": "DRUG"})
        return spans


# ---------------------------------------------------------------------------
# LF 2: ADR gazetteer
# ---------------------------------------------------------------------------

class ADRGazetteerAnnotator(LFAnnotator):
    def __init__(self, adr_terms: Set[str]):
        super().__init__("lf_adr_gazetteer")
        self.adr_terms = adr_terms

    def annotate(self, doc: Doc) -> List[Dict]:
        spans = []
        for start_char, end_char, _ in _longest_match(doc.text, self.adr_terms, "ADR"):
            tok_span = _char_to_token_span(doc, start_char, end_char)
            if tok_span is not None:
                spans.append({"start": tok_span.start, "end": tok_span.end,
                               "start_char": start_char, "end_char": end_char, "label": "ADR"})
        return spans


# ---------------------------------------------------------------------------
# LF 3: Trigger proximity
# ---------------------------------------------------------------------------

class TriggerProximityAnnotator(LFAnnotator):
    """Emits ADR spans found within 15 tokens of a trigger word that is
    itself within 15 tokens of a DRUG span."""

    def __init__(self, drug_terms: Dict[str, str], adr_terms: Set[str], window: int = 15):
        super().__init__("lf_trigger_proximity")
        self.drug_terms = drug_terms
        self.adr_terms = adr_terms
        self.window = window

    def annotate(self, doc: Doc) -> List[Dict]:
        # Run dependent LFs in-place to find drug/ADR positions
        drug_hits = _longest_match(doc.text, self.drug_terms, "DRUG")
        adr_hits = _longest_match(doc.text, self.adr_terms, "ADR")

        drug_token_ends: List[int] = []
        for sc, ec, _ in drug_hits:
            tok = _char_to_token_span(doc, sc, ec)
            if tok:
                drug_token_ends.append(tok.end)

        # Build trigger word token positions
        trigger_positions: List[int] = []
        text_lower = doc.text.lower()
        for tw in sorted(TRIGGER_WORDS, key=len, reverse=True):
            pos = 0
            while True:
                idx = text_lower.find(tw, pos)
                if idx == -1:
                    break
                end = idx + len(tw)
                tok = _char_to_token_span(doc, idx, end)
                if tok:
                    trigger_positions.append(tok.start)
                pos = idx + 1

        if not drug_token_ends or not trigger_positions:
            return []

        # For each drug span, look for a trigger within window tokens
        triggered_trigger_toks: List[int] = []
        for drug_end in drug_token_ends:
            for trig_tok in trigger_positions:
                if abs(trig_tok - drug_end) <= self.window:
                    triggered_trigger_toks.append(trig_tok)

        if not triggered_trigger_toks:
            return []

        # For each triggered trigger, look for ADR span within window tokens
        new_spans: List[Dict] = []
        for sc, ec, _ in adr_hits:
            tok = _char_to_token_span(doc, sc, ec)
            if tok is None:
                continue
            for trig_tok in triggered_trigger_toks:
                if abs(tok.start - trig_tok) <= self.window:
                    new_spans.append({
                        "start": tok.start,
                        "end": tok.end,
                        "start_char": sc,
                        "end_char": ec,
                        "label": "ADR",
                    })
                    break
        return new_spans


# ---------------------------------------------------------------------------
# LF 4: Negation
# ---------------------------------------------------------------------------

class NegationAnnotator(LFAnnotator):
    """Marks ADR spans that are preceded (within 5 tokens) by a negation cue.

    Emits the same span with label "NEG_ADR" so the aggregator can treat
    it as counter-evidence for contains_adr = 1.
    """

    def __init__(self, adr_terms: Set[str], left_window: int = 5):
        super().__init__("lf_negation")
        self.adr_terms = adr_terms
        self.left_window = left_window

    def annotate(self, doc: Doc) -> List[Dict]:
        adr_hits = _longest_match(doc.text, self.adr_terms, "ADR")
        text_lower = doc.text.lower()

        # Pre-compute negation cue token positions
        neg_token_set: Set[int] = set()
        for cue in sorted(NEGATION_CUES, key=len, reverse=True):
            pos = 0
            while True:
                idx = text_lower.find(cue, pos)
                if idx == -1:
                    break
                end = idx + len(cue)
                before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
                after_ok = end == len(text_lower) or not text_lower[end].isalpha()
                if before_ok and after_ok:
                    tok = _char_to_token_span(doc, idx, end)
                    if tok:
                        for ti in range(tok.start, tok.end):
                            neg_token_set.add(ti)
                pos = idx + 1

        negated: List[Dict] = []
        for sc, ec, _ in adr_hits:
            tok = _char_to_token_span(doc, sc, ec)
            if tok is None:
                continue
            # Check if any negation token falls within left_window tokens before span
            for neg_ti in neg_token_set:
                if 0 <= tok.start - neg_ti <= self.left_window:
                    negated.append({
                        "start": tok.start,
                        "end": tok.end,
                        "start_char": sc,
                        "end_char": ec,
                        "label": "NEG_ADR",
                    })
                    break
        return negated


# ---------------------------------------------------------------------------
# Build all annotators
# ---------------------------------------------------------------------------

def build_annotators(
    drug_terms: Dict[str, str],
    adr_terms: Set[str],
) -> Tuple[DrugGazetteerAnnotator, ADRGazetteerAnnotator,
           TriggerProximityAnnotator, NegationAnnotator]:
    return (
        DrugGazetteerAnnotator(drug_terms),
        ADRGazetteerAnnotator(adr_terms),
        TriggerProximityAnnotator(drug_terms, adr_terms),
        NegationAnnotator(adr_terms),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _apply_all_lfs(doc: Doc, annotators) -> Doc:
    for ann in annotators:
        ann(doc)
    return doc


def _spans_from_doc(doc: Doc, lf_name: str) -> List[Dict]:
    return doc.user_data.get("lf_spans", {}).get(lf_name, [])


def _aggregate_lightweight(doc: Doc) -> Tuple[int, List[Dict]]:
    """Lightweight aggregation without skweak.

    Logic:
      - Collect all ADR spans from lf_drug_gazetteer (no ADRs), lf_adr_gazetteer,
        and lf_trigger_proximity.
      - Subtract spans covered by lf_negation (NEG_ADR).
      - contains_adr = 1 if any non-negated ADR span exists.

    LF weights (precision proxy): trigger_proximity > adr_gazetteer.
    """
    adr_gazetteer_spans = _spans_from_doc(doc, "lf_adr_gazetteer")
    trigger_spans = _spans_from_doc(doc, "lf_trigger_proximity")
    neg_spans = _spans_from_doc(doc, "lf_negation")

    neg_positions: Set[Tuple[int, int]] = {(s["start"], s["end"]) for s in neg_spans}

    surviving: List[Dict] = []
    for span in adr_gazetteer_spans + trigger_spans:
        key = (span["start"], span["end"])
        if key not in neg_positions:
            surviving.append(span)

    # Deduplicate by token position
    seen: Set[Tuple[int, int]] = set()
    deduped: List[Dict] = []
    for span in surviving:
        key = (span["start"], span["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(span)

    return (1 if deduped else 0), deduped


def _aggregate_with_skweak(docs: List[Doc], annotators) -> List[Tuple[int, List[Dict]]]:
    """HMM aggregation via skweak. Mutates docs with skweak span annotations."""
    try:
        import skweak  # noqa: F401 — confirm importable
        from skweak import heuristics as sk_heuristics, aggregation as sk_agg
    except ImportError:
        return None

    # skweak wraps our annotators as SpanAnnotator subclasses.
    # Since our LFs already ran and stored results in doc.user_data, we wrap them
    # as trivial passthrough annotators that replay stored spans into doc.spans.

    class _ReplayAnnotator(sk_heuristics.SpanAnnotator):
        def __init__(self, lf_name: str, label_map: Dict[str, str]):
            super().__init__(lf_name)
            self.lf_name = lf_name
            self.label_map = label_map

        def find_spans(self, doc: Doc) -> Iterator[Tuple[int, int, str]]:
            for s in doc.user_data.get("lf_spans", {}).get(self.lf_name, []):
                mapped = self.label_map.get(s["label"])
                if mapped:
                    yield s["start"], s["end"], mapped

    # Map labels for skweak's span key
    drug_replay = _ReplayAnnotator("lf_drug_gazetteer", {"DRUG": "DRUG"})
    adr_replay = _ReplayAnnotator("lf_adr_gazetteer", {"ADR": "ADR"})
    trigger_replay = _ReplayAnnotator("lf_trigger_proximity", {"ADR": "ADR"})
    # Negation mapped to "O" so HMM treats it as non-entity evidence
    neg_replay = _ReplayAnnotator("lf_negation", {"NEG_ADR": "O"})

    sk_annotators = [drug_replay, adr_replay, trigger_replay, neg_replay]
    for ann in sk_annotators:
        docs = list(ann.pipe(docs))

    hmm = sk_agg.HMM("hmm", ["DRUG", "ADR"])
    try:
        docs = list(hmm.fit_and_aggregate(docs))
    except Exception:
        return None

    results: List[Tuple[int, List[Dict]]] = []
    for doc in docs:
        agg_spans = doc.spans.get("hmm", [])
        # Negation positions from raw LF
        neg_positions: Set[Tuple[int, int]] = {
            (s["start"], s["end"])
            for s in doc.user_data.get("lf_spans", {}).get("lf_negation", [])
        }
        surviving = [
            {"start": sp.start, "end": sp.end,
             "start_char": sp.start_char, "end_char": sp.end_char,
             "label": sp.label_}
            for sp in agg_spans
            if sp.label_ == "ADR" and (sp.start, sp.end) not in neg_positions
        ]
        results.append((1 if surviving else 0, surviving))
    return results


def aggregate_batch(
    docs: List[Doc],
    annotators,
    use_skweak: bool = True,
) -> List[Tuple[int, List[Dict]]]:
    """Apply all LFs and aggregate.

    Returns list of (contains_adr, adr_spans) per doc.
    Tries skweak HMM first; falls back to lightweight aggregation if unavailable.
    """
    for doc in docs:
        _apply_all_lfs(doc, annotators)

    if use_skweak:
        result = _aggregate_with_skweak(docs, annotators)
        if result is not None:
            return result

    return [_aggregate_lightweight(doc) for doc in docs]


def predict_single(
    text: str,
    nlp,
    annotators,
    use_skweak: bool = True,
) -> Dict:
    """Run all LFs on a single text and return full LF output dict."""
    doc = nlp(text)
    _apply_all_lfs(doc, annotators)

    lf_spans = doc.user_data.get("lf_spans", {})

    # Per-LF binary (standalone)
    lf_adr_binary = {
        name: (1 if any(s["label"] == "ADR" for s in spans) else 0)
        for name, spans in lf_spans.items()
        if name != "lf_negation"
    }
    neg_fires = bool(lf_spans.get("lf_negation", []))
    # Negation LF standalone signal: 0 if it fires on any ADR
    lf_adr_binary["lf_negation"] = 0 if neg_fires else -1  # -1 = abstain

    # Aggregated prediction
    contains_adr, adr_spans = _aggregate_lightweight(doc)
    if use_skweak:
        result = _aggregate_with_skweak([doc], annotators)
        if result:
            contains_adr, adr_spans = result[0]

    return {
        "text": text,
        "contains_adr": contains_adr,
        "adr_spans": adr_spans,
        "drug_spans": lf_spans.get("lf_drug_gazetteer", []),
        "lf_standalone": lf_adr_binary,
        "neg_fires": neg_fires,
    }
