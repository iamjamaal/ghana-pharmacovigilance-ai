"""
Step 3 — Validate LF quality on the fda_newsletter gold fold.

Loads the ~99 manually-reviewed fda_newsletter sentences from
output/gold/ghana_adr_gold_classification.jsonl, applies all 4 LFs
plus aggregation, then reports:

  - Overall aggregated precision / recall / F1
  - Per-LF standalone precision / recall / F1
  - Which LF is the primary noise source (lowest standalone precision)

Target: aggregated precision >= 0.70

Usage:
    python src/weak_supervision/validate_silver.py
    python src/weak_supervision/validate_silver.py --gold output/gold/ghana_adr_gold_classification.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import spacy

# lfs.py is a sibling module in src/weak_supervision/
from lfs import (
    load_gazetteers,
    build_annotators,
    aggregate_batch,
    _spans_from_doc,
)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Per-LF standalone binary prediction
# ---------------------------------------------------------------------------

def _lf_standalone_predict(doc, lf_name: str) -> int:
    """Returns 1 if the LF emits at least one ADR span, 0 otherwise.

    For lf_negation: returns 0 if it fires (negation → no ADR), -1 (abstain) if not.
    """
    spans = _spans_from_doc(doc, lf_name)
    if lf_name == "lf_negation":
        return 0 if spans else -1
    return 1 if any(s["label"] == "ADR" for s in spans) else 0


def _lf_contribution_to_fp(lf_name: str, doc, gold_label: int) -> bool:
    """True if this LF produces a false positive on this sentence."""
    pred = _lf_standalone_predict(doc, lf_name)
    if lf_name == "lf_negation":
        return False  # negation doesn't produce FPs by design
    return pred == 1 and gold_label == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate LFs on fda_newsletter gold")
    parser.add_argument(
        "--gold",
        default=None,
        help="Path to ghana_adr_gold_classification.jsonl",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    gold_path = Path(args.gold) if args.gold else (
        repo_root / "output" / "gold" / "ghana_adr_gold_classification.jsonl"
    )

    if not gold_path.exists():
        print(f"ERROR: gold file not found: {gold_path}")
        sys.exit(1)

    # Load fda_newsletter gold rows (manually reviewed, label != -1)
    gold_rows: List[Dict] = []
    with open(gold_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (
                row.get("source_type") == "fda_newsletter"
                and row.get("manually_reviewed", False)
                and row.get("label", -1) != -1
            ):
                gold_rows.append(row)

    if not gold_rows:
        print("ERROR: no fda_newsletter manually-reviewed rows found in gold file")
        sys.exit(1)

    print(f"Loaded {len(gold_rows)} fda_newsletter gold sentences")
    pos_count = sum(1 for r in gold_rows if r["label"] == 1)
    print(f"  Positive (contains_adr=1): {pos_count}")
    print(f"  Negative (contains_adr=0): {len(gold_rows) - pos_count}")
    print(f"  Positive rate: {100*pos_count/len(gold_rows):.1f}%\n")

    print("Loading spaCy model…")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    print("Loading gazetteers…")
    drug_terms, adr_terms = load_gazetteers()
    annotators = build_annotators(drug_terms, adr_terms)
    ann_names = [a.name for a in annotators]

    texts = [r["text"] for r in gold_rows]
    gold_labels = [r["label"] for r in gold_rows]

    print("Applying LFs and aggregating…")
    docs = list(nlp.pipe(texts))
    # aggregate_batch applies all LFs internally then aggregates
    agg_results = aggregate_batch(docs, annotators)
    agg_preds = [c for c, _ in agg_results]

    # ---------------------------------------------------------------------------
    # Aggregated metrics
    # ---------------------------------------------------------------------------
    tp = sum(1 for p, g in zip(agg_preds, gold_labels) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(agg_preds, gold_labels) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(agg_preds, gold_labels) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(agg_preds, gold_labels) if p == 0 and g == 0)

    agg_p, agg_r, agg_f = prf(tp, fp, fn)

    print("\n" + "=" * 55)
    print("AGGREGATED LF METRICS (fda_newsletter gold)")
    print("=" * 55)
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    target_str = "PASS (>= 0.70)" if agg_p >= 0.70 else "FAIL (< 0.70)  <-- BELOW TARGET"
    print(f"  Precision : {agg_p:.3f}  {target_str}")
    print(f"  Recall    : {agg_r:.3f}")
    print(f"  F1        : {agg_f:.3f}")

    # ---------------------------------------------------------------------------
    # Per-LF standalone metrics
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("PER-LF STANDALONE METRICS")
    print("-" * 55)

    lf_precisions: Dict[str, float] = {}
    for lf_name in ann_names:
        lf_preds = [_lf_standalone_predict(doc, lf_name) for doc in docs]
        # For aggregation, treat abstain (-1) as 0
        lf_preds_binary = [max(0, p) for p in lf_preds]
        lf_tp = sum(1 for p, g in zip(lf_preds_binary, gold_labels) if p == 1 and g == 1)
        lf_fp = sum(1 for p, g in zip(lf_preds_binary, gold_labels) if p == 1 and g == 0)
        lf_fn = sum(1 for p, g in zip(lf_preds_binary, gold_labels) if p == 0 and g == 1)
        lf_p, lf_r, lf_f = prf(lf_tp, lf_fp, lf_fn)
        lf_precisions[lf_name] = lf_p
        abstains = sum(1 for p in lf_preds if p == -1)
        print(f"  {lf_name:<28}  P={lf_p:.3f}  R={lf_r:.3f}  F1={lf_f:.3f}"
              f"  TP={lf_tp} FP={lf_fp} FN={lf_fn}"
              + (f"  abstains={abstains}" if abstains else ""))

    # ---------------------------------------------------------------------------
    # Noise source analysis
    # ---------------------------------------------------------------------------
    if agg_p < 0.70:
        print("\n" + "-" * 55)
        print("NOISE SOURCE ANALYSIS  (precision < 0.70 — diagnosing main FP source)")
        print("-" * 55)
        # Count per-LF FP contributions
        fp_counts: Dict[str, int] = {name: 0 for name in ann_names if name != "lf_negation"}
        for doc, gold in zip(docs, gold_labels):
            for lf_name in ann_names:
                if _lf_contribution_to_fp(lf_name, doc, gold):
                    fp_counts[lf_name] += 1
        sorted_fps = sorted(fp_counts.items(), key=lambda x: -x[1])
        for lf_name, count in sorted_fps:
            print(f"  {lf_name:<28}  FP contributions: {count}")
        noisy_lf = sorted_fps[0][0] if sorted_fps else "unknown"
        print(f"\n  Primary noise source: {noisy_lf} (P={lf_precisions.get(noisy_lf, 0):.3f})")
        print("  Recommendation: tighten this LF's matching criteria or expand negation cues.")
    else:
        print("\n  Precision target met. LF quality sufficient for silver label generation.")

    # ---------------------------------------------------------------------------
    # FP examples
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("FALSE POSITIVE EXAMPLES (aggregated, up to 5)")
    print("-" * 55)
    shown = 0
    for doc, pred, gold, row in zip(docs, agg_preds, gold_labels, gold_rows):
        if pred == 1 and gold == 0 and shown < 5:
            adr_spans = [s for s in doc.user_data.get("lf_spans", {}).get("lf_adr_gazetteer", [])]
            terms = [doc.text[s["start_char"]:s["end_char"]] for s in adr_spans]
            print(f"  [{row['id'][:8]}] {row['text'][:100]}…")
            print(f"           matched ADR terms: {terms[:3]}")
            shown += 1
    if shown == 0:
        print("  (none)")

    print("\n" + "=" * 55)
    print(f"RESULT: aggregated P={agg_p:.3f} R={agg_r:.3f} F1={agg_f:.3f} on {len(gold_rows)} newsletter sentences")
    target_ok = "PASS" if agg_p >= 0.70 else "FAIL"
    print(f"PRECISION TARGET (>= 0.70): {target_ok}")  # noqa
    print("=" * 55)


if __name__ == "__main__":
    main()
