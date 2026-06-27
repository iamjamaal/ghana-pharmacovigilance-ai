#!/usr/bin/env python3
"""
scripts/phase5_per_source_threshold.py
========================================
Per-source CLF threshold sweep using Phase 2b checkpoints.

Phase 4 used a single global threshold (t=0.55). This script finds the
optimal threshold independently for each fold by sweeping on the val split,
then measures the actual gain on the LOSO test fold.

Two-stage approach:
  Stage 1 — val sweep: for each fold, sweep t=[0.25..0.80] on cls_val.jsonl
             to select the per-fold optimal threshold.
  Stage 2 — test eval: run inference on the held-out test fold at both the
             global t=0.55 and the per-fold optimal t; compare F1.

Checkpoints used: Phase 2b clf_best (models_phase2b_*/clf_best/)
Val files:        models_v1aug_*/cls_val.jsonl (same CLF training data)
Test rows:        gold reviewed rows filtered by source_type (not augmented)

Usage (run from ghana-adr-pipeline/):
    python scripts/phase5_per_source_threshold.py
"""

import sys, json, importlib.util, logging
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
DEMO_DIR = ROOT / "ghana-adr-demo"
REPORTS  = ROOT / "reports" / "loso"

# ── Checkpoints ───────────────────────────────────────────────────────────────
PHASE2B_CLF_DIRS = {
    "case_report":           REPORTS / "models_phase2b_case_report",
    "cohort_study":          REPORTS / "models_phase2b_cohort_study",
    "fda_newsletter":        REPORTS / "models_phase2b_fda_newsletter",
    "qualitative_interview": REPORTS / "models_phase2b_qualitative_interview",
}

V1AUG_FOLD_DIRS = {
    "case_report":           REPORTS / "models_v1aug_case_report",
    "cohort_study":          REPORTS / "models_v1aug_cohort_study",
    "fda_newsletter":        REPORTS / "models_v1aug_fda_newsletter",
    "qualitative_interview": REPORTS / "models_v1aug_qualitative_interview",
}

GOLD_CLF = ROOT / "output" / "gold" / "ghana_adr_gold_classification.jsonl"
AUG_CLF  = ROOT / "output" / "gold" / "ghana_adr_gold_classification_augmented.jsonl"

GLOBAL_T = 0.55
CLF_MAXLEN = 128
THRESHOLDS = [round(t * 0.05, 2) for t in range(5, 17)]  # 0.25 .. 0.80 step 0.05

FOLD_SOURCES = [
    "case_report",
    "cohort_study",
    "fda_newsletter",
    "qualitative_interview",
]

# ── Load override patterns from inference engine ──────────────────────────────
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

log.info("Loading inference_engine override patterns...")
_ie = _load_module("ie", DEMO_DIR / "inference_engine.py")
_has_negation    = _ie.has_negation
_INTOLERANCE     = _ie.INTOLERANCE_PATTERNS
_DRUG_TERMS_PAT  = _ie._DRUG_TERMS_PATTERN
_ADR_TERMS_PAT   = _ie._ADR_TERMS_PATTERN


def apply_overrides(text: str, raw_pred: int) -> int:
    """Apply Phase 4 override logic (entity-gated intolerance + negation)."""
    if _has_negation(text):
        return 0
    has_entity = bool(_DRUG_TERMS_PAT.search(text) or _ADR_TERMS_PAT.search(text))
    if has_entity and any(p.search(text) for p in _INTOLERANCE):
        return 1
    return raw_pred


def load_val_rows(fold_dir: Path) -> list:
    path = fold_dir / "cls_val.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Val file missing: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_test_rows(source: str) -> list:
    """Return reviewed gold CLF rows for the given source (excludes augmented)."""
    rows = []
    with open(GOLD_CLF, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("source_type") == source
                    and r.get("manually_reviewed")
                    and not r.get("augmented")):
                rows.append(r)
    return rows


def run_inference(model, tokenizer, rows: list, device) -> list:
    """Return list of prob_adr floats for each row."""
    probs = []
    model.eval()
    with torch.no_grad():
        for r in rows:
            enc = tokenizer(
                r["text"], truncation=True, max_length=CLF_MAXLEN,
                padding="max_length", return_tensors="pt"
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device)
            ).logits
            p = torch.softmax(logits, dim=1).squeeze()
            probs.append(float(p[1].item()))
    return probs


def metrics_at_threshold(rows: list, probs: list, t: float) -> dict:
    labels = [int(r["label"]) for r in rows]
    preds  = [apply_overrides(r["text"], int(prob >= t)) for r, prob in zip(rows, probs)]
    return {
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "p":  round(float(precision_score(labels, preds, zero_division=0)), 4),
        "r":  round(float(recall_score(labels, preds, zero_division=0)), 4),
        "n":  len(labels),
        "pos": int(sum(labels)),
    }


def main():
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )

    val_results  = {}   # val_results[fold][t]  = {f1, p, r}
    test_results = {}   # test_results[fold][t] = {f1, p, r}
    per_fold_best_t = {}

    for fold in FOLD_SOURCES:
        ckpt_dir = PHASE2B_CLF_DIRS[fold] / "clf_best"
        val_dir  = V1AUG_FOLD_DIRS[fold]

        if not ckpt_dir.exists():
            log.warning(f"[SKIP] {fold}: checkpoint missing at {ckpt_dir}")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"FOLD: {fold}")
        log.info(f"{'='*60}")

        val_rows  = load_val_rows(val_dir)
        test_rows = load_test_rows(fold)

        log.info(f"  Val rows: {len(val_rows)} (pos={sum(r['label'] for r in val_rows)})")
        log.info(f"  Test rows: {len(test_rows)} (pos={sum(int(r['label']) for r in test_rows)})")

        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)

        log.info("  Running val inference...")
        val_probs  = run_inference(model, tokenizer, val_rows, device)
        log.info("  Running test inference...")
        test_probs = run_inference(model, tokenizer, test_rows, device)

        del model  # free memory between folds

        # ── Stage 1: sweep on val ─────────────────────────────────────────────
        val_results[fold]  = {}
        test_results[fold] = {}

        log.info(f"\n  {'t':>5}  {'val F1':>8}  {'val P':>8}  {'val R':>8}  "
                 f"{'test F1':>8}  {'test P':>8}  {'test R':>8}")
        log.info(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

        for t in THRESHOLDS:
            vm = metrics_at_threshold(val_rows, val_probs, t)
            tm = metrics_at_threshold(test_rows, test_probs, t)
            val_results[fold][t]  = vm
            test_results[fold][t] = tm
            marker = " <-- global" if t == GLOBAL_T else ""
            log.info(
                f"  {t:>5.2f}  {vm['f1']:>8.4f}  {vm['p']:>8.4f}  {vm['r']:>8.4f}  "
                f"{tm['f1']:>8.4f}  {tm['p']:>8.4f}  {tm['r']:>8.4f}{marker}"
            )

        # Stage 1 selection: best val F1 threshold
        best_t = max(THRESHOLDS, key=lambda t: val_results[fold][t]["f1"])
        per_fold_best_t[fold] = best_t
        log.info(f"\n  >> Val-selected threshold: t={best_t}  "
                 f"val F1={val_results[fold][best_t]['f1']:.4f}  "
                 f"test F1={test_results[fold][best_t]['f1']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-SOURCE THRESHOLD SWEEP — RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Fold':30s}  {'Val-best t':>10}  {'Test F1 @ t=0.55':>17}  "
          f"{'Test F1 @ best t':>17}  {'Delta':>7}")
    print("-" * 90)

    global_test_f1s  = []
    persrc_test_f1s  = []

    folds_done = [f for f in FOLD_SOURCES if f in per_fold_best_t]
    for fold in folds_done:
        best_t  = per_fold_best_t[fold]
        global_m = test_results[fold].get(GLOBAL_T, {})
        best_m   = test_results[fold].get(best_t, {})
        delta = best_m.get('f1', 0) - global_m.get('f1', 0)
        sign  = "+" if delta >= 0 else ""
        print(f"{fold:30s}  {best_t:>10.2f}  {global_m.get('f1',0):>17.4f}  "
              f"{best_m.get('f1',0):>17.4f}  {sign}{delta:>.4f}")
        global_test_f1s.append(global_m.get("f1", 0))
        persrc_test_f1s.append(best_m.get("f1", 0))

    print("-" * 90)
    macro_global = sum(global_test_f1s) / len(global_test_f1s) if global_test_f1s else 0
    macro_persrc = sum(persrc_test_f1s) / len(persrc_test_f1s) if persrc_test_f1s else 0
    delta_macro  = macro_persrc - macro_global
    sign = "+" if delta_macro >= 0 else ""
    print(f"{'MACRO':30s}  {'':>10}  {macro_global:>17.4f}  {macro_persrc:>17.4f}  "
          f"{sign}{delta_macro:.4f}")

    print(f"\nPer-fold thresholds selected: {per_fold_best_t}")
    if delta_macro > 0:
        print(f"\nGAIN: +{delta_macro*100:.2f}pp macro CLF F1 with per-source thresholds")
    else:
        print(f"\nNo macro gain from per-source thresholds (global t=0.55 remains best)")

    # ── Detailed test metrics per fold ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED TEST METRICS — GLOBAL vs PER-FOLD THRESHOLD")
    print("=" * 70)
    for fold in folds_done:
        best_t   = per_fold_best_t[fold]
        g = test_results[fold].get(GLOBAL_T, {})
        b = test_results[fold].get(best_t, {})
        print(f"\n{fold} (n={g.get('n','?')}, pos={g.get('pos','?')}):")
        print(f"  t=0.55 (global):      F1={g.get('f1',0):.4f}  P={g.get('p',0):.4f}  R={g.get('r',0):.4f}")
        print(f"  t={best_t:.2f} (per-fold): F1={b.get('f1',0):.4f}  P={b.get('p',0):.4f}  R={b.get('r',0):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "reports" / "phase5_per_source_threshold.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "global_threshold": GLOBAL_T,
            "thresholds_swept": THRESHOLDS,
            "per_fold_best_t": per_fold_best_t,
            "macro_global_t":   round(macro_global, 4),
            "macro_per_fold_t": round(macro_persrc, 4),
            "macro_delta":      round(delta_macro, 4),
            "val_results":  {fold: {str(t): v for t, v in fold_data.items()}
                             for fold, fold_data in val_results.items()},
            "test_results": {fold: {str(t): v for t, v in fold_data.items()}
                             for fold, fold_data in test_results.items()},
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
