#!/usr/bin/env python3
"""
scripts/phase7_hybrid_eval.py
================================
Hybrid: Phase 6 CLF + Phase 7 NER.

CLF: Phase 6 standalone (models_phase2b_*/clf_best), Phase 6 thresholds.
NER: Phase 7 standalone (models_phase7_*/ner_best).

Hypothesis: Phase 6 CLF is the precision-recall ceiling (macro 0.713);
Phase 7 NER is the new best (macro 0.655). Combining them should give the
best of both without retraining.

Usage (from ghana-adr-pipeline/):
    python scripts/phase7_hybrid_eval.py
"""

import sys, json, importlib.util
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import (
    classification_report as seq_report,
    f1_score        as seq_f1,
    precision_score as seq_prec,
    recall_score    as seq_rec,
)

ROOT     = Path(__file__).parent.parent
DEMO_DIR = ROOT / "ghana-adr-demo"
GOLD_CLF = ROOT / "output" / "gold" / "ghana_adr_gold_classification.jsonl"
GOLD_NER = ROOT / "output" / "gold" / "ghana_adr_gold_ner.jsonl"
REPORTS  = ROOT / "reports" / "loso"
DAPT_DIR = ROOT / "models" / "pubmedbert-ghana-dapt"

FOLD_SOURCES = ["case_report", "cohort_study", "fda_newsletter", "qualitative_interview"]

CLF_MAXLEN = 128
NER_MAXLEN = 128

# Phase 6 CLF thresholds (from Phase 5 sweep, used in Phase 6 LOSO)
PHASE6_THRESHOLD = {
    "case_report":           0.70,
    "cohort_study":          0.55,
    "fda_newsletter":        0.30,
    "qualitative_interview": 0.60,
}

NER_LABELS = [
    "O",
    "B-DRUG", "I-DRUG",
    "B-ADR",  "I-ADR",
    "B-SEVERITY",     "I-SEVERITY",
    "B-PATIENT_DEMO", "I-PATIENT_DEMO",
]
ID2LABEL    = {i: l for i, l in enumerate(NER_LABELS)}
ENTITY_TAGS = ["DRUG", "ADR", "SEVERITY", "PATIENT_DEMO"]

P6_CLF_DIRS = {s: REPORTS / f"models_phase2b_{s}" for s in FOLD_SOURCES}
P7_NER_DIRS = {s: REPORTS / f"models_phase7_{s}"  for s in FOLD_SOURCES}

PHASE6_STANDALONE = {
    "case_report":           {"clf_f1": 0.787, "ner_f1": 0.638},
    "cohort_study":          {"clf_f1": 0.776, "ner_f1": 0.754},
    "fda_newsletter":        {"clf_f1": 0.667, "ner_f1": 0.564},
    "qualitative_interview": {"clf_f1": 0.610, "ner_f1": 0.647},
}
PHASE7_STANDALONE = {
    "case_report":           {"clf_f1": 0.7273, "ner_f1": 0.598},
    "cohort_study":          {"clf_f1": 0.7308, "ner_f1": 0.785},
    "fda_newsletter":        {"clf_f1": 0.6667, "ner_f1": 0.587},
    "qualitative_interview": {"clf_f1": 0.5373, "ner_f1": 0.650},
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("[hybrid] importing inference engine...")
_ie = _load("ie", DEMO_DIR / "inference_engine.py")
has_negation    = _ie.has_negation
INTOLERANCE     = _ie.INTOLERANCE_PATTERNS
_DRUG_TERMS_PAT = _ie._DRUG_TERMS_PATTERN
_ADR_TERMS_PAT  = _ie._ADR_TERMS_PATTERN
sev_rules       = _ie.apply_severity_rules
gh_synonyms     = _ie.apply_ghanaian_synonyms


def load_gold_clf():
    rows = []
    with open(GOLD_CLF, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("manually_reviewed"):
                rows.append(r)
    return rows


def load_gold_ner():
    rows = []
    with open(GOLD_NER, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("manually_reviewed"):
                r["source_type"] = r.get("meta", {}).get("source_type", "")
                rows.append(r)
    return rows


def _spans_to_bio(text, entities, offsets):
    cl = ["O"] * len(text)
    for e in entities:
        s   = e.get("start", 0)
        end = min(e.get("end", 0), len(text))
        lbl = e.get("label", "O")
        for i in range(s, end):
            cl[i] = f"B-{lbl}" if i == s else f"I-{lbl}"
    return [cl[cs] if cs < len(cl) else "O" for cs, ce in offsets if cs != ce]


def eval_clf_fold(ckpt, test_rows, tokenizer, device, threshold):
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for r in test_rows:
            enc    = tokenizer(r["text"], truncation=True, max_length=CLF_MAXLEN,
                               padding="max_length", return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            prob   = torch.softmax(logits, dim=1).squeeze()[1].item()
            pred   = int(prob >= threshold)
            if has_negation(r["text"]):
                pred = 0
            elif ((_DRUG_TERMS_PAT.search(r["text"]) or _ADR_TERMS_PAT.search(r["text"]))
                  and any(p.search(r["text"]) for p in INTOLERANCE)):
                pred = 1
            preds.append(pred)
            labels.append(int(r["label"]))
    del model
    return {
        "clf_precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "clf_recall":    round(float(recall_score(   labels, preds, zero_division=0)), 4),
        "clf_f1":        round(float(f1_score(       labels, preds, zero_division=0)), 4),
    }


def eval_ner_fold(ckpt, test_rows, tokenizer, device):
    model = AutoModelForTokenClassification.from_pretrained(str(ckpt)).to(device)
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for r in test_rows:
            text = r["text"]
            enc  = tokenizer(text, truncation=True, max_length=NER_MAXLEN,
                             return_offsets_mapping=True, return_tensors="pt")
            offs   = enc["offset_mapping"].squeeze().tolist()
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            pids = torch.argmax(logits, dim=2).squeeze().tolist()
            if isinstance(pids, int): pids = [pids]

            pred_ents, cur = [], None
            for pid, (cs, ce) in zip(pids, offs):
                if cs == ce: continue
                lbl = ID2LABEL.get(pid, "O")
                if lbl.startswith("B-"):
                    if cur: pred_ents.append(cur)
                    cur = {"label": lbl[2:], "start": cs, "end": ce, "text": text[cs:ce]}
                elif lbl.startswith("I-") and cur and lbl[2:] == cur["label"]:
                    cur["end"] = ce; cur["text"] = text[cur["start"]:ce]
                else:
                    if cur: pred_ents.append(cur)
                    cur = None
            if cur: pred_ents.append(cur)

            pred_ents = sev_rules(text, pred_ents)
            pred_ents = gh_synonyms(text, pred_ents)

            all_pred.append(_spans_to_bio(text, pred_ents, offs))
            all_true.append(_spans_to_bio(text, r.get("entities", []), offs))

    del model

    report  = seq_report(all_true, all_pred, output_dict=True, zero_division=0)
    per_tag = {}
    for tag in ENTITY_TAGS:
        e = report.get(tag, {})
        per_tag[tag] = {"f1": round(float(e.get("f1-score", 0.0)), 4),
                        "support": int(e.get("support", 0))}
        if per_tag[tag]["support"] < 10:
            per_tag[tag]["f1_note"] = "noisy (n<10)"
    return {
        "ner_overall_f1": round(float(seq_f1(  all_true, all_pred, zero_division=0)), 4),
        "ner_precision":  round(float(seq_prec(all_true, all_pred, zero_division=0)), 4),
        "ner_recall":     round(float(seq_rec( all_true, all_pred, zero_division=0)), 4),
        "ner_per_tag":    per_tag,
    }


def main():
    device = torch.device("cpu")
    print(f"\n{'='*64}")
    print(f"  Ghana ADR -- Phase 7 Hybrid Eval")
    print(f"  CLF: Phase 6 standalone  thresholds: {PHASE6_THRESHOLD}")
    print(f"  NER: Phase 7 standalone  (new best)")
    print(f"{'='*64}")

    print("  Loading tokenizer from DAPT...")
    tokenizer = AutoTokenizer.from_pretrained(str(DAPT_DIR))

    print("  Loading gold data...")
    gold_clf = load_gold_clf()
    gold_ner = load_gold_ner()
    print(f"  Gold: {len(gold_clf)} clf  {len(gold_ner)} ner")

    results = {}
    for source in FOLD_SOURCES:
        test_clf = [r for r in gold_clf
                    if r.get("source_type") == source and not r.get("augmented")]
        test_ner = [r for r in gold_ner
                    if r.get("source_type") == source and not r.get("augmented")]

        print(f"\n  [{source}]  clf_test={len(test_clf)}  ner_test={len(test_ner)}")

        ckpt6_clf = P6_CLF_DIRS[source] / "clf_best"
        ckpt7_ner = P7_NER_DIRS[source] / "ner_best"

        for label, ckpt in [("P6 CLF", ckpt6_clf), ("P7 NER", ckpt7_ner)]:
            if not ckpt.exists():
                print(f"  ERROR: missing checkpoint [{label}] {ckpt}", file=sys.stderr)
                sys.exit(1)

        t   = PHASE6_THRESHOLD[source]
        p6s = PHASE6_STANDALONE[source]
        p7s = PHASE7_STANDALONE[source]

        print(f"    -> Phase 6 CLF (t={t})...")
        clf_res = eval_clf_fold(ckpt6_clf, test_clf, tokenizer, device, t)
        print(f"       CLF F1={clf_res['clf_f1']:.4f}  P={clf_res['clf_precision']:.4f}  R={clf_res['clf_recall']:.4f}"
              f"  (Ph6={p6s['clf_f1']:.3f}  Ph7={p7s['clf_f1']:.3f})")

        print(f"    -> Phase 7 NER standalone...")
        ner_res = eval_ner_fold(ckpt7_ner, test_ner, tokenizer, device)
        print(f"       NER F1={ner_res['ner_overall_f1']:.4f}"
              f"  (Ph6={p6s['ner_f1']:.3f}  Ph7={p7s['ner_f1']:.3f})")
        for tag in ENTITY_TAGS:
            t_val = ner_res["ner_per_tag"].get(tag, {})
            note  = "  <- noisy" if t_val.get("f1_note") else ""
            print(f"       {tag:<14} F1={t_val['f1']:.4f}  support={t_val['support']}{note}")

        results[source] = {**clf_res, **ner_res}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  RESULTS -- Phase 7 Hybrid (CLF=Ph6 | NER=Ph7)")
    print(f"{'='*72}")
    print(f"  {'Source':<28}  {'CLF':>6} {'ΔvPh6':>6} {'ΔvPh7':>6}  {'NER':>6} {'ΔvPh6':>6} {'ΔvPh7':>6}")
    print("  " + "-" * 72)

    clf_f1s, ner_f1s = [], []
    for source in FOLD_SOURCES:
        r   = results[source]
        p6s = PHASE6_STANDALONE[source]
        p7s = PHASE7_STANDALONE[source]
        clf = r["clf_f1"]
        ner = r["ner_overall_f1"]
        print(f"  {source:<28}  "
              f"{clf:>6.3f} {clf-p6s['clf_f1']:>+6.3f} {clf-p7s['clf_f1']:>+6.3f}  "
              f"{ner:>6.3f} {ner-p6s['ner_f1']:>+6.3f} {ner-p7s['ner_f1']:>+6.3f}")
        clf_f1s.append(clf)
        ner_f1s.append(ner)

    macro_clf = sum(clf_f1s) / len(clf_f1s)
    macro_ner = sum(ner_f1s) / len(ner_f1s)
    p6_macro_clf = sum(PHASE6_STANDALONE[s]["clf_f1"] for s in FOLD_SOURCES) / 4
    p6_macro_ner = sum(PHASE6_STANDALONE[s]["ner_f1"] for s in FOLD_SOURCES) / 4
    p7_macro_clf = sum(PHASE7_STANDALONE[s]["clf_f1"] for s in FOLD_SOURCES) / 4
    p7_macro_ner = sum(PHASE7_STANDALONE[s]["ner_f1"] for s in FOLD_SOURCES) / 4

    print("  " + "-" * 72)
    print(f"  {'macro-avg':<28}  "
          f"{macro_clf:>6.3f} {macro_clf-p6_macro_clf:>+6.3f} {macro_clf-p7_macro_clf:>+6.3f}  "
          f"{macro_ner:>6.3f} {macro_ner-p6_macro_ner:>+6.3f} {macro_ner-p7_macro_ner:>+6.3f}")
    print()
    print(f"  Phase 6 standalone:  macro CLF={p6_macro_clf:.3f}  macro NER={p6_macro_ner:.3f}")
    print(f"  Phase 7 standalone:  macro CLF={p7_macro_clf:.3f}  macro NER={p7_macro_ner:.3f}")
    print(f"  Phase 7 hybrid:      macro CLF={macro_clf:.3f}  macro NER={macro_ner:.3f}")

    out = {
        "config": {
            "clf": "Phase 6 standalone",
            "ner": "Phase 7 standalone",
            "phase6_thresholds": PHASE6_THRESHOLD,
        },
        "results": results,
        "macro_clf": round(macro_clf, 4),
        "macro_ner": round(macro_ner, 4),
        "phase6_standalone": {"macro_clf": round(p6_macro_clf, 4), "macro_ner": round(p6_macro_ner, 4)},
        "phase7_standalone": {"macro_clf": round(p7_macro_clf, 4), "macro_ner": round(p7_macro_ner, 4)},
    }
    out_path = ROOT / "reports" / "phase7_hybrid.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print("  Phase 7 hybrid eval complete.\n")


if __name__ == "__main__":
    main()
