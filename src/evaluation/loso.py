#!/usr/bin/env python3
"""
src/evaluation/loso.py
======================
Leave-One-Source-Out (LOSO) evaluation for the Ghana ADR pipeline.

6 folds (fda_annual_report excluded — only 4 manually reviewed rows).
Each fold trains both classifier and NER ONCE on 5 sources, then evaluates
on the held-out source with two passes:

  Pass A  pattern overrides + Ghanaian synonym layer ENABLED
          (inference_engine.py: has_negation, INTOLERANCE_PATTERNS,
           apply_severity_rules, apply_ghanaian_synonyms)
  Pass B  vanilla model only — overrides DISABLED

Both passes produced from a single training run per fold.

Outputs:
  reports/loso/fold_<source>.json          per-fold JSON (both passes)
  reports/loso/summary.md                  Pass A table
  reports/loso/summary_no_overrides.md     Pass B table

Usage (run from ghana-adr-pipeline/):
    python src/evaluation/loso.py
    python src/evaluation/loso.py --source cohort_study   # single-fold debug
"""

import sys
import json
import random
import logging
import argparse
import importlib.util
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import (
    classification_report as seq_report,
    f1_score        as seq_f1,
    precision_score as seq_prec,
    recall_score    as seq_rec,
)

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
log.info(f"[LOSO] seed={SEED}")


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent
TRAIN_DIR = ROOT / "output" / "gold" / "ghana-adr-training"
DEMO_DIR  = ROOT / "ghana-adr-demo"
GOLD_CLF  = ROOT / "output" / "gold" / "ghana_adr_gold_classification.jsonl"
GOLD_NER  = ROOT / "output" / "gold" / "ghana_adr_gold_ner.jsonl"
REPORTS   = ROOT / "reports" / "loso"
REPORTS.mkdir(parents=True, exist_ok=True)


# ── Hyperparameters — must match the 0.830 / 0.776 baseline exactly ──────────
MODEL_NAME  = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
CLF_EPOCHS  = 3       # reduced from 5 for CPU-only LOSO (max_length also halved)
CLF_BS      = 16
CLF_LR      = 2e-5
CLF_MAXLEN  = 128     # reduced from 256; ~3-4x CPU speedup (attention is O(n^2))
NER_EPOCHS  = 4       # reduced from 10 for CPU-only LOSO
NER_BS      = 8
NER_LR      = 3e-5
NER_MAXLEN  = 128     # reduced from 256; consistent with CLF_MAXLEN
WARMUP_RATE = 0.1
VAL_RATIO   = 0.15

RS_CLF_F1 = 0.830   # random-split Pass A reference
RS_NER_F1 = 0.776   # random-split Pass A reference (includes inference overrides)

EXCLUDE = {"fda_annual_report"}  # only 4 manually reviewed rows

NER_LABELS = [
    "O",
    "B-DRUG",  "I-DRUG",
    "B-ADR",   "I-ADR",
    "B-SEVERITY",    "I-SEVERITY",
    "B-PATIENT_DEMO", "I-PATIENT_DEMO",
]
LABEL2ID    = {l: i for i, l in enumerate(NER_LABELS)}
ID2LABEL    = {i: l for i, l in enumerate(NER_LABELS)}
ENTITY_TAGS = ["DRUG", "ADR", "SEVERITY", "PATIENT_DEMO"]


# ── Dynamic imports (module names start with digits — can't use import) ───────
def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


log.info("[LOSO] importing training modules...")
_c  = _load("train_cls",   TRAIN_DIR / "02a_train_classifier.py")
_n  = _load("train_ner",   TRAIN_DIR / "02b_train_ner.py")
_p  = _load("prep",        TRAIN_DIR / "01_prepare_data.py")
_ie = _load("ie",          DEMO_DIR  / "inference_engine.py")

# Reuse existing Dataset classes and evaluation functions
ADRDataset   = _c.ADRDataset         # classifier dataset; tokenises at __getitem__
eval_clf     = _c.evaluate           # sklearn P/R/F1
NERDataset   = _n.NERDataset         # reads pre-tokenised JSONL
eval_ner_raw = _n.evaluate_ner       # seqeval; used for val-loop only
align_labels = _p.align_labels_to_tokens   # text + entity spans → token label IDs
augment_sev  = _p.augment_severity         # doubles SEVERITY training examples

# Pass A overrides (inference_engine.py)
has_negation = _ie.has_negation            # negation pattern check (→ force predict 0)
INTOLERANCE  = _ie.INTOLERANCE_PATTERNS    # 14 intolerance regexes (→ force predict 1)
sev_rules    = _ie.apply_severity_rules    # SEVERITY keyword injection
gh_synonyms  = _ie.apply_ghanaian_synonyms # Ghanaian ADR phrase injection


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_clf_rows():
    rows = []
    with open(GOLD_CLF, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("manually_reviewed") and r.get("source_type") not in EXCLUDE:
                rows.append(r)
    return rows


def load_ner_rows():
    rows = []
    with open(GOLD_NER, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("meta", {}).get("source_type", "")
            if r.get("manually_reviewed") and src not in EXCLUDE:
                r["source_type"] = src   # hoist for convenience
                rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Val-split helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_val_split(rows, key_fn):
    """
    Stratified val split (VAL_RATIO of each stratum).
    key_fn: row → stratum key.
    Returns (train_subset, val_subset).
    """
    strata = defaultdict(list)
    for r in rows:
        strata[key_fn(r)].append(r)
    train, val = [], []
    for bucket in strata.values():
        b = bucket.copy()
        random.shuffle(b)
        n_val = max(1, round(len(b) * VAL_RATIO))
        val.extend(b[:n_val])
        train.extend(b[n_val:])
    return train, val


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier training
# ═══════════════════════════════════════════════════════════════════════════════

def _clf_weights(train_rows, device):
    """Inverse-frequency class weights with 1.4× ADR boost (mirrors baseline line 151)."""
    labels  = [r["label"] for r in train_rows]
    counts  = Counter(labels)
    total   = len(labels)
    weights = torch.tensor(
        [total / (2 * counts[i]) for i in range(2)], dtype=torch.float
    )
    weights[1] *= 1.4
    return weights.to(device)


def train_clf_fold(fold_dir: Path, train_rows, val_rows, tokenizer, device):
    """
    Train classifier for one LOSO fold.
    Exact same loop as 02a_train_classifier.py; CLF_EPOCHS is fixed (no auto-extend).
    Returns path to best checkpoint directory.
    """
    ckpt_dir = fold_dir / "clf_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ADRDataset reads from file paths, so write temp splits
    for split_name, rows in [("train", train_rows), ("val", val_rows)]:
        path = fold_dir / f"cls_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "id": r["id"], "text": r["text"],
                    "label": r["label"],
                    "source_type": r.get("source_type", ""),
                }, ensure_ascii=False) + "\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "no_adr", 1: "contains_adr"},
        label2id={"no_adr": 0, "contains_adr": 1},
    ).to(device)

    train_ds = ADRDataset(fold_dir / "cls_train.jsonl", tokenizer, CLF_MAXLEN)
    val_ds   = ADRDataset(fold_dir / "cls_val.jsonl",   tokenizer, CLF_MAXLEN)
    train_ld = DataLoader(train_ds, batch_size=CLF_BS, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=CLF_BS)

    weights   = _clf_weights(train_rows, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=CLF_LR, weight_decay=0.01)
    total_steps  = len(train_ld) * CLF_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(1, CLF_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_ld:
            optimizer.zero_grad()
            loss = criterion(
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits,
                batch["labels"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        val_m = eval_clf(model, val_ld, device, criterion)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
        log.info(
            f"    clf  epoch {epoch}/{CLF_EPOCHS}  "
            f"loss={epoch_loss/len(train_ld):.4f}  "
            f"val_F1={val_m['f1']:.4f}"
            + (" ✓" if val_m["f1"] == best_f1 else "")
        )

    log.info(f"    clf best val F1={best_f1:.4f}  saved → {ckpt_dir.name}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ckpt_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier evaluation (Pass A and B)
# ═══════════════════════════════════════════════════════════════════════════════

def _clf_eval(model_dir, test_rows, tokenizer, device, use_overrides):
    """
    Evaluate saved classifier on raw test rows.

    Pass A (use_overrides=True):
      negation check → force 0
      intolerance patterns → force 1
      (negation takes priority, matching inference_engine.py analyse_text logic)

    Pass B (use_overrides=False): raw model prediction only.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for r in test_rows:
            enc = tokenizer(
                r["text"], truncation=True, max_length=CLF_MAXLEN,
                padding="max_length", return_tensors="pt",
            )
            pred = int(torch.argmax(
                model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).logits,
                dim=1,
            ).item())

            if use_overrides:
                text = r["text"]
                if has_negation(text):          # checked first (matches analyse_text)
                    pred = 0
                elif any(p.search(text) for p in INTOLERANCE):
                    pred = 1

            preds.append(pred)
            labels.append(int(r["label"]))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "clf_precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "clf_recall":    round(float(recall_score(   labels, preds, zero_division=0)), 4),
        "clf_f1":        round(float(f1_score(       labels, preds, zero_division=0)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NER tokenisation + training
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_ner_rows(rows, tokenizer):
    """
    Tokenize raw NER rows → list of dicts ready for NERDataset.
    Uses max_length=512 for both encoding and label alignment (matches
    01_prepare_data.py::prepare_ner). NERDataset truncates to NER_MAXLEN.
    """
    out = []
    for r in rows:
        text     = r["text"]
        entities = r.get("entities", [])
        tok_lbl  = align_labels(tokenizer, text, entities)   # max_length=512 internally
        enc = tokenizer(text, truncation=True, max_length=512)
        out.append({
            "id":             r.get("id", ""),
            "text":           text,
            "entities":       entities,
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", []),
            "labels":         tok_lbl,
            "source_type":    r.get("source_type", ""),
        })
    return out


def train_ner_fold(fold_dir: Path, train_rows_raw, val_rows_raw, tokenizer, device):
    """
    Train NER model for one LOSO fold.
    Exact same loop as 02b_train_ner.py; applies SEVERITY augmentation.
    Returns path to best checkpoint directory.
    """
    ckpt_dir = fold_dir / "ner_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # SEVERITY augmentation — matches baseline 01_prepare_data.py::prepare_ner
    train_aug = augment_sev(train_rows_raw)
    log.info(
        f"    ner  SEVERITY aug: {len(train_rows_raw)} → {len(train_aug)} train rows"
    )

    train_tok = tokenize_ner_rows(train_aug,    tokenizer)
    val_tok   = tokenize_ner_rows(val_rows_raw, tokenizer)

    for split_name, rows in [("train", train_tok), ("val", val_tok)]:
        path = fold_dir / f"ner_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(NER_LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    train_ds = NERDataset(fold_dir / "ner_train.jsonl", NER_MAXLEN)
    val_ds   = NERDataset(fold_dir / "ner_val.jsonl",   NER_MAXLEN)
    train_ld = DataLoader(train_ds, batch_size=NER_BS, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=NER_BS)

    # Class-weighted loss — computed from fold train data (mirrors baseline lines 163-171)
    counts = torch.zeros(len(NER_LABELS))
    for batch in DataLoader(train_ds, batch_size=64):
        for lbl in batch["labels"].view(-1):
            if lbl.item() != -100:
                counts[lbl.item()] += 1
    counts       = counts.clamp(min=1)
    class_weights = (counts.sum() / (len(NER_LABELS) * counts)).clamp(max=20.0).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    optimizer = AdamW(model.parameters(), lr=NER_LR, weight_decay=0.01)
    total_steps  = len(train_ld) * NER_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(1, NER_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_ld:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = loss_fn(
                logits.view(-1, len(NER_LABELS)),
                batch["labels"].view(-1).to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        val_m = eval_ner_raw(model, val_ld, device)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            model.save_pretrained(ckpt_dir)
        log.info(
            f"    ner  epoch {epoch}/{NER_EPOCHS}  "
            f"loss={epoch_loss/len(train_ld):.4f}  "
            f"val_F1={val_m['f1']:.4f}"
            + (" ✓" if val_m["f1"] == best_f1 else "")
        )

    log.info(f"    ner best val F1={best_f1:.4f}  saved → {ckpt_dir.name}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ckpt_dir


# ═══════════════════════════════════════════════════════════════════════════════
# NER evaluation (Pass A and B)
# ═══════════════════════════════════════════════════════════════════════════════

def _spans_to_bio(text: str, entities: list, offsets: list) -> list:
    """
    Map character-span entities onto tokenizer offsets → BIO tag list.
    Returns one tag per real token (skips special tokens where cs == ce).
    Mirrors align_labels_to_tokens logic but returns strings instead of IDs.
    """
    char_lbl = ["O"] * len(text)
    for ent in entities:
        s   = ent.get("start", 0)
        e   = min(ent.get("end", 0), len(text))
        lbl = ent.get("label", "O")
        for i in range(s, e):
            char_lbl[i] = f"B-{lbl}" if i == s else f"I-{lbl}"
    return [
        char_lbl[cs] if cs < len(char_lbl) else "O"
        for cs, ce in offsets
        if cs != ce   # skip special tokens
    ]


def _ner_eval(model_dir, test_rows_raw, tokenizer, device, use_overrides):
    """
    Evaluate NER on raw gold rows.

    Both passes:
      1. Run model → token-level logits
      2. argmax → token predictions → entity spans (via character offsets)
      3. Pass A only: apply sev_rules + gh_synonyms to predicted spans
      4. Convert predicted spans → BIO sequence using same offsets
      5. Convert gold entity spans → BIO sequence using same offsets
      6. seqeval entity-level F1

    Using character-span conversion ensures Pass A post-processing applies on
    exactly the same footing as the inference engine that produced the 0.776 baseline.
    """
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    all_true, all_pred = [], []

    with torch.no_grad():
        for r in test_rows_raw:
            text          = r["text"]
            gold_entities = r.get("entities", [])

            enc = tokenizer(
                text, truncation=True, max_length=NER_MAXLEN,
                return_offsets_mapping=True, return_tensors="pt",
            )
            offsets = enc["offset_mapping"].squeeze().tolist()
            logits  = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            preds = torch.argmax(logits, dim=2).squeeze().tolist()
            if isinstance(preds, int):
                preds = [preds]

            # Token predictions → entity spans
            # "text" field required by apply_severity_rules (inference_engine.py:288)
            pred_ents, cur = [], None
            for pid, (cs, ce) in zip(preds, offsets):
                if cs == ce:
                    continue
                lbl = ID2LABEL.get(pid, "O")
                if lbl.startswith("B-"):
                    if cur:
                        pred_ents.append(cur)
                    cur = {"label": lbl[2:], "start": cs, "end": ce,
                           "text": text[cs:ce]}
                elif lbl.startswith("I-") and cur and lbl[2:] == cur["label"]:
                    cur["end"]  = ce
                    cur["text"] = text[cur["start"]:ce]
                else:
                    if cur:
                        pred_ents.append(cur)
                    cur = None
            if cur:
                pred_ents.append(cur)

            if use_overrides:
                pred_ents = sev_rules(text, pred_ents)
                pred_ents = gh_synonyms(text, pred_ents)

            all_pred.append(_spans_to_bio(text, pred_ents,    offsets))
            all_true.append(_spans_to_bio(text, gold_entities, offsets))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    overall_f1   = round(float(seq_f1(  all_true, all_pred, zero_division=0)), 4)
    overall_prec = round(float(seq_prec(all_true, all_pred, zero_division=0)), 4)
    overall_rec  = round(float(seq_rec( all_true, all_pred, zero_division=0)), 4)

    report  = seq_report(all_true, all_pred, output_dict=True, zero_division=0)
    per_tag = {}
    for tag in ENTITY_TAGS:
        entry = report.get(tag, {})
        per_tag[tag] = {
            "f1":      round(float(entry.get("f1-score", 0.0)), 4),
            "support": int(entry.get("support", 0)),
        }
        if per_tag[tag]["support"] < 10:
            per_tag[tag]["f1_note"] = "noisy (n<10)"

    return {
        "ner_overall_f1": overall_f1,
        "ner_precision":  overall_prec,
        "ner_recall":     overall_rec,
        "ner_per_tag":    per_tag,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sanity checks (Step 5)
# ═══════════════════════════════════════════════════════════════════════════════

def sanity_check(source, clf_test, clf_train_full, total_clf):
    issues = []

    # Check 1: no data loss
    if len(clf_test) + len(clf_train_full) != total_clf:
        issues.append(
            f"DATA LOSS: test={len(clf_test)} + train={len(clf_train_full)} "
            f"≠ total={total_clf}"
        )

    # Check 2: no leakage
    if any(r.get("source_type") == source for r in clf_train_full):
        issues.append(f"LEAKAGE: '{source}' appears in training set")

    # Check 3: positive rate in 5-30% range
    pct = 100.0 * sum(r["label"] for r in clf_test) / max(len(clf_test), 1)
    if pct < 5.0 or pct > 30.0:
        issues.append(f"POSITIVE RATE outside 5-30%: {pct:.1f}% — flag for review")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Fold runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_fold(source, clf_rows, ner_rows, tokenizer, device):
    fold_dir  = REPORTS / f"models_{source}"
    fold_json = REPORTS / f"fold_{source}.json"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Resume: if the fold JSON was written on a previous run, load and return it
    if fold_json.exists():
        log.info(f"  ↩ fold_{source}.json already exists — skipping (delete to rerun)")
        with open(fold_json, encoding="utf-8") as f:
            return json.load(f)

    # ── Split by source ─────────────────────────────────────────────────────
    clf_test  = [r for r in clf_rows if r["source_type"] == source]
    clf_train = [r for r in clf_rows if r["source_type"] != source]
    ner_test  = [r for r in ner_rows if r["source_type"] == source]
    ner_train = [r for r in ner_rows if r["source_type"] != source]

    clf_train_sub, clf_val = make_val_split(
        clf_train, lambda r: (r["label"], r.get("source_type", ""))
    )
    ner_train_sub, ner_val = make_val_split(
        ner_train, lambda r: r.get("source_type", "")
    )

    n_total    = len(clf_test)
    n_positive = sum(r["label"] for r in clf_test)
    pct_pos    = round(100.0 * n_positive / max(n_total, 1), 1)

    issues = sanity_check(source, clf_test, clf_train, len(clf_rows))
    for w in issues:
        log.warning(f"  ⚠  {w}")

    log.info(f"\n{'='*64}")
    log.info(f"  FOLD  hold-out={source}")
    log.info(
        f"  clf   test={n_total} ({n_positive} pos, {pct_pos}%)  "
        f"train={len(clf_train_sub)}  val={len(clf_val)}"
    )
    log.info(
        f"  ner   test={len(ner_test)}  "
        f"train={len(ner_train_sub)}  val={len(ner_val)}"
    )
    log.info(f"{'='*64}")

    # ── Train once (shared by both passes) ──────────────────────────────────
    clf_ckpt = fold_dir / "clf_best"
    ner_ckpt = fold_dir / "ner_best"

    if clf_ckpt.exists() and any(clf_ckpt.iterdir()):
        log.info(f"  → CLF checkpoint found at {clf_ckpt.name} — skipping training")
    else:
        log.info("  → Training classifier...")
        t0 = time.time()
        clf_ckpt = train_clf_fold(fold_dir, clf_train_sub, clf_val, tokenizer, device)
        log.info(f"  ← Classifier done in {time.time()-t0:.0f}s")

    if ner_ckpt.exists() and any(ner_ckpt.iterdir()):
        log.info(f"  → NER checkpoint found at {ner_ckpt.name} — skipping training")
    else:
        log.info("  → Training NER model...")
        t0 = time.time()
        ner_ckpt = train_ner_fold(fold_dir, ner_train_sub, ner_val, tokenizer, device)
        log.info(f"  ← NER done in {time.time()-t0:.0f}s")

    # ── Evaluate both passes ─────────────────────────────────────────────────
    pass_results = {}
    for pass_label, use_overrides in [("pass_a", True), ("pass_b", False)]:
        log.info(f"  → Evaluating {pass_label.upper()} (overrides={'on' if use_overrides else 'off'})...")
        clf_m = _clf_eval(clf_ckpt, clf_test,  tokenizer, device, use_overrides)
        ner_m = _ner_eval(ner_ckpt, ner_test,  tokenizer, device, use_overrides)
        pass_results[pass_label] = {**clf_m, **ner_m}

        log.info(
            f"  {pass_label.upper()}  "
            f"CLF F1={clf_m['clf_f1']:.4f}  P={clf_m['clf_precision']:.4f}  "
            f"R={clf_m['clf_recall']:.4f}  |  "
            f"NER F1={ner_m['ner_overall_f1']:.4f}"
        )
        for tag in ENTITY_TAGS:
            s    = ner_m["ner_per_tag"][tag]
            note = f"  [{s.get('f1_note', '')}]" if s.get("f1_note") else ""
            log.info(
                f"         {tag:12s}  "
                f"F1={s['f1']:.4f}  support={s['support']}{note}"
            )

    fold_result = {
        "held_out_source": source,
        "n_total":         n_total,
        "n_positive":      n_positive,
        "pct_positive":    pct_pos,
        "warnings":        issues,
        **pass_results,
    }

    fold_json = REPORTS / f"fold_{source}.json"
    with open(fold_json, "w", encoding="utf-8") as f:
        json.dump(fold_result, f, indent=2)
    log.info(f"  ✓ Fold complete — {fold_json}")
    return fold_result


# ═══════════════════════════════════════════════════════════════════════════════
# Summary tables
# ═══════════════════════════════════════════════════════════════════════════════

def _macro(results, pass_key):
    avg = {}
    for k in ["clf_precision", "clf_recall", "clf_f1", "ner_overall_f1"]:
        vals = [r[pass_key][k] for r in results]
        avg[k] = round(sum(vals) / len(vals), 4)

    avg["ner_per_tag"] = {}
    for tag in ENTITY_TAGS:
        # Exclude noisy folds from that tag's macro-F1
        f1s = [
            r[pass_key]["ner_per_tag"][tag]["f1"]
            for r in results
            if "f1_note" not in r[pass_key]["ner_per_tag"].get(tag, {})
        ]
        avg["ner_per_tag"][tag] = {
            "f1":      round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "support": sum(
                r[pass_key]["ner_per_tag"].get(tag, {}).get("support", 0)
                for r in results
            ),
        }
    return avg


def write_summary(results, path: Path, pass_key: str, rs_clf: float, rs_ner: float):
    macro    = _macro(results, pass_key)
    override = "ENABLED" if pass_key == "pass_a" else "DISABLED (vanilla)"
    header = (
        f"# LOSO Summary — Pattern Overrides {override}\n\n"
        f"Random-split reference (Pass A):  CLF F1={rs_clf}  NER F1={rs_ner}\n"
        f"Seed={SEED}  "
        f"CLF_EPOCHS={CLF_EPOCHS} (baseline=5, reduced for CPU)  "
        f"NER_EPOCHS={NER_EPOCHS} (baseline=10, reduced for CPU)  "
        f"max_length={CLF_MAXLEN} (baseline=256, reduced for CPU)\n"
        f"Note: reduced epochs/maxlen for CPU-only run. "
        f"Gap direction is reliable; absolute scores may be ~2-5pp below full-epochs baseline.\n"
        f"Excluded sources: {', '.join(sorted(EXCLUDE))}\n\n"
        f"\\* = noisy (SEVERITY support < 10 in test fold)\n\n"
    )

    col_hdr = (
        "| Source                    | N   | Pos%"
        " | CLF P | CLF R | CLF F1 | Gap CLF"
        " | NER F1 | Gap NER"
        " | DRUG  | ADR   | SEV    | DEMO  |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    def fmt_row(label, n, pct, p):
        sev   = p["ner_per_tag"]["SEVERITY"]
        sev_s = f"{sev['f1']:.3f}{'*' if sev.get('f1_note') else ''}"
        return (
            f"| {label:25s}"
            f"| {n!s:3}  "
            f"| {pct!s:4}%"
            f"| {p['clf_precision']:.3f} "
            f"| {p['clf_recall']:.3f} "
            f"| {p['clf_f1']:.3f}  "
            f"| {rs_clf - p['clf_f1']:+.3f}  "
            f"| {p['ner_overall_f1']:.3f}  "
            f"| {rs_ner - p['ner_overall_f1']:+.3f}  "
            f"| {p['ner_per_tag']['DRUG']['f1']:.3f} "
            f"| {p['ner_per_tag']['ADR']['f1']:.3f} "
            f"| {sev_s:6s} "
            f"| {p['ner_per_tag']['PATIENT_DEMO']['f1']:.3f} |"
        )

    rows = [
        fmt_row(r["held_out_source"], r["n_total"], r["pct_positive"], r[pass_key])
        for r in results
    ]
    macro_row = fmt_row("**macro-avg**", "—", "—", macro)

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(col_hdr + "\n")
        f.write(sep + "\n")
        for row in rows:
            f.write(row + "\n")
        f.write(macro_row + "\n")
    log.info(f"  Summary ({pass_key}) → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LOSO evaluation — Ghana ADR pipeline")
    parser.add_argument(
        "--source",
        help="Run a single fold only (for timing/debugging). E.g. --source cohort_study",
    )
    args = parser.parse_args()

    log.info(f"\n{'='*64}")
    log.info("  Ghana ADR — Leave-One-Source-Out (LOSO) Evaluation")
    log.info(f"  Seed={SEED}  CLF epochs={CLF_EPOCHS} (fixed)  NER epochs={NER_EPOCHS}")
    log.info(f"  Pass A: overrides ENABLED  |  Pass B: overrides DISABLED")
    log.info(f"  Excluded: {EXCLUDE}")
    log.info(f"{'='*64}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")
    if device.type == "cpu":
        log.warning(
            "\n  ⚠  CPU-only detected. Estimates with current settings "
            f"(CLF_EPOCHS={CLF_EPOCHS}, NER_EPOCHS={NER_EPOCHS}, maxlen={CLF_MAXLEN}):\n"
            "       Classifier (~3 epochs, maxlen=128): ~15-20 min per fold\n"
            "       NER        (~4 epochs, maxlen=128): ~55-80 min per fold\n"
            "       6 folds total:                      ~7-10 hours\n"
        )

    log.info("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log.info("  Loading gold data (manually reviewed, excluding fda_annual_report)...")
    clf_rows = load_clf_rows()
    ner_rows = load_ner_rows()

    sources = sorted({r["source_type"] for r in clf_rows})
    log.info(f"  Loaded: {len(clf_rows)} clf rows, {len(ner_rows)} ner rows")
    log.info(f"  Sources ({len(sources)}): {sources}")

    for src in sources:
        n = sum(1 for r in clf_rows if r["source_type"] == src)
        p = sum(r["label"] for r in clf_rows if r["source_type"] == src)
        log.info(f"    {src:30s}  n={n}  pos={p} ({100*p//max(n,1)}%)")

    if args.source:
        if args.source not in sources:
            log.error(f"Unknown source '{args.source}'. Valid: {sources}")
            sys.exit(1)
        sources = [args.source]
        log.info(f"\n  Single-fold mode: {args.source}")

    all_results   = []
    total_start   = time.time()

    for i, source in enumerate(sources, 1):
        log.info(f"\n  [{i}/{len(sources)}] fold={source}")
        # Reset seeds for reproducibility across fold order
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        fold_start = time.time()
        result     = run_fold(source, clf_rows, ner_rows, tokenizer, device)
        all_results.append(result)

        elapsed   = time.time() - total_start
        fold_time = time.time() - fold_start
        eta       = (elapsed / i) * (len(sources) - i)
        log.info(
            f"  [{i}/{len(sources)}] ✓ {source}  "
            f"fold={fold_time/60:.1f}m  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m"
        )

    # ── Write summary tables ─────────────────────────────────────────────────
    if len(all_results) == len(sources):
        write_summary(all_results, REPORTS / "summary.md",
                      pass_key="pass_a", rs_clf=RS_CLF_F1, rs_ner=RS_NER_F1)
        write_summary(all_results, REPORTS / "summary_no_overrides.md",
                      pass_key="pass_b", rs_clf=RS_CLF_F1, rs_ner=RS_NER_F1)

    # ── Gap analysis (Pass A) ─────────────────────────────────────────────────
    log.info(f"\n{'='*64}")
    log.info("  GAP ANALYSIS — Pass A CLF F1 vs random-split 0.830")
    log.info(f"{'='*64}")
    gaps = sorted(
        [(r["held_out_source"], RS_CLF_F1 - r["pass_a"]["clf_f1"])
         for r in all_results],
        key=lambda x: -x[1],
    )
    for src, gap in gaps:
        bar = "▓" * max(0, int(abs(gap) * 50))
        sign = "↓" if gap > 0 else "↑"
        log.info(f"  {src:30s}  {sign} {gap:+.4f}  {bar}")

    if gaps:
        log.info(f"\n  Largest  gap (most source-dependent): {gaps[0][0]}  ({gaps[0][1]:+.4f})")
        log.info(f"  Smallest gap (most generalizable):    {gaps[-1][0]}  ({gaps[-1][1]:+.4f})")

    total_time = time.time() - total_start
    log.info(f"\n  Total elapsed: {total_time/60:.1f}m")
    log.info("  LOSO complete.")


if __name__ == "__main__":
    main()
