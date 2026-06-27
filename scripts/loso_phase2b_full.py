#!/usr/bin/env python3
"""
scripts/loso_phase2b_full.py
=============================
Full 4-fold LOSO evaluation on Phase 2b model for the competition report.

Phase 2b model definition:
  CLF: gold + aug only (no silver) — same training data as v1+aug for all folds
  NER: gold + aug + silver (287 DailyMed inserts) — adds silver to all folds

Strategy to avoid unnecessary retraining:
  CLF — reuse existing checkpoints (identical training data):
    fda_newsletter         : models_neronly_newsletter/clf_best  (Phase 2b CLF)
    case_report            : models_v1aug_case_report/clf_best   (v1+aug CLF = Phase 2b CLF)
    cohort_study           : models_v1aug_cohort_study/clf_best  (v1+aug CLF = Phase 2b CLF)
    qualitative_interview  : models_v1aug_qualitative_interview/clf_best

  NER — reuse Phase 2 checkpoint for newsletter; train fresh with silver for others:
    fda_newsletter         : models_silver_newsletter/ner_best  (Phase 2 NER)
    case_report, cohort_study, qualitative_interview: train new NER with silver

Output:
  reports/loso/fold_phase2b_<source>.json   per-fold metrics
  reports/loso/summary_phase2b.md           4-fold table + macro-avg + delta vs v1+aug

Usage (run from ghana-adr-pipeline/):
    python scripts/loso_phase2b_full.py
    python scripts/loso_phase2b_full.py --source cohort_study
"""

import sys, json, random, logging, argparse, importlib.util, time, hashlib, shutil
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

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "output" / "gold" / "ghana-adr-training"
DEMO_DIR  = ROOT / "ghana-adr-demo"
GOLD_CLF  = ROOT / "output" / "gold" / "ghana_adr_gold_classification.jsonl"
GOLD_NER  = ROOT / "output" / "gold" / "ghana_adr_gold_ner.jsonl"
AUG_CLF      = ROOT / "output" / "gold" / "ghana_adr_gold_classification_augmented.jsonl"
AUG_NER      = ROOT / "output" / "gold" / "ghana_adr_gold_ner_augmented.jsonl"
DEMO_AUG_NER = ROOT / "output" / "gold" / "ghana_adr_gold_ner_augmented_demo.jsonl"
SILVER            = ROOT / "data" / "silver" / "inserts_v1.jsonl"
NEWSLETTER_SILVER = ROOT / "data" / "silver" / "newsletter_silver_v1.jsonl"
REPORTS   = ROOT / "reports" / "loso"
REPORTS.mkdir(parents=True, exist_ok=True)

# Existing checkpoint sources
CLF_SEED_DIRS = {
    "fda_newsletter":        REPORTS / "models_neronly_newsletter",   # Phase 2b
    "case_report":           REPORTS / "models_v1aug_case_report",    # v1+aug = Phase 2b
    "cohort_study":          REPORTS / "models_v1aug_cohort_study",
    "qualitative_interview": REPORTS / "models_v1aug_qualitative_interview",
}
NER_SEED_DIRS = {
    # fda_newsletter removed: now trains fresh with newsletter silver (Step A)
    # Previously: REPORTS / "models_silver_newsletter"
}
# Warm-start: fine-tune from a FROZEN seed (separate from fold output dir) for N epochs (Step B)
# Seed is created automatically in main() by copying models_phase2b_<source>/ner_best once.
NER_WARMSTART_DIRS = {
    "qualitative_interview": REPORTS / "models_phase4c_interview_ner_seed",
}
NER_WARMSTART_EPOCHS = 2

# ── Hyperparameters — match all prior LOSO runs ───────────────────────────────
MODEL_NAME  = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
CLF_EPOCHS  = 3
CLF_BS      = 16
CLF_LR      = 2e-5
CLF_MAXLEN  = 128
NER_EPOCHS  = 4
NER_BS      = 8
NER_LR      = 3e-5
NER_MAXLEN  = 128
WARMUP_RATE   = 0.1
VAL_RATIO     = 0.15
# Step 4.1: global threshold (kept as fallback)
# Step 5.0: per-source thresholds from val-sweep + domain-informed newsletter prior
# case_report: t=0.70 (val peak, +4.84pp test); cohort_study: t=0.55 (already optimal)
# fda_newsletter: t=0.30 (val flat — domain-informed; lower t recovers recall, P stays >0.90)
# qualitative_interview: t=0.55 (t=0.60 gave -0.18pp; threshold can't fix precision collapse)
CLF_THRESHOLD = 0.55
CLF_THRESHOLD_PER_SOURCE = {
    "case_report":           0.70,
    "cohort_study":          0.55,
    "fda_newsletter":        0.30,
    "qualitative_interview": 0.55,
}

FOLD_SOURCES = [
    "case_report",
    "cohort_study",
    "fda_newsletter",
    "qualitative_interview",
]
SKIP_AS_FOLD = {"fda_annual_report", "fda_guideline", "surveillance_evaluation"}

NER_LABELS = [
    "O",
    "B-DRUG",  "I-DRUG",
    "B-ADR",   "I-ADR",
    "B-SEVERITY",     "I-SEVERITY",
    "B-PATIENT_DEMO", "I-PATIENT_DEMO",
]
LABEL2ID    = {l: i for i, l in enumerate(NER_LABELS)}
ID2LABEL    = {i: l for i, l in enumerate(NER_LABELS)}
ENTITY_TAGS = ["DRUG", "ADR", "SEVERITY", "PATIENT_DEMO"]

# v1+aug baseline scores for delta column
V1AUG_BASELINE = {
    "case_report":           {"clf_f1": 0.746, "ner_overall_f1": 0.607},
    "cohort_study":          {"clf_f1": 0.768, "ner_overall_f1": 0.659},
    "fda_newsletter":        {"clf_f1": 0.605, "ner_overall_f1": 0.494},
    "qualitative_interview": {"clf_f1": 0.500, "ner_overall_f1": 0.588},
}

# ── Dynamic imports ────────────────────────────────────────────────────────────
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

log.info("[Phase 2b LOSO] importing training modules...")
_c  = _load("train_cls", TRAIN_DIR / "02a_train_classifier.py")
_n  = _load("train_ner", TRAIN_DIR / "02b_train_ner.py")
_p  = _load("prep",      TRAIN_DIR / "01_prepare_data.py")
_ie = _load("ie",        DEMO_DIR  / "inference_engine.py")

ADRDataset   = _c.ADRDataset
eval_clf     = _c.evaluate
NERDataset   = _n.NERDataset
eval_ner_raw = _n.evaluate_ner
align_labels = _p.align_labels_to_tokens
augment_sev  = _p.augment_severity
has_negation       = _ie.has_negation
INTOLERANCE        = _ie.INTOLERANCE_PATTERNS
_DRUG_TERMS_PAT    = _ie._DRUG_TERMS_PATTERN
_ADR_TERMS_PAT     = _ie._ADR_TERMS_PATTERN
sev_rules          = _ie.apply_severity_rules
gh_synonyms        = _ie.apply_ghanaian_synonyms


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

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


def load_aug_clf():
    rows = []
    with open(AUG_CLF, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_aug_ner():
    rows = []
    with open(AUG_NER, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            r["source_type"] = r.get("meta", {}).get("source_type",
                                    r.get("source_type", ""))
            rows.append(r)
    return rows


def load_demo_aug_ner():
    """PATIENT_DEMO augmented NER rows (Step 4.2). Returns [] if file absent."""
    if not DEMO_AUG_NER.exists():
        return []
    rows = []
    with open(DEMO_AUG_NER, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            r["source_type"] = r.get("meta", {}).get("source_type",
                                    r.get("source_type", ""))
            rows.append(r)
    return rows


def load_silver_ner():
    """Silver NER rows only (no CLF — Phase 2b fix)."""
    rows = []
    with open(SILVER, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            sid  = "silver_" + __import__("hashlib").md5(
                r["sentence"].encode()).hexdigest()[:10]
            text = r["sentence"]
            entities = []
            for span in r.get("adr_spans", []):
                entities.append({"label": "ADR",
                                  "start": span["start_char"], "end": span["end_char"],
                                  "text": text[span["start_char"]:span["end_char"]]})
            for span in r.get("drug_spans", []):
                entities.append({"label": "DRUG",
                                  "start": span["start_char"], "end": span["end_char"],
                                  "text": text[span["start_char"]:span["end_char"]]})
            rows.append({"id": sid, "text": text, "entities": entities,
                          "source_type": "dailymed_insert", "silver": True})
    return rows


def load_newsletter_silver():
    """Newsletter silver NER rows from DrugLens harvest (Step A). Returns [] if file absent."""
    if not NEWSLETTER_SILVER.exists():
        log.warning(f"  [newsletter silver] file not found: {NEWSLETTER_SILVER.name} -- skipping")
        return []
    rows = []
    with open(NEWSLETTER_SILVER, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid  = "nl_" + __import__("hashlib").md5(
                r["sentence"].encode()).hexdigest()[:10]
            text = r["sentence"]
            entities = []
            for span in r.get("adr_spans", []):
                entities.append({"label": "ADR",
                                  "start": span["start_char"], "end": span["end_char"],
                                  "text": text[span["start_char"]:span["end_char"]]})
            for span in r.get("drug_spans", []):
                entities.append({"label": "DRUG",
                                  "start": span["start_char"], "end": span["end_char"],
                                  "text": text[span["start_char"]:span["end_char"]]})
            rows.append({"id": sid, "text": text, "entities": entities,
                          "source_type": "fda_newsletter", "silver": True})
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Val-split helper
# ═══════════════════════════════════════════════════════════════════════════════

def make_val_split(rows, key_fn):
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
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def _clf_weights(train_rows, device, adr_boost: float = 1.4):
    labels = [r["label"] for r in train_rows]
    counts = Counter(labels)
    total  = len(labels)
    w = torch.tensor([total / (2 * counts[i]) for i in range(2)], dtype=torch.float)
    w[1] *= adr_boost
    return w.to(device)


def train_clf_fold(fold_dir, train_rows, val_rows, tokenizer, device,
                   adr_boost: float = 1.4):
    ckpt = fold_dir / "clf_best"
    ckpt.mkdir(parents=True, exist_ok=True)

    for split_name, rows in [("train", train_rows), ("val", val_rows)]:
        with open(fold_dir / f"cls_{split_name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "text": r["text"],
                    "label": r["label"], "source_type": r.get("source_type", "")},
                    ensure_ascii=False) + "\n")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "no_adr", 1: "contains_adr"},
        label2id={"no_adr": 0, "contains_adr": 1},
    ).to(device)

    train_ds = ADRDataset(fold_dir / "cls_train.jsonl", tokenizer, CLF_MAXLEN)
    val_ds   = ADRDataset(fold_dir / "cls_val.jsonl",   tokenizer, CLF_MAXLEN)
    tl = DataLoader(train_ds, batch_size=CLF_BS, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=CLF_BS)

    crit = torch.nn.CrossEntropyLoss(weight=_clf_weights(train_rows, device, adr_boost=adr_boost))
    opt  = AdamW(model.parameters(), lr=CLF_LR, weight_decay=0.01)
    sch  = get_linear_schedule_with_warmup(
        opt, int(len(tl) * CLF_EPOCHS * WARMUP_RATE), len(tl) * CLF_EPOCHS)

    best = 0.0
    for ep in range(1, CLF_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tl:
            opt.zero_grad()
            loss = crit(
                model(input_ids=batch["input_ids"].to(device),
                      attention_mask=batch["attention_mask"].to(device)).logits,
                batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            epoch_loss += loss.item()
        vm = eval_clf(model, vl, device, crit)
        if vm["f1"] > best:
            best = vm["f1"]
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
        log.info(f"    clf  ep{ep}/{CLF_EPOCHS}  loss={epoch_loss/len(tl):.4f}  val_F1={vm['f1']:.4f}")

    log.info(f"    clf best val F1={best:.4f}")
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return ckpt


def _tokenize_ner(rows, tokenizer):
    out = []
    for r in rows:
        tok = align_labels(tokenizer, r["text"], r.get("entities", []))
        enc = tokenizer(r["text"], truncation=True, max_length=512)
        out.append({"id": r.get("id", ""), "text": r["text"],
                    "entities": r.get("entities", []),
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "token_type_ids": enc.get("token_type_ids", []),
                    "labels": tok, "source_type": r.get("source_type", "")})
    return out


def train_ner_fold(fold_dir, train_rows_raw, val_rows_raw, tokenizer, device,
                   seed_ckpt=None, epochs=None):
    """Train NER fold.

    seed_ckpt: if provided, warm-start from this local checkpoint instead of MODEL_NAME.
    epochs: override NER_EPOCHS (used for warmstart fine-tuning with fewer steps).
    """
    ckpt = fold_dir / "ner_best"
    ckpt.mkdir(parents=True, exist_ok=True)

    n_epochs = epochs if epochs is not None else NER_EPOCHS

    train_aug = augment_sev(train_rows_raw)
    log.info(f"    ner  SEVERITY aug: {len(train_rows_raw)} -> {len(train_aug)} train rows")

    train_tok = _tokenize_ner(train_aug,    tokenizer)
    val_tok   = _tokenize_ner(val_rows_raw, tokenizer)

    for split_name, rows in [("train", train_tok), ("val", val_tok)]:
        with open(fold_dir / f"ner_{split_name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if seed_ckpt is not None:
        log.info(f"    ner  warm-start from: {Path(seed_ckpt).name}")
        model = AutoModelForTokenClassification.from_pretrained(
            seed_ckpt, num_labels=len(NER_LABELS),
            id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True).to(device)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, num_labels=len(NER_LABELS),
            id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    tds = NERDataset(fold_dir / "ner_train.jsonl", NER_MAXLEN)
    vds = NERDataset(fold_dir / "ner_val.jsonl",   NER_MAXLEN)
    tl  = DataLoader(tds, batch_size=NER_BS, shuffle=True)
    vl  = DataLoader(vds, batch_size=NER_BS)

    cnt = torch.zeros(len(NER_LABELS))
    for batch in DataLoader(tds, batch_size=64):
        for lbl in batch["labels"].view(-1):
            if lbl.item() != -100: cnt[lbl.item()] += 1
    cnt     = cnt.clamp(min=1)
    wt      = (cnt.sum() / (len(NER_LABELS) * cnt)).clamp(max=20.0).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=wt, ignore_index=-100)

    opt = AdamW(model.parameters(), lr=NER_LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(
        opt, int(len(tl) * n_epochs * WARMUP_RATE), len(tl) * n_epochs)

    best = 0.0
    for ep in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tl:
            opt.zero_grad()
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            loss = loss_fn(logits.view(-1, len(NER_LABELS)),
                           batch["labels"].view(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            epoch_loss += loss.item()
        vm = eval_ner_raw(model, vl, device)
        if vm["f1"] > best:
            best = vm["f1"]
            model.save_pretrained(ckpt)
        log.info(f"    ner  ep{ep}/{n_epochs}  loss={epoch_loss/len(tl):.4f}  val_F1={vm['f1']:.4f}")

    log.info(f"    ner best val F1={best:.4f}")
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return ckpt


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation — Pass A (overrides ON)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_clf_fold(ckpt, test_rows, tokenizer, device, source=None):
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for r in test_rows:
            enc  = tokenizer(r["text"], truncation=True, max_length=CLF_MAXLEN,
                             padding="max_length", return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device)).logits
            probs  = torch.softmax(logits, dim=1).squeeze()
            t      = CLF_THRESHOLD_PER_SOURCE.get(source, CLF_THRESHOLD)
            pred   = int(probs[1].item() >= t)
            if has_negation(r["text"]): pred = 0
            elif ((_DRUG_TERMS_PAT.search(r["text"]) or _ADR_TERMS_PAT.search(r["text"]))
                  and any(p.search(r["text"]) for p in INTOLERANCE)):
                pred = 1
            preds.append(pred)
            labels.append(int(r["label"]))
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return {
        "clf_precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "clf_recall":    round(float(recall_score(   labels, preds, zero_division=0)), 4),
        "clf_f1":        round(float(f1_score(       labels, preds, zero_division=0)), 4),
    }


def _spans_to_bio(text, entities, offsets):
    cl = ["O"] * len(text)
    for e in entities:
        s   = e.get("start", 0)
        end = min(e.get("end", 0), len(text))
        lbl = e.get("label", "O")
        for i in range(s, end):
            cl[i] = f"B-{lbl}" if i == s else f"I-{lbl}"
    return [cl[cs] if cs < len(cl) else "O" for cs, ce in offsets if cs != ce]


def eval_ner_fold(ckpt, test_rows, tokenizer, device):
    model = AutoModelForTokenClassification.from_pretrained(ckpt).to(device)
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
    if device.type == "cuda": torch.cuda.empty_cache()

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


# ═══════════════════════════════════════════════════════════════════════════════
# Fold runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_fold(source, gold_clf, gold_ner, aug_clf, aug_ner,
             silver_ner, tokenizer, device, demo_aug_ner=None,
             newsletter_silver_ner=None, skip_warmstart=False,
             interview_adr_boost=1.0):
    fold_dir  = REPORTS / f"models_phase2b_{source}"
    fold_json = REPORTS / f"fold_phase2b_{source}.json"
    fold_dir.mkdir(parents=True, exist_ok=True)

    if fold_json.exists():
        log.info(f"  <- {fold_json.name} exists -- skipping (delete to rerun)")
        with open(fold_json, encoding="utf-8") as f:
            return json.load(f)

    clf_test       = [r for r in gold_clf
                      if r.get("source_type") == source and not r.get("augmented")]
    clf_train_gold = [r for r in gold_clf if r.get("source_type") != source]
    clf_train_aug  = [r for r in aug_clf  if r.get("augmented")]

    ner_test       = [r for r in gold_ner
                      if r.get("source_type") == source and not r.get("augmented")]
    ner_train_gold = [r for r in gold_ner if r.get("source_type") != source]
    ner_train_aug  = [r for r in aug_ner  if r.get("augmented")]

    # CLF: gold+aug only (no silver)
    clf_subtrain, clf_val = make_val_split(
        clf_train_gold, lambda r: (r["label"], r.get("source_type", "")))
    clf_subtrain = clf_subtrain + clf_train_aug

    # NER: gold+aug+silver+demo_aug+newsletter_silver(newsletter fold only)
    _demo_aug = ([r for r in demo_aug_ner if r.get("augmented")]
                 if demo_aug_ner else [])
    ner_train_demo = [r for r in _demo_aug if r.get("source_type") != source]
    nl_silver = (newsletter_silver_ner or []) if source == "fda_newsletter" else []
    ner_subtrain, ner_val = make_val_split(
        ner_train_gold, lambda r: r.get("source_type", ""))
    ner_subtrain = ner_subtrain + ner_train_aug + ner_train_demo + silver_ner + nl_silver

    n_total    = len(clf_test)
    n_positive = sum(r["label"] for r in clf_test)
    pct_pos    = round(100.0 * n_positive / max(n_total, 1), 1)

    log.info(f"\n{'='*64}")
    log.info(f"  FOLD  hold-out={source}  [Phase 2b: CLF=gold+aug, NER=gold+aug+silver]")
    log.info(f"  clf  test={n_total} ({n_positive} pos, {pct_pos}%)"
             f"  subtrain={len(clf_subtrain)} (gold={len(clf_subtrain)-len(clf_train_aug)}"
             f", aug={len(clf_train_aug)}, silver=0)  val={len(clf_val)}")
    log.info(f"  ner  test={len(ner_test)}"
             f"  subtrain={len(ner_subtrain)}"
             f" (gold={len(ner_subtrain)-len(ner_train_aug)-len(ner_train_demo)-len(silver_ner)-len(nl_silver)}"
             f", aug={len(ner_train_aug)}, demo_aug={len(ner_train_demo)}"
             f", silver={len(silver_ner)}, nl_silver={len(nl_silver)})  val={len(ner_val)}")
    log.info(f"{'='*64}")

    clf_ckpt = fold_dir / "clf_best"
    ner_ckpt = fold_dir / "ner_best"

    # ── CLF: seed from existing checkpoint (identical training data) ──────────
    # qualitative_interview uses interview_adr_boost (default 1.0, precision recovery); others use 1.4
    clf_adr_boost = interview_adr_boost if source == "qualitative_interview" else 1.4
    clf_seed = CLF_SEED_DIRS[source] / "clf_best"
    if clf_ckpt.exists() and any(clf_ckpt.iterdir()):
        log.info("  -> CLF checkpoint found -- skipping")
    elif clf_seed.exists() and any(clf_seed.iterdir()) and source != "qualitative_interview":
        shutil.copytree(clf_seed, clf_ckpt)
        log.info(f"  -> CLF seeded from {clf_seed.parent.name}/clf_best")
    else:
        log.info(f"  -> Training CLF (adr_boost={clf_adr_boost})...")
        t0 = time.time()
        clf_ckpt = train_clf_fold(fold_dir, clf_subtrain, clf_val, tokenizer, device,
                                  adr_boost=clf_adr_boost)
        log.info(f"  <- CLF done in {(time.time()-t0)/60:.1f}m")

    # ── NER: train fresh with silver (newsletter fold uses newsletter silver too, Step A)
    ner_seed_dir = NER_SEED_DIRS.get(source)
    ner_seed = (ner_seed_dir / "ner_best") if ner_seed_dir else None
    ner_warmstart_dir = None if skip_warmstart else NER_WARMSTART_DIRS.get(source)
    ner_warmstart = (ner_warmstart_dir / "ner_best") if ner_warmstart_dir else None

    if ner_ckpt.exists() and any(ner_ckpt.iterdir()):
        log.info("  -> NER checkpoint found -- skipping")
    elif ner_seed and ner_seed.exists() and any(ner_seed.iterdir()):
        shutil.copytree(ner_seed, ner_ckpt)
        log.info(f"  -> NER seeded from Phase 2 ({ner_seed.parent.name}/ner_best)")
    elif ner_warmstart and ner_warmstart.exists() and any(ner_warmstart.iterdir()):
        log.info(f"  -> NER warm-start from Phase 4c ({NER_WARMSTART_EPOCHS} epochs)...")
        t0 = time.time()
        ner_ckpt = train_ner_fold(fold_dir, ner_subtrain, ner_val, tokenizer, device,
                                  seed_ckpt=str(ner_warmstart),
                                  epochs=NER_WARMSTART_EPOCHS)
        log.info(f"  <- NER warm-start done in {(time.time()-t0)/60:.1f}m")
    else:
        log.info("  -> Training NER (gold+aug+silver)...")
        t0 = time.time()
        ner_ckpt = train_ner_fold(fold_dir, ner_subtrain, ner_val, tokenizer, device)
        log.info(f"  <- NER done in {(time.time()-t0)/60:.1f}m")

    log.info("  -> Evaluating (Pass A -- overrides ON)...")
    clf_m = eval_clf_fold(clf_ckpt, clf_test, tokenizer, device, source=source)
    ner_m = eval_ner_fold(ner_ckpt, ner_test, tokenizer, device)

    base = V1AUG_BASELINE.get(source, {})
    clf_d = round(clf_m["clf_f1"] - base.get("clf_f1", 0), 4)
    ner_d = round(ner_m["ner_overall_f1"] - base.get("ner_overall_f1", 0), 4)

    log.info(f"  CLF  P={clf_m['clf_precision']:.4f}  R={clf_m['clf_recall']:.4f}"
             f"  F1={clf_m['clf_f1']:.4f}  (vs v1+aug: {clf_d:+.4f})")
    log.info(f"  NER  overall F1={ner_m['ner_overall_f1']:.4f}  (vs v1+aug: {ner_d:+.4f})")
    for tag in ENTITY_TAGS:
        t = ner_m["ner_per_tag"][tag]
        note = "  <- noisy" if t.get("f1_note") else ""
        log.info(f"       {tag:<14} F1={t['f1']:.4f}  support={t['support']}{note}")

    result = {
        "held_out_source": source,
        "silver_mode":     "NER only (gold+aug+silver NER, gold+aug CLF)",
        "n_total":         n_total,
        "n_positive":      n_positive,
        "pct_positive":    pct_pos,
        "clf_train_rows":  len(clf_subtrain),
        "ner_train_rows":  len(ner_subtrain),
        "ner_silver_rows": len(silver_ner),
        "v1aug_clf_f1":    base.get("clf_f1"),
        "v1aug_ner_f1":    base.get("ner_overall_f1"),
        "clf_delta_vs_v1aug": clf_d,
        "ner_delta_vs_v1aug": ner_d,
        **clf_m,
        **ner_m,
    }
    with open(fold_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"  Saved -> {fold_json.name}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════

def _macro(results):
    avg = {}
    for k in ["clf_precision", "clf_recall", "clf_f1", "ner_overall_f1"]:
        avg[k] = round(sum(r[k] for r in results) / len(results), 4)
    avg["clf_delta_vs_v1aug"] = round(sum(r.get("clf_delta_vs_v1aug", 0) for r in results) / len(results), 4)
    avg["ner_delta_vs_v1aug"] = round(sum(r.get("ner_delta_vs_v1aug", 0) for r in results) / len(results), 4)
    avg["ner_per_tag"] = {}
    for tag in ENTITY_TAGS:
        f1s = [r["ner_per_tag"][tag]["f1"]
               for r in results if "f1_note" not in r["ner_per_tag"].get(tag, {})]
        avg["ner_per_tag"][tag] = {
            "f1":      round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "support": sum(r["ner_per_tag"].get(tag, {}).get("support", 0) for r in results),
        }
    return avg


def write_summary(results):
    path  = REPORTS / "summary_phase2b.md"
    macro = _macro(results)

    header = (
        "# LOSO Summary — Phase 2b (Pass A, overrides ON)\n\n"
        "Model: CLF = gold+aug only (no silver) | NER = gold+aug+silver (287 DailyMed inserts)\n"
        "CLF checkpoints: v1+aug for case_report/cohort_study/qualitative_interview; "
        "Phase 2b for fda_newsletter\n"
        "NER checkpoints: Phase 2 seeded for fda_newsletter; trained fresh for other folds\n"
        f"Seed={SEED}  CLF_EPOCHS={CLF_EPOCHS}  NER_EPOCHS={NER_EPOCHS}  max_length={CLF_MAXLEN}\n"
        "\\* = noisy (support < 10 in test fold)\n"
        "Delta vs v1+aug baseline; macro = straight mean; noisy per-tag folds excluded.\n\n"
    )

    col_hdr = (
        "| Source                 | N   | Pos%"
        " | CLF P | CLF R | CLF F1 | ΔCLF"
        " | NER F1 | ΔNER  | DRUG  | ADR   | SEV    | DEMO  |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    def fmt_row(label, n, pct, r):
        sev   = r["ner_per_tag"]["SEVERITY"]
        sev_s = f"{sev['f1']:.3f}{'*' if sev.get('f1_note') else ''}"
        clf_d = r.get("clf_delta_vs_v1aug", 0)
        ner_d = r.get("ner_delta_vs_v1aug", 0)
        return (
            f"| {label:<22}"
            f"| {str(n):<4}"
            f"| {str(pct):<5}"
            f"| {r['clf_precision']:.3f} "
            f"| {r['clf_recall']:.3f} "
            f"| {r['clf_f1']:.3f}  "
            f"| {clf_d:+.3f}"
            f"| {r['ner_overall_f1']:.3f}  "
            f"| {ner_d:+.3f}"
            f"| {r['ner_per_tag']['DRUG']['f1']:.3f} "
            f"| {r['ner_per_tag']['ADR']['f1']:.3f} "
            f"| {sev_s:<6} "
            f"| {r['ner_per_tag']['PATIENT_DEMO']['f1']:.3f} |"
        )

    fold_rows = [fmt_row(r["held_out_source"], r["n_total"], r["pct_positive"], r)
                 for r in results]
    macro_row = fmt_row("**macro-avg**", "—", "—", macro)

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(col_hdr + "\n")
        f.write(sep + "\n")
        for row in fold_rows:
            f.write(row + "\n")
        f.write(macro_row + "\n")

    log.info(f"  Summary -> {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="4-fold LOSO — Phase 2b")
    parser.add_argument("--source", help="Single fold. E.g. --source cohort_study")
    parser.add_argument("--skip-demo-aug", action="store_true",
                        help="Skip PATIENT_DEMO augmentation (NER recovery mode)")
    parser.add_argument("--skip-warmstart", action="store_true",
                        help="Skip NER warm-start from Phase 4c checkpoint")
    parser.add_argument("--interview-adr-boost", type=float, default=1.0,
                        help="ADR class-weight boost for qualitative_interview CLF (default: 1.0)")
    args = parser.parse_args()

    log.info(f"\n{'='*64}")
    log.info("  Ghana ADR -- Phase 2b LOSO (4 folds, Pass A)")
    log.info(f"  CLF: gold+aug only | NER: gold+aug+silver")
    log.info(f"  Seed={SEED}  CLF_EPOCHS={CLF_EPOCHS}  NER_EPOCHS={NER_EPOCHS}  maxlen={CLF_MAXLEN}")
    log.info(f"{'='*64}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")
    if device.type == "cpu":
        log.warning(
            "\n  CPU-only. CLF seeded from checkpoints; only NER trains fresh:\n"
            f"    NER per fold: ~{NER_EPOCHS} ep x ~65 min = ~{NER_EPOCHS*65} min\n"
            f"    3 new NER folds total: ~{3*NER_EPOCHS*65} min\n"
        )

    log.info("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log.info("  Loading data...")
    gold_clf     = load_gold_clf()
    gold_ner     = load_gold_ner()
    aug_clf      = load_aug_clf()
    aug_ner      = load_aug_ner()
    demo_aug_ner        = [] if args.skip_demo_aug else load_demo_aug_ner()
    silver_ner          = load_silver_ner()
    newsletter_silver   = load_newsletter_silver()

    log.info(f"  Gold: {len(gold_clf)} clf  {len(gold_ner)} ner"
             f"  Aug: {len(aug_clf)} clf  {len(aug_ner)} ner"
             f"  Demo aug NER: {len(demo_aug_ner)}"
             f"  Silver NER: {len(silver_ner)}"
             f"  Newsletter Silver NER: {len(newsletter_silver)}")

    # Create frozen NER warmstart seeds (Step B): copy Phase 2b ner_best to a separate
    # immutable dir so deleting fold_dir/ner_best doesn't destroy the warmstart source.
    for ws_source, ws_dir in NER_WARMSTART_DIRS.items():
        src_ner  = REPORTS / f"models_phase2b_{ws_source}" / "ner_best"
        seed_ner = ws_dir / "ner_best"
        if src_ner.exists() and not seed_ner.exists():
            log.info(f"  Creating warmstart seed for {ws_source}: {ws_dir.name}/ner_best")
            ws_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_ner, seed_ner)
        elif not src_ner.exists():
            log.warning(f"  [warmstart] Phase 2b NER checkpoint missing for {ws_source} -- warm-start disabled")

    sources = FOLD_SOURCES
    if args.source:
        if args.source not in sources:
            log.error(f"Unknown source '{args.source}'. Valid: {sources}")
            sys.exit(1)
        sources = [args.source]

    all_results = []
    total_start = time.time()

    for i, source in enumerate(sources, 1):
        log.info(f"\n  [{i}/{len(sources)}] fold={source}")
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

        fold_start = time.time()
        result     = run_fold(source, gold_clf, gold_ner, aug_clf, aug_ner,
                              silver_ner, tokenizer, device,
                              demo_aug_ner=demo_aug_ner,
                              newsletter_silver_ner=newsletter_silver,
                              skip_warmstart=args.skip_warmstart,
                              interview_adr_boost=args.interview_adr_boost)
        all_results.append(result)

        elapsed   = time.time() - total_start
        eta       = (elapsed / i) * (len(sources) - i)
        log.info(f"  [{i}/{len(sources)}] done  fold={(time.time()-fold_start)/60:.1f}m"
                 f"  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    if len(all_results) == len(sources):
        path = write_summary(all_results)

        log.info(f"\n{'='*64}")
        log.info("  RESULTS — Phase 2b LOSO")
        log.info(f"{'='*64}")
        for r in all_results:
            sev = r["ner_per_tag"]["SEVERITY"]
            log.info(
                f"  {r['held_out_source']:<28}"
                f"  CLF_F1={r['clf_f1']:.3f} ({r['clf_delta_vs_v1aug']:+.3f})"
                f"  NER_F1={r['ner_overall_f1']:.3f} ({r['ner_delta_vs_v1aug']:+.3f})"
                f"  DRUG={r['ner_per_tag']['DRUG']['f1']:.3f}"
                f"  ADR={r['ner_per_tag']['ADR']['f1']:.3f}"
                f"  SEV={sev['f1']:.3f}{'*' if sev.get('f1_note') else ''}"
                f"  DEMO={r['ner_per_tag']['PATIENT_DEMO']['f1']:.3f}"
            )
        log.info(f"\n  Report: {path}")

    log.info(f"\n  Total elapsed: {(time.time()-total_start)/60:.1f}m")
    log.info("  Phase 2b LOSO complete.")


if __name__ == "__main__":
    main()
