#!/usr/bin/env python3
"""
run_dapt.py — Domain-Adaptive Pre-Training (DAPT) on the assembled corpus.

Base model : microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
Corpus     : corpora/dapt/*.txt (all files concatenated)
Output     : models/pubmedbert-ghana-dapt/

Settings (per Phase-3 spec):
  max_length=128, mlm_probability=0.15, num_train_epochs=2
  learning_rate=5e-5, effective batch=32 (per_device=8, grad_accum=4)
  save_steps=500, eval_strategy=steps, eval_steps=500
  load_best_model_at_end=True

Stopping condition: if eval loss drop < 0.05 after epoch 1, training stops
and reports the likely cause (corpus too small / too similar to pretraining data).

Usage:
  python scripts/run_dapt.py              # full run
  python scripts/run_dapt.py --smoke-test # 20-step sanity check
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ── paths ─────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DAPT_DIR  = ROOT / "corpora" / "dapt"
MODEL_OUT = ROOT / "models" / "pubmedbert-ghana-dapt"

BASE_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# ── hyper-parameters ──────────────────────────────────────────────────────────────
MAX_LENGTH      = 128
MLM_PROBABILITY = 0.15
NUM_EPOCHS      = 2
LR              = 5e-5
PER_DEVICE_BS   = 8
GRAD_ACCUM      = 4   # effective batch = 32
SAVE_STEPS      = 500
EVAL_STEPS      = 500
EVAL_FRAC       = 0.05
SEED            = 42

STOP_THRESHOLD  = 0.05   # minimum eval-loss drop required after epoch 1


# ── corpus loading ────────────────────────────────────────────────────────────────

def load_corpus(dapt_dir: Path, pmc_limit: int = None) -> list:
    """Load all .txt files; return list of non-empty sentence strings."""
    all_lines = []
    txt_files = sorted(dapt_dir.glob("*.txt"))
    if not txt_files:
        sys.exit(f"ERROR: no .txt files found in {dapt_dir}")
    for f in txt_files:
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if pmc_limit and "pmc" in f.name:
            lines = lines[:pmc_limit]
        print(f"  {f.name}: {len(lines):,} sentences")
        all_lines.extend(lines)
    return all_lines


# ── tokenisation ──────────────────────────────────────────────────────────────────

def tokenise(lines: list, tokenizer) -> Dataset:
    """Tokenize sentences into fixed-length blocks for MLM."""
    def _batch_encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    ds = Dataset.from_dict({"text": lines})
    ds = ds.map(_batch_encode, batched=True, batch_size=1000,
                remove_columns=["text"],
                desc="Tokenising")
    return ds


# ── stopping-condition callback ───────────────────────────────────────────────────

class EpochOneLossGate(TrainerCallback):
    """Stop if eval loss hasn't dropped >= STOP_THRESHOLD by end of epoch 1."""

    def __init__(self, threshold: float = STOP_THRESHOLD):
        self.threshold = threshold
        self.initial_eval_loss = None

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl,
                    metrics=None, **kwargs):
        if metrics is None:
            return
        loss = metrics.get("eval_loss")
        if loss is None:
            return

        if self.initial_eval_loss is None:
            self.initial_eval_loss = loss
            print(f"\n[LossGate] Initial eval loss recorded: {loss:.4f}")
            return

        # Check at end of epoch 1 (step count crosses steps_per_epoch)
        if state.epoch and state.epoch >= 1.0 and state.epoch < 1.1:
            drop = self.initial_eval_loss - loss
            print(
                f"\n[LossGate] Epoch 1 eval loss: {loss:.4f} "
                f"(drop={drop:+.4f}, threshold={self.threshold})"
            )
            if drop < self.threshold:
                print(
                    f"[LossGate] STOPPING: loss dropped only {drop:.4f} < {self.threshold}.\n"
                    "  Likely cause: corpus too small or too similar to PubMedBERT's\n"
                    "  original pretraining data (PMC African papers overlap heavily).\n"
                    "  Recommendation: upweight DrugLens / AJOL / inserts relative to PMC,\n"
                    "  or extend the non-PMC corpus before re-running DAPT."
                )
                control.should_training_stop = True


# ── main ──────────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 20 steps on 200 sentences to verify the loop")
    parser.add_argument("--pmc-limit", type=int, default=None,
                        help="Cap sentences loaded from pmc_*.txt files (e.g. 25000 for Tier A overnight run)")
    args = parser.parse_args()

    print("=" * 64)
    mode = "SMOKE TEST" if args.smoke_test else "FULL RUN"
    print(f"DAPT — {mode}")
    print(f"Base model : {BASE_MODEL}")
    print(f"Output     : {MODEL_OUT}")
    print("=" * 64)

    # ── load corpus ──────────────────────────────────────────────────────────────
    print("\nLoading corpus...")
    lines = load_corpus(DAPT_DIR, pmc_limit=args.pmc_limit)
    print(f"  Total sentences: {len(lines):,}")
    if not lines:
        sys.exit("ERROR: corpus is empty — run assemble_dapt_corpus.py first")

    random.seed(SEED)
    random.shuffle(lines)

    if args.smoke_test:
        lines = lines[:200]
        print(f"  Smoke-test: using {len(lines)} sentences")

    # ── train/eval split (5% eval) ───────────────────────────────────────────────
    n_eval = max(1, int(len(lines) * EVAL_FRAC))
    eval_lines  = lines[:n_eval]
    train_lines = lines[n_eval:]
    print(f"  Train: {len(train_lines):,} | Eval: {len(eval_lines):,}")

    # ── tokenizer + model ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer and model from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.1f}M")

    # ── tokenise datasets ────────────────────────────────────────────────────────
    print("\nTokenising...")
    train_ds = tokenise(train_lines, tokenizer)
    eval_ds  = tokenise(eval_lines,  tokenizer)
    print(f"  Train dataset: {len(train_ds):,} examples")
    print(f"  Eval  dataset: {len(eval_ds):,} examples")

    # ── data collator ────────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    # ── training arguments ───────────────────────────────────────────────────────
    if args.smoke_test:
        t_args = TrainingArguments(
            output_dir=str(MODEL_OUT / "smoke"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=20,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="no",
            logging_steps=5,
            learning_rate=LR,
            seed=SEED,
            use_cpu=True,
            report_to="none",
        )
    else:
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        t_args = TrainingArguments(
            output_dir=str(MODEL_OUT),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=100,
            seed=SEED,
            use_cpu=True,       # CPU-only machine
            dataloader_num_workers=0,
            report_to="none",
            save_total_limit=2,
        )

    # ── trainer ──────────────────────────────────────────────────────────────────
    callbacks = [] if args.smoke_test else [EpochOneLossGate(STOP_THRESHOLD)]

    trainer = Trainer(
        model=model,
        args=t_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ── initial eval loss ────────────────────────────────────────────────────────
    print("\nComputing initial eval loss (before training)...")
    t0 = time.time()
    init_metrics = trainer.evaluate()
    init_loss = init_metrics.get("eval_loss", float("nan"))
    print(f"  Initial eval loss: {init_loss:.4f} (perplexity: {math.exp(init_loss):.2f})")

    # ── train ────────────────────────────────────────────────────────────────────
    print(f"\nStarting {'smoke-test ' if args.smoke_test else ''}training...")
    train_start = time.time()
    resume_from = None
    if not args.smoke_test:
        ckpt_dirs = sorted(MODEL_OUT.glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1]))
        if ckpt_dirs:
            resume_from = str(ckpt_dirs[-1])
            print(f"  Resuming from: {resume_from}")
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - train_start

    # ── final eval ───────────────────────────────────────────────────────────────
    final_metrics = trainer.evaluate()
    final_loss = final_metrics.get("eval_loss", float("nan"))

    print("\n" + "=" * 64)
    print("DAPT RESULTS")
    print("=" * 64)
    print(f"  Starting eval loss : {init_loss:.4f}  (PPL {math.exp(init_loss):.2f})")
    print(f"  Final eval loss    : {final_loss:.4f}  (PPL {math.exp(final_loss):.2f})")
    print(f"  Loss drop          : {init_loss - final_loss:+.4f}")
    print(f"  Training time      : {elapsed/3600:.2f} h  ({elapsed:.0f} s)")
    print(f"  Train steps        : {train_result.global_step}")

    if args.smoke_test:
        print("\n[Smoke test passed] Loss computed and training ran without crash.")
        print("Re-run without --smoke-test to begin the full DAPT.")
        return

    # ── save ─────────────────────────────────────────────────────────────────────
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print(f"\nModel saved to: {MODEL_OUT}")

    drop = init_loss - final_loss
    if drop >= STOP_THRESHOLD:
        print(f"  SUCCESS: loss dropped {drop:.4f} >= {STOP_THRESHOLD} threshold")
    else:
        print(
            f"  NOTE: total loss drop {drop:.4f} < {STOP_THRESHOLD}.\n"
            "  DAPT may not have adapted meaningfully. "
            "Consider adding more non-PMC text (DrugLens, AJOL, inserts) "
            "to shift the domain further from PubMedBERT's pretraining data."
        )


if __name__ == "__main__":
    main()
