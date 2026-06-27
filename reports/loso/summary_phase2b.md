# LOSO Summary — Phase 2b (Pass A, overrides ON)

Model: CLF = gold+aug only (no silver) | NER = gold+aug+silver (287 DailyMed inserts)
CLF checkpoints: v1+aug for case_report/cohort_study; Phase 2b for fda_newsletter/qualitative_interview
NER checkpoints: trained fresh for all folds
Seed=42  CLF_EPOCHS=3  NER_EPOCHS=4  max_length=128
\* = noisy (support < 10 in test fold)
Delta vs v1+aug baseline; macro = straight mean; noisy per-tag folds excluded.

---

## Phase 6 Final — Steps A+B+C+D (Newsletter Silver + Interview CLF/NER + adr_boost tuning)

Steps applied:
- **Step A**: 74 DrugLens newsletter silver rows added to fda_newsletter NER subtrain (`newsletter_silver_v1.jsonl`)
- **Step B**: qualitative_interview NER warm-started 2 epochs from Phase 4c checkpoint (PATIENT_DEMO recovery)
- **Step C**: qualitative_interview CLF retrained with `adr_boost=1.0` (initial precision recovery)
- **Step D**: qualitative_interview CLF retrained with `adr_boost=0.8` (+0.3pp CLF vs Step C)

Per-source thresholds: case_report=0.70, cohort_study=0.55, fda_newsletter=0.30, qualitative_interview=0.55
Scripts: `scripts/loso_phase2b_full.py --interview-adr-boost 0.8` + `scripts/phase5_per_source_threshold.py`

| Source                 | t_CLF | N   | Pos% | CLF P | CLF R | CLF F1 | ΔCLF vs Ph5 | NER F1 | ΔNER vs Ph5 | DRUG  | ADR   | DEMO  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| case_report           | 0.70  | 44  | 56.8 | 0.667 | 0.960 | 0.787  | +0.000 | 0.638  | +0.000 | 0.836 | 0.618 | 0.222 |
| cohort_study          | 0.55  | 123 | 48.0 | 0.789 | 0.763 | 0.776  | +0.000 | 0.754  | +0.000 | 0.760 | 0.897 | 0.043 |
| fda_newsletter        | 0.30  | 99  | 27.3 | 0.933 | 0.519 | 0.667  | +0.000 | 0.564  | +0.025 | 0.616 | 0.586 | 0.074 |
| qualitative_interview | 0.55  | 78  | 25.6 | 0.474 | 0.900 | 0.621  | +0.065 | 0.647  | +0.023 | 0.531 | 0.845 | 0.381 |
| **macro-avg**         | —     | —   | —    | —     | —     | **0.713** | **+0.017** | **0.651** | **+0.012** | 0.686 | 0.737 | 0.180 |

Delta vs Phase 5 baseline (CLF=0.696, NER=0.639).

---

## Phase 5 Final — Per-Source CLF Threshold (Step 5.0)

Per-fold thresholds from val sweep + domain-informed newsletter prior.
No retraining. CLF/NER checkpoints unchanged from Phase 4.

| Source                 | t_CLF | N   | Pos% | CLF P | CLF R | CLF F1 | ΔCLF vs Ph4 | NER F1 | DRUG  | ADR   | DEMO  |
|---|---|---|---|---|---|---|---|---|---|---|---|
| case_report           | 0.70  | 44  | 56.8 | 0.667 | 0.960 | 0.787  | +0.048 | 0.638  | 0.836 | 0.618 | 0.080 |
| cohort_study          | 0.55  | 123 | 48.0 | 0.789 | 0.763 | 0.776  | +0.000 | 0.754  | 0.760 | 0.897 | 0.043 |
| fda_newsletter        | 0.30  | 99  | 27.3 | 0.933 | 0.519 | 0.667  | +0.067 | 0.539  | 0.576 | 0.585 | 0.080 |
| qualitative_interview | 0.55  | 78  | 25.6 | 0.385 | 1.000 | 0.556  | +0.000 | 0.624  | 0.481 | 0.826 | 0.355 |
| **macro-avg**         | —     | —   | —    | —     | —     | **0.696** | **+0.028** | **0.639** | 0.663 | 0.732 | 0.140 |

Delta vs Phase 4 baseline (CLF=0.668, NER=0.639).
Script: `scripts/phase5_per_source_threshold.py` | Results: `reports/phase5_per_source_threshold.json`

---

## Phase 4 Final (Steps 4.0–4c)

Improvements over Phase 2b: INTOLERANCE gate fix (4.0), CLF threshold=0.55 (4.1),
case_report NER retrain with drug aug bonus (4b), NER recovery for cohort/interview (4c).

| Source                 | N   | Pos% | CLF P | CLF R | CLF F1 | ΔCLF | NER F1 | ΔNER  | DRUG  | ADR   | SEV    | DEMO  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| case_report           | 44  | 56.8 | 0.600 | 0.960 | 0.739  | -0.007| 0.638  | +0.031| 0.836 | 0.618 | 0.000* | 0.222 |
| cohort_study          | 123 | 48.0 | 0.789 | 0.763 | 0.776  | +0.008| 0.754  | +0.095| 0.760 | 0.897 | 0.210* | 0.043 |
| fda_newsletter        | 99  | 27.3 | 0.923 | 0.444 | 0.600  | -0.005| 0.539  | +0.045| 0.576 | 0.585 | 0.242* | 0.080 |
| qualitative_interview | 78  | 25.6 | 0.385 | 1.000 | 0.556  | +0.056| 0.624  | +0.036| 0.481 | 0.826 | 0.615* | 0.355 |
| **macro-avg**         | —   | —    | 0.674 | 0.792 | **0.668** | **+0.018** | **0.639** | **+0.008** | 0.663 | 0.732 | — | 0.175 |

Delta vs Phase 2b baseline (CLF=0.650, NER=0.631).

## Step 4.4 — Ensemble Eval (Phase 2b + Phase 3 DAPT)

Rule applied **uniformly to all 4 folds**: CLF = average Phase 2b + Phase 3 DAPT softmax
(threshold=0.55); NER = Phase 2b base with Phase 3 PATIENT_DEMO injection where Phase 2b=O
and Phase 3 confidence > 0.70. Script: `scripts/phase4_ensemble_eval.py`.
Full results: `reports/phase4_ensemble.json`.

| Source                 | CLF F1 | ΔCLF  | NER F1 | ΔNER   | DEMO  | ΔDEMO  |
|---|---|---|---|---|---|---|
| case_report           | 0.758  | +0.019 | 0.638  | +0.000 | 0.222 | +0.000 |
| cohort_study          | 0.785  | +0.009 | 0.754  | +0.000 | 0.043 | -0.000 |
| fda_newsletter        | 0.634  | +0.034 | 0.539  | -0.000 | 0.080 | +0.000 |
| qualitative_interview | 0.526  | -0.030 | 0.600  | -0.024 | 0.258 | -0.097 |
| **macro-avg**         | **0.676** | **+0.008** | **0.633** | **-0.006** | | |

Delta vs Phase 4 baseline (CLF=0.668, NER=0.639).

**Best configuration:** CLF ensemble (averaging) + Phase 4 NER for all folds.
Macro CLF = **0.676** (+2.6pp vs Phase 2b, +0.8pp vs Phase 4). Macro NER = **0.639** (unchanged).

---

## Phase 6 Ensemble — Phase 6 + Phase 3 DAPT (script: `scripts/phase6_ensemble_eval.py`)

| Source | CLF F1 | ΔCLF vs Ph6 | NER F1 | ΔNER vs Ph6 | DEMO | ΔDEMO |
|---|---|---|---|---|---|---|
| case_report | 0.787 | −0.000 | 0.638 | +0.000 | 0.222 | +0.000 |
| cohort_study | 0.785 | +0.009 | 0.754 | +0.000 | 0.043 | +0.000 |
| fda_newsletter | 0.577 | −0.090 | 0.564 | +0.000 | 0.074 | +0.000 |
| qualitative_interview | 0.571 | −0.050 | 0.614 | −0.033 | 0.254 | −0.127 |
| **macro** | **0.680** | **−0.033** | **0.642** | **−0.009** | | |

**Phase 6 ensemble is worse on both metrics.** Two failure modes identical to Phase 4.4:
- Newsletter CLF: t=0.30 is too low for averaged softmax — Phase 3 DAPT pulls borderline negatives over threshold (−9.0pp CLF).
- Interview NER/DEMO: Phase 3 DAPT makes high-confidence wrong DEMO predictions, overwriting Phase 6 correct O labels (−12.7pp DEMO).

**Do not use ensemble with Phase 6 checkpoints.** Phase 6 standalone is the best single-model configuration.
Full results: `reports/phase6_ensemble.json`

---

## Progressive Summary

| Configuration | Macro CLF | Macro NER | Notes |
|---|---|---|---|
| Phase 2b baseline | 0.650 | 0.631 | |
| Phase 4 (Steps 4.0–4c) | 0.668 | 0.639 | INTOLERANCE fix, threshold=0.55, NER retrain |
| Phase 4 + CLF ensemble | 0.676 | 0.639 | Dual-model CLF inference required |
| Phase 5 (per-source thresholds) | 0.696 | 0.639 | No retrain; threshold tuning only |
| Phase 6 Steps A+B+C (adr_boost=1.0) | 0.710 | 0.651 | Newsletter silver NER + interview CLF/NER retrain |
| **Phase 6 Step D (adr_boost=0.8)** | **0.713** | **0.651** | Further interview CLF tuning; **current best** |
| Phase 6 + Phase 3 ensemble | 0.680 | 0.642 | Worse than standalone; do not use |
| NER oracle ceiling | 0.696 | 0.647 | Newsletter NER swap (test-set selection, not comparable) |
