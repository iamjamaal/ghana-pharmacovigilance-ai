# Final Evaluation — Phase 2b vs Phase 3 DAPT

> **Phase 2b**: LOSO fold-specific models (CLF = gold+aug, NER = gold+aug+silver).  
> **Phase 3 DAPT**: LOSO fold-specific models trained from DAPT backbone  
> (`models/pubmedbert-ghana-dapt/`, PPL 6.11→4.55, 1,582 steps).  
> Evaluation mode: Pass A (negation + intolerance overrides ON).  
> Phase 2b metrics re-run from saved checkpoints.  
> Phase 3 LOSO metrics from recorded fold runs — fold checkpoints were not retained.

## Phase 2b — LOSO Fold Results (re-run from checkpoints)

\* = noisy (support < 10 in test fold)

| Source | N | Pos% | CLF P | CLF R | CLF F1 | NER F1 | DRUG | ADR | SEV | DEMO |
|---|---|---|---|---|---|---|---|---|---|---|
| case_report | 44 | 56.8 | 0.595 | 1.000 | 0.746 | 0.606 | 0.746 | 0.603 | 0.000* | 0.222* |
| cohort_study | 123 | 48.0 | 0.727 | 0.814 | 0.768 | 0.754 | 0.760 | 0.897 | 0.210* | 0.043 |
| fda_newsletter | 99 | 27.3 | 0.857 | 0.444 | 0.585 | 0.539 | 0.576 | 0.585 | 0.242* | 0.080 |
| qualitative_interview | 78 | 25.6 | 0.333 | 1.000 | 0.500 | 0.624 | 0.481 | 0.826 | 0.615* | 0.355 |
| **macro-avg** | — | — | — | — | **0.650** | **0.631** | — | — | — | — |

## Phase 3 DAPT — LOSO Fold Results (recorded fold runs)

\* = noisy (support < 10 in test fold)

| Source | N | Pos% | CLF P | CLF R | CLF F1 | NER F1 | DRUG | ADR | SEV | DEMO |
|---|---|---|---|---|---|---|---|---|---|---|
| case_report | 44 | 56.8 | 0.610 | 1.000 | 0.758 | 0.603 | 0.757 | 0.594 | 0.000* | 0.000* |
| cohort_study | 123 | 48.0 | 0.640 | 0.966 | 0.770 | 0.730 | 0.736 | 0.875 | 0.205* | 0.000 |
| fda_newsletter | 99 | 27.3 | 0.636 | 0.518 | 0.571 | 0.572 | 0.605 | 0.612 | 0.242* | 0.276 |
| qualitative_interview | 78 | 25.6 | 0.328 | 1.000 | 0.494 | 0.586 | 0.556 | 0.824 | 0.533* | 0.164 |
| **macro-avg** | — | — | — | — | **0.648** | **0.623** | — | — | — | — |

## Final Comparison Table

| Source | N | Phase 2b CLF | DAPT CLF | Phase 2b NER | DAPT NER |
|---|---|---|---|---|---|
| case_report | 44 | 0.746 | 0.758 | 0.606 | 0.603 |
| cohort_study | 123 | 0.768 | 0.770 | 0.754 | 0.730 |
| fda_newsletter | 99 | 0.585 | 0.571 | 0.539 | 0.572 |
| qualitative_interview | 78 | 0.500 | 0.494 | 0.624 | 0.586 |
| **macro** | — | **0.650** | **0.648** | **0.631** | **0.623** |

## In-Distribution Comparison (Random Split)

Single 70/15/15 split of all gold data (test n≈65, 35% positive).

| Metric | Phase 2b | Phase 3 DAPT | Delta |
|---|---|---|---|
| CLF F1 | 0.775 | 0.808 | +0.033 |
| NER F1 | 0.764 | 0.765 | +0.001 |

> DAPT improves in-distribution CLF (+3.3 pp) but does not improve LOSO macro generalization (macro CLF −0.2 pp, macro NER −0.8 pp). Phase 2b is the better out-of-domain generalizer.

## Manifest Keys (`output/gold/manifest_v1.json`)

```json
[
  "version",
  "created",
  "files",
  "ner_duplicate_pairs",
  "clf_duplicate_pairs"
]
```
