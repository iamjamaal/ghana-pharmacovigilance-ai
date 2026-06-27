# LOSO Summary — Phase 7 (Pass A, overrides ON)

Model: CLF = gold+aug+synthetic_phase7 (754 synthetic) | NER = gold+aug+silver (DailyMed+newsletter)
Base model: models/pubmedbert-ghana-dapt (DAPT, PPL 6.11→4.55)
CLF: trained fresh from DAPT (synthetic data changes training set)
NER: warm-start from Phase 6 NER for qualitative_interview; fresh from DAPT for others
Synthetic priorities covered: 5,6,7,8,10,11,15 | Deferred: 1,4 (real data needed)
Seed=42  CLF_EPOCHS=3  NER_EPOCHS=4  max_length=128
\* = noisy (support < 10 in test fold)
Delta vs Phase 6 baseline; macro = straight mean; noisy per-tag folds excluded.

| Source                 | N   | Pos% | CLF P | CLF R | CLF F1 | ΔCLF | NER F1 | ΔNER  | DRUG  | ADR   | SEV    | DEMO  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| case_report           | 44  | 56.8 | 0.667 | 0.800 | 0.727  | -0.019| 0.598  | -0.009| 0.862 | 0.545 | 0.000* | 0.000 |
| cohort_study          | 123 | 48.0 | 0.844 | 0.644 | 0.731  | -0.037| 0.785  | +0.126| 0.823 | 0.884 | 0.210* | 0.154 |
| fda_newsletter        | 99  | 27.3 | 0.708 | 0.630 | 0.667  | +0.062| 0.587  | +0.092| 0.626 | 0.634 | 0.229* | 0.154 |
| qualitative_interview | 78  | 25.6 | 0.383 | 0.900 | 0.537  | +0.037| 0.650  | +0.062| 0.560 | 0.862 | 0.500* | 0.375 |
| **macro-avg**         | —   | —    | 0.651 | 0.743 | 0.665  | +0.011| 0.655  | +0.068| 0.718 | 0.732 | 0.000  | 0.228 |
