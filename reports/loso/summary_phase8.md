# LOSO Summary — Phase 8 (Pass A, overrides ON)

CLF training: gold+aug+synthetic_phase7 (754) + synthetic_phase8 (359) + synthetic_phase8b (302) + openfda_harvest (329) = 1744 synthetic total
NER training: unchanged from Phase 7 (gold+aug+silver DailyMed+newsletter)
Base model: models/pubmedbert-ghana-dapt (DAPT, PPL 6.11→4.55)
NER warm-start: Phase 7 NER best for qualitative_interview fold
New synthetic priorities: P16–P26 (FDA/reg, temporal, severity, asymptomatic...)
                          P27–P31 (Akan/Ewe/Northern/Pidgin/code-switched dialect)
OpenFDA: 329 real ICSRs for Ghana-relevant drugs (antimalarials, ARVs, anti-TB)
Baseline to beat (Phase 7): CLF macro=0.724, NER macro=0.655
Seed=42  CLF_EPOCHS=3  NER_EPOCHS=4  max_length=128
\* = noisy (support < 10 in test fold)
Delta vs Phase 7 per-fold baseline; macro = straight mean.

| Source                 | N   | Pos% | CLF P | CLF R | CLF F1 | ΔCLF | NER F1 | ΔNER  | DRUG  | ADR   | SEV    | DEMO  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| case_report           | 44  | 56.8 | 0.792 | 0.760 | 0.775  | +0.048| 0.601  | +0.003| 0.767 | 0.583 | 0.000* | 0.250 |
| cohort_study          | 123 | 48.0 | 0.775 | 0.644 | 0.704  | -0.027| 0.756  | -0.029| 0.775 | 0.889 | 0.205* | 0.040 |
| fda_newsletter        | 99  | 27.3 | 0.750 | 0.667 | 0.706  | +0.039| 0.580  | -0.007| 0.612 | 0.634 | 0.229* | 0.160 |
| qualitative_interview | 78  | 25.6 | 0.409 | 0.900 | 0.562  | +0.025| 0.602  | -0.049| 0.490 | 0.860 | 0.615* | 0.200 |
| **macro-avg**         | —   | —    | 0.682 | 0.743 | 0.687  | +0.021| 0.634  | -0.020| 0.661 | 0.741 | 0.000  | 0.133 |
