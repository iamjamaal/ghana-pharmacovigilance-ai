# LOSO Summary — Phase 9 (FP exonerators + malaise + hedged signal)

CLF training: gold+aug+syn_ph7 (754) + syn_ph8 (359) + syn_ph8b (302) + openfda (329) + syn_ph9 (195) = 1939 synthetic total
NER training: unchanged from Phase 7 (gold+aug+silver DailyMed+newsletter)
Base model: models/pubmedbert-ghana-dapt (DAPT, PPL 6.11→4.55)
NER warm-start: Phase 7 NER best for qualitative_interview fold
New synthetic priorities: P32–P36 (investigation exonerates, not-started, drug-holiday,
                                   objective-normalises, coincidental illness)
                          P37–P40 (asym lab ADR, sev-min, malaise idioms, hedged signal)
Baseline to beat (Phase 7 Hybrid): CLF macro=0.724, NER macro=0.655
Seed=42  CLF_EPOCHS=3  NER_EPOCHS=4  max_length=128
\* = noisy (support < 10 in test fold)
Delta vs Phase 7 per-fold baseline; macro = straight mean.

| Source                 | N   | Pos% | CLF P | CLF R | CLF F1 | ΔCLF | NER F1 | ΔNER  | DRUG  | ADR   | SEV    | DEMO  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| case_report           | 44  | 56.8 | 0.724 | 0.840 | 0.778  | +0.002| 0.577  | -0.024| 0.862 | 0.500 | 0.000* | 0.000 |
| cohort_study          | 123 | 48.0 | 0.765 | 0.441 | 0.559  | -0.145| 0.763  | +0.007| 0.773 | 0.906 | 0.205* | 0.113 |
| fda_newsletter        | 99  | 27.3 | 0.750 | 0.778 | 0.764  | +0.058| 0.579  | -0.000| 0.612 | 0.616 | 0.235* | 0.296 |
| qualitative_interview | 78  | 25.6 | 0.360 | 0.900 | 0.514  | -0.048| 0.700  | +0.099| 0.571 | 0.842 | 0.615* | 0.557 |
| **macro-avg**         | —   | —    | 0.650 | 0.740 | 0.654  | -0.033| 0.655  | +0.020| 0.705 | 0.716 | 0.000  | 0.322 |
