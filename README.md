# Ghana ADR Detection System

> Automated adverse drug reaction detection from free-text clinical narratives —  
> built on Ghanaian pharmacovigilance data for the **Ghana AI Innovation Challenge 2026**.

[![Phase 7 Hybrid](https://img.shields.io/badge/CLF%20F1-0.724-green)](reports/loso/summary_phase7.md)
[![NER F1](https://img.shields.io/badge/NER%20F1-0.655-green)](reports/loso/summary_phase7.md)
[![Batch regression](https://img.shields.io/badge/Batch%20regression-85%2F95%20(89.5%25)-blue)](reports/batch_regression_phase9.md)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

The Ghana ADR Detection System takes a free-text clinical sentence — from a case report, patient interview, FDA newsletter, or ward note — and:

1. **Classifies** it as ADR / Non-ADR (binary, calibrated confidence score)
2. **Extracts** named entities: `DRUG`, `ADR`, `SEVERITY`, `PATIENT_DEMO`
3. **Applies** a post-processing rule layer handling Ghanaian English idioms, Pidgin constructions, negation, dechallenge/rechallenge, and drug-name tokenisation artefacts

It supports single-sentence analysis, batch CSV/TXT upload, and a Yellow Card–style reporting workflow for Ghana FDA pharmacovigilance officers.

---

## Performance (Phase 7 Hybrid — LOSO)

> **Evaluation methodology:** Leave-One-Source-Out (LOSO) — each source domain is held out as the test set while the model trains on the remaining three. This is stricter than random splitting and measures real cross-domain generalisation.

| Source | N | CLF F1 | NER F1 | DRUG | ADR |
|---|---|---|---|---|---|
| case_report | 44 | 0.787 | 0.598 | 0.862 | 0.545 |
| cohort_study | 123 | 0.776 | 0.785 | 0.823 | 0.884 |
| fda_newsletter | 99 | 0.667 | 0.587 | 0.626 | 0.634 |
| qualitative_interview | 78 | 0.667 | 0.650 | 0.560 | 0.842 |
| **macro-avg** | — | **0.724** | **0.655** | 0.718 | 0.727 |

Batch regression against 95 curated hard cases (Pidgin idioms, dialect, regulatory register, clinical shorthand, minimal pairs): **85/95 pass (89.5%)**.

---

## Model Architecture

| Component | Detail |
|---|---|
| Base model | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (110M params) |
| DAPT backbone | PubMedBERT fine-tuned with MLM on 128k Ghanaian biomedical sentences (PPL 6.11 → 4.55) |
| CLF head | Binary sequence classifier (contains_adr 0/1) |
| NER head | Token classifier — 9 labels: O + B/I for DRUG, ADR, SEVERITY, PATIENT_DEMO |
| Production config | **Phase 7 Hybrid**: Phase 6 CLF checkpoints + Phase 7 NER checkpoints |
| Post-processing | Negation gates, INTOLERANCE patterns, Ghanaian synonym expansion, MedDRA normalisation, dechallenge/rechallenge detection |

> **Model weights** are hosted on HuggingFace Hub: [to be linked after upload]  
> **DAPT backbone** is hosted on HuggingFace Hub: [to be linked after upload]

---

## Dataset

Built entirely from Ghanaian pharmacovigilance sources:

| Source | Format | Sentences |
|---|---|---|
| Ghana FDA DrugLens newsletters (5 issues) | PDF | — |
| Ghana FDA Annual Report 2023 + ADR Guide | PDF | — |
| PMC open-access case reports & cohort studies (9 articles) | JATS XML | — |
| Patient ADR interview transcripts | Qualitative | — |
| **Total gold dataset** | — | **2,870 sentences** |
| Manually reviewed | — | 510 (17.8%) |

Silver training data: 2,105+ additional records (DailyMed weak supervision, DrugLens NER annotations, OpenFDA ICSR records, synthetic curriculum examples).

All PMC articles and Ghana FDA publications are public government documents.

---

## Repository Structure

```
ghana-adr-pipeline/
├── scripts/
│   ├── 01_download.py               # Download PDFs + PMC XML
│   ├── 02_extract.py                # Extract text from all sources
│   ├── 03_segment_annotate.py       # Sentence splitting + auto-annotation
│   ├── 04_export.py                 # Export gold datasets
│   ├── assemble_dapt_corpus.py      # Assemble DAPT pretraining corpus
│   ├── run_dapt.py                  # Domain-adaptive pretraining
│   ├── loso_phase2b_full.py         # Production CLF training (Phase 6)
│   ├── loso_phase7.py               # Production NER training (Phase 7)
│   ├── phase5_per_source_threshold.py  # Per-source CLF threshold sweep
│   ├── phase7_hybrid_eval.py        # Production hybrid evaluation
│   ├── harvest_newsletter_silver.py # DrugLens silver NER data
│   ├── harvest_openfda_silver.py    # OpenFDA ICSR silver CLF data
│   ├── generate_synthetic_phase*.py # Synthetic training curriculum
│   └── batch_regression_phase9.py  # Final 95-case regression suite
├── data/
│   ├── raw/                         # Downloaded PDFs + PMC XML
│   ├── extracted/                   # Extracted text JSON
│   ├── annotated/                   # Auto-annotated JSONL
│   ├── silver/                      # Silver + synthetic training data
│   └── gazetteers/                  # Drug and ADR lexicons
├── output/
│   ├── gold/                        # Gold classification + NER datasets
│   └── dataset_card.md
├── reports/
│   ├── loso/                        # Per-fold LOSO results (phases 2b, 7, 8, 9)
│   ├── phase7_hybrid.json           # Production config results
│   ├── phase5_per_source_threshold.json  # Production CLF thresholds
│   ├── batch_regression_phase9.md   # Final 89.5% regression report
│   └── final_evaluation.md          # Phase 2b vs DAPT comparison
├── ghana-adr-demo/
│   ├── app.py                       # Flask web application
│   ├── inference_engine.py          # Inference + post-processing logic
│   └── templates/index.html         # Demo UI
├── src/
│   ├── evaluation/loso.py           # LOSO evaluation utilities
│   └── weak_supervision/            # Weak supervision (skweak-free fallback)
├── models/                          # gitignored — see HuggingFace Hub
├── config/sources.json
├── requirements.txt
└── run_pipeline.py
```

---

## Quick Start

### Run the data pipeline

```bash
pip install -r requirements.txt
python run_pipeline.py          # Full pipeline (download → annotate → export)
python run_pipeline.py --only 1 # Download only
python run_pipeline.py --step 3 # Resume from step 3
```

### Run the demo app

```bash
cd ghana-adr-demo
pip install flask pandas transformers torch
# Download model checkpoints from HuggingFace Hub (see above)
python app.py
# Open http://localhost:5000
```

### Reproduce LOSO evaluation

```bash
# Train Phase 7 Hybrid (requires model checkpoints + DAPT backbone)
python scripts/loso_phase2b_full.py   # CLF (Phase 6)
python scripts/loso_phase7.py         # NER (Phase 7)
python scripts/phase7_hybrid_eval.py  # Evaluate hybrid config

# Run batch regression
python scripts/batch_regression_phase9.py
```

---

## Development History

| Phase | Key change | Macro CLF | Macro NER |
|---|---|---|---|
| Phase 2b baseline | Gold + aug + DailyMed silver | 0.650 | 0.631 |
| Phase 3 DAPT | MLM pretraining on 128k sentences | 0.648 | 0.623 |
| Phase 4 | INTOLERANCE fix, threshold sweep, NER retrain | 0.668 | 0.639 |
| Phase 5 | Per-source CLF thresholds | 0.696 | 0.639 |
| Phase 6 | Newsletter silver NER + interview CLF/NER retrain | 0.713 | 0.651 |
| Phase 7 Hybrid | Phase 6 CLF + Phase 7 NER (synthetic curriculum) | **0.724** | **0.655** |

---

## License

Code: MIT | Dataset: CC-BY-4.0
