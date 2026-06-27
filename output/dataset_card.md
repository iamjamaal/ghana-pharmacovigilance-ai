---
language: en
license: cc-by-4.0
task_categories:
  - token-classification
  - text-classification
task_ids:
  - named-entity-recognition
  - binary-classification
tags:
  - pharmacovigilance
  - adverse-drug-reactions
  - ghana
  - healthcare
  - nlp
  - africa
pretty_name: Ghana ADR Dataset
size_categories:
  - 1K<n<10K
---

# Ghana Adverse Drug Reaction (ADR) NLP Dataset

## Dataset Description

A curated corpus of adverse drug reaction mentions extracted from publicly
available Ghanaian health sources, built for the **Ghana AI Innovation
Challenge 2026** (Ghana AI Summit & Awards).

**Primary Ghanaian data sources:**
- Ghana FDA DrugLens Newsletters (Issues 3, 5, 7, 9, 10)
- Ghana FDA Annual Report 2023
- Open-access case reports from Korle-Bu, KATH, and regional hospitals
- Patient ADR interview transcripts from 28 Ghanaian health facilities
- COVID-19 vaccine AEFI surveillance data from Ghana

### Dataset Statistics

| Metric | Value |
|---|---|
| Total sentences | 2,870 |
| Manually reviewed (gold) | 510 |
| ADR-positive sentences (gold) | 164 (32.2% of reviewed) |
| ADR-negative sentences (gold) | 345 |
| Uncertain | 1 |
| Total drug mentions | 765 |
| Total ADR mentions | 630 |
| Drug–ADR relations | 4,202 |
| Source documents | 15 |
| Year range | 2014–2025 |

### Source Distribution

| Source Type | Total Sentences | Manually Reviewed |
|---|---|---|
| fda_newsletter | 675 | 99 |
| qualitative_interview | 611 | 120 |
| cohort_study | 512 | 161 |
| fda_annual_report | 387 | 4 |
| surveillance_evaluation | 375 | 55 |
| fda_guideline | 194 | 27 |
| case_report | 116 | 44 |

### Top 15 Drugs Mentioned (gold annotations)

| Drug | Mentions |
|---|---|
| AZT (zidovudine) | 55 |
| 3TC (lamivudine) | 55 |
| hydroxychloroquine | 52 |
| chloroquine | 32 |
| azithromycin | 29 |
| AstraZeneca | 27 |
| zidovudine | 26 |
| LPV/r (lopinavir-ritonavir) | 26 |
| Moderna | 20 |
| Sputnik V | 20 |
| lamivudine | 17 |
| codeine | 16 |
| artesunate-amodiaquine | 13 |
| hyoscine butylbromide | 13 |
| lopinavir | 12 |

### Top 15 ADR Terms (gold annotations)

| ADR | Mentions |
|---|---|
| dizziness | 23 |
| headache | 23 |
| anaemia | 23 |
| nausea | 18 |
| vomiting | 17 |
| diarrhoea | 16 |
| oculogyric crisis | 13 |
| fatigue | 12 |
| weakness | 11 |
| poor vision | 10 |
| abdominal pain | 10 |
| restlessness | 10 |
| fever | 9 |
| loss of appetite | 8 |
| dyspnoea | 8 |

### Model Performance (PubMedBERT fine-tuned on gold subset)

| Task | Model | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|
| Sentence classification | BiomedNLP-PubMedBERT | 0.784 | 0.833 | 0.741 |
| NER — DRUG | BiomedNLP-PubMedBERT | 0.77 | 0.74 | 0.79 |
| NER — ADR | BiomedNLP-PubMedBERT | 0.80 | 0.77 | 0.82 |
| NER — PATIENT_DEMO | BiomedNLP-PubMedBERT | 0.53 | 0.48 | 0.59 |
| NER — overall | BiomedNLP-PubMedBERT | 0.746 | 0.726 | 0.766 |

## Annotation Schema

- **DRUG**: Medication names (brand or generic)
- **ADR**: Adverse drug reaction symptoms or conditions
- **SEVERITY**: Severity indicators (mild, moderate, severe, fatal)
- **PATIENT_DEMO**: Patient demographics (age, sex, region)
- **CAUSES**: Relation linking DRUG → ADR

Sentence-level label: `contains_adr` (binary)

## Usage

```python
import json

# Load full dataset
with open("ghana_adr_dataset.jsonl") as f:
    data = [json.loads(line) for line in f]

# Filter ADR-positive sentences
adr_sentences = [row for row in data if row["contains_adr"] == 1]

# Get all drug-ADR relations
relations = []
for row in data:
    for rel in row.get("relations", []):
        relations.append(rel)
```

## Licensing

- DrugLens newsletters: Ghana Government Publication (public domain)
- PMC articles: CC-BY (Creative Commons Attribution)
- Dataset compilation: CC-BY-4.0

## Citation

```bibtex
@dataset{ghana_adr_2026,
  title={Ghana ADR NLP Dataset},
  author={[Your Team Name]},
  year={2026},
  note={Built for the Ghana AI Innovation Challenge 2026},
  url={https://github.com/[your-repo]}
}
```

## Ethical Considerations

- No individual patient identifiers are included
- All source data is from publicly available, open-access publications
- The dataset is intended for research and development of pharmacovigilance tools
- Auto-annotations are preliminary and require manual review before clinical use

Generated on 2026-05-28 by the Ghana ADR Pipeline.
