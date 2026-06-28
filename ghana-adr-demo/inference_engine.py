"""
inference_engine.py
===================
Inference logic for the Ghana ADR demo. Loads both CLF and NER models,
runs the post-processing rule layer, and merges hyphenated drug-name
fragments produced by NER tokenisation artefacts.

Fix: "Co" + "-trimoxazole" → "Co-trimoxazole" (and all gazetteer multi-part drugs)
"""

import json
import re
import torch
from pathlib import Path
from difflib import get_close_matches
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

# ── MODEL CONFIG ─────────────────────────────────────────────────────────────
_ROOT          = Path(__file__).parent.parent
CLF_MODEL_PATH = "reports/loso/models_phase2b_cohort_study/clf_best"
NER_MODEL_PATH = "reports/loso/models_phase7_cohort_study/ner_best"
BACKBONE       = "Phase7-Hybrid (cohort_study fold)"
DESCRIPTION    = (
    "PubMedBERT DAPT fine-tuned on Ghana ADR data; Phase 7 Hybrid: "
    "Phase 6 CLF + Phase 7 NER; macro LOSO CLF=0.724, NER=0.655; "
    "cohort_study fold: CLF=0.776, NER=0.785, threshold=0.55"
)

# Phase 6 cohort_study threshold (val sweep selected t=0.55)
CLF_THRESHOLD = 0.55

CLS_MODEL_DIR = _ROOT / CLF_MODEL_PATH
NER_MODEL_DIR = _ROOT / NER_MODEL_PATH
# ─────────────────────────────────────────────────────────────────────────────

NER_LABELS = [
    "O", "B-DRUG", "I-DRUG", "B-ADR", "I-ADR",
    "B-SEVERITY", "I-SEVERITY", "B-PATIENT_DEMO", "I-PATIENT_DEMO"
]
ID2LABEL = {i: l for i, l in enumerate(NER_LABELS)}

ENTITY_COLORS = {
    "DRUG":         "#22c55e",
    "ADR":          "#ef4444",
    "SEVERITY":     "#f97316",
    "PATIENT_DEMO": "#3b82f6",
}

ENTITY_BG = {
    "DRUG":         "#dcfce7",
    "ADR":          "#fee2e2",
    "SEVERITY":     "#ffedd5",
    "PATIENT_DEMO": "#dbeafe",
}

# ── 1A: SEVERITY keyword set ──
SEVERITY_TERMS = {
    "severe", "mild", "moderate", "serious", "fatal", "life-threatening",
    "mild-to-moderate", "moderate-to-severe", "grade 1", "grade 2",
    "grade 3", "grade 4", "severe adverse", "serious adverse", "lethal",
    "minor", "critical", "acute",
}

# ── 1B: Phantom entity blacklist (BERT subword artefacts) ──
MIN_ENTITY_LEN = 2
SUBWORD_BLACKLIST = {
    "laxis", "tion", "sis", "ine", "ase", "ing", "ment", "ness",
    "ity", "ate", "ous", "al", "ic", "er", "ed", "ly", "##",
}

# ── 1B2: Hospital / facility name guard ──
# Covers BERT tokeniser splits (e.g. "Korle" → ["Kor","##le"]) that make
# the model tag "le Bu" from "Korle Bu Teaching Hospital" as a DRUG entity.
_HOSPITAL_NAME_FRAGMENTS = frozenset({
    "korle bu", "korle", "le bu", "kath", "komfo anokye",
    "ridge hospital", "ridge", "tema general", "tema",
    "37 military", "military hospital", "cocoa clinic",
    "lekma", "pantang", "ho teaching", "cape coast teaching",
    "tamale teaching", "wa regional", "bolgatanga regional",
})
_HOSPITAL_CONTEXT_WORDS = (
    "teaching hospital", "hospital", "medical centre", "medical center",
    "clinic", "infirmary", "health centre", "health center",
    "polyclinic", "dispensary",
)
_HOSPITAL_CONTEXT_RADIUS = 60  # characters after entity end to scan


def _is_hospital_fragment(ent: dict, text: str) -> bool:
    """Return True if ent is a fragment of a hospital/facility name."""
    if ent["text"].strip().lower() in _HOSPITAL_NAME_FRAGMENTS:
        return True
    window = text[ent["end"]: ent["end"] + _HOSPITAL_CONTEXT_RADIUS].lower()
    return any(hw in window for hw in _HOSPITAL_CONTEXT_WORDS)


# ── 1C: Proximity relation guard ──
MAX_RELATION_DISTANCE = 150

# ── 1D: Confidence uncertain zone ──
CONF_UNCERTAIN_LOW  = 0.45
CONF_UNCERTAIN_HIGH = 0.55

# ── 1E: Ghanaian-English ADR synonym expansion ──
GHANAIAN_ADR_SYNONYMS = {
    "whole body was weak":      "weakness",
    "whole body weak":          "weakness",
    "body was weak":            "weakness",
    "body weakness":            "weakness",
    "testes swollen":           "testicular swelling",
    "testis swollen":           "testicular swelling",
    "noise in their ears":      "tinnitus",
    "noise in his ears":        "tinnitus",
    "noise in her ears":        "tinnitus",
    "ringing in the ears":      "tinnitus",
    "poor vision":              "visual impairment",
    "could not see clearly":    "visual impairment",
    "blurred vision":           "visual impairment",
    "body pains":               "myalgia",
    "body pain":                "myalgia",
    "joint pain":               "arthralgia",
    "loss of appetite":         "anorexia",
    "no appetite":              "anorexia",
    "skin rash":                "rash",
    "skin peeling":             "skin exfoliation",
    "yellow eyes":              "jaundice",
    "yellow skin":              "jaundice",
    "swollen legs":             "peripheral oedema",
    "swollen feet":             "peripheral oedema",
    "difficulty breathing":     "dyspnoea",
    "hard to breathe":          "dyspnoea",
    "could not breathe":        "dyspnoea",
    "chest pain":               "chest pain",
    "heart racing":             "tachycardia",
    "fast heartbeat":           "tachycardia",
}

# ── Bug 1: Negation patterns — checked BEFORE classifier ──
NEGATION_PATTERNS = [
    re.compile(r'\bnot\s+a\s+drug\s+side\s+effect\b', re.IGNORECASE),
    re.compile(r'\bnot\s+a\s+(drug\s+)?(side\s+effect|adverse\s+reaction|adr)\b', re.IGNORECASE),
    re.compile(r'\bcould\s+have\s+been\s+due\s+to\s+.{0,80}\bnot\s+a\b', re.IGNORECASE),
    re.compile(r'\bnot\s+(necessarily\s+)?due\s+to\s+the\s+(drug|medication|medicine|treatment)\b', re.IGNORECASE),
    re.compile(r'\b(infection|disease|condition)\b.{0,40}\bnot\s+a\s+(drug|medicine)', re.IGNORECASE),
    re.compile(r'\bexcluded?\s+.{0,30}(drug|medication|medicine)', re.IGNORECASE),
    re.compile(r'\bdetermined\s+to\s+be\s+unrelated\b', re.IGNORECASE),
    re.compile(r'\bunrelated\s+to\s+(?:the\s+)?(?:\w+\s+)?(?:drug|medication|medicine|treatment)', re.IGNORECASE),
    # Allow optional modifier (new/further/additional/significant) between "no" and
    # "adverse/side"; extended completion verbs include was/have been + noted/recorded.
    re.compile(r'\bno\s+(?:new\s+|further\s+|additional\s+|significant\s+)?(?:adverse|side)(?:\s+drug)?\s+(?:effect|reaction|event)s?\s+(?:(?:were|was|have\s+been)\s+)?(?:observed|reported|noted|found|detected|documented|recorded)\b', re.IGNORECASE),
    re.compile(r'\bno\s+toxicity\b', re.IGNORECASE),
    re.compile(r'\bno\s+adr\b', re.IGNORECASE),
    # Patient self-report: "denied any [symptoms]"
    re.compile(r'\bdenied\s+any\b', re.IGNORECASE),
    re.compile(r'\bdenied\s+(?:having|experiencing|suffering\s+from)\b', re.IGNORECASE),
    # Direct experience negation: "did not experience / develop / suffer…"
    re.compile(r'\bdid\s+not\s+(?:experience|develop|suffer|exhibit|report|display|show|complain)\b', re.IGNORECASE),
    re.compile(r'\b(?:has|have|had)\s+not\s+(?:experienced|developed|suffered|reported|exhibited|shown|displayed)\b', re.IGNORECASE),
    # "without (any / development of any) adverse reaction/effect"
    re.compile(r'\bwithout\s+(?:any\s+)?(?:adverse|side|untoward)\s+(?:drug\s+)?(?:reaction|effect|event)s?\b', re.IGNORECASE),
    re.compile(r'\bwithout\s+(?:(?:any|the)\s+)?(?:development|incidence|occurrence|report|evidence)\s+of\s+(?:any\s+)?(?:adverse|side)\s+(?:drug\s+)?(?:reaction|effect|event)s?\b', re.IGNORECASE),
    # "No adverse effects." / "No adverse reactions." without a completion verb
    re.compile(r'\bno\s+(?:new\s+|further\s+|additional\s+)?(?:adverse|side)\s+(?:drug\s+)?(?:effects?|reactions?|events?)\b', re.IGNORECASE),
    # "no side effects", "no complaints", "no symptoms" — routine follow-up negation
    re.compile(r'\bno\s+(?:further\s+)?(?:side\s+effects?|complaints?|symptoms?|adverse\s+events?)\b', re.IGNORECASE),
    # "no intervention required/needed/warranted" — mild/self-limiting event context
    re.compile(r'\bno\s+(?:further\s+)?intervention\s+(?:was\s+)?(?:required|necessary|needed|warranted)\b', re.IGNORECASE),
    # Clinical wound/examination negation: "no erythema", "no oedema", etc.
    re.compile(r'\bno\s+(?:erythema|oedema|edema|swelling|tenderness|discharge|purulence|redness|bruising|induration|inflammation|haemorrhage|hemorrhage|jaundice|icterus)\b', re.IGNORECASE),
    # Fix C: explicit wellness / no-problem frames (treatment success, non-ADR)
    re.compile(r'\bno\s+problems?\s+with\s+(?:the\s+)?(?:tablet|drug|medicine|medication|treatment)\b', re.IGNORECASE),
    re.compile(r'\bdiet\s+and\s+exercise\s+only\b', re.IGNORECASE),
    re.compile(r'\blifestyle\s+(?:changes?|modifications?)\s+only\b', re.IGNORECASE),
    re.compile(r'\bmanaging\s+(?:my|his|her|their)\s+\w+\s+with\s+diet\b', re.IGNORECASE),
    # Clinician confirms expected/normal pharmacological effect — not an ADR
    re.compile(r'\b(?:nurse|doctor|pharmacist|physician|clinician)\s+(?:say|said|told\s+(?:me|us|him|her))\s+(?:it\s+)?(?:is\s+|was\s+)?normal\b', re.IGNORECASE),
    # Batch-4 fixes
    re.compile(r'\bdenies\s+any\b', re.IGNORECASE),                                                                                                                                       # #61
    re.compile(r'\bdenied?\s+all\b', re.IGNORECASE),                                                                                                                                       # #63
    re.compile(r'\btolerating\s+\S+(?:\s+\S+)?\s+well\b', re.IGNORECASE),                                                                                                                 # #64
    re.compile(r'\bnot\s+related\s+to\s+(?:the\s+)?(?:drug|medicine|medication|treatment|tablet)\b', re.IGNORECASE),                                                                      # #65
    re.compile(r'\bnot\s+(?:from\s+)?(?:the\s+)?(?:drug|medicine|medication|treatment|tablet)\b', re.IGNORECASE),                                                                         # #81 #83
    re.compile(r'\b(?:nurse|doctor|pharmacist|physician|clinician)\s+(?:say|said|told\s+(?:me|us|him|her|them)).{0,100}\bfor\s+the\s+first\s+(?:few\s+)?(?:days?\b|weeks?\b)', re.IGNORECASE),  # #73
    re.compile(r'\b(?:nurse|doctor|pharmacist|physician|clinician)\s+say\b.{0,80}\balways\b', re.IGNORECASE),                                                                              # #86
    # Batch-5 fixes
    re.compile(r'\bvoiced?\s+no\s+(?:concerns?|complaints?)\b', re.IGNORECASE),                                                                                                           # #109
    re.compile(r'\bno\s+untoward\s+(?:effects?|reactions?|events?)\b', re.IGNORECASE),                                                                                                    # #111
    # NOTE: "well tolerated" intentionally excluded — it is a positive safety
    # assessment, not a negation frame; ADR entities must be preserved.
]

TOLERANCE_PHRASES = re.compile(
    r'\b(well[\s-]tolerated|generally\s+tolerated|adequately\s+tolerated|'
    r'good\s+tolerability|acceptable\s+tolerability|acceptable\s+safety\s+profile|'
    r'favourable\s+safety)\b',
    re.IGNORECASE,
)

# Bibliographic context: general drug safety statement, not a patient event report.
# Combined with TOLERANCE_PHRASES → skip classifier entirely.
_LITERATURE_CONTEXT_PATTERN = re.compile(
    r'\b(?:(?:well\s+)?documented|reported|described)\s+in\s+the\s+'
    r'(?:medical\s+|published\s+)?literature\b',
    re.IGNORECASE,
)

# Fix 1.2: "has not developed normally" in a fetal/pregnancy context describes an in-utero
# drug-induced abnormality — the "not" is part of the ADR signal, not a negation of ADR.
_FETAL_ABNORMALITY_GUARD = re.compile(
    r'(?:not\s+developed?\s+normally|scan\s+shows|ultrasound\s+(?:shows|found)|'
    r'baby\s+has\s+not|fetal\s+abnormali)',
    re.IGNORECASE
)
_FETAL_CONTEXT_KEYWORDS = frozenset(['baby', 'scan', 'fetal', 'foetal', 'pregnancy', 'weeks pregnant'])

# Fix 1.7: negation-then-pivot (#142, #147) — "no complaints/side effects" before a contrast
# conjunction ("but", "however") means the negation covers only the pre-drug or baseline period.
# The pivot then introduces the current ADR, so the negation rule must not suppress it.
_NEGATION_PIVOT_GUARD = re.compile(
    r'\bno\s+(?:side\s+effects?|complaints?|symptoms?)\b'
    r'.{0,70}'
    r'\b(?:but|however|though|although|yet|since\s+then|and\s+then)\b',
    re.IGNORECASE | re.DOTALL
)


def has_negation(text: str) -> bool:
    """Return True only for genuine negation frames; tolerance language is excluded."""
    if TOLERANCE_PHRASES.search(text):
        return False
    # Fix 1.2: fetal abnormality — "not developed normally" is the ADR, not a negation of ADR
    if _FETAL_ABNORMALITY_GUARD.search(text):
        text_lower = text.lower()
        if any(kw in text_lower for kw in _FETAL_CONTEXT_KEYWORDS):
            return False
    # Fix 1.7: "no complaints/side effects" before a contrast pivot — the negation applies
    # to the pre-drug baseline period, not the current state; the pivot introduces the ADR
    if _NEGATION_PIVOT_GUARD.search(text):
        return False
    for pattern in NEGATION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── Fix A: Resolution / treatment-success suppressor ──
# Catches sentences where an event is described as having resolved or never occurred.
_RESOLUTION_PATTERNS = [
    re.compile(r'\b(?:completely|totally|fully|entirely)\s+(?:gone|resolved|cleared|settled|disappeared)\b', re.IGNORECASE),
    re.compile(r'\b(?:fever|symptoms?|pain|swelling|rash|cough|upset)\b.{0,40}\b(?:gone|resolved|cleared|settled|subsided|disappeared)\b', re.IGNORECASE),
    re.compile(r'\bcleared\s+up\b', re.IGNORECASE),
    re.compile(r'\bsettled\s+(?:after|within|by|down|$|[.!,])\b', re.IGNORECASE),
    re.compile(r'\bit\s+(?:has\s+)?settled\b', re.IGNORECASE),
    re.compile(r'\bfeeling\s+(?:much\s+)?better\s+now\b', re.IGNORECASE),
    re.compile(r'\bno\s+issues?\s+at\s+all\b', re.IGNORECASE),
    re.compile(r'\bdoing\s+(?:very\s+)?(?:well|fine|good)\b', re.IGNORECASE),
    re.compile(r'\bgoing\s+well\b', re.IGNORECASE),
    re.compile(r'\bfinished\s+the\s+(?:full\s+)?course\b', re.IGNORECASE),
    # Batch-3 fixes
    re.compile(r'\bno\s+problems?\s+at\s+all\b', re.IGNORECASE),
    re.compile(r'\bfinished\s+(?:my|the)\b.{0,25}\btreatment\b', re.IGNORECASE),
    re.compile(r'\bno\s+problem\b', re.IGNORECASE),
    re.compile(r'\b(?:fever|temperature)\b.{0,30}\bcome\s+down\b', re.IGNORECASE),
    re.compile(r'\bgrowing\s+(?:well|normally)\b', re.IGNORECASE),
    # Batch-4 fixes
    re.compile(r'\bam\s+fine\s+now\b', re.IGNORECASE),                                                              # #66
    re.compile(r'\b(?:side\s+effect|problem|reaction)\s+don\s+(?:pass|go|finish)\b', re.IGNORECASE),                # #67 Pidgin
    re.compile(r'\bwith\s+no\s+issues?\b', re.IGNORECASE),                                                          # #69
    # Phase 7 Fix 1.4 — extended recovery / clinician-reassurance language
    re.compile(r'\bit\s+(?:went|passed|cleared|settled)\s+(?:away|on\s+its\s+own|quickly|within|in)\b', re.IGNORECASE),
    re.compile(r'\bfine\s+(?:the\s+next\s+(?:morning|day)|by\s+the\s+(?:second|third|next)\s+day)\b', re.IGNORECASE),
    re.compile(r'\bwas\s+fine\s+by\s+the\b', re.IGNORECASE),
    re.compile(r'\bfelt\s+fine\b', re.IGNORECASE),
    re.compile(r'\bnothing\s+serious\b', re.IGNORECASE),
    re.compile(r'\b(?:midwife|nurse|doctor|health\s+worker)\s+(?:told|said|confirmed|explained)\s+(?:me\s+)?it\s+(?:was|is)\s+(?:normal|expected|fine)\b', re.IGNORECASE),
    re.compile(r'\btook\s+paracetamol\s+and\s+(?:was\s+fine|recovered|felt\s+better)\b', re.IGNORECASE),
    re.compile(r'\ball\s+(?:checks?|tests?|readings?)\s+(?:are|were)\s+normal\b', re.IGNORECASE),
    re.compile(r'\bno\s+(?:fever|vomiting|diarrhoea|rash|swelling|unusual\s+crying)\b', re.IGNORECASE),
]

# Guard: explicit ADR-onset or serious-harm language means the event DID occur (even if later resolved).
_ACTIVE_HARM_GUARD = re.compile(
    r'\b(?:experienced|developed|suffered|presenting\s+with|presented\s+with|'
    r'complained\s+of|caused\b|induced\b|triggered\b|'
    r'requiring\s+(?:immediate|urgent|medical)|required\s+(?:immediate|urgent|medical)|'
    r'hospitalised|hospitaliz(?:ed|ing)|emergency|could\s+not\s+breathe|collapsed|seizure|'
    r'anaphylaxis|respiratory\s+depression|cardiac\s+arrest|ICU|intensive\s+care)\b',
    re.IGNORECASE,
)


# Fix 1.3: "appetite/hearing/vision is completely gone" describes an ongoing symptom (anorexia,
# ototoxicity, visual impairment), not a resolved event — "gone" here means ABSENT, not RESOLVED.
_SYMPTOM_GONE_GUARD = re.compile(
    r'\b(?:appetite|energy|strength|feeling|sensation|hearing|vision|sight)\b'
    r'.{0,15}\bgone\b',
    re.IGNORECASE | re.DOTALL
)

# Fix 1.8: "took/on [drug] for [period] with no problem" is a historical tolerance clause (#132):
# the patient was once tolerating the drug but a new ADR has since developed. "no problem"
# refers to the past tolerance window, not a resolution of the current adverse event.
_HISTORICAL_TOLERANCE_GUARD = re.compile(
    r'\b(?:took|on|taking|tolerated?|used?)\b.{0,60}\bwith\s+no\s+problem\b',
    re.IGNORECASE | re.DOTALL
)

# Fix 1.1: positive dechallenge — cessation of drug followed by symptom resolution confirms ADR
# causation and must not be suppressed by the resolution rule.
_POSITIVE_DECHALLENGE = re.compile(
    r'(?:stopped|discontinued|ceased|stopped\s+taking)\s+(?:the\s+)?(?:[\w,]+\s+){0,6}'
    r'(?:and|then|after\s+which)\s+(?:the\s+)?(?:[\w,]+\s+){0,8}'
    r'(?:disappeared|resolved|cleared|gone|improved)\b',
    re.IGNORECASE | re.DOTALL
)


def has_resolution_language(text: str) -> bool:
    """Return True if the text describes a resolved or successful outcome with no ADR onset."""
    if _ACTIVE_HARM_GUARD.search(text):
        return False
    # Fix 1.3: symptom "gone" in the sense of "absent since starting drug" is an ongoing ADR
    if _SYMPTOM_GONE_GUARD.search(text) and 'since' in text.lower():
        return False
    # Fix 1.8: historical tolerance clause — "took [drug] for [period] with no problem"
    # means the patient once tolerated the drug; "no problem" is not a resolution of a current ADR
    if _HISTORICAL_TOLERANCE_GUARD.search(text):
        return False
    return any(p.search(text) for p in _RESOLUTION_PATTERNS)


# ── Fix B: Indication-frame ("drug for symptom") suppressor ──
# Detects "took/prescribed X for [condition]" where the condition is the reason for the
# drug, not a reaction to it. A causal-pivot guard prevents suppressing compound sentences
# like "took X for Y but then developed Z".
_INDICATION_FOR_PATTERN = re.compile(
    r'\b(?:take|took|taken|prescribed|given|use|used|using|started|started\s+on|'
    r'initiated|administered|recommended|dispensed)\b'
    r'.{0,30}'
    r'\bfor\b'
    r'.{0,50}'
    r'\b(?:malaria|fever|pain|headache|infection|cough|diarrhoea|diarrhea|nausea|'
    r'vomiting|inflammation|hypertension|diabetes|epilepsy|HIV|tuberculosis|TB|'
    r'anaemia|anemia|anxiety|depression|arthritis|ulcer|pneumonia|prophylaxis)\b',
    re.IGNORECASE | re.DOTALL,
)

_INDICATION_CAUSAL_GUARD = re.compile(
    r'(?:'
    r'\b(?:but|however|after|then|subsequently|later|and\s+then)\b.{0,80}'
    r'\b(?:react|experience|develop|suffer|cause|trigger|start|begin|report)\b'
    r'|'
    r'\b(?:dropped|declined|fell|decreased|elevated|worsened)\s+from\s+\d'
    r')',
    re.IGNORECASE | re.DOTALL,
)


def is_indication_frame(text: str) -> bool:
    """Return True if the drug is described as taken FOR a condition (indication, not reaction)."""
    if not _INDICATION_FOR_PATTERN.search(text):
        return False
    return not _INDICATION_CAUSAL_GUARD.search(text)


# ── FIX 1: Absence-of-evidence patterns ──
# Nominal-bridge negation: "no [N] of [ADR]" — not caught by token-window negation.
ABSENCE_OF_EVIDENCE_PATTERNS = [
    re.compile(r'\bno\s+\w*\s*(?:documented\s+)?cases?\s+of\b', re.IGNORECASE),
    re.compile(r'\bno\s+reported\s+(?:cases?\s+of\s+)?\b', re.IGNORECASE),
    re.compile(r'\bnot\s+been\s+reported\b', re.IGNORECASE),
    re.compile(r'\bno\s+(?:published\s+|available\s+)?evidence\s+of\b', re.IGNORECASE),
    re.compile(r'\bno\s+(?:causal\s+|known\s+)?association\b', re.IGNORECASE),
    re.compile(r'\babsence\s+of\b', re.IGNORECASE),
    re.compile(r'\bhas\s+not\s+been\s+(?:documented|observed|identified|established)\b', re.IGNORECASE),
    re.compile(r'\bnot\s+associated\s+with\b', re.IGNORECASE),
    re.compile(r'\bno\s+\w+\s+(?:were|was|have\s+been)\s+(?:identified|found|detected|observed)\b', re.IGNORECASE),
]


def is_absence_of_evidence(text: str) -> bool:
    """Return True for population-level absence assertions (nominal-bridge negation)."""
    for pattern in ABSENCE_OF_EVIDENCE_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ── FIX 3A: Concessive ADR patterns ──
# Sentences with positive safety framing followed by explicit ADR acknowledgement.
_CONCESSIVE_ADR_RE = re.compile(
    r'(?:generally\s+safe|well[\s-]tolerated|considered\s+safe|regarded\s+as\s+safe)'
    r'.{0,80}'
    r'(?:though|however|but|despite|although|nevertheless|nonetheless|still|yet)'
    r'.{0,60}'
    r'(?:adr|adverse\s+(?:drug\s+)?reactions?|side\s+effects?|adverse\s+events?)',
    re.IGNORECASE | re.DOTALL,
)


def check_concessive_adr(text: str) -> bool:
    """Return True for sentences acknowledging ADRs despite a positive safety framing."""
    return bool(_CONCESSIVE_ADR_RE.search(text))


# ── Bug 2: Drug intolerance + epidemiological pattern overrides ──
INTOLERANCE_PATTERNS = [
    re.compile(r"could not tolerate", re.IGNORECASE),
    re.compile(r"intolerant to", re.IGNORECASE),
    re.compile(r"discontinued\s+due\s+to", re.IGNORECASE),
    re.compile(r"discontinued.{0,40}due\s+to.{0,20}(side\s+effects?|adverse|reaction|toxicity)", re.IGNORECASE),
    re.compile(r"discontinuation\s+due\s+to", re.IGNORECASE),
    re.compile(r"withdrawn\s+(due\s+to|because\s+of)", re.IGNORECASE),
    re.compile(r"stopped\s+(due\s+to|because\s+of).{0,30}(side\s+effect|adverse|toxicity|reaction)", re.IGNORECASE),
    re.compile(r"withheld\s+due\s+to", re.IGNORECASE),
    re.compile(r"switched\s+(from|off).{0,20}due\s+to", re.IGNORECASE),
    re.compile(r"treatment\s+was\s+(stopped|halted|suspended)\s+(due\s+to|because\s+of)", re.IGNORECASE),
    re.compile(r"(significantly|more)\s+(likely|common)\s+to\s+develop.{0,60}(while|on|with)", re.IGNORECASE),
    re.compile(r"side\s+effects?\s+(of|include|reported\s+for|associated\s+with)", re.IGNORECASE),
    re.compile(r"adverse\s+(event|reaction|effect)s?\s+(reported|observed|noted|found)", re.IGNORECASE),
    re.compile(r"(reported|experienced|developed|suffered).{0,60}(after|following|while\s+on|due\s+to)", re.IGNORECASE),
]

# ── FIX 2: Lightweight entity-presence gate for pattern overrides ──
# Pattern overrides must not fire when no named drug or specific ADR is present.
# Intentionally excludes generic phrases like "drug", "side effects", "adverse reactions"
# since those are exactly what the intolerance patterns already match.
_DRUG_TERMS_PATTERN = re.compile(
    r'\b(?:chloroquine|artemether|artesunate|amodiaquine|lumefantrine|'
    r'zidovudine|co-trimoxazole|cotrimoxazole|trimoxazole|spaq|'
    r'hydroxychloroquine|sulphadoxine|pyrimethamine|efavirenz|nevirapine|'
    r'lamivudine|tenofovir|abacavir|doxycycline|rifampicin|isoniazid|'
    r'pyrazinamide|ethambutol|quinine|mefloquine|coartem|asaq|fansidar|'
    r'dihydroartemisinin|piperaquine|azithromycin|amoxicillin|penicillin|'
    r'metformin|aspirin|ibuprofen|paracetamol|acetaminophen)\b',
    re.IGNORECASE,
)
_ADR_TERMS_PATTERN = re.compile(
    r'\b(?:fever|vomiting|nausea|headache|rash|diarrhoea|diarrhea|anaemia|anemia|'
    r'jaundice|convulsions?|anaphylaxis|tinnitus|myalgia|arthralgia|oedema|edema|'
    r'hepatotoxicity|nephrotoxicity|thrombocytopenia|neutropenia|hepatitis|'
    r'pancreatitis|neuropathy|cardiomyopathy|bradycardia|tachycardia|'
    r'hypotension|hypertension|pruritus|urticaria|dyspnoea|dyspnea|'
    r'fatigue|weakness|dizziness|vertigo|abdominal\s+pain|'
    r'gastrointestinal\s+(?:discomfort|symptoms?|upset))\b',
    re.IGNORECASE,
)

# ── Phase 2: MedDRA lookup (inline — no separate file needed) ──
MEDDRA_LOOKUP = {
    "dizziness":            "Dizziness",
    "headache":             "Headache",
    "anaemia":              "Anaemia",
    "anemia":               "Anaemia",
    "nausea":               "Nausea",
    "vomiting":             "Vomiting",
    "diarrhoea":            "Diarrhoea",
    "diarrhea":             "Diarrhoea",
    "fatigue":              "Fatigue",
    "weakness":             "Asthenia",
    "body weakness":        "Asthenia",
    "oculogyric crisis":    "Oculogyric crisis",
    "poor vision":          "Vision blurred",
    "visual impairment":    "Vision blurred",
    "tinnitus":             "Tinnitus",
    "myalgia":              "Myalgia",
    "body pain":            "Myalgia",
    "body pains":           "Myalgia",
    "abdominal pain":       "Abdominal pain",
    "restlessness":         "Agitation",
    "fever":                "Pyrexia",
    "loss of appetite":     "Decreased appetite",
    "anorexia":             "Decreased appetite",
    "dyspnoea":             "Dyspnoea",
    "jaundice":             "Jaundice",
    "rash":                 "Rash",
    "skin rash":            "Rash",
    "peripheral oedema":    "Oedema peripheral",
    "tachycardia":          "Tachycardia",
    "arthralgia":           "Arthralgia",
    "joint pain":           "Arthralgia",
    "testicular swelling":  "Testicular swelling",
    "chest pain":           "Chest pain",
    "cardiac arrest":       "Cardiac arrest",
    "heart failure":        "Cardiac failure",
    "respiratory failure":  "Respiratory failure",
    "renal failure":        "Renal failure",
    "liver failure":        "Hepatic failure",
    "hepatic failure":      "Hepatic failure",
}

# ── 1F: Gazetteer-anchored hyphenated/slash drug merging ──
# Built once at import time from drugs.json.
# Only forms explicitly in the gazetteer are eligible for merging — prevents
# false extension into constructions like "rifampicin-induced" or
# "metformin-associated" that are pervasive in pharmacovigilance text.

def _load_hyphenated_drug_forms():
    """Return (forms_set, prefixes_set) from drugs.json for multi-part names."""
    gaz_path = _ROOT / "data" / "gazetteers" / "drugs.json"
    try:
        with open(gaz_path, encoding="utf-8") as fh:
            drugs = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return set(), set()

    forms    = set()
    prefixes = set()
    for entry in drugs:
        all_names = (
            [entry.get("canonical", "")]
            + entry.get("synonyms", [])
            + entry.get("lay_terms", [])
        )
        for name in all_names:
            name_l = name.lower().strip()
            if "-" not in name_l and "/" not in name_l:
                continue
            forms.add(name_l)
            # Prefix = part before the first connector
            sep = min(
                name_l.index("-") if "-" in name_l else len(name_l),
                name_l.index("/") if "/" in name_l else len(name_l),
            )
            prefix = name_l[:sep].strip()
            if len(prefix) >= 2:
                prefixes.add(prefix)
    return forms, prefixes


HYPHENATED_DRUG_FORMS, HYPHENATED_DRUG_PREFIXES = _load_hyphenated_drug_forms()


def merge_hyphenated_drugs(text: str, entities: list) -> list:
    """
    Post-processing: merge DRUG entity fragments split at a hyphen/slash connector.

    The NER tokeniser splits "Co-trimoxazole" into subword tokens, so the model
    may emit B-DRUG for "Co" and leave "-trimoxazole" unlabelled.  This function
    detects that pattern and re-joins the fragments.

    Merge is gated on the gazetteer: the completed span must match a known
    multi-part drug form (case-insensitive), which prevents false extension into
    constructions like "rifampicin-induced" or "metformin-associated".

    Handles:
      Co-trimoxazole, artemether-lumefantrine, dihydroartemisinin-piperaquine,
      lopinavir/ritonavir, sulfadoxine-pyrimethamine, and every other gazetteer
      entry whose canonical/synonym/lay_term contains '-' or '/'.
    """
    if not entities or not HYPHENATED_DRUG_PREFIXES:
        return entities

    result = []
    skip   = set()

    for i, ent in enumerate(entities):
        if i in skip:
            continue
        if ent["label"] != "DRUG":
            result.append(ent)
            continue

        ent = dict(ent)

        # Strip any trailing connector the model accidentally included in the span
        prefix = ent["text"].rstrip("-/").lower()
        if prefix not in HYPHENATED_DRUG_PREFIXES:
            result.append(ent)
            continue

        end = ent["end"]

        # Locate the connector character
        if ent["text"].endswith(("-", "/")):
            # Model included '-' at the tail of the entity span
            suffix_start = end
        elif end < len(text) and text[end] in ("-", "/"):
            # Connector is the first character after the entity span
            suffix_start = end + 1
        else:
            result.append(ent)
            continue

        # Nothing useful after the connector
        if suffix_start >= len(text) or text[suffix_start].isspace():
            result.append(ent)
            continue

        # Read the suffix word (stop at whitespace or sentence-boundary punctuation)
        word_end = suffix_start
        while word_end < len(text) and not text[word_end].isspace() \
              and text[word_end] not in (",", ";", "(", ")", "[", "]", "."):
            word_end += 1

        if word_end == suffix_start:
            result.append(ent)
            continue

        candidate = text[ent["start"]:word_end].lower()

        # Gate: only merge if the candidate is a known gazetteer form
        if candidate not in HYPHENATED_DRUG_FORMS:
            result.append(ent)
            continue

        # If the suffix overlaps a separately-labelled DRUG entity, absorb it
        if i + 1 < len(entities):
            nxt = entities[i + 1]
            if nxt["label"] == "DRUG" and suffix_start <= nxt["start"] < word_end:
                word_end = max(word_end, nxt["end"])
                skip.add(i + 1)

        ent = dict(ent, text=text[ent["start"]:word_end], end=word_end)
        result.append(ent)

    return result


# ── 1G: ADR gazetteer fallback (multi-word clinical terms) ──
# Loaded from adrs.json at import time.  Only multi-word terms are used so that
# short ambiguous words ("rash", "fever") aren't injected outside NER context.
# This makes adding terms to adrs.json immediately effective at inference time.

def _load_adr_gazetteer():
    """Return sorted list of (term_lower, meddra_pt) for multi-word ADR terms."""
    gaz_path = _ROOT / "data" / "gazetteers" / "adrs.json"
    try:
        with open(gaz_path, encoding="utf-8") as fh:
            adrs = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    terms = []
    for entry in adrs:
        term = entry.get("term", "").strip()
        if len(term.split()) >= 2:          # multi-word only
            terms.append((term.lower(), entry.get("meddra_pt")))
    # Longest match first to prevent partial overlaps
    terms.sort(key=lambda x: -len(x[0]))
    return terms


ADR_GAZETTEER_TERMS = _load_adr_gazetteer()


def apply_adr_gazetteer(text, entities):
    """
    1G: Inject ADR entities for multi-word gazetteer terms the NER missed.

    Runs after the BIO-corrected NER output and Ghanaian synonym expansion,
    so it only fills gaps.  Restricted to multi-word terms (2+ words) to keep
    precision high — single-word terms like 'rash' or 'fever' are too ambiguous
    to match without the NER's context window.
    """
    if not ADR_GAZETTEER_TERMS:
        return entities

    covered = set()
    for ent in entities:
        for i in range(ent["start"], ent["end"]):
            covered.add(i)

    text_lower = text.lower()
    extras = []
    for term, meddra_pt in ADR_GAZETTEER_TERMS:
        idx = 0
        while True:
            pos = text_lower.find(term, idx)
            if pos == -1:
                break
            end = pos + len(term)
            if pos not in covered:
                extras.append({
                    "text":  text[pos:end],
                    "label": "ADR",
                    "start": pos,
                    "end":   end,
                })
                for i in range(pos, end):
                    covered.add(i)
            idx = end

    combined = entities + extras
    combined.sort(key=lambda e: e["start"])
    return combined


# Fix 1.5: third-party voice — "my friend/colleague/classmate had X" is not the narrator's ADR.
# Only suppresses if the first-person narrator is NOT also reporting their own reaction.
_THIRD_PARTY_EXTENDED = re.compile(
    r'\bmy\s+(?:friend|colleague|co-worker|classmate|roommate|uncle|aunt|cousin)\s+'
    r'(?:had|experienced|developed|suffered|got|has\s+had)\s+',
    re.IGNORECASE
)


def first_person_also_affected(text: str) -> bool:
    """Returns True if the narrator is also reporting their own reaction."""
    return bool(re.search(r'\bI\s+(?:also|too|myself|have\s+also)\b', text, re.IGNORECASE))


# Fix 1.6: aggregate/population-level document headers — these describe known ADR populations
# in periodic safety reports, not individual patient events.
_AGGREGATE_DOC_HEADERS = re.compile(
    r'(?:periodic\s+safety\s+update|PSUR|national\s+AEFI\s+surveillance\s+summary|'
    r'signal\s+assessment\s+(?:memo|report)|reporting\s+(?:quarter|period)\s*:\s*Q\d|'
    r'cumulative\s+(?:serious\s+)?adverse\s+events\s+reported)',
    re.IGNORECASE
)


def load_models():
    """Load both models. Call this once at startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_tok   = AutoTokenizer.from_pretrained(str(CLS_MODEL_DIR))
    cls_model = AutoModelForSequenceClassification.from_pretrained(str(CLS_MODEL_DIR))
    cls_model.to(device).eval()

    # NER checkpoint omitted tokenizer.save_pretrained(); use CLF tokenizer (same DAPT backbone)
    ner_tok   = AutoTokenizer.from_pretrained(str(CLS_MODEL_DIR))
    ner_model = AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_DIR))
    ner_model.to(device).eval()

    return cls_tok, cls_model, ner_tok, ner_model, device


def classify_sentence(text, tokenizer, model, device):
    """Run binary classifier. Returns result with optional uncertain flag."""
    enc = tokenizer(
        text, truncation=True, max_length=256,
        padding=True, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1).squeeze()

    prob_adr = round(probs[1].item(), 4)
    label    = int(prob_adr >= CLF_THRESHOLD)

    uncertain = CONF_UNCERTAIN_LOW <= prob_adr <= CONF_UNCERTAIN_HIGH

    rule_triggered = False
    # FIX 2: only fire pattern override when a named drug or specific ADR term is
    # present — prevents pronoun references and meta-language from triggering overrides.
    has_named_entity = bool(
        _DRUG_TERMS_PATTERN.search(text) or _ADR_TERMS_PATTERN.search(text)
    )
    if has_named_entity and not TOLERANCE_PHRASES.search(text):
        for pattern in INTOLERANCE_PATTERNS:
            if pattern.search(text):
                label = 1
                rule_triggered = True
                break

    return {
        "contains_adr":   bool(label),
        "confidence":     round(probs[label].item(), 4),
        "prob_no_adr":    round(probs[0].item(), 4),
        "prob_adr":       prob_adr,
        "uncertain":      uncertain,
        "rule_triggered": rule_triggered,
    }


def extract_entities(text, tokenizer, model, device):
    """Run NER model and return entity spans. Filters phantom artefacts (1B)."""
    enc = tokenizer(
        text, truncation=True, max_length=256,
        return_offsets_mapping=True, return_tensors="pt"
    )
    offsets = enc["offset_mapping"].squeeze().tolist()
    enc_no_offset = {k: v.to(device) for k, v in enc.items()
                     if k != "offset_mapping"}

    with torch.no_grad():
        logits = model(**enc_no_offset).logits
        preds  = torch.argmax(logits, dim=2).squeeze().tolist()

    entities = []
    current  = None
    for pred_id, (cs, ce) in zip(preds, offsets):
        if cs == ce:
            continue
        label = ID2LABEL.get(pred_id, "O")
        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"text": text[cs:ce], "label": label[2:], "start": cs, "end": ce}
        elif label.startswith("I-") and current and label[2:] == current["label"]:
            current["text"] = text[current["start"]:ce]
            current["end"]  = ce
        elif label.startswith("I-"):
            # BIO violation: orphan I- (or type mismatch, e.g. I-ADR after B-SEVERITY).
            # Treat as an implicit B- so fragments like "cardiac arrest" aren't dropped
            # when the model emits B-SEVERITY on "fatal" then I-ADR on "cardiac"/"arrest".
            if current:
                entities.append(current)
            current = {"text": text[cs:ce], "label": label[2:], "start": cs, "end": ce}
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)

    # 1B: filter phantom entities
    filtered = []
    for ent in entities:
        t = ent["text"].strip()
        if len(t) < MIN_ENTITY_LEN:
            continue
        if t.lower() in SUBWORD_BLACKLIST:
            continue
        if re.fullmatch(r"[\d\s\W]+", t):
            continue
        # 1B2: drop DRUG spans that are hospital/facility name fragments
        if ent["label"] == "DRUG" and _is_hospital_fragment(ent, text):
            continue
        filtered.append(ent)

    return filtered


# ── FIX 4: Adjective-only ADR entity filter ──
_ADJ_ONLY_BLACKLIST = frozenset({
    "persistent", "chronic", "recurrent", "intermittent", "acute", "sudden",
    "mild", "moderate", "severe", "serious", "fatal", "life-threatening",
    "occasional", "frequent", "rare", "common", "associated", "related",
    "prolonged", "transient", "temporary", "permanent",
    "allergic",  # "for allergic reaction" — adjective describing indication, not ADR
})


def filter_adr_entities(entities: list) -> list:
    """Drop ADR entities whose span is entirely adjective modifiers (no nominal head)."""
    filtered = []
    for ent in entities:
        if ent["label"] != "ADR":
            filtered.append(ent)
            continue
        tokens = ent["text"].strip().lower().split()
        if tokens and all(t in _ADJ_ONLY_BLACKLIST for t in tokens):
            continue
        filtered.append(ent)
    return filtered


def apply_severity_rules(text, entities):
    """1A: Re-label known severity terms as SEVERITY; detect missed ones."""
    severity_lower = SEVERITY_TERMS
    result  = []
    covered = set()

    for ent in entities:
        if ent["text"].lower().strip() in severity_lower:
            ent = dict(ent, label="SEVERITY")
        result.append(ent)
        for i in range(ent["start"], ent["end"]):
            covered.add(i)

    text_lower = text.lower()
    for term in sorted(severity_lower, key=len, reverse=True):
        idx = 0
        while True:
            pos = text_lower.find(term, idx)
            if pos == -1:
                break
            end = pos + len(term)
            if pos not in covered:
                result.append({
                    "text":  text[pos:end],
                    "label": "SEVERITY",
                    "start": pos,
                    "end":   end,
                })
                for i in range(pos, end):
                    covered.add(i)
            idx = end

    result.sort(key=lambda e: e["start"])
    return result


def apply_ghanaian_synonyms(text, entities):
    """1E: Inject ADR entities for Ghanaian-English expressions the NER missed."""
    covered = set()
    for ent in entities:
        for i in range(ent["start"], ent["end"]):
            covered.add(i)

    text_lower = text.lower()
    extras = []
    for phrase, canonical in sorted(GHANAIAN_ADR_SYNONYMS.items(), key=lambda x: -len(x[0])):
        idx = 0
        while True:
            pos = text_lower.find(phrase, idx)
            if pos == -1:
                break
            end = pos + len(phrase)
            if pos not in covered:
                extras.append({
                    "text":      text[pos:end],
                    "label":     "ADR",
                    "start":     pos,
                    "end":       end,
                    "canonical": canonical,
                })
                for i in range(pos, end):
                    covered.add(i)
            idx = end

    combined = entities + extras
    combined.sort(key=lambda e: e["start"])
    return combined


def normalise_entities(entities):
    """Phase 2: Add meddra_pt field to each ADR entity via lookup + fuzzy match."""
    all_keys = list(MEDDRA_LOOKUP.keys())
    for ent in entities:
        if ent["label"] not in ("ADR", "SEVERITY"):
            continue
        raw = ent.get("canonical", ent["text"]).lower().strip()
        pt  = MEDDRA_LOOKUP.get(raw)
        if pt is None:
            matches = get_close_matches(raw, all_keys, n=1, cutoff=0.82)
            if matches:
                pt = MEDDRA_LOOKUP[matches[0]]
        ent["meddra_pt"] = pt
    return entities


_EXCLUSION_PHRASES = (
    "not associated with", "no association", "not caused by",
    "unrelated to", "independent of", "not attributed to",
)


def extract_relations(entities, text: str = ""):
    """1C: Proximity-constrained DRUG→ADR relation extraction (1:N).

    FIX 5: accepts text to skip pairs where explicit exclusion language
    appears between the two entity spans.
    """
    drugs = [e for e in entities if e["label"] == "DRUG"]
    adrs  = [e for e in entities if e["label"] == "ADR"]
    relations = []
    text_lower = text.lower() if text else ""
    for d in drugs:
        for a in adrs:
            dist = abs(d["start"] - a["start"])
            if dist > MAX_RELATION_DISTANCE:
                continue
            if text_lower:
                lo = min(d["end"], a["end"])
                hi = max(d["start"], a["start"])
                between = text_lower[lo:hi] if lo < hi else ""
                if any(phrase in between for phrase in _EXCLUSION_PHRASES):
                    continue
            relations.append({
                "drug":     d["text"],
                "adr":      a["text"],
                "type":     "CAUSES",
                "distance": dist,
            })
    relations.sort(key=lambda r: r["distance"])
    return relations


def highlight_text(text, entities):
    """Return HTML with entity spans highlighted."""
    if not entities:
        return f"<span style='line-height:1.8'>{_escape(text)}</span>"

    sorted_ents = sorted(entities, key=lambda e: e["start"])
    result = []
    cursor = 0

    for ent in sorted_ents:
        start, end = ent["start"], ent["end"]
        if start < cursor:
            continue
        result.append(_escape(text[cursor:start]))
        label  = ent["label"]
        color  = ENTITY_COLORS.get(label, "#888")
        bg     = ENTITY_BG.get(label, "#f0f0f0")
        pt_tag = ""
        if ent.get("meddra_pt"):
            pt_tag = (
                f'<span style="font-size:8px;opacity:0.7;margin-left:3px">'
                f'[{ent["meddra_pt"]}]</span>'
            )
        result.append(
            f'<mark style="background:{bg};color:{color};border-radius:4px;'
            f'padding:1px 4px;margin:0 1px;font-weight:600;border:1px solid {color}44">'
            f'{_escape(ent["text"])}'
            f'<sup style="font-size:9px;margin-left:2px;opacity:0.8">{label}</sup>'
            f'{pt_tag}'
            f'</mark>'
        )
        cursor = end

    result.append(_escape(text[cursor:]))
    return f"<span style='line-height:2'>{''.join(result)}</span>"


def split_sentences(text):
    """Split paragraph into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def analyse_text(text, cls_tok, cls_mod, ner_tok, ner_mod, device):
    """Full pipeline for a single sentence."""
    # FIX 1: absence-of-evidence — highest priority; entities are negated, discard all
    if is_absence_of_evidence(text):
        return {
            "text":              text,
            "contains_adr":      False,
            "confidence":        1.0,
            "prob_no_adr":       1.0,
            "prob_adr":          0.0,
            "uncertain":         False,
            "rule_triggered":    False,
            "negation_override": True,
            "entities":          [],
            "relations":         [],
            "highlighted_html":  f"<span style='line-height:1.8'>{_escape(text)}</span>",
        }

    # Fix 1.6: aggregate/PSUR document header — population-level report, not a patient event
    if _AGGREGATE_DOC_HEADERS.search(text):
        return {
            "text":              text,
            "contains_adr":      False,
            "confidence":        0.15,
            "prob_no_adr":       0.85,
            "prob_adr":          0.15,
            "uncertain":         False,
            "rule_triggered":    False,
            "negation_override": True,
            "entities":          [],
            "relations":         [],
            "highlighted_html":  f"<span style='line-height:1.8'>{_escape(text)}</span>",
        }

    # Tolerance phrase + bibliographic context: general safety statement, not a patient event.
    # "acceptable tolerability … well documented in the literature" must not fire as ADR.
    if TOLERANCE_PHRASES.search(text) and _LITERATURE_CONTEXT_PATTERN.search(text):
        ents = extract_entities(text, ner_tok, ner_mod, device)
        ents = merge_hyphenated_drugs(text, ents)
        ents = [e for e in ents if e["label"] == "DRUG"]
        ents = normalise_entities(ents)
        html = highlight_text(text, ents)
        return {
            "text":              text,
            "contains_adr":      False,
            "confidence":        1.0,
            "prob_no_adr":       1.0,
            "prob_adr":          0.0,
            "uncertain":         False,
            "rule_triggered":    False,
            "negation_override": True,
            "entities":          ents,
            "relations":         [],
            "highlighted_html":  html,
        }

    # FIX 3A: concessive ADR — positive safety frame + explicit ADR acknowledgement
    if check_concessive_adr(text):
        ents = extract_entities(text, ner_tok, ner_mod, device)
        ents = filter_adr_entities(ents)                   # FIX 4
        ents = merge_hyphenated_drugs(text, ents)          # 1F
        ents = apply_severity_rules(text, ents)
        ents = apply_ghanaian_synonyms(text, ents)
        ents = apply_adr_gazetteer(text, ents)             # 1G
        ents = normalise_entities(ents)
        rels = extract_relations(ents, text)               # FIX 5
        html = highlight_text(text, ents)
        return {"text": text, "contains_adr": True, "confidence": 1.0,
                "prob_no_adr": 0.0, "prob_adr": 1.0, "uncertain": False,
                "rule_triggered": True, "negation_override": False,
                "entities": ents, "relations": rels, "highlighted_html": html}

    # Bug 1: negation pre-check — runs before classifier
    if has_negation(text):
        ents = extract_entities(text, ner_tok, ner_mod, device)
        ents = filter_adr_entities(ents)                   # FIX 4
        ents = merge_hyphenated_drugs(text, ents)          # 1F
        ents = apply_severity_rules(text, ents)
        ents = [e for e in ents if e["label"] != "ADR"]   # strip ADR, keep DRUG/SEVERITY/PATIENT_DEMO
        ents = normalise_entities(ents)
        html = highlight_text(text, ents)
        return {
            "text":              text,
            "contains_adr":      False,
            "confidence":        1.0,
            "prob_no_adr":       1.0,
            "prob_adr":          0.0,
            "uncertain":         False,
            "rule_triggered":    False,
            "negation_override": True,
            "entities":          ents,
            "relations":         [],
            "highlighted_html":  html,
        }

    # Fix 1.1: positive dechallenge — "stopped drug → symptom disappeared" proves causation;
    # must fire BEFORE the resolution rule, which would otherwise suppress it as Non-ADR.
    if _POSITIVE_DECHALLENGE.search(text):
        ents = extract_entities(text, ner_tok, ner_mod, device)
        ents = filter_adr_entities(ents)
        ents = merge_hyphenated_drugs(text, ents)
        ents = apply_severity_rules(text, ents)
        ents = apply_ghanaian_synonyms(text, ents)
        ents = apply_adr_gazetteer(text, ents)
        ents = normalise_entities(ents)
        rels = extract_relations(ents, text)
        html = highlight_text(text, ents)
        return {"text": text, "contains_adr": True, "confidence": 1.0,
                "prob_no_adr": 0.0, "prob_adr": 1.0, "uncertain": False,
                "rule_triggered": True, "negation_override": False,
                "entities": ents, "relations": rels, "highlighted_html": html}

    # Fix A: resolution / treatment-success language — runs before classifier
    if has_resolution_language(text):
        ents = extract_entities(text, ner_tok, ner_mod, device)
        ents = filter_adr_entities(ents)
        ents = merge_hyphenated_drugs(text, ents)
        ents = apply_severity_rules(text, ents)
        ents = [e for e in ents if e["label"] != "ADR"]
        ents = normalise_entities(ents)
        html = highlight_text(text, ents)
        return {
            "text":              text,
            "contains_adr":      False,
            "confidence":        1.0,
            "prob_no_adr":       1.0,
            "prob_adr":          0.0,
            "uncertain":         False,
            "rule_triggered":    False,
            "negation_override": True,
            "entities":          ents,
            "relations":         [],
            "highlighted_html":  html,
        }

    # Fix B: indication-frame detection — runs before classifier
    if is_indication_frame(text):
        ents = extract_entities(text, ner_tok, ner_mod, device)
        ents = filter_adr_entities(ents)
        ents = merge_hyphenated_drugs(text, ents)
        ents = apply_severity_rules(text, ents)
        ents = [e for e in ents if e["label"] != "ADR"]
        ents = normalise_entities(ents)
        html = highlight_text(text, ents)
        return {
            "text":              text,
            "contains_adr":      False,
            "confidence":        1.0,
            "prob_no_adr":       1.0,
            "prob_adr":          0.0,
            "uncertain":         False,
            "rule_triggered":    False,
            "negation_override": True,
            "entities":          ents,
            "relations":         [],
            "highlighted_html":  html,
        }

    # Fix 1.5: third-party voice — narrator reports someone else's reaction, not their own
    if _THIRD_PARTY_EXTENDED.search(text) and not first_person_also_affected(text):
        ents = extract_entities(text, ner_tok, ner_mod, device)
        ents = filter_adr_entities(ents)
        ents = merge_hyphenated_drugs(text, ents)
        ents = apply_severity_rules(text, ents)
        ents = [e for e in ents if e["label"] != "ADR"]
        ents = normalise_entities(ents)
        html = highlight_text(text, ents)
        return {
            "text":              text,
            "contains_adr":      False,
            "confidence":        1.0,
            "prob_no_adr":       1.0,
            "prob_adr":          0.0,
            "uncertain":         False,
            "rule_triggered":    False,
            "negation_override": True,
            "entities":          ents,
            "relations":         [],
            "highlighted_html":  html,
        }

    cls  = classify_sentence(text, cls_tok, cls_mod, device)
    ents = extract_entities(text, ner_tok, ner_mod, device)
    ents = filter_adr_entities(ents)                       # FIX 4
    ents = merge_hyphenated_drugs(text, ents)              # 1F
    ents = apply_severity_rules(text, ents)
    ents = apply_ghanaian_synonyms(text, ents)
    ents = apply_adr_gazetteer(text, ents)                 # 1G
    ents = normalise_entities(ents)
    rels = extract_relations(ents, text)                   # FIX 5
    html = highlight_text(text, ents)
    return {"text": text, **cls, "negation_override": False,
            "entities": ents, "relations": rels, "highlighted_html": html}


def _escape(text):
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
