#!/usr/bin/env python3
"""
scripts/batch_regression_phase8.py
====================================
Re-run the 95 known batch-2–11 failures against the Phase 9 Hybrid model
(Phase 8 CLF cohort_study + Phase 7 NER cohort_study, t=0.55) and the
inference_engine_v2.py rule layer.

Baseline comparison: Phase 7 Hybrid (40/95, 42.1%).

Run (from ghana-adr-pipeline/):
    python scripts/batch_regression_phase8.py

Output: reports/batch_regression_phase9.md
"""

import sys, importlib.util
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

ROOT     = Path(__file__).parent.parent
DEMO_DIR = ROOT / "ghana-adr-demo"
REPORTS  = ROOT / "reports"

# Phase 9 Hybrid: Phase 8 CLF + Phase 7 NER (cohort_study fold, t=0.55)
CLF_DIR = ROOT / "reports" / "loso" / "models_phase9_cohort_study" / "clf_best"
NER_DIR = ROOT / "reports" / "loso" / "models_phase7_cohort_study" / "ner_best"
CLF_THRESHOLD = 0.55

# ── Load inference_engine_v2 ──────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("ie_v2", DEMO_DIR / "inference_engine.py")
IE   = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IE)

# ── Batch test cases ──────────────────────────────────────────────────────────
BATCH_CASES = [
    # ── Batch 2 ──────────────────────────────────────────────────────────────
    (2, 20, True,
     "My doctor said my potassium is high but I don't feel anything wrong. "
     "She wants to change the BP medicine.",
     "Asymptomatic lab ADR — patient denies symptoms; lab finding IS the ADR"),

    # ── Batch 3 ──────────────────────────────────────────────────────────────
    (3, 31, True,
     "Since dem start give me the TB injection every morning, "
     "my ear dey ring and I no fit hear well again.",
     "Pidgin — aminoglycoside ototoxicity"),
    (3, 37, True,
     "I dey take my diabetes drug and also dey drink the herbal tea my grandmother prepare. "
     "Since then my sugar dey go too low and I dey dizzy.",
     "Pidgin — drug-herb interaction hypoglycaemia"),
    (3, 55, True,
     "I take the sleeping tablet but then I start walk around the house in the night "
     "and I no know what I dey do.",
     "Pidgin — somnambulism from hypnotic"),

    # ── Batch 4 ──────────────────────────────────────────────────────────────
    (4, 77, True,
     "The diarrhoea after the antibiotic is manageable. I just drink plenty of water. "
     "It's not a big deal.",
     "Severity minimisation — English coping language"),
    (4, 78, True,
     "My stomach dey do small small since I start the tablet. "
     "It no be serious thing, I still dey take am.",
     "Pidgin reduplication 'small small' + severity minimiser"),
    (4, 82, False,
     "He has been nauseated and vomiting since admission. He is jaundiced from hepatitis B. "
     "Currently being started on TB drugs.",
     "Drug not yet started — 'currently being started on'"),
    (4, 84, False,
     "Patient on metformin. Complains of nausea and abdominal pain. "
     "Ultrasound shows acute cholecystitis as the cause.",
     "Investigation exonerates drug — imaging finds alternative aetiology"),
    (4, 85, False,
     "Since starting ARVs the patient has had persistent diarrhoea. "
     "However, stool culture grew Cryptosporidium — ongoing OI treatment initiated.",
     "Microbiology exonerates drug — OI identified as cause"),

    # ── Batch 5 ──────────────────────────────────────────────────────────────
    (5, 91, True,
     "My body dey do me since I start the injection. My system don scatter.",
     "Pidgin illness idiom — generalised dysregulation"),
    (5, 92, True,
     "The medicine enter my head. I no dey think straight since I take am.",
     "Pidgin illness idiom — CNS confusion"),
    (5, 95, True,
     "After I take the drug, blood dey run cold and my body dey shake.",
     "Pidgin — rigors/chills"),
    (5, 96, True,
     "My body was somehow after I started the tablets.",
     "Ghanaian English idiom — malaise"),
    (5, 97, True,
     "Tummy run since dem start give me the treatment.",
     "Pidgin — diarrhoea"),
    (5, 99, True,
     "My body dey hot inside but outside cold since I dey take the medicine.",
     "Pidgin — drug fever with rigors"),
    (5, 101, True,
     "The tablet dey help my sugar but my legs dey pain me oo, especially the joints.",
     "Pidgin concessive — 'dey help X but dey cause Y'"),
    (5, 112, True,
     "I prayed about it but since I started this white tablet, "
     "my body has not been the same. I feel like poison entered me.",
     "Spiritual framing before drug-symptom causal statement"),
    (5, 117, True,
     "Adverse drug reaction report. Drug: quinolone antibiotic. "
     "Reaction: QTc interval prolongation (520 ms). Drug withdrawn.",
     "Formal structured ADR case report register"),
    (5, 118, True,
     "PHARMACOVIGILANCE REPORT. Reaction: Stevens-Johnson syndrome. "
     "Causality assessment: probable.",
     "Regulatory PV report register"),
    (5, 119, True,
     "Signal detection summary: disproportionality analysis identified an "
     "unexpected reporting ratio for peripheral neuropathy with metronidazole.",
     "Pharmacoepidemiology signal detection language"),

    # ── Batch 6 ──────────────────────────────────────────────────────────────
    (6, 122, False,
     "He takes metformin sometimes — maybe three or four times a week. "
     "His stomach is fine and the sugar is controlled.",
     "Non-adherence context — intermittent dosing"),
    (6, 124, False,
     "He was skipping ARV doses for three days while travelling. "
     "Now presenting with mild headache.",
     "Drug holiday — headache not ADR"),
    (6, 127, True,
     "My stomach runs every day but I don't know which one is causing it "
     "— I am on many tablets.",
     "Polypharmacy attribution uncertainty — uncertainty =/= Non-ADR"),
    (6, 130, False,
     "She is having headaches. "
     "The headaches started before any of the current medications were added.",
     "Pre-existing condition explicitly stated"),
    (6, 132, True,
     "She was on metformin for five years with no problem. "
     "But now she is complaining of ankle swelling since her dose was increased.",
     "ARCH FIX 1.8 — historical tolerance clause; negation rule must not fire"),
    (6, 142, True,
     "I had no complaints at first. But then I noticed my skin started "
     "changing colour after the second week of the antibiotic.",
     "ARCH FIX 1.7 — negation pivot guard"),
    (6, 145, False,
     "The drug is not working and he is feeling worse. "
     "However, his HbA1c improved from 9.8% to 7.1%.",
     "Contradictory signal — objective improvement overrides subjective complaint"),
    (6, 146, True,
     "I dey pass water too much — every thirty minutes I run to toilet "
     "since I start the blood pressure tablet.",
     "Pidgin — urinary frequency"),
    (6, 147, True,
     "I had no side effects initially. But the latest ECG shows prolonged "
     "QT interval — the cardiologist says it is drug-related.",
     "ARCH FIX 1.7 — negation pivot guard"),
    (6, 148, False,
     "She feels the drug is making her worse — more pain, more swelling. "
     "Yet her ESR and CRP have normalised.",
     "Nocebo — subjective worsening with objective normalisation"),
    (6, 150, True,
     "My friend who is a nurse look at me and say my eyes yellow small "
     "since I start the hepatitis drug.",
     "Third-party observation of jaundice in Pidgin"),

    # ── Batch 7 ──────────────────────────────────────────────────────────────
    (7, 151, True,
     "My pikin dey shake after we give am the injection. "
     "The whole body dey vibrate and the eyes go up.",
     "Pidgin caregiver — paediatric post-injection seizure"),
    (7, 158, True,
     "14:25 metronidazole 400 mg IV commenced. 14:45 patient complained of nausea. "
     "15:10 vomiting x1. No further action taken. Observations stable.",
     "Nursing DAR timestamped record"),
    (7, 160, True,
     "Ward round note: patient showing involuntary jaw movements and lip smacking "
     "since 0600 hours. Currently on haloperidol.",
     "Ward round clinical shorthand"),
    (7, 161, True,
     "Discharge summary: acute allergic reaction secondary to co-trimoxazole. "
     "Manifestations: urticaria, angioedema, stridor.",
     "Discharge summary — explicit 'secondary to [drug]' causal framing"),
    (7, 162, True,
     "Impression: presentation consistent with drug-induced lupus "
     "most likely attributable to hydralazine.",
     "Physician impression with differential diagnosis causality"),
    (7, 166, True,
     "CHPS compound report: client reported swelling of both legs "
     "since starting blood pressure medication two months ago.",
     "CHPS field report format"),
    (7, 170, True,
     "During mass drug administration, one recipient presented with generalised rash "
     "and vomiting within one hour of receiving artesunate-amodiaquine.",
     "Field surveillance during mass drug distribution"),
    (7, 171, True,
     "ICSR. Reaction term (MedDRA): cardiac arrhythmia. "
     "Seriousness criterion: hospitalisation. Causality assessment: probable.",
     "ICSR structured ADR report"),
    (7, 172, False,
     "Periodic safety update Q3 2025: adverse events reported for "
     "efavirenz-containing regimens in Ghana: 14.",
     "ARCH FIX 1.6 — PSUR aggregate document; INTOLERANCE rule must not override"),
    (7, 174, True,
     "Market withdrawal notification: lot recalled due to reports of "
     "severe renal toxicity in three confirmed cases.",
     "Market withdrawal — regulatory enforcement language"),
    (7, 175, True,
     "Signal assessment memo: reporting rate for peripheral neuropathy is "
     "suggestive but not confirmed. Recommendation: enhanced surveillance.",
     "Hedged signal assessment memo"),
    (7, 176, True,
     "The tablet they approved is spoiling people's kidneys. "
     "My own kidney has been affected since I started taking it.",
     "Patient FDA complaint — 'spoiling' as harm idiom"),
    (7, 177, True,
     "It made me feel terrible for three days. I want compensation from the clinic.",
     "Compensation framing — vague 'terrible' + legal intent"),
    (7, 178, True,
     "I told the doctor but he didn't listen. "
     "My legs are heavy and I cannot walk well since I started the new medication.",
     "Physician-dismissal framing — near threshold"),
    (7, 179, True,
     "The yellow tablet dey make my mouth dry and my eyes dey pain me small "
     "since I start taking it.",
     "Pidgin voice-note — anticholinergic dry mouth / eye pain"),

    # ── Batch 8 — Minimal pairs ───────────────────────────────────────────────
    (8, 182, False,
     "Before I began the ARV treatment, my knee was already painful. "
     "The ARV has not caused it.",
     "P01-B temporal exonerator 'before'"),
    (8, 183, True,
     "My joints hurt badly from taking this drug. I know it is from the tablet.",
     "P02-A causal attribution below threshold"),
    (8, 186, False,
     "My friend had a severe reaction to the malaria injection — "
     "her face swelled and she could not breathe.",
     "FIX 1.5 — third-party voice; narrator not affected"),
    (8, 188, False,
     "I had a rash after starting the drug but it has resolved — it lasted only one day.",
     "FIX 1.4 — 'has resolved' resolution language"),
    (8, 190, False,
     "My legs were swelling two years ago from an old medication but not the current one.",
     "P05-B remote past tense ignored"),
    (8, 192, False,
     "The doctor warned me I might get a headache from the tablet but I took it and I felt fine.",
     "FIX 1.4 — 'felt fine' resolution"),
    (8, 194, False,
     "My stomach no dey pain me at all since I start the new tablet.",
     "P07-B Pidgin negation 'no dey' ignored"),
    (8, 196, False,
     "I have had diarrhoea but I don't think it was the drug — "
     "I had not slept well that week and my diet was poor.",
     "P08-B self-attribution to alternative cause"),
    (8, 197, True,
     "I stopped the ARV and the rash disappeared completely within three days of stopping.",
     "ARCH FIX 1.1 — positive dechallenge"),
    (8, 198, False,
     "I stopped taking the drug but the rash did not change at all — still there.",
     "P09-B negative dechallenge"),
    (8, 200, False,
     "I tried the medication again and this time I had no itching at all.",
     "P10-B negative rechallenge"),
    (8, 204, False,
     "My ALT is 48 U/L — only 1.2 times the upper limit of normal — on this medication.",
     "P12-B lab sub-threshold — numeric multiplier not parsed"),
    (8, 206, True,
     "After I take am, I start sick. The whole body dey shake.",
     "P13-B Pidgin temporal 'After I take am'"),
    (8, 209, True,
     "I am certain the new blood pressure drug is causing my joint pain. "
     "Nothing else has changed in my life recently.",
     "P15-A patient certainty + 'nothing else has changed'"),
    (8, 210, True,
     "I am not sure if the drug is causing my joint pain or if it is something else, "
     "but it started right after I began the medication.",
     "P15-B patient hedging fully suppresses ADR"),

    # ── Batch 9 — Demographic fairness ───────────────────────────────────────
    (9, 212, False,
     "This elderly woman has been on metformin for ten years. No new problems "
     "with the new statin that was added last month.",
     "Elderly baseline — 'No new problems' not extracted"),
    (9, 213, True,
     "After, my head. Too much. And my belly. I go toilet many times. "
     "Started the new tablet last week.",
     "Low-literacy telegraphic style"),
    (9, 219, True,
     "My body is not at rest since starting this medicine. "
     "The hospital man say it could be the drug.",
     "Northern Ghanaian English — akathisia idiom 'body is not at rest'"),
    (9, 221, True,
     "I have problems with my manhood since starting this blood pressure medicine. "
     "I cannot perform as a man should.",
     "Ghanaian male reproductive idiom — erectile dysfunction"),
    (9, 223, True,
     "Since starting the new tablet, I dey thirsty all the time and I dey pass water too much. "
     "But I think I am just not drinking enough water.",
     "Adolescent polydipsia — self-rationalises away ADR"),
    (9, 225, True,
     "The ARV is doing me anyhow since I started. My flesh is paining me all over.",
     "Akan-influenced English — 'doing me anyhow', 'flesh is paining'"),
    (9, 229, True,
     "My tummy hurts a lot and I feel like vomiting every time I take the pill.",
     "Child voice ADR description"),
    (9, 230, True,
     "Anka me ho ye me since I dey take am. My whole body dey pain me.",
     "Multilingual Twi + Pidgin + English"),
    (9, 231, True,
     "Since starting the blood pressure tablet, I dey feel dizzy sometimes. "
     "I sit down small and it passes. I cannot afford to be sick.",
     "Market trader minimisation — dizziness managed"),
    (9, 232, True,
     "I do not want to say what it is but since starting the new tablet, "
     "I have been hearing voices at night.",
     "Stigma concealment — 'hearing voices since starting new tablet'"),
    (9, 234, False,
     "I take my iron supplement every day during pregnancy. I feel well. "
     "The pregnancy is going smoothly.",
     "Routine iron supplementation in pregnancy — no ADR"),
    (9, 235, True,
     "My appetite is completely gone since I started the metformin.",
     "ARCH FIX 1.3 — 'appetite gone' = anorexia ADR, not resolution"),
    (9, 238, True,
     "Inside my head, something is turning and my two eyes are paining me "
     "since starting the injection.",
     "Ewe-influenced English — vertigo and ocular pain idioms"),

    # ── Batch 10 — Maternal/obstetric ────────────────────────────────────────
    (10, 244, False,
     "At each visit, reflexes were checked every hour as instructed for MgSO4 monitoring. "
     "All readings were normal.",
     "Uneventful MgSO4 monitoring — 'reflexes' keyword triggers"),
    (10, 248, False,
     "After the misoprostol I felt cold briefly but it passed quickly. "
     "The midwife said it was expected.",
     "FIX 1.4 — 'passed quickly. midwife said expected'"),
    (10, 249, True,
     "The scan shows the baby has not developed normally since I started "
     "taking the drug at 12 weeks pregnant.",
     "ARCH FIX 1.2 — fetal abnormality; 'has not' triggers negation guard"),
    (10, 251, True,
     "My baby has jaundice. The doctors say it could be from the medicine "
     "I was on during pregnancy.",
     "Neonatal jaundice — indirect maternal drug framing"),
    (10, 253, True,
     "Intrapartum record: patient hyperstimulated following oxytocin infusion. "
     "FHR decelerations noted.",
     "Midwife intrapartum clinical shorthand"),
    (10, 254, True,
     "The TBA gave me herbs to speed up the labour. "
     "After that, I started bleeding very badly.",
     "TBA report — traditional uterotonic with haemorrhage"),
    (10, 255, True,
     "Since starting the magnesium infusion, everything feels unreal. "
     "I feel detached from the room and from myself.",
     "Postpartum depersonalisation ADR"),
    (10, 256, True,
     "My belle dey pain me too much since dem start the drip — like fire.",
     "Pidgin obstetric — uterine pain"),
    (10, 262, False,
     "After the injection I was cold and shaky but the midwife told me "
     "it was normal and it passed within ten minutes.",
     "FIX 1.4 — clinician reassurance + 'passed within ten minutes'"),
    (10, 266, True,
     "Since the delivery injection, I have been having strange thoughts. "
     "I thought the baby had been replaced by someone else.",
     "Postpartum psychosis — 'thought baby had been replaced'"),
    (10, 269, False,
     "I had some nausea in the first two days of the antimalarial but it settled. "
     "Baby is moving normally.",
     "FIX 1.4 — 'nausea...but it settled'"),

    # ── Batch 11 — Vaccine / AEFI ────────────────────────────────────────────
    (11, 272, False,
     "After the BCG my baby was a little unsettled but fed well and slept normally tonight. "
     "No fever.",
     "Post-vaccination minor unsettledness; 'fed well' + 'no fever'"),
    (11, 276, False,
     "After the vaccine, we noticed a runny nose. But the whole family had been sick "
     "with the same thing for a week already — before the vaccine.",
     "Coincidental URTI — family cluster predates vaccination"),
    (11, 278, False,
     "I had a mild headache for two days after the vaccine. It went away on its own.",
     "FIX 1.4 — 'went away on its own'"),
    (11, 280, False,
     "My arm was sore for about two days after the injection. Nothing serious. Normal I think.",
     "FIX 1.4 — 'nothing serious'"),
    (11, 285, False,
     "No vomiting, no diarrhoea, no unusual crying after the hexavalent vaccine. "
     "Routine visit — all normal.",
     "FIX 1.4 — 'no vomiting, no diarrhoea...' negation/resolution"),
    (11, 290, False,
     "My arm was slightly sore at the injection site. Fine the next morning.",
     "FIX 1.4 — 'fine the next morning'"),
    (11, 291, True,
     "I received the COVID vaccine and felt fine afterwards. Walking out of the clinic "
     "I collapsed on the way out of the building. I woke up on the floor with nurses around me.",
     "Post-vaccination vasovagal syncope — 'collapsed', 'woke up on the floor'"),
    (11, 292, False,
     "My child was feverish the night after the MMR vaccine. "
     "Took paracetamol and was fine by the second day.",
     "FIX 1.4 — 'took paracetamol and was fine by the second day'"),
    (11, 293, False,
     "Three weeks after the measles vaccine, my child had a rash. "
     "The doctor says it looks like actual measles — coincidence.",
     "Coincidental measles post-vaccination — 'coincidence' noted"),
    (11, 296, False,
     "I felt fine after the vaccine. Just a small pain where the needle went.",
     "FIX 1.4 — 'felt fine' + trivial injection site pain"),
]

# Phase 7 Hybrid per-batch baseline (pass/tested)
PHASE8_BATCH_SCORES = {
    2:  (0,  1),
    3:  (3,  3),
    4:  (1,  5),
    5:  (10, 11),
    6:  (8, 11),
    7:  (14, 15),
    8:  (15, 15),
    9:  (13, 13),
    10: (11, 11),
    11: (9, 10),
}
PHASE8_TOTAL = (84, 95)

# Cases that should pass due to arch/rule fixes already present in inference_engine_v2
ARCH_FIX_IDS       = {132, 142, 147, 172, 197, 235, 249}
RESOLUTION_FIX_IDS = {186, 188, 192, 248, 262, 269, 272, 278, 280, 285, 290, 292}
EXPECTED_FIX_IDS   = ARCH_FIX_IDS | RESOLUTION_FIX_IDS

# Clusters targeted by Phase 8 synthetic data
PHASE8_TARGET_IDS = {
    # FDA/regulatory register (P16)
    117, 118, 119, 171, 174, 175,
    # Non-adherence / drug-not-started FPs (P17)
    82, 122, 124,
    # Investigation exonerates FPs (P18)
    84, 85,
    # Temporal minimal pairs (P19)
    182, 190, 196, 198, 200,
    # Contradictory signal FPs (P20)
    145, 148,
    # Asymptomatic lab ADR (P21)
    20,
    # Severity minimisation (P22)
    77,
    # Polypharmacy (P26)
    127,
    # Post-vaccination syncope (P25)
    291,
    # Vaccine coincidental illness (P24)
    276, 293,
    # Akan/Ewe/Northern dialect (P27-P29)
    219, 225, 238,
    # Pidgin (P30)
    31, 37, 55, 78, 91, 92, 97, 99, 146, 150,
    # Code-switched (P31)
    230,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  Ghana ADR -- Batch Regression Test (Phase 9 Hybrid)")
    print(f"  CLF: Phase 8 cohort_study  NER: Phase 7 cohort_study  t={CLF_THRESHOLD}")
    print(f"{'='*70}")

    print("  Loading tokenizer from DAPT...")
    tok = IE.AutoTokenizer.from_pretrained(str(ROOT / "models" / "pubmedbert-ghana-dapt"))
    print("  Loading CLF (Phase 8 cohort_study)...")
    cls_mod = IE.AutoModelForSequenceClassification.from_pretrained(str(CLF_DIR)).to(device)
    cls_mod.eval()
    print("  Loading NER (Phase 7 cohort_study)...")
    ner_mod = IE.AutoModelForTokenClassification.from_pretrained(str(NER_DIR)).to(device)
    ner_mod.eval()

    results  = []
    by_batch = {}

    for (batch, cid, expected_adr, text, note) in BATCH_CASES:
        res      = IE.analyse_text(text, tok, cls_mod, tok, ner_mod, device)
        prob_adr = res["prob_adr"]
        pred_adr = res["contains_adr"]
        passed   = (pred_adr == expected_adr)
        results.append((batch, cid, expected_adr, pred_adr, prob_adr, passed, note))
        by_batch.setdefault(batch, []).append((cid, expected_adr, pred_adr, prob_adr, passed, note))

    # ── Build report ─────────────────────────────────────────────────────────
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("  RESULTS -- Phase 9 Hybrid Batch Regression")
    lines.append(f"{'='*70}\n")

    total_pass = total_fail = 0
    p8_target_pass = p8_target_total = 0
    newly_fixed    = []
    regressed      = []

    for b in sorted(by_batch):
        batch_cases = by_batch[b]
        passed_here = sum(1 for *_, p, _ in batch_cases if p)
        n           = len(batch_cases)
        p7_pass, p7_n = PHASE8_BATCH_SCORES.get(b, (0, n))
        delta = passed_here - p7_pass

        sign = f"+{delta}" if delta > 0 else str(delta)
        lines.append(f"  Batch {b:2d}  ({n} failures re-tested)  "
                     f"{passed_here}/{n} now pass  "
                     f"(Phase 7 Hybrid: {p7_pass}/{p7_n}  delta={sign})")

        for (cid, exp, pred, prob, ok, note) in batch_cases:
            marker  = "[+]" if ok else "[x]"
            truth   = "ADR"     if exp  else "Non-ADR"
            got     = "ADR"     if pred else "Non-ADR"
            tags    = []
            if ok  and cid in EXPECTED_FIX_IDS: tags.append("ARCH/RULE FIX")
            if ok  and cid in PHASE8_TARGET_IDS: tags.append("P8 TARGET FIXED")
            if not ok and cid in EXPECTED_FIX_IDS: tags.append("STILL FAILING (expected fixed)")
            tag_str = "  <- " + ", ".join(tags) if tags else ""

            lines.append(
                f"    {marker} #{cid:3d}  truth={truth:8s}  pred={got:8s}  "
                f"prob={prob:.3f}  {note[:52]}{tag_str}"
            )
            if ok:
                total_pass += 1
            else:
                total_fail += 1

            if cid in PHASE8_TARGET_IDS:
                p8_target_total += 1
                if ok:
                    p8_target_pass += 1

        lines.append("")

    p7_macro  = PHASE8_TOTAL[0]
    p7_denom  = PHASE8_TOTAL[1]
    total_n   = total_pass + total_fail
    delta_tot = total_pass - p7_macro

    lines.append(f"{'='*70}")
    lines.append(f"  OVERALL   {total_pass}/{total_n} pass  "
                 f"({100*total_pass/total_n:.1f}%)")
    lines.append(f"  Phase 8 Hybrid baseline:  {p7_macro}/{p7_denom}  "
                 f"({100*p7_macro/p7_denom:.1f}%)")
    sign = f"+{delta_tot}" if delta_tot >= 0 else str(delta_tot)
    lines.append(f"  Delta vs Phase 8 Hybrid:  {sign} cases")
    lines.append(f"  Phase 8 target clusters:  {p8_target_pass}/{p8_target_total} pass  "
                 f"(was {sum(1 for *_, p, _ in results if p and _[-2] in PHASE8_TARGET_IDS or False)}/{p8_target_total})")
    lines.append(f"{'='*70}\n")

    output = "\n".join(lines)
    print(output)

    out_path = REPORTS / "batch_regression_phase9.md"
    md_lines = [
        "# Batch Regression -- Phase 9 Hybrid",
        "",
        "**Model:** Phase 9 CLF (cohort_study) + Phase 7 NER (cohort_study)  ",
        "**Threshold:** 0.55  ",
        "**Rule layer:** inference_engine_v2.py  ",
        "**Baseline:** Phase 7 Hybrid (40/95, 42.1%)  ",
        "",
        "```",
        output.strip(),
        "```",
    ]
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
