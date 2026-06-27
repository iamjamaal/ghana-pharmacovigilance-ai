#!/usr/bin/env python3
"""
scripts/generate_synthetic_phase8.py
======================================
Generate synthetic_phase8.jsonl covering the 11 retraining priorities
identified from the Phase 7 batch regression (55 remaining model-level
failures).

Priorities covered:
  P16 — FDA/Regulatory register (ICSR, signal memo, withdrawal, PV report)
  P17 — Non-adherence / drug-not-yet-started FPs
  P18 — Investigation-exonerates FPs
  P19 — Temporal minimal pairs (extended, supplements v1)
  P20 — Contradictory signal FPs (objective normalises)
  P21 — Asymptomatic lab ADR FNs
  P22 — Severity-minimisation FNs
  P23 — Postpartum psychiatric (extends P7 in v2)
  P24 — Vaccine coincidental illness (extends P6 in v2)
  P25 — Post-vaccination syncope FNs
  P26 — Polypharmacy attribution uncertainty FNs

IMPORTANT: adr_spans and drug_spans are always empty [].
This data is CLF-only. Do NOT pass to NER training.

Output: data/silver/synthetic_phase8.jsonl
Run:    python scripts/generate_synthetic_phase8.py
"""

import json
from pathlib import Path

ROOT   = Path(__file__).parent.parent
OUTDIR = ROOT / "data" / "silver"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT    = OUTDIR / "synthetic_phase8.jsonl"


def make(drug, sentence, priority, tag, n, contains_adr):
    return {
        "drug":         drug,
        "sentence":     sentence,
        "source":       "synthetic_phase8",
        "setid":        f"synthetic_phase8_p{priority}_{tag}_{n:03d}",
        "contains_adr": int(contains_adr),
        "adr_spans":    [],
        "drug_spans":   [],
    }


# ════════════════════════════════════════════════════════════════════════════
# Priority 16 — FDA/Regulatory Register
# Addresses batch failures: #117, #118, #119, #171, #174, #175
# Strategy: authentic Ghana FDA PV register format, ICSR E2B field syntax,
# signal assessment memos, market withdrawal letters, annual PV report excerpts.
# All 130 examples are ADR=1 (regulatory text assumes reportable reaction).
# ════════════════════════════════════════════════════════════════════════════

P16_ADR = [
    # ── ICSR E2B-style case reports (Ghana FDA format) ──────────────────────
    ("cotrimoxazole",   "GH-FADR-2021-00143: Suspect drug — Cotrimoxazole 960 mg daily. Indication: PCP prophylaxis. Reaction: Stevens-Johnson syndrome. MedDRA PT: Stevens-Johnson syndrome. Onset: 14 days. Seriousness: hospitalisation. Outcome: recovered with sequelae."),
    ("lamivudine",      "Patient ID GH-2019-0041. Suspect drug: Lamivudine 150 mg BD. Indication: HIV management. Adverse event: peripheral neuropathy. Latency: 8 weeks. Seriousness criteria: disability. Causality: probable."),
    ("artesunate",      "ICSR reference GH-MAL-2022-0088. Drug: Artesunate injection. Indication: severe malaria. Reaction reported: acute haemolytic anaemia. MedDRA PT: Haemolytic anaemia. Onset: day 7 post-treatment. Seriousness: life-threatening."),
    ("efavirenz",       "GH-ARV-2020-0312. Suspect drug: Efavirenz 600 mg nocte. Indication: HIV-1. Adverse event: suicidal ideation. WHO causality: possible. Seriousness: hospitalisation required. Patient outcome: recovered after drug discontinuation."),
    ("quinine",         "Pharmacovigilance case GH-2023-MAL-0056. Drug: Quinine IV infusion. Indication: severe P. falciparum malaria. ADR: hypoglycaemia. Onset: 6 hours post-infusion. Seriousness: life-threatening. Managed with IV dextrose."),
    ("rifampicin",      "GH-TB-2021-00289. Case: 34-year-old female. Suspect drug: Rifampicin 600 mg OD. Indication: pulmonary TB. ADR: drug-induced hepatitis. LFT: AST 410 U/L, ALT 380 U/L. Seriousness: hospitalisation. Outcome: recovered on drug cessation."),
    ("nevirapine",      "Case ID: GH-FADR-2020-0177. Drug: Nevirapine 200 mg BD. Indication: PMTCT. Reaction: drug hypersensitivity syndrome. Features: fever, rash, hepatitis. Seriousness: life-threatening. Causality: certain. Drug withdrawn."),
    ("pyrazinamide",    "ICSR GH-2022-TB-0091. Suspect drug: Pyrazinamide. Indication: DS-TB treatment. Adverse reaction: acute gout. MedDRA PT: Gout. Onset: 3 weeks. Seriousness: non-serious but required dose interruption. Urate 0.62 mmol/L."),
    ("isoniazid",       "Ghana FDA ICSR GH-TB-2023-0204. Drug: Isoniazid 300 mg OD. Reaction: peripheral neuropathy. Onset: 6 weeks. Risk factor: no pyridoxine co-prescription. Seriousness: disability/incapacity. Causality: certain."),
    ("tenofovir",       "Case GH-HIV-2021-0389. Suspect drug: Tenofovir disoproxil fumarate 300 mg OD. Indication: ART. ADR: Fanconi syndrome. Creatinine elevated 2.8x baseline. Seriousness: hospitalisation. Outcome: resolved after switch to TAF."),
    ("amodiaquine",     "GH-FADR-2023-0072. Drug: Artesunate-Amodiaquine. Indication: uncomplicated malaria. Reaction: agranulocytosis. Neutrophil count: 0.2 × 10⁹/L. Seriousness: life-threatening. Outcome: recovered following G-CSF treatment."),
    ("dapsone",         "Case reference GH-DERM-2022-0031. Drug: Dapsone 100 mg OD. Indication: leprosy. Adverse event: haemolytic anaemia. Hb drop from 12.4 to 7.1 g/dL. Seriousness: hospitalisation. Causality: certain. G6PD status: deficient."),
    ("haloperidol",     "GH-FADR-2020-0441. Drug: Haloperidol 10 mg IM depot. Indication: schizophrenia. Adverse event: neuroleptic malignant syndrome. Temperature 40.2°C, rigidity, CK 4800 U/L. Seriousness: life-threatening. Outcome: ICU admission required."),
    ("methotrexate",    "Case GH-HAEM-2022-0018. Drug: Methotrexate 15 mg weekly. Indication: rheumatoid arthritis. Reaction: pneumonitis. Chest X-ray: bilateral infiltrates. DLCO reduced. Seriousness: life-threatening. Causality: probable."),
    ("chloroquine",     "GH-FADR-2021-0099. Drug: Chloroquine 250 mg OD. Indication: lupus. Adverse event: retinopathy. Ophthalmology: bull's-eye maculopathy on OCT. Seriousness: disability. Causality: certain. Duration of therapy: 4 years."),
    ("gentamicin",      "ICSR GH-SURG-2023-0055. Drug: Gentamicin 320 mg IV OD. Indication: sepsis. ADR: nephrotoxicity. Creatinine rise from 80 to 340 µmol/L. Seriousness: hospitalisation. Causality: probable. Duration: 10 days."),
    ("diclofenac",      "GH-FADR-2022-0308. Drug: Diclofenac 75 mg BD. Indication: osteoarthritis. Reaction: upper GI bleed. Endoscopy: two gastric ulcers. Seriousness: hospitalisation. Causality: probable. Risk factors: no PPI co-prescription."),
    ("amoxicillin",     "Case GH-PAED-2021-0122. Drug: Amoxicillin 500 mg TDS. Indication: pneumonia. ADR: drug-induced urticaria. Onset: 3 days. Seriousness: non-serious. Causality: certain. No anaphylaxis. Antihistamine treatment sufficient."),
    ("spironolactone",  "GH-FADR-2023-0191. Drug: Spironolactone 100 mg OD. Indication: heart failure. ADR: hyperkalaemia. Potassium: 6.8 mmol/L. ECG: peaked T-waves. Seriousness: life-threatening. Managed with calcium gluconate and Kayexalate."),
    ("phenytoin",       "Case GH-NEURO-2022-0074. Drug: Phenytoin 300 mg OD. Indication: epilepsy. Adverse event: gingival hyperplasia with secondary infection. Seriousness: non-serious. Causality: certain. Referred to dental review."),
    # ── Signal assessment memos ──────────────────────────────────────────────
    ("cotrimoxazole",   "Pharmacovigilance signal assessment: cotrimoxazole and Stevens-Johnson syndrome. Identified via ICSR cluster analysis. PRR: 12.4 (95% CI 8.1–19.0). Cases: 27 in reporting period. Signal confirmed. Regulatory action: label update, healthcare provider communication issued."),
    ("efavirenz",       "Signal assessment memo — efavirenz and suicidal behaviour. Disproportionality statistic: ROR 4.8. Cases: 14 spontaneous reports in 18 months, 3 completed suicides. Seriousness: fatal. Action: label amendment, prescriber alert, monitoring guidance."),
    ("artesunate",      "Signal: artesunate and post-treatment haemolysis. Cases reported: 22 with late-onset haemolytic anaemia 2–3 weeks post-treatment. PRR: 6.2. WHO Uppsala Monitoring Centre verification pending. Preliminary action: healthcare alert issued."),
    ("nevirapine",      "Ghana FDA Safety Signal: Nevirapine-associated hypersensitivity syndrome. Cluster of 11 cases in HIV-positive women initiating ART. CD4 count at initiation >250 cells/mm³ identified as risk factor. Risk minimisation measures: CD4 threshold prescribing restriction."),
    ("haloperidol",     "PV signal review — haloperidol depot and neuroleptic malignant syndrome. 8 cases in 12 months, 2 fatalities. Disproportionate compared to oral haloperidol. Regulatory action: Dear Healthcare Professional letter issued. Monitoring requirements updated."),
    ("quinine",         "Signal memo: quinine IV and hypoglycaemia in severe malaria. 34 cases identified, 6 deaths attributed. Risk factors: prolonged infusion, co-administration with insulin. Action: revised dosing guidance, mandatory glucose monitoring protocol."),
    ("amodiaquine",     "Pharmacovigilance signal — artesunate-amodiaquine and agranulocytosis. Spontaneous reports: 19 in West Africa cluster. PRR: 8.3 for combination vs. artesunate monotherapy. Causality: confirmed in 12 cases. Regulatory review initiated."),
    ("isoniazid",       "Signal assessment: isoniazid and hepatotoxicity in Ghanaian population. 48 reports in 2023 annual review. Rate: 1.8 per 1000 patient-years, exceeding expected 1.0. Risk factors: slow acetylators, alcohol use. Label revision recommended."),
    # ── Market withdrawal / Dear HCP letters ────────────────────────────────
    ("valsartan",       "Market withdrawal notification — Ghana FDA. Product: Valsartan-containing medicines. Reason: detection of N-nitrosodimethylamine (NDMA) impurity above acceptable daily intake. All batches recalled. Healthcare providers should switch patients to alternative ARBs immediately."),
    ("metformin",       "Ghana Food and Drug Authority Advisory: Voluntary recall of Metformin Extended-Release tablets, lot numbers GH-MET-2021-003 to GH-MET-2021-009. Reason: NDMA levels exceeding FDA limits. Patients should continue diabetes management with immediate-release formulation or alternative."),
    ("ranitidine",      "URGENT: Ghana FDA market withdrawal — Ranitidine (Zantac) and all generic equivalents. Post-market surveillance identified NDMA impurity levels that increase with storage temperature. All dispensing must cease. Patients should be transitioned to famotidine or PPIs."),
    ("codeine",         "Ghana FDA Healthcare Professional Communication: Codeine-containing medicines restricted for use in children under 12 years and nursing mothers. Pharmacogenomic risk: ultra-rapid CYP2D6 metabolisers may develop respiratory depression at standard doses. Cases of death reported."),
    ("diclofenac",      "Ghana FDA Drug Safety Update: Diclofenac sodium associated with increased cardiovascular risk. Analysis of spontaneous reports and published data indicates elevated myocardial infarction risk. Prescribers should avoid use in patients with established CVD. Label amendments in progress."),
    # ── Annual PV report excerpts (pharmacovigilance surveillance language) ──
    ("artemether-lumefantrine", "2023 Annual Pharmacovigilance Report — Ghana FDA. Artemether-lumefantrine: 234 adverse event reports received. Serious reactions: 12 (5.1%). Most common: prolonged QTc (n=7), haematological reactions (n=3), neuropsychiatric (n=2). Signal monitoring ongoing."),
    ("efavirenz",       "Annual PV Report 2022: Efavirenz neuropsychiatric ADR surveillance. Total reports: 89. Central nervous system reactions constituted 67% of efavirenz ADRs. Suicidality signals (ideation/attempts): 8 cases, rate 0.4 per 100 patient-years on EFV-based ART. Monitoring intensified."),
    ("tenofovir",       "Ghana FDA 2023 Annual Pharmacovigilance Review. Tenofovir renal toxicity monitoring programme: 156 enrolled patients. Creatinine elevation >1.5x baseline: 18 patients (11.5%). Tubular dysfunction (Fanconi): 4 cases. Monitoring frequency increased to 3-monthly."),
    ("rifampicin",      "Annual Surveillance Report 2023: Drug-induced liver injury monitoring in TB patients. Rifampicin-associated hepatotoxicity rate: 6.2% (95/1532 patients). Severe hepatitis requiring drug withdrawal: 23 cases. Two fatalities. Highest risk: first 8 weeks of therapy."),
    ("cotrimoxazole",   "Spontaneous reporting system summary 2022: Cotrimoxazole remains the most frequently implicated drug in serious cutaneous ADRs in Ghana. 41 SJS/TEN cases; 7 deaths. Rate: 1.2 per 10,000 courses. Risk communication to prescribers reinforced."),
    # ── WHO pharmacovigilance training / assessment language ─────────────────
    ("haloperidol",     "WHO causality assessment — haloperidol and tardive dyskinesia. Naranjo score: 8 (definite ADR). Onset: 18 months of exposure. Dechallenge: partial improvement. Rechallenge not performed. Assessment: drug-induced tardive dyskinesia. Documented for PV database."),
    ("amoxicillin-clavulanate", "WHO causality category: Probable. Drug: Amoxicillin-clavulanate. Reaction: cholestatic jaundice. Onset: 10 days. Bilirubin 4.2 mg/dL, ALP 3x ULN. Other causes excluded by investigation. Temporal relationship clear. Dechallenge positive. Rechallenge not done."),
    ("metformin",       "WHO-UMC causality assessment. Drug: Metformin 2000 mg/day. ADR: lactic acidosis. pH 7.18, lactate 12.4 mmol/L. Onset: acute renal injury precipitating event. Seriousness: fatal. Causality: certain. Concomitant iodinated contrast exposure identified as trigger."),
    ("atorvastatin",    "WHO causality assessment form. Drug: Atorvastatin 40 mg OD. ADR: rhabdomyolysis. CK >40,000 U/L. Onset: 3 months post-initiation. Co-medication: clarithromycin (CYP3A4 inhibitor). Seriousness: life-threatening. Causality: certain (drug interaction)."),
    ("carbamazepine",   "PV case narrative: Carbamazepine-induced agranulocytosis. Patient: 28-year-old epileptic. Duration: 6 weeks therapy. ANC: 0.1 × 10⁹/L. Hospitalised. Treated with G-CSF. WHO causality: certain. HLA-B*15:02 testing negative. Mechanism: idiosyncratic."),
    # ── AEFI classification (WHO-AEFI causality) ─────────────────────────────
    ("BCG vaccine",     "AEFI investigation report — BCG osteitis. Case: 4-month-old, BCG administered at birth. Onset: 3 months. Radiology: lytic lesion right humerus. Culture: M. bovis BCG strain. WHO causality classification: vaccine product-related reaction. Immunodeficiency screen: negative."),
    ("OPV vaccine",     "AEFI causality assessment: vaccine-derived polio (VDPV) case. Oral polio vaccine, type 2 component. Child: 18 months, unimmunised siblings. Acute flaccid paralysis onset: 30 days post-OPV. WHO causality: vaccine-derived. Outbreak investigation initiated."),
    ("yellow fever vaccine", "AEFI serious adverse event: yellow fever vaccine-associated viscerotropic disease. Patient: 62-year-old male. Onset: 4 days post-vaccination. Multi-organ failure, jaundice, haemorrhage. Case-fatality ratio. WHO causality: vaccine product-related. Risk factor: age >60."),
    ("COVID-19 vaccine","AEFI causality assessment — COVID-19 mRNA vaccine and anaphylaxis. Onset: 8 minutes post-injection. Clinical criteria: bronchospasm, urticaria, hypotension. Serum tryptase elevated. Treated with IM adrenaline. WHO causality: vaccine product-related reaction (definite anaphylaxis)."),
    ("meningococcal vaccine", "AEFI report — meningococcal vaccine and Guillain-Barré syndrome. Onset: 21 days post-vaccination. EMG: demyelinating pattern. Cerebrospinal fluid: albuminocytological dissociation. Hospital admission 14 days. WHO causality: probable vaccine-related reaction."),
    # ── Pharmacovigilance regulatory inspection language ─────────────────────
    ("isoniazid",       "Regulatory inspection finding: failure to report 23 suspected INH hepatotoxicity cases within the mandatory 15-day expedited reporting window. Cases identified retrospectively from hospital discharge summaries. Total unreported serious ADRs in audit period: 23. Corrective action required."),
    ("cotrimoxazole",   "Pharmacovigilance inspection outcome: systematic under-reporting of SJS/TEN cases associated with cotrimoxazole. Audit of dermatology ward admissions revealed 11 unreported cases. Root cause: healthcare staff unaware of mandatory ADR reporting. Training programme mandated."),
    ("artesunate",      "Post-approval surveillance commitment: Phase IV study of artesunate post-treatment haemolysis in Ghanaian patients. Enrolment target: 500 patients. Primary endpoint: haemoglobin change at day 21. Secondary: transfusion requirements. Interim safety data submitted to Ghana FDA."),
    # ── Structured case narrative (PV database entry format) ─────────────────
    ("metformin",       "Case narrative: 71-year-old female, DM type 2. Metformin 1000 mg BD ongoing. Admitted for elective procedure; IV contrast administered without metformin withholding. Developed lactic acidosis 24 hours post-contrast. pH 7.12. ICU admission. Outcome: recovered. Lesson: pre-procedure metformin protocol breach."),
    ("rifampicin",      "Case narrative: 29-year-old male, TB-HIV co-infection. Rifampicin 600 mg OD + ART (efavirenz). Developed hepatitis at week 5: ALT 8x ULN, jaundice. Naranjo score: rifampicin 6, efavirenz 4. Drug-induced liver injury — probable rifampicin, possible efavirenz contribution. Both withdrawn."),
    ("tenofovir",       "Clinical case entered into National ADR Database: Tenofovir nephropathy in HIV patient. TDF therapy 3 years. Gradual creatinine rise noted at 18 months (ignored). At 36 months: creatinine 285 µmol/L, phosphate wasting, glycosuria. Proximal tubule dysfunction. TDF discontinued; switch to TAF. Partial recovery."),
    ("nevirapine",      "ADR case GH-NVP-2022: Female patient CD4 312 cells/mm³. Started nevirapine-based ART. Day 12: fever, diffuse rash, elevated transaminases. Classic nevirapine hypersensitivity syndrome. Stopped immediately. Prednisolone 1 mg/kg started. Recovered in 14 days. Switched to efavirenz."),
    ("quinine",         "Case entry: Recurrent hypoglycaemia during IV quinine treatment, falciparum malaria, 3rd trimester pregnancy. Serial blood glucose: 2.1, 1.8, 2.3 mmol/L. Continuous glucose monitoring initiated. Mechanism: quinine-stimulated insulin release + pregnancy insulin sensitisation. Managed with 10% dextrose co-infusion."),
    # ── PSUR / risk management language ─────────────────────────────────────
    ("efavirenz",       "Periodic Safety Update Report (PSUR) extract — Efavirenz. Review period: 2021–2023. Serious psychiatric events: 42 reports (suicidal ideation n=18, attempted suicide n=8, completed suicide n=3, psychosis n=13). Cumulative rate: 0.8 per 1000 patient-years. Risk minimisation: updated patient information, prescriber checklist."),
    ("cotrimoxazole",   "PSUR executive summary — Cotrimoxazole. Serious cutaneous reactions: cumulative 89 SJS/TEN cases in African countries 2020–2023. Mortality rate 8.9%. Risk factors identified: high-dose regimen, slow acetylator genotype, Asian descent less relevant in sub-Saharan context. Label revision underway."),
    ("artemether-lumefantrine", "Risk Management Plan — Artemether-lumefantrine. Identified risk: QTc prolongation. Characterisation: 14/234 patients QTc >500 ms post-dose. Risk factors: female, electrolyte imbalance. Minimisation: ECG monitoring in high-risk patients, concomitant QT-prolonging drug avoidance, patient counselling."),
    ("amodiaquine",     "Post-marketing commitment report: Amodiaquine agranulocytosis surveillance. Enhanced pharmacovigilance programme year 2. Cases detected: 6 definite, 8 probable. Denominator: 1.2 million treatment courses. Rate: 1.2 per 100,000 courses. Benefit-risk assessment: remains favourable for malaria treatment."),
    ("haloperidol",     "Risk communication — haloperidol depot and metabolic syndrome. 5-year follow-up data: 38% of patients on long-acting injectable developed clinically significant weight gain (>7% body weight). 22% developed new-onset T2DM. Monitoring programme: annual fasting glucose, lipids, weight, waist circumference."),
    # ── Ghana EPI / AEFI programme language ─────────────────────────────────
    ("pentavalent vaccine", "AEFI surveillance report: Cluster investigation — pentavalent vaccine, Northern Region. 4 serious AEFI cases at one health facility. Two cases of hypotonic-hyporesponsive episode (HHE), one high-pitched cry >3 hours, one febrile convulsion. Vaccine vial condition check: normal. Lot number analysis pending."),
    ("measles-rubella vaccine", "National AEFI investigation: Severe allergic reaction following measles-rubella vaccination. Anaphylaxis onset within 15 minutes of injection. Adrenaline given at vaccination site. Recovered. Causality: vaccine product-related (latex allergy excluded; gelatin hypersensitivity suspected). Batch recall review not indicated."),
    ("BCG vaccine",     "AEFI programme report: BCG lymphadenitis cluster, Accra District. 7 cases ipsilateral axillary lymphadenopathy >1 cm, 2 cases suppurative. All same BCG batch. Investigation: no cold-chain breach identified. Batch potency test: in range. Probable high-dose injection technique variance. Training conducted."),
    ("OPV vaccine",     "AEFI report: Vaccine-associated paralytic poliomyelitis (VAPP). Immunocompromised recipient. OPV dose 1. Onset: 7–30 days post-administration. Virus isolation: Sabin type 1 from stool. WHO classification: vaccine-associated (definite). Switched to IPV for sibling contacts. Cluster investigation negative."),
    ("yellow fever vaccine", "Serious AEFI — yellow fever vaccine-associated neurotropic disease (YEL-AND). Encephalitis onset 12 days post-vaccination. CSF: pleocytosis, YFV antigen positive. Patient: 45-year-old thymus dysfunction (risk factor). WHO causality: vaccine product-related. Outcome: partial neurological recovery at 6 months."),
    # ── Additional structured PV record formats ──────────────────────────────
    ("methotrexate",    "Adverse drug reaction report form — completed by prescribing physician. Drug: Methotrexate 25 mg weekly. Reaction: pancytopenia. Haematology: Hb 6.1, WBC 1.8, Plt 42. Onset: 10 weeks. Concurrent NSAID use identified as probable interaction (reduced renal clearance). Seriousness: hospitalisation. Folate co-prescription omitted."),
    ("atorvastatin",    "Statin-associated muscle symptoms (SAMS) registry entry. Drug: Atorvastatin 40 mg. Duration: 7 months. Symptom onset: bilateral proximal myalgia, weakness. CK 1,840 U/L (3.8x ULN). Drug-related: confirmed (resolved on dechallenge, recurred on rechallenge with pravastatin). Switched to rosuvastatin 5 mg — tolerating well."),
    ("carbamazepine",   "Case summary submitted to West African Pharmacovigilance Network. Drug: Carbamazepine 400 mg BD. ADR: SJS. Onset: 3 weeks. Mucosal involvement: oral, ocular. Skin detachment: <10% BSA. HLA-B*15:02 not tested (resource-limited). Seriousness: hospitalisation, ophthalmology consult. Outcome: recovered, residual ocular scarring."),
    ("rifampicin",      "Pharmacovigilance case — drug-drug interaction leading to ADR. Rifampicin 600 mg (CYP3A4 inducer) added to stable warfarin regimen. INR decreased from 2.8 to 1.1 within 2 weeks. Prescriber not informed of interaction. DVT recurrence. ADR type: pharmacokinetic drug interaction. Anticoagulation management modified."),
    ("clofazimine",     "ICSR: Clofazimine-induced skin discolouration, leprosy treatment. Patient: 24-year-old female, MB leprosy. Onset: 2 months. Extent: diffuse red-brown/black pigmentation, face and limbs. Expected effect but patient not counselled. Seriousness: not serious. Causality: certain. Patient information leaflet updated to describe discolouration."),
    ("doxycycline",     "Pharmacovigilance database entry. Drug: Doxycycline 100 mg BD (malaria prophylaxis). ADR: oesophageal ulceration. Onset: day 4. Endoscopy confirmed mid-oesophageal ulcer. Risk factor: tablet taken without adequate water, lying down immediately after. Seriousness: hospitalisation. Causality: certain. Patient counselling update."),
    # ── WHO essential medicines post-marketing surveillance ──────────────────
    ("chloroquine",     "Post-market surveillance report: Chloroquine retinal toxicity screening programme, 2023. 180 patients with >5 years chloroquine exposure for lupus/RA screened. Maculopathy on OCT: 22 patients (12.2%). Severe/central scotoma: 3 cases. Rate exceeds published estimates for African populations. Ophthalmology co-management protocol established."),
    ("phenobarbitone",  "Post-marketing ADR report — phenobarbitone and cognitive impairment in children. Longitudinal cohort: 52 paediatric epilepsy patients, average 3.2 years exposure. Cognitive testing: significant deficit in attention, memory vs. carbamazepine group (p<0.001). Reporting to national programme. Policy review: switch to levetiracetam considered."),
    ("gentamicin",      "National monitoring programme — aminoglycoside nephrotoxicity and ototoxicity. Prospective surveillance: 312 patients receiving gentamicin. Nephrotoxicity (creatinine rise >50% baseline): 42 (13.5%). Audiometry-confirmed ototoxicity: 18 (5.8%). Risk factors: duration >7 days, trough >2 mg/L, pre-existing renal impairment. TDM service established."),
    ("furosemide",      "ADR case series: furosemide-associated hypokalaemia and cardiac arrhythmia in Ghanaian patients. 8 cases in 6-month review. Potassium at ADR recognition: range 2.0–2.8 mmol/L. Three cases required IV replacement. Two developed AF. Contributing factor: co-prescription without potassium supplementation monitoring. Guideline revision initiated."),
    ("glibenclamide",   "FADR surveillance report: glibenclamide-associated hypoglycaemia, frail elderly patients. 15 serious cases in 2023. Age range 72–89. Episodes: coma (n=4), seizure (n=3), fall-related injury (n=5). Contributing factors: irregular meal times, deteriorating renal function. Regulatory action: contraindication in patients >70 years with GFR <60."),
    # ── Additional ICSR-style to reach volume ────────────────────────────────
    ("lopinavir/ritonavir", "GH-ARV-2021-0244. Drug: Lopinavir/ritonavir 400/100 mg BD. Indication: HIV-2. ADR: pancreatitis. Onset: 9 weeks. Lipase 8x ULN. CT pancreas: oedematous pancreatitis. Seriousness: hospitalisation. Causality: probable (lipid-elevating mechanism). Switched to integrase inhibitor-based regimen."),
    ("amphotericin B",  "ICSR GH-FUNG-2022-0031. Drug: Amphotericin B deoxycholate 0.7 mg/kg/day. Indication: cryptococcal meningitis. ADR: infusion-related reaction. Fever 39.4°C, rigors, hypotension during infusion. Seriousness: medically significant. Management: pre-medication with hydrocortisone, paracetamol. Liposomal formulation not available."),
    ("ciclosporin",     "GH-TRANS-2021-0008. Drug: Ciclosporin 3 mg/kg/day. Indication: renal transplant. ADR: nephrotoxicity. Creatinine: 340 µmol/L (baseline 110). Renal biopsy: striped interstitial fibrosis. Ciclosporin level: 680 ng/mL. Seriousness: hospitalisation, life-threatening. Dose reduction implemented; tacrolimus conversion."),
    ("amiodarone",      "Case GH-CARD-2023-0017. Drug: Amiodarone 200 mg OD. Indication: persistent AF. ADR: thyrotoxicosis (amiodarone-induced thyroid disease type 1). TSH undetectable, fT4 elevated 3x. Atrial fibrillation worsened. Seriousness: hospitalisation. Causality: certain. Carbimazole added; amiodarone discontinued."),
    ("trimethoprim",    "GH-FADR-2022-0389. Drug: Trimethoprim 200 mg BD. Indication: recurrent UTI prophylaxis. ADR: hyperkalaemia. K+ 6.1 mmol/L. Mechanism: trimethoprim blocks potassium excretion via ENaC. Concurrent ACE inhibitor identified as additional risk factor. Seriousness: medically significant. Both drugs dose-adjusted."),
    ("pyrimethamine",   "ICSR: Pyrimethamine + sulfadoxine (SP) and megaloblastic anaemia in HIV patient. SP given for IPT-P in pregnancy. Haemoglobin 7.2 g/dL, MCV 108 fL, hypersegmented neutrophils. Concurrent ART with zidovudine (additional cause). Causality: probable SP contribution. Folate supplementation dose increased."),
    ("dexamethasone",   "Case GH-PAED-2022-0098. Drug: Dexamethasone 0.15 mg/kg/dose. Indication: bacterial meningitis adjunct therapy. ADR: severe hyperglycaemia. BG: 22 mmol/L day 2 of treatment. Insulin infusion required. Seriousness: medically significant. Causality: certain (dose-dependent steroid effect). BG monitoring every 4 hours during steroid course."),
    ("hydroxychloroquine", "GH-RHEUM-2023-0041. Drug: Hydroxychloroquine 400 mg OD. Indication: SLE. Duration: 6 years. ADR: QTc prolongation. QTc 524 ms on ECG. Symptomatic: palpitations, near-syncope. Seriousness: medically significant. Hydroxychloroquine dose reduced; cardiology review. Baseline and annual ECG monitoring protocol formalised."),
    ("acyclovir",       "ICSR GH-INFECT-2021-0077. Drug: Acyclovir 10 mg/kg IV TDS. Indication: herpes encephalitis. ADR: acute kidney injury. Creatinine rise from 90 to 520 µmol/L within 72 hours. Mechanism: acyclovir crystalluria. Contributing factor: inadequate hydration. Seriousness: hospitalisation. Acyclovir dose reduced; IV fluids at 500 mL/hr."),
    ("lamotrigine",     "Adverse event report GH-NEURO-2023-0062. Drug: Lamotrigine 100 mg BD. Indication: bipolar disorder. ADR: drug rash with eosinophilia and systemic symptoms (DRESS). Onset: 4 weeks. Rash, fever, eosinophilia 2800 cells/µL, lymphadenopathy, raised LFTs. Seriousness: hospitalisation. Lamotrigine withdrawn; systemic steroids given."),
    ("warfarin",        "GH-HAEM-2022-0118. Drug: Warfarin 5 mg OD. Indication: AF, CHA₂DS₂-VASc 4. ADR: major bleeding (haematuria + Hb drop 4 g/dL). INR at event: 6.2. Contributory factors: recent antibiotic course (reduced gut flora, reduced vitamin K), intercurrent illness reducing oral intake. Seriousness: hospitalisation. Vitamin K IV administered."),
    ("lithium",         "Case GH-PSYCH-2022-0033. Drug: Lithium carbonate 800 mg/day. Indication: bipolar I. ADR: lithium toxicity. Serum lithium: 2.4 mmol/L. Features: coarse tremor, confusion, diarrhoea, vomiting. Precipitant: dehydration from gastroenteritis reducing lithium clearance. Seriousness: hospitalisation. Managed with IV hydration, lithium held."),
    ("furosemide",      "ICSR — furosemide and ototoxicity. Drug: furosemide 500 mg IV bolus (loop dose for acute pulmonary oedema). ADR: sudden sensorineural hearing loss, bilateral. Onset: 30 minutes post-infusion. Audiometry: flat loss 60 dB. Mechanism: cochlear endolymph osmotic disruption. Seriousness: disability. Irreversible hearing loss."),
    ("chlorpromazine",  "GH-FADR-2020-0193. Drug: Chlorpromazine 300 mg OD. Indication: acute psychosis. ADR: obstructive jaundice (chlorpromazine cholestasis). Bilirubin 6.4 mg/dL, ALP 4x ULN, minimal ALT rise. Onset: 4 weeks. Eosinophilia present. Causality: certain (hypersensitivity cholestasis). Drug withdrawn; ursodeoxycholic acid prescribed."),
    ("nifedipine",      "PV database entry GH-CARD-2022-0144. Drug: Nifedipine modified-release 30 mg OD. Indication: hypertension. ADR: peripheral oedema, bilateral, pitting to mid-shin. Onset: 6 weeks. Mechanism: arterial vasodilation without venous component. Seriousness: non-serious, but treatment-limiting. Switched to amlodipine with addition of ACE inhibitor."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 17 — Non-adherence / drug-not-yet-started FPs
# Addresses batch failures: #82, #122, #124
# Sentences where patient hasn't started drug, is between doses, or stopped
# drug — current symptoms pre-date drug use.  All Non-ADR.
# ════════════════════════════════════════════════════════════════════════════

P17_NON_ADR = [
    # Drug prescribed but not yet started
    ("metformin",       "I was prescribed metformin last week but I haven't started taking it yet — I'm still experiencing the stomach pains I had before. They're from the H. pylori infection I've had for months."),
    ("lisinopril",      "The doctor gave me lisinopril for my blood pressure but I picked up the prescription two days ago and haven't taken a single tablet — I still have the cough I've had since my chest infection in December."),
    ("amoxicillin",     "I was prescribed amoxicillin for the tooth abscess but I only have the prescription, I haven't filled it yet. My face is still swollen from the infection."),
    ("atorvastatin",    "I'm supposed to start the cholesterol tablet next week when I come back for my review. I haven't taken any of it yet. My muscle aches are from the gym — nothing to do with a tablet I haven't touched."),
    ("metformin",       "She was being initiated on metformin when we noticed the elevated creatinine — the decision was made to delay starting until renal function is checked. The nausea she has is from the hyperglycaemia itself, pre-medication."),
    ("cotrimoxazole",   "We are currently initiating her on cotrimoxazole prophylaxis but have not yet dispensed the first dose — the rash she presented with today predates any drug exposure."),
    ("isoniazid",       "He was prescribed the full TB preventive therapy course at today's visit. He has not yet taken any isoniazid. The fatigue he reports has been present for the past three months — unrelated to drug not yet started."),
    ("rifampicin",      "The patient collected the anti-TB prescription today and is yet to fill it at the pharmacy — the yellow discolouration of urine she describes cannot be from rifampicin she has not yet taken."),
    ("efavirenz",       "ARV initiation planned for next clinic visit. No efavirenz dispensed today. The vivid dreams she reports are a long-standing feature she has had since childhood, not a drug effect."),
    ("doxycycline",     "I'm about to start the malaria prophylaxis for my trip next week. Haven't taken any yet. The heartburn I have now is from the spicy food I had last night, not from tablets I haven't opened."),
    # Drug holiday / stopped before symptom onset
    ("diclofenac",      "I stopped the diclofenac three weeks ago because I finished the course. The stomach pain started this week, well after the drug was out of my system — this is a new problem, not from the NSAID I stopped."),
    ("contraceptive pill", "She discontinued the combined oral contraceptive pill six months ago. The irregular bleeding she presents with now is unrelated to the pill she stopped half a year ago."),
    ("metformin",       "I ran out of my metformin tablets and have been without them for two weeks. The nausea I have now started before I ran out, and has persisted — it's the poorly controlled diabetes, not the tablet I've stopped taking."),
    ("amlodipine",      "He discontinued amlodipine himself about a month ago because he felt fine. The ankle swelling he now has is consistent with heart failure decompensation following the drug discontinuation, not an effect of the amlodipine."),
    ("haloperidol",     "The patient stopped taking the haloperidol depot three months ago without telling us. The tremors he has now are withdrawal-associated — not a new adverse drug reaction."),
    # Pre-existing condition predating drug start
    ("glibenclamide",   "His hypoglycaemic episodes were documented on the previous admission before glibenclamide was ever prescribed — he had an insulinoma. The drug was initiated after those episodes and is not their cause."),
    ("co-trimoxazole",  "The skin rash was present for two weeks before co-trimoxazole was started. Dermatology reviewed and confirmed psoriasis flare predating any drug exposure."),
    ("amoxicillin-clavulanate", "She had abnormal liver function tests at baseline before amoxicillin-clavulanate was prescribed — pre-existing fatty liver disease. The elevated enzymes are not new and are not drug-induced."),
    ("isoniazid",       "The peripheral neuropathy was documented at the pre-treatment assessment before isoniazid was started — existing diabetic neuropathy. Drug cannot be the cause of a condition present before treatment."),
    ("metformin",       "The diarrhoea is listed as a pre-existing condition — longstanding IBS, documented in the notes from 2018. Metformin has only been prescribed in this consultation. The symptom predates the drug by three years."),
    # Drug not dispensed / prescription error
    ("warfarin",        "The warfarin prescription was written but the pharmacy dispensed placebo due to a dispensing error — the patient's bleeding is not from warfarin she never actually received."),
    ("lithium",         "Lithium was added to her prescription three weeks ago. She told me today she never collected it from the pharmacy — the tremor and polyuria she has are from her pre-existing anxiety and diabetes insipidus, not from an uncollected medication."),
    ("carbamazepine",   "Dispensing records confirm carbamazepine was not dispensed at last visit despite being on the prescription. The patient hasn't been taking any anticonvulsant for the past month. Her rash is from the viral infection she developed last week."),
    # Drug recently started but symptom clearly predated
    ("metformin",       "She started metformin only yesterday, and the vomiting she has today has been going on for five days before that — this is the gastroenteritis going around in her community, not a metformin reaction from a single dose."),
    ("atorvastatin",    "He took his first atorvastatin tablet this morning. The muscle aches he describes are three weeks old — present for weeks before any statin exposure. Almost certainly delayed-onset post-viral myalgia."),
    # Between courses / drug-free interval
    ("artemether-lumefantrine", "She completed the three-day artemether-lumefantrine course five weeks ago. The joint pains she has now have been present for two weeks — well outside the drug elimination period. Rheumatological cause being investigated."),
    ("doxycycline",     "He finished the doxycycline course two months ago. The photosensitivity reaction he had resolved completely within a week of stopping. The current sun sensitivity is from sun damage to his skin, not an ongoing drug effect."),
    ("amoxicillin",     "The penicillin course ended three weeks ago. Any urticarial reaction from amoxicillin would have resolved by now — the chronic urticaria she has is a primary immune condition under investigation, unrelated to a long-discontinued antibiotic."),
    ("rifampicin",      "He completed the TB treatment course including rifampicin 8 months ago. He was declared treatment success. The joint pains and fatigue he now reports are from post-TB syndrome — not from rifampicin taken eight months ago."),
    # Topical/non-systemic route
    ("hydrocortisone cream", "She's been using hydrocortisone 1% cream topically for the eczema. The moon face and weight gain she describes are not from topical hydrocortisone at 1% — systemic absorption at this dose is negligible. Another cause of Cushingoid features is being sought."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 18 — Investigation-Exonerates FPs
# Addresses batch failures: #84, #85
# Strategy: investigation result reveals alternative non-drug diagnosis.
# Pattern: [test] shows/grew/revealed/confirmed [non-drug cause of symptom].
# All Non-ADR.
# ════════════════════════════════════════════════════════════════════════════

P18_NON_ADR = [
    # Imaging exonerates drug
    ("atorvastatin",    "The ultrasound showed gallstones in the gallbladder — the doctor said that's what's been causing the right upper quadrant pain, not the atorvastatin. The statin can continue."),
    ("metformin",       "The abdominal CT showed a perforated appendix. The surgeon confirmed the pain and vomiting were from appendicitis, not from the metformin. The metformin was unrelated."),
    ("isoniazid",       "The chest X-ray showed a large pleural effusion from malignancy. The breathlessness was from the cancer pressing on the lung, not from the TB medication."),
    ("amlodipine",      "Echocardiography confirmed severe right heart failure as the cause of the bilateral ankle oedema. The amlodipine was not responsible — the heart failure was the cause."),
    ("haloperidol",     "Brain MRI showed a bilateral subdural haematoma. The confusion was from head injury following a fall, not from the haloperidol. The drug was not the cause."),
    ("efavirenz",       "The brain CT showed a large ischaemic stroke in the right hemisphere. The neurological symptoms were from the stroke, not from the efavirenz. Neurology team confirmed drug not implicated."),
    ("tenofovir",       "The renal ultrasound showed hydronephrosis from a ureteric calculus. The acute kidney injury was from obstruction, not from tenofovir. Urology consultation arranged."),
    ("rifampicin",      "The liver biopsy showed hepatocellular carcinoma. The jaundice and elevated transaminases were from the malignancy, not from rifampicin. The tumour explained all the abnormalities."),
    ("cotrimoxazole",   "Dermoscopy confirmed the skin lesion was a melanoma, not a drug rash. Dermatology removed it surgically. Cotrimoxazole was not implicated."),
    ("diclofenac",      "Endoscopy showed a large duodenal ulcer with H. pylori on CLO test. The bleeding was from the pre-existing peptic ulcer disease caused by H. pylori infection, not from the diclofenac."),
    # Culture / microbiology exonerates drug
    ("amoxicillin",     "Stool culture grew Salmonella typhi. The diarrhoea, fever, and abdominal pain were from typhoid fever, not from the amoxicillin. The antibiotic was switched to ciprofloxacin to treat the causative organism."),
    ("co-trimoxazole",  "Blood cultures grew Staphylococcus aureus — the rigors and fever were from bacteraemia, not from the cotrimoxazole. IV flucloxacillin was started and the patient improved rapidly."),
    ("metronidazole",   "Stool microscopy and culture confirmed Entamoeba histolytica. The bloody diarrhoea was from amoebic dysentery. The metronidazole was actually treating the infection — the symptoms were from the disease, not the drug."),
    ("amoxicillin-clavulanate", "Throat swab grew Group A Streptococcus. The fever and sore throat were from streptococcal pharyngitis — the antibiotic was the treatment, not the cause. The rash was a Streptococcal scarlet fever rash, not drug-induced."),
    ("gentamicin",      "Bronchoalveolar lavage grew Aspergillus fumigatus. The pulmonary infiltrates were from invasive aspergillosis — the gentamicin was treating a concomitant bacterial infection and was not responsible for the lung changes."),
    ("doxycycline",     "Serology confirmed acute Q fever (Coxiella burnetii IgM positive). The hepatitis was from Q fever infection, not from doxycycline. In fact doxycycline is the treatment of choice for Q fever."),
    # Biopsy / histology exonerates drug
    ("isoniazid",       "Liver biopsy showed granulomatous hepatitis consistent with TB hepatic involvement, not drug-induced hepatitis. The isoniazid was treating the very disease that was causing the liver abnormalities."),
    ("methotrexate",    "Lung biopsy confirmed lymphoma with pulmonary involvement. The bilateral infiltrates were from the cancer, not from methotrexate. Oncology reviewed and initiated chemotherapy."),
    ("carbamazepine",   "Skin biopsy confirmed lichen planus — a chronic autoimmune skin condition. The oral and cutaneous lesions were not from carbamazepine but from the primary inflammatory skin disease."),
    ("atorvastatin",    "Muscle biopsy showed inflammatory myopathy (polymyositis). The weakness and elevated CK were from the autoimmune condition, not from the atorvastatin. Rheumatology initiated immunosuppression."),
    # Blood test / biochemistry exonerates drug
    ("metformin",       "Vitamin B12 levels came back critically low — the neuropathy was from B12 deficiency, not from metformin. Supplementation started and symptoms improved rapidly within weeks."),
    ("atorvastatin",    "TSH was markedly elevated — 48 mIU/L. The muscle aches and fatigue were from hypothyroidism, not from the atorvastatin. Levothyroxine started and symptoms resolved without stopping the statin."),
    ("amlodipine",      "A random cortisol was undetectable. The weakness, dizziness, and weight loss were from undiagnosed Addison's disease, not from the amlodipine. Emergency hydrocortisone given."),
    ("cotrimoxazole",   "G6PD deficiency confirmed on enzyme assay. The haemolytic anaemia was from a congenital G6PD deficiency exposed by intercurrent infection, not directly caused by the cotrimoxazole at the dose given."),
    ("isoniazid",       "Anti-LKM-1 antibodies positive. The hepatitis was from autoimmune hepatitis type 2 — the diagnosis predated TB treatment. Isoniazid was discontinued and azathioprine started for the autoimmune condition."),
    # ECG / cardiac investigation exonerates drug
    ("amiodarone",      "The 24-hour Holter monitor showed pre-existing Wolff-Parkinson-White syndrome with accessory pathway — the palpitations were from the WPW, not from the amiodarone which was actually trying to control the arrhythmia."),
    ("azithromycin",    "Formal QTc measurement on ECG: 498 ms. Family history of sudden cardiac death obtained — LQTS genetic testing confirmed congenital long QT syndrome type 1. The QTc prolongation was genetic, not azithromycin-induced."),
    # Urine / renal investigation exonerates drug
    ("tenofovir",       "Renal tract ultrasound and proteinuria testing: 24-hour urine protein 4.2 g. Renal biopsy — FSGS (focal segmental glomerulosclerosis) confirmed. The nephropathy was from HIV-associated nephropathy, not from tenofovir."),
    ("cotrimoxazole",   "Urine microscopy: 30 red cells per high-power field. Cystoscopy: transitional cell carcinoma of the bladder. The haematuria was from bladder cancer, not from the cotrimoxazole."),
    # Neurological investigation exonerates drug
    ("phenytoin",       "EEG showed ictal activity not controlled by phenytoin — the confusion was from non-convulsive status epilepticus from the underlying epilepsy, not from phenytoin toxicity. Phenytoin level was subtherapeutic."),
    ("haloperidol",     "Lumbar puncture and MRI confirmed viral encephalitis. The agitation, confusion, and fever were from the encephalitis, not from the haloperidol. Antiviral treatment started."),
    # Genetic / metabolic investigation exonerates
    ("valproate",       "Metabolic screen confirmed MELAS syndrome (mitochondrial myopathy). The elevated lactate and hepatic dysfunction were from the mitochondrial disease, not from valproate — though valproate was appropriately discontinued given mitochondrial disease."),
    ("atorvastatin",    "Genetic testing confirmed familial hypercholesterolaemia (LDLR pathogenic variant). The muscle enzyme elevation predates statin use — baseline CK was 600 U/L before any statin was ever prescribed. Normal for this patient."),
    # Clinical reassessment exonerates drug
    ("metformin",       "Timeline review confirmed: the weight loss began 8 months ago — two years before metformin was prescribed. The ongoing weight loss is from the underlying gastrointestinal malignancy, not from metformin."),
    ("rifampicin",      "Medication reconciliation revealed the yellow discolouration of urine and skin was explained by the rifampicin, which is an expected, harmless effect — not an adverse event. The jaundice the family reported was actually the orange discolouration from rifampicin excretion, not true jaundice. Bilirubin was normal."),
    ("isoniazid",       "Ophthalmology review confirmed optic nerve drusen, a pre-existing congenital condition. The visual changes were not from isoniazid but from the pre-existing optic anomaly. Visual fields were actually stable."),
    ("doxycycline",     "The photosensitivity dermatitis resolved six months after stopping doxycycline — however patch testing confirmed the patient has a primary photoallergy to UV-A independent of any medication. The doxycycline may have unmasked it but was not the underlying cause."),
    ("warfarin",        "Repeat INR was 1.8 — within therapeutic range. Investigation of the joint haemarthrosis showed haemophilia A on factor VIII assay. The bleeding disorder was congenital, not from warfarin at therapeutic levels."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 19 — Temporal Minimal Pairs Extended
# Addresses batch failures: #182, #190, #196, #198, #200, #204
# Supplements the 160 examples already in synthetic_phase7_v1.jsonl.
# Focus: remote past tense, negative dechallenge, negative rechallenge,
# symptom predates drug, future tense.
# ════════════════════════════════════════════════════════════════════════════

P19_ADR = [
    # Currently active ADR (positive label)
    ("metformin",       "I started metformin two weeks ago and I've been having bad diarrhoea every morning since then — it's new, it wasn't there before the tablet."),
    ("atorvastatin",    "The muscle pain began three days after I started atorvastatin and it's been getting worse every week since I've been taking it."),
    ("cotrimoxazole",   "Since starting cotrimoxazole I've developed this generalised rash — it appeared on day 4 of treatment and is spreading."),
    ("efavirenz",       "Every night since I started efavirenz I have very vivid nightmares and I wake up confused — this only started with the new drug."),
    ("lisinopril",      "This dry cough started two weeks into taking lisinopril. I didn't have any cough before this tablet."),
    ("isoniazid",       "The tingling in my hands and feet started about a month after I began the TB treatment — I had no neuropathy before starting isoniazid."),
    ("amlodipine",      "My ankles have been swelling more and more since I started amlodipine three weeks ago — they were fine before the blood pressure tablet."),
    ("haloperidol",     "Since they gave me the haloperidol injection last week I cannot sit still — my legs keep moving and I have to pace up and down all the time."),
    ("co-trimoxazole",  "After starting co-trimoxazole my child developed yellowing of the eyes. This wasn't there before the antibiotic."),
    ("rifampicin",      "Since starting rifampicin my urine is orange-red — I know this is the drug, it wasn't like this before I started the TB treatment."),
    ("tenofovir",       "My creatinine has been rising steadily since starting tenofovir — it was normal before I switched to this ARV combination."),
    ("quinine",         "Since starting the quinine drip I've been having severe hypoglycaemia episodes — three times in the past 24 hours and it wasn't happening before the quinine."),
    ("diclofenac",      "I've been having black stools for the past three days — since I started the diclofenac for my knee. No bleeding before the NSAID."),
    ("carbamazepine",   "Two weeks after starting carbamazepine I developed double vision and dizziness that gets worse when I stand — these are new symptoms since the tablet."),
    ("doxycycline",     "Since starting the doxycycline my oesophagus has been burning terribly — especially in the morning when I take the tablet lying in bed."),
    ("artemether-lumefantrine", "After the second dose of artemether-lumefantrine I started having heart palpitations. I haven't had this before and it started within hours of the dose."),
    ("phenytoin",       "My gums have been enlarging and bleeding since I started phenytoin — I can see they are swollen compared to before the anti-epileptic."),
    ("glibenclamide",   "I keep having episodes of shaking and sweating since I started glibenclamide — my blood sugar drops very low and I've had three hypoglycaemic episodes this week."),
    ("clomiphene",      "Since starting clomiphene I've been having blurred vision and seeing coloured halos — this started on day 3 of the tablet and wasn't there before."),
    ("metoclopramide",  "After the metoclopramide injection my neck stiffened and my eyes went upwards — it was frightening, it happened within minutes of the injection."),
]

P19_NON_ADR = [
    # Remote past tense — ADR resolved, no longer active
    ("metformin",       "I had diarrhoea when I first started metformin three years ago, but it went away after two weeks. I've had no problem with it since then."),
    ("atorvastatin",    "When I started the statin five years ago I had some muscle aches for the first few weeks, but those resolved and I've been taking it without any trouble ever since."),
    ("cotrimoxazole",   "I had a rash from cotrimoxazole when I was twelve. I'm now 34 and I've never taken it since. I do not currently have any symptoms from that drug."),
    ("efavirenz",       "The bad dreams were terrible in the first month of efavirenz, back in 2019. They settled completely and I've had no issues with it for the past four years."),
    ("lisinopril",      "The cough from lisinopril resolved when I switched to an ARB two years ago. I've been on amlodipine since then with no cough."),
    ("isoniazid",       "I had neuropathy from isoniazid during my TB treatment last year, but it fully resolved three months after completing the course. I have no symptoms now."),
    # Negative dechallenge — stopped drug, symptoms continued = drug not cause
    ("amlodipine",      "I stopped taking amlodipine two months ago thinking it was causing the swelling, but the ankle oedema has continued exactly the same as before. Clearly not the amlodipine."),
    ("metformin",       "We stopped the metformin to see if it was causing the abdominal pain. Three weeks off the tablet and the pain is just the same. The metformin is not responsible."),
    ("atorvastatin",    "The muscle pains continued for four weeks after stopping atorvastatin — unchanged in character and severity. The statin is not the cause. The cause is being investigated."),
    ("efavirenz",       "She discontinued efavirenz six weeks ago. The psychiatric symptoms have not improved at all since then — ruling out efavirenz as the cause."),
    ("haloperidol",     "The tremor persisted unchanged for three months after the haloperidol was stopped. It is a Parkinsonian tremor from an underlying movement disorder, not haloperidol-induced."),
    ("carbamazepine",   "I stopped carbamazepine completely two months ago — the skin rash is still here, unchanged. Dermatology confirmed it's lichen planus, not a drug rash."),
    # Negative rechallenge — restarted drug, no recurrence = original event not drug
    ("amoxicillin",     "He had a rash once attributed to amoxicillin in 2018. We re-challenged with a full course this admission and he tolerated it perfectly with no reaction."),
    ("diclofenac",      "After investigating, we cautiously re-prescribed diclofenac at a lower dose. She has been taking it for three weeks without any recurrence of the stomach symptoms."),
    ("cotrimoxazole",   "The previous urticaria when he took cotrimoxazole was re-evaluated — drug provocation test was negative. He is now tolerating cotrimoxazole without any reaction."),
    ("isoniazid",       "Following recovery from the liver enzyme elevation, isoniazid was cautiously re-introduced at half dose. Liver function has remained completely normal for 8 weeks of rechallenge."),
    ("metformin",       "She had mild nausea previously attributed to metformin. We restarted at the same dose after three months off — she's been tolerating it for six weeks with no symptoms."),
    # Symptom predates drug start
    ("glibenclamide",   "The patient's hypoglycaemic episodes were documented two years before glibenclamide was prescribed — he had an insulinoma. The drug had nothing to do with those events."),
    ("lisinopril",      "The dry cough is listed in the notes from 2015 — long before lisinopril was first prescribed in 2021. This is a chronic condition, not caused by the ACEi."),
    ("haloperidol",     "The movement disorder was present on admission — documented as tremor predating all current medications by 3 years. Haloperidol was started after the movement disorder was already established."),
    # Future — symptom not yet present, drug not yet causing it
    ("metformin",       "She is being started on metformin today. There are no symptoms related to the drug at this time. Follow-up in two weeks to assess tolerability."),
    ("efavirenz",       "ART with efavirenz is being initiated today. Neuropsychiatric side effects may develop — the patient has been counselled to report vivid dreams or mood changes if they occur."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 20 — Contradictory Signal FPs (Objective Normalises)
# Addresses batch failures: #145, #148
# Patient reports subjective complaint but objective investigations are normal.
# ════════════════════════════════════════════════════════════════════════════

P20_NON_ADR = [
    ("atorvastatin",    "I feel tired all the time since starting the statin, but my haemoglobin is 13.4, my thyroid is completely normal, and my CK is 142 U/L — all within range. There is no biochemical evidence of a drug effect."),
    ("metformin",       "She complains of pain in her legs since starting metformin, but the nerve conduction studies are completely normal and her B12 is 410 pmol/L — no evidence of neuropathy. The pain is functional."),
    ("lisinopril",      "He reports a cough which he attributes to lisinopril, but on objective assessment he has no wheeze, spirometry is completely normal, and the cough has been present on review of old notes from before the ACEi was started."),
    ("co-trimoxazole",  "She reports a rash that she thinks is from the co-trimoxazole, but on examination the skin is entirely normal — no erythema, no urticaria, no pruritus. No objective rash found."),
    ("efavirenz",       "He reports memory problems since starting efavirenz, but formal cognitive testing (MoCA 28/30) is completely normal. Neuropsychological assessment found no deficit attributable to the medication."),
    ("amlodipine",      "She feels her ankles are swelling, but on examination there is no pitting oedema and the ankle circumference measurements are unchanged from the baseline visit. No objective evidence of ankle oedema."),
    ("rifampicin",      "He reports nausea with the rifampicin but his weight is stable, he is eating normally, and his electrolytes and liver function tests are all within normal range. No objective evidence of drug-induced toxicity."),
    ("tenofovir",       "She reports kidney pain she attributes to tenofovir, but creatinine is 72 µmol/L (stable), eGFR 88, dipstick urinalysis negative. Renal ultrasound normal. No objective evidence of nephrotoxicity."),
    ("haloperidol",     "She reports tremor from the haloperidol, but on examination there is no resting tremor, no rigidity, and normal AIMS score. No objective extrapyramidal signs detected."),
    ("isoniazid",       "He reports tingling in his hands attributed to INH neuropathy, but monofilament testing is normal at all sites, vibration is intact bilaterally, and nerve conduction velocity is within normal range."),
    ("cotrimoxazole",   "She reports breathlessness she attributes to co-trimoxazole, but pulse oximetry is 99% on air, respiratory rate 14, peak flow 480 L/min (at predicted), and auscultation is entirely clear. No objective respiratory compromise."),
    ("atorvastatin",    "The patient reports muscle weakness from the statin, but manual muscle testing reveals 5/5 power throughout, CK is 210 U/L (upper limit of normal for his age), and EMG is normal. Functional overlay considered."),
    ("glibenclamide",   "She is concerned about hypoglycaemia from glibenclamide, but glucose monitoring log shows 18 readings over 2 weeks with a range of 5.1–9.3 mmol/L — no documented episode below 4.0 mmol/L."),
    ("metformin",       "She reports worsening of her gastrointestinal symptoms on metformin, but stool diary and bowel frequency are unchanged from pre-treatment baseline — average 1.8 bowel movements per day both before and after starting."),
    ("diclofenac",      "He reports heartburn from the diclofenac, but endoscopy showed completely normal gastric and oesophageal mucosa. No gastritis, no ulceration, no erosions despite two months of diclofenac use."),
    ("amlodipine",      "She reports flushing and facial redness that she thinks is from amlodipine, but on examination skin colouration is normal, blood pressure is well-controlled, and rosacea screen by dermatology is the more likely diagnosis."),
    ("efavirenz",       "He says efavirenz is affecting his heart — palpitations and chest tightness. ECG is normal sinus rhythm, 24-hour Holter shows no arrhythmia, troponin negative. Cardiac cause excluded. Somatic anxiety disorder referred to psychiatry."),
    ("rifampicin",      "She attributes her hair thinning to the rifampicin, but dermatology examination shows a normal scalp with no alopecia pattern, and hair pull test is negative. Hair density is within the normal range for her age."),
    ("carbamazepine",   "He says his balance is off since starting carbamazepine, but formal neurological assessment shows normal gait, Romberg negative, coordination intact, nystagmus absent. Carbamazepine level is within the low therapeutic range."),
    ("isoniazid",       "She reports visual changes from isoniazid, but ophthalmology assessment shows visual acuity 6/6 bilaterally, colour vision normal, optic disc and retina normal. No evidence of isoniazid optic neuropathy."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 21 — Asymptomatic Lab ADR
# Addresses batch failure: #20
# ADR is a laboratory abnormality — patient has no subjective symptoms.
# ════════════════════════════════════════════════════════════════════════════

P21_ADR = [
    ("tenofovir",       "I feel completely fine, but my creatinine has risen to 180 µmol/L since starting tenofovir. The doctor says the drug is causing kidney damage even though I have no symptoms."),
    ("isoniazid",       "I have no symptoms at all — no pain, no nausea — but my ALT came back at 3 times the upper limit of normal on my monthly liver function test. The nurse says the TB tablet is affecting my liver."),
    ("atorvastatin",    "I feel perfectly well, but my CK came back at 2,400 U/L on routine monitoring — almost 5 times the upper limit. The doctor says this is a statin-related muscle effect even though I'm not in any pain."),
    ("methotrexate",    "No symptoms at all, but my FBC showed neutrophils of 1.0 × 10⁹/L on routine testing. The rheumatologist says the methotrexate is suppressing my bone marrow even though I feel fine."),
    ("glibenclamide",   "I feel completely well, but the glucometer caught a blood sugar of 2.4 mmol/L without any symptoms. The doctor says this is a silent hypoglycaemic episode from the glibenclamide."),
    ("co-trimoxazole",  "No complaints at all from the patient, but the full blood count shows haemoglobin dropped from 13.2 to 9.8 g/dL in 6 weeks. The doctor diagnosed haemolysis from co-trimoxazole — asymptomatic at this stage."),
    ("rifampicin",      "He feels well and reports no symptoms, but his bilirubin has risen from 8 to 42 µmol/L over 4 weeks. The physician says rifampicin-related hepatotoxicity is developing subclinically."),
    ("amiodarone",      "She has no symptoms and feels fine, but her TSH has become suppressed at 0.01 mIU/L. The endocrinologist says the amiodarone is causing subclinical thyrotoxicosis even without symptoms."),
    ("ciclosporin",     "The patient feels well, but creatinine has risen from 100 to 175 µmol/L. The transplant team says this silent renal deterioration is consistent with calcineurin inhibitor nephrotoxicity and requires dose adjustment."),
    ("dapsone",         "No symptoms reported, but the full blood count showed methaemoglobin levels at 8% on routine monitoring — a subclinical but clinically relevant dapsone-induced methaemoglobinaemia."),
    ("phenytoin",       "He is asymptomatic and feels well, but his phenytoin level came back at 28 µg/mL — in the toxic range. The team says the drug level itself represents a measurable adverse effect requiring dose reduction."),
    ("warfarin",        "No bleeding symptoms, but INR returned at 7.8 on routine monitoring. The pharmacist says the supratherapeutic INR is a drug effect even without current bleeding — it requires urgent reversal."),
    ("lithium",         "The patient reports feeling well, but serum lithium was 1.6 mmol/L — above the therapeutic range. No overt toxicity symptoms yet, but this concentration itself constitutes a lithium ADR requiring dose reduction."),
    ("metformin",       "She has no gastrointestinal complaints and feels fine, but her B12 level was 108 pmol/L — deficient. The doctor says long-term metformin has caused subclinical B12 malabsorption without symptoms."),
    ("amoxicillin-clavulanate", "He feels completely well, but LFTs from routine post-treatment check showed ALP at 4x ULN, bilirubin slightly raised. Clavulanate-associated cholestasis confirmed on rechallenge history — asymptomatic hepatotoxicity."),
    ("hydroxychloroquine", "No visual symptoms at all, but OCT screening at her annual review showed early bull's-eye maculopathy. The ophthalmologist says hydroxychloroquine is causing retinal changes before she can notice any visual loss."),
    ("gentamicin",      "He reports no hearing changes, but audiometry at day 7 of gentamicin therapy shows 20 dB threshold shift at 8 kHz — early high-frequency ototoxicity detectable on testing before any subjective hearing complaint."),
    ("furosemide",      "She feels fine but routine U&E shows potassium of 2.6 mmol/L — asymptomatic hypokalaemia from furosemide requiring oral potassium supplementation despite absence of cardiac symptoms."),
    ("spironolactone",  "No symptoms, but potassium returned at 6.4 mmol/L — hyperkalaemia from spironolactone. ECG monitoring initiated despite the patient reporting he feels completely normal."),
    ("chloroquine",     "She reports no visual symptoms, but the annual visual field test shows a small paracentral scotoma not present last year. The ophthalmologist says early chloroquine retinopathy is developing silently."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 22 — Severity Minimisation FNs
# Addresses batch failure: #77
# Patient downplays real ADR with minimising language.
# ════════════════════════════════════════════════════════════════════════════

P22_ADR = [
    ("metformin",       "The nausea is manageable — just a bit of queasiness every morning when I take the metformin. It's not a big deal but it's there every single day since I started."),
    ("co-trimoxazole",  "There's a rash on my arms — it's not too bad, just a few spots from the cotrimoxazole. I'm coping with it but I thought I should mention it since it wasn't there before."),
    ("efavirenz",       "The dreams are strange but I can live with them — very vivid and sometimes disturbing since I started the efavirenz, but it's manageable compared to not treating the virus."),
    ("amlodipine",      "My ankles are a little puffy — nothing dramatic, just mild swelling from the blood pressure tablet. I haven't needed to change my shoes but it's new since I started the medication."),
    ("isoniazid",       "There's a mild tingling in my feet — not painful, just there constantly since the TB tablets. I cope with it, it's not stopping me doing anything, but it's been there for two months now."),
    ("diclofenac",      "I get a bit of heartburn after taking the tablets — nothing serious, I just bear it and take the diclofenac with food. But it's there every time since I started."),
    ("haloperidol",     "My hands shake a bit since the injection — it's manageable, I can still do everything, but there's definitely a new tremor that wasn't there before the haloperidol."),
    ("atorvastatin",    "There's some aching in my legs — mild, nothing I can't handle. I take it as part of getting older, but it did start after the cholesterol tablet and it is new."),
    ("rifampicin",      "My urine is orange-red every time since starting the TB treatment. I've gotten used to it — it's not painful or anything, just discoloured. But it's definitely from the drug."),
    ("lisinopril",      "There's a tickly cough — not bad at all, just annoying. I cope with it, I only notice it at night really, but it's been there every night since starting the blood pressure tablet."),
    ("doxycycline",     "A slight burning in my chest when I take the doxycycline — mild enough that I didn't think to mention it. But it is there every morning and I suppose it is a reaction to the tablet."),
    ("co-trimoxazole",  "The itching is mild — I wouldn't call it a rash exactly, just some itchiness on my back since starting the antibiotic. I'm managing fine but it's definitely new."),
    ("artemether-lumefantrine", "A bit of nausea and dizziness after each dose — I manage it by lying down for a while. Not severe at all but it's there with every tablet of the malaria medicine."),
    ("metformin",       "Just a loose stool in the morning — I cope with it, it's not urgent diarrhoea, just softer than normal since I started the diabetes tablet. I didn't think it was worth mentioning."),
    ("efavirenz",       "The dizziness in the morning is mild — I sit on the bed for a minute before standing and it passes. I've adjusted my routine around it but it is from the efavirenz, it started the first week."),
    ("glibenclamide",   "I feel a bit shaky in the mid-morning — mild, it passes when I eat a biscuit. I cope fine. But I check my sugar and it's always around 3.2 mmol/L at those times — it is a real low from the tablet."),
    ("amlodipine",      "My face gets a bit flushed and warm after taking the tablet — I can live with it, it only lasts about thirty minutes. But it happens every day since the blood pressure medicine."),
    ("haloperidol",     "I feel restless sometimes — a bit hard to sit still, but manageable. I wouldn't say it's severe. But since the depot injection I do feel the need to move, it's new."),
    ("tenofovir",       "Just a bit of bone aching — not severe at all. I take paracetamol and I'm fine. But it started after the ARV change to tenofovir and it is new."),
    ("isoniazid",       "There's a slight yellow tinge to my eyes — mild, my wife noticed it first. I feel fine in myself. But the nurse checked my bilirubin and it's mildly elevated from the TB tablet."),
    ("methotrexate",    "The mouth ulcers are small — just two or three, manageable. I use the mouthwash and they heal. But they come back with every weekly methotrexate dose."),
    ("carbamazepine",   "A little bit of double vision when I look to the side — not always there, just sometimes. I can live with it but it is new since the anticonvulsant and my doctor should know."),
    ("doxycycline",     "My skin catches the sun easily since starting the tablet — just mild sunburn if I'm outside for even a short time. I now wear a hat and manage fine, but the photosensitivity is real and new."),
    ("cotrimoxazole",   "A mild headache every day since starting the antibiotic — I take paracetamol and it goes. Not severe enough to worry about, but it's been there every day for a week and it's from the co-trimoxazole."),
    ("glibenclamide",   "I get a bit dizzy before lunch — mild. It passes with a glucose tablet. I'm managing. But it's a hypoglycaemic episode from the glibenclamide and it happens every few days."),
    ("atorvastatin",    "My liver enzymes were borderline raised — twice the upper limit — but I feel absolutely fine. The doctor says it's a mild statin effect and we're watching it. Still, it is an ADR even if I feel well."),
    ("metformin",       "There's a metallic taste in my mouth since I started the tablet — a bit unpleasant but I cope with it. It's there all the time. It's from the metformin."),
    ("amoxicillin",     "Just a small red area on my forearm — I'm not sure if it counts as a rash. It's not spreading, not bothering me much. But it appeared on day 3 of the antibiotic and it is new."),
    ("haloperidol",     "A mild stiffness in my neck since the injection — not painful, just a bit stiff. I can manage. But it's new since the haloperidol and I should report it to the clinic."),
    ("rifampicin",      "My tears are slightly orange-pink when I cry — I only notice on the tissue. It's strange but not distressing. It's the rifampicin staining my secretions and it's a real drug effect."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 23 — Postpartum Psychiatric (Extended)
# Extends P7 coverage from v2. Addresses failures: #255, #266.
# Focus: postpartum psychosis, delirium, drug-induced psychiatric ADR
# in postpartum period.
# ════════════════════════════════════════════════════════════════════════════

P23_ADR = [
    ("methyldopa",      "After starting methyldopa for postpartum hypertension I began hearing voices at night — the voices were not there before the medication. The psychiatrist confirmed this was a drug-related psychosis."),
    ("bromocriptine",   "After bromocriptine was started to suppress lactation she became acutely psychotic — she believed strangers were entering the house and refused to care for her baby. It was the bromocriptine."),
    ("corticosteroids", "She was given high-dose steroids for postpartum inflammatory arthritis and within 48 hours developed frank psychosis — paranoia, auditory hallucinations. Steroid-induced psychosis confirmed."),
    ("metoclopramide",  "After metoclopramide for postpartum nausea she developed acute dystonic reaction with oculogyric crisis — eyes fixed upward, unable to look down. Emergency treatment required."),
    ("methyldopa",      "Since starting methyldopa after delivery she has been profoundly depressed and cannot bond with her baby. The obstetrician believes the methyldopa is contributing to her postpartum depression."),
    ("bromocriptine",   "Three days after starting bromocriptine she had a grand mal seizure. There was no prior history of epilepsy. The seizure was attributed to bromocriptine — the drug is not recommended postpartum."),
    ("opioid analgesic","After postpartum opioid analgesia she became delirious — confused, agitated, hallucinating. The opioid was the precipitant of the delirium in the postpartum period."),
    ("magnesium sulfate", "She remained drowsy and confused 24 hours after the magnesium infusion was stopped — magnesium toxicity was confirmed by serum level 4.2 mmol/L. The altered consciousness was a drug ADR."),
    ("methyldopa",      "Since methyldopa was started for gestational hypertension she describes feeling like she is 'not herself' — depersonalised, watching herself from outside. This is a recognised methyldopa psychiatric effect."),
    ("nifedipine",      "After the postpartum nifedipine dose she became extremely dizzy and had a syncopal episode — hypotension from the calcium channel blocker precipitated a near-collapse in the immediate postpartum period."),
    ("corticosteroids", "High-dose corticosteroids for postpartum lupus flare induced hypomania — she became elated, didn't sleep for two nights, was unusually energetic. Steroid-induced mood disturbance confirmed by psychiatry."),
    ("ergometrine",     "Ergometrine given postpartum for haemorrhage management caused a severe hypertensive emergency — blood pressure 210/130. She had a severe headache and photophobia. The drug was causative."),
    ("codeine",         "She was prescribed codeine for postpartum perineal pain and developed neonatal respiratory depression in the breastfed infant — rapid metaboliser phenotype causing high morphine levels in breast milk."),
    ("haloperidol",     "Started haloperidol for postpartum psychosis — she developed acute akathisia within 48 hours, unable to sit still, extremely distressed. The akathisia was from the haloperidol."),
    ("lithium",         "Lithium was restarted in the postpartum period — she developed lithium toxicity with confusion and ataxia because her renal clearance had not returned to pre-pregnancy levels. Toxic drug level confirmed."),
    ("fluoxetine",      "Fluoxetine was started for postpartum depression. Within two weeks she developed hypomania — elevated mood, decreased sleep need, pressured speech. SSRI-induced hypomanic switch confirmed."),
    ("ondansetron",     "Ondansetron given for postpartum nausea caused QTc prolongation to 510 ms — documented on ECG monitoring. The prolonged QTc is an adverse drug effect."),
    ("clonidine",       "After clonidine for postpartum hypertension she became profoundly sedated and unable to care for her baby — excessive CNS depression from the alpha-2 agonist."),
    ("misoprostol",     "Following misoprostol administration she developed high fever, rigors, and tachycardia — pyrexia as an adverse effect of misoprostol."),
    ("oxytocin",        "Prolonged high-dose oxytocin infusion led to hyponatraemia — sodium 122 mmol/L. She became confused and had a brief seizure. The antidiuretic effect of oxytocin at high doses caused the electrolyte disturbance."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 24 — Vaccine Coincidental Illness FPs (Extended)
# Extends P6 AEFI coverage from v2. Addresses failures: #276, #293, #296.
# More family cluster / community outbreak scenarios.
# ════════════════════════════════════════════════════════════════════════════

P24_NON_ADR = [
    # Family cluster illness (same illness in unvaccinated family members)
    ("pentavalent vaccine", "My baby had diarrhoea after the pentavalent vaccination, but my two-year-old who was not vaccinated and my husband both had the same diarrhoea the same week. It was a family gastroenteritis going around, not from the vaccine."),
    ("measles-rubella vaccine", "My daughter developed a runny nose and mild fever three days after her MR vaccine, but her older brother who hadn't been vaccinated had the same cold that week. It was a seasonal viral illness, not a vaccine reaction."),
    ("COVID-19 vaccine",  "I had a fever and cough three days after the COVID vaccine. My husband, who hasn't been vaccinated at all, came down with the exact same symptoms the day after me. It was definitely a family respiratory illness."),
    ("BCG vaccine",       "The baby had a slight fever the week after BCG. But the grandmother, the father, and the older sibling — none of them vaccinated for BCG recently — all had the same fever that week. Community viral illness confirmed by the clinic."),
    ("yellow fever vaccine", "Both my children had the yellow fever vaccine. Only one had fever afterwards — but three of their classmates who were NOT vaccinated also had fever the same week from a viral infection going around their school."),
    ("pentavalent vaccine", "Four babies at the same health facility received the pentavalent vaccine on the same day. All four had mild fever in the next two days — but so did three unvaccinated babies from the same community. Community URTI confirmed."),
    ("HPV vaccine",       "My daughter had headaches and fatigue after her HPV vaccine. Her best friend, who hadn't received any vaccine recently, had the same symptoms the same week. Their whole class had a viral illness."),
    ("COVID-19 vaccine",  "She had a sore throat and mild fever starting 4 days post-vaccination. But the illness was already present in her household — two family members were sick before she received the vaccine. The illness predated her vaccination."),
    # Community outbreak coinciding with vaccination campaign
    ("OPV vaccine",       "During the polio vaccination campaign in the district, a gastroenteritis outbreak occurred affecting both vaccinated and unvaccinated children equally. The diarrhoea cases are not from the oral polio vaccine."),
    ("measles-rubella vaccine", "A measles-rubella vaccination campaign ran the same week as a dengue fever outbreak in the community. The fever cases presenting after vaccination are from dengue, not from the vaccine — dengue NS1 confirmed."),
    ("meningococcal vaccine", "The meningococcal vaccination campaign coincided with a community URTI outbreak. Both vaccinated and unvaccinated individuals developed respiratory symptoms — the illness is from the community respiratory pathogen, not the meningococcal vaccine."),
    ("COVID-19 vaccine",  "A rotavirus outbreak was occurring in the district during the COVID-19 vaccination campaign. Children presenting with vomiting and diarrhoea had rotavirus PCR positive — not a COVID vaccine adverse event."),
    ("pentavalent vaccine", "A malaria transmission peak coincided with the EPI vaccination dates in the northern region. Children with fever after vaccination tested positive for malaria — the fever is from malaria, not the pentavalent vaccine."),
    ("yellow fever vaccine", "During the yellow fever catch-up campaign, a community diarrhoea outbreak was identified in the same catchment area. Stool cultures grew Salmonella — the febrile illness with diarrhoea is not from the yellow fever vaccine."),
    ("BCG vaccine",       "The BCG mass vaccination event for newborns coincided with a neonatal URTI cluster from respiratory syncytial virus circulating in the postnatal ward. The respiratory symptoms are from RSV, not BCG."),
    # Illness onset before vaccination or outside vaccine window
    ("pentavalent vaccine", "The mother reported cough and fever after vaccination, but the clinic notes show the illness was documented at the pre-vaccination check before the injection was given. The symptoms predated vaccination."),
    ("COVID-19 vaccine",  "He developed fever 12 days after the COVID vaccine — well outside the expected 1–3 day window for vaccine reactions. Investigation confirmed influenza A. The illness is a coincidental influenza infection, not vaccine-related."),
    ("measles-rubella vaccine", "The rash appeared on day 1 post-vaccination — too early for a vaccine-related reaction (expected vaccine rash occurs days 7–12 post-MR). The immediate rash is a coincidental viral exanthem, not from the vaccine."),
    ("meningococcal vaccine", "The seizure occurred 3 weeks after meningococcal vaccination — beyond the biological plausibility window for vaccine-related seizures. EEG confirmed a new-onset epilepsy unrelated to vaccination."),
    # Multiple vaccinations — other vaccine more likely or neither
    ("COVID-19 vaccine",  "She received both COVID and influenza vaccines on the same day. She developed flu-like illness — however, influenza rapid antigen test was positive. The illness is from the live attenuated influenza nasal spray, not the COVID vaccine."),
    ("hexavalent vaccine","He received hexavalent and rotavirus vaccines simultaneously. The diarrhoea that followed is characteristic of rotavirus gastroenteritis (explosive watery diarrhoea, rotavirus PCR positive) — coincidental acquisition, not from the hexavalent."),
    # Healthcare worker confirmation of coincidence
    ("pentavalent vaccine", "The nurse investigated the cluster of post-vaccination illnesses and confirmed via case finding that the same viral illness affected unvaccinated children in the same ward — this is a nosocomial viral outbreak, not an AEFI cluster."),
    ("BCG vaccine",       "Epidemiological investigation of the post-BCG fever cluster confirmed an RSV outbreak on the postnatal ward. All cases, including non-vaccinated babies in adjacent rooms, had RSV-positive nasopharyngeal swabs. BCG cleared."),
    ("COVID-19 vaccine",  "Follow-up of the post-vaccination illness cluster: contact tracing identified a superspreader event at a funeral one week before vaccine rollout. The illness presenting 5 days after vaccination was incubating before any vaccine was given."),
    ("yellow fever vaccine", "The AEFI investigation of post-yellow-fever-vaccination jaundice found a hepatitis E outbreak in the same water supply area. Hepatitis E IgM was positive in all cases. The jaundice is not from the vaccine."),
    # Temporal coincidence only, no biological link
    ("HPV vaccine",       "She was diagnosed with lupus one year after receiving the HPV vaccine series. Thorough investigation found no biological link — lupus was already developing based on retrospective symptom review before the first vaccine dose."),
    ("pentavalent vaccine","Onset of type 1 diabetes diagnosed 3 months after pentavalent vaccination. Islet autoantibodies were positive months before the vaccination based on stored samples. The diabetes is not caused by the vaccine."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 25 — Post-vaccination Syncope FNs
# Addresses batch failure: #291
# True AEFI: vasovagal syncope/collapse requiring medical attention.
# ════════════════════════════════════════════════════════════════════════════

P25_ADR = [
    ("COVID-19 vaccine",  "After the COVID vaccine injection he collapsed in the waiting area — he lost consciousness for about two minutes and fell to the ground. He was helped to a bed and observed for an hour before being discharged."),
    ("HPV vaccine",       "She fainted immediately after the HPV injection and fell from the chair, hitting her head on the floor. She required a CT scan to exclude intracranial injury — the fall caused a laceration."),
    ("meningococcal vaccine", "Three minutes after the meningococcal vaccine she lost consciousness completely. She was pale and sweating before collapsing — vasovagal syncope confirmed. She was kept lying down for 30 minutes."),
    ("COVID-19 vaccine",  "She felt faint, then collapsed onto the floor of the vaccination centre. She was unconscious for 90 seconds. Staff administered first aid and she recovered, but the episode was documented as a post-vaccination syncope AEFI."),
    ("tetanus vaccine",   "He passed out immediately after the tetanus injection — woke up on the floor surrounded by nurses. He had no warning. The collapse required emergency response and observation before discharge."),
    ("HPV vaccine",       "Post-HPV vaccination syncopal episode in a 14-year-old. She collapsed, was caught by a nurse who was present, and was unresponsive for about one minute. She had a minor head injury from the partial fall."),
    ("yellow fever vaccine", "He felt dizzy, began to sweat profusely, and then woke up on the floor. He had no memory of falling. The vaccination team documented it as a vasovagal syncopal episode — a recognised AEFI following any injection."),
    ("pentavalent vaccine", "The caregiver collapsed while holding the baby after the pentavalent injection — she fainted and the nurse quickly took the infant. She was treated for vasovagal syncope at the clinic."),
    ("COVID-19 vaccine",  "She described becoming very hot and then everything went dark — she woke up with clinic staff around her. The syncopal episode lasted approximately two minutes and occurred within 5 minutes of the COVID vaccine."),
    ("meningococcal vaccine", "He collapsed in the waiting room 8 minutes after the meningococcal vaccination. His blood pressure was 80/40 at that point. He was treated with IV fluids and supine positioning and recovered fully — vasovagal syncope."),
    ("BCG vaccine",       "The infant's father fainted during the BCG vaccination — a bystander syncopal response to witnessing the injection. He fell and sustained a wrist fracture from the fall. Documented as injection-associated syncope, caregiver."),
    ("COVID-19 vaccine",  "Within minutes of vaccination she felt her heart pounding and then collapsed — she had bitten her tongue in the fall. This was a serious AEFI requiring documentation: post-vaccination vasovagal syncope with injury."),
    ("tetanus toxoid",    "He received tetanus toxoid and within 3 minutes began to feel faint — we reclined the chair but he lost consciousness before we could do so fully. He was managed on site and monitored for 30 minutes."),
    ("HPV vaccine",       "She woke up on the floor of the clinic after her first HPV dose. No prodrome reported. A tonic-clonic movement was observed by the nurse for approximately 15 seconds during the syncopal episode — anoxic seizure during syncope."),
    ("COVID-19 vaccine",  "Collapse post-COVID vaccination — patient found on floor in cubicle. Pulse present, breathing spontaneously. Regained consciousness within 90 seconds. Injury: minor facial laceration from the fall. AEFI form completed."),
    ("meningococcal vaccine", "She sat up to receive the injection and within minutes said she couldn't see properly and then slumped in the chair — syncopal episode. She was placed in the recovery position and observed for one hour."),
    ("yellow fever vaccine", "He reported feeling unwell immediately post-injection, took two steps, and collapsed. He required oxygen administration and was kept lying flat for 45 minutes before being allowed to leave the facility."),
    ("HPV vaccine",       "A cluster of three syncopal episodes in a school-based HPV vaccination programme — three students fainted after the injection within a 30-minute period. All recovered without sequelae but were documented as vaccination-related AEFIs."),
    ("COVID-19 vaccine",  "Post-vaccination syncope with prolonged loss of consciousness of 5 minutes — she required ambulance transfer for observation. ECG and blood glucose were normal. Vasovagal syncope confirmed, prolonged episode."),
    ("tetanus vaccine",   "She collapsed in the chair immediately after the injection, slid to the floor before the nurse could catch her. She had a brief tonic-clonic seizure associated with the syncopal event. AEFI documented: syncope with anoxic seizure."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 26 — Polypharmacy Attribution Uncertainty
# Addresses batch failure: #127
# ════════════════════════════════════════════════════════════════════════════

P26_ADR = [
    ("multiple medications", "I don't know which tablet is causing my rash — I take five different medicines and the rash started two weeks after my cardiologist changed three of them at the same time."),
    ("multiple medications", "She was started on ramipril, atorvastatin, and amlodipine all at the same visit. Three weeks later she has a dry cough and ankle swelling — we need to work out which drug is responsible."),
    ("multiple medications", "My stomach has been terrible since they changed my medicines in January. They adjusted four different drugs at once and now I have daily nausea and vomiting. I don't know which one to blame."),
    ("multiple medications", "He was on ten tablets for his heart failure and diabetes. His muscle pains could be from the statin, the diuretic causing electrolyte loss, or the new antihypertensive — we cannot attribute it to a single drug."),
    ("multiple medications", "She was initiated on isoniazid, rifampicin, pyrazinamide, and ethambutol simultaneously. The liver enzymes are now elevated — standard practice is to attribute this to the combination, but we cannot isolate the causative agent."),
    ("multiple medications", "I'm on warfarin, antibiotics, and the new blood pressure tablet all at once. My INR has gone haywire — is it the antibiotic interaction, the new drug, or both? I have a bruise on my arm that appeared overnight."),
    ("multiple medications", "Her rash appeared two weeks after starting HAART — efavirenz, tenofovir, and lamivudine together. We don't know if it's the efavirenz, the tenofovir, or a combination hypersensitivity reaction."),
    ("multiple medications", "He takes metformin, glibenclamide, and lisinopril. His renal function has worsened — is it the NSAID his private doctor added, the ACE inhibitor, or the metformin in the context of renal impairment? Uncertain polypharmacy ADR."),
    ("multiple medications", "She was switched from first-line to second-line anti-TB drugs all at once after failure. The neuropathy she develops — is it from the linezolid, the cycloserine, or exacerbation of the pre-existing INH neuropathy?"),
    ("multiple medications", "I changed three medications last month. Now I have severe diarrhoea — is it the new metformin brand, the PPI that was added, or the iron supplement? I've been trying to work out which one to stop first."),
    ("multiple medications", "She is on prednisolone, azathioprine, and hydroxychloroquine for lupus. Her bone marrow suppression could be from the azathioprine, the lupus itself, or an interaction between all three agents."),
    ("multiple medications", "Multiple new drugs were started after his coronary bypass — aspirin, clopidogrel, atorvastatin, bisoprolol, ramipril. His new muscle weakness — could be statin, could be the beta-blocker, could be both. Causality is unclear."),
    ("multiple medications", "He's on six different antihypertensives and his potassium is dangerously low. Is it the furosemide, the thiazide, a combination effect, or his inadequate dietary intake? Attribution is uncertain across this complex regimen."),
    ("multiple medications", "She started azithromycin and haloperidol on the same day — both prolong the QT interval. The dangerous arrhythmia she developed three days later is from the additive drug interaction, but reporting requires attribution."),
    ("multiple medications", "He takes eight tablets and cannot identify which one gives him the metallic taste. It could be metronidazole, metformin, or the zinc supplement — I've told him to bring his full blister pack so we can review each one."),
    ("multiple medications", "Her pruritus started after we added ferrous sulphate, cotrimoxazole, and folate in the same prescription. We stopped cotrimoxazole first as the most likely cause, but the itch has only partially improved — may be iron too."),
    ("multiple medications", "Four drugs were changed on the same ward round. The patient is now confused. Is it the opioid, the corticosteroid, the antibiotic that crosses the blood-brain barrier, or the anticholinergic? Polypharmacy-induced delirium."),
    ("multiple medications", "She was discharged on twelve tablets. Three weeks later she presents with jaundice — which hepatotoxic drug is responsible? We have to do structured dechallenge and rechallenge one drug at a time to find the cause."),
    ("multiple medications", "He's on dual antiplatelet therapy and developed a GI bleed. Is it the aspirin, the clopidogrel, or the additive effect of both combined? Both carry bleeding risk; the clinical picture is of combined drug-induced bleeding."),
    ("multiple medications", "Three weeks after starting a new regimen she has a haemoglobin of 6.5 g/dL. Possible causes: cotrimoxazole haemolysis, zidovudine marrow suppression, rifampicin haemolysis — or a combination of all three in this TB-HIV co-infected patient."),
]


# ════════════════════════════════════════════════════════════════════════════
# Assembly
# ════════════════════════════════════════════════════════════════════════════

def build_records():
    records = []

    # P16 — FDA/regulatory register
    n = 1
    for drug, sent in P16_ADR:
        records.append(make(drug, sent, 16, "fda_reg_adr", n, True)); n += 1

    # P17 — Non-adherence / drug-not-started
    n = 1
    for drug, sent in P17_NON_ADR:
        records.append(make(drug, sent, 17, "nonadherence_nonadr", n, False)); n += 1

    # P18 — Investigation-exonerates
    n = 1
    for drug, sent in P18_NON_ADR:
        records.append(make(drug, sent, 18, "invx_exonerates_nonadr", n, False)); n += 1

    # P19 — Temporal minimal pairs extended
    n = 1
    for drug, sent in P19_ADR:
        records.append(make(drug, sent, 19, "temporal_adr", n, True)); n += 1
    n = 1
    for drug, sent in P19_NON_ADR:
        records.append(make(drug, sent, 19, "temporal_nonadr", n, False)); n += 1

    # P20 — Contradictory signal
    n = 1
    for drug, sent in P20_NON_ADR:
        records.append(make(drug, sent, 20, "contradict_nonadr", n, False)); n += 1

    # P21 — Asymptomatic lab ADR
    n = 1
    for drug, sent in P21_ADR:
        records.append(make(drug, sent, 21, "asymp_lab_adr", n, True)); n += 1

    # P22 — Severity minimisation
    n = 1
    for drug, sent in P22_ADR:
        records.append(make(drug, sent, 22, "sev_min_adr", n, True)); n += 1

    # P23 — Postpartum psychiatric
    n = 1
    for drug, sent in P23_ADR:
        records.append(make(drug, sent, 23, "postpartum_psych_adr", n, True)); n += 1

    # P24 — Vaccine coincidental illness
    n = 1
    for drug, sent in P24_NON_ADR:
        records.append(make(drug, sent, 24, "vacc_coincidental_nonadr", n, False)); n += 1

    # P25 — Post-vaccination syncope
    n = 1
    for drug, sent in P25_ADR:
        records.append(make(drug, sent, 25, "vacc_syncope_adr", n, True)); n += 1

    # P26 — Polypharmacy attribution
    n = 1
    for drug, sent in P26_ADR:
        records.append(make(drug, sent, 26, "polypharm_adr", n, True)); n += 1

    return records


def main():
    records = build_records()
    adr_count     = sum(1 for r in records if r["contains_adr"])
    non_adr_count = len(records) - adr_count

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} examples to {OUT}")
    print(f"  ADR:     {adr_count}")
    print(f"  Non-ADR: {non_adr_count}")

    from collections import Counter
    by_prio = Counter()
    for r in records:
        p = r["setid"].split("_")[2]
        by_prio[p] += 1
    print("\nPer-priority counts:")
    for p in sorted(by_prio, key=lambda x: int(x.lstrip("p"))):
        print(f"  {p}: {by_prio[p]}")

    print("\nNOTE: Use this data for CLF training ONLY.")
    print("      Do not pass to NER — adr_spans are empty.")


if __name__ == "__main__":
    main()
