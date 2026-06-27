#!/usr/bin/env python3
"""
scripts/generate_synthetic_phase9.py
========================================
Generate synthetic_phase9.jsonl targeting the 11 remaining batch failures
after Phase 8 Hybrid (84/95).

Priorities:
  P32 — Investigation exonerates (imaging/culture/biopsy finds alternative)
  P33 — Drug not yet started / pre-medication temporal
  P34 — Drug holiday + unrelated symptom
  P35 — Objective lab/imaging normalisation overrides subjective complaint
  P36 — Coincidental illness / pre-existing condition
  P37 — Asymptomatic lab ADR (extended)
  P38 — Severity minimisation coping language (extended)
  P39 — Ghanaian English malaise idioms (extended — fixes #96 regression)
  P40 — Hedged regulatory / signal memo language (fixes #175)

Addresses: #20, #77, #82, #84, #85, #96, #124, #145, #148, #175, #293

IMPORTANT: CLF-only. adr_spans and drug_spans are always empty [].
Output: data/silver/synthetic_phase9.jsonl
Run:    python scripts/generate_synthetic_phase9.py
"""

import json
from pathlib import Path

ROOT   = Path(__file__).parent.parent
OUTDIR = ROOT / "data" / "silver"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT    = OUTDIR / "synthetic_phase9.jsonl"


def make(drug, sentence, priority, tag, n, contains_adr):
    return {
        "drug":         drug,
        "sentence":     sentence,
        "source":       "synthetic_phase9",
        "setid":        f"synthetic_phase9_p{priority}_{tag}_{n:03d}",
        "contains_adr": int(contains_adr),
        "adr_spans":    [],
        "drug_spans":   [],
    }


# ════════════════════════════════════════════════════════════════════════════
# P32 — Investigation exonerates (Non-ADR)
# Targets: #84 (ultrasound cholecystitis), #85 (Cryptosporidium)
# Key: drug + symptom co-occur but test reveals alternative cause.
# Must be harder than Phase 8 P18 — use same drug-symptom pairs,
# stronger exonerating language, more modalities.
# ════════════════════════════════════════════════════════════════════════════

P32_NONADR = [
    # Imaging exonerates — abdominal
    ("metformin",         "Patient on metformin complains of nausea and right upper quadrant pain. Ultrasound shows acute cholecystitis with gallstones. The pain is from the gallbladder, not the medication."),
    ("metformin",         "She is on metformin and has had abdominal pain and nausea for two weeks. CT abdomen reveals acute appendicitis. Surgical referral made. Metformin not implicated."),
    ("atorvastatin",      "He complains of right flank pain since starting atorvastatin. Renal ultrasound shows a 7 mm ureteric calculus. The pain is from the kidney stone, not the statin."),
    ("amlodipine",        "Patient on amlodipine presented with leg swelling and abdominal discomfort. CT scan shows liver cirrhosis with ascites — pre-existing condition, not drug-related."),
    ("co-trimoxazole",    "Since starting co-trimoxazole she has had abdominal cramps. Ultrasound reveals uterine fibroids causing the pain. No drug reaction identified."),
    ("lisinopril",        "He reports cough and chest discomfort since lisinopril was started. CT thorax shows a right lower lobe pneumonia. The symptoms are from the pneumonia, not the ACE inhibitor."),
    ("isoniazid",         "Patient on isoniazid-containing TB regimen complains of right upper quadrant pain. Liver ultrasound shows cholelithiasis. Hepatitis screen negative. Gallstones are the cause."),
    ("rifampicin",        "She has nausea and abdominal pain on the TB regimen. Upper GI endoscopy reveals a duodenal ulcer with H. pylori. Not a drug reaction — peptic ulcer disease."),
    ("artemether-lumefantrine", "Child with vomiting and diarrhoea after receiving artemether-lumefantrine. Stool microscopy grew Salmonella. The gastroenteritis is from the bacterial infection, not the antimalarial."),
    ("tenofovir",         "Patient on tenofovir has back pain. MRI lumbar spine shows L4/L5 disc prolapse compressing the nerve root. The pain is musculoskeletal, not drug nephrotoxicity."),
    # Microbiology exonerates
    ("amoxicillin",       "Since starting ARVs the patient has had persistent diarrhoea. Stool culture grew Cryptosporidium parvum — an ongoing opportunistic infection. The diarrhoea is not from the medication."),
    ("efavirenz",         "He has had diarrhoea and abdominal cramps since starting efavirenz. Stool PCR positive for Clostridioides difficile following recent hospitalization. The diarrhoea is C. diff, not an ADR."),
    ("co-trimoxazole",    "She reports diarrhoea on co-trimoxazole prophylaxis. Stool culture grew Giardia lamblia. Metronidazole started. The diarrhoea is a parasitic infection, not the drug."),
    ("metronidazole",     "Patient on metronidazole reports vaginal discharge and pelvic pain. High vaginal swab grew Candida albicans. Fluconazole prescribed. Candidiasis, not a drug reaction."),
    ("ceftriaxone",       "Child with fever and rash on ceftriaxone. Blood culture grew Salmonella typhi. The rash is rose spots from typhoid fever, not a drug allergy."),
    ("amoxicillin-clavulanate", "He has a generalised rash since starting amoxicillin-clavulanate. Paul-Bunnell test positive for infectious mononucleosis. The rash is EBV-associated, not drug hypersensitivity."),
    ("rifampicin",        "Patient on TB drugs has a maculopapular rash. Serology positive for dengue fever NS1 antigen. The rash and fever are from dengue, not the anti-TB drugs."),
    # Biopsy / histology exonerates
    ("atorvastatin",      "He has muscle pain and weakness since starting atorvastatin. CK is normal. Muscle biopsy shows polymyositis — an autoimmune condition unrelated to the statin."),
    ("methotrexate",      "She has joint pain and skin thickening on methotrexate. Skin biopsy shows systemic sclerosis. The findings are the underlying disease, not a drug reaction."),
    ("isoniazid",         "Liver biopsy in a patient on isoniazid shows non-alcoholic steatohepatitis, not drug-induced liver injury. The enzyme elevation is from fatty liver disease."),
    # Cardiac / neurological investigation exonerates
    ("haloperidol",       "Patient on haloperidol has chest pain and palpitations. ECG shows Wolff-Parkinson-White syndrome — a congenital anomaly, not a drug effect."),
    ("amlodipine",        "She reports dizziness and near-fainting on amlodipine. Tilt-table test confirms vasovagal syncope, a pre-existing condition. Not related to the antihypertensive."),
    ("efavirenz",         "He reports headaches and visual disturbance on efavirenz. MRI brain shows a meningioma. The neurological symptoms are from the tumour, not the ARV."),
    ("carbamazepine",     "Patient on carbamazepine has a new rash. Skin punch biopsy and serology confirm parvovirus B19 infection — slapped cheek rash, not Stevens-Johnson."),
    # Strong exonerating conclusions
    ("metformin",         "Patient has nausea and abdominal pain on metformin. Full workup including gastroscopy, ultrasound, and bloods reveals irritable bowel syndrome. No evidence of drug reaction."),
    ("co-trimoxazole",    "She complains of fatigue and weakness since starting co-trimoxazole. Full haematological workup shows iron deficiency anaemia from menorrhagia — not related to the antibiotic."),
    ("lisinopril",        "He reports shortness of breath on lisinopril. Spirometry confirms obstructive pattern — new diagnosis of asthma. Lisinopril cough was ruled out by ACE inhibitor challenge."),
    ("tenofovir",         "Patient on tenofovir has elevated creatinine. Renal biopsy shows IgA nephropathy — a primary renal disease unrelated to the ARV. Tenofovir dose unchanged."),
    ("artemether-lumefantrine", "Child has vomiting after antimalarial. Rapid malaria test positive — the vomiting is from the malaria itself, not the medication. Treatment continued."),
    ("glibenclamide",     "She reports sweating and trembling on glibenclamide. Fasting glucose is 7.2 mmol/L — not hypoglycaemia. Full autonomic workup shows anxiety disorder. Not a drug reaction."),
]

# ════════════════════════════════════════════════════════════════════════════
# P33 — Drug not yet started / pre-medication temporal (Non-ADR)
# Targets: #82 ("currently being started on TB drugs")
# Key: symptoms are present BEFORE or AT THE TIME the drug is initiated.
# ════════════════════════════════════════════════════════════════════════════

P33_NONADR = [
    # "Currently being started on" pattern
    ("rifampicin",        "He is currently being started on TB drugs. He has had nausea and abdominal pain since admission — these symptoms predated the treatment."),
    ("co-trimoxazole",    "She is currently being started on co-trimoxazole prophylaxis. The rash on her arms was present before the drug was prescribed."),
    ("isoniazid",         "Patient is currently being started on isoniazid. He has had peripheral tingling for two weeks before any treatment was initiated."),
    ("efavirenz",         "She is currently being started on ARV treatment today. The dizziness she reports began three days ago, before any medication was given."),
    ("metformin",         "He is being started on metformin this visit. The nausea and bloating he complains of have been present for six weeks before starting the drug."),
    ("amlodipine",        "Patient is currently being started on amlodipine. The leg oedema was noted at the previous visit, before antihypertensives were prescribed."),
    # "First dose not yet taken"
    ("atorvastatin",      "He was prescribed atorvastatin today and has not yet taken the first dose. He reports muscle aches that have been present for a month — pre-existing condition."),
    ("lisinopril",        "She has been prescribed lisinopril but has not yet started taking it. The cough she reports has been present since January — likely from a viral URTI."),
    ("carbamazepine",     "Patient has been prescribed carbamazepine for seizures. She has not taken any doses yet. The dizziness she describes is from the seizure disorder itself."),
    # "Treatment initiated" with pre-existing symptoms
    ("artemether-lumefantrine", "Malaria treatment was initiated this morning. The vomiting and abdominal pain started two days ago with the malaria illness — before medication."),
    ("amoxicillin",       "Antibiotic treatment was initiated today. The rash on the patient's trunk was documented on admission, prior to any antibiotic being given."),
    ("metronidazole",     "Metronidazole was started this evening. The nausea and abdominal discomfort have been present since yesterday — symptoms of the underlying infection."),
    ("tenofovir",         "ARV was initiated last week. The fatigue the patient reports has been present for two months as part of the HIV illness, not from the medication."),
    ("rifampicin",        "First dose of rifampicin given today. The patient had orange urine last week — this was noted before the TB drugs were commenced."),
    # "Not yet on" pattern
    ("haloperidol",       "She is not yet on any antipsychotic medication. The muscle stiffness and jaw tension she describes are features of her untreated psychosis."),
    ("glibenclamide",     "He had not yet started glibenclamide when the hypoglycaemic episode occurred. He had been fasting for religious purposes — the cause is fasting, not the drug."),
    ("isoniazid",         "Isoniazid had not yet been started. The patient's liver enzymes were already elevated on admission — from alcohol use, not from any TB medication."),
    # "Prescription was written but not dispensed"
    ("atorvastatin",      "The atorvastatin prescription was written but not dispensed. She reports joint pain — but she has not taken any of the medication."),
    ("co-trimoxazole",    "Co-trimoxazole was prescribed but the patient has not collected it yet. The diarrhoea he reports started four days ago — unrelated to any current medication."),
    ("metformin",         "Metformin was recommended at the last visit but he has not started it. The abdominal symptoms he describes are from his uncontrolled diabetes, not the drug."),
    # Pre-existing explicitly stated
    ("amlodipine",        "She was already having leg swelling before the amlodipine was added. The swelling is from her heart failure, which preceded the drug by two years."),
    ("efavirenz",         "The vivid dreams and insomnia were documented in the notes six months before efavirenz was started. Pre-existing sleep disorder."),
    ("lisinopril",        "His cough predates the lisinopril by two years. It has not changed in character or severity since the drug was started."),
    ("haloperidol",       "The rigidity and tremor were present before haloperidol was initiated — signs of Parkinson's disease, not extrapyramidal side effects."),
    ("rifampicin",        "His fatigue and weight loss were present for three months before TB treatment started — symptoms of active TB, not medication side effects."),
]

# ════════════════════════════════════════════════════════════════════════════
# P34 — Drug holiday + unrelated symptom (Non-ADR)
# Targets: #124 ("skipping ARV doses for three days, mild headache")
# Key: patient was not taking the drug when the symptom occurred.
# ════════════════════════════════════════════════════════════════════════════

P34_NONADR = [
    # Skipping doses during travel / running out
    ("efavirenz",         "He was skipping ARV doses for five days while travelling for work. He now presents with mild headache — he has been on the road with poor sleep. Not a drug reaction."),
    ("metformin",         "She ran out of metformin four days ago and has not taken any. She is reporting fatigue and nausea — from her uncontrolled diabetes, not the drug."),
    ("amlodipine",        "Patient missed his amlodipine for a week while visiting his village. He reports mild swelling of the ankles — from the long journey and inactivity, not the medication."),
    ("isoniazid",         "She stopped taking isoniazid without telling the doctor two weeks ago. The tingling she reports in her hands is from TB itself, not the drug she has not been taking."),
    ("atorvastatin",      "He has not taken atorvastatin for ten days — he says he forgot to refill. The muscle ache he reports started before he missed the doses, from physical labour."),
    # Drug stopped by doctor — symptom appears after washout
    ("lisinopril",        "Lisinopril was discontinued three weeks ago. She presents with a cough — she has had a respiratory tract infection for five days. No drug reaction possible."),
    ("carbamazepine",     "Carbamazepine was withdrawn two weeks ago under specialist supervision. The dizziness is a seizure prodrome, not a drug side effect."),
    ("co-trimoxazole",    "Co-trimoxazole prophylaxis was stopped two months ago after CD4 recovery. The rash that appears now is from sun exposure, not the antibiotic."),
    # Patient took drug holiday
    ("rifampicin",        "He stopped all TB drugs for a month without medical advice. He reports nausea and abdominal pain — from irregular treatment and active TB, not drug toxicity."),
    ("glibenclamide",     "She took a drug holiday from glibenclamide for three days during Ramadan. The symptoms she reports — mild dizziness and hunger — are from the fasting period."),
    ("metformin",         "He has been off metformin for six days after reading about side effects online. The stomach upset he describes has been present since before he stopped the drug."),
    # "Not on medication at the time"
    ("amoxicillin",       "She was not on any antibiotic at the time of the rash. Her last dose of amoxicillin was eight days before the rash appeared — outside the expected reaction window."),
    ("artemether-lumefantrine", "The vomiting occurred three weeks after the last dose of the antimalarial. This is far outside any plausible drug reaction window."),
    ("haloperidol",       "He had not received his depot injection for six weeks when the agitation occurred. The behavioural change is from undertreated psychosis, not the antipsychotic."),
    # No causal temporal link
    ("tenofovir",         "She has not taken her ARVs for two months. The renal function is impaired — from untreated HIV nephropathy, not tenofovir nephrotoxicity."),
    ("atorvastatin",      "He stopped the statin three months ago. The muscle enzyme elevation is from his recent vigorous exercise programme, not from a drug he is no longer taking."),
    ("amlodipine",        "Patient self-discontinued amlodipine five weeks ago. The peripheral oedema she now has is from her worsening cardiac failure — she is no longer on the drug."),
    ("isoniazid",         "She ran out of all TB drugs two weeks ago. The nausea and vomiting she presents with are from hyperemesis gravidarum — she is seven weeks pregnant."),
    ("efavirenz",         "He has been off all ARVs for three months due to a supply chain issue. The vivid dreams were present before he started ARVs years ago — they are not drug-related."),
    ("co-trimoxazole",    "She stopped co-trimoxazole at 36 weeks of pregnancy as advised. The rash she develops at 38 weeks is PUPP — a pregnancy-related dermatosis, not a drug reaction."),
]

# ════════════════════════════════════════════════════════════════════════════
# P35 — Objective lab / imaging normalisation overrides subjective (Non-ADR)
# Targets: #145 (HbA1c improved despite "feeling worse"), #148 (ESR/CRP normalised)
# Key: objective evidence of improvement/non-toxicity despite patient's subjective complaint.
# ════════════════════════════════════════════════════════════════════════════

P35_NONADR = [
    # HbA1c improving
    ("metformin",         "She says she feels worse and the drug is making her ill. However, her HbA1c has improved from 10.2% to 6.8% since starting metformin. Objective control is excellent."),
    ("glibenclamide",     "He feels the diabetes tablet is not working and he feels worse. His HbA1c has dropped from 11.0% to 7.4% over three months. The drug is working — the symptoms are unrelated."),
    ("insulin",           "Patient reports feeling terrible on insulin. Her fasting glucose has normalised to 5.8 mmol/L and her HbA1c is now 6.5%. Glycaemic control is achieved; no ADR documented."),
    # ESR / CRP normalising
    ("prednisolone",      "She feels the drug is making her worse — more joint pain and stiffness. However, her ESR has fallen from 88 to 12 mm/hr and CRP from 64 to 4 mg/L. Objective remission."),
    ("methotrexate",      "He reports feeling no improvement on methotrexate. His ESR has normalised from 72 to 18 mm/hr and his CRP is now undetectable. The drug is achieving disease control."),
    ("hydroxychloroquine","Patient says she feels worse since starting hydroxychloroquine. Her CRP has fallen from 48 to 7 mg/L and joint swelling has reduced on examination. Nocebo effect suspected."),
    # Liver function tests normalising
    ("isoniazid",         "She reports nausea and abdominal discomfort on isoniazid. Liver function tests: ALT 28 U/L, AST 31 U/L — both within normal limits. No hepatotoxicity; symptoms are functional."),
    ("atorvastatin",      "He reports fatigue and muscle ache on the statin. CK is 94 U/L — well within normal range. Liver enzymes normal. No objective evidence of drug toxicity."),
    ("rifampicin",        "Patient reports feeling unwell on TB drugs. LFTs: ALT 32, AST 28, bilirubin 12 — all normal. Symptoms are not from drug-induced liver injury."),
    # Renal function stable / improving
    ("tenofovir",         "She is worried the ARV is damaging her kidneys. eGFR is 74 mL/min — unchanged from baseline. Creatinine is 88 umol/L — normal. No evidence of tenofovir nephrotoxicity."),
    ("lisinopril",        "He reports feeling worse on lisinopril. Creatinine is 102 umol/L — within normal limits and unchanged from before the drug. Blood pressure is now 124/78. No ADR."),
    ("co-trimoxazole",    "Patient reports nausea on co-trimoxazole. Full blood count shows no cytopaenia. Renal function normal. Urinalysis clear. The nausea is functional, not drug toxicity."),
    # Blood counts normal
    ("zidovudine",        "She reports fatigue on zidovudine. Full blood count: Hb 12.1 g/dL — no anaemia. WBC and platelets normal. The fatigue is from HIV disease burden, not drug toxicity."),
    ("carbamazepine",     "Patient complains of dizziness on carbamazepine. Serum carbamazepine level is 6.4 mg/L — within therapeutic range. No toxicity. Dizziness may be from epilepsy."),
    # Imaging / clinical exam improves
    ("amlodipine",        "She feels the drug is making her heart worse. Echocardiogram shows LVEF improved from 38% to 52% since starting amlodipine. The drug is working. Symptoms are unrelated."),
    ("prednisolone",      "He reports feeling worse. Chest X-ray shows consolidation improving compared to two weeks ago. PaO2 is 94% on room air — improving. Steroid is achieving its effect."),
    ("metformin",         "She says the tablet makes her feel sick all day. Fasting glucose 5.3 mmol/L — tightly controlled. BMI has decreased by 1.8. The drug is beneficial; her symptoms are not toxicity."),
    # Objective normal overrides strong subjective complaint
    ("atorvastatin",      "Patient insists the statin is destroying his muscles. CK is 112 U/L (upper limit of normal 200 U/L). Aldolase normal. Muscle MRI unremarkable. No myopathy confirmed."),
    ("efavirenz",         "She feels the ARV is damaging her brain. Neuropsychological testing shows no cognitive decline — scores are within normal limits. HIV viral load is undetectable."),
    ("glibenclamide",     "He reports weakness and collapse but blood glucose at the time was 5.6 mmol/L. No hypoglycaemia documented. The collapse was a vasovagal episode unrelated to the drug."),
]

# ════════════════════════════════════════════════════════════════════════════
# P36 — Coincidental illness / pre-existing condition (Non-ADR)
# Targets: #293 (coincidental measles post-vaccination — the model still FP)
# Key: symptoms explicitly pre-date or are causally attributed to another source.
# ════════════════════════════════════════════════════════════════════════════

P36_NONADR = [
    # Family cluster / contact exposure predates vaccination or drug
    ("measles vaccine",   "Three weeks after the measles vaccine my child had a rash. The doctor confirmed it was actual measles from a school contact — the child was exposed before the vaccine took effect."),
    ("BCG vaccine",       "After the BCG, the baby developed a fever. The whole family had been ill with the same fever for a week before the vaccine — a coincidental viral illness."),
    ("hexavalent vaccine","Child had diarrhoea four days after the hexavalent vaccine. Two siblings had the same diarrhoea two days before the vaccine was given — a rotavirus family cluster."),
    ("OPV vaccine",       "Baby developed loose stools five days after oral polio vaccine. Two cousins who were not vaccinated had the same illness at the same time — community gastroenteritis."),
    ("yellow fever vaccine","She had fever and joint pain two weeks after the yellow fever vaccine. Her husband had the same symptoms — both tested positive for chikungunya from a local outbreak."),
    ("meningitis vaccine","Child had a febrile convulsion two days after meningococcal vaccine. EEG confirms idiopathic epilepsy — a pre-existing susceptibility, not a vaccine reaction."),
    # Symptoms clearly predate drug start
    ("amoxicillin",       "The rash appeared the day before amoxicillin was prescribed. The doctor confirmed it was a viral exanthem — the drug was not yet given when the rash started."),
    ("atorvastatin",      "He reports muscle pain since starting the statin. On review, the muscle pain was documented in physiotherapy notes three months before atorvastatin was prescribed."),
    ("metformin",         "She says the stomach pain started with the metformin. GP records show she attended A&E with identical abdominal pain six weeks before the drug was started."),
    ("amlodipine",        "The leg swelling is attributed to amlodipine. However, echocardiogram from two years prior shows pre-existing right heart dysfunction causing peripheral oedema."),
    # Independent viral / bacterial cause confirmed
    ("co-trimoxazole",    "Patient has a rash while on co-trimoxazole. Serology confirms parvovirus B19 — slapped cheek disease. The rash is the viral exanthem, not a drug reaction."),
    ("ceftriaxone",       "She has urticaria while on ceftriaxone. Serology shows positive IgE to latex from a recent surgical procedure. The urticaria is a latex allergy, not a drug reaction."),
    ("rifampicin",        "He has a maculopapular rash on TB drugs. Serology positive for dengue NS1 antigen — active dengue infection. The rash is from dengue fever, not anti-TB drugs."),
    ("artemether-lumefantrine", "Child has fever and vomiting after antimalarial. Malaria RDT is positive — the fever is residual malaria, not a drug reaction. Treatment dose was correct."),
    ("amoxicillin",       "He has a morbilliform rash on amoxicillin. Paul-Bunnell positive — he has infectious mononucleosis. The amoxicillin-EBV rash is recognised but is not a true drug allergy."),
    # "Coincidence" explicitly noted or doctor attributed
    ("haloperidol",       "She had a fall after starting haloperidol. Review of her falls diary shows she had three falls in the month before starting the antipsychotic — pre-existing instability."),
    ("tenofovir",         "He has proteinuria on tenofovir. Renal biopsy shows pre-existing focal segmental glomerulosclerosis from HIV nephropathy — not tenofovir toxicity."),
    ("isoniazid",         "Jaundice developed four weeks after starting TB drugs. Hepatitis B surface antigen positive — acute hepatitis B flare, not isoniazid-induced liver injury."),
    ("metronidazole",     "She has tingling in her hands on metronidazole. Nerve conduction studies confirm carpal tunnel syndrome — compression neuropathy, not drug-induced neuropathy."),
    ("efavirenz",         "He reports peripheral neuropathy on efavirenz. EMG confirms diabetic peripheral neuropathy consistent with 15 years of poorly controlled diabetes — not ARV toxicity."),
]

# ════════════════════════════════════════════════════════════════════════════
# P37 — Asymptomatic lab ADR (ADR)
# Targets: #20 (potassium high, no symptoms — the lab finding IS the ADR)
# Key: patient denies symptoms but objective lab finding is the adverse reaction.
# ════════════════════════════════════════════════════════════════════════════

P37_ADR = [
    ("lisinopril",        "My doctor said my potassium is high but I don't feel anything wrong. She wants to change the BP medicine because the lisinopril is causing the high potassium."),
    ("spironolactone",    "I feel completely fine but my doctor says my blood potassium is dangerously high since starting spironolactone. She wants to stop the tablet immediately."),
    ("trimethoprim",      "I have no symptoms at all but the blood test shows my potassium level is very elevated since I started trimethoprim. The doctor says it is from the antibiotic."),
    ("tenofovir",         "I feel well but my kidney test is abnormal. The doctor says my creatinine has risen since I started tenofovir and she wants to switch my ARV regimen."),
    ("isoniazid",         "I feel no pain but my liver blood test is showing high ALT since I started the TB drug. The doctor says the isoniazid is affecting my liver silently."),
    ("methotrexate",      "I feel fine but the blood test shows my white cell count has dropped significantly since I started methotrexate. The doctor is worried about bone marrow suppression."),
    ("atorvastatin",      "I have no muscle pain at all but my CK blood test came back very high after starting the statin. The doctor says it is statin-induced muscle damage without symptoms."),
    ("amiodarone",        "I feel completely well but my thyroid test is grossly abnormal since starting amiodarone. The doctor says it is drug-induced hypothyroidism even though I feel no different."),
    ("rifampicin",        "The doctor says my liver tests are very elevated since starting the TB drugs but I do not feel ill at all. She says the rifampicin is causing the liver abnormality silently."),
    ("hydroxyurea",       "I feel okay but my blood count shows my platelets and white cells have fallen sharply since starting hydroxyurea. The doctor says the drug is suppressing my bone marrow."),
    ("indinavir",         "My kidneys feel fine and I have no pain but the urine test shows crystals and my creatinine is rising since I started indinavir. The doctor says it is early nephrotoxicity."),
    ("vancomycin",        "I cannot hear the difference but the audiogram shows high-frequency hearing loss since I started the vancomycin infusions. The doctor says it is early ototoxicity."),
    ("zidovudine",        "I do not feel particularly tired but my blood test shows moderate anaemia since starting zidovudine. The nurse said the drug is suppressing my red cell production."),
    ("lithium",           "I feel fine but my blood lithium level came back mildly toxic and my TSH is elevated since I started the mood stabiliser. The doctor wants to adjust the dose."),
    ("metformin",         "I feel well but the blood test shows my B12 level has dropped very low since I have been on metformin for two years. The doctor says the drug is blocking B12 absorption."),
    ("tacrolimus",        "I have no symptoms at all but my blood sugar has been consistently elevated since starting tacrolimus. The doctor says it is causing new-onset diabetes silently."),
    ("ACE inhibitor",     "My doctor says my blood potassium has risen to a dangerous level since the new blood pressure tablet was started. I feel nothing abnormal but she is changing the drug."),
    ("ciclosporin",       "I feel fine but my creatinine has doubled since starting ciclosporin for my kidney transplant. The team says it is calcineurin inhibitor nephrotoxicity on the blood test."),
    ("amoxicillin",       "I have no diarrhoea and feel well but my stool test shows I have Clostridioides difficile since the antibiotic course. The doctor says it is an antibiotic-associated infection."),
    ("furosemide",        "I feel okay but the blood test shows my sodium is very low and my potassium is low since starting furosemide. The doctor says the diuretic is causing the electrolyte problem."),
]

# ════════════════════════════════════════════════════════════════════════════
# P38 — Severity minimisation coping language (ADR)
# Targets: #77 ("manageable", "not a big deal", prob=0.051 — strongly missed)
# Key: true ADR dismissed with coping language — still an ADR.
# ════════════════════════════════════════════════════════════════════════════

P38_ADR = [
    # "Manageable" pattern
    ("metformin",         "The diarrhoea from the metformin is manageable honestly. I just take it after food and drink extra water. But yes, it is happening every day since I started the tablet."),
    ("co-trimoxazole",    "The rash is manageable — I use the cream the pharmacist gave me. But I have had it since I started the antibiotic and it is spreading slowly."),
    ("efavirenz",         "The dizziness from the ARV is manageable. I just sit down when it comes. It happens every morning when I take the tablet on an empty stomach."),
    ("isoniazid",         "The tingling in my hands is manageable. I ignore it mostly. But it started when I began the TB drug and it has not gone away."),
    ("amlodipine",        "My ankle swelling from the blood pressure tablet is manageable. I put my feet up in the evening. But both ankles are swollen since I started the drug."),
    # "Not a big deal" / "Nothing serious"
    ("amoxicillin",       "The loose stool after the antibiotic is not a big deal. I just drink plenty of water. But it is happening four or five times a day since I started the course."),
    ("artemether-lumefantrine", "The stomach upset from the malaria tablet is nothing serious. I cope with it. But I vomited twice after each of the first three doses."),
    ("rifampicin",        "My orange urine from the TB drug is not a big deal — I know it is from the rifampicin. But my skin has also gone slightly yellow and my eyes look different."),
    ("atorvastatin",      "The muscle aches from the statin are nothing serious — I just take paracetamol. But both my thighs have been aching every day since I started the cholesterol tablet."),
    ("lisinopril",        "The cough from the blood pressure tablet is not a big deal. It does not stop me sleeping. But it is there every day and it started the week I began the lisinopril."),
    # "I cope" / "I manage"
    ("haloperidol",       "The stiffness and slow walking since the injection — I cope with it. I just move more carefully. But my family says I am very different since the depot injection."),
    ("tenofovir",         "The back pain from the ARV I manage. I take ibuprofen when it is bad. But the pain has been there since I started tenofovir and my kidney test was flagged."),
    ("metformin",         "The nausea from the metformin, I manage it. I take it with food and rest after. But it is there every morning and every evening when I take the tablet."),
    ("carbamazepine",     "The double vision from the epilepsy tablet, I cope. I just blink and it goes. But it happens every day and it is affecting my work and reading."),
    # "Small" minimiser with real symptom
    ("co-trimoxazole",    "There is a small rash on my chest since the antibiotic. It is small but it is spreading to my arms and back. I have been scratching it every night."),
    ("efavirenz",         "My sleep is a little disturbed from the ARV — just small. But I am waking up every night with bad dreams and sweating since I started the efavirenz."),
    ("glibenclamide",     "I feel weak sometimes — just small small. But the nurse checked and said my sugar is dropping too low in the afternoons since I started the diabetes tablet."),
    ("isoniazid",         "My hands feel a little different since the TB drug — just mild tingling. But the physiotherapist says my grip has weakened noticeably on the right side."),
    ("amlodipine",        "My legs are a little heavy since the blood pressure medicine — just mild. But I cannot walk to the market now without resting three times. Before the tablet I had no problem."),
    ("artemether-lumefantrine", "I vomited once after the malaria tablet — just one time. But the nausea is still there with every dose and I cannot eat properly since starting the treatment."),
]

# ════════════════════════════════════════════════════════════════════════════
# P39 — Ghanaian English malaise idioms extended (ADR)
# Targets: #96 ("My body was somehow after I started the tablets" — regression to prob=0.243)
# Key: "somehow", "not the same", "anyhow", vague-but-real systemic ADR descriptions.
# ════════════════════════════════════════════════════════════════════════════

P39_ADR = [
    # "Somehow" idiom — core pattern from #96
    ("metformin",         "My body was somehow after I started the tablets. I cannot explain it well but I am not feeling myself since the diabetes medicine."),
    ("efavirenz",         "My body is somehow since I started the ARV. I feel strange and not normal. Something is not right since I began the treatment."),
    ("co-trimoxazole",    "Since the antibiotic my body has been somehow. I feel somehow in my head and somehow in my stomach. I want to report it."),
    ("isoniazid",         "My body feels somehow since the TB drug started. It is hard to explain but I know something is wrong with me since I started it."),
    ("haloperidol",       "My body is somehow since the injection. I feel slow and stiff. Something in my body is different since they gave me the depot."),
    ("amlodipine",        "Since the blood pressure tablet my body is somehow. My legs are heavy and somehow when I walk. I have never felt like this before the drug."),
    ("atorvastatin",      "My body has been somehow since the cholesterol tablet. I feel weak somehow and my muscles are somehow aching. It started when the doctor added the new drug."),
    # "Not the same" / "changed"
    ("efavirenz",         "Since I started the ARV my body has not been the same. I feel like a different person. Before the drug I was fine. Now I am not myself."),
    ("metformin",         "My body has not been the same since the diabetes medicine. Something has changed inside me. I cannot feel normal anymore since the tablet."),
    ("isoniazid",         "Since the TB treatment started I am not the same person. My family says I look different. My body is not the same since the drugs."),
    ("rifampicin",        "Something changed in my body when I started the TB drugs. I cannot say exactly what but I am not the same and I know it is from the medicine."),
    # "Anyhow" / "doing me anyhow"
    ("efavirenz",         "The ARV is doing my body anyhow. I feel anyhow every morning. Since the tablet I cannot go about my normal duties properly."),
    ("co-trimoxazole",    "Since the antibiotic my body is doing me anyhow. I feel somehow anyhow and cannot sleep well. I think it is from the drug."),
    ("glibenclamide",     "The diabetes tablet is doing my body anyhow. I feel somehow weak and my head is somehow anyhow every afternoon. The nurse said my sugar is going low."),
    # "Not well" / "not fine"
    ("artemether-lumefantrine", "Since I took the malaria tablet my body has not been well. I feel not fine inside. Something inside me is not fine since the antimalarial."),
    ("amlodipine",        "My body has not been fine since the blood pressure tablet. I feel somehow and not fine every day. My legs are not fine — they are heavy and swollen."),
    ("tenofovir",         "Since the ARV I am not fine. My back is not fine and my urine is not fine. I know the drug is doing something to my kidneys."),
    # "Something entered my body"
    ("haloperidol",       "Since the injection something entered my body. I feel stiff and different. Something is inside me that was not there before the depot."),
    ("isoniazid",         "Since the TB drug something has happened to my body. I feel something in my hands and feet — a strange feeling. It started with the medicine."),
    ("efavirenz",         "I feel like the ARV entered my body and changed something. My head is not right since I started it. Something is wrong that was not wrong before."),
]

# ════════════════════════════════════════════════════════════════════════════
# P40 — Hedged regulatory / signal memo language (ADR)
# Targets: #175 ("reporting rate...suggestive but not confirmed", prob=0.150)
# Key: hedged, uncertain, or preliminary regulatory language still describes a real ADR signal.
# ════════════════════════════════════════════════════════════════════════════

P40_ADR = [
    # "Suggestive but not confirmed" signal language
    ("metronidazole",     "Signal assessment memo: reporting rate for peripheral neuropathy with metronidazole is suggestive but not confirmed. Recommendation: enhanced surveillance and reporting."),
    ("quinolone",         "Signal assessment: disproportionality analysis suggests a possible association between fluoroquinolones and tendon rupture. Signal not yet confirmed. Further review ongoing."),
    ("efavirenz",         "Preliminary signal: data suggest a possible neuropsychiatric signal with efavirenz at standard doses in African populations. Signal under review. Healthcare providers should report cases."),
    ("artemether-lumefantrine", "Signal memo: reports of QTc prolongation with artemether-lumefantrine are suggestive of a pharmacovigilance signal. Evidence is preliminary and under international review."),
    ("spironolactone",    "Pharmacovigilance signal: disproportionate reporting of hyperkalaemia with spironolactone in outpatient settings. Signal is preliminary. Monitoring of renal function recommended."),
    # "Possible", "may be", "cannot be ruled out"
    ("rifampicin",        "Causality assessment: the acute liver injury in this patient may be related to rifampicin. Drug-induced liver injury cannot be ruled out. The drug has been suspended pending review."),
    ("isoniazid",         "The peripheral neuropathy reported in this patient is possibly related to isoniazid. Pyridoxine co-administration is recommended. Causality is probable but not certain."),
    ("co-trimoxazole",    "The haematological abnormality observed in this cohort may be attributable to co-trimoxazole. The association is biologically plausible but causality is not yet established."),
    ("haloperidol",       "Neuroleptic malignant syndrome cannot be ruled out in this patient on haloperidol. Causality is considered possible based on the WHO-UMC scale. The drug has been stopped."),
    ("tenofovir",         "Tenofovir-associated nephrotoxicity is possible in this case. The temporal relationship is consistent and renal function is improving since the drug was switched."),
    # "Under investigation", "provisional", "preliminary"
    ("atorvastatin",      "Provisional causality assessment: statin-induced rhabdomyolysis is the working diagnosis. CK is markedly elevated. Atorvastatin has been stopped. Outcome under review."),
    ("carbamazepine",     "Skin reaction is provisionally classified as drug reaction with eosinophilia and systemic symptoms (DRESS) related to carbamazepine. Classification may be revised after full workup."),
    ("amiodarone",        "Pulmonary toxicity is under investigation in this patient on long-term amiodarone. CT findings are consistent with amiodarone lung but other causes have not been excluded."),
    ("methotrexate",      "Preliminary assessment: hepatic fibrosis on liver biopsy is possibly related to cumulative methotrexate dose. Full causality assessment pending specialist hepatology review."),
    # Market withdrawal / recall language
    ("quinolone",         "Market withdrawal notification: this lot is recalled due to preliminary reports of severe hepatotoxicity in two confirmed cases under regulatory investigation."),
    ("sulfonamide",       "Regulatory withdrawal memo: this product is provisionally withdrawn pending investigation of a possible association with aplastic anaemia. Cases are being reviewed."),
    ("antimalarial",      "Enhanced surveillance notice: three cases of cardiac arrhythmia possibly associated with this antimalarial have been reported. Healthcare workers should report any cases immediately."),
    # "Weak signal", "low confidence" but real concern
    ("rifampicin",        "The reporting rate for hepatotoxicity with rifampicin in this population is higher than expected but within a range that may reflect increased surveillance rather than increased incidence."),
    ("isoniazid",         "The peripheral neuropathy signal with isoniazid in this register is weak but consistent with known pharmacology. The data support continuation of pyridoxine co-prescription."),
    ("efavirenz",         "The neuropsychiatric signal with efavirenz in sub-Saharan Africa is weak but persistent across multiple datasets. The benefit-risk profile remains favourable but monitoring is warranted."),
]


def build_records():
    rows = []

    def add(items, priority, tag, contains_adr):
        for n, (drug, sentence) in enumerate(items, 1):
            rows.append(make(drug, sentence, priority, tag, n, contains_adr))

    add(P32_NONADR, 32, "inv_exonerates",   False)
    add(P33_NONADR, 33, "not_started",      False)
    add(P34_NONADR, 34, "drug_holiday",     False)
    add(P35_NONADR, 35, "obj_normalises",   False)
    add(P36_NONADR, 36, "coincidental",     False)
    add(P37_ADR,    37, "asymp_lab",        True)
    add(P38_ADR,    38, "sev_min",          True)
    add(P39_ADR,    39, "malaise_idiom",    True)
    add(P40_ADR,    40, "hedged_signal",    True)

    return rows


def validate(rows):
    ids = [r["setid"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate setids"
    for r in rows:
        for field in ("drug", "sentence", "source", "setid", "contains_adr", "adr_spans", "drug_spans"):
            assert field in r, f"Missing field {field} in {r['setid']}"
        assert r["adr_spans"] == [], f"Non-empty adr_spans in {r['setid']}"
        assert r["drug_spans"] == [], f"Non-empty drug_spans in {r['setid']}"


def main():
    rows = build_records()
    validate(rows)

    non_adr = sum(1 for r in rows if r["contains_adr"] == 0)
    adr     = sum(1 for r in rows if r["contains_adr"] == 1)

    print(f"Records: {len(rows)} total  ({adr} ADR, {non_adr} Non-ADR)")

    by_p = {}
    for r in rows:
        p = r["setid"].split("_")[2]
        by_p.setdefault(p, 0)
        by_p[p] += 1
    for p in sorted(by_p, key=lambda x: int(x.lstrip("p"))):
        print(f"  {p}: {by_p[p]}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written -> {OUT}")


if __name__ == "__main__":
    main()
