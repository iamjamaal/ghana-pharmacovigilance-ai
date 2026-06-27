#!/usr/bin/env python3
"""
scripts/generate_synthetic_phase7_v2.py
========================================
Generate synthetic_phase7_v2.jsonl covering the 7 synthetically-feasible
retraining priorities from MODEL_FINDINGS.md Section 6 not covered by v1.

v1 covered:  priorities 2, 3, 9, 12, 13, 14
v2 covers:   priorities 5, 6, 7, 8, 10, 11, 15
Deferred:    priorities 1 (FDA/regulatory), 4 (Akan/Ewe/Hausa) — need real records

IMPORTANT: all adr_spans and drug_spans are left empty [].
This data is CLF-only. Do NOT pass it to NER training — empty spans would
teach the NER model that these sentences contain no entities.
The loso_phase7.py script enforces this by loading synthetic separately.

Output: data/silver/synthetic_phase7_v2.jsonl
Run:    python scripts/generate_synthetic_phase7_v2.py
"""

import json
from pathlib import Path

ROOT   = Path(__file__).parent.parent
OUTDIR = ROOT / "data" / "silver"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT    = OUTDIR / "synthetic_phase7_v2.jsonl"


def make(drug, sentence, priority, tag, n, contains_adr):
    return {
        "drug":         drug,
        "sentence":     sentence,
        "source":       "synthetic_phase7",
        "setid":        f"synthetic_phase7_p{priority}_{tag}_{n:03d}",
        "contains_adr": int(contains_adr),
        "adr_spans":    [],
        "drug_spans":   [],
    }


# ════════════════════════════════════════════════════════════════════════════
# Priority 5 — Pidgin ADR augmentation (ototoxicity, drug-herb, CNS-behaviour)
# MODEL_FINDINGS Section 2.15 — systematic sub-threshold Pidgin FNs
# Strategy: minimal pairs — same drug/symptom vocabulary, flip temporal/causal
# structure for Non-ADR counterpart. 60 examples (30 ADR + 30 Non-ADR)
# ════════════════════════════════════════════════════════════════════════════

P5_ADR = [
    # Ototoxicity (aminoglycosides, quinine, furosemide)
    ("streptomycin",  "Since dem start give me the streptomycin injection every morning, my ear dey ring and I no fit hear well again."),
    ("gentamicin",    "After I begin take the gentamicin injection, my two ears dey make noise all the time and I no fit hear what people dey say."),
    ("streptomycin",  "I dey take streptomycin for three weeks now and my ear dey do me somehow — the ringing no stop and my hearing don reduce plenty."),
    ("quinine",       "Since I start the quinine injection for severe malaria, my head dey ring and my hearing don go down small small."),
    ("gentamicin",    "My ear dey pain me and I no fit hear well since dem give me the gentamicin. The doctor say na the injection cause am."),
    ("furosemide",    "Since I begin use furosemide every morning, I no fit hear as before — everything dey sound far away and my ear dey ring."),
    # Drug-herb interaction
    ("metformin",     "I dey take my diabetes drug and I also dey drink the herbal tea my grandmother prepare. Since then my sugar dey go too low and I dey dizzy all the time."),
    ("glibenclamide", "I dey use the white sugar tablet and also dey boil the bitter leaf for drink. Since then I dey shake plenty and I faint one time — doctor say my sugar go down too low."),
    ("metformin",     "I mix the metformin with the herbal root my neighbour give me. After that, my head dey swim and I dey sweat cold sweat — hospital say blood sugar low."),
    ("glibenclamide", "I take the diabetes medicine with the herbal tonic from the market. My body start shake and I no fit stand well — sugar drop too low."),
    # CNS/behavioural ADRs
    ("efavirenz",     "Since I start efavirenz I start walk around my house in the night and I no know what I dey do. My wife tell me I dey talk plenty in my sleep."),
    ("haloperidol",   "After dem give me the haloperidol injection, I no fit sit down — my legs dey move by themselves and I dey pace up and down. Na the injection cause am."),
    ("diazepam",      "Since I start take the diazepam tablet, I dey feel like say something dey move inside my head and my thoughts dey mix up all the time."),
    ("efavirenz",     "Since efavirenz start, I dey see strange things when I close my eyes for sleep — like bad dream but I still dey wake. The nurse say na the drug."),
    ("chlorpromazine","After I take the chlorpromazine, my whole body don stiffen and my mouth open and no fit close. I go hospital emergency and dem say na the drug reaction."),
    # Additional Pidgin symptom types from MODEL_FINDINGS
    ("artesunate-amodiaquine", "Since I begin take artesunate-amodiaquine my body dey hot inside but outside cold and I dey shake — na drug fever the nurse say."),
    ("isoniazid",     "My body dey do me anyhow since I start the TB tablet — system scatter and I no fit work well at all."),
    ("lopinavir/ritonavir", "Medicine enter my head since I start the lopinavir tablet — I no dey think straight and everything dey confuse me."),
    ("co-trimoxazole","Since I start co-trimoxazole my pikin dey shake after I give am the tablet. The whole body dey vibrate and the eyes go up — doctor say na reaction."),
    ("tenofovir",     "Since I begin tenofovir my body dey do me somehow — I no fit explain but nothing dey normal. Nurse say na the ARV tablet."),
    ("amlodipine",    "I dey pass water too much since I start the blood pressure tablet — every thirty minutes I run to toilet. Doctor say na the amlodipine."),
    ("co-trimoxazole","My pikin eyes yellow small after I give am co-trimoxazole for two weeks. Doctor say na the tablet cause am — jaundice from the drug."),
    ("isoniazid",     "Tummy run dey worry me plenty since I start the TB injection — I go toilet more than five times per day. Nurse say na the isoniazid."),
    ("haloperidol",   "Since dem give me the injection, my body no dey rest — I dey pace up and down, legs no fit keep still. Na akathisia from the drug."),
    ("pyrazinamide",  "Blood dey run cold and body dey shake since I start the pyrazinamide tablet — shivers plenty and temperature go up. Doctor say na drug fever."),
    # Concessive constructions ("dey help X but dey cause Y")
    ("metformin",     "The metformin tablet dey help my sugar but my stomach dey pain me every morning oo — I dey vomit after I take am."),
    ("efavirenz",     "Efavirenz dey keep my viral load down but my head dey confuse me every night — I dey see things and no fit sleep well at all."),
    ("isoniazid",     "The TB tablet dey work for my chest but since I start am my legs and hands dey pain me — doctor say na nerve damage from the isoniazid."),
    ("amlodipine",    "The amlodipine tablet dey bring my pressure down but my two legs dey swell since I start am — I no fit wear my shoe well again."),
    ("co-trimoxazole","Co-trimoxazole dey protect me from infection but since I start am my skin don come out with plenty rash all over my body."),
]

P5_NON_ADR = [
    # Ototoxicity Non-ADR counterparts (ear problem before drug / alternative cause)
    ("streptomycin",  "My ear was ringing before dem even give me the streptomycin injection — the doctor say na wax inside my ear cause am, not the drug."),
    ("gentamicin",    "The hearing problem dey before I start gentamicin — I don have problem with my ear since I was small. The doctor confirm say na my old condition."),
    ("streptomycin",  "I stop the streptomycin one week ago and my ear ringing no change at all — still the same. Doctor say na ear infection, not the drug."),
    ("quinine",       "I had ear noise before they give me quinine injection — it started when I had that fever two months ago. The doctor say the quinine never cause am."),
    ("gentamicin",    "My hearing was already going down before I start the gentamicin. Doctor check and say na age — I don be old man and my ear dey do am since."),
    # Drug-herb Non-ADR counterparts
    ("metformin",     "I dey take metformin and also dey use herbal tea but my sugar dey normal — doctor check and say both together no cause any problem for me."),
    ("glibenclamide", "I try the herbal together with my diabetes tablet but dem no affect me — blood sugar check fine and doctor say my body tolerate am well."),
    # CNS/behavioural Non-ADR counterparts
    ("efavirenz",     "I walk in my sleep before I even start efavirenz — I been do am since secondary school. My family confirm say na old habit."),
    ("haloperidol",   "My legs were restless before dem give me the haloperidol injection. Doctor check and say na the anxiety cause am, not the injection."),
    ("diazepam",      "The confusion I dey have was already there before I start the diazepam — it started after I fall and hit my head. Doctor say no be the tablet."),
    # Other Pidgin Non-ADR
    ("artesunate-amodiaquine", "I was shaking before I even take the artesunate — the malaria itself give me the chills. After I finish the drug the shaking stop — na the disease, not the drug."),
    ("isoniazid",     "My body dey do me somehow before I start the TB tablet — the disease itself was making me weak. After I start treatment I start feel better."),
    ("co-trimoxazole","My pikin was yellow before I give am co-trimoxazole — doctor say na malaria cause the jaundice, not the tablet. Drug was innocent."),
    ("tenofovir",     "I don dey feel somehow for months before I even start tenofovir — the HIV itself was making me weak. Drug no cause am."),
    ("amlodipine",    "I no dey pass water more than normal. My doctor check and say my kidney fine and the amlodipine no cause any problem with my water."),
    # Concessive Non-ADR counterparts
    ("metformin",     "The metformin dey help my sugar and I feel fine — no stomach pain, no vomiting. I dey take am well well and no problem at all."),
    ("efavirenz",     "Efavirenz dey work for my viral load and I no have problem with my sleep — no strange dreams, no confusion. I dey tolerate am well."),
    ("isoniazid",     "The TB tablet dey treat my chest well and I no feel any nerve pain in my hands or legs. I dey take am without any problem."),
    ("amlodipine",    "Amlodipine dey bring my pressure down and my legs no dey swell — doctor check and say everything fine with the drug."),
    ("co-trimoxazole","Co-trimoxazole dey protect me and I no develop any rash or reaction. I dey take am every day without any side effect."),
    # Additional Pidgin non-specific
    ("chlorpromazine","I no have any stiffness or movement problem since I start the chlorpromazine tablet. Nurse say I dey tolerate am well."),
    ("efavirenz",     "I take efavirenz for three months now and no strange thing happen — no bad dream, no confusion. My body accept the drug well."),
    ("pyrazinamide",  "I was already having rigors from the TB infection before I start pyrazinamide. After I start the drug, the shaking reduce — na the treatment working."),
    ("haloperidol",   "I dey sit still fine since I start the haloperidol — no restless legs, no pacing. The injection agree with me."),
    ("metformin",     "My tummy was upset before I start metformin — I been have that problem for years. Doctor confirm the metformin no cause any stomach issue for me."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 6 — AEFI expected/minor vs serious (MODEL_FINDINGS Section 2.12)
# 80 Non-ADR (expected minor reactions) + 20 ADR (serious reportable AEFIs)
# Key: resolution language + clinician confirmation = Non-ADR
# ════════════════════════════════════════════════════════════════════════════

P6_NON_ADR = [
    # Post-BCG expected reactions
    ("BCG vaccine",   "She was a little unsettled the evening after BCG but fed well and slept normally tonight. No fever. She was fine by morning."),
    ("BCG vaccine",   "The baby cried more than usual the first evening after BCG vaccination. She settled overnight and was completely normal by the next day."),
    ("BCG vaccine",   "Small redness at the BCG injection site — the nurse said this is normal and expected. It went away on its own within a week."),
    ("BCG vaccine",   "The BCG site was slightly raised and red. Our health worker said it was the expected response. It passed and healed normally over two weeks."),
    # Post-pentavalent / hexavalent expected reactions
    ("pentavalent vaccine", "After the pentavalent injection my baby had a mild fever of 37.8°C. I gave paracetamol and she was fine by the next morning. No other problems."),
    ("pentavalent vaccine", "Baby was fussy and cried a lot after the pentavalent shot. She slept it off and was completely normal the next day. Nothing serious."),
    ("pentavalent vaccine", "A bit of redness and swelling at the injection site after pentavalent. It was sore for about two days and then nothing serious. Normal I think."),
    ("hexavalent vaccine",  "No vomiting, no diarrhoea, no unusual crying after the hexavalent vaccine. Routine check — all normal."),
    ("hexavalent vaccine",  "Baby was slightly irritable after the hexavalent injection but she fed well and slept normally. Fine the next morning."),
    ("hexavalent vaccine",  "Mild fever for one day after hexavalent. Took paracetamol and was fine by the second day. The nurse confirmed it was an expected reaction."),
    # Post-measles-rubella expected reactions
    ("measles-rubella vaccine", "A mild rash appeared 8 days after measles-rubella vaccination. The doctor said it was a common mild vaccine reaction and it went away on its own."),
    ("measles-rubella vaccine", "Low-grade fever five days after measles-rubella shot. It lasted one day and passed quickly. The midwife told me it was normal."),
    ("measles-rubella vaccine", "My daughter had a mild rash after the measles vaccine. The health worker confirmed it was expected and it cleared within three days."),
    # Post-rotavirus expected reactions
    ("rotavirus vaccine", "My baby had loose stool for one day after the rotavirus drops. The nurse said this is expected and it settled by the next day."),
    ("rotavirus vaccine", "Mild diarrhoea for 24 hours after rotavirus vaccine. All checks were normal and it went away on its own. Health worker said it was fine."),
    # Post-COVID-19 expected reactions
    ("COVID-19 vaccine",  "Sore arm for two days after COVID vaccination. I felt a bit off on day one — tired and mild headache. Fine the next morning."),
    ("COVID-19 vaccine",  "Arm was painful and I was feverish the night after the COVID jab. I took paracetamol and was fine by the second day. Nothing serious."),
    ("COVID-19 vaccine",  "I felt tired for one day after the COVID vaccine and had a mild headache. It went away on its own by the following morning. Fine after that."),
    ("COVID-19 vaccine",  "Slightly sore arm and mild body aches after COVID vaccination. I felt a bit off for the first day. Fine the next morning."),
    ("COVID-19 vaccine",  "Low-grade fever on the evening after my second COVID dose. I took paracetamol and was fine by the next morning. No other issues."),
    # Post-yellow fever expected reactions
    ("yellow fever vaccine", "Mild headache and low-grade fever for two days after yellow fever vaccination. It went away on its own. The doctor confirmed it was a normal vaccine response."),
    ("yellow fever vaccine", "I felt tired and had a sore arm after the yellow fever jab. Took paracetamol and was fine the next day. Nothing unusual beyond mild expected effects."),
    # Post-meningococcal expected reactions
    ("meningococcal vaccine", "Soreness at the meningococcal injection site for about two days. Nothing serious — the nurse said it was normal and it passed quickly."),
    ("meningococcal vaccine", "Mild fever after the meningococcal vaccine. Took paracetamol and was fine by the second day. All readings were normal at follow-up."),
    # Post-HPV expected reactions
    ("HPV vaccine",    "My daughter had a sore arm and felt tired after her first HPV vaccine dose. She was fine the next morning. The nurse said it was expected."),
    ("HPV vaccine",    "Mild headache and slight dizziness after HPV vaccination. She sat down and it passed within twenty minutes. Nurse said it was fine."),
    # Post-OPV expected reactions
    ("OPV vaccine",    "Baby was slightly fussy after OPV drops. He fed well and settled within a few hours. No fever, no unusual crying. All normal."),
    ("OPV vaccine",    "My son cried briefly after the oral polio vaccine but settled quickly. No fever, no rash. The nurse confirmed it was nothing to worry about."),
    # Coincidental illness cases
    ("pentavalent vaccine", "My son had a cough and runny nose after the pentavalent vaccination. The doctor said the whole family had been sick with the same thing — coincidental URTI, not from the vaccine."),
    ("measles-rubella vaccine", "My child developed a rash all over her body after the measles vaccine. The doctor examined her and said it looks like actual measles from a coincidental exposure — the vaccine cannot cause measles rash this quickly."),
    ("COVID-19 vaccine",  "I developed a cough and fever two days after my COVID vaccine. My husband and two colleagues had the exact same illness that week — we all had flu. Not from the vaccine."),
    ("pentavalent vaccine", "Baby had diarrhoea after the vaccination visit. The health worker checked and found that three other children in the compound had the same — it was a community gastroenteritis."),
    # Clinician-confirmed below-threshold
    ("BCG vaccine",    "I felt fine after the BCG vaccination. Just a small pain where the needle went. The nurse said everything was completely normal."),
    ("pentavalent vaccine", "No vomiting, no diarrhoea, no unusual crying after the vaccination. The nurse checked the baby thoroughly. Routine follow-up — all normal."),
    ("COVID-19 vaccine",  "I felt completely fine after the COVID vaccine. Just a small pain where the needle went. Nothing else at all. The nurse said my response was excellent."),
    ("hexavalent vaccine",  "No rash, no fever, no unusual crying after the hexavalent shot. The nurse checked the baby and said all was well. Normal expected follow-up."),
    ("meningococcal vaccine", "I felt fine. Just a small pain where the needle went for one day. The health worker confirmed it was a normal mild reaction below the AEFI reporting threshold."),
    # Post-artesunate injection (not strictly vaccine but similar context)
    ("artesunate injection", "The child was unsettled for a few hours after the artesunate injection. She settled and was fine by the next morning. Nurse said it was expected."),
    ("artesunate injection", "Minor fever for one day after the artesunate injection for malaria. Took paracetamol and was fine by the second day. Health worker said all was normal."),
    # More minor reaction variants
    ("pentavalent vaccine", "Baby felt warm but not feverish after the pentavalent injection. She fed normally and slept well. Nothing serious — it passed on its own."),
    ("BCG vaccine",    "Tiny lump at BCG injection site after two weeks — health worker said it was expected BCG response and it resolved on its own over four weeks."),
    ("COVID-19 vaccine",  "Mild chills and body aches the evening after COVID vaccination. Took paracetamol and felt fine by the next morning. No serious symptoms."),
    ("yellow fever vaccine", "I felt a bit off on day two after yellow fever vaccination — mild headache and tiredness. It went away on its own by day three."),
    ("HPV vaccine",    "Sore arm and mild headache after second HPV dose. It was fine the next morning. The nurse confirmed it was within the normal expected range."),
    ("meningococcal vaccine", "Mild redness and warmth at the injection site after meningococcal vaccine. It went away on its own within three days. All normal."),
    ("rotavirus vaccine", "Baby passed one loose stool after rotavirus drops. No fever, no other symptoms. The nurse said it was an expected minor response. All normal."),
    ("measles-rubella vaccine", "Slight fever of 37.6°C after MR vaccine. Took paracetamol and was fine by the next morning. Midwife told me it was normal and expected."),
    ("pentavalent vaccine", "My daughter cried after the injection but was consolable. She slept normally and woke up fine. Nothing concerning. All checks normal."),
    ("hexavalent vaccine",  "Mild swelling at the hexavalent injection site. It was sore for two days then nothing serious. All readings were normal at follow-up."),
    ("COVID-19 vaccine",  "Fatigue and mild muscle aches on day one after COVID vaccination. I rested and was completely fine by day two. Normal vaccine response."),
    ("BCG vaccine",    "Small scab forming at BCG injection site at two weeks — health worker confirmed this is the expected BCG response and not an adverse event."),
    ("pentavalent vaccine", "Low-grade fever of 37.9°C after pentavalent vaccine. Gave paracetamol syrup and baby was fine by the next morning. Nurse confirmed it was expected."),
    ("yellow fever vaccine", "Mild injection site pain after yellow fever vaccine. It was fine the next morning. Health worker said it was completely normal."),
    ("COVID-19 vaccine",  "Swollen lymph node under my arm after COVID vaccine on that side. Doctor said this is a known expected reaction that resolves on its own within weeks."),
    ("HPV vaccine",    "My daughter fainted briefly immediately after the HPV vaccination. She recovered in two minutes. Nurse said vasovagal syncope is expected — she was observed and discharged normally."),
    ("meningococcal vaccine", "Headache for one day after meningococcal vaccine. I took paracetamol and was fine the next day. The nurse said it was within expected effects."),
    ("pentavalent vaccine", "Baby was unsettled for one evening after the vaccination. She slept normally and was fine by morning. Nothing serious and all checks normal."),
    ("COVID-19 vaccine",  "I had a slightly sore arm for two days after my COVID booster. Nothing else. The pharmacist said it was completely within the expected response range."),
    ("BCG vaccine",    "The baby was slightly warm but not feverish after BCG. She fed well and slept through. Fine the next morning. Nurse told us it was normal."),
    ("hexavalent vaccine",  "Minor fussiness after hexavalent vaccination. Baby settled when held and was normal by the next morning. No fever, no rash, no vomiting. All normal."),
    ("artesunate injection", "Mild shivering after the artesunate injection — the nurse said it was expected and it passed within fifteen minutes. No further problems."),
    ("rotavirus vaccine", "Baby had very soft stool once after the rotavirus drops. No other symptoms. Health worker said this is expected. All fine."),
    ("OPV vaccine",    "Baby cried briefly after the oral polio drops but was consoled quickly. No fever, no rash. All normal at follow-up. Nothing concerning."),
    ("measles-rubella vaccine", "Low-grade fever after MR vaccination — temperature 37.7°C. Took paracetamol and was fine by the next morning. Nurse confirmed it was a normal vaccine response."),
    ("pentavalent vaccine", "The injection site was red and slightly swollen for one day after pentavalent vaccine. It was nothing serious and went away on its own. Nurse confirmed normal reaction."),
    ("COVID-19 vaccine",  "Second COVID dose caused mild fatigue and chills that evening. Paracetamol helped and I was fully normal by morning. Health worker said it was expected."),
    ("HPV vaccine",    "My daughter had a mild headache for one day after her HPV booster. She took paracetamol and was fine by the next morning. All normal."),
    ("meningococcal vaccine", "Slight redness at injection site after meningococcal vaccine. It resolved on its own within two days. Nothing else — all checks were normal."),
    ("BCG vaccine",    "BCG reaction at six weeks — small pustule forming at the site. Health worker confirmed this is the expected BCG response and should not be reported as an adverse event."),
    ("pentavalent vaccine", "After the pentavalent shot, my baby had mild fever and was fussy. I gave paracetamol syrup and she was fine by the next morning. Nurse said it was normal."),
]

P6_ADR = [
    # Serious AEFIs (reportable)
    ("pentavalent vaccine",    "After the pentavalent vaccination my baby went limp — completely floppy and not responding normally. Temperature was 40.1°C. We rushed to hospital. Doctor said it was a hypotonic hyporesponsive episode from the vaccine."),
    ("pentavalent vaccine",    "The baby had convulsions within three hours of the pentavalent vaccine — full tonic-clonic seizure lasting five minutes. Admitted to hospital. Serious AEFI confirmed."),
    ("BCG vaccine",            "Extensive swelling and pus at the BCG injection site spreading beyond 3 cm — the health worker said this was an extensive local reaction requiring medical review. Not the normal BCG response."),
    ("BCG vaccine",            "The BCG injection site became a large abscess after two months. The baby was brought to the clinic and the doctor confirmed it was a serious BCG injection site reaction requiring incision and drainage."),
    ("yellow fever vaccine",   "My sister's eyes became yellow two days after the yellow fever vaccination and she developed severe liver pain and high fever. Doctor said it was yellow fever vaccine-associated viscerotropic disease — a very serious reaction."),
    ("measles-rubella vaccine","My son developed severe encephalitis one week after the measles-rubella vaccination — high fever, seizures, and altered consciousness. Admitted to neurological ward. Serious AEFI."),
    ("pentavalent vaccine",    "The baby received pentavalent vaccine then three hours later the arm became massively swollen from shoulder to wrist and she was screaming in pain. Brought to emergency — serious local reaction."),
    ("COVID-19 vaccine",       "I developed sudden difficulty breathing and my face swelled within thirty minutes of the COVID vaccine. The nurse gave me adrenaline injection immediately. Confirmed anaphylaxis from the vaccine."),
    ("COVID-19 vaccine",       "I collapsed on the way out of the vaccination centre after my COVID shot. I woke up on the floor with nurses around me. They said it was a vasovagal syncope with loss of consciousness — I was observed for one hour before discharge."),
    ("rotavirus vaccine",      "My baby passed red jelly-like material in his nappy two days after the rotavirus vaccine and cried non-stop with his legs drawn up. Doctor diagnosed intussusception — emergency surgery performed."),
    ("OPV vaccine",            "My child developed weakness in one leg six weeks after the oral polio drops. The weakness got worse over two weeks and she cannot walk now. Doctors said it may be vaccine-derived poliovirus paralysis."),
    ("pentavalent vaccine",    "Baby was inconsolably crying for more than three hours after the pentavalent injection — high-pitched cry that we could not settle. She was admitted overnight. Persistent crying AEFI confirmed."),
    ("COVID-19 vaccine",       "Within minutes of the COVID vaccine I developed hives all over my body, throat tightening, and my blood pressure dropped. Treated with adrenaline in clinic — anaphylactic reaction to vaccine."),
    ("hexavalent vaccine",     "Three hours after the hexavalent vaccine my baby became very pale, floppy, and unresponsive with a weak pulse. Emergency admission — doctor diagnosed hypotonic hyporesponsive episode. Serious AEFI report filed."),
    ("yellow fever vaccine",   "The patient developed progressive liver failure with jaundice and coagulopathy ten days after yellow fever vaccination. ICU admission required. Yellow fever vaccine-associated viscerotropic disease confirmed."),
    ("BCG vaccine",            "The baby developed generalised BCG disease with widespread lymph node swelling and fever. The immunologist confirmed BCG dissemination due to underlying immune deficiency. Serious AEFI."),
    ("measles-rubella vaccine","High fever of 40.5°C and a generalised seizure eight days after measles vaccine. Child admitted to paediatric ward. Doctor confirmed febrile seizure as serious AEFI."),
    ("pentavalent vaccine",    "The baby's arm swelled to twice its size within an hour of the pentavalent injection with red streaks going up the arm. Emergency admission — doctor treated serious injection site infection."),
    ("COVID-19 vaccine",       "I developed chest pain and shortness of breath three days after my COVID-19 booster. ECG and troponin confirmed myocarditis. Cardiologist confirmed vaccine-associated myocarditis — serious AEFI."),
    ("HPV vaccine",            "My daughter collapsed immediately after the HPV vaccine — not vasovagal, she had a tonic-clonic seizure lasting two minutes. Admitted for observation. Neurologist assessed as serious AEFI."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 7 — Obstetric: midwife, TBA, postpartum psychiatric ADRs
# MODEL_FINDINGS Section 2.9 — institutional voice and Pidgin FNs; FP transient effects
# 20 ADR + 20 Non-ADR = 40 examples
# ════════════════════════════════════════════════════════════════════════════

P7_ADR = [
    # Midwife intrapartum ADR records
    ("oxytocin",    "08:45 Oxytocin infusion started per protocol. 09:30 — hyperstimulation noted: more than five contractions in ten minutes. FHR decelerations on CTG — late decelerations. Infusion stopped immediately. Paediatrician alerted. Assessed as oxytocin-induced hyperstimulation."),
    ("oxytocin",    "Uterine tachysystole following oxytocin: six contractions in ten minutes with fetal bradycardia on trace. CTG Category III. Oxytocin ceased, repositioned left lateral, oxygen given. Obstetric registrar called. ADR — uterine hyperstimulation."),
    ("misoprostol", "16:00 — Misoprostol 25mcg administered per cervical ripening protocol. 18:30 — tetanic contraction lasting 3 minutes with severe fetal bradycardia to 80 bpm. Emergency LSCS performed for fetal distress. Misoprostol-induced uterine hyperstimulation confirmed."),
    ("misoprostol", "Patient received misoprostol for induction of labour. Subsequently developed uterine rupture with sudden loss of FHR on CTG and severe abdominal pain. Laparotomy performed. Causality assessment: probable misoprostol ADR."),
    ("magnesium sulphate", "MgSO4 infusion for eclampsia prophylaxis. Urinary output fell to 15 ml/hour and respiratory rate 8/min with loss of patellar reflexes — signs of magnesium toxicity. Infusion stopped, calcium gluconate administered. Confirmed MgSO4 overdose reaction."),
    ("oxytocin",    "Midwife note 14:20: uterine hyperstimulation with prolonged contraction five minutes duration. FHR showing variable decelerations with poor recovery. Oxytocin rate reduced and tocolysis requested. Reaction to oxytocin — documented as intrapartum ADR."),
    # TBA / community midwife reports
    ("herbal uterotonic", "She was having fits after the traditional birth attendant gave her something to speed up the labour. Her blood pressure was very high and she lost consciousness. She was rushed to the health centre — they said the traditional medicine caused the problem."),
    ("herbal uterotonic", "TBA gave the woman herbal medicine to hasten delivery. Contractions became continuous and the baby's heartbeat became very weak. The TBA brought her in. Doctor said the traditional medicine caused uterine hyperstimulation."),
    ("unknown uterotonic", "The traditional medicine man gave her something to drink for the delivery. After that she had very strong non-stop pain and the baby could not come. Health worker said it was an abnormal contraction from the traditional drug."),
    # Postpartum psychiatric ADRs
    ("pethidine",   "Patient administered pethidine for postpartum pain. Developed marked dysphoria and depersonalisation — reported feeling she was watching herself from outside. 'I am not in my own body' — she was terrified. Opioid-induced dysphoric reaction confirmed."),
    ("methyldopa",  "Two weeks postpartum on methyldopa for chronic hypertension. Presenting with persistent low mood, psychomotor slowing, flat affect and inability to bond with baby. Psychiatry reviewed — methyldopa-induced depressive syndrome."),
    ("morphine",    "Patient post-caesarean on morphine PCA. At 36 hours she became acutely confused and reported seeing people in the room who were not there. Opioid-induced delirium confirmed — morphine dose reduced and haloperidol administered."),
    ("pethidine",   "Postpartum patient — thought the baby had been replaced by another baby. Confused, agitated and suspicious. After pethidine injection for pain. Opioid-induced psychosis assessed and antipsychotic medication started."),
    ("methyldopa",  "Week three postpartum. Patient on methyldopa for pre-eclampsia. Presenting with severe postpartum depression and suicidal ideation. Psychiatrist consulted and confirmed methyldopa as contributing factor. Drug switched."),
    # Pidgin obstetric ADRs
    ("oxytocin",    "My belle dey pain me too much — like fire inside — since dem put the drip for my hand for speed up the delivery. The pain no stop at all, it continuous."),
    ("misoprostol", "After dem put the misoprostol tablet for ripen my womb, pain dey come one after another without stopping — no rest between. Midwife say contraction too too much from the tablet."),
    ("oxytocin",    "Since dem start the injection to bring the labour, my womb never rest at all — pain constant. Midwife check baby heartbeat and run call doctor — she say na the drip cause problem for baby."),
    ("herbal uterotonic", "After I drink the something the TBA give me, my belly start pain like I go burst. Baby heartbeat go down and dem take me hospital emergency — doctor say na the local medicine cause am."),
    ("magnesium sulphate", "After the MgSO4 drip start, I no fit breathe well and my legs go heavy — reflex check show nothing. Nurse say magnesium too high in blood — drip stop and dem give me injection to reverse."),
    ("pethidine",   "After the pethidine injection everything feel unreal — like say I watching myself from far away, I no dey in my body. This na not normal feeling from pain medicine."),
]

P7_NON_ADR = [
    # Uneventful oxytocin monitoring
    ("oxytocin",    "Oxytocin infusion commenced at 08:00 for augmentation of labour. Contractions three in ten minutes, well-tolerated. FHR baseline 140 bpm, reactive. Progress normal. No adverse event noted. Delivery at 15:30 — good outcome."),
    ("oxytocin",    "Labour augmented with oxytocin — contractions established at four in ten minutes with good FHR variability throughout. Patient tolerated the infusion well with no adverse effects. Normal vaginal delivery achieved."),
    ("misoprostol", "Misoprostol 25 mcg administered for cervical ripening. Mild cervical cramping for twenty minutes — midwife informed patient this was expected. Contractions established normally thereafter. Uncomplicated progress."),
    ("misoprostol", "I felt cold briefly after the misoprostol tablet but it passed quickly. The midwife told me it was expected. Baby moving normally after. Everything went fine."),
    # Uneventful MgSO4 monitoring
    ("magnesium sulphate", "MgSO4 infusion for eclampsia prophylaxis. Reflexes were checked every hour as instructed. Urinary output satisfactory at 50 ml/hour. Respiratory rate 16/min. No signs of toxicity throughout infusion."),
    ("magnesium sulphate", "Patient on MgSO4 — 4g loading dose then 1g/hour maintenance. Hourly monitoring of reflexes, urine output and respiratory rate all within normal parameters. Uneventful antenatal monitoring. No adverse event."),
    # Uneventful post-delivery injection
    ("oxytocin",    "Post-delivery oxytocin 10 units IM administered. Patient was cold and shaky for ten minutes — the midwife told her it was normal and it passed within ten minutes. Uterus well-contracted. Observation complete, discharged."),
    ("oxytocin",    "Routine post-partum Syntocinon injection. Mild shivering noted — midwife confirmed this is expected and it resolved spontaneously within fifteen minutes. Post-partum check satisfactory."),
    # Iron supplementation in pregnancy
    ("ferrous sulphate", "I have been taking my iron every day in pregnancy. I feel well and my energy is better than before. Doctor checked and everything is fine with me and the baby."),
    ("ferrous sulphate", "Pregnant patient on ferrous sulphate and folic acid. Mild constipation on day two — health worker advised high fibre diet and it resolved. No ADR, drug continued. Baby scan normal."),
    # Uneventful antimalarial in pregnancy
    ("artemether-lumefantrine", "I had some nausea in the first two days of malaria treatment but it settled after I took the tablet with food. Baby has been moving normally throughout. No adverse event."),
    ("artemether-lumefantrine", "Maternal treatment with artemether-lumefantrine for uncomplicated malaria in second trimester. Nausea mild and resolved by day three. Fetal movements normal throughout. Outcome: healthy delivery."),
    # Uneventful antihypertensive in pregnancy
    ("methyldopa",  "Patient on methyldopa 250mg TDS for gestational hypertension. Blood pressure well-controlled. No symptoms of depression or slowing noted at follow-up. Patient coping well. Drug continued."),
    ("labetalol",   "Inpatient blood pressure management with labetalol IV. BP responded well. Patient reported mild tiredness which is expected from the drug. No chest pain, no dyspnoea. Normal fetal monitoring. Discharged stable."),
    # Uneventful antibiotics in pregnancy
    ("amoxicillin", "I took the antibiotic for the infection in pregnancy and I had no side effect at all. Baby moving well and scan showed normal development."),
    ("co-trimoxazole", "Patient on co-trimoxazole prophylaxis in pregnancy — no allergic reaction, no rash. Full blood count normal. Baby at twenty weeks scan — normal anatomy. Drug continued without issue."),
    # Normal TBA report
    ("herbal supplement", "The traditional birth attendant said the mother had a normal delivery at home with no complications. She did not give any traditional medicine. Both mother and baby are well."),
    ("herbal supplement", "Community midwife report: uncomplicated home delivery. No uterotonics administered by TBA. Mother and baby healthy. No adverse event to report."),
    # Postpartum normal
    ("ferrous sulphate", "Two weeks postpartum on ferrous sulphate for anaemia. Patient reports feeling well and bonding with the baby normally. No mood changes. Iron levels improving on repeat blood test."),
    ("ferrous sulphate", "Postpartum iron supplementation — patient tolerating well, energy improving, no gastrointestinal side effects. Baby breastfeeding normally. No adverse event."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 8 — Fetal / neonatal ADR framing (MODEL_FINDINGS Section 2.10)
# Mother narrating fetal/neonatal ADR
# 10 ADR + 10 Non-ADR = 20 examples
# ════════════════════════════════════════════════════════════════════════════

P8_ADR = [
    # Fetal abnormality from maternal drug
    ("artemisinin",  "The scan shows the baby has not developed normally since I started taking the drug at twelve weeks pregnant. The doctor thinks it is from the artemisinin I took in the first trimester."),
    ("artemisinin",  "The ultrasound found that the baby had not developed normally and the doctor said it may be from the artemisinin medication I was given before I knew I was pregnant."),
    ("methotrexate", "My baby was born with bone problems — the doctor said it was because I was taking methotrexate before we knew about the pregnancy. The drug affected the baby's development."),
    ("phenytoin",    "My child has a cleft lip and learning problems. The neurologist said it is fetal phenytoin syndrome from the epilepsy drug I was on throughout pregnancy before they switched me."),
    ("warfarin",     "The baby was born with the warfarin embryopathy — the rheumatologist confirmed that the nose bone and bone development problems are from the blood thinner I took in the first trimester."),
    # Neonatal ADR narrated by mother
    ("co-trimoxazole","My baby's skin turned yellow in the first week — the doctor said it is because of the co-trimoxazole I was taking during pregnancy. The drug caused the baby's jaundice."),
    ("co-trimoxazole","The baby has jaundice and the neonatologist told me it is from the medicine I was on during pregnancy — the co-trimoxazole. My baby has a G6PD deficiency and the drug triggered the jaundice."),
    ("isoniazid",    "My baby was born and started shaking on day two. The doctor said it is neonatal withdrawal from the isoniazid — the anti-TB drug I was on throughout pregnancy. The baby needed treatment."),
    ("diazepam",     "My newborn was very floppy and could not suck well. The neonatal team said it was because of the diazepam I was given before delivery — the baby had neonatal sedation syndrome."),
    ("pethidine",    "My baby was born not breathing well — very slow breathing and could not cry. The midwife said it was because I had pethidine for pain before delivery and it crossed to the baby. They gave the baby an antidote injection."),
]

P8_NON_ADR = [
    # Normal fetal development on drugs
    ("ferrous sulphate", "I have been on ferrous sulphate and folic acid throughout the pregnancy. The scan at twenty weeks showed completely normal baby development. No concerns."),
    ("methyldopa",  "Patient on methyldopa for gestational hypertension throughout pregnancy. Anatomy scan at twenty weeks — all structures normal. No evidence of drug effect on fetal development."),
    ("labetalol",   "I was on labetalol for blood pressure in pregnancy. All my baby scans were normal and my baby was born healthy with good APGAR scores. No problems at all."),
    ("amoxicillin", "I took amoxicillin for a urine infection in the second trimester. Baby scan was normal and delivery was uncomplicated. Baby is fine and developing normally."),
    # Neonatal jaundice from non-drug cause
    ("ferrous sulphate", "My baby has jaundice but the paediatrician confirmed it is normal physiological jaundice — nothing to do with the iron tablets I was taking. The bilirubin is coming down with phototherapy."),
    ("co-trimoxazole","The baby has mild jaundice but the neonatologist tested for G6PD and it was normal. The jaundice is physiological — not related to the co-trimoxazole I took. Baby is fine."),
    # Birth defects from genetic cause
    ("ferrous sulphate", "My baby was born with Down syndrome. The geneticist confirmed it is a chromosomal condition — completely unrelated to any medication I took during pregnancy."),
    ("ferrous sulphate", "The baby has a heart defect that was found on the antenatal scan. The cardiologist explained it is a structural defect that developed before eight weeks and is unrelated to any drug I took."),
    # Neonatal transition — not drug
    ("ferrous sulphate", "My newborn had some breathing difficulty in the first minutes after birth. The neonatologist said it was normal transitional breathing and confirmed no drug exposure caused it. Baby is now well."),
    ("ferrous sulphate", "Baby was born healthy with APGAR 9 at 1 minute and 10 at 5 minutes. No jaundice, no feeding problems. I was only on folic acid and iron throughout — no adverse effect on the baby."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 10 — Age-specific voices (MODEL_FINDINGS Section 2.7)
# Child, adolescent, elderly, low-literacy
# 25 ADR + 25 Non-ADR = 50 examples
# ════════════════════════════════════════════════════════════════════════════

P10_ADR = [
    # Child self-report (age 5–12 voice)
    ("amoxicillin",  "My tummy hurts a lot since I started taking the medicine. I feel like vomiting after every tablet. Mummy says I have to take it but it makes me feel yucky."),
    ("co-trimoxazole","My tummy hurts and I feel sick every time I take the white tablet. I told my mummy and she said I should tell the nurse."),
    ("artemether-lumefantrine", "The malaria medicine makes me feel bad inside my tummy. I want to vomit after I take it and my head hurts a lot."),
    ("amoxicillin",  "After I take the antibiotic tablet my tummy hurts so much I cannot sleep. Mummy had to rub my tummy and I still felt sick. I don't want to take it anymore."),
    ("co-trimoxazole","I am having a bad rash all over my body since I started the white tablet. It is itchy and I am scratching all the time. Mummy took me to the clinic."),
    ("artesunate",   "Since they gave me the injection for malaria I feel dizzy and my head is paining me plenty. I cannot go to school."),
    ("artemether-lumefantrine", "I feel like there is something turning in my tummy after I take the malaria tablet. I vomited two times. My teacher sent me home."),
    ("amoxicillin",  "My skin got spots all over since I started taking the tablet. I showed teacher and she said I should go to the nurse. It is itchy everywhere."),
    # Adolescent ADR with minimization / peer-attribution
    ("metformin",    "I have been drinking a lot of water since I started the diabetes tablet. I thought maybe it was because I was not drinking enough water before. But I am drinking plenty now and still thirsty."),
    ("valproate",    "My hair is falling out since I started the new epilepsy tablet. I told my mum but she said maybe it's just stress from school. But it's a lot of hair and it started exactly when I started the medicine."),
    ("isotretinoin", "Since I started the acne tablet my lips are very dry and cracked and my skin peels all the time. I thought it was the dry harmattan weather but it started when I started the drug."),
    ("metformin",    "I have been passing urine so many times since I started the diabetes medicine — even at night. My friend said maybe I am drinking too much water. But I am not — the medicine is making me pass too much water."),
    ("doxycycline",  "I am sensitive to the sun since I started the antibiotic. I got a bad sunburn reaction on my face even from a short walk outside. It started with the drug."),
    # Elderly ADR reports
    ("amlodipine",   "Since my doctor started me on this new blood pressure tablet last month, my ankles have been swelling badly. I can no longer wear my church shoes. My daughter says it is the new medicine."),
    ("simvastatin",  "Since I started the cholesterol tablet I have terrible muscle pain in my thighs and calves. I cannot climb the stairs properly anymore. My children say it started with the new tablet."),
    ("metformin",    "I am having bad loose stool multiple times a day since I started the diabetes medicine. I am eighty years old and this has never happened to me before. It is the tablet causing it."),
    ("amlodipine",   "My legs are very heavy and swollen since the new blood pressure medicine was started. I had to buy bigger shoes. At my age this is embarrassing and I know it is the tablet."),
    ("atenolol",     "Since they gave me the new heart tablet my head feels heavy and I am slow — I cannot think clearly. My children say I am not myself since the new medicine."),
    # Low-literacy telegraphic ADR
    ("artesunate",   "After tablet. My head. Pain too much. I no fit work. I go toilet plenty. Since I start the medicine."),
    ("isoniazid",    "Start tablet. Three days. Legs. Hands. Dey pain me. Numbness. I go hospital. Doctor say na the TB medicine."),
    ("amoxicillin",  "After. Antibiotic. My belly. Too much. Run toilet many times. Vomit also. Since start drug."),
    ("efavirenz",    "I start the HIV tablet. Night. Bad dreams. No fit sleep. Wake up plenty. Na the efavirenz. Doctor say."),
    ("metformin",    "Sugar tablet. Start it. Now stomach. Pain. Plenty. Loose stool. Every morning. Since tablet start."),
    ("co-trimoxazole","Take the white tablet. Two days. Skin come out. Rash everywhere. Itching plenty. Na the tablet."),
]

P10_NON_ADR = [
    # Child — pre-existing or unrelated
    ("amoxicillin",  "My daughter had a tummy ache before I started giving her the antibiotic — it was there from the stomach infection. The doctor said the drug is fine and the pain is from the illness, not the medicine."),
    ("co-trimoxazole","The child had a rash before we started the co-trimoxazole — the nurse confirmed it was from the food allergy that was already there. The medicine did not cause it."),
    ("artemether-lumefantrine","My son said his tummy hurt but the doctor checked and said it was from the malaria itself. The medicine is fine — no side effect."),
    ("amoxicillin",  "My daughter felt a bit nauseous but the doctor said it was from the throat infection, not the antibiotic. She finished the course and is now fine."),
    ("artesunate",   "My child was dizzy but the doctor confirmed it was from the fever itself, not from the injection. After the fever came down the dizziness stopped."),
    # Adolescent — non-ADR
    ("metformin",    "My daughter has been well on the diabetes tablet. No excessive thirst, no unusual urination. She plays sports normally and says she has no complaints from the medicine."),
    ("valproate",    "My son started the epilepsy tablet three months ago. No hair loss, no side effects. He is doing well at school and his seizures are controlled."),
    ("doxycycline",  "I have been on the antibiotic for two weeks with no sun sensitivity or skin reaction. I go outside normally and have no complaints from the drug."),
    ("isotretinoin", "My daughter has been on the acne treatment for one month. No severe dryness, no mood changes. She is tolerating it well and the acne is improving."),
    ("metformin",    "I am taking the diabetes medicine and passing urine normally — not more than before. Doctor checked my kidneys and they are fine. No problem from the tablet."),
    # Elderly — "no new problems"
    ("amlodipine",   "I have been on the new blood pressure tablet for three months. No new problems with the medication. My ankles are fine and I sleep well. My daughter says I am back to my old self."),
    ("simvastatin",  "I started the cholesterol tablet six weeks ago. No muscle pain, no weakness. I walk to the market normally and have no complaints from the new medicine."),
    ("metformin",    "The diabetes tablet has not caused me any problems. No diarrhoea, no nausea. I am eighty and I am tolerating it well. Doctor says my sugar is well-controlled."),
    ("atenolol",     "I have been on the heart tablet for six months. No dizziness, no slowing of my thinking. My family says I am sharp as ever. No new problems with this medicine."),
    ("amlodipine",   "My legs are not swelling since I started the blood pressure tablet. I can still wear my normal shoes. No new problems — the medicine is agreeing with me."),
    # Low-literacy — no ADR
    ("artesunate",   "Take tablet. Feel fine. No problem. Finish treatment. All good."),
    ("isoniazid",    "TB medicine. Take every day. No pain. Legs fine. Hands fine. All normal."),
    ("amoxicillin",  "Antibiotic. Take it. No problem. Belly fine. No vomiting. All fine."),
    ("efavirenz",    "HIV tablet. Take it. Sleep normal. No bad dream. Fine. Three months. No problem."),
    ("metformin",    "Sugar tablet. Take every day. Stomach fine. No loose stool. All normal."),
    # General adolescent/child non-ADR
    ("co-trimoxazole","My child took the full course of co-trimoxazole and had no reaction at all. No rash, no fever. Doctor confirmed the drug was well tolerated."),
    ("amoxicillin",  "My son finished the full antibiotic course without any side effect. He said his tummy was fine the whole time. No problems at all."),
    ("artemether-lumefantrine","My daughter completed malaria treatment with no side effects. No vomiting, no dizziness. She was back at school within three days."),
    ("ferrous sulphate","I am a sixteen-year-old girl taking iron tablets for anaemia. I have no side effects — no constipation, no nausea. My haemoglobin is improving."),
    ("valproate",    "I am a teenage girl on valproate for epilepsy. I have no hair loss, no weight gain. Periods are regular. I am tolerating the drug well."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 11 — Clinical record shorthand (MODEL_FINDINGS Section 2.14)
# Discharge summaries, nursing DAR, ward notes, physician impression
# 20 ADR + 20 Non-ADR = 40 examples
# ════════════════════════════════════════════════════════════════════════════

P11_ADR = [
    # Discharge summaries — explicit ADR as principal diagnosis
    ("co-trimoxazole", "Discharge Summary. Diagnosis: Acute allergic reaction secondary to co-trimoxazole. Presentation: urticaria, angioedema, and stridor. Management: adrenaline IM, antihistamine, steroids. Drug permanently discontinued."),
    ("gentamicin",     "Discharge Summary. Principal diagnosis: gentamicin-induced acute kidney injury. Creatinine on admission 95 μmol/L; peak creatinine on day 4: 380 μmol/L. Gentamicin held. Renal function recovering at discharge."),
    ("hydralazine",    "Discharge Summary — Impression: Drug-induced lupus syndrome most likely attributable to hydralazine. ANA positive, anti-histone antibodies positive. Hydralazine ceased. Commenced hydroxychloroquine. Rheumatology review arranged."),
    ("isoniazid",      "Discharge Summary. Principal diagnosis: isoniazid-induced peripheral neuropathy. Patient on TB therapy for four months. Tingling and numbness in both feet. Pyridoxine 50mg added; isoniazid dose reduced. Neurology follow-up."),
    ("methotrexate",   "Discharge Summary — Diagnosis: methotrexate-induced hepatotoxicity. LFTs on admission: ALT 480 U/L (12× ULN). Methotrexate held. LFTs improving at discharge: ALT 95 U/L. Gastroenterology review for further management."),
    ("efavirenz",      "Discharge Summary. Diagnosis: efavirenz-induced neuropsychiatric syndrome. Patient presented with acute psychosis, hallucinations, and aggression. Efavirenz substituted to dolutegravir. Psychiatry inpatient review completed. Improved on discharge."),
    ("co-trimoxazole", "D/C Summary — Allergic reaction secondary to co-trimoxazole: drug withdrawn D1. Urticaria and angioedema resolved after two days antihistamine. Discharged with drug allergy alert documented. Co-trimoxazole contraindicated."),
    ("rifampicin",     "Discharge Summary. Principal diagnosis: Drug-induced cholestatic hepatitis secondary to rifampicin. Jaundice on D7 of TB therapy. Bili 180 μmol/L. Rifampicin withheld. LFTs normalising. Sequential reintroduction planned."),
    # Nursing DAR timestamped notes
    ("metronidazole",  "DAR 09:45 — Patient on metronidazole 400mg TDS. Reported nausea at 14:30 and vomiting x2 at 15:10. Observations: T37.2, BP 110/70. Dr [initials] notified 15:15. Metronidazole dose reviewed; anti-emetic prescribed. 18:00 — patient settled."),
    ("isoniazid",      "Nursing note 06:00: Patient reports tingling and burning pain in both hands and feet. Started on isoniazid four weeks ago. New complaint — not present on previous shift. Reported to medical team. Pyridoxine prescribed 06:30."),
    ("quinine",        "DAR 11:30: IV quinine infusion running. Patient developed tinnitus and reported visual blurring at 12:00. Quinine rate reduced per protocol. Medical team notified. 14:00: symptoms improved. Cinchonism documented."),
    ("ampicillin",     "22:15 — Patient developed generalised urticaria within 30 minutes of first ampicillin IV dose. Infusion stopped immediately. Adrenaline 0.5mg IM given. Antihistamine IV. Dr called stat. Drug allergy confirmed."),
    # Ward round notes
    ("haloperidol",    "Ward round — Day 3. Patient on haloperidol 5mg IM. Overnight: involuntary jaw movements and lip smacking since 0600 hours. Nurse reported rhythmic facial movements. Extrapyramidal reaction. Haloperidol dose reduced, procyclidine added."),
    ("chlorpromazine", "Ward round — D2 chlorpromazine. Patient holding head fixed upward, unable to look down, neck extended. Eyes deviated upward. Diagnosed oculogyric crisis. Biperiden given IM with resolution. Drug dose adjusted."),
    ("isoniazid",      "Ward round. Patient five weeks post TB treatment commencement. New complaint: numbness hands and feet since one week. Examination: stocking-glove sensory loss. Impression: isoniazid-induced peripheral neuropathy. Pyridoxine added."),
    ("gentamicin",     "Ward round D5. Gentamicin continued for sepsis. Creatinine today 290 μmol/L (was 95 μmol/L on admission). Drug nephrotoxicity suspected. Drug level sent. Nephrology review requested. Gentamicin held pending result."),
    # Physician impression / clinical notes
    ("hydralazine",    "Impression: clinical picture consistent with drug-induced systemic lupus erythematosus. Most likely attributable to hydralazine commenced eight months ago. ANA 1:320, anti-histone antibody positive. Causality: probable."),
    ("isoniazid",      "Impression: Drug-induced hepatotoxicity most likely secondary to isoniazid. ALT 420 U/L at week six of TB treatment. Patient symptomatic with nausea and right upper quadrant pain. Drug held pending liver biopsy result."),
    ("efavirenz",      "Impression: Efavirenz-induced neuropsychiatric toxicity. Patient commenced efavirenz eight weeks ago and presenting with vivid nightmares, disorientation, and aggression. Causality: probable. Switch to dolutegravir recommended."),
    ("co-trimoxazole", "Impression: Adverse drug reaction secondary to co-trimoxazole. Presentation: Stevens-Johnson syndrome, target lesions on trunk and oral mucosal involvement. SCORTEN score calculated. Dermatology review requested. Drug permanently contraindicated."),
]

P11_NON_ADR = [
    # Uneventful discharge summaries
    ("amoxicillin",    "Discharge Summary. Diagnosis: community-acquired pneumonia. Treatment: amoxicillin-clavulanate. Patient completed seven-day course without any adverse drug reaction. Discharged well. No drug allergy documented."),
    ("isoniazid",      "Discharge Summary. Patient completed initial two months of TB therapy (HRZE). No drug reaction recorded. LFTs monitored — within normal limits throughout. Discharged to outpatient follow-up. Continuing treatment."),
    ("metformin",      "Discharge Summary. Type 2 diabetes mellitus. Metformin commenced during admission. Well tolerated. No hypoglycaemia, no lactic acidosis risk factors. Patient educated on drug. Discharged on metformin 500mg BD."),
    ("co-trimoxazole", "Discharge Summary. HIV — started co-trimoxazole prophylaxis during admission. No allergic reaction. No rash. Discharge with prophylaxis confirmed. Patient counselled."),
    ("amlodipine",     "Discharge Summary. Hypertension. New antihypertensive started — amlodipine 5mg OD. BP well controlled. No ankle oedema, no headache. Patient tolerating drug well. Discharged."),
    # Normal nursing DAR notes
    ("ferrous sulphate","DAR 08:00 — Patient on ferrous sulphate. Reports mild constipation. Health worker advised high fibre diet and fluids. No nausea, no pain. Normal bowel motion expected. Drug continued."),
    ("amoxicillin",     "Nursing note: Day 3 oral amoxicillin for wound infection. No signs of allergic reaction. Wound healing well. Patient afebrile. No adverse effects documented. Continue as prescribed."),
    ("isoniazid",       "Nursing note 09:00: Patient on TB therapy, month four. No new complaints. LFTs checked this week — normal. Patient reports no tingling or numbness. Continue current regimen."),
    ("metronidazole",   "DAR 10:30 — Patient received IV metronidazole for abdominal infection. No nausea, no vomiting. Patient comfortable. Obs stable: T37.1, BP 120/78. Infusion completed without incident. Continue as charted."),
    ("co-trimoxazole",  "Nursing note 14:00: Patient on co-trimoxazole day 5. No rash observed. Patient feels well. No itching, no fever. Drug tolerated without adverse effect. Continue as prescribed."),
    # Normal ward round notes
    ("isoniazid",       "Ward round D30 of TB therapy. Patient well. No peripheral neuropathy symptoms. LFTs reviewed — within normal range. Compliance confirmed. No side effects reported. Continue first-line TB treatment."),
    ("metformin",       "Ward round. Diabetes management review. Metformin 1g BD — well tolerated. No diarrhoea, no nausea. HbA1c improving. Renal function normal. Continue current regimen."),
    ("amlodipine",      "Ward round. Patient on amlodipine for six months. BP at target. No ankle oedema, no headache. Heart rate normal. Drug well tolerated. Continue."),
    ("efavirenz",       "Ward round. HIV review — patient on efavirenz-based regimen. Viral load suppressed. No neuropsychiatric symptoms reported. Sleep normal. Patient happy with regimen. Continue."),
    ("co-trimoxazole",  "Ward round. Day 3 co-trimoxazole for Pneumocystis prophylaxis. No rash, no fever, no nausea. FBC normal. G6PD normal. Drug well tolerated. Continue as planned."),
    # Normal physician impressions
    ("amoxicillin",     "Impression: uncomplicated community pneumonia responding well to amoxicillin. No drug adverse effects. Patient improving clinically. Continue antibiotic for full seven-day course."),
    ("metformin",       "Impression: type 2 diabetes mellitus with good glycaemic control on metformin. No evidence of lactic acidosis risk. Renal function normal. Continue metformin."),
    ("isoniazid",       "Impression: TB treatment progressing well. No drug-induced hepatotoxicity or peripheral neuropathy. Patient compliant and tolerating all four drugs without adverse effect."),
    ("amlodipine",      "Impression: essential hypertension, well controlled on amlodipine monotherapy. No adverse drug effects. Continue current management."),
    ("co-trimoxazole",  "Impression: HIV — co-trimoxazole prophylaxis well tolerated. No allergic reaction documented after three months. Continue prophylaxis per guidelines."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 15 — Male reproductive ADR (MODEL_FINDINGS Section 2.7)
# Ghanaian idiomatic framing — erectile dysfunction and sexual ADRs
# 10 ADR + 10 Non-ADR = 20 examples
# ════════════════════════════════════════════════════════════════════════════

P15_ADR = [
    ("amlodipine",    "Since I started the blood pressure medicine I am having problems with my manhood. I cannot perform as a man should and my wife is not happy. I know it is the tablet causing it."),
    ("atenolol",      "Since the doctor gave me this blood pressure tablet my man-power has gone down. I struggle to satisfy my wife. I did not have this problem before the medicine."),
    ("spironolactone","Since starting spironolactone my breasts are growing and I feel shame in front of my colleagues. I am a man and this is not right. It started exactly when the drug was started."),
    ("cimetidine",    "The stomach tablet has caused me problems in the bedroom. I cannot function as a man since I started it. My wife has noticed and I am embarrassed. It is the drug."),
    ("metoclopramide","Since they gave me the injection for nausea, my breasts are tender and swollen. As a man this is very embarrassing. The doctor said it is from the injection."),
    ("haloperidol",   "Since starting the psychiatric tablet I have no interest in my wife at all. My manhood does not work. I told the doctor and he confirmed it is from the haloperidol."),
    ("metoprolol",    "The heart tablet has affected my ability to perform in the bedroom. I have problems with erection since I started metoprolol. This never happened before."),
    ("finasteride",   "Since I started the prostate tablet my interest in my wife has gone completely. I cannot perform and I told my doctor — he said it is a known side effect of the drug."),
    ("spironolactone","I am ashamed to say it but since spironolactone, my chest is swollen and I cannot have relations with my wife. As a man in Ghana this is very difficult to live with."),
    ("atenolol",      "I told my pharmacist that since the blood pressure tablet, I have been unable to be a man for my wife. He said this is a side effect and I should discuss with my doctor about changing the drug."),
]

P15_NON_ADR = [
    ("amlodipine",    "I have been on the blood pressure tablet for six months and I have no problems in the bedroom. My manhood is working normally and my wife is satisfied. No side effect."),
    ("atenolol",      "I discussed concerns about the heart tablet with my doctor. He checked and confirmed that my problems in the bedroom are from my age and diabetes — not from the atenolol. The drug is fine."),
    ("spironolactone","I have been on spironolactone for three months. No breast swelling, no changes in my manhood. I am tolerating the drug well and my wife says I am fine."),
    ("metoprolol",    "My doctor told me that the problem with my manhood started before I was put on the heart tablet. My blood sugar has been high for years and that is the cause — not the metoprolol."),
    ("haloperidol",   "I told the doctor about my concerns but he examined me and said my problem in the bedroom is from the depression itself, not from the haloperidol tablet. The drug is not the cause."),
    ("amlodipine",    "I have had problems with my manhood since long before the blood pressure medicine. I was checked by the urologist and he confirmed it is vascular — the amlodipine is not the cause."),
    ("atenolol",      "I am on atenolol for my heart and I have no problems at all in the bedroom. I perform normally as a man. The drug has not affected me in that way."),
    ("cimetidine",    "I finished the stomach treatment and my manhood is functioning normally. I had no problems in the bedroom during or after the cimetidine course."),
    ("metoprolol",    "My doctor explained that the problem I described started two years before the heart tablet was prescribed. The cause is my long-standing diabetes — not the metoprolol."),
    ("finasteride",   "I discussed the prostate tablet with my doctor and he confirmed that my libido issues started before the finasteride — they are from stress and my age, not from the drug."),
]


# ════════════════════════════════════════════════════════════════════════════
# Assembly
# ════════════════════════════════════════════════════════════════════════════

def build_records():
    records = []
    n = 1

    # P5 — Pidgin augmentation
    for drug, sent in P5_ADR:
        records.append(make(drug, sent, 5, "pidgin_adr", n, True)); n += 1
    for drug, sent in P5_NON_ADR:
        records.append(make(drug, sent, 5, "pidgin_nonadr", n, False)); n += 1

    # P6 — AEFI
    n = 1
    for drug, sent in P6_NON_ADR:
        records.append(make(drug, sent, 6, "aefi_nonadr", n, False)); n += 1
    n = 1
    for drug, sent in P6_ADR:
        records.append(make(drug, sent, 6, "aefi_adr", n, True)); n += 1

    # P7 — Obstetric
    n = 1
    for drug, sent in P7_ADR:
        records.append(make(drug, sent, 7, "obstetric_adr", n, True)); n += 1
    n = 1
    for drug, sent in P7_NON_ADR:
        records.append(make(drug, sent, 7, "obstetric_nonadr", n, False)); n += 1

    # P8 — Fetal/neonatal
    n = 1
    for drug, sent in P8_ADR:
        records.append(make(drug, sent, 8, "fetal_adr", n, True)); n += 1
    n = 1
    for drug, sent in P8_NON_ADR:
        records.append(make(drug, sent, 8, "fetal_nonadr", n, False)); n += 1

    # P10 — Age-specific
    n = 1
    for drug, sent in P10_ADR:
        records.append(make(drug, sent, 10, "age_adr", n, True)); n += 1
    n = 1
    for drug, sent in P10_NON_ADR:
        records.append(make(drug, sent, 10, "age_nonadr", n, False)); n += 1

    # P11 — Clinical record shorthand
    n = 1
    for drug, sent in P11_ADR:
        records.append(make(drug, sent, 11, "clinical_adr", n, True)); n += 1
    n = 1
    for drug, sent in P11_NON_ADR:
        records.append(make(drug, sent, 11, "clinical_nonadr", n, False)); n += 1

    # P15 — Male reproductive
    n = 1
    for drug, sent in P15_ADR:
        records.append(make(drug, sent, 15, "male_repro_adr", n, True)); n += 1
    n = 1
    for drug, sent in P15_NON_ADR:
        records.append(make(drug, sent, 15, "male_repro_nonadr", n, False)); n += 1

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

    # Per-priority summary
    from collections import Counter
    by_prio = Counter()
    for r in records:
        p = r["setid"].split("_")[2]
        by_prio[p] += 1
    print("\nPer-priority counts:")
    for p in sorted(by_prio):
        print(f"  p{p}: {by_prio[p]}")

    print("\nNOTE: Use this data for CLF training ONLY.")
    print("      Do not pass to NER — adr_spans are empty.")


if __name__ == "__main__":
    main()
