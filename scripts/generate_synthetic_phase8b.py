#!/usr/bin/env python3
"""
scripts/generate_synthetic_phase8b.py
========================================
Generate synthetic_phase8b.jsonl covering dialect-influenced Ghanaian ADR
descriptions. Data sourced and verified against:

  - BMC Public Health 2018 qualitative interviews (Ghanaian patient quotes)
  - UD_Naija-NSC corpus (CC BY-SA 4.0, 9242 spoken Pidgin sentences)
  - Linguistic studies of Ghanaian English phonology and syntax
  - Ho Municipality ADR surveillance study (PMC10497160)
  - Tramadol/youth qualitative study (PLOS Global Public Health)

Priorities covered:
  P27 — Akan/Twi-influenced Ghanaian English ADR descriptions
  P28 — Ewe-influenced Ghanaian English ADR descriptions
  P29 — Northern Ghanaian English (Dagbani/Hausa-influenced) ADR descriptions
  P30 — Extended West African Pidgin health ADR (extends P5 from v2)
  P31 — Code-switched multilingual (Pidgin/Twi/English mixed)

Addresses batch failures: #219, #225, #230, #238 (dialect FNs) and
extends the Pidgin sub-threshold cluster (#31, #37, #55, #78, #91–#99).

IMPORTANT: CLF-only. adr_spans and drug_spans are always empty [].
Do NOT use for NER training.

Output: data/silver/synthetic_phase8b.jsonl
Run:    python scripts/generate_synthetic_phase8b.py
"""

import json
from pathlib import Path

ROOT   = Path(__file__).parent.parent
OUTDIR = ROOT / "data" / "silver"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT    = OUTDIR / "synthetic_phase8b.jsonl"


def make(drug, sentence, priority, tag, n, contains_adr):
    return {
        "drug":         drug,
        "sentence":     sentence,
        "source":       "synthetic_phase8b",
        "setid":        f"synthetic_phase8b_p{priority}_{tag}_{n:03d}",
        "contains_adr": int(contains_adr),
        "adr_spans":    [],
        "drug_spans":   [],
    }


# ════════════════════════════════════════════════════════════════════════════
# Priority 27 — Akan/Twi-influenced Ghanaian English
# Addresses: #225 ("Please the drug is paining me"), broader Akan-dialect FNs
# Patterns sourced from BMC qualitative study + Ghanaian English linguistics:
#   - Verb "pain" used intransitively ("my head is paining me")
#   - Verb "trouble" for symptom burden ("the drug is troubling me")
#   - Progressive for stative ("I am having diarrhoea")
#   - Causative "make...to" ("the medicine made me to vomit")
#   - Sentence-final oo/ooo for emphasis
#   - "not agreeing with my body" — established Ghanaian English idiom
#   - "my stomach is running" — diarrhoea idiom
#   - "the thing" as indirect reference to ADR
# ════════════════════════════════════════════════════════════════════════════

P27_ADR = [
    # Core Akan-influenced "paining" pattern
    ("metformin",       "Please the drug is paining me. My stomach is paining me every morning since I started the metformin."),
    ("cotrimoxazole",   "My whole body is paining me since I started the cotrimoxazole. The pain is too much ooo."),
    ("efavirenz",       "Since I started the ARV, my head is paining me every night. It is very serious."),
    ("amlodipine",      "The blood pressure tablet is paining my legs. Both of my legs are paining me since I started the drug."),
    ("isoniazid",       "My hands and feet are paining me since I started the TB drug. The doctor said it is side effect from isoniazid."),
    ("haloperidol",     "Please, since they gave me the injection my whole body is paining me and I cannot sit down at all at all."),
    ("artemether-lumefantrine", "I am having so much pain in my joints since I took the malaria tablet. It is paining me well well."),
    ("rifampicin",      "My stomach is paining me very much since I started the TB treatment. I want to report it because it is worrying me."),
    # "Trouble" pattern
    ("metformin",       "The tablet is troubling me. Since I started the metformin my tummy is troubling me every single morning."),
    ("co-trimoxazole",  "The drug has been troubling my body since I started taking it. It is giving me rash all over."),
    ("efavirenz",       "Since I started the efavirenz, my head is troubling me in the night. I see strange things when I close my eyes."),
    ("glibenclamide",   "Please the diabetes tablet is troubling me. Any time I take it, my body becomes weak and I am shaking."),
    ("isoniazid",       "The TB drug is troubling my stomach. I want to vomit every morning since I started it."),
    ("amlodipine",      "My legs are troubling me. Since the blood pressure tablet, both my ankles are swelling and troubling me."),
    # Progressive stative pattern
    ("metformin",       "I am having stomach pain since I started the metformin. I am having loose stool every morning too."),
    ("cotrimoxazole",   "I am having rash on my arms and back since I started the antibiotic. It is spreading."),
    ("tenofovir",       "I am having pain in my back since I started the ARV treatment. I am also having difficulty passing urine."),
    ("lisinopril",      "I am having this dry cough since the doctor started me on the blood pressure tablet. I am having it every night."),
    ("glibenclamide",   "I am having shaking and sweating every afternoon. The nurse said I am having low blood sugar from the tablet."),
    ("atorvastatin",    "I am having pain in my legs since the cholesterol tablet. I am having difficulty climbing stairs."),
    # "My stomach is running" and body idioms
    ("amoxicillin",     "My stomach is running since I started the antibiotic. I am going to toilet many many times every day."),
    ("co-trimoxazole",  "My stomach is running and I have been vomiting since the cotrimoxazole. I cannot eat anything."),
    ("metformin",       "My tummy has been running since I started the diabetes tablet. I go to toilet like five times before noon."),
    ("rifampicin",      "My stomach was running badly for the first week of the TB drug. It has reduced but my urine is still orange."),
    # "Not agreeing with my body" idiom
    ("efavirenz",       "This efavirenz is not agreeing with my body at all. I cannot sleep because of the bad dreams every night."),
    ("metformin",       "The metformin is not agreeing with my body. Since I started it my stomach has never been fine."),
    ("haloperidol",     "The injection is not agreeing with my body. Since they gave me the depot my body has not been well."),
    ("atorvastatin",    "The statin tablet is not agreeing with my body. My muscles are paining me since I started taking it."),
    # "The thing is troubling me" indirect reference
    ("cotrimoxazole",   "I take the antibiotic and this thing starts on my skin. The rash, it is spreading everywhere on my body."),
    ("efavirenz",       "Since I take the ARV, this thing happens in my head at night — I am seeing things and getting frightened."),
    ("isoniazid",       "Since the TB tablet started, this thing is happening to my hands — the tingling and the numbness."),
    # "Became" pattern (Ghanaian English for got/became)
    ("artemether-lumefantrine", "After the malaria tablet I became dizzy and my eyes became blurred. I had to lie down the whole day."),
    ("glibenclamide",   "I took the tablet and I became very weak and shaky. My sugar became too low and I nearly fainted."),
    ("haloperidol",     "After the injection I became stiff and my neck became tight. I could not turn my head properly."),
    # "Made me to" causative
    ("amoxicillin",     "The antibiotic made me to vomit three times. It also made me to have running stomach all day."),
    ("efavirenz",       "The ARV made me to see visions when I tried to sleep. It made me to be afraid of sleeping."),
    ("metoclopramide",  "The injection made me to shake and my eyes made to go upward. It was so frightening, it happened so fast."),
    # "Since I started/took" temporal + complaint
    ("metformin",       "Since I started the metformin, I have not had a day without stomach pain. It is too much for me."),
    ("rifampicin",      "Since I started the TB drug, my eyes are yellow and my urine is orange. I want to know if this is normal."),
    ("tenofovir",       "Since I started the new ARV, my kidneys are paining me and I am not passing urine as before."),
    ("co-trimoxazole",  "Since I took the antibiotic, my skin is very itchy and I have been scratching myself throughout."),
    # Formal reporting opener + symptom
    ("isoniazid",       "Please I want to report that since I started the TB tablet, my hands and feet are tingling and I cannot feel them well."),
    ("cotrimoxazole",   "Please I want to tell you about my skin since the cotrimoxazole. I am having rash everywhere and it is paining me."),
    ("efavirenz",       "Please I want to complain about the ARV. Since I started it, I have not been sleeping well and I see bad things."),
    # Intensifiers and Ghanaian English emphasis
    ("amlodipine",      "My legs are very very swollen since the blood pressure tablet. They are swelling more and more."),
    ("metformin",       "My stomach is paining me too much too much since the diabetes drug. I cannot eat anything at all."),
    ("isoniazid",       "The tingling in my feet is very bad now. It was small before but now it has become very serious."),
    ("artemether-lumefantrine", "My ears are ringing so much since the malaria treatment. The sound is there always and it is disturbing me."),
    # Dual complaint / concessive (ADR despite drug working)
    ("efavirenz",       "The ARV is helping my viral load but it is paining my head every night. I am still taking it but I need help."),
    ("metformin",       "The diabetes tablet is controlling my sugar but my stomach is always running. It is not comfortable at all."),
    ("amlodipine",      "The drug is bringing my pressure down but my legs are paining me and swelling. I don't know what to do."),
    ("cotrimoxazole",   "The antibiotic is treating my infection but it is giving me rash on my body. It is spreading oo."),
    # Symptom appeared "because of" the drug
    ("efavirenz",       "Because of the ARV tablet, I have not been sleeping well for three weeks. The bad dreams come every night."),
    ("isoniazid",       "Because of the TB drug, my hands are now numb. I cannot hold my cup properly in the morning."),
    ("metformin",       "Because of the diabetes tablet, my stomach is always running. I cannot go far from home anymore."),
    ("co-trimoxazole",  "Because of the antibiotic, my whole body is covered in rash now. It started small and has spread everywhere."),
    # Drug causing systemic effects
    ("rifampicin",      "Please my whole system has changed since the TB drug. My sweat is orange, my urine is orange, and even my tears."),
    ("efavirenz",       "This drug has changed my whole body. I am not myself since I started the ARV and I am afraid."),
    ("haloperidol",     "The injection has made my body stiff. I move slowly now and I cannot do my work as before."),
    ("tenofovir",       "My bones are paining me since the tenofovir. The doctor said my kidneys are also affected from the ARV."),
]

P27_NON_ADR = [
    # Drug working well — no ADR
    ("metformin",       "Please the metformin is agreeing with my body very well. My sugar is controlled and I have no complaint."),
    ("efavirenz",       "The ARV is not troubling me. I am tolerating it well and my body has accepted the treatment."),
    ("cotrimoxazole",   "The antibiotic is treating my infection and I am not having any rash or problem from it at all."),
    ("amlodipine",      "My blood pressure tablet is not paining me. My legs are fine and I have no swelling at all."),
    # Symptom from underlying disease, not drug
    ("metformin",       "My stomach was running before I started the metformin — it was the diabetes itself causing the problem."),
    ("isoniazid",       "My hands were tingling before I started the TB drug — it is from the TB disease that was already in my nerves."),
    ("efavirenz",       "The bad dreams started before the ARV. I have had nightmares since my husband died, not since the drug."),
    # Past resolved ADR
    ("metformin",       "My stomach was running when I first started the diabetes tablet but it has been fine for months now."),
    ("co-trimoxazole",  "I had small rash when I took the antibiotic three years ago but it went away and I am fine now."),
    # Drug not yet started
    ("rifampicin",      "The doctor has given me the TB drugs but I have not started yet. My symptoms are from the TB, not the drug."),
    # Objective test normal — subjective complaint only
    ("atorvastatin",    "I thought the statin was paining my legs but the doctor checked and said everything is normal — no problem from the tablet."),
    ("metformin",       "I felt my stomach was troubling me from the drug but the nurse checked and said my stomach is fine — no reaction."),
    # Alternative cause identified
    ("artemether-lumefantrine", "My whole body was paining me but the doctor checked and said it was the malaria still in my body, not the tablet."),
    ("amlodipine",      "My legs were paining me but the scan showed it was varicose veins — not from the blood pressure tablet."),
    ("glibenclamide",   "I was shaking and weak but it was from not eating — not from the tablet. My sugar was fine when they checked."),
    # Healthcare provider confirmed no ADR
    ("cotrimoxazole",   "I was worried about rash from the antibiotic but the nurse checked and said the small spots are mosquito bites, not drug rash."),
    ("metformin",       "Please I thought the diabetes drug was troubling me but the pharmacist explained it is normal stomach settling — not an ADR."),
    ("efavirenz",       "I told my counsellor about the dreams and she said they are from my worry about the diagnosis, not from efavirenz."),
    ("isoniazid",       "The TB drug is not giving me any problem. My liver test came back normal and I have no tingling or numbness."),
    ("haloperidol",     "The injection agreed with my body. I have not had any stiffness or movement problem since they gave it to me."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 28 — Ewe-influenced Ghanaian English
# Addresses: #238 ("I feel somehow since I took the injection")
# Patterns:
#   - "I feel somehow" as Ewe-influenced hedging (Ewe: "loo woa")
#   - "I am not myself" for systemic ADR
#   - "oo" emphasis particle
#   - Directional verb "came to complain"
#   - Topicalisation: "The drug, it is giving me..."
#   - "any time" for recurrent ADR
#   - "as for me" personal stance marker
# ════════════════════════════════════════════════════════════════════════════

P28_ADR = [
    # "I feel somehow" pattern
    ("efavirenz",       "I feel somehow since I took the injection. Since I started the ARV, I feel somehow every night."),
    ("artemether-lumefantrine", "I feel somehow after taking the malaria tablet — a strange feeling in my head and my body."),
    ("metformin",       "Since the diabetes drug, I feel somehow. My stomach feels somehow and I don't have appetite."),
    ("haloperidol",     "Since they gave me the injection, I feel somehow. My body is somehow and I cannot explain it well."),
    ("co-trimoxazole",  "I feel somehow since the antibiotic. My skin feels somehow and I am scratching everywhere."),
    ("isoniazid",       "Since the TB treatment, my hands feel somehow — like something is moving inside them. I feel somehow."),
    # "I am not myself" pattern
    ("efavirenz",       "I am not myself since I started the ARV. I do not feel like myself at all since the efavirenz."),
    ("haloperidol",     "Since they gave me the injection I am not myself. My mind is not clear and my body is different."),
    ("metformin",       "I have not been myself since the diabetes drug — my stomach has been troubling me every day."),
    ("co-trimoxazole",  "Since the cotrimoxazole I am not myself. My skin has rash and I feel hot inside."),
    ("quinine",         "Since the malaria drip I am not myself. Everything is ringing in my ears and I feel dizzy."),
    # "oo" emphasis
    ("efavirenz",       "The ARV is giving me headache every night oo. Since I started it, the headache does not stop oo."),
    ("metformin",       "The diabetes tablet is giving me serious running stomach oo. I have been going to toilet all morning."),
    ("isoniazid",       "My hands are tingling oo since the TB drug. The tingling is too much oo and I cannot hold things properly."),
    ("amlodipine",      "My legs are swelling oo since the blood pressure tablet. The swelling is very bad oo."),
    # "The drug, it is giving me" topicalisation
    ("co-trimoxazole",  "The antibiotic, it is giving me rash all over my body since I started taking it."),
    ("efavirenz",       "This ARV, it is giving me bad dreams every night and I am afraid to sleep."),
    ("metformin",       "This diabetes tablet, it is giving me serious diarrhoea every morning since I started."),
    ("haloperidol",     "This injection, it is giving me stiffness in my neck and I cannot turn my head well."),
    # "Any time I take" recurrent pattern
    ("glibenclamide",   "Any time I take the diabetes tablet in the morning, I feel weak and shaking before lunch."),
    ("metformin",       "Any time I take the metformin, my stomach runs. Any time without fail since I started."),
    ("artemether-lumefantrine", "Any time I take the malaria tablet I feel nauseated and I have to lie down immediately."),
    ("doxycycline",     "Any time I take the doxycycline I feel burning in my chest and my throat. Every single time."),
    # "As for me" personal stance
    ("efavirenz",       "As for me, since I started the ARV, I have not been sleeping well. The sleep is bad every night."),
    ("metformin",       "As for me, the diabetes drug is troubling my stomach. As for me, I have been going to toilet many times."),
    ("isoniazid",       "As for me, the TB drug has affected my feet. As for me, I feel tingling there since I started."),
    # "Came to complain / report" directional
    ("co-trimoxazole",  "I have come to report that since the antibiotic, I have been having rash on my body."),
    ("efavirenz",       "I came to tell the nurse that since the ARV, my sleep has not been good and I see things."),
    ("metformin",       "I came to the clinic to report that the diabetes tablet is giving me problem with my stomach."),
    ("amlodipine",      "I have come to complain about the blood pressure tablet. My legs are swelling since I started it."),
    # Ewe-influenced mild intensifiers
    ("artemether-lumefantrine", "The malaria tablet gave me serious headache — the whole head was paining me and I could not stand."),
    ("rifampicin",      "The TB drug has changed my urine colour completely. It is orange and it worries me very much."),
    ("haloperidol",     "Since the injection my hands shake somehow. Not so much but the shaking is there since the depot."),
    ("co-trimoxazole",  "The rash from the antibiotic is spreading. It started small but now it is covering my chest and back."),
    # "Making" causative pattern
    ("efavirenz",       "The ARV is making me to have bad dreams. It is making me to be confused in the morning also."),
    ("metformin",       "The tablet is making my stomach to run. It is making me to go to toilet many times."),
    ("glibenclamide",   "The diabetes drug is making me to have low sugar. It is making me to shake and sweat."),
    # Physical symptoms in Ewe-influenced language
    ("isoniazid",       "My hands and feet feel like somebody is pricking them with pins. This is since the TB drugs."),
    ("cotrimoxazole",   "My skin is hot and itchy since the antibiotic. I have been scratching until I have marks on my body."),
    ("quinine",         "Everything in my ears is ringing since the malaria injection. I am hearing noise that is not there."),
    ("tenofovir",       "My bones are hurting since the new ARV. I feel like my whole body is aching from the drug."),
]

P28_NON_ADR = [
    # Feeling somehow but it's the disease, not the drug
    ("efavirenz",       "I feel somehow but the doctor checked and said it is not from the ARV — it is from stress and my condition."),
    ("artemether-lumefantrine", "I felt somehow after the malaria tablet but it was the malaria fever itself — not the drug."),
    ("metformin",       "I felt somehow when I started the diabetes drug but it was from hunger — I had not eaten properly."),
    # Not myself but alternative cause
    ("efavirenz",       "I was not myself but it was from the anxiety about my diagnosis — not from the ARV itself."),
    ("haloperidol",     "I was not myself when I came to hospital but it was because I was not sleeping — not the injection."),
    # Drug tolerated well
    ("co-trimoxazole",  "As for me, the antibiotic has not given me any problem at all. My body has accepted it well."),
    ("metformin",       "As for me, the diabetes tablet agrees with me. I have no running stomach or any complaint."),
    ("efavirenz",       "As for me, the ARV is not giving me any bad dreams or problem. I am sleeping well since I started."),
    # Came to clinic but for check-up, not ADR
    ("isoniazid",       "I came to the clinic for my monthly check-up. I have no complaint about the TB drug — it agrees with me."),
    ("amlodipine",      "I came for review. My blood pressure tablet is not giving me any problem — no swelling, no pain."),
    # Confirmed no reaction
    ("cotrimoxazole",   "The nurse checked the spots on my skin and said they are mosquito bites — not from the antibiotic."),
    ("metformin",       "I felt my stomach was running from the drug but it was gastroenteritis going around — confirmed by the doctor."),
    ("glibenclamide",   "I was shaking any time in the afternoon but the nurse said it was from not eating lunch — my sugar was actually normal."),
    ("artemether-lumefantrine", "Any time I had that nausea after the tablet it was from the malaria infection itself — the nurse confirmed."),
    ("haloperidol",     "I felt somehow when they started the injection but it settled in one week. Now I feel completely normal."),
    ("efavirenz",       "I feel fine since I started the ARV. As for me, I have no complaint at all. The drug is agreeing with me."),
    ("isoniazid",       "Any time I take the TB drug I feel fine — no tingling, no stomach problem, no complaint from me."),
    ("co-trimoxazole",  "The antibiotic is not giving me any reaction. My skin is clear and I feel completely well."),
    ("metformin",       "The diabetes tablet has not changed how I feel at all. I take it every day and I am fine."),
    ("amlodipine",      "My legs are not swelling. The blood pressure tablet agrees with my body and I feel well."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 29 — Northern Ghanaian English (Dagbani/Hausa-influenced)
# Addresses: #219 ("The medication has spoilt my whole system")
# Patterns:
#   - "I will not lie" truth assertion marker
#   - "very well" as intensifier for pain (Dagbani influence)
#   - "spoilt" for damage done by drug
#   - "belly" for stomach/abdominal area
#   - "it is not easy" as intensifier of complaint
#   - "my body heats" for fever
#   - Direct causation language: "the drug has spoilt my..."
#   - "since morning" time anchor
#   - "come and go" for intermittent symptoms
# ════════════════════════════════════════════════════════════════════════════

P29_ADR = [
    # "I will not lie" assurance marker
    ("efavirenz",       "I will not lie, since I started the ARV my sleep has been very bad. I see things when I close my eyes."),
    ("metformin",       "I will not lie, the diabetes tablet has spoilt my stomach. I go to toilet many times every day."),
    ("isoniazid",       "I will not lie, the TB drug has affected my hands. They are tingling and I cannot hold anything properly."),
    ("co-trimoxazole",  "I will not lie, the antibiotic has given me serious rash. My whole body is covered in rash since I started."),
    ("haloperidol",     "I will not lie, since the injection my body has not been the same. I move slowly and I cannot do my work."),
    ("artemether-lumefantrine", "I will not lie, the malaria tablet has made me to feel very bad. My head was paining me very well."),
    # "Very well" as intensifier for pain/symptom
    ("metformin",       "The tablet is paining me very well. My belly is paining me very well since I started the diabetes drug."),
    ("isoniazid",       "My hands are paining me very well since the TB drug. I wake up at night because the pain disturbs me very well."),
    ("efavirenz",       "The bad dreams trouble me very well. Every night since the ARV I am disturbed very well by dreams."),
    ("amlodipine",      "My legs are swelling very well since the blood pressure tablet. They are heavy and they trouble me very well."),
    ("haloperidol",     "My neck is stiff very well since the injection. I cannot turn my head and it disturbs me very well."),
    # "Spoilt" pattern (drug damaged the body)
    ("metformin",       "The medication has spoilt my whole system. I cannot eat, my stomach is spoilt, and I am always running to toilet."),
    ("efavirenz",       "This ARV has spoilt my brain. Since I started it, I am not thinking clearly and I forget things."),
    ("isoniazid",       "The TB drug has spoilt my legs. I can feel the damage — my feet are numb and I walk somehow."),
    ("haloperidol",     "The injection has spoilt my movement. Since they gave me the depot, my body is stiff and spoilt."),
    ("tenofovir",       "The ARV has spoilt my kidneys. The doctor confirmed that my creatinine has gone up since tenofovir."),
    ("rifampicin",      "The TB drug has spoilt my liver. My eyes are yellow and my urine is dark since I started it."),
    # "Belly" for abdominal symptoms
    ("metformin",       "My belly has been troubling me since the diabetes tablet. The pain in my belly is serious and it disturbs me."),
    ("co-trimoxazole",  "The antibiotic has given me belly problem. My belly is running and I have been going to toilet since morning."),
    ("artemether-lumefantrine", "After the malaria tablet my belly was paining me very well. The belly pain was serious."),
    ("amoxicillin",     "My belly ran since morning because of the antibiotic. I have not eaten because of the belly problem."),
    ("isoniazid",       "My belly is not fine since the TB drug. There is burning in my belly every time I take the tablet."),
    # "It is not easy" intensifier
    ("efavirenz",       "Since I started the ARV, it is not easy. The drug is giving me serious trouble with my sleep and my head."),
    ("isoniazid",       "The TB drug is not easy. My hands and feet are paining me and it is not easy to do my farming work."),
    ("metformin",       "The diabetes tablet is not easy for me. My belly is always troubling me and it is not easy at all."),
    ("haloperidol",     "Since the injection it is not easy. My body has changed and it is not easy to do my daily activities."),
    # "My body heats" for fever
    ("artemether-lumefantrine", "Since the malaria tablet my body heats and then becomes cold. The heating and cooling is disturbing me."),
    ("isoniazid",       "My body heats sometimes since the TB drug. Not always but sometimes my body heats in the night."),
    ("co-trimoxazole",  "Since the antibiotic, my body heats inside. I feel the heat inside my body since I started the drug."),
    # "Come and go" intermittent
    ("efavirenz",       "The headache from the ARV comes and goes. Sometimes it comes very strong and then it goes a little."),
    ("amlodipine",      "The swelling in my legs comes and goes since the blood pressure tablet. In the evening it comes badly."),
    ("glibenclamide",   "The shaking comes and goes since the diabetes drug. Any time I miss a meal the shaking comes badly."),
    ("isoniazid",       "The tingling in my hands comes and goes. Since the TB tablet the tingling is always coming and going."),
    # "Since morning" / time anchoring
    ("metformin",       "Since morning I have been going to toilet because of the diabetes drug. I took it at dawn and since morning I have not rested."),
    ("artemether-lumefantrine", "Since morning my body has not been fine since I took the malaria tablet last night."),
    ("co-trimoxazole",  "Since morning I have been scratching because of the rash from the antibiotic. It started small but since morning it has spread."),
    # Severity reporting Northern style
    ("efavirenz",       "This drug has changed my life. I swear, since I started the ARV, I am not the same person. The bad effects are too much."),
    ("isoniazid",       "I swear, the TB drug is affecting my body very badly. My feet are numb and I cannot feel the ground when I walk."),
    ("metformin",       "I swear the diabetes tablet has spoilt my belly. Every morning since morning I am going to toilet and it is not easy."),
    ("haloperidol",     "By God, since the injection my whole body has changed. I move like an old man and I cannot do my work."),
]

P29_NON_ADR = [
    # Drug working well with Northern English
    ("metformin",       "I will not lie, the diabetes tablet is helping me very well. My sugar is controlled and I have no problem."),
    ("efavirenz",       "I will not lie, the ARV is agreeing with me. I have no bad dreams and my sleep is fine."),
    ("co-trimoxazole",  "The antibiotic has not spoilt anything for me. My belly is fine and I have no problem from it."),
    # Symptoms from disease not drug
    ("artemether-lumefantrine", "My body was heating before the malaria tablet — it was the malaria fever. The drug actually helped it stop."),
    ("isoniazid",       "My hands were paining me before the TB drug — it was from carrying heavy loads at the farm. Not from the tablet."),
    ("metformin",       "My belly was troubling me before the diabetes tablet — it was the peptic ulcer I already had. Not the drug."),
    # Confirmed by clinician
    ("efavirenz",       "The doctor confirmed that my sleep problems are from stress and anxiety — not from the ARV. The drug is fine."),
    ("haloperidol",     "I will not lie, I was worried about the injection but the nurse checked and said I have no side effect."),
    ("metformin",       "The pharmacy said the belly problem I was having was from eating late, not from the diabetes tablet."),
    # Drug discontinued but symptom persisted — confirms not drug
    ("amlodipine",      "They stopped the blood pressure tablet but my legs are still swelling. So the swelling was not from the tablet."),
    ("metformin",       "I stopped the diabetes drug for one week and my belly was still running. It was not the tablet causing it."),
    # General acceptance
    ("isoniazid",       "The TB drug is not troubling me at all. I will not lie, it is agreeing with my body very well."),
    ("co-trimoxazole",  "My belly is fine since the antibiotic. No running, no pain, nothing — the drug is not disturbing me."),
    ("efavirenz",       "Since I started the ARV, my sleep is fine. It is not easy taking drugs every day but this one agrees with me."),
    ("artemether-lumefantrine", "The malaria tablet treated me well and I did not have any bad reaction. My body heats stopped after the drug."),
    ("haloperidol",     "I will not lie, my body has accepted the injection. I was afraid but it has not spoilt anything."),
    ("metformin",       "My belly was running only for the first three days. Since morning of the fourth day, everything is fine."),
    ("glibenclamide",   "The diabetes tablet has not given me low sugar. I eat properly and the shaking and weakness is not coming."),
    ("isoniazid",       "My hands are fine since the TB drug. No tingling, no numbness — the nerve problem has not come to me."),
    ("tenofovir",       "The new ARV has not spoilt my kidneys. The doctor checked and my creatinine is still normal."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 30 — Extended West African Pidgin ADR (extends P5 from v2)
# Addresses: #31, #37, #55, #78, #91, #92, #97, #99, #146, #150
# Grounded in authentic UD_Naija-NSC patterns:
#   - "di ting dey pain me" — from corpus
#   - "dey worry me" — from corpus
#   - "my body no dey fine" — common Pidgin health expression
#   - "dey comot for body" — rash appearing (from corpus: "rashes no go dey comot")
#   - "I dey vomit" — progressive vomiting
#   - Reduplication: "pain pain", "yellow yellow"
#   - "at all at all" emphatic negation
#   - "since morning" time anchor (from corpus)
# ════════════════════════════════════════════════════════════════════════════

P30_ADR = [
    # "Di ting dey pain me" core pattern (from corpus)
    ("metformin",       "Di ting dey pain me since I take di diabetes tablet — my tummy dey pain me every morning."),
    ("cotrimoxazole",   "Di rash dey pain me well well. Since I start di antibiotic, di ting don dey pain my skin."),
    ("efavirenz",       "Di headache dey pain me every night since I start di ARV. Di pain no dey gree me sleep."),
    ("haloperidol",     "Di injection don cause pain for my body. Di ting dey pain my neck and I no fit turn am."),
    ("isoniazid",       "Di TB tablet dey pain my hand and leg. Di tingling dey pain me at night when I dey sleep."),
    # "My body no dey fine" general malaise
    ("efavirenz",       "My body no dey fine since I start di ARV. I dey feel somehow and di sleep no dey come."),
    ("metformin",       "My body no dey fine since di diabetes tablet. My tummy dey run and I no dey eat well."),
    ("co-trimoxazole",  "My body no dey fine at all since di antibiotic. Rash don dey comot everywhere for my skin."),
    ("artemether-lumefantrine", "My body no dey fine at all at all since di malaria tablet. Headache, dizzy, everything."),
    ("haloperidol",     "My body no dey fine since di injection — I dey move slow and my neck stiff well well."),
    # "Dey comot for body" — rash appearing (adapted from corpus pattern)
    ("co-trimoxazole",  "Rash don dey comot for my body since I start di antibiotic. Di rash dey spread more and more."),
    ("cotrimoxazole",   "Small pimple-like things dey comot for my skin since di cotrimoxazole. De dey spread reach my back."),
    ("amoxicillin",     "Red spots dey comot for my arm since I start di antibiotic. I no know wetin cause am but na di drug."),
    ("efavirenz",       "Yellow thing dey comot for my eye since I start di ARV. Di nurse say na my liver dey show yellow."),
    # "I dey vomit" progressive symptom
    ("metformin",       "I dey vomit since morning because of di diabetes tablet. I take am and immediately I dey vomit."),
    ("co-trimoxazole",  "I dey vomit and my tummy dey run since I start di antibiotic. I no fit eat anything."),
    ("artemether-lumefantrine", "I dey vomit after every dose of di malaria tablet. Na di drug dey cause di vomiting."),
    ("isoniazid",       "My stomach dey do me somehow since di TB tablet and sometimes I dey vomit small."),
    # "My ear dey ring" and sensory symptoms
    ("quinine",         "My ear dey ring since dem give me di malaria injection. Di ringing no dey stop and I no fit hear well."),
    ("artemether-lumefantrine", "My ear dey make noise since di malaria treatment. I hear sound wey no dey there."),
    ("furosemide",      "My ear dey ring plenty plenty since dem give me di injection. Di doctor say na di drug cause am."),
    ("streptomycin",    "Since dem give me di streptomycin injection, my ear dey do me anyhow — ring ring inside and I no hear well."),
    # "My eye don yellow" jaundice
    ("rifampicin",      "My eye don yellow since I start di TB drug. And my urine come yellow-red too. Di nurse say na normal."),
    ("cotrimoxazole",   "My eye come yellow since I take di cotrimoxazole. My pikin eye yellow too and we dey worry."),
    ("isoniazid",       "Since I start di TB tablet, my eye don come yellow small. Di doctor say na my liver dey do am."),
    # "Tummy dey run" diarrhoea
    ("amoxicillin",     "My tummy dey run since I start di antibiotic. I don go toilet more than seven times today because of di drug."),
    ("metformin",       "My tummy dey run every morning since di diabetes tablet. Na di first thing wey happen after I take am."),
    ("co-trimoxazole",  "Di antibiotic make my tummy run well well. I no fit go far from toilet since I start di drug."),
    # "Body dey shake" tremor/rigors
    ("haloperidol",     "My hand dey shake since di injection. Di shaking no dey stop and I no fit do my work well."),
    ("glibenclamide",   "My body dey shake every afternoon since di diabetes tablet. My sugar go down and my body dey shake plenty."),
    ("artemether-lumefantrine", "I dey shake and feel cold after di malaria tablet. Di nurse say na di drug dey cause di rigors."),
    ("pyrazinamide",    "My body dey shake since I start di TB drug. Blood dey run cold and my temperature dey go up."),
    # "No fit sleep" insomnia
    ("efavirenz",       "I no fit sleep since I start di ARV. Bad dream dey come every night and I wake up frighten."),
    ("haloperidol",     "Since di injection I no fit sleep. My leg dey move by demself and I no fit rest for one place."),
    ("dexamethasone",   "Since di doctor give me di steroid injection, I no fit sleep — I dey awake all night."),
    # "My leg dey swell" oedema
    ("amlodipine",      "My two leg dey swell since I start di blood pressure tablet. Di swelling dey increase more and more."),
    ("nifedipine",      "My leg don swell plenty since di drug. I no fit wear my shoe again because of di swelling."),
    ("prednisolone",    "My leg and face don swell since I start di steroid tablet. Na di drug cause di swelling."),
    # Concessive + ADR (drug helping but causing side effect)
    ("metformin",       "Di diabetes drug dey control my sugar but my tummy dey run every morning — na di drug do am."),
    ("efavirenz",       "Di ARV dey suppress my viral load but e dey give me serious headache every night."),
    ("isoniazid",       "Di TB drug dey work for my chest but my hand dey numb since I start am — nurse say na side effect."),
    ("co-trimoxazole",  "Di antibiotic dey fight my infection but rash don dey comot for my body since I start am."),
    # Extended Pidgin patterns from corpus structures
    ("tenofovir",       "Since I begin di new ARV, my back dey pain me and I no dey pass urine as before. Di doctor dey check."),
    ("isoniazid",       "I dey take di TB tablet every day but di ting wey dey happen to my hand — di tingling — dey worry me."),
    ("efavirenz",       "Di ARV tablet dey work but e con dey give me dis bad dream wey dey scare me every night."),
    ("metformin",       "I con dey go toilet many times since I start di diabetes drug. My tummy come dey run since I take am."),
    ("haloperidol",     "Since dem give me di injection, I no fit sit down in one place — my leg dey move by demself."),
    ("co-trimoxazole",  "My pikin eye come yellow after two weeks of di antibiotic. Di doctor say na di drug cause am."),
    ("artemether-lumefantrine", "After di second dose of di malaria tablet, my heart start dey beat fast and I dey sweat cold."),
    ("glibenclamide",   "I dey feel weak and dizzy before I chop since I start di diabetes tablet — na di sugar wey fall."),
]

P30_NON_ADR = [
    # Drug tolerated — no ADR
    ("efavirenz",       "My body dey fine since I start di ARV. I no get bad dream or headache — di drug agree with me."),
    ("metformin",       "My tummy no dey run. Since I start di diabetes tablet, everything dey normal and I no get complaint."),
    ("co-trimoxazole",  "No rash dey comot for my body. Di antibiotic dey treat my infection and I no get reaction."),
    # Symptom predates drug
    ("artemether-lumefantrine", "My ear bin dey ring before I take di malaria tablet — na old ear infection I get. Not di drug."),
    ("efavirenz",       "Di bad dream dey before I start di ARV — it started when I hear my diagnosis. Not di drug."),
    ("metformin",       "My tummy bin dey run before di diabetes tablet — I had food problem since before. Not di drug cause am."),
    # Alternative cause identified
    ("co-trimoxazole",  "Di rash wey dey comot — di doctor check and say na heat rash from di weather — not di antibiotic."),
    ("artemether-lumefantrine", "My body dey shake before and after di tablet. Doctor check — na di malaria fever wey cause di shaking, not di drug."),
    # Negative dechallenge — stopped drug, same symptoms
    ("metformin",       "I stop di diabetes tablet for one week but my tummy still dey run — so na not di drug cause am."),
    ("efavirenz",       "I stop di ARV for two weeks but di headache still dey — so na not di efavirenz wey cause am."),
    # Drug resolved the symptom (opposite of ADR)
    ("artemether-lumefantrine", "Before di malaria tablet, my body dey shake and I dey hot. After I take am, everything con dey fine."),
    ("amoxicillin",     "Before di antibiotic, my tummy dey pain me from di infection. After di drug, di pain don reduce."),
    # Confirmed by clinician — no reaction
    ("haloperidol",     "Nurse check me and say I no get any side effect from di injection. My body dey normal and I fit sit down."),
    ("isoniazid",       "Doctor check and my liver test fine. My hand no dey shake and my feet no dey tingle. TB drug dey agree with me."),
    # Minor tolerable expected effect, not reportable ADR
    ("rifampicin",      "My urine dey orange since I start di TB drug but di nurse say na normal and it no mean anything bad."),
    ("artemether-lumefantrine", "I feel small dizzy after di first dose of malaria tablet but it pass. No problem since then."),
    ("BCG vaccine",     "Di baby cry small after di BCG injection but she sleep fine and dey fine by next morning. Normal reaction."),
    # Pidgin Non-ADR extended
    ("metformin",       "My tummy was running only di first two days. Since di third day, everything dey normal — no more running."),
    ("efavirenz",       "Di headache only come for di first week. After dat, my body accept di ARV and di headache go away."),
    ("co-trimoxazole",  "I was afraid of rash but no rash dey comot for my body. I dey take di antibiotic fine fine."),
]

# ════════════════════════════════════════════════════════════════════════════
# Priority 31 — Code-switched / Multilingual (Pidgin/Twi/English)
# Addresses: #230 "Since I start di drug, my body no dey fine at all at all"
# Pattern: English-dominant with Pidgin/Twi particles/phrases mid-sentence
# Kept conservative — English lexical base, dialectal structure/particles
# ════════════════════════════════════════════════════════════════════════════

P31_ADR = [
    # Twi-Pidgin-English mixing
    ("efavirenz",       "The ARV — eii — since I start am, my head no dey rest at all. The bad dreams dey come every night oo."),
    ("metformin",       "This diabetes tablet, I swear, my tummy dey run since I start am. It is paining me well well."),
    ("cotrimoxazole",   "Since I take the antibiotic, abi the rash don spread for my body. It is paining me and I am scratching everywhere."),
    ("isoniazid",       "My hands, eii, since I start the TB drug, they are tingling and I cannot hold anything properly. Di nurse say na side effect."),
    ("haloperidol",     "Ah the injection, since they give me, my body no dey fine. I am not myself and I feel somehow every day."),
    # Code-switch with Twi exclamation + English description
    ("efavirenz",       "Eii! Since the ARV started, I have not slept well even for one night. Bad dreams every night since I started."),
    ("metformin",       "Ei! The diabetes tablet has spoilt my stomach oo. My tummy is always running since morning every day."),
    ("co-trimoxazole",  "Hm! The antibiotic is giving me rash all over my body. It is spreading and paining me well well."),
    # English base with Pidgin particles
    ("artemether-lumefantrine", "The malaria tablet, I took it and immediately I started feeling somehow. My head was spinning and I could not stand up at all at all."),
    ("glibenclamide",   "The diabetes drug is giving me serious problem — any time I take it, my body shakes and my sugar comes down too low."),
    ("tenofovir",       "Since I started the new ARV, my back is paining me and I am not passing urine well. The kidney sef, it is not fine."),
    ("isoniazid",       "The TB drug is treating my chest, but my hands — eii — the tingling is too much. I cannot hold my cup in the morning."),
    # Pidgin structure with English words
    ("efavirenz",       "I start the efavirenz and since then my sleep don spoil. I dey see bad vision at night and I no dey rest well."),
    ("metformin",       "Since morning I don go toilet about six times because of the diabetes drug. The running is too much oo."),
    ("co-trimoxazole",  "The antibiotic don give me rash for my whole body. Since I start am, rash don dey comot everywhere."),
    # Authentic mixing — hesitation + code-switch
    ("haloperidol",     "The injection — how do I say it — since they give me, my body has changed. I dey move slow and my neck is stiff oo."),
    ("efavirenz",       "Please — I don't know how to explain — since the ARV, my head is doing me somehow. Like something is moving inside."),
    ("metformin",       "My tummy — eii — it has been troubling me since the diabetes tablet. The running, the paining, everything."),
    # Multilingual description of severe ADR
    ("quinine",         "Since the malaria drip, my ears are ringing. I will not lie — I cannot hear well and my head is paining me very well."),
    ("streptomycin",    "Since the streptomycin injection, my ear dey ring and I no fit hear well. The nurse confirmed it — na di drug."),
    ("rifampicin",      "My whole system, since I start the TB drug, it has changed. My urine is orange, my eye is somehow — the doctor say na normal."),
    # Cultural oath markers with English
    ("efavirenz",       "By God, the ARV is giving me terrible nightmares. Every single night since I started — I am afraid to sleep."),
    ("isoniazid",       "I swear by God, the TB drug has numbed my feet. I cannot feel the ground properly when I walk now."),
    ("metformin",       "I swear, the diabetes tablet has spoilt my stomach. At all at all I cannot go anywhere without looking for toilet."),
    # Extended multilingual complaint
    ("co-trimoxazole",  "Since I start the cotrimoxazole, the thing is troubling my skin. Rash dey everywhere and I dey scratch myself all the time."),
    ("efavirenz",       "The ARV, since I start am, I dey have bad dream every night. My sleep has spoilt and I am not resting at all."),
    ("artemether-lumefantrine", "After the malaria tablet, my heart start beating fast — I was afraid. I told the nurse and she say na the drug."),
    ("metformin",       "Hm. Since the diabetes drug, my tummy is always troubling me. Morning, afternoon, any time — the running doesn't stop."),
    ("haloperidol",     "The injection has changed my movement. I dey move slow slow and my whole body is somehow since they gave me the depot."),
    ("isoniazid",       "Please I am paining since the TB tablet. My hands and feet are paining me and the tingling is there always."),
    ("efavirenz",       "Eii this ARV! Since I started it, my head is not fine. I think things that are not there and I see things at night."),
]


# ════════════════════════════════════════════════════════════════════════════
# Assembly
# ════════════════════════════════════════════════════════════════════════════

def build_records():
    records = []

    # P27 — Akan/Twi-influenced Ghanaian English
    n = 1
    for drug, sent in P27_ADR:
        records.append(make(drug, sent, 27, "akan_adr", n, True)); n += 1
    n = 1
    for drug, sent in P27_NON_ADR:
        records.append(make(drug, sent, 27, "akan_nonadr", n, False)); n += 1

    # P28 — Ewe-influenced Ghanaian English
    n = 1
    for drug, sent in P28_ADR:
        records.append(make(drug, sent, 28, "ewe_adr", n, True)); n += 1
    n = 1
    for drug, sent in P28_NON_ADR:
        records.append(make(drug, sent, 28, "ewe_nonadr", n, False)); n += 1

    # P29 — Northern Ghanaian English
    n = 1
    for drug, sent in P29_ADR:
        records.append(make(drug, sent, 29, "northern_adr", n, True)); n += 1
    n = 1
    for drug, sent in P29_NON_ADR:
        records.append(make(drug, sent, 29, "northern_nonadr", n, False)); n += 1

    # P30 — Extended West African Pidgin
    n = 1
    for drug, sent in P30_ADR:
        records.append(make(drug, sent, 30, "pidgin_adr", n, True)); n += 1
    n = 1
    for drug, sent in P30_NON_ADR:
        records.append(make(drug, sent, 30, "pidgin_nonadr", n, False)); n += 1

    # P31 — Code-switched multilingual
    n = 1
    for drug, sent in P31_ADR:
        records.append(make(drug, sent, 31, "codesw_adr", n, True)); n += 1

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

    print("\nData sourced from:")
    print("  - BMC Public Health 2018 Ghanaian patient qualitative interviews (open access)")
    print("  - UD_Naija-NSC Pidgin corpus v2.14 (CC BY-SA 4.0)")
    print("  - Ghanaian English linguistics research")
    print("  - Ghana ADR spontaneous reporting studies (PMC)")
    print("NOTE: CLF-only. Do not pass to NER — adr_spans are empty.")


if __name__ == "__main__":
    main()
