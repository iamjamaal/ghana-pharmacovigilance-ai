# Batch Regression -- Phase 8 Hybrid

**Model:** Phase 8 CLF (cohort_study) + Phase 7 NER (cohort_study)  
**Threshold:** 0.55  
**Rule layer:** inference_engine_v2.py  
**Baseline:** Phase 7 Hybrid (40/95, 42.1%)  

```
======================================================================
  RESULTS -- Phase 8 Hybrid Batch Regression
======================================================================

  Batch  2  (1 failures re-tested)  0/1 now pass  (Phase 7 Hybrid: 0/1  delta=0)
    [x] # 20  truth=ADR       pred=Non-ADR   prob=0.464  Asymptomatic lab ADR — patient denies symptoms; lab 

  Batch  3  (3 failures re-tested)  3/3 now pass  (Phase 7 Hybrid: 0/3  delta=+3)
    [+] # 31  truth=ADR       pred=ADR       prob=0.997  Pidgin — aminoglycoside ototoxicity  <- P8 TARGET FIXED
    [+] # 37  truth=ADR       pred=ADR       prob=0.997  Pidgin — drug-herb interaction hypoglycaemia  <- P8 TARGET FIXED
    [+] # 55  truth=ADR       pred=ADR       prob=0.995  Pidgin — somnambulism from hypnotic  <- P8 TARGET FIXED

  Batch  4  (5 failures re-tested)  1/5 now pass  (Phase 7 Hybrid: 0/5  delta=+1)
    [x] # 77  truth=ADR       pred=Non-ADR   prob=0.051  Severity minimisation — English coping language
    [+] # 78  truth=ADR       pred=ADR       prob=0.721  Pidgin reduplication 'small small' + severity minimi  <- P8 TARGET FIXED
    [x] # 82  truth=Non-ADR   pred=ADR       prob=0.992  Drug not yet started — 'currently being started on'
    [x] # 84  truth=Non-ADR   pred=ADR       prob=0.996  Investigation exonerates drug — imaging finds altern
    [x] # 85  truth=Non-ADR   pred=ADR       prob=0.985  Microbiology exonerates drug — OI identified as caus

  Batch  5  (11 failures re-tested)  10/11 now pass  (Phase 7 Hybrid: 4/11  delta=+6)
    [+] # 91  truth=ADR       pred=ADR       prob=0.996  Pidgin illness idiom — generalised dysregulation  <- P8 TARGET FIXED
    [+] # 92  truth=ADR       pred=ADR       prob=0.993  Pidgin illness idiom — CNS confusion  <- P8 TARGET FIXED
    [+] # 95  truth=ADR       pred=ADR       prob=0.997  Pidgin — rigors/chills
    [x] # 96  truth=ADR       pred=Non-ADR   prob=0.243  Ghanaian English idiom — malaise
    [+] # 97  truth=ADR       pred=ADR       prob=0.973  Pidgin — diarrhoea  <- P8 TARGET FIXED
    [+] # 99  truth=ADR       pred=ADR       prob=0.991  Pidgin — drug fever with rigors  <- P8 TARGET FIXED
    [+] #101  truth=ADR       pred=ADR       prob=0.996  Pidgin concessive — 'dey help X but dey cause Y'
    [+] #112  truth=ADR       pred=ADR       prob=0.995  Spiritual framing before drug-symptom causal stateme
    [+] #117  truth=ADR       pred=ADR       prob=0.997  Formal structured ADR case report register  <- P8 TARGET FIXED
    [+] #118  truth=ADR       pred=ADR       prob=0.997  Regulatory PV report register  <- P8 TARGET FIXED
    [+] #119  truth=ADR       pred=ADR       prob=0.676  Pharmacoepidemiology signal detection language  <- P8 TARGET FIXED

  Batch  6  (11 failures re-tested)  8/11 now pass  (Phase 7 Hybrid: 3/11  delta=+5)
    [+] #122  truth=Non-ADR   pred=Non-ADR   prob=0.009  Non-adherence context — intermittent dosing  <- P8 TARGET FIXED
    [x] #124  truth=Non-ADR   pred=ADR       prob=0.986  Drug holiday — headache not ADR
    [+] #127  truth=ADR       pred=ADR       prob=0.990  Polypharmacy attribution uncertainty — uncertainty =  <- P8 TARGET FIXED
    [+] #130  truth=Non-ADR   pred=Non-ADR   prob=0.012  Pre-existing condition explicitly stated
    [+] #132  truth=ADR       pred=ADR       prob=0.995  ARCH FIX 1.8 — historical tolerance clause; negation  <- ARCH/RULE FIX
    [+] #142  truth=ADR       pred=ADR       prob=0.994  ARCH FIX 1.7 — negation pivot guard  <- ARCH/RULE FIX
    [x] #145  truth=Non-ADR   pred=ADR       prob=0.988  Contradictory signal — objective improvement overrid
    [+] #146  truth=ADR       pred=ADR       prob=0.997  Pidgin — urinary frequency  <- P8 TARGET FIXED
    [+] #147  truth=ADR       pred=ADR       prob=0.995  ARCH FIX 1.7 — negation pivot guard  <- ARCH/RULE FIX
    [x] #148  truth=Non-ADR   pred=ADR       prob=0.992  Nocebo — subjective worsening with objective normali
    [+] #150  truth=ADR       pred=ADR       prob=0.996  Third-party observation of jaundice in Pidgin  <- P8 TARGET FIXED

  Batch  7  (15 failures re-tested)  14/15 now pass  (Phase 7 Hybrid: 9/15  delta=+5)
    [+] #151  truth=ADR       pred=ADR       prob=0.996  Pidgin caregiver — paediatric post-injection seizure
    [+] #158  truth=ADR       pred=ADR       prob=0.835  Nursing DAR timestamped record
    [+] #160  truth=ADR       pred=ADR       prob=0.996  Ward round clinical shorthand
    [+] #161  truth=ADR       pred=ADR       prob=0.997  Discharge summary — explicit 'secondary to [drug]' c
    [+] #162  truth=ADR       pred=ADR       prob=0.989  Physician impression with differential diagnosis cau
    [+] #166  truth=ADR       pred=ADR       prob=0.994  CHPS field report format
    [+] #170  truth=ADR       pred=ADR       prob=0.993  Field surveillance during mass drug distribution
    [+] #171  truth=ADR       pred=ADR       prob=0.997  ICSR structured ADR report  <- P8 TARGET FIXED
    [+] #172  truth=Non-ADR   pred=Non-ADR   prob=0.150  ARCH FIX 1.6 — PSUR aggregate document; INTOLERANCE   <- ARCH/RULE FIX
    [+] #174  truth=ADR       pred=ADR       prob=0.984  Market withdrawal — regulatory enforcement language  <- P8 TARGET FIXED
    [x] #175  truth=ADR       pred=Non-ADR   prob=0.150  Hedged signal assessment memo
    [+] #176  truth=ADR       pred=ADR       prob=0.995  Patient FDA complaint — 'spoiling' as harm idiom
    [+] #177  truth=ADR       pred=ADR       prob=0.996  Compensation framing — vague 'terrible' + legal inte
    [+] #178  truth=ADR       pred=ADR       prob=0.996  Physician-dismissal framing — near threshold
    [+] #179  truth=ADR       pred=ADR       prob=0.997  Pidgin voice-note — anticholinergic dry mouth / eye 

  Batch  8  (15 failures re-tested)  15/15 now pass  (Phase 7 Hybrid: 6/15  delta=+9)
    [+] #182  truth=Non-ADR   pred=Non-ADR   prob=0.008  P01-B temporal exonerator 'before'  <- P8 TARGET FIXED
    [+] #183  truth=ADR       pred=ADR       prob=0.994  P02-A causal attribution below threshold
    [+] #186  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.5 — third-party voice; narrator not affected  <- ARCH/RULE FIX
    [+] #188  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'has resolved' resolution language  <- ARCH/RULE FIX
    [+] #190  truth=Non-ADR   pred=Non-ADR   prob=0.011  P05-B remote past tense ignored  <- P8 TARGET FIXED
    [+] #192  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'felt fine' resolution  <- ARCH/RULE FIX
    [+] #194  truth=Non-ADR   pred=Non-ADR   prob=0.195  P07-B Pidgin negation 'no dey' ignored
    [+] #196  truth=Non-ADR   pred=Non-ADR   prob=0.011  P08-B self-attribution to alternative cause  <- P8 TARGET FIXED
    [+] #197  truth=ADR       pred=ADR       prob=1.000  ARCH FIX 1.1 — positive dechallenge  <- ARCH/RULE FIX
    [+] #198  truth=Non-ADR   pred=Non-ADR   prob=0.006  P09-B negative dechallenge  <- P8 TARGET FIXED
    [+] #200  truth=Non-ADR   pred=Non-ADR   prob=0.009  P10-B negative rechallenge  <- P8 TARGET FIXED
    [+] #204  truth=Non-ADR   pred=Non-ADR   prob=0.011  P12-B lab sub-threshold — numeric multiplier not par
    [+] #206  truth=ADR       pred=ADR       prob=0.997  P13-B Pidgin temporal 'After I take am'
    [+] #209  truth=ADR       pred=ADR       prob=0.984  P15-A patient certainty + 'nothing else has changed'
    [+] #210  truth=ADR       pred=ADR       prob=0.886  P15-B patient hedging fully suppresses ADR

  Batch  9  (13 failures re-tested)  13/13 now pass  (Phase 7 Hybrid: 5/13  delta=+8)
    [+] #212  truth=Non-ADR   pred=Non-ADR   prob=0.009  Elderly baseline — 'No new problems' not extracted
    [+] #213  truth=ADR       pred=ADR       prob=0.997  Low-literacy telegraphic style
    [+] #219  truth=ADR       pred=ADR       prob=0.988  Northern Ghanaian English — akathisia idiom 'body is  <- P8 TARGET FIXED
    [+] #221  truth=ADR       pred=ADR       prob=0.995  Ghanaian male reproductive idiom — erectile dysfunct
    [+] #223  truth=ADR       pred=ADR       prob=0.996  Adolescent polydipsia — self-rationalises away ADR
    [+] #225  truth=ADR       pred=ADR       prob=0.997  Akan-influenced English — 'doing me anyhow', 'flesh   <- P8 TARGET FIXED
    [+] #229  truth=ADR       pred=ADR       prob=0.996  Child voice ADR description
    [+] #230  truth=ADR       pred=ADR       prob=0.997  Multilingual Twi + Pidgin + English  <- P8 TARGET FIXED
    [+] #231  truth=ADR       pred=ADR       prob=0.997  Market trader minimisation — dizziness managed
    [+] #232  truth=ADR       pred=ADR       prob=0.995  Stigma concealment — 'hearing voices since starting 
    [+] #234  truth=Non-ADR   pred=Non-ADR   prob=0.011  Routine iron supplementation in pregnancy — no ADR
    [+] #235  truth=ADR       pred=ADR       prob=0.990  ARCH FIX 1.3 — 'appetite gone' = anorexia ADR, not r  <- ARCH/RULE FIX
    [+] #238  truth=ADR       pred=ADR       prob=0.997  Ewe-influenced English — vertigo and ocular pain idi  <- P8 TARGET FIXED

  Batch 10  (11 failures re-tested)  11/11 now pass  (Phase 7 Hybrid: 6/11  delta=+5)
    [+] #244  truth=Non-ADR   pred=Non-ADR   prob=0.000  Uneventful MgSO4 monitoring — 'reflexes' keyword tri
    [+] #248  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'passed quickly. midwife said expected'  <- ARCH/RULE FIX
    [+] #249  truth=ADR       pred=ADR       prob=0.968  ARCH FIX 1.2 — fetal abnormality; 'has not' triggers  <- ARCH/RULE FIX
    [+] #251  truth=ADR       pred=ADR       prob=0.981  Neonatal jaundice — indirect maternal drug framing
    [+] #253  truth=ADR       pred=ADR       prob=0.995  Midwife intrapartum clinical shorthand
    [+] #254  truth=ADR       pred=ADR       prob=0.995  TBA report — traditional uterotonic with haemorrhage
    [+] #255  truth=ADR       pred=ADR       prob=0.995  Postpartum depersonalisation ADR
    [+] #256  truth=ADR       pred=ADR       prob=0.997  Pidgin obstetric — uterine pain
    [+] #262  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — clinician reassurance + 'passed within ten  <- ARCH/RULE FIX
    [+] #266  truth=ADR       pred=ADR       prob=0.995  Postpartum psychosis — 'thought baby had been replac
    [+] #269  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'nausea...but it settled'  <- ARCH/RULE FIX

  Batch 11  (10 failures re-tested)  9/10 now pass  (Phase 7 Hybrid: 7/10  delta=+2)
    [+] #272  truth=Non-ADR   pred=Non-ADR   prob=0.000  Post-vaccination minor unsettledness; 'fed well' + '  <- ARCH/RULE FIX
    [+] #276  truth=Non-ADR   pred=Non-ADR   prob=0.031  Coincidental URTI — family cluster predates vaccinat  <- P8 TARGET FIXED
    [+] #278  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'went away on its own'  <- ARCH/RULE FIX
    [+] #280  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'nothing serious'  <- ARCH/RULE FIX
    [+] #285  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'no vomiting, no diarrhoea...' negation/re  <- ARCH/RULE FIX
    [+] #290  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'fine the next morning'  <- ARCH/RULE FIX
    [+] #291  truth=ADR       pred=ADR       prob=0.993  Post-vaccination vasovagal syncope — 'collapsed', 'w  <- P8 TARGET FIXED
    [+] #292  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'took paracetamol and was fine by the seco  <- ARCH/RULE FIX
    [x] #293  truth=Non-ADR   pred=ADR       prob=0.982  Coincidental measles post-vaccination — 'coincidence
    [+] #296  truth=Non-ADR   pred=Non-ADR   prob=0.000  FIX 1.4 — 'felt fine' + trivial injection site pain

======================================================================
  OVERALL   84/95 pass  (88.4%)
  Phase 7 Hybrid baseline:  40/95  (42.1%)
  Delta vs Phase 7 Hybrid:  +44 cases
  Phase 8 target clusters:  28/38 pass  (was 0/38)
======================================================================
```