# IMNCI rule expansion — held-out per-rule + per-case audit (2026-05-07)

**Source artifacts:**
- `eval/scripts/audit_rule_firings.py` — read-only re-runs the v2 `_matches_rule` logic against held-out test split (n=131), producing per-rule firing counts and per-case rule traces. Does NOT touch engine.py.
- `eval/analysis/2026-05-07/_rule_firing_audit.json` — full audit output.
- `eval/results/run_rules_v2_orig_loras_20260506_180705.json` — held-out re-eval with sprint 1 LoRAs + v2 rules.
- `eval/analysis/2026-05-06/red_failure_modes.md` — sprint 1 7-case missed-RED list.

## Schema-audit gap surfaced and routed around

The existing `engine_overrides` log captures only **escalating** matches — rules whose `minimum_triage_level` is higher than the current level when checked. A rule that matches but doesn't escalate (because the model already said RED) does NOT appear in overrides. The user's per-rule audit requires every match.

Rather than patch the engine and re-run inference, I wrote a read-only audit script (`audit_rule_firings.py`) that re-applies the v2 `_matches_rule` logic against each held-out test case in a single pass, with per-filter trace. No new artifacts had to be generated; the verbal_symptoms / tamil_question / age_group / duration_days fields were already in the test JSONLs from sprint 1.

## Per-rule firing summary (held-out test, n=131)

| Rule | min level | total fires | TP (gold matches) | FP (over-trigger) | chief-only-match miss | Notes |
|---|---|---|---|---|---|---|
| **ADULT-CARDIAC-001** | RED | 0 | 0 | 0 | 0 | chief regex never matched any case |
| **ADULT-ANAPHYLAXIS-001** | RED | **1** | **1** | 0 | 0 | derm_test_040 caught — single confirmed RED catch |
| **ADULT-HEAD-TRAUMA-001** | RED | 0 | 0 | 0 | 0 | chief regex never matched any case |
| **ADULT-RESPIRATORY-001** | RED | 0 | 0 | 0 | **2** | matched 2 chiefs but co-signals filtered both out |
| **ANIMAL-BITE-RESPIRATORY-001** | RED | 0 | 0 | 0 | 0 | chief regex never matched any case |
| **NEW-ONSET-JAUNDICE-001** | RED | 0 | 0 | 0 | 0 | no jaundice cases in held-out set |
| (existing) IMNCI-002 | RED | 0 | — | — | 2 | fever in chief but age != infant for those 2 |
| (existing) TN-002 | YELLOW | 0 | — | — | 2 | fever in chief but duration < 4d for those 2 |
| (existing) TN-004 | YELLOW | 2 | 1 | 1 | 1 | cough ≥ 21d — fired on 1 RED + 1 GREEN |

**Headline:** of the 6 new adult-emergency rules, only **ADULT-ANAPHYLAXIS-001 fired at all** on this held-out set. Five of the six new rules had zero fires. The +0.083 RED-recall delta from the rule expansion is entirely from one rule catching one case (derm_test_040).

## Per-case verdicts on the 7 sprint 1 missed REDs

For each, I list pre-engine model prediction + chief-complaint match against the closest semantic rule + reason for non-fire.

### 1. triage_test_039 — Acute MI presentation · age=adult · duration=1d

- **Verbal symptoms:** `இடது மார்பில் கடுமையான வலி மற்றும் இடது கையில் மரத்துப்போன உணர்வு` ("severe pain in left chest + numbness in left arm")
- **Closest rule:** ADULT-CARDIAC-001 (chest pain pattern)
- **Outcome:** chief regex did NOT match.
- **Why:** the rule's Tamil chief pattern is `மார்பு\s*(வலி|இறுக்க|நெருக்கம்)` — bare nominative `மார்பு` followed (optionally with whitespace) by a pain word. The case uses the **locative case** `மார்பில்` ("in the chest") with two intervening words (`கடுமையான வலி` = "severe pain"). The regex requires the pain word to follow `மார்பு` directly.
- **Verdict:** **caught_by_new_rule = False; missed_pattern_match_failure (Tamil locative `-இல்` + intervening adjective)**
- **Bucket: A** (regex too narrow for Tamil morphology — fixable with targeted regex tightening)

### 2. triage_test_040 — Dog bite + tingling at bite site · age=adult · duration=1d

- **Verbal symptoms:** `நாயினால் கடிக்கப்பட்டது, கடித்த இடத்தில் மதததப்பு மற்றும் துடிப்பு உணர்வு` ("bitten by dog + tingling and throbbing at bite site")
- **Closest rule:** ANIMAL-BITE-RESPIRATORY-001
- **Outcome:** chief regex did NOT match.
- **Why:** the rule's Tamil chief pattern is `நாய்\s*கடி` (bare `நாய்` + whitespace + `கடி`). The case uses **instrumental case** `நாயினால்` ("by dog") + `கடிக்கப்பட்டது` (passive bitten). The instrumental suffix `-இனால்` is fused to `நாய்`, breaking the regex.
- Additionally, the case's clinical RED flag is **paresthesia at bite site** (early rabies prodrome), not respiratory distress — so even if the chief regex matched, the required co-signal (respiratory distress) wouldn't fire.
- **Verdict:** **caught_by_new_rule = False; missed_pattern_match_failure (Tamil instrumental `-இனால்`) AND no rule covers paresthesia/early-rabies presentation**
- **Bucket: A and B (mixed)** — chief regex would benefit from broader Tamil case handling, but even with that, this case needs a rule we don't have ("animal bite + neurological prodrome → RED").

### 3. triage_test_042 — Lung inflammation, smoker · age=adult · duration=10d

- **Verbal symptoms:** `மூச்சுத்திணறல், கடுமையான இருமல் மற்றும் நுரையீரலில் வீக்கம்` ("dyspnea, severe cough, lung inflammation")
- **Closest rule:** ADULT-RESPIRATORY-001
- **Outcome:** chief regex did NOT match.
- **Why:** the rule's Tamil chief pattern is `மூச்சு\s*(விசில்|திண|இடைய)` — `மூச்சு` (breath) + whitespace + truncated form. The case uses the **compound** `மூச்சுத்திணறல்` (with sandhi `த்` connecting the morphemes — no space). The regex requires whitespace.
- **Verdict:** **missed_pattern_match_failure (Tamil compound with sandhi)**
- **Bucket: A** (regex too narrow for compound morphology — fixable)

### 4. triage_test_044 — Wheezing pneumonia · age=adult · duration=2d

- **Verbal symptoms:** `சுவாசப்பாதையில் வீசும் சத்தம் (wheezing) மற்றும் படுக்கும்போது மூச்சுத்திணறல்` ("wheezing in airway + dyspnea while lying down")
- **Closest rule:** ADULT-RESPIRATORY-001
- **Outcome:** chief regex DID match (`wheez` literal). Co-signals failed.
- **Why:** rule requires one of: `cannot speak in complete sentences`, `cyanosis`, `accessory muscle use`, `tripod position`, `பேச முடியவில்` (cannot speak), `நீல உதடு/நிற` (blue lips), `மார்பு உள்வாங்க` (chest indrawing). The case mentions "wheezing while lying down" but no severe-respiratory-distress marker.
- The patient is on antibiotics + albuterol inhaler for known pneumonia. Clinically borderline — defensible YELLOW (wheezing + nocturnal dyspnea on treatment) or RED (active wheezing with positional dyspnea suggests deterioration).
- **Verdict:** **caught_by_new_rule = False; chief matched, severe-distress co-signals reasonably absent**
- **Bucket: C** (borderline RED — rule's design correctly defers RED to clearly-severe presentations; the gold label is the borderline case the user warned about)

### 5. derm_test_040 — Anaphylaxis-pattern (chemical sensitivity) · age=adult · duration=1d

- **Verbal symptoms:** `வேதிப்பொருள் உணர்திறன், தோல் தடிப்பு, அரிப்பு, நாக்கு வீக்கம், அதிக தாகம்` ("chemical sensitivity, skin rash, itching, **tongue swelling**, extreme thirst")
- **Closest rule:** ADULT-ANAPHYLAXIS-001
- **Outcome:** **rule FIRED ✓** — chief regex matched on `நாக்கு வீக்க` (tongue swelling); duration_max=1 OK; no negative scoping triggered.
- **Verdict:** **caught_by_new_rule** ✓ (the only sprint 1 missed RED that the new rules caught)

### 6. maternal_test_027 — Post-fall syncope · age=adult · duration=1d

- **Verbal symptoms:** `விழுந்த பிறகு மயக்கம், குமட்டல் மற்றும் உட்காரும் போது வலி` ("after falling: dizziness, nausea, pain on sitting")
- **Closest rule:** ADULT-HEAD-TRAUMA-001
- **Outcome:** chief regex did NOT match.
- **Why:** rule requires `head injury` keywords or specifically `கீழே\s*விழ` ("fall down"). The chief uses bare `விழுந்த` ("having fallen") without the directional `கீழே` ("downward"). The narrative does have `கீழே விழுந்தேன்`, but chief regex matches against `verbal_symptoms` only.
- Even if the head-trauma chief regex were broadened, the patient's chief complaint is **post-fall transient syncope**, not head trauma — she fell on her buttocks, didn't lose consciousness, but felt momentarily dizzy on standing. Required co-signal `(LOC|AMS|persistent vomiting)` wouldn't match.
- This is a **post-fall orthostatic / vasovagal syncope** case — there is no syncope rule in the 6 new rules, and per user instruction we don't add a 7th.
- **Verdict:** **missed_no_rule_applies** (chief is "post-fall syncope" — no canonical pattern in the 6 rules)
- **Bucket: B** (rule layer ceiling; clinically real but not covered by any of our patterns)

### 7. maternal_test_028 — Chemo patient with rectal bleeding + mucus · age=adult · duration=7d

- **Verbal symptoms:** `மலம் கழிக்கும்போது இரத்தம் மற்றும் слизи (mucous) வெளியேறுதல்` ("blood and mucus during defecation")
- **Closest rule:** none of the 6 — closest existing v1 rule is IMNCI-002 (fever) which doesn't apply.
- **Outcome:** no new rule's chief regex matched.
- **Why:** No rule covers GI bleeding in chemo / immunocompromised patient. The chemo-related risk (febrile neutropenia, GI mucositis with active bleeding) is real but doesn't pattern-match any of the 6 designed rules. None of (cardiac, anaphylaxis, head trauma, severe respiratory, animal bite + respiratory, new-onset jaundice) applies.
- **Verdict:** **missed_no_rule_applies**
- **Bucket: B** (rule layer ceiling)

## Bucket distribution

| Bucket | Definition | Count | Cases |
|---|---|---|---|
| A — regex tightening fixable | Chief regex would match a tightened pattern | **3** | triage_test_039, triage_test_040, triage_test_042 |
| B — rule-layer ceiling | No canonical pattern; needs new rule (out of scope this sprint) or model-level fix | **2** | maternal_test_027, maternal_test_028 |
| C — borderline RED | Defensible YELLOW per the rule's own design; rule correctly defers | **1** | triage_test_044 |
| Caught by new rule | derm_test_040 — ADULT-ANAPHYLAXIS-001 fires | **1** | derm_test_040 |

**Bucket A is the dominant remaining-failure pattern (3 of 6 misses, 50%).** Per the user's spec ("if Bucket A is the dominant pattern: tighten rules before Task 6"), targeted regex tightening with new unit tests is in scope.

## Proposed targeted tightenings (not yet applied — awaiting approval)

The user's spec says: *"Do not tighten rules speculatively. Do not add new rules beyond the 6 already approved. The action from this analysis is documentation + (if Bucket A is the dominant pattern) targeted regex tightening with new unit tests."*

I am surfacing the proposed changes here for review. They have NOT been written into `imnci_rules_v2.json` yet.

### Proposed tightening 1 — ADULT-CARDIAC-001 chief pattern

Add Tamil case-inflected forms of `மார்பு` and allow intervening words:

```
old:  chest\s*(pain|press(ure|ing)?|tight(ness)?|discomf(ort)?)|cardiac|மார்பு\s*(வலி|இறுக்க|நெருக்கம்)
new:  chest\s*(pain|press(ure|ing)?|tight(ness)?|discomf(ort)?)|cardiac|மார்[பு][ிீு]?[ல்னை]*\s*[஀-௿\s]{0,30}?(வலி|இறுக்க|நெருக்கம்|பீடிப்)
```

Rationale: `மார்[பு][ிீு]?[ல்னை]*` matches `மார்பு / மார்பில் / மார்பின் / மார்பை / மார்புக்கு` (bare + locative + genitive + accusative + dative). The `[஀-௿\s]{0,30}?` allows up to 30 Tamil characters between the chest noun and the pain word (non-greedy). Catches "மார்பில் கடுமையான வலி".

New positive unit test: `triage_test_039`'s exact verbal_symptoms.
New negative unit test: a non-cardiac chest mention like "தோல் மார்பு" (chest skin) without pain.

### Proposed tightening 2 — ANIMAL-BITE-RESPIRATORY-001 chief pattern

Add Tamil case-inflected forms of `நாய்/பூனை/பாம்பு`:

```
old:  ...|நாய்\s*கடி|பூனை\s*கடி|பாம்பு\s*கடி
new:  ...|(நாய்|நாயின்|நாயினால்|நாயை)\s*[஀-௿\s]{0,15}?(கடி|கடிக்க|கடிக்கப்பட்)|(பூனை|பூனையின்|பூனையால்|பூனையை)\s*[஀-௿\s]{0,15}?(கடி|கடிக்க)|(பாம்பு|பாம்பின்|பாம்பால்|பாம்பை)\s*[஀-௿\s]{0,15}?(கடி|கடிக்க)
```

Rationale: handles instrumental `-இனால்`, genitive `-இன்`, accusative `-ஐ`, and the passive verb form `கடிக்கப்பட்டது` ("was bitten").

Caveat: triage_test_040's clinical RED flag is **paresthesia at bite site (early rabies)**, not respiratory distress. Even with this regex tightening, the required co-signal `(dyspn|airway swell|hypotens)` wouldn't match for this case. This tightening alone catches **0 of the 3** Bucket A cases for the rule's intended trigger; the case becomes a Bucket B miss after regex tightening because we have no rabies-prodrome rule. Documenting this honestly: tightening the regex is correct hygiene but doesn't move RED recall on this specific case.

### Proposed tightening 3 — ADULT-RESPIRATORY-001 chief pattern

Allow Tamil compound forms (no whitespace required between morphemes):

```
old:  wheez|stridor|severe\s*(dyspn|breathl)|respiratory\s*distress|cannot\s*breathe|gasping|மூச்சு\s*(விசில்|திண|இடைய)
new:  wheez|stridor|severe\s*(dyspn|breathl)|respiratory\s*distress|cannot\s*breathe|gasping|மூச்சு[த்ு]?\s*(விசில்|திண|இடைய|திணறல்|திணற)
```

Rationale: `[த்ு]?` allows the optional sandhi consonant `த்` and the alternate vowel before `திணறல்` (the full noun "dyspnea"). Catches `மூச்சுத்திணறல்` (compound) AND `மூச்சு திண` (separated).

New positive unit test: `triage_test_042`'s exact verbal_symptoms.
New negative unit test: a non-respiratory mention like "மூச்சு போய் வந்தது" (came back to my breath = relief, not distress).

### Expected delta from proposed tightenings

| Case | Bucket pre-tighten | Catches after tighten? | Notes |
|---|---|---|---|
| triage_test_039 | A | likely yes — chief regex now matches `மார்பில் ... வலி`; required co-signal `radiation+jaw OR arm OR dyspn` should match the case's narrative mention of arm numbness | net +1 RED catch |
| triage_test_040 | A→B (mixed) | no — chief regex matches but co-signal (respiratory distress in patient) doesn't apply | net +0 (still missed; clinical pattern is paresthesia, not anaphylaxis) |
| triage_test_042 | A | likely yes — chief regex now matches `மூச்சுத்திணறல்`; co-signal needs to match — narrative mentions "shortness of breath + viral infection diagnosis" but doesn't mention cyanosis/speech difficulty/accessory muscle use | depends on co-signal interpretation; possibly net +0 if the rule's design correctly requires severe-distress co-signal |

**Realistic expected RED-recall improvement from these three tightenings: +1 case (triage_test_039 catches via cardiac rule).** From 6/12 to 7/12 = 0.583. Not transformative.

The bigger conclusion: **the rule layer is approaching its ceiling on this dataset.** 5 of the remaining 6 missed REDs are either Bucket B (no canonical pattern in our 6 rules) or Bucket C (genuinely borderline). Further RED-recall improvement requires either model-level changes (Stream 1 retrains) or a substantively expanded rule set (out of scope this sprint).

## What this means for Task 6 framing

Per the user's decision tree:
- "If most misses are Bucket A: tighten rules before Task 6" — Bucket A is dominant (3/6 = 50%) but the realistic catch-rate from tightening is only +1 case.
- "If most misses are Bucket B: accept the rule ceiling, document it, proceed to Task 6 with honest framing" — also defensible: 2 Bucket B + 1 Bucket C = 50% of misses are not rule-fixable.

**Recommendation:** apply the 3 proposed tightenings (small, well-bounded, with new unit tests), then proceed to Task 6 with honest framing that the rule layer's RED-recall ceiling on this dataset is approximately **0.58–0.60** even with broader regex coverage. Going beyond requires the model layer (Stream 1) or significantly more rules (out of scope).

I have NOT applied the tightenings yet. Awaiting user sign-off on the proposed regex changes + the bucket framing.

## Honest gaps in this audit

- The audit re-runs `_matches_rule` standalone, not through the engine's full pipeline. Behaviour matches the engine for the cases tested; if the engine's rule iteration order or the confidence-floor interaction produces different end states, the audit wouldn't catch it. Mitigation: the rule-set delta eval (`run_rules_v2_orig_loras_*.json`) does run end-to-end and confirms the 1-case RED-recall improvement that this audit attributes to ADULT-ANAPHYLAXIS-001.
- The audit shows TN-004 firing on a GREEN case (1 false positive), which the rule-set delta eval may have over-escalated to YELLOW. Did not investigate further; flagging for completeness.
- New rules not exercised on this held-out test set (CARDIAC, HEAD-TRAUMA, ANIMAL-BITE-RESPIRATORY, NEW-ONSET-JAUNDICE — all 0 fires) cannot be validated empirically here. The unit tests in `test_engine.py` give synthetic positive/negative coverage; held-out coverage requires more held-out data containing those clinical patterns.
