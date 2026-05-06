# RED-recall failure-mode analysis — 2026-05-06

**Source artifacts:** `eval/results/run_diag_{routed,triage,derm,maternal}_20260506_150*.json` (held-out test, n=131, single seed @ T=0, with the observability patch landed today: pre-engine level/confidence/overrides + per-class logprobs).

**Headline number under investigation:** RED recall = 0.4167 on the routed config (12 true REDs, 5 caught, 7 missed).

## Bottom line

Across all four routing configurations, **100% of missed REDs are in bucket C** — the model said YELLOW *before* the protocol engine ran, with high reported confidence (mean 0.89), and the engine had nothing to escalate against. Buckets A (model said GREEN high-conf), B (model said GREEN low-conf — engine should have escalated), D (engine fired but didn't help), and E (engine downgraded RED) are all empty across all 4 configs.

The engine is *not* the failure point. The model is. And the engine couldn't have rescued these cases because:

- Per-rule pattern check on the routed config's 7 missed REDs: **6 of 7 had no IMNCI rule whose pattern matched the case symptoms.**
- The 1 remaining case (`maternal_test_028`) had a pattern match (IMNCI-002 = fever), but the rule was filtered out by its `age_group` constraint (the rule is pediatric; the patient is 44).

So the diagnosis is two-part:

1. **Model**: under-classifies adult-emergency presentations (acute MI, anaphylaxis, post-fall head trauma, dog bite + respiratory distress, severe wheezing) as YELLOW with confidence 0.85–0.9.
2. **Engine**: rule set is pediatric-IMNCI-only. There is no adult-cardiac-pain rule, no anaphylaxis rule, no animal-bite rule, no head-trauma rule. So even if the engine were inclined to help, it has nothing to fire on.

These are independent gaps; both have to close to lift RED recall.

## Method

Patched `run_eval.py:_real_predict` to capture, *before* `engine.apply` mutates the `TriageResult` in-place:

- `pre_engine_level` (the model-only level, captured into a local string before any engine call)
- `pre_engine_confidence` (model's reported confidence, ditto)
- `pre_engine_escalation_flag`
- `engine_overrides` (the `list[ProtocolOverride]` that the engine returns and that the previous codepath threw away into `_overrides`)
- `class_logprobs` (top-K logprobs at the level position from a 1-token probe with the prompt primed up to `{"level": "`; requires `logits_all=True` on the Llama load)

Patch is observability-only — no inference behaviour change. Re-ran all 4 routing configs on the held-out test split (single seed, T=0).

## Bucket counts across all 4 configs

| Config | A: pre=GREEN, conf≥0.7 | B: pre=GREEN, conf<0.7 (engine should have escalated) | C: pre=YELLOW | D: engine fired but final≠RED | E: engine downgrade RED→lower | **total missed REDs** |
|---|---|---|---|---|---|---|
| routed | 0 | 0 | **7** | 0 | 0 | 7 |
| triage-only | 0 | 0 | **7** | 0 | 0 | 7 |
| derm-only | 0 | 0 | **9** | 0 | 0 | 9 |
| maternal-only | 0 | 0 | **8** | 0 | 0 | 8 |

Bucket E (engine-downgrade) is empty across all configs — confirms the engine's invariant ("never downgrade urgency") holds in practice, not just in the source code. The user-requested 5th bucket is verified zero.

## Representative cases (bucket C — routed config, all 7 listed)

For each, I show the model's pre-engine state, the per-class probabilities from the 1-token logprobs probe, the engine override trace, and which IMNCI rules' regex would have matched the case symptoms (independent of the rule's own age/duration filters, just to show whether the engine had *any* signal to work with).

### 1. `triage_test_039` — acute MI presentation (chest pain + left-arm numbness), adult, day 1

- Symptoms (Tamil): "ER visit, severe left-chest pain radiating from sternum to left arm, left-arm numbness."
- Pre-engine: level=YELLOW, confidence=0.90, escalation_flag=True
- Class probabilities: YELLOW=0.976, RED=0.019, GREEN=0.000
- Engine overrides: `[]`
- RED-pattern matches: `(none)` — no IMNCI rule covers adult cardiac pain.

### 2. `triage_test_040` — dog bite + acute respiratory distress, adult, day 1

- Symptoms: dog bite 10 min ago + heavy laboured breathing + tingling at bite site.
- Pre-engine: level=YELLOW, confidence=0.90, escalation_flag=True
- Class probabilities: YELLOW=0.888, RED=0.091, GREEN=0.000
- Engine overrides: `[]`
- RED-pattern matches: `(none)` — no rabies / animal-bite rule, and the breathing-distress rule (IMNCI-004) requires specific Tamil phrases ("சாம்பல் நிறம்", "சுவாசிக்க கஷ்டம்") that don't appear in the verbal symptoms field.

### 3. `triage_test_042` — pneumonia / lung inflammation in smoker, adult, day 10

- Symptoms: shortness of breath + severe cough + X-ray showed inflammation around lungs.
- Pre-engine: level=YELLOW, confidence=0.90
- Class probabilities: YELLOW=0.702, RED=0.293, GREEN=0.000
- Engine overrides: `[]`
- RED-pattern matches: `(none)` — `மூச்சுத்திணறல்` (shortness of breath) is not in IMNCI-004's pattern, which uses `மூச்சு விட முடியவில்` instead.

### 4. `triage_test_044` — wheezing pneumonia at sleep, adult, day 2

- Symptoms: wheezing during sleep + already-diagnosed pneumonia + on antibiotics + albuterol inhaler.
- Pre-engine: level=YELLOW, confidence=0.90
- Class probabilities: YELLOW=0.970, RED=0.018, GREEN=0.000
- Engine overrides: `[]`
- RED-pattern matches: `(none)` — wheezing isn't in any RED-level rule pattern.

### 5. `derm_test_040` — chemical-sensitivity anaphylactic-pattern, adult, day 1

- Symptoms: multiple chemical sensitivities, skin rash, itching, **tongue swelling**, extreme thirst.
- Pre-engine: level=YELLOW, confidence=0.85, escalation_flag=False
- Class probabilities: YELLOW=0.960, RED=0.036, GREEN=0.000
- Engine overrides: `[]`
- RED-pattern matches: `(none)` — no anaphylaxis rule. Tongue swelling is a textbook stridor precursor; no current rule catches it.

### 6. `maternal_test_027` — post-fall syncope, adult, day 1

- Symptoms: fell down 3 stairs onto buttocks, **fainted briefly on standing up**, nausea, persistent dizziness.
- Pre-engine: level=YELLOW, confidence=0.90, escalation_flag=False
- Class probabilities: YELLOW=0.991, RED=0.008, GREEN=0.000
- Engine overrides: `[]`
- RED-pattern matches: `(none)` — no head-trauma / loss-of-consciousness rule.

### 7. `maternal_test_028` — chemo patient, rectal bleeding + mucus, adult, day 7

- Symptoms: 44-year-old on Ixempra chemo (4th cycle), passing blood + mucus from rectum.
- Pre-engine: level=YELLOW, confidence=0.90, escalation_flag=False
- Class probabilities: YELLOW=0.990, GREEN=0.007, RED=0.0004
- Engine overrides: `[]`
- RED-pattern matches: `IMNCI-002` (fever) — but the case **does not** have a fever; the patient narrative just *mentions* the word "காய்ச்சல்" (fever) in passing. The pattern match here is a false positive against narrative, not a real fever finding. The engine correctly **filtered this rule out** because its `age_group` constraint targets pediatric cases, not 44-year-olds. Rule logic is fine; the engine's correct decision is "this rule does not apply." There's no rule that does apply to chemo patients with GI bleeding.

## Summary of why the engine couldn't help

Tabulating the 7 routed missed REDs against the IMNCI rule set (15 rules, 7 with `minimum_triage_level=RED`):

| Case | Clinical pattern | Closest existing RED rule | Why engine did not fire |
|---|---|---|---|
| triage_039 | Acute MI (chest pain, arm numbness) | (none) | No adult-cardiac-pain rule |
| triage_040 | Dog bite + respiratory distress | IMNCI-004 (severe breathing) | Pattern phrasing mismatch (Tamil specific) |
| triage_042 | Lung inflammation, dyspnoea | IMNCI-004 | Same pattern phrasing mismatch |
| triage_044 | Wheezing pneumonia | IMNCI-004 | "Wheezing" is not in any pattern |
| derm_040 | Anaphylactic-pattern (tongue swelling) | (none) | No anaphylaxis rule |
| maternal_027 | Post-fall syncope | (none) | No head-trauma / LOC rule |
| maternal_028 | Chemo + GI bleeding | IMNCI-002 | Rule's pediatric age_group filter excluded the case (correctly) |

So 5 of 7 have no semantically related RED rule at all. 2 of 7 have a related rule whose pattern is too narrow (Tamil phrasing) or too age-restricted to fire.

## What this memo is *not*

- **Not a fix proposal.** The diagnosis is "model under-predicts adult emergencies AND engine rule-set is pediatric-only." Both gaps need separate decisions: (a) fix the model? add adult emergencies to training data? expand training? (b) expand the rule set to cover adult emergencies? loosen age filters? Both are out of scope for this sprint.
- **Not a router fix proposal.** All four routing configs show the same pattern (bucket C dominant). The router is irrelevant to RED recall on this test set — every single LoRA under-classifies these cases.
- **Not a confidence-floor fix proposal.** The escalation rule (escalate one level if confidence < 0.7) couldn't help here because the model is over-confident on its YELLOW prediction (mean 0.89). Lowering the threshold would catch some cases but at the cost of escalating many true-YELLOWs to RED — a separate tradeoff the next sprint should evaluate empirically, not adjust by intuition.

## Sequencing note for the next sprint

If the next sprint touches the engine, the cheapest first step is widening the IMNCI rule patterns and adding adult emergency rules (cardiac chest pain, anaphylaxis with airway/tongue involvement, head trauma + LOC, severe-wheezing pattern). That alone could lift RED recall by ~3-4 cases (cases 2, 4, 5, 6 above) without any model retraining.

If the next sprint touches the model, the right experiment is *to confirm* whether triage/derm/maternal LoRAs all show the same YELLOW-overconfidence on adult emergencies, or whether one of them generalises better — Task 1 of this same investigation sprint will provide that data.

Do not act on either gap until both diagnoses (this memo + the specialist-diagnosis memo) are in.
