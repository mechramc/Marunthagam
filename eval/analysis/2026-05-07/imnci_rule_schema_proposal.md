# IMNCI rule expansion — schema proposal (BLOCKER for Task 4 rule code)

**Status:** awaiting user approval. No rule code written until this is signed off.

## Why a schema decision is needed before any rule is written

The current engine matches `condition_pattern` regex against the full Tamil narrative. From `eval/scripts/run_eval.py:446`:

```python
triage, overrides = engine.apply(
    triage,
    symptoms=case.tamil_question.strip() or case.verbal_symptoms,
    ...
)
```

So `symptoms` is whatever the patient typed — often a long narrative containing many anatomical and symptom mentions, only some of which are the chief complaint. Running a regex like `chest pain|cardiac|MI` against that produces false positives whenever the patient's narrative *mentions* chest pain in passing — for example, a query about a knee injury that ends "I had chest pain six years ago, doctor said it was nothing." That should not fire a cardiac rule.

This was already an empirical problem in sprint 1: in `red_failure_modes.md` row `maternal_test_028`, the engine's IMNCI-002 (fever) pattern matched because the patient's narrative *mentioned* the word `காய்ச்சல்` even though the case was about chemo + GI bleeding, not fever. The rule-engine correctly skipped it via the age-group filter, but only because that filter happened to apply. With a cardiac rule, no equivalent filter exists.

## Proposed schema

The training data already extracts a structured chief-complaint field (`function_call_args.verbal_symptoms`). Use it.

```python
# New schema fields per rule
{
  "id": "ADULT-CARDIAC-001",
  "minimum_triage_level": "RED",

  # Match against the structured chief complaint, NOT the full narrative.
  "chief_complaint_pattern": "chest\\s*(pain|press|tight|discomf)|cardiac|மார்பு\\s*(வலி|இறுக்க)",

  # Optional REQUIRED additional signals — rule fires only if ALL listed
  # patterns match somewhere in chief complaint OR narrative. Each entry is
  # a regex; ALL must match. Use to express AND-combinations like
  # "chest pain AND (jaw radiation OR dyspnea OR tachycardia)".
  "required_co_signals": [
    "(radiat|jaw|left\\s*arm|dyspn|short\\s*of\\s*breath|tachycard|sweat|diapho)|(மார்|தோள்|கை.*மரத்|மூச்சு)"
  ],

  # Optional NEGATIVE scoping — rule does NOT fire if any of these match in
  # chief complaint or narrative. Used for "ignore if patient self-attributes
  # to anxiety AND no symptom-pattern signals" — but cardiac rule explicitly
  # ignores self-attribution per user direction (so this list stays empty
  # for ADULT-CARDIAC-001).
  "negative_scoping": [],

  # Existing fields preserved
  "age_group": "adult|elderly|adolescent",   # OR pattern for permissive scoping
  "duration_days_max": null,                  # null = any duration
  "duration_days_min": null,
  "override_reason": "Cardiac-pattern symptoms (chest pain + radiation/dyspnea/tachycardia). Symptom pattern alone requires same-day cardiology evaluation regardless of patient self-attribution."
}
```

### Match algorithm

```
def matches(rule, chief_complaint, narrative, age_group, duration_days):
    full_text = f"{chief_complaint}\n{narrative}"

    # 1. Chief complaint must match the primary pattern (NOT the full narrative).
    if not re.search(rule["chief_complaint_pattern"], chief_complaint, re.IGNORECASE):
        return False

    # 2. ALL required co-signals must match somewhere (chief OR narrative).
    for co_pat in rule.get("required_co_signals", []):
        if not re.search(co_pat, full_text, re.IGNORECASE):
            return False

    # 3. Any negative scoping pattern means the rule is suppressed.
    for neg_pat in rule.get("negative_scoping", []):
        if re.search(neg_pat, full_text, re.IGNORECASE):
            return False

    # 4. Age and duration constraints (preserved).
    if not _age_matches(rule["age_group"], age_group):
        return False
    if rule.get("duration_days_min") and duration_days < rule["duration_days_min"]:
        return False
    if rule.get("duration_days_max") and duration_days > rule["duration_days_max"]:
        return False

    return True
```

### Backwards compatibility

Existing 15 IMNCI rules use `condition_pattern`. Two options:

**A. Migrate all old rules to new schema.** Translate each `condition_pattern` to `chief_complaint_pattern` (same regex; just a name change). Add empty `required_co_signals` and `negative_scoping`. Keep the 15 rules behaviour-identical. Slight diff in matching only because the old code matched against full narrative; new code matches chief complaint only — so existing rules become *more* specific, which is what the chemo+fever case showed we wanted anyway.

**B. Support both schemas.** Old rules with `condition_pattern` continue using the old behaviour; new rules use the new schema. Avoids touching working code. Confusing long-term.

**Recommendation: A.** The behaviour change is in the desired direction (fewer false positives) and exposing it now lets us re-validate the existing 15 rules at the same time we add the 6 new ones.

### Age-group syntax

Existing rules use `age_group: "infant"` (single value). To express adult-emergency rules that should fire on adolescent + adult + elderly but not infant/child, use a pipe-separated form: `"adolescent|adult|elderly"`. The `_age_matches` helper checks membership.

## Six new rules to add (informed by `red_failure_modes.md` and the user's spotcheck)

For each, I will write the regex + co-signals + age/duration constraints + override reason, write 1 positive + 1 negative unit test, and surface for review *before* committing the rule code.

| ID | Min level | Trigger | Notes |
|---|---|---|---|
| **ADULT-CARDIAC-001** | RED | chest pain/pressure/tightness in chief complaint **AND** (radiation OR dyspnea OR tachycardia OR diaphoresis) anywhere | adolescent+. **No negative scoping for self-attribution** per user direction — symptom pattern alone is sufficient. (Catches `triage_test_039`.) |
| **ADULT-ANAPHYLAXIS-001** | RED | tongue swelling OR lip swelling OR airway/breathing distress in chief complaint **AND** (rash OR allergy OR ingestion OR sting) anywhere | any age. (Catches `derm_test_040` chemical sensitivity case with tongue swelling.) |
| **ADULT-HEAD-TRAUMA-001** | RED | (fall OR head injury OR struck) in chief complaint **AND** (LOC OR fainted OR confusion OR vomiting OR persistent headache) anywhere | any age. (Catches `maternal_test_027` post-fall syncope.) |
| **ADULT-RESPIRATORY-001** | RED | (severe wheezing OR severe dyspnea OR pneumonia + worsening) in chief complaint **AND** age ≥ adolescent | extends IMNCI-004 to adults. (Catches `triage_test_042` and `triage_test_044`.) |
| **ANIMAL-BITE-PEP-001** | YELLOW | (dog bite OR cat bite OR animal bite OR rabies) in chief complaint **AND** (incomplete OR unfinished OR halfway OR PEP series) anywhere | any age. **YELLOW not RED** per user direction — it's an urgent compliance question, not same-hour. (Catches the spotcheck rabies-PEP case.) |
| **NEW-ONSET-JAUNDICE-001** | RED | (jaundice OR yellow skin OR yellow sclera OR காமாலை) in chief complaint **AND NOT** (known liver disease OR diagnosed hepatitis) | any age. (Catches `derm_train_284` per user spotcheck — new-onset acute jaundice in undiagnosed patient.) |

## What I need from you

1. **Approve schema option A** (migrate existing 15 rules to new schema while adding 6 new ones).
2. **Approve the 6 rules listed above** — names, levels, trigger logic. Most consequential: confirm ADULT-CARDIAC-001 has no self-attribution negative scoping (the user already said this in the previous turn, just double-checking before code is written).
3. **Approve test format**: each new rule has a positive case (rule fires) + a negative case (rule does not fire), expressed as `(chief_complaint, narrative, age_group, duration_days, expected_fire: bool)` tuples in a pytest-style file.

Once approved I will:
- Write the schema migration (existing rules translated to new fields).
- Add the 6 new rules with the patterns specified above.
- Write 12 unit tests (6 pos + 6 neg).
- Patch `engine.py:_matches_rule` to consume the new schema.
- Patch `run_eval.py` to pass `chief_complaint` (= `verbal_symptoms`) and `narrative` (= `tamil_question`) separately.
- Re-run held-out test with original LoRAs + expanded rules to isolate the rule-set delta.

Until all three approvals land, no rule code, no engine changes, no eval re-runs.
