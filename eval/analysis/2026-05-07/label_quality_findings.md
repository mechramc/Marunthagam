# Label-quality spot-check findings — 2026-05-07

**Source:** `eval/analysis/2026-05-07/label_quality_spotcheck_LABELED.csv` (80 hand-rated rows: 40 triage + 40 derm, 20 GREEN + 20 YELLOW per specialist, seed-42 sampled from `training/data/formatted/<spec>/train.jsonl`).

## Bottom line

The training labels for the GREEN class — particularly in triage — are systematically biased toward under-triage. **All 12 disagreements (out of 80) are in the same direction: the rater (clinician) wants higher acuity than the gold label assigned.** Triage YELLOW labels are clean. Derm has a separate problem: 22% of cases are routed to the derm specialist when the underlying clinical content is not dermatology at all (poison control, hepatology, pulmonology).

This finding **inverts the original sprint 2 plan** for Task 3 (class-balanced CE retraining). With 30% of triage-GREEN labels actually being YELLOW per clinical review, class-balanced retraining on the current data would *upweight the noisiest class* and push the model away from clinical correctness, not toward it. Relabeling has to come first.

User's decision: **relabel the GREEN class in triage at minimum** (per message 2026-05-07).

## Agreement statistics

### By (specialist, gold class)

| Specialist | Gold class | n | Agree | Disagree | Agreement % |
|---|---|---|---|---|---|
| triage | GREEN | 20 | 14 | 6 | **70%** |
| triage | YELLOW | 20 | 20 | 0 | **100%** |
| derm | GREEN | 20 | 16 | 4 | 80% |
| derm | YELLOW | 20 | 18 | 2 | 90% |
| **Total** | — | **80** | **68** | **12** | **85%** |

### Direction of disagreement

| Direction | Count |
|---|---|
| Toward **higher acuity** (label under-triages) | **12 / 12** |
| Toward lower acuity (label over-triages) | 0 / 12 |
| Same level, different class | 0 / 12 |

The data systematically under-triages. Not a single case in the sample was over-triaged.

### Wrong-specialist routing in derm

9 of 40 derm cases (22%) flagged as belonging to a different specialist:

- Poison/overdose: 1
- Hepatology (elevated ALT, jaundice, ESLD): 2
- Pulmonology (CXR with TB scars, infiltrates): 2
- GI / general surgery (splenic cyst): 1
- Vascular (post-DVT presentation): 1
- Ortho (rotator cuff): 1
- Hepatology with cosmetic concern (xanthelasma + lipid panel — borderline): 1

Mechanism: per `acquire_sources.py`, the keyword regex routed examples into the derm bucket whenever any answer text mentioned skin-related vocabulary, even if the chief complaint was non-dermatologic. This was already on the sprint 1 known-issues list (item B3 in the original pending-work plan). The label-quality data confirms the contamination rate empirically.

### Projected relabeled distributions (if 80-row sample rate generalises)

| Train set | Current G/Y/R | Current % | Projected G/Y/R | Projected % |
|---|---|---|---|---|
| triage (n=351) | 90 / 209 / 52 | 25.6 / 59.5 / 14.8 | 63 / 236 / 52 | 17.9 / 67.2 / 14.8 |
| derm (n=328) | 142 / 164 / 22 | 43.3 / 50.0 / 6.7 | 114 / 176 / 38 | 34.8 / 53.7 / 11.6 |

After projected relabel, **triage YELLOW prior rises from 59.5% to 67.2%.** The class imbalance gets *worse*, not better. Class-balanced CE on relabeled data would still need careful tuning, but the underlying signal is at least clean.

## What this changes about sprint 2

### Task 3 — superseded by relabel workflow

The original Task 3 ("retrain triage and derm with class-balanced CE") cannot run on the current data without baking under-triage bias more deeply. Replace with:

**Task 3a — Relabel triage GREEN training cases** (90 rows): user reviews each, sets corrected label. Gates: (a) review complete, (b) per-class distribution recomputed, (c) decide whether to also relabel triage val (11 GREEN) and test (12 GREEN) so the eval denominator matches the training labels.

**Task 3b — Retrain triage with relabeled data**, *without* class-balanced CE first. Standard 3-epoch run, seed 42, gate on GREEN recall ≥ 0.50 on relabeled own-training-data. If that fails, *then* try class-balanced CE on relabeled data — but the pri or rationale for class balancing is now to handle the fact that GREEN is even rarer post-relabel, not to fight collapse.

**Task 3c — Decide on derm.** The wrong-specialist routing in derm is a data-acquisition bug (`acquire_sources.py`), not just a labeling bug. A simple re-label won't fix it; the bucket itself contains non-derm cases. Options:
- **Drop the contaminated cases** from derm-train: small re-acquire pass that filters on chief-complaint regex rather than answer text. Likely loses ~20% of derm training data.
- **Move the contaminated cases to triage**: those cases are clinically real triage queries; they belong in triage, not derm. Increases triage train slightly.
- **Keep the contamination, accept derm-LoRA isn't really a derm specialist.** This is the current state. Defensible if we ship maternal-only or routed.

User decision needed before we run derm relabel.

### Task 4 — unchanged but now higher priority

The IMNCI rule expansion still stands and is independent of the relabeling. Adult-emergency rules (acute MI, anaphylaxis, head trauma + LOC, severe wheezing, animal bite + respiratory distress) remain on the list. The user's labels even exposed two new candidates from the spotcheck:

- **New cardiac-pattern rule** (informed by spotcheck cases #2 — chest pressure radiating to jaw + dyspnea + tachycardia): the user explicitly flagged this as a case where the model must escalate based on symptom pattern alone, not patient self-attribution. Worth a rule.
- **Animal bite + incomplete rabies PEP** (informed by spotcheck case #6): a dedicated rule for "dog bite + PEP series in progress" → YELLOW (not RED, since it's a compliance question, but should not be GREEN).

### Task 5 — unchanged

Safety classifier rebuild is fully independent. Continue with the validation-set approach.

### Task 6 — gated on Task 3b

Held-out re-eval cannot run until at least triage-relabeling + retrain is complete.

## Proposed workflow for triage GREEN relabel

The cheapest and most reproducible path:

1. I generate `eval/analysis/2026-05-07/triage_green_relabel.csv` containing all 90 triage-train GREEN cases with the same column shape as the spotcheck CSV.
2. User fills `my_label` (GREEN | YELLOW | RED) and `notes` for each.
3. I write a small script that:
   - Reads the user's relabels
   - Updates `training/data/formatted/triage/train.jsonl` to overwrite the gold `level` field for any case whose label changed
   - Saves a backup of the original at `training/data/formatted/triage/train_v1_pre_relabel.jsonl`
   - Logs a diff: which cases changed, from what to what
4. User reviews the diff log and approves before any retraining starts.

Estimated user effort: ~15–25 min for 90 cases at the pace of the 80-row spotcheck. Scope-creep to triage val/test (11+12 GREEN cases) adds ~5 min and gives us clean test labels too — strongly recommend including these.

Want me to generate the 90-case relabel CSV now, or do you want to scope it differently first (e.g., only the cases where the model and gold disagree, ~50-60 candidates)?
