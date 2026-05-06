# Derm contamination move + derm-clean retrain findings (S3-CW-1)

**Date:** 2026-05-07
**Status:** swap canceled — keep sprint-1 derm-LoRA in production stack.

## Summary (for the writeup)

We identified that 21% of the derm specialist's training data was misrouted from other clinical domains (pulmonology, hepatology, GI surgery, etc.). We constructed a cleaned dataset and retrained derm-LoRA on it. On head-to-head evaluation against the original sprint-1 derm-LoRA on identical cleaned test data, the cleaned-data model underperformed (F1 0.538 vs 0.581, RED recall 0.000 vs 1.000). We attribute this to the 15% reduction in training data being more costly than the noise reduction was beneficial in this low-resource specialist regime. Production therefore uses the original sprint-1 derm-LoRA. The contamination diagnosis and cleaned dataset remain documented as data hygiene artifacts — separately valuable from the model decision.

## What we did

1. Applied `apply_derm_contamination.py` (49 MOVE_TO_TRIAGE + 1 RELABEL_ONLY) per user-completed verdicts. Six derm/triage JSONLs backed up to `*_v2_pre_derm_move.jsonl`. Diff log at `derm_contamination_diff.json`.
2. Retrained derm-LoRA on the contamination-cleaned data: `training/outputs/derm-clean-seed42/`. eval_loss = 2.112 (vs sprint-1 derm-seed42 eval_loss = 2.018 — slightly higher).
3. Evaluated both LoRAs on the **same** cleaned derm-test split (n=35) for a fair head-to-head.

## Result

Direct head-to-head on identical data:

| Model | F1 | RED recall | macro F1 |
|---|---|---|---|
| **sprint-1 derm (current production)** | **0.581** | **1.000** | **0.556** |
| derm-clean (retrained on cleaned data) | 0.538 | 0.000 | 0.369 |

Per-class on cleaned derm-test (n=35; 17G / 17Y / 1R):

| Class | sprint-1 (P / R / F1) | derm-clean (P / R / F1) |
|---|---|---|
| GREEN (n=17) | 0.857 / 0.353 / 0.500 | 1.000 / 0.353 / 0.522 |
| YELLOW (n=17) | 0.560 / 0.824 / 0.667 | 0.500 / 0.706 / 0.585 |
| RED (n=1) | 0.333 / 1.000 / 0.500 | 0.000 / 0.000 / 0.000 |

## Verdict

Per the gate criterion in the Sprint 3 plan: *"CW-1 fails if derm-LoRA on cleaned data scores LOWER held-out F1 than the existing sprint-1 derm-LoRA. Action: revert from backup."*

**0.538 < 0.581 → gate failed.** Production stack keeps sprint-1 derm-LoRA. The derm-clean adapter remains on disk at `training/outputs/derm-clean-seed42/final/` for diagnostic reference.

## Why this happened

Two interacting effects on n=35 with only 1 gold-RED case:

1. **Lost training signal.** Cleaning derm-train removed 41 cases (12.5%). Even if those cases were non-dermatology by chief complaint, they were still text-completion supervision the LoRA used to learn the JSON schema and Tamil reasoning style. Smaller train + same hyperparameters = slightly worse fit.
2. **Single RED case is fragile.** sprint-1 derm catches the 1 RED, derm-clean doesn't. With n=1, that swing dominates RED recall (0.0 vs 1.0) and macro F1. A larger held-out RED set would give a more confident answer.

## What stays applied

The contamination MOVE itself is correct data hygiene and stays. Concretely:

- `training/data/formatted/derm/{train,val,test}.jsonl` are now the cleaned versions (49 cases removed).
- `training/data/formatted/triage/{train,val,test}.jsonl` gained 41 + 2 + 6 = 49 cases (legitimate triage queries that were misrouted by `acquire_sources.py`'s permissive keyword regex).
- Backups at `*_v2_pre_derm_move.jsonl` for audit and reversal if needed.

This means **future retrains of triage on the bigger (post-move) triage train (392 rows vs 351)** will benefit from the cleaner data acquisition, even though derm-clean as a model swap doesn't help in this sprint. That's a Sprint 4 lever if there's appetite to retrain B-triage on the post-move train data.

## What this finding tells us about contamination-cleaning as an intervention

In general, removing wrongly-routed training data is correct hygiene but doesn't automatically improve specialist quality. The data-acquisition fix and the model-quality fix are two separate levers:

- **Data hygiene:** the moved cases now enrich triage, where they belong. Auditable. Future-friendly.
- **Specialist quality:** independent question. May require more data, more epochs, or different sampling rather than a smaller-but-cleaner training set.

For the writeup, the framing is: "we identified data acquisition contamination, fixed it for honesty, and the specialist swap didn't help in this measurement window — but the cleaner data acquisition is still the right state to ship."
