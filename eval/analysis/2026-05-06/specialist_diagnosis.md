# Specialist diagnosis — 2026-05-06

**Source artifacts:**
- Loss curves: `training/outputs/<spec>-seed{42,137,256}/checkpoint-{63,66,69}/trainer_state.json`
- Training data: `training/data/formatted/<spec>/{train,val,test}.jsonl`
- Held-out diag re-runs (with logprobs + override traces): `eval/results/run_diag_{routed,triage,derm,maternal}_20260506_150*.json`
- Cross-domain training-set evals: `eval/results/run_cross_{triage,derm,maternal}_on_{triage,derm,maternal}_train_20260506_15*.json`

**Headline finding for the next sprint:** Triage-LoRA and derm-LoRA are **undertrained** with severe class collapse to YELLOW. Maternal-LoRA is partially undertrained but generalises better because its training distribution was less class-imbalanced — not because of any architectural advantage.

**Verdict:** **both, partially — with a clear severity ordering.** Triage and derm collapsed catastrophically; maternal underfit gracefully.

This means the router rework (B1) **must come after** triage and derm are retrained. Optimising routing over signal that has collapsed to "always YELLOW" is meaningless.

---

## Finding 1 — Loss curves

All three specialists were trained for 3 epochs at the same hyperparameters (rank=32, alpha=64, lr=2e-4, batch=4, grad-accum=4, warmup=50). Loss-history (every 10 train steps + every epoch on val):

### Triage (best=1.9042, eval_loss[42] = 2.579 → 2.136 → 1.904)
| step | seed42 train | seed137 train | seed256 train |
|---|---|---|---|
| 10 | 14.523 | 14.555 | 14.532 |
| 20 | 6.940 | 6.966 | 6.941 |
| 30 | 3.583 | 3.601 | 3.586 |
| 40 | 2.113 | 2.115 | 2.105 |
| 50 | 1.525 | 1.520 | 1.515 |
| 60 | 1.272 | 1.270 | 1.267 |

### Derm (best=2.0183, eval_loss[42] = 2.723 → 2.220 → 2.018)
| step | seed42 train | seed137 train | seed256 train |
|---|---|---|---|
| 10 | 14.622 | 14.655 | 14.633 |
| 20 | 6.930 | 6.954 | 6.937 |
| 30 | 3.559 | 3.565 | 3.562 |
| 40 | 2.100 | 2.101 | 2.103 |
| 50 | 1.510 | 1.515 | 1.506 |
| 60 | 1.310 | 1.311 | 1.303 |

### Maternal (best=1.9122, eval_loss[42] = 2.575 → 2.137 → 1.945)
| step | seed42 train | seed137 train | seed256 train |
|---|---|---|---|
| 10 | 14.478 | 14.511 | 14.492 |
| 20 | 6.998 | 7.005 | 6.995 |
| 30 | 3.601 | 3.615 | 3.613 |
| 40 | 2.164 | 2.179 | 2.171 |
| 50 | 1.481 | 1.495 | 1.485 |
| 60 | 1.299 | 1.308 | 1.303 |

**Observations:**

- All three follow nearly identical training-loss trajectories (descent from ~14.5 to ~1.27 at step 60). Hyperparameters are not the differentiator.
- **Eval loss is monotonically decreasing across all three epochs for all three specialists.** The slope between epoch 2 and epoch 3 is meaningful (~0.10-0.20 per epoch). **None of the three converged** — they were stopped while the eval loss was still descending.
- Derm has the highest final eval loss (2.018), worse than triage (1.904) and maternal (1.912). That ordering is consistent with derm having the lowest in-distribution F1 (see Finding 4).
- All seeds within a specialist agree to within ~0.05 eval loss — seed-to-seed variance is small.

The single sentence: **none of the three LoRAs has converged at 3 epochs**, and derm is the worst-fit of the three even on its own validation set.

---

## Finding 2 — Training set sizes and label distributions

| Spec | train n | val n | test n | train G/Y/R | YELLOW % | RED % |
|---|---|---|---|---|---|---|
| triage | 351 | 43 | 45 | 90 / 209 / 52 | **59.5%** | 14.8% |
| derm | 328 | 41 | 41 | 142 / 164 / 22 | 50.0% | 6.7% |
| maternal | 353 | 44 | 45 | 195 / 134 / 24 | **38.0%** | 6.8% |

**Held-out test split (n=131) gold label distribution: 42% GREEN, 49% YELLOW, 9% RED.**

**Observations:**

- Sizes are within 8% of each other (328 vs 353). No specialist is meaningfully under-supplied; this is *not* a data-volume problem.
- Class distributions are **wildly different**: triage's training data is 60% YELLOW; maternal's is 38% YELLOW. Derm sits between at 50%.
- Triage has the highest RED proportion (14.8% — supporting that the dataset includes more emergency presentations) but also the highest YELLOW dominance.
- The two specialists that show class-collapse on held-out (triage and derm — both over-predict YELLOW) have the highest YELLOW prior in their training data. The one that doesn't collapse (maternal) has the lowest YELLOW prior.

This pattern is consistent with class-imbalance-driven prior collapse: the model learns to bet the prior, and when one class dominates, that prior overwhelms what would otherwise be discriminative features.

---

## Finding 3 — Per-class confidence calibration on held-out test

For each LoRA, binned by class probability from the 1-token logprobs probe at the level position. Top-1 calibration (rows where the model's pre-engine pred is the class being binned) — empirical accuracy in each bin:

### Triage-only LoRA (held-out test, n=131)
| Class | P([0.0,0.2)) | [0.2,0.4) | [0.4,0.6) | [0.6,0.8) | [0.8,1.01) |
|---|---|---|---|---|---|
| GREEN | n=0 | n=0 | n=2, acc=1.000 | n=3, acc=1.000 | **n=6, acc=1.000** |
| YELLOW | n=0 | n=2, acc=1.000 | n=5, acc=0.800 | n=9, acc=0.667 | **n=91, acc=0.495** |
| RED | n=0 | n=0 | n=1, acc=0.000 | n=2, acc=0.500 | **n=10, acc=0.400** |

### Derm-only LoRA
| Class | [0.0,0.2) | [0.2,0.4) | [0.4,0.6) | [0.6,0.8) | [0.8,1.01) |
|---|---|---|---|---|---|
| GREEN | n=1, acc=1.000 | n=2, acc=0.500 | n=2, acc=1.000 | n=2, acc=1.000 | **n=13, acc=0.923** |
| YELLOW | n=0 | n=1, acc=1.000 | n=4, acc=0.250 | n=6, acc=0.667 | **n=90, acc=0.544** |
| RED | n=0 | n=0 | n=0 | n=2, acc=0.000 | **n=8, acc=0.375** |

### Maternal-only LoRA
| Class | [0.0,0.2) | [0.2,0.4) | [0.4,0.6) | [0.6,0.8) | [0.8,1.01) |
|---|---|---|---|---|---|
| GREEN | n=0 | n=0 | n=4, acc=0.750 | n=2, acc=1.000 | **n=29, acc=0.862** |
| YELLOW | n=0 | n=0 | n=11, acc=0.545 | n=11, acc=0.727 | **n=63, acc=0.619** |
| RED | n=0 | n=0 | n=0 | n=1, acc=0.000 | **n=10, acc=0.400** |

### Routed (KALAVAI) — for reference
| Class | [0.0,0.2) | [0.2,0.4) | [0.4,0.6) | [0.6,0.8) | [0.8,1.01) |
|---|---|---|---|---|---|
| GREEN | n=0 | n=2, acc=0.500 | n=2, acc=1.000 | n=2, acc=1.000 | **n=21, acc=0.905** |
| YELLOW | n=0 | n=1, acc=1.000 | n=6, acc=0.500 | n=9, acc=0.667 | **n=74, acc=0.581** |
| RED | n=0 | n=0 | n=1, acc=0.000 | n=4, acc=0.250 | **n=9, acc=0.444** |

**Observations:**

- **GREEN class is well-calibrated** for all three LoRAs (when `P(GREEN) ≥ 0.8`, empirical accuracy is 0.86–1.00). The model is honest when it's confident GREEN.
- **YELLOW class is severely overconfident across all three LoRAs**:
  - Triage: 91 cases at `P(YELLOW) ≥ 0.8`, but empirical accuracy is **0.495** — the model says "almost certainly YELLOW" but is right less than half the time.
  - Derm: 90 cases at `P(YELLOW) ≥ 0.8`, empirical accuracy 0.544.
  - Maternal: 63 cases at `P(YELLOW) ≥ 0.8`, empirical accuracy 0.619 — best of the three but still 19 percentage points below the model's claim.
  - Maternal's middle-bin counts are the highest (11 cases at `P(YELLOW) ∈ [0.4, 0.6)`, vs 5–6 for others), meaning maternal more often expresses uncertainty when uncertain — a healthier behaviour.
- **RED class is poorly calibrated** for all three: at `P(RED) ≥ 0.8`, empirical accuracy is 0.375–0.444. When the model "screams RED", it's wrong ~60% of the time. (Saving grace: this is rare — only 8–10 cases per config.)
- The confidence floor (`< 0.7 → escalate one level`) **cannot rescue the YELLOW-overconfidence problem** because the model puts probability mass at 0.85–0.99 on YELLOW for the cases that are actually RED. Lowering the threshold would catch some, but at the cost of escalating many true-YELLOWs to RED — that tradeoff has to be evaluated empirically, not adjusted by intuition.

The single sentence: **the model is honest when GREEN, dishonestly confident when YELLOW, and badly miscalibrated on RED — all three LoRAs share this pattern, with maternal least bad.**

---

## Finding 4 — Cross-domain training-set evaluation (the decisive experiment)

Per your Task 1 step 4 instruction. Each LoRA evaluated on each specialist's `train.jsonl`. Per-set numbers reported separately, no aggregation.

### Each LoRA on its own training data (in-distribution baseline)

| LoRA | Train set | n | Weighted F1 | Macro F1 | RED recall | GREEN recall | YELLOW recall |
|---|---|---|---|---|---|---|---|
| triage-LoRA | triage-train | 351 | **0.5569** | 0.469 | 0.481 | **0.078** | 0.904 |
| derm-LoRA | derm-train | 328 | **0.5190** | 0.495 | 0.455 | **0.197** | 0.933 |
| maternal-LoRA | maternal-train | 353 | **0.6180** | 0.572 | 0.500 | 0.549 | 0.724 |

### Maternal-LoRA on the other specialists' training data (cross-domain)

| LoRA | Train set | n | Weighted F1 | Macro F1 | RED recall | GREEN recall | YELLOW recall |
|---|---|---|---|---|---|---|---|
| **maternal-LoRA** | **triage-train** | 351 | **0.6426** | 0.575 | 0.481 | 0.311 | 0.871 |
| **maternal-LoRA** | **derm-train** | 328 | **0.6375** | 0.588 | 0.455 | 0.444 | 0.860 |

### Per the user's spec — what these numbers mean

> "If maternal beats triage-LoRA on triage's own training data, that's a damning finding about triage-LoRA."

**It does. By +0.086 weighted F1 (0.6426 vs 0.5569).** Maternal-LoRA — which has never seen any triage-train row — fits triage-train better than triage-LoRA does. The same pattern holds for derm: maternal-LoRA scores +0.118 above derm-LoRA on derm's own data (0.6375 vs 0.5190).

> "If maternal loses on training data but wins on held-out, triage-LoRA overfit."

This is **not** the pattern we see. Maternal *wins* on the others' training data. So the alternative diagnosis (triage and derm overfit, generalise badly) is ruled out — they don't even fit the data they were trained on.

### Where the gap comes from — per-class

The dominant gap is **GREEN recall**:

- triage-LoRA on triage-train: GREEN recall **0.078** (catches 7 / 90 true GREENs)
- maternal-LoRA on triage-train: GREEN recall **0.311** (catches 28 / 90 true GREENs — 4× more)

- derm-LoRA on derm-train: GREEN recall **0.197** (catches 28 / 142 true GREENs)
- maternal-LoRA on derm-train: GREEN recall **0.444** (catches 63 / 142 true GREENs — 2.3× more)

YELLOW recall is similar (~0.87–0.93 across configs — all LoRAs over-predict YELLOW). RED recall is essentially tied across configs on the same data. The discriminator is GREEN — triage-LoRA and derm-LoRA have **collapsed onto a "predict YELLOW" prior** and refuse to commit to GREEN even when the training data clearly labels something GREEN. Maternal-LoRA preserved the GREEN class because its training distribution had 55% GREEN — strong enough to keep that decision boundary alive.

---

## Verdict

**Both, partially — but the severity is asymmetric.**

- **Triage-LoRA: severely undertrained / class-collapsed.** Only 7.8% GREEN recall on its own training data, against 60% YELLOW prior. The LoRA has memorised "predict YELLOW" rather than learning the task.
- **Derm-LoRA: severely undertrained / class-collapsed.** Same pattern at 19.7% GREEN recall, 93% YELLOW recall. Derm also has the highest eval loss (2.018) of the three.
- **Maternal-LoRA: mildly undertrained but not class-collapsed.** Eval loss still descending at epoch 3, but the model preserved meaningful predictions across all three classes. Maternal is the *least* miscalibrated, has the strongest in-distribution fit among the three (F1 0.618 vs 0.557 / 0.519), and outperforms the other two on their own data.

**Why maternal escaped collapse:** maternal-LoRA's training set was the most balanced toward GREEN (55%) and least YELLOW-heavy (38%). Triage and derm had >50% YELLOW, which combined with under-training produced prior collapse. Maternal's distribution kept the GREEN decision boundary alive long enough for the LoRA to learn it.

**This is not a router problem.** Across all 4 routing configs (routed / triage-only / derm-only / maternal-only), the missed-RED bucket is identical (all in bucket C — model says YELLOW with high confidence, regardless of which LoRA is in the slot). The signal coming out of triage-LoRA and derm-LoRA has collapsed; switching how you route into them doesn't help.

**Sequencing for the next sprint:**

1. **First, retrain triage-LoRA and derm-LoRA** with one or more of the following (decision is for the next sprint, this memo is diagnosis only):
   - More epochs (eval_loss was still descending at 3) — cheapest first try.
   - Class-balanced loss weights (e.g. inverse-frequency weighting on the cross-entropy) to break the YELLOW prior collapse.
   - A lower learning rate — current 2e-4 may be overshooting the GREEN boundary on the imbalanced data.
   - Hard-negative mining: oversample the misclassified-as-YELLOW GREEN cases from validation back into training.
   The cheapest experiment is "train for 6 epochs instead of 3" — the loss curves predict eval_loss will keep dropping; if it does and GREEN recall recovers above 0.5, that's the fix.

2. **Optionally retrain maternal-LoRA the same way** to see if it improves; not strictly necessary because it didn't collapse.

3. **Then re-run the held-out eval** with the new LoRAs.

4. **Only then touch the router.** The user's Task 5 instruction explicitly anticipated this case: "the router fix has to come AFTER specialist retraining or we'll be optimizing routing over weak signal." Confirmed by data.

5. **Independently from the model work, expand the IMNCI rule set** to cover the adult-emergency cases the held-out RED failures exposed (acute MI, anaphylaxis, head trauma + LOC, severe wheezing, animal bite + respiratory distress). See `red_failure_modes.md`. Those rule additions could lift RED recall on the current LoRAs; combined with retraining they should push it through the 0.90 target.

This memo is diagnosis-only. No code changes, no router rework, no retraining triggered yet. Decision lies with the user.
