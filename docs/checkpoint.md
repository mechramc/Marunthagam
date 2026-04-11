# Marunthagam — Checkpoint Log

Each checkpoint is written before every `git commit` + `git push`.
Verifier agent signs off before any push.

---

## Checkpoint Format

```
### [YYYY-MM-DD HH:MM] Checkpoint N — <description>

**Commit:** <SHA>
**Tasks completed:** T?.? — <name>
**Tests passing:** <test count>
**Verifier:** ✅ Approved / ❌ Issues (see notes)
**Notes:** <any issues found or deferred>
```

---

## Checkpoints

### [2026-04-10 20:48] Checkpoint 2 — Dataset pipeline + baseline eval

**Tasks completed:** T4.1–T4.5
**Tests passing:** 31 (9 engine + 10 logger + 12 handler)
**Verifier:** ✅ Approved — all splits non-empty, disclaimer enforced, 4-turn format correct
**Files:** training/data/README.md, training/scripts/translate_dataset.py, training/data/fixtures/ (3×10 JSONL), training/scripts/format_training_data.py
**Notes:** val.jsonl uses max(1, round(n×VAL_RATIO)) to guarantee non-empty with small fixture groups. test.jsonl is empty for 10-example fixtures (expected — real data will have hundreds).

### [2026-04-10 14:30] Checkpoint 1 — Protocol engine + Function calling

**Tasks completed:** T1.1–T1.3, T2.1–T2.7, T3.1–T3.5
**Tests passing:** 21 (9 engine + 12 handler)
**Verifier:** ✅ Approved — all 8 checklist groups passed
**Files:** schema.sql, engine.py, imnci_rules.json (15 rules), load_rules.py, test_engine.py, schemas.py, handler.py, test_handler.py, baseline_examples.json, baseline_eval.py, baseline.yaml
**Notes:** Verifier noted edge case: RED + low confidence doesn't set escalation_flag (reasonable behavior, future test TODO)

### [2026-04-10 13:39] Checkpoint 0 — Initial scaffold

**Commit:** cc2ab81
**Tasks completed:** Infrastructure (repo structure, CLAUDE.md, protocol schemas, plan)
**Tests passing:** N/A (no test files yet)
**Verifier:** N/A (initial scaffold)
**Notes:** Spec read, full plan written, folder structure created. Week 1 execution begins next.
