# Marunthagam — Sprint Status

**Last updated:** 2026-04-10
**Sprint deadline:** 2026-05-18 (38 days remaining)
**Overall progress:** 31% (15/49 atomic tasks complete)

---

## Current Focus

**Phase:** Week 1 — Foundation
**Active task:** T4.1 — Dataset pipeline + T5+ training scripts (parallel)

---

## Phase Progress

| Phase | Week | Tasks | Done | Status |
|-------|------|-------|------|--------|
| Foundation | Apr 10–16 | T1–T4 (28 tasks) | 0 | 🔲 Not started |
| Fine-Tuning | Apr 17–23 | T5–T6 (11 tasks) | 0 | 🔲 Blocked on data |
| App Build | Apr 24–30 | T7–T8 (14 tasks) | 0 | 🔲 Not started |
| Evaluation | May 1–7 | T9–T10 (16 tasks) | 0 | 🔲 Blocked on models |
| Submission | May 8–18 | T11 (6 tasks) | 0 | 🔲 Not started |

---

## Completed Tasks

- ✅ T1.1 — eval/data/baseline_examples.json (20 Tamil triage fixtures, 7G/7Y/6R)
- ✅ T1.2 — training/configs/baseline.yaml
- ✅ T1.3 — training/scripts/baseline_eval.py
- ✅ T2.1 — inference/protocol_engine/schema.sql (no PII, CHECK constraints)
- ✅ T2.2 — inference/protocol_engine/rules/imnci_rules.json (15 rules)
- ✅ T2.3 — inference/protocol_engine/load_rules.py (idempotent INSERT OR IGNORE)
- ✅ T2.4 — inference/protocol_engine/engine.py (ProtocolEngine, confidence floor, disclaimer)
- ✅ T2.5 — inference/protocol_engine/test_engine.py (9 tests)
- ✅ T2.6 — All 9 engine tests pass
- ✅ T3.1 — inference/function_calling/schemas.py (Pydantic v2, disclaimer validator)
- ✅ T3.2 — inference/function_calling/handler.py (tool_call extraction + fallback)
- ✅ T3.3 — inference/function_calling/test_handler.py (12 tests)
- ✅ T3.4 — All 12 handler tests pass
- ✅ T2.7/T3.5 — Verifier approved (21 total tests passing)

---

## Blockers

*(none yet)*

---

## Risk Flags

- [ ] E4B Tamil quality after fine-tuning (Medium likelihood, High impact) — Mitigation: 26B-A4B demo fallback
- [ ] Native audio in GGUF (High likelihood, Medium impact) — Mitigation: Whisper-small-Tamil offline
- [ ] KALAVAI fusion gain (Medium likelihood, High impact) — Mitigation: Report result honestly
- [ ] Scope (High likelihood, High impact) — Mitigation: MVP path defined in plan

---

## MVP Gate (Must-Ship Minimum)

- [ ] LoRA-Triage fine-tuned (1 specialist min)
- [ ] triage_classify() function calling working
- [ ] Protocol grounding engine (SQLite + IMNCI)
- [ ] Evaluation: F1/P/R per class, 3 seeds
- [ ] CLI demo on llama.cpp
- [ ] Technical write-up
- [ ] Demo video
- [ ] GitHub + HuggingFace weights
