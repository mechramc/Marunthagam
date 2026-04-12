# Marunthagam — Sprint Status

**Last updated:** 2026-04-12
**Sprint deadline:** 2026-05-18 (36 days remaining)
**Overall progress:** 90% (all code complete — pending: LoRA training + GGUF weights + demo video)

---

## Current Focus

**Phase:** Week 4/5 — Evaluation + Submission prep
**Active task:** T5/T6 execution — LoRA training on RTX 5090 (3 seeds × 3 specialists)
**Blocked on:** Actual GGUF weights (training not yet run)

---

## Phase Progress

| Phase | Week | Tasks | Done | Status |
|-------|------|-------|------|--------|
| Foundation | Apr 10–16 | T1–T4 | ✅ | Complete |
| Fine-Tuning | Apr 17–23 | T5–T6 | ✅ | Scripts complete — training pending |
| App Build | Apr 24–30 | T7–T8 | ✅ | Complete |
| Evaluation | May 1–7 | T9–T10 | ✅ | Scripts complete — real eval pending weights |
| Submission | May 8–18 | T11 | ✅ | README + docs complete — video + HF upload pending |

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
- ✅ T4.1 — training/data/README.md (data pipeline docs, 8 sections)
- ✅ T4.2 — training/scripts/translate_dataset.py (Gemma 4 31B automated translation)
- ✅ T4.3 — training/data/fixtures/ (30 fixture entries: 10 each triage/derm/maternal)
- ✅ T4.4 — training/scripts/format_training_data.py (Gemma 4 chat format, 80/10/10 split)
- ✅ T4.5 — Format pipeline validated (val non-empty, disclaimer enforced, correct 4-turn format)
- ✅ T5.1 — training/scripts/train_lora.py (parameterized Unsloth QLoRA trainer, 3 seeds)
- ✅ T5.2 — training/configs/lora_triage.yaml, lora_derm.yaml, lora_maternal.yaml
- ✅ T6.1 — training/scripts/train_router.py (KalavaiRouter: nn.Linear + softmax)
- ✅ T6.2 — training/scripts/export_gguf.py + training/configs/router.yaml
- ✅ T6.3 — Week 2 training scripts verified (LoRA + router + GGUF export, all guarded imports)
- ✅ T8.1 — inference/protocol_engine/logger.py (encrypted SQLite interaction logger)
- ✅ T8.2 — inference/protocol_engine/test_logger.py (12 tests)
- ✅ T8.3 — All 12 logger tests pass
- ✅ T8.4 — Logger + full inference suite verified (33 total tests green)
- ✅ T7.1 — android JNI: CMakeLists.txt, marunthagam_jni.cpp
- ✅ T7.2 — android Kotlin: LlamaWrapper.kt, TriageEngine.kt, TriageLogDao.kt
- ✅ T7.3 — android UI: activity_triage.xml (Material 3, disclaimer always visible)
- ✅ T7.4 — android/build.gradle.kts (root + app), settings.gradle.kts
- ✅ T7.5 — android/app/src/main/AndroidManifest.xml (largeHeap, INTERNET removed)
- ✅ T7.6 — android MainActivity.kt (ViewBinding, TriageEngine integration, coroutines)
- ✅ T7.7 — android strings.xml, arrays.xml (Tamil UI strings + age groups)
- ✅ T7.8 — android gradle wrapper properties, proguard-rules.pro, adaptive icons
- ✅ T9.1 — eval/scripts/run_eval.py (full suite, mock + real, 3-seed aggregation)
- ✅ T9.2 — eval/scripts/eval_triage.py (per-class P/R/F1, RED recall, bootstrap CI)
- ✅ T9.3 — eval/scripts/eval_safety.py (100 adversarial prompts, refusal rate)
- ✅ T9.4 — eval/scripts/eval_latency.py (TTFT + throughput, phone/workstation targets)
- ✅ T9.5 — eval/data/adversarial_prompts.json (100 prompts, 5 categories, Tamil+English)
- ✅ T9.6 — eval/scripts/ablation_rank.py (rank 4/8/16/32/64 comparison, break-even)
- ✅ T10.1 — dashboard React+D3: package.json, vite.config.js, App.jsx, main.jsx
- ✅ T10.2 — dashboard components: TriageHeatmap, TriageTrendChart, AlertPanel, StatsSummary
- ✅ T10.3 — dashboard pages: Overview, MapView, AlertsView, TrendsView
- ✅ T10.4 — dashboard i18n: ta.js (Tamil strings centralized), mockData.js
- ✅ T11.1 — README.md (11-section submission doc, ASCII tier diagram, eval targets)
- ✅ T11.2 — docs/architecture.md (KALAVAI, protocol engine, privacy, inference pipeline)
- ✅ T11.3 — docs/protocol_spec.md (Open Protocol v1.0, schema, privacy guarantees)
- ✅ eval/notebooks/results_analysis.ipynb (7-section analysis notebook)

---

## Remaining Work (Blocking Submission)

- [ ] **Run LoRA training** — `python train_lora.py --config lora_triage.yaml --seed 42/137/256` (RTX 5090)
- [ ] **Run KALAVAI router training** — after all 3 specialists complete
- [ ] **Export GGUF** — `python export_gguf.py --checkpoint outputs/fused-best`
- [ ] **Run real eval** — `python run_eval.py --model models/fused-E4B-Q4_K_M.gguf --seeds 42,137,256`
- [ ] **Fill eval table in README.md** — replace "Pending" with actual results
- [ ] **Record demo video** — 3-minute CLI demo (Android optional)
- [ ] **Publish HuggingFace weights** — murailabs/marunthagam-*
- [ ] **Submit to Kaggle** by May 18, 2026

---

## Blockers

- **GGUF weights not yet trained** — all eval scripts, Android inference, and dashboard live data blocked on actual model training

---

## MVP Gate (Must-Ship Minimum)

- [ ] LoRA-Triage fine-tuned (1 specialist min)
- [x] triage_classify() function calling working
- [x] Protocol grounding engine (SQLite + IMNCI)
- [ ] Evaluation: F1/P/R per class, 3 seeds
- [ ] CLI demo on llama.cpp
- [x] Technical write-up
- [ ] Demo video
- [ ] GitHub + HuggingFace weights

---

## Risk Flags

- [ ] E4B Tamil quality after fine-tuning (Medium likelihood, High impact) — Mitigation: 26B-A4B demo fallback
- [ ] Native audio in GGUF (High likelihood, Medium impact) — Mitigation: Whisper-small-Tamil offline
- [ ] KALAVAI fusion gain (Medium likelihood, High impact) — Mitigation: Report result honestly
- [ ] Scope (Low likelihood now — most code done) — MVP path defined in plan
