# Marunthagam - Sprint Status

**Last updated:** 2026-04-14
**Sprint deadline:** 2026-05-18 (34 days remaining)
**Overall progress:** 92% (dashboard + Android builds verified; pending: LoRA training, GGUF weights, real evals, demo video)

---

## Current Focus

**Phase:** Week 4/5 - Evaluation + Submission prep  
**Active task:** T5/T6 execution - LoRA training on RTX 5090 (3 seeds x 3 specialists)  
**Blocked on:** Actual GGUF weights (training not yet run)

---

## Phase Progress

| Phase | Week | Tasks | Done | Status |
|-------|------|-------|------|--------|
| Foundation | Apr 10-16 | T1-T4 | Completed | Complete |
| Fine-Tuning | Apr 17-23 | T5-T6 | Scripts complete | Training pending |
| App Build | Apr 24-30 | T7-T8 | Completed | Build verified |
| Evaluation | May 1-7 | T9-T10 | Scripts complete | Real eval pending weights |
| Submission | May 8-18 | T11 | Docs complete | Video + HF upload pending |

---

## Completed Tasks

- Completed T1.1 - `eval/data/baseline_examples.json` (20 Tamil triage fixtures, 7G/7Y/6R)
- Completed T1.2 - `training/configs/baseline.yaml`
- Completed T1.3 - `training/scripts/baseline_eval.py`
- Completed T2.1 - `inference/protocol_engine/schema.sql` (no PII, CHECK constraints)
- Completed T2.2 - `inference/protocol_engine/rules/imnci_rules.json` (15 rules)
- Completed T2.3 - `inference/protocol_engine/load_rules.py` (idempotent `INSERT OR IGNORE`)
- Completed T2.4 - `inference/protocol_engine/engine.py` (ProtocolEngine, confidence floor, disclaimer)
- Completed T2.5 - `inference/protocol_engine/test_engine.py` (9 tests)
- Completed T2.6 - All 9 engine tests pass
- Completed T3.1 - `inference/function_calling/schemas.py` (Pydantic v2, disclaimer validator)
- Completed T3.2 - `inference/function_calling/handler.py` (tool call extraction + fallback)
- Completed T3.3 - `inference/function_calling/test_handler.py` (12 tests)
- Completed T3.4 - All 12 handler tests pass
- Completed T2.7/T3.5 - Verifier approved (21 total tests passing)
- Completed T4.1 - `training/data/README.md` (data pipeline docs, 8 sections)
- Completed T4.2 - `training/scripts/translate_dataset.py` (Gemma 4 31B automated translation)
- Completed T4.3 - `training/data/fixtures/` (30 fixture entries: 10 each triage/derm/maternal)
- Completed T4.4 - `training/scripts/format_training_data.py` (Gemma 4 chat format, 80/10/10 split)
- Completed T4.5 - Format pipeline validated (val non-empty, disclaimer enforced, correct 4-turn format)
- Completed T5.1 - `training/scripts/train_lora.py` (parameterized Unsloth QLoRA trainer, 3 seeds)
- Completed T5.2 - `training/configs/lora_triage.yaml`, `lora_derm.yaml`, `lora_maternal.yaml`
- Completed T6.1 - `training/scripts/train_router.py` (KalavaiRouter: `nn.Linear` + `softmax`)
- Completed T6.2 - `training/scripts/export_gguf.py` + `training/configs/router.yaml`
- Completed T6.3 - Week 2 training scripts verified (LoRA + router + GGUF export, all guarded imports)
- Completed T8.1 - `inference/protocol_engine/logger.py` (encrypted SQLite interaction logger)
- Completed T8.2 - `inference/protocol_engine/test_logger.py` (12 tests)
- Completed T8.3 - All 12 logger tests pass
- Completed T8.4 - Logger + full inference suite verified (33 total tests green)
- Completed T7.1 - Android JNI scaffolding: `CMakeLists.txt`, `marunthagam_jni.cpp`
- Completed T7.2 - Android Kotlin inference layer: `LlamaWrapper.kt`, `TriageEngine.kt`, `TriageLogDao.kt`
- Completed T7.3 - Android UI: `activity_triage.xml` (Material 3, disclaimer always visible)
- Completed T7.4 - Android Gradle project files: root + app `build.gradle.kts`, `settings.gradle.kts`
- Completed T7.5 - `android/app/src/main/AndroidManifest.xml`
- Completed T7.6 - `android/MainActivity.kt` (ViewBinding, TriageEngine integration, coroutines)
- Completed T7.7 - `strings.xml`, `arrays.xml` (Tamil UI strings + age groups)
- Completed T7.8 - Android resources and release config
- Completed T7.9 - Android Gradle wrapper added (`gradlew`, `gradlew.bat`, wrapper JAR)
- Completed T7.10 - Android JNI bridge updated for current `llama.cpp` API
- Completed T7.11 - Android debug build verified (`.\\gradlew.bat assembleDebug`)
- Completed T9.1 - `eval/scripts/run_eval.py` (full suite, mock + real, 3-seed aggregation)
- Completed T9.2 - `eval/scripts/eval_triage.py` (per-class P/R/F1, RED recall, bootstrap CI)
- Completed T9.3 - `eval/scripts/eval_safety.py` (100 adversarial prompts, refusal rate)
- Completed T9.4 - `eval/scripts/eval_latency.py` (TTFT + throughput, phone/workstation targets)
- Completed T9.5 - `eval/data/adversarial_prompts.json` (100 prompts, 5 categories, Tamil+English)
- Completed T9.6 - `eval/scripts/ablation_rank.py` (rank 4/8/16/32/64 comparison, break-even)
- Completed T10.1 - Dashboard React+D3 shell: `package.json`, `vite.config.js`, `App.jsx`, `main.jsx`
- Completed T10.2 - Dashboard components: `TriageHeatmap`, `TriageTrendChart`, `AlertPanel`, `StatsSummary`
- Completed T10.3 - Dashboard pages: `Overview`, `MapView`, `AlertsView`, `TrendsView`
- Completed T10.4 - Dashboard i18n: `ta.js`, `mockData.js`
- Completed T10.5 - Dashboard production build verified (`npm run build`)
- Completed T11.1 - `README.md` submission doc
- Completed T11.2 - `docs/architecture.md`
- Completed T11.3 - `docs/protocol_spec.md`
- Completed `eval/notebooks/results_analysis.ipynb`

---

## Remaining Work (Blocking Submission)

- Run LoRA training - `python train_lora.py --config lora_triage.yaml --seed 42/137/256`
- Run KALAVAI router training on real embeddings
- Export GGUF - `python export_gguf.py --checkpoint outputs/fused-best`
- Run real eval - `python run_eval.py --model models/fused-E4B-Q4_K_M.gguf --seeds 42,137,256`
- Fill eval table in `README.md` with real numbers
- Record demo video
- Publish Hugging Face weights
- Submit to Kaggle by 2026-05-18
- Decide how Android should source `llama.cpp` long-term: vendored tree, submodule, or bootstrap step

---

## Blockers

- GGUF weights not yet trained - all real evals and end-to-end demo quality remain blocked on actual model artifacts
- Android runtime still needs a real GGUF file on device - build is verified, runtime demo still depends on shipping model

---

## MVP Gate (Must-Ship Minimum)

- [ ] LoRA-Triage fine-tuned (minimum one specialist)
- [x] `triage_classify()` function calling working
- [x] Protocol grounding engine (SQLite + IMNCI)
- [ ] Evaluation: F1/P/R per class, 3 seeds
- [x] Dashboard build verified
- [x] Android APK build verified
- [ ] CLI demo on `llama.cpp` with real fused model
- [x] Technical write-up
- [ ] Demo video
- [ ] GitHub + Hugging Face weights

---

## Risk Flags

- [ ] E4B Tamil quality after fine-tuning (Medium likelihood, High impact) - Mitigation: 26B-A4B demo fallback
- [ ] Native audio in GGUF (High likelihood, Medium impact) - Mitigation: Whisper-small-Tamil offline
- [ ] KALAVAI fusion gain (Medium likelihood, High impact) - Mitigation: report result honestly
- [ ] Android dependency management for `llama.cpp` (Medium likelihood, Medium impact) - Mitigation: formalize vendoring/submodule approach before wider onboarding
