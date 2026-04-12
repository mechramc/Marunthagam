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

### [2026-04-12 04:38] Checkpoint 5 — Task 9+7+11: eval suite completion, Android scaffold, README + docs

**Commit:** 26a1736
**Tasks completed:** T9.2–T9.4 (eval_safety, eval_latency, adversarial_prompts), T7.4–T7.8 (Android Gradle + Manifest + MainActivity), T11.1–T11.3 (README, architecture.md, protocol_spec.md)
**Tests passing:** 33 (unchanged — new code is scripts/docs, no new pytest suites)
**Verifier:** ✅ Approved — eval scripts run clean in mock mode, Android namespace consistency confirmed, README structure matches spec
**Files:** eval_safety.py, eval_latency.py, adversarial_prompts.json (100 prompts), build.gradle.kts (app+root), settings.gradle.kts, gradle.properties, gradle-wrapper.properties, AndroidManifest.xml, MainActivity.kt, strings.xml, arrays.xml, adaptive icons, README.md, docs/architecture.md, docs/protocol_spec.md
**Notes:** Latency mock reports FAIL on workstation targets (expected — phone-tier TTFT ~2s vs workstation target <1s). Safety mock reports 98/100 (2 intentional slip-throughs for realistic testing). Android needs: llama.cpp source in cpp/llama.cpp/, NDK r26+, local.properties.

### [2026-04-12 03:00] Checkpoint 4 — Week 3 build: eval scripts, Android JNI, React dashboard, results notebook

**Commit:** 3a00b7a
**Tasks completed:** T7.1–T7.3 (Android JNI + Kotlin + layout), T9.1 (run_eval.py + eval_triage.py + ablation_rank.py), T10.1–T10.4 (full React+D3 dashboard), notebook
**Tests passing:** 33 (all prior tests green)
**Verifier:** ✅ Approved — eval scripts run mock mode, privacy fix applied (kv_cache_clear on prefill failure), disclaimer moved outside hidden CardView, D3 null guards added, Tamil i18n centralized
**Files:** eval/scripts/ (3 files), eval/notebooks/results_analysis.ipynb, android JNI (6 files), dashboard (21 files), training/data/formatted_test/ (9 JSONL splits)
**Notes:** Critical privacy fix: added kv_cache_clear() on JNI prefill error path to prevent cross-patient context bleed. Dashboard uses Tamil Nadu geohash prefixes tf7/tf8.

### [2026-04-10 23:12] Checkpoint 3 — Week 2 training scripts + inference logger

**Tasks completed:** T5.1–T5.2, T6.1–T6.3, T8.1–T8.4
**Tests passing:** 33 (9 engine + 12 logger + 12 handler)
**Verifier:** ✅ Approved — all guarded imports confirmed, KalavaiRouter architecture correct, 12 logger tests pass, PII guarantee verified
**Files:** train_lora.py, lora_*.yaml (×3), train_router.py, router.yaml, export_gguf.py, logger.py, test_logger.py
**Notes:** embed_text in train_router.py uses random-normal stub (documented TODO); router.yaml embedding_dim=768 placeholder (real E4B dim=2560, will update before Week 4 eval). Logger test_logger.py now covers get_pending_sync and full PII forbidden column set.

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
