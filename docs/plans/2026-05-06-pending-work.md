# Marunthagam — Pending Work

**Date opened:** 2026-05-06
**Hackathon deadline:** 2026-05-18 (12 days)
**Scope discipline:** No skipping items below without explicit user approval. Each item names what was previously skipped and what "done" looks like.

---

## A. Evaluation gaps (skipped during the 2026-05-05 sprint)

These were marked "Not run / Not measured" in `README.md`. The headline F1 = 0.8174 / RED recall = 0.9231 numbers ship; the items below are still required to honour the eval scope in `CLAUDE.md` and `docs/architecture.md`.

### A1. Safety refusal rate (target: 100% on 100 adversarial prompts)
- Script: `eval/scripts/eval_safety.py` (already written, never invoked).
- Data: `eval/data/adversarial_prompts.json` (100 prompts in 5 categories, Tamil + English).
- Pre-req: same per-specialist GGUFs `run_eval.py` uses; same `_llama_cpp_setup` DLL bridge.
- Likely fix needed: `eval_safety.py` currently calls `llama-cli` subprocess (same architectural bug we hit in `run_eval.py`). Patch to use `llama_cpp.Llama` Python binding with the same prompt-priming + JSON-output scheme.
- Done = `eval/results/safety_*.json` written; refusal rate reported per category; cite in README.

### A2. Workstation TTFT + throughput
- Script: `eval/scripts/eval_latency.py` (already written, never invoked).
- Targets: TTFT < 1s, > 30 tok/s on workstation; TTFT < 3s, > 8 tok/s on phone.
- Same llama-cli → llama_cpp.Llama patch needed.
- Done = `eval/results/latency_*.json`; numbers in README. Phone TTFT may require Android deployment; if so, mark explicitly as deferred to A6.

### A3. Ablation — KALAVAI fusion gain (target: +3% over best specialist)
- Compare current routed eval (Weighted F1 0.8174) against:
  - Each single specialist GGUF used for ALL 50 cases (3 single-model runs).
  - Pick the highest, compare to fused.
- Script: re-run `run_eval.py --model <single_specialist_gguf>` 3× and tabulate.
- Done = README has a fusion-gain table with mean ± std for each single + fused.

### A4. Per-domain specialist gain (target: +5% over generalist)
- Need a "generalist" baseline. Two options (pick at session start):
  - **Option A:** base Gemma 4 E4B (no LoRA) → straightforward, defensible.
  - **Option B:** train a fourth LoRA on the mixed corpus (all 3 specialists merged) → faithful to the spec's "generalist" framing, costs ~3 min training.
- Eval the baseline on each specialist's fixture cases; compare to that specialist's per-case F1.
- Done = README has a per-domain-gain table with one row per specialist (baseline F1 vs specialist F1).

### A5. Tamil fluency (chrF++ target: > 0.60)
- The current eval prompt forces a JSON-only response — no `next_steps_tamil` text to score.
- Need a second eval pass that asks the model to produce `next_steps_tamil`, then computes chrF++ against the gold next-steps in the fixtures.
- `sacrebleu` (already a training dep) supports chrF++.
- Done = chrF++ score per specialist + overall; numbers in README.

### A6. Phone TTFT (target: TTFT < 3s, > 8 tok/s)
- Requires Android device + APK install or `llama-cli` running on Termux/ARM.
- Lowest-friction path: ship the triage GGUF onto a connected device via `adb push`, run the existing Android app's inference path, log timing. Android build was already verified 2026-04-14 but the `llama.cpp` JNI may have drifted.
- Done = phone latency numbers in README, or explicit "deferred — no device available".

### A7. LoRA rank ablation
- Script: `eval/scripts/ablation_rank.py` (already written, never invoked).
- Trains LoRAs at ranks 4 / 8 / 16 / 32 / 64 and reports break-even point.
- Costs: 5 ranks × 1 specialist (likely triage) × ~3 min/run = ~15 min compute.
- Done = ablation chart in `eval/notebooks/results_analysis.ipynb` and a paragraph in README.

### A8. Use the held-out test split, not just fixtures
- Current eval runs on `training/data/fixtures/{specialist}_reviewed.jsonl` (10 rows × 3 specialists + 20 baseline = 50 cases).
- The 80/10/10 test split (`training/data/formatted/{specialist}/test.jsonl`, **131 rows total**) has never been evaluated — that's our actual held-out test set, and the right denominator for the headline F1.
- Done = re-run `run_eval.py` against the test split; report two F1 numbers (fixtures + test split) so the writeup is honest about both.

---

## B. Architectural polish

### B1. KALAVAI router triage-class collapse
- Current val accuracy 0.615 (above 0.33 random) but **triage class P/R/F1 = 0.00** — never gets predicted.
- Root cause hypothesis: triage is the catch-all class with no distinctive vocabulary; mean-pool of E4B last hidden state gives derm/maternal a more separable signature.
- Things to try (in order, time-boxed):
  1. CLS-token / first-token pooling instead of mean-pool.
  2. Class weights in the cross-entropy loss to upweight triage.
  3. Hard-negative mining: explicitly mine derm-flavoured-but-actually-triage cases from the data.
  4. A small MLP (Linear → ReLU → Linear) instead of a single linear layer.
- Done = router val_acc > 0.75 AND triage F1 > 0.5, OR documented decision to ship with the current router.

### B2. Protocol engine usage in Android inference path
- The eval now wires `inference/protocol_engine` post-LLM. Check that the Android side does the same — `android/.../TriageEngine.kt` should call into the protocol engine before showing the user a level.
- Currently the Android JNI bridge talks to the model only; protocol engine is Python-only.
- Options: port engine to Kotlin, OR ship a single rules JSON the Kotlin side parses, OR run the engine via embedded Python (Chaquopy).
- Done = decision logged + (if going with port) Kotlin version of `engine.py` with passing tests.

### B3. Derm bucket leakage in source acquisition
- `acquire_sources.py` keyword regex put oral/ENT cases ("sore tongue/gums", "ear infection") into the derm bucket because the answer text mentioned skin-related words.
- Fixes (pick one): tighten regex to question-text only, OR add a specialist-LLM re-bucket pass on the 1500 raw rows.
- Done = re-acquire + re-translate + re-label + re-train derm only (the rest is unchanged).

### B4. Hindi devanagari leakage in Tamil translations
- ~1 in 100 translated rows contains stray Hindi characters (saw "ठीक ऊपर" in row 2 of triage smoke).
- Lightweight fix: post-translation validator that flags any non-Tamil/non-Latin script and re-prompts the 31B with stricter instruction.
- Done = validator added to `translate_dataset.py`; rerun on affected files; re-train.

---

## C. Submission deliverables

### C1. HuggingFace weights upload (3 GGUFs)
- Script: `training/scripts/upload_to_hf.py` (ready, supports `--public`).
- **Blocked on user approval for public publication** (harness denied unprompted public push 2026-05-05). Decide public vs private at session start.
- Done = three model cards live at `mechramc/marunthagam-{triage,derm,maternal}-E4B-Q4_K_M`, linked from README.

### C2. Demo video
- Not started. Spec calls for a recording showing the offline phone-side experience (CLI demo acceptable per CLAUDE.md MVP gate).
- Easiest first cut: screencap of `llama-cli` (or our Python binding) responding to a sample Tamil patient query, narrating the JSON → protocol-engine → final triage flow.
- Done = `docs/demo.mp4` or hosted link in README.

### C3. README submission polish
- Eval table is now real. Need: a clear quickstart, architecture diagram, link to HF weights, link to demo video, link to dataset description.
- Done = README reads like a hackathon submission, not an in-progress spec.

### C4. Status doc refresh (`docs/status.md`)
- Currently dated 2026-04-14 — wildly stale. Says "92% complete; pending: LoRA training, GGUF weights, real evals, demo video".
- Today (2026-05-05) we landed everything except demo video, HF upload, and the items in section A above.
- Done = status.md reflects 2026-05-06 reality with concrete checklist of what remains for May 18.

### C5. Architecture doc refresh (`docs/architecture.md`)
- Confirm it describes the shipped pipeline: ChatDoctor → 31B translate → 31B label → format → QLoRA E4B × 3 → GGUF Q4_K_M → router (E4B mean-pool) → protocol engine.
- Done = doc matches code; diagram present.

### C6. Kaggle submission
- Final submission to Gemma 4 Good Hackathon by 2026-05-18.
- Done = submission registered with link to GitHub repo + HF weights + demo video.

---

## D. Quality / hygiene

### D1. CPU thermal mitigation (background)
- Killed 7 idle Codex processes 2026-05-05; iCUE / ArmouryCrate / Adobe / Blitz / Microsoft Copilot still running. Not critical, just noise.
- Done = if temps creep, revisit with HWiNFO64 baseline.

### D2. wandb logging
- Currently disabled (`WANDB_DISABLED=true`) because the wandb service-startup polling races on Windows. Acceptable for now since per-run logs land in `training/logs/`.
- If we re-train (router, generalist baseline, ablation runs), consider re-enabling with `WANDB_MODE=offline` so logs are local-first and can be synced later.

### D3. Stop committing `.claude/settings.local.json` accidentally
- It's project-local Claude permission state. Either gitignore it project-wide, or let the harness keep mutating it.
- Done = decision logged.

---

## Order-of-operations recommendation for the next session

Time-box estimates assume the GPU and llama-cpp-python stack from 2026-05-05 are still warm.

1. **A8** (held-out test eval, ~5 min) — confirms the headline number on the right denominator.
2. **A1 + A2** (safety + latency, ~30 min) — easy wins; both scripts exist.
3. **A3** (fusion ablation, ~10 min) — re-runs of `run_eval.py` only.
4. **A4** (per-domain gain, ~15 min for option A / ~20 min for option B).
5. **A5** (Tamil fluency chrF++, ~30 min) — requires a second eval pass.
6. **A7** (LoRA rank ablation, ~15 min) — last optional quantitative ablation.
7. **B1** (router improvement) — only if A4 leaves time and the router is on the demo critical path.
8. **C1** (HF upload, ~30 min once approved) — needs user "go public" call.
9. **C2** (demo video) — once the system above is fully numbers-backed.
10. **C3 / C4 / C5** (doc polish) — last 90 minutes.
11. **C6** (Kaggle submission) — final step.

Items deliberately *not* on the critical path: B2 (Android protocol port), B3 (derm rebucket), B4 (Tamil-only validator), A6 (phone TTFT). These are nice-to-haves, surface after A and C are done.
