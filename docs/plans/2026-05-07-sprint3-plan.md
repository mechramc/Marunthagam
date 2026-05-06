# Sprint 3 — Submission prep + Sprint 3 deferred items

**Date opened:** 2026-05-07
**Hackathon deadline:** 2026-05-18 (~11 days)
**Premise:** Sprint 2 closed with the production stack shipped (routed inference, B-retrained triage + sprint-1 derm + sprint-1 maternal + v2.1 IMNCI rules + v2 multilingual safety classifier). All Sprint 2 commits pushed; smoke test passed 25/25. This sprint covers everything between "submission-ready" and "submitted," plus the Sprint 3 deferred work the team chose not to defer further.

---

## Day-by-day shape

### Days 1–2: cheap wins + submission-prep groundwork (parallel)

**Cheap wins** (low blast radius, real ROI):
- **S3-CW-1**  Derm contamination move + derm-LoRA retrain. Apply `apply_derm_contamination.py` (49 MOVE + 1 RELABEL_ONLY, all dry-run-clean). Retrain derm-LoRA on cleaned data (~5 min). Eval on held-out, isolate per-domain delta. Estimated **~1.5 h**.
- **S3-CW-2**  YELLOW/RED label-quality spotcheck. Generate 40-row CSV (20 YELLOW + 20 RED from triage train). User reviews. Apply diffs. Estimated **~1 h building + user review time + 0.5 h applying**.
- **S3-CW-3**  chrF++ replacement metric. Use `paraphrase-multilingual-mpnet-base-v2` cosine similarity on the existing 131 chrF eval rows. Report side-by-side with chrF++. Estimated **~1.5 h**.

**Submission-prep groundwork** (parallel CPU/IO):
- **S3-SUB-2**  GGUF export of B-retrained triage adapter. Base model may already be cached from rank-16 export. ~30–45 min if cached, ~1.5 h if not.
- **S3-SUB-3**  Build `inference/cli_demo.py` (referenced in README, missing on disk). ~1.5 h.
- **S3-AND-1**  Android emulator setup. Already past the install step (emulator + system image landed). Create AVD, install APK, push GGUF, validate end-to-end UI flow. ~1 h once GGUF is exported.

### Days 3–5: medium lifts

- **S3-MED-1**  Tier 3 dashboard data-integration. Wire `interaction_log` aggregation → React/D3. ~1 day.
- **S3-MED-3**  Multi-seed Task 6 (seeds 137 + 256). Background work; gives 3-seed std bars on the headline number. ~30 min training × 2 seeds + ~12 min HF+PEFT eval × 2 = ~1.5 h, fits in background.
- **S3-MED-2**  Image multimodal smoke test (only if 20–40 labeled derm images can be sourced quickly). Use existing mmproj GGUF + llama-cpp-python multimodal API. **Gated on test image availability** — skipped otherwise.

### Days 6–8: Tier 2 attempt with Day-8 cancel point

- **S3-BIG-1**  Tier 2 (Gemma 4 26B-A4B clinic model). Train rank-8 LoRA (smaller to fit memory budget on 32GB RTX 5090). GGUF export. Held-out eval. **Hard cancel point at Day 8**: if Tier 2 isn't beating E4B held-out F1 by Day 8 EOD, drop from submission and frame as "designed but not deployed in time" rather than shipping numbers that don't strengthen the story.

### Days 9–11: submission

- **S3-FIN-1**  Demo video. Record once Tier 2 outcome is known so video reflects shipped reality not aspiration. CLI demo + emulator UI flow + dashboard if S3-MED-1 landed. No phone-tier latency claims.
- **S3-FIN-2**  README final pass. Reconcile with what actually shipped. Add HF link.
- **S3-FIN-3**  Kaggle / hackathon submission registration. Final step.
- **S3-SUB-1**  HuggingFace weights upload — BLOCKED on user "go public" approval. Once approved, ~30 min.
- **Buffer:** 1 day for things going wrong.

### Skipped explicitly

- **Audio input on Tier 1.** No Android device available; emulator can't credibly demo audio either. Out of scope; documented in README as Sprint 4+ work.
- **Phone TTFT claim.** Emulator latency would be misleading. README already says "deferred — no physical device." Stays that way.

---

## Cancellation criteria

If any of the cheap wins fails its acceptance criterion, STOP that thread and report — same gate-driven discipline as Sprints 1+2. Specifically:

- **S3-CW-1** fails if derm-LoRA on cleaned data scores LOWER held-out F1 than the existing sprint-1 derm-LoRA. That would mean the contamination removal hurt more than it helped (extremely unlikely but possible). Action: revert from backup.
- **S3-CW-3** fails if the sentence-transformer scores produce no useful signal (e.g., everything > 0.95 or everything < 0.20). Action: report and stay with chrF++ caveat in README.
- **S3-BIG-1** fails Day 8 if Tier 2 26B held-out F1 ≤ E4B held-out F1 (0.6491). Action: drop from submission per spec above.

---

## Acceptance for "submission-ready"

By Day 11 EOD:

- [ ] HF weights uploaded (3 LoRAs + GGUFs publicly accessible)
- [ ] B-retrained triage GGUF exists in `training/models/`
- [ ] CLI demo script runs end-to-end (`python inference/cli_demo.py` produces valid triage output for a Tamil query)
- [ ] Android emulator validated end-to-end (APK + GGUF + triage flow), screenshots in repo
- [ ] `docs/status.md` and `docs/architecture.md` reflect Sprint 2 + Sprint 3 reality
- [ ] Demo video published
- [ ] README has HF + video links
- [ ] Kaggle/hackathon submission registered

Sprint 3 deferred items NOT on the critical path (chrF replacement, multi-seed Task 6, Tier 3 dashboard, Tier 2 26B, image multimodal) are bonuses — strengthen the submission if they land, but don't block submission if they don't.
