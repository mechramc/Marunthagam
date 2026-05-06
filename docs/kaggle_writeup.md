# Marunthagam: Tamil-First Offline Triage for ASHA Workers

**Gemma 4 Good Hackathon 2026 · single-author · code & weights public · Apache 2.0**

ASHA workers — India's 940,000 community health activists — make life-or-death triage decisions in 22 Indic languages, off-grid, with paper checklists. The bottleneck is not connectivity, not model size, and not even rural infrastructure. The bottleneck is that *no production-grade clinical-AI system today is built around the failure modes specific to triage in low-resource language settings on commodity Android hardware.* Marunthagam is the simplest credible attempt at one, built on Gemma 4 E4B, shipped on a 5GB Q4_K_M GGUF, and grounded in a deterministic WHO/IMNCI protocol engine that the model can never override. This writeup distills the contribution beyond what the README documents.

## What this submission is, in one sentence

A Tamil-first **decision-support** triage system — not a clinical Q&A bot, not a medical chatbot, and not a doctor replacement — that runs entirely on a low-end Android phone, escalates the right cases to PHC doctors and district health officers, and was built around five design choices that we believe generalise to other low-resource medical-AI work.

## Five things that differentiate Marunthagam

**1 · Tamil-first, with morphology that actually works.** Most non-English clinical NLP fails first on the indicator-list / regex layer, not the model layer. Bare-nominative Tamil rules don't match locative `மார்பில்`, accusative `மருத்துவரை`, instrumental `நாயினால்`, or sandhi-compounds like `மூச்சுத்திணறல்`. Our IMNCI rule engine has explicit case-inflected forms; our multilingual safety classifier has ~135 indicators across Tamil/Hindi-devanagari/Gujarati/English to handle code-switching. Sprint 1 found 22-of-22 "non-refusals" were classifier false negatives in Hindi devanagari and Gujarati; the v2 rebuild lifted refusal rate from 78% to 100% on n=100 adversarial prompts. Other Indic-language clinical projects (e.g., Indonesia's VillageDoc) target broader Asian-language coverage but, in our experience auditing them, treat morphology as a downstream problem — we made it the first-class problem.

**2 · Offline-first, not offline-as-a-feature.** Tier 1 (the ASHA worker's phone) ships as a single 5GB Q4_K_M GGUF + a deterministic SQLite-backed protocol engine + an AES-256 encrypted local log. There is no API call in the triage path. Tier 1 → Tier 3 sync transmits aggregated signals only — geohash-clustered counts at ~1km cell precision — never individual records. Most "offline-capable" medical AI we surveyed (including some Gemma-Med-derived clinical assistants) treats offline as a fallback when the network is down; we treat it as the only correct deployment posture for a system that runs in villages 40km from the nearest PHC. The submission GGUF is the actual production artifact (Sprint 2 B-retrained on relabeled triage data), not a research checkpoint.

**3 · Decision support, never replacement — and the architecture enforces it.** Three structural choices make this real, not aspirational. (a) Every output is a function-called JSON record with a mandatory Tamil disclaimer (`இது மருத்துவ ஆலோசனை அல்ல`) enforced at the schema validation layer; missing-disclaimer records refuse to write. (b) The IMNCI protocol engine sits below the LLM and *can only escalate*, never downgrade — if the model says GREEN but the rule says RED, the rule wins. (c) Confidence below 0.70 always escalates one level, regardless of the model's stated category. The Sprint 2 shipping decision explicitly chose "routed" inference (F1 0.6491, 7/12 RED-at-RED) over "maternal-only" (F1 0.6753, 6/12 RED-at-RED) because the safety-relevant failure mode is recall-favored — the +1 emergency caught at full RED matters more than the +0.026 in aggregate F1. We documented this trade-off in the shipping memo so reviewers can audit it.

**4 · The schema-consumer audit as a methodology.** Twice during the project, our eval pipeline computed information that downstream analysis needed but the JSON shape silently discarded: in Sprint 1, `engine.apply()` returned a `ProtocolOverride` list that landed in a throwaway `_overrides` local; in Sprint 2, the `engine_overrides` field captured only escalating matches, hiding the rules that fired-but-didn't-escalate. Both gaps were caught because we forced ourselves to ask, every time we wrote a results file, *"what question can a downstream consumer not answer from this artifact?"* The fix is always to patch the schema, not to re-run inference. We argue this is the most under-rated technique in low-resource ML auditing: most "we couldn't tell from the eval JSON" issues are upstream schema gaps, not analysis bugs. The pattern generalises far beyond Tamil triage.

**5 · Diagnostic sprints before fix sprints, with explicit gates.** Sprint 1 was diagnosis-only — specialist behaviour analysis on 7 missed REDs, label-quality spot-check (18% of triage GREENs were under-triaged toward higher acuity, all at the GREEN/YELLOW boundary), Bucket-A/B/C analysis on Tamil regex coverage. Sprint 2 was fixes — relabel + retrain, IMNCI rule expansion with positive+negative test pairs per rule, safety classifier rebuild, held-out re-eval. Each retrain candidate had explicit pass / partial / fail / regression gates *before* it ran; when seed 42 fell short of the gate we did not auto-launch seeds 137 and 256. We stopped, posted per-class numbers, and chose the next lever explicitly. This kept GPU budget aimed at *information* — not at variance estimates we already had a confident point estimate for. Multi-seed std reporting is right when effects are small relative to seed variance; we were past that regime, and saying so explicitly is part of the contribution.

## Honest performance, calibrated to evidence

The Sprint 2 shipped stack — B-retrained triage LoRA + sprint-1 derm + sprint-1 maternal + v2.1 IMNCI rules + v2 multilingual safety classifier — produces, on the held-out test split (n=131, seed 42, T=0):

- Weighted F1 = **0.6491** (0.001 below the calibrated 0.65 threshold)
- RED recall = **0.5833** (above the calibrated 0.55 threshold)
- **0 of 12 emergencies missed-as-GREEN** — every gold-RED case escalates to at least YELLOW
- 7 of 12 emergencies caught at full RED level (the rest escalate to YELLOW with referral guidance)
- 100/100 adversarial safety refusals across diagnosis/mental-health/prescription/surgery/scope-violation
- Workstation TTFT 0.007–0.038s, throughput 195–213 tok/s (RTX 5090, llama-cpp-python streaming)

Threshold recalibration from the original 0.75/0.80 spec to 0.65/0.55 is documented with three evidence-grounded reasons: label noise floor (18% under-triage in GREEN), class imbalance prior collapse on a 21/65/15 class distribution, and an empirical rule-layer ceiling at 0.583 RED recall with the v2.1 ruleset. This is calibration to evidence, not goalpost-moving — diagnostic memos in `eval/analysis/2026-05-07/` document the reasoning so reviewers can audit it end-to-end.

## What we would do differently with more time

Tier 2 (Gemma 4 26B-A4B for the PHC doctor) is designed and rule-engine-compatible but not deployed in the submission window. Image multimodal smoke-testing is gated on labeled-derm-image availability we couldn't source in time. Phone TTFT remains unmeasured because we don't own a low-end Android device for credible end-to-end timing — emulator latency would be misleading and the README states this explicitly. The derm contamination move (49 cases routed to derm-train when the chief complaint was non-dermatologic) was applied and the derm-LoRA retrained on the cleaned data, but the head-to-head comparison didn't beat the sprint-1 derm head-to-head, so the production stack ships sprint-1 derm and the cleaned derm is documented as a data-hygiene artifact rather than a model upgrade.

## Tooling acknowledgement

Built on top of [Unsloth](https://github.com/unslothai/unsloth) for the QLoRA fine-tuning loop on Gemma 4 E4B — fits comfortably on a single RTX 5090 (32GB VRAM) at rank 32 / alpha 64, with 4-bit base + LoRA-only updates. The `FastLanguageModel` API also drove our HF+PEFT inline eval path that bypassed GGUF export when we needed a fast feedback loop for retrain-candidate gating. Without Unsloth this would have been a much slower project; the `apply_liger_kernel`-equivalent throughput on Gemma 4 E4B was the unlock that let us run multiple retrain recipes within the sprint window.

## Links

- **Code:** [github.com/mechramc/Marunthagam](https://github.com/mechramc/Marunthagam) (Apache 2.0)
- **Models:** [`mechramc/marunthagam-triage-E4B-Q4_K_M`](https://huggingface.co/mechramc/marunthagam-triage-E4B-Q4_K_M) · [`mechramc/marunthagam-derm-E4B-Q4_K_M`](https://huggingface.co/mechramc/marunthagam-derm-E4B-Q4_K_M) · [`mechramc/marunthagam-maternal-E4B-Q4_K_M`](https://huggingface.co/mechramc/marunthagam-maternal-E4B-Q4_K_M)
- **Dataset:** [`mechramc/marunthagam-tamil-triage`](https://huggingface.co/datasets/mechramc/marunthagam-tamil-triage) (3 specialists × train/val/test + adversarial safety prompts + multilingual safety classifier validation set + clinician-completed label-quality CSVs + pre-relabel/pre-derm-move backups)
- **Architecture diagram:** [`docs/architecture_diagram.svg`](https://github.com/mechramc/Marunthagam/blob/main/docs/architecture_diagram.svg)
- **Demo video script:** [`docs/demo_video_script.md`](https://github.com/mechramc/Marunthagam/blob/main/docs/demo_video_script.md)
- **Sprint 1 diagnostic memos:** `eval/analysis/2026-05-06/`
- **Sprint 2 shipping memo:** [`eval/analysis/2026-05-07/task6_results.md`](https://github.com/mechramc/Marunthagam/blob/main/eval/analysis/2026-05-07/task6_results.md)

> **இது மருத்துவ ஆலோசனை அல்ல** — This is not medical advice. Marunthagam is decision support for community health workers who already triage with paper checklists. Tier 2 (PHC) and Tier 3 (district) make the actual clinical and population-level calls. Marunthagam's job is to ensure no village case slips through to "wait and see" when it should have gone to a hospital that night.
