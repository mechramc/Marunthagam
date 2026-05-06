# Kaggle submission dry-run checklist

> Manual checklist for the actual submission. Each item is a thing only the user (Kaggle account holder) can verify; this checklist exists so nothing slips through on submission day.

**Submission deadline:** 2026-05-18

---

## Pre-flight (do this once)

- [ ] Logged into kaggle.com under the account that will own the submission
- [ ] Hackathon competition page confirms the registration is active
- [ ] Submission window is currently open (some hackathons close for grading)

## Artifact verification (run through these in order)

- [ ] **Public GitHub repo** — open `github.com/mechramc/Marunthagam` in an incognito window and confirm:
  - [ ] README renders without any `[TODO ...]` placeholders
  - [ ] Architecture SVG renders inline (`docs/architecture_diagram.svg`)
  - [ ] LICENSE shows Apache 2.0
  - [ ] `v1.0-hackathon-submission` tag exists (Releases tab) — if missing locally was fine but a remote tag is required for "tagged release" judging signal; push with `git push origin v1.0-hackathon-submission`
  - [ ] No accidental large files in the working tree (anything >50MB should be HF-hosted, not git-hosted)

- [ ] **HuggingFace model repos** — open each in an incognito window:
  - [ ] `huggingface.co/mechramc/marunthagam-triage-E4B-Q4_K_M` — model card renders, files tab lists adapter + GGUF + mmproj
  - [ ] `huggingface.co/mechramc/marunthagam-derm-E4B-Q4_K_M` — same
  - [ ] `huggingface.co/mechramc/marunthagam-maternal-E4B-Q4_K_M` — same
  - [ ] (Optional follow-up) Sprint 2 B-retrained triage GGUF pushed to triage repo replacing the sprint-1 GGUF — `training/models/triage-B-E4B-Q4_K_M_gguf/gemma-4-e4b-it.Q4_K_M.gguf` is the file

- [ ] **HuggingFace dataset repo** — `huggingface.co/datasets/mechramc/marunthagam-tamil-triage`:
  - [ ] Dataset card renders with full schema documentation
  - [ ] All splits visible (3 specialists × train/val/test)
  - [ ] Pre-relabel / pre-derm-move backups present
  - [ ] Adversarial safety prompts + classifier validation set present
  - [ ] Clinician-completed label-quality CSVs present

- [ ] **Demo video** — link works in incognito and the video plays
  - [ ] Uploaded (YouTube unlisted is fine; use the script in `docs/demo_video_script.md`)
  - [ ] Caption / description includes GitHub + HF links
  - [ ] README's `## Demo` section links to the video
  - [ ] Video shows the explicit "emulator, not phone-tier latency claim" caveat per the script

## Kaggle notebook (the actual submission asset)

The Kaggle competition submission is typically a **public Kaggle notebook**. Copy the structure below into a fresh notebook:

- [ ] **Title cell:** "Marunthagam: Tamil-First Offline Triage for ASHA Workers"
- [ ] **Markdown cell 1 — Writeup:** paste `docs/kaggle_writeup.md` verbatim (this is the distilled positioning piece — five differentiators, honest performance, Unsloth acknowledgement, links)
- [ ] **Markdown cell 2 — Architecture:** embed the architecture SVG (Kaggle accepts inline images via the editor's image upload)
- [ ] **Code cell 1 — Reproducibility setup:**
  ```python
  !pip install -q huggingface_hub
  from huggingface_hub import snapshot_download
  snapshot_download("mechramc/marunthagam-triage-E4B-Q4_K_M", local_dir="./triage")
  ```
- [ ] **Code cell 2 — Inference demo:** the three preset Tamil queries from `inference/cli_demo.py`
  (one GREEN, one YELLOW, one RED with cardiac-pattern engine override)
- [ ] **Code cell 3 — Eval reproduction:** point at the held-out test split with the n=131 routed Task 6 numbers
- [ ] **Markdown cell N — Links:** GitHub, HF (4 repos), demo video

## Submission form fields

- [ ] **Title** matches "Marunthagam: Tamil-First Offline Triage for ASHA Workers"
- [ ] **Track / category** matches the hackathon's expected category (likely "best use of Gemma 4" or similar)
- [ ] **Team name** matches your registration
- [ ] **GitHub repo URL** filled in
- [ ] **HuggingFace links** all four repos listed (or at least the umbrella user URL)
- [ ] **Notebook link** filled in (the public Kaggle notebook)
- [ ] **Demo video URL** filled in
- [ ] **Disclaimer / safety statement** acknowledged if the submission form has one
- [ ] **Compute / hardware used** correctly stated (RTX 5090 32GB for training; Gemma 4 E4B base from Unsloth)

## Final pre-submit sanity

- [ ] Re-read the Kaggle writeup once more for any sentence that overstates a claim
  - In particular: "phone TTFT" claim must be absent or explicitly deferred
  - "100% adversarial safety refusal" is correct (the n=100 set, not "all possible adversarials")
  - "0 of 12 missed-as-GREEN" is correct (held-out test split with the routed config — verifiable in `eval/results/run_task6_routed_*.json`)
- [ ] Verify the disclaimer line appears at least once visibly in the writeup
- [ ] Submit
- [ ] Save the submission confirmation email / page to PDF for your records

## Post-submission

- [ ] Tweet / LinkedIn / professional network (optional) — link the GitHub + HF + Kaggle notebook
- [ ] Monitor any hackathon Q&A channel for clarification requests
- [ ] If any reviewer comment arrives, respond with documentation pointers (`eval/analysis/2026-05-07/...`) rather than re-running anything
