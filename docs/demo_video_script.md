# Demo video script — Marunthagam (Sprint 3 deliverable)

**Target length:** 3-4 minutes
**Audience:** Gemma 4 Good Hackathon judges + the broader research community
**Recording approach:** screen capture (CLI + Android emulator + dashboard) with voice-over

The video does NOT claim phone-tier latency from emulator runs — that
caveat is explicit in voice-over and matches what the README documents.

---

## Opening (~20s) — what is Marunthagam, what we built

> "Marunthagam — Tamil for 'place of medicine.' An offline Tamil-first
> triage AI for ASHA workers, India's village-level community health
> workers. We built it on Gemma 4 E4B for the Gemma 4 Good Hackathon.
>
> ASHA workers serve rural villages with no doctor on call, intermittent
> connectivity, and clinical decisions made on paper checklists. The
> system runs entirely on-device — model, protocol engine, encrypted
> log database — no network call required to produce a triage decision."

**Show:** project README hero diagram (3-tier architecture), the disclaimer
"இது மருத்துவ ஆலோசனை அல்ல" rendered prominently.

---

## Scene 1 (~45s) — the working triage path

**Show:** terminal running `python inference/cli_demo.py --models-dir training/models`

> "Here's the production stack. Three preset Tamil queries — one GREEN
> (mild cough, low-acuity), one YELLOW (persistent abdominal pain), one
> RED (cardiac-pattern chest pain with arm numbness). The CLI runs the
> same inference path the held-out evaluation uses: routed inference
> across three specialist LoRAs, output validated against the
> `triage_classify()` schema, then passed through the deterministic
> IMNCI protocol engine.
>
> Notice the cardiac case: the model outputs YELLOW with confidence 0.9.
> Then `ADULT-CARDIAC-001` fires in the engine and upgrades it to RED.
> That's the safety architecture: the LLM proposes, the protocol engine
> can only escalate, never downgrade."

**Highlight in capture:** 
- Three JSON outputs, one per preset
- The `engine_overrides` field showing `ADULT-CARDIAC-001`
- The disclaimer string at the bottom of every output

---

## Scene 2 (~45s) — the diagnostic finding that changed the project

**Show:** `eval/analysis/2026-05-06/red_failure_modes.md` and the cross-specialist matrix figure.

> "When we first ran held-out evaluation, F1 was 0.61 — well below the
> original 0.80 target. Diagnostic Sprint 1 surfaced *why*. We
> hand-reviewed 113 triage GREEN training labels with a clinician.
> 18% were under-triaged — chest-pain-with-arm-numbness labeled GREEN,
> post-fall syncope labeled GREEN, persistent palpitations labeled GREEN.
> The minority class carried the noise.
>
> Sprint 2 was the fix. We relabeled, retrained, expanded the IMNCI
> rules to handle Tamil case-inflected forms, and rebuilt the safety
> classifier to handle Hindi devanagari and Gujarati when the model
> code-switches. Smoke test passes 25 of 25."

**Highlight:**
- The 22-of-22 false-negative finding from the safety classifier
- The Tamil morphology examples (மார்பில் locative, நாயினால் instrumental, மூச்சுத்திணறல் compound)

---

## Scene 3 (~45s) — Android emulator, the offline path

**Show:** Android emulator running the app, the user typing a Tamil query, the triage card rendering.

> "Tier 1 deployment: the Android app loaded with our Q4_K_M GGUF.
> Here we're using a Pixel 6 emulator on Android 14. ASHA worker types
> the symptom in Tamil — 'severe chest pain on the left side, arm
> numbness.' The app calls llama.cpp through JNI, parses the JSON,
> applies the protocol engine.
>
> The triage card shows up with the cardiac-pattern RED escalation,
> the next steps in plain Tamil, the protocol references, the
> mandatory disclaimer. No network call.
>
> *Honest caveat: this is an emulator. We don't claim phone-tier
> latency from emulator runs — that measurement is deferred until we
> can run on a real device. The README states this explicitly.*"

**Highlight:**
- The Tamil keyboard input
- The triage card UI elements (level / confidence / next steps / references / disclaimer)
- The offline indicator

---

## Scene 4 (~30s) — Tier 3 district health dashboard

**Show:** dashboard built and running locally — `npm run dev`, the Overview/Map/Alerts/Trends pages.

> "Tier 3 — district health officer's dashboard. React + D3, consumes
> aggregated signals only. No individual patient records ever flow up
> from Tier 1. Geohash precision is capped at ~1km — outbreak mapping
> works, individual identification doesn't.
>
> The dashboard surfaces escalation hotspots, RED-case clusters, and
> trend deviations from baseline. ASHA workers' own data privacy is
> protected by aggregation."

**Highlight:**
- The map view rendering geohash cells
- The trend chart with G/Y/R counts per day
- The alerts panel

---

## Scene 5 (~45s) — what generalises

**Show:** the README's "What generalises" section, scrolling through.

> "What we believe generalises beyond this submission:
>
> One — clinical relabeling on the GREEN class is non-optional for triage
> data. The minority class carries the most labeling noise. Relabel
> before you retrain.
>
> Two — Tamil regex needs morphology-aware patterns. Bare nominative
> forms miss locative, accusative, instrumental, dative cases. Hindi
> and Gujarati script must be covered when the model code-switches.
>
> Three — schema-consumer audits catch silent data loss. We surfaced
> two cases where the eval pipeline computed information it then
> threw away in throwaway local variables. Patch the schema, not the
> analysis.
>
> The model performance numbers are the evidence those processes
> produced something real. F1 0.6491 on held-out. Zero of twelve
> emergencies missed-as-GREEN. Hundred percent adversarial safety
> refusal."

---

## Closing (~20s) — links + thanks

**Show:** the GitHub URL, HuggingFace URLs (mechramc/marunthagam-*).

> "Code on GitHub at github.com/mechramc/Marunthagam.
>
> Models and dataset on HuggingFace at huggingface.co/mechramc.
>
> Submission for the Gemma 4 Good Hackathon. Thank you."

---

## Recording checklist

- [ ] CLI demo runs cleanly (use `--models-dir training/models`, no `--mock`)
- [ ] Emulator running, APK installed, B-retrained GGUF pushed to `/data/data/<app>/files/` via adb push
- [ ] Dashboard running: `cd dashboard && npm run dev`
- [ ] HF upload finished — repo URLs render with model cards
- [ ] Tamil text rendering correctly in capture (verify font support)
- [ ] Audio: voice-over in clear English, no background noise

## Tools

- OBS Studio for screen capture (free, robust)
- Audio: built-in mic should be fine for voice-over; pop filter optional
- Editing: ffmpeg for stitching + trimming, no need for full NLE

## Delivery

Upload to YouTube (unlisted) or directly into the submission package as
an mp4. Paste link into README and into the hackathon submission form.
