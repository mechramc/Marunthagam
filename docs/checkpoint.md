# Marunthagam - Checkpoint Log

Each checkpoint is written before every `git commit` + `git push`.

---

## Checkpoint Format

```text
### [YYYY-MM-DD HH:MM] Checkpoint N - <description>

**Commit:** <SHA or pending>
**Tasks completed:** T?.? - <name>
**Tests passing:** <test count / build verification>
**Verifier:** Approved / Issues (see notes)
**Notes:** <any issues found or deferred>
```

---

## Checkpoints

### [2026-04-14 16:40] Checkpoint 6 - Dashboard + Android build readiness verified

**Commit:** pending
**Tasks completed:** T10.5 (dashboard production build), T7.9-T7.11 (Android wrapper, JNI API update, debug APK build)
**Tests passing:** `pytest -q` in `inference` (33 passed), `npm run build` in `dashboard`, `.\\gradlew.bat assembleDebug` in `android`
**Verifier:** Approved - dashboard and Android now build on this machine with the required local SDK/JDK setup
**Files:** `dashboard/package-lock.json`, `android/gradlew`, `android/gradlew.bat`, `android/gradle/wrapper/gradle-wrapper.jar`, `android/gradle/wrapper/gradle-wrapper.properties`, `android/app/src/main/kotlin/com/marunthagam/inference/LlamaWrapper.kt`, `android/app/src/main/cpp/CMakeLists.txt`, `android/app/src/main/cpp/marunthagam_jni.cpp`
**Notes:** Android build additionally depends on vendored `android/app/src/main/cpp/llama.cpp/` and a local SDK path in `android/local.properties`. Runtime still requires a real GGUF file on device.

### [2026-04-12 04:38] Checkpoint 5 - Task 9+7+11: eval suite completion, Android scaffold, README + docs

**Commit:** 26a1736
**Tasks completed:** T9.2-T9.4, T7.4-T7.8, T11.1-T11.3
**Tests passing:** 33 (unchanged - new code was scripts/docs, no new pytest suites)
**Verifier:** Approved - eval scripts run clean in mock mode, Android namespace consistency confirmed, README structure matches spec
**Notes:** Latency mock reports fail on workstation targets (expected in mock mode). Safety mock reports 98/100 by design. Android still needed `llama.cpp`, NDK, and local SDK wiring at this checkpoint.

### [2026-04-12 03:00] Checkpoint 4 - Week 3 build: eval scripts, Android JNI, React dashboard, results notebook

**Commit:** 3a00b7a
**Tasks completed:** T7.1-T7.3, T9.1, T10.1-T10.4
**Tests passing:** 33
**Verifier:** Approved
**Notes:** Privacy fix applied on JNI prefill failure. Dashboard uses Tamil Nadu geohash prefixes `tf7`/`tf8`.

### [2026-04-10 23:12] Checkpoint 3 - Week 2 training scripts + inference logger

**Commit:** 8014255
**Tasks completed:** T5.1-T5.2, T6.1-T6.3, T8.1-T8.4
**Tests passing:** 33
**Verifier:** Approved
**Notes:** Router still uses placeholder embeddings and placeholder embedding dimension in config.

### [2026-04-10 20:48] Checkpoint 2 - Dataset pipeline + baseline eval

**Commit:** bea0075
**Tasks completed:** T4.1-T4.5
**Tests passing:** 31
**Verifier:** Approved
**Notes:** Validation split is forced non-empty for small fixture groups.

### [2026-04-10 14:30] Checkpoint 1 - Protocol engine + function calling

**Commit:** a9d85d8
**Tasks completed:** T1.1-T1.3, T2.1-T2.7, T3.1-T3.5
**Tests passing:** 21
**Verifier:** Approved
**Notes:** Edge case noted: RED + low confidence escalation flag behavior left as future test follow-up.

### [2026-04-10 13:39] Checkpoint 0 - Initial scaffold

**Commit:** cc2ab81
**Tasks completed:** Infrastructure (repo structure, plan, protocol schemas)
**Tests passing:** N/A
**Verifier:** N/A
**Notes:** Week 1 execution began from this scaffold.
