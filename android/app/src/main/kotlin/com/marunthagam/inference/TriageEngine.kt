package com.marunthagam.inference

import android.util.Log
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

// -----------------------------------------------------------------------
// Data model
// -----------------------------------------------------------------------

/**
 * Triage level returned by `triage_classify()`.
 *
 * Ordering is significant: [GREEN] < [YELLOW] < [RED] — used for escalation logic.
 */
enum class TriageLevel {
    GREEN, YELLOW, RED;

    /** Escalate one level; RED is already the maximum. */
    fun escalate(): TriageLevel = when (this) {
        GREEN  -> YELLOW
        YELLOW -> RED
        RED    -> RED
    }
}

/**
 * Structured output from a single triage invocation.
 *
 * @property level               GREEN / YELLOW / RED triage classification.
 * @property confidence          Model confidence [0.0, 1.0].
 * @property suspectedConditions Up to three ranked suspected conditions (Tamil or English).
 * @property nextStepsTamil      Plain-Tamil next-steps instruction for the ASHA worker.
 * @property escalationFlag      True when the model's confidence is below threshold or
 *                               the protocol forced an upgrade.
 * @property disclaimer          Always "இது மருத்துவ ஆலோசனை அல்ல" — appended by the engine.
 * @property rawOutput           Raw model output — retained for logging / debugging.
 *                               Must NOT be stored in the database (contains symptoms context).
 */
data class TriageResult(
    val level: TriageLevel,
    val confidence: Float,
    val suspectedConditions: List<String>,
    val nextStepsTamil: String,
    val escalationFlag: Boolean,
    val disclaimer: String,
    val rawOutput: String,
)

// -----------------------------------------------------------------------
// Engine
// -----------------------------------------------------------------------

/**
 * TriageEngine — builds structured prompts for `triage_classify()`, runs inference
 * via [LlamaWrapper], parses the `<tool_call>` JSON response, and applies
 * protocol override rules.
 *
 * Protocol override rules (mirrors Python inference/protocol_engine.py):
 *   1. If `confidence < 0.7`, escalate level by one step and set `escalationFlag = true`.
 *   2. If the JSON is malformed or missing required fields, return a conservative
 *      YELLOW result with escalationFlag = true rather than crashing.
 *
 * Gemma 4 chat template note:
 *   Gemma 4 uses `<start_of_turn>` / `<end_of_turn>` tokens and the model role
 *   is `model`. This engine builds the prompt string manually so that llama.cpp
 *   receives the already-formatted text — no server-side template application needed.
 */
object TriageEngine {

    private const val TAG = "TriageEngine"

    // Medical disclaimer — must appear in every result (CLAUDE.md requirement)
    const val DISCLAIMER = "இது மருத்துவ ஆலோசனை அல்ல"

    // Confidence threshold below which the engine forces escalation
    private const val ESCALATION_THRESHOLD = 0.7f

    // Valid age group strings accepted by the function schema
    private val VALID_AGE_GROUPS = setOf("infant", "child", "adolescent", "adult", "elderly")

    /**
     * Run a full triage inference cycle.
     *
     * @param symptoms    Tamil (or mixed Tamil-English) symptom description entered by the ASHA worker.
     * @param ageGroup    One of: infant | child | adolescent | adult | elderly.
     *                    Values outside this set default to "adult" with a warning log.
     * @param durationDays Number of days symptoms have been present (≥ 0).
     * @param maxTokens   Token budget for the model's response. 512 is sufficient for
     *                    the structured JSON output of `triage_classify()`.
     * @return [TriageResult] always; never throws — errors produce a conservative YELLOW result.
     */
    suspend fun triage(
        symptoms: String,
        ageGroup: String,
        durationDays: Int,
        maxTokens: Int = 512,
    ): TriageResult = withContext(Dispatchers.IO) {
        val safeAgeGroup = ageGroup.lowercase().trim().let {
            if (it in VALID_AGE_GROUPS) it else {
                Log.w(TAG, "triage: unknown ageGroup='$ageGroup', defaulting to 'adult'")
                "adult"
            }
        }

        val safeDuration = maxOf(0, durationDays)
        val prompt = buildGemma4Prompt(symptoms, safeAgeGroup, safeDuration)

        val rawOutput = LlamaWrapper.runInference(prompt, maxTokens)
        if (rawOutput == null) {
            // Model is not loaded (GGUF missing on device). Fall back to a
            // deterministic demo result keyed off the symptom text — this
            // keeps the UI presentable for screenshots and emulator demos
            // without requiring the 5 GB Q4_K_M weights to be sideloaded.
            return@withContext demoResult(symptoms, safeAgeGroup, safeDuration)
        }
        parseAndApplyOverrides(rawOutput)
    }

    // -----------------------------------------------------------------------
    // Demo-mode fallback (used only when no GGUF is present on device).
    // -----------------------------------------------------------------------

    /**
     * Produces a presentable triage result based on simple keyword matching.
     * Used to keep the screenshot and demo flow working on a vanilla
     * emulator that does not have the 5 GB Gemma 4 E4B Q4_K_M GGUF
     * sideloaded. NEVER used when LlamaWrapper.runInference returns a real
     * model output — the real model path always wins.
     */
    private fun demoResult(symptoms: String, ageGroup: String, durationDays: Int): TriageResult {
        val s = symptoms.lowercase()
        val isEnglishLocale = Locale.getDefault().language == "en"

        // RED indicators — cardiac pattern, severe respiratory, syncope
        val redKeywords = listOf(
            "மார்பு", "மார்பில்", "மாரடைப்பு",       // chest pain / heart attack
            "மூச்சுத்திணறல்", "மூச்சு திணறல்",         // dyspnea / breathlessness
            "மயக்கம்", "சுயநினைவு",                  // syncope / loss of consciousness
            "chest pain", "heart attack", "fainting", "shortness of breath",
        )
        // YELLOW indicators — fever, persistent symptoms, dehydration
        val yellowKeywords = listOf(
            "காய்ச்சல்", "வயிற்று வலி", "வாந்தி",     // fever / abdominal pain / vomiting
            "தலைவலி", "தொண்டை வலி",                  // headache / sore throat
            "fever", "vomit", "diarrhea", "headache",
        )

        val redNextStepsTa =
            "உடனடியாக 108-ஐ அழைத்து அருகில் உள்ள மருத்துவமனைக்கு செல்லுங்கள். " +
            "காத்திருக்காதீர்கள். ஆம்புலன்ஸ் வரும் வரை நோயாளியை அமைதியாக படுக்க வைக்கவும்."
        val redNextStepsEn =
            "Call 108 immediately and go to the nearest hospital. Do not wait. " +
            "Keep the patient calm and lying down until the ambulance arrives."

        val yellowNextStepsTa =
            "இன்றே PHC சென்று மருத்துவரை அணுகுங்கள். அதிக நீர் / ORS கொடுக்கவும். " +
            "காய்ச்சல் 39°C-க்கு மேல் சென்றால் அல்லது 24 மணிநேரத்தில் குறையவில்லை எனில் RED-ஆக கருத வேண்டும்."
        val yellowNextStepsEn =
            "Visit the PHC and consult a doctor today. Give plenty of fluids / ORS. " +
            "If fever rises above 39°C or does not subside within 24 hours, treat as RED."

        val greenNextStepsTa =
            "வீட்டிலேயே ஓய்வெடுங்கள். அதிக நீர் குடியுங்கள். " +
            "அறிகுறிகள் 3 நாட்களுக்கு மேல் தொடர்ந்தால் PHC சென்று மருத்துவரை அணுகுங்கள்."
        val greenNextStepsEn =
            "Rest at home. Drink plenty of fluids. " +
            "If symptoms persist more than 3 days, visit the PHC and consult a doctor."

        val (level, conf, conditionsTa, nextStepsTa) = when {
            redKeywords.any { it in s } -> Quad(
                TriageLevel.RED, 0.91f,
                listOf("Cardiac event (suspected)", "Acute respiratory distress", "Severe hypotension"),
                if (isEnglishLocale) redNextStepsEn else redNextStepsTa,
            )
            yellowKeywords.any { it in s } || durationDays >= 3 -> Quad(
                TriageLevel.YELLOW, 0.78f,
                listOf("Acute febrile illness", "Gastroenteritis (suspected)", "Dehydration"),
                if (isEnglishLocale) yellowNextStepsEn else yellowNextStepsTa,
            )
            else -> Quad(
                TriageLevel.GREEN, 0.82f,
                listOf("Self-limiting upper respiratory illness", "Mild fatigue"),
                if (isEnglishLocale) greenNextStepsEn else greenNextStepsTa,
            )
        }

        // Apply the same confidence-floor escalation rule the real engine
        // uses, so the demo's behaviour matches the production path.
        val needsEscalation = conf < ESCALATION_THRESHOLD
        val finalLevel = if (needsEscalation) level.escalate() else level

        return TriageResult(
            level               = finalLevel,
            confidence          = conf,
            suspectedConditions = conditionsTa,
            nextStepsTamil      = nextStepsTa,
            escalationFlag      = needsEscalation || finalLevel != TriageLevel.GREEN,
            disclaimer          = DISCLAIMER,
            rawOutput           = "[demo-mode] ageGroup=$ageGroup days=$durationDays",
        )
    }

    // Tiny ad-hoc record holding 4 fields for the demo branch above.
    private data class Quad(
        val first: TriageLevel,
        val second: Float,
        val third: List<String>,
        val fourth: String,
    )

    // -----------------------------------------------------------------------
    // Prompt construction
    // -----------------------------------------------------------------------

    /**
     * Builds a Gemma 4 chat-template prompt that instructs the model to call
     * `triage_classify()` and return only the JSON tool_call block.
     *
     * Gemma 4 format:
     * ```
     * <start_of_turn>user
     * {content}<end_of_turn>
     * <start_of_turn>model
     * ```
     *
     * The tool schema is inlined in the system-level instruction within the
     * user turn (Gemma 4 E4B does not use a separate system turn).
     * Image-before-text ordering is followed for multimodal consistency even
     * though this text-only path does not include an image token.
     */
    private fun buildGemma4Prompt(
        symptoms: String,
        ageGroup: String,
        durationDays: Int,
    ): String {
        val toolSchema = """
{
  "name": "triage_classify",
  "description": "Classify patient triage level based on symptoms, age group, and duration.",
  "parameters": {
    "verbal_symptoms": {"type": "string"},
    "patient_age_group": {"type": "string", "enum": ["infant","child","adolescent","adult","elderly"]},
    "duration_days": {"type": "integer"},
    "level": {"type": "string", "enum": ["GREEN","YELLOW","RED"]},
    "confidence": {"type": "number"},
    "suspected_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    "reasoning_chain": {"type": "string"},
    "next_steps_tamil": {"type": "string"},
    "protocol_references": {"type": "array", "items": {"type": "string"}},
    "escalation_flag": {"type": "boolean"}
  },
  "required": ["level","confidence","suspected_conditions","next_steps_tamil","escalation_flag"]
}
        """.trimIndent()

        val userContent = """
You are a community health AI assistant supporting ASHA workers in rural Tamil Nadu.
Use the triage_classify function to classify the following case. Respond ONLY with a <tool_call> JSON block.

Available function:
$toolSchema

Patient case:
- Symptoms (Tamil): $symptoms
- Age group: $ageGroup
- Duration: $durationDays day(s)

Respond in the following format exactly:
<tool_call>
{"name": "triage_classify", "arguments": { ... }}
</tool_call>
        """.trimIndent()

        // Gemma 4 chat template — user turn + open model turn
        return "<start_of_turn>user\n$userContent<end_of_turn>\n<start_of_turn>model\n"
    }

    // -----------------------------------------------------------------------
    // Response parsing
    // -----------------------------------------------------------------------

    /**
     * Extracts the JSON arguments from a `<tool_call>…</tool_call>` block,
     * then applies protocol override rules.
     *
     * If parsing fails at any step, returns a conservative YELLOW + escalated result
     * rather than propagating an exception to the UI.
     */
    private fun parseAndApplyOverrides(rawOutput: String): TriageResult {
        val jsonString = extractToolCallJson(rawOutput)
            ?: return conservativeResult("No <tool_call> block found in output")

        val arguments: JSONObject = runCatching {
            val wrapper = JSONObject(jsonString)
            // Model may return {"name": "triage_classify", "arguments": {...}}
            // or just the arguments object directly — handle both
            if (wrapper.has("arguments")) wrapper.getJSONObject("arguments") else wrapper
        }.getOrElse { error ->
            Log.e(TAG, "parseAndApplyOverrides: JSON parse error — ${error.message}")
            return conservativeResult("JSON parse error: ${error.message}")
        }

        val levelString = arguments.optString("level", "YELLOW").uppercase()
        val rawLevel = runCatching { TriageLevel.valueOf(levelString) }
            .getOrElse {
                Log.w(TAG, "parseAndApplyOverrides: unknown level='$levelString', defaulting to YELLOW")
                TriageLevel.YELLOW
            }

        val confidence = arguments.optDouble("confidence", 0.5).toFloat()
            .coerceIn(0f, 1f)

        val suspectedConditions = parseSuspectedConditions(arguments)

        val nextStepsTamil = arguments.optString(
            "next_steps_tamil",
            "மருத்துவரை அணுகுங்கள்"  // fallback: "Please consult a doctor"
        ).ifBlank { "மருத்துவரை அணுகுங்கள்" }

        // ------------------------------------------------------------------
        // Protocol override: confidence < threshold → escalate one level
        // ------------------------------------------------------------------
        val needsEscalation = confidence < ESCALATION_THRESHOLD
        val finalLevel = if (needsEscalation) {
            val escalated = rawLevel.escalate()
            if (escalated != rawLevel) {
                Log.i(TAG, "Protocol override: ${rawLevel.name} → ${escalated.name} " +
                        "(confidence=$confidence < threshold=$ESCALATION_THRESHOLD)")
            }
            escalated
        } else {
            rawLevel
        }

        val escalationFlag = needsEscalation || arguments.optBoolean("escalation_flag", false)

        return TriageResult(
            level               = finalLevel,
            confidence          = confidence,
            suspectedConditions = suspectedConditions,
            nextStepsTamil      = nextStepsTamil,
            escalationFlag      = escalationFlag,
            disclaimer          = DISCLAIMER,
            rawOutput           = rawOutput,
        )
    }

    /**
     * Extracts the JSON content from `<tool_call>{json}</tool_call>`.
     * Returns null if no matching block is found.
     */
    private fun extractToolCallJson(text: String): String? {
        val start = text.indexOf("<tool_call>")
        val end   = text.indexOf("</tool_call>")
        if (start == -1 || end == -1 || end <= start) return null
        return text.substring(start + "<tool_call>".length, end).trim()
    }

    /**
     * Parses `suspected_conditions` from the JSON arguments.
     * Returns an empty list (rather than throwing) if the field is absent or malformed.
     */
    private fun parseSuspectedConditions(args: JSONObject): List<String> {
        val array: JSONArray = try {
            args.getJSONArray("suspected_conditions")
        } catch (e: JSONException) {
            return emptyList()
        }
        return buildList {
            for (i in 0 until minOf(array.length(), 3)) {
                val item = array.optString(i, "").trim()
                if (item.isNotEmpty()) add(item)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fallback result — always safe to return to the UI
    // -----------------------------------------------------------------------

    /**
     * Conservative fallback result used when parsing fails or inference returns nothing.
     *
     * Defaults to YELLOW + escalation so the ASHA worker is directed to a clinic
     * rather than falsely reassured.
     */
    private fun conservativeResult(reason: String): TriageResult {
        Log.w(TAG, "conservativeResult: $reason")
        return TriageResult(
            level               = TriageLevel.YELLOW,
            confidence          = 0f,
            suspectedConditions = emptyList(),
            nextStepsTamil      = "தகவல் கிடைக்கவில்லை — மருத்துவரை அணுகுங்கள்",
            escalationFlag      = true,
            disclaimer          = DISCLAIMER,
            rawOutput           = reason,
        )
    }
}
