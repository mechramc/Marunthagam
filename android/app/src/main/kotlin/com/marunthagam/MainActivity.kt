package com.marunthagam

import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.marunthagam.databinding.ActivityTriageBinding
import com.marunthagam.inference.LlamaWrapper
import com.marunthagam.inference.TriageEngine
import com.marunthagam.inference.TriageLevel
import com.marunthagam.inference.TriageResult
import com.marunthagam.storage.TriageLog
import com.marunthagam.storage.TriageLogDao
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * MainActivity — primary ASHA worker screen for Marunthagam triage.
 *
 * Lifecycle contract:
 *   [onStart]  → initialise TriageEngine (load GGUF model into llama.cpp)
 *   [onStop]   → free the loaded model to release the ~5 GB of device RAM
 *
 * We use onStart/onStop rather than onCreate/onDestroy so that the model is
 * released when the app is backgrounded (e.g. an incoming call) and reloaded
 * when it returns to foreground. This avoids the model sitting in RAM while
 * the app is not in use, which would cause the OS to kill the process on
 * low-memory devices.
 *
 * ViewBinding: ActivityTriageBinding is generated from activity_triage.xml.
 * All view references come from binding.* — no findViewById() calls.
 */
class MainActivity : AppCompatActivity() {

    private val tag = "MainActivity"

    private lateinit var binding: ActivityTriageBinding
    private lateinit var triageLogDao: TriageLogDao

    // Internal age group values that map 1:1 to the Tamil display labels in
    // R.array.age_groups_tamil (indices must stay aligned with arrays.xml)
    private val ageGroupValues: Array<String> by lazy {
        resources.getStringArray(R.array.age_groups)
    }

    // Display labels populated into the AutoCompleteTextView dropdown
    private val ageGroupLabels: Array<String> by lazy {
        resources.getStringArray(R.array.age_groups_tamil)
    }

    // Standard path where ASHA workers place the GGUF file after downloading
    // via USB or a one-time clinic sync. Users are instructed to put the file
    // at this exact path in the setup guide.
    private val modelPath: String by lazy {
        File(
            Environment.getExternalStorageDirectory(),
            "marunthagam/gemma4-e4b-q4_k_m.gguf"
        ).absolutePath
    }

    // ---------------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTriageBinding.inflate(layoutInflater)
        setContentView(binding.root)

        triageLogDao = TriageLogDao(this)

        setupAgeGroupDropdown()
        setupAssessButton()
    }

    override fun onStart() {
        super.onStart()
        // Load model on every foreground entry so it is available when the user
        // starts filling in the form. Loading is async — the UI remains interactive
        // but the button will show loading state while the model warms up.
        loadModel()
    }

    override fun onStop() {
        super.onStop()
        // Release ~5 GB of RAM when the activity is no longer visible.
        lifecycleScope.launch {
            LlamaWrapper.freeModel()
            Log.i(tag, "onStop: model freed")
        }
    }

    // ---------------------------------------------------------------------------
    // Setup helpers
    // ---------------------------------------------------------------------------

    /**
     * Wires the age group [AutoCompleteTextView] to the Tamil display labels
     * from [R.array.age_groups_tamil], defaulting the selection to "adult"
     * (index 3) as the most common ASHA worker patient demographic.
     */
    private fun setupAgeGroupDropdown() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_dropdown_item_1line,
            ageGroupLabels,
        )
        binding.spinnerAgeGroup.setAdapter(adapter)
        // Pre-select "பெரியவர்" (adult) — index 3
        binding.spinnerAgeGroup.setText(ageGroupLabels[3], false)
    }

    private fun setupAssessButton() {
        binding.btnAssess.setOnClickListener {
            val symptoms = binding.etSymptoms.text?.toString()?.trim().orEmpty()

            if (symptoms.isEmpty()) {
                Snackbar.make(
                    binding.root,
                    getString(R.string.error_symptoms_empty),
                    Snackbar.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            runTriage(symptoms)
        }
    }

    // ---------------------------------------------------------------------------
    // Model loading
    // ---------------------------------------------------------------------------

    private fun loadModel() {
        lifecycleScope.launch {
            setInferenceInProgress(true)

            val modelFile = File(modelPath)
            if (!modelFile.exists()) {
                Log.e(tag, "loadModel: GGUF not found at $modelPath")
                withContext(Dispatchers.Main) {
                    Snackbar.make(
                        binding.root,
                        getString(R.string.error_model_not_found),
                        Snackbar.LENGTH_LONG
                    ).show()
                }
                setInferenceInProgress(false)
                return@launch
            }

            val success = LlamaWrapper.loadModel(modelPath)
            if (!success) {
                Log.e(tag, "loadModel: LlamaWrapper.loadModel returned false")
                withContext(Dispatchers.Main) {
                    Snackbar.make(
                        binding.root,
                        getString(R.string.error_model_not_found),
                        Snackbar.LENGTH_LONG
                    ).show()
                }
            }

            setInferenceInProgress(false)
        }
    }

    // ---------------------------------------------------------------------------
    // Triage inference
    // ---------------------------------------------------------------------------

    private fun runTriage(symptoms: String) {
        val selectedLabel = binding.spinnerAgeGroup.text?.toString().orEmpty()
        // Map Tamil display label back to the internal English value that
        // TriageEngine.triage() expects (e.g. "பெரியவர்" → "adult")
        val ageGroupIndex = ageGroupLabels.indexOf(selectedLabel)
        val ageGroup = if (ageGroupIndex >= 0) ageGroupValues[ageGroupIndex] else "adult"

        val durationDays = binding.etDurationDays.text?.toString()?.toIntOrNull() ?: 0

        setInferenceInProgress(true)
        hideResultCard()

        lifecycleScope.launch {
            val result = TriageEngine.triage(
                symptoms = symptoms,
                ageGroup = ageGroup,
                durationDays = durationDays,
            )

            // Persist anonymised log entry (no symptoms text stored — privacy)
            withContext(Dispatchers.IO) {
                triageLogDao.insertLog(
                    TriageLog(
                        ageGroup       = ageGroup,
                        durationDays   = durationDays,
                        triageLevel    = result.level,
                        confidence     = result.confidence,
                        escalationFlag = result.escalationFlag,
                    )
                )
            }

            setInferenceInProgress(false)
            showResult(result)
        }
    }

    // ---------------------------------------------------------------------------
    // Result rendering
    // ---------------------------------------------------------------------------

    /**
     * Renders a [TriageResult] into the result card views.
     *
     * Triage level badge colours:
     *   GREEN  → #4CAF50 (Material Green 500)
     *   YELLOW → #FFC107 (Material Amber 500)
     *   RED    → #F44336 (Material Red 500)
     */
    private fun showResult(result: TriageResult) {
        // Level badge
        val (levelText, levelColor) = when (result.level) {
            TriageLevel.GREEN  -> "GREEN"  to Color.parseColor("#4CAF50")
            TriageLevel.YELLOW -> "YELLOW" to Color.parseColor("#FFC107")
            TriageLevel.RED    -> "RED"    to Color.parseColor("#F44336")
        }
        binding.chipTriageLevel.text = levelText
        binding.chipTriageLevel.chipBackgroundColor = ColorStateList.valueOf(levelColor)

        // Confidence as percentage
        val confidencePct = (result.confidence * 100).toInt()
        binding.tvConfidence.text = "$confidencePct%"

        // Escalation warning chip
        binding.chipEscalation.visibility =
            if (result.escalationFlag) View.VISIBLE else View.GONE

        // Next steps in Tamil
        binding.tvNextSteps.text = result.nextStepsTamil

        binding.cardResult.visibility = View.VISIBLE
    }

    private fun hideResultCard() {
        binding.cardResult.visibility = View.GONE
    }

    // ---------------------------------------------------------------------------
    // Loading state management
    // ---------------------------------------------------------------------------

    /**
     * Toggles the inference progress indicator and the assess button enabled state.
     * Called on the main thread — safe to call from lifecycleScope.launch {} directly
     * because lifecycleScope dispatches on Main by default.
     */
    private fun setInferenceInProgress(inProgress: Boolean) {
        binding.progressInference.visibility = if (inProgress) View.VISIBLE else View.GONE
        binding.btnAssess.isEnabled = !inProgress
    }
}
