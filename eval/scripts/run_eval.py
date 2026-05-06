"""
Marunthagam — Full Evaluation Suite Orchestrator.

Loads test cases from all three specialist fixture files (triage, derm, maternal),
calls triage_classify() via real llama.cpp subprocess or a deterministic mock,
computes classification metrics, and saves results to eval/results/.

Usage:
    python run_eval.py --mock
    python run_eval.py --model /path/to/model.gguf
    python run_eval.py --mock --seeds 42,137,256
    python run_eval.py --model /path/to/model.gguf --seeds 42,137,256
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

# Make scripts/_llama_cpp_setup importable so the cu12 DLL dirs are registered
# before any llama_cpp import. Safe no-op on non-Windows hosts.
_TRAINING_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "training", "scripts")
)
if _TRAINING_SCRIPTS not in sys.path:
    sys.path.insert(0, _TRAINING_SCRIPTS)
import _llama_cpp_setup  # noqa: F401  -- registers cu12 DLL dirs on Windows

# Make the protocol_engine package importable so we can apply the deterministic
# IMNCI safety floor on top of the LLM's prediction.
_PROTOCOL_ENGINE_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "inference", "protocol_engine")
)
if _PROTOCOL_ENGINE_PKG not in sys.path:
    sys.path.insert(0, _PROTOCOL_ENGINE_PKG)
from engine import ProtocolEngine, TriageResult  # noqa: E402

_PROTOCOL_DB_PATH = os.path.join(_PROTOCOL_ENGINE_PKG, "data", "protocol.db")

# Reconfigure stdout/stderr to UTF-8 so Tamil text and box-drawing characters
# render correctly on Windows (which defaults to CP1252 in some environments).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Module-level constants — no magic numbers
# ---------------------------------------------------------------------------
TARGET_F1: float = 0.80
TARGET_RED_RECALL: float = 0.90
TARGET_CHRF: float = 0.60
DISCLAIMER_TEXT: str = "இது மருத்துவ ஆலோசனை அல்ல"
TRIAGE_LEVELS: list[str] = ["GREEN", "YELLOW", "RED"]
SPECIALISTS: list[str] = ["triage", "derm", "maternal"]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "training" / "data" / "fixtures"
EVAL_DATA_DIR = REPO_ROOT / "eval" / "data"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
FORMATTED_DIR = REPO_ROOT / "training" / "data" / "formatted"

from run_logger import RunLogger  # noqa: E402  (after sys.path setup above)

# Mock confidence values by level — deterministic and clinically plausible
_MOCK_CONFIDENCE: dict[str, float] = {
    "GREEN": 0.91,
    "YELLOW": 0.82,
    "RED": 0.95,
}

# Mock adds small per-seed noise to simulate real variation
_MOCK_NOISE_STD: float = 0.02


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single evaluation case loaded from a fixture or baseline file."""
    specialist: str                          # triage | derm | maternal
    verbal_symptoms: str
    age_group: str
    duration_days: int
    gold_level: str                          # GREEN | YELLOW | RED
    case_id: Optional[str] = None
    tamil_question: str = ""                 # natural Tamil patient utterance,
                                             # matches the LoRA training prompt
                                             # format; falls back to verbal_symptoms


@dataclass
class PredictedOutput:
    """Normalised output from triage_classify(), real or mock."""
    level: str
    confidence: float
    escalation_flag: bool
    reasoning_chain: str
    next_steps_tamil: str
    disclaimer: str = DISCLAIMER_TEXT
    # Observability fields — populated only when real (non-mock) inference runs.
    # These let downstream analysis distinguish model output from engine output.
    pre_engine_level: Optional[str] = None
    pre_engine_confidence: Optional[float] = None
    pre_engine_escalation_flag: Optional[bool] = None
    engine_overrides: list[dict] = field(default_factory=list)
    class_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class SeedResult:
    """Metrics for a single seed run."""
    seed: int
    weighted_f1: float
    macro_f1: float
    red_recall: float
    green_f1: float
    yellow_f1: float
    red_f1: float
    escalation_rate: float
    n_cases: int
    per_class_report: dict[str, dict[str, float]]
    per_specialist: dict[str, dict[str, float]] = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)


@dataclass
class AggregatedResult:
    """Mean ± std across multiple seed runs."""
    seeds: list[int]
    weighted_f1_mean: float
    weighted_f1_std: float
    macro_f1_mean: float
    macro_f1_std: float
    red_recall_mean: float
    red_recall_std: float
    escalation_rate_mean: float
    escalation_rate_std: float
    seed_results: list[SeedResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _extract_gold_level(record: dict) -> str:
    """
    Extract gold triage level from a fixture record.

    Fixture records store the gold label in triage_result.level.
    Baseline records store it in gold_level directly.
    """
    if "triage_result" in record and "level" in record["triage_result"]:
        return record["triage_result"]["level"]
    if "gold_level" in record:
        return record["gold_level"]
    raise ValueError(f"Cannot determine gold level from record keys: {list(record.keys())}")


def load_fixture(specialist: str) -> list[TestCase]:
    """
    Load test cases from a specialist fixture JSONL file.

    Each line is a JSON object with function_call_args and triage_result.
    """
    fixture_path = FIXTURES_DIR / f"{specialist}_reviewed.jsonl"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    cases: list[TestCase] = []
    with open(fixture_path, encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {fixture_path}:{line_num}: {exc}"
                ) from exc

            args = record.get("function_call_args", {})
            gold_level = _extract_gold_level(record)

            cases.append(TestCase(
                specialist=specialist,
                verbal_symptoms=args.get("verbal_symptoms", record.get("symptom_description", "")),
                age_group=args.get("patient_age_group", record.get("age_group", "adult")),
                duration_days=int(args.get("duration_days", record.get("duration_days", 1))),
                gold_level=gold_level,
                case_id=f"{specialist}_{line_num:03d}",
                tamil_question=record.get("tamil_question", ""),
            ))

    return cases


def load_baseline_examples() -> list[TestCase]:
    """Load the 20 baseline evaluation examples."""
    baseline_path = EVAL_DATA_DIR / "baseline_examples.json"
    if not baseline_path.exists():
        return []

    with open(baseline_path, encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)

    cases: list[TestCase] = []
    for idx, record in enumerate(records, start=1):
        cases.append(TestCase(
            specialist="triage",
            verbal_symptoms=record.get("symptom_description", ""),
            age_group=record.get("age_group", "adult"),
            duration_days=int(record.get("duration_days", 1)),
            gold_level=record["gold_level"],
            case_id=f"baseline_{idx:03d}",
        ))
    return cases


def load_all_cases() -> list[TestCase]:
    """Load all test cases: fixtures for each specialist + baseline examples."""
    all_cases: list[TestCase] = []
    for specialist in SPECIALISTS:
        try:
            cases = load_fixture(specialist)
            all_cases.extend(cases)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping {specialist}", file=sys.stderr)

    baseline = load_baseline_examples()
    all_cases.extend(baseline)
    return all_cases


def load_test_split_specialist(specialist: str) -> list[TestCase]:
    """
    Load the held-out test split for one specialist (training/data/formatted/<s>/test.jsonl).

    Each line is a chat-format record:
        {"messages": [
            {"role": "user", "content": "<tamil_question>"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "triage_classify",
              "arguments": {"verbal_symptoms": ..., "patient_age_group": ..., "duration_days": ..., ...}}}]},
            {"role": "tool", "content": "<JSON triage_result with level, confidence, suspected_conditions, ...>"},
            {"role": "assistant", "content": "<next_steps_tamil>"}
        ]}
    """
    path = FORMATTED_DIR / specialist / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}")

    cases: list[TestCase] = []
    with open(path, encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            messages = record.get("messages", [])
            if not messages:
                continue

            tamil_question = ""
            args: dict = {}
            gold_tool_payload: dict = {}
            next_steps_tamil = ""

            for msg in messages:
                role = msg.get("role")
                if role == "user" and not tamil_question:
                    tamil_question = msg.get("content") or ""
                elif role == "assistant" and msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = {}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                elif role == "tool":
                    raw_tool = msg.get("content")
                    if isinstance(raw_tool, str):
                        try:
                            gold_tool_payload = json.loads(raw_tool)
                        except json.JSONDecodeError:
                            gold_tool_payload = {}
                    elif isinstance(raw_tool, dict):
                        gold_tool_payload = raw_tool
                elif role == "assistant" and not msg.get("tool_calls"):
                    if not next_steps_tamil and msg.get("content"):
                        next_steps_tamil = msg["content"]

            gold_level = str(gold_tool_payload.get("level", "")).upper()
            if gold_level not in TRIAGE_LEVELS:
                # Skip rows we cannot evaluate (no gold label).
                continue

            cases.append(TestCase(
                specialist=specialist,
                verbal_symptoms=args.get("verbal_symptoms", "") or tamil_question,
                age_group=args.get("patient_age_group", "adult"),
                duration_days=int(args.get("duration_days", 1) or 1),
                gold_level=gold_level,
                case_id=f"{specialist}_test_{line_num:03d}",
                tamil_question=tamil_question,
            ))

    return cases


def load_all_test_split_cases() -> list[TestCase]:
    """Load held-out test split rows across all specialists."""
    all_cases: list[TestCase] = []
    for specialist in SPECIALISTS:
        try:
            cases = load_test_split_specialist(specialist)
            print(f"  Loaded {len(cases)} cases from test.jsonl for {specialist}")
            all_cases.extend(cases)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping {specialist}", file=sys.stderr)
    return all_cases


# ---------------------------------------------------------------------------
# Mock inference
# ---------------------------------------------------------------------------

def _mock_predict(case: TestCase, seed: int) -> PredictedOutput:
    """
    Deterministic mock of triage_classify().

    Returns the gold level with high confidence, plus small Gaussian noise
    seeded per (case_id, seed) for reproducible but non-identical results.
    Introduces a ~10% error rate for realism: one level below gold.
    """
    rng = random.Random(f"{case.case_id}-{seed}")
    noise_rng = random.Random(f"{case.case_id}-{seed}-noise")

    base_confidence = _MOCK_CONFIDENCE[case.gold_level]
    confidence = base_confidence + noise_rng.gauss(0, _MOCK_NOISE_STD)
    confidence = max(0.0, min(1.0, confidence))

    # 10% chance of one-level degradation (GREEN→YELLOW, YELLOW→RED, RED→RED)
    error_roll = rng.random()
    level = case.gold_level
    if error_roll < 0.10:
        idx = TRIAGE_LEVELS.index(case.gold_level)
        level = TRIAGE_LEVELS[min(idx + 1, len(TRIAGE_LEVELS) - 1)]
        confidence = max(0.0, confidence - 0.15)

    escalation_flag = confidence < 0.70 or level == "RED"

    return PredictedOutput(
        level=level,
        confidence=round(confidence, 4),
        escalation_flag=escalation_flag,
        reasoning_chain="[MOCK] பரிசோதனை முடிவு",
        next_steps_tamil="[MOCK] அடுத்த படி",
        disclaimer=DISCLAIMER_TEXT,
    )


# ---------------------------------------------------------------------------
# Real llama.cpp inference
# ---------------------------------------------------------------------------

# Gemma 4 chat-template prompt. Uses the Gemma 4 turn delimiters that the
# LoRA was trained against (<|turn>...<turn|>, NOT Gemma 3's <start_of_turn>).
# We append a Tamil instruction asking for a structured JSON classification,
# then constrain the model's output via a GBNF grammar (see _TRIAGE_GRAMMAR).
_LLAMA_PROMPT_TEMPLATE = (
    "<|turn>user\n"
    "{user_message}\n\n"
    "Classify this case. Output ONE JSON object only, no other text:\n"
    '{{"level": "GREEN" | "YELLOW" | "RED", "confidence": 0.0-1.0, '
    '"escalation_flag": true | false}}<turn|>\n'
    "<|turn>model\n"
    "{{"
)


# GBNF grammar that forces output to be a valid triage JSON object with the
# fields the eval needs. Removing tool_call ambiguity — the model can still
# fall back to natural prose without grammar, but with grammar it MUST emit
# this shape.
_TRIAGE_GRAMMAR = r"""
root        ::= "{" ws "\"level\"" ws ":" ws level ws "," ws
                "\"confidence\"" ws ":" ws confidence ws "," ws
                "\"escalation_flag\"" ws ":" ws bool ws "}"
level       ::= "\"GREEN\"" | "\"YELLOW\"" | "\"RED\""
confidence  ::= "0." [0-9] [0-9]?
              | "1.0"
              | "1.00"
bool        ::= "true" | "false"
ws          ::= [ \t\n]*
"""


# Cached llama_cpp.Llama instances keyed by absolute model path so each GGUF
# is loaded exactly once per process even when --models-dir routes per case.
_LLM_CACHE: dict[str, "Llama"] = {}


def _get_llm(model_path: str) -> "Llama":
    abs_path = os.path.abspath(model_path)
    cached = _LLM_CACHE.get(abs_path)
    if cached is not None:
        return cached
    from llama_cpp import Llama  # local import: avoids cost when --mock is used
    llm = Llama(
        model_path=abs_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
        # logits_all enables the per-position logits needed by `logprobs=N`
        # in our class-probability probe. ~+30% memory; no effect on the main
        # generation behaviour.
        logits_all=True,
    )
    _LLM_CACHE[abs_path] = llm
    return llm


_PROTOCOL_ENGINE: Optional[ProtocolEngine] = None


def _get_protocol_engine() -> Optional[ProtocolEngine]:
    """Open the IMNCI protocol DB once per process. Returns None if missing."""
    global _PROTOCOL_ENGINE
    if _PROTOCOL_ENGINE is not None:
        return _PROTOCOL_ENGINE
    if not os.path.exists(_PROTOCOL_DB_PATH):
        return None
    _PROTOCOL_ENGINE = ProtocolEngine(_PROTOCOL_DB_PATH)
    return _PROTOCOL_ENGINE


# ---------------------------------------------------------------------------
# Per-class logprobs probe (observability only — does not affect inference)
# ---------------------------------------------------------------------------
# After the main JSON completion, we run a single 1-token probe with the
# prompt primed up to `{"level": "` so the next token is the level token.
# top_logprobs at that position give us the model's distribution over
# {GREEN, YELLOW, RED} as actually decoded by the GGUF tokenizer.

# Prompt suffix that makes the very next token the level value.
_LOGPROBS_PROBE_SUFFIX = '{"level": "'

# Tamil: not relevant — we accumulate over Latin token strings the tokenizer
# emits. Different tokenizers split "GREEN"/"YELLOW"/"RED" differently;
# we attribute a token's logprob to whichever class label it is a prefix of.
_CLASS_TOKENS = {"GREEN": "GREEN", "YELLOW": "YELLOW", "RED": "RED"}


def _probe_class_logprobs(llm: "Llama", base_prompt: str) -> dict[str, float]:
    """
    Re-run a single-token probe to capture {GREEN, YELLOW, RED} logprobs.

    Returns a dict {class_label: logprob} for whichever of the three classes
    appear in the top-K returned by llama_cpp. Missing classes are absent
    from the dict (NOT zeroed) so the caller can distinguish "not in top-K"
    from "exactly 0".
    """
    # Drop the trailing `{` the main template primes and replace with the probe
    # suffix that opens the level value's quoted string.
    if base_prompt.endswith("{"):
        probe_prompt = base_prompt[:-1] + _LOGPROBS_PROBE_SUFFIX
    else:
        probe_prompt = base_prompt + _LOGPROBS_PROBE_SUFFIX

    try:
        completion = llm(
            probe_prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=20,  # generous — tokenizer may split labels across tokens
        )
    except Exception:
        return {}

    choice = completion.get("choices", [{}])[0]
    lp = choice.get("logprobs") or {}
    top_at_pos = lp.get("top_logprobs") or []
    if not top_at_pos:
        return {}

    # First emitted token's top-K logprobs as {token_str: logprob}
    top0 = top_at_pos[0] if isinstance(top_at_pos[0], dict) else {}

    # Attribute each candidate token to a class label by prefix match.
    # The first token of "GREEN" might be "G", "GR", "GRE", or "GREEN" — all
    # acceptable as evidence the model picked GREEN. Same for the others.
    out: dict[str, float] = {}
    for tok_str, logp in top0.items():
        stripped = tok_str.lstrip()
        if not stripped:
            continue
        upper = stripped.upper()
        for cls, full in _CLASS_TOKENS.items():
            if full.startswith(upper) and upper:
                # Keep the highest logprob seen for this class (most likely
                # tokenisation among multiple matches).
                if cls not in out or logp > out[cls]:
                    out[cls] = float(logp)
                break

    return out


_TRIAGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "level": {"type": "string", "enum": ["GREEN", "YELLOW", "RED"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "escalation_flag": {"type": "boolean"},
    },
    "required": ["level", "confidence", "escalation_flag"],
    "additionalProperties": False,
}


def _real_predict(case: TestCase, model_path: str) -> PredictedOutput:
    """
    Call the LoRA in-process (llama-cpp-python) with a Gemma 4 chat-template
    prompt and a JSON schema response format that forces the model to emit a
    triage JSON object {"level": ..., "confidence": ..., "escalation_flag": ...}.
    On any parse anomaly we fall back to a low-confidence GREEN with
    escalation_flag=True so the case is still counted in metrics.
    """
    user_message = case.tamil_question.strip() or case.verbal_symptoms
    prompt = _LLAMA_PROMPT_TEMPLATE.format(user_message=user_message)

    llm = _get_llm(model_path)
    completion = llm(
        prompt,
        max_tokens=128,
        temperature=0.0,
        stop=["<turn|>", "<|turn>", "\n\n"],
    )
    # Re-prepend the "{" we primed the prompt with so the parser sees a full JSON object.
    raw_output = "{" + completion["choices"][0]["text"]

    import re
    json_block = re.search(r"\{.*\}", raw_output, re.DOTALL)
    parsed: Optional[dict] = None
    if json_block:
        try:
            parsed = json.loads(json_block.group(0))
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        return PredictedOutput(
            level="GREEN",
            confidence=0.0,
            escalation_flag=True,
            reasoning_chain=f"[PARSE FAIL] {raw_output[:200]}",
            next_steps_tamil="",
        )

    # Support multiple shapes: flat triage_result, {"result": ...}, or
    # {"name": "triage_classify", "arguments": {...}} from a tool_call.
    output_data = parsed
    if isinstance(parsed.get("triage_result"), dict):
        output_data = parsed["triage_result"]
    elif isinstance(parsed.get("result"), dict):
        output_data = parsed["result"]
    elif isinstance(parsed.get("arguments"), dict):
        output_data = parsed["arguments"]

    level = str(output_data.get("level", "GREEN")).upper()
    if level not in TRIAGE_LEVELS:
        # Treat unexpected levels as a low-confidence safety escalation rather
        # than crashing the whole eval run.
        return PredictedOutput(
            level="GREEN",
            confidence=0.0,
            escalation_flag=True,
            reasoning_chain=f"[BAD LEVEL '{level}']",
            next_steps_tamil="",
        )

    try:
        confidence = float(output_data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    escalation_flag = bool(output_data.get("escalation_flag", confidence < 0.70))

    # ---- OBSERVABILITY: snapshot pre-engine state into IMMUTABLE LOCALS ----
    # MUST happen BEFORE engine.apply runs. engine.apply mutates the
    # TriageResult in place (engine.py:100 `result.level = current_level`),
    # so reading triage.level after the call would give post-engine values.
    # Capture into plain str/float/bool — never reference triage.* afterwards
    # for these snapshots.
    pre_engine_level: str = level
    pre_engine_confidence: float = confidence
    pre_engine_escalation_flag: bool = escalation_flag

    # ---- OBSERVABILITY: per-class logprobs probe (1 extra forward pass) ----
    # Called BEFORE engine.apply so the LLM-only signal is captured even if
    # the engine later overrides the level. Probe is independent of the
    # main completion; no inference behaviour change for the main call.
    class_logprobs = _probe_class_logprobs(llm, prompt)

    # Apply IMNCI / TN protocol rules and the confidence floor. The engine only
    # upgrades urgency (never downgrades), so this is a safety floor.
    engine_overrides: list[dict] = []
    engine = _get_protocol_engine()
    if engine is not None:
        triage = TriageResult(
            level=level,
            confidence=confidence,
            suspected_conditions=[],
            reasoning_chain=str(output_data.get("reasoning_chain", "")),
            next_steps_tamil=str(output_data.get("next_steps_tamil", "")),
            protocol_references=[],
            escalation_flag=escalation_flag,
        )
        # v2 engine schema (2026-05-07): chief complaint matched only
        # against verbal_symptoms; narrative used for co-signals + negative
        # scoping. Eliminates the v1 false-positive where rules fired on
        # narrative mentions of unrelated symptoms.
        triage, overrides = engine.apply(
            triage,
            chief_complaint=(case.verbal_symptoms or "").strip(),
            narrative=(case.tamil_question or "").strip(),
            age_group=case.age_group,
            duration_days=case.duration_days,
        )
        # Capture override trace BEFORE we read mutated fields.
        engine_overrides = [
            {
                "rule_id": o.rule_id,
                "original_level": o.original_level,
                "overridden_to": o.overridden_to,
                "reason": o.reason,
            }
            for o in overrides
        ]
        level = triage.level
        escalation_flag = triage.escalation_flag

    return PredictedOutput(
        level=level,
        confidence=confidence,
        escalation_flag=escalation_flag,
        reasoning_chain=str(output_data.get("reasoning_chain", "")),
        next_steps_tamil=str(output_data.get("next_steps_tamil", "")),
        pre_engine_level=pre_engine_level,
        pre_engine_confidence=pre_engine_confidence,
        pre_engine_escalation_flag=pre_engine_escalation_flag,
        engine_overrides=engine_overrides,
        class_logprobs=class_logprobs,
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    cases: list[TestCase],
    predictions: list[PredictedOutput],
    seed: int,
) -> SeedResult:
    """
    Compute per-class and aggregate classification metrics for one seed run.

    RED recall is treated as a critical safety metric.
    """
    gold_labels = [c.gold_level for c in cases]
    pred_labels = [p.level for p in predictions]
    escalation_flags = [p.escalation_flag for p in predictions]

    report_dict: dict[str, dict[str, float]] = classification_report(
        gold_labels,
        pred_labels,
        labels=TRIAGE_LEVELS,
        output_dict=True,
        zero_division=0,
    )  # type: ignore[assignment]

    weighted_f1 = f1_score(
        gold_labels, pred_labels, average="weighted", labels=TRIAGE_LEVELS, zero_division=0
    )
    macro_f1 = f1_score(
        gold_labels, pred_labels, average="macro", labels=TRIAGE_LEVELS, zero_division=0
    )

    red_stats = report_dict.get("RED", {})
    red_recall = float(red_stats.get("recall", 0.0))

    escalation_rate = sum(escalation_flags) / max(len(escalation_flags), 1)

    # Per-specialist slice — useful for A4 (cross-specialist matrix) and A3 (fusion).
    per_specialist: dict[str, dict[str, float]] = {}
    by_spec: dict[str, list[int]] = {}
    for i, case in enumerate(cases):
        by_spec.setdefault(case.specialist, []).append(i)
    for spec, idxs in by_spec.items():
        if not idxs:
            continue
        gold_subset = [gold_labels[i] for i in idxs]
        pred_subset = [pred_labels[i] for i in idxs]
        try:
            wf1 = f1_score(gold_subset, pred_subset, average="weighted",
                           labels=TRIAGE_LEVELS, zero_division=0)
            mf1 = f1_score(gold_subset, pred_subset, average="macro",
                           labels=TRIAGE_LEVELS, zero_division=0)
            sub_report = classification_report(
                gold_subset, pred_subset, labels=TRIAGE_LEVELS,
                output_dict=True, zero_division=0,
            )  # type: ignore[assignment]
            sub_red = float(sub_report.get("RED", {}).get("recall", 0.0))
        except Exception:
            wf1, mf1, sub_red = 0.0, 0.0, 0.0
        per_specialist[spec] = {
            "n": len(idxs),
            "weighted_f1": round(float(wf1), 4),
            "macro_f1": round(float(mf1), 4),
            "red_recall": round(sub_red, 4),
        }

    predictions_out = [
        {
            "case_id": case.case_id,
            "specialist": case.specialist,
            "gold": gold_labels[i],
            "pred": pred_labels[i],
            "confidence": predictions[i].confidence,
            "escalation_flag": predictions[i].escalation_flag,
            # Observability: model-only signal vs engine-applied signal.
            "pre_engine_level": predictions[i].pre_engine_level,
            "pre_engine_confidence": predictions[i].pre_engine_confidence,
            "pre_engine_escalation_flag": predictions[i].pre_engine_escalation_flag,
            "engine_overrides": predictions[i].engine_overrides,
            "class_logprobs": predictions[i].class_logprobs,
        }
        for i, case in enumerate(cases)
    ]

    return SeedResult(
        seed=seed,
        weighted_f1=round(float(weighted_f1), 4),
        macro_f1=round(float(macro_f1), 4),
        red_recall=round(red_recall, 4),
        green_f1=round(float(report_dict.get("GREEN", {}).get("f1-score", 0.0)), 4),
        yellow_f1=round(float(report_dict.get("YELLOW", {}).get("f1-score", 0.0)), 4),
        red_f1=round(float(report_dict.get("RED", {}).get("f1-score", 0.0)), 4),
        escalation_rate=round(escalation_rate, 4),
        n_cases=len(cases),
        per_class_report={
            level: {
                "precision": round(float(report_dict.get(level, {}).get("precision", 0.0)), 4),
                "recall": round(float(report_dict.get(level, {}).get("recall", 0.0)), 4),
                "f1": round(float(report_dict.get(level, {}).get("f1-score", 0.0)), 4),
                "support": int(report_dict.get(level, {}).get("support", 0)),
            }
            for level in TRIAGE_LEVELS
        },
        per_specialist=per_specialist,
        predictions=predictions_out,
    )


def aggregate_seed_results(seed_results: list[SeedResult]) -> AggregatedResult:
    """Compute mean ± std across seed runs."""
    def _stats(values: list[float]) -> tuple[float, float]:
        arr = np.array(values, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    wf1_mean, wf1_std = _stats([r.weighted_f1 for r in seed_results])
    mf1_mean, mf1_std = _stats([r.macro_f1 for r in seed_results])
    rr_mean, rr_std = _stats([r.red_recall for r in seed_results])
    esc_mean, esc_std = _stats([r.escalation_rate for r in seed_results])

    return AggregatedResult(
        seeds=[r.seed for r in seed_results],
        weighted_f1_mean=round(wf1_mean, 4),
        weighted_f1_std=round(wf1_std, 4),
        macro_f1_mean=round(mf1_mean, 4),
        macro_f1_std=round(mf1_std, 4),
        red_recall_mean=round(rr_mean, 4),
        red_recall_std=round(rr_std, 4),
        escalation_rate_mean=round(esc_mean, 4),
        escalation_rate_std=round(esc_std, 4),
        seed_results=seed_results,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _pass_fail(value: float, target: float) -> str:
    return "PASS" if value >= target else "FAIL"


def print_summary_table(result: SeedResult | AggregatedResult) -> None:
    """Print a clean human-readable summary to stdout."""
    separator = "─" * 60
    print(separator)
    print("  MARUNTHAGAM EVAL RESULTS")
    print(separator)

    if isinstance(result, SeedResult):
        print(f"  Seed:            {result.seed}")
        print(f"  Total cases:     {result.n_cases}")
        print(f"  Weighted F1:     {result.weighted_f1:.4f}   target >{TARGET_F1}  "
              f"→ {_pass_fail(result.weighted_f1, TARGET_F1)}")
        print(f"  Macro F1:        {result.macro_f1:.4f}")
        print(f"  RED recall:      {result.red_recall:.4f}   target >{TARGET_RED_RECALL}  "
              f"→ {_pass_fail(result.red_recall, TARGET_RED_RECALL)}")
        print(f"  Escalation rate: {result.escalation_rate:.4f}")
        print()
        print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for level in TRIAGE_LEVELS:
            stats = result.per_class_report[level]
            print(f"  {level:<10} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
                  f"{stats['f1']:>10.4f} {stats['support']:>10d}")
    else:
        print(f"  Seeds:           {result.seeds}")
        print(f"  Weighted F1:     {result.weighted_f1_mean:.4f} ± {result.weighted_f1_std:.4f}"
              f"   target >{TARGET_F1}  "
              f"→ {_pass_fail(result.weighted_f1_mean, TARGET_F1)}")
        print(f"  Macro F1:        {result.macro_f1_mean:.4f} ± {result.macro_f1_std:.4f}")
        print(f"  RED recall:      {result.red_recall_mean:.4f} ± {result.red_recall_std:.4f}"
              f"   target >{TARGET_RED_RECALL}  "
              f"→ {_pass_fail(result.red_recall_mean, TARGET_RED_RECALL)}")
        print(f"  Escalation rate: {result.escalation_rate_mean:.4f} ± {result.escalation_rate_std:.4f}")

    print(separator)


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

def _resolve_model_for_case(
    case: TestCase,
    single_model: Optional[str],
    models_by_specialist: Optional[dict[str, str]],
) -> str:
    """Pick the GGUF for a case: prefer per-specialist routing, fall back to single model."""
    if models_by_specialist is not None:
        path = models_by_specialist.get(case.specialist)
        if path is None:
            raise RuntimeError(
                f"No model registered for specialist {case.specialist!r} in models-dir."
            )
        return path
    assert single_model is not None
    return single_model


def run_single_seed(
    cases: list[TestCase],
    seed: int,
    model_path: Optional[str],
    use_mock: bool,
    models_by_specialist: Optional[dict[str, str]] = None,
) -> SeedResult:
    """Run inference and compute metrics for one seed."""
    random.seed(seed)
    np.random.seed(seed)

    predictions: list[PredictedOutput] = []
    for case in cases:
        if use_mock:
            pred = _mock_predict(case, seed=seed)
        else:
            target_model = _resolve_model_for_case(case, model_path, models_by_specialist)
            pred = _real_predict(case, target_model)
        predictions.append(pred)

    return compute_metrics(cases, predictions, seed=seed)


def discover_specialist_models(models_dir: str) -> dict[str, str]:
    """
    Resolve per-specialist GGUF paths under a models directory exported by
    `run_gguf_export.sh`. Convention:
        <models_dir>/<specialist>-E4B-Q4_K_M_gguf/gemma-4-e4b-it.Q4_K_M.gguf
    """
    resolved: dict[str, str] = {}
    base = os.path.abspath(models_dir)
    for specialist in SPECIALISTS:
        candidate = os.path.join(
            base,
            f"{specialist}-E4B-Q4_K_M_gguf",
            "gemma-4-e4b-it.Q4_K_M.gguf",
        )
        if not os.path.exists(candidate):
            raise RuntimeError(
                f"Missing GGUF for specialist {specialist!r}: {candidate}"
            )
        resolved[specialist] = candidate
    return resolved


def run_eval(
    model_path: Optional[str],
    use_mock: bool,
    seeds: list[int],
    models_by_specialist: Optional[dict[str, str]] = None,
    use_test_split: bool = False,
    run_logger: Optional[RunLogger] = None,
    output_tag: Optional[str] = None,
) -> AggregatedResult | SeedResult:
    """
    Main eval entry point.

    Loads all test cases, runs inference for each seed, aggregates results,
    and saves to eval/results/.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if use_test_split:
        print(f"Loading HELD-OUT test split from {FORMATTED_DIR}/<specialist>/test.jsonl ...")
        cases = load_all_test_split_cases()
        case_source = "test_split"
    else:
        print(f"Loading test cases from {FIXTURES_DIR} and {EVAL_DATA_DIR} ...")
        cases = load_all_cases()
        case_source = "fixtures+baseline"
    if not cases:
        raise RuntimeError(
            "No test cases loaded. Check that data files exist."
        )
    print(f"  Loaded {len(cases)} cases ({case_source})")
    if run_logger is not None:
        run_logger.merge_manifest(case_source=case_source, n_cases=len(cases))

    if use_mock:
        mode = "MOCK"
    elif models_by_specialist is not None:
        mode = f"REAL (per-specialist: {sorted(models_by_specialist)})"
    else:
        mode = f"REAL ({model_path})"
    print(f"  Inference mode: {mode}")
    print(f"  Seeds: {seeds}\n")

    seed_results: list[SeedResult] = []
    for seed in seeds:
        print(f"  Running seed {seed} ...")
        start_time = time.monotonic()
        result = run_single_seed(
            cases,
            seed=seed,
            model_path=model_path,
            use_mock=use_mock,
            models_by_specialist=models_by_specialist,
        )
        elapsed = time.monotonic() - start_time
        print(f"    Weighted F1={result.weighted_f1:.4f}  RED recall={result.red_recall:.4f}  "
              f"({elapsed:.1f}s)")
        seed_results.append(result)
        if run_logger is not None:
            run_logger.log_event(
                "seed_done",
                seed=seed,
                weighted_f1=result.weighted_f1,
                macro_f1=result.macro_f1,
                red_recall=result.red_recall,
                escalation_rate=result.escalation_rate,
                duration_s=round(elapsed, 3),
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"_{output_tag}" if output_tag else ""
    output_path = RESULTS_DIR / f"run{tag}_{timestamp}.json"

    if len(seeds) > 1:
        aggregated = aggregate_seed_results(seed_results)
        print()
        print_summary_table(aggregated)
        payload = {
            "timestamp": timestamp,
            "model": model_path if model_path else "mock",
            "mode": mode,
            "seeds": seeds,
            "n_cases": len(cases),
            "targets": {
                "weighted_f1": TARGET_F1,
                "red_recall": TARGET_RED_RECALL,
                "chrf": TARGET_CHRF,
            },
            "aggregated": {
                "weighted_f1_mean": aggregated.weighted_f1_mean,
                "weighted_f1_std": aggregated.weighted_f1_std,
                "macro_f1_mean": aggregated.macro_f1_mean,
                "macro_f1_std": aggregated.macro_f1_std,
                "red_recall_mean": aggregated.red_recall_mean,
                "red_recall_std": aggregated.red_recall_std,
                "escalation_rate_mean": aggregated.escalation_rate_mean,
                "escalation_rate_std": aggregated.escalation_rate_std,
            },
            "seed_results": [
                {
                    "seed": r.seed,
                    "weighted_f1": r.weighted_f1,
                    "macro_f1": r.macro_f1,
                    "red_recall": r.red_recall,
                    "escalation_rate": r.escalation_rate,
                    "n_cases": r.n_cases,
                    "per_class": r.per_class_report,
                    "per_specialist": r.per_specialist,
                    "predictions": r.predictions,
                }
                for r in seed_results
            ],
        }
        payload["case_source"] = case_source
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to: {output_path}")
        if run_logger is not None:
            run_logger.attach_result(output_path)
        return aggregated

    # Single seed
    single = seed_results[0]
    print()
    print_summary_table(single)
    payload = {
        "timestamp": timestamp,
        "model": model_path if model_path else "mock",
        "mode": mode,
        "seed": single.seed,
        "n_cases": single.n_cases,
        "targets": {
            "weighted_f1": TARGET_F1,
            "red_recall": TARGET_RED_RECALL,
            "chrf": TARGET_CHRF,
        },
        "weighted_f1": single.weighted_f1,
        "macro_f1": single.macro_f1,
        "red_recall": single.red_recall,
        "escalation_rate": single.escalation_rate,
        "per_class": single.per_class_report,
        "per_specialist": single.per_specialist,
        "predictions": single.predictions,
    }
    payload["case_source"] = case_source
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")
    if run_logger is not None:
        run_logger.attach_result(output_path)
    return single


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_seeds(raw: str) -> list[int]:
    """Parse comma-separated seed list, e.g. '42,137,256'."""
    parts = raw.split(",")
    seeds: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid seed value '{part}' — must be an integer."
            ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed must be provided.")
    return seeds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Marunthagam full evaluation suite. Runs triage_classify() on all "
            "specialist fixture cases, computes P/R/F1 and RED recall, and saves "
            "results to eval/results/."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        metavar="GGUF_PATH",
        help="Path to a single quantised GGUF model file for real inference.",
    )
    model_group.add_argument(
        "--models-dir",
        metavar="MODELS_DIR",
        help=(
            "Directory containing per-specialist GGUFs from run_gguf_export.sh. "
            "Each case is routed to <dir>/<specialist>-E4B-Q4_K_M_gguf/"
            "gemma-4-e4b-it.Q4_K_M.gguf based on its fixture specialist tag."
        ),
    )
    model_group.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use the deterministic mock predictor instead of a real model. "
            "Simulates ~90%% accuracy with seed-specific noise — useful for "
            "testing the eval pipeline before model weights are available."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        metavar="SEED_LIST",
        help=(
            "Comma-separated list of random seeds, e.g. '42,137,256'. "
            "When multiple seeds are given, results are aggregated as mean ± std. "
            "Default: 42."
        ),
    )
    parser.add_argument(
        "--test-split",
        action="store_true",
        help=(
            "Evaluate against the held-out 80/10/10 test split "
            "(training/data/formatted/<specialist>/test.jsonl, 131 rows total) "
            "instead of the small fixture set. Use this for the headline F1 number."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        metavar="TAG",
        help="Optional tag inserted into the result filename (e.g. 'test_split', 'fusion_only_triage').",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        seeds = _parse_seeds(args.seeds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    model_path: Optional[str] = args.model if not args.mock else None
    models_by_specialist: Optional[dict[str, str]] = None
    if args.models_dir:
        try:
            models_by_specialist = discover_specialist_models(args.models_dir)
        except RuntimeError as exc:
            parser.error(str(exc))

    with RunLogger(kind="run_eval", args=args) as logger:
        logger.merge_manifest(
            model_path=model_path,
            models_by_specialist=models_by_specialist,
            seeds=seeds,
            use_test_split=args.test_split,
        )
        try:
            run_eval(
                model_path=model_path,
                use_mock=args.mock,
                seeds=seeds,
                models_by_specialist=models_by_specialist,
                use_test_split=args.test_split,
                run_logger=logger,
                output_tag=args.tag,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            logger.merge_manifest(error_message=str(exc))
            sys.exit(1)


if __name__ == "__main__":
    main()
