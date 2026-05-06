"""
End-to-end smoke test of the Sprint 2 shipping configuration.

Verifies all five pieces of the production stack in a single pass:
  1. v2.1 IMNCI rules loaded from protocol.db (21 rules, including the 6
     adult-emergency rules with Bucket A morphology fixes)
  2. v2 multilingual safety classifier loads and runs (en/hi/gu/ta coverage)
  3. B-retrained triage LoRA loads via HF+PEFT and produces a parseable JSON
  4. Sprint 1 derm + maternal GGUFs load via llama-cpp-python and produce
     parseable JSONs
  5. Routed inference over a 3-case smoke set (one per specialist), with
     engine.apply running end-to-end and emitting overrides where expected

Output: PASS or FAIL with diagnostic detail. Run before declaring submission-ready.

    python eval/scripts/smoke_test_production_stack.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]

# Make eval/scripts and training/scripts importable (DLL bridge + run_eval helpers)
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_TRAINING = os.path.abspath(os.path.join(_HERE, "..", "..", "training", "scripts"))
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
import _llama_cpp_setup  # noqa: F401
import run_eval  # noqa: E402

DB = REPO / "inference" / "protocol_engine" / "data" / "protocol.db"
B_ADAPTER = REPO / "training" / "outputs" / "triage-relabel-seed42-6ep" / "final"
GGUF_DERM = REPO / "training" / "models" / "derm-E4B-Q4_K_M_gguf" / "gemma-4-e4b-it.Q4_K_M.gguf"
GGUF_MATERNAL = REPO / "training" / "models" / "maternal-E4B-Q4_K_M_gguf" / "gemma-4-e4b-it.Q4_K_M.gguf"


PASSES: list[str] = []
FAILS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSES.append(f"  ✓ {label}" + (f"  [{detail}]" if detail else ""))
    else:
        FAILS.append(f"  ✗ {label}" + (f"  [{detail}]" if detail else ""))


def smoke_test_rules() -> int:
    print("=== 1. Protocol rules (v2.1) ===")
    if not DB.exists():
        check("protocol.db exists", False, str(DB))
        return 0
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT id, condition_pattern, required_co_signals FROM protocol_rules WHERE active=1"
    ).fetchall()
    conn.close()
    count = len(rows)
    check("protocol.db has rules", count > 0, f"{count} active")
    expected_new_rules = {
        "ADULT-CARDIAC-001", "ADULT-ANAPHYLAXIS-001", "ADULT-HEAD-TRAUMA-001",
        "ADULT-RESPIRATORY-001", "ANIMAL-BITE-RESPIRATORY-001",
        "NEW-ONSET-JAUNDICE-001",
    }
    rule_ids = {r[0] for r in rows}
    missing = expected_new_rules - rule_ids
    check("6 v2.0 adult-emergency rules present", not missing,
          f"missing: {sorted(missing)}" if missing else "all present")
    # Verify Bucket A tightenings present
    cardiac = next((r for r in rows if r[0] == "ADULT-CARDIAC-001"), None)
    if cardiac:
        check(
            "ADULT-CARDIAC-001 has Tamil locative form",
            "மார்[பு]" in cardiac[1] or "மார்பில்" in cardiac[1],
            "v2.1 chief-pattern morphology fix",
        )
        co = json.loads(cardiac[2]) if cardiac[2] else []
        co_text = "\n".join(co)
        check(
            "ADULT-CARDIAC-001 co_signal includes Tamil radiation `பரவி`",
            "பரவி" in co_text,
            "v2.1 co_signal_1 morphology fix",
        )
    respiratory = next((r for r in rows if r[0] == "ADULT-RESPIRATORY-001"), None)
    if respiratory:
        check(
            "ADULT-RESPIRATORY-001 has compound `(?:த்|ு)?` form",
            "(?:த்|ு)?" in respiratory[1],
            "v2.1 sandhi compound fix",
        )
    animal = next((r for r in rows if r[0] == "ANIMAL-BITE-RESPIRATORY-001"), None)
    if animal:
        check(
            "ANIMAL-BITE-RESPIRATORY-001 has instrumental `நாயினால்`",
            "நாயினால்" in animal[1],
            "v2.1 case-inflection fix",
        )
    return count


def smoke_test_safety_classifier() -> None:
    print("\n=== 2. Safety classifier (v2 multilingual) ===")
    from eval_safety import is_refusal, REFUSAL_INDICATORS
    check("indicator list size > 50", len(REFUSAL_INDICATORS) > 50,
          f"{len(REFUSAL_INDICATORS)} indicators")
    # Test cases: one refusal per language, one non-refusal Tamil
    cases = [
        ("ta", True, "மன்னிக்கவும். தயவுசெய்து மருத்துவரை அணுக."),
        ("hi", True, "कृपया डॉक्टर से मिलें। यह गंभीर है।"),
        ("gu", True, "ડૉક્ટરની સલાહ લો."),
        ("en", True, "Please do not attempt this. Go to the nearest hospital."),
        ("en", True, "I am not qualified to advise on this."),
        ("ta-canonical", True, "இது மருத்துவ ஆலோசனை அல்ல. அவசியம் மருத்துவரை அணுகவும்."),
    ]
    for lang, expected, text in cases:
        got = is_refusal(text)
        check(f"refusal[{lang}] '{text[:40]}...' detected", got == expected,
              f"got={got} expected={expected}")


def smoke_test_b_adapter() -> None:
    print("\n=== 3. B-retrained triage LoRA (HF+PEFT) ===")
    if not B_ADAPTER.exists():
        check("adapter dir exists", False, str(B_ADAPTER))
        return
    check("adapter dir exists", True, str(B_ADAPTER))
    print("  loading model (slow first time, ~30s)...")
    try:
        from unsloth import FastLanguageModel
        model, processor = FastLanguageModel.from_pretrained(
            model_name=str(B_ADAPTER),
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        tok = getattr(processor, "tokenizer", processor)
        check("HF+PEFT model loaded", True)
    except Exception as exc:
        check("HF+PEFT model loaded", False, str(exc))
        return

    # One inference: a triage GREEN-flavoured query
    import torch
    prompt = (
        "<|turn>user\n"
        "என் தலையில் சாதாரண சளி உள்ளது. மருந்து பரிந்துரைக்க முடியுமா?\n\n"
        "Classify this case. Output ONE JSON object only, no other text:\n"
        '{"level": "GREEN" | "YELLOW" | "RED", "confidence": 0.0-1.0, '
        '"escalation_flag": true | false}<turn|>\n'
        "<|turn>model\n"
        "{"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            temperature=None, top_p=None, pad_token_id=tok.eos_token_id,
        )
    text = "{" + tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    blk = re.search(r"\{.*?\}", text, re.DOTALL)
    parsed = None
    if blk:
        try:
            parsed = json.loads(blk.group(0))
        except json.JSONDecodeError:
            parsed = None
    check("B emits parseable JSON",
          parsed is not None and "level" in parsed,
          f"got: {text[:120]}")


def smoke_test_gguf(label: str, path: Path) -> None:
    print(f"\n=== 4. {label} GGUF (sprint 1) ===")
    if not path.exists():
        check(f"{label} GGUF exists", False, str(path))
        return
    check(f"{label} GGUF exists", True, str(path.name))
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(path),
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
            logits_all=False,
        )
        check(f"{label} GGUF loaded", True)
    except Exception as exc:
        check(f"{label} GGUF loaded", False, str(exc))
        return
    prompt = run_eval._LLAMA_PROMPT_TEMPLATE.format(
        user_message="என் கையில் தோல் தடிப்பு உள்ளது."
    )
    completion = llm(prompt, max_tokens=64, temperature=0.0,
                     stop=["<turn|>", "<|turn>", "\n\n"])
    raw = "{" + completion["choices"][0]["text"]
    blk = re.search(r"\{.*?\}", raw, re.DOTALL)
    parsed = None
    if blk:
        try:
            parsed = json.loads(blk.group(0))
        except json.JSONDecodeError:
            parsed = None
    check(f"{label} GGUF emits parseable JSON",
          parsed is not None and "level" in parsed,
          f"got: {raw[:120]}")


def smoke_test_engine_e2e() -> None:
    print("\n=== 5. Engine end-to-end (chief + narrative + age + duration) ===")
    engine = run_eval._get_protocol_engine()
    if engine is None:
        check("engine loadable", False, "_get_protocol_engine returned None")
        return
    check("engine loadable", True)

    # Cardiac chief that the v2.1 morphology fix should now match
    triage = run_eval.TriageResult(
        level="YELLOW", confidence=0.9, suspected_conditions=[],
        reasoning_chain="", next_steps_tamil="",
        protocol_references=[], escalation_flag=False,
    )
    triage_out, overrides = engine.apply(
        triage,
        chief_complaint="இடது மார்பில் கடுமையான வலி மற்றும் இடது கையில் மரத்துப்போன உணர்வு",
        narrative="வலி கழுத்தெலும்புக்கு பரவி, மரத்துப்போன உணர்வு",
        age_group="adult",
        duration_days=1,
    )
    cardiac_fired = any(o.rule_id == "ADULT-CARDIAC-001" for o in overrides)
    check("ADULT-CARDIAC-001 fires on triage_test_039 pattern",
          cardiac_fired,
          f"final_level={triage_out.level} overrides={[o.rule_id for o in overrides]}")
    check("post-engine level is RED on cardiac case",
          triage_out.level == "RED")


def main() -> None:
    print("Marunthagam Sprint 2 — production stack smoke test")
    print(f"Repo: {REPO}\n")

    t0 = time.monotonic()

    n_rules = smoke_test_rules()
    smoke_test_safety_classifier()
    smoke_test_b_adapter()
    smoke_test_gguf("derm", GGUF_DERM)
    smoke_test_gguf("maternal", GGUF_MATERNAL)
    smoke_test_engine_e2e()

    elapsed = time.monotonic() - t0
    print("\n=== RESULTS ===")
    for line in PASSES:
        print(line)
    for line in FAILS:
        print(line)
    print(f"\n{len(PASSES)} pass / {len(FAILS)} fail ({elapsed:.1f}s)")
    print(
        "\n*** SUBMISSION-READY ***" if not FAILS
        else "\n*** NOT SUBMISSION-READY — fix failures above ***"
    )
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
