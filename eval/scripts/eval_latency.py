"""
Marunthagam — Latency Benchmark.

Measures Time To First Token (TTFT) and decode throughput (tokens/second)
across multiple prompt lengths, averaged over N runs.

Real inference uses llama-cpp-python in streaming mode, so TTFT is the
wall-clock delta from sending the prompt to the first generated token —
not a guess based on a fraction of total elapsed time.

Targets (from CLAUDE.md):
  Phone  (E4B):      TTFT < 3.0s,  throughput > 8 tok/s
  Workstation (26B): TTFT < 1.0s,  throughput > 30 tok/s

Usage:
    python eval_latency.py --mock
    python eval_latency.py --model /path/to/model.gguf
    python eval_latency.py --models-dir training/models
    python eval_latency.py --mock --n-runs 20 --prompt-lengths 50,200,500,1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make scripts/_llama_cpp_setup importable so the cu12 DLL dirs are registered
# before any llama_cpp import. Safe no-op on non-Windows hosts.
_TRAINING_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "training", "scripts")
)
if _TRAINING_SCRIPTS not in sys.path:
    sys.path.insert(0, _TRAINING_SCRIPTS)
import _llama_cpp_setup  # noqa: F401, E402  -- registers cu12 DLL dirs on Windows

# Reconfigure stdout/stderr to UTF-8 so Tamil text renders correctly on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from run_logger import RunLogger  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants — performance targets
# ---------------------------------------------------------------------------

# Phone tier (Gemma 4 E4B, GGUF Q4_K_M on Android)
PHONE_TTFT_TARGET: float = 3.0        # seconds
PHONE_THROUGHPUT_TARGET: float = 8.0  # tokens per second

# Workstation tier
WORKSTATION_TTFT_TARGET: float = 1.0
WORKSTATION_THROUGHPUT_TARGET: float = 30.0

DEFAULT_N_RUNS: int = 5
DEFAULT_PROMPT_LENGTHS: str = "50,200,500"
DEFAULT_GENERATE_TOKENS: int = 128

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "eval" / "results"

SPECIALISTS: list[str] = ["triage", "derm", "maternal"]

# Mock latency parameters — realistic E4B-class device simulation
_MOCK_TTFT_BASE: float = 1.8
_MOCK_TTFT_PER_TOKEN: float = 0.002
_MOCK_TTFT_NOISE_STD: float = 0.15
_MOCK_THROUGHPUT_BASE: float = 11.5
_MOCK_THROUGHPUT_NOISE_STD: float = 1.2
_MOCK_OUTPUT_TOKENS_BASE: int = 128
_MOCK_OUTPUT_TOKENS_NOISE: int = 32

_PROMPT_TEMPLATE_TAMIL = "குழந்தைக்கு காய்ச்சல் இருக்கிறது மற்றும் வாந்தி வருகிறது "
_PROMPT_TEMPLATE_EN = "patient has symptoms of fever and vomiting "


def _generate_prompt(approx_tokens: int) -> str:
    """Generate a Tamil/English mixed prompt of approximately N tokens (~3 chars/token)."""
    target_chars = approx_tokens * 3
    tamil_chunk = _PROMPT_TEMPLATE_TAMIL * 10
    english_chunk = _PROMPT_TEMPLATE_EN * 10
    combined = (tamil_chunk + english_chunk) * 20
    return combined[:target_chars]


# ---------------------------------------------------------------------------
# Latency data structures
# ---------------------------------------------------------------------------

class RunResult:
    __slots__ = ("ttft_s", "throughput_toks", "output_tokens", "prompt_length", "total_s")

    def __init__(
        self,
        ttft_s: float,
        throughput_toks: float,
        output_tokens: int,
        prompt_length: int,
        total_s: float,
    ) -> None:
        self.ttft_s = ttft_s
        self.throughput_toks = throughput_toks
        self.output_tokens = output_tokens
        self.prompt_length = prompt_length
        self.total_s = total_s


class PromptLengthResult:
    """Aggregated stats over N runs for one prompt length."""

    def __init__(self, prompt_length: int, runs: list[RunResult]) -> None:
        self.prompt_length = prompt_length
        self.runs = runs
        ttfts = [r.ttft_s for r in runs]
        tputs = [r.throughput_toks for r in runs]
        self.avg_ttft = sum(ttfts) / len(runs)
        self.avg_throughput = sum(tputs) / len(runs)
        self.min_ttft = min(ttfts)
        self.max_ttft = max(ttfts)
        self.min_throughput = min(tputs)
        self.max_throughput = max(tputs)
        self.median_ttft = sorted(ttfts)[len(ttfts) // 2]
        self.median_throughput = sorted(tputs)[len(tputs) // 2]


# ---------------------------------------------------------------------------
# Mock inference
# ---------------------------------------------------------------------------

def _mock_run(prompt_length: int, run_index: int) -> RunResult:
    rng = random.Random(f"latency-{prompt_length}-{run_index}")
    ttft = (
        _MOCK_TTFT_BASE
        + prompt_length * _MOCK_TTFT_PER_TOKEN
        + rng.gauss(0, _MOCK_TTFT_NOISE_STD)
    )
    ttft = max(0.5, ttft)
    throughput = _MOCK_THROUGHPUT_BASE + rng.gauss(0, _MOCK_THROUGHPUT_NOISE_STD)
    throughput = max(1.0, throughput)
    output_tokens = _MOCK_OUTPUT_TOKENS_BASE + rng.randint(
        -_MOCK_OUTPUT_TOKENS_NOISE, _MOCK_OUTPUT_TOKENS_NOISE,
    )
    decode_s = output_tokens / throughput
    return RunResult(
        ttft_s=round(ttft, 3),
        throughput_toks=round(throughput, 2),
        output_tokens=output_tokens,
        prompt_length=prompt_length,
        total_s=round(ttft + decode_s, 3),
    )


# ---------------------------------------------------------------------------
# Real inference (llama-cpp-python streaming, real TTFT)
# ---------------------------------------------------------------------------

_LLM_CACHE: dict[str, "Llama"] = {}


def _get_llm(model_path: str) -> "Llama":
    abs_path = os.path.abspath(model_path)
    cached = _LLM_CACHE.get(abs_path)
    if cached is not None:
        return cached
    from llama_cpp import Llama  # local import: avoids cost in --mock mode
    llm = Llama(
        model_path=abs_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )
    _LLM_CACHE[abs_path] = llm
    return llm


def _real_run(
    prompt_length: int,
    model_path: str,
    max_tokens: int,
) -> RunResult:
    """
    Measure TTFT and decode throughput using llama-cpp-python streaming.

    TTFT = wall-clock delta from llm(...) call to first token chunk.
    Throughput = (output_tokens - 1) / (total_elapsed - ttft).
    """
    prompt = _generate_prompt(prompt_length)
    llm = _get_llm(model_path)

    start = time.monotonic()
    first_token_t: Optional[float] = None
    output_tokens = 0
    output_text_parts: list[str] = []

    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    for chunk in stream:
        if first_token_t is None:
            first_token_t = time.monotonic()
        output_tokens += 1
        try:
            output_text_parts.append(chunk["choices"][0]["text"])
        except (KeyError, IndexError, TypeError):
            pass

    end = time.monotonic()

    if first_token_t is None or output_tokens == 0:
        # Model emitted nothing — record a degenerate run.
        return RunResult(
            ttft_s=round(end - start, 3),
            throughput_toks=0.0,
            output_tokens=0,
            prompt_length=prompt_length,
            total_s=round(end - start, 3),
        )

    ttft = first_token_t - start
    decode_time = max(end - first_token_t, 1e-3)
    # Throughput counts the decode-only tokens (excluding the first-token prefill cost).
    throughput = max(output_tokens - 1, 1) / decode_time

    return RunResult(
        ttft_s=round(ttft, 4),
        throughput_toks=round(throughput, 2),
        output_tokens=output_tokens,
        prompt_length=prompt_length,
        total_s=round(end - start, 4),
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _pass_fail_ttft(avg_ttft: float, target: float) -> str:
    return "PASS" if avg_ttft <= target else "FAIL"


def _pass_fail_throughput(avg_throughput: float, target: float) -> str:
    return "PASS" if avg_throughput >= target else "FAIL"


def print_latency_table(
    length_results: list[PromptLengthResult],
    ttft_target: float,
    throughput_target: float,
    tier_label: str,
) -> None:
    sep = "─" * 76
    print(sep)
    print(f"  MARUNTHAGAM LATENCY BENCHMARK — {tier_label}")
    print(f"  Targets: TTFT < {ttft_target}s  |  Throughput > {throughput_target} tok/s")
    print(sep)
    header = (
        f"  {'Prompt Len':>12} | {'Avg TTFT (s)':>14} | "
        f"{'Throughput (tok/s)':>20} | {'Status':>10}"
    )
    print(header)
    print(f"  {'─'*12}-+-{'─'*14}-+-{'─'*20}-+-{'─'*10}")
    for lr in length_results:
        ttft_status = _pass_fail_ttft(lr.avg_ttft, ttft_target)
        tput_status = _pass_fail_throughput(lr.avg_throughput, throughput_target)
        combined = "PASS" if ttft_status == "PASS" and tput_status == "PASS" else "FAIL"
        print(
            f"  {lr.prompt_length:>12} | {lr.avg_ttft:>14.3f} | "
            f"{lr.avg_throughput:>20.2f} | {combined:>10}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def discover_specialist_models(models_dir: str) -> dict[str, str]:
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


def run_latency_benchmark(
    model_path: Optional[str],
    use_mock: bool,
    n_runs: int,
    prompt_lengths: list[int],
    output_path: Path,
    max_tokens: int = DEFAULT_GENERATE_TOKENS,
    models_by_specialist: Optional[dict[str, str]] = None,
    run_logger: Optional[RunLogger] = None,
) -> dict[str, list[PromptLengthResult]]:
    """
    Run latency benchmark across prompt lengths. Returns dict
    keyed by model label (e.g. 'triage', or the model basename).
    """
    if use_mock:
        mode = "MOCK"
    elif models_by_specialist is not None:
        mode = f"REAL (per-specialist: {sorted(models_by_specialist)})"
    else:
        mode = f"REAL ({model_path})"
    print(f"Latency benchmark — mode: {mode}")
    print(f"Prompt lengths: {prompt_lengths}")
    print(f"Runs per length: {n_runs}")
    print(f"Max generate tokens: {max_tokens}\n")

    # Build (label, model_path) pairs we'll bench.
    benchmarks: list[tuple[str, Optional[str]]] = []
    if use_mock:
        benchmarks = [("mock", None)]
    elif models_by_specialist is not None:
        for spec in SPECIALISTS:
            benchmarks.append((spec, models_by_specialist[spec]))
    else:
        assert model_path is not None
        benchmarks = [(Path(model_path).stem, model_path)]

    all_by_model: dict[str, list[PromptLengthResult]] = {}
    payload_records: list[dict] = []

    for label, mpath in benchmarks:
        print(f"  ─── Benchmarking model: {label} ───")
        per_length: list[PromptLengthResult] = []
        for prompt_length in prompt_lengths:
            print(
                f"    prompt_length={prompt_length} ({n_runs} runs) ...",
                end=" ", flush=True,
            )
            runs: list[RunResult] = []
            for run_index in range(n_runs):
                if use_mock:
                    run = _mock_run(prompt_length, run_index)
                else:
                    assert mpath is not None
                    run = _real_run(prompt_length, mpath, max_tokens)
                runs.append(run)
                if run_logger is not None:
                    run_logger.log_event(
                        "latency_run",
                        model=label,
                        prompt_length=prompt_length,
                        run_index=run_index,
                        ttft_s=run.ttft_s,
                        throughput_toks=run.throughput_toks,
                        output_tokens=run.output_tokens,
                    )
            lr = PromptLengthResult(prompt_length=prompt_length, runs=runs)
            per_length.append(lr)
            print(
                f"avg TTFT={lr.avg_ttft:.3f}s  throughput={lr.avg_throughput:.2f} tok/s "
                f"(median TTFT={lr.median_ttft:.3f}s)"
            )
        all_by_model[label] = per_length
        print()

        # Phone-tier and workstation-tier tables for this model.
        print_latency_table(
            per_length,
            ttft_target=PHONE_TTFT_TARGET,
            throughput_target=PHONE_THROUGHPUT_TARGET,
            tier_label=f"PHONE TIER targets · {label}",
        )
        print()
        print_latency_table(
            per_length,
            ttft_target=WORKSTATION_TTFT_TARGET,
            throughput_target=WORKSTATION_THROUGHPUT_TARGET,
            tier_label=f"WORKSTATION TIER targets · {label}",
        )
        print()

        payload_records.append({
            "model_label": label,
            "model_path": mpath,
            "results": [
                {
                    "prompt_length": lr.prompt_length,
                    "avg_ttft_s": round(lr.avg_ttft, 4),
                    "min_ttft_s": round(lr.min_ttft, 4),
                    "max_ttft_s": round(lr.max_ttft, 4),
                    "median_ttft_s": round(lr.median_ttft, 4),
                    "avg_throughput_toks": round(lr.avg_throughput, 4),
                    "min_throughput_toks": round(lr.min_throughput, 4),
                    "max_throughput_toks": round(lr.max_throughput, 4),
                    "median_throughput_toks": round(lr.median_throughput, 4),
                    "n_runs": len(lr.runs),
                    "phone_pass": (
                        lr.avg_ttft <= PHONE_TTFT_TARGET
                        and lr.avg_throughput >= PHONE_THROUGHPUT_TARGET
                    ),
                    "workstation_pass": (
                        lr.avg_ttft <= WORKSTATION_TTFT_TARGET
                        and lr.avg_throughput >= WORKSTATION_THROUGHPUT_TARGET
                    ),
                    "raw_runs": [
                        {
                            "ttft_s": r.ttft_s,
                            "throughput_toks": r.throughput_toks,
                            "output_tokens": r.output_tokens,
                            "total_s": r.total_s,
                        }
                        for r in lr.runs
                    ],
                }
                for lr in per_length
            ],
        })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "mode": mode,
        "n_runs": n_runs,
        "max_tokens": max_tokens,
        "targets": {
            "phone": {"ttft_s": PHONE_TTFT_TARGET, "throughput_toks": PHONE_THROUGHPUT_TARGET},
            "workstation": {
                "ttft_s": WORKSTATION_TTFT_TARGET,
                "throughput_toks": WORKSTATION_THROUGHPUT_TARGET,
            },
        },
        "benchmarks": payload_records,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")
    if run_logger is not None:
        run_logger.attach_result(output_path)

    return all_by_model


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_prompt_lengths(raw: str) -> list[int]:
    parts = raw.split(",")
    lengths: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
            if value <= 0:
                raise ValueError(f"Prompt length must be positive, got {value}")
            lengths.append(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid prompt length '{part}' — must be a positive integer."
            ) from exc
    if not lengths:
        raise argparse.ArgumentTypeError("At least one prompt length must be provided.")
    return sorted(set(lengths))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Marunthagam latency benchmark. Streams llama-cpp-python to measure "
            "real TTFT and decode throughput across prompt lengths. Reports pass/fail "
            "against phone (E4B) and workstation targets."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        metavar="GGUF_PATH",
        help="Path to a single GGUF model file.",
    )
    model_group.add_argument(
        "--models-dir",
        metavar="MODELS_DIR",
        help=(
            "Directory containing per-specialist GGUFs. Each specialist is "
            "benchmarked separately so we can compare triage / derm / maternal."
        ),
    )
    model_group.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock latency for pipeline testing.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        metavar="N",
        help=f"Number of inference runs to average per prompt length. Default: {DEFAULT_N_RUNS}.",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default=DEFAULT_PROMPT_LENGTHS,
        metavar="LENGTHS",
        help=f"Comma-separated prompt token lengths. Default: '{DEFAULT_PROMPT_LENGTHS}'.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_GENERATE_TOKENS,
        metavar="N",
        help=f"Max tokens to generate per run. Default: {DEFAULT_GENERATE_TOKENS}.",
    )
    timestamp_default = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / f"latency_{timestamp_default}.json",
        metavar="OUTPUT_JSON",
        help="Path to save results JSON.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        prompt_lengths = _parse_prompt_lengths(args.prompt_lengths)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    model_path: Optional[str] = args.model if not args.mock else None
    models_by_specialist: Optional[dict[str, str]] = None
    if args.models_dir:
        try:
            models_by_specialist = discover_specialist_models(args.models_dir)
        except RuntimeError as exc:
            parser.error(str(exc))

    with RunLogger(kind="eval_latency", args=args) as logger:
        logger.merge_manifest(
            model_path=model_path,
            models_by_specialist=models_by_specialist,
            prompt_lengths=prompt_lengths,
            n_runs=args.n_runs,
            max_tokens=args.max_tokens,
        )
        try:
            run_latency_benchmark(
                model_path=model_path,
                use_mock=args.mock,
                n_runs=args.n_runs,
                prompt_lengths=prompt_lengths,
                output_path=args.output,
                max_tokens=args.max_tokens,
                models_by_specialist=models_by_specialist,
                run_logger=logger,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            logger.merge_manifest(error_message=str(exc))
            sys.exit(1)


if __name__ == "__main__":
    main()
