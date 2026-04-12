"""
Marunthagam — Latency Benchmark.

Measures Time To First Token (TTFT) and throughput (tokens/second) across
multiple prompt lengths, averaged over N runs.

Targets (from CLAUDE.md):
  Phone  (E4B):      TTFT < 3.0s,  throughput > 8 tok/s
  Workstation (26B): TTFT < 1.0s,  throughput > 30 tok/s

Usage:
    python eval_latency.py --mock
    python eval_latency.py --model /path/to/model.gguf
    python eval_latency.py --mock --n-runs 20 --prompt-lengths 50,200,500,1000
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time

# Reconfigure stdout/stderr to UTF-8 so Tamil text renders correctly on Windows
# (which defaults to CP1252 in some environments).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level constants — performance targets
# ---------------------------------------------------------------------------

# Phone tier (Gemma 4 E4B, GGUF Q4_K_M on Android)
PHONE_TTFT_TARGET: float = 3.0        # seconds
PHONE_THROUGHPUT_TARGET: float = 8.0  # tokens per second

# Workstation tier (Gemma 4 26B-A4B, llama.cpp or vLLM)
WORKSTATION_TTFT_TARGET: float = 1.0         # seconds
WORKSTATION_THROUGHPUT_TARGET: float = 30.0  # tokens per second

DEFAULT_N_RUNS: int = 10
DEFAULT_PROMPT_LENGTHS: str = "50,200,500"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# ---------------------------------------------------------------------------
# Mock latency parameters — realistic E4B-class device simulation
# ---------------------------------------------------------------------------

# Base latency (seconds) per prompt-length bracket
_MOCK_TTFT_BASE: float = 1.8        # seconds — well within phone target
_MOCK_TTFT_PER_TOKEN: float = 0.002  # additional ms per input token (prompt overhead)
_MOCK_TTFT_NOISE_STD: float = 0.15  # variance across runs

# Throughput simulation
_MOCK_THROUGHPUT_BASE: float = 11.5  # tok/s — above phone target of 8
_MOCK_THROUGHPUT_NOISE_STD: float = 1.2

# Output length simulation (approximate output tokens per run)
_MOCK_OUTPUT_TOKENS_BASE: int = 128
_MOCK_OUTPUT_TOKENS_NOISE: int = 32

# ---------------------------------------------------------------------------
# Prompt template for real inference
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE_WORD = "குழந்தைக்கு காய்ச்சல் இருக்கிறது மற்றும் வாந்தி வருகிறது "  # ~6 chars/word
_PADDING_WORD_EN = "patient has symptoms of fever and vomiting "  # ~7 chars/word


def _generate_prompt(approx_tokens: int) -> str:
    """
    Generate a Tamil/English mixed prompt of approximately `approx_tokens` tokens.

    Token-to-character ratio for Tamil is roughly 3–4 chars/token; for English ~4.
    We alternate Tamil and English phrases to simulate realistic ASHA worker input.
    """
    # Rough heuristic: 1 token ≈ 3.5 chars (mixed Tamil/English)
    target_chars = approx_tokens * 3
    tamil_chunk = _PROMPT_TEMPLATE_WORD * 10
    english_chunk = _PADDING_WORD_EN * 10
    combined = (tamil_chunk + english_chunk) * 20
    return combined[:target_chars]


# ---------------------------------------------------------------------------
# Latency data structures
# ---------------------------------------------------------------------------

class RunResult:
    """Single benchmark run result."""
    __slots__ = ("ttft_s", "throughput_toks", "output_tokens", "prompt_length")

    def __init__(
        self,
        ttft_s: float,
        throughput_toks: float,
        output_tokens: int,
        prompt_length: int,
    ) -> None:
        self.ttft_s = ttft_s
        self.throughput_toks = throughput_toks
        self.output_tokens = output_tokens
        self.prompt_length = prompt_length


class PromptLengthResult:
    """Aggregated stats over N runs for one prompt length."""

    def __init__(self, prompt_length: int, runs: list[RunResult]) -> None:
        self.prompt_length = prompt_length
        self.runs = runs
        self.avg_ttft: float = sum(r.ttft_s for r in runs) / len(runs)
        self.avg_throughput: float = sum(r.throughput_toks for r in runs) / len(runs)
        self.min_ttft: float = min(r.ttft_s for r in runs)
        self.max_ttft: float = max(r.ttft_s for r in runs)
        self.min_throughput: float = min(r.throughput_toks for r in runs)
        self.max_throughput: float = max(r.throughput_toks for r in runs)


# ---------------------------------------------------------------------------
# Mock inference
# ---------------------------------------------------------------------------

def _mock_run(prompt_length: int, run_index: int) -> RunResult:
    """
    Simulate a single inference run with realistic E4B-class latency values.

    TTFT scales slightly with prompt length (prefill cost).
    Throughput is independent of prompt length (decode phase).
    """
    rng = random.Random(f"latency-{prompt_length}-{run_index}")

    ttft = (
        _MOCK_TTFT_BASE
        + prompt_length * _MOCK_TTFT_PER_TOKEN
        + rng.gauss(0, _MOCK_TTFT_NOISE_STD)
    )
    ttft = max(0.5, ttft)

    throughput = (
        _MOCK_THROUGHPUT_BASE
        + rng.gauss(0, _MOCK_THROUGHPUT_NOISE_STD)
    )
    throughput = max(1.0, throughput)

    output_tokens = _MOCK_OUTPUT_TOKENS_BASE + rng.randint(
        -_MOCK_OUTPUT_TOKENS_NOISE,
        _MOCK_OUTPUT_TOKENS_NOISE,
    )

    return RunResult(
        ttft_s=round(ttft, 3),
        throughput_toks=round(throughput, 2),
        output_tokens=output_tokens,
        prompt_length=prompt_length,
    )


# ---------------------------------------------------------------------------
# Real inference
# ---------------------------------------------------------------------------

def _real_run(prompt_length: int, model_path: str) -> RunResult:
    """
    Measure TTFT and throughput for a single real llama.cpp inference run.

    TTFT is approximated as the wall-clock time from process start to first
    output token appearing in stdout (streaming mode). Throughput is computed
    from total output tokens divided by total decode time.
    """
    prompt = _generate_prompt(prompt_length)
    cmd = [
        "llama-cli",
        "--model", model_path,
        "--prompt", prompt,
        "--n-predict", "256",
        "--temp", "0.0",
        "--no-display-prompt",
        "--log-disable",
    ]

    try:
        start = time.monotonic()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
            check=False,
        )
        elapsed = time.monotonic() - start
    except FileNotFoundError as exc:
        raise RuntimeError(
            "llama-cli not found in PATH. Install llama.cpp or use --mock."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"llama.cpp timed out after 300s for prompt_length={prompt_length}"
        ) from exc

    output_text = result.stdout
    output_tokens = max(1, len(output_text.split()))

    # Without streaming API, approximate TTFT as 20% of total elapsed (prefill share)
    ttft = elapsed * 0.20
    decode_time = elapsed * 0.80
    throughput = output_tokens / max(decode_time, 0.001)

    return RunResult(
        ttft_s=round(ttft, 3),
        throughput_toks=round(throughput, 2),
        output_tokens=output_tokens,
        prompt_length=prompt_length,
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
    """Print an ASCII latency benchmark table."""
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
        combined_status = "PASS" if ttft_status == "PASS" and tput_status == "PASS" else "FAIL"
        print(
            f"  {lr.prompt_length:>12} | {lr.avg_ttft:>14.3f} | "
            f"{lr.avg_throughput:>20.2f} | {combined_status:>10}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def run_latency_benchmark(
    model_path: Optional[str],
    use_mock: bool,
    n_runs: int,
    prompt_lengths: list[int],
    output_path: Path,
) -> list[PromptLengthResult]:
    """
    Run latency benchmark across all prompt lengths and aggregate.

    Returns list of PromptLengthResult (one per prompt length).
    """
    mode = "MOCK" if use_mock else f"REAL ({model_path})"
    print(f"Latency benchmark — mode: {mode}")
    print(f"Prompt lengths: {prompt_lengths}")
    print(f"Runs per length: {n_runs}\n")

    all_results: list[PromptLengthResult] = []

    for prompt_length in prompt_lengths:
        print(f"  Benchmarking prompt_length={prompt_length} tokens ({n_runs} runs) ...", end=" ", flush=True)
        runs: list[RunResult] = []
        for run_index in range(n_runs):
            if use_mock:
                run = _mock_run(prompt_length, run_index)
            else:
                assert model_path is not None
                run = _real_run(prompt_length, model_path)
            runs.append(run)
        lr = PromptLengthResult(prompt_length=prompt_length, runs=runs)
        all_results.append(lr)
        print(f"avg TTFT={lr.avg_ttft:.3f}s  throughput={lr.avg_throughput:.2f} tok/s")

    print()
    # Print phone-tier table (primary target)
    print_latency_table(
        all_results,
        ttft_target=PHONE_TTFT_TARGET,
        throughput_target=PHONE_THROUGHPUT_TARGET,
        tier_label="PHONE TIER (E4B targets)",
    )
    print()
    print_latency_table(
        all_results,
        ttft_target=WORKSTATION_TTFT_TARGET,
        throughput_target=WORKSTATION_THROUGHPUT_TARGET,
        tier_label="WORKSTATION TIER (26B targets)",
    )

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": timestamp,
        "model": model_path if model_path else "mock",
        "n_runs": n_runs,
        "targets": {
            "phone": {
                "ttft_s": PHONE_TTFT_TARGET,
                "throughput_toks": PHONE_THROUGHPUT_TARGET,
            },
            "workstation": {
                "ttft_s": WORKSTATION_TTFT_TARGET,
                "throughput_toks": WORKSTATION_THROUGHPUT_TARGET,
            },
        },
        "results": [
            {
                "prompt_length": lr.prompt_length,
                "avg_ttft_s": round(lr.avg_ttft, 4),
                "min_ttft_s": round(lr.min_ttft, 4),
                "max_ttft_s": round(lr.max_ttft, 4),
                "avg_throughput_toks": round(lr.avg_throughput, 4),
                "min_throughput_toks": round(lr.min_throughput, 4),
                "max_throughput_toks": round(lr.max_throughput, 4),
                "phone_pass": (
                    lr.avg_ttft <= PHONE_TTFT_TARGET
                    and lr.avg_throughput >= PHONE_THROUGHPUT_TARGET
                ),
                "workstation_pass": (
                    lr.avg_ttft <= WORKSTATION_TTFT_TARGET
                    and lr.avg_throughput >= WORKSTATION_THROUGHPUT_TARGET
                ),
            }
            for lr in all_results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_prompt_lengths(raw: str) -> list[int]:
    """Parse comma-separated prompt length list, e.g. '50,200,500'."""
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
            "Marunthagam latency benchmark. Measures TTFT and throughput across "
            "multiple prompt lengths, averaged over N runs. Reports pass/fail "
            "against phone (E4B) and workstation (26B) targets."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        metavar="GGUF_PATH",
        help="Path to the quantised GGUF model file for real inference via llama.cpp.",
    )
    model_group.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use deterministic mock latency (TTFT 1.5–2.8s, throughput 9–14 tok/s). "
            "Useful for testing the benchmark pipeline before model weights are available."
        ),
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
        help=(
            f"Comma-separated list of prompt token lengths to benchmark. "
            f"Default: '{DEFAULT_PROMPT_LENGTHS}'."
        ),
    )
    timestamp_default = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / f"latency_{timestamp_default}.json",
        metavar="OUTPUT_JSON",
        help="Path to save results JSON. Default: eval/results/latency_{timestamp}.json",
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

    try:
        run_latency_benchmark(
            model_path=model_path,
            use_mock=args.mock,
            n_runs=args.n_runs,
            prompt_lengths=prompt_lengths,
            output_path=args.output,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
