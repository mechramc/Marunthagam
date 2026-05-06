#!/usr/bin/env bash
# Export the best-seed checkpoint per specialist to Q4_K_M GGUF for phone deployment.
set -euo pipefail

PY="C:/Users/mechr/.unsloth/studio/unsloth_studio/Scripts/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "models"

# Best seed per specialist (chosen by lowest eval_loss across 3 seeds).
declare -A BEST=(
    [triage]=42
    [derm]=42
    [maternal]=256
)

for sp in triage derm maternal; do
    seed="${BEST[$sp]}"
    ckpt="outputs/${sp}-seed${seed}/final"
    out="models/${sp}-E4B-Q4_K_M"
    echo
    echo "============================================================"
    echo "EXPORT: ${sp} (seed=${seed}) -> ${out}.gguf"
    echo "Started: $(date -Iseconds)"
    echo "============================================================"
    "$PY" -u scripts/export_gguf.py \
        --checkpoint "$ckpt" \
        --output "$out" \
        --quantization q4_k_m \
        2>&1 | tee "$LOG_DIR/export_${sp}.log"
done

echo
echo "GGUF EXPORT COMPLETE"
ls -la models/*.gguf 2>&1 || true
