#!/usr/bin/env bash
# Sequential translate + label across all three specialists, then format.
# Run from training/ directory:
#     bash scripts/run_full_pipeline.sh
set -euo pipefail

PY="C:/Users/mechr/.unsloth/studio/unsloth_studio/Scripts/python"
MODEL="models/gemma-4-31B-it-Q4_K_M.gguf"
LOG_DIR="logs"
mkdir -p data/reviewed data/formatted "$LOG_DIR"

phase() {
    echo
    echo "============================================================"
    echo "$1"
    echo "Started: $(date -Iseconds)"
    echo "============================================================"
}

for sp in triage derm maternal; do
    phase "TRANSLATE: $sp"
    "$PY" scripts/translate_dataset.py \
        --source "data/raw/${sp}_en.jsonl" \
        --output "data/reviewed/${sp}_pending.jsonl" \
        --model "$MODEL" \
        --batch-size 25 \
        2>&1 | tee "$LOG_DIR/translate_${sp}.log"

    phase "LABEL: $sp"
    "$PY" scripts/label_triage.py \
        --input "data/reviewed/${sp}_pending.jsonl" \
        --output "data/reviewed/${sp}_approved.jsonl" \
        --model "$MODEL" \
        --progress-every 25 \
        2>&1 | tee "$LOG_DIR/label_${sp}.log"
done

phase "FORMAT: all specialists"
for sp in triage derm maternal; do
    echo "--- format $sp ---"
    "$PY" scripts/format_training_data.py \
        --reviewed "data/reviewed/${sp}_approved.jsonl" \
        --specialist "$sp" \
        --output-dir "data/formatted" \
        --seed 42 \
        2>&1 | tee "$LOG_DIR/format_${sp}.log"
done

phase "DONE"
echo "Translated + labeled + formatted at $(date -Iseconds)"
echo "See data/formatted/{triage,derm,maternal}/{train,val,test}.jsonl"
