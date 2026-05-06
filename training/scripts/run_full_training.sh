#!/usr/bin/env bash
# Sequential LoRA fine-tuning across 3 specialists × 3 seeds = 9 runs.
# Run from training/ directory:
#     bash scripts/run_full_training.sh
set -euo pipefail

PY="C:/Users/mechr/.unsloth/studio/unsloth_studio/Scripts/python"
LOG_DIR="logs"
SEEDS=(42 137 256)
SPECIALISTS=(triage derm maternal)
mkdir -p "$LOG_DIR"

# Disable wandb — its service-startup polling races on Windows and aborts
# training. Metrics are visible in stdout/per-run log files.
export WANDB_DISABLED=true
export WANDB_MODE=disabled

phase() {
    echo
    echo "============================================================"
    echo "$1"
    echo "Started: $(date -Iseconds)"
    echo "============================================================"
}

run=0
for sp in "${SPECIALISTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run=$((run + 1))
        phase "RUN ${run}/9: train ${sp} seed=${seed}"
        "$PY" -u scripts/train_lora.py \
            --config "configs/lora_${sp}.yaml" \
            --seed "$seed" \
            2>&1 | tee "$LOG_DIR/train_${sp}_seed${seed}.log"
    done
done

phase "ALL TRAINING COMPLETE"
echo "Outputs at: outputs/{triage,derm,maternal}-seed{42,137,256}/final"
ls -la outputs/ 2>&1 || true
