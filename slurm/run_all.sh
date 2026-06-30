#!/usr/bin/env bash
# Submit the full SARizard pipeline as a SLURM dependency chain.
# Run this once from the repo root; every stage is resumable (existing outputs are skipped).
#
# Required before running:
#   export SURROGATE_CSV=/path/to/protacdb2.0_zinc_chembl_dataset.csv
#
# Optional overrides (all have defaults in slurm/env.sh):
#   export REPO_DIR=/path/to/SARizard   # default: directory containing this script
#   export MAIN_ENV=sarizard             # default: sarizard
#   export ACCELERATOR=gpu               # default: gpu
#
# Usage:
#   export SURROGATE_CSV=/data/protacdb2.0_zinc_chembl_dataset.csv
#   bash slurm/run_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# validate SURROGATE_CSV before submitting anything
if [[ -z "${SURROGATE_CSV:-}" || "${SURROGATE_CSV:-}" == EDIT_* ]]; then
    echo "ERROR: SURROGATE_CSV is not set" >&2
    echo "  export SURROGATE_CSV=/path/to/protacdb2.0_zinc_chembl_dataset.csv" >&2
    echo "  Download from the URL in pretraining/features/surrogate_target.py" >&2
    exit 1
fi

# generate per-flavor finetuning configs from the baseline templates now so we can count
# the recipes for the finetune array; generation reads only flavor metadata and baseline
# YAMLs, so it works before pretrain runs and before the foundation files exist
echo "generating per-flavor finetuning configs..."
conda run -n "$MAIN_ENV" python -m configs.generate
N_RECIPES=$(ls "$REPO_DIR"/configs/*/*.yaml 2>/dev/null | grep -v '/_baseline/' | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: configs.generate produced no recipes; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"
echo ""

# submit stages in order; each stage depends on the previous completing without errors
# (afterok waits for ALL array tasks to succeed before releasing the next stage)

JOB_CORPUS=$(sbatch --parsable "$SCRIPT_DIR/prepare_corpus.sbatch")
echo "corpus     job=$JOB_CORPUS"

JOB_TARGETS=$(sbatch --parsable \
    --dependency=afterok:"$JOB_CORPUS" \
    --export=ALL,SURROGATE_CSV="$SURROGATE_CSV" \
    "$SCRIPT_DIR/compute_targets.sbatch")
echo "targets    job=$JOB_TARGETS  (after corpus $JOB_CORPUS)"

JOB_PRETRAIN=$(sbatch --parsable \
    --dependency=afterok:"$JOB_TARGETS" \
    "$SCRIPT_DIR/pretrain.sbatch")
echo "pretrain   job=$JOB_PRETRAIN  (after targets $JOB_TARGETS)"

JOB_FINETUNE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_PRETRAIN" \
    --array=0-$((N_RECIPES - 1)) \
    "$SCRIPT_DIR/finetune.sbatch")
echo "finetune   job=$JOB_FINETUNE  (after pretrain $JOB_PRETRAIN)"

JOB_ANALYZE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_FINETUNE" \
    "$SCRIPT_DIR/analyze.sbatch")
echo "analyze    job=$JOB_ANALYZE  (after finetune $JOB_FINETUNE)"

echo ""
echo "all stages submitted; monitor with:"
echo "  watch squeue -u \$USER"
echo "  tail -f $REPO_DIR/slurm/logs/analyze_${JOB_ANALYZE}.out"
