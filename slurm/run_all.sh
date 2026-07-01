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

# generate per-(flavor, seed) finetuning configs from the baseline templates now so we can
# count the recipes for the finetune array; generation reads only flavor metadata and baseline
# YAMLs, so it works before pretrain runs and before the foundation files exist
echo "generating per-(flavor, seed) finetuning configs (seeds: $FLAVOR_SEEDS)..."
conda run -n "$MAIN_ENV" python -m sarizard.configs.generate --seeds $FLAVOR_SEEDS
N_RECIPES=$(flavor_recipe_list | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: configs.generate produced no recipes; check configs/generate.py" >&2
    exit 1
fi
# size the arrays from the registry and seed set rather than fixed ranges, so adding a flavor
# or a seed needs no edits to the sbatch headers. Targets and split are per flavor (seed
# independent); pretrain is per (flavor, seed).
N_FLAVORS=$(flavor_list | wc -l | tr -d ' ')
if [[ "$N_FLAVORS" -eq 0 ]]; then
    echo "ERROR: flavor registry is empty; check sarizard/pretraining/flavors.py" >&2
    exit 1
fi
N_SEEDS=$(wc -w <<<"$FLAVOR_SEEDS" | tr -d ' ')
N_PRETRAIN=$(( N_FLAVORS * N_SEEDS ))
echo "  $N_FLAVORS flavors (targets/split array 0-$((N_FLAVORS - 1)))"
echo "  $N_FLAVORS x $N_SEEDS seeds = $N_PRETRAIN pretrain tasks (array 0-$((N_PRETRAIN - 1)))"
echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"
echo ""

# submit stages in order; each stage depends on the previous completing without errors.
# afterok partial-failure contract: the dependent stage is released only if EVERY array task
# of the prior stage exits 0. If one flavor or recipe task fails, the next stage is cancelled
# (SLURM marks it DependencyNeverSatisfied) and the rest of the chain never runs. To recover,
# fix the failing task's cause and re-run this script: every stage is resumable (it skips
# flavors/recipes whose outputs already exist), so only the gaps are recomputed and the chain
# is re-armed from there. Inspect a failed task's log under slurm/logs/<stage>_<jobid>_<taskid>.out.

JOB_CORPUS=$(sbatch --parsable "$SCRIPT_DIR/prepare_corpus.sbatch")
echo "corpus     job=$JOB_CORPUS"

JOB_TARGETS=$(sbatch --parsable \
    --dependency=afterok:"$JOB_CORPUS" \
    --array=0-$((N_FLAVORS - 1)) \
    --export=ALL,SURROGATE_CSV="$SURROGATE_CSV" \
    "$SCRIPT_DIR/compute_targets.sbatch")
echo "targets    job=$JOB_TARGETS  (after corpus $JOB_CORPUS)"

JOB_SPLIT=$(sbatch --parsable \
    --dependency=afterok:"$JOB_TARGETS" \
    --array=0-$((N_FLAVORS - 1)) \
    "$SCRIPT_DIR/split.sbatch")
echo "split      job=$JOB_SPLIT  (after targets $JOB_TARGETS)"

JOB_PRETRAIN=$(sbatch --parsable \
    --dependency=afterok:"$JOB_SPLIT" \
    --array=0-$((N_PRETRAIN - 1)) \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS" \
    "$SCRIPT_DIR/pretrain.sbatch")
echo "pretrain   job=$JOB_PRETRAIN  (after split $JOB_SPLIT, $N_PRETRAIN flavor x seed)"

JOB_FINETUNE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_PRETRAIN" \
    --array=0-$((N_RECIPES - 1)) \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS" \
    "$SCRIPT_DIR/finetune.sbatch")
echo "finetune   job=$JOB_FINETUNE  (after pretrain $JOB_PRETRAIN)"

JOB_ANALYZE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_FINETUNE" \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS" \
    "$SCRIPT_DIR/analyze.sbatch")
echo "analyze    job=$JOB_ANALYZE  (after finetune $JOB_FINETUNE)"

echo ""
echo "all stages submitted; monitor with:"
echo "  watch squeue -u \$USER"
echo "  tail -f $REPO_DIR/slurm/logs/analyze_${JOB_ANALYZE}.out"
