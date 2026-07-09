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

# FOUNDATION_SEED set => finetune-only replication: every FLAVOR_SEED finetunes the one
# foundation pretrained at FOUNDATION_SEED, so we skip corpus/targets/split/pretrain (those
# outputs already exist) and submit only finetune + analyze. Unset => the legacy full pipeline.
FINETUNE_ONLY=""
[[ -n "${FOUNDATION_SEED:-}" ]] && FINETUNE_ONLY=1

# validate SURROGATE_CSV before submitting anything; only the targets stage needs it, so a
# finetune-only run (which skips that stage) does not require it
if [[ -z "$FINETUNE_ONLY" && ( -z "${SURROGATE_CSV:-}" || "${SURROGATE_CSV:-}" == EDIT_* ) ]]; then
    echo "ERROR: SURROGATE_CSV is not set" >&2
    echo "  export SURROGATE_CSV=/path/to/protacdb2.0_zinc_chembl_dataset.csv" >&2
    echo "  Download from the URL in pretraining/features/surrogate_target.py" >&2
    exit 1
fi

# generate per-(flavor, seed) finetuning configs from the baseline templates now so we can
# count the recipes for the finetune array; generation reads only flavor metadata and baseline
# YAMLs, so it works before pretrain runs and before the foundation files exist
if [[ -n "$FINETUNE_ONLY" ]]; then
    echo "finetune-only: generating $FLAVOR_SEEDS finetunes of the s$FOUNDATION_SEED foundations..."
    conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
        --seeds $FLAVOR_SEEDS --foundation-seed "$FOUNDATION_SEED"
else
    echo "generating per-(flavor, seed) finetuning configs (seeds: $FLAVOR_SEEDS)..."
    conda run -n "$MAIN_ENV" python -m sarizard.configs.generate --seeds $FLAVOR_SEEDS
fi
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
# finetune-only preflight: every recipe initializes from foundations/<flavor>__s<FOUNDATION_SEED>_mp.pt,
# so gate on all of them existing before submitting, rather than discovering the gap task by task
if [[ -n "$FINETUNE_ONLY" ]]; then
    MISSING=""
    while IFS= read -r flavor; do
        [[ -n "$flavor" ]] || continue
        foundation="foundations/${flavor}__s${FOUNDATION_SEED}_mp.pt"
        [[ -f "$REPO_DIR/$foundation" ]] || MISSING+=" $foundation"
    done < <(flavor_list)
    if [[ -n "$MISSING" ]]; then
        echo "ERROR: finetune-only requires the s$FOUNDATION_SEED foundations; missing:$MISSING" >&2
        exit 1
    fi
    echo "  finetune-only: $N_SEEDS seeds x $N_FLAVORS flavors finetune the s$FOUNDATION_SEED foundations"
    echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"
else
    echo "  $N_FLAVORS flavors (targets/split array 0-$((N_FLAVORS - 1)))"
    echo "  $N_FLAVORS x $N_SEEDS seeds = $N_PRETRAIN pretrain tasks (array 0-$((N_PRETRAIN - 1)))"
    echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"
fi
echo ""

# submit stages in order; each stage depends on the previous completing without errors.
# afterok partial-failure contract: the dependent stage is released only if EVERY array task
# of the prior stage exits 0. If one flavor or recipe task fails, the next stage is cancelled
# (SLURM marks it DependencyNeverSatisfied) and the rest of the chain never runs. To recover,
# fix the failing task's cause and re-run this script: every stage is resumable (it skips
# flavors/recipes whose outputs already exist), so only the gaps are recomputed and the chain
# is re-armed from there. Inspect a failed task's log under slurm/logs/<stage>_<jobid>_<taskid>.out.

# finetune-only skips corpus/targets/split/pretrain (the pinned foundations already exist) and
# submits the finetune array with no upstream dependency; the legacy path builds the chain first.
FINETUNE_DEPENDENCY=""
if [[ -z "$FINETUNE_ONLY" ]]; then
    JOB_CORPUS=$(sbatch --parsable "$SCRIPT_DIR/prepare_corpus.sbatch")
    echo "corpus     job=$JOB_CORPUS"

    JOB_TARGETS=$(sbatch --parsable \
        --dependency=afterok:"$JOB_CORPUS" \
        --array=0-$((N_FLAVORS - 1)) \
        --export=ALL,SURROGATE_CSV="$SURROGATE_CSV" \
        "$SCRIPT_DIR/compute_targets.sbatch")
    echo "targets    job=$JOB_TARGETS  (after corpus $JOB_CORPUS)"

    # a derived flavor (e.g. osmordred_pca80) has no calculator of its own; compute_targets.sbatch
    # skips it, and this single-task stage builds its target.zarr afterward, from the base
    # flavor's raw target the targets stage above just computed. Only submitted when the current
    # registry actually has a derived flavor, so a registry with none skips straight to split.
    SPLIT_DEPENDENCY="afterok:$JOB_TARGETS"
    HAS_DERIVED=$(conda run -n "$MAIN_ENV" python -c "
from sarizard.pretraining.flavors import flavor_names, get_flavor
print('1' if any(get_flavor(f).derived_from for f in flavor_names()) else '')
")
    if [[ -n "$HAS_DERIVED" ]]; then
        JOB_PCA=$(sbatch --parsable \
            --dependency=afterok:"$JOB_TARGETS" \
            "$SCRIPT_DIR/osmordred_pca_targets.sbatch")
        echo "pca-targets job=$JOB_PCA  (after targets $JOB_TARGETS)"
        SPLIT_DEPENDENCY="afterok:$JOB_TARGETS,afterok:$JOB_PCA"
    fi

    JOB_SPLIT=$(sbatch --parsable \
        --dependency="$SPLIT_DEPENDENCY" \
        --array=0-$((N_FLAVORS - 1)) \
        "$SCRIPT_DIR/split.sbatch")
    echo "split      job=$JOB_SPLIT  (after $SPLIT_DEPENDENCY)"

    JOB_PRETRAIN=$(sbatch --parsable \
        --dependency=afterok:"$JOB_SPLIT" \
        --array=0-$((N_PRETRAIN - 1)) \
        --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS" \
        "$SCRIPT_DIR/pretrain.sbatch")
    echo "pretrain   job=$JOB_PRETRAIN  (after split $JOB_SPLIT, $N_PRETRAIN flavor x seed)"
    FINETUNE_DEPENDENCY="--dependency=afterok:$JOB_PRETRAIN"
fi

JOB_FINETUNE=$(sbatch --parsable \
    $FINETUNE_DEPENDENCY \
    --array=0-$((N_RECIPES - 1)) \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS" \
    "$SCRIPT_DIR/finetune.sbatch")
if [[ -n "$FINETUNE_ONLY" ]]; then
    echo "finetune   job=$JOB_FINETUNE  ($N_RECIPES recipes off the s$FOUNDATION_SEED foundations)"
else
    echo "finetune   job=$JOB_FINETUNE  (after pretrain $JOB_PRETRAIN)"
fi

JOB_ANALYZE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_FINETUNE" \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS" \
    "$SCRIPT_DIR/analyze.sbatch")
echo "analyze    job=$JOB_ANALYZE  (after finetune $JOB_FINETUNE)"

echo ""
echo "all stages submitted; monitor with:"
echo "  watch squeue -u \$USER"
echo "  tail -f $REPO_DIR/slurm/logs/analyze_${JOB_ANALYZE}.out"
