#!/usr/bin/env bash
# Drive the osmordred_surrogate control end to end: compute osmordred descriptors on
# surrogate_adme's Novartis corpus, pretrain one foundation (seed 42) under the shared regime,
# finetune the 24 endpoints at 5 seeds off that foundation, then evaluate and print the
# comparison against the stock baseline. The control is standalone: it is excluded from
# flavor_names() so it never enters the sweep or the report card, and its metrics land in a
# dedicated results/osmordred_surrogate_metrics.csv, not the shared results/metrics.csv.
#
# The pre-finetune stages (target -> split -> pretrain) run as a SLURM dependency chain; the
# finetune stage then runs in batches via submit_batched.sh, which BLOCKS for hours, so launch
# this from a persistent shell (interactive allocation or nohup). Every stage is resumable.
#
# Prereq: surrogate_adme's corpus is already on disk
# (cache/targets/surrogate_adme/corpus_smiles.parquet), written by its own target computation.
#
# Usage (export REPO_DIR to dodge a stale interactive SLURM_SUBMIT_DIR, per the repo convention):
#   REPO_DIR=/scratch/choderaj/westd/SARizard bash slurm/run_osmordred_surrogate.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

FLAVOR=osmordred_surrogate
FOUNDATION_SEED=42
FINETUNE_SEEDS="1 2 3 4 5"

# scope every registry-driven array stage (targets/split/pretrain/finetune) to just this flavor;
# it is standalone, so flavor_list() would otherwise never emit it
export FLAVOR_SUBSET="$FLAVOR"
source "$SCRIPT_DIR/env.sh"

CORPUS_SRC="$REPO_DIR/cache/targets/surrogate_adme/corpus_smiles.parquet"
if [[ ! -f "$CORPUS_SRC" ]]; then
    echo "ERROR: $CORPUS_SRC is missing; compute the surrogate_adme target first" >&2
    exit 1
fi

FOUNDATION="foundations/${FLAVOR}__s${FOUNDATION_SEED}_mp.pt"

# Phase A: build the seed-42 foundation (target on the Novartis corpus -> split -> pretrain),
# skipped when the foundation already exists so a rerun jumps straight to finetuning.
FINETUNE_DEPENDENCY=""
if [[ -f "$REPO_DIR/$FOUNDATION" ]]; then
    echo "foundation $FOUNDATION exists; skipping the build, finetuning off it directly"
else
    JOB_TARGETS=$(sbatch --parsable --array=0-0 --export=ALL "$SCRIPT_DIR/compute_targets.sbatch")
    echo "targets   job=$JOB_TARGETS  (osmordred on the Novartis corpus)"

    JOB_SPLIT=$(sbatch --parsable \
        --dependency=afterok:"$JOB_TARGETS" --array=0-0 --export=ALL \
        "$SCRIPT_DIR/split.sbatch")
    echo "split     job=$JOB_SPLIT  (after targets $JOB_TARGETS, order_fix prescaling)"

    JOB_PRETRAIN=$(sbatch --parsable \
        --dependency=afterok:"$JOB_SPLIT" --array=0-0 --export=ALL,FLAVOR_SEEDS="$FOUNDATION_SEED" \
        "$SCRIPT_DIR/pretrain.sbatch")
    echo "pretrain  job=$JOB_PRETRAIN  (after split $JOB_SPLIT, seed $FOUNDATION_SEED foundation)"
    FINETUNE_DEPENDENCY="--dependency=afterok:$JOB_PRETRAIN"
fi

# generate the 5-seed finetune recipes off the one pinned s42 foundation (each seed a replicate);
# reads only flavor metadata and templates, so it runs before the foundation exists
echo "generating $FINETUNE_SEEDS finetunes of the s$FOUNDATION_SEED foundation..."
conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
    --flavors "$FLAVOR" --seeds $FINETUNE_SEEDS --foundation-seed "$FOUNDATION_SEED"

# Phase B: batched finetune (blocks until all 120 recipes have model.pth). The first batch waits
# on pretrain via $FINETUNE_DEPENDENCY when the foundation was built this run.
export FLAVOR_SEEDS="$FINETUNE_SEEDS"
N_RECIPES=$(flavor_recipe_list | wc -l | tr -d ' ')
echo "finetune  running $N_RECIPES recipes in batches (submit_batched.sh)..."
"$SCRIPT_DIR/submit_batched.sh" "$SCRIPT_DIR/finetune.sbatch" flavor_recipe_list \
    --export=ALL,FLAVOR_SEEDS="$FINETUNE_SEEDS",FLAVOR_SUBSET="$FLAVOR" $FINETUNE_DEPENDENCY
echo "finetune  complete (all $N_RECIPES recipes)"

# Phase C: evaluate the control into its dedicated CSV and print the comparison (GPU); no SLURM
# dependency since every finetune is verified complete above
JOB_ANALYZE=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/osmordred_surrogate_analyze.sbatch")
echo "analyze   job=$JOB_ANALYZE"
echo ""
echo "control finetuned; evaluate+print submitted. Watch the comparison with:"
echo "  tail -f $REPO_DIR/slurm/logs/osurr_analyze_${JOB_ANALYZE}.out"
