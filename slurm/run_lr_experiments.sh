#!/usr/bin/env bash
# Submit the finetune learning-rate experiments as a SLURM dependency chain.
# Run this AFTER run_all.sh has finished (the LR recipes reuse the flavor-sweep foundations and
# compare against the frozen sweep's own results). Every stage is resumable.
#
# The experiments repeat the finetuning from each flavor foundation with the MPNN backbone
# partially unfrozen (reduced: mpnn_lr = ffn_lr/10) or fully unfrozen (unlocked: mpnn_lr =
# ffn_lr), then compare both against the frozen sweep (mpnn_lr = 0). The frozen-warmup protocol
# in TODO.md is not included: it needs a two-phase training schedule anvil cannot express.
#
# Optional overrides (defaults in slurm/env.sh):
#   export LR_MODES="reduced unlocked"    # backbone protocols to sweep
#   export FLAVOR_SEEDS="1 2 3"           # must match the flavor sweep that produced the foundations
#   export REPO_DIR=...  MAIN_ENV=...  ACCELERATOR=...
#
# Usage:
#   bash slurm/run_lr_experiments.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# preflight: the LR recipes point at the flavor-sweep foundations, so at least one must exist
if ! ls "$REPO_DIR"/foundations/*__s*_mp.pt >/dev/null 2>&1; then
    echo "ERROR: no flavor-sweep foundations found under $REPO_DIR/foundations" >&2
    echo "  run slurm/run_all.sh first (its pretrain stage writes <flavor>__s<seed>_mp.pt)" >&2
    exit 1
fi

read -ra MODES <<<"$LR_MODES"
echo "LR modes: ${MODES[*]}   seeds: $FLAVOR_SEEDS"

# generate LR recipes for each mode off the existing flavor foundations; this only reads
# templates and flavor metadata, so it runs before any job queues
echo "generating LR finetuning configs..."
for mode in "${MODES[@]}"; do
    conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
        --seeds $FLAVOR_SEEDS --mpnn-lr-mode "$mode" --label-prefix "lr_${mode}"
done
N_RECIPES=$(lr_recipe_list | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no LR recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES LR recipes (finetune array 0-$((N_RECIPES - 1)))"
echo ""

# submit finetune then analyze; analyze also reads the frozen sweep's existing results
JOB_FINETUNE=$(sbatch --parsable \
    --array=0-$((N_RECIPES - 1)) \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS",LR_MODES="$LR_MODES" \
    "$SCRIPT_DIR/lr_finetune.sbatch")
echo "lr-finetune   job=$JOB_FINETUNE  ($N_RECIPES recipes)"

JOB_ANALYZE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_FINETUNE" \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS",LR_MODES="$LR_MODES" \
    "$SCRIPT_DIR/lr_analyze.sbatch")
echo "lr-analyze    job=$JOB_ANALYZE  (after lr-finetune $JOB_FINETUNE)"

echo ""
echo "LR experiments submitted; when done, read plots/lr_ranking_r2.csv"
echo "  (mean delta and win count of each mode vs the frozen sweep)"
