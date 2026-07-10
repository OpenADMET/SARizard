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

# FOUNDATION_SEED set => finetune-only replication: every FLAVOR_SEED finetunes the one
# foundation pretrained at FOUNDATION_SEED, so the seeds are finetune replicates off that
# foundation rather than one foundation per seed. This script never pretrains regardless (the
# LR recipes reuse the flavor-sweep foundations); FOUNDATION_SEED only pins which foundation.
FINETUNE_ONLY=""
[[ -n "${FOUNDATION_SEED:-}" ]] && FINETUNE_ONLY=1

# preflight: the LR recipes point at the flavor-sweep foundations. Finetune-only needs every
# pinned s<FOUNDATION_SEED> foundation; the legacy path just needs at least one to exist.
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
elif ! ls "$REPO_DIR"/foundations/*__s*_mp.pt >/dev/null 2>&1; then
    echo "ERROR: no flavor-sweep foundations found under $REPO_DIR/foundations" >&2
    echo "  run slurm/run_all.sh first (its pretrain stage writes <flavor>__s<seed>_mp.pt)" >&2
    exit 1
fi

read -ra MODES <<<"$LR_MODES"
echo "LR modes: ${MODES[*]}   seeds: $FLAVOR_SEEDS"

# generate LR recipes for each mode off the existing flavor foundations; this only reads
# templates and flavor metadata, so it runs before any job queues. Finetune-only pins the
# foundation seed so every FLAVOR_SEED finetunes the one s<FOUNDATION_SEED> foundation.
echo "generating LR finetuning configs..."
foundation_flag=()
[[ -n "$FINETUNE_ONLY" ]] && foundation_flag=(--foundation-seed "$FOUNDATION_SEED")
for mode in "${MODES[@]}"; do
    conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
        --seeds $FLAVOR_SEEDS --mpnn-lr-mode "$mode" --label-prefix "lr_${mode}" \
        "${foundation_flag[@]}"
done
N_RECIPES=$(lr_recipe_list | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no LR recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES LR recipes (finetune array 0-$((N_RECIPES - 1)))"
echo ""

# run the finetune array in batches of BATCH_SIZE (default 50) via submit_batched.sh: submit a
# batch, wait for it, rerun its failures, then the next. It blocks until every LR recipe has a
# trained model.pth (or exits nonzero, stopping set -e before analyze), so this runs for hours;
# launch this script from a persistent shell (interactive allocation or nohup). The LR recipes
# point at foundations that already exist, so no upstream SLURM dependency is passed.
echo "lr-finetune   running in batches (submit_batched.sh)..."
"$SCRIPT_DIR/submit_batched.sh" "$SCRIPT_DIR/lr_finetune.sbatch" lr_recipe_list \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS",LR_MODES="$LR_MODES"
echo "lr-finetune   complete ($N_RECIPES recipes)"

# every finetune is verified complete, so analyze needs no SLURM dependency
JOB_ANALYZE=$(sbatch --parsable \
    --export=ALL,FLAVOR_SEEDS="$FLAVOR_SEEDS",LR_MODES="$LR_MODES" \
    "$SCRIPT_DIR/lr_analyze.sbatch")
echo "lr-analyze    job=$JOB_ANALYZE"

echo ""
echo "LR experiments submitted; when done, read plots/lr_ranking_r2.csv"
echo "  (mean delta and win count of each mode vs the frozen sweep)"
