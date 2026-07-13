#!/usr/bin/env bash
# Generate and finetune the stock-CheMeleon reference: the released CheMeleon checkpoint
# (anvil downloads it, no local pretraining) finetuned on every ADMET endpoint under the
# same recipe templates every flavor uses. This is the external "how much does any of our
# custom pretraining help over off-the-shelf CheMeleon" reference for the report card
# (sarizard.analysis.report_card's --baseline-flavor).
#
# Multi-seed: STOCK_SEEDS (default "1 2 3 4") finetunes the checkpoint once per seed so the
# baseline carries the same per-cell error bars the flavors do, tagged
# chemeleon_stock[_<mode>]__s<seed>. The existing single-seed run (bare chemeleon_stock, seed
# 42) stays on disk and averages in alongside these, so the frozen baseline is 5 seeds total
# (42 plus 1-4). report_card/meta_model collapse the __s<seed> variants back to one baseline
# column and average them.
#
# One protocol per invocation via STOCK_LR_MODE (frozen/reduced/unlocked), matching the flavor
# LR-experiment driver pattern; run it three times (once per protocol) to cover all three.
# Corpus/regime-independent, so unlike run_all.sh it never needs rerunning when either changes.
# Resumable: an existing results/<label>/<recipe>/ dir with a model.pth is skipped.
#
# The finetune runs through submit_batched.sh, which blocks for the duration (submit a batch,
# wait, rerun bad-node casualties, next batch), so launch this from a persistent shell. The
# durable pattern is a standalone cpu job:
#   sbatch --partition=cpu --time=1-00:00:00 --job-name=stock-driver-frozen \
#       --export=ALL,REPO_DIR="$PWD",STOCK_LR_MODE=frozen,STOCK_SEEDS="1 2 3 4" \
#       --wrap="bash slurm/run_stock_baseline.sh"
#
# Usage:
#   STOCK_LR_MODE=frozen STOCK_SEEDS="1 2 3 4" bash slurm/run_stock_baseline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "stock-CheMeleon baseline: mode=$STOCK_LR_MODE seeds='$STOCK_SEEDS'"

# generate the per-seed recipes for this protocol (one recipe set per seed, gitignored)
for seed in $STOCK_SEEDS; do
    conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
        --stock-baseline --mpnn-lr-mode "$STOCK_LR_MODE" --finetune-seed "$seed"
done

N_RECIPES=$(stock_recipe_list | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no stock-baseline recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES recipes (STOCK_LR_MODE=$STOCK_LR_MODE x $(wc -w <<<"$STOCK_SEEDS") seeds)"

# drive the finetune in bad-node-safe batches; the array index maps against stock_recipe_list,
# the same order finetune_stock_baseline.sbatch enumerates
bash "$SCRIPT_DIR/submit_batched.sh" \
    "$SCRIPT_DIR/finetune_stock_baseline.sbatch" stock_recipe_list --export=ALL

echo ""
echo "stock-baseline finetune complete for mode=$STOCK_LR_MODE."
echo "regenerate the report cards / metrics with the averaged baseline via:"
if [[ "$STOCK_LR_MODE" == "frozen" ]]; then
    echo "  sbatch slurm/analyze.sbatch      # folds every chemeleon_stock[__s<seed>] dir in"
else
    echo "  bash slurm/run_lr_experiments.sh # rerun analyze for the $STOCK_LR_MODE protocol"
fi
