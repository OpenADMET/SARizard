#!/usr/bin/env bash
# Finetune the stock-CheMeleon reference on every endpoint, one GPU array task per recipe.
# anvil downloads the released CheMeleon checkpoint itself (from_foundation stays "chemeleon"
# in these recipes); no pretraining or foundation conversion is needed first.
# Writes results/chemeleon_stock/<recipe>/. Resumable: an existing result dir is skipped.
#
# Generate the recipes first (gitignored), or let run_stock_baseline.sh do both:
#   python -m sarizard.configs.generate --stock-baseline
#   sbatch --array=0-$(($(ls configs/chemeleon_stock/*.yaml | wc -l) - 1)) \
#       slurm/finetune_stock_baseline.sbatch
#SBATCH --job-name=sar-stock-finetune
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-23
#SBATCH --output=slurm/logs/stock_finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

mapfile -t RECIPES < <(ls "$REPO_DIR"/configs/chemeleon_stock/*.yaml 2>/dev/null | sort)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no recipes found; run: python -m sarizard.configs.generate --stock-baseline" >&2
    exit 1
fi
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#RECIPES[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#RECIPES[@]} recipes; nothing to do"
    exit 0
fi
RECIPE="${RECIPES[$SLURM_ARRAY_TASK_ID]}"
NAME=$(basename "$RECIPE" .yaml)
OUT="results/chemeleon_stock/$NAME"
echo "task=$SLURM_ARRAY_TASK_ID recipe=$RECIPE -> $OUT"

if [[ -d "$OUT" ]]; then
    echo "$OUT exists; skipping"
    exit 0
fi

openadmet anvil --recipe-path "$RECIPE" --output-dir "$OUT/"
