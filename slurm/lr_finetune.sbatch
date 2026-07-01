#!/usr/bin/env bash
# Finetune the LR experiments, one GPU array task per (lr_mode, flavor, seed, endpoint) recipe.
# Each task runs anvil on one generated LR recipe and writes results/lr_<mode>__<flavor>__s<seed>/
# <endpoint>/. The recipes point at the flavor-sweep foundations, so those must already exist
# (run_all.sh's pretrain stage). Resumable: a recipe whose result dir already exists is skipped.
#
# Generate the LR recipes first (run_lr_experiments.sh does this), then size --array to the
# recipe count: (source slurm/env.sh; lr_recipe_list | wc -l).
#SBATCH --job-name=sar-lr-finetune
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-0
#SBATCH --output=slurm/logs/lr_finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

mapfile -t RECIPES < <(lr_recipe_list)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no LR recipes found; run slurm/run_lr_experiments.sh (it calls configs.generate)" >&2
    exit 1
fi
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#RECIPES[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#RECIPES[@]} recipes; nothing to do"
    exit 0
fi
RECIPE="${RECIPES[$SLURM_ARRAY_TASK_ID]}"
LABEL=$(basename "$(dirname "$RECIPE")")   # lr_<mode>__<flavor>__s<seed>
NAME=$(basename "$RECIPE" .yaml)           # endpoint
OUT="results/$LABEL/$NAME"
echo "task=$SLURM_ARRAY_TASK_ID recipe=$RECIPE -> $OUT"

if [[ -d "$OUT" ]]; then
    echo "$OUT exists; skipping"
    exit 0
fi

openadmet anvil --recipe-path "$RECIPE" --output-dir "$OUT/"
