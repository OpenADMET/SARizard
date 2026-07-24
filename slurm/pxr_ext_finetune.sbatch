#!/usr/bin/env bash
# Finetune the PXR external-test rerun, one GPU array task per (flavor, seed, phase) recipe.
# Each task runs anvil on one generated pxr_ext recipe and writes
# results/pxr_ext__<flavor>__s<seed>/pxr_phase{1,2}_st/. The recipes point at the flavor-sweep
# s42 foundations (stock uses the downloaded checkpoint), so those must already exist. Resumable:
# a recipe whose result dir already exists is skipped.
#
# Generate the recipes first (run_pxr_ext.sh does this), then size --array to the recipe count:
# (source slurm/env.sh; pxr_ext_recipe_list | wc -l). run_pxr_ext.sh drives this via submit_batched.
#SBATCH --job-name=sar-pxrext-finetune
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-0
#SBATCH --output=slurm/logs/pxr_ext_finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

mapfile -t RECIPES < <(pxr_ext_recipe_list)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no pxr_ext recipes found; run slurm/run_pxr_ext.sh (it calls configs.generate)" >&2
    exit 1
fi
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#RECIPES[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#RECIPES[@]} recipes; nothing to do"
    exit 0
fi
RECIPE="${RECIPES[$SLURM_ARRAY_TASK_ID]}"
LABEL=$(basename "$(dirname "$RECIPE")")   # pxr_ext__<flavor>__s<seed>
NAME=$(basename "$RECIPE" .yaml)           # pxr_phase1_st or pxr_phase2_st
OUT="results/$LABEL/$NAME"
echo "task=$SLURM_ARRAY_TASK_ID recipe=$RECIPE -> $OUT"

if [[ -d "$OUT" ]]; then
    echo "$OUT exists; skipping"
    exit 0
fi

openadmet anvil --recipe-path "$RECIPE" --output-dir "$OUT/"
