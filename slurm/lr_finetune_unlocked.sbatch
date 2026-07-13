#!/usr/bin/env bash
# Finetune the unlocked-protocol LR experiments, one GPU array task per (flavor, seed, endpoint)
# recipe. Identical to lr_finetune.sbatch except it enumerates only the unlocked recipes
# (lr_unlocked_recipe_list) instead of the full lr_* glob, so an unlocked-only driver runs
# concurrently with a reduced-only driver (lr_finetune.sbatch + lr_recipe_list) without their
# array indices aliasing onto each other's recipes. Each task runs anvil on one generated unlocked
# recipe and writes results/lr_unlocked__<flavor>__s<seed>/<endpoint>/. The recipes point at the
# flavor-sweep foundations, so those must already exist. Resumable: a recipe whose result dir
# already exists is skipped.
#
# Generate the unlocked recipes first (configs.generate --mpnn-lr-mode unlocked), then size
# --array to the recipe count: (source slurm/env.sh; lr_unlocked_recipe_list | wc -l).
#SBATCH --job-name=sar-lr-finetune
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-0
#SBATCH --output=slurm/logs/lr_finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

mapfile -t RECIPES < <(lr_unlocked_recipe_list)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no unlocked LR recipes found; run configs.generate --mpnn-lr-mode unlocked first" >&2
    exit 1
fi
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#RECIPES[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#RECIPES[@]} recipes; nothing to do"
    exit 0
fi
RECIPE="${RECIPES[$SLURM_ARRAY_TASK_ID]}"
LABEL=$(basename "$(dirname "$RECIPE")")   # lr_unlocked__<flavor>__s<seed>
NAME=$(basename "$RECIPE" .yaml)           # endpoint
OUT="results/$LABEL/$NAME"
echo "task=$SLURM_ARRAY_TASK_ID recipe=$RECIPE -> $OUT"

if [[ -d "$OUT" ]]; then
    echo "$OUT exists; skipping"
    exit 0
fi

openadmet anvil --recipe-path "$RECIPE" --output-dir "$OUT/"
