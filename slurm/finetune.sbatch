#!/usr/bin/env bash
# Finetune every flavor on every endpoint, one GPU array task per (flavor, endpoint) recipe.
# Each task runs anvil on one generated recipe and writes results/<flavor>/<recipe>/.
# Resumable: a recipe whose result directory already exists is skipped.
#
# Generate the recipes first (they are gitignored). Prefer submitting through run_all.sh,
# which sizes the array from the actual recipe count. To submit standalone, size the array to
# match the generated recipes (flavor_recipe_list counts only registry flavors, not ablations):
#   python -m sarizard.configs.generate
#   sbatch --array=0-$(($(source slurm/env.sh; flavor_recipe_list | wc -l) - 1)) slurm/finetune.sbatch
# The --array header below is a conservative default; the command line above overrides it.
#SBATCH --job-name=sar-finetune
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-311
#SBATCH --output=slurm/logs/finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

# enumerate generated per-flavor recipes (registry flavors only, deterministic sorted order);
# flavor_recipe_list excludes the _baseline templates and any ablation_* dirs under configs/
mapfile -t RECIPES < <(flavor_recipe_list)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no recipes found; run: python -m sarizard.configs.generate" >&2
    exit 1
fi
# guard against an array range wider than the recipe count (e.g. the default header on a
# registry that produced fewer recipes); such tasks are no-ops rather than indexing errors
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#RECIPES[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#RECIPES[@]} recipes; nothing to do"
    exit 0
fi
RECIPE="${RECIPES[$SLURM_ARRAY_TASK_ID]}"
FLAVOR=$(basename "$(dirname "$RECIPE")")
NAME=$(basename "$RECIPE" .yaml)
OUT="results/$FLAVOR/$NAME"
echo "task=$SLURM_ARRAY_TASK_ID recipe=$RECIPE -> $OUT"

if [[ -d "$OUT" ]]; then
    echo "$OUT exists; skipping"
    exit 0
fi

openadmet anvil --recipe-path "$RECIPE" --output-dir "$OUT/"
