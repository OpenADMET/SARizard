#!/usr/bin/env bash
# Finetune each external pretrained foundation on every endpoint, one GPU array task per
# (name, seed, protocol, endpoint) recipe. Each task runs anvil on one generated recipe and
# writes results/<label>/<endpoint>/. Resumable: a recipe whose result directory already
# exists is skipped.
#
# Generate the recipes first (run_external_foundations.sh does this), then size --array to the
# recipe count: source slurm/env.sh; extfound_recipe_list | wc -l
# (4 names x 24 endpoints x 5 seeds x 3 protocols = 1440 by default).
#SBATCH --job-name=sar-extfound-finetune
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-1439
#SBATCH --output=slurm/logs/extfound_finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

# enumerate generated external-foundation recipes (deterministic sorted order); the same env.sh
# function submit_batched.sh uses, so the array-index-to-recipe mapping matches when driven in batches
mapfile -t RECIPES < <(extfound_recipe_list)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no external-foundation recipes found; run run_external_foundations.sh" >&2
    exit 1
fi
# guard against an array range wider than the recipe count (the default header assumes the full
# 1440; a scoped run via EXTFOUND_NAMES/SEEDS/LR_MODES produces fewer)
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#RECIPES[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#RECIPES[@]} recipes; nothing to do"
    exit 0
fi
RECIPE="${RECIPES[$SLURM_ARRAY_TASK_ID]}"
LABEL=$(basename "$(dirname "$RECIPE")")   # <name>__s<seed> or lr_<mode>__<name>__s<seed>
NAME=$(basename "$RECIPE" .yaml)           # endpoint
OUT="results/$LABEL/$NAME"
echo "task=$SLURM_ARRAY_TASK_ID recipe=$RECIPE -> $OUT"

if [[ -d "$OUT" ]]; then
    echo "$OUT exists; skipping"
    exit 0
fi

openadmet anvil --recipe-path "$RECIPE" --output-dir "$OUT/"
