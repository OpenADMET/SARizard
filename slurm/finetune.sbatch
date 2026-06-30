#!/usr/bin/env bash
# Finetune every flavor on every endpoint, one GPU array task per (flavor, endpoint) recipe.
# Each task runs anvil on one generated recipe and writes results/<flavor>/<recipe>/.
# Resumable: a recipe whose result directory already exists is skipped.
#
# Generate the recipes first (they are gitignored), then submit from the repo root:
#   python -m sarizard.configs.generate
#   sbatch slurm/finetune.sbatch
# The array range must cover flavors x endpoints (13 x 24 = 312 -> 0-311). Adjust to match the
# number of generated recipes: ls configs/*/*.yaml | grep -v /_baseline/ | wc -l
#SBATCH --job-name=sar-finetune
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-311
#SBATCH --output=slurm/logs/finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

# enumerate generated per-flavor recipes (deterministic sorted order), excluding the templates
mapfile -t RECIPES < <(ls configs/*/*.yaml 2>/dev/null | grep -v '/_baseline/')
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no recipes found; run: python -m sarizard.configs.generate" >&2
    exit 1
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
