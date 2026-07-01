#!/usr/bin/env bash
# Finetune each ablation foundation on every endpoint, one GPU array task per
# (ablation, endpoint) recipe. Each task runs anvil on one generated ablation recipe and
# writes results/ablation_<name>/<endpoint>/. Resumable: a recipe whose result directory
# already exists is skipped.
#
# Generate the ablation recipes first (run_ablations.sh does this), then size --array to the
# recipe count: ls configs/ablation_*/*.yaml | wc -l  (7 ablations x 24 endpoints = 168 per
# finetune protocol; the count grows with ABLATION_LR_MODES, e.g. 504 across frozen+reduced+unlocked).
#SBATCH --job-name=sar-abl-finetune
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-167
#SBATCH --output=slurm/logs/abl_finetune_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

# enumerate generated ablation recipes (deterministic sorted order)
mapfile -t RECIPES < <(ls configs/ablation_*/*.yaml 2>/dev/null)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "no ablation recipes found; run run_ablations.sh (it calls configs.generate)" >&2
    exit 1
fi
RECIPE="${RECIPES[$SLURM_ARRAY_TASK_ID]}"
LABEL=$(basename "$(dirname "$RECIPE")")   # ablation_<name>
NAME=$(basename "$RECIPE" .yaml)           # endpoint
OUT="results/$LABEL/$NAME"
echo "task=$SLURM_ARRAY_TASK_ID recipe=$RECIPE -> $OUT"

if [[ -d "$OUT" ]]; then
    echo "$OUT exists; skipping"
    exit 0
fi

openadmet anvil --recipe-path "$RECIPE" --output-dir "$OUT/"
