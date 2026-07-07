#!/usr/bin/env bash
# Build the PCA-compressed osmordred targets (osmordred_pca80/90/95 by default): prescale
# PCA_BASE_FLAVOR with the "full" recipe (order-fixed winsorize/z-score plus correlation
# drop, low-variance drop, and Yeo-Johnson), fit PCA once on the train rows, and write one
# target.zarr per threshold in PCA_THRESHOLDS. Single CPU task, not an array: one shared fit
# feeds every threshold, and pca_target.py itself computes all thresholds from one PCA fit.
# Depends on PCA_BASE_FLAVOR's own raw target.zarr already existing (compute_targets.sbatch).
# Resumable: skips prescaling if the full-recipe store exists, and skips PCA if every
# threshold's target.zarr already exists.
#
# Submit after compute_targets.sbatch, before split.sbatch:
#   sbatch --dependency=afterok:<targets job> slurm/osmordred_pca_targets.sbatch
#SBATCH --job-name=sar-pca-targets
#SBATCH --partition=cpu
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/pca_targets_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

BASE="$PCA_BASE_FLAVOR"
read -ra THRESHOLDS <<<"$PCA_THRESHOLDS"
echo "pca base flavor=$BASE thresholds=${THRESHOLDS[*]}"

BASE_TARGET="$REPO_DIR/cache/targets/$BASE/target.zarr"
if [[ ! -d "$BASE_TARGET" ]]; then
    echo "missing $BASE_TARGET; run slurm/compute_targets.sbatch for $BASE first" >&2
    exit 1
fi

# check whether every threshold's target.zarr already exists (all-or-nothing skip, since one
# PCA fit produces every threshold together)
ALL_EXIST=1
for t in "${THRESHOLDS[@]}"; do
    NAME=$(conda run -n "$MAIN_ENV" python -c \
        "from sarizard.pretraining.pca_target import threshold_flavor_name; print(threshold_flavor_name('$BASE', $t))")
    if [[ ! -d "$REPO_DIR/cache/targets/$NAME/target.zarr" ]]; then
        ALL_EXIST=0
    fi
done
if [[ "$ALL_EXIST" -eq 1 ]]; then
    echo "every PCA threshold target.zarr already exists for $BASE; skipping"
    exit 0
fi

FULL_PRESCALED="$REPO_DIR/cache/prescaled/${BASE}_full/prescaled.zarr"
FULL_SUMMARY="$REPO_DIR/cache/prescaled/${BASE}_full/prescaling_summary.json"

# pretraining/ scripts run script-style from their own directory
cd "$REPO_DIR/sarizard/pretraining"

if [[ ! -d "$FULL_PRESCALED" ]]; then
    python prescaling.py --input-zarr "$BASE_TARGET" --output-zarr "$FULL_PRESCALED" \
        --ablation full --summary "$FULL_SUMMARY"
else
    echo "$FULL_PRESCALED exists; skipping the full-recipe prescale"
fi

CHUNK_ROWS_OUT=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.config import CORPUS_CHUNK_ROWS; print(CORPUS_CHUNK_ROWS)")
PCA_SUMMARY="$REPO_DIR/cache/targets/${BASE}_pca_summary.json"
python pca_target.py --input-zarr "$FULL_PRESCALED" --output-dir "$REPO_DIR/cache/targets" \
    --base-flavor "$BASE" --thresholds "${THRESHOLDS[@]}" --chunk-rows-out "$CHUNK_ROWS_OUT" \
    --summary "$PCA_SUMMARY" --force
