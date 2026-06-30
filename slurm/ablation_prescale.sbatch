#!/usr/bin/env bash
# Prescale the shared raw target under each ablation config, one CPU array task per ablation.
# Each task runs prescaling.py (clean/winsorize/corr/yeo-johnson/low-var/z-score, per the
# ablation's toggles, fit on the train chunks) then split.py --prescaled to slice the result
# into train/val. Resumable: an ablation whose split already exists is skipped.
#
# The array range must cover the ablation registry (7 ablations -> 0-6); update it if the
# registry grows. Submit from the repo root (or let run_ablations.sh do it).
#SBATCH --job-name=sar-abl-prescale
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-6
#SBATCH --output=slurm/logs/abl_prescale_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

mapfile -t ABLATIONS < <(ablation_list)
ABLATION="${ABLATIONS[$SLURM_ARRAY_TASK_ID]}"
FLAVOR="$ABLATION_FLAVOR"
echo "task=$SLURM_ARRAY_TASK_ID ablation=$ABLATION flavor=$FLAVOR"

SPLIT_DIR="$REPO_DIR/cache/ablations/$ABLATION/split"
if [[ -d "$SPLIT_DIR/train_rescaled.zarr" ]]; then
    echo "$SPLIT_DIR/train_rescaled.zarr exists; skipping"
    exit 0
fi

TARGET_ZARR="$REPO_DIR/cache/targets/$FLAVOR/target.zarr"
if [[ ! -d "$TARGET_ZARR" ]]; then
    echo "missing $TARGET_ZARR; run slurm/ablation_target.sbatch first" >&2
    exit 1
fi
PRESCALED="$REPO_DIR/cache/ablations/$ABLATION/prescaled.zarr"
SUMMARY="$REPO_DIR/cache/ablations/$ABLATION/prescaling_summary.json"
CORPUS="$REPO_DIR/corpus/corpus_250k.parquet"

# prescaling.py and split.py run script-style from the pretraining directory
cd "$REPO_DIR/sarizard/pretraining"
python prescaling.py --input-zarr "$TARGET_ZARR" --output-zarr "$PRESCALED" \
    --ablation "$ABLATION" --summary "$SUMMARY" --force
python split.py --input-zarr "$PRESCALED" --input-smiles "$CORPUS" \
    --outdir "$SPLIT_DIR" --flavor "$FLAVOR" --prescaled --force
