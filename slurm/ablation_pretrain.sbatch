#!/usr/bin/env bash
# Pretrain a foundation from each ablation's prescaled target, one GPU array task per
# ablation. The backbone, corpus, and regime are identical across ablations; only the
# target prescaling differs, so the downstream report card isolates the prescaling effect.
# Resumable: an ablation whose foundation already exists is skipped.
#
# The array range must cover the ablation registry (7 ablations -> 0-6). Submit from the repo
# root after ablation_prescale (or let run_ablations.sh do it).
#SBATCH --job-name=sar-abl-pretrain
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-6
#SBATCH --output=slurm/logs/abl_pretrain_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

mapfile -t ABLATIONS < <(ablation_list)
ABLATION="${ABLATIONS[$SLURM_ARRAY_TASK_ID]}"
FLAVOR="$ABLATION_FLAVOR"
FOUNDATION="ablation_${ABLATION}_mp.pt"
echo "task=$SLURM_ARRAY_TASK_ID ablation=$ABLATION flavor=$FLAVOR -> $FOUNDATION"

if [[ -f "foundations/$FOUNDATION" ]]; then
    echo "foundations/$FOUNDATION exists; skipping"
    exit 0
fi

SPLIT_DIR="$REPO_DIR/cache/ablations/$ABLATION/split"
if [[ ! -d "$SPLIT_DIR/train_rescaled.zarr" ]]; then
    echo "missing $SPLIT_DIR; run slurm/ablation_prescale.sbatch first" >&2
    exit 1
fi
RUN_DIR="$REPO_DIR/sarizard/pretraining/runs/ablation_$ABLATION"

cd "$REPO_DIR/sarizard/pretraining"
python train.py --flavor "$FLAVOR" --input-dir "$SPLIT_DIR" --output-dir "$RUN_DIR" \
    --foundation-name "$FOUNDATION"
