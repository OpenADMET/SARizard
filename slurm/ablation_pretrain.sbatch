#!/usr/bin/env bash
# Pretrain a foundation from each ablation's prescaled target, one GPU array task per
# ablation. The backbone, corpus, and regime are identical across ablations; only the
# target prescaling differs, so the downstream report card isolates the prescaling effect.
# Resumable: an ablation whose foundation already exists is skipped.
#
# One task per (ablation, seed): the array spans ABLATIONS x ABLATION_SEEDS, row-major in
# ablation then seed. Size --array to that product (run_ablations.sh does it):
#   sbatch --array=0-$(($(source slurm/env.sh; ablation_list | wc -l) * \
#       $(source slurm/env.sh; wc -w <<<"$ABLATION_SEEDS") - 1)) slurm/ablation_pretrain.sbatch
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

# map the flat array index to an (ablation, seed) pair, ablation-major
mapfile -t ABLATIONS < <(ablation_list)
read -ra SEEDS <<<"$ABLATION_SEEDS"
N_SEEDS=${#SEEDS[@]}
N_TASKS=$(( ${#ABLATIONS[@]} * N_SEEDS ))
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_TASKS" ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= $N_TASKS (ablations x seeds); nothing to do"
    exit 0
fi
ABLATION="${ABLATIONS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % N_SEEDS))]}"
FLAVOR="$ABLATION_FLAVOR"
FOUNDATION="ablation_${ABLATION}__s${SEED}_mp.pt"
echo "task=$SLURM_ARRAY_TASK_ID ablation=$ABLATION seed=$SEED flavor=$FLAVOR -> $FOUNDATION"

if [[ -f "foundations/$FOUNDATION" ]]; then
    echo "foundations/$FOUNDATION exists; skipping"
    exit 0
fi

# the prescaled split is seed-independent (prescaling is deterministic), so every seed of an
# ablation reads the one split that ablation_prescale produced
SPLIT_DIR="$REPO_DIR/cache/ablations/$ABLATION/split"
if [[ ! -d "$SPLIT_DIR/train_rescaled.zarr" ]]; then
    echo "missing $SPLIT_DIR; run slurm/ablation_prescale.sbatch first" >&2
    exit 1
fi
RUN_DIR="$REPO_DIR/sarizard/pretraining/runs/ablation_${ABLATION}__s${SEED}"

cd "$REPO_DIR/sarizard/pretraining"
python train.py --flavor "$FLAVOR" --input-dir "$SPLIT_DIR" --output-dir "$RUN_DIR" \
    --foundation-name "$FOUNDATION" --seed "$SEED"
