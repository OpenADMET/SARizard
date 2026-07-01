#!/usr/bin/env bash
# Pretrain each flavor's foundation, one GPU array task per (flavor, seed). Each task reads the
# flavor's pre-built split (from split.sbatch), pretrains the D-MPNN at its seed under the shared
# regime, and exports the foundation checkpoint (train.py writes it to foundations/). Resumable:
# a (flavor, seed) whose foundation already exists is skipped.
#
# One task per (flavor, seed): the array spans FLAVORS x FLAVOR_SEEDS, flavor-major. Prefer
# submitting through run_all.sh, which sizes the array. Standalone (after split.sbatch):
#   sbatch --array=0-$(($(source slurm/env.sh; flavor_list | wc -l) * \
#       $(source slurm/env.sh; wc -w <<<"$FLAVOR_SEEDS") - 1)) slurm/pretrain.sbatch
#SBATCH --job-name=sar-pretrain
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-12
#SBATCH --output=slurm/logs/pretrain_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

# map the flat array index to a (flavor, seed) pair, flavor-major
mapfile -t FLAVORS < <(flavor_list)
read -ra SEEDS <<<"$FLAVOR_SEEDS"
N_SEEDS=${#SEEDS[@]}
N_TASKS=$(( ${#FLAVORS[@]} * N_SEEDS ))
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_TASKS" ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= $N_TASKS (flavors x seeds); nothing to do"
    exit 0
fi
FLAVOR="${FLAVORS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % N_SEEDS))]}"
FOUNDATION="${FLAVOR}__s${SEED}_mp.pt"
echo "task=$SLURM_ARRAY_TASK_ID flavor=$FLAVOR seed=$SEED -> $FOUNDATION"

if [[ -f "foundations/$FOUNDATION" ]]; then
    echo "foundations/$FOUNDATION exists; skipping"
    exit 0
fi

# the split is seed-independent and built once per flavor by split.sbatch
SPLIT_DIR="$REPO_DIR/cache/splits/$FLAVOR"
if [[ ! -d "$SPLIT_DIR/train_rescaled.zarr" ]]; then
    echo "missing $SPLIT_DIR; run slurm/split.sbatch first" >&2
    exit 1
fi
RUN_DIR="$REPO_DIR/sarizard/pretraining/runs/${FLAVOR}__s${SEED}"

# train.py runs script-style from the pretraining directory; pass it absolute paths
cd "$REPO_DIR/sarizard/pretraining"
python train.py --flavor "$FLAVOR" --input-dir "$SPLIT_DIR" --output-dir "$RUN_DIR" \
    --foundation-name "$FOUNDATION" --seed "$SEED"
