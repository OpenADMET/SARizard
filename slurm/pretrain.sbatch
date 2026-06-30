#!/usr/bin/env bash
# Pretrain every flavor's foundation, one GPU array task per flavor. Each task splits the
# flavor's cached target into train/val, pretrains the D-MPNN under the shared regime, and
# exports the foundation checkpoint (train.py writes it to foundations/). Resumable: a flavor
# whose foundation already exists is skipped.
#
# EDIT the resource directives below for your cluster, then submit from the repo root (after
# compute_targets has produced cache/targets/<flavor>/target.zarr):
#   sbatch slurm/pretrain.sbatch
# The array range must cover the registry (13 flavors -> 0-12).
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

mapfile -t FLAVORS < <(flavor_list)
FLAVOR="${FLAVORS[$SLURM_ARRAY_TASK_ID]}"
echo "task=$SLURM_ARRAY_TASK_ID flavor=$FLAVOR"

if [[ -f "foundations/${FLAVOR}_mp.pt" ]]; then
    echo "foundations/${FLAVOR}_mp.pt exists; skipping"
    exit 0
fi

# absolute paths (layout mirrors analysis/paths.py); split.py and train.py run script-style
# from the pretraining directory, so pass them absolute paths
TARGET_ZARR="$REPO_DIR/cache/targets/$FLAVOR/target.zarr"
if [[ ! -d "$TARGET_ZARR" ]]; then
    echo "missing $TARGET_ZARR; run slurm/compute_targets.sbatch first" >&2
    exit 1
fi
SPLIT_DIR="$REPO_DIR/cache/splits/$FLAVOR"
# surrogate_adme uses its own corpus written by compute_target; all other flavors share 250K
if [[ "$FLAVOR" == "surrogate_adme" ]]; then
    CORPUS="$REPO_DIR/cache/targets/surrogate_adme/corpus_smiles.parquet"
else
    CORPUS="$REPO_DIR/corpus/corpus_250k.parquet"
fi
RUN_DIR="$REPO_DIR/sarizard/pretraining/runs/$FLAVOR"

cd "$REPO_DIR/sarizard/pretraining"
python split.py --input-zarr "$TARGET_ZARR" --input-smiles "$CORPUS" \
    --outdir "$SPLIT_DIR" --flavor "$FLAVOR" --force
python train.py --flavor "$FLAVOR" --input-dir "$SPLIT_DIR" --output-dir "$RUN_DIR"
