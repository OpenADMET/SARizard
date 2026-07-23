#!/usr/bin/env bash
# Split each flavor's cached target into a rescaled train/val split, one CPU array task per
# flavor. The split is seed-independent, so it is built once per flavor here and every seed's
# pretraining task reads it (this keeps concurrent seed tasks from racing on the same split).
# Resumable: a flavor whose split already exists is skipped.
#
# Prefer submitting through run_all.sh, which sizes the array from the registry. Standalone
# (after compute_targets has produced cache/targets/<flavor>/target.zarr):
#   sbatch --array=0-$(($(source slurm/env.sh; flavor_list | wc -l) - 1)) slurm/split.sbatch
#SBATCH --job-name=sar-split
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-12
#SBATCH --output=slurm/logs/split_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

mapfile -t FLAVORS < <(flavor_list)
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#FLAVORS[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#FLAVORS[@]} flavors; nothing to do"
    exit 0
fi
FLAVOR="${FLAVORS[$SLURM_ARRAY_TASK_ID]}"
echo "task=$SLURM_ARRAY_TASK_ID flavor=$FLAVOR"

SPLIT_DIR="$REPO_DIR/cache/splits/$FLAVOR"
if [[ -d "$SPLIT_DIR/train_rescaled.zarr" ]]; then
    echo "$SPLIT_DIR/train_rescaled.zarr exists; skipping"
    exit 0
fi

TARGET_ZARR="$REPO_DIR/cache/targets/$FLAVOR/target.zarr"
if [[ ! -d "$TARGET_ZARR" ]]; then
    echo "missing $TARGET_ZARR; run slurm/compute_targets.sbatch first" >&2
    exit 1
fi
# split.py runs script-style from the pretraining directory; pass absolute paths
cd "$REPO_DIR/sarizard/pretraining"

DERIVED_FROM=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').derived_from or '')")
CORPUS_FROM=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').corpus_from or '')")

# surrogate_adme uses its own corpus written by compute_target, and a corpus_from flavor borrows
# that companion corpus; all other flavors share whichever corpus this run targets (CORPUS_FILE)
if [[ "$FLAVOR" == "surrogate_adme" ]]; then
    CORPUS="$REPO_DIR/cache/targets/surrogate_adme/corpus_smiles.parquet"
elif [[ -n "$CORPUS_FROM" ]]; then
    CORPUS="$REPO_DIR/cache/targets/$CORPUS_FROM/corpus_smiles.parquet"
else
    CORPUS="$REPO_DIR/$CORPUS_FILE"
fi
KIND=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').kind)")

if [[ -n "$DERIVED_FROM" ]]; then
    # a derived flavor's target.zarr (e.g. osmordred_pca80, written by
    # slurm/osmordred_pca_targets.sbatch) is already cleaned, PCA-projected, and z-scored;
    # treat it exactly like a prescaled input, no further order_fix prescaling
    python split.py --input-zarr "$TARGET_ZARR" --input-smiles "$CORPUS" \
        --outdir "$SPLIT_DIR" --flavor "$FLAVOR" --prescaled --force
elif [[ "$KIND" == "continuous" ]]; then
    # continuous flavors get order_fix prescaling (winsorize before z-score, see
    # wiki/Prescaling Ablation.md) ahead of the split, the Milestone-6 override of the
    # Milestone-5 default (chemeleon_baseline); binary/fingerprint flavors skip rescaling
    # entirely regardless (their kind selects that inside split.py), so they run split.py
    # directly on the raw target as before
    PRESCALED="$REPO_DIR/cache/prescaled/$FLAVOR/prescaled.zarr"
    SUMMARY="$REPO_DIR/cache/prescaled/$FLAVOR/prescaling_summary.json"
    python prescaling.py --input-zarr "$TARGET_ZARR" --output-zarr "$PRESCALED" \
        --ablation order_fix --summary "$SUMMARY" --force
    python split.py --input-zarr "$PRESCALED" --input-smiles "$CORPUS" \
        --outdir "$SPLIT_DIR" --flavor "$FLAVOR" --prescaled --force
else
    python split.py --input-zarr "$TARGET_ZARR" --input-smiles "$CORPUS" \
        --outdir "$SPLIT_DIR" --flavor "$FLAVOR" --force
fi
