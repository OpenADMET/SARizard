#!/usr/bin/env bash
# Compute every flavor's pretraining target, one array task per flavor. Each task computes
# the raw target in the flavor's own conda environment, then packs it to zarr in the main
# environment. CPU only. Resumable: a flavor whose target.zarr already exists is skipped.
#
# EDIT the resource directives below for your cluster, then submit from the repo root:
#   sbatch slurm/compute_targets.sbatch
# The array range must cover the registry (13 flavors -> 0-12); update it if the registry grows.
#SBATCH --job-name=sar-targets
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-12
#SBATCH --output=slurm/logs/targets_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"

# surrogate_adme reads the Novartis released CSV directly; set this path before submitting
SURROGATE_CSV="${SURROGATE_CSV:-EDIT_PATH_TO_protacdb2.0_zinc_chembl_dataset.csv}"

# map this array task to a flavor and look up the environment that computes it
mapfile -t FLAVORS < <(flavor_list)
FLAVOR="${FLAVORS[$SLURM_ARRAY_TASK_ID]}"
FLAVOR_ENV=$(conda run -n "$MAIN_ENV" python -c \
    "from pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').env)")
echo "task=$SLURM_ARRAY_TASK_ID flavor=$FLAVOR env=$FLAVOR_ENV"

if [[ -d "cache/targets/$FLAVOR/target.zarr" ]]; then
    echo "cache/targets/$FLAVOR/target.zarr exists; skipping"
    exit 0
fi

if [[ "$FLAVOR" == "surrogate_adme" ]]; then
    # surrogate_adme runs in the main env; compute_target writes target.npy + corpus_smiles.parquet
    conda activate "$MAIN_ENV"
    python -m pretraining.features.compute_target --flavor surrogate_adme \
        --csv-path "$SURROGATE_CSV"
    python -m pretraining.features.pack_target --flavor surrogate_adme
else
    # compute the raw target in the flavor's isolated environment (writes target.npy)
    conda activate "$FLAVOR_ENV"
    python -m pretraining.features.compute_target --flavor "$FLAVOR"

    # pack the .npy into the chunked target.zarr the trainer reads, in the main environment
    conda activate "$MAIN_ENV"
    python -m pretraining.features.pack_target --flavor "$FLAVOR"
fi
