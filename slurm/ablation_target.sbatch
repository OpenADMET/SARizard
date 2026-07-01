#!/usr/bin/env bash
# Compute the shared raw target for the prescaling ablation triage: the representative
# continuous flavor (ABLATION_FLAVOR, default osmordred). Single CPU task. Every ablation
# prescales this one raw target store, so it is built once here. Resumable: skipped if the
# target.zarr already exists.
#
# Submit from the repo root (or let run_ablations.sh do it):
#   sbatch slurm/ablation_target.sbatch
#SBATCH --job-name=sar-abl-target
#SBATCH --partition=cpu
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/abl_target_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"

FLAVOR="$ABLATION_FLAVOR"
FLAVOR_ENV=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').env)")
echo "ablation flavor=$FLAVOR env=$FLAVOR_ENV corpus=$CORPUS_FILE"

if [[ -d "cache/targets/$FLAVOR/target.zarr" ]]; then
    echo "cache/targets/$FLAVOR/target.zarr exists; skipping"
    exit 0
fi

# compute the raw target in the flavor's isolated environment, then pack to zarr in main
conda activate "$FLAVOR_ENV"
python -m sarizard.pretraining.features.compute_target --flavor "$FLAVOR" \
    --corpus "$REPO_DIR/$CORPUS_FILE"
conda activate "$MAIN_ENV"
python -m sarizard.pretraining.features.pack_target --flavor "$FLAVOR"
