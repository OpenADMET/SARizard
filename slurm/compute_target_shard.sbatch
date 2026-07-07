#!/usr/bin/env bash
# Compute one row-shard of a slow flavor's target, one array task per shard, so a flavor whose
# whole-corpus compute would exceed wall time (the 3D conformer flavors, jazzy) finishes in
# parallel instead. Set FLAVOR and NUM_SHARDS, and size --array to 0-(NUM_SHARDS-1). Each task
# computes its slice in the flavor's isolated environment and writes cache/targets/<flavor>/
# shards/; slurm/merge_target.sbatch then concatenates them and packs the zarr.
#
#   sbatch --export=ALL,FLAVOR=usrcat,NUM_SHARDS=20 --array=0-19 slurm/compute_target_shard.sbatch
#SBATCH --job-name=sar-tgtshard
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-19
#SBATCH --output=slurm/logs/tgtshard_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"

: "${FLAVOR:?set FLAVOR (e.g. usrcat) via --export}"
: "${NUM_SHARDS:?set NUM_SHARDS (must match the --array size) via --export}"

# look up the environment that computes this flavor
FLAVOR_ENV=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').env)")
echo "shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS flavor=$FLAVOR env=$FLAVOR_ENV"

# pin worker count to the allocation so we do not oversubscribe the node's cores
conda activate "$FLAVOR_ENV"
python -m sarizard.pretraining.features.compute_target \
    --flavor "$FLAVOR" --corpus "$REPO_DIR/$CORPUS_FILE" \
    --num-shards "$NUM_SHARDS" --shard-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"
