#!/usr/bin/env bash
# Merge a sharded flavor's row-shards into target.npy and pack it to target.zarr. Merging is
# calculator-free (a validated numpy concatenation), so it runs in the main environment.
# Depends afterok on the shard array; set FLAVOR and NUM_SHARDS to match that job.
#
#   sbatch --export=ALL,FLAVOR=usrcat,NUM_SHARDS=20 \
#       --dependency=afterok:<shard_job_id> slurm/merge_target.sbatch
#SBATCH --job-name=sar-tgtmerge
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/tgtmerge_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"

: "${FLAVOR:?set FLAVOR via --export}"
: "${NUM_SHARDS:?set NUM_SHARDS via --export}"

conda activate "$MAIN_ENV"
# concatenate shards (validates completeness), then pack the full target into the zarr
python -m sarizard.pretraining.features.compute_target \
    --flavor "$FLAVOR" --merge-shards --num-shards "$NUM_SHARDS"
python -m sarizard.pretraining.features.pack_target --flavor "$FLAVOR"
