#!/usr/bin/env bash
# Download and prepare the shared 250K pretraining corpus from Zenodo. Single task, CPU only.
# Resumable: if the corpus parquet already exists this job exits immediately.
#
# Submit from the repo root:
#   sbatch slurm/prepare_corpus.sbatch
#SBATCH --job-name=sar-corpus
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/corpus_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

if [[ -f "corpus/corpus_250k.parquet" ]]; then
    echo "corpus/corpus_250k.parquet exists; skipping"
    exit 0
fi

python -m corpus.prepare_corpus
