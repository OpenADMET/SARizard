#!/usr/bin/env bash
# Download and prepare the shared pretraining corpus from Zenodo (CORPUS_FILE/CORPUS_N in
# env.sh, default the 250K screening corpus). Single task, CPU only. Resumable: if the corpus
# parquet already exists this job exits immediately.
#
# Submit from the repo root:
#   sbatch slurm/prepare_corpus.sbatch
#   CORPUS_FILE=corpus/corpus_full.parquet CORPUS_N=1000000 sbatch slurm/prepare_corpus.sbatch
#SBATCH --job-name=sar-corpus
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/corpus_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

if [[ -f "$CORPUS_FILE" ]]; then
    echo "$CORPUS_FILE exists; skipping"
    exit 0
fi

python -m sarizard.corpus.prepare_corpus --n "$CORPUS_N" --out "$REPO_DIR/$CORPUS_FILE"
