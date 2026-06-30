#!/usr/bin/env bash
# Run the analysis pipeline after all finetuning is done: collect test metrics, write the
# report-card heatmap, and fit the stacking meta-model. Single task, GPU for evaluate.
# Resumable at the file level: evaluate caches per-model y_pred.npy files and skips existing
# ones, so a partial run can be continued by resubmitting.
#
# Submit from the repo root (or let run_all.sh do it):
#   sbatch slurm/analyze.sbatch
#SBATCH --job-name=sar-analyze
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/analyze_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

# reload test-set predictions and write metrics.csv
python -m analysis.evaluate --accelerator "$ACCELERATOR"

# report-card heatmap (endpoints x flavors, R²)
python -m analysis.report_card --metric r2

# stacking meta-model vs best single flavor
python -m analysis.meta_model

echo "analysis complete; results in analysis/plots/ and results/metrics.csv"
