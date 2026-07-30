#!/usr/bin/env bash
# Evaluate the PXR external-test rerun and print the per-phase flavor ranking against the stock
# baseline. Writes a DEDICATED metrics CSV (results/pxr_ext_metrics.csv), never the shared
# results/metrics.csv, and renders no report card: this is a standalone arm printed to the terminal.
# GPU for the evaluate pass (uncached result dirs need model.predict on the external test sets).
#
# Submit after every pxr_ext finetune is on disk (run_pxr_ext.sh drives this):
#   sbatch slurm/pxr_ext_analyze.sbatch
#SBATCH --job-name=sar-pxrext-analyze
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=iscn008,iscf008
#SBATCH --output=slurm/logs/pxr_ext_analyze_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

PXR_CSV="$REPO_DIR/results/pxr_ext_metrics.csv"

# every pxr_ext__<flavor>__s<seed> label that was generated; evaluate skips any without a result dir
mapfile -t LABELS < <(ls -d "$REPO_DIR"/configs/pxr_ext__*/ | xargs -n1 basename)

# reload the external-test predictions for exactly these labels into the dedicated CSV
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors "${LABELS[@]}" --out "$PXR_CSV"

# print the per-phase flavor ranking and the family-wise (Dunnett) delta vs stock
python -m sarizard.analysis.pxr_ext_report --metrics-csv "$PXR_CSV"

echo "pxr_ext comparison printed above; dedicated metrics at $PXR_CSV (report card untouched)"
