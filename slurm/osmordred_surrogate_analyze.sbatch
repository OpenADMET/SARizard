#!/usr/bin/env bash
# Evaluate the osmordred_surrogate control and print its comparison against the stock baseline,
# the sweep osmordred (same target, PubChem corpus), and surrogate_adme (same corpus, surrogate
# target). Writes a DEDICATED metrics CSV (results/osmordred_surrogate_metrics.csv), never the
# shared results/metrics.csv, and renders no report card: the control is a standalone arm printed
# to the terminal only. GPU for the evaluate pass (uncached control result dirs need model.predict).
#
# Submit after the control's 5 finetune seeds are on disk (run_osmordred_surrogate.sh drives this):
#   sbatch slurm/osmordred_surrogate_analyze.sbatch
#SBATCH --job-name=sar-osurr-analyze
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# exclude the known-bad GPU nodes by default (uncorrectable-ECC/CUDA-init faults have repeatedly
# killed single-GPU evaluate passes); override on the CLI if needed
#SBATCH --exclude=iscn008,iscf008
#SBATCH --output=slurm/logs/osurr_analyze_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

CONTROL_CSV="$REPO_DIR/results/osmordred_surrogate_metrics.csv"

# the labels to evaluate into the dedicated CSV: the control (seeds 1-5), the stock baseline
# (bare seed-42 run plus seeds 1-5), and the two context arms (sweep osmordred and surrogate_adme,
# each seeds 1-5 plus the bare seed-42 sweep run). evaluate skips any label with no result dir, so
# listing a seed that was never run is harmless.
LABELS=(chemeleon_stock)
for base in osmordred_surrogate chemeleon_stock osmordred surrogate_adme; do
    for seed in 1 2 3 4 5 42; do
        LABELS+=("${base}__s${seed}")
    done
done

# reload the test-set predictions for exactly these labels into the dedicated CSV; the control
# dirs are uncached so this needs the GPU, the others reuse their cached y_pred.npy
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors "${LABELS[@]}" --out "$CONTROL_CSV"

# print the comparison table, the Welch test vs baseline, and the chemical-space read
python -m sarizard.analysis.control_report --metrics-csv "$CONTROL_CSV"

echo "control comparison printed above; dedicated metrics at $CONTROL_CSV (report card untouched)"
