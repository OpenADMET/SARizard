#!/usr/bin/env bash
# One-off: render a full report card for the archived 250K flavor sweep
# (archive/flavor_sweep_250k/), now that the stock-CheMeleon baseline and meta-model
# reference columns exist. Submit after slurm/run_stock_baseline.sh's finetune completes:
#   sbatch --dependency=afterok:<stock-finetune job> slurm/report_card_250k.sbatch
#SBATCH --job-name=sar-250k-report
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/report_card_250k_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"

# evaluate the stock-CheMeleon baseline into its own tidy CSV; kept separate from the live
# results/metrics.csv so it doesn't race the full-corpus rerun's own analyze stage
conda activate "$OPENADMET_ENV"
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors chemeleon_stock --results results \
    --out results/metrics_chemeleon_stock.csv

# merge the archived 250K flavor-sweep metrics with the (corpus-independent) stock-baseline
# rows into one derived CSV; the frozen archive files themselves are not modified
conda activate "$MAIN_ENV"
python - <<'PY'
import pandas as pd
from pathlib import Path

archive_metrics = Path("archive/flavor_sweep_250k/results/metrics.csv")
stock_metrics = Path("results/metrics_chemeleon_stock.csv")
out = Path("archive/flavor_sweep_250k/results/metrics_with_references.csv")

frame = pd.concat([pd.read_csv(archive_metrics), pd.read_csv(stock_metrics)], ignore_index=True)
frame.to_csv(out, index=False)
print(f"wrote {out} ({len(frame)} rows, {frame['flavor'].nunique()} flavors)")
PY

# render the 250K report card: the 10 archived flavors, plus the stock-CheMeleon baseline
# and the archived meta-model as labeled reference columns
python -m sarizard.analysis.report_card --metric r2 \
    --metrics-csv archive/flavor_sweep_250k/results/metrics_with_references.csv \
    --meta-model-csv archive/flavor_sweep_250k/results/meta_model_lgbm.csv \
    --out plots/report_card_250k_r2.png

echo "wrote plots/report_card_250k_r2.png and plots/report_card_250k_r2.csv"
