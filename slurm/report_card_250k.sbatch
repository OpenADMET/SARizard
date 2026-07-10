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

# render the two report cards for the 250K archive (the 10 archived flavors plus the
# stock-CheMeleon baseline) into their own subdir so the fixed basenames do not clobber the
# live full-corpus cards: an R² card (red=0, green=1, baseline as the first column) and a
# MAE %-change-vs-baseline card
python -m sarizard.analysis.report_card \
    --metrics-csv archive/flavor_sweep_250k/results/metrics_with_references.csv \
    --out-dir plots/report_card_250k

echo "wrote plots/report_card_250k/report_card_r2.png and report_card_mae_delta.png (+ CSVs)"
