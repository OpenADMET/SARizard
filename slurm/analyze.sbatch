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
conda activate "$OPENADMET_ENV"

# the per-(flavor, seed) result labels (<flavor>__s<seed>) to collect; report_card and
# meta_model average the seeds back to one column/feature per flavor
mapfile -t FLAVORS < <(flavor_list)
read -ra SEEDS <<<"$FLAVOR_SEEDS"
LABELS=()
for flavor in "${FLAVORS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        LABELS+=("${flavor}__s${seed}")
    done
done

# the stock-CheMeleon reference (slurm/run_stock_baseline.sh) is a corpus-independent run; fold
# every frozen stock seed variant into the same metrics.csv whenever present, so report_card's
# --baseline-flavor finds them without a separate evaluate pass. report_card/meta_model collapse
# the __s<seed> variants back to the bare chemeleon_stock label and average the seeds. The bare
# dir is the original single-seed (seed 42) run; the __s<seed> dirs are the added replicates.
# Match only frozen stock dirs (chemeleon_stock, chemeleon_stock__s<seed>); the reduced/unlocked
# baselines (chemeleon_stock_<mode>...) belong to run_lr_experiments.sh's own analyze pass.
for stock_dir in "$REPO_DIR"/results/chemeleon_stock "$REPO_DIR"/results/chemeleon_stock__s*; do
    [[ -d "$stock_dir" ]] && LABELS+=("$(basename "$stock_dir")")
done

# reload test-set predictions and write metrics.csv
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" --flavors "${LABELS[@]}"

# two report cards (endpoints x flavors), seed variants collapsed: an R² card (red=0, green=1)
# with the stock baseline as the first column, and a MAE %-change-vs-baseline card
python -m sarizard.analysis.report_card

# stacking meta-model vs best single flavor (scored per seed on that seed's aligned split,
# then averaged across seeds; seeds are not stacked at the raw-prediction level because the
# multi-task endpoints resample their test set per finetune seed)
python -m sarizard.analysis.meta_model --flavors "${LABELS[@]}"

echo "analysis complete; results in plots/ and results/metrics.csv"
