#!/usr/bin/env bash
# Analyze the finetune LR experiments: collect metrics for the frozen flavor sweep and every
# LR variant, then render the frozen-vs-reduced-vs-unlocked comparison. Single GPU task
# (evaluate runs inference). Resumable: evaluate caches predictions per result dir.
#
# Submit from the repo root after lr_finetune (or let run_lr_experiments.sh do it). The frozen
# baseline comes from the flavor sweep's own results (results/<flavor>__s<seed>/), so run_all.sh
# must have finished first.
#SBATCH --job-name=sar-lr-analyze
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/lr_analyze_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

# labels to collect: the frozen flavor-sweep variants plus every generated LR variant. The
# frozen labels (<flavor>__s<seed>) give lr_report its baseline; LR labels come from the dirs.
mapfile -t FLAVORS < <(flavor_list)
read -ra SEEDS <<<"$FLAVOR_SEEDS"
LABELS=()
for flavor in "${FLAVORS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        LABELS+=("${flavor}__s${seed}")
    done
done
for dir in "$REPO_DIR"/configs/lr_*/; do
    [[ -d "$dir" ]] || continue
    LABELS+=("$(basename "$dir")")
done

# collect per-(variant, endpoint) metrics into a dedicated CSV, then compare the LR modes
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors "${LABELS[@]}" --out results/lr_metrics.csv
python -m sarizard.analysis.lr_report --metric r2

echo "LR experiments complete; see plots/lr_ranking_r2.* and plots/lr_report_r2.csv"
