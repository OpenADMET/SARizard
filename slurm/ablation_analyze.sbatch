#!/usr/bin/env bash
# Analyze the prescaling ablation triage: collect test metrics for every ablation, then
# render the ablation report card and ranking that pick the production prescaling recipe.
# Single GPU task (evaluate runs inference). Resumable: evaluate caches predictions per
# result dir and skips existing ones.
#
# Submit from the repo root after ablation_finetune (or let run_ablations.sh do it).
#SBATCH --job-name=sar-abl-analyze
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/abl_analyze_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$MAIN_ENV"

# the ablation result labels (ablation_<name>) evaluate should collect
mapfile -t ABLATIONS < <(ablation_list)
LABELS=()
for ablation in "${ABLATIONS[@]}"; do
    LABELS+=("ablation_$ablation")
done

# collect per-(ablation, endpoint) metrics into a dedicated CSV, then compare ablations
python -m analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors "${LABELS[@]}" --out results/ablation_metrics.csv
python -m analysis.prescaling_report --metric r2

echo "ablation triage complete; see analysis/plots/prescaling_report_r2.* and ranking"
