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
conda activate "$OPENADMET_ENV"

# the ablation result labels evaluate should collect: every generated (ablation, seed) variant
# and its finetune-protocol variants (ablation_<name>__s<seed>[__reduced|__unlocked]), taken from
# the config dirs so analyze covers exactly the protocols that were finetuned. prescaling_report
# averages the seeds per ablation, reports each protocol, and (when more than one protocol is
# present) writes a cross-protocol comparison of each recipe's mean metric.
mapfile -t LABELS < <(ablation_label_list)
if [[ ${#LABELS[@]} -eq 0 ]]; then
    echo "no ablation recipe dirs found under configs/; run run_ablations.sh first" >&2
    exit 1
fi

# collect per-(variant, endpoint) metrics into a dedicated CSV, then compare ablations
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors "${LABELS[@]}" --out results/ablation_metrics.csv
python -m sarizard.analysis.prescaling_report --metric r2

echo "ablation triage complete; see plots/prescaling_report_r2*.* and prescaling_mode_comparison_r2.*"
