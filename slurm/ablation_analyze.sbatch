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

# add the stock-CheMeleon baseline the cards diverge against: for each finetuned protocol, its
# per-seed baseline dirs (chemeleon_stock__s<seed> for frozen, chemeleon_stock_<mode>__s<seed>
# otherwise), plus the legacy bare frozen dir. prescaling_report looks these up by label as the
# reference column and the MAE-delta baseline, so they must be in ablation_metrics.csv too.
# Derive the protocols from the ablation result dirs actually present rather than ABLATION_LR_MODES,
# so a submit that does not export the env var (e.g. a manual rerun) cannot silently drop a
# protocol's baseline and skip its MAE-delta card. Frozen is always present.
MODES=(frozen)
compgen -G "$REPO_DIR/configs/ablation_*__s*__reduced" >/dev/null && MODES+=(reduced)
compgen -G "$REPO_DIR/configs/ablation_*__s*__unlocked" >/dev/null && MODES+=(unlocked)
for mode in "${MODES[@]}"; do
    if [[ "$mode" == "frozen" ]]; then base="chemeleon_stock"; else base="chemeleon_stock_${mode}"; fi
    for sdir in "$REPO_DIR"/results/${base}__s* "$REPO_DIR"/results/${base}; do
        [[ -d "$sdir" ]] && LABELS+=("$(basename "$sdir")")
    done
done

# endpoints dropped from the reduced cards, matching lr_analyze.sbatch: the cyp1a2 and
# ExpansionRx LogD single-task rows, each of which has a multi-task sibling measuring the same
# endpoint that stays on the card. prescaling_report scopes them to the reduced protocol, so the
# frozen and unlocked cards keep every row and their AVERAGE rows are not comparable to reduced's
CARD_EXCLUDE_RECIPES="${CARD_EXCLUDE_RECIPES:-cyp1a2_st expansionrx_logd_st_rand}"

# collect per-(variant, endpoint) metrics into a dedicated CSV, then compare ablations
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors "${LABELS[@]}" --out results/ablation_metrics.csv
exclude=()
[[ -n "$CARD_EXCLUDE_RECIPES" ]] && exclude=(--exclude-recipe $CARD_EXCLUDE_RECIPES)
python -m sarizard.analysis.prescaling_report --metric r2 "${exclude[@]}"

echo "ablation triage complete; see plots/prescaling_report_r2*.* and prescaling_mode_comparison_r2.*"
