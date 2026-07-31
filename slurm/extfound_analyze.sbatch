#!/usr/bin/env bash
# Analyze the external-foundation sweep: collect test metrics for every external foundation and
# the stock-CheMeleon baseline into a dedicated metrics CSV, then render one standalone report
# card per finetune protocol (frozen/reduced/unlocked) showing just the baseline column and the
# external foundations. Single GPU task (evaluate runs inference). Resumable: evaluate caches
# predictions per result dir and skips existing ones.
#
# Submit from the repo root after extfound_finetune (or let run_external_foundations.sh do it).
#SBATCH --job-name=sar-extfound-analyze
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# exclude the known-bad GPU nodes whose ECC/CUDA-init faults have repeatedly killed the
# single-GPU evaluate pass (see analyze.sbatch); override on the CLI if needed
#SBATCH --exclude=iscn008,iscf008
#SBATCH --output=slurm/logs/extfound_analyze_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

EXT_METRICS="results/external_metrics.csv"
CARD_DIR="plots/external_foundations"
mkdir -p "$CARD_DIR"

# the external-foundation result labels to collect (every protocol's <name>__s<seed> variants),
# plus the stock-CheMeleon baseline dirs for each protocol so report_card's --baseline-flavor
# finds them in the same CSV. report_card collapses the __s<seed> variants back to one column and
# averages the seeds.
mapfile -t LABELS < <(extfound_label_list)
for mode in $EXTFOUND_LR_MODES; do
    if [[ "$mode" == "frozen" ]]; then
        base="chemeleon_stock"
    else
        base="chemeleon_stock_${mode}"
    fi
    for stock_dir in "$REPO_DIR"/results/"$base" "$REPO_DIR"/results/"${base}"__s*; do
        [[ -d "$stock_dir" ]] && LABELS+=("$(basename "$stock_dir")")
    done
done

# reload test-set predictions into a dedicated CSV, so this sweep never overwrites the flavor
# sweep's results/metrics.csv
python -m sarizard.analysis.evaluate \
    --accelerator "$ACCELERATOR" --flavors "${LABELS[@]}" --out "$EXT_METRICS"

# endpoints dropped from the reduced cards, matching lr_analyze.sbatch: the cyp1a2 and
# ExpansionRx LogD single-task rows, each of which has a multi-task sibling measuring the same
# endpoint that stays on the card. Scoped to reduced; the frozen and unlocked cards keep every
# row, so their AVERAGE rows run over a different endpoint set and are not comparable to reduced's
CARD_EXCLUDE_RECIPES="${CARD_EXCLUDE_RECIPES:-cyp1a2_st expansionrx_logd_st_rand}"

# one standalone card pair (R² and MAE %-change) per protocol: the columns are exactly the
# external foundations (--columns), and the baseline is that protocol's stock-CheMeleon run.
# frozen reads bare-name rows; reduced/unlocked filter the lr_<mode>__ rows and strip the prefix.
for mode in $EXTFOUND_LR_MODES; do
    exclude=()
    [[ "$mode" == "reduced" && -n "$CARD_EXCLUDE_RECIPES" ]] && \
        exclude=(--exclude-recipe $CARD_EXCLUDE_RECIPES)
    if [[ "$mode" == "frozen" ]]; then
        python -m sarizard.analysis.report_card \
            --metrics-csv "$EXT_METRICS" --out-dir "$CARD_DIR" \
            --columns $EXTFOUND_NAMES --baseline-flavor chemeleon_stock
    else
        python -m sarizard.analysis.report_card \
            --metrics-csv "$EXT_METRICS" --out-dir "$CARD_DIR" --lr-mode "$mode" \
            --columns $EXTFOUND_NAMES --baseline-flavor "chemeleon_stock_${mode}" \
            "${exclude[@]}"
    fi
done

echo "external-foundation analysis complete; cards in $CARD_DIR, metrics in $EXT_METRICS"
