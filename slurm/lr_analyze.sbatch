#!/usr/bin/env bash
# Analyze the finetune LR experiments: collect metrics for the frozen flavor sweep and the LR
# variants of the requested protocols, render each protocol's two report cards (R² with per-cell
# seed std, and MAE %-change gated by a per-cell significance test against that protocol's stock
# baseline), then the frozen-vs-LR ranking. Single GPU task (evaluate runs inference). Resumable:
# evaluate caches predictions per result dir.
#
# Scoped by LR_MODES (default "reduced unlocked"): only those protocols' lr_<mode>__ recipe dirs
# and chemeleon_stock_<mode> baselines are evaluated and carded, so a still-running protocol can be
# left out by narrowing LR_MODES (e.g. LR_MODES=reduced while unlocked finetunes).
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
# exclude the known-bad GPU nodes: their ECC/CUDA-init faults fail every uncached model.predict
# in the evaluate pass (see analyze.sbatch). Override on the CLI if needed.
#SBATCH --exclude=iscn008,iscf008
#SBATCH --output=slurm/logs/lr_analyze_%j.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"
conda activate "$OPENADMET_ENV"

read -ra MODES <<<"${LR_MODES:-reduced unlocked}"

# labels to collect: the frozen flavor-sweep variants (lr_report's baseline), plus, for each
# requested protocol, its lr_<mode>__ recipe dirs and its multi-seed stock baseline dirs
# (chemeleon_stock_<mode>__s<seed>). report_card/meta_model collapse the seeds per label.
mapfile -t FLAVORS < <(flavor_list)
read -ra SEEDS <<<"$FLAVOR_SEEDS"
LABELS=()
for flavor in "${FLAVORS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        LABELS+=("${flavor}__s${seed}")
    done
done
for mode in "${MODES[@]}"; do
    for dir in "$REPO_DIR"/configs/lr_${mode}__*/; do
        [[ -d "$dir" ]] && LABELS+=("$(basename "$dir")")
    done
    for sdir in "$REPO_DIR"/results/chemeleon_stock_${mode}__s*; do
        [[ -d "$sdir" ]] && LABELS+=("$(basename "$sdir")")
    done
done

# collect per-(variant, endpoint) metrics into a dedicated CSV
python -m sarizard.analysis.evaluate --accelerator "$ACCELERATOR" \
    --flavors "${LABELS[@]}" --out results/lr_metrics.csv

# render each protocol's two report cards against that protocol's own stock baseline; the cards
# read the shared lr_metrics.csv and filter to the mode's rows internally (--lr-mode)
for mode in "${MODES[@]}"; do
    python -m sarizard.analysis.report_card --lr-mode "$mode" \
        --metrics-csv results/lr_metrics.csv --baseline-flavor "chemeleon_stock_${mode}"
done

# frozen-vs-LR ranking across whatever protocols are present; non-fatal so a partial-mode run
# still leaves the report cards above
python -m sarizard.analysis.lr_report --metric r2 || echo "lr_report skipped (partial LR modes)"

echo "LR analyze complete for modes '${MODES[*]}'; see plots/report_card_*_<mode>.* and lr_metrics.csv"
