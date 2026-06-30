#!/usr/bin/env bash
# Submit the full prescaling-ablation triage as a SLURM dependency chain.
# Run this once from the repo root, before the flavor sweep, to decide which descriptor
# prescaling recipe to bake into the core workflow. Every stage is resumable (existing
# outputs are skipped).
#
# The triage drives one representative continuous flavor (ABLATION_FLAVOR, default osmordred)
# through every prescaling ablation: prescale -> pretrain -> finetune on all endpoints ->
# compare. The backbone, corpus, and regime are identical across ablations, so the report
# card isolates the prescaling effect.
#
# Required before running:
#   conda env create -f envs/osmordred.yml     # the ABLATION_FLAVOR's target environment
#
# Optional overrides (defaults in slurm/env.sh):
#   export ABLATION_FLAVOR=osmordred
#   export REPO_DIR=...  MAIN_ENV=...  ACCELERATOR=...
#
# Usage:
#   bash slurm/run_ablations.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# number of ablations sets the prescale/pretrain array ranges
mapfile -t ABLATIONS < <(ablation_list)
N_ABL=${#ABLATIONS[@]}
if [[ "$N_ABL" -eq 0 ]]; then
    echo "ERROR: no ablations registered in pretraining/prescaling.py" >&2
    exit 1
fi
echo "ablations ($N_ABL): ${ABLATIONS[*]}"

# generate per-ablation finetuning recipes pointing at each ablation foundation; this only
# reads templates and writes YAML, so it runs now even though the foundations do not exist yet
echo "generating per-ablation finetuning configs..."
for ablation in "${ABLATIONS[@]}"; do
    conda run -n "$MAIN_ENV" python -m configs.generate \
        --foundation "$REPO_DIR/foundations/ablation_${ablation}_mp.pt" \
        --out-subdir "ablation_${ablation}"
done
N_RECIPES=$(ls "$REPO_DIR"/configs/ablation_*/*.yaml 2>/dev/null | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no ablation recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"
echo ""

# submit the chain; each stage waits for all tasks of the previous to succeed (afterok)
JOB_CORPUS=$(sbatch --parsable "$SCRIPT_DIR/prepare_corpus.sbatch")
echo "corpus     job=$JOB_CORPUS"

JOB_TARGET=$(sbatch --parsable \
    --dependency=afterok:"$JOB_CORPUS" \
    "$SCRIPT_DIR/ablation_target.sbatch")
echo "target     job=$JOB_TARGET  (after corpus $JOB_CORPUS)"

JOB_PRESCALE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_TARGET" \
    --array=0-$((N_ABL - 1)) \
    "$SCRIPT_DIR/ablation_prescale.sbatch")
echo "prescale   job=$JOB_PRESCALE  (after target $JOB_TARGET)"

JOB_PRETRAIN=$(sbatch --parsable \
    --dependency=afterok:"$JOB_PRESCALE" \
    --array=0-$((N_ABL - 1)) \
    "$SCRIPT_DIR/ablation_pretrain.sbatch")
echo "pretrain   job=$JOB_PRETRAIN  (after prescale $JOB_PRESCALE)"

JOB_FINETUNE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_PRETRAIN" \
    --array=0-$((N_RECIPES - 1)) \
    "$SCRIPT_DIR/ablation_finetune.sbatch")
echo "finetune   job=$JOB_FINETUNE  (after pretrain $JOB_PRETRAIN)"

JOB_ANALYZE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_FINETUNE" \
    "$SCRIPT_DIR/ablation_analyze.sbatch")
echo "analyze    job=$JOB_ANALYZE  (after finetune $JOB_FINETUNE)"

echo ""
echo "ablation triage submitted; monitor with:"
echo "  watch squeue -u \$USER"
echo "  tail -f $REPO_DIR/slurm/logs/abl_analyze_${JOB_ANALYZE}.out"
echo "when done, read analysis/plots/prescaling_ranking_r2.csv to pick the production recipe"
