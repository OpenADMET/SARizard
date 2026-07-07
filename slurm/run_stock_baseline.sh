#!/usr/bin/env bash
# Generate and finetune the stock-CheMeleon reference: the released CheMeleon checkpoint
# (anvil downloads it, no local pretraining) finetuned on every ADMET endpoint under the
# same recipe templates every flavor uses. This is the external "how much does any of our
# custom pretraining help over off-the-shelf CheMeleon" reference for the report card
# (sarizard.analysis.report_card's --baseline-flavor).
#
# One-time: unlike run_all.sh, this does not depend on the corpus or pretraining regime, so it
# does not need to be rerun when either changes. Resumable: an existing results/chemeleon_stock/
# <recipe>/ dir is skipped.
#
# Usage:
#   bash slurm/run_stock_baseline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "generating stock-CheMeleon baseline recipes..."
conda run -n "$MAIN_ENV" python -m sarizard.configs.generate --stock-baseline
N_RECIPES=$(ls "$REPO_DIR"/configs/chemeleon_stock/*.yaml 2>/dev/null | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no stock-baseline recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"

JOB_FINETUNE=$(sbatch --parsable \
    --array=0-$((N_RECIPES - 1)) \
    "$SCRIPT_DIR/finetune_stock_baseline.sbatch")
echo "stock-finetune job=$JOB_FINETUNE  ($N_RECIPES recipes)"

echo ""
echo "when done, results/chemeleon_stock/ is picked up automatically by:"
echo "  python -m sarizard.analysis.evaluate --flavors chemeleon_stock <other flavors...>"
echo "  python -m sarizard.analysis.report_card --metric r2   # includes it as a reference column"
