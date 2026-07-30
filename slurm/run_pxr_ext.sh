#!/usr/bin/env bash
# Drive the PXR external-test rerun end to end: finetune every flavor (plus the stock baseline) on
# the PXR training pool under the reduced protocol, then evaluate each on the two OpenADMET
# PXR-challenge test phases (Phase 1 and Phase 2) and print the per-phase flavor ranking.
#
# Unlike the sweep PXR endpoint (inline Butina ClusterSplitter, split moves with the seed), this
# uses a fixed 90/10 train/val on data/pxr_pec50.parquet and the two challenge phases as external
# held-out test sets (data/splits/pxr_*; build them with
# `python -m sarizard.analysis.build_pxr_ext_splits`). It is standalone: namespaced pxr_ext__* so it
# never enters the flavor sweep, the LR experiments, the stock baseline, or the report card, and its
# metrics land in a dedicated results/pxr_ext_metrics.csv.
#
# The finetune stage runs in batches via submit_batched.sh, which BLOCKS for hours, so launch this
# from a persistent shell (interactive allocation or a durable cpu wrap job). Every stage is
# resumable: a recipe whose result dir already exists is skipped.
#
# Usage (export REPO_DIR to dodge a stale interactive SLURM_SUBMIT_DIR, per the repo convention):
#   REPO_DIR=/scratch/choderaj/westd/SARizard bash slurm/run_pxr_ext.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPT_DIR/env.sh"

FOUNDATION_SEED=42
FINETUNE_SEEDS="${FINETUNE_SEEDS:-1 2 3 4 5}"

# labels to run: the whole registry plus the stock reference by default. PXR_FLAVORS narrows it,
# which is what deepening one arm needs: adding control seeds with PXR_FLAVORS=chemeleon_stock
# leaves the treatment arms at their existing seeds, where generating over the full registry would
# silently add the same seeds to all 15 flavors and change what the comparison measures. Exported
# so submit_batched.sh and the sbatch --export=ALL array see the same scoping the generation used.
if [[ -z "${PXR_FLAVORS:-}" ]]; then
    PXR_FLAVORS="$(conda run -n "$MAIN_ENV" python -c \
        "from sarizard.pretraining.flavors import flavor_names; print(' '.join(flavor_names()))") chemeleon_stock"
fi
export PXR_FLAVORS FINETUNE_SEEDS

# preflight: the recipes point at the flavor-sweep s42 foundations (stock downloads its own), and
# the external-test split files must already exist
MISSING=""
for flavor in $PXR_FLAVORS; do
    # the stock reference downloads the released checkpoint, so it has no foundation to check
    [[ "$flavor" == "chemeleon_stock" ]] && continue
    foundation="foundations/${flavor}__s${FOUNDATION_SEED}_mp.pt"
    [[ -f "$REPO_DIR/$foundation" ]] || MISSING+=" $foundation"
done
if [[ -n "$MISSING" ]]; then
    echo "ERROR: missing s$FOUNDATION_SEED foundations:$MISSING (run slurm/run_all.sh first)" >&2
    exit 1
fi
for f in pxr_ext_train pxr_ext_val pxr_test_phase1 pxr_test_phase2; do
    [[ -f "$REPO_DIR/data/splits/$f.csv" ]] || {
        echo "ERROR: data/splits/$f.csv missing; run: python -m sarizard.analysis.build_pxr_ext_splits" >&2
        exit 1
    }
done

# generate the reduced-protocol pxr_ext recipes for the selected labels, all finetuning off the
# one s42 foundation (chemeleon_stock downloads the released checkpoint instead)
echo "generating pxr_ext recipes (reduced, seeds $FINETUNE_SEEDS, flavors: $PXR_FLAVORS)..."
conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
    --baseline-dir "$REPO_DIR/configs/_pxr_ext" \
    --flavors $PXR_FLAVORS \
    --seeds $FINETUNE_SEEDS --foundation-seed "$FOUNDATION_SEED" \
    --mpnn-lr-mode reduced --label-prefix pxr_ext

N_RECIPES=$(pxr_ext_recipe_list | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no pxr_ext recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "finetune  running $N_RECIPES recipes in batches (submit_batched.sh)..."
"$SCRIPT_DIR/submit_batched.sh" "$SCRIPT_DIR/pxr_ext_finetune.sbatch" pxr_ext_recipe_list \
    --export=ALL
echo "finetune  complete (all $N_RECIPES recipes)"

# evaluate into the dedicated CSV and print the per-phase ranking (GPU); no SLURM dependency since
# every finetune is verified complete above
JOB_ANALYZE=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/pxr_ext_analyze.sbatch")
echo "analyze   job=$JOB_ANALYZE"
echo ""
echo "pxr_ext finetuned; evaluate+print submitted. Watch the ranking with:"
echo "  tail -f $REPO_DIR/slurm/logs/pxr_ext_analyze_${JOB_ANALYZE}.out"
