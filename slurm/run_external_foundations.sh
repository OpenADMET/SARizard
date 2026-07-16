#!/usr/bin/env bash
# Submit the external-foundation sweep as a SLURM dependency chain: finetune the same endpoints
# on four pretrained checkpoints that carry no target/pretraining in this repo (different
# pretraining datasets and sizes), then render a standalone report card comparing them against
# the shared 5-seed stock-CheMeleon baseline. This is the flavor sweep's finetune-only path
# pointed at external foundations instead of registry-flavor foundations; every stage is
# resumable (existing outputs are skipped).
#
# The four checkpoints are converted (openadmet {hyper_parameters, state_dict}) MPNN foundations.
# The driver copies each into foundations/<name>__s42_mp.pt (so configs.generate can resolve it
# relative to the repo root; a symlink would resolve back out of the tree), validates its format
# and message-passing dims against an existing repo foundation, then finetunes it at 5 seeds under
# all three protocols, matching the flavor legs and the 5-seed stock baseline.
#
# Runs the finetune in batches via submit_batched.sh, which blocks for hours, so launch from a
# persistent shell (interactive allocation or nohup).
#
# Optional overrides (defaults in slurm/env.sh):
#   export EXTFOUND_NAMES="molpile_1M molpile_5M molpile_10M expansion_gen"
#   export EXTFOUND_SEEDS="1 2 3 4 5"
#   export EXTFOUND_LR_MODES="frozen reduced unlocked"
#   export REPO_DIR=...  MAIN_ENV=...  OPENADMET_ENV=...
#
# Usage:
#   bash slurm/run_external_foundations.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# the pinned foundation seed the copied checkpoints are labelled with; every finetune seed is a
# replicate off this one foundation per name (there is no per-seed pretraining here)
FOUNDATION_SEED=42

# source checkpoint for each EXTFOUND_NAMES entry; edit here to add or move a checkpoint
declare -A EXTFOUND_SOURCES=(
    [molpile_1M]="/home/westd1/myscratch/foundation-models/datafiles/foundation_models/molpile_1M_converted.pt"
    [molpile_5M]="/home/westd1/myscratch/foundation-models/datafiles/foundation_models/molpile_5M_converted.pt"
    [molpile_10M]="/home/westd1/myscratch/foundation-models/datafiles/foundation_models/molpile_10M_converted.pt"
    [expansion_gen]="/home/westd1/myscratch/202606_generative_foundation_models/expansion_gen/best_converted.pt"
)

echo "external foundations: $EXTFOUND_NAMES"
echo "seeds: $EXTFOUND_SEEDS   protocols: $EXTFOUND_LR_MODES"

# copy each external checkpoint into foundations/<name>__s42_mp.pt (skip if already copied); a
# copy, not a symlink, because configs.generate resolves the foundation path and relative_to the
# repo root, and resolve() would follow a symlink back out of the tree
mkdir -p "$REPO_DIR/foundations"
for name in $EXTFOUND_NAMES; do
    src="${EXTFOUND_SOURCES[$name]:-}"
    if [[ -z "$src" ]]; then
        echo "ERROR: no source checkpoint mapped for '$name' in EXTFOUND_SOURCES" >&2
        exit 1
    fi
    if [[ ! -f "$src" ]]; then
        echo "ERROR: source checkpoint missing: $src" >&2
        exit 1
    fi
    dest="$REPO_DIR/foundations/${name}__s${FOUNDATION_SEED}_mp.pt"
    if [[ -f "$dest" ]]; then
        echo "  $name: $dest exists; keeping"
    else
        cp "$src" "$dest"
        echo "  $name: copied $src -> $dest"
    fi
done

# validate every copied foundation against an existing repo foundation before fanning out: the
# openadmet loader expects {hyper_parameters, state_dict} and bakes the message-passing input
# dims (d_v, d_e) into the checkpoint, so a dim mismatch degrades the foundation silently rather
# than erroring. Fail here instead of discovering it 1440 finetune tasks in.
echo "validating foundation format and message-passing dims..."
conda run -n "$OPENADMET_ENV" python - "$REPO_DIR" $EXTFOUND_NAMES <<'PY'
import sys
from pathlib import Path

import torch

repo = Path(sys.argv[1])
names = sys.argv[2:]

# a known-good in-repo foundation as the dims reference
refs = sorted(p for p in (repo / "foundations").glob("*__s42_mp.pt")
              if not p.name.startswith(tuple(f"{n}__" for n in names)))
if not refs:
    sys.exit("no existing repo foundation to validate dims against")
ref_hp = torch.load(refs[0], weights_only=True)["hyper_parameters"]
ref_dims = (ref_hp["d_v"], ref_hp["d_e"])

for name in names:
    path = repo / "foundations" / f"{name}__s42_mp.pt"
    ckpt = torch.load(path, weights_only=True)
    missing = {"hyper_parameters", "state_dict"} - set(ckpt)
    if missing:
        sys.exit(f"{name}: checkpoint missing keys {missing}; not an openadmet foundation")
    hp = ckpt["hyper_parameters"]
    dims = (hp["d_v"], hp["d_e"])
    if dims != ref_dims:
        sys.exit(f"{name}: message-passing dims {dims} != reference {ref_dims} "
                 f"({refs[0].name}); featurizer mismatch, would degrade the foundation")
    print(f"  {name}: ok (d_v={dims[0]}, d_e={dims[1]})")
print("all external foundations validated")
PY

# generate finetuning recipes for each (name, seed, protocol) off the copied foundation: frozen
# carries no suffix, reduced/unlocked add lr_<mode>__ so the protocols land in their own result
# dirs and report_card/filter_lr_mode pick them apart the same way they do for the flavors. This
# only reads templates and writes YAML.
echo "generating per-(name, seed, protocol) finetuning configs..."
for name in $EXTFOUND_NAMES; do
    foundation="$REPO_DIR/foundations/${name}__s${FOUNDATION_SEED}_mp.pt"
    for seed in $EXTFOUND_SEEDS; do
        for mode in $EXTFOUND_LR_MODES; do
            if [[ "$mode" == "frozen" ]]; then
                out_subdir="${name}__s${seed}"
            else
                out_subdir="lr_${mode}__${name}__s${seed}"
            fi
            conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
                --foundation "$foundation" --out-subdir "$out_subdir" \
                --finetune-seed "$seed" --mpnn-lr-mode "$mode"
        done
    done
done
N_RECIPES=$(extfound_recipe_list | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no external-foundation recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"
echo ""

# run the finetune array in batches via submit_batched.sh (submit a batch, wait, rerun its
# failures, then the next); the recipes point at foundations that already exist, so no upstream
# SLURM dependency. Propagate the sweep scoping so a scoped run's sbatch tasks enumerate the same
# recipe set through extfound_recipe_list.
EXPORT="ALL,EXTFOUND_NAMES=$EXTFOUND_NAMES,EXTFOUND_SEEDS=$EXTFOUND_SEEDS,EXTFOUND_LR_MODES=$EXTFOUND_LR_MODES"
echo "finetune   running in batches (submit_batched.sh)..."
"$SCRIPT_DIR/submit_batched.sh" "$SCRIPT_DIR/extfound_finetune.sbatch" extfound_recipe_list \
    --export="$EXPORT"
echo "finetune   complete ($N_RECIPES recipes)"

# every finetune is verified complete, so analyze needs no SLURM dependency
JOB_ANALYZE=$(sbatch --parsable --export="$EXPORT" "$SCRIPT_DIR/extfound_analyze.sbatch")
echo "analyze    job=$JOB_ANALYZE"

echo ""
echo "external-foundation sweep submitted; when done, read the cards in plots/external_foundations/"
echo "  tail -f $REPO_DIR/slurm/logs/extfound_analyze_${JOB_ANALYZE}.out"
