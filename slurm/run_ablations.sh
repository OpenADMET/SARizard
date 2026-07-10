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
#   conda activate sarizard-osmordred && bash envs/build_osmordred.sh && conda deactivate
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

# FOUNDATION_SEED set => finetune-only replication: every ABLATION_SEED finetunes the one
# ablation foundation pretrained at FOUNDATION_SEED, so we skip corpus/target/prescale/pretrain
# (those outputs already exist) and submit only finetune + analyze. Each seed labels its recipes
# ablation_<name>__s<seed>[__<mode>] off that pinned foundation, and prescaling_report averages
# the seeds per ablation. Unset => the legacy full pipeline (one foundation per seed).
FINETUNE_ONLY=""
[[ -n "${FOUNDATION_SEED:-}" ]] && FINETUNE_ONLY=1

# ablations x seeds set the array ranges; prescale is seed-independent (one task per ablation)
mapfile -t ABLATIONS < <(ablation_list)
N_ABL=${#ABLATIONS[@]}
if [[ "$N_ABL" -eq 0 ]]; then
    echo "ERROR: no ablations registered in pretraining/prescaling.py" >&2
    exit 1
fi
read -ra SEEDS <<<"$ABLATION_SEEDS"
N_SEEDS=${#SEEDS[@]}
N_PRETRAIN=$(( N_ABL * N_SEEDS ))
echo "ablations ($N_ABL): ${ABLATIONS[*]}"
echo "seeds ($N_SEEDS): ${SEEDS[*]}  (flavor: $ABLATION_FLAVOR)"

# finetune-only preflight: every recipe initializes from the one s<FOUNDATION_SEED> foundation
# per ablation, so gate on all of them existing before submitting rather than discovering the
# gap task by task
if [[ -n "$FINETUNE_ONLY" ]]; then
    MISSING=""
    for ablation in "${ABLATIONS[@]}"; do
        foundation="foundations/ablation_${ablation}__s${FOUNDATION_SEED}_mp.pt"
        [[ -f "$REPO_DIR/$foundation" ]] || MISSING+=" $foundation"
    done
    if [[ -n "$MISSING" ]]; then
        echo "ERROR: finetune-only requires the s$FOUNDATION_SEED ablation foundations; missing:$MISSING" >&2
        exit 1
    fi
    echo "finetune-only: $N_SEEDS seeds finetune the s$FOUNDATION_SEED ablation foundations"
fi

# generate finetuning recipes for each (ablation, seed, protocol) variant, pointing at that
# variant's foundation; this only reads templates and writes YAML, so it runs before the
# foundations exist. Frozen carries no suffix; reduced/unlocked add __<mode> so each protocol
# gets its own recipes and result dir off the same foundation (ABLATION_LR_MODES selects them).
# In finetune-only mode every seed's recipes point at the one s<FOUNDATION_SEED> foundation and
# write the seed into the recipe (--finetune-seed), so the seeds are finetune replicates; the
# out-subdir still labels by the finetune seed so their result dirs and the seed-averaging split.
read -ra LR_MODE_LIST <<<"$ABLATION_LR_MODES"
echo "protocols (${#LR_MODE_LIST[@]}): ${LR_MODE_LIST[*]}"
echo "generating per-(ablation, seed, protocol) finetuning configs..."
for ablation in "${ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # pin the foundation seed when finetune-only, else it tracks the finetune seed (legacy)
        foundation_seed="${FOUNDATION_SEED:-$seed}"
        for mode in "${LR_MODE_LIST[@]}"; do
            suffix=""
            [[ "$mode" != "frozen" ]] && suffix="__${mode}"
            extra=()
            [[ -n "$FINETUNE_ONLY" ]] && extra=(--finetune-seed "$seed")
            conda run -n "$MAIN_ENV" python -m sarizard.configs.generate \
                --foundation "$REPO_DIR/foundations/ablation_${ablation}__s${foundation_seed}_mp.pt" \
                --out-subdir "ablation_${ablation}__s${seed}${suffix}" \
                --mpnn-lr-mode "$mode" "${extra[@]}"
        done
    done
done
N_RECIPES=$(ls "$REPO_DIR"/configs/ablation_*/*.yaml 2>/dev/null | wc -l | tr -d ' ')
if [[ "$N_RECIPES" -eq 0 ]]; then
    echo "ERROR: no ablation recipes generated; check configs/generate.py" >&2
    exit 1
fi
echo "  $N_RECIPES recipes (finetune array 0-$((N_RECIPES - 1)))"
echo ""

# submit the chain; each stage waits for all tasks of the previous to succeed (afterok).
# finetune-only skips corpus/target/prescale/pretrain (the pinned foundations already exist) and
# submits the finetune array with no upstream dependency; the legacy path builds the chain first.
FINETUNE_DEPENDENCY=""
if [[ -z "$FINETUNE_ONLY" ]]; then
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
        --array=0-$((N_PRETRAIN - 1)) \
        "$SCRIPT_DIR/ablation_pretrain.sbatch")
    echo "pretrain   job=$JOB_PRETRAIN  (after prescale $JOB_PRESCALE, $N_PRETRAIN ablation x seed)"
    FINETUNE_DEPENDENCY="--dependency=afterok:$JOB_PRETRAIN"
fi

# run the finetune array in batches of BATCH_SIZE (default 50) via submit_batched.sh: submit a
# batch, wait for it, rerun its failures, then the next. It blocks until every ablation recipe has
# a trained model.pth (or exits nonzero, stopping set -e before analyze), so this runs for hours;
# launch this script from a persistent shell (interactive allocation or nohup). $FINETUNE_DEPENDENCY
# gates the first batch on pretrain in the legacy path and is empty in finetune-only mode.
echo "finetune   running in batches (submit_batched.sh)..."
"$SCRIPT_DIR/submit_batched.sh" "$SCRIPT_DIR/ablation_finetune.sbatch" ablation_recipe_list \
    $FINETUNE_DEPENDENCY
echo "finetune   complete ($N_RECIPES recipes)"

# every finetune is verified complete, so analyze needs no SLURM dependency
JOB_ANALYZE=$(sbatch --parsable \
    "$SCRIPT_DIR/ablation_analyze.sbatch")
echo "analyze    job=$JOB_ANALYZE"

echo ""
echo "ablation triage submitted; monitor with:"
echo "  watch squeue -u \$USER"
echo "  tail -f $REPO_DIR/slurm/logs/abl_analyze_${JOB_ANALYZE}.out"
echo "when done, read plots/prescaling_ranking_r2.csv to pick the production recipe"
echo "  (with several ABLATION_LR_MODES, also plots/prescaling_mode_comparison_r2.csv)"
