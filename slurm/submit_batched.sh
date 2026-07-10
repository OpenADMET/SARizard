#!/usr/bin/env bash
# Drive a finetune array in fixed-size batches: submit BATCH_SIZE tasks, wait for the whole batch
# to finish, rerun any that failed, and only then submit the next batch. This caps how many jobs
# are queued at once and catches a bad-node failure per batch instead of letting one poison a
# 1800-task array and cascade-cancel the analyze stage through its afterok dependency.
#
# A finetune casualty (a node hardware fault or a wall-clock timeout) leaves its result dir with
# the data-prep files but no trained model.pth, so model.pth is the per-recipe success marker; a
# batch is done when every one of its recipes has one. Between attempts the partial dir is removed
# so the sbatch skip-if-exists guard reruns the recipe instead of silently no-op'ing it.
#
# Blocks until every recipe is complete or a batch exhausts its retries (then exits nonzero, so a
# caller with set -e stops before submitting analyze). It waits for hours, so run it inside a
# persistent shell: an interactive allocation, or `nohup bash slurm/run_all.sh >run.log 2>&1 &`.
# The runners (run_all.sh, run_lr_experiments.sh, run_ablations.sh) call it in place of one big
# sbatch --array, then submit analyze once it returns success (no SLURM dependency needed, since
# every finetune is already verified complete).
#
# Usage:
#   submit_batched.sh <sbatch_script> <recipe_list_fn> [sbatch_passthru_args...]
#     <sbatch_script>    the finetune sbatch to run (finetune.sbatch, lr_finetune.sbatch,
#                        ablation_finetune.sbatch)
#     <recipe_list_fn>   an env.sh function that prints the recipe paths one per line, in the SAME
#                        order the sbatch script enumerates them (flavor_recipe_list,
#                        lr_recipe_list, ablation_recipe_list); the array index maps to this order,
#                        so the two must agree for the result-dir bookkeeping to line up
#     passthru           extra args forwarded to every sbatch call (e.g.
#                        --export=ALL,FLAVOR_SEEDS=..., or --dependency=afterok:<pretrain job>)
#
# Knobs (environment, with defaults):
#   BATCH_SIZE=50                 tasks submitted per batch
#   MAX_RETRIES=3                 rerun attempts per batch before giving up
#   EXCLUDE_NODES=iscn008,iscf008 --exclude list for every submit (set empty to disable)
#   POLL_INTERVAL=30              seconds between squeue polls while a batch runs

# match env.sh: set -u is omitted on purpose (conda's activation scripts reference unbound vars)
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SBATCH_SCRIPT="${1:?usage: submit_batched.sh <sbatch_script> <recipe_list_fn> [sbatch args...]}"
RECIPE_FN="${2:?missing recipe_list_fn}"
shift 2
PASSTHRU=("$@")

BATCH_SIZE="${BATCH_SIZE:-50}"
MAX_RETRIES="${MAX_RETRIES:-3}"
EXCLUDE_NODES="${EXCLUDE_NODES-iscn008,iscf008}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

EXCLUDE_ARG=()
[[ -n "$EXCLUDE_NODES" ]] && EXCLUDE_ARG=(--exclude="$EXCLUDE_NODES")

# enumerate recipes in the sbatch script's own order so index i here is index i there
mapfile -t RECIPES < <("$RECIPE_FN")
N=${#RECIPES[@]}
if [[ "$N" -eq 0 ]]; then
    echo "submit_batched: '$RECIPE_FN' produced no recipes; generate them first" >&2
    exit 1
fi

# result dir for recipe index $1: results/<recipe-parent-dir>/<recipe-stem>, matching every
# finetune sbatch's OUT=results/$LABEL/$NAME
out_dir() {
    local recipe="${RECIPES[$1]}"
    echo "results/$(basename "$(dirname "$recipe")")/$(basename "$recipe" .yaml)"
}
# a finetune is complete iff its trained weights exist; a casualty leaves the data-prep dir
# (dataloaders) but never model.pth, so that file is the authoritative, sacct-independent marker
is_done() { [[ -f "$(out_dir "$1")/model.pth" ]]; }

# wait until every task of array job $1 has left the queue (pending, running, or completing)
wait_for_job() {
    local jid="$1"
    sleep 5
    while [[ -n "$(squeue -j "$jid" -h -o '%t' 2>/dev/null)" ]]; do
        sleep "$POLL_INTERVAL"
    done
}

echo "submit_batched: $N recipes, batches of $BATCH_SIZE, exclude='${EXCLUDE_NODES:-none}', retries=$MAX_RETRIES"

batch_start=0
while (( batch_start < N )); do
    batch_end=$(( batch_start + BATCH_SIZE ))
    (( batch_end > N )) && batch_end=$N
    echo ""
    echo "=== batch $(( batch_start / BATCH_SIZE + 1 )): recipes $batch_start-$(( batch_end - 1 )) of $N ==="

    # skip recipes already complete so a rerun of the driver resumes rather than redoing work
    todo=()
    for (( i = batch_start; i < batch_end; i++ )); do
        is_done "$i" || todo+=("$i")
    done
    if [[ ${#todo[@]} -eq 0 ]]; then
        echo "all $(( batch_end - batch_start )) already complete; skipping"
        batch_start=$batch_end
        continue
    fi

    attempt=0
    while [[ ${#todo[@]} -gt 0 ]]; do
        attempt=$(( attempt + 1 ))
        if (( attempt > MAX_RETRIES )); then
            echo "submit_batched: batch $batch_start-$(( batch_end - 1 )) still has ${#todo[@]} failing tasks after $MAX_RETRIES attempts:" >&2
            for i in "${todo[@]}"; do echo "  task $i -> $(out_dir "$i")" >&2; done
            exit 1
        fi
        # clear each casualty's partial dir so the sbatch skip-guard reruns it, not no-ops it
        for i in "${todo[@]}"; do
            d="$(out_dir "$i")"
            [[ -d "$d" ]] && rm -rf "$d"
        done
        arr="$(IFS=,; echo "${todo[*]}")"
        echo "attempt $attempt/$MAX_RETRIES: submitting ${#todo[@]} tasks (array=$arr)"
        jid=$(sbatch --parsable "${EXCLUDE_ARG[@]}" --array="$arr" "${PASSTHRU[@]}" "$SBATCH_SCRIPT")
        echo "  job=$jid; waiting for completion..."
        wait_for_job "$jid"
        # recompute failures from the trained-model marker rather than sacct state
        failed=()
        for i in "${todo[@]}"; do
            is_done "$i" || failed+=("$i")
        done
        if [[ ${#failed[@]} -gt 0 ]]; then
            echo "  ${#failed[@]}/${#todo[@]} failed (node fault or timeout); rerunning"
        else
            echo "  batch complete"
        fi
        todo=("${failed[@]}")
    done
    batch_start=$batch_end
done

echo ""
echo "submit_batched: all $N recipes complete"
