#!/usr/bin/env bash
# Shared runtime configuration for the SARizard SLURM jobs, sourced by each job script after
# its #SBATCH block. The #SBATCH resource directives (partition, account, time, gpu) are read
# from each script's header BEFORE this file runs, so set those in the headers (or override on
# the sbatch command line); set everything else here or export it before submitting.
#
# set -u is intentionally omitted: conda's activation scripts reference unbound shell vars.
set -eo pipefail

# repo root; submit from the repo root so SLURM_SUBMIT_DIR points here, or export REPO_DIR
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# main conda environment: target computation/packing, pretraining, and split
MAIN_ENV="${MAIN_ENV:-sarizard}"

# environment for the openadmet-models CLI (finetune) and analysis: openadmet-models declares
# its deps only in its conda-env file, and that stack (pandas 2.x, torch 2.7, py3.12) conflicts
# with the main training env (pandas 3.x, torch 2.12, py3.11), so it lives in its own env
OPENADMET_ENV="${OPENADMET_ENV:-openadmet}"

# lightning accelerator for training and prediction
ACCELERATOR="${ACCELERATOR:-gpu}"

# the representative continuous flavor driven through the prescaling ablation triage
ABLATION_FLAVOR="${ABLATION_FLAVOR:-osmordred}"

# training seeds for the prescaling triage (space-separated). Multiple seeds estimate the
# seed-driven variance the prescaling effect must clear; kept to the single ABLATION_FLAVOR.
ABLATION_SEEDS="${ABLATION_SEEDS:-42}"

# finetune protocols to run the prescaling triage under (space-separated subset of
# frozen/reduced/unlocked). Default frozen keeps the single-protocol triage; add reduced and
# unlocked to check that the winning recipe holds once the MPNN backbone can adapt. Each extra
# protocol adds a full finetune pass off the same ablation foundations
# (configs/ablation_<name>__s<seed>[__<mode>]), so the finetune array grows with the count.
ABLATION_LR_MODES="${ABLATION_LR_MODES:-frozen}"

# training seeds for the flavor sweep (and the LR experiments, which reuse its foundations).
# Each seed is a separate pretraining run -> its own foundation, recipes, and results, tagged
# <flavor>__s<seed>; the report averages them back to one column per flavor. Add seeds and
# re-run to accumulate: existing (flavor, seed) foundations/results are skipped.
FLAVOR_SEEDS="${FLAVOR_SEEDS:-42}"

# finetune learning-rate experiments: the backbone protocols to sweep off the flavor-sweep
# foundations (frozen is the flavor sweep itself, so it is not repeated here)
LR_MODES="${LR_MODES:-reduced unlocked}"

# make `conda activate` work in a non-interactive batch shell
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$REPO_DIR"
mkdir -p slurm/logs

# print the registry flavor list, one per line, using the main environment. the sed drops
# blank lines: conda run appends a trailing newline that would otherwise become an empty
# mapfile element and an off-by-one in every array that sizes itself from this list
flavor_list() {
    conda run -n "$MAIN_ENV" python -c \
        "from sarizard.pretraining.flavors import flavor_names; print('\n'.join(flavor_names()))" \
        | sed '/^$/d'
}

# print the prescaling ablation list, one per line, using the main environment. sed drops the
# trailing blank line conda run adds (see flavor_list) so N_ABL and the array range stay exact
ablation_list() {
    conda run -n "$MAIN_ENV" python -c \
        "from sarizard.pretraining.prescaling import ablation_names; print('\n'.join(ablation_names()))" \
        | sed '/^$/d'
}

# print the generated finetune recipe paths for the flavor sweep, one per line: registry
# flavors x FLAVOR_SEEDS (configs/<flavor>__s<seed>/). Scoping to those exact dirs keeps the
# sweep from sweeping up ablation (ablation_*) or LR-experiment (lr_*) recipe dirs under configs/.
flavor_recipe_list() {
    local flavor seed
    local -a seeds
    read -ra seeds <<<"$FLAVOR_SEEDS"
    while IFS= read -r flavor; do
        [[ -n "$flavor" ]] || continue
        for seed in "${seeds[@]}"; do
            ls "$REPO_DIR/configs/${flavor}__s${seed}"/*.yaml 2>/dev/null
        done
    done < <(flavor_list)
}

# print the generated LR-experiment recipe paths (configs/lr_<mode>__<flavor>__s<seed>/),
# one per line, for the finetune array in run_lr_experiments.sh
lr_recipe_list() {
    ls "$REPO_DIR"/configs/lr_*/*.yaml 2>/dev/null
}

# print the generated ablation result labels (config dir basenames), one per line: every
# (ablation, seed) variant and its finetune-protocol variants (ablation_<name>__s<seed>, plus
# the __reduced/__unlocked suffixes the MPNN-LR sweep adds). This mirrors ablation_finetune's
# own recipe glob, so analyze evaluates exactly the protocols that were finetuned; evaluate skips
# any label without a result dir, and prescaling_report groups the labels back by protocol.
ablation_label_list() {
    local dir
    for dir in "$REPO_DIR"/configs/ablation_*__s*/; do
        [[ -d "$dir" ]] || continue
        basename "$dir"
    done | sort -u
}
