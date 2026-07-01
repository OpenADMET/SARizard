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

# main conda environment: training, target packing, finetuning, and analysis
MAIN_ENV="${MAIN_ENV:-sarizard}"

# lightning accelerator for training and prediction
ACCELERATOR="${ACCELERATOR:-gpu}"

# the representative continuous flavor driven through the prescaling ablation triage
ABLATION_FLAVOR="${ABLATION_FLAVOR:-osmordred}"

# training seeds for the prescaling triage (space-separated). Multiple seeds estimate the
# seed-driven variance the prescaling effect must clear; kept to the single ABLATION_FLAVOR.
ABLATION_SEEDS="${ABLATION_SEEDS:-42}"

# training seeds for the flavor sweep (and the LR experiments, which reuse its foundations).
# Each seed is a separate pretraining run -> its own foundation, recipes, and results, tagged
# <flavor>__s<seed>; the report averages them back to one column per flavor. Add seeds and
# re-run to accumulate: existing (flavor, seed) foundations/results are skipped.
FLAVOR_SEEDS="${FLAVOR_SEEDS:-42}"

# make `conda activate` work in a non-interactive batch shell
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$REPO_DIR"
mkdir -p slurm/logs

# print the registry flavor list, one per line, using the main environment
flavor_list() {
    conda run -n "$MAIN_ENV" python -c \
        "from sarizard.pretraining.flavors import flavor_names; print('\n'.join(flavor_names()))"
}

# print the prescaling ablation list, one per line, using the main environment
ablation_list() {
    conda run -n "$MAIN_ENV" python -c \
        "from sarizard.pretraining.prescaling import ablation_names; print('\n'.join(ablation_names()))"
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
