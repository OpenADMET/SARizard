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

# print the generated finetune recipe paths for registry flavors only, one per line.
# Scoping to configs/<flavor>/ (rather than a bare configs/*/*.yaml glob) keeps the flavor
# sweep from sweeping up ablation recipe dirs written under configs/ by configs.generate.
flavor_recipe_list() {
    local flavor
    while IFS= read -r flavor; do
        [[ -n "$flavor" ]] || continue
        ls "$REPO_DIR/configs/$flavor"/*.yaml 2>/dev/null
    done < <(flavor_list)
}
