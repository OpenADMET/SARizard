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

# make `conda activate` work in a non-interactive batch shell
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$REPO_DIR"
mkdir -p slurm/logs

# print the registry flavor list, one per line, using the main environment
flavor_list() {
    conda run -n "$MAIN_ENV" python -c \
        "from pretraining.flavors import flavor_names; print('\n'.join(flavor_names()))"
}

# print the prescaling ablation list, one per line, using the main environment
ablation_list() {
    conda run -n "$MAIN_ENV" python -c \
        "from pretraining.prescaling import ablation_names; print('\n'.join(ablation_names()))"
}
