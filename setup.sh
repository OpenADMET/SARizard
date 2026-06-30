#!/usr/bin/env bash
# One-shot cluster setup: build every conda environment from envs/*.yml, install the SARizard
# package into each, run the test suite, and print an "okay" status if everything passed.
#
# Each environment's YAML already pins the runtime deps its calculator needs (and the isolated
# envs pin conflicting numpy/rdkit/python versions on purpose), so SARizard is installed with
# --no-deps to avoid clobbering those pins and --ignore-requires-python so the editable install
# also lands in the py3.8/3.10 isolated envs (the package metadata targets py3.11+). The main
# env additionally gets pytest and, if the sibling checkout is present, openadmet-models.
#
# Usage:
#   bash setup.sh                 # build/install all envs, then test
#   bash setup.sh main osmordred  # restrict to specific envs (by envs/<name>.yml basename)
#   FORCE=1 bash setup.sh         # recreate envs that already exist
#   SKIP_OSMORDRED_BUILD=1 ...    # skip the slow osmordred source build
#   OPENADMET_DIR=/path bash ...  # override the openadmet-models checkout location
#
# Exit status is 0 only if every requested step succeeds; the final line is "okay" on success.

# no `set -e`: steps are run through run_step so one failure is recorded and reported rather
# than aborting the whole setup midway
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

MAIN_ENV="sarizard"
OPENADMET_DIR="${OPENADMET_DIR:-$REPO_DIR/../openadmet-models}"
FORCE="${FORCE:-0}"
SKIP_OSMORDRED_BUILD="${SKIP_OSMORDRED_BUILD:-0}"

# result tracking: a status line per step, plus a failure flag and a warnings list
declare -a SUMMARY=()
declare -a WARNINGS=()
FAILED=0

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN:\033[0m %s\n' "$*" >&2; WARNINGS+=("$*"); }

# run a labelled step; record OK/FAIL and keep going so the summary is complete
run_step() {
    local label="$1"; shift
    log "$label"
    if "$@"; then
        SUMMARY+=("OK    $label")
        return 0
    fi
    SUMMARY+=("FAIL  $label")
    FAILED=1
    return 1
}

# is a conda env present (matched on the name column of `conda env list`)
env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

# create one env from a YAML, honoring FORCE; idempotent otherwise
create_env() {
    local yaml="$1" name="$2"
    if env_exists "$name"; then
        if [[ "$FORCE" == "1" ]]; then
            conda env remove -n "$name" -y || return 1
        else
            echo "env $name already exists; skipping create (FORCE=1 to recreate)"
            return 0
        fi
    fi
    conda env create -f "$yaml"
}

# build osmordred from source into its env (C++ extension, no PyPI/conda release)
build_osmordred() {
    if [[ "$SKIP_OSMORDRED_BUILD" == "1" ]]; then
        echo "SKIP_OSMORDRED_BUILD=1; not building osmordred"
        return 0
    fi
    conda run --no-capture-output -n sarizard-osmordred bash envs/build_osmordred.sh
}

# install the SARizard package into one env without disturbing its pinned deps
install_sarizard() {
    local name="$1"
    conda run --no-capture-output -n "$name" \
        python -m pip install --ignore-requires-python --no-deps -e .
}

# main-env extras: pytest for the test suite, openadmet-models if the checkout is present
install_main_extras() {
    conda run --no-capture-output -n "$MAIN_ENV" python -m pip install pytest || return 1
    if [[ -d "$OPENADMET_DIR" ]]; then
        conda run --no-capture-output -n "$MAIN_ENV" python -m pip install -e "$OPENADMET_DIR" \
            || return 1
    else
        warn "openadmet-models not found at $OPENADMET_DIR; finetune/evaluate need it"
        warn "  (the test suite does not, so this does not block 'okay')"
    fi
    return 0
}

# run the test suite in the main env
run_tests() {
    conda run --no-capture-output -n "$MAIN_ENV" python -m pytest tests/ -q
}

# ── preflight ────────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH; load it (e.g. module load miniforge) and retry" >&2
    exit 1
fi
# make `conda run` / activation work in a non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# ── select which env YAMLs to process ────────────────────────────────────────
declare -a YAMLS=()
if [[ "$#" -gt 0 ]]; then
    for base in "$@"; do
        if [[ -f "envs/$base.yml" ]]; then
            YAMLS+=("envs/$base.yml")
        else
            echo "ERROR: envs/$base.yml does not exist" >&2
            exit 1
        fi
    done
else
    # main first so the env that runs the tests is ready early
    YAMLS+=("envs/main.yml")
    for yaml in envs/*.yml; do
        [[ "$yaml" == "envs/main.yml" ]] && continue
        YAMLS+=("$yaml")
    done
fi

# ── build + install each environment ─────────────────────────────────────────
for yaml in "${YAMLS[@]}"; do
    name="$(awk -F': *' '/^name:/{print $2; exit}' "$yaml")"
    if [[ -z "$name" ]]; then
        warn "no name: field in $yaml; skipping"
        continue
    fi

    # create the env; skip its dependent steps if creation fails
    if ! run_step "create env $name ($yaml)" create_env "$yaml" "$name"; then
        continue
    fi

    # osmordred needs its source build before the package install
    if [[ "$name" == "sarizard-osmordred" ]]; then
        run_step "build osmordred into $name" build_osmordred
    fi

    # install the SARizard package (editable, no deps) into this env
    run_step "install sarizard into $name" install_sarizard "$name"

    # the main env also needs pytest (and openadmet-models if available)
    if [[ "$name" == "$MAIN_ENV" ]]; then
        run_step "install main extras (pytest, openadmet-models)" install_main_extras
    fi
done

# ── run the test suite in the main env ───────────────────────────────────────
if env_exists "$MAIN_ENV"; then
    run_step "test suite (env $MAIN_ENV)" run_tests
else
    SUMMARY+=("FAIL  test suite (env $MAIN_ENV missing)")
    FAILED=1
fi

# ── summary ──────────────────────────────────────────────────────────────────
echo
echo "===== SETUP SUMMARY ====="
printf '%s\n' "${SUMMARY[@]}"
if [[ "${#WARNINGS[@]}" -gt 0 ]]; then
    echo
    echo "warnings:"
    printf '  - %s\n' "${WARNINGS[@]}"
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
    echo "okay"
    exit 0
fi
echo "setup incomplete: one or more steps failed (see FAIL lines above)"
exit 1
