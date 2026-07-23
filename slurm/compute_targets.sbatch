#!/usr/bin/env bash
# Compute every flavor's pretraining target, one array task per flavor. Each task computes
# the raw target in the flavor's own conda environment, then packs it to zarr in the main
# environment. CPU only. Resumable: a flavor whose target.zarr already exists is skipped.
#
# EDIT the resource directives below for your cluster. Prefer submitting through run_all.sh,
# which sizes the array from the registry. To submit standalone, size the array to the flavor
# count (the --array header below is a conservative default the command line overrides):
#   sbatch --array=0-$(($(source slurm/env.sh; flavor_list | wc -l) - 1)) slurm/compute_targets.sbatch
#SBATCH --job-name=sar-targets
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-12
#SBATCH --output=slurm/logs/targets_%A_%a.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh"

# surrogate_adme reads the Novartis released CSV directly; defaults to the cached download
# under cache/surrogate/ (gitignored). Override SURROGATE_CSV to point elsewhere
SURROGATE_CSV="${SURROGATE_CSV:-cache/surrogate/protacdb2.0_zinc_chembl_dataset.csv}"

# map this array task to a flavor and look up the environment that computes it
mapfile -t FLAVORS < <(flavor_list)
if [[ "$SLURM_ARRAY_TASK_ID" -ge ${#FLAVORS[@]} ]]; then
    echo "task $SLURM_ARRAY_TASK_ID >= ${#FLAVORS[@]} flavors; nothing to do"
    exit 0
fi
FLAVOR="${FLAVORS[$SLURM_ARRAY_TASK_ID]}"
FLAVOR_ENV=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').env)")
DERIVED_FROM=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').derived_from or '')")
CORPUS_FROM=$(conda run -n "$MAIN_ENV" python -c \
    "from sarizard.pretraining.flavors import get_flavor; print(get_flavor('$FLAVOR').corpus_from or '')")
echo "task=$SLURM_ARRAY_TASK_ID flavor=$FLAVOR env=$FLAVOR_ENV derived_from=${DERIVED_FROM:-none}"

if [[ -n "$DERIVED_FROM" ]]; then
    # a derived flavor (e.g. osmordred_pca80) has no calculator of its own; its target.zarr
    # is written later by slurm/osmordred_pca_targets.sbatch, from the base flavor's raw
    # target, once this array (including the base flavor) has completed
    echo "$FLAVOR is derived from $DERIVED_FROM; nothing to compute here, skipping"
    exit 0
fi

if [[ -d "cache/targets/$FLAVOR/target.zarr" ]]; then
    echo "cache/targets/$FLAVOR/target.zarr exists; skipping"
    exit 0
fi

CORPUS="$REPO_DIR/$CORPUS_FILE"
# a flavor with corpus_from borrows another flavor's companion corpus_smiles.parquet (written
# by that flavor's compute_target) rather than the shared corpus, so its target is computed on
# those molecules; that companion parquet must already exist (its source flavor ran first)
if [[ -n "$CORPUS_FROM" ]]; then
    CORPUS="$REPO_DIR/cache/targets/$CORPUS_FROM/corpus_smiles.parquet"
    if [[ ! -f "$CORPUS" ]]; then
        echo "missing $CORPUS; compute the $CORPUS_FROM target first" >&2
        exit 1
    fi
fi
if [[ "$FLAVOR" == "surrogate_adme" ]]; then
    # surrogate_adme runs in the main env; compute_target writes target.npy + corpus_smiles.parquet
    conda activate "$MAIN_ENV"
    python -m sarizard.pretraining.features.compute_target --flavor surrogate_adme \
        --csv-path "$SURROGATE_CSV"
    python -m sarizard.pretraining.features.pack_target --flavor surrogate_adme
else
    # compute the raw target in the flavor's isolated environment (writes target.npy)
    conda activate "$FLAVOR_ENV"
    python -m sarizard.pretraining.features.compute_target --flavor "$FLAVOR" --corpus "$CORPUS"

    # pack the .npy into the chunked target.zarr the trainer reads, in the main environment
    conda activate "$MAIN_ENV"
    python -m sarizard.pretraining.features.pack_target --flavor "$FLAVOR"
fi
