"""Single source of truth for on-disk locations.

Repo-root scripts and the analysis package import these helpers instead of hardcoding
paths. The vendored ``pretraining/`` scripts run from their own directory and use relative
paths instead; this module is for code launched from the repo root.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# inputs
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
CORPUS_DIR = REPO_ROOT / "corpus"
CORPUS_SMILES = CORPUS_DIR / "corpus_250k.parquet"

# regenerable caches and artifacts (gitignored)
CACHE_DIR = REPO_ROOT / "cache"
TARGETS_DIR = CACHE_DIR / "targets"  # cache/targets/<flavor>/target.zarr
SPLITS_CACHE_DIR = CACHE_DIR / "splits"  # cache/splits/<flavor>/{train,val}_rescaled.zarr
FOUNDATIONS_DIR = REPO_ROOT / "foundations"  # foundations/<flavor>_mp.pt
PRETRAIN_RUNS_DIR = REPO_ROOT / "pretraining" / "runs"  # pretraining/runs/<flavor>/<timestamp>/

# finetuning
CONFIGS_DIR = REPO_ROOT / "configs"  # configs/<flavor>/<endpoint>.yaml
RESULTS_DIR = REPO_ROOT / "results"  # results/<flavor>/<endpoint>/
METRICS_CSV = RESULTS_DIR / "metrics.csv"  # tidy per-(flavor, endpoint) metrics from evaluate.py

# analysis outputs
PLOTS_DIR = REPO_ROOT / "analysis" / "plots"


SURROGATE_CORPUS_SMILES = TARGETS_DIR / "surrogate_adme" / "corpus_smiles.parquet"


def flavor_corpus(flavor: str) -> Path:
    """Return the corpus parquet path for a flavor.

    Most flavors share the 250K PubChem corpus. ``surrogate_adme`` uses its own
    corpus derived from the Novartis released dataset, written alongside ``target.npy``
    by ``compute_target --flavor surrogate_adme``.
    """
    if flavor == "surrogate_adme":
        return SURROGATE_CORPUS_SMILES
    return CORPUS_SMILES


def target_npy(flavor: str) -> Path:
    """Return the raw per-flavor target memmap path (written by calculators)."""
    return TARGETS_DIR / flavor / "target.npy"


def target_zarr(flavor: str) -> Path:
    """Return the raw per-flavor target store path (packed from the memmap)."""
    return TARGETS_DIR / flavor / "target.zarr"


def split_dir(flavor: str) -> Path:
    """Return the per-flavor train/val split directory consumed by pretraining."""
    return SPLITS_CACHE_DIR / flavor


def foundation_path(flavor: str) -> Path:
    """Return the converted foundation checkpoint path for a flavor."""
    return FOUNDATIONS_DIR / f"{flavor}_mp.pt"


def results_dir(flavor: str, endpoint: str) -> Path:
    """Return the finetuning result directory for a flavor and endpoint."""
    return RESULTS_DIR / flavor / endpoint
