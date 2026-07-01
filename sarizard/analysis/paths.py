"""Single source of truth for on-disk locations.

Repo-root scripts and the analysis package import these helpers instead of hardcoding
paths. The vendored ``sarizard/pretraining/`` scripts run from their own directory and use
relative paths instead; this module is for code launched from the repo root.
"""

from __future__ import annotations

from pathlib import Path

# this file is sarizard/analysis/paths.py, so the repo root is two parents up
REPO_ROOT = Path(__file__).resolve().parents[2]

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
PRETRAIN_RUNS_DIR = REPO_ROOT / "sarizard" / "pretraining" / "runs"  # <flavor>/<timestamp>/

# finetuning
CONFIGS_DIR = REPO_ROOT / "configs"  # configs/<flavor>/<endpoint>.yaml
RESULTS_DIR = REPO_ROOT / "results"  # results/<flavor>/<endpoint>/
METRICS_CSV = RESULTS_DIR / "metrics.csv"  # tidy per-(flavor, endpoint) metrics from evaluate.py

# analysis outputs (repo-root artifact dir, gitignored)
PLOTS_DIR = REPO_ROOT / "plots"


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
    """Return the converted foundation checkpoint path for a flavor (unseeded label)."""
    return FOUNDATIONS_DIR / f"{flavor}_mp.pt"


def results_dir(flavor: str, endpoint: str) -> Path:
    """Return the finetuning result directory for a flavor and endpoint."""
    return RESULTS_DIR / flavor / endpoint


# ── seed variants ────────────────────────────────────────────────────────────
# Every experiment (flavor sweep, prescaling triage, LR experiments) can run a base unit at
# several training seeds to separate its effect from seed-driven variance. The seed is tagged
# onto the base label as ``<base>__s<seed>``, so each variant gets its own foundation, recipes,
# and result dir, and the reports collapse the variants back to one column per base unit.
SEED_TAG = "__s"


def seed_variant_label(base: str, seed: int) -> str:
    """Tag a base experiment label with a seed (``<base>__s<seed>``)."""
    return f"{base}{SEED_TAG}{seed}"


def parse_seed_variant(label: str) -> tuple[str, int | None]:
    """Split ``<base>__s<seed>`` into ``(base, seed)``; seed is ``None`` when absent.

    Plain labels (no ``__s<seed>`` suffix) round-trip to ``(label, None)``, so a report can
    collapse seeded and unseeded labels uniformly.
    """
    base, sep, seed = label.rpartition(SEED_TAG)
    if sep and seed.isdigit():
        return base, int(seed)
    return label, None


def flavor_variant_label(flavor: str, seed: int) -> str:
    """Return the label for one ``(flavor, seed)`` variant of the flavor sweep."""
    return seed_variant_label(flavor, seed)


def foundation_variant_path(flavor: str, seed: int) -> Path:
    """Return the foundation checkpoint path for one ``(flavor, seed)`` sweep variant."""
    return FOUNDATIONS_DIR / f"{flavor_variant_label(flavor, seed)}_mp.pt"


# prescaling ablation triage (run before the flavor sweep to fix the production recipe)
ABLATIONS_CACHE_DIR = CACHE_DIR / "ablations"  # cache/ablations/<ablation>/


def ablation_label(ablation: str) -> str:
    """Return the result/config label for an ablation (namespaced from flavors)."""
    return f"ablation_{ablation}"


def ablation_variant_label(ablation: str, seed: int) -> str:
    """Return the label for one ``(ablation, seed)`` variant of the prescaling triage.

    The triage drives a single flavor (osmordred) through each prescaling ablation at one or
    more training seeds, so the prescaling effect can be read against seed-driven variance.
    Each variant gets its own foundation, recipes, and result dir; the report aggregates the
    seeds back to one column per ablation.
    """
    return seed_variant_label(ablation_label(ablation), seed)


def parse_ablation_variant(label: str) -> tuple[str, int | None]:
    """Split a label into ``(ablation_name, seed)``; seed is ``None`` when absent.

    Accepts both seeded variant labels (``ablation_<name>__s<seed>``) and plain ablation
    labels (``ablation_<name>``), so the report can collapse either form to its ablation.
    """
    base, seed = parse_seed_variant(label)
    name = base[len("ablation_"):] if base.startswith("ablation_") else base
    return name, seed


def ablation_prescaled_zarr(ablation: str) -> Path:
    """Return the prescaled target store for an ablation."""
    return ABLATIONS_CACHE_DIR / ablation / "prescaled.zarr"


def ablation_split_dir(ablation: str) -> Path:
    """Return the train/val split directory for an ablation."""
    return ABLATIONS_CACHE_DIR / ablation / "split"


def ablation_foundation_name(ablation: str) -> str:
    """Return the exported foundation filename for an ablation."""
    return f"{ablation_label(ablation)}_mp.pt"


def ablation_variant_foundation_name(ablation: str, seed: int) -> str:
    """Return the exported foundation filename for one ``(ablation, seed)`` variant."""
    return f"{ablation_variant_label(ablation, seed)}_mp.pt"
