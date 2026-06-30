"""Surrogate-ADME target (25-dim), read directly from the Novartis released CSV.

The Novartis surrogate-ADME work (Peteani et al., Nat Commun 2024,
DOI 10.1038/s41467-024-49979-3, CC BY 4.0) releases a 273,706-row CSV (Supplementary Data 1)
with 25 precomputed ADME predictions per molecule. Unlike every other flavor, which computes
its target on the shared 250K PubChem corpus, surrogate_adme uses the Novartis molecules and
their precomputed labels directly: the released dataset IS the pretraining corpus for this
flavor.

This module writes two files to ``cache/targets/surrogate_adme/``:

- ``target.npy``: ``(n, 25)`` float32 array of ADME predictions, NaN where a value is missing.
- ``corpus_smiles.parquet``: one ``SMILES`` column of the n canonical SMILES that passed
  RDKit parsing. This parquet replaces the shared 250K corpus when ``split.py`` runs for
  this flavor.

Released dataset (download once, not redistributed here):
  https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-49979-3/MediaObjects/41467_2024_49979_MOESM4_ESM.zip
The CSV inside is ``protacdb2.0_zinc_chembl_dataset.csv`` with a ``smiles`` column and 25
``pred(...)`` target columns.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger

from sarizard.analysis.paths import TARGETS_DIR

logger = logging.getLogger(__name__)

N_TASKS = 25
SMILES_COLUMN = "smiles"
NON_TARGET_COLUMNS = frozenset({"", "index", "Unnamed: 0", SMILES_COLUMN, "Public Source"})

FLAVOR_DIR = TARGETS_DIR / "surrogate_adme"
DEFAULT_OUT_NPY = FLAVOR_DIR / "target.npy"
DEFAULT_OUT_SMILES = FLAVOR_DIR / "corpus_smiles.parquet"


def _target_columns(csv_path: Path) -> list[str]:
    """Return the 25 prediction columns from the released CSV header, in file order."""
    with csv_path.open(newline="") as handle:
        header = next(csv.reader(handle))
    cols = [c for c in header if c not in NON_TARGET_COLUMNS]
    if len(cols) != N_TASKS:
        logger.warning("expected %d target columns, found %d: %s", N_TASKS, len(cols), cols)
    return cols


def build_from_csv(
    csv_path: Path,
    out_npy: Path = DEFAULT_OUT_NPY,
    out_smiles: Path = DEFAULT_OUT_SMILES,
    *,
    force: bool = False,
) -> int:
    """Read the Novartis CSV, canonicalize SMILES, and write target.npy + corpus_smiles.parquet.

    Parameters
    ----------
    csv_path : path-like
        The released ``protacdb2.0_zinc_chembl_dataset.csv``.
    out_npy : path-like, optional
        Output path for the ``(n, 25)`` target array. Defaults to the standard cache location.
    out_smiles : path-like, optional
        Output parquet path for the canonical SMILES column. Defaults to the standard cache
        location alongside ``target.npy``.
    force : bool, optional
        Overwrite existing output files. Default ``False``.

    Returns
    -------
    int
        Number of rows kept after SMILES canonicalization.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    FileExistsError
        If ``out_npy`` exists and ``force`` is ``False``.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; download the Novartis surrogate dataset first"
            " (see this module's docstring for the URL)"
        )
    if out_npy.exists() and not force:
        raise FileExistsError(f"{out_npy} exists; pass force=True to overwrite")

    RDLogger.DisableLog("rdApp.*")
    target_cols = _target_columns(csv_path)
    n_cols = len(target_cols)

    # read CSV row by row; 273K rows x 25 cols is ~26 MB float32, comfortable in memory
    kept_smiles: list[str] = []
    kept_targets: list[list[float]] = []
    n_total = 0
    n_parse_fail = 0

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n_total += 1
            mol = Chem.MolFromSmiles(row.get(SMILES_COLUMN, "") or "")
            if mol is None:
                n_parse_fail += 1
                continue
            kept_smiles.append(Chem.MolToSmiles(mol))
            vals: list[float] = []
            for col in target_cols:
                raw = row.get(col, "")
                try:
                    vals.append(float(raw))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
            kept_targets.append(vals)

    n_kept = len(kept_smiles)
    logger.info(
        "read %d CSV rows: %d kept, %d failed SMILES parse",
        n_total,
        n_kept,
        n_parse_fail,
    )

    arr = np.array(kept_targets, dtype=np.float32)  # (n_kept, n_cols)
    if arr.shape[1] != n_cols:
        raise ValueError(f"target array has {arr.shape[1]} columns, expected {n_cols}")

    # write target.npy (standard npy format; pack_target reads it with np.load)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), arr)
    logger.info("wrote %s shape=%s", out_npy, arr.shape)

    # write companion corpus parquet so split.py uses these molecules as the pretraining corpus
    out_smiles.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"SMILES": kept_smiles}).write_parquet(out_smiles)
    logger.info("wrote %s (%d rows)", out_smiles, n_kept)

    return n_kept
