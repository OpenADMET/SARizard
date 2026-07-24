"""Materialize the fixed PXR external-test split files for the reduced-protocol PXR rerun.

The standard PXR endpoint splits ``data/pxr_pec50.parquet`` with an inline Butina
``ClusterSplitter``, so the split membership moves with the finetune seed. This module instead
builds a fixed evaluation: it holds out the two OpenADMET PXR-challenge test phases (downloaded
from the ``openadmet/pxr-challenge-train-test`` Hugging Face dataset) as external test sets and
splits ``pxr_pec50.parquet`` once into a shared 90/10 train/val, so every flavor and seed trains
on the same molecules and is scored on the same held-out compounds.

The challenge files already carry the target as a ``pEC50`` column in the same ``-log10(molarity)``
convention as the training ``PXR_pEC50``, so no log transform is applied; the column is renamed and
used as-is. Test SMILES are RDKit-canonicalized into ``OPENADMET_CANONICAL_SMILES`` to match the
recipes' ``input_col``. Any test molecule that also appears in the training set (by InChIKey) is
dropped from train to prevent leakage.

Writes four CSVs under ``data/splits/`` that the ``configs/_pxr_ext`` recipes read via
``train_resource``/``val_resource``/``test_resource``. Run in the main environment (needs RDKit and
a Hugging Face token for the gated dataset):

    python -m sarizard.analysis.build_pxr_ext_splits
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from sarizard.analysis.paths import REPO_ROOT

logger = logging.getLogger(__name__)

HF_DATASET = "hf://datasets/openadmet/pxr-challenge-train-test"
PHASE_FILES = {
    1: "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    2: "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
}
TRAIN_PARQUET = REPO_ROOT / "data" / "pxr_pec50.parquet"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# columns the recipes read (input_col + target_cols); InChIKey rides along for the leakage check
SMILES_COL = "OPENADMET_CANONICAL_SMILES"
INCHIKEY_COL = "OPENADMET_INCHIKEY"
TARGET_COL = "PXR_pEC50"
VAL_FRACTION = 0.10
SPLIT_SEED = 42


def _canonicalize(smiles: str) -> tuple[str | None, str | None, int]:
    """Return ``(canonical_smiles, inchikey, n_fragments)`` for one SMILES, or NaNs on failure."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None, None, 0
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol), len(Chem.GetMolFrags(mol))


def load_test_phase(phase: int) -> pd.DataFrame:
    """Load one challenge test phase, canonicalize its SMILES, and harmonize columns.

    Parameters
    ----------
    phase : int
        Challenge phase (1 or 2), keying into the Hugging Face dataset files.

    Returns
    -------
    pandas.DataFrame
        Columns ``Molecule Name``, ``OPENADMET_CANONICAL_SMILES``, ``OPENADMET_INCHIKEY``,
        ``PXR_pEC50``, with unparseable rows dropped.
    """
    raw = pd.read_csv(f"{HF_DATASET}/{PHASE_FILES[phase]}")
    canon = raw["SMILES"].map(_canonicalize)
    out = pd.DataFrame(
        {
            "Molecule Name": raw["Molecule Name"].to_numpy(),
            SMILES_COL: [c[0] for c in canon],
            INCHIKEY_COL: [c[1] for c in canon],
            TARGET_COL: raw["pEC50"].to_numpy(),
        }
    )
    n_bad = int(out[SMILES_COL].isna().sum())
    n_salt = int(sum(c[2] > 1 for c in canon))
    if n_bad or n_salt:
        logger.warning("phase %d: %d unparseable, %d multi-fragment SMILES", phase, n_bad, n_salt)
    out = out.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)
    logger.info("phase %d: %d test molecules (pEC50 %.2f-%.2f)", phase, len(out),
                out[TARGET_COL].min(), out[TARGET_COL].max())
    return out


def build(out_dir: Path = SPLITS_DIR) -> None:
    """Write the four PXR external-test split CSVs into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    # external held-out test sets, one per challenge phase
    tests = {phase: load_test_phase(phase) for phase in PHASE_FILES}
    test_keys = set().union(*(set(df[INCHIKEY_COL]) for df in tests.values()))

    # training pool: pxr_pec50, minus any molecule that leaks into a test phase
    train_pool = pd.read_parquet(TRAIN_PARQUET)[[SMILES_COL, INCHIKEY_COL, TARGET_COL]]
    n_raw = len(train_pool)
    train_pool = train_pool[~train_pool[INCHIKEY_COL].isin(test_keys)].reset_index(drop=True)
    if n_raw != len(train_pool):
        logger.info("dropped %d train molecules overlapping a test phase", n_raw - len(train_pool))

    # one fixed 90/10 train/val split, shared across every flavor and seed
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(train_pool))
    n_val = round(VAL_FRACTION * len(train_pool))
    val_rows = train_pool.iloc[np.sort(order[:n_val])]
    train_rows = train_pool.iloc[np.sort(order[n_val:])]

    train_rows.to_csv(out_dir / "pxr_ext_train.csv", index=False)
    val_rows.to_csv(out_dir / "pxr_ext_val.csv", index=False)
    for phase, df in tests.items():
        df.to_csv(out_dir / f"pxr_test_phase{phase}.csv", index=False)

    logger.info("wrote train=%d val=%d, test phase1=%d phase2=%d -> %s",
                len(train_rows), len(val_rows), len(tests[1]), len(tests[2]), out_dir)


def main() -> None:
    """Build the PXR external-test split files from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR, help="split output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(args.out_dir)


if __name__ == "__main__":
    main()
