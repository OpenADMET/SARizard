"""Prepare the shared pretraining corpus: a fixed 250K subset of the CheMeleon SMILES.

Downloads ``cleaned_pubchem_1MM.smiles`` from the CheMeleon Training Data record on Zenodo
(DOI 10.5281/zenodo.15733575, CC-BY-4.0), canonicalizes with RDKit, and downsamples to a
fixed subset with a single seed. The resulting ``corpus/corpus_250k.parquet`` (one ``SMILES``
column) is the one molecule set every flavor computes its target on, so the report-card
columns stay comparable.

The source is already cleaned by the CheMeleon authors, so this step canonicalizes and
drops unparseable rows rather than re-running a full standardization that would diverge from
their corpus.

Usage:
    python -m corpus.prepare_corpus              # download and build the 250K parquet
    python -m corpus.prepare_corpus --n 100000   # a smaller smoke-test corpus
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger

from analysis.paths import CORPUS_DIR, CORPUS_SMILES

logger = logging.getLogger(__name__)

# fixed Zenodo source (CheMeleon Training Data, version record 15733575)
SOURCE_URL = "https://zenodo.org/api/records/15733575/files/cleaned_pubchem_1MM.smiles/content"
SOURCE_NAME = "cleaned_pubchem_1MM.smiles"

DEFAULT_N = 250_000
DEFAULT_SEED = 42


def download_source(dest: Path) -> Path:
    """Download the 1M SMILES source to ``dest`` if not already present.

    Parameters
    ----------
    dest : pathlib.Path
        Local path for the downloaded ``.smiles`` file.

    Returns
    -------
    pathlib.Path
        The local path, downloaded or cached.
    """
    if dest.exists():
        logger.info("using cached source at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s -> %s", SOURCE_URL, dest)
    # fixed first-party https URL (Zenodo); not user-controlled
    urlretrieve(SOURCE_URL, dest)  # noqa: S310
    return dest


def _read_smiles(path: Path) -> list[str]:
    """Read raw SMILES from a one-per-line file, taking the first whitespace token."""
    smiles: list[str] = []
    with path.open() as handle:
        for line in handle:
            token = line.strip().split()
            if token:
                smiles.append(token[0])
    return smiles


def canonical_subset(raw: list[str], n: int, seed: int) -> tuple[list[str], int]:
    """Shuffle, canonicalize, and collect ``n`` valid canonical SMILES.

    Parameters
    ----------
    raw : list of str
        Raw SMILES strings from the source file.
    n : int
        Number of valid canonical SMILES to collect.
    seed : int
        Seed for the shuffle, so the subset is reproducible.

    Returns
    -------
    smiles : list of str
        ``n`` canonical SMILES (or fewer if the source is exhausted).
    n_failed : int
        Count of source rows that failed to parse.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(raw))
    kept: list[str] = []
    n_failed = 0
    for idx in order:
        mol = Chem.MolFromSmiles(raw[idx])
        if mol is None:
            n_failed += 1
            continue
        kept.append(Chem.MolToSmiles(mol))
        if len(kept) >= n:
            break
    return kept, n_failed


def main() -> None:
    """Download, canonicalize, downsample, and write the shared corpus parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="corpus size (default 250000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="downsample seed")
    parser.add_argument(
        "--source",
        type=Path,
        default=CORPUS_DIR / SOURCE_NAME,
        help="local source .smiles path; downloaded from Zenodo if absent",
    )
    parser.add_argument(
        "--out", type=Path, default=CORPUS_SMILES, help="output parquet path"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # parse failures are counted via None checks below, so silencing per-molecule logs is safe
    RDLogger.DisableLog("rdApp.*")

    source = download_source(args.source)
    raw = _read_smiles(source)
    logger.info("read %d raw SMILES from %s", len(raw), source.name)

    smiles, n_failed = canonical_subset(raw, args.n, args.seed)
    if len(smiles) < args.n:
        logger.warning("collected only %d valid SMILES (requested %d)", len(smiles), args.n)
    logger.info("kept %d canonical SMILES; %d failed to parse", len(smiles), n_failed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"SMILES": smiles}).write_parquet(args.out)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
