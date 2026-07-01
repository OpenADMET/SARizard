"""Jazzy hydration-energy target (6-dim), computed in the isolated jazzy env.

Jazzy (AstraZeneca) is an empirical model of the free energy of hydration and hydrogen-bond
donor/acceptor strengths. ``molecular_vector_from_smiles`` returns six continuous values
(sdc, sdx, sa, dga, dgp, dgtot) from an embedded, minimised conformer, deterministic for a
fixed embedding seed. Jazzy pins ``rdkit==2024.3.1`` exactly, so it gets its own environment.

A molecule jazzy cannot process raises ``JazzyError``, and jazzy also leaks bare exceptions
(e.g. ``IndexError`` on exotic atoms during UFF typing); either way the row becomes all-NaN
and the masked pretraining loss skips it. Conformer minimisation reuses the shared force field and
seed from ``config`` so the target is reproducible alongside the other 3D-dependent flavors.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from jazzy.api import molecular_vector_from_smiles

from sarizard.pretraining.config import CONFORMER_FORCE_FIELD, CONFORMER_SEED

logger = logging.getLogger(__name__)

# fixed key order so the six target columns are stable across runs
JAZZY_KEYS = ("sdc", "sdx", "sa", "dga", "dgp", "dgtot")
TARGET_DIM = len(JAZZY_KEYS)


def _calculate(smiles: str) -> np.ndarray:
    """Return Jazzy's six-value vector for one SMILES, NaN on failure."""
    try:
        vector = molecular_vector_from_smiles(
            smiles,
            minimisation_method=CONFORMER_FORCE_FIELD,
            embedding_seed=CONFORMER_SEED,
        )
    # jazzy raises JazzyError for molecules it rejects, but also leaks bare exceptions
    # (e.g. IndexError on exotic atoms like Pb/La during UFF typing); any failure scatters
    # a NaN row rather than crashing the whole block, per this module's contract
    except Exception as err:  # noqa: BLE001 - isolate per-molecule calculator failures
        logger.debug("jazzy failed for %s: %s", smiles, err)
        return np.full(TARGET_DIM, np.nan, dtype=np.float32)
    return np.array([vector[key] for key in JAZZY_KEYS], dtype=np.float32)


def build_compute_fn(n_jobs: int = -1) -> Callable[[Sequence[str]], np.ndarray]:
    """Return the Jazzy calculator backed by a persistent process pool.

    Parameters
    ----------
    n_jobs : int, optional
        Worker processes; ``-1`` (default) uses all CPUs.

    Returns
    -------
    Callable[[Sequence[str]], numpy.ndarray]
        Maps a block of SMILES to an ``(n, 6)`` float32 array, NaN rows for failures.
    """
    max_workers = os.cpu_count() if n_jobs in (-1, 0, None) else n_jobs
    executor = ProcessPoolExecutor(max_workers=max_workers)

    def compute(smiles: Sequence[str]) -> np.ndarray:
        rows = list(executor.map(_calculate, list(smiles), chunksize=16))
        return np.asarray(rows, dtype=np.float32)

    return compute
