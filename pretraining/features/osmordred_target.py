"""osmordred 2D descriptor target (3585-dim), computed in the isolated osmordred env.

Reuses the vendored ``_osmordred.calculate`` (a performant Mordred reimplementation) over
a process pool. ``calculate`` returns an all-NaN row for an unparseable or failing
molecule, so the cached target stays aligned with the corpus and the masked loss skips it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


def build_compute_fn(n_jobs: int = -1) -> Callable[[Sequence[str]], np.ndarray]:
    """Return the osmordred calculator backed by a persistent process pool.

    Parameters
    ----------
    n_jobs : int, optional
        Worker processes; ``-1`` (default) uses all CPUs.

    Returns
    -------
    Callable[[Sequence[str]], numpy.ndarray]
        Maps a block of SMILES to an ``(n, 3585)`` float32 array, NaN rows for failures.
    """
    # imported here so the heavy osmordred extension loads only in its own environment
    from pretraining.features._osmordred import DESCRIPTOR_COUNT, calculate

    max_workers = os.cpu_count() if n_jobs in (-1, 0, None) else n_jobs
    executor = ProcessPoolExecutor(max_workers=max_workers)

    def compute(smiles: Sequence[str]) -> np.ndarray:
        rows = list(executor.map(calculate, list(smiles), chunksize=16))
        out = np.asarray(rows, dtype=np.float32)
        if out.shape != (len(smiles), DESCRIPTOR_COUNT):
            raise ValueError(
                f"osmordred returned {out.shape}, expected {(len(smiles), DESCRIPTOR_COUNT)}"
            )
        return out

    return compute
