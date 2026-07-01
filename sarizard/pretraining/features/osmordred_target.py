"""osmordred 2D descriptor target (3585-dim), computed in the isolated osmordred env.

Reuses the vendored ``_osmordred.calculate`` (a performant Mordred reimplementation) over
a process pool. ``calculate`` usually returns an all-NaN row for an unparseable or failing
molecule, but for some inputs (odd elements, isotopes) it returns a short, misaligned row;
``compute`` coerces every row to the fixed width and scatters a full NaN row for any that is
not exactly ``DESCRIPTOR_COUNT``, so the cached target stays aligned with the corpus and the
masked loss skips the failures.
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
    from sarizard.pretraining.features._osmordred import DESCRIPTOR_COUNT, calculate

    max_workers = os.cpu_count() if n_jobs in (-1, 0, None) else n_jobs
    executor = ProcessPoolExecutor(max_workers=max_workers)

    def compute(smiles: Sequence[str]) -> np.ndarray:
        # calculate() is documented to return an all-NaN row on failure, but for some
        # inputs (odd elements, isotopes like [1H]Br / [252Cf]) it silently drops descriptor
        # blocks and returns a short, misaligned row. stacking those ragged rows with a bare
        # np.asarray crashes, and a short row cannot be aligned to the 3585-column schema, so
        # coerce per row and scatter a full NaN row for any that is not exactly DESCRIPTOR_COUNT
        rows = list(executor.map(calculate, list(smiles), chunksize=16))
        out = np.full((len(smiles), DESCRIPTOR_COUNT), np.nan, dtype=np.float32)
        n_malformed = 0
        for i, (smi, row) in enumerate(zip(smiles, rows)):
            try:
                arr = np.asarray(row, dtype=np.float32).ravel()
            except (ValueError, TypeError):
                arr = np.empty(0, dtype=np.float32)
            if arr.size == DESCRIPTOR_COUNT:
                out[i] = arr
            else:
                n_malformed += 1
                logger.warning(
                    "osmordred returned width %d (expected %d) for %r; scattering NaN row",
                    arr.size,
                    DESCRIPTOR_COUNT,
                    smi,
                )
        if n_malformed:
            logger.info("osmordred: %d/%d rows malformed and NaN-filled", n_malformed, len(smiles))
        return out

    return compute
