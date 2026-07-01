"""Environment-universal NumPy memmap writer for per-flavor targets.

The isolated calculator environments pin old Python and conflicting stacks, so they
cannot depend on zarr 3.x (Python 3.11+). This module depends on numpy only and writes
the raw target array to a ``.npy`` memmap, filling failed rows with NaN. The downstream
masked losses ignore NaN rows, so a molecule that fails its calculator drops out of the
pretraining target without corrupting the rest.

``pack_target`` (zarr, main environment) converts the resulting ``.npy`` into the chunked
``target.zarr`` the training pipeline reads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    # a calculator maps a list of SMILES to an (n, target_dim) float array, NaN where a
    # molecule failed; the row order must match the input order so the cache stays aligned
    # with the shared corpus parquet. defined here (not at module scope) because the ml_qm
    # env pins Python 3.8, where subscripting collections.abc.Callable at runtime raises
    # TypeError; annotations are stringized by __future__ so nothing evaluates it at runtime
    ComputeFn = Callable[[Sequence[str]], np.ndarray]

logger = logging.getLogger(__name__)

DTYPE = np.float32


def open_target_memmap(path: Path, n_rows: int, target_dim: int) -> np.memmap:
    """Create a NaN-filled ``(n_rows, target_dim)`` float32 memmap at ``path``.

    Parameters
    ----------
    path : pathlib.Path
        Output ``.npy`` path; parent directories are created.
    n_rows : int
        Number of molecules (rows), matching the shared corpus length.
    target_dim : int
        Number of target columns for this flavor.

    Returns
    -------
    numpy.memmap
        A writable memmap initialised to NaN, so any row never written stays NaN.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(path, mode="w+", dtype=DTYPE, shape=(n_rows, target_dim))
    memmap[:] = np.nan
    memmap.flush()
    return memmap


def fill_streaming(
    memmap: np.memmap, smiles: Sequence[str], compute_fn: ComputeFn, block_rows: int
) -> int:
    """Fill ``memmap`` block by block from ``compute_fn``.

    Parameters
    ----------
    memmap : numpy.memmap
        Target memmap from :func:`open_target_memmap`.
    smiles : sequence of str
        Corpus SMILES in row order.
    compute_fn : ComputeFn
        Maps a block of SMILES to an ``(len(block), target_dim)`` float array.
    block_rows : int
        Number of molecules computed per call; the compute granularity, decoupled from
        the eventual zarr storage chunking.

    Returns
    -------
    int
        Count of rows that came back all-NaN (a failed molecule).
    """
    n_rows, target_dim = memmap.shape
    n_failed = 0
    for start in range(0, n_rows, block_rows):
        end = min(start + block_rows, n_rows)
        block = np.asarray(compute_fn(smiles[start:end]), dtype=DTYPE)
        if block.shape != (end - start, target_dim):
            raise ValueError(
                f"compute_fn returned {block.shape}, expected {(end - start, target_dim)}"
            )
        memmap[start:end] = block
        memmap.flush()
        n_failed += int(np.isnan(block).all(axis=1).sum())
        logger.info("rows %d-%d / %d written", start, end, n_rows)
    return n_failed


def fill_array(memmap: np.memmap, array: np.ndarray) -> int:
    """Write a fully precomputed ``(n_rows, target_dim)`` array into ``memmap``.

    Used by learned-model flavors that produce the whole target at once rather than
    streaming. Returns the count of all-NaN rows.
    """
    array = np.asarray(array, dtype=DTYPE)
    if array.shape != memmap.shape:
        raise ValueError(f"array shape {array.shape} != memmap shape {memmap.shape}")
    memmap[:] = array
    memmap.flush()
    return int(np.isnan(array).all(axis=1).sum())
