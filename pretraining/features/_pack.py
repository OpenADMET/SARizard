"""Pack a cached ``target.npy`` into the chunked ``target.zarr`` the trainer reads.

Runs in the main ``sarizard`` environment, where zarr 3.x is available. The storage
chunking is fixed (``CORPUS_CHUNK_ROWS``) so the chunk-based train/val split is identical
across flavors and the pretraining batch size is uniform; see ``pretraining/config.py``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import zarr

logger = logging.getLogger(__name__)

DTYPE = np.float32


def _safe_remove_zarr(path: Path) -> None:
    """Remove an existing ``.zarr`` store, refusing anything that is not one."""
    if path.suffix != ".zarr":
        raise ValueError(f"refusing to remove non-zarr path {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def pack_npy_to_zarr(
    npy_path: Path, zarr_path: Path, chunk_rows: int, *, force: bool = False
) -> None:
    """Convert a ``.npy`` target memmap into a chunked, uncompressed zarr array.

    Parameters
    ----------
    npy_path : pathlib.Path
        Source ``target.npy`` written by a calculator.
    zarr_path : pathlib.Path
        Destination ``target.zarr`` store.
    chunk_rows : int
        Rows per zarr chunk; fixed across flavors for split and batch uniformity.
    force : bool, optional
        Overwrite an existing destination store. Default ``False``.

    Raises
    ------
    FileNotFoundError
        If ``npy_path`` does not exist.
    FileExistsError
        If ``zarr_path`` exists and ``force`` is ``False``.
    """
    if not npy_path.exists():
        raise FileNotFoundError(f"{npy_path} not found; compute the target first")
    if zarr_path.exists():
        if not force:
            raise FileExistsError(f"{zarr_path} exists; pass force=True to overwrite")
        _safe_remove_zarr(zarr_path)

    array = np.load(npy_path, mmap_mode="r")
    n_rows, target_dim = array.shape
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.create_array(
        store=str(zarr_path),
        shape=(n_rows, target_dim),
        chunks=(chunk_rows, target_dim),
        dtype=DTYPE,
        compressors=None,  # uncompressed for fast random chunk reads during training
        fill_value=np.nan,
    )

    # copy in large blocks so a wide target never loads whole into memory
    block = max(chunk_rows, 8192)
    for start in range(0, n_rows, block):
        end = min(start + block, n_rows)
        store[start:end, :] = np.asarray(array[start:end], dtype=DTYPE)
    logger.info("packed %s -> %s (%d x %d)", npy_path.name, zarr_path, n_rows, target_dim)
