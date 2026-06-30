"""Shared chunk-based train/val split for the pretraining corpus.

The target store is chunked at a fixed row count (``CORPUS_CHUNK_ROWS``), and the
train/val split is taken at chunk granularity so an entire chunk lands wholly in one
side. Both ``split.py`` (which slices the target store into train/val zarrs) and
``prescaling.py`` (which fits its transforms on the train rows only, to avoid leakage)
derive the split from this one helper, so the train rows prescaling fits on are exactly
the train rows training reads.
"""

from __future__ import annotations

import numpy as np

SPLIT_SEED = 42
TRAIN_FRACTION = 0.9


def train_val_chunk_indices(
    n_chunks: int, *, seed: int = SPLIT_SEED, train_frac: float = TRAIN_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Partition chunk indices into shuffled train and validation sets.

    The last chunk is dropped before splitting because it may be partial (fewer than
    ``CORPUS_CHUNK_ROWS`` rows); excluding it keeps every emitted chunk full and the
    batch size uniform.

    Parameters
    ----------
    n_chunks : int
        Total number of chunks in the target store (``zarr_array.nchunks``).
    seed : int, optional
        Seed for the shuffle, so the split is reproducible across the split and
        prescaling steps. Default ``SPLIT_SEED``.
    train_frac : float, optional
        Fraction of chunks assigned to the train side. Default ``TRAIN_FRACTION``.

    Returns
    -------
    train_chunks : numpy.ndarray
        Sorted chunk indices assigned to training.
    val_chunks : numpy.ndarray
        Sorted chunk indices assigned to validation.
    """
    # drop the last (possibly partial) chunk so every emitted chunk is full
    chunk_indices = np.arange(n_chunks)[:-1]
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(chunk_indices)
    # split on the full chunk count to match the historical split.py boundary exactly
    split_idx = int(train_frac * n_chunks)
    train_chunks = np.sort(chunk_indices[:split_idx])
    val_chunks = np.sort(chunk_indices[split_idx:])
    return train_chunks, val_chunks


def chunk_row_ranges(chunks: np.ndarray, rows_per_chunk: int) -> list[tuple[int, int]]:
    """Expand chunk indices into ``(start_row, end_row)`` ranges.

    Parameters
    ----------
    chunks : numpy.ndarray
        Chunk indices (each maps to a full ``rows_per_chunk`` block).
    rows_per_chunk : int
        Number of rows per chunk.

    Returns
    -------
    list of (int, int)
        Half-open row ranges, one per chunk, in the order ``chunks`` is given.
    """
    return [(int(c) * rows_per_chunk, (int(c) + 1) * rows_per_chunk) for c in chunks]
