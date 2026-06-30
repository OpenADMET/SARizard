"""minimol learned-embedding target (512-dim), computed in the isolated minimol env.

minimol is a distilled GNN molecular fingerprinter (Graphcore Research) that maps a SMILES
to a fixed 512-dim continuous embedding, deterministic per molecule and MSE-regressable.
It runs on CPU or CUDA (no Graphcore IPU needed) and batches internally.

minimol does not fail gracefully on bad SMILES (graphium returns the raw string and only
counts failures), so SMILES are pre-filtered with RDKit and only valid molecules are sent.
The corpus is already RDKit-canonical, so failures here should be rare; a length-mismatch
guard falls back to per-molecule featurization to keep rows aligned.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

EMBED_DIM = 512


def build_compute_fn(batch_size: int = 100) -> Callable[[Sequence[str]], np.ndarray]:
    """Return the minimol calculator.

    Parameters
    ----------
    batch_size : int, optional
        minimol's internal featurization batch size. Default 100.

    Returns
    -------
    Callable[[Sequence[str]], numpy.ndarray]
        Maps a block of SMILES to an ``(n, 512)`` float32 array, NaN rows for failures.
    """
    import torch
    from minimol import Minimol

    model = Minimol(batch_size=batch_size)

    def _stack(fps: list) -> np.ndarray:
        return torch.stack(fps).to(torch.float32).cpu().numpy()

    def compute(smiles: Sequence[str]) -> np.ndarray:
        out = np.full((len(smiles), EMBED_DIM), np.nan, dtype=np.float32)
        valid = [(i, s) for i, s in enumerate(smiles) if Chem.MolFromSmiles(s) is not None]
        if not valid:
            return out
        idx = [i for i, _ in valid]
        valid_smiles = [s for _, s in valid]
        fps = model(valid_smiles)
        if len(fps) == len(valid_smiles):
            out[idx] = _stack(fps)
            return out
        # minimol kept failures in the list and misaligned; recover per molecule
        logger.warning(
            "minimol returned %d for %d inputs; per-molecule fallback", len(fps), len(valid_smiles)
        )
        for i, smi in zip(idx, valid_smiles, strict=True):
            try:
                out[i] = _stack(model([smi]))[0]
            except Exception as err:  # noqa: BLE001 - leave NaN for this molecule
                logger.warning("minimol failed on a molecule: %s", err)
        return out

    return compute
