"""ML-QM target (24-dim), computed in the isolated qmdesc env.

qmdesc (Guan et al.) predicts QM atom-level and bond-level descriptors from a SMILES with a
bundled D-MPNN, deterministically and in RDKit atom order. ``ReactivityDescriptorHandler``
returns flattened per-atom arrays (partial_charge, fukui_neu, fukui_elec, NMR) and per-bond
arrays (bond_order, bond_length). Each of the six is pooled with (mean, std, min, max) into
a fixed 24-dim molecule vector, so the variable-length atom/bond outputs become a regression
target. Pooling discards per-atom resolution; regressing the descriptors as node/edge targets
is the richer alternative noted in TODO.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# fixed key order, so the 24-dim target columns are well-defined and stable
DESCRIPTOR_KEYS = ("partial_charge", "fukui_neu", "fukui_elec", "NMR", "bond_order", "bond_length")
N_STATS = 4  # mean, std, min, max
POOL_DIM = len(DESCRIPTOR_KEYS) * N_STATS  # 24


def _pool(values: np.ndarray) -> np.ndarray:
    """Pool a 1D descriptor array to (mean, std, min, max); NaN if empty."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.full(N_STATS, np.nan, dtype=np.float64)
    return np.array([arr.mean(), arr.std(), arr.min(), arr.max()], dtype=np.float64)


def build_compute_fn() -> Callable[[Sequence[str]], np.ndarray]:
    """Return the qmdesc calculator.

    Returns
    -------
    Callable[[Sequence[str]], numpy.ndarray]
        Maps a block of SMILES to an ``(n, 24)`` float32 array, NaN rows for failures.
    """
    from qmdesc import ReactivityDescriptorHandler

    handler = ReactivityDescriptorHandler()

    def compute(smiles: Sequence[str]) -> np.ndarray:
        out = np.full((len(smiles), POOL_DIM), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles):
            try:
                result = handler.predict(smi)
            except Exception as err:  # noqa: BLE001 - leave NaN for this molecule
                logger.warning("qmdesc failed on a molecule: %s", err)
                continue
            out[i] = np.concatenate([_pool(result[key]) for key in DESCRIPTOR_KEYS])
        return out

    return compute
