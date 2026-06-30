"""Direct-compute descriptor and fingerprint targets via scikit-fingerprints (skfp).

Eight flavors are built here. The 2D flavors (rdkit2d, erg, ecfp, atompair, pubchem)
transform SMILES directly; the 3D flavors (usrcat, whim, e3fp) need a generated conformer
first. Invalid SMILES and conformer failures are scattered back as NaN rows so the cached
target stays aligned with the shared corpus, and the masked pretraining loss skips them.

API verified against skfp 2.0.0. Output widths: rdkit2d 200, erg 315, ecfp 2048,
atompair 2048, pubchem 881, usrcat 60, whim 114, e3fp 1024.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
from rdkit import Chem

from sarizard.pretraining.config import CONFORMER_FORCE_FIELD, CONFORMER_NUM, CONFORMER_SEED

logger = logging.getLogger(__name__)

# 2D flavors: a builder returning a configured skfp transformer. Every fingerprint
# parameter that fixes the output is set explicitly (chemoinformatics rule: pin them).
_2D_BUILDERS: dict[str, Callable[[int], object]] = {}
_3D_BUILDERS: dict[str, Callable[[int], object]] = {}


def _register_builders() -> None:
    """Populate the builder tables, importing skfp lazily inside this call."""
    from skfp.fingerprints import (
        AtomPairFingerprint,
        E3FPFingerprint,
        ECFPFingerprint,
        ERGFingerprint,
        PubChemFingerprint,
        RDKit2DDescriptorsFingerprint,
        USRCATFingerprint,
        WHIMFingerprint,
    )

    _2D_BUILDERS.update(
        {
            # 200 RDKit 2D physicochemical descriptors, raw (rescaled later by split.py)
            "rdkit2d": lambda n: RDKit2DDescriptorsFingerprint(normalized=False, n_jobs=n),
            # 315-dim continuous extended reduced graph pharmacophore
            "erg": lambda n: ERGFingerprint(n_jobs=n),
            # 2048-bit ECFP4 (radius 2), binary bits; chirality pinned off (the standard ECFP4
            # invariant set) so the target is reproducible across skfp/rdkit versions
            "ecfp": lambda n: ECFPFingerprint(
                fp_size=2048, radius=2, include_chirality=False, count=False, n_jobs=n
            ),
            # 2048-bit topological atom-pair, binary bits; chirality pinned off explicitly
            "atompair": lambda n: AtomPairFingerprint(
                fp_size=2048, include_chirality=False, count=False, n_jobs=n
            ),
            # 881-bit PubChem substructure keys
            "pubchem": lambda n: PubChemFingerprint(count=False, n_jobs=n),
        }
    )
    _3D_BUILDERS.update(
        {
            # 60-dim USRCAT shape + pharmacophore moments; errors="NaN" yields NaN rows
            "usrcat": lambda n: USRCATFingerprint(errors="NaN", n_jobs=n),
            # 114-dim WHIM holistic geometry (no per-row error mode; handled by fallback)
            "whim": lambda n: WHIMFingerprint(n_jobs=n),
            # 1024-bit E3FP 3D circular substructure bits
            "e3fp": lambda n: E3FPFingerprint(fp_size=1024, n_jobs=n),
        }
    )


def is_skfp_flavor(name: str) -> bool:
    """Return whether ``name`` is computed by this module."""
    if not _2D_BUILDERS:
        _register_builders()
    return name in _2D_BUILDERS or name in _3D_BUILDERS


def _parse_valid(smiles: Sequence[str]) -> tuple[list[Chem.Mol], list[int]]:
    """Parse SMILES, returning valid mols and their indices into the input."""
    mols: list[Chem.Mol] = []
    idx: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            idx.append(i)
    return mols, idx


def _safe_transform(transformer: object, mols: list[Chem.Mol], target_dim: int) -> np.ndarray:
    """Transform ``mols`` to ``(len(mols), target_dim)``, isolating per-molecule failures.

    Tries the whole batch first (fast path); on any exception falls back to per-molecule
    transforms so one bad molecule does not void the block, scattering NaN for failures.
    """
    try:
        return np.asarray(transformer.transform(mols), dtype=np.float32)
    except Exception as err:  # noqa: BLE001 - fall back to isolate the offending molecule
        logger.warning("batch transform failed (%s); retrying per molecule", err)
        out = np.full((len(mols), target_dim), np.nan, dtype=np.float32)
        for i, mol in enumerate(mols):
            try:
                out[i] = np.asarray(transformer.transform([mol]), dtype=np.float32)[0]
            except Exception as inner:  # noqa: BLE001 - leave NaN for this molecule
                logger.warning("molecule transform failed: %s", inner)
        return out


def _build_2d(name: str, n_jobs: int, target_dim: int) -> Callable[[Sequence[str]], np.ndarray]:
    """Build a compute_fn for a 2D skfp flavor."""
    transformer = _2D_BUILDERS[name](n_jobs)

    def compute(smiles: Sequence[str]) -> np.ndarray:
        out = np.full((len(smiles), target_dim), np.nan, dtype=np.float32)
        mols, idx = _parse_valid(smiles)
        if mols:
            out[idx] = _safe_transform(transformer, mols, target_dim)
        return out

    return compute


def _build_3d(name: str, n_jobs: int, target_dim: int) -> Callable[[Sequence[str]], np.ndarray]:
    """Build a compute_fn for a 3D skfp flavor (generates one conformer per molecule)."""
    from skfp.preprocessing import ConformerGenerator

    transformer = _3D_BUILDERS[name](n_jobs)
    # one reproducible conformer per molecule; settings recorded in config for the
    # methodology watch-item (3D targets are only approximately reproducible)
    confgen = ConformerGenerator(
        num_conformers=CONFORMER_NUM,
        optimize_force_field=CONFORMER_FORCE_FIELD,
        random_state=CONFORMER_SEED,
        n_jobs=n_jobs,
    )

    def compute(smiles: Sequence[str]) -> np.ndarray:
        out = np.full((len(smiles), target_dim), np.nan, dtype=np.float32)
        mols, idx = _parse_valid(smiles)
        if not mols:
            return out
        # embed conformers per molecule so a single embedding failure does not void the
        # block; only molecules that get a conf_id reach the fingerprint
        conf_mols: list[Chem.Mol] = []
        conf_idx: list[int] = []
        for global_i, mol in zip(idx, mols, strict=True):
            try:
                conf_mols.append(confgen.transform([mol])[0])
                conf_idx.append(global_i)
            except Exception as err:  # noqa: BLE001 - leave NaN for this molecule
                logger.warning("conformer generation failed: %s", err)
        if conf_mols:
            out[conf_idx] = _safe_transform(transformer, conf_mols, target_dim)
        return out

    return compute


def build_compute_fn(
    name: str, n_jobs: int, target_dim: int
) -> Callable[[Sequence[str]], np.ndarray]:
    """Return the calculator for an skfp flavor.

    Parameters
    ----------
    name : str
        Flavor name (one of the eight skfp flavors).
    n_jobs : int
        Parallel jobs passed to the skfp transformer and conformer generator.
    target_dim : int
        Expected output width, used to shape failed rows.

    Returns
    -------
    Callable[[Sequence[str]], numpy.ndarray]
        Maps a block of SMILES to an ``(n, target_dim)`` float32 array with NaN rows for
        molecules that failed.
    """
    if not _2D_BUILDERS:
        _register_builders()
    if name in _2D_BUILDERS:
        return _build_2d(name, n_jobs, target_dim)
    if name in _3D_BUILDERS:
        return _build_3d(name, n_jobs, target_dim)
    raise KeyError(f"{name!r} is not an skfp flavor")
