"""Shared molecular standardization for the corpus and surrogate builders.

Both the shared-corpus builder and the surrogate-ADME builder canonicalize SMILES before
anything downstream featurizes them. They route that through one helper here so the salt and
solvent stripping is identical: keep the largest (organic-preferring) fragment, then emit
canonical SMILES. Stripping is applied uniformly across every flavor's molecule set, so the
report-card columns stay comparable; molecules that are already a single fragment pass through
with only canonicalization, so a clean corpus barely changes.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# reused across the serial corpus loop; constructed once to avoid per-call setup
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()


def standardize_to_canonical(smiles: str) -> tuple[str | None, bool]:
    """Parse, strip salts and solvents, and return canonical SMILES.

    Parameters
    ----------
    smiles : str
        A raw SMILES string from a corpus or released dataset.

    Returns
    -------
    canonical : str or None
        Canonical SMILES of the largest fragment, or ``None`` if the input does not parse or
        leaves no recoverable fragment (callers count and drop these rather than featurize a
        defect).
    stripped : bool
        Whether a salt or solvent fragment was removed (the input had more than one fragment).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False
    # only invoke the chooser for multi-fragment inputs; single-fragment molecules are kept
    # exactly (just canonicalized), so an already-clean corpus is unchanged
    multi_fragment = len(Chem.GetMolFrags(mol)) > 1
    parent = _LARGEST_FRAGMENT.choose(mol) if multi_fragment else mol
    if parent is None or parent.GetNumAtoms() == 0:
        return None, False
    return Chem.MolToSmiles(parent), multi_fragment
