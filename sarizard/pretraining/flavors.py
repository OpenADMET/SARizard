"""Registry of foundation flavors: the single source of per-flavor metadata.

A flavor is one self-supervised pretraining target for the shared D-MPNN backbone.
The only intended differences between flavors are the target block and, for binary
fingerprint targets, the loss (BCE rather than MSE) and the skipped target rescaling.
Everything else (the shared corpus, the DEFAULT graph featurizer, mean aggregation, and
the pretraining regime) is held fixed so the downstream report-card columns stay
comparable.

``target_dim`` is recorded for reference only; the pretraining head sizes itself from
the cached target array at train time, so a wrong or unknown value here never silently
mis-shapes a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["continuous", "binary"]
Source = Literal["direct", "model"]


@dataclass(frozen=True)
class Flavor:
    """One pretraining target definition.

    Parameters
    ----------
    name : str
        Flavor key, used for cache, foundation, config, and result directory names.
    kind : {"continuous", "binary"}
        ``continuous`` targets train with MSE and are winsorized and z-scored;
        ``binary`` targets train with BCE and skip rescaling.
    source : {"direct", "model"}
        ``direct`` targets are computed in-repo from the graph; ``model`` targets are
        produced by running a learned model over the shared corpus in an isolated env.
    target_dim : int or None
        Number of target columns, for reference only. ``None`` when not known until
        the target is computed.
    env : str
        Conda environment that computes this flavor's target.
    description : str
        Short human-readable summary.
    derived_from : str or None, optional
        For a flavor whose target is not computed by its own calculator but derived from
        another flavor's already-computed target (e.g. ``osmordred_pca80`` is PCA fit on
        ``osmordred``'s fully-prescaled descriptor matrix), the base flavor name. ``None``
        for every flavor with its own calculator. ``compute_targets.sbatch`` skips a
        derived flavor's own target stage (nothing to compute independently); the
        derivation runs later in ``split.sbatch``, after the base flavor's raw target
        exists.
    """

    name: str
    kind: Kind
    source: Source
    target_dim: int | None
    env: str
    description: str
    derived_from: str | None = None


FLAVORS: dict[str, Flavor] = {
    "osmordred": Flavor(
        "osmordred", "continuous", "direct", 3585, "sarizard-osmordred",
        "osmordred 2D physicochemical descriptors (performant Mordred reimplementation)",
    ),
    "rdkit2d": Flavor(
        "rdkit2d", "continuous", "direct", 200, "sarizard",
        "curated RDKit 2D physicochemical descriptors (scikit-fingerprints)",
    ),
    "erg": Flavor(
        "erg", "continuous", "direct", 315, "sarizard",
        "extended reduced-graph pharmacophore, continuous (scikit-fingerprints)",
    ),
    "ecfp": Flavor(
        "ecfp", "binary", "direct", 2048, "sarizard",
        "ECFP/Morgan circular substructure bits (scikit-fingerprints)",
    ),
    "atompair": Flavor(
        "atompair", "binary", "direct", 2048, "sarizard",
        "topological atom-pair bits (scikit-fingerprints)",
    ),
    "pubchem": Flavor(
        "pubchem", "binary", "direct", 881, "sarizard",
        "PubChem substructure keys (scikit-fingerprints)",
    ),
    "usrcat": Flavor(
        "usrcat", "continuous", "direct", 60, "sarizard",
        "USRCAT 3D shape plus pharmacophore moments (needs conformers)",
    ),
    "whim": Flavor(
        "whim", "continuous", "direct", 114, "sarizard",
        "WHIM 3D holistic geometry descriptors (needs conformers)",
    ),
    "e3fp": Flavor(
        "e3fp", "binary", "direct", 1024, "sarizard",
        "E3FP 3D circular substructure bits (needs conformers)",
    ),
    "jazzy": Flavor(
        "jazzy", "continuous", "direct", 6, "sarizard-jazzy",
        "Jazzy hydration free energy and H-bond strengths (sdc, sdx, sa, dga, dgp, dgtot)",
    ),
    "minimol": Flavor(
        "minimol", "continuous", "model", 512, "sarizard-minimol",
        "minimol learned molecular embedding, distilled (Graphcore minimol)",
    ),
    "surrogate_adme": Flavor(
        "surrogate_adme", "continuous", "direct", 25, "sarizard",
        "25 surrogate ADME predictions read directly from the Novartis released dataset",
    ),
    "osmordred_pca80": Flavor(
        "osmordred_pca80", "continuous", "direct", None, "sarizard",
        "osmordred, full-recipe prescaled then PCA-compressed to 80% explained variance",
        derived_from="osmordred",
    ),
    "osmordred_pca90": Flavor(
        "osmordred_pca90", "continuous", "direct", None, "sarizard",
        "osmordred, full-recipe prescaled then PCA-compressed to 90% explained variance",
        derived_from="osmordred",
    ),
    "osmordred_pca95": Flavor(
        "osmordred_pca95", "continuous", "direct", None, "sarizard",
        "osmordred, full-recipe prescaled then PCA-compressed to 95% explained variance",
        derived_from="osmordred",
    ),
}


def get_flavor(name: str) -> Flavor:
    """Return the flavor with the given name.

    Parameters
    ----------
    name : str
        Flavor key.

    Returns
    -------
    Flavor
        The registered flavor.

    Raises
    ------
    KeyError
        If no flavor with that name is registered.
    """
    try:
        return FLAVORS[name]
    except KeyError as err:
        raise KeyError(
            f"unknown flavor {name!r}; known flavors: {', '.join(FLAVORS)}"
        ) from err


def flavor_names() -> list[str]:
    """Return all registered flavor names in definition order."""
    return list(FLAVORS)
