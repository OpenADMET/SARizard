"""Export a pretrained foundation into the checkpoint format openadmet-models consumes.

A SARizard pretraining run produces a Chemprop ``MPNN`` (message passing block plus a
descriptor-regression head). openadmet's ``ChemPropModel`` loads a foundation with
``torch.load(path, weights_only=True)`` and expects a dict
``{"hyper_parameters": <BondMessagePassing kwargs>, "state_dict": <mp state>}``; it
rebuilds the block with ``BondMessagePassing(**hyper_parameters)`` and discards any head.

``extract_foundation`` produces exactly that, keeping only plain-scalar hyperparameters so
the file is loadable under ``weights_only=True``. ``train.py`` calls ``save_foundation``
inline after training; the CLI here re-converts existing run checkpoints.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing

# dual import: script-style when run from pretraining/ (sbatch), package-style when imported
# from the repo root (tests)
try:
    from flavors import flavor_names
except ImportError:
    from sarizard.pretraining.flavors import flavor_names

logger = logging.getLogger(__name__)

# Default on-disk locations relative to the sarizard/pretraining/ working directory.
# runs/ sits beside this code; foundations/ lives at the repo root (two levels up).
RUNS_DIR = Path("runs")
FOUNDATIONS_DIR = Path("..") / ".." / "foundations"


def extract_foundation(mpnn: MPNN) -> dict:
    """Extract the openadmet-format foundation dict from a trained MPNN.

    Parameters
    ----------
    mpnn : chemprop.models.MPNN
        A trained model whose ``message_passing`` block is the foundation.

    Returns
    -------
    dict
        ``{"hyper_parameters": dict, "state_dict": dict}`` where the hyperparameters are
        the plain-scalar ``BondMessagePassing`` kwargs (no ``cls`` key, no module-valued
        transforms) so the file loads under ``torch.load(weights_only=True)``.

    Raises
    ------
    TypeError
        If the model's message passing block is not a ``BondMessagePassing`` (openadmet
        rebuilds it as one).
    """
    mp = mpnn.message_passing
    if not isinstance(mp, BondMessagePassing):
        raise TypeError(
            f"foundation requires a BondMessagePassing block; got {type(mp).__name__}. "
            "openadmet rebuilds the foundation as BondMessagePassing(**hyper_parameters)."
        )
    # keep only JSON-plain hyperparameters; drop the 'cls' key and any module-valued
    # transforms (None or nn.Module) so weights_only loading accepts the file
    hyper_parameters = {}
    for key, value in dict(mp.hparams).items():
        plain = _plain_scalar(value)
        if plain is not None:
            hyper_parameters[key] = plain
    return {"hyper_parameters": hyper_parameters, "state_dict": mp.state_dict()}


def _plain_scalar(value: object) -> int | float | str | bool | None:
    """Coerce a hyperparameter to a weights_only-safe plain scalar, or None to drop it.

    chemprop stores ``activation`` as an ``Activation`` str-enum, which passes an
    ``isinstance(str)`` check but pickles as a ``chemprop.nn.utils.Activation`` class global
    that ``torch.load(weights_only=True)`` refuses. Collapsing str subclasses to a plain
    ``str`` (and dropping None/module-valued transforms) keeps the foundation loadable.
    """
    if isinstance(value, bool):  # before int: bool is an int subclass
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return str(value)  # plain str, even for a str-enum subclass like Activation
    return None


def save_foundation(mpnn: MPNN, out_path: Path) -> Path:
    """Write the foundation extracted from ``mpnn`` to ``out_path``.

    Parameters
    ----------
    mpnn : chemprop.models.MPNN
        Trained model to export.
    out_path : pathlib.Path
        Destination ``.pt`` file; parent directories are created.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extract_foundation(mpnn), out_path)
    logger.info("wrote foundation to %s", out_path)
    return out_path


def convert_checkpoint(ckpt_path: Path, out_path: Path) -> Path:
    """Convert one Lightning checkpoint into a foundation file.

    Parameters
    ----------
    ckpt_path : pathlib.Path
        A first-party Chemprop MPNN Lightning checkpoint from a pretraining run.
    out_path : pathlib.Path
        Destination foundation ``.pt`` file.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    # first-party checkpoint from our own pretraining run; load_from_checkpoint
    # reconstructs the module (it needs the saved hyperparameters, so it is not a
    # weights_only load). The exported foundation is loaded weights_only downstream.
    mpnn = MPNN.load_from_checkpoint(ckpt_path, map_location="cpu")
    return save_foundation(mpnn, out_path)


def _latest_best_ckpt(flavor: str) -> Path | None:
    """Return the most recent run's checkpoint for a flavor, or None if absent."""
    runs = sorted((RUNS_DIR / flavor).glob("*/checkpoints/*.ckpt"))
    return runs[-1] if runs else None


def main() -> None:
    """Convert pretraining checkpoints to foundation files from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="convert the latest run of every flavor")
    group.add_argument("--ckpt", type=Path, help="a single checkpoint to convert")
    parser.add_argument("--flavor", help="flavor name (required with --ckpt; names the output)")
    parser.add_argument("--out", type=Path, help="output path (defaults to foundations/<flavor>_mp.pt)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.all:
        for flavor in flavor_names():
            ckpt = _latest_best_ckpt(flavor)
            if ckpt is None:
                logger.info("no checkpoint found for %s; skipping", flavor)
                continue
            convert_checkpoint(ckpt, FOUNDATIONS_DIR / f"{flavor}_mp.pt")
        return

    if not args.flavor:
        parser.error("--flavor is required with --ckpt")
    out = args.out or FOUNDATIONS_DIR / f"{args.flavor}_mp.pt"
    convert_checkpoint(args.ckpt, out)


if __name__ == "__main__":
    main()
