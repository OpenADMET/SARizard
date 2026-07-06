"""Pretrain one foundation flavor: regress its cached descriptor target from the graph.

Adapted from how-to-train-your-chemeleon. The differences from upstream are the
SARizard compatibility invariants and the per-flavor loss:

- mean aggregation (openadmet rebuilds foundations with MeanAggregation), not NormAggregation
- the DEFAULT graph featurizer (matches openadmet's ChemPropFeaturizer dims), not RIGR
- continuous targets train with masked MSE and a RegressionFFN; binary fingerprint targets
  train with masked BCE and a BinaryClassificationFFN
- the trained message-passing block is exported to the openadmet foundation format inline
- gradient clipping and mixed precision (bf16/16, matching ../foundation-models/pretraining),
  added after every prescaling-ablation run diverged mid-training on the unclipped fp32 regime

Usage:
    python train.py --flavor osmordred --input-dir <split_dir> --output-dir runs/osmordred
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import polars
import torch
import zarr
from chemprop.featurizers import (
    MultiHotAtomFeaturizer,
    MultiHotBondFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.models import MPNN
from chemprop.nn import (
    BinaryClassificationFFN,
    BondMessagePassing,
    MeanAggregation,
    RegressionFFN,
    metrics,
)
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rdkit.rdBase import BlockLogs
from torch.utils.data import DataLoader

from config import (
    DROPOUT_FRACTION,
    EPOCHS,
    FEATURIZER,
    FINAL_LEARNING_RATE,
    FNN_ACTIVATION,
    FNN_HIDDEN_LAYERS,
    FNN_HIDDEN_SIZE,
    GRADIENT_CLIP_VAL,
    INITIAL_LEARNING_RATE,
    MAXIMUM_LEARNING_RATE,
    MP_ACTIVATION,
    MP_DEPTH,
    MP_HIDDEN_SIZE,
    PATIENCE,
    WARMUP_EPOCHS,
)
from convert_checkpoint import FOUNDATIONS_DIR, save_foundation
from dataset import ChempropChunkwiseZarrDataset
from flavors import get_flavor
from losses import RandomDropoutBCE, RandomDropoutMSE

logger = logging.getLogger(__name__)

# seed shared across flavors so the only intended difference is the target block
SEED = 42


def default_precision() -> str:
    """Return the fastest stable mixed-precision mode for the current GPU.

    Matches ../foundation-models/pretraining/run_pretraining.py: bf16 avoids the fp16
    overflow risk on descriptor targets that still have heavy tails after prescaling.
    """
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


def _build_featurizer() -> SimpleMoleculeMolGraphFeaturizer:
    """Build the DEFAULT graph featurizer that matches openadmet's ChemPropFeaturizer."""
    if FEATURIZER.upper() != "DEFAULT":
        raise ValueError(
            f"FEATURIZER must be DEFAULT for openadmet compatibility, got {FEATURIZER!r}; "
            "RIGR changes d_v/d_e and breaks loading via from_foundation (see AGENTS.md)."
        )
    return SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=MultiHotAtomFeaturizer.v2(),
        bond_featurizer=MultiHotBondFeaturizer(),
    )


def _build_model(
    kind: str,
    n_features: int,
    featurizer: SimpleMoleculeMolGraphFeaturizer,
    dropout_fraction: float = DROPOUT_FRACTION,
) -> MPNN:
    """Assemble the MPNN with a kind-appropriate head, loss, and validation metrics.

    Parameters
    ----------
    kind : {"continuous", "binary"}
        The flavor target kind; selects MSE+RegressionFFN or BCE+BinaryClassificationFFN.
    n_features : int
        Number of target columns, read from the cached target store.
    featurizer : SimpleMoleculeMolGraphFeaturizer
        The graph featurizer, used for the message-passing input dimensions.
    dropout_fraction : float, optional
        Masked-pretext target-dropout fraction (default the shared ``DROPOUT_FRACTION``
        regime constant). Override per flavor when the regime default keeps too few
        targets per step for a narrow target block.

    Returns
    -------
    chemprop.models.MPNN
        The assembled model. ``metrics[-1]`` is a deterministic minimization metric so
        ``val_loss`` (logged by chemprop as ``metrics[-1]``) drives early stopping.
    """
    mp = BondMessagePassing(
        d_v=featurizer.atom_fdim,
        d_e=featurizer.bond_fdim,
        d_h=MP_HIDDEN_SIZE,
        depth=MP_DEPTH,
        activation=MP_ACTIVATION,
    )
    ffn_kwargs = {
        "n_tasks": n_features,
        "input_dim": MP_HIDDEN_SIZE,
        "hidden_dim": FNN_HIDDEN_SIZE,
        "n_layers": FNN_HIDDEN_LAYERS,
        "activation": FNN_ACTIVATION,
    }
    if kind == "continuous":
        predictor = RegressionFFN(
            criterion=RandomDropoutMSE(dropout_fraction=dropout_fraction), **ffn_kwargs
        )
        # last metric is a deterministic MSE -> val_loss for early stopping
        metric_list = [metrics.MAE(), metrics.R2Score(), metrics.RMSE(), metrics.MSE()]
    elif kind == "binary":
        predictor = BinaryClassificationFFN(
            criterion=RandomDropoutBCE(dropout_fraction=dropout_fraction), **ffn_kwargs
        )
        # last metric is a deterministic BCE (on logits) -> val_loss for early stopping
        metric_list = [metrics.BinaryAUROC(), metrics.BCELoss()]
    else:
        raise ValueError(f"unknown flavor kind {kind!r}")

    return MPNN(
        mp,
        MeanAggregation(),
        predictor=predictor,
        metrics=metric_list,
        init_lr=INITIAL_LEARNING_RATE,
        max_lr=MAXIMUM_LEARNING_RATE,
        final_lr=FINAL_LEARNING_RATE,
        warmup_epochs=WARMUP_EPOCHS,
    )


def main() -> None:
    """Pretrain one flavor end to end and export its foundation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", required=True, help="flavor name (see flavors.py)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="split dir with train_rescaled.zarr, val_rescaled.zarr, and the smiles parquets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="run directory; a timestamped subdirectory is created inside it",
    )
    parser.add_argument(
        "--foundation-name",
        default=None,
        help="exported foundation filename (default <flavor>_mp.pt); set for ablation runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            "training seed (default 42, shared across the flavor sweep so the only intended "
            "difference is the target block); vary it only for the osmordred prescaling "
            "triage to estimate seed-driven variance"
        ),
    )
    parser.add_argument(
        "--dropout-fraction",
        type=float,
        default=DROPOUT_FRACTION,
        help=(
            "masked-pretext target-dropout fraction (default the shared DROPOUT_FRACTION "
            "regime constant, 0.85); override only for a narrow target block where the "
            "default keeps too few targets per step (e.g. ml_qm, surrogate_adme), as a "
            "recorded per-flavor deviation, not a change to the regime for every flavor"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    BlockLogs()
    seed_everything(args.seed, workers=True)

    flavor = get_flavor(args.flavor)
    run_dir = args.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
    run_dir.mkdir(parents=True, exist_ok=True)

    training_store = args.input_dir / "train_rescaled.zarr"
    validation_store = args.input_dir / "val_rescaled.zarr"
    n_features = zarr.open_array(training_store, mode="r").shape[1]
    logger.info("flavor=%s kind=%s n_features=%d", flavor.name, flavor.kind, n_features)

    featurizer = _build_featurizer()
    train_smiles = polars.read_parquet(args.input_dir / "train_smiles.parquet")["SMILES"].to_list()
    val_smiles = polars.read_parquet(args.input_dir / "val_smiles.parquet")["SMILES"].to_list()
    train_dl = DataLoader(
        ChempropChunkwiseZarrDataset(train_smiles, training_store, featurizer),
        batch_size=None,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        ChempropChunkwiseZarrDataset(val_smiles, validation_store, featurizer),
        batch_size=None,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    model = _build_model(flavor.kind, n_features, featurizer, dropout_fraction=args.dropout_fraction)

    # monitor a deterministic validation metric. chemprop appends a clone of the
    # random-dropout criterion as metrics[-1] and logs it as val_loss, so continuous
    # flavors instead monitor the deterministic val/mse from metrics[:-1]. The binary
    # per-batch metrics are computed on sigmoid probabilities (not logits), so for binary
    # we fall back to val_loss, the masked BCE on logits, which is correct if mildly noisy.
    monitor, mode = ("val/mse", "min") if flavor.kind == "continuous" else ("val_loss", "min")

    trainer = Trainer(
        max_epochs=EPOCHS,
        precision=default_precision(),
        gradient_clip_val=GRADIENT_CLIP_VAL,
        logger=TensorBoardLogger(run_dir, name="tensorboard_logs", default_hp_metric=False),
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor=monitor, mode=mode, patience=PATIENCE),
            ModelCheckpoint(
                monitor=monitor, mode=mode, save_top_k=2, dirpath=run_dir / "checkpoints"
            ),
        ],
    )
    trainer.fit(model, train_dl, val_dl)

    # reload the best checkpoint, then export the foundation in openadmet format
    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info("best checkpoint: %s", best_ckpt)
    best = MPNN.load_from_checkpoint(best_ckpt, map_location="cpu")
    foundation_name = args.foundation_name or f"{flavor.name}_mp.pt"
    save_foundation(best, run_dir / foundation_name)
    save_foundation(best, FOUNDATIONS_DIR / foundation_name)

    # record provenance next to the run for the report card and reproducibility
    (run_dir / "foundation.json").write_text(
        json.dumps(
            {
                "flavor": flavor.name,
                "kind": flavor.kind,
                "n_features": n_features,
                "seed": args.seed,
                "featurizer": "DEFAULT",
                "aggregation": "mean",
                "best_checkpoint": best_ckpt,
                "regime": {
                    "epochs": EPOCHS,
                    "patience": PATIENCE,
                    "mp_hidden": MP_HIDDEN_SIZE,
                    "mp_depth": MP_DEPTH,
                    "gradient_clip_val": GRADIENT_CLIP_VAL,
                    "precision": default_precision(),
                    "dropout_fraction": args.dropout_fraction,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
