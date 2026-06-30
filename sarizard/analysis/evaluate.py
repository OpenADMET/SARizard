"""Collect per-(flavor, endpoint) test metrics from finetuning result directories.

For each ``results/<flavor>/<recipe>/`` directory written by ``anvil``, this reloads the
trained model, predicts on the held-out test split, and computes regression metrics per
target column. Predictions are cached as ``data/y_pred.npy`` inside each result dir so the
meta-model (sarizard.analysis.meta_model) can stack per-flavor predictions without re-inferring.

Adapted from the sibling igm ``analysis/analyze.py`` evaluation path. Run in the main
environment (it imports openadmet). The light plotting step lives in ``report_card.py`` and
reads only the tidy CSV this writes, so it needs neither openadmet nor a GPU.

Usage:
    python -m sarizard.analysis.evaluate --accelerator gpu   # all flavors -> results/metrics.csv
    python -m sarizard.analysis.evaluate --flavors ecfp jazzy --force
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from openadmet.models.inference.inference import load_anvil_model_and_metadata
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sarizard.analysis.metrics_spec import dataset_of
from sarizard.analysis.paths import METRICS_CSV, RESULTS_DIR
from sarizard.pretraining.flavors import flavor_names

logger = logging.getLogger(__name__)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for one endpoint against a train-mean baseline."""
    mse = float(mean_squared_error(y_true, y_pred))
    baseline = float(np.mean(np.abs(y_true - y_train.mean())))
    rae = float(np.mean(np.abs(y_true - y_pred)) / baseline) if baseline > 0 else float("nan")
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "kendall": float(kendalltau(y_true, y_pred).statistic),
        "rae": rae,
    }


def evaluate_result_dir(
    result_dir: Path, flavor: str, accelerator: str, *, force: bool = False
) -> list[dict]:
    """Evaluate one result dir, returning one row per target column.

    Parameters
    ----------
    result_dir : pathlib.Path
        An ``anvil`` output directory (contains ``recipe_components/`` and ``data/``).
    flavor : str
        The flavor this result belongs to (the result dir's parent name).
    accelerator : str
        Passed to ``model.predict`` ("gpu" or "cpu").
    force : bool, optional
        Recompute predictions even if a cache exists.

    Returns
    -------
    list of dict
        One metrics row per target column, tagged with flavor, recipe, and dataset.
    """
    model, feat, _, data_spec = load_anvil_model_and_metadata(result_dir)
    data_dir = result_dir / "data"
    x_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")

    # record the model's target column order so the meta-model can align its cached predictions
    # without re-loading openadmet; the prediction array columns follow data_spec.target_cols
    (data_dir / "target_cols.json").write_text(json.dumps(list(data_spec.target_cols)))

    cache = data_dir / "y_pred.npy"
    if cache.exists() and not force:
        preds = np.load(cache)
    else:
        test_dl = feat.featurize(x_test[data_spec.input_col])[0]
        preds = model.predict(test_dl, accelerator=accelerator)
        np.save(cache, preds)

    recipe = result_dir.name
    dataset = dataset_of(recipe)
    rows: list[dict] = []
    for i, col in enumerate(data_spec.target_cols):
        mask = y_test[col].notna().to_numpy()
        if mask.sum() == 0:
            logger.warning("no test labels for %s/%s:%s", flavor, recipe, col)
            continue
        y_true = y_test[col].to_numpy()[mask]
        y_pred = np.asarray(preds)[mask, i]
        train_col = y_train[col].dropna().to_numpy()
        row = {"flavor": flavor, "dataset": dataset, "recipe": recipe, "endpoint": col}
        row.update(_metrics(y_true, y_pred, train_col))
        row["n_test"] = int(mask.sum())
        row["n_train"] = int(train_col.size)
        rows.append(row)
    return rows


def collect(
    results_root: Path, flavors: list[str], accelerator: str, *, force: bool = False
) -> pd.DataFrame:
    """Evaluate every ``results/<flavor>/<recipe>/`` dir into one tidy DataFrame."""
    rows: list[dict] = []
    for flavor in flavors:
        flavor_dir = results_root / flavor
        if not flavor_dir.is_dir():
            continue
        for result_dir in sorted(p for p in flavor_dir.iterdir() if p.is_dir()):
            # each result dir is independent: log and skip a failed one (missing file, corrupt
            # checkpoint, inference error) so one bad dir does not abort the whole sweep
            try:
                rows.extend(evaluate_result_dir(result_dir, flavor, accelerator, force=force))
            except Exception:
                logger.exception("skipping %s: evaluation failed", result_dir)
    return pd.DataFrame(rows)


def main() -> None:
    """Evaluate all requested flavors and write the tidy metrics CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavors", nargs="*", default=None, help="flavor subset (default all)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="results root")
    parser.add_argument("--out", type=Path, default=METRICS_CSV, help="tidy metrics CSV")
    parser.add_argument("--accelerator", default="gpu", help="predict accelerator")
    parser.add_argument("--force", action="store_true", help="recompute cached predictions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    flavors = args.flavors or flavor_names()
    frame = collect(args.results, flavors, args.accelerator, force=args.force)
    if frame.empty:
        raise SystemExit(f"no results found under {args.results}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows, %d flavors)", args.out, len(frame), frame["flavor"].nunique())


if __name__ == "__main__":
    main()
