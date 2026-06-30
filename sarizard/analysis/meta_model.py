"""Meta-model: stack per-flavor finetuned predictions per endpoint and test for a lift.

For each endpoint, the per-flavor test-set predictions become the features of a small
stacking model (LGBM, random forest, or MLP) that learns to combine them, and its
cross-validated score is compared to the best single flavor on the same molecules. The
question is whether an ensemble of foundations beats the best single foundation.

Leakage discipline: the flavor predictions are on the held-out finetuning test set (the
flavor models never trained on these molecules). The meta-model is the only thing being
fit here, so it is cross-validated over that test set: its reported score is out-of-fold,
never trained on the rows it scores. A single flavor's prediction is a fixed vector (nothing
is fit to the test set), so its direct test score is already honest and needs no CV.

This step depends only on numpy, pandas, scikit-learn, lightgbm, and matplotlib (the flavor
predictions are read from cached ``y_pred.npy`` files), so it runs without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.meta_model                       # LGBM stacker
    python -m sarizard.analysis.meta_model --estimator rf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - set backend before importing pyplot
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from sarizard.analysis.metrics_spec import dataset_of  # noqa: E402
from sarizard.analysis.paths import PLOTS_DIR, RESULTS_DIR  # noqa: E402
from sarizard.pretraining.flavors import flavor_names  # noqa: E402

logger = logging.getLogger(__name__)

ESTIMATORS = ("lgbm", "rf", "mlp")


def _make_estimator(name: str, seed: int):
    """Build a stacking estimator by name (lightgbm imported lazily)."""
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, random_state=seed, verbosity=-1)
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    raise ValueError(f"unknown estimator {name!r}")


def collect_predictions(results_root: Path, flavors: list[str]) -> dict:
    """Gather per-flavor test predictions, keyed by (dataset, recipe, endpoint).

    Returns
    -------
    dict
        ``{(dataset, recipe, endpoint): {"y": ndarray, "preds": {flavor: ndarray}}}``. The
        test split is identical across flavors of the same recipe (only the foundation
        differs), so the per-flavor vectors for an endpoint are molecule-aligned.
    """
    store: dict[tuple[str, str, str], dict] = {}
    for flavor in flavors:
        flavor_dir = results_root / flavor
        if not flavor_dir.is_dir():
            continue
        for result_dir in sorted(p for p in flavor_dir.iterdir() if p.is_dir()):
            pred_path = result_dir / "data" / "y_pred.npy"
            test_path = result_dir / "data" / "y_test.csv"
            if not (pred_path.exists() and test_path.exists()):
                continue
            preds = np.asarray(np.load(pred_path))
            y_test = pd.read_csv(test_path)
            for i, col in enumerate(y_test.columns):
                mask = y_test[col].notna().to_numpy()
                if mask.sum() == 0:
                    continue
                key = (dataset_of(result_dir.name), result_dir.name, col)
                entry = store.setdefault(key, {"y": y_test[col].to_numpy()[mask], "preds": {}})
                entry["preds"][flavor] = preds[mask, i]
    return store


def _evaluate_endpoint(entry: dict, estimator: str, n_splits: int, seed: int) -> dict | None:
    """Cross-validate the stacker for one endpoint and score the best single flavor."""
    y = np.asarray(entry["y"], dtype=float)
    flavors = [flavor for flavor in flavor_names() if flavor in entry["preds"]]
    folds = min(n_splits, len(y) // 2)
    if len(flavors) < 2 or folds < 2:
        return None

    features = np.column_stack([entry["preds"][flavor] for flavor in flavors])
    oof = np.full(len(y), np.nan)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(features):
        model = _make_estimator(estimator, seed)
        model.fit(features[train_idx], y[train_idx])
        oof[test_idx] = model.predict(features[test_idx])

    single_r2 = {flavor: float(r2_score(y, entry["preds"][flavor])) for flavor in flavors}
    best_flavor = max(single_r2, key=single_r2.get)
    meta_r2 = float(r2_score(y, oof))
    return {
        "n_test": int(len(y)),
        "n_flavors": len(flavors),
        "estimator": estimator,
        "meta_r2": meta_r2,
        "meta_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "best_single_flavor": best_flavor,
        "best_single_r2": single_r2[best_flavor],
        "delta_r2": meta_r2 - single_r2[best_flavor],
    }


def run(
    results_root: Path, flavors: list[str], estimator: str, n_splits: int, seed: int
) -> pd.DataFrame:
    """Evaluate the stacker against the best single flavor for every endpoint."""
    store = collect_predictions(results_root, flavors)
    rows: list[dict] = []
    for (dataset, recipe, endpoint), entry in store.items():
        result = _evaluate_endpoint(entry, estimator, n_splits, seed)
        if result is None:
            logger.info("skipping %s:%s (need >=2 flavors and enough rows)", recipe, endpoint)
            continue
        rows.append({"dataset": dataset, "recipe": recipe, "endpoint": endpoint, **result})
    return pd.DataFrame(rows)


def plot_delta(frame: pd.DataFrame, estimator: str, out_png: Path) -> None:
    """Bar chart of meta minus best-single R-squared per endpoint."""
    ordered = frame.sort_values("delta_r2")
    labels = ordered["dataset"] + " · " + ordered["endpoint"]
    colors = ["tab:green" if d > 0 else "tab:red" for d in ordered["delta_r2"]]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(ordered) + 2.0), constrained_layout=True)
    ax.barh(np.arange(len(ordered)), ordered["delta_r2"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("meta R² − best single flavor R²")
    ax.set_title(f"Stacking lift ({estimator}); green = ensemble beats best single", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the meta-model comparison and write its CSV, plot, and summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", default="lgbm", choices=ESTIMATORS, help="stacking model")
    parser.add_argument("--flavors", nargs="*", default=None, help="flavor subset (default all)")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="results root")
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    flavors = args.flavors or flavor_names()
    frame = run(args.results, flavors, args.estimator, args.folds, args.seed)
    if frame.empty:
        raise SystemExit(f"no stackable endpoints under {args.results} (need >=2 flavors each)")

    out_csv = args.results / f"meta_model_{args.estimator}.csv"
    frame.to_csv(out_csv, index=False)
    plot_delta(frame, args.estimator, PLOTS_DIR / f"meta_model_{args.estimator}.png")

    wins = int((frame["delta_r2"] > 0).sum())
    logger.info(
        "%s: meta beats best single on %d/%d endpoints; mean delta R2 = %+.4f; wrote %s",
        args.estimator, wins, len(frame), float(frame["delta_r2"].mean()), out_csv,
    )


if __name__ == "__main__":
    main()
