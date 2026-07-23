"""Meta-model: stack per-flavor finetuned predictions per endpoint and test for a lift.

For each endpoint, the per-flavor test-set predictions become the features of a small
stacking model (LGBM, random forest, or MLP) that learns to combine them, and its
cross-validated score is compared to the best single flavor on the same molecules. The
question is whether an ensemble of foundations beats the best single foundation.

Leakage discipline: the flavor predictions are on the held-out finetuning test set (the
flavor models never trained on these molecules). The meta-model is the only thing being
fit here, so it is cross-validated over that test set: its reported score is out-of-fold,
never trained on the rows it scores. The best-single-flavor baseline is scored out-of-fold
the same way: the winning flavor is chosen on each fold's training rows and scored on its
held-out rows, so selecting the baseline never peeks at the labels the meta-model is scored
against. A given flavor's prediction is a fixed vector, but choosing the best among several
on the full test set would fit the selection to the test labels.

This step depends only on numpy, pandas, scikit-learn, lightgbm, and matplotlib (the flavor
predictions are read from cached ``y_pred.npy`` files), so it runs without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.meta_model                       # LGBM stacker
    python -m sarizard.analysis.meta_model --estimator rf
"""

from __future__ import annotations

import argparse
import json
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
from sarizard.analysis.paths import PLOTS_DIR, RESULTS_DIR, parse_seed_variant  # noqa: E402
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


def collect_predictions(
    results_root: Path, flavors: list[str], *, strip_prefix: str = ""
) -> dict:
    """Gather per-flavor test predictions, grouped by (dataset, recipe, endpoint) then seed.

    Parameters
    ----------
    results_root : pathlib.Path
        Root under which each label in ``flavors`` names a result dir.
    flavors : list of str
        The result-dir labels to read (typically ``<flavor>__s<seed>`` seed variants, or
        ``lr_<mode>__<flavor>__s<seed>`` for a learning-rate protocol).
    strip_prefix : str, optional
        A namespace prefix (e.g. ``lr_reduced__``) to strip from each label's base before it
        becomes the per-flavor feature key. This maps a protocol's ``lr_<mode>__<flavor>``
        labels back to the bare flavor name, so the stacker's feature columns match the flavor
        registry the same way the frozen sweep's do. Default keeps the base unchanged.

    Returns
    -------
    dict
        ``{(dataset, recipe, endpoint): {seed: {"y": ndarray, "preds": {flavor: ndarray}}}}``.
        At one finetune seed every flavor shares the same train/val/test split (only the
        foundation differs), so that seed's per-flavor vectors are molecule-aligned and
        stackable. Across seeds the split is reseeded (the multi-task endpoints resample the
        test set per seed), so seeds are kept in separate buckets here and combined only at the
        metric level by ``run``, never by averaging raw predictions of differing length.
        Non-seeded labels (e.g. ``chemeleon_stock``) land under seed ``None``.
    """
    # accumulate one bucket per (endpoint, seed); flavors within a bucket share the split
    store: dict[tuple[str, str, str], dict] = {}
    for flavor in flavors:
        base, seed = parse_seed_variant(flavor)
        # map a protocol-namespaced label (lr_<mode>__<flavor>) back to the bare flavor key so
        # the registry filter in _evaluate_endpoint matches it as it does the frozen sweep
        if strip_prefix and base.startswith(strip_prefix):
            base = base[len(strip_prefix):]
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
            # align prediction columns to the model's target order. evaluate.py writes that
            # order to target_cols.json; fall back to the y_test columns for older result dirs
            cols_path = result_dir / "data" / "target_cols.json"
            target_cols = (
                json.loads(cols_path.read_text())
                if cols_path.exists()
                else list(y_test.columns)
            )
            for i, col in enumerate(target_cols):
                mask = y_test[col].notna().to_numpy()
                if mask.sum() == 0:
                    continue
                key = (dataset_of(result_dir.name), result_dir.name, col)
                by_seed = store.setdefault(key, {})
                entry = by_seed.setdefault(seed, {"y": y_test[col].to_numpy()[mask], "preds": {}})
                entry["preds"][base] = preds[mask, i]
    return store


def _evaluate_endpoint(entry: dict, estimator: str, n_splits: int, seed: int) -> dict | None:
    """Cross-validate the stacker against a fold-wise best-single-flavor baseline.

    Both the stacker and the baseline are scored out-of-fold so the comparison is symmetric:
    within each fold the best single flavor is chosen on the training rows only, then scored on
    the held-out rows. Picking the best flavor by its score over the full test set would select
    on the same labels the meta-model is scored against, an optimistic baseline.
    """
    y = np.asarray(entry["y"], dtype=float)
    flavors = [flavor for flavor in flavor_names() if flavor in entry["preds"]]
    folds = min(n_splits, len(y) // 2)
    if len(flavors) < 2 or folds < 2:
        return None

    features = np.column_stack([entry["preds"][flavor] for flavor in flavors])
    oof = np.full(len(y), np.nan)
    baseline_oof = np.full(len(y), np.nan)
    fold_winners: list[str] = []
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(features):
        model = _make_estimator(estimator, seed)
        model.fit(features[train_idx], y[train_idx])
        oof[test_idx] = model.predict(features[test_idx])
        # fold-wise baseline: select on the training rows, score on the held-out rows
        train_r2 = [r2_score(y[train_idx], features[train_idx, j]) for j in range(len(flavors))]
        best_j = int(np.argmax(train_r2))
        baseline_oof[test_idx] = features[test_idx, best_j]
        fold_winners.append(flavors[best_j])

    meta_r2 = float(r2_score(y, oof))
    baseline_r2 = float(r2_score(y, baseline_oof))
    # report the flavor selected in the most folds as the representative single baseline
    best_flavor = max(set(fold_winners), key=fold_winners.count)
    return {
        "n_test": int(len(y)),
        "n_flavors": len(flavors),
        "estimator": estimator,
        "meta_r2": meta_r2,
        "meta_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "best_single_flavor": best_flavor,
        "best_single_r2": baseline_r2,
        "delta_r2": meta_r2 - baseline_r2,
    }


def _aggregate_seeds(seed_results: list[dict], estimator: str) -> dict:
    """Average an endpoint's per-seed stacker scores into one row with seed error bars.

    Each element of ``seed_results`` is one finetune seed's ``_evaluate_endpoint`` output. The
    seeds reuse different random test splits (the multi-task endpoints resample per seed), so
    their raw predictions are not comparable row-for-row; combining them at the metric level (a
    mean with a standard deviation) is what makes the stacker's score a seed average with error
    bars, like every other report-card cell. The single-flavor baseline is averaged the same
    way, and the representative winner is the flavor chosen in the most seeds.
    """
    meta = np.array([r["meta_r2"] for r in seed_results], dtype=float)
    delta = np.array([r["delta_r2"] for r in seed_results], dtype=float)
    winners = [r["best_single_flavor"] for r in seed_results]
    # sample std across seeds; a single seed has no spread, so report 0.0 rather than nan
    single_seed = len(seed_results) < 2
    return {
        "n_seeds": len(seed_results),
        "n_flavors": max(r["n_flavors"] for r in seed_results),
        "n_test": int(round(float(np.mean([r["n_test"] for r in seed_results])))),
        "estimator": estimator,
        "meta_r2": float(meta.mean()),
        "meta_r2_std": 0.0 if single_seed else float(meta.std(ddof=1)),
        "meta_rmse": float(np.mean([r["meta_rmse"] for r in seed_results])),
        "best_single_flavor": max(set(winners), key=winners.count),
        "best_single_r2": float(np.mean([r["best_single_r2"] for r in seed_results])),
        "delta_r2": float(delta.mean()),
        "delta_r2_std": 0.0 if single_seed else float(delta.std(ddof=1)),
    }


def run(
    results_root: Path,
    flavors: list[str],
    estimator: str,
    n_splits: int,
    seed: int,
    *,
    strip_prefix: str = "",
) -> pd.DataFrame:
    """Score the stacker per finetune seed and average across seeds, for every endpoint."""
    store = collect_predictions(results_root, flavors, strip_prefix=strip_prefix)
    rows: list[dict] = []
    for (dataset, recipe, endpoint), by_seed in store.items():
        # evaluate each seed on its own aligned split, then average the scores across seeds;
        # None sorts last so a stray non-seeded bucket never orders against an int seed
        seed_results = []
        for seed_key in sorted(by_seed, key=lambda s: (s is None, s)):
            result = _evaluate_endpoint(by_seed[seed_key], estimator, n_splits, seed)
            if result is not None:
                seed_results.append(result)
        if not seed_results:
            logger.info("skipping %s:%s (need >=2 flavors and enough rows)", recipe, endpoint)
            continue
        rows.append(
            {
                "dataset": dataset,
                "recipe": recipe,
                "endpoint": endpoint,
                **_aggregate_seeds(seed_results, estimator),
            }
        )
    return pd.DataFrame(rows)


def plot_delta(frame: pd.DataFrame, estimator: str, out_png: Path) -> None:
    """Bar chart of meta minus best-single R-squared per endpoint."""
    ordered = frame.sort_values("delta_r2")
    labels = ordered["dataset"] + " · " + ordered["endpoint"]
    colors = ["tab:green" if d > 0 else "tab:red" for d in ordered["delta_r2"]]

    # seed spread as x error bars when the CSV carries it (single-seed rows report 0.0)
    xerr = ordered["delta_r2_std"] if "delta_r2_std" in ordered else None
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(ordered) + 2.0), constrained_layout=True)
    ax.barh(
        np.arange(len(ordered)), ordered["delta_r2"], color=colors,
        xerr=xerr, error_kw={"elinewidth": 0.8, "ecolor": "0.3"},
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(ordered)), labels=list(labels), fontsize=12)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("meta R² − best single flavor R²", fontsize=14)
    ax.set_title(
        f"Stacking lift ({estimator}); green = ensemble beats best single", fontsize=15
    )
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
    parser.add_argument(
        "--lr-mode", choices=("reduced", "unlocked"), default=None,
        help="score one learning-rate protocol: read its lr_<mode>__<flavor> result dirs "
        "(passed via --flavors), strip the prefix back to the bare flavor, and write a "
        "mode-scoped meta_model_<estimator>_<mode>.csv so the three protocols do not collide",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    flavors = args.flavors or flavor_names()
    strip_prefix = f"lr_{args.lr_mode}__" if args.lr_mode else ""
    frame = run(
        args.results, flavors, args.estimator, args.folds, args.seed, strip_prefix=strip_prefix
    )
    if frame.empty:
        raise SystemExit(f"no stackable endpoints under {args.results} (need >=2 flavors each)")

    # mode-scoped output names keep the three protocols' meta-models from overwriting each other
    suffix = f"_{args.lr_mode}" if args.lr_mode else ""
    out_csv = args.results / f"meta_model_{args.estimator}{suffix}.csv"
    frame.to_csv(out_csv, index=False)
    plot_delta(frame, args.estimator, PLOTS_DIR / f"meta_model_{args.estimator}{suffix}.png")

    wins = int((frame["delta_r2"] > 0).sum())
    mean_seeds = float(frame["n_seeds"].mean()) if "n_seeds" in frame else 1.0
    logger.info(
        "%s: meta beats best single on %d/%d endpoints (scores averaged over ~%.1f seeds); "
        "mean delta R2 = %+.4f; wrote %s",
        args.estimator, wins, len(frame), mean_seeds, float(frame["delta_r2"].mean()), out_csv,
    )


if __name__ == "__main__":
    main()
