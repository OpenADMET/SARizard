"""Report-card heatmap: endpoints (rows) by foundation flavors (columns), one metric.

Reads the tidy metrics CSV written by ``analysis.evaluate`` and renders a heatmap where each
cell is one selectable metric (default R-squared) for a (flavor, endpoint) pair. Color is
row-relative (per endpoint), so the best flavor for each endpoint is greenest regardless of
the metric's absolute scale, and direction follows the metric (higher-better vs lower-better).
The raw value is annotated in each cell. A CSV of the underlying metric matrix is written
alongside the figure.

This step depends only on pandas, numpy, and matplotlib, so it runs without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.report_card                 # R-squared
    python -m sarizard.analysis.report_card --metric rmse
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

from sarizard.analysis.metrics_spec import (  # noqa: E402
    DATASETS,
    HIGHER_IS_BETTER,
    METRIC_COLUMNS,
    METRIC_LABELS,
)
from sarizard.analysis.paths import METRICS_CSV, PLOTS_DIR  # noqa: E402
from sarizard.pretraining.flavors import flavor_names  # noqa: E402

logger = logging.getLogger(__name__)


def build_matrix(
    frame: pd.DataFrame, metric: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """Pivot the tidy metrics into an endpoints-by-columns matrix for one metric.

    Parameters
    ----------
    frame : pandas.DataFrame
        Tidy metrics with columns flavor, dataset, endpoint, and the metric columns.
    metric : str
        Which metric column to display.
    columns : list of str, optional
        Column order for the pivot (values of the ``flavor`` field). Defaults to the flavor
        registry order; pass an explicit list (e.g. ablation labels) to order by something
        other than the registry. Only columns present in ``frame`` are kept.

    Returns
    -------
    pandas.DataFrame
        Rows are ``"<dataset> · <endpoint>"`` ordered by dataset then endpoint; columns are
        ``columns`` (or registry flavors) that appear in ``frame``.
    """
    frame = frame.copy()
    frame["row"] = frame["dataset"] + " · " + frame["endpoint"]
    rank = {dataset: i for i, dataset in enumerate(DATASETS)}
    ordered = (
        frame[["dataset", "endpoint", "row"]]
        .drop_duplicates()
        .assign(_rank=lambda d: d["dataset"].map(lambda x: rank.get(x, len(DATASETS))))
        .sort_values(["_rank", "endpoint"])
    )
    order = columns if columns is not None else flavor_names()
    present = set(frame["flavor"])
    keep = [col for col in order if col in present]
    pivot = frame.pivot_table(index="row", columns="flavor", values=metric, aggfunc="mean")
    return pivot.reindex(index=ordered["row"].tolist(), columns=keep)


def _row_relative(values: np.ndarray, higher_better: bool) -> np.ndarray:
    """Min-max normalize each row to [0, 1] so the best flavor per endpoint maps to 1."""
    normed = np.full(values.shape, np.nan, dtype=float)
    for i in range(values.shape[0]):
        row = values[i]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            continue
        lo, hi = np.nanmin(row[finite]), np.nanmax(row[finite])
        if hi == lo:
            normed[i, finite] = 0.5
            continue
        unit = (row - lo) / (hi - lo)
        normed[i] = unit if higher_better else 1.0 - unit
    return normed


def plot_report_card(pivot: pd.DataFrame, metric: str, out_png: Path, out_csv: Path) -> None:
    """Render and save the report-card heatmap and its metric matrix CSV."""
    values = pivot.to_numpy(dtype=float)
    normed = _row_relative(values, metric in HIGHER_IS_BETTER)
    n_rows, n_cols = values.shape

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("lightgrey")  # missing (flavor, endpoint) cells

    fig, ax = plt.subplots(
        figsize=(1.15 * n_cols + 3.0, 0.42 * n_rows + 2.0), constrained_layout=True
    )
    im = ax.imshow(
        np.ma.masked_invalid(normed), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0
    )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="left", fontsize=9)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # annotate each cell with the raw metric value
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="black")

    label = METRIC_LABELS.get(metric, metric)
    direction = "higher better" if metric in HIGHER_IS_BETTER else "lower better"
    ax.set_title(f"Report card: {label} ({direction}); color is row-relative", fontsize=11, pad=28)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["worst for endpoint", "best for endpoint"], fontsize=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pivot.to_csv(out_csv)
    logger.info("wrote %s and %s", out_png, out_csv)


def main() -> None:
    """Build and save the report card for the requested metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="r2", choices=METRIC_COLUMNS, help="metric to display")
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV, help="tidy metrics CSV")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate first")
    frame = pd.read_csv(args.metrics_csv)
    pivot = build_matrix(frame, args.metric)
    out_png = args.out or (PLOTS_DIR / f"report_card_{args.metric}.png")
    out_csv = out_png.with_suffix(".csv")
    plot_report_card(pivot, args.metric, out_png, out_csv)


if __name__ == "__main__":
    main()
