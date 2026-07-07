"""Report-card heatmap: endpoints (rows) by foundation flavors (columns), one metric.

Reads the tidy metrics CSV written by ``analysis.evaluate`` and renders a heatmap where each
cell is one selectable metric (default R-squared) for a (flavor, endpoint) pair. Color is
row-relative (per endpoint), so the best flavor for each endpoint is greenest regardless of
the metric's absolute scale, and direction follows the metric (higher-better vs lower-better).
The raw value is annotated in each cell. A CSV of the underlying metric matrix is written
alongside the figure.

Two reference columns can be appended after a blank spacer column, clearly labeled and
visually separated from the per-flavor columns with a divider line: the stock-CheMeleon
baseline (``slurm/run_stock_baseline.sh``, an external reference that used a different
corpus and pretraining regime than our flavors) and the LGBM meta-model (stacked
out-of-fold predictions across every flavor, a ceiling reference rather than a deployable
single foundation). Neither is a candidate a reader should "pick" the way they would a
flavor column; the divider and reference labels keep that distinction visible in the plot.

This step depends only on pandas, numpy, and matplotlib, so it runs without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.report_card                 # R-squared, with references
    python -m sarizard.analysis.report_card --metric rmse
    python -m sarizard.analysis.report_card --no-references # flavor columns only
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
from sarizard.analysis.paths import METRICS_CSV, PLOTS_DIR, parse_seed_variant  # noqa: E402
from sarizard.pretraining.flavors import flavor_names  # noqa: E402

logger = logging.getLogger(__name__)


def collapse_seed_variants(frame: pd.DataFrame, column: str = "flavor") -> pd.DataFrame:
    """Map ``<base>__s<seed>`` labels in ``column`` back to their base label.

    Seeds are run as separate variants (own foundation, recipes, results); collapsing the
    labels lets ``build_matrix`` average the seed replicates into one cell per (endpoint, base)
    via its ``aggfunc="mean"`` pivot. Plain labels (no seed suffix) pass through unchanged. The
    base may itself carry a namespace prefix (``ablation_<name>``, ``lr_<mode>__<flavor>``).
    """
    frame = frame.copy()
    frame[column] = frame[column].map(lambda label: parse_seed_variant(label)[0])
    return frame


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


def build_reference_series(frame: pd.DataFrame, flavor: str, metric: str) -> pd.Series:
    """Extract one flavor's per-endpoint metric as a ``"<dataset> · <endpoint>"``-indexed Series.

    Used for a reference flavor (e.g. ``chemeleon_stock``) that should appear as a single
    extra report-card column rather than take part in ``build_matrix``'s registry-ordered
    flavor columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Tidy metrics with columns flavor, dataset, endpoint, and the metric columns.
    flavor : str
        The flavor label to extract (a value of the ``flavor`` column).
    metric : str
        Which metric column to extract.

    Returns
    -------
    pandas.Series
        Indexed like ``build_matrix``'s rows; empty if ``flavor`` has no rows in ``frame``.
    """
    subset = frame[frame["flavor"] == flavor]
    if subset.empty:
        return pd.Series(dtype=float)
    row = subset["dataset"] + " · " + subset["endpoint"]
    return pd.Series(subset[metric].to_numpy(), index=row)


def meta_model_series(meta_model_csv: Path, metric: str) -> pd.Series:
    """Read a ``meta_model.py`` output CSV into a ``"<dataset> · <endpoint>"``-indexed Series.

    Parameters
    ----------
    meta_model_csv : pathlib.Path
        Output of ``sarizard.analysis.meta_model`` (columns include dataset, endpoint,
        meta_r2, meta_rmse).
    metric : str
        Report-card metric being displayed. The meta-model only reports r2 and rmse (see
        ``meta_model._evaluate_endpoint``); any other metric returns an empty Series.

    Returns
    -------
    pandas.Series
        Indexed like ``build_matrix``'s rows; empty if the metric isn't one the meta-model
        reports, or the CSV doesn't exist.
    """
    column = {"r2": "meta_r2", "rmse": "meta_rmse"}.get(metric)
    if column is None or not meta_model_csv.exists():
        return pd.Series(dtype=float)
    frame = pd.read_csv(meta_model_csv)
    row = frame["dataset"] + " · " + frame["endpoint"]
    return pd.Series(frame[column].to_numpy(), index=row)


SPACER_COLUMN = " "  # single space: a distinct, blank column label


def augment_with_references(
    pivot: pd.DataFrame,
    *,
    baseline: pd.Series | None = None,
    baseline_label: str = "chemeleon baseline (stock, external)",
    meta_model: pd.Series | None = None,
    meta_model_label: str = "meta-model (LGBM, all flavors)",
) -> tuple[pd.DataFrame, int]:
    """Append reference columns after a blank spacer column, separate from the flavor columns.

    Parameters
    ----------
    pivot : pandas.DataFrame
        The per-flavor matrix from ``build_matrix``.
    baseline : pandas.Series, optional
        Stock-CheMeleon reference values, indexed like ``pivot``'s rows (see
        ``build_reference_series``). Omitted (or empty) skips this column.
    baseline_label : str, optional
        Column label for the baseline reference.
    meta_model : pandas.Series, optional
        Meta-model reference values, indexed like ``pivot``'s rows (see
        ``meta_model_series``). Omitted (or empty) skips this column.
    meta_model_label : str, optional
        Column label for the meta-model reference.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The augmented matrix, and the column index of the spacer (``pivot.shape[1]`` if no
        reference was added), so the caller can draw a divider there.
    """
    n_flavor_cols = pivot.shape[1]
    references = [
        (label, series)
        for label, series in ((baseline_label, baseline), (meta_model_label, meta_model))
        if series is not None and not series.empty
    ]
    if not references:
        return pivot, n_flavor_cols
    out = pivot.copy()
    out[SPACER_COLUMN] = np.nan
    for label, series in references:
        out[label] = series.reindex(out.index)
    return out, n_flavor_cols


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


def plot_report_card(
    pivot: pd.DataFrame, metric: str, out_png: Path, out_csv: Path, *, divider_at: int | None = None
) -> None:
    """Render and save the report-card heatmap and its metric matrix CSV.

    Parameters
    ----------
    pivot : pandas.DataFrame
        Rows are endpoints, columns are flavors, optionally followed by a blank spacer
        column and reference columns (``augment_with_references``).
    metric : str
        Which metric ``pivot`` holds (drives the title and color direction).
    out_png : pathlib.Path
        Heatmap image output path.
    out_csv : pathlib.Path
        Path for the underlying metric matrix CSV.
    divider_at : int, optional
        Column index of the blank spacer column (``augment_with_references``'s second
        return value). When given and less than the column count, a white gap and divider
        lines are drawn there to visually separate reference columns from flavor columns,
        and the columns after it are bold-labeled as references rather than flavors.
    """
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

    has_references = divider_at is not None and divider_at < n_cols
    if has_references:
        # paint the spacer column white (distinct from the "missing data" lightgrey) and
        # bound it with divider lines, so the reference columns read as a separate block
        ax.axvspan(divider_at - 0.5, divider_at + 0.5, color="white", zorder=2)
        ax.axvline(divider_at - 0.5, color="black", linewidth=1.2, zorder=3)
        ax.axvline(divider_at + 0.5, color="black", linewidth=1.2, zorder=3)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="left", fontsize=9)
    if has_references:
        # bold the reference-column labels (everything past the spacer) so they read as
        # "not a flavor you can pick" rather than blending into the flavor list
        for i, label in enumerate(ax.get_xticklabels()):
            if i > divider_at:
                label.set_fontweight("bold")
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
    title = f"Report card: {label} ({direction}); color is row-relative"
    if has_references:
        title += " across flavors + references"
    ax.set_title(title, fontsize=11, pad=28)
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
    parser.add_argument(
        "--no-references", action="store_true",
        help="omit the stock-CheMeleon baseline and meta-model reference columns",
    )
    parser.add_argument(
        "--baseline-flavor", default="chemeleon_stock",
        help="flavor label for the stock-CheMeleon reference column (see "
        "slurm/run_stock_baseline.sh); skipped if absent from --metrics-csv",
    )
    parser.add_argument(
        "--meta-model-csv", type=Path, default=None,
        help="meta_model.py output CSV for the meta-model reference column "
        "(default results/meta_model_lgbm.csv next to --metrics-csv's results dir); "
        "skipped if it doesn't exist",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate first")
    frame = pd.read_csv(args.metrics_csv)
    # average any per-seed variants back to one column per flavor before pivoting
    frame = collapse_seed_variants(frame)
    pivot = build_matrix(frame, args.metric)

    divider_at = pivot.shape[1]
    if not args.no_references:
        baseline = build_reference_series(frame, args.baseline_flavor, args.metric)
        meta_csv = args.meta_model_csv or (args.metrics_csv.parent / "meta_model_lgbm.csv")
        meta = meta_model_series(meta_csv, args.metric)
        pivot, divider_at = augment_with_references(pivot, baseline=baseline, meta_model=meta)

    out_png = args.out or (PLOTS_DIR / f"report_card_{args.metric}.png")
    out_csv = out_png.with_suffix(".csv")
    plot_report_card(pivot, args.metric, out_png, out_csv, divider_at=divider_at)


if __name__ == "__main__":
    main()
