"""Report-card heatmaps: endpoints (rows) by foundation flavors (columns).

Two cards are rendered per setup from the tidy metrics CSV written by ``analysis.evaluate``:

- an R-squared card colored on a fixed red-to-green scale (red = 0, green = 1), with the
  stock-CheMeleon baseline as the first column (separated from the flavor block by a blank
  spacer) and a final AVERAGE row that means each column across all endpoints;
- a delta card whose cells are the percentage change in MAE relative to the stock-CheMeleon
  baseline (green where a flavor's MAE beats the baseline, red where it is worse), flavor columns
  only, with the same AVERAGE row. A cell is painted white unless the flavor's per-seed MAE
  differs significantly from the baseline's (two-sample Welch t-test, p at or below
  ``SIGNIFICANCE_ALPHA``), so only differences the seed spread supports carry color.

The R² card annotates every endpoint cell (flavors and the multi-seed baseline column) with a
``±`` seed standard deviation under its value; the delta card annotates each cell with its change
and the test p-value. The AVERAGE row shows a bare mean, since its spread is over endpoints, not
seeds.

Both cards group the endpoint rows by their source dataset (asap, chembl, expansionrx, ...)
with a bold black separator line and a bold source label per group.

This step depends only on pandas, numpy, and matplotlib, so it runs without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.report_card                 # both cards, frozen sweep
    python -m sarizard.analysis.report_card --lr-mode reduced
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.stats import ttest_ind  # noqa: E402

from sarizard.analysis.metrics_spec import DATASETS, dataset_of  # noqa: E402
from sarizard.analysis.paths import METRICS_CSV, PLOTS_DIR, parse_seed_variant  # noqa: E402
from sarizard.pretraining.flavors import flavor_names  # noqa: E402

logger = logging.getLogger(__name__)

# significance threshold for the MAE-delta card: a flavor whose per-seed MAE does not differ from
# the baseline's at this level (two-sample Welch t-test, p above the threshold) is painted white,
# so only differences the seed spread supports carry color
SIGNIFICANCE_ALPHA = 0.05

# cap the MAE-delta diverging color scale at this magnitude (percentage points) in both
# directions, so a few large outliers do not wash out the scale; larger changes saturate at the
# end color while their annotation still shows the true value
DELTA_EXTENT_CAP = 25.0

# reference-column labels and the blank labels used for the spacer column(s) and spacer row;
# the spacers carry no data (painted white) and their tick labels are blanked at render
BASELINE_LABEL = "chemeleon\nbaseline"
AVERAGE_LABEL = "AVERAGE"
_SPACER_LEFT = " "  # figure space: a unique, blank column label before the flavor block
_SPACER_ROW = " "

# green-white-red diverging map for the MAE-delta card: green = MAE below baseline (better),
# white = no change, red = MAE above baseline (worse)
_DELTA_CMAP = LinearSegmentedColormap.from_list("mae_delta", ["#1a9850", "#ffffff", "#d73027"])

# report-card font sizes (points), nudged up for legibility at print scale
FONT_TITLE = 13
FONT_AXIS = 11  # x tick labels and the per-group source labels
FONT_YTICK = 10  # endpoint row labels
FONT_CELL = 8  # per-cell value and error-bar/p-value annotation
FONT_CBAR = 9  # colorbar tick labels

# source group whose last endpoint gets a thicker separator line directly after it
EMPHASIS_SOURCE = "pxr"

# clearance (inches) left of the grid for the rotated per-group source labels, enough to clear
# the longest endpoint tick label. Converted to an axes fraction per card from the grid width
# (~1.15 in/column, matching the figsize) so narrow ablation cards get the same absolute gap as
# the wide flavor card rather than a width-scaled one that collides on the narrow layout.
GROUP_LABEL_INCHES = 5.5


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


def filter_lr_mode(frame: pd.DataFrame, mode: str, column: str = "flavor") -> pd.DataFrame:
    """Keep one learning-rate protocol's rows and strip the ``lr_<mode>__`` prefix.

    The LR-experiment metrics CSV (``results/lr_metrics.csv``) namespaces every row's flavor as
    ``lr_<mode>__<flavor>``. Selecting one mode and stripping the prefix rewrites the labels back
    to the bare flavor name, so a reduced or unlocked card reuses the registry-ordered flavor
    columns exactly as the frozen card does. Rows for other modes, and un-prefixed rows (bare
    flavors, reference labels), are dropped, so pass the full frame to the reference-series
    builders separately if their labels are not prefixed.

    Parameters
    ----------
    frame : pandas.DataFrame
        Seed-collapsed tidy metrics (call ``collapse_seed_variants`` first).
    mode : str
        The LR protocol to keep (e.g. ``reduced`` or ``unlocked``).
    column : str, optional
        The label column to filter and rewrite.

    Returns
    -------
    pandas.DataFrame
        Only this mode's rows, with ``column`` rewritten to the bare flavor name.
    """
    prefix = f"lr_{mode}__"
    kept = frame[frame[column].str.startswith(prefix)].copy()
    kept[column] = kept[column].str.slice(len(prefix))
    return kept


def prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the dataset from the recipe and build a disambiguated per-row identity.

    The ``dataset`` column is recomputed from ``recipe`` via ``metrics_spec.dataset_of`` so the
    grouping always matches the current rules regardless of what an older metrics CSV stored
    (e.g. the CYP recipes now group under ``openadmet_cyp``). The row identity is
    ``"<dataset> · <endpoint>"``, with the recipe appended as ``" (<recipe>)"`` wherever the same
    ``(dataset, endpoint)`` is produced by more than one recipe, so a single-task and a
    multi-task model of the same endpoint (e.g. cyp1a2 under ``cyp1a2_st`` and ``cyp_mt``, or
    ``LOG_CLint_HLM`` under a chembl single-task and multi-task recipe) stay separate, labeled
    rows instead of silently averaging together.

    Parameters
    ----------
    frame : pandas.DataFrame
        Tidy metrics with at least ``recipe`` and ``endpoint`` columns.

    Returns
    -------
    pandas.DataFrame
        A copy with the recomputed ``dataset`` column and a new ``row`` identity column.
    """
    frame = frame.copy()
    frame["dataset"] = frame["recipe"].map(dataset_of)
    frame["row"] = frame["dataset"] + " · " + frame["endpoint"]
    # tag only the rows whose (dataset, endpoint) is produced by more than one recipe
    ambiguous = frame.groupby("row")["recipe"].transform("nunique") > 1
    frame.loc[ambiguous, "row"] = frame["row"] + " (" + frame["recipe"] + ")"
    return frame


def build_matrix(
    frame: pd.DataFrame, metric: str, columns: list[str] | None = None, *, aggfunc: str = "mean"
) -> pd.DataFrame:
    """Pivot the prepared metrics into an endpoints-by-columns matrix for one metric.

    Parameters
    ----------
    frame : pandas.DataFrame
        Metrics already run through ``prepare_rows`` (carries ``dataset``, ``endpoint``, ``row``,
        ``flavor``, and the metric columns).
    metric : str
        Which metric column to display.
    columns : list of str, optional
        Column order for the pivot (values of the ``flavor`` field). Defaults to the flavor
        registry order; pass an explicit list (e.g. ablation labels) to order by something
        other than the registry. Only columns present in ``frame`` are kept.
    aggfunc : str, optional
        How to combine the seed replicates that ``collapse_seed_variants`` maps to one flavor
        label. Defaults to ``"mean"`` (the displayed value); pass ``"std"`` to build the matching
        per-cell standard deviation for the error-bar annotation. A cell with a single seed is
        NaN under ``"std"`` (no spread defined).

    Returns
    -------
    pandas.DataFrame
        Rows are the ``row`` identities ordered by dataset then endpoint; columns are
        ``columns`` (or registry flavors) that appear in ``frame``.
    """
    frame = frame.copy()
    if "row" not in frame.columns:
        # an unprepared frame (e.g. the ablation report) keys rows by dataset and endpoint
        # directly, without the recipe disambiguation prepare_rows adds
        frame["row"] = frame["dataset"] + " · " + frame["endpoint"]
    rank = {dataset: i for i, dataset in enumerate(DATASETS)}
    ordered = (
        frame[["dataset", "endpoint", "row"]]
        .drop_duplicates()
        .assign(_rank=lambda d: d["dataset"].map(lambda x: rank.get(x, len(DATASETS))))
        .sort_values(["_rank", "endpoint", "row"])
    )
    order = columns if columns is not None else flavor_names()
    present = set(frame["flavor"])
    keep = [col for col in order if col in present]
    pivot = frame.pivot_table(index="row", columns="flavor", values=metric, aggfunc=aggfunc)
    return pivot.reindex(index=ordered["row"].tolist(), columns=keep)


def build_reference_series(
    frame: pd.DataFrame, flavor: str, metric: str, *, agg: str = "mean"
) -> pd.Series:
    """Extract one flavor's per-endpoint metric as a ``"<dataset> · <endpoint>"``-indexed Series.

    Used for a reference flavor (e.g. ``chemeleon_stock``) that should appear as a single
    extra report-card column, or supply the baseline the MAE-delta card is measured against,
    rather than take part in ``build_matrix``'s registry-ordered flavor columns. When the
    baseline is run at several seeds (collapsed to one label by ``collapse_seed_variants``), the
    per-endpoint duplicates are the seed replicates.

    Parameters
    ----------
    frame : pandas.DataFrame
        Metrics already run through ``prepare_rows`` (carries the ``row`` identity and the
        metric columns).
    flavor : str
        The flavor label to extract (a value of the ``flavor`` column).
    metric : str
        Which metric column to extract.
    agg : str, optional
        How to combine the seed replicates per endpoint. Defaults to ``"mean"`` (the displayed
        baseline value); pass ``"std"`` for the baseline error bar. A single-seed baseline is NaN
        under ``"std"``.

    Returns
    -------
    pandas.Series
        Indexed by the ``row`` identity like ``build_matrix``'s rows; empty if ``flavor`` has no
        rows in ``frame``.
    """
    subset = frame[frame["flavor"] == flavor]
    if subset.empty:
        return pd.Series(dtype=float)
    # the prepared row identity separates single-task and multi-task variants; fall back to the
    # plain dataset+endpoint key for an unprepared frame, collapsing seed duplicates by agg so the
    # index stays unique and aligned to the pivot
    if "row" in subset.columns:
        row = subset["row"]
    else:
        row = subset["dataset"] + " · " + subset["endpoint"]
    return pd.Series(subset[metric].to_numpy(), index=row).groupby(level=0).agg(agg)


def mae_delta_matrix(mae_matrix: pd.DataFrame, baseline_mae: pd.Series) -> pd.DataFrame:
    """Percentage change in MAE relative to the baseline, per (endpoint, flavor).

    Each cell is ``100 * (mae_flavor - mae_baseline) / mae_baseline``, so a negative value
    means the flavor's MAE is below the baseline (an improvement) and a positive value means it
    is worse. Rows where the baseline MAE is missing or zero become NaN (no defined change).

    Parameters
    ----------
    mae_matrix : pandas.DataFrame
        Per-flavor MAE matrix from ``build_matrix(frame, "mae")``.
    baseline_mae : pandas.Series
        Baseline MAE per endpoint, indexed like ``mae_matrix``'s rows (see
        ``build_reference_series``).

    Returns
    -------
    pandas.DataFrame
        Same shape as ``mae_matrix``; cells are percentage changes, NaN where undefined.
    """
    baseline = baseline_mae.reindex(mae_matrix.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 100.0 * mae_matrix.sub(baseline, axis=0).div(baseline, axis=0)
    return delta.where(np.isfinite(delta))


def mae_delta_std(
    mae_matrix: pd.DataFrame,
    mae_std_matrix: pd.DataFrame,
    baseline_mae: pd.Series,
    baseline_mae_std: pd.Series,
) -> pd.DataFrame:
    """Propagate the seed spread of both sides into the MAE %-change error bar.

    The delta cell is ``100 * (F/B - 1)`` for flavor mean MAE ``F`` and baseline mean MAE ``B``.
    Treating the flavor and baseline seed spreads as independent (the stock seeds and flavor
    seeds are separate finetune runs), first-order propagation gives
    ``sigma_delta = 100 * sqrt((sF / B)^2 + (F * sB / B^2)^2)`` for per-seed standard deviations
    ``sF`` (flavor) and ``sB`` (baseline). A cell is NaN where either spread or the baseline is
    undefined (single-seed side, or missing/zero baseline).

    Parameters
    ----------
    mae_matrix, mae_std_matrix : pandas.DataFrame
        Per-flavor mean and standard-deviation MAE matrices from ``build_matrix(frame, "mae",
        aggfunc=...)``.
    baseline_mae, baseline_mae_std : pandas.Series
        Baseline mean and standard-deviation MAE per endpoint, indexed like the matrices' rows.

    Returns
    -------
    pandas.DataFrame
        Same shape as ``mae_matrix``; cells are the propagated percentage-point standard
        deviations, NaN where undefined.
    """
    base = baseline_mae.reindex(mae_matrix.index)
    base_std = baseline_mae_std.reindex(mae_matrix.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        flavor_term = mae_std_matrix.div(base, axis=0)
        baseline_term = mae_matrix.mul(base_std, axis=0).div(base**2, axis=0)
        sigma = 100.0 * np.sqrt(flavor_term**2 + baseline_term**2)
    return sigma.where(np.isfinite(sigma))


def _per_seed_values(frame: pd.DataFrame, metric: str) -> dict[tuple[str, str], np.ndarray]:
    """Map ``(row, flavor)`` to the array of per-seed ``metric`` values behind that cell.

    After ``collapse_seed_variants`` the seed replicates share a flavor label but stay separate
    rows, so grouping by the displayed row identity and flavor recovers each cell's seed sample.
    """
    grouped = frame.groupby(["row", "flavor"])[metric].apply(lambda s: s.to_numpy(dtype=float))
    return dict(grouped)


def mae_significance_pvalues(
    matrix_frame: pd.DataFrame, frame: pd.DataFrame, baseline_flavor: str, mae_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Two-sample p-value per (endpoint, flavor) for the flavor's MAE differing from baseline.

    Each cell runs a two-sample Welch t-test (unequal variance) between the flavor's per-seed MAE
    sample and the stock baseline's per-seed MAE sample for that endpoint. The seed sets are not
    paired (the baseline and flavor seeds only partly overlap), so an unpaired test is used. A cell
    is NaN where either side has fewer than two seeds or zero variance, which the card treats as
    not significant (painted white).

    Parameters
    ----------
    matrix_frame : pandas.DataFrame
        Prepared, seed-collapsed metrics for the flavor columns (carries ``row``, ``flavor``,
        ``mae``); the same frame ``build_matrix`` pivots.
    frame : pandas.DataFrame
        Prepared, seed-collapsed metrics that include the baseline flavor's rows.
    baseline_flavor : str
        The baseline flavor label (e.g. ``chemeleon_stock``).
    mae_matrix : pandas.DataFrame
        The per-flavor MAE matrix whose index and columns the returned p-values align to.

    Returns
    -------
    pandas.DataFrame
        Same shape as ``mae_matrix``; each cell is the test p-value, NaN where undefined.
    """
    flavor_samples = _per_seed_values(matrix_frame, "mae")
    baseline_samples = _per_seed_values(frame[frame["flavor"] == baseline_flavor], "mae")
    pvalues = pd.DataFrame(np.nan, index=mae_matrix.index, columns=mae_matrix.columns)
    for row in mae_matrix.index:
        base = baseline_samples.get((row, baseline_flavor))
        if base is None or np.size(base) < 2 or np.ptp(base) == 0:
            continue
        for flavor in mae_matrix.columns:
            sample = flavor_samples.get((row, flavor))
            if sample is None or np.size(sample) < 2 or np.ptp(sample) == 0:
                continue
            pvalues.loc[row, flavor] = float(ttest_ind(sample, base, equal_var=False).pvalue)
    return pvalues


def assemble_r2_card(
    flavor_matrix: pd.DataFrame, baseline: pd.Series
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Order the R-squared card columns: baseline first (behind a spacer), then the flavors.

    Parameters
    ----------
    flavor_matrix : pandas.DataFrame
        Per-flavor R-squared matrix from ``build_matrix``.
    baseline : pandas.Series
        Stock-CheMeleon baseline per endpoint; a non-empty series becomes the first column,
        separated from the flavor block by a blank spacer column.

    Returns
    -------
    tuple of (pandas.DataFrame, list of int, list of int)
        The assembled matrix, the column indices of the blank spacer columns, and the column
        indices of the reference columns (the baseline) so the caller can bold them.
    """
    index = flavor_matrix.index
    # (name, series, is_reference, is_spacer) in final left-to-right order
    entries: list[tuple[str, pd.Series, bool, bool]] = []
    if not baseline.empty:
        entries.append((BASELINE_LABEL, baseline.reindex(index), True, False))
        entries.append((_SPACER_LEFT, pd.Series(np.nan, index=index), False, True))
    for col in flavor_matrix.columns:
        entries.append((col, flavor_matrix[col], False, False))

    matrix = pd.concat([series.rename(name) for name, series, _, _ in entries], axis=1)
    spacer_cols = [i for i, (_, _, _, is_spacer) in enumerate(entries) if is_spacer]
    ref_cols = [i for i, (_, _, is_ref, _) in enumerate(entries) if is_ref]
    return matrix, spacer_cols, ref_cols


def append_average_row(matrix: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append a blank spacer row then an AVERAGE row meaning each column over the current rows.

    The mean is taken over the endpoint rows present before the append (skipping NaN), so the
    AVERAGE row summarizes each column across all endpoints; spacer columns stay NaN.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The matrix with the two extra rows, and the row index of the AVERAGE row.
    """
    average = matrix.mean(axis=0, skipna=True).rename(AVERAGE_LABEL)
    spacer = pd.Series(np.nan, index=matrix.columns, name=_SPACER_ROW)
    out = pd.concat([matrix, spacer.to_frame().T, average.to_frame().T])
    return out, len(out) - 1


def source_groups(index: pd.Index) -> list[tuple[int, int, str]]:
    """Return contiguous ``(start, end, source)`` runs from ``"<source> · <endpoint>"`` rows.

    ``build_matrix`` orders rows by dataset, so each source's endpoints are contiguous; the
    runs drive the bold separator lines and per-group source labels on the card.
    """
    sources = [str(row).split(" · ", 1)[0] for row in index]
    groups: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(sources) + 1):
        if i == len(sources) or sources[i] != sources[start]:
            groups.append((start, i, sources[start]))
            start = i
    return groups


def _endpoint_labels(index: pd.Index) -> list[str]:
    """Strip the ``"<source> · "`` prefix from endpoint rows; keep AVERAGE and spacer labels."""
    labels = []
    for row in index:
        text = str(row)
        labels.append(text.split(" · ", 1)[1] if " · " in text else text)
    return labels


def _draw_hline(ax, y: float, n_cols: int, spacer_cols: list[int], *, linewidth: float) -> None:
    """Draw a horizontal separator at ``y`` across the data columns, broken over spacer columns.

    Splitting the rule at each blank spacer column keeps it from crossing the white reference
    gap; the segments span the real cells on either side only.
    """
    x = -0.5
    for col in sorted(spacer_cols):
        if col - 0.5 > x:
            ax.plot([x, col - 0.5], [y, y], color="black", linewidth=linewidth, zorder=3)
        x = col + 0.5
    ax.plot([x, n_cols - 0.5], [y, y], color="black", linewidth=linewidth, zorder=3)


def plot_card(
    matrix: pd.DataFrame,
    out_png: Path,
    out_csv: Path,
    *,
    cmap,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate,
    title: str,
    cbar_ticks: list[float],
    cbar_labels: list[str],
    spacer_cols: list[int],
    ref_cols: list[int],
    groups: list[tuple[int, int, str]],
    average_row: int,
    aux: pd.DataFrame | None = None,
    color_values: pd.DataFrame | None = None,
    emphasis_source: str | None = None,
) -> None:
    """Render one report-card heatmap and write its underlying matrix CSV.

    Parameters
    ----------
    matrix : pandas.DataFrame
        The fully assembled card (endpoint rows, then a spacer row and the AVERAGE row; columns
        may include blank spacer columns and reference columns). Drives the cell annotations, and
        the cell colors unless ``color_values`` is given.
    out_png, out_csv : pathlib.Path
        Image and matrix-CSV output paths.
    cmap : matplotlib colormap
        Colormap for the cell values.
    norm : matplotlib.colors.Normalize, optional
        Color normalization; when omitted, ``vmin``/``vmax`` bound a linear scale.
    vmin, vmax : float, optional
        Linear color bounds used when ``norm`` is None.
    annotate : callable
        Maps a finite cell value and its auxiliary value to an annotation string,
        ``annotate(value, aux)``; ``aux`` is NaN where none is defined. The auxiliary value is the
        seed standard deviation on the R² card and the significance p-value on the MAE-delta card.
    title : str
        Figure title.
    cbar_ticks, cbar_labels : list
        Colorbar tick positions and their labels.
    spacer_cols : list of int
        Column indices of blank spacers to paint white and bound with divider lines.
    ref_cols : list of int
        Column indices whose x labels are bold (reference columns).
    groups : list of (int, int, str)
        Source-group runs over the endpoint rows (see ``source_groups``); each draws a bold
        boundary line and a bold source label.
    average_row : int
        Row index of the AVERAGE row, separated from the endpoint block by a bold line.
    aux : pandas.DataFrame, optional
        Per-cell auxiliary value aligned to ``matrix`` (same index and columns); passed as the
        second argument to ``annotate``. When omitted, ``annotate`` receives NaN for every cell.
    color_values : pandas.DataFrame, optional
        Per-cell value driving the color, aligned to ``matrix``; used when the color should differ
        from the annotated value (e.g. the MAE-delta card paints a non-significant cell white by
        setting its color value to the norm center while still annotating the real change). When
        omitted, the annotated ``matrix`` drives the color.
    emphasis_source : str, optional
        Source-group name (see ``groups``) whose last endpoint gets a thicker separator line
        directly after it, over and above the normal group boundary. When omitted, no group is
        emphasized.
    """
    values = matrix.to_numpy(dtype=float)
    # align the auxiliary matrix to the value matrix so the annotation loop indexes them in lockstep
    aux_values = (
        aux.reindex(index=matrix.index, columns=matrix.columns).to_numpy(dtype=float)
        if aux is not None
        else np.full_like(values, np.nan)
    )
    # the color layer defaults to the annotated values, but a card may drive color from a separate
    # matrix (e.g. significance-gated deltas) while still annotating the true value
    color_layer = (
        color_values.reindex(index=matrix.index, columns=matrix.columns).to_numpy(dtype=float)
        if color_values is not None
        else values
    )
    n_rows, n_cols = values.shape
    cmap = cmap.copy()
    cmap.set_bad("lightgrey")  # missing (flavor, endpoint) cells

    fig, ax = plt.subplots(
        figsize=(1.15 * n_cols + 3.5, 0.42 * n_rows + 2.5), constrained_layout=True
    )
    imshow_kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.imshow(np.ma.masked_invalid(color_layer), aspect="auto", cmap=cmap, **imshow_kwargs)

    # the blank spacer row sits directly above the AVERAGE row; its white band is where the
    # vertical divider lines break so no rule crosses empty space
    spacer_row_top, spacer_row_bottom = average_row - 1.5, average_row - 0.5

    # paint the spacer columns and the spacer row white (distinct from missing-data lightgrey);
    # bound each spacer column with divider lines split at the white spacer row so no line
    # crosses that empty band
    for col in spacer_cols:
        ax.axvspan(col - 0.5, col + 0.5, color="white", zorder=2)
        for x in (col - 0.5, col + 0.5):
            ax.plot([x, x], [-0.5, spacer_row_top], color="black", linewidth=1.2, zorder=3)
            ax.plot(
                [x, x], [spacer_row_bottom, n_rows - 0.5], color="black", linewidth=1.2, zorder=3
            )
    ax.axhspan(spacer_row_top, spacer_row_bottom, color="white", zorder=2)

    # bold source-group separators (each broken across the white spacer columns) and per-group
    # labels on the left margin; the label x is a fixed absolute gap left of the grid, so a narrow
    # card clears its row labels as well as the wide one does (see GROUP_LABEL_INCHES)
    group_label_x = -GROUP_LABEL_INCHES / (1.15 * n_cols)
    for start, end, source in groups:
        if start > 0:
            _draw_hline(ax, start - 0.5, n_cols, spacer_cols, linewidth=2.2)
        ax.text(
            group_label_x, (start + end - 1) / 2.0, source, transform=ax.get_yaxis_transform(),
            rotation=90, ha="center", va="center", fontsize=FONT_AXIS, fontweight="bold",
        )
    # emphasis line directly after the requested source group's last endpoint, matching the
    # weight of the other group separators
    if emphasis_source is not None:
        for _, end, source in groups:
            if source == emphasis_source:
                _draw_hline(ax, end - 0.5, n_cols, spacer_cols, linewidth=2.2)
    # bold line above the AVERAGE row
    _draw_hline(ax, average_row - 0.5, n_cols, spacer_cols, linewidth=2.2)

    # pin the view to the imshow extent so the added line segments do not re-margin the axes
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # place ticks only on real columns, so a blank spacer column carries no tick mark or label
    x_positions = [i for i in range(n_cols) if i not in spacer_cols]
    ax.set_xticks(
        x_positions, labels=[str(matrix.columns[i]) for i in x_positions],
        rotation=45, ha="left", fontsize=FONT_AXIS,
    )
    for pos, label in zip(x_positions, ax.get_xticklabels()):
        if pos in ref_cols:
            label.set_fontweight("bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # likewise skip the blank spacer row (directly above AVERAGE) so it carries no tick mark
    spacer_row_index = average_row - 1
    endpoint_labels = _endpoint_labels(matrix.index)
    y_positions = [i for i in range(n_rows) if i != spacer_row_index]
    ax.set_yticks(
        y_positions, labels=[endpoint_labels[i] for i in y_positions], fontsize=FONT_YTICK
    )
    for pos, label in zip(y_positions, ax.get_yticklabels()):
        if pos == average_row:
            label.set_fontweight("bold")

    # annotate each cell with its value and, where defined, its auxiliary (error bar or p-value)
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    j, i, annotate(value, aux_values[i, j]),
                    ha="center", va="center", fontsize=FONT_CELL, color="black",
                )

    ax.set_title(title, fontsize=FONT_TITLE, pad=28)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels, fontsize=FONT_CBAR)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    matrix.to_csv(out_csv)
    logger.info("wrote %s and %s", out_png, out_csv)


def _blank_summary_rows(aux: pd.DataFrame) -> pd.DataFrame:
    """Null the auxiliary annotation (error bar or p-value) on the spacer and AVERAGE rows.

    ``append_average_row`` would otherwise mean the per-cell auxiliary values into the AVERAGE row,
    conflating a per-cell quantity (seed spread, or a per-cell significance test) with the
    endpoint-to-endpoint AVERAGE. The AVERAGE row shows a bare mean, so its auxiliary is nulled
    along with the spacer row.
    """
    aux = aux.copy()
    aux.iloc[-2:, :] = np.nan
    return aux


def render_r2_card(
    matrix_frame: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_flavor: str,
    out_png: Path,
    *,
    columns: list[str] | None = None,
    title_prefix: str = "Report card",
) -> None:
    """Assemble and render the R-squared card (red = 0, green = 1) with the baseline column.

    Parameters
    ----------
    matrix_frame : pandas.DataFrame
        Prepared, seed-collapsed metrics for the card columns (carries ``row``, ``flavor``, and
        the metric columns).
    frame : pandas.DataFrame
        Prepared, seed-collapsed metrics that include the baseline flavor's rows.
    baseline_flavor : str
        Baseline flavor label for the reference column (e.g. ``chemeleon_stock``).
    out_png : pathlib.Path
        Image output path; the matrix CSV is written alongside it.
    columns : list of str, optional
        Column order for the card, values of the ``flavor`` field. Defaults to the flavor
        registry order; the ablation report passes its bare ablation names.
    title_prefix : str, optional
        Leading text of the figure title, so a reused card can read e.g. ``Ablation report
        card`` instead of the default ``Report card``.
    """
    flavor_r2 = build_matrix(matrix_frame, "r2", columns=columns)
    flavor_r2_std = build_matrix(matrix_frame, "r2", columns=columns, aggfunc="std")
    baseline = build_reference_series(frame, baseline_flavor, "r2")
    baseline_std = build_reference_series(frame, baseline_flavor, "r2", agg="std")
    matrix, spacer_cols, ref_cols = assemble_r2_card(flavor_r2, baseline)
    std, _, _ = assemble_r2_card(flavor_r2_std, baseline_std)
    groups = source_groups(flavor_r2.index)
    matrix, average_row = append_average_row(matrix)
    std, _ = append_average_row(std)
    plot_card(
        matrix, out_png, out_png.with_suffix(".csv"),
        cmap=plt.get_cmap("RdYlGn"), vmin=0.0, vmax=1.0,
        annotate=lambda v, s: f"{v:.3f}" if not np.isfinite(s) else f"{v:.3f}\n±{s:.3f}",
        title=f"{title_prefix}: R² (red = 0, green = 1; ± is the seed standard deviation)",
        cbar_ticks=[0.0, 0.5, 1.0], cbar_labels=["0.0", "0.5", "1.0"],
        spacer_cols=spacer_cols, ref_cols=ref_cols, groups=groups, average_row=average_row,
        aux=_blank_summary_rows(std), emphasis_source=EMPHASIS_SOURCE,
    )


def _format_pvalue(p: float) -> str:
    """Render a significance p-value compactly for a cell annotation."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "p<.001"
    return f"p={p:.3f}"


def render_mae_delta_card(
    matrix_frame: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_flavor: str,
    out_png: Path,
    *,
    columns: list[str] | None = None,
    title_prefix: str = "Report card",
) -> None:
    """Render the MAE %-change card, coloring only cells that differ significantly from baseline.

    Each cell's color still encodes the percentage change in MAE (green better, red worse), but a
    cell whose per-seed MAE does not differ significantly from the baseline's (two-sample Welch
    t-test p above ``SIGNIFICANCE_ALPHA``) is painted white regardless of the change, so the card
    highlights only differences the seed spread supports. Every cell is annotated with its change
    and its p-value.

    Parameters
    ----------
    matrix_frame, frame, baseline_flavor, out_png
        As in :func:`render_r2_card`.
    columns : list of str, optional
        Column order for the card. Defaults to the flavor registry order; the ablation report
        passes its bare ablation names.
    title_prefix : str, optional
        Leading text of the figure title (default ``Report card``).
    """
    baseline_mae = build_reference_series(frame, baseline_flavor, "mae")
    if baseline_mae.empty:
        logger.warning(
            "no %s MAE in the metrics; skipping the MAE-delta card", baseline_flavor
        )
        return
    mae = build_matrix(matrix_frame, "mae", columns=columns)
    delta = mae_delta_matrix(mae, baseline_mae)
    pvalues = mae_significance_pvalues(matrix_frame, frame, baseline_flavor, mae)
    # paint a non-significant (or untestable) cell white by driving its color to the norm center,
    # while the annotation still shows the real change; leave a missing delta as NaN (grey)
    significant = pvalues.le(SIGNIFICANCE_ALPHA)
    color_delta = delta.mask((~significant) & delta.notna(), 0.0)

    groups = source_groups(delta.index)
    matrix, average_row = append_average_row(delta)
    color_matrix, _ = append_average_row(color_delta)
    # the AVERAGE row summarizes across endpoints, not a per-cell test, so color it by its true
    # mean change rather than the significance-gated values
    color_matrix.iloc[-1] = matrix.iloc[-1].to_numpy()
    pvalue_matrix, _ = append_average_row(pvalues)

    finite = matrix.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    # symmetric scale centered on 0% (the baseline), capped at DELTA_EXTENT_CAP so a few large
    # outliers do not wash out the rest; changes beyond the cap saturate at the end color while
    # their annotation still shows the true value
    observed = max(float(np.abs(finite).max()) if finite.size else 1.0, 1e-6)
    extent = min(observed, DELTA_EXTENT_CAP)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-extent, vmax=extent)
    plot_card(
        matrix, out_png, out_png.with_suffix(".csv"),
        cmap=_DELTA_CMAP, norm=norm,
        annotate=lambda v, p: f"{v:+.0f}%\n{_format_pvalue(p)}".rstrip(),
        title=f"{title_prefix}: MAE % change vs chemeleon baseline (green = lower MAE / better, "
        f"red = worse; white where p > {SIGNIFICANCE_ALPHA:g}, two-sample Welch t-test on the "
        "seeds)",
        cbar_ticks=[-extent, 0.0, extent],
        cbar_labels=[f"-{extent:.0f}%", "0% (baseline or not significant)", f"+{extent:.0f}%"],
        spacer_cols=[], ref_cols=[], groups=groups, average_row=average_row,
        aux=_blank_summary_rows(pvalue_matrix), color_values=color_matrix,
        emphasis_source=EMPHASIS_SOURCE,
    )


def main() -> None:
    """Build and save both report cards (R-squared and MAE % change) for the requested setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV, help="tidy metrics CSV")
    parser.add_argument(
        "--out-dir", type=Path, default=PLOTS_DIR, help="directory for the two card PNGs"
    )
    parser.add_argument(
        "--baseline-flavor", default="chemeleon_stock",
        help="flavor label for the stock-CheMeleon reference / MAE-delta baseline (see "
        "slurm/run_stock_baseline.sh); the R² baseline column and the whole MAE-delta card are "
        "skipped if it is absent from --metrics-csv",
    )
    parser.add_argument(
        "--lr-mode", choices=("reduced", "unlocked"), default=None,
        help="render a learning-rate-experiment setup: filter --metrics-csv (typically "
        "results/lr_metrics.csv) to this protocol's lr_<mode>__<flavor> rows and strip the "
        "prefix so the columns match the flavor registry. The references still read from the "
        "full frame, so pass the protocol's stock baseline via --baseline-flavor "
        "(chemeleon_stock_<mode>). The frozen setup needs no filter (bare-flavor metrics.csv)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate first")
    # re-derive the dataset from the recipe and build the disambiguated row identity once on the
    # full frame, so build_matrix and the reference series share one consistent set of row labels
    frame = prepare_rows(collapse_seed_variants(pd.read_csv(args.metrics_csv)))
    # for a reduced/unlocked setup, keep only that protocol's rows as bare-flavor columns; the
    # references still read from the full frame so the mode's stock baseline survives
    matrix_frame = frame
    if args.lr_mode is not None:
        matrix_frame = filter_lr_mode(frame, args.lr_mode)
        if matrix_frame.empty:
            raise SystemExit(
                f"--lr-mode {args.lr_mode} matched no lr_{args.lr_mode}__ rows in "
                f"{args.metrics_csv}; point it at results/lr_metrics.csv"
            )

    suffix = f"_{args.lr_mode}" if args.lr_mode else ""
    render_r2_card(
        matrix_frame, frame, args.baseline_flavor,
        args.out_dir / f"report_card_r2{suffix}.png",
    )
    render_mae_delta_card(
        matrix_frame, frame, args.baseline_flavor,
        args.out_dir / f"report_card_mae_delta{suffix}.png",
    )


if __name__ == "__main__":
    main()
