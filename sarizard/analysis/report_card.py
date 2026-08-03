"""Report-card heatmaps: endpoints (rows) by foundation flavors (columns).

Two cards are rendered per setup from the tidy metrics CSV written by ``analysis.evaluate``:

- an R-squared card colored on a fixed red-to-green scale (red = 0, green = 1), with the
  stock-CheMeleon baseline as the first column (separated from the flavor block by a heavy
  vertical rule) and a final AVERAGE row that means each column across all endpoints;
- a delta card whose cells are the percentage change in MAE relative to the stock-CheMeleon
  baseline (green where a flavor's MAE beats the baseline, red where it is worse), flavor columns
  only, with the same AVERAGE row. A cell is painted white unless the flavor's per-seed MAE
  differs significantly from the baseline's (Dunnett's test, p at or below
  ``SIGNIFICANCE_ALPHA``), so only differences the seed spread supports carry color. Each
  endpoint row is one comparison family: its flavors are all measured against the same baseline,
  so they are corrected together rather than tested one cell at a time. Its AVERAGE row is gated
  the same way, off a separate Dunnett test on each flavor's per-seed mean change across all the
  card's endpoints, so a column that only looks better overall is painted white too.

The R² card annotates every endpoint cell (flavors and the multi-seed baseline column) with a
``±`` seed standard deviation under its value, and shows a bare mean on its AVERAGE row; the
delta card annotates each cell with its change and the test p-value, its AVERAGE row included.

Both cards group the endpoint rows by their source dataset (asap, chembl, expansionrx, ...),
bracketing each group in a left-margin box labelled with the source's display name and its
split strategy (the single-endpoint hERG and PXR groups are the exception: their box is one row
tall, too short for a legible rotated label, so their name and split lead the row label instead).
The styling follows the sibling information-gain-metric repo's heatmaps so the two projects'
figures read as one family.

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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# match the sibling information-gain-metric repo's heatmap styling, so the two projects' figures
# read as one family: a clean white ground with no axis spines, cells separated by white gridlines
# rather than drawn borders, and the dataset grouping carried by left-margin boxes
plt.style.use("seaborn-v0_8-white")

from sarizard.analysis import card_html  # noqa: E402
from sarizard.analysis.metrics_spec import DATASETS, dataset_of  # noqa: E402
from sarizard.analysis.multicomp import MIN_GROUP_SIZE, dunnett_pvalues  # noqa: E402
from sarizard.analysis.paths import METRICS_CSV, PLOTS_DIR, parse_seed_variant  # noqa: E402
from sarizard.pretraining.flavors import flavor_names  # noqa: E402

logger = logging.getLogger(__name__)

# significance threshold for the MAE-delta card: a flavor whose per-seed MAE does not differ from
# the baseline's at this level (Dunnett's test, family-wise within the endpoint row, p above the
# threshold) is painted white, so only differences the seed spread supports carry color
SIGNIFICANCE_ALPHA = 0.05

# cap the MAE-delta diverging color scale at this magnitude (percentage points) in both
# directions, so a few large outliers do not wash out the scale; larger changes saturate at the
# end color while their annotation still shows the true value
DELTA_EXTENT_CAP = 25.0

# column label for the reference column, and row label for the across-endpoint summary
BASELINE_LABEL = "chemeleon\nbaseline"
AVERAGE_LABEL = "AVERAGE"

# pivot key standing in for a missing seed number, so an unseeded label (the legacy single-run
# baseline) still groups as one replicate instead of being dropped by the pivot's NaN index
_UNSEEDED_KEY = -1

# width of the rule separating the reference column from the block it is compared against,
# heavy enough to read as a division of the card rather than as another cell border
_DIVIDER_LINEWIDTH = 3.0

# green-white-red diverging map for the MAE-delta card: green = MAE below baseline (better),
# white = no change, red = MAE above baseline (worse)
_DELTA_CMAP = LinearSegmentedColormap.from_list("mae_delta", ["#1a9850", "#ffffff", "#d73027"])

# report-card font sizes (points), following the sibling repo's heatmap scale but stepped up
# across the board for legibility: these cards print at roughly 22 x 20 inches, where that
# repo's scale reads small. FONT_CELL stays below its 13 pt default because these cells carry
# two lines (a value over its error bar or p-value) where that repo's carry one, which is the
# adjustment its renderer anticipates
FONT_AXIS = 11  # x tick labels, bold, and the per-group source labels
FONT_YTICK = 10  # endpoint row labels
FONT_CELL = 10  # per-cell value and error-bar/p-value annotation
FONT_CBAR = 9  # colorbar tick labels

# cell grid geometry (inches), and the fixed margins the figure size adds around it: the
# left-margin group boxes, the endpoint row labels, the colorbar, and the height taken by the
# rotated column labels. Sizing from the grid plus fixed margins (rather than the grid
# plus one lump) keeps the cells the same size on a narrow ablation card and a wide flavor one.
# The per-column width is below the sibling repo's 1.55 because these cards carry 15 to 17
# columns against its handful of metric columns, which at 1.55 would run past 30 inches wide.
# The three text-carrying margins track the font scale above, so a font bump does not eat into
# the space the labels it enlarges are allotted
CELL_ROW_INCHES = 0.55
CELL_COL_INCHES = 1.05
DATASET_LABEL_INCHES = 1.3
TICK_LABEL_INCHES = 2.2
COLORBAR_INCHES = 1.0
FIG_HEIGHT_PAD_INCHES = 2.2

# correcting the figure size so the laid-out grid actually hits the cell geometry above (see
# _fit_cells): how many measure-and-grow passes to run, and how close counts as done
_FIT_PASSES = 3
_FIT_TOLERANCE_INCHES = 0.01

# how finely the HTML legend samples the colormap (see card_html)
_LEGEND_SAMPLES = 48

# output resolution, matching the sibling repo's 600. These cards are roughly 22 x 20 inches, so
# that lands a flavor card near 14000 px wide and 150 megapixels: heavy to open, but the cell
# annotations and the small group-box labels stay crisp when the card is zoomed or printed
_DPI = 600

# source group whose last endpoint gets a thicker separator line directly after it
EMPHASIS_SOURCE = "pxr"

# Display names and split strategy per dataset group, for the left-margin group boxes. The names
# follow the sibling information-gain-metric repo's analyze.py, except for the three
# ChEMBL-derived groups: that repo names them "ChEMBL" and "ChEMBL 37" (the latter shared by the
# CYP and hERG groups), which reads as though only one of the three came out of ChEMBL and puts
# the same label on two different assays. All three are ChEMBL extracts, so they are separated by
# the assay they measure instead of by a release number. The hERG group needs no qualifier: it is
# one row, and that row already reads "hERG pIC50". The split types were re-derived from this
# repo's own recipes rather than copied on trust: a template carrying train_resource/test_resource
# uses the dataset's predefined split, one without uses anvil's inline ClusterSplitter.
_DATASET_DISPLAY: dict[str, str] = {
    "asap": "ASAP",
    "asap_potency": "ASAP",
    "biogen": "Biogen",
    "chembl": "ChEMBL CLint",
    "expansionrx": "ExpRx",
    "openadmet_cyp": "ChEMBL CYP",
    "herg": "ChEMBL",
    "pxr": "Octant",
}
_SPLIT_TYPE: dict[str, str] = {
    "asap": "predefined",
    "asap_potency": "predefined",
    "biogen": "cluster",
    "chembl": "cluster",
    "expansionrx": "predefined",
    "openadmet_cyp": "cluster",
    "herg": "cluster",
    "pxr": "cluster",
}

# sources whose name and split strategy move out of the group box and onto the row label. Each
# is a one-endpoint group, so its bracket is one row tall and the rotated label inside it shrinks
# past legibility; set horizontally in front of the endpoint it reads at the same size as every
# other row label. Their brackets stay, carrying no text
_INLINE_SOURCES = frozenset({"herg", "pxr"})

# Endpoint column -> short row label, also from the sibling repo, so a row reads "CLint HLM"
# rather than "LOG_CLint_HLM". A disambiguating "(<recipe>)" suffix survives the mapping
_COL_DISPLAY: dict[str, str] = {
    "LOG_CLint_HLM": "CLint HLM",
    "LOG_CLint_MLM": "CLint MLM",
    "LOG_CLint_RLM": "CLint RLM",
    "LOG_MDR1": "MDR1",
    "LogD": "LogD",
    "LOG_KSOL": "KSOL",
    "LOG_SOL": "SOL",
    "LOG_CACO2_PAPP": "Caco2 Papp",
    "LOG_CACO2_EFFLUX": "Caco2 Efflux",
    "LOG_MPPB": "MPPB",
    "LOG_MBPB": "MBPB",
    "pIC50_MERS_Mpro": "MERS pIC50",
    "pIC50_SARS2_Mpro": "SARS2 pIC50",
    "OPENADMET_LOGAC50_cyp3a4": "CYP3A4 IC50",
    "OPENADMET_LOGAC50_cyp2c9": "CYP2C9 IC50",
    "OPENADMET_LOGAC50_cyp2d6": "CYP2D6 IC50",
    "OPENADMET_LOGAC50_cyp1a2": "CYP1A2 IC50",
    "pchembl_value_mean": "hERG pIC50",
    "PXR_pEC50": "PXR pEC50",
}

# slack between the row labels and the group boxes' right edge, in inches, so a label the
# TICK_LABEL_INCHES allowance underestimates does not butt straight into the bracket
_BOX_GAP_INCHES = 0.1

# cell text flips to white once the cell color is this far from the colormap's midpoint, where
# the RdYlGn and delta ramps are dark enough at both ends to swallow black text
_TEXT_FLIP_DISTANCE = 0.36

# fitting a rotated group-box label to its box height: the average width of a bold glyph in em,
# and the fraction of the box the label is allowed to fill so adjacent short groups keep a gap
_BOLD_EM_WIDTH = 0.68
_GROUP_LABEL_SLACK = 0.85


def collapse_seed_variants(frame: pd.DataFrame, column: str = "flavor") -> pd.DataFrame:
    """Map ``<base>__s<seed>`` labels in ``column`` back to their base label.

    Seeds are run as separate variants (own foundation, recipes, results); collapsing the
    labels lets ``build_matrix`` average the seed replicates into one cell per (endpoint, base)
    via its ``aggfunc="mean"`` pivot. Plain labels (no seed suffix) pass through unchanged. The
    base may itself carry a namespace prefix (``ablation_<name>``, ``lr_<mode>__<flavor>``).

    The stripped seed is kept in a ``seed`` column (NaN for an unseeded label) so a caller that
    needs to line one seed's rows up across endpoints, as the AVERAGE-row test does, can recover
    the replicate identity the collapsed label no longer carries.
    """
    frame = frame.copy()
    parsed = frame[column].map(parse_seed_variant)
    frame[column] = parsed.map(lambda base_seed: base_seed[0])
    frame["seed"] = parsed.map(lambda base_seed: base_seed[1])
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
    """Family-wise p-value per (endpoint, flavor) for the flavor's MAE differing from baseline.

    One endpoint row is one comparison family: its flavors are all measured against the same stock
    baseline on the same molecules, so they are corrected together with Dunnett's test rather than
    tested pairwise. Correcting per row and not across the whole card is deliberate, since each
    endpoint asks its own question; error across the card's rows is therefore not controlled.

    This replaced a per-cell unpaired Welch t-test, which ran one uncorrected test per cell and so
    let the card's false-positive count grow with the number of flavors shown. The seed sets are
    still unpaired (the baseline and flavor seeds only partly overlap, and the seed-randomized
    splits resample per seed), which Dunnett assumes; what changed is the multiplicity, not the
    pairing.

    A cell is NaN where the comparison is undefined (fewer than ``MIN_GROUP_SIZE`` seeds on either
    side, or no residual variance anywhere in the row), which the card treats as not significant
    and paints white.

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
        if base is None or np.size(base) < MIN_GROUP_SIZE:
            continue
        # the family is the flavors actually shown in this row, so a standalone card (--columns)
        # pays a correction sized to its own column set rather than the full registry
        samples = {
            flavor: sample
            for flavor in mae_matrix.columns
            if (sample := flavor_samples.get((row, flavor))) is not None
        }
        for flavor, pvalue in dunnett_pvalues(samples, base).items():
            pvalues.loc[row, flavor] = pvalue
    return pvalues


def assemble_r2_card(
    flavor_matrix: pd.DataFrame, baseline: pd.Series
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Order the R-squared card columns: baseline first, then the flavors behind a heavy rule.

    Parameters
    ----------
    flavor_matrix : pandas.DataFrame
        Per-flavor R-squared matrix from ``build_matrix``.
    baseline : pandas.Series
        Stock-CheMeleon baseline per endpoint; a non-empty series becomes the first column,
        divided from the flavor block by a heavy vertical rule rather than a gap, so the two
        blocks read as one grid with a boundary in it.

    Returns
    -------
    tuple of (pandas.DataFrame, list of int, list of int)
        The assembled matrix, the column indices whose left edge carries a divider rule, and
        the column indices of the reference columns (the baseline) so the caller can bold them.
    """
    index = flavor_matrix.index
    # (name, series, is_reference) in final left-to-right order
    entries: list[tuple[str, pd.Series, bool]] = []
    if not baseline.empty:
        entries.append((BASELINE_LABEL, baseline.reindex(index), True))
    for col in flavor_matrix.columns:
        entries.append((col, flavor_matrix[col], False))

    matrix = pd.concat([series.rename(name) for name, series, _ in entries], axis=1)
    ref_cols = [i for i, (_, _, is_ref) in enumerate(entries) if is_ref]
    # the rule sits at the left edge of the first non-reference column, so it divides the
    # baseline from the block measured against it
    divider_cols = [] if baseline.empty else [1]
    return matrix, divider_cols, ref_cols


def append_average_row(matrix: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append an AVERAGE row meaning each column over the current rows.

    The mean is taken over the endpoint rows present before the append (skipping NaN), so the
    AVERAGE row summarizes each column across all endpoints. The row follows the endpoint block
    directly: the group bracket closing above it is what marks it off.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The matrix with the extra row, and the row index of the AVERAGE row.
    """
    average = matrix.mean(axis=0, skipna=True).rename(AVERAGE_LABEL)
    out = pd.concat([matrix, average.to_frame().T])
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


def _humanize(label: str) -> str:
    """Space out the underscores in a tick label.

    Column names and unmapped endpoint names arrive in their on-disk snake_case form
    (``osmordred_pca80``, ``chembl_clint_hlm_st``), which reads as an identifier rather than a
    label; the card shows them as words. Only tick labels go through this, so the underlying
    matrix CSV keeps the original names.
    """
    return label.replace("_", " ")


def _endpoint_labels(index: pd.Index) -> list[str]:
    """Row tick labels: the source prefix dropped and the endpoint given its short display name.

    The ``"<source> · "`` prefix is redundant once the left-margin group boxes name the source,
    and the endpoint itself is shortened via ``_COL_DISPLAY`` (``LOG_CLint_HLM`` reads
    ``CLint HLM``). A disambiguating ``" (<recipe>)"`` suffix is preserved, since it is the only
    thing separating two rows measuring the same endpoint. An endpoint missing from the display
    map falls through rather than being dropped. Underscores become spaces throughout, so a
    fallthrough name and a recipe suffix read as words. The AVERAGE label passes through.

    A source in :data:`_INLINE_SOURCES` is the exception: its name and split strategy lead the
    row label instead of sitting in the group box, so a one-row group's source stays readable.
    The endpoint keeps the end of the string, where the right-aligned labels meet the grid.
    """
    labels = []
    for row in index:
        text = str(row)
        source, _, endpoint = text.partition(" · ")
        if not endpoint:
            source, endpoint = "", text
        name, sep, suffix = endpoint.partition(" (")
        label = _humanize(_COL_DISPLAY.get(name, name) + sep + suffix)
        if source in _INLINE_SOURCES:
            label = f"{_group_label(source, inline=True)} {label}"
        labels.append(label)
    return labels


def _draw_hline(ax, y: float, n_cols: int, *, linewidth: float) -> None:
    """Draw a horizontal separator at ``y`` across the full width of the data columns."""
    ax.plot([-0.5, n_cols - 0.5], [y, y], color="black", linewidth=linewidth, zorder=3)


def _merge_by_display(groups: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Fuse adjacent source runs that share a display name into one box.

    Two source keys can map to the same name (``asap`` and ``asap_potency`` are both ASAP). Where
    such runs are adjacent they are one dataset as far as the reader is concerned, so they get one
    bracket rather than two identically labelled boxes stacked on each other.
    """
    merged: list[tuple[int, int, str]] = []
    for start, end, source in groups:
        display = _DATASET_DISPLAY.get(source, source)
        if merged and _DATASET_DISPLAY.get(merged[-1][2], merged[-1][2]) == display:
            merged[-1] = (merged[-1][0], end, merged[-1][2])
            continue
        merged.append((start, end, source))
    return merged


def _group_label(source: str, *, inline: bool = False) -> str:
    """Return a source's display name with its split strategy, for the group box or a row label.

    The group box stacks the two on separate lines and reads them rotated; a source listed in
    :data:`_INLINE_SOURCES` instead gets an empty box and, with ``inline``, a one-line form that
    :func:`_endpoint_labels` sets horizontally in front of its endpoint.
    """
    split = f"({_SPLIT_TYPE.get(source, 'cluster')})"
    name = _DATASET_DISPLAY.get(source, source)
    if source in _INLINE_SOURCES:
        return f"{name} {split}" if inline else ""
    return f"{name}\n{split}"


def _group_label_fontsize(label: str, n_rows: int) -> float:
    """Shrink a group-box label until its longest line fits the box height.

    The label is rotated upright, so its length runs along the rows and a one- or two-row group
    has very little room. Bold glyphs average around 0.68 em, and the fit is left a little slack
    on top of that, so a single-row group's label stays inside its own bracket instead of running
    into its neighbours'. Never grows past ``FONT_AXIS``.
    """
    longest = max((len(line) for line in label.splitlines()), default=1)
    available_points = n_rows * CELL_ROW_INCHES * 72.0 * _GROUP_LABEL_SLACK
    return max(4.5, min(float(FONT_AXIS), available_points / (_BOLD_EM_WIDTH * longest)))


def _group_box_bounds(n_cols: int) -> tuple[float, float]:
    """Return the group boxes' ``(left, right)`` x bounds in axes fraction, for ``n_cols`` columns.

    The boxes sit outside the grid, past the row labels, so their natural units are the inches
    the figure-size rule allots to each margin. Expressing them as a fixed axes fraction instead
    would put them a different physical distance out on every card, and on a narrow one (the
    external-foundation cards carry five columns against the flavor cards' seventeen) that
    distance collapses to less than the row labels need and the labels run into the brackets.
    The grid's width is what the figure size gives it, ``CELL_COL_INCHES`` per column.
    """
    grid_inches = CELL_COL_INCHES * n_cols
    right = -(TICK_LABEL_INCHES + _BOX_GAP_INCHES) / grid_inches
    return right - DATASET_LABEL_INCHES / grid_inches, right


def _draw_group_boxes(
    ax,
    groups: list[tuple[int, int, str]],
    n_cols: int,
    average_row: int,
) -> None:
    """Bracket each source's rows with a left-margin box carrying its name and split strategy.

    One box per contiguous source run: a rule along the top of the run reaching from the box's
    left edge across the grid, a vertical closing the box on the left, a bottom rule on the last
    run only (each run's top rule closes the one above it), and the source's display name over
    its split strategy set vertically in the margin. An :data:`_INLINE_SOURCES` run gets the
    bracket without the text, its name having moved onto the row label.

    The bracket column then carries on past the last run and around the AVERAGE row, so the
    summary label sits inside the same column as the endpoint labels it summarizes rather than
    floating below where the column stops.

    Each rule runs the full width of the grid, crossing the divider that separates the baseline
    column from the block compared against it.
    """
    # x in axes fraction, y in data coordinates, so the boxes track the rows while sitting at a
    # fixed distance from the grid regardless of how many columns the card has
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    box_x_l, box_x_r = _group_box_bounds(n_cols)
    groups = _merge_by_display(groups)
    for index, (start, end, source) in enumerate(groups):
        y_top, y_bottom = start - 0.5, end - 0.5
        for y in (y_top, y_bottom) if index == len(groups) - 1 else (y_top,):
            ax.plot([box_x_l, 0.0], [y, y], transform=trans, color="black", lw=1.0, clip_on=False)
            _draw_hline(ax, y, n_cols, linewidth=1.0)
        ax.plot(
            [box_x_l, box_x_l],
            [y_top, y_bottom],
            transform=trans,
            color="black",
            lw=1.0,
            clip_on=False,
        )
        # an inline source's bracket is drawn but left empty; its name rides the row label
        label = _group_label(source)
        if not label:
            continue
        ax.text(
            (box_x_l + box_x_r) / 2,
            (start + end - 1) / 2.0,
            label,
            transform=trans,
            ha="center",
            va="center",
            fontsize=_group_label_fontsize(label, end - start),
            fontweight="bold",
            rotation=90,
            clip_on=False,
        )

    # carry the column down past the last group and around AVERAGE: the vertical continues
    # uninterrupted and a rule closes under the summary
    if groups:
        ax.plot(
            [box_x_l, box_x_l],
            [groups[-1][1] - 0.5, average_row + 0.5],
            transform=trans,
            color="black",
            lw=1.0,
            clip_on=False,
        )
        ax.plot(
            [box_x_l, 0.0],
            [average_row + 0.5, average_row + 0.5],
            transform=trans,
            color="black",
            lw=1.0,
            clip_on=False,
        )
        _draw_hline(ax, average_row + 0.5, n_cols, linewidth=1.0)


def _html_card(
    matrix: pd.DataFrame,
    *,
    cell_text: list[list[str]],
    cell_color: list[list[str]],
    cell_light: list[list[bool]],
    groups: list[tuple[int, int, str]],
    divider_cols: list[int],
    average_row: int,
    emphasis_source: str | None,
    norm,
    cmap,
    cbar_ticks: list[float],
    cbar_labels: list[str],
    title: str,
) -> card_html.HtmlCard:
    """Translate a drawn card into the resolved form :mod:`card_html` renders.

    Takes the cell strings, colors, and contrast flips the PNG just used, and adds the pieces
    the table needs in its own terms: the group runs merged and labelled as the left-margin
    boxes are, and the colorbar flattened into gradient stops. Sampling the ramp through the
    same norm keeps a diverging card's off-center midpoint in the right place on the legend.
    """
    merged = [
        (start, end, _group_label(source)) for start, end, source in _merge_by_display(groups)
    ]
    emphasis_rows = [
        end - 1 for _, end, source in groups if emphasis_source and source == emphasis_source
    ]

    # sample the ramp in data space and position each stop by where the norm puts it, so a
    # TwoSlopeNorm's center lands on the legend where it lands on the cells
    low, high = float(norm.vmin), float(norm.vmax)
    samples = np.linspace(low, high, _LEGEND_SAMPLES)
    legend_stops = [(float(norm(v)), to_hex(cmap(norm(v)))) for v in samples]
    legend_ticks = [
        (float(norm(tick)), label)
        for tick, label in zip(cbar_ticks, cbar_labels, strict=True)
        if low <= tick <= high
    ]

    return card_html.HtmlCard(
        row_labels=_endpoint_labels(matrix.index),
        col_labels=[_humanize(str(column)) for column in matrix.columns],
        text=cell_text,
        color=cell_color,
        light_text=cell_light,
        groups=merged,
        divider_cols=divider_cols,
        average_row=average_row,
        emphasis_rows=emphasis_rows,
        legend_stops=legend_stops,
        legend_ticks=legend_ticks,
        title=_humanize(title),
    )


def _fit_cells(fig, ax, n_rows: int, n_cols: int, passes: int = _FIT_PASSES) -> None:
    """Grow the figure until each cell is exactly ``CELL_COL_INCHES`` by ``CELL_ROW_INCHES``.

    The figure-size rule budgets a fixed number of inches per margin, but constrained layout
    then takes whatever the decorations actually need out of the axes, so the grid comes out
    smaller than the budget by a roughly constant amount: about 1.2 inches, dominated by the
    colorbar's gap and the rotated column labels overhanging the right edge. That barely dents a
    seventeen-column flavor card but squashes a four-column external-foundation one, which is
    what the fixed-margin sizing was meant to prevent (a card's cells would otherwise be a
    quarter narrower than another's).

    Measuring the laid-out axes and growing the figure by the shortfall puts the difference
    where it belongs, since the decorations keep their size in inches while the axes absorbs the
    change. A second pass settles the parts that do scale with the figure, the colorbar being
    sized off the axes height.
    """
    target_w, target_h = CELL_COL_INCHES * n_cols, CELL_ROW_INCHES * n_rows
    for _ in range(passes):
        fig.canvas.draw()
        fig_w, fig_h = fig.get_size_inches()
        position = ax.get_position()
        grid_w, grid_h = position.width * fig_w, position.height * fig_h
        if abs(grid_w - target_w) < _FIT_TOLERANCE_INCHES and (
            abs(grid_h - target_h) < _FIT_TOLERANCE_INCHES
        ):
            return
        fig.set_size_inches(fig_w + (target_w - grid_w), fig_h + (target_h - grid_h))


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
    cbar_ticks: list[float],
    cbar_labels: list[str],
    divider_cols: list[int],
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
        The fully assembled card (endpoint rows, then the AVERAGE row; the columns may include
        reference columns). Drives the cell annotations, and the cell colors unless
        ``color_values`` is given.
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
    cbar_ticks, cbar_labels : list
        Colorbar tick positions and their labels.
    divider_cols : list of int
        Column indices whose left edge carries a heavy vertical rule, dividing the reference
        column from the block compared against it.
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
    # align the auxiliary matrix to the value matrix so the annotation loop indexes them
    # in lockstep
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

    # figure size is built from the grid plus each fixed margin it has to carry, the sibling
    # repo's sizing rule, so the cells stay the same size on a narrow ablation card and a wide
    # flavor one instead of being squeezed by whatever the labels need
    fig, ax = plt.subplots(
        figsize=(
            DATASET_LABEL_INCHES + TICK_LABEL_INCHES + CELL_COL_INCHES * n_cols + COLORBAR_INCHES,
            max(4.0, CELL_ROW_INCHES * n_rows + FIG_HEIGHT_PAD_INCHES),
        ),
        layout="constrained",
    )
    imshow_kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.imshow(
        np.ma.masked_invalid(color_layer),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        **imshow_kwargs,
    )

    # normalized cell colors in [0, 1], so the annotation can flip to white where the ramp goes
    # dark at either end; im.norm is whichever of norm/vmin-vmax was passed above
    with np.errstate(invalid="ignore"):
        normed = np.ma.filled(im.norm(np.ma.masked_invalid(color_layer)), 0.5)

    # separate the cells with white gridlines on the minor ticks rather than drawn borders
    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    # divide the reference column from the block compared against it with a heavy rule running
    # the full height of the grid, in place of a gap between the two
    for col in divider_cols:
        x = col - 0.5
        ax.plot(
            [x, x],
            [-0.5, n_rows - 0.5],
            color="black",
            linewidth=_DIVIDER_LINEWIDTH,
            zorder=4,
        )

    # left-margin boxes carrying each source's display name and split strategy, bracketing its
    # endpoint rows
    _draw_group_boxes(ax, groups, n_cols, average_row)
    # emphasis line directly after the requested source group's last endpoint, heavier than the
    # group brackets so the endpoint the study leans on stays findable
    if emphasis_source is not None:
        for _, end, source in groups:
            if source == emphasis_source:
                _draw_hline(ax, end - 0.5, n_cols, linewidth=1.8)
    # no rule above AVERAGE: the last group's closing rule already ends the endpoint block, and
    # the bracket column closes under the summary

    # pin the view to the imshow extent so the added line segments do not re-margin the axes
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    x_positions = list(range(n_cols))
    ax.set_xticks(
        x_positions,
        labels=[_humanize(str(matrix.columns[i])) for i in x_positions],
        rotation=45,
        ha="left",
        fontsize=FONT_AXIS,
        fontweight="bold",
    )
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    endpoint_labels = _endpoint_labels(matrix.index)
    y_positions = list(range(n_rows))
    ax.set_yticks(
        y_positions, labels=[endpoint_labels[i] for i in y_positions], fontsize=FONT_YTICK
    )
    for pos, label in zip(y_positions, ax.get_yticklabels(), strict=True):
        if pos == average_row:
            label.set_fontweight("bold")

    # annotate each cell with its value and, where defined, its auxiliary (error bar or p-value),
    # flipping the text to white once the cell color is dark enough to swallow black. The same
    # strings, colors, and flips are collected for the HTML rendering, so the two cannot drift
    cell_text = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    cell_color = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    cell_light = [[False for _ in range(n_cols)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            if np.isfinite(value):
                dark = abs(float(normed[i, j]) - 0.5) > _TEXT_FLIP_DISTANCE
                text = annotate(value, aux_values[i, j])
                cell_text[i][j] = text
                cell_light[i][j] = dark
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=FONT_CELL,
                    color="white" if dark else "black",
                )
            if np.isfinite(color_layer[i, j]):
                cell_color[i][j] = to_hex(cmap(im.norm(color_layer[i, j])))

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels, fontsize=FONT_CBAR)

    _fit_cells(fig, ax, n_rows, n_cols)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    matrix.to_csv(out_csv)
    logger.info("wrote %s and %s", out_png, out_csv)

    # the same card as a scalable, selectable HTML table, written beside the PNG
    card_html.write(
        _html_card(
            matrix,
            cell_text=cell_text,
            cell_color=cell_color,
            cell_light=cell_light,
            groups=groups,
            divider_cols=divider_cols,
            average_row=average_row,
            emphasis_source=emphasis_source,
            norm=im.norm,
            cmap=cmap,
            cbar_ticks=cbar_ticks,
            cbar_labels=cbar_labels,
            title=out_png.stem,
        ),
        out_png.with_suffix(".html"),
    )


def _blank_summary_rows(aux: pd.DataFrame, *, keep_average: bool = False) -> pd.DataFrame:
    """Null the auxiliary annotation (error bar or p-value) on the AVERAGE row.

    ``append_average_row`` would otherwise mean the per-cell auxiliary values into the AVERAGE row,
    conflating a per-cell quantity (seed spread, or a per-cell significance test) with the
    endpoint-to-endpoint AVERAGE, which the row shows as a bare mean.

    ``keep_average`` spares the row for a caller that has already replaced it with a quantity
    computed for it directly, as the MAE-delta card does with its own across-endpoint Dunnett
    p-values.
    """
    aux = aux.copy()
    if not keep_average:
        aux.iloc[-1, :] = np.nan
    return aux


def _per_seed_average_delta(
    frame: pd.DataFrame, flavor: str, rows: pd.Index, baseline_mae: pd.Series
) -> np.ndarray:
    """Per-seed mean MAE %-change across ``rows`` for one flavor, one value per finetune seed.

    Raw MAE cannot be averaged across endpoints (they carry different units and ranges), so each
    endpoint is first expressed as the percentage change against that endpoint's baseline mean,
    which is what the card's cells already show, and those scale-free changes are then meaned.
    Averaging a seed's endpoints before comparing groups keeps the endpoint-to-endpoint
    correlation within a seed inside the seed-level spread instead of counting each endpoint as an
    independent observation.
    """
    subset = frame[frame["flavor"] == flavor]
    if subset.empty:
        return np.empty(0, dtype=float)

    # an unseeded label (the legacy single-run baseline) still forms one group of its own, so give
    # it a sentinel key rather than dropping it from the pivot
    table = (
        subset.assign(seed=subset["seed"].fillna(_UNSEEDED_KEY))
        .pivot_table(index="seed", columns="row", values="mae", aggfunc="mean")
        .reindex(columns=rows)
    )
    base = baseline_mae.reindex(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 100.0 * table.sub(base, axis=1).div(base, axis=1)
    per_seed = delta.where(np.isfinite(delta)).mean(axis=1, skipna=True)
    return per_seed.dropna().to_numpy(dtype=float)


def mae_average_pvalues(
    matrix_frame: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_flavor: str,
    mae_matrix: pd.DataFrame,
    baseline_mae: pd.Series,
) -> pd.Series:
    """Family-wise p-value per flavor for the AVERAGE row's across-endpoint mean change.

    The AVERAGE row asks a different question from the cells above it: not whether a flavor beats
    the baseline on one endpoint, but whether its mean change across every endpoint on the card
    differs from the baseline's. That is its own family of many flavors against one control, so it
    gets its own Dunnett test rather than inheriting or averaging the per-cell p-values, which
    would conflate a per-endpoint result with a summary across endpoints.

    Each group is one value per finetune seed: that seed's mean MAE %-change across the card's
    endpoints (see ``_per_seed_average_delta``). The control group is the baseline's own seeds put
    through the same aggregation, so it is centered near zero and carries the baseline's seed
    spread. Because every group is expressed against the same per-endpoint baseline means, the
    control and the treatments are not independent of those denominators; read the row as a
    summary of the card, not as a standalone claim of overall superiority.

    Two consequences of the pooling are worth knowing before reading a white AVERAGE cell. The
    verdict is driven by the mean shift against the *family's* pooled spread, not against the
    column's own, so one unusually noisy column raises the bar for every other column, and two
    columns with equal mean shifts get equal p-values however different their own spreads are.
    And where the seed-by-endpoint grid is ragged (a flavor short a finetune), this function's
    group mean is a mean over seeds of a mean over endpoints while the row above it means the
    per-endpoint values, so the two can differ slightly; the gap is under a tenth of a
    percentage point on the current sweep.

    Parameters
    ----------
    matrix_frame : pandas.DataFrame
        Prepared, seed-collapsed metrics for the flavor columns (carries ``row``, ``flavor``,
        ``seed``, ``mae``).
    frame : pandas.DataFrame
        Prepared, seed-collapsed metrics that include the baseline flavor's rows.
    baseline_flavor : str
        The baseline flavor label (e.g. ``chemeleon_stock``).
    mae_matrix : pandas.DataFrame
        The per-flavor MAE matrix whose columns the returned p-values align to, and whose index
        fixes the endpoint set the average runs over.
    baseline_mae : pandas.Series
        Baseline mean MAE per endpoint, indexed like ``mae_matrix``'s rows.

    Returns
    -------
    pandas.Series
        One p-value per column of ``mae_matrix``, NaN where the comparison is undefined (fewer
        than ``MIN_GROUP_SIZE`` seeds on either side, or no residual variance in the family).
    """
    rows = mae_matrix.index
    control = _per_seed_average_delta(frame, baseline_flavor, rows, baseline_mae)
    samples = {
        flavor: values
        for flavor in mae_matrix.columns
        if (values := _per_seed_average_delta(matrix_frame, flavor, rows, baseline_mae)).size
    }
    pvalues = pd.Series(np.nan, index=mae_matrix.columns, dtype=float)
    for flavor, pvalue in dunnett_pvalues(samples, control).items():
        pvalues[flavor] = pvalue
    return pvalues


def render_r2_card(
    matrix_frame: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_flavor: str,
    out_png: Path,
    *,
    columns: list[str] | None = None,
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
    """
    flavor_r2 = build_matrix(matrix_frame, "r2", columns=columns)
    flavor_r2_std = build_matrix(matrix_frame, "r2", columns=columns, aggfunc="std")
    baseline = build_reference_series(frame, baseline_flavor, "r2")
    baseline_std = build_reference_series(frame, baseline_flavor, "r2", agg="std")
    matrix, divider_cols, ref_cols = assemble_r2_card(flavor_r2, baseline)
    std, _, _ = assemble_r2_card(flavor_r2_std, baseline_std)
    groups = source_groups(flavor_r2.index)
    matrix, average_row = append_average_row(matrix)
    std, _ = append_average_row(std)
    plot_card(
        matrix,
        out_png,
        out_png.with_suffix(".csv"),
        cmap=plt.get_cmap("RdYlGn"),
        vmin=0.0,
        vmax=1.0,
        annotate=lambda v, s: f"{v:.3f}" if not np.isfinite(s) else f"{v:.3f}\n±{s:.3f}",
        cbar_ticks=[0.0, 0.5, 1.0],
        cbar_labels=["0.0", "0.5", "1.0"],
        divider_cols=divider_cols,
        ref_cols=ref_cols,
        groups=groups,
        average_row=average_row,
        aux=_blank_summary_rows(std),
        emphasis_source=EMPHASIS_SOURCE,
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
) -> None:
    """Render the MAE %-change card, coloring only cells that differ significantly from baseline.

    Each cell's color still encodes the percentage change in MAE (green better, red worse), but a
    cell whose per-seed MAE does not differ significantly from the baseline's (Dunnett p above
    ``SIGNIFICANCE_ALPHA``) is painted white regardless of the change, so the card highlights only
    differences the seed spread supports. Every cell is annotated with its change and its p-value.
    The row's flavors form one comparison family, so those p-values are already family-wise; see
    :func:`mae_significance_pvalues`.

    The AVERAGE row is gated on its own test rather than on the cells above it: each flavor's
    per-seed mean change across every endpoint on the card, against the baseline put through the
    same aggregation, corrected as one family (see :func:`mae_average_pvalues`). So a column can
    carry a visible mean improvement and still be painted white when the seed spread does not
    separate it from the baseline overall.

    Parameters
    ----------
    matrix_frame, frame, baseline_flavor, out_png
        As in :func:`render_r2_card`.
    columns : list of str, optional
        Column order for the card. Defaults to the flavor registry order; the ablation report
        passes its bare ablation names.
    """
    baseline_mae = build_reference_series(frame, baseline_flavor, "mae")
    if baseline_mae.empty:
        logger.warning("no %s MAE in the metrics; skipping the MAE-delta card", baseline_flavor)
        return
    mae = build_matrix(matrix_frame, "mae", columns=columns)
    delta = mae_delta_matrix(mae, baseline_mae)
    pvalues = mae_significance_pvalues(matrix_frame, frame, baseline_flavor, mae)
    # paint a non-significant (or untestable) cell white by driving its color to the norm center,
    # while the annotation still shows the real change; leave a missing delta as NaN (grey)
    significant = pvalues.le(SIGNIFICANCE_ALPHA)
    color_delta = delta.mask((~significant) & delta.notna(), 0.0)

    # the AVERAGE row runs its own family of tests on the across-endpoint mean, so it is gated by
    # those p-values rather than by anything derived from the per-cell tests above it
    average_pvalues = mae_average_pvalues(matrix_frame, frame, baseline_flavor, mae, baseline_mae)
    average_significant = average_pvalues.le(SIGNIFICANCE_ALPHA)

    groups = source_groups(delta.index)
    matrix, average_row = append_average_row(delta)
    color_matrix, _ = append_average_row(color_delta)
    average_delta = matrix.iloc[-1]
    color_matrix.iloc[-1] = average_delta.mask(
        (~average_significant) & average_delta.notna(), 0.0
    ).to_numpy()
    pvalue_matrix, _ = append_average_row(pvalues)
    pvalue_matrix.iloc[-1] = average_pvalues.reindex(pvalue_matrix.columns).to_numpy()

    finite = matrix.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    # symmetric scale centered on 0% (the baseline), capped at DELTA_EXTENT_CAP so a few large
    # outliers do not wash out the rest; changes beyond the cap saturate at the end color while
    # their annotation still shows the true value
    observed = max(float(np.abs(finite).max()) if finite.size else 1.0, 1e-6)
    extent = min(observed, DELTA_EXTENT_CAP)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-extent, vmax=extent)
    plot_card(
        matrix,
        out_png,
        out_png.with_suffix(".csv"),
        cmap=_DELTA_CMAP,
        norm=norm,
        annotate=lambda v, p: f"{v:+.0f}%\n{_format_pvalue(p)}".rstrip(),
        cbar_ticks=[-extent, 0.0, extent],
        cbar_labels=[f"-{extent:.0f}%", "0% (baseline or not significant)", f"+{extent:.0f}%"],
        divider_cols=[],
        ref_cols=[],
        groups=groups,
        average_row=average_row,
        aux=_blank_summary_rows(pvalue_matrix, keep_average=True),
        color_values=color_matrix,
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
        "--baseline-flavor",
        default="chemeleon_stock",
        help="flavor label for the stock-CheMeleon reference / MAE-delta baseline (see "
        "slurm/run_stock_baseline.sh); the R² baseline column and the whole MAE-delta card are "
        "skipped if it is absent from --metrics-csv",
    )
    parser.add_argument(
        "--lr-mode",
        choices=("reduced", "unlocked"),
        default=None,
        help="render a learning-rate-experiment setup: filter --metrics-csv (typically "
        "results/lr_metrics.csv) to this protocol's lr_<mode>__<flavor> rows and strip the "
        "prefix so the columns match the flavor registry. The references still read from the "
        "full frame, so pass the protocol's stock baseline via --baseline-flavor "
        "(chemeleon_stock_<mode>). The frozen setup needs no filter (bare-flavor metrics.csv)",
    )
    parser.add_argument(
        "--exclude-recipe",
        nargs="*",
        default=None,
        dest="exclude_recipes",
        metavar="RECIPE",
        help="drop these recipes' rows from the card entirely (e.g. --exclude-recipe cyp1a2_st "
        "expansionrx_logd_st_rand). Filtering happens before the row identities are built, so a "
        "sibling recipe left alone on a (dataset, endpoint) also loses its disambiguating "
        "'(<recipe>)' suffix. The AVERAGE row and its test run over the endpoints that remain, "
        "so a card rendered with exclusions is not comparable to one rendered without them",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="explicit column set (values of the flavor field), overriding the registry-flavor "
        "default so a standalone card shows an arbitrary foundation set (e.g. the external "
        "checkpoints: --columns molpile_1M molpile_5M molpile_10M expansion_gen). The baseline "
        "(--baseline-flavor) is still added as the R² card's first column and the MAE-delta "
        "reference. Under --lr-mode pass the bare foundation name (the lr_<mode>__ prefix is "
        "stripped like the flavors')",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate first")
    raw = pd.read_csv(args.metrics_csv)
    # drop excluded recipes before the row identities are built, so a sibling recipe left alone
    # on a (dataset, endpoint) sheds its now-redundant "(<recipe>)" suffix. A name that matches
    # nothing is a typo that would otherwise silently exclude nothing, so refuse it
    if args.exclude_recipes:
        unknown = sorted(set(args.exclude_recipes) - set(raw["recipe"].unique()))
        if unknown:
            raise SystemExit(
                f"--exclude-recipe: no rows in {args.metrics_csv} for {', '.join(unknown)}"
            )
        raw = raw[~raw["recipe"].isin(args.exclude_recipes)]
    # re-derive the dataset from the recipe and build the disambiguated row identity once on the
    # full frame, so build_matrix and the reference series share one consistent set of row labels
    frame = prepare_rows(collapse_seed_variants(raw))
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

    # an explicit --columns set overrides the registry-flavor default, so a standalone card can
    # show just the external foundations; None keeps the registry order build_matrix defaults to
    columns = args.columns

    suffix = f"_{args.lr_mode}" if args.lr_mode else ""
    render_r2_card(
        matrix_frame,
        frame,
        args.baseline_flavor,
        args.out_dir / f"report_card_r2{suffix}.png",
        columns=columns,
    )
    render_mae_delta_card(
        matrix_frame,
        frame,
        args.baseline_flavor,
        args.out_dir / f"report_card_mae_delta{suffix}.png",
        columns=columns,
    )


if __name__ == "__main__":
    main()
