"""Summary boxplots of a report card's AVERAGE row, one box per card column.

A report card's AVERAGE row compresses every endpoint into one number per column, which hides
both what that mean is made of and how precisely it is known. These figures unpack it: each
column becomes a box over the per-endpoint cells the AVERAGE means, carrying a separate error
bar for the seed uncertainty on the mean itself. The two spreads answer different questions, so
they are drawn as different marks:

- the **box and whiskers** are the endpoint-to-endpoint spread (quartiles over the card's
  endpoint rows, whiskers at 1.5 IQR, outliers as points). A wide box means the column's
  performance depends heavily on which endpoint is asked, whatever its average;
- the **error bar** is the finetune-seed spread of the across-endpoint mean (one value per seed,
  standard deviation across seeds). This is the precision of the AVERAGE itself, and it is the
  quantity the card's AVERAGE-row Dunnett test works on.

Two figures are produced per card pair, matching the two cards:

- the R² summary, whose columns follow the R² card (the stock-CheMeleon baseline first, behind a
  divider, then the flavors). Its boxes are drawn black: R² carries no significance test of its
  own, so coloring them would imply one;
- the MAE %-change summary, whose flavor columns are filled with the exact color their AVERAGE
  cell earns on the MAE-delta card (green better, red worse, white where Dunnett's test does not
  separate the column from the baseline, grey where undefined). The color is read from the card
  itself via ``report_card.average_cell_colors`` rather than recomputed, so the box and the cell
  can never disagree.

Both write a CSV of the plotted quantities alongside the PNG. Depends only on pandas, numpy, and
matplotlib, so it runs without openadmet or a GPU.

Usage:
    # flavor sweep, reduced protocol
    python -m sarizard.analysis.average_summary --lr-mode reduced \
        --metrics-csv results/lr_metrics.csv --baseline-flavor chemeleon_stock_reduced \
        --exclude-recipe cyp1a2_st expansionrx_logd_st_rand

    # external foundations, reduced protocol
    python -m sarizard.analysis.average_summary --lr-mode reduced \
        --metrics-csv results/external_metrics.csv --baseline-flavor chemeleon_stock_reduced \
        --columns molpile_1M molpile_5M molpile_10M expansion_gen \
        --out-dir plots/external_foundations \
        --exclude-recipe cyp1a2_st expansionrx_logd_st_rand

    # prescaling ablations, reduced protocol
    python -m sarizard.analysis.average_summary --lr-mode reduced \
        --metrics-csv results/ablation_metrics.csv --baseline-flavor chemeleon_stock_reduced \
        --ablations --prefix ablation_average_summary \
        --exclude-recipe cyp1a2_st expansionrx_logd_st_rand
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
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# share the report cards' ground styling so the summaries read as part of the same figure family
plt.style.use("seaborn-v0_8-white")

from sarizard.analysis.paths import (  # noqa: E402
    BASELINE_LR_MODE,
    METRICS_CSV,
    PLOTS_DIR,
    parse_lr_mode,
)
from sarizard.analysis.report_card import (  # noqa: E402
    BASELINE_LABEL,
    DELTA_CMAP,
    SIGNIFICANCE_ALPHA,
    UNSEEDED_KEY,
    assemble_r2_card,
    average_cell_colors,
    build_mae_delta_card,
    build_matrix,
    build_reference_series,
    collapse_seed_variants,
    filter_lr_mode,
    per_seed_average_delta,
    prepare_rows,
)
from sarizard.pretraining.prescaling import ablation_names  # noqa: E402

logger = logging.getLogger(__name__)

# font sizes (points), a step below the report cards' scale: these are single-panel figures a
# few inches tall rather than the cards' 20-inch grids, so the card sizes would overwhelm them
FONT_LABEL = 13  # axis labels
FONT_TICK = 11  # column and value tick labels
FONT_ANNOT = 10  # p-value sub-labels and the legend

# box geometry: the width each column's box fills of its one-unit slot, and the figure inches
# allotted per column plus the fixed margin for the axis labels and rotated tick labels
_BOX_WIDTH = 0.62
_COL_INCHES = 0.85
# fixed width for the axis labels and the legend, which is parked outside the axes on the right
_FIG_MARGIN_INCHES = 5.0
_FIG_HEIGHT_INCHES = 6.5

# output resolution: these are ordinary single-panel figures, so they take the repo's plotting
# default rather than the report cards' 600 dpi (which those need only for their small cell text)
_DPI = 300

# unfilled boxes for the R² figure, where no significance test backs a fill color
_UNFILLED = "none"
_LINE_COLOR = "black"


def _per_seed_average_metric(
    frame: pd.DataFrame, flavor: str, rows: pd.Index, metric: str
) -> np.ndarray:
    """Per-seed mean of ``metric`` across ``rows`` for one flavor, one value per finetune seed.

    The seed replicates share a flavor label after ``collapse_seed_variants`` but stay separate
    rows, so pivoting seed against endpoint and meaning along the endpoints recovers what each
    seed contributed to the AVERAGE. Averaging a seed's endpoints before comparing seeds keeps
    the endpoint-to-endpoint correlation within a seed inside the seed-level spread, instead of
    counting each endpoint as an independent observation.

    This is the R² counterpart of :func:`report_card.per_seed_average_delta`, which has to
    express each endpoint against a baseline first because raw MAE cannot be averaged across
    endpoints; R² is already scale-free, so it is meaned directly.

    Parameters
    ----------
    frame : pandas.DataFrame
        Prepared, seed-collapsed metrics carrying ``row``, ``flavor``, ``seed``, and ``metric``.
    flavor : str
        The flavor label to summarize.
    rows : pandas.Index
        The endpoint rows the average runs over, fixing the endpoint set across columns.
    metric : str
        Metric column to average.

    Returns
    -------
    numpy.ndarray
        One value per seed that has any of ``rows``; empty when the flavor is absent.
    """
    subset = frame[frame["flavor"] == flavor]
    if subset.empty:
        return np.empty(0, dtype=float)
    # an unseeded label (the legacy single-run baseline) still forms one group of its own, so give
    # it a sentinel key rather than dropping it from the pivot
    table = (
        subset.assign(seed=subset["seed"].fillna(UNSEEDED_KEY))
        .pivot_table(index="seed", columns="row", values=metric, aggfunc="mean")
        .reindex(columns=rows)
        .mean(axis=1, skipna=True)
    )
    return table.dropna().to_numpy(dtype=float)


def _format_pvalue(pvalue: float) -> str:
    """Render an AVERAGE-row p-value for a tick sub-label, matching the card's annotation."""
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.001:
        return "p<.001"
    return f"p={pvalue:.3f}"


def _box_stats(values: np.ndarray) -> dict[str, float]:
    """Quartiles and 1.5-IQR whisker ends of a box, matching what matplotlib draws."""
    q1, median, q3 = (float(x) for x in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    inside_low = values[values >= q1 - 1.5 * iqr]
    inside_high = values[values <= q3 + 1.5 * iqr]
    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "whisker_low": float(inside_low.min()) if inside_low.size else q1,
        "whisker_high": float(inside_high.max()) if inside_high.size else q3,
    }


def _draw_summary(
    samples: dict[str, np.ndarray],
    averages: pd.Series,
    seed_samples: dict[str, np.ndarray],
    out_png: Path,
    out_csv: Path,
    *,
    ylabel: str,
    colors: pd.Series | None = None,
    sub_labels: dict[str, str] | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
    divider_after: int | None = None,
    legend_handles: list[Patch] | None = None,
    extra_csv: pd.DataFrame | None = None,
) -> None:
    """Draw one summary figure and write the quantities behind it.

    Parameters
    ----------
    samples : dict of str to numpy.ndarray
        Per-column per-endpoint values, in the left-to-right column order. These are what the box
        and whiskers describe.
    averages : pandas.Series
        Each column's AVERAGE-row value, where the error-bar marker is centered. Taken from the
        card rather than re-meaned from ``seed_samples``, which can differ by a hair where the
        seed-by-endpoint grid is ragged.
    seed_samples : dict of str to numpy.ndarray
        Per-column per-seed across-endpoint means; their standard deviation is the error bar.
    out_png, out_csv : pathlib.Path
        Image and summary-CSV output paths.
    ylabel : str
        Y-axis label.
    colors : pandas.Series, optional
        Per-column box fill color. Omitted leaves the boxes unfilled with black edges.
    sub_labels : dict of str to str, optional
        A second line appended under a column's tick label (the MAE figure puts its p-value
        there).
    reference : float, optional
        Y value for a horizontal reference line (the baseline's own average, or zero change).
    reference_label : str, optional
        Legend label for that line.
    divider_after : int, optional
        Draw a heavy vertical rule after this column index, matching the R² card's rule between
        the baseline column and the block measured against it.
    legend_handles : list of matplotlib.patches.Patch, optional
        Extra legend entries (the MAE figure's color key).
    extra_csv : pandas.DataFrame, optional
        Further per-column columns to join into the CSV (the MAE figure's p-values).
    """
    columns = list(samples)
    positions = np.arange(len(columns), dtype=float)
    width = max(_COL_INCHES * len(columns) + _FIG_MARGIN_INCHES, 7.0)
    fig, ax = plt.subplots(figsize=(width, _FIG_HEIGHT_INCHES), layout="constrained")

    # the endpoint-to-endpoint spread: one box per column over the cells the AVERAGE means
    artists = ax.boxplot(
        [samples[column] for column in columns],
        positions=positions,
        widths=_BOX_WIDTH,
        patch_artist=True,
        medianprops={"color": _LINE_COLOR, "linewidth": 1.4},
        boxprops={"edgecolor": _LINE_COLOR, "linewidth": 1.2},
        whiskerprops={"color": _LINE_COLOR, "linewidth": 1.0},
        capprops={"color": _LINE_COLOR, "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": "none",
            "markeredgecolor": "grey",
        },
    )
    for column, patch in zip(columns, artists["boxes"], strict=True):
        patch.set_facecolor(_UNFILLED if colors is None else colors.get(column, _UNFILLED))

    # the seed spread of the across-endpoint mean, a different quantity from the box, so it is
    # drawn as its own mark rather than folded into the whiskers
    seed_sd = np.array(
        [
            np.std(seed_samples[column], ddof=1) if seed_samples[column].size > 1 else np.nan
            for column in columns
        ]
    )
    ax.errorbar(
        positions,
        averages.reindex(columns).to_numpy(dtype=float),
        yerr=seed_sd,
        fmt="D",
        markersize=4.5,
        color=_LINE_COLOR,
        markerfacecolor="white",
        markeredgewidth=1.2,
        capsize=4.0,
        linestyle="none",
        zorder=4,
        label="AVERAGE ± seed SD",
    )

    if reference is not None:
        ax.axhline(
            reference, color="dimgrey", linewidth=1.2, linestyle="--", label=reference_label
        )
    if divider_after is not None:
        ax.axvline(divider_after + 0.5, color=_LINE_COLOR, linewidth=3.0)

    tick_labels = [
        column if sub_labels is None else f"{column}\n{sub_labels.get(column, '')}".rstrip()
        for column in columns
    ]
    ax.set_xticks(positions, labels=tick_labels, rotation=45, ha="right", fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_xlim(-0.7, len(columns) - 0.3)
    ax.grid(axis="y", color="lightgrey", linewidth=0.6)
    ax.set_axisbelow(True)

    # the legend sits outside the axes: a narrow card (six ablation recipes) leaves no in-axes
    # corner the color key fits without covering a box
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + (legend_handles or []),
        fontsize=FONT_ANNOT,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    # the plotted quantities, so a number in the figure can be read off without re-deriving it
    table = pd.DataFrame(
        [
            {
                "column": column,
                "n_endpoints": int(samples[column].size),
                "average": float(averages.get(column, np.nan)),
                **_box_stats(samples[column]),
                "n_seeds": int(seed_samples[column].size),
                "seed_sd": float(sd) if np.isfinite(sd) else np.nan,
            }
            for column, sd in zip(columns, seed_sd, strict=True)
        ]
    ).set_index("column")
    if extra_csv is not None:
        table = table.join(extra_csv)
    table.to_csv(out_csv)
    logger.info("wrote %s and %s", out_png, out_csv)


def render_r2_summary(
    matrix_frame: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_flavor: str,
    out_png: Path,
    *,
    columns: list[str] | None = None,
) -> None:
    """Draw the R² card's AVERAGE row as one box per column, boxes black.

    The columns and their order follow the R² card exactly, baseline first behind the same heavy
    divider. Nothing is colored: the R² card runs no significance test, so a fill would assert
    one that was never made. The MAE summary is where the significance verdicts live.

    Parameters
    ----------
    matrix_frame : pandas.DataFrame
        Prepared, seed-collapsed metrics for the card columns.
    frame : pandas.DataFrame
        Prepared, seed-collapsed metrics that include the baseline flavor's rows.
    baseline_flavor : str
        Baseline flavor label (e.g. ``chemeleon_stock_reduced``).
    out_png : pathlib.Path
        Image output path; the summary CSV is written alongside it.
    columns : list of str, optional
        Column order, values of the ``flavor`` field. Defaults to the flavor registry order.
    """
    flavor_r2 = build_matrix(matrix_frame, "r2", columns=columns)
    baseline_r2 = build_reference_series(frame, baseline_flavor, "r2")
    matrix, divider_cols, _ = assemble_r2_card(flavor_r2, baseline_r2)
    rows = matrix.index

    # the baseline column is drawn from the full frame under its own flavor label; every other
    # column is a card column of matrix_frame
    sources = {
        column: (frame, baseline_flavor) if column == BASELINE_LABEL else (matrix_frame, column)
        for column in matrix.columns
    }
    samples = {
        column: matrix[column].dropna().to_numpy(dtype=float)
        for column in matrix.columns
        if matrix[column].notna().any()
    }
    seed_samples = {
        column: _per_seed_average_metric(source, flavor, rows, "r2")
        for column, (source, flavor) in sources.items()
        if column in samples
    }
    averages = matrix.mean(axis=0, skipna=True)

    reference = float(averages[BASELINE_LABEL]) if BASELINE_LABEL in averages else None
    _draw_summary(
        samples,
        averages,
        seed_samples,
        out_png,
        out_png.with_suffix(".csv"),
        ylabel="R² across endpoints",
        reference=reference,
        reference_label="stock-CheMeleon AVERAGE",
        divider_after=(divider_cols[0] - 1) if divider_cols else None,
    )


def render_mae_delta_summary(
    matrix_frame: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_flavor: str,
    out_png: Path,
    *,
    columns: list[str] | None = None,
) -> None:
    """Draw the MAE-delta card's AVERAGE row as one box per column, colored by significance.

    Each box is filled with the color its AVERAGE cell carries on the card: green where the
    column's mean MAE across endpoints is significantly below the baseline's, red where it is
    significantly above, white where Dunnett's test does not separate the two at
    ``SIGNIFICANCE_ALPHA``, grey where the change is undefined. The p-value driving that verdict
    is printed under the column's label, so a white box can be read as "tested, not separated"
    rather than as an omission.

    The box itself is the endpoint-to-endpoint spread of the per-endpoint changes, which is not
    what the test looks at: the test compares per-seed across-endpoint means, and that spread is
    the error bar. A column can therefore carry a wide box and a significant color at once.

    Parameters
    ----------
    matrix_frame, frame, baseline_flavor, out_png
        As in :func:`render_r2_summary`.
    columns : list of str, optional
        Column order, values of the ``flavor`` field. Defaults to the flavor registry order.
    """
    card = build_mae_delta_card(matrix_frame, frame, baseline_flavor, columns=columns)
    if card is None:
        logger.warning("no %s MAE in the metrics; skipping the MAE summary", baseline_flavor)
        return

    rows = card.delta.index
    samples = {
        column: card.delta[column].dropna().to_numpy(dtype=float)
        for column in card.delta.columns
        if card.delta[column].notna().any()
    }
    seed_samples = {
        column: per_seed_average_delta(matrix_frame, column, rows, card.baseline_mae)
        for column in samples
    }
    colors = average_cell_colors(card)
    sub_labels = {
        column: _format_pvalue(card.average_pvalues.get(column, np.nan)) for column in samples
    }
    # the key's swatches are the card's own ramp ends and center, so a legend entry cannot drift
    # from the fills it explains
    legend_handles = [
        Patch(facecolor=to_hex(DELTA_CMAP(end)), edgecolor=_LINE_COLOR, label=label)
        for end, label in (
            (0.0, "lower MAE than stock (p ≤ 0.05)"),
            (1.0, "higher MAE than stock (p ≤ 0.05)"),
            (0.5, "not significant (p > 0.05)"),
        )
    ]
    extra = pd.DataFrame(
        {
            "average_pvalue": card.average_pvalues,
            "significant": card.average_pvalues.le(SIGNIFICANCE_ALPHA),
        }
    )
    extra.index.name = "column"

    _draw_summary(
        samples,
        card.average_delta,
        seed_samples,
        out_png,
        out_png.with_suffix(".csv"),
        ylabel="MAE change vs stock CheMeleon (%)",
        colors=colors,
        sub_labels=sub_labels,
        reference=0.0,
        reference_label="stock-CheMeleon baseline",
        legend_handles=legend_handles,
        extra_csv=extra,
    )


def main() -> None:
    """Render the AVERAGE-row summary boxplots for one card setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV, help="tidy metrics CSV")
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR, help="directory for the PNGs")
    parser.add_argument(
        "--baseline-flavor",
        default="chemeleon_stock",
        help="flavor label for the stock-CheMeleon reference column and the MAE-delta baseline",
    )
    parser.add_argument(
        "--lr-mode",
        choices=("reduced", "unlocked"),
        default=None,
        help="finetune protocol to summarize; omit for frozen. Selects the lr_<mode>__-prefixed "
        "rows as report_card does, or under --ablations the __<mode>-suffixed ones",
    )
    parser.add_argument(
        "--exclude-recipe",
        nargs="*",
        default=None,
        dest="exclude_recipes",
        metavar="RECIPE",
        help="drop these recipes' rows before the row identities are built, so the summary runs "
        "over the same endpoints as the card it summarizes",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="explicit column set (values of the flavor field), overriding the registry-flavor "
        "default (e.g. the external checkpoints: --columns molpile_1M molpile_5M molpile_10M "
        "expansion_gen)",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="summarize the prescaling ablations: read the protocol off the labels' __<mode> "
        "suffix, strip the ablation_ prefix, and use the prescaling recipe names as the columns",
    )
    parser.add_argument(
        "--prefix",
        default="average_summary",
        help="output filename stem; the metric and the LR mode are appended",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate first")
    raw = pd.read_csv(args.metrics_csv)
    # exclusions land before the row identities are built, so a sibling recipe left alone on a
    # (dataset, endpoint) sheds its now-redundant "(<recipe>)" suffix exactly as on the card
    if args.exclude_recipes:
        unknown = sorted(set(args.exclude_recipes) - set(raw["recipe"].unique()))
        if unknown:
            raise SystemExit(
                f"--exclude-recipe: no rows in {args.metrics_csv} for {', '.join(unknown)}"
            )
        raw = raw[~raw["recipe"].isin(args.exclude_recipes)]

    # the two label schemes carry the finetune protocol differently: the flavor and external
    # sweeps namespace it as an lr_<mode>__ prefix, the ablations tag it as a __<mode> suffix
    # (see paths.parse_lr_mode), so the protocol has to be peeled off before the frame is built
    if args.ablations:
        parsed = raw["flavor"].map(parse_lr_mode)
        raw = raw.assign(
            mpnn_lr_mode=parsed.map(lambda base_mode: base_mode[1]),
            flavor=parsed.map(lambda base_mode: base_mode[0]),
        )
    frame = prepare_rows(collapse_seed_variants(raw))

    columns = args.columns
    matrix_frame = frame
    if args.ablations:
        mode = args.lr_mode or BASELINE_LR_MODE
        matrix_frame = frame[
            (frame["mpnn_lr_mode"] == mode) & frame["flavor"].str.startswith("ablation_")
        ]
        if matrix_frame.empty:
            raise SystemExit(f"no ablation rows for protocol {mode} in {args.metrics_csv}")
        # strip the ablation_ prefix so the columns read as bare recipe names, as the cards do
        matrix_frame = matrix_frame.assign(
            flavor=matrix_frame["flavor"].str.removeprefix("ablation_")
        )
        present = set(matrix_frame["flavor"])
        columns = [name for name in ablation_names() if name in present]
    elif args.lr_mode is not None:
        matrix_frame = filter_lr_mode(frame, args.lr_mode)
        if matrix_frame.empty:
            raise SystemExit(
                f"--lr-mode {args.lr_mode} matched no lr_{args.lr_mode}__ rows in "
                f"{args.metrics_csv}"
            )

    suffix = f"_{args.lr_mode}" if args.lr_mode else ""
    render_r2_summary(
        matrix_frame,
        frame,
        args.baseline_flavor,
        args.out_dir / f"{args.prefix}_r2{suffix}.png",
        columns=columns,
    )
    render_mae_delta_summary(
        matrix_frame,
        frame,
        args.baseline_flavor,
        args.out_dir / f"{args.prefix}_mae_delta{suffix}.png",
        columns=columns,
    )


if __name__ == "__main__":
    main()
