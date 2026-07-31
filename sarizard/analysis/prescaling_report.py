"""Compare prescaling ablations against the stock-CheMeleon base model, per finetune protocol.

Reads the tidy metrics CSV produced by ``analysis.evaluate`` for the ablation results (where the
``flavor`` column holds ``ablation_<name>__s<seed>`` labels, optionally with a ``__reduced`` or
``__unlocked`` finetune-protocol suffix) plus the stock-CheMeleon baseline rows
(``chemeleon_stock[_<mode>]__s<seed>``), and renders, per finetune LR protocol:

- the two flavor-style report cards (reused from :mod:`sarizard.analysis.report_card`): an R²
  card whose cells are the 5-seed mean with a per-cell ``±`` seed standard deviation, with the
  stock-CheMeleon base model as the baseline column; and an MAE %-change card whose cells compare
  each ablation's per-seed MAE against the stock baseline's per-seed MAE, painted white where
  Dunnett's test gives ``p > SIGNIFICANCE_ALPHA`` (the row's ablations are corrected together as
  one family against the shared baseline);
- a summary CSV and bar chart ranking each ablation by its mean metric across endpoints and the
  number of endpoints it wins, the read used to pick the production prescaling recipe.

When more than one protocol is present it also writes a cross-mode comparison (each ablation's
mean metric under frozen, reduced, and unlocked side by side), so the preprocessing choice can be
checked for stability once the backbone is allowed to move. Frozen artifacts keep their
unsuffixed filenames; the LR protocols add a ``_<mode>`` suffix.

The baseline is the released CheMeleon checkpoint finetuned directly (``chemeleon_stock``), not a
prescaling recipe, so the cards read "how much does each prescaling of an osmordred foundation
help over off-the-shelf CheMeleon". Depends only on pandas, numpy, and matplotlib, so it runs
without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.prescaling_report --metric r2                 # all protocols
    python -m sarizard.analysis.prescaling_report --metric r2 --mpnn-lr-mode frozen
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
    HIGHER_IS_BETTER,
    METRIC_COLUMNS,
    METRIC_LABELS,
)
from sarizard.analysis.paths import (  # noqa: E402
    BASELINE_LR_MODE,
    LR_MODES,
    PLOTS_DIR,
    RESULTS_DIR,
    parse_lr_mode,
)
from sarizard.analysis.report_card import (  # noqa: E402
    build_matrix,
    collapse_seed_variants,
    prepare_rows,
    render_mae_delta_card,
    render_r2_card,
)
from sarizard.configs.generate import stock_baseline_label  # noqa: E402
from sarizard.pretraining.prescaling import ablation_names  # noqa: E402

logger = logging.getLogger(__name__)

ABLATION_METRICS_CSV = RESULTS_DIR / "ablation_metrics.csv"

# summary bar-chart font sizes (points), enlarged to sit consistently with the report cards
# rendered by report_card.plot_card; these charts are smaller figures, so they run a touch below
# the card fonts rather than matching them one for one
SUMMARY_FONT_TITLE = 15
SUMMARY_FONT_LABEL = 14  # axis labels
SUMMARY_FONT_TICK = 13  # tick labels
SUMMARY_FONT_ANNOT = 12  # bar-value annotations and legend

# collapse_seed_variants is the shared helper (report_card); re-exported so callers importing it
# from this module keep working, and it maps ablation_<name>__s<seed> -> ablation_<name>
__all__ = ["collapse_seed_variants", "rank_ablations", "report_one_mode", "mode_comparison"]


def _strip(label: str) -> str:
    """Map an ``ablation_<name>`` result label back to the plain ablation name."""
    prefix = "ablation_"
    return label[len(prefix) :] if label.startswith(prefix) else label


def rank_ablations(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank ablations by mean metric across endpoints and endpoint wins.

    Parameters
    ----------
    pivot : pandas.DataFrame
        Endpoints-by-ablation metric matrix (columns are plain ablation names).
    metric : str
        Metric name, used only to decide the win direction.

    Returns
    -------
    pandas.DataFrame
        One row per ablation with ``mean``, ``median``, and ``wins`` columns, sorted best
        first for this metric.
    """
    higher = metric in HIGHER_IS_BETTER
    # best ablation per endpoint (row), counting wins; ignore all-NaN rows
    if higher:
        winners = pivot.idxmax(axis=1)
    else:
        winners = pivot.idxmin(axis=1)
    wins = winners.value_counts()
    summary = pd.DataFrame(
        {
            "mean": pivot.mean(axis=0),
            "median": pivot.median(axis=0),
            "wins": wins.reindex(pivot.columns).fillna(0).astype(int),
        }
    )
    return summary.sort_values("mean", ascending=not higher)


def plot_ranking(summary: pd.DataFrame, metric: str, out_png: Path) -> None:
    """Render a horizontal bar chart of the per-ablation mean metric."""
    label = METRIC_LABELS.get(metric, metric)
    fig, ax = plt.subplots(figsize=(8.0, 0.6 * len(summary) + 1.6), constrained_layout=True)
    order = summary.iloc[::-1]  # best at top
    ax.barh(order.index, order["mean"], color="steelblue")
    for y, (value, wins) in enumerate(zip(order["mean"], order["wins"], strict=True)):
        ax.text(value, y, f"  {value:.3f} ({wins} wins)", va="center", fontsize=SUMMARY_FONT_ANNOT)
    # headroom so the bigger bar-end annotations do not clip on the right spine
    ax.set_xlim(right=float(order["mean"].max()) * 1.3)
    ax.tick_params(labelsize=SUMMARY_FONT_TICK)
    ax.set_xlabel(f"mean {label} across endpoints", fontsize=SUMMARY_FONT_LABEL)
    ax.set_title(f"Prescaling ablation ranking ({label})", fontsize=SUMMARY_FONT_TITLE)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mode_comparison(comparison: pd.DataFrame, metric: str, out_png: Path) -> None:
    """Plot each ablation's mean metric under each finetune LR protocol as grouped bars."""
    label = METRIC_LABELS.get(metric, metric)
    modes = list(comparison.columns)
    ablations = list(comparison.index)
    positions = np.arange(len(ablations))
    height = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(8.5, 0.8 * len(ablations) + 1.6), constrained_layout=True)
    # one offset bar per protocol so a recipe's three protocols sit side by side per ablation
    for i, mode in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * height
        ax.barh(positions + offset, comparison[mode].to_numpy(), height=height, label=mode)
    ax.set_yticks(positions, labels=ablations, fontsize=SUMMARY_FONT_TICK)
    ax.tick_params(axis="x", labelsize=SUMMARY_FONT_TICK)
    ax.set_xlabel(f"mean {label} across endpoints", fontsize=SUMMARY_FONT_LABEL)
    ax.set_title(
        f"Prescaling ablation by finetune LR protocol ({label})", fontsize=SUMMARY_FONT_TITLE
    )
    ax.legend(title="mpnn_lr", fontsize=SUMMARY_FONT_ANNOT, title_fontsize=SUMMARY_FONT_ANNOT)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_one_mode(
    mode_frame: pd.DataFrame, full_frame: pd.DataFrame, metric: str, mode: str, out_dir: Path
) -> pd.Series | None:
    """Render one protocol's two report cards and ranking; return its per-ablation mean series.

    Writes the R² card (5-seed mean with ± seed std) and the MAE %-change card (each ablation's
    seeds vs the stock-CheMeleon baseline's seeds, Dunnett's test across the row's ablations,
    white where not significant) via the shared flavor-card renderers, plus the ranking CSV and
    bar chart. Frozen keeps the unsuffixed filenames; the LR protocols add a ``_<mode>`` suffix.

    Parameters
    ----------
    mode_frame : pandas.DataFrame
        Prepared, seed-collapsed ablation rows for this protocol only, with the ``ablation_``
        prefix already stripped from ``flavor`` so the columns read as bare ablation names.
    full_frame : pandas.DataFrame
        The whole prepared, seed-collapsed frame, so the card renderers can look up this
        protocol's stock-CheMeleon baseline rows by label.
    metric : str
        Metric column the ranking summarizes (the cards themselves are R² and MAE).
    mode : str
        The protocol these rows belong to, one of :data:`sarizard.analysis.paths.LR_MODES`.
    out_dir : pathlib.Path
        Directory the cards, ranking CSV, and bar chart are written to.

    Returns
    -------
    pandas.Series or None
        Per-ablation mean metric (index is the plain ablation name), or ``None`` when the frame
        holds no ablation results for this protocol.
    """
    columns = [name for name in ablation_names() if name in set(mode_frame["flavor"])]
    if not columns:
        return None
    suffix = "" if mode == BASELINE_LR_MODE else f"_{mode}"
    baseline_flavor = stock_baseline_label(mode)

    # the two flavor-style cards: R² with per-cell seed std, and MAE %-change vs the stock
    # baseline gated by the unpaired seed t-test (both inherit the flavor-card cosmetics)
    render_r2_card(
        mode_frame,
        full_frame,
        baseline_flavor,
        out_dir / f"ablation_report_card_r2{suffix}.png",
        columns=columns,
    )
    render_mae_delta_card(
        mode_frame,
        full_frame,
        baseline_flavor,
        out_dir / f"ablation_report_card_mae_delta{suffix}.png",
        columns=columns,
    )

    # ranking read used to pick the production recipe
    pivot = build_matrix(mode_frame, metric, columns=columns)
    summary = rank_ablations(pivot, metric)
    summary.to_csv(out_dir / f"prescaling_ranking_{metric}{suffix}.csv")
    plot_ranking(summary, metric, out_dir / f"prescaling_ranking_{metric}{suffix}.png")
    logger.info("protocol %s: best by mean %s is %s", mode, metric, summary.index[0])
    return summary["mean"]


def mode_comparison(per_mode_mean: dict[str, pd.Series]) -> pd.DataFrame:
    """Assemble a per-ablation mean-metric matrix across protocols, ordered frozen to unlocked."""
    comparison = pd.DataFrame(per_mode_mean)
    return comparison[[mode for mode in LR_MODES if mode in comparison.columns]]


def main() -> None:
    """Build the ablation report cards and ranking per protocol, plus a cross-protocol read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="r2", choices=METRIC_COLUMNS, help="ranking metric")
    parser.add_argument(
        "--metrics-csv", type=Path, default=ABLATION_METRICS_CSV, help="tidy ablation metrics CSV"
    )
    parser.add_argument(
        "--mpnn-lr-mode",
        default="all",
        choices=(*LR_MODES, "all"),
        help="finetune LR protocol to report; 'all' builds every present protocol and, when more "
        "than one is present, a cross-protocol comparison",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTS_DIR,
        help="directory for the cards, rankings, and comparison; point at an archived run's plots "
        "dir to render it without overwriting the live report card (default: live plots dir)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate on the ablations")
    raw = pd.read_csv(args.metrics_csv)

    # peel the finetune-protocol suffix off the ablation labels first (ablations carry the
    # __<mode> tag; the stock-baseline labels encode the mode with a single underscore and so
    # round-trip to the frozen tag, which is fine since they are found by explicit label below),
    # then seed-collapse and build the disambiguated row identity once on the whole frame so the
    # card columns and the baseline reference share one consistent set of row labels
    parsed = raw["flavor"].map(parse_lr_mode)
    raw["mpnn_lr_mode"] = parsed.map(lambda base_mode: base_mode[1])
    raw["flavor"] = parsed.map(lambda base_mode: base_mode[0])
    full = prepare_rows(collapse_seed_variants(raw))

    present = [mode for mode in LR_MODES if mode in set(full["mpnn_lr_mode"])]
    requested = list(LR_MODES) if args.mpnn_lr_mode == "all" else [args.mpnn_lr_mode]
    modes = [mode for mode in present if mode in requested]
    if not modes:
        raise SystemExit(f"no ablation results for protocol(s) {requested} in {args.metrics_csv}")

    # one pair of cards + ranking per protocol; keep each protocol's mean series for the comparison
    per_mode_mean: dict[str, pd.Series] = {}
    for mode in modes:
        # this protocol's ablation rows, with the ablation_ prefix stripped to bare column names
        mode_frame = full[
            (full["mpnn_lr_mode"] == mode) & full["flavor"].str.startswith("ablation_")
        ].copy()
        mode_frame["flavor"] = mode_frame["flavor"].map(_strip)
        mean_series = report_one_mode(mode_frame, full, args.metric, mode, args.out_dir)
        if mean_series is None:
            logger.warning("protocol %s: no ablation results; skipping", mode)
            continue
        per_mode_mean[mode] = mean_series

    if not per_mode_mean:
        raise SystemExit("no ablation results found in the metrics CSV")

    # cross-protocol read: does the winning recipe hold once the backbone can move?
    if len(per_mode_mean) > 1:
        comparison = mode_comparison(per_mode_mean)
        comp_csv = args.out_dir / f"prescaling_mode_comparison_{args.metric}.csv"
        comparison.to_csv(comp_csv)
        plot_mode_comparison(
            comparison, args.metric, args.out_dir / f"prescaling_mode_comparison_{args.metric}.png"
        )
        higher = args.metric in HIGHER_IS_BETTER
        winners = {
            mode: (comparison[mode].idxmax() if higher else comparison[mode].idxmin())
            for mode in comparison.columns
        }
        if len(set(winners.values())) == 1:
            winner = next(iter(winners.values()))
            logger.info("ranking is stable: %s wins under every protocol", winner)
        else:
            logger.info("winning recipe shifts across protocols: %s", winners)
            logger.info("read %s before baking in a recipe", comp_csv)
    else:
        logger.info("only protocol %s present; no cross-protocol comparison written", modes[0])


if __name__ == "__main__":
    main()
