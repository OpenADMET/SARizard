"""Compare prescaling ablations: endpoints by ablation, ranking, and a cross-protocol read.

Reads the tidy metrics CSV produced by ``analysis.evaluate`` for the ablation results (where the
``flavor`` column holds ``ablation_<name>__s<seed>`` labels, optionally with a ``__reduced`` or
``__unlocked`` finetune-protocol suffix) and renders, per finetune LR protocol:

- a report-card heatmap, endpoints (rows) by ablation (columns), reusing the row-relative
  coloring from ``analysis.report_card``;
- a summary CSV and bar chart ranking each ablation by its mean metric across endpoints and
  the number of endpoints it wins, the read used to pick the production prescaling recipe.

When more than one protocol is present it also writes a cross-mode comparison (each ablation's
mean metric under frozen, reduced, and unlocked side by side), so the preprocessing choice can be
checked for stability once the backbone is allowed to move rather than assumed from the frozen
ranking alone. Frozen artifacts keep their unsuffixed filenames; the LR protocols add a
``_<mode>`` suffix.

Depends only on pandas, numpy, and matplotlib, so it runs without openadmet or a GPU.

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
    ablation_label,
    parse_lr_mode,
)
from sarizard.analysis.report_card import (  # noqa: E402
    build_matrix,
    collapse_seed_variants,
    plot_report_card,
)
from sarizard.pretraining.prescaling import ablation_names  # noqa: E402

logger = logging.getLogger(__name__)

ABLATION_METRICS_CSV = RESULTS_DIR / "ablation_metrics.csv"

# collapse_seed_variants is the shared helper (report_card); imported here so callers importing
# it from this module keep working, and it maps ablation_<name>__s<seed> -> ablation_<name>
__all__ = ["collapse_seed_variants", "rank_ablations", "report_one_mode", "mode_comparison"]


def _strip(label: str) -> str:
    """Map an ``ablation_<name>`` result label back to the plain ablation name."""
    prefix = "ablation_"
    return label[len(prefix):] if label.startswith(prefix) else label


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
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(summary) + 1.5), constrained_layout=True)
    order = summary.iloc[::-1]  # best at top
    ax.barh(order.index, order["mean"], color="steelblue")
    for y, (value, wins) in enumerate(zip(order["mean"], order["wins"], strict=True)):
        ax.text(value, y, f"  {value:.3f} ({wins} wins)", va="center", fontsize=8)
    ax.set_xlabel(f"mean {label} across endpoints")
    ax.set_title(f"Prescaling ablation ranking ({label})", fontsize=11)
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
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(ablations) + 1.5), constrained_layout=True)
    # one offset bar per protocol so a recipe's three protocols sit side by side per ablation
    for i, mode in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * height
        ax.barh(positions + offset, comparison[mode].to_numpy(), height=height, label=mode)
    ax.set_yticks(positions)
    ax.set_yticklabels(ablations)
    ax.set_xlabel(f"mean {label} across endpoints")
    ax.set_title(f"Prescaling ablation by finetune LR protocol ({label})", fontsize=11)
    ax.legend(title="mpnn_lr", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_one_mode(frame: pd.DataFrame, metric: str, mode: str) -> pd.Series | None:
    """Build and save the report card and ranking for one protocol; return its mean series.

    Writes the endpoints-by-ablation heatmap, the ranking CSV, and the ranking bar chart for this
    finetune LR protocol (frozen keeps the unsuffixed filenames, the LR protocols add a
    ``_<mode>`` suffix), then returns the per-ablation mean metric for the cross-protocol read.

    Parameters
    ----------
    frame : pandas.DataFrame
        Tidy metrics for a single protocol, with ``ablation_<name>__s<seed>`` labels in
        ``flavor`` (the protocol suffix already stripped).
    metric : str
        Metric column to report.
    mode : str
        The protocol these rows belong to, one of :data:`sarizard.analysis.paths.LR_MODES`; sets
        the output-filename suffix.

    Returns
    -------
    pandas.Series or None
        Per-ablation mean metric (index is the plain ablation name), or ``None`` when the frame
        holds no ablation results for this metric.
    """
    collapsed = collapse_seed_variants(frame)
    columns = [ablation_label(name) for name in ablation_names()]
    pivot = build_matrix(collapsed, metric, columns=columns)
    pivot.columns = [_strip(col) for col in pivot.columns]
    if pivot.empty or pivot.shape[1] == 0:
        return None
    suffix = "" if mode == BASELINE_LR_MODE else f"_{mode}"
    out_png = PLOTS_DIR / f"prescaling_report_{metric}{suffix}.png"
    plot_report_card(pivot, metric, out_png, out_png.with_suffix(".csv"))
    summary = rank_ablations(pivot, metric)
    summary.to_csv(PLOTS_DIR / f"prescaling_ranking_{metric}{suffix}.csv")
    plot_ranking(summary, metric, PLOTS_DIR / f"prescaling_ranking_{metric}{suffix}.png")
    logger.info("protocol %s: best by mean %s is %s", mode, metric, summary.index[0])
    return summary["mean"]


def mode_comparison(per_mode_mean: dict[str, pd.Series]) -> pd.DataFrame:
    """Assemble a per-ablation mean-metric matrix across protocols, ordered frozen to unlocked."""
    comparison = pd.DataFrame(per_mode_mean)
    return comparison[[mode for mode in LR_MODES if mode in comparison.columns]]


def main() -> None:
    """Build the ablation report card and ranking per protocol, plus a cross-protocol read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="r2", choices=METRIC_COLUMNS, help="metric to show")
    parser.add_argument(
        "--metrics-csv", type=Path, default=ABLATION_METRICS_CSV, help="tidy ablation metrics CSV"
    )
    parser.add_argument(
        "--mpnn-lr-mode", default="all", choices=(*LR_MODES, "all"),
        help="finetune LR protocol to report; 'all' builds every present protocol and, when more "
        "than one is present, a cross-protocol comparison",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate on the ablations")
    frame = pd.read_csv(args.metrics_csv)

    # tag each result with its finetune LR protocol and strip the suffix so the seed collapse and
    # ablation-name parsing downstream operate on the frozen-style base label
    parsed = frame["flavor"].map(parse_lr_mode)
    frame["mpnn_lr_mode"] = parsed.map(lambda base_mode: base_mode[1])
    frame["flavor"] = parsed.map(lambda base_mode: base_mode[0])

    present = [mode for mode in LR_MODES if mode in set(frame["mpnn_lr_mode"])]
    requested = list(LR_MODES) if args.mpnn_lr_mode == "all" else [args.mpnn_lr_mode]
    modes = [mode for mode in present if mode in requested]
    if not modes:
        raise SystemExit(f"no ablation results for protocol(s) {requested} in {args.metrics_csv}")

    # one report card + ranking per protocol; keep each protocol's mean series for the comparison
    per_mode_mean: dict[str, pd.Series] = {}
    for mode in modes:
        mean_series = report_one_mode(frame[frame["mpnn_lr_mode"] == mode], args.metric, mode)
        if mean_series is None:
            logger.warning("protocol %s: no ablation results; skipping", mode)
            continue
        per_mode_mean[mode] = mean_series

    if not per_mode_mean:
        raise SystemExit("no ablation results found in the metrics CSV")

    # cross-protocol read: does the winning recipe hold once the backbone can move?
    if len(per_mode_mean) > 1:
        comparison = mode_comparison(per_mode_mean)
        comp_csv = PLOTS_DIR / f"prescaling_mode_comparison_{args.metric}.csv"
        comparison.to_csv(comp_csv)
        plot_mode_comparison(
            comparison, args.metric, PLOTS_DIR / f"prescaling_mode_comparison_{args.metric}.png"
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
