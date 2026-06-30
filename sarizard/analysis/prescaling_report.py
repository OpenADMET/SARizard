"""Compare prescaling ablations: endpoints by ablation, plus an overall ranking.

Reads the tidy metrics CSV produced by ``analysis.evaluate`` for the ablation results
(where the ``flavor`` column holds ``ablation_<name>`` labels) and renders two artifacts:

- a report-card heatmap, endpoints (rows) by ablation (columns), reusing the row-relative
  coloring from ``analysis.report_card``;
- a summary CSV and bar chart ranking each ablation by its mean metric across endpoints and
  the number of endpoints it wins, the read used to pick the production prescaling recipe.

Depends only on pandas, numpy, and matplotlib, so it runs without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.prescaling_report --metric r2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - set backend before importing pyplot
import pandas as pd  # noqa: E402

from sarizard.analysis.metrics_spec import (  # noqa: E402
    HIGHER_IS_BETTER,
    METRIC_COLUMNS,
    METRIC_LABELS,
)
from sarizard.analysis.paths import PLOTS_DIR, RESULTS_DIR, ablation_label  # noqa: E402
from sarizard.analysis.report_card import build_matrix, plot_report_card  # noqa: E402
from sarizard.pretraining.prescaling import ablation_names  # noqa: E402

logger = logging.getLogger(__name__)

ABLATION_METRICS_CSV = RESULTS_DIR / "ablation_metrics.csv"


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


def main() -> None:
    """Build the ablation report card and ranking for the requested metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="r2", choices=METRIC_COLUMNS, help="metric to show")
    parser.add_argument(
        "--metrics-csv", type=Path, default=ABLATION_METRICS_CSV, help="tidy ablation metrics CSV"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate on the ablations")
    frame = pd.read_csv(args.metrics_csv)

    # order columns by the ablation ladder, keep only those that produced results
    columns = [ablation_label(name) for name in ablation_names()]
    pivot = build_matrix(frame, args.metric, columns=columns)
    pivot.columns = [_strip(col) for col in pivot.columns]
    if pivot.empty or pivot.shape[1] == 0:
        raise SystemExit("no ablation results found in the metrics CSV")

    out_png = PLOTS_DIR / f"prescaling_report_{args.metric}.png"
    plot_report_card(pivot, args.metric, out_png, out_png.with_suffix(".csv"))

    summary = rank_ablations(pivot, args.metric)
    summary_csv = PLOTS_DIR / f"prescaling_ranking_{args.metric}.csv"
    summary.to_csv(summary_csv)
    plot_ranking(summary, args.metric, PLOTS_DIR / f"prescaling_ranking_{args.metric}.png")
    logger.info("best by mean %s: %s", args.metric, summary.index[0])
    logger.info("wrote %s and ranking artifacts", out_png)


if __name__ == "__main__":
    main()
