"""Compare the finetune learning-rate experiments: frozen vs reduced vs unlocked MPNN.

The flavor sweep finetunes with a frozen backbone (``mpnn_lr=0``). The LR experiments repeat
the finetuning from the same foundations with the backbone partially unfrozen (``reduced``,
``mpnn_lr = ffn_lr/10``) or fully unfrozen (``unlocked``, ``mpnn_lr = ffn_lr``). This reads the
combined metrics CSV, whose ``flavor`` column holds the frozen labels (``<flavor>__s<seed>``)
and the LR labels (``lr_<mode>__<flavor>__s<seed>``), averages the seeds, and reports how each
mode moves a metric relative to the frozen baseline: a per-mode mean delta and win count, plus
the full ``(flavor, endpoint)`` by mode matrix. The gap between frozen and unlocked is the cost
of the clean frozen ablation; a positive reduced delta says partial unfreezing helps.

Depends only on pandas, numpy, and matplotlib, so it runs without openadmet or a GPU.

Usage:
    python -m sarizard.analysis.lr_report --metric r2
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
from sarizard.analysis.paths import PLOTS_DIR, RESULTS_DIR, parse_seed_variant  # noqa: E402
from sarizard.pretraining.flavors import flavor_names  # noqa: E402

logger = logging.getLogger(__name__)

LR_METRICS_CSV = RESULTS_DIR / "lr_metrics.csv"
BASELINE_MODE = "frozen"
MODE_ORDER = ("frozen", "reduced", "unlocked")


def parse_lr_variant(label: str) -> tuple[str, str]:
    """Split a result label into ``(mpnn_lr_mode, flavor)``.

    Frozen sweep labels (``<flavor>__s<seed>``) have no ``lr_`` prefix and map to ``frozen``;
    LR labels (``lr_<mode>__<flavor>__s<seed>``) carry the mode in their prefix.
    """
    base, _seed = parse_seed_variant(label)
    if base.startswith("lr_"):
        prefix, _, flavor = base.partition("__")
        return prefix[len("lr_"):], flavor
    return BASELINE_MODE, base


def mode_flavor_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Tag each row with its mode and flavor, keep known flavors, average seeds.

    Returns one row per ``(flavor, endpoint, mode)`` with the seed-averaged metric.
    """
    frame = frame.copy()
    parsed = frame["flavor"].map(parse_lr_variant)
    frame["mode"] = parsed.map(lambda mf: mf[0])
    frame["flavor_base"] = parsed.map(lambda mf: mf[1])
    frame = frame[frame["flavor_base"].isin(set(flavor_names()))]
    return frame.groupby(["flavor_base", "endpoint", "mode"])[metric].mean().reset_index()


def build_mode_matrix(grouped: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot the seed-averaged metric to ``(flavor · endpoint)`` rows by mode columns."""
    grouped = grouped.copy()
    grouped["row"] = grouped["flavor_base"] + " · " + grouped["endpoint"]
    pivot = grouped.pivot_table(index="row", columns="mode", values=metric, aggfunc="mean")
    return pivot[[mode for mode in MODE_ORDER if mode in pivot.columns]]


def summarize(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-mode mean metric and, for the non-frozen modes, delta and wins vs frozen.

    ``wins_vs_frozen`` counts ``(flavor, endpoint)`` pairs where the mode beats frozen in the
    metric's own direction (higher for r2/spearman/kendall, lower for the error metrics).
    """
    higher = metric in HIGHER_IS_BETTER
    records: list[dict] = []
    for mode in pivot.columns:
        column = pivot[mode]
        record = {"mode": mode, "mean": float(column.mean()), "n": int(column.notna().sum())}
        if BASELINE_MODE in pivot.columns and mode != BASELINE_MODE:
            delta = column - pivot[BASELINE_MODE]
            wins = (delta > 0) if higher else (delta < 0)
            record["mean_delta_vs_frozen"] = float(delta.mean())
            record["wins_vs_frozen"] = int(wins.sum())
        records.append(record)
    return pd.DataFrame(records).set_index("mode")


def plot_deltas(summary: pd.DataFrame, metric: str, out_png: Path) -> None:
    """Bar chart of each non-frozen mode's mean metric delta versus the frozen baseline."""
    label = METRIC_LABELS.get(metric, metric)
    deltas = summary.loc[summary.index != BASELINE_MODE, "mean_delta_vs_frozen"].dropna()
    fig, ax = plt.subplots(figsize=(7.0, 0.7 * len(deltas) + 1.6), constrained_layout=True)
    colors = ["tab:green" if value > 0 else "tab:red" for value in deltas]
    ax.barh(deltas.index, deltas.to_numpy(), color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.tick_params(labelsize=13)
    ax.set_xlabel(f"mean {label} delta vs frozen", fontsize=14)
    ax.set_title(f"Finetune LR experiments ({label}); green beats frozen", fontsize=15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the LR-experiment comparison for the requested metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="r2", choices=METRIC_COLUMNS, help="metric to compare")
    parser.add_argument(
        "--metrics-csv", type=Path, default=LR_METRICS_CSV, help="tidy combined metrics CSV"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.metrics_csv.exists():
        raise SystemExit(f"{args.metrics_csv} not found; run analysis.evaluate on the LR results")
    grouped = mode_flavor_frame(pd.read_csv(args.metrics_csv), args.metric)
    pivot = build_mode_matrix(grouped, args.metric)
    if pivot.empty:
        raise SystemExit("no LR-experiment results found in the metrics CSV")
    if BASELINE_MODE not in pivot.columns:
        logger.warning("no frozen baseline rows found; deltas will be omitted")

    summary = summarize(pivot, args.metric)
    matrix_csv = PLOTS_DIR / f"lr_report_{args.metric}.csv"
    summary_csv = PLOTS_DIR / f"lr_ranking_{args.metric}.csv"
    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(matrix_csv)
    summary.to_csv(summary_csv)
    if "mean_delta_vs_frozen" in summary.columns:
        plot_deltas(summary, args.metric, PLOTS_DIR / f"lr_ranking_{args.metric}.png")
    logger.info("wrote %s and %s", matrix_csv, summary_csv)


if __name__ == "__main__":
    main()
