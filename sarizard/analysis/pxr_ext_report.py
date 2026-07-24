"""Summarize the PXR external-test rerun: per-flavor, per-phase R2/MAE against the stock baseline.

Reads the dedicated tidy metrics CSV that ``evaluate`` writes for the ``pxr_ext__*`` labels and
prints, per challenge phase, one row per flavor with its mean test R2 and MAE across the finetune
seeds (with across-seed std), the delta vs stock, and its Welch t-test. Printed to the terminal as
a standalone arm; it never touches the report card or ``results/metrics.csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sarizard.analysis.paths import parse_seed_variant

logger = logging.getLogger(__name__)

LABEL_PREFIX = "pxr_ext__"
STOCK_FLAVOR = "chemeleon_stock"
PHASES = {"pxr_phase1_st": 1, "pxr_phase2_st": 2}


def _flavor_of(label: str) -> str:
    """Return the base flavor of a ``pxr_ext__<flavor>__s<seed>`` label."""
    base = parse_seed_variant(label)[0]
    return base[len(LABEL_PREFIX):] if base.startswith(LABEL_PREFIX) else base


def _per_seed(frame: pd.DataFrame, flavor: str, metric: str) -> np.ndarray:
    """Return one metric value per seed for a flavor within an already phase-filtered frame."""
    rows = frame[frame["flavor"].map(_flavor_of) == flavor]
    return rows[metric].to_numpy(dtype=np.float64)


def print_phase(frame: pd.DataFrame, phase_recipe: str, phase: int) -> None:
    """Print the flavor ranking for one challenge phase, each flavor tested against stock."""
    sub = frame[frame["recipe"] == phase_recipe]
    if sub.empty:
        logger.warning("no rows for %s", phase_recipe)
        return
    flavors = sorted(set(sub["flavor"].map(_flavor_of)))
    stock_r2 = _per_seed(sub, STOCK_FLAVOR, "r2")

    print(f"\nPXR external test, phase {phase} ({phase_recipe})\n{'=' * 74}")
    print(f"{'flavor':<20}{'R2':>8}{'std':>7}{'MAE':>8}{'std':>7}{'seeds':>6}{'dR2 vs stock':>14}")
    stats_rows = []
    for fl in flavors:
        r2 = _per_seed(sub, fl, "r2")
        mae = _per_seed(sub, fl, "mae")
        if r2.size == 0:
            continue
        r2_std = float(r2.std(ddof=1)) if r2.size > 1 else 0.0
        mae_std = float(mae.std(ddof=1)) if mae.size > 1 else 0.0
        stats_rows.append((fl, float(r2.mean()), r2_std, float(mae.mean()), mae_std, r2.size))
    # rank by mean R2 descending; stock is shown in place so its row is easy to find
    for fl, r2m, r2s, maem, maes, n in sorted(stats_rows, key=lambda t: t[1], reverse=True):
        delta = ""
        if fl != STOCK_FLAVOR and stock_r2.size >= 2:
            fr2 = _per_seed(sub, fl, "r2")
            if fr2.size >= 2:
                t = stats.ttest_ind(fr2, stock_r2, equal_var=False)
                star = "*" if t.pvalue <= 0.05 else " "
                delta = f"{r2m - stock_r2.mean():+.3f}{star}"
        print(f"{fl:<20}{r2m:>8.3f}{r2s:>7.3f}{maem:>8.3f}{maes:>7.3f}{n:>6}{delta:>14}")
    print("  (* delta R2 vs stock significant at Welch p<=0.05)")


def main() -> None:
    """Print the PXR external-test comparison from a metrics CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True, help="pxr_ext tidy metrics CSV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frame = pd.read_csv(args.metrics_csv)
    for phase_recipe, phase in PHASES.items():
        print_phase(frame, phase_recipe, phase)


if __name__ == "__main__":
    main()
