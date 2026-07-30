"""Summarize the PXR external-test rerun: per-flavor, per-phase R2/MAE against the stock baseline.

Reads the dedicated tidy metrics CSV that ``evaluate`` writes for the ``pxr_ext__*`` labels and
prints, per challenge phase, one row per flavor with its mean test R2 and MAE across the finetune
seeds (with across-seed std), the delta vs stock, and its family-wise significance. Printed to the
terminal as a standalone arm; it never touches the report card or ``results/metrics.csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sarizard.analysis.multicomp import MIN_GROUP_SIZE, dunnett_pvalues
from sarizard.analysis.paths import parse_seed_variant

logger = logging.getLogger(__name__)

LABEL_PREFIX = "pxr_ext__"
STOCK_FLAVOR = "chemeleon_stock"
PHASES = {"pxr_phase1_st": 1, "pxr_phase2_st": 2}

# a flavor whose per-seed R2 does not differ from stock's at this level (Dunnett's test,
# family-wise within the phase) prints without a significance star
SIGNIFICANCE_ALPHA = 0.05


def _flavor_of(label: str) -> str:
    """Return the base flavor of a ``pxr_ext__<flavor>__s<seed>`` label."""
    base = parse_seed_variant(label)[0]
    return base[len(LABEL_PREFIX):] if base.startswith(LABEL_PREFIX) else base


def _per_seed(frame: pd.DataFrame, flavor: str, metric: str) -> np.ndarray:
    """Return one metric value per seed for a flavor within an already phase-filtered frame."""
    rows = frame[frame["flavor"].map(_flavor_of) == flavor]
    return rows[metric].to_numpy(dtype=np.float64)


def print_phase(frame: pd.DataFrame, phase_recipe: str, phase: int) -> None:
    """Print the flavor ranking for one challenge phase, each flavor tested against stock.

    One phase is one comparison family: every flavor is measured against the same stock seeds on
    the same held-out molecules, so they are corrected together with Dunnett's test rather than
    tested pairwise. A pairwise test here would run one uncorrected comparison per flavor and let
    the false-positive count grow with the number of flavors shown.
    """
    sub = frame[frame["recipe"] == phase_recipe]
    if sub.empty:
        logger.warning("no rows for %s", phase_recipe)
        return
    flavors = sorted(set(sub["flavor"].map(_flavor_of)))
    stock_r2 = _per_seed(sub, STOCK_FLAVOR, "r2")

    # the family is every non-stock flavor with enough seeds to carry a comparison; a flavor
    # dropped here also shrinks the correction the others pay
    samples = {
        flavor: values
        for flavor in flavors
        if flavor != STOCK_FLAVOR
        and (values := _per_seed(sub, flavor, "r2")).size >= MIN_GROUP_SIZE
    }
    pvalues: dict[str, float] = {}
    if stock_r2.size >= MIN_GROUP_SIZE and samples:
        pvalues = dunnett_pvalues(samples, stock_r2)

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
        if fl in pvalues:
            # a NaN p-value means the comparison is undefined, which prints as not significant
            star = "*" if pvalues[fl] <= SIGNIFICANCE_ALPHA else " "
            delta = f"{r2m - stock_r2.mean():+.3f}{star}"
        print(f"{fl:<20}{r2m:>8.3f}{r2s:>7.3f}{maem:>8.3f}{maes:>7.3f}{n:>6}{delta:>14}")
    print(
        f"  (* delta R2 vs stock significant at Dunnett p<={SIGNIFICANCE_ALPHA:g}, "
        f"family-wise across the {len(pvalues)} flavors in this phase)"
    )


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
