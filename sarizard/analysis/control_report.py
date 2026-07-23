"""Print the osmordred_surrogate control comparison to the terminal.

The ``surrogate_adme`` flavor is confounded two ways against the sweep ``osmordred``: it
pretrains on the Novartis molecules (a different corpus) and on the 25 surrogate ADME
predictions (a different target). The ``osmordred_surrogate`` control holds the target
identical to the sweep osmordred (3585 osmordred descriptors) while borrowing
``surrogate_adme``'s Novartis corpus, so the only thing that separates it from the sweep
osmordred is the corpus, and the only thing that separates it from ``surrogate_adme`` is the
target. Reading its downstream R-squared against those two arms tells whether the surrogate
flavor's strength came from its chemical space or from the surrogate target itself.

This module reads the tidy metrics CSV that ``evaluate`` writes for the relevant labels and
prints, per condition, the mean test R-squared over endpoints averaged across finetune seeds
(with the across-seed standard deviation), a Welch t-test of the control against the stock
baseline, and an automated read of which arm the control lands nearest. It never writes to the
report card or the shared ``results/metrics.csv``.
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


def per_seed_endpoint_means(frame: pd.DataFrame, base: str) -> np.ndarray:
    """Return one mean-R-squared value per finetune seed for a base flavor label.

    Groups the tidy metrics rows whose flavor label collapses to ``base`` by their seed and
    means the ``r2`` column over that seed's endpoint rows, so each returned value is one
    seed's endpoint-averaged R-squared. Rows without a seed suffix (a lone unseeded run) count
    as one group.

    Parameters
    ----------
    frame : pandas.DataFrame
        Tidy metrics with ``flavor`` and ``r2`` columns (one row per flavor/seed/endpoint).
    base : str
        Base flavor label to select, e.g. ``osmordred_surrogate`` or ``chemeleon_stock``.

    Returns
    -------
    numpy.ndarray
        One R-squared per seed, ascending by seed. Empty if no rows match ``base``.
    """
    parsed = frame["flavor"].map(parse_seed_variant)
    bases = parsed.map(lambda pair: pair[0])
    seeds = parsed.map(lambda pair: pair[1])
    selected = frame.loc[bases == base, ["r2"]].assign(seed=seeds[bases == base].to_numpy())
    if selected.empty:
        return np.empty(0, dtype=np.float64)
    per_seed = selected.groupby("seed", dropna=False)["r2"].mean()
    return per_seed.to_numpy(dtype=np.float64)


def _summarize(frame: pd.DataFrame, base: str) -> tuple[float, float, int]:
    """Return ``(mean, std, n_seeds)`` of a base flavor's per-seed endpoint means."""
    values = per_seed_endpoint_means(frame, base)
    if values.size == 0:
        return float("nan"), float("nan"), 0
    # std over seeds (ddof=1) needs >=2 seeds; report 0.0 for a single seed rather than NaN
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), std, int(values.size)


def print_comparison(
    frame: pd.DataFrame,
    subject: str,
    baseline: str,
    context: dict[str, str],
) -> None:
    """Print the control comparison table, a Welch test, and the chemical-space read.

    Parameters
    ----------
    subject : str
        The control flavor's base label (``osmordred_surrogate``).
    baseline : str
        The reference base label to test against (``chemeleon_stock``).
    context : dict of str to str
        Additional base labels to print for context, mapped to a short description
        (e.g. ``{"surrogate_adme": "Novartis corpus, surrogate target"}``).
    """
    order = [subject, baseline, *context]
    descriptions = {
        subject: "Novartis corpus, osmordred target (the control)",
        baseline: "stock CheMeleon (baseline)",
        **context,
    }

    print(f"\nosmordred_surrogate control: chemical space vs surrogate target\n{'=' * 68}")
    print(f"{'condition':<24}{'mean R2':>9}{'±std':>8}{'seeds':>7}  description")
    stats_by_base: dict[str, tuple[float, float, int]] = {}
    for base in order:
        mean, std, n = _summarize(frame, base)
        stats_by_base[base] = (mean, std, n)
        print(f"{base:<24}{mean:>9.3f}{std:>8.3f}{n:>7}  {descriptions.get(base, '')}")

    # Welch t-test of the control's per-seed means against the baseline's: unequal variances
    # and possibly unequal seed counts, so Welch rather than a pooled or paired test (the seeds
    # are independent finetune replicates, not paired across conditions)
    subj_seeds = per_seed_endpoint_means(frame, subject)
    base_seeds = per_seed_endpoint_means(frame, baseline)
    subj_mean = stats_by_base[subject][0]
    base_mean = stats_by_base[baseline][0]
    print(f"\nControl vs baseline ({baseline}):")
    if subj_seeds.size >= 2 and base_seeds.size >= 2:
        test = stats.ttest_ind(subj_seeds, base_seeds, equal_var=False)
        verdict = "significant" if test.pvalue <= 0.05 else "not significant"
        print(
            f"  delta mean R2 = {subj_mean - base_mean:+.3f}"
            f"  (Welch t={test.statistic:+.2f}, p={test.pvalue:.3f}, {verdict})"
        )
    else:
        print(f"  delta mean R2 = {subj_mean - base_mean:+.3f}  (too few seeds for a t-test)")

    # chemical-space read: whichever context arm the control lands nearest tells the story
    if context:
        distances = {
            base: abs(subj_mean - stats_by_base[base][0])
            for base in context
            if stats_by_base[base][2] > 0
        }
        if distances:
            nearest = min(distances, key=distances.get)
            print(
                f"\nRead: the control lands nearest {nearest} "
                f"(|delta| = {distances[nearest]:.3f}); "
                f"{context.get(nearest, '')}."
            )


def main() -> None:
    """Print the osmordred_surrogate control comparison from a metrics CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="tidy metrics CSV holding the control, baseline, and context labels",
    )
    parser.add_argument("--subject", default="osmordred_surrogate", help="control base label")
    parser.add_argument("--baseline", default="chemeleon_stock", help="baseline base label")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frame = pd.read_csv(args.metrics_csv)
    context = {
        "surrogate_adme": "same corpus as the control, surrogate ADME target",
        "osmordred": "same target as the control, PubChem sweep corpus",
    }
    print_comparison(frame, args.subject, args.baseline, context)


if __name__ == "__main__":
    main()
