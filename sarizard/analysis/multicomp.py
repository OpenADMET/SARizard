"""Multiple-comparison tests for many-treatments-against-one-control comparisons.

Every significance question in this repo has the same shape: several flavors (or ablation
recipes, or external foundations) are each compared against one stock-CheMeleon control, on the
same endpoint, from the same finetune seeds. Testing each pair in isolation ignores that the
comparisons form a family, so the chance of at least one false positive grows with the number of
flavors. Dunnett's test is built for exactly this design: it tests every treatment against the
control while controlling the family-wise error rate across the family, and it pools the
within-group variance across all groups, which buys error degrees of freedom that a two-group
Welch test cannot reach at five seeds.

The caller decides what a family is. The report card treats one endpoint row as one family (each
row asks its own question, so its flavors are corrected together), which leaves error across rows
uncontrolled by design.

Dunnett's pooled variance assumes the groups share a common variance. That holds well enough for
comparisons within one endpoint, where every group is the same metric on the same molecules, but
it is an assumption, not something five seeds per group can verify: Levene has almost no power at
this sample size. Report the observed spread alongside any result that leans on it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import dunnett

# minimum observations per group for a comparison to be testable at all; below this the group has
# no spread to estimate and the cell is reported as undefined rather than guessed at
MIN_GROUP_SIZE = 2

# scipy's Dunnett p-values come from a quasi-Monte-Carlo integration of the multivariate t, so
# they carry a small amount of Monte Carlo noise; a fixed seed makes a re-render reproducible
DUNNETT_SEED = 0


def dunnett_pvalues(samples: dict[str, np.ndarray], control: np.ndarray) -> dict[str, float]:
    """Two-sided Dunnett p-values for each treatment against a shared control.

    All treatments in ``samples`` are corrected together as one family, so a p-value here is
    already family-wise and should be compared against the nominal alpha directly, with no
    further correction.

    Parameters
    ----------
    samples : dict of str to numpy.ndarray
        Treatment label to that treatment's observations (one per finetune seed). Treatments with
        fewer than ``MIN_GROUP_SIZE`` observations are dropped from the family and returned as
        NaN, so they neither borrow from nor contribute to the pooled variance.
    control : numpy.ndarray
        The control group's observations, shared by every comparison.

    Returns
    -------
    dict of str to float
        One p-value per key of ``samples``; NaN where the comparison is undefined (too few
        observations in either group, or no residual variance anywhere in the family, which
        leaves the pooled estimate at zero).

    Examples
    --------
    >>> import numpy as np
    >>> control = np.array([0.30, 0.29, 0.31, 0.30, 0.29])
    >>> samples = {
    ...     "better": np.array([0.40, 0.41, 0.39, 0.40, 0.41]),
    ...     "same": np.array([0.30, 0.31, 0.29, 0.30, 0.30]),
    ... }
    >>> p = dunnett_pvalues(samples, control)
    >>> bool(p["better"] < 0.05), bool(p["same"] > 0.05)
    (True, True)
    """
    results = dict.fromkeys(samples, float("nan"))
    control = np.asarray(control, dtype=float)
    if control.size < MIN_GROUP_SIZE:
        return results

    # keep only the treatments that can carry a comparison; the family size is what survives here,
    # so a dropped treatment also shrinks the correction the others pay
    testable = {
        label: np.asarray(values, dtype=float)
        for label, values in samples.items()
        if np.size(values) >= MIN_GROUP_SIZE
    }
    if not testable:
        return results

    # Dunnett pools the within-group variance; with no residual variance anywhere the pooled
    # estimate is zero and the statistic is undefined, so report NaN rather than a spurious 0 or 1
    groups = [control, *testable.values()]
    residual = sum(float(np.sum((g - g.mean()) ** 2)) for g in groups)
    if residual <= 0.0:
        return results

    labels = list(testable)
    outcome = dunnett(*testable.values(), control=control, rng=DUNNETT_SEED)
    for label, pvalue in zip(labels, np.atleast_1d(outcome.pvalue), strict=True):
        results[label] = float(pvalue)
    return results
