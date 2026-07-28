---
tags: [flavor, status/yellow]
---
# osmordred PCA targets

> **Summary:** Three registry flavors (`osmordred_pca80`, `osmordred_pca90`, `osmordred_pca95`)
> that pretrain against PCA component scores of the [[osmordred]] descriptor matrix instead of
> the raw 3585-dim block. They ask whether a smaller, decorrelated target trains a better
> foundation, or just a cheaper one. One page covers all three: they differ only in the
> explained-variance threshold and behave identically.

- Target: PCA component scores, MSE · Source: derived from [[osmordred]]'s cached target, no
  calculator of its own (the `derived_from` field on the flavor registry)
- Component counts (`cache/targets/osmordred_pca_summary.json`): 70 at 80.1% variance, 147 at
  90.1%, 237 at 95.0%
- On the [[Report Card]] and in the [[Meta-Model]] as three ordinary columns

## How the target is built

`sarizard/pretraining/pca_target.py` runs the descriptor matrix through the full prescaling
pipeline (the `full` recipe), then fits PCA on the result. The fit uses **train rows only**, once
at the largest requested threshold, and the smaller thresholds slice a prefix of the fitted
components; the same transform is applied to val. This keeps the no-leakage invariant in
[[Shared Corpus and Regime]]: the PCA basis never sees a held-out molecule. Because these
flavors arrive pre-prescaled, the split stage skips its own prescaling step for them.

## Result

Mean R² across the 32 endpoint-columns, 5 finetune seeds, with full [[osmordred]] and the
same-protocol [[Stock CheMeleon]] baseline for reference:

| flavor | frozen | reduced | unlocked |
|---|---|---|---|
| osmordred (3585 dims) | 0.301 ± 0.013 | 0.341 ± 0.010 | 0.298 ± 0.022 |
| osmordred_pca80 (70) | 0.315 ± 0.010 | 0.338 ± 0.009 | 0.304 ± 0.014 |
| osmordred_pca90 (147) | 0.315 ± 0.022 | 0.336 ± 0.008 | 0.287 ± 0.025 |
| osmordred_pca95 (237) | 0.304 ± 0.009 | 0.345 ± 0.019 | 0.283 ± 0.014 |
| [[Stock CheMeleon]] | 0.294 ± 0.010 | 0.316 ± 0.014 | 0.337 ± 0.008 |

**Compression is free, not a win.** All three land within one seed standard deviation of full
osmordred under every protocol, and the thresholds do not order consistently with each other
(pca80 leads frozen and unlocked, pca95 leads reduced), so the spread between them is seed noise
rather than an explained-variance effect. A 15-to-50x narrower, decorrelated target trains an
equally transferable foundation for a fraction of the pretraining cost, but not a better one.

The secondary motivation does not apply. A narrow target changes the masked-pretext dropout's
keep-count-per-step math, but all three counts stay well above the 30-dim override threshold in
[[Shared Corpus and Regime]], so they pretrain under the standard regime exactly as full
osmordred does.

On the [[PXR External Test]] the three sit in the lower half and below stock on both phases,
tracking [[osmordred]] rather than diverging from it.

## Related

- Base flavor: [[osmordred]] · Regime: [[Shared Corpus and Regime]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
