---
tags: [flavor, status/red]
---
# erg

> **Summary:** 315-dim extended reduced-graph pharmacophore, regressed with MSE. A continuous
> pharmacophore target that abstracts the molecule to its feature topology.

- Target: 315 continuous values · Loss: MSE · Source: direct compute (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

Pharmacophore features encode binding-relevant pattern, so this may favor target-engagement
endpoints: [[Potency]], [[CYP Inhibition]], and [[hERG]], more than the purely property-driven
families.

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.265 ± 0.005, significantly **below** the 0.294 ± 0.010 stock baseline, the only
non-fingerprint flavor with a significant frozen deficit; reduced brings it back to a tie (0.322)
and unlocked is a tie too (0.303). Beats stock on 16 of
32 endpoint-columns frozen, so its deficit is spread rather than concentrated.

The one place it stands out is the [[PXR External Test]], where it is the second-best model on
phase 1 (0.336) and the only flavor other than [[surrogate_adme]] above stock there.

## Related

- Regime: [[Shared Corpus and Regime]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
