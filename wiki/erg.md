---
tags: [flavor, status/planned]
---
# erg

> **Summary:** 315-dim extended reduced-graph pharmacophore, regressed with MSE. A continuous
> pharmacophore target that abstracts the molecule to its feature topology.

- Target: 315 continuous values · Loss: MSE · Source: direct compute (scikit-fingerprints)
- Calculator: `pretraining/features/skfp_targets.py`

## Hypothesis

Pharmacophore features encode binding-relevant pattern, so this may favor target-engagement
endpoints: [[Potency]], [[CYP Inhibition]], and [[hERG]], more than the purely property-driven
families.

## Related

- Regime: [[Shared Corpus and Regime]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
