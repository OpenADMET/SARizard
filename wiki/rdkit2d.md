---
tags: [flavor, status/planned]
---
# rdkit2d

> **Summary:** 200 curated RDKit 2D physicochemical descriptors, regressed with MSE. A compact
> physchem target, the low-dimensional counterpart to [[osmordred]].

- Target: 200 continuous descriptors · Loss: MSE · Source: direct compute (scikit-fingerprints)
- Calculator: `pretraining/features/skfp_targets.py`

## Hypothesis

A focused physchem block should serve [[Solubility]] and [[Lipophilicity]] well, and tests
whether 200 well-chosen descriptors match the much wider [[osmordred]] target on
[[Permeability]] and [[Clearance]].

## Related

- Regime: [[Shared Corpus and Regime]] · Wider sibling: [[osmordred]]
- Scored on the [[Report Card]]; stacked in the [[Meta-Model]]
