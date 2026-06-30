---
tags: [flavor, status/planned]
---
# whim

> **Summary:** 114 WHIM descriptors: holistic 3D geometry capturing size, shape, symmetry, and
> atom-property distribution, regressed with MSE. A conformer-dependent target (RDKit ETKDG +
> MMFF94, seeded), so only approximately reproducible.

- Target: 114 continuous values · Loss: MSE · Source: direct compute, conformers (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

Whole-molecule geometry may inform distribution and shape-sensitive endpoints: [[Permeability]],
[[Potency]], and [[hERG]]. It contrasts with [[usrcat]]'s shape-plus-pharmacophore framing on
the same conformers.

## Related

- 3D siblings: [[usrcat]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
