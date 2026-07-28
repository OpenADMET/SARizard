---
tags: [flavor, status/red]
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

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.210 ± 0.024, significantly below the 0.294 ± 0.010 stock baseline, and
significantly below under reduced (0.276). Under unlocked it climbs to 0.304 and the deficit is
no longer significant, the largest frozen-to-unlocked recovery of any flavor: its pretext needs
backbone adaptation to pay off at all. Beats stock on only 8 of 32
endpoint-columns frozen, the fewest of any flavor. The weakest continuous-descriptor target in
the study.

Holistic 3D geometry on a single generated conformer per molecule appears to carry little that
transfers; the conformer dependence also makes the target only approximately reproducible.

## Related

- 3D siblings: [[usrcat]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
