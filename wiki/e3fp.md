---
tags: [flavor, status/planned]
---
# e3fp

> **Summary:** 1024-bit E3FP 3D circular substructure fingerprint, trained with masked BCE.
> Binary and conformer-dependent: it combines the leaky-pretext caveat of the fingerprint
> flavors with the approximate reproducibility of the 3D flavors.

- Target: 1024 binary bits · Loss: BCE · Source: direct compute, conformers (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

3D substructure bits add geometry the 2D fingerprints lack, but the target is still largely
determined by the graph plus a conformer, so expect a weak column with a possible edge on
shape-driven endpoints ([[Potency]], [[hERG]]) over the 2D leaky fingerprints.

## Related

- Leaky fingerprints: [[ecfp]], [[atompair]], [[pubchem]] · 3D siblings: [[usrcat]], [[whim]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
