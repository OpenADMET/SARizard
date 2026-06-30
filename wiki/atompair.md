---
tags: [flavor, status/planned]
---
# atompair

> **Summary:** 2048-bit topological atom-pair fingerprint, trained with masked BCE. Like
> [[ecfp]], a binary fingerprint is deterministic from the graph, so it is a leaky, likely weak
> pretext whose report-card position is a result, not a bug.

- Target: 2048 binary bits · Loss: BCE · Source: direct compute (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

Atom-pair encoding emphasizes through-bond distance pattern, a different bias than ECFP's local
environments. It tests whether the choice of fingerprint among leaky pretexts matters at all.

## Related

- Sibling leaky fingerprints: [[ecfp]], [[pubchem]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
