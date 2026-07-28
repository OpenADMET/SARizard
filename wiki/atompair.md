---
tags: [flavor, status/yellow]
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

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.270 ± 0.011 against a 0.294 ± 0.010 stock baseline, below it but not
significantly once the flavors are corrected as one family (p=0.08); reduced and unlocked are
statistical ties too, so it never separates from stock under any protocol. Beats stock on 11 of 32 endpoint-columns frozen, its best being
expansionrx MPPB (+0.25). Consistent with the leaky-and-weak-pretext prior for binary
fingerprint targets.

## Related

- Sibling leaky fingerprints: [[ecfp]], [[pubchem]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
