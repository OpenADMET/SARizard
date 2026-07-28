---
tags: [flavor, status/yellow]
---
# pubchem

> **Summary:** 881-bit PubChem substructure keys, trained with masked BCE. The coarsest of the
> binary fingerprints, a fixed dictionary of substructure presence. Deterministic from the
> graph, so a leaky and likely weak pretext.

- Target: 881 binary bits · Loss: BCE · Source: direct compute (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

A small fixed key set is the lowest-capacity fingerprint target. If even this matches the
richer leaky fingerprints, it confirms that the fingerprint family teaches little beyond what
the graph already provides.

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.270 ± 0.014 against a 0.294 ± 0.010 stock baseline, below it but not
significantly once corrected family-wise (p=0.08); reduced (0.303) and unlocked (0.306) are ties
as well, so it never separates from stock under any protocol. Beats stock on only 12 of 32 endpoint-columns frozen.
The best of the four binary fingerprint flavors, which is a low bar. Consistent with the
leaky-and-weak-pretext prior: a key-set the message-passing network can read off the graph
teaches it little.

## Related

- Sibling leaky fingerprints: [[ecfp]], [[atompair]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
