---
tags: [flavor, status/planned]
---
# ecfp

> **Summary:** 2048-bit ECFP4 circular substructure fingerprint, trained with masked BCE. A
> binary fingerprint target is deterministic from the input graph, so it is a leaky and likely
> weak self-supervised pretext; its report-card position is itself a result.

- Target: 2048 binary bits · Loss: BCE · Source: direct compute (scikit-fingerprints)
- Calculator: `sarizard/pretraining/features/skfp_targets.py`

## Hypothesis

Substructure presence carries motif information relevant to [[Potency]] and [[CYP Inhibition]],
but because the target is a deterministic function of the graph the backbone already sees, the
pretext may teach little. Expect a weak-to-baseline column.

## Related

- Sibling leaky fingerprints: [[atompair]], [[pubchem]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
