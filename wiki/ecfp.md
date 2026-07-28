---
tags: [flavor, status/red]
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

## Result (frozen sweep, 5 seeds)
Frozen mean R² 0.214 ± 0.020, significantly below the 0.294 ± 0.010 stock baseline, and
significantly below under reduced (0.253) and unlocked (0.243) too: worst or near-worst on the
card under every protocol. The [[PXR External Test]] is starker still, at 0.036 on phase 1
against stock's 0.325.

The clearest confirmation of the leaky-pretext prior in the study. ECFP4 bits are a
deterministic function of the graph the network already sees, so predicting them asks it to
learn almost nothing.

## Related

- Sibling leaky fingerprints: [[atompair]], [[pubchem]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
