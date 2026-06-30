---
tags: [flavor, status/planned]
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

## Related

- Sibling leaky fingerprints: [[ecfp]], [[atompair]], [[e3fp]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
