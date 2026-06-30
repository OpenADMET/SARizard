---
tags: [flavor, status/planned]
---
# minimol

> **Summary:** 512-dim minimol embedding, regressed with MSE. A learned, distilled GNN
> fingerprint (Graphcore) trained on large multitask molecular data; this flavor distills that
> representation into our backbone.

- Target: 512 continuous values · Loss: MSE · Source: learned model (isolated env, graphium stack)
- Calculator: `sarizard/pretraining/features/minimol_target.py`

## Hypothesis

A general-purpose learned embedding carries broad, task-agnostic structure, so it may be the
strongest all-rounder across families and a reliable backbone. The open question is whether a
learned target beats hand-built descriptors at their own specialties.

## Related

- Learned-model siblings: [[surrogate_adme]], [[ml_qm]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
