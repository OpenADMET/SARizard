---
tags: [flavor, status/planned]
---
# minimol

> **Summary:** 512-dim minimol embedding, regressed with MSE. A learned, distilled GNN
> fingerprint (Graphcore) trained on large multitask molecular data; this flavor distills that
> representation into our backbone.

- Target: 512 continuous values · Loss: MSE · Source: learned model (isolated env, graphium stack)
- Calculator: `sarizard/pretraining/features/minimol_target.py`

## Status

Milestone 7, in progress (its siblings [[surrogate_adme]] and [[ml_qm]] are held out; see
their Status sections). Not implicated by the target-dropout-fraction blocker in `TODO.md`
(512 dims is comparably wide to osmordred's 3585). The cached target was found to be
entirely NaN before this run: `envs/minimol.yml` left `scipy` unpinned, so it resolved to
1.15.3, which dropped `float16` sparse-matrix support that `graphium`'s featurizer needs,
so every calculator call failed silently. Fixed by pinning `scipy<1.13`.

## Hypothesis

A general-purpose learned embedding carries broad, task-agnostic structure, so it may be the
strongest all-rounder across families and a reliable backbone. The open question is whether a
learned target beats hand-built descriptors at their own specialties.

## Related

- Learned-model siblings: [[surrogate_adme]], [[ml_qm]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
