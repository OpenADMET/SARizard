---
tags: [flavor, status/green]
---
# minimol

> **Summary:** 512-dim minimol embedding, regressed with MSE. A learned, distilled GNN
> fingerprint (Graphcore) trained on large multitask molecular data; this flavor distills that
> representation into our backbone.

- Target: 512 continuous values · Loss: MSE · Source: learned model (isolated env, graphium stack)
- Calculator: `sarizard/pretraining/features/minimol_target.py`

## Status

Complete: pretrained, finetuned at 5 seeds under all three protocols, and evaluated. Not
implicated by the target-dropout question (512 dims is comparably wide to osmordred's 3585), so
it pretrains under the standard masked-pretext regime.

One bug is worth remembering: the cached target was once entirely NaN. `envs/minimol.yml` left
`scipy` unpinned, so it resolved to 1.15.3, which dropped the `float16` sparse-matrix support
`graphium`'s featurizer needs, and every calculator call failed with no per-row signal to catch
it. Fixed by pinning `scipy<1.13`.

## Hypothesis

A general-purpose learned embedding carries broad, task-agnostic structure, so it may be the
strongest all-rounder across families and a reliable backbone. The open question is whether a
learned target beats hand-built descriptors at their own specialties.

## Result (frozen sweep, 5 seeds)
Frozen mean R² **0.343 ± 0.006** against a 0.294 ± 0.010 stock baseline, significant, and the
strongest same-corpus flavor on the card. Holds under reduced (0.371) but drops to 0.308 under
unlocked, below stock though not significantly so, the usual wash-out once the backbone can move
freely. Beats
stock on 27 of 32 endpoint-columns frozen, and takes [[hERG]] outright (0.220 against stock
0.148).

A learned 512-dim embedding transferring better than any direct-compute descriptor block is the
cleanest positive result for the learned-flavor family, and unlike [[surrogate_adme]] it runs on
the shared corpus, so it is a fair column.

## Related

- Learned-model sibling: [[surrogate_adme]]
- Regime: [[Shared Corpus and Regime]] · [[Report Card]] · [[Meta-Model]]
