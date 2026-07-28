---
tags: [control, status/complete]
---
# PXR External Test

> **Summary:** Re-evaluates every flavor on two fixed external PXR hold-outs instead of the
> sweep's seed-dependent internal split. It exists because the [[Report Card]]'s PXR column,
> where [[rdkit2d]] led, was the study's cleanest specialization signal, and that signal rests on
> a split that moves with the finetune seed. On fixed external molecules it does not reproduce.

- Test sets: OpenADMET PXR-challenge Phase 1 (253 molecules) and Phase 2 (260), from the
  `openadmet/pxr-challenge-train-test` HF dataset
- Train/val: a single fixed 90/10 split of `pxr_pec50.parquet` (1950/217, seed 42) shared across
  every flavor and seed, so the finetune seed varies only head init and training
- 160 finetunes: 16 models (15 flavors + stock) × 5 seeds × 2 phases, reduced protocol
- Standalone by construction: dedicated `results/pxr_ext_metrics.csv`, never the shared
  `results/metrics.csv` or the [[Report Card]]

## Why the internal split was suspect

The sweep's PXR endpoint splits with an inline Butina `ClusterSplitter` whose `random_seed` is
the finetune seed, so split membership changes between seeds. A flavor's PXR cell therefore
averages over five different evaluation sets, and a flavor can look strong partly through the
splits it happened to draw. Pinning the split and moving the test set outside the training
distribution entirely removes both effects.

Data handling: the challenge `pEC50` is already −log10(molarity), matching `PXR_pEC50`, so it is
used as-is with no log transform; test SMILES are RDKit-canonicalized; no test molecule overlaps
train by InChIKey and the two phases are disjoint.

## Result

Sorted by phase 1; an asterisk marks a delta against stock significant at Welch p ≤ 0.05.

| flavor | Phase 1 R² | Phase 2 R² |
|---|---|---|
| [[surrogate_adme]] | 0.361 ± 0.027 | 0.415 ± 0.028 |
| [[erg]] | 0.336 ± 0.032 | 0.348 ± 0.015* |
| **[[Stock CheMeleon]]** | **0.325 ± 0.043** | **0.413 ± 0.021** |
| [[minimol]] | 0.299 ± 0.036 | 0.372 ± 0.027* |
| [[jazzy]] | 0.287 ± 0.038 | 0.321 ± 0.055* |
| [[atompair]] | 0.280 ± 0.044 | 0.385 ± 0.033 |
| [[usrcat]] | 0.245 ± 0.026* | 0.334 ± 0.046* |
| [[osmordred]] | 0.241 ± 0.054* | 0.331 ± 0.054* |
| [[rdkit2d]] | 0.234 ± 0.078 | 0.336 ± 0.024* |
| osmordred_pca95 | 0.216 ± 0.028* | 0.333 ± 0.033* |
| osmordred_pca80 | 0.204 ± 0.059* | 0.359 ± 0.024* |
| [[whim]] | 0.157 ± 0.041* | 0.230 ± 0.040* |
| osmordred_pca90 | 0.134 ± 0.033* | 0.300 ± 0.013* |
| [[pubchem]] | 0.134 ± 0.047* | 0.258 ± 0.030* |
| [[ecfp]] | 0.036 ± 0.043* | 0.100 ± 0.045* |
| [[e3fp]] | 0.027 ± 0.024* | 0.065 ± 0.035* |

(The three `osmordred_pca` rows are the [[osmordred PCA targets]].)

**No pretrained flavor significantly beats stock on either phase.** [[surrogate_adme]] is the
only flavor above stock on both, and neither margin is significant (+0.036, +0.003). Most
flavors land significantly below.

**[[rdkit2d]] is the load-bearing result.** It led the internal Butina-split PXR column; here it
is mid-pack and below stock on both phases. A specialization measured on an internally generated
split did not survive a fixed external hold-out, which is a caution about how much weight any
single report-card cell can carry, not just about PXR.

Phase 2 is easier than phase 1 for every model (stock 0.413 against 0.325), so the phases differ
in difficulty as well as membership. The binary fingerprints fail far worse here than on the
report card, which fits their leaky-pretext read but is much starker on genuinely held-out
molecules.

## Related

- The column it re-tests: [[Report Card]] · Protocol: [[Finetune Protocols]]
- Reference: [[Stock CheMeleon]]
- Other standalone studies: [[osmordred_surrogate]], [[External Foundations]]
