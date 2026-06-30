# Findings: fit-to-purpose CheMeleon foundations

Source of record for headline results. Per-flavor and per-endpoint detail lives in the
Obsidian wiki under `wiki/`; where the wiki and this file disagree, this file wins.

## The question

Does the descriptor target a CheMeleon-style foundation is pretrained against determine
which ADMET endpoints and endpoint families it serves best? And does stacking the
per-flavor finetuned predictions into a meta-model beat the best single foundation?

## Status

Scaffolding. No training results yet. This section will lead with the report card (the
endpoint-by-flavor matrix for the selected metric) once the first flavors finish.

## Report card

To be filled. The artifact is `analysis/report_card.py`: rows are endpoints across all
benchmark sets, columns are foundation flavors, each cell is one selectable metric
(default R-squared). The read to capture here: which flavor wins each endpoint family,
and whether any flavor dominates or is dominated overall.

## Per-flavor read

To be filled, one short paragraph per flavor as results land. Expected priors to test:

- Continuous descriptor flavors (osmordred, rdkit2d, erg) should be the strongest general foundations.
- Binary fingerprint flavors (ecfp, atompair, pubchem, e3fp) are leaky/weak pretexts and
  may underperform; confirm or refute.
- 3D flavors (usrcat, whim, e3fp) encode information absent from the 2D graph; test
  whether that helps any endpoint family.
- Learned-model flavors (minimol, surrogate_adme, ml_qm) distill another model's
  knowledge; test whether that transfers better than hand-crafted descriptor targets.

## Meta-model

To be filled. Whether the stacked ensemble of foundations beats the best single flavor
per endpoint, and which flavors carry the ensemble.
